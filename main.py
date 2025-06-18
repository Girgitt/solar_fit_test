# ========================================================================== #
# Polynomial regression CLI tool: fit/update/execute solar model alignment
# Adds sun-elevation & time-of-day features, supports model save/load via JSON
# ========================================================================== #

import datetime as dt
import argparse
import matplotlib.pyplot as plt

from utils import *

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
OUT_DIR = Path("./plots")  # all PNGs go here
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# ARGUMENT PARSING
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--action", choices=["update", "execute"], required=True,
                    help="Specify whether to 'update' (train/save) or 'execute' (load/apply) the model")
parser.add_argument("--model_id", default="default",
                    help="Model identifier used for saving/loading coefficients")
parser.add_argument("--csv", required=True,
                    help="Path to CSV file with input data")
args = parser.parse_args()

model_path = Path(f"model_config__{args.model_id}.json")

# ----------------------------------------------------------------------------
# LOAD AND PREPROCESS DATA
# ----------------------------------------------------------------------------
df = pd.read_csv(args.csv, parse_dates=["time"])
#df["power_hi.common@sensor_1:VALUE"] *= 2  # scale input by factor of 2
df["sun_elev"] = df["time"].apply(lambda t: solar_elevation(52.2297, 21.0122, 2, t))

# smooth time-of-day encodings
df["hour_decimal"] = (df["time"].dt.hour
                      + df["time"].dt.minute / 60
                      + df["time"].dt.second / 3600)
hour_angle = 2 * np.pi * df["hour_decimal"] / 24
df["hour_sin"] = np.sin(hour_angle)
df["hour_cos"] = np.cos(hour_angle)
df = df.dropna(subset=["power_reference.common@sensor_1:VALUE", "power_hi.common@sensor_1:VALUE"])

X = df[["power_hi.common@sensor_1:VALUE", "sun_elev", "hour_sin", "hour_cos"]]

# ----------------------------------------------------------------------------
# ACTION: UPDATE (fit and save model) or EXECUTE (load and predict)
# ----------------------------------------------------------------------------
if args.action == "update":
    y = df["power_reference.common@sensor_1:VALUE"]
    fit_hourly_models(df, args.model_id, log_dir='./logs')

elif args.action == "execute":
    df = execute_hourly_prediction(df, args.model_id, log_dir='./logs')

    mask = df["pred_reference_hourly"].notna()
    y_true = df.loc[mask, "power_reference.common@sensor_1:VALUE"]
    y_pred = df.loc[mask, "pred_reference_hourly"]

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    save_model_metrics(dt, args, r2, mae, 'model_name', 'linear_regression')
    print(f"[Hourly models] R² = {r2:.3f}   MAE = {mae:.1f} W")

# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

out_dir = Path("./plots")
model_id = "tmp_v1"

plt.figure(figsize=(9, 4))
plt.plot(df["time"], df["power_hi.common@sensor_1:VALUE"], label="Power HI sensor", linewidth=0.9)
plt.plot(df["time"], df["power_reference.common@sensor_1:VALUE"], label="Power Reference (actual)", linewidth=0.9)
plt.title("Raw input series over time")
plt.xlabel("Time")
plt.ylabel("Power [W]")
plt.legend()
plt.grid(True)
plt.tight_layout()
fname = out_dir / f"series_vs_time_hourly_{model_id}_{timestamp}.png"
plt.savefig(fname, dpi=300)
print("Saved:", fname)
plt.show()

if "pred_reference_hourly" in df:
    df_sorted = df[df["pred_reference_hourly"].notna()].sort_values("power_hi.common@sensor_1:VALUE")

    plt.figure(figsize=(6, 5))
    plt.scatter(df["power_hi.common@sensor_1:VALUE"],
                df["power_reference.common@sensor_1:VALUE"],
                s=8, alpha=0.3, label="Actual reference")
    plt.plot(df_sorted["power_hi.common@sensor_1:VALUE"],
             df_sorted["pred_reference_hourly"],
             linewidth=2, label="Hourly model prediction")

    plt.title("Hourly model fit: output vs input")
    plt.xlabel("Power HI sensor [W]")
    plt.ylabel("Reference power [W]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = out_dir / f"fit_curve_hourly_{model_id}_{timestamp}.png"
    plt.savefig(fname, dpi=300)
    print("Saved:", fname)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(df["time"], df["power_reference.common@sensor_1:VALUE"], label="Power Reference (actual)", linewidth=0.9)
    plt.plot(df["time"], df["pred_reference_hourly"], label="Hourly model output", linewidth=0.9)
    plt.title("Reference sensor vs hourly model over time")
    plt.xlabel("Time")
    plt.ylabel("Reference power [W]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = out_dir / f"reference_vs_fit_hourly_{model_id}_{timestamp}.png"
    plt.savefig(fname, dpi=300)
    print("Saved:", fname)
    plt.show()

# ----------------------------------------------------------------------------
# OPTIONAL: C++-friendly COEFF DUMP FOR MICROCONTROLLER
# ----------------------------------------------------------------------------
'''
if args.action == "update":
    names = poly.get_feature_names_out()
    coeff = np.r_[model.coef_]
    template = textwrap.dedent("""\
        // Auto-generated coefficients (float32)
        constexpr float INTERCEPT = {inter:.8f}f;
        constexpr float COEF[{n}] = {{
        {coef_body}
        }};
    """)
    body = ",\n".join([f"    /*{n:>4}*/ {c:.8f}f" for n, c in zip(names, coeff)])
    print(template.format(inter=model.intercept_, coef_body=body, n=len(names)))

    save_model_metrics('model_name', 'linear_regression')
'''
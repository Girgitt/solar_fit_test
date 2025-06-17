# ========================================================================== #
# Polynomial regression CLI tool: fit/update/execute solar model alignment
# Adds sun-elevation & time-of-day features, supports model save/load via JSON
# ========================================================================== #

import math
import datetime as dt
import argparse
import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
OUT_DIR = Path("./plots")  # all PNGs go here
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 2. ARGUMENT PARSING
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
# 3. HELPER: very-lightweight solar-elevation approximation
# ----------------------------------------------------------------------------
def solar_elevation(lat, lon, tz_offset, dt_local):
    n = dt_local.timetuple().tm_yday
    lt = dt_local.hour + dt_local.minute / 60 + dt_local.second / 3600  # local clock time
    B = math.radians((360 / 365) * (n - 81))
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)  # Eq. of Time [min]
    lstm = 15 * tz_offset
    tc = 4 * (lon - lstm) + eot  # Time-corr [min]
    lst = lt + tc / 60  # Local solar time
    omega = math.radians(15 * (lst - 12))  # Hour angle
    delta = math.radians(23.45 * math.sin(math.radians(360 * (284 + n) / 365)))
    phi = math.radians(lat)
    cos_z = (math.sin(phi) * math.sin(delta) +
             math.cos(phi) * math.cos(delta) * math.cos(omega))
    z = math.acos(max(-1, min(1, cos_z)))  # clamp
    return math.degrees(math.pi / 2 - z)

# ----------------------------------------------------------------------------
# 4. MODEL STORAGE HELPERS
# ----------------------------------------------------------------------------
def save_model_to_json(model, poly, path):
    data = {
        "intercept": model.intercept_,
        "coefficients": model.coef_.tolist(),
        "features": poly.get_feature_names_out().tolist()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Model saved to {path}")

def load_model_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["intercept"], data["coefficients"], data["features"]

# ----------------------------------------------------------------------------
# 5. LOAD AND PREPROCESS DATA
# ----------------------------------------------------------------------------
df = pd.read_csv(args.csv, parse_dates=["time"])
#df["power_hi.common@sensor_1:VALUE"] *= 2  # scale input by factor of 2
df["sun_elev"] = df["time"].apply(lambda t: solar_elevation(52.2297, 21.0122, 2, t))

# 3b. smooth time-of-day encodings
df["hour_decimal"] = (df["time"].dt.hour
                      + df["time"].dt.minute / 60
                      + df["time"].dt.second / 3600)
hour_angle = 2 * np.pi * df["hour_decimal"] / 24
df["hour_sin"] = np.sin(hour_angle)
df["hour_cos"] = np.cos(hour_angle)
df = df.dropna(subset=["power_reference.common@sensor_1:VALUE", "power_hi.common@sensor_1:VALUE"])

X = df[["power_hi.common@sensor_1:VALUE", "sun_elev", "hour_sin", "hour_cos"]]
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# ----------------------------------------------------------------------------
# 6. ACTION: UPDATE (fit and save model) or EXECUTE (load and predict)
# ----------------------------------------------------------------------------
if args.action == "update":
    y = df["power_reference.common@sensor_1:VALUE"]
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.20, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"R² = {r2:.3f}   MAE = {mae:.1f} W")

    df["pred_reference"] = model.predict(X_poly)
    save_model_to_json(model, poly, model_path)

elif args.action == "execute":
    intercept, coeffs, _ = load_model_from_json(model_path)
    df["pred_reference"] = np.dot(X_poly, np.array(coeffs)) + intercept

    # Calculate R² and MAE for full data
    y_true = df["power_reference.common@sensor_1:VALUE"]
    y_pred = df["pred_reference"]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"R² = {r2:.3f}   MAE = {mae:.1f} W")

# ----------------------------------------------------------------------------
# 7. PLOTS
# ----------------------------------------------------------------------------
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

plt.figure(figsize=(9, 4))
plt.plot(df["time"], df["power_hi.common@sensor_1:VALUE"], label="Power HI sensor", linewidth=0.9)
plt.plot(df["time"], df["power_reference.common@sensor_1:VALUE"], label="Power Reference (actual)", linewidth=0.9)
plt.title("Raw input series over time")
plt.xlabel("Time")
plt.ylabel("Power [W]")
plt.legend()
plt.grid(True)
plt.tight_layout()
fname = OUT_DIR / f"series_vs_time_{timestamp}.png"
plt.savefig(fname, dpi=300)
print("Saved:", fname)
plt.show()

if "pred_reference" in df:
    df_sorted = df.sort_values("power_hi.common@sensor_1:VALUE")
    plt.figure(figsize=(6, 5))
    plt.scatter(df["power_hi.common@sensor_1:VALUE"], df["power_reference.common@sensor_1:VALUE"], s=8, alpha=0.3, label="Actual reference")
    plt.plot(df_sorted["power_hi.common@sensor_1:VALUE"], df_sorted["pred_reference"], linewidth=2, label="Model prediction")
    plt.title("Model fit: output vs input")
    plt.xlabel("Power HI sensor [W]")
    plt.ylabel("Reference power [W]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = OUT_DIR / f"fit_curve_{timestamp}.png"
    plt.savefig(fname, dpi=300)
    print("Saved:", fname)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(df["time"], df["power_reference.common@sensor_1:VALUE"], label="Power Reference (actual)", linewidth=0.9)
    plt.plot(df["time"], df["pred_reference"], label="Model output (fitted)", linewidth=0.9)
    plt.title("Reference sensor vs fitted model over time")
    plt.xlabel("Time")
    plt.ylabel("Reference power [W]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = OUT_DIR / f"reference_vs_fit_{timestamp}.png"
    plt.savefig(fname, dpi=300)
    print("Saved:", fname)
    plt.show()

# ----------------------------------------------------------------------------
# 8. OPTIONAL: C++-friendly COEFF DUMP FOR MICROCONTROLLER
# ----------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------
    # 9. SAVE METRICS TO JSON
    # ----------------------------------------------------------------------------
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / 'model_metrics.json'

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics_list = json.load(f)
    else:
        metrics_list = []

    metrics_list.append({
        "timestamp": dt.datetime.now().isoformat(),
        "action": args.action,
        "model_id": args.model_id,
        "csv": args.csv,
        "r2": round(r2, 4),
        "mae": round(mae, 2)
    })

    with open(metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=2)

    print(f"Metrics saved to {metrics_path}")


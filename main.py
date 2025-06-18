# ========================================================================== #
# Polynomial regression CLI tool: fit/update/execute solar model alignment
# Adds sun-elevation & time-of-day features, supports model save/load via JSON
# ========================================================================== #

import argparse

from utils import *

def main():
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

        print(f"[Hourly models] R² = {r2:.3f}   MAE = {mae:.1f} W")

    plot_model_outputs(df, model_id=args.model_id, prefix="linear", show=True)


if __name__ == '__main__':
    main()

    
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


# ========================================================================== #
# Polynomial regression CLI tool: fit/update/execute solar model alignment
# Adds sun-elevation & time-of-day features, supports model save/load via JSON
# ========================================================================== #

'''
python main.py --action=update --model_id=hi_fit_mixed --csv=./data/eds_trend__power_hi.csv
python main.py --action=execute --model_id=hi_fit_mixed --csv=./data/eds_trend__power_hi.csv
'''

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

    data_columns = [col for col in df.columns if col != "time"]

    print("Available data columns:")
    for i, col in enumerate(data_columns):
        print(f"{i}: {col}")

    print(data_columns[0])
    print(data_columns[1])

    #df["power_hi.common@sensor_1:VALUE"] *= 2  # scale input by factor of 2
    df["sun_elev"] = df["time"].apply(lambda t: solar_elevation(52.2297, 21.0122, 2, t))

    # smooth time-of-day encodings
    df["hour_decimal"] = (df["time"].dt.hour
                          + df["time"].dt.minute / 60
                          + df["time"].dt.second / 3600)
    hour_angle = 2 * np.pi * df["hour_decimal"] / 24
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)

    # Adjust proper header names considering readed csv file!
    #df = df.dropna(subset=["power_reference.common@sensor_1:VALUE", "power_hi.common@sensor_1:VALUE"])
    sensor_value_1 = data_columns[0]
    sensor_value_ref = data_columns[1]
    df = df.dropna(subset=[sensor_value_1, sensor_value_ref])

    X = df[[sensor_value_1, "sun_elev", "hour_sin", "hour_cos"]]

    # ----------------------------------------------------------------------------
    # ACTION: UPDATE (fit and save model) or EXECUTE (load and predict)
    # ----------------------------------------------------------------------------
    if args.action == "update":
        y = df[sensor_value_1]
        fit_hourly_models(df,
                          args.model_id,
                          log_dir='./logs',
                          sensor_value_1=sensor_value_1,
                          sensor_value_ref=sensor_value_ref,)

    elif args.action == "execute":
        df = execute_hourly_prediction(df,
                                       args.model_id,
                                       log_dir='./logs',
                                       sensor_value_1=sensor_value_1,)

        mask = df["pred_reference_hourly"].notna()
        y_true = df.loc[mask, sensor_value_ref]
        y_pred = df.loc[mask, "pred_reference_hourly"]

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        model_path = Path(f"./logs/{args.model_id}.json")
        with open(model_path, "r") as f:
            model_data = json.load(f)

        df_metrics = pd.DataFrame(model_data)
        df_metrics["hour"] = pd.to_datetime(df_metrics["hour"])
        df_metrics = df_metrics.sort_values("hour")

        identify_and_export_weak_hours(df_metrics,
                                       min_r2=0.8,
                                       max_mae=40.0,
                                       model_id=args.model_id)

        group_and_export_models(df_metrics,
                                output_path=f"./logs/grouped_{args.model_id}.json",
                                tol_a=0.02,
                                tol_b=1.0)

        plot_weak_hourly_segments(df,
                                  weak_hours_path=f"./logs/weak_hours_{args.model_id}.json",
                                  model_data_path=f"./logs/{args.model_id}.json",
                                  sensor_value_1=sensor_value_1,
                                  sensor_value_ref=sensor_value_ref,)

        smooth_models(models_list_path=f"./logs/{args.model_id}.json",
                      blend_minutes=10,
                      time_delta=timedelta(minutes=1),
                      output_path=f"./logs/smoothed_{args.model_id}.json")

    plot_model_outputs(df,
                       model_id=args.model_id,
                       prefix="linear",
                       show=True,
                       sensor_value_1=sensor_value_1,
                       sensor_value_ref=sensor_value_ref)

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


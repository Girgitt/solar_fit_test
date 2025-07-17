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
    sensor_values = [
        data_columns[2],
        data_columns[3],
        data_columns[4],
    ]
    sensor_value_ref = data_columns[6]
    df = df.dropna(subset=sensor_values + [sensor_value_ref])

    # ----------------------------------------------------------------------------
    # ACTION: UPDATE (fit and save model) or EXECUTE (load and predict)
    # ----------------------------------------------------------------------------
    if args.action == "update":

        fit_hourly_models(
            df=df,
            model_id=args.model_id,
            log_dir="./logs",
            sensor_values=sensor_values,
            sensor_value_ref=sensor_value_ref,
        )

    elif args.action == "execute":

        df = execute_hourly_prediction(
            df=df,
            model_id=args.model_id,
            log_dir="./logs",
            sensor_values=sensor_values,
        )

        #results = {}
        results = evaluate_predictions_per_sensor(
            df=df,
            model_id=args.model_id,
            sensor_values=sensor_values,
            sensor_value_ref="power_reference.common@sensor_1:VALUE"
        )

        for sensor, res in results.items():
            df_metrics = res["df_metrics"]

            identify_and_export_weak_hours(
                df_metrics,
                min_r2=0.8,
                max_mae=40.0,
                model_id=f"{args.model_id}__{sensor.replace('@', '_').replace(':', '_')}"
            )

            group_and_export_models(
                df_metrics,
                output_path=f"./logs/grouped_{args.model_id}__{sensor.replace('@', '_').replace(':', '_')}.json",
                tol_a=0.02,
                tol_b=1.0
            )

        plot_weak_hourly_segments(
            df=df,
            weak_hours_path=f"./logs/weak_hours_{args.model_id}.json",
            model_data_path=f"./logs/{args.model_id}.json",
            output_dir="./plots/weak_hours",
            sensor_values=sensor_values,
            sensor_value_ref=sensor_value_ref,
        )

        smooth_models(models_list_path=f"./logs/{args.model_id}.json",
                      blend_minutes=10,
                      time_delta=timedelta(minutes=1),
                      output_path=f"./logs/smoothed_{args.model_id}.json")

    plot_model_outputs(
        df=df,
        model_id="multi_sensor_model",
        sensor_values=sensor_values,
        sensor_value_ref=sensor_value_ref,
    )

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


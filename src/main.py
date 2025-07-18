# ========================================================================== #
# Polynomial regression CLI tool: fit/update/execute solar model alignment
# Adds sun-elevation & time-of-day features, supports model save/load via JSON
# ========================================================================== #

# ----------------------------------------------------------------------------
# ACTION: UPDATE (fit and save model) or EXECUTE (load and predict)
# ----------------------------------------------------------------------------

'''
python src/main.py --action=update --model_id=hi_fit_mixed --csv=./data/eds_trend__power_hi.csv
python src/main.py --action=execute --model_id=hi_fit_mixed --csv=./data/eds_trend__power_hi.csv
'''

import argparse
from utils import *

def main():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    LOG_DIR = ROOT_DIR / "logs"
    PLOT_DIR = ROOT_DIR / "plots"
    DATA_DIR = ROOT_DIR / "data"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    args = argument_parsing(parser)
    model_path = Path(f"model_config__{args.model_id}.json")

    df = pd.read_csv(args.csv, parse_dates=["time"])
    data_columns = [col for col in df.columns if col != "time"]

    print_available_data_columns(data_columns)
    sensor_values, sensor_value_ref, df = select_available_data_columns_to_process(data_columns, df)

    model_parameters = ModelParameters(
        df=df,
        args = args,
        log_dir = LOG_DIR,
        plot_dir = PLOT_DIR,
        sensor_values = sensor_values,
        sensor_value_ref = sensor_value_ref,
        min_r2 = 0.8,
        max_mae = 40.0,
        tolerance_min = 0.02,
        tolerance_max = 1.0,
        blend_minutes = 10,
        timedelta = 1,
    )

    if args.action == "update":
        update_function(model_parameters)

    elif args.action == "execute":
        execute_function(model_parameters)

    plot_model_outputs(
        df=model_parameters.df,
        model_id="multi_sensor_model",
        sensor_values=model_parameters.sensor_values,
        sensor_value_ref=model_parameters.sensor_value_ref,
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


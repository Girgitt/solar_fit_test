# ========================================================================== #
# Polynomial regression CLI tool: fit/update/execute solar model alignment
# Adds sun-elevation & time-of-day features, supports model save/load via JSON
# ========================================================================== #

# ----------------------------------------------------------------------------
# ACTION: UPDATE (fit and save model) or EXECUTE (load and predict)
# ----------------------------------------------------------------------------

'''
python src/main.py --action=update --model_id=hi_fit_mixed --csv=./dataeds_trend__power_hi.csv
python src/main.py --action=execute --model_id=hi_fit_mixed --csv=./data/eds_trend__power_hi.csv

python src/pvtools/main.py --action=update --model_id=1_day_timestamp_3s --csv=./data/1_day_timestamp_3s.csv
python src/pvtools/main.py --action=execute --model_id=high_sunshine_frequent_cover_1_day --csv=./data/high_sunshine_frequent_cover_1_day.csv
'''

import argparse
import pandas as pd

from pathlib import Path

from pvtools.io_file.reader import load_dataframe_from_csv
from pvtools.utils.utilities import argument_parsing, check_if_csv_contains_timezone_info, print_available_data_columns, \
    select_available_data_columns_to_process
from pvtools.utils.update_function import update_function
from pvtools.utils.execute_function import execute_function
from pvtools.config.params import ModelParameters, ClearSkyParameters, ClearSkyCalculatedValues
from pvtools.preprocess.preprocess_data import preprocess_data, sanitize_filename

def main():
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    LOG_DIR = ROOT_DIR / "logs"
    PLOT_DIR = ROOT_DIR / "plots"
    DATA_DIR = ROOT_DIR / "data"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    args = argument_parsing(parser)
    model_path = Path(f"model_config__{args.model_id}.json")

    check_if_csv_contains_timezone_info(args.csv)

    df = pd.read_csv(args.csv, parse_dates=["time"])
    df = preprocess_data(
        df=df,
        save_dir=Path(args.csv),
        target_timedelta='1min' # available formats: 'xs' 'xmin' 'xh' 'xms' where x is a number
    )

    # use measurement_limitations after calibration - otherwise VEML values are too low!!

    # to get sunny periods for VEML's I need to do calibrtion first!
    # Then designate sunny periods and do calibration again (only for sunny periods)!

    # Second method is better I think. It takes sunny period for DAVIS and uses it for all VAML's

    data_columns = [col for col in df.columns if col != "time"]
    time_column = df["time"]

    print_available_data_columns(data_columns)
    sensor_names, sensor_name_ref, df = select_available_data_columns_to_process(data_columns, df)

    model_parameters = ModelParameters(
        df=df,
        df_time = time_column,
        args = args,
        log_dir = LOG_DIR,
        data_filename_dir = Path(args.csv),
        plot_dir = PLOT_DIR,
        sensor_names = sensor_names,
        sensor_name_ref = sensor_name_ref
    )

    clear_sky_parameters = ClearSkyParameters(
        start_time=model_parameters.df_time.iloc[0],
        end_time=model_parameters.df_time.iloc[-1],
        warsaw_lat=52.22977,
        warsaw_lon=21.01178,
        tz='Europe/Warsaw',
        altitude=170,
        name='Warsaw',
        frequency='1min',
        albedo=0.2,
        surface_tilt=30,  # degrees from horizontal
        surface_azimuth = 180,  # south-facing
    )

    clear_sky_calculated_values = ClearSkyCalculatedValues(
        poa=load_dataframe_from_csv(Path(DATA_DIR / "calculated_data" / model_parameters.data_filename_dir.stem / "poa_values.csv")),
        clearsky_periods=load_dataframe_from_csv(
            Path(DATA_DIR /
                 "calculated_data" / model_parameters.data_filename_dir.stem /
                 f"{sanitize_filename(model_parameters.sensor_name_ref)}_sunny_periods.csv"
                 ))
    )

    if args.action == "update":
        update_function(model_parameters, clear_sky_parameters)

    elif args.action == "execute":
        execute_function(model_parameters, clear_sky_calculated_values)

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


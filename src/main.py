# ========================================================================== #
# Polynomial regression CLI tool: fit/update/execute solar model alignment
# Adds sun-elevation & time-of-day features, supports model save/load via JSON
# ========================================================================== #

# ----------------------------------------------------------------------------
# ACTION: UPDATE (fit and save model) or EXECUTE (load and predict)
# ----------------------------------------------------------------------------

'''
python src/main.py --action=update --model_id=25-09-04_08 --csv=./data/org/25-09-04_08.csv

python src/main.py --action=update --model_id=hi_fit_mixed --csv=./dataeds_trend__power_hi.csv
python src/main.py --action=execute --model_id=hi_fit_mixed --csv=./data/eds_trend__power_hi.csv

python src/pvtools/main.py --action=update --model_id=1_day_timestamp_3s --csv=./data/1_day_timestamp_3s.csv
python src/pvtools/main.py --action=execute --model_id=high_sunshine_frequent_cover_1_day --csv=./data/high_sunshine_frequent_cover_1_day.csv
'''

import argparse
import pandas as pd

from pathlib import Path

from pvtools.io_file.reader import load_dataframe_from_csv
from pvtools.utils.utilities import argument_parsing, print_available_data_columns, select_available_data_columns_to_process
from pvtools.utils.update_function import update_function
from pvtools.utils.execute_function import execute_function
from pvtools.config.params import ModelParameters, ClearSkyParameters, ClearSkyCalculatedValues
from pvtools.preprocess.preprocess_data import preprocess_data, sanitize_filename, ensure_dataframe_contains_valid_data, ensure_datetime_contains_timezone

def main():
    ROOT_DIR = Path(__file__).resolve().parent.parent#.parent
    LOG_DIR = ROOT_DIR / "logs"
    PLOT_DIR = ROOT_DIR / "plots"
    DATA_DIR = ROOT_DIR / "data"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    args = argument_parsing(parser)
    target_frequency='1min'

    df = pd.read_csv(args.csv, parse_dates=["time"])
    df_filtered = preprocess_data(
        df=df,
        target_timedelta=target_frequency, # available formats: 'xs' 'xmin' 'xh' 'xms' where x is a number
        save_dir=Path(args.csv)
    )

    data_columns = [col for col in df_filtered.columns if col != "time"]

    print_available_data_columns(data_columns)
    sensor_names, sensor_name_ref, df_filtered = select_available_data_columns_to_process(data_columns, df_filtered)

    # use measurement_limitations after calibration - otherwise VEML values are too low!!

    # to get sunny periods for VEML's I need to do calibrtion first!
    # Then designate sunny periods and do calibration again (only for sunny periods)!

    # Second method is better I think. It takes sunny period for DAVIS and uses it for all VAML's

    model_parameters = ModelParameters(
        df=df_filtered,
        df_time = df_filtered["time"],
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
        frequency=target_frequency,
        albedo=0.2,
        surface_tilt=30,  # degrees from horizontal
        surface_azimuth = 180,  # south-facing
    )

    if args.action == "update":
        update_function(model_parameters, clear_sky_parameters)

    elif args.action == "execute":

        clear_sky_calculated_values = ClearSkyCalculatedValues(
            poa=load_dataframe_from_csv(
                Path(DATA_DIR / "calculated_data" / model_parameters.data_filename_dir.stem / "poa_values.csv")),
            clearsky_periods=load_dataframe_from_csv(
                Path(DATA_DIR /
                     "calculated_data" / model_parameters.data_filename_dir.stem /
                     f"{sanitize_filename(model_parameters.sensor_name_ref)}_sunny_periods.csv"
                     ))
        )

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


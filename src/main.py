# ========================================================================== #
# Polynomial regression CLI tool: fit/update/execute solar model alignment
# Adds sun-elevation & time-of-day features, supports model save/load via JSON
# ========================================================================== #

# ----------------------------------------------------------------------------
# ACTION: UPDATE (fit and save model) or EXECUTE (load and predict)
# ----------------------------------------------------------------------------

'''
python src/main.py --action=update --model_id=25-09-04_08 --csv=./data/org/25-09-04_08.csv --calibration=linear --sensors 0 1 2 --reference 3
'''

import os
import logging

import argparse
import pandas as pd

from pathlib import Path

from pvtools.utils.utilities import initialize_dirs_for_base_dir, select_available_data_columns_to_process, \
    print_available_data_columns, argument_parsing
from pvtools.utils.update_function import update_function

from pvtools.utils.execute_function import execute_function
from pvtools.config.params import ModelParameters, ClearSkyParameters, ClearSkyCalculatedValues
from pvtools.preprocess.preprocess_data import preprocess_data


def get_logging_format():
    return '%(asctime)s : %(levelname)s [%(processName)s-%(threadName)s %(name)s.%(funcName)s:%(lineno)d] %(message)s'


log = logging.getLogger('main_thd')

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(get_logging_format(),
                              datefmt='%b %d %H:%M:%S')
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)


def main():

    parser = argparse.ArgumentParser()
    args = argument_parsing(parser)
    target_frequency = '1min'

    arg_data_dir = args.data_dir if args.data_dir else None

    if arg_data_dir is None:
        data_dir = Path(os.getcwd())
    else:
        data_dir = Path(arg_data_dir)

    log_dir, plot_dir, data_dir = initialize_dirs_for_base_dir(data_dir)

    df = pd.read_csv(args.csv, parse_dates=["time"])
    df_filtered = preprocess_data(
        df=df,
        target_timedelta=target_frequency, # available formats: 'xs' 'xmin' 'xh' 'xms' where x is a number
        save_dir=Path(args.csv),
    )

    data_columns = [col for col in df_filtered.columns if col != "time"]

    print_available_data_columns(data_columns)
    sensor_names, sensor_name_ref, df_filtered = select_available_data_columns_to_process(
        data_columns=data_columns,
        df=df_filtered,
        sensors_chosen=args.sensors,
        sensor_ref_chosen=args.reference
    )

    model_parameters = ModelParameters(
        df=df_filtered,
        df_time = df_filtered["time"],
        args = args,
        log_dir = log_dir,
        data_dir = data_dir, # data/
        filename= Path(args.csv).stem, # data/org/filename.csv
        plot_dir = plot_dir,
        sensor_names = sensor_names,
        sensor_name_ref = sensor_name_ref
    )

    clearsky_parameters = ClearSkyParameters(
        start_time=model_parameters.df_time.iloc[0],
        end_time=model_parameters.df_time.iloc[-1],
        warsaw_lat=52.22977,
        warsaw_lon=21.01178,
        tz='Europe/Warsaw',
        altitude=170,
        name='Warsaw',
        frequency=target_frequency,
        albedo=0.2,
        surface_tilt=0,  # degrees from horizontal
        surface_azimuth = 180,  # south-facing
    )

    clearsky_calculated_values = ClearSkyCalculatedValues(
        poa=pd.DataFrame(),
        clearsky_periods=pd.Series(),
        cloudy_periods=pd.Series()
    )

    if args.action == "update":
        update_function(model_parameters, clearsky_parameters, clearsky_calculated_values)

    elif args.action == "execute":
        execute_function(model_parameters, clearsky_calculated_values)


if __name__ == '__main__':
    main()


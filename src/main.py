# ========================================================================== #
# Polynomial regression CLI tool: fit/update/execute solar model alignment
# Adds sun-elevation & time-of-day features, supports model save/load via JSON
# ========================================================================== #

# ----------------------------------------------------------------------------
# ACTION: UPDATE (fit and save model) or EXECUTE (load and predict)
# ----------------------------------------------------------------------------

'''
python ../src/main.py --action=update --model_id=test_update_1 --csv=../data/org/25-09-04_08.csv --calibration=linear --sensors 0 1 2 --reference 3 --start_time_hour 4 --start_time_minute 0 --end_time_hour 18 --end_time_minute 0 --latitude 52.22977 --longtitude 21.01178 --timezone=Europe/Warsaw --altitude 170 --name=Warsaw --frequency=1min --albedo 0.25 --surface_tilt 0 --surface_azimuth 180 --project_dir=./test_update_1
python ../src/main.py --action=execute --model_id=test_execute_1 --csv=../data/org/25-09-26__25-10-02.csv --calibration=linear --sensors 0 1 2 --start_time_hour 4 --start_time_minute 0 --end_time_hour 18 --end_time_minute 0 --latitude 52.22977 --longtitude 21.01178 --timezone=Europe/Warsaw --altitude 170 --name=Warsaw --frequency=1min --albedo 0.25 --surface_tilt 0 --surface_azimuth 180 --project_dir ./test_execute_1 --calibration_metrics_dir ../test_update_1/logs/25-09-04_08
'''

import os
import logging

import argparse
import pandas as pd

from pathlib import Path
from datetime import time

from pvtools.utils.utilities import initialize_dirs_for_base_dir, initialize_dirs_for_loading_dependencies, \
    select_available_data_columns_to_process, print_available_data_columns, argument_parsing
from pvtools.utils.update_function import update_function

from pvtools.utils.execute_function import execute_function
from pvtools.config.params import ModelData, ModelDirectories, ClearSkyParameters, ClearSkyCalculatedValues, ModelTimes
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
    target_frequency = args.frequency

    if args.action == "update" and args.reference == None:
        raise(AttributeError("Cannot update without reference sensor!"))

    if args.reference == None and args.calibration_metrics_dir == None:
        raise(AttributeError("Cannot execute without specified calibration metrics directory!"))

    arg_project_dir = args.project_dir if args.project_dir else None

    if arg_project_dir is None:
        project_dir = Path(os.getcwd()).resolve()
    else:
        project_dir = Path(arg_project_dir).resolve()

    log_dir, plot_dir, data_dir = initialize_dirs_for_base_dir(project_dir)
    load_metrics_dir = initialize_dirs_for_loading_dependencies(args.calibration_metrics_dir)

    df = pd.read_csv(args.csv, parse_dates=["time"])

    start_daytime = time(args.start_time_hour, args.start_time_minute) # 4:00 GMT -> 6:00 UTC+2
    end_daytime = time(args.end_time_hour, args.end_time_minute) # 17:00 GMT -> 19:00 UTC+2

    df_filtered = preprocess_data(
        df=df,
        target_timedelta=target_frequency, # available formats: 'xs' 'xmin' 'xh' 'xms' where x is a number
        start_daytime=start_daytime,
        end_daytime=end_daytime,
        save_dir=Path(data_dir),
        filename=Path(args.csv).stem,
    )

    data_columns = [col for col in df_filtered.columns if col != "time"]

    print_available_data_columns(data_columns)
    sensor_names, sensor_name_ref, df_filtered = select_available_data_columns_to_process(
        data_columns=data_columns,
        df=df_filtered,
        sensors_chosen=args.sensors,
        sensor_ref_chosen=args.reference
    )

    model_data = ModelData(
        df=df_filtered,
        df_time=df_filtered["time"],
        sensor_names=sensor_names,
        sensor_name_ref=sensor_name_ref,
    )

    model_dirs = ModelDirectories(
        project_dir=project_dir,
        log_dir=log_dir,
        data_dir=data_dir,
        plot_dir=plot_dir,
        filename=Path(args.csv).stem,
        load_metrics_dir=load_metrics_dir,
    )

    model_times = ModelTimes(
        start_time=model_data.df_time.iloc[0],
        end_time=model_data.df_time.iloc[-1],
        frequency=target_frequency,
        divided_linear_regression_interval=args.divided_linear_regression_intervals,
        start_daytime_cut=start_daytime,
        end_daytime_cut=end_daytime,
    )

    clearsky_params = ClearSkyParameters(
        warsaw_lat=args.latitude,
        warsaw_lon=args.longtitude,
        tz=args.timezone,
        altitude=args.altitude,
        name=args.name,
        albedo=args.albedo,
        surface_tilt=args.surface_tilt,  # degrees from horizontal
        surface_azimuth=args.surface_azimuth,  # south-facing
    )

    clearsky_cal_val = ClearSkyCalculatedValues(
        poa=pd.DataFrame(),
        clearsky_periods=pd.Series(),
        cloudy_periods=pd.Series()
    )

    if args.action == "update":
        update_function(
            model_data=model_data,
            model_dirs=model_dirs,
            model_times=model_times,
            clearsky_params=clearsky_params,
            clearsky_cal_val=clearsky_cal_val,
        )

    elif args.action == "execute":
        execute_function(
            model_data=model_data,
            model_dirs=model_dirs,
            model_times=model_times,
            clearsky_params=clearsky_params,
            clearsky_cal_val=clearsky_cal_val,
            calibration_method=args.calibration
        )


if __name__ == '__main__':
    main()


import numpy as np
import pandas as pd

from pathlib import Path
from typing import TypeAlias, Literal
from datetime import time

from pvtools.config.params import ModelParameters, ModelDirectories, ClearSkyParameters, ClearSkyCalculatedValues
from pvtools.modeling.calculate_calibration_parameters import linear_regression, divided_linear_regression, polynominal_regression, \
    decision_tree_regression, mlp_regression
from pvtools.solar_domain.clearsky import clear_sky, detect_clearsky_periods
from pvtools.solar_domain.determine_orientation import determine_system_azimuth_and_tilt
from pvtools.utils.apply_mask_for_dataframe import apply_mask_for_dataframe
from pvtools.solar_domain.measurement_limitations import limit_sensor_ref_irradiance_to_clear_sky_model

Period_type: TypeAlias = Literal['sunny', 'cloudy']

def update_function(
        model_parameters: ModelParameters,
        model_directories: ModelDirectories,
        clear_sky_parameters: ClearSkyParameters,
        clearsky_calculated_values: ClearSkyCalculatedValues,
        start_time: time = time(4, 0), # 4:00 GMT -> 6:00 UTC+2
        end_time: time = time(17, 0), # 17:00 GMT -> 19:00 UTC+2
) -> None:
    df_sunny_cutted_short, df_cloudy_cutted_short = process_solar_data_with_clearsky_detection_and_masking(
        model_parameters=model_parameters,
        model_directories=model_directories,
        clearsky_parameters=clear_sky_parameters,
        clearsky_calculated_values=clearsky_calculated_values,
        start_time=start_time,
        end_time=end_time
    )

    calculate_regression(
        df=df_sunny_cutted_short,
        model_parameters=model_parameters,
        model_directories=model_directories,
        period="sunny"
    )

    calculate_regression(
        df=df_cloudy_cutted_short,
        model_parameters=model_parameters,
        model_directories=model_directories,
        period="cloudy"
    )


def process_solar_data_with_clearsky_detection_and_masking(
        model_parameters: ModelParameters,
        model_directories: ModelDirectories,
        clearsky_parameters: ClearSkyParameters,
        clearsky_calculated_values: ClearSkyCalculatedValues,
        start_time: time = time(4, 0),
        end_time: time = time(17, 0),
) -> [pd.DataFrame, pd.DataFrame]:

    poa = clear_sky(
        clearsky_parameters=clearsky_parameters,
        show=False,
        start_time=start_time,
        end_time=end_time,
        save_dir_plot=model_directories.plot_dir / model_directories.filename,
        save_dir=model_directories.data_dir,
        filename=model_directories.filename
    )

    clearsky_calculated_values.poa = poa

    df_limited = limit_sensor_ref_irradiance_to_clear_sky_model(
        df=model_parameters.df,
        clearsky_df=clearsky_calculated_values.poa,
        sensor_name_ref=model_parameters.sensor_name_ref,
        poa_global_name='poa_global',
        save_dir=model_directories.data_dir,
        filename=model_directories.filename
    )

    model_parameters.df = df_limited

    clearsky_periods_all, cloudy_periods_all = detect_clearsky_periods(
        poa=poa,
        df=model_parameters.df,
        sensor_name_ref=model_parameters.sensor_name_ref,
        save_dir=model_directories.data_dir,
        filename=model_directories.filename
    )

    determine_system_azimuth_and_tilt(
        clear_sky_parameters=clearsky_parameters,
        df=model_parameters.df,
        sunny_mask=clearsky_periods_all,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        tilts=np.arange(0, 30, 1),  # None
        azimuths=np.arange(170, 190, 1)  # None
    )

    df_sunny_periods_cutted_short = apply_mask_for_dataframe(
        data_filename=model_directories.filename,
        sensor_name_ref=model_parameters.sensor_name_ref,
        period_type="sunny",
        save_dir=model_directories.data_dir
    )

    df_cloudy_periods_cutted_short = apply_mask_for_dataframe(
        data_filename=model_directories.filename,
        sensor_name_ref=model_parameters.sensor_name_ref,
        period_type="cloudy",
        save_dir=model_directories.data_dir
    )

    return df_sunny_periods_cutted_short, df_cloudy_periods_cutted_short


def calculate_regression(
        df: pd.DataFrame,
        model_parameters: ModelParameters,
        model_directories: ModelDirectories,
        period: Period_type
) -> None:

    linear_regression(
        df=df,
        period=period,
        log_dir=model_directories.log_dir,
        data_filename=model_directories.filename,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    divided_linear_regression(
        df=df,
        period=period,
        log_dir=model_directories.log_dir,
        data_filename=model_directories.filename,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    polynominal_regression(
        df=df,
        period=period,
        log_dir=model_directories.log_dir,
        data_filename=model_directories.filename,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    decision_tree_regression(
        df=df,
        period=period,
        log_dir=model_directories.log_dir,
        data_filename=model_directories.filename,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    mlp_regression(
        df=df,
        period=period,
        log_dir=model_directories.log_dir,
        data_filename=model_directories.filename,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )
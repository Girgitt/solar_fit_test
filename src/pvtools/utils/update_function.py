import numpy as np
import pandas as pd

from pathlib import Path

from pvtools.config.params import ModelParameters, ClearSkyParameters, ClearSkyCalculatedValues
from pvtools.modeling.calculate_calibration_parameters import linear_regression, divided_linear_regression, polynominal_regression, \
    decision_tree_regression, mlp_regression
from pvtools.solar_domain.clearsky import clear_sky, detect_clearsky_periods
from pvtools.solar_domain.determine_orientation import determine_system_azimuth_and_tilt
from pvtools.utils.apply_sunny_mask import apply_sunny_mask
from pvtools.solar_domain.measurement_limitations import limit_sensor_ref_irradiance_to_clear_sky_model

def update_function(
        model_parameters: ModelParameters,
        clear_sky_parameters: ClearSkyParameters,
        clearsky_calculated_values: ClearSkyCalculatedValues,
) -> None:
    df_sunny = process_solar_data_with_clearsky_detection_and_masking(
        model_parameters=model_parameters,
        clearsky_parameters=clear_sky_parameters,
        clearsky_calculated_values=clearsky_calculated_values
    )

    #model_parameters.df = df_sunny

    calculate_regression(model_parameters)

def process_solar_data_with_clearsky_detection_and_masking(
        model_parameters: ModelParameters,
        clearsky_parameters: ClearSkyParameters,
        clearsky_calculated_values: ClearSkyCalculatedValues
) -> pd.DataFrame:
    poa = clear_sky(
        clearsky_parameters=clearsky_parameters,
        show=False,
        save_dir_plot=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        save_dir=model_parameters.data_dir,
        filename=model_parameters.filename
    )

    clearsky_calculated_values.poa = poa

    df_limited = limit_sensor_ref_irradiance_to_clear_sky_model(
        df=model_parameters.df,
        clearsky_df=clearsky_calculated_values.poa,
        sensor_name_ref=model_parameters.sensor_name_ref,
        poa_global_name='poa_global',
        save_dir=model_parameters.data_dir,
        filename=model_parameters.filename
    )

    model_parameters.df = df_limited

    clearsky_periods = detect_clearsky_periods(
        poa=poa,
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        save_dir=model_parameters.data_dir,
        filename=model_parameters.filename
    )

    clearsky_calculated_values.clearsky_periods = clearsky_periods

    determine_system_azimuth_and_tilt(
        clear_sky_parameters=clearsky_parameters,
        df=model_parameters.df,
        sunny_mask=clearsky_periods,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        tilts=np.arange(0, 30, 1),  # None
        azimuths=np.arange(170, 190, 1)  # None
    )

    df_sunny_periods = apply_sunny_mask(
        data_filename=model_parameters.filename,
        sensor_name_ref=model_parameters.sensor_name_ref,
        save_dir=model_parameters.data_dir
    )

    return df_sunny_periods

def calculate_regression(model_parameters: ModelParameters) -> None:
    linear_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename=model_parameters.filename,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    divided_linear_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename=model_parameters.filename,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    polynominal_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename=model_parameters.filename,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    decision_tree_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename=model_parameters.filename,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    mlp_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename=model_parameters.filename,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )
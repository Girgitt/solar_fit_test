import numpy as np

from pathlib import Path

from pvtools.config.params import ModelParameters, ClearSkyParameters
from pvtools.modeling.calibrate import linear_regression, divided_linear_regression, polynominal_regression, \
    decision_tree_regression, mlp_regression
from pvtools.solar_domain.clearsky import clear_sky, detect_clearsky_periods
from pvtools.solar_domain.determine_orientation import determine_system_azimuth_and_tilt
from pvtools.utils.apply_sunny_mask import apply_sunny_mask

def update_function(
        model_parameters: ModelParameters,
        clear_sky_parameters: ClearSkyParameters
) -> None:
    calculate_regression(model_parameters)
    process_solar_data_with_clearsky_detection_and_masking(
        model_parameters=model_parameters,
        clear_sky_parameters=clear_sky_parameters
    )

def calculate_regression(model_parameters: ModelParameters) -> None:
    linear_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename_dir=model_parameters.data_filename_dir,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    divided_linear_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename_dir=model_parameters.data_filename_dir,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    polynominal_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename_dir=model_parameters.data_filename_dir,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    decision_tree_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename_dir=model_parameters.data_filename_dir,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    mlp_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename_dir=model_parameters.data_filename_dir,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

def process_solar_data_with_clearsky_detection_and_masking(
        model_parameters: ModelParameters,
        clear_sky_parameters: ClearSkyParameters
) -> None:
    poa = clear_sky(
        clear_sky_parameters=clear_sky_parameters,
        show=False,
        save_dir_plot=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        save_dir_data=model_parameters.data_filename_dir
    )

    clearsky_periods = detect_clearsky_periods(
        poa=poa,
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        save_dir=model_parameters.data_filename_dir
    )

    determine_system_azimuth_and_tilt(
        clear_sky_parameters=clear_sky_parameters,
        df=model_parameters.df,
        sunny_mask=clearsky_periods,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        tilts=np.arange(0, 30, 1),  # None
        azimuths=np.arange(170, 190, 1)  # None
    )

    apply_sunny_mask(
        data_dir=model_parameters.data_filename_dir,
        data_filename=model_parameters.data_filename_dir,
        sensor_name_ref=model_parameters.sensor_name_ref,
        save_dir=model_parameters.data_filename_dir.parent
    )




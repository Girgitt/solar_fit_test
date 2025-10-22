import pandas as pd
import numpy as np

from pathlib import Path
from typing import TypeAlias, Literal
from datetime import time
from argparse import ArgumentParser, Namespace

from pvtools.calibration.calibrate import (calibrate_by_linear_regression, calibrate_by_divided_linear_regression,
    calibrate_by_polynominal_regression, calibrate_by_decision_tree_regression, calibrate_by_mlp_regression,
    calibrate_by_fuzzy_linear_regression)
from pvtools.config.params import ModelParameters, ModelDirectories, ClearSkyParameters, ClearSkyCalculatedValues
from pvtools.visualisation.plotter import (plot_from_dataframe, plot_predicted_data, plot_poa_vs_reference,
    plot_poa_reference_with_clearsky_periods)
from pvtools.io_file.reader import load_dataframe_from_csv
from pvtools.utils.utilities import (load_filtered_and_calculated_data_needed_for_execute_function_no_periods_detected,
                                     load_filtered_and_calculated_data_needed_for_execute_function_periods_detected,
                                     sanitize_filename, create_calibrated_dataframe,
                                     check_if_sunny_cloudy_periods_exists)
from pvtools.postprocess.postprocess_data import postprocess_data, merge_sunny_and_cloudy_calibrated_dataframes
from pvtools.solar_domain.clearsky import clear_sky, detect_clearsky_periods
from pvtools.solar_domain.determine_orientation import determine_system_azimuth_and_tilt
from pvtools.utils.apply_mask_for_dataframe import apply_mask_for_dataframe


Period_type: TypeAlias = Literal['sunny', 'cloudy']


def execute_function(
        model_parameters: ModelParameters,
        model_directories: ModelDirectories,
        clearsky_parameters: ClearSkyParameters,
        clearsky_calculated_values: ClearSkyCalculatedValues,
        start_time: time = time(4, 0),
        end_time: time = time(17, 0),
        surface_tilt: int = 0,
        surface_azimuth: int = 180,
        calibration_method: str = "linear"
) -> None:

    clearsky_calculated_values.clearsky_periods=None
    clearsky_calculated_values.cloudy_periods=None

    if not check_if_sunny_cloudy_periods_exists(model_directories):
        [df, poa] = load_filtered_and_calculated_data_needed_for_execute_function_no_periods_detected(model_directories)

        if model_parameters.sensor_name_ref is not None:
            clearsky_periods_all, cloudy_periods_all = detect_clearsky_periods(
                poa=clearsky_calculated_values.poa,
                df=model_parameters.df,
                sensor_name_ref=model_parameters.sensor_name_ref,
                save_dir=model_directories.data_dir,
                filename=model_directories.filename
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

            df_cutted_short_periods = merge_sunny_and_cloudy_calibrated_dataframes(
                df_sunny=df_sunny_periods_cutted_short,
                df_cloudy=df_cloudy_periods_cutted_short
            )

            model_parameters.df = df_cutted_short_periods

        model_parameters.df = df

    else:
        [df, df_sunny, df_cloudy, poa, clearsky_periods, cloudy_periods] = (
            load_filtered_and_calculated_data_needed_for_execute_function_periods_detected(
                model_directories=model_directories,
                sensor_name_ref=model_parameters.sensor_name_ref
            ))

        clearsky_calculated_values.clearsky_periods = clearsky_periods
        clearsky_calculated_values.cloudy_periods = cloudy_periods

        df_cutted_short_periods = merge_sunny_and_cloudy_calibrated_dataframes(
            df_sunny=df_sunny,
            df_cloudy=df_cloudy
        )

        model_parameters.df = df_cutted_short_periods

    clearsky_calculated_values.poa = poa

    if surface_tilt is not 0:
        determine_system_azimuth_and_tilt(
            clear_sky_parameters=clearsky_parameters,
            df=df,
            sunny_mask=clearsky_periods_all,
            sensor_names=model_parameters.sensor_names,
            sensor_name_ref=model_parameters.sensor_name_ref,
            tilts=np.arange(surface_tilt-10, surface_tilt+10, 1),
            azimuths=np.arange(surface_azimuth-10, surface_azimuth+10, 1)
        )

    calibrate(
        df=model_parameters.df,
        poa=clearsky_calculated_values.poa,
        model_parameters=model_parameters,
        model_directories=model_directories
    )

    df_calibrated = create_calibrated_dataframe(
        model_parameters=model_parameters,
        model_directories=model_directories,
        calibration_method=calibration_method
    )

    postprocess_df = postprocess_data(
        df=df_calibrated,
        sensor_names=model_parameters.sensor_names,
        data_dir=model_directories.data_dir,
        filename=model_directories.filename,
        clearsky_df=clearsky_calculated_values.poa,
        poa_global_name='poa_global',
        calibration_method=calibration_method
    )

    model_parameters.df = postprocess_df

    plot(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        data_dir=model_directories.data_dir,
        plot_dir=model_directories.plot_dir,
        log_dir=model_directories.log_dir,
        filename=model_directories.filename,
        poa=clearsky_calculated_values.poa,
        clearsky_periods=clearsky_calculated_values.clearsky_periods,
        calibration_method=calibration_method
    )


def calibrate(
        df: pd.DataFrame,
        poa: pd.DataFrame,
        model_parameters: ModelParameters,
        model_directories: ModelDirectories
) -> None:

    calibrate_by_linear_regression(
        df=df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        load_params_dir=model_directories.load_metrics_dir,
        save_dir=model_directories.log_dir,
        filename=model_directories.filename,
        period_flag=False
    )

    '''
    calibrate_by_fuzzy_linear_regression(
        df=df,
        poa=poa,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=model_parameters.filename
    )

    calibrate_by_divided_linear_regression(
        df=df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=model_parameters.filename
    )

    calibrate_by_polynominal_regression(
        df=df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=model_parameters.filename
    )

    calibrate_by_decision_tree_regression(
        df=df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=model_parameters.filename
    )

    calibrate_by_mlp_regression(
        df=df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=model_parameters.filename
    )
    '''


def plot(
        df: pd.DataFrame,
        sensor_names: list[str],
        sensor_name_ref: str,
        data_dir: Path,
        plot_dir: Path,
        log_dir: Path,
        filename: str,
        poa: pd.DataFrame,
        clearsky_periods: pd.DataFrame,
        calibration_method: str,
) -> None:

    df_org = load_dataframe_from_csv(Path(data_dir / "filtered" / f"{filename}.csv"))
    df_org.columns = [sanitize_filename(name) for name in df_org.columns]

    df_filtered = load_dataframe_from_csv(Path(data_dir / "filtered" / "calibrated" / filename /
                                               f"{calibration_method}.csv"))

    plot_from_dataframe(
        df=df_org,
        save_dir=plot_dir / filename,
        filename="org_series_vs_time.png",
        sensor_names=sensor_names,
        sensor_name_ref=sensor_name_ref,
        show=True,
        title="Original series vs. time"
    )

    plot_from_dataframe(
        df=df_filtered,
        save_dir=plot_dir / filename,
        filename=f"predicted_series_vs_time_{calibration_method}.png",
        sensor_names=sensor_names,
        sensor_name_ref=sensor_name_ref,
        show=True,
        title="Predicted series vs. time"
    )

    plot_predicted_data(
        calibration_method_dir=log_dir / filename,
        show=False,
        save_dir=plot_dir / filename,
    )

    if sensor_name_ref is not None:
        plot_poa_vs_reference(
            poa_global=poa['poa_global'],
            sensor_reference=df[sensor_name_ref],
            save_dir=plot_dir / filename,
            show=True,
        )

        plot_poa_reference_with_clearsky_periods(
            poa_global=poa['poa_global'],
            sensor_reference=df[sensor_name_ref],
            sunny=clearsky_periods['if_sunny'],
            save_dir=plot_dir / filename,
            show=True,
        )

    # FIXME - plot calibrated values and poa on one graph, and reference (if calibrated with other metrics)
    # FIXME with calibrated sensors!
import pandas as pd
import logging

from typing import TypeAlias, Literal

from pvtools.calibration.calibrate import (calibrate_by_linear_regression, calibrate_by_divided_linear_regression,
                                           calibrate_by_polynominal_regression, calibrate_by_decision_tree_regression,
                                           calibrate_by_mlp_regression,
                                           calibrate_by_fuzzy_linear_regression,
                                           calibrate_by_divided_linear_regression_mean)
from pvtools.config.params import ModelData, ModelDirectories, ClearSkyParameters, ClearSkyCalculatedValues, ModelTimes
from pvtools.visualisation.plotter import (plot_from_dataframe, plot_poa_vs_reference,
    plot_poa_reference_with_clearsky_periods)
from pvtools.utils.utilities import (load_filtered_and_calculated_data_needed_for_execute_function_no_periods_detected,
                                     load_filtered_and_calculated_data_needed_for_execute_function_periods_detected,
                                     create_calibrated_dataframe, check_if_calibration_method_available,
                                     check_if_sunny_cloudy_periods_exists, create_dataframe_with_sensor_values_and_poa)
from pvtools.postprocess.postprocess_data import postprocess_data, merge_sunny_and_cloudy_calibrated_dataframes
from pvtools.solar_domain.clearsky import clear_sky, detect_clearsky_periods
from pvtools.solar_domain.determine_orientation import determine_system_azimuth_and_tilt
from pvtools.utils.apply_mask_for_dataframe import apply_mask_for_dataframe
from pvtools.preprocess.preprocess_data import delete_night_period
from pvtools.visualisation.plotter import plot_clear_sky, plot_poa_components

log = logging.getLogger("calibrate")

Period_type: TypeAlias = Literal['sunny', 'cloudy']


def execute_function(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
        clearsky_params: ClearSkyParameters,
        clearsky_cal_val: ClearSkyCalculatedValues,
        calibration_method: str = "linear"
) -> None:

    poa, cs = clear_sky(
        clearsky_params=clearsky_params,
        model_dirs=model_dirs,
        model_times=model_times
    )

    poa = delete_night_period(
        df=poa,
        start=model_times.start_daytime_cut,
        end=model_times.end_daytime_cut,
    )

    filename = model_dirs.filename
    save_dir_plot = model_dirs.plot_dir / filename

    plot_clear_sky(cs, save_dir=save_dir_plot, show=False)
    plot_poa_components(poa, save_dir=save_dir_plot, show=False)

    clearsky_cal_val.poa = poa
    clearsky_periods = None
    cloudy_periods = None
    period_flag=False

    if model_data.sensor_name_ref is not None:
        clearsky_periods, cloudy_periods, df_cutted_short_periods = run_full_clearsky_data_pipeline(
            model_data=model_data,
            model_dirs=model_dirs,
            clearsky_cal_val=clearsky_cal_val
        )

        #model_data.df = df_cutted_short_periods
        period_flag=True

        plot_poa_reference_with_clearsky_periods(
            poa_global=poa[["time", "poa_global"]],
            sensor_reference=model_data.df[["time", model_data.sensor_name_ref]],
            sunny=clearsky_periods,
            save_dir=save_dir_plot,
            show=False
        )

    else:
        df = load_filtered_and_calculated_data_needed_for_execute_function_no_periods_detected(
            data_dir=model_dirs.data_dir,
            filename=model_dirs.filename,
        )

        model_data.df = df

    clearsky_cal_val.clearsky_periods = clearsky_periods
    clearsky_cal_val.cloudy_periods = cloudy_periods
    clearsky_cal_val.poa = poa

    model_data.df["if_sunny"] = clearsky_periods

    surface_tilt = clearsky_params.surface_tilt

    if surface_tilt != 0:
        clearsky_params.surface_tilt, clearsky_params.surface_azimuth = determine_system_azimuth_and_tilt(
            model_data=model_data,
            model_times=model_times,
            clearsky_params=clearsky_params,
            sunny_mask=clearsky_periods
        )

    calibration_directory = check_if_calibration_method_available(
        log_dir=model_dirs.log_dir,
        filename=model_dirs.filename,
        calibration_method=calibration_method
    )

    calibrate(
        clearsky_cal_val=clearsky_cal_val,
        model_data=model_data,
        model_dirs=model_dirs,
        model_times=model_times,
        period_flag=period_flag,
        calibration_method=calibration_method
    )

    df_calibrated = create_calibrated_dataframe(
        model_parameters=model_data,
        model_directories=model_dirs,
        calibration_directory=calibration_directory
    )

    df_postprocess = postprocess_data(
        df=df_calibrated,
        sensor_names=model_data.sensor_names,
        data_dir=model_dirs.data_dir,
        filename=model_dirs.filename,
        clearsky_df=clearsky_cal_val.poa,
        poa_global_name='poa_global',
        calibration_method=calibration_method
    )

    model_data.df = df_postprocess

    [df_org,
     df_postprocess_calibrated_sensor_data_with_poa_global,
     df_calibrated_sensor_data_with_poa_global,
     df_org_sensor_data_with_poa_global] = (
        create_dataframe_with_sensor_values_and_poa(
            df_postprocess=df_postprocess,
            df_calibrated=df_calibrated,
            sensor_names=model_data.sensor_names,
            poa = clearsky_cal_val.poa,
            data_dir=model_dirs.data_dir,
            filename=model_dirs.filename
    ))

    list_of_dataframes = [
        df_postprocess,
        df_org,
        df_postprocess_calibrated_sensor_data_with_poa_global,
        df_calibrated_sensor_data_with_poa_global,
        df_org_sensor_data_with_poa_global
    ]

    plot(
        dataframes=list_of_dataframes,
        model_data=model_data,
        model_dirs=model_dirs,
        clearsky_cal_val=clearsky_cal_val,
        calibration_method=calibration_method
    )


def calibrate(
        clearsky_cal_val: ClearSkyCalculatedValues,
        model_data: ModelData,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
        period_flag: bool,
        calibration_method: str
) -> None:

    if calibration_method == "linear":
        calibrate_by_linear_regression(
            model_data=model_data,
            model_dirs=model_dirs,
            period_flag=period_flag
        )

    elif calibration_method == "fuzzy":
        calibrate_by_fuzzy_linear_regression(
            model_data=model_data,
            model_dirs=model_dirs,
            clearsky_cal_val=clearsky_cal_val,
            period_flag=period_flag
        )

    elif calibration_method == "divided":
        calibrate_by_divided_linear_regression(
            model_data=model_data,
            model_dirs=model_dirs,
            period_flag=period_flag
        )

    elif calibration_method == "divided_mean":
        calibrate_by_divided_linear_regression_mean(
            model_data=model_data,
            model_dirs=model_dirs,
            model_times=model_times,
            period_flag=period_flag
        )

    elif calibration_method == "poly":
        calibrate_by_polynominal_regression(
            model_data=model_data,
            model_dirs=model_dirs,
            period_flag=period_flag
        )

    elif calibration_method == "decision_tree":
        calibrate_by_decision_tree_regression(
            model_data=model_data,
            model_dirs=model_dirs,
            period_flag=period_flag
        )

    elif calibration_method == "mlp":
        calibrate_by_mlp_regression(
            model_data=model_data,
            model_dirs=model_dirs,
            period_flag=period_flag
        )
    else:
        raise ValueError(f"Unsupported calibration method: {calibration_method}")


def plot(
        dataframes: list[pd.DataFrame],
        model_data: ModelData,
        model_dirs: ModelDirectories,
        clearsky_cal_val: ClearSkyCalculatedValues,
        calibration_method: str,
) -> None:

    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref
    plot_dir = model_dirs.plot_dir
    filename = model_dirs.filename
    poa = clearsky_cal_val.poa
    clearsky_periods = clearsky_cal_val.clearsky_periods

    save_dir = plot_dir / filename

    [df_postprocess,
     df_org,
     df_postprocess_calibrated_sensor_data_with_poa_global,
     df_calibrated_sensor_data_with_poa_global,
     df_org_sensor_data_with_poa_global] = dataframes

    plot_from_dataframe(
        df=df_org,
        save_dir=save_dir,
        filename="org_series_vs_time.png",
        sensor_names=sensor_names,
        sensor_name_ref=sensor_name_ref,
        show=True,
        title="Original series vs time"
    )

    plot_from_dataframe(
        df=df_postprocess,
        save_dir=save_dir,
        filename=f"calibrated_series_vs_time_{calibration_method}.png",
        sensor_names=sensor_names,
        sensor_name_ref=sensor_name_ref,
        show=True,
        title="Calibrated series vs time"
    )

    plot_from_dataframe(
        df=df_org_sensor_data_with_poa_global,
        save_dir=save_dir,
        filename="org_series_with_poa_vs_time.png",
        sensor_names=sensor_names,
        sensor_name_ref="poa_global",
        show=True,
        title="Original series with poa global vs time"
    )

    plot_from_dataframe(
        df=df_postprocess_calibrated_sensor_data_with_poa_global,
        save_dir=save_dir,
        filename=f"postprocess_calibrated_series_with_poa_vs_time_{calibration_method}.png",
        sensor_names=sensor_names,
        sensor_name_ref="poa_global",
        show=True,
        title="Postprocess calibrated series with poa global vs time"
    )

    plot_from_dataframe(
        df=df_calibrated_sensor_data_with_poa_global,
        save_dir=save_dir,
        filename=f"calibrated_series_with_poa_vs_time_{calibration_method}.png",
        sensor_names=sensor_names,
        sensor_name_ref="poa_global",
        show=True,
        title="Calibrated series with poa global vs time"
    )

    if sensor_name_ref is not None:
        plot_poa_vs_reference(
            poa_global=poa['poa_global'],
            sensor_reference=df_postprocess[sensor_name_ref],
            save_dir=save_dir,
            show=True,
        )

def run_full_clearsky_data_pipeline(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        clearsky_cal_val: ClearSkyCalculatedValues,
) -> list[pd.DataFrame]:

    clearsky_periods_all, cloudy_periods_all = detect_clearsky_periods(
        poa=clearsky_cal_val.poa,
        df=model_data.df,
        sensor_name_ref=model_data.sensor_name_ref,
        save_dir=model_dirs.data_dir,
        filename=model_dirs.filename
    )

    df_sunny_periods_cutted_short = apply_mask_for_dataframe(
        data_filename=model_dirs.filename,
        sensor_name_ref=model_data.sensor_name_ref,
        period_type="sunny",
        save_dir=model_dirs.data_dir
    )

    df_cloudy_periods_cutted_short = apply_mask_for_dataframe(
        data_filename=model_dirs.filename,
        sensor_name_ref=model_data.sensor_name_ref,
        period_type="cloudy",
        save_dir=model_dirs.data_dir
    )

    df_cutted_short_periods = merge_sunny_and_cloudy_calibrated_dataframes(
        df_sunny=df_sunny_periods_cutted_short,
        df_cloudy=df_cloudy_periods_cutted_short
    )

    return_dfs = [
        clearsky_periods_all,
        cloudy_periods_all,
        df_cutted_short_periods
    ]

    return return_dfs
import pandas as pd
import logging

from typing import TypeAlias, Literal
from pathlib import Path

from pvtools.calibration.calibrate_to_reference import (calibrate_by_linear_regression,
                                                        calibrate_by_divided_linear_regression,
                                                        calibrate_by_polynominal_regression,
                                                        calibrate_by_decision_tree_regression,
                                                        calibrate_by_mlp_regression,
                                                        calibrate_by_fuzzy_linear_regression,
                                                        calibrate_by_divided_linear_regression_mean)
from pvtools.calibration.calibrate_to_poa.gaussian_process import gaussian_process_pipeline
from pvtools.calibration.calibrate_to_poa.ransac import ransac_pipeline
from pvtools.calibration.calibrate_to_poa.clearsky_utils import (clearsky_detection_by_frequency_method,
                                                                 frequency_analysis, low_frequency_mask,
                                                                 two_medians_mask, derivative_df,
                                                                 relative_derivative_mask,
                                                                 determine_signal_amplification_scale,
                                                                 compute_scale_factor)
from pvtools.config.params import ModelData, ModelDirectories, ClearSkyParameters, ClearSkyCalculatedValues, ModelTimes
from pvtools.visualisation.plotter import (plot_from_dataframe, plot_poa_vs_reference,
                                           plot_poa_reference_with_clearsky_periods,
                                           plot_sensors_calibrated_directly_to_poa,
                                           plot_clear_sky, plot_poa_components,
                                           tmp_plot_check_masks, tmp_plot_smoothed_vemls,
                                           tmp_plot_smoothed_derivs_vemls, tmp_plot_evenelope,
                                           tmp_plot_evenelope_scaled,
                                           tmp_plot_scaled_sensor_vs_reference)
from pvtools.utils.utilities import (load_filtered_and_calculated_data_needed_for_execute_function_no_periods_detected,
                                     load_filtered_and_calculated_data_needed_for_execute_function_periods_detected,
                                     create_calibrated_dataframe, check_if_calibration_method_available,
                                     check_if_sunny_cloudy_periods_exists, create_dataframe_with_sensor_values_and_poa)
from pvtools.postprocess.postprocess_data import postprocess_data, merge_sunny_and_cloudy_calibrated_dataframes
from pvtools.solar_domain.clearsky import clear_sky, detect_clearsky_periods, detect_clearsky_periods_v2
from pvtools.solar_domain.determine_orientation import determine_system_azimuth_and_tilt
from pvtools.utils.apply_mask_for_dataframe import apply_mask_for_dataframe
from pvtools.preprocess.preprocess_data import delete_night_period

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

    model_data.df["if_sunny"] = model_data.df["time"].map(clearsky_periods)

    surface_tilt = clearsky_params.surface_tilt

    #FIXME - funkcja determine_system_azimuth_and_tilt potrzebuje sensora referencyjnego (nie zawsze podany)
    '''
    if surface_tilt != 0:
        clearsky_params.surface_tilt, clearsky_params.surface_azimuth = determine_system_azimuth_and_tilt(
            model_data=model_data,
            model_times=model_times,
            clearsky_params=clearsky_params,
            sunny_mask=clearsky_periods
        )
    '''

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

    calibrate_directly_to_poa(
        model_data=model_data,
        clearsky_cal_val=clearsky_cal_val,
        model_dirs=model_dirs,
        model_times=model_times
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


def calibrate_directly_to_poa(
        model_data: ModelData,
        clearsky_cal_val: ClearSkyCalculatedValues,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
) -> None:

    for sensor_name in model_data.sensor_names:
        frequency_mask = low_frequency_mask(
            sensor=model_data.df[sensor_name],
            sampling_sec=60,
            low_freq_max=0.002,
            window_sec=14400, # 4hrs
            thershold=0.80
        )

        frequency_mask.index = model_data.df["time"]

        two_medians_mask_ = two_medians_mask(
            sensor=model_data.df[sensor_name],
            time=model_data.df["time"],
            short_window="30min",
            long_window="4h",
            rel_threshold=0.05,
            min_run_length=10
        )

        # ---------------------------------------------------------------------------------------------

        derivative_df_ = derivative_df(
            sensor=model_data.df[sensor_name],
            poa_global=clearsky_cal_val.poa["poa_global"],
            time=model_data.df["time"],
            window_length=240,
            polyorder=1,
            delta=1.0
        )

        tmp_plot_smoothed_vemls(
            result_df=derivative_df_,
            title="Check smoothness",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"smoothness_{sensor_name}",
            show=False
        )


        tmp_plot_smoothed_derivs_vemls(
            result_df=derivative_df_,
            title="Check derivative",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"derivative_{sensor_name}",
            show=False
        )

        # ---------------------------------------------------------------------------------------------

        relative_derivative_mask_ = relative_derivative_mask(
            sensor=model_data.df[sensor_name],
            poa_global=clearsky_cal_val.poa["poa_global"],
            time=model_data.df["time"],
            window_length=240,
            polyorder=1,
            delta=1.0
        )

        tmp_plot_smoothed_vemls(
            result_df=relative_derivative_mask_,
            title="Check relative smoothness",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"smoothness_relative_{sensor_name}",
            show=False
        )

        tmp_plot_smoothed_derivs_vemls(
            result_df=relative_derivative_mask_,
            title="Check relative derivative",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"derivative_relivative_{sensor_name}",
            show=False
        )

        tmp_plot_check_masks(
            result_df=relative_derivative_mask_,
            title="Check derivative relative mask",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"mask_derivative_relative_{sensor_name}",
            show=False
        )

        # ---------------------------------------------------------------------------------------------

        evenelope, sensor_smooth = determine_signal_amplification_scale(
            sensor=model_data.df[sensor_name],
            poa_global=clearsky_cal_val.poa["poa_global"],
            time=model_data.df["time"],
            smooth_window=30,
            polyorder=3,
            minimum_disatnce_between_peaks=10,
            smoothing_factor=2000
        )

        tmp_plot_evenelope(
            sensor_smooth=sensor_smooth,
            evenelope=evenelope,
            title="Upper evenelope",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"evenelope_{sensor_name}",
            show=False
        )

        evenelope = pd.Series(evenelope, index=model_data.df["time"])

        factor = compute_scale_factor(
            envelope=evenelope,
            poa_global=clearsky_cal_val.poa["poa_global"],
            poa_min=20.0,
            env_min=2.0,
            use_median=False,
        )

        tmp_plot_evenelope_scaled(
            evenelope=factor * evenelope,
            sensor=factor * model_data.df[sensor_name],
            poa_global=clearsky_cal_val.poa["poa_global"],
            title="Upper evenelope scaled",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"evenelope_scaled_{sensor_name}",
            show=False
        )

        tmp_plot_scaled_sensor_vs_reference(
            sensor=factor * model_data.df[sensor_name],
            reference=model_data.df["irr_dav_1_VALUE"], # or MAX_VALUE - this is temporary hardcoded
            title="Scaled sensor vs reference",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"evenelope_scaled_vs_reference{sensor_name}",
            show=False
        )

        # ---------------------------------------------------------------------------------------------

        df_freq_mask = pd.DataFrame({
            "sensor": model_data.df[sensor_name],
            "poa_global": clearsky_cal_val.poa["poa_global"],
            "mask": frequency_mask
        })



        df_two_medians_mask = pd.DataFrame({
            "sensor": model_data.df[sensor_name],
            "poa_global": clearsky_cal_val.poa["poa_global"],
            "mask": two_medians_mask_
        })

        tmp_plot_check_masks(
            result_df=df_freq_mask,
            title="Check frequency mask",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"mask_freq_{sensor_name}",
            show=False
        )

        tmp_plot_check_masks(
            result_df=df_two_medians_mask,
            title="Check two medians mask",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"mask_two_medians_{sensor_name}",
            show=False
        )

        ransac_freq_mask = ransac_pipeline(
            sensor=model_data.df[sensor_name],
            poa_global=clearsky_cal_val.poa["poa_global"],
            clearsky_mask=frequency_mask,
            time=model_data.df["time"]
        )

        ransac_two_medians_mask = ransac_pipeline(
            sensor=model_data.df[sensor_name],
            poa_global=clearsky_cal_val.poa["poa_global"],
            clearsky_mask=two_medians_mask_,
            time=model_data.df["time"]
        )

        plot_sensors_calibrated_directly_to_poa(
            result_df=ransac_freq_mask,
            title="RANSAC calibration directly to POA with frequency mask",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"direct_calibration_to_poa_by_ransac_freq_mask_{sensor_name}",
            show=False
        )

        plot_sensors_calibrated_directly_to_poa(
            result_df=ransac_two_medians_mask,
            title="Frequency calibration directly to POA with two medians mask",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"direct_calibration_to_poa_by_ransac_two_medians_mask_{sensor_name}",
            show=False
        )
        
        gp_freq_mask = gaussian_process_pipeline(
            sensor=model_data.df[sensor_name],
            poa_global=clearsky_cal_val.poa["poa_global"],
            clearsky_mask=frequency_mask,
            time=model_data.df["time"]
        )

        gp_two_medians_mask = gaussian_process_pipeline(
            sensor=model_data.df[sensor_name],
            poa_global=clearsky_cal_val.poa["poa_global"],
            clearsky_mask=two_medians_mask_,
            time=model_data.df["time"]
        )

        plot_sensors_calibrated_directly_to_poa(
            result_df=gp_freq_mask,
            title="Gaussian Process Regression calibration directly to POA with frequency mask",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"direct_calibration_to_poa_by_gaussian_freq_mask_{sensor_name}",
            show=False
        )

        plot_sensors_calibrated_directly_to_poa(
            result_df=gp_two_medians_mask,
            title="Gaussian Process Regression calibration directly to POA with two medians mask",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename),
            filename=f"direct_calibration_to_poa_by_gaussian_two_medians_mask_{sensor_name}",
            show=False
        )


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

    '''
    clearsky_periods_all, cloudy_periods_all = detect_clearsky_periods(
        poa=clearsky_cal_val.poa,
        df=model_data.df,
        sensor_name_ref=model_data.sensor_name_ref,
        save_dir=model_dirs.data_dir,
        filename=model_dirs.filename
    )
    '''

    clearsky_periods_all, cloudy_periods_all = detect_clearsky_periods_v2(
        measured=model_data.df[model_data.sensor_name_ref],
        clearsky=clearsky_cal_val.poa["poa_global"],  # cs["ghi"],
        times=model_data.df["time"],
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
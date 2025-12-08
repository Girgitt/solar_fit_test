import pandas as pd

from pathlib import Path

from pvtools.calibration.calibrate_to_reference.calibrate_linear_regression import (calibrate_by_linear_regression,
                                                                                    calibrate_by_fuzzy_linear_regression)
from pvtools.calibration.calibrate_to_reference.calibrate_divided_regression import (calibrate_by_divided_linear_regression,
                                                                                     calibrate_by_divided_linear_regression_mean)
from pvtools.calibration.calibrate_to_reference.calibrate_polynominal_regression import calibrate_by_polynominal_regression
from pvtools.calibration.calibrate_to_reference.calibrate_decision_tree import calibrate_by_decision_tree_regression
from pvtools.calibration.calibrate_to_reference.calibrate_mlp import calibrate_by_mlp_regression
from pvtools.calibration.calibrate_to_poa.ransac import ransac_pipeline
from pvtools.calibration.calibrate_to_poa.clearsky_utils import (clearsky_detection_by_frequency_method,
                                                                 frequency_analysis, frequency_mask,
                                                                 two_medians_mask, create_derivative_df,
                                                                 create_relative_derivative_mask_df,
                                                                 determine_signal_amplification_scale,
                                                                 compute_gain_factor)
from pvtools.config.params import ModelData, ModelDirectories, ClearSkyCalculatedValues, ModelTimes
from pvtools.visualisation.plotter import (plot_sensors_calibrated_directly_to_poa,
                                           tmp_plot_check_masks, tmp_plot_smoothed_vemls,
                                           tmp_plot_smoothed_derivs_vemls, tmp_plot_evenelope,
                                           tmp_plot_evenelope_scaled,
                                           tmp_plot_scaled_sensor_vs_reference,
                                           plot_universal)


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
        model_dirs: ModelDirectories
) -> None:

    for sensor_name in model_data.sensor_names:

        df_combined = compute_mask_method(
            model_data=model_data,
            clearsky_cal_val=clearsky_cal_val,
            sensor_name=sensor_name
        )

        evenelope, gain_factor = compute_amplifying_signal_envelope_method(
            model_data=model_data,
            clearsky_cal_val=clearsky_cal_val,
            sensor_name=sensor_name
        )

        sensor_gain = pd.Series(
            data=df_combined["sensor"] * gain_factor,
            index=df_combined.index,
            name="sensor_gain"
        )

        evenelope_gain = pd.Series(
            data=evenelope * gain_factor,
            index=evenelope.index,
            name="evenelope_gain"
        )

        sensor_ref = model_data.df[model_data.sensor_name_ref]
        sensor_ref = sensor_ref.reset_index(drop=True)

        df_combined = pd.concat(
            [df_combined, sensor_gain, evenelope, evenelope_gain, sensor_ref],
            axis=1
        )

        dict_series_description = {
            "time": "time",
            "sensor": "sensor",
            "sensor_smooth": "sensor smooth",
            "sensor_d_dt": "sensor derivative",
            "poa_global": "poa global",
            "poa_global_d_dt": "poa global derivative",
            "frequency_mask": "frequency mask",
            "two_medians_mask": "two medainas mask",
            "derivative_mask": "derivative mask",
            "ransac_freq_mask_calibration": "RANSAC frequency mask calibration",
            "ransac_two_medians_mask_calibration": "RANSAC two medainas mask calibration",
            "ransac_derivative_mask_calibration": "RANSAC derivative mask calibration",
            "sensor_gain": "sensor scaled by evenelope factor",
            "evenelope": "evenelope",
            "evenelope_gain": "evenelope scaled",
            f"{model_data.sensor_name_ref}": "sensor reference",
        }

        #FIXME - move all plots to the separate function. Pass subdirectory as an input parameter
        # to declare it in one place!

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor", "sensor_smooth", "poa_global"],
            dict_series_description=dict_series_description,
            mask=None,
            title="Check smoothness in universal function",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"smoothness_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor_d_dt", "poa_global_d_dt"],
            dict_series_description=dict_series_description,
            mask=None,
            title="Check derivative in universal function",
            ylabel="-",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"derivative_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor", "sensor_smooth", "poa_global"],
            dict_series_description=dict_series_description,
            mask=None,
            title="Check relative smoothness",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"smoothness_relative_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor_d_dt", "poa_global_d_dt"],
            dict_series_description=dict_series_description,
            mask=None,
            title="Check relative derivative",
            ylabel="-",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"derivative_relative_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor", "poa_global"],
            dict_series_description=dict_series_description,
            mask="derivative_mask",
            title="Check relative derivative mask",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"mask_derivative_relative_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor", "evenelope"],
            dict_series_description=dict_series_description,
            mask=None,
            title="Upper envenelope",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"envenelope_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor_gain", "evenelope_gain", "poa_global"],
            dict_series_description=dict_series_description,
            mask=None,
            title="Upper envenelope scaled",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"envenelope_scaled_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=[f"{model_data.sensor_name_ref}", "sensor_gain"],
            dict_series_description=dict_series_description,
            mask=None,
            title="Upper envenelope scaled",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"envenelope_scaled_vs_reference_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor", "poa_global"],
            dict_series_description=dict_series_description,
            mask="frequency_mask",
            title="Frequency mask",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"mask_freq_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor", "poa_global"],
            dict_series_description=dict_series_description,
            mask="two_medians_mask",
            title="Two medinas mask",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"mask_two_medians_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor", "poa_global"],
            dict_series_description=dict_series_description,
            mask="derivative_mask",
            title="Derivative mask",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"mask_deriv_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor", "poa_global", "ransac_freq_mask_calibration"],
            dict_series_description=dict_series_description,
            mask="frequency_mask",
            title="RANSAC calibration directly to POA with frequency mask",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"direct_calibration_to_poa_by_ransac_freq_mask_{sensor_name}",
            show=False
        )

        plot_universal(
            result_df=df_combined,
            data_series_names=["sensor", "poa_global", "ransac_two_medians_mask_calibration"],
            dict_series_description=dict_series_description,
            mask="two_medians_mask",
            title="RANSAC calibration directly to POA with two medians mask",
            ylabel="Irradiance W/m²",
            xlabel="Time",
            save_dir=Path(model_dirs.plot_dir / model_dirs.filename / "test_universal_plot"),
            filename=f"direct_calibration_to_poa_by_ransac_two_medians_mask_{sensor_name}",
            show=False
        )


def compute_mask_method(
        model_data: ModelData,
        clearsky_cal_val: ClearSkyCalculatedValues,
        sensor_name: str
) -> pd.DataFrame:

    df_frequency_mask = frequency_mask(
        sensor=model_data.df[sensor_name],
        poa_global=clearsky_cal_val.poa["poa_global"],
        time=model_data.df["time"],
        sampling_sec=60,
        low_freq_max=0.002,
        window_sec=14400,  # 4hrs
        thershold=0.80
    )

    df_two_medians_mask = two_medians_mask(
        sensor=model_data.df[sensor_name],
        poa_global=clearsky_cal_val.poa["poa_global"],
        time=model_data.df["time"],
        short_window="30min",
        long_window="4h",
        rel_threshold=0.05,
        min_run_length=10
    )

    df_derivative = create_derivative_df(
        sensor=model_data.df[sensor_name],
        poa_global=clearsky_cal_val.poa["poa_global"],
        time=model_data.df["time"],
        window_length=240,
        polyorder=1,
        delta=1.0
    )

    df_relative_derivative_mask = create_relative_derivative_mask_df(
        sensor=model_data.df[sensor_name],
        poa_global=clearsky_cal_val.poa["poa_global"],
        time=model_data.df["time"],
        window_length=240,
        polyorder=1,
        delta=1.0
    )

    ransac_freq_mask = ransac_pipeline(
        sensor=df_frequency_mask["sensor"],
        poa_global=df_frequency_mask["poa_global"],
        clearsky_mask=df_frequency_mask["mask"],
        time=df_frequency_mask["time"]
    )

    ransac_two_medians_mask = ransac_pipeline(
        sensor=df_two_medians_mask["sensor"],
        poa_global=df_two_medians_mask["poa_global"],
        clearsky_mask=df_two_medians_mask["mask"],
        time=df_two_medians_mask["time"]
    )

    ransac_relative_derivative_mask = ransac_pipeline(
        sensor=df_relative_derivative_mask["sensor"],
        poa_global=df_relative_derivative_mask["poa_global"],
        clearsky_mask=df_relative_derivative_mask["mask"],
        time=df_relative_derivative_mask["time"]
    )

    return_dfs_v2 = pd.DataFrame({
        "time": model_data.df["time"],
        "sensor": model_data.df[sensor_name],
        "poa_global": clearsky_cal_val.poa["poa_global"],
        "frequency_mask": df_frequency_mask["mask"],
        "two_medians_mask": df_two_medians_mask["mask"],
        "sensor_smooth": df_derivative["sensor_smooth"],
        "sensor_d_dt": df_derivative["sensor_d_dt"],
        "poa_global_d_dt": df_derivative["poa_global_d_dt"],
        "derivative_mask": df_relative_derivative_mask["mask"],
        "ransac_freq_mask_calibration": ransac_freq_mask["sensor_cal"],
        "ransac_two_medians_mask_calibration": ransac_two_medians_mask["sensor_cal"],
        "ransac_derivative_mask_calibration": ransac_relative_derivative_mask["sensor_cal"],
    })

    return return_dfs_v2


def compute_amplifying_signal_envelope_method(
        model_data: ModelData,
        clearsky_cal_val: ClearSkyCalculatedValues,
        sensor_name: str
) -> tuple[pd.Series, float]:

    evenelope, sensor_smooth = determine_signal_amplification_scale(
        sensor=model_data.df[sensor_name],
        poa_global=clearsky_cal_val.poa["poa_global"],
        time=model_data.df["time"],
        smooth_window=30,
        polyorder=3,
        minimum_disatnce_between_peaks=10,
        smoothing_factor=2000
    )

    evenelope = pd.Series(
        data=evenelope,
        index=model_data.df[sensor_name].index,
        name="evenelope"
    )

    gain_factor = compute_gain_factor(
        envelope=evenelope,
        poa_global=clearsky_cal_val.poa["poa_global"],
        poa_min=20.0,
        env_min=2.0,
        use_median=False,
    )

    return evenelope, gain_factor
import pandas as pd

from pathlib import Path

from pvtools.config.params import ModelData, ModelDirectories, ClearSkyCalculatedValues
from pvtools.visualisation.plotter import plot_from_dataframe, plot_poa_vs_reference, plot_universal


def plot_calibrated_to_reference(
        dataframes: list[pd.DataFrame],
        model_data: ModelData,
        model_dirs: ModelDirectories,
        clearsky_cal_val: ClearSkyCalculatedValues,
        calibration_method: str,
) -> None:
    """
    Plot multiple graphs with data, basic sensors calibrated by reference to POA.

    Graphs include:

    * Original data from basic sensors and reference sensor
    * Calibrated data from basic sensors and reference sensor
    * Reference sensor and POA
    """

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


def plot_calibrated_to_poa(
        df: pd.DataFrame,
        dict_: dict[str, str],
        model_dirs: ModelDirectories,
        model_data: ModelData,
        sensor_name: str
) -> None:
    """
    Plot multiple graphs with data, directly calibrated basic sensors to POA.

    Graphs include:

    * Smoothness
    * Derivatives
    * Relative Derivatives
    * Derivative mask
    * Frequency mask
    * Two medinas mask
    * RANSAC calibration
    """

    direct_calibration_plotting_dir = Path(model_dirs.plot_dir / model_dirs.filename / "direct_calibration_to_poa")

    plot_universal(
        result_df=df,
        data_series_names=["sensor", "sensor_smooth", "poa_global"],
        dict_series_description=dict_,
        mask=None,
        title="Check smoothness",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"smoothness_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor_d_dt", "poa_global_d_dt"],
        dict_series_description=dict_,
        mask=None,
        title="Check derivatives",
        ylabel="-",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"derivative_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor", "sensor_smooth", "poa_global"],
        dict_series_description=dict_,
        mask=None,
        title="Check relative smoothness",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"smoothness_relative_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor_d_dt", "poa_global_d_dt"],
        dict_series_description=dict_,
        mask=None,
        title="Check relative derivative",
        ylabel="-",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"derivative_relative_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor", "poa_global"],
        dict_series_description=dict_,
        mask="derivative_mask",
        title="Check relative derivative mask",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"mask_derivative_relative_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor", "evenelope"],
        dict_series_description=dict_,
        mask=None,
        title="Upper envenelope",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"envenelope_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor_gain", "evenelope_gain", "poa_global"],
        dict_series_description=dict_,
        mask=None,
        title="Upper envenelope scaled",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"envenelope_scaled_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=[f"{model_data.sensor_name_ref}", "sensor_gain"],
        dict_series_description=dict_,
        mask=None,
        title="Upper envenelope scaled",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"envenelope_scaled_vs_reference_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor", "poa_global"],
        dict_series_description=dict_,
        mask="frequency_mask",
        title="Frequency mask",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"mask_freq_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor", "poa_global"],
        dict_series_description=dict_,
        mask="two_medians_mask",
        title="Two medinas mask",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"mask_two_medians_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor", "poa_global"],
        dict_series_description=dict_,
        mask="derivative_mask",
        title="Derivative mask",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"mask_deriv_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor", "poa_global", "ransac_freq_mask_calibration"],
        dict_series_description=dict_,
        mask="frequency_mask",
        title="RANSAC calibration directly to POA with frequency mask",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"direct_calibration_to_poa_by_ransac_freq_mask_{sensor_name}",
        show=False
    )

    plot_universal(
        result_df=df,
        data_series_names=["sensor", "poa_global", "ransac_two_medians_mask_calibration"],
        dict_series_description=dict_,
        mask="two_medians_mask",
        title="RANSAC calibration directly to POA with two medians mask",
        ylabel="Irradiance W/m²",
        xlabel="Time",
        save_dir=direct_calibration_plotting_dir,
        filename=f"direct_calibration_to_poa_by_ransac_two_medians_mask_{sensor_name}",
        show=False
    )
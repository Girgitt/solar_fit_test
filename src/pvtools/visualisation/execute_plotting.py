import pandas as pd

from pvtools.config.params import ModelData, ModelDirectories, ClearSkyParameters, ClearSkyCalculatedValues, ModelTimes
from pvtools.visualisation.plotter import (plot_from_dataframe, plot_poa_vs_reference,
                                           plot_poa_reference_with_clearsky_periods,
                                           plot_sensors_calibrated_directly_to_poa,
                                           plot_clear_sky, plot_poa_components,
                                           tmp_plot_check_masks, tmp_plot_smoothed_vemls,
                                           tmp_plot_smoothed_derivs_vemls, tmp_plot_evenelope,
                                           tmp_plot_evenelope_scaled,
                                           tmp_plot_scaled_sensor_vs_reference)


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
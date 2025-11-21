import pandas as pd

from typing import TypeAlias, Literal

from pvtools.config.params import ModelData, ModelDirectories, ClearSkyParameters, ClearSkyCalculatedValues, ModelTimes
from pvtools.modeling.calculate_calibration_parameters import (linear_regression, divided_linear_regression,
                                                               polynominal_regression, decision_tree_regression,
                                                               mlp_regression)
from pvtools.solar_domain.clearsky import clear_sky, detect_clearsky_periods, detect_clearsky_periods_v2
from pvtools.solar_domain.determine_orientation import determine_system_azimuth_and_tilt
from pvtools.utils.apply_mask_for_dataframe import apply_mask_for_dataframe
from pvtools.solar_domain.measurement_limitations import limit_sensor_ref_irradiance_to_clear_sky_model
from pvtools.visualisation.plotter import (plot_clear_sky, plot_poa_components,
                                           plot_poa_reference_with_clearsky_periods, plot_from_dataframe)
from pvtools.preprocess.preprocess_data import delete_night_period

Period_type: TypeAlias = Literal['sunny', 'cloudy', 'all']

def update_function(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
        clearsky_params: ClearSkyParameters,
        clearsky_cal_val: ClearSkyCalculatedValues,
) -> None:

    df = model_data.df

    plot_from_dataframe(
            df=model_data.df,
            save_dir=model_dirs.plot_dir /  model_dirs.filename,
            filename="sensors_and_reference_vs_time.png",
            sensor_names=model_data.sensor_names,
            sensor_name_ref=model_data.sensor_name_ref,
            show=False,
            title="Sensors and reference vs time"
    )

    df_sunny_cutted_short, df_cloudy_cutted_short = process_solar_data_with_clearsky_detection_and_masking(
        model_data=model_data,
        model_dirs=model_dirs,
        model_times=model_times,
        clearsky_params=clearsky_params,
        clearsky_cal_val=clearsky_cal_val,
    )

    calculate_regression(
        df=df,
        model_data=model_data,
        model_dirs=model_dirs,
        model_times=model_times,
        period="all"
    )

    if not df_sunny_cutted_short.empty:
        calculate_regression(
            df=df_sunny_cutted_short,
            model_data=model_data,
            model_dirs=model_dirs,
            model_times=model_times,
            period="sunny"
        )

    if not df_cloudy_cutted_short.empty:
        calculate_regression(
            df=df_cloudy_cutted_short,
            model_data=model_data,
            model_dirs=model_dirs,
            model_times=model_times,
            period="cloudy"
        )


def process_solar_data_with_clearsky_detection_and_masking(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
        clearsky_params: ClearSkyParameters,
        clearsky_cal_val: ClearSkyCalculatedValues,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    poa, cs = clear_sky(
        clearsky_params=clearsky_params,
        model_dirs=model_dirs,
        model_times=model_times
    )

    poa = delete_night_period(
        df=poa,
        start=model_times.start_daytime_cut,
        end=model_times.end_daytime_cut
    )

    filename = model_dirs.filename
    save_dir_plot = model_dirs.plot_dir / filename

    plot_clear_sky(cs, save_dir=save_dir_plot, show=False)
    plot_poa_components(poa, save_dir=save_dir_plot, show=False)

    clearsky_cal_val.poa = poa

    #FIXME - does it have any sense?
    '''
    df_limited = limit_sensor_ref_irradiance_to_clear_sky_model(
        df=model_data.df,
        clearsky_df=clearsky_cal_val.poa,
        sensor_name_ref=model_data.sensor_name_ref,
        poa_global_name='poa_global',
        save_dir=model_dirs.data_dir,
        filename=model_dirs.filename
    )
    
    model_data.df = df_limited
    '''

    '''
    clearsky_periods_all, cloudy_periods_all = detect_clearsky_periods(
        poa=poa,
        df=model_data.df,
        sensor_name_ref=model_data.sensor_name_ref,
        save_dir=model_dirs.data_dir,
        filename=model_dirs.filename
    )
    '''

    clearsky_periods_all, cloudy_periods_all = detect_clearsky_periods_v2(
        measured=model_data.df[model_data.sensor_name_ref],
        clearsky=poa["poa_global"], #cs["ghi"],
        times=model_data.df["time"],
        sensor_name_ref=model_data.sensor_name_ref,
        save_dir=model_dirs.data_dir,
        filename=model_dirs.filename
    )


    '''
    if clearsky_params.surface_tilt != 0:
        clearsky_params.surface_tilt, clearsky_params.surface_azimuth = determine_system_azimuth_and_tilt(
            model_data=model_data,
            model_times=model_times,
            clearsky_params=clearsky_params,
            sunny_mask=clearsky_periods_all,
        )
    '''

    df_sunny_periods = apply_mask_for_dataframe(
        data_filename=model_dirs.filename,
        sensor_name_ref=model_data.sensor_name_ref,
        period_type="sunny",
        save_dir=model_dirs.data_dir
    )

    df_cloudy_periods = apply_mask_for_dataframe(
        data_filename=model_dirs.filename,
        sensor_name_ref=model_data.sensor_name_ref,
        period_type="cloudy",
        save_dir=model_dirs.data_dir
    )

    plot_poa_reference_with_clearsky_periods(
        poa_global=poa[["time", "poa_global"]],
        sensor_reference=model_data.df[["time", model_data.sensor_name_ref]],
        sunny=clearsky_periods_all,
        save_dir=save_dir_plot,
        show=False
    )

    return df_sunny_periods, df_cloudy_periods


def calculate_regression(
        df: pd.DataFrame,
        model_data: ModelData,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
        period: Period_type
) -> None:

    linear_regression(
        df=df,
        period=period,
        model_data=model_data,
        model_dirs=model_dirs,
    )

    divided_linear_regression(
        df=df,
        period=period,
        model_data=model_data,
        model_dirs=model_dirs,
        model_times=model_times
    )

    polynominal_regression(
        df=df,
        period=period,
        model_data=model_data,
        model_dirs=model_dirs,
    )

    decision_tree_regression(
        df=df,
        period=period,
        model_data=model_data,
        model_dirs=model_dirs,
    )

    mlp_regression(
        df=df,
        period=period,
        model_data=model_data,
        model_dirs=model_dirs,
    )
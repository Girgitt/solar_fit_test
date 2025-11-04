import pandas as pd

from typing import TypeAlias, Literal

from pvtools.config.params import ModelData, ModelDirectories, ClearSkyParameters, ClearSkyCalculatedValues, ModelTimes
from pvtools.modeling.calculate_calibration_parameters import (linear_regression, divided_linear_regression,
                                                               polynominal_regression, decision_tree_regression,
                                                               mlp_regression)
from pvtools.solar_domain.clearsky import clear_sky, detect_clearsky_periods
from pvtools.solar_domain.determine_orientation import determine_system_azimuth_and_tilt
from pvtools.utils.apply_mask_for_dataframe import apply_mask_for_dataframe
from pvtools.solar_domain.measurement_limitations import limit_sensor_ref_irradiance_to_clear_sky_model
from pvtools.visualisation.plotter import plot_clear_sky, plot_poa_components, plot_poa_reference_with_clearsky_periods
from pvtools.preprocess.preprocess_data import delete_night_period

Period_type: TypeAlias = Literal['sunny', 'cloudy']

def update_function(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
        clearsky_params: ClearSkyParameters,
        clearsky_cal_val: ClearSkyCalculatedValues,
) -> None:

    df = model_data.df

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

    calculate_regression(
        df=df_sunny_cutted_short,
        model_data=model_data,
        model_dirs=model_dirs,
        model_times=model_times,
        period="sunny"
    )

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
    df_limited = limit_sensor_ref_irradiance_to_clear_sky_model(
        df=model_data.df,
        clearsky_df=clearsky_cal_val.poa,
        sensor_name_ref=model_data.sensor_name_ref,
        poa_global_name='poa_global',
        save_dir=model_dirs.data_dir,
        filename=model_dirs.filename
    )

    model_data.df = df_limited

    clearsky_periods_all, cloudy_periods_all = detect_clearsky_periods(
        poa=poa,
        df=model_data.df,
        sensor_name_ref=model_data.sensor_name_ref,
        save_dir=model_dirs.data_dir,
        filename=model_dirs.filename
    )

    model_data.df = df_limited

    #FIXME - jesli przechylenie rozne od 0!
    determine_system_azimuth_and_tilt(
        model_data=model_data,
        model_times=model_times,
        clearsky_params=clearsky_params,
        sunny_mask=clearsky_periods_all,
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

    plot_poa_reference_with_clearsky_periods(
        poa_global=poa[["time", "poa_global"]],
        sensor_reference=model_data.df[["time", model_data.sensor_name_ref]],
        sunny=clearsky_periods_all,
        save_dir=save_dir_plot,
        show=False
    )

    return df_sunny_periods_cutted_short, df_cloudy_periods_cutted_short


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
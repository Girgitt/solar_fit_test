import pandas as pd

from typing import TypeAlias, Literal

from pvtools.solar_domain.measurement_limitations import limit_sensors_irradiance_to_clear_sky_model, remove_negative_measurements
from pvtools.io_file.reader import load_and_merge_calibrated_data_from_each_sensor
from pvtools.io_file.writer import save_dataframe_to_csv
from pvtools.config.params import ModelParameters

Period_type: TypeAlias = Literal['sunny', 'cloudy']

def postprocess_data(
        df: pd.DataFrame,
        df_sunny: pd.DataFrame,
        df_cloudy: pd.DataFrame,
        model_parameters: ModelParameters,
        clearsky_df: pd.DataFrame,
        poa_global_name: str = 'poa_global'
) -> pd.DataFrame:

    df = load_and_merge_calibrated_data_from_each_sensor(
        df=df,
        model_parameters=model_parameters,
        period="sunny"
    )
    '''
    df_sunny = load_and_merge_calibrated_data_from_each_sensor(
        df=df_sunny,
        model_parameters=model_parameters,
        period="sunny"
    )

    df_cloudy = load_and_merge_calibrated_data_from_each_sensor(
        df=df_cloudy,
        model_parameters=model_parameters,
        period="cloudy"
    )
    '''

    df = merge_sunny_and_cloudy_calibrated_dataframes(
        df_sunny=df_sunny,
        df_cloudy=df_cloudy,
    )

    limit_df = limit_sensors_irradiance_to_clear_sky_model(
        df=df,
        clearsky_df=clearsky_df,
        sensor_names=model_parameters.sensor_names,
        poa_global_name=poa_global_name
    )

    result_df = remove_negative_measurements(df=limit_df)

    save_dataframe_to_csv(
        df=result_df,
        output_path=model_parameters.data_dir / "filtered" / "calibrated" / model_parameters.filename / f"{model_parameters.args.calibration}.csv",
        index=False,
        index_label=None,
    )

    return result_df

def merge_sunny_and_cloudy_calibrated_dataframes(
        df_sunny: pd.DataFrame,
        df_cloudy: pd.DataFrame
) -> pd.DataFrame:
    df_combined = pd.concat([df_sunny, df_cloudy])

    df_combined = df_combined.sort_values('time').drop_duplicates('time')

    return df_combined
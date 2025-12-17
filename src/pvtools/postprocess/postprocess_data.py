from pathlib import Path

import pandas as pd

from typing import TypeAlias, Literal

from pvtools.solar_domain.measurement_limitations import limit_sensors_irradiance_to_clear_sky_model, remove_negative_measurements
from pvtools.io_file.writer import save_dataframe_to_csv
from pvtools.config.params import ModelData, ModelDirectories

Period_type: TypeAlias = Literal['sunny', 'cloudy']

def postprocess_data(
        df: pd.DataFrame,
        sensor_names: list[str],
        data_dir: Path,
        filename: str,
        clearsky_df: pd.DataFrame,
        poa_global_name: str = 'poa_global',
        calibration_method: str = "linear"
) -> pd.DataFrame:

    '''
    limit_df = limit_sensors_irradiance_to_clear_sky_model(
        df=df,
        clearsky_df=clearsky_df,
        sensor_names=sensor_names,
        poa_global_name=poa_global_name
    )

    result_df = remove_negative_measurements(df=limit_df)
    '''

    save_dataframe_to_csv(
        df=df,
        output_path= Path(data_dir) / "filtered" / "calibrated" / filename / f"{calibration_method}.csv",
        index=False,
        index_label=None,
    )

    return df

def merge_sunny_and_cloudy_calibrated_dataframes(
        df_sunny: pd.DataFrame,
        df_cloudy: pd.DataFrame
) -> pd.DataFrame:

    """
    Merge calibrated sunny and cloudy datasets while preserving chronological order.
    """

    df_combined = pd.concat([df_sunny, df_cloudy])

    df_combined = df_combined.sort_values('time').drop_duplicates('time')

    return df_combined
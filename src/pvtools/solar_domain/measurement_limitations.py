import numpy as np
import pandas as pd

from pathlib import Path

from pvtools.io_file.writer import save_dataframe_to_csv

def limit_measured_irradiance_to_clear_sky_model(
        df: pd.DataFrame,
        clearsky_df: pd.DataFrame,
        sensor_name_ref: str = 'irr_dav_1',
        poa_global_name: str = 'poa_global',
        save_dir: Path = None,
        filename: str = None
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or not isinstance(clearsky_df, pd.DataFrame):
        raise TypeError("Expected 'df' and 'clear_sky_df' to be a pandas DataFrame")

    if 'time' not in df.columns or 'time' not in clearsky_df.columns:
        raise ValueError("'time' column needs to be provided!")

    mismatched_times = set(df['time']) - set(clearsky_df['time'])
    if mismatched_times:
        raise ValueError("Timestamps are mismatched!")

    df = df.copy()

    merged = (df[['time', sensor_name_ref]].merge(clearsky_df[['time', poa_global_name]], on='time', how='inner'))

    limited_df = merged.copy()
    limited_df[sensor_name_ref] = limited_df[sensor_name_ref].clip(upper=limited_df[poa_global_name])

    df[sensor_name_ref] = limited_df[sensor_name_ref]

    if save_dir is not None:
        save_dir = Path(save_dir)
        output_path = save_dir / "filtered" / f"{filename}.csv"
        save_dataframe_to_csv(df, output_path, index=False)

    return df

def remove_negative_measurements(
        df: pd.DataFrame,
        save_dir: Path = None,
        filename: str = None
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected 'df' to be a pandas DataFrame")

    original_df = df.copy()
    df = df.copy()

    for col in df.columns:
        if col == 'time':
            continue
        df[col] = df[col].clip(lower=0)

    if_changed = df.equals(original_df)

    if save_dir is not None and not if_changed:
        save_dir = Path(save_dir)
        output_path = save_dir / "filtered" / f"{filename}.csv"
        save_dataframe_to_csv(df, output_path, index=False)

    return df
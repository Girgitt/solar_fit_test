import pandas as pd
import numpy as np
import re

from datetime import time
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from pvtools.io_file.writer import save_dataframe_to_csv

def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaled_array = scaler.fit_transform(df[numeric_cols])
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaled_array

    return df_scaled

def sanitize_filename(name: str) -> str:
    name = name.split("@")[-1]
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def preprocess_data(
        df: pd.DataFrame,
        save_dir: Path = None,
        target_timedelta: str = '1min'
) -> pd.DataFrame:
    df = df.copy()

    df = ensure_datetime(df=df)
    ensure_target_frequency_is_lower_than_measurements(
        df=df,
        target_timedelta=target_timedelta
    )

    df_filtered = delete_night_period(
        df=df,
        start=time(3,0),
        end=time(18,0)
    )

    df_avereged = average_measurements(
        df=df_filtered,
        target_timedelta=target_timedelta
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        output_path = save_dir.parent / "filtered" / (save_dir.stem + save_dir.suffix)
        print(f"AAA: {output_path}")
        save_dataframe_to_csv(df_avereged, output_path, index=False)

    return df_avereged

def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df['time'].dtype):
        df['time'] = pd.to_datetime(df['time'])

    return df

def ensure_target_frequency_is_lower_than_measurements(
        df: pd.DataFrame,
        target_timedelta: str = '1min'
) -> None:
    if len(df.index) >= 2:
        actual_timedelta = df['time'][1] - df['time'][0] # all data has same timedelta
    else:
        raise ValueError(f"Not enough samples!")

    if actual_timedelta:
        actual_freq = pd.to_timedelta(actual_timedelta)
        target_freq = pd.to_timedelta(target_timedelta)

        if actual_freq > target_freq:
            print(f"[INFO] Data has already less frequent measurements: {actual_freq.total_seconds()}s > {target_freq.total_seconds()}s. Nothing to do.")
            return

def delete_night_period(
        df: pd.DataFrame,
        start: time = time(3,0), # 3:00 GMT -> 5:00 UTC+2
        end: time = time(18,0), # 18:00 GMT -> 20:00 UTC+2
) -> pd.DataFrame:
    df_time_only = df['time'].dt.tz_convert(None).dt.time

    mask = df_time_only.between(start, end)
    df_filtered = df[mask]

    return df_filtered

def average_measurements(
        df: pd.DataFrame,
        target_timedelta: str = '1min',
) -> pd.DataFrame:
    df.set_index('time', inplace=True)
    df_resampled = df.resample(target_timedelta).mean()
    df_resampled = df_resampled.reset_index()

    return df_resampled


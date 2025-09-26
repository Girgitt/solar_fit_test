import pandas as pd
import numpy as np
import re

from datetime import time
from zoneinfo import ZoneInfo
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from pvtools.io_file.writer import save_dataframe_to_csv

def preprocess_data(
        df: pd.DataFrame,
        target_timedelta: str = '1min',
        save_dir: Path = None
) -> pd.DataFrame:
    df = df.copy()

    df = ensure_dataframe_contains_valid_data(df=df)
    df = ensure_datetime_contains_timezone(
        df=df,
        tz_name='Europe/Warsaw'
    )

    df_filtered = delete_night_period(
        df=df,
        start=time(3, 0),  # 3:00 GMT -> 5:00 UTC+2
        end=time(18, 0)  # 18:00 GMT -> 20:00 UTC+2
    )

    if check_if_target_frequency_is_lower_than_measurements(df=df, target_timedelta=target_timedelta) is False:
        df_avereged = average_measurements(
            df=df_filtered,
            target_timedelta=target_timedelta
        )
    else:
        df_avereged = df_filtered

    if save_dir is not None:
        save_dir = Path(save_dir)
        output_path = save_dir.parent / "filtered" / (save_dir.stem + save_dir.suffix)
        save_dataframe_to_csv(df_avereged, output_path, index=False)

    return df_avereged

def ensure_dataframe_contains_valid_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    cleaned_df = df.dropna(how="any").reset_index(drop=True)

    return cleaned_df

def ensure_datetime_contains_timezone(
        df: pd.DataFrame,
        tz_name: str = 'Europe/Warsaw',
        save_dir: Path = None,
) -> pd.DataFrame:
    df = df.copy()

    if "time" not in df.columns:
        raise KeyError("DataFrame must contain a 'time' column")

    tzinfo = ZoneInfo(tz_name)
    col = df["time"]

    if pd.api.types.is_numeric_dtype(col):
        df["time"] = pd.to_datetime(col, unit="s", utc=True).dt.tz_convert(tzinfo)

    elif pd.api.types.is_datetime64_any_dtype(col):
        if col.dt.tz is None:
            df["time"] = col.dt.tz_localize(tzinfo)
        else:
            df["time"] = col
    else:
        try:
            int_secs = col.astype("int64")
            df["time"] = pd.to_datetime(int_secs, unit="s", utc=True).dt.tz_convert(tzinfo)
        except Exception as e:
            raise TypeError(f"Unsupported 'time' format, cannot convert: {e}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        output_path = save_dir.parent / f"{save_dir.stem+save_dir.suffix}_a"
        save_dataframe_to_csv(df, output_path, index=False)

    return df

def delete_night_period(
        df: pd.DataFrame,
        start: time = time(3,0), # 3:00 GMT -> 5:00 UTC+2
        end: time = time(18,0), # 18:00 GMT -> 20:00 UTC+2
) -> pd.DataFrame:
    df_time_only = df['time'].dt.tz_convert(None).dt.time

    mask = df_time_only.between(start, end)
    df_filtered = df[mask]

    df_filtered = df_filtered.set_index("time").resample("1min").mean()
    df_filtered = df_filtered.dropna(how="all").reset_index()

    return df_filtered

def check_if_target_frequency_is_lower_than_measurements(
        df: pd.DataFrame,
        target_timedelta: str = '1min'
) -> bool:
    if len(df.index) >= 2:
        measured_timedelta = df['time'][1] - df['time'][0] # all data has same timedelta
    else:
        raise ValueError(f"Not enough samples!")

    if measured_timedelta:
        measured_freq = pd.to_timedelta(measured_timedelta)
        target_freq = pd.to_timedelta(target_timedelta)

        if measured_freq > target_freq:
            print(f"[INFO] Data has already less frequent measurements: {measured_freq.total_seconds()}s > {target_freq.total_seconds()}s. Nothing to do.")
            return True

    return False

def average_measurements(
        df: pd.DataFrame,
        target_timedelta: str = '1min',
) -> pd.DataFrame:
    df.set_index('time', inplace=True)
    df_resampled = df.resample(target_timedelta).mean()
    df_resampled = df_resampled.reset_index()

    return df_resampled

def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected 'df' to be a pandas DataFrame")

    df = df.copy()

    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaled_array = scaler.fit_transform(df[numeric_cols])
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaled_array

    return df_scaled

def sanitize_filename(name: str) -> str:
    name = name.split("@")[-1]
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)


import math
import numpy as np
import pandas as pd


from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo

def argument_parsing(parser: ArgumentParser) -> Namespace:
    parser.add_argument("--action", choices=["update", "execute"], required=True,
                        help="Specify whether to 'update' (train/save) or 'execute' (load/apply) the model")
    parser.add_argument("--model_id", default="default",
                        help="Model identifier used for saving/loading coefficients")
    parser.add_argument("--csv", required=True,
                        help="Path to CSV file with input data")

    return parser.parse_args()

def print_available_data_columns(data_columns: List[str]) -> None:
    print("Available data columns:")
    for i, col in enumerate(data_columns):
        print(f"{i}: {col}")

def select_available_data_columns_to_process(
        data_columns: List[str],
        df: pd.DataFrame
) -> Tuple[List[str], str, pd.DataFrame]:
    sensor_names = [
        data_columns[0],
        data_columns[1],
        data_columns[2],
    ]
    sensor_name_ref = data_columns[3]
    df = df.dropna(subset=sensor_names + [sensor_name_ref])

    return sensor_names, sensor_name_ref, df

def check_if_csv_contains_timezone_info__v1(file_path: str) -> None:
    file_path = Path(file_path)

    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])

    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize(ZoneInfo('Europe/Warsaw'))
        df.to_csv(file_path, index=False)
        print('Time does not contain timezone info. Timezone added and file updated')
    else:
        print('[INFO] Time already has timezone. No changes made')

def check_if_csv_contains_timezone_info(file_path: str, tz_name: str = "Europe/Warsaw") -> None:
    """
    Reads a CSV with a 'time' column and ensures it has timezone info.
    - If 'time' is numeric epoch (s/ms/us/ns), interpret as UTC and convert to tz_name.
    - If 'time' parses to naive datetimes, tz-localize to tz_name (treating them as local times).
    - If 'time' already has tz info, leave it unchanged.

    Notes:
    * Mixed formats in one column are not supported; majority-numeric is treated as epoch.
    * Epoch detection uses value magnitude to infer unit.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    if "time" not in df.columns:
        raise KeyError("CSV must contain a 'time' column.")

    tz = ZoneInfo(tz_name)
    col = df["time"]

    # --- Helper: decide if column is (mostly) numeric epoch and infer unit
    num = pd.to_numeric(col, errors="coerce")
    numeric_ratio = num.notna().mean() if len(num) else 0.0

    def infer_epoch_unit(values: pd.Series) -> str:
        """Infer epoch unit from magnitude (median of non-NaN absolute values)."""
        v = values.dropna().abs()
        if v.empty:
            # Default to seconds if empty after dropna (shouldn't happen if we call it correctly)
            return "s"
        med = float(np.median(v))
        # Rough thresholds for modern timestamps
        if med >= 1e17:
            return "ns"
        elif med >= 1e14:
            return "us"
        elif med >= 1e11:
            return "ms"
        else:
            # Covers typical seconds-since-epoch (~1e9 today), and also older dates
            return "s"

    # Treat as epoch if the column is mostly numeric (>= 90%)
    if numeric_ratio >= 0.90:
        unit = infer_epoch_unit(num)
        # Parse as UTC instants, then convert to local tz
        dt = pd.to_datetime(num, unit=unit, utc=True)
        df["time"] = dt.dt.tz_convert(tz)
        df.to_csv(file_path, index=False)
        print(f"[INFO] Detected epoch timestamps (~{unit}). Parsed as UTC and converted to {tz_name}. File updated.")
        return df

    # Otherwise: parse as datetimes (strings)
    dt = pd.to_datetime(col, errors="coerce", utc=False)

    if dt.isna().all():
        raise ValueError('Could not parse any datetimes from the "time" column.')

    # If already timezone-aware, keep as-is
    if getattr(dt.dt, "tz", None) is not None:
        print("[INFO] Time already has timezone. No changes made")
        return df

    # Naive datetimes: interpret as local time in tz_name
    df["time"] = dt.dt.tz_localize(tz)
    df.to_csv(file_path, index=False)
    print("Time does not contain timezone info. Timezone added and file updated")

    return df

def solar_elevation(
        lat: float,
        lon: float,
        tz_offset: int,
        dt_local: datetime
) -> float:
    n = dt_local.timetuple().tm_yday
    lt = dt_local.hour + dt_local.minute / 60 + dt_local.second / 3600  # local clock time
    B = math.radians((360 / 365) * (n - 81))
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)  # Eq. of Time [min]
    lstm = 15 * tz_offset
    tc = 4 * (lon - lstm) + eot  # Time-corr [min]
    lst = lt + tc / 60  # Local solar time
    omega = math.radians(15 * (lst - 12))  # Hour angle
    delta = math.radians(23.45 * math.sin(math.radians(360 * (284 + n) / 365)))
    phi = math.radians(lat)
    cos_z = (math.sin(phi) * math.sin(delta) +
             math.cos(phi) * math.cos(delta) * math.cos(omega))
    z = math.acos(max(-1, min(1, cos_z)))  # clamp

    return math.degrees(math.pi / 2 - z)


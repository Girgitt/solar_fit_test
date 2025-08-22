import pandas as pd
import numpy as np
from typing import Optional, List
from pathlib import Path
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from save_functions import save_dataframe_to_csv
from calibrate_methods import sanitize_filename

def cut_reference_values_by_poa_global(
        poa_global: pd.Series,
        sensor_reference: pd.Series,
        save_dir: Optional[Path] = None,
) -> None:
    pass

def determine_curve_interpolating_maxima(
        df: pd.DataFrame,
        sensor_names: List[str] = None,
        sensor_name_ref: str = None,
        save_dir: Optional[Path] = None,
) -> None:
    save_dir = Path(save_dir)
    df = df.copy()

    if 'time in df.columns':
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.set_index('time').sort_index()

    df_interp = df.copy()

    peak_rows = []

    if sensor_names is None:
        raise(ValueError("Parameter 'sensor_names' must be a list of column names."))

    sensor_names.append(sensor_name_ref)

    for idx, sensor_col in enumerate(sensor_names):
        peaks, props = find_peaks(df[sensor_col].to_numpy())
        peak_times = df.index[peaks]
        peak_values = df[sensor_col].iloc[peaks].to_numpy()

        for i, (ts, val) in enumerate(zip(peak_times, peak_values)):
            peak_rows.append({
                "time": ts,
                "value": val,
                "index": int(peaks[i])
            })

        df_peaks = pd.DataFrame(peak_rows).sort_values(["time"])
        s_name = sanitize_filename(sensor_col)
        output_path_peaks = save_dir.parent / "interpolated" / save_dir.stem / (s_name + "_peaks" + save_dir.suffix)
        save_dataframe_to_csv(df=df_peaks, output_path=output_path_peaks)

        interpolation_function = interp1d(peak_times.astype(np.int64), peak_values, kind='linear', bounds_error=False, fill_value="extrapolate")
        df_interp[sensor_col] = interpolation_function(df[sensor_col].index.astype(np.int64))

    output_path_interp = save_dir.parent / "interpolated" / save_dir.stem / (save_dir.stem + "_interp" + save_dir.suffix)
    save_dataframe_to_csv(df=df_interp, output_path=output_path_interp)
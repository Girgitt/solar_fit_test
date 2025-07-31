import numpy as np
import pandas as pd

from dataclasses import dataclass
from argparse import Namespace
from pathlib import Path

@dataclass
class ModelParameters:
    df: pd.DataFrame
    df_time: pd.DataFrame
    args: Namespace
    log_dir: Path
    data_filename_dir: Path
    plot_dir: str
    sensor_names: np.ndarray
    sensor_name_ref: np.ndarray
    min_r2: float
    max_mae: float
    tolerance_min: float
    tolerance_max: float
    blend_minutes: int
    timedelta: int
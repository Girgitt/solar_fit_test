import numpy as np
import pandas as pd

from dataclasses import dataclass
from argparse import Namespace

@dataclass
class ModelParameters:
    df: pd.DataFrame
    args: Namespace
    log_dir: str
    plot_dir: str
    sensor_values: np.ndarray
    sensor_value_ref: np.ndarray
    min_r2: float
    max_mae: float
    tolerance_min: float
    tolerance_max: float
    blend_minutes: int
    timedelta: int
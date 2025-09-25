import numpy as np
import pandas as pd

from dataclasses import dataclass
from argparse import Namespace
from pathlib import Path
from pvlib.location import Location
from typing import TypedDict, List

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

@dataclass
class ClearSkyParameters:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    warsaw_lat: float
    warsaw_lon: float
    tz: str
    altitude: int
    name: str
    frequency: str
    albedo: float
    surface_tilt: int  # degrees from horizontal
    surface_azimuth: int # 180 - south facing

@dataclass
class ClearSkyCalculatedValues:
    poa: pd.DataFrame
    clearsky_periods: pd.DataFrame

@dataclass
class SolarDataForLocationAndTime:
    location: Location
    times: pd.DatetimeIndex
    solar_position: pd.DataFrame
    clear_sky: pd.DataFrame

class DatatypeCoefficientsForDividedLinearRegression(TypedDict):
    hour: str
    a: float
    b: float

class DatatypeCoefficientsForMLPRegression(TypedDict):
    layer_1_weights: List[List[float]]
    layer_1_biases: List[float]
    layer_2_weights: List[List[float]]
    layer_2_biases: List[float]
    output_weights: List[List[float]]
    output_biases: List[float]

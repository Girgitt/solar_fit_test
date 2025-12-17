import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from datetime import time
from typing import TypedDict

from pandas import Timedelta


class ModelData:
    df: pd.DataFrame
    df_time: pd.Series
    sensor_names: list[str]
    sensor_name_ref: str


class ModelDirectories:
    project_dir: Path
    log_dir: Path
    data_dir: Path
    plot_dir: Path
    filename: str
    load_metrics_dir: Path


class ModelTimes:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    frequency: Timedelta
    divided_linear_regression_interval: str
    start_daytime_cut: time
    end_daytime_cut: time


class ClearSkyParameters:
    warsaw_lat: float
    warsaw_lon: float
    tz: str
    altitude: int
    name: str
    albedo: float
    surface_tilt: int  # degrees from horizontal
    surface_azimuth: int # 180 - south facing


class ClearSkyCalculatedValues:
    poa: pd.DataFrame
    clearsky_periods: pd.Series
    cloudy_periods: pd.Series


class DatatypeCoefficientsForDividedLinearRegression(TypedDict):
    hour: pd.Timestamp
    a: float
    b: float


class DatatypeScalersForMLPRegression(TypedDict):
    x_scaler_mean: list[float]
    x_scaler_scale: list[float]
    y_scaler_mean: list[float]
    y_scaler_scale: list[float]
    activation: str


class DatatypeCoefficientsForMLPRegression(TypedDict):
    layer_1_weights: list[list[float]]
    layer_1_biases: list[float]
    layer_2_weights: list[list[float]]
    layer_2_biases: list[float]
    output_weights: list[list[float]]
    output_biases: list[float]


class DatatypeMLPRegressionParameters(TypedDict):
    coefficients: DatatypeCoefficientsForMLPRegression
    scalers: DatatypeScalersForMLPRegression

import os
import math
import pandas as pd

from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import List, Tuple
from pathlib import Path

from pvtools.config.params import ModelParameters
from pvtools.io_file.reader import load_dataframe_from_csv
from pvtools.preprocess.preprocess_data import sanitize_filename


def argument_parsing(parser: ArgumentParser) -> Namespace:
    parser.add_argument("--action", choices=["update", "execute"], required=True,
                        help="Specify whether to 'update' (train/save) or 'execute' (load/apply) the model")
    parser.add_argument("--model_id", default="default",
                        help="Model identifier used for saving/loading coefficients")
    parser.add_argument("--csv", required=True,
                        help="Path to CSV file with input data")
    parser.add_argument("--calibration",
                        choices=["linear", "divided_linear", "decision_tree", "poly", "mlp"],
                        default="linear",
                        help="Defines which calibration method use to calibrate sensors")
    parser.add_argument("--sensors", type=int, nargs="+",required=True,
                        help="List of sensors to calibrate."
                             "Number of specified column, counting from 0, skipping time column."
                             "Accept multiple numbers separated by space.")
    parser.add_argument("--reference", type=int, required=True,
                        help="Number of reference sensors"
                             "Number of specified column, counting from 0, skipping time column."
                             "Accept single number.")

    parser.add_argument("--data_dir",
                        help="force specific data directory to store logs, plots etc. <current working dir>")

    return parser.parse_args()


def print_available_data_columns(data_columns: List[str]) -> None:
    print("Available data columns:")
    for i, col in enumerate(data_columns):
        print(f"{i}: {col}")


def select_available_data_columns_to_process(
        data_columns: List[str],
        df: pd.DataFrame,
        sensors_chosen: List[int],
        sensor_ref_chosen: int
) -> Tuple[List[str], str, pd.DataFrame]:
    n = len(data_columns)

    if not sensors_chosen:
        raise ValueError("[ERROR] 'sensors_chosen' cannot be empty")
    if not isinstance(sensor_ref_chosen, int):
        raise TypeError("[ERROR] 'sensor_ref_chosen' must be an int")

    sensors_chosen = list(dict.fromkeys(sensors_chosen))

    bad_sensors = [i for i in sensors_chosen if not isinstance(i, int) or i < 0 or i >= n]
    if bad_sensors:
        raise IndexError(f"[ERROR] Sensor index out of range: {bad_sensors}; there are {n} columns")
    if sensor_ref_chosen < 0 or sensor_ref_chosen >= n:
        raise IndexError(f"[ERROR] Reference index out of range: {sensor_ref_chosen}; there are {n} columns")

    if sensor_ref_chosen in sensors_chosen:
        raise ValueError(f"[ERROR] Sensors and reference overlap at index: {sensor_ref_chosen}")

    sensor_names = [data_columns[i] for i in sensors_chosen]
    sensor_name_ref = data_columns[sensor_ref_chosen]

    df_out = df.dropna(subset=sensor_names + [sensor_name_ref])

    return sensor_names, sensor_name_ref, df_out


def load_filtered_and_calculated_data_needed_for_execute_function(
        model_parameters: ModelParameters
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_dataframe_from_csv(
        Path(model_parameters.data_dir / "filtered" / f"{model_parameters.filename}.csv"))

    poa = load_dataframe_from_csv(
        Path(model_parameters.data_dir / "calculated_data" / model_parameters.filename / "poa_values.csv"))

    clearsky_periods = load_dataframe_from_csv(
        Path(model_parameters.data_dir / "calculated_data" / model_parameters.filename /
             f"{sanitize_filename(model_parameters.sensor_name_ref)}_sunny_periods.csv"))

    return df, poa, clearsky_periods


def initialize_dirs_for_base_dir(data_dir_path):
    LOG_DIR = Path(os.path.join(data_dir_path, "logs"))
    PLOT_DIR = Path(os.path.join(data_dir_path, "plots"))
    DATA_DIR = Path(os.path.join(data_dir_path, "data"))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    return LOG_DIR, PLOT_DIR, DATA_DIR

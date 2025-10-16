import os
import pandas as pd

from argparse import ArgumentParser, Namespace
from typing import List, Tuple
from pathlib import Path
from pandas import DataFrame

from pvtools.config.params import ModelParameters
from pvtools.io_file.reader import load_dataframe_from_csv, load_and_merge_calibrated_data_from_each_sensor
from pvtools.io_file.writer import save_dataframe_to_csv
from pvtools.preprocess.preprocess_data import sanitize_filename


def argument_parsing(parser: ArgumentParser) -> Namespace:
    parser.add_argument("--action", choices=["update", "execute"], required=True,
                        help="Specify whether to 'update' (train/save) or 'execute' (load/apply) the model")
    parser.add_argument("--model_id", default="default",
                        help="Model identifier used for saving/loading coefficients")
    parser.add_argument("--csv", required=True,
                        help="Path to CSV file with input data")
    parser.add_argument("--calibration",
                        choices=["linear", "fuzzy", "divided_linear", "decision_tree", "poly", "mlp"],
                        default="linear",
                        help="Defines which calibration method use to calibrate sensors")
    parser.add_argument("--sensors", type=int, nargs="+",required=True,
                        help="List of sensors to calibrate."
                             " Number of specified column, counting from 0, skipping time column"
                             " Accept multiple numbers separated by space")
    parser.add_argument("--reference", type=int, required=True,
                        help="Number of reference sensors."
                             " Number of specified column, counting from 0, skipping time column."
                             " Accept single number")
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
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    df = load_dataframe_from_csv(
        Path(model_parameters.data_dir / "filtered" / f"{model_parameters.filename}.csv"))

    df_sunny = load_dataframe_from_csv(
        Path(model_parameters.data_dir / "filtered" / "sunny_periods" / f"{model_parameters.filename}_all.csv"))

    df_cloudy = load_dataframe_from_csv(
        Path(model_parameters.data_dir / "filtered" / "cloudy_periods" / f"{model_parameters.filename}_all.csv"))

    poa = load_dataframe_from_csv(
        Path(model_parameters.data_dir / "calculated_data" / model_parameters.filename / "poa_values.csv"))

    sensor_name = sanitize_filename(model_parameters.sensor_name_ref)

    clearsky_periods = load_dataframe_from_csv(
        Path(model_parameters.data_dir / "calculated_data" / model_parameters.filename /
             f"{sensor_name}_sunny_periods_all.csv"))

    cloudy_periods = load_dataframe_from_csv(
        Path(model_parameters.data_dir / "calculated_data" / model_parameters.filename /
             f"{sensor_name}_cloudy_periods_all.csv"))

    df_sunny['if_sunny'] = True
    df_cloudy['if_sunny'] = False

    return df, df_sunny, df_cloudy, poa, clearsky_periods, cloudy_periods


def initialize_dirs_for_base_dir(data_dir_path):
    log_dir = Path(os.path.join(data_dir_path, "logs"))
    plot_dir = Path(os.path.join(data_dir_path, "plots"))
    data_dir = Path(os.path.join(data_dir_path, "data"))

    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return log_dir, plot_dir, data_dir

def create_calibrated_dataframe(
        df: pd.DataFrame,
        model_parameters: ModelParameters
) -> pd.DataFrame:

    df = load_and_merge_calibrated_data_from_each_sensor(
        df=df,
        model_parameters=model_parameters
    )

    save_dir = Path(model_parameters.data_dir / "filtered" / "calibrated" / model_parameters.filename /
                    f"{model_parameters.args.calibration}.csv"
                    )

    save_dataframe_to_csv(
        df=df,
        output_path=save_dir
    )

    return df



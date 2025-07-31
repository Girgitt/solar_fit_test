import math
from argparse import ArgumentParser
from datetime import datetime
from typing import Tuple

from calibrate_methods import *
from analyze_calibration import *

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

def update_function(model_parameters: ModelParameters) -> None:
    linear_regression(
        df=model_parameters.df,
        log_dir = model_parameters.log_dir,
        data_filename_dir = model_parameters.data_filename_dir,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    divided_linear_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename_dir=model_parameters.data_filename_dir,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    polynominal_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename_dir=model_parameters.data_filename_dir,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    decision_tree_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename_dir=model_parameters.data_filename_dir,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

    mlp_regression(
        df=model_parameters.df,
        log_dir=model_parameters.log_dir,
        data_filename_dir=model_parameters.data_filename_dir,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
    )

def execute_function(model_parameters: ModelParameters) -> None:
    calibrate_by_linear_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=Path(model_parameters.args.csv).stem
    )

    calibrate_by_divided_linear_regression(
        df=model_parameters.df,
        df_time=model_parameters.df_time,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=Path(model_parameters.args.csv).stem
    )

    calibrate_by_polynominal_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=Path(model_parameters.args.csv).stem
    )

    calibrate_by_decision_tree_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=Path(model_parameters.args.csv).stem
    )

    calibrate_by_mlp_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=Path(model_parameters.args.csv).stem
    )

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


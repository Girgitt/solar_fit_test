import os
import pandas as pd

from argparse import ArgumentParser, Namespace
from typing import Tuple, Optional
from pathlib import Path
from pandas import DataFrame

from pvtools.config.params import ModelParameters, ModelDirectories
from pvtools.io_file.reader import load_dataframe_from_csv, load_and_merge_calibrated_data_from_each_sensor
from pvtools.io_file.writer import save_dataframe_to_csv
from pvtools.preprocess.preprocess_data import sanitize_filename


def argument_parsing(parser: ArgumentParser) -> Namespace:
    parser.add_argument("--action",
                        choices=["update", "execute"],
                        required=True,
                        help="Specify whether to 'update' (train/save) or 'execute' (load/apply) the model")

    parser.add_argument("--model_id",
                        default="default",
                        help="Model identifier used for saving/loading coefficients")

    parser.add_argument("--csv",
                        required=True,
                        type=Path,
                        help="Path to CSV file with input data")

    parser.add_argument("--calibration",
                        choices=["linear", "fuzzy", "divided_linear", "decision_tree", "poly", "mlp"],
                        default="linear",
                        help="Defines which calibration method use to calibrate sensors")

    parser.add_argument("--sensors",
                        type=int,
                        nargs="+",
                        required=True,
                        help="List of sensors to calibrate."
                             " Number of specified column, counting from 0, skipping time column"
                             " Accept multiple numbers separated by space")

    parser.add_argument("--reference",
                        type=int,
                        default=None,
                        required=False,
                        help="Number of reference sensors."
                             " Number of specified column, counting from 0, skipping time column."
                             " Accept single number")

    parser.add_argument("--project_dir",
                        type=Path,
                        default=Path.cwd(),
                        help="Force specific data directory to store logs, plots etc. "
                             "(default: current working directory)")

    parser.add_argument("--calibration_metrics_dir",
                        type=Path,
                        default=Path.cwd(),
                        help="Directory which contains all metrics needed for calibration. "
                             "(default: current working directory)")

    parser.add_argument("--start_time_hour",
                      type=int,
                      default=4,
                      help="Start time (hour) for filter only day time period (GMT)")

    parser.add_argument("--start_time_minute",
                      type=int,
                      default=0,
                      help="Start time (minute) for filter only day time period")

    parser.add_argument("--end_time_hour",
                      type=int,
                      default=17,
                      help="Start time (hour) for filter only day time period (GMT)")

    parser.add_argument("--end_time_minute",
                      type=int,
                      default=0,
                      help="Start time (minute) for filter only day time period")

    parser.add_argument("--latitude",
                      type=float,
                      default=52.22977,
                      required=True,
                      help="Decimal latitude coordinates of measurement station (default Warsaw)")

    parser.add_argument("--longtitude",
                      type=float,
                      default=21.01178,
                      required=True,
                      help="Decimal longtitude coordinates of measurement station (default Warsaw)")

    parser.add_argument("--timezone",
                      type=str,
                      default="Europe/Warsaw",
                      required=True,
                      help="Time zone of measurement station (default Europe/Warsaw)."
                           " Check 'pytz.all_timezones' for all available options.")

    parser.add_argument("--altitude",
                      type=int,
                      default=100,
                      required=True,
                      help="Altitude of measurement station in meters")

    parser.add_argument("--name",
                      type=str,
                      default="Warsaw",
                      help="Name for measurement station")

    parser.add_argument("--frequency",
                      type=str,
                      default="1min",
                      required=True,
                      help="Target timestamps for filterenig dataset."
                           " Available formats: 'xs' 'xmin' 'xh' 'xms' where x is a number")

    parser.add_argument("--albedo",
                      type=float,
                      default=0.25,
                      required=True,
                      help="Ratio of reflected solar irradiance to global horizontal irradiance (unitless)."
                           " Default 0.25")

    parser.add_argument("--surface_tilt",
                      type=int,
                      default=0,
                      required=True,
                      help="Surface tilt of the sensor in degrees (default 0, horizontal)")

    parser.add_argument("--surface_azimuth",
                      type=int,
                      default=0,
                      required=True,
                      help="Surface azimuth of the sensor in degrees (default 180, south)")

    return parser.parse_args()


def print_available_data_columns(data_columns: list[str]) -> None:
    print("Available data columns:")
    for i, col in enumerate(data_columns):
        print(f"{i}: {col}")


def select_available_data_columns_to_process(
        data_columns: list[str],
        df: pd.DataFrame,
        sensors_chosen: list[int],
        sensor_ref_chosen: Optional[int]
) -> Tuple[list[str], Optional[str], pd.DataFrame]:

    n = len(data_columns)

    if not sensors_chosen:
        raise ValueError("[ERROR] 'sensors_chosen' cannot be empty")

    if sensor_ref_chosen is not None and not isinstance(sensor_ref_chosen, int):
        raise TypeError("[ERROR] 'sensor_ref_chosen' must be an int or None")

    sensors_chosen = list(dict.fromkeys(sensors_chosen))

    bad_sensors = [i for i in sensors_chosen if not isinstance(i, int) or i < 0 or i >= n]
    if bad_sensors:
        raise IndexError(f"[ERROR] Sensor index out of range: {bad_sensors}; there are {n} columns")

    if sensor_ref_chosen is not None:
        if sensor_ref_chosen < 0 or sensor_ref_chosen >= n:
            raise IndexError(f"[ERROR] Reference index out of range: {sensor_ref_chosen}; there are {n} columns")
        if sensor_ref_chosen in sensors_chosen:
            raise ValueError(f"[ERROR] Sensors and reference overlap at index: {sensor_ref_chosen}")

    sensor_names = [data_columns[i] for i in sensors_chosen]
    sensor_name_ref: Optional[str] = None if sensor_ref_chosen is None else data_columns[sensor_ref_chosen]

    subset = list(sensor_names)
    if sensor_name_ref is not None:
        subset.append(sensor_name_ref)
    df_out = df.dropna(subset=subset)

    return sensor_names, sensor_name_ref, df_out


def load_filtered_and_calculated_data_needed_for_execute_function_periods_detected(
        model_directories: ModelDirectories,
        sensor_name_ref: str
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    df = load_dataframe_from_csv(
        Path(model_directories.data_dir / "filtered" / f"{model_directories.filename}.csv"))

    df_sunny = load_dataframe_from_csv(
        Path(model_directories.data_dir / "filtered" / "sunny_periods" / f"{model_directories.filename}_all.csv"))

    df_cloudy = load_dataframe_from_csv(
        Path(model_directories.data_dir / "filtered" / "cloudy_periods" / f"{model_directories.filename}_all.csv"))

    poa = load_dataframe_from_csv(
        Path(model_directories.data_dir / "calculated_data" / model_directories.filename / "poa_values.csv"))

    sensor_name = sanitize_filename(sensor_name_ref)

    clearsky_periods = load_dataframe_from_csv(
        Path(model_directories.data_dir / "calculated_data" / model_directories.filename /
             f"{sensor_name}_sunny_periods_all.csv"))

    cloudy_periods = load_dataframe_from_csv(
        Path(model_directories.data_dir / "calculated_data" / model_directories.filename /
             f"{sensor_name}_cloudy_periods_all.csv"))

    df_sunny['if_sunny'] = True
    df_cloudy['if_sunny'] = False

    return df, df_sunny, df_cloudy, poa, clearsky_periods, cloudy_periods

def load_filtered_and_calculated_data_needed_for_execute_function_no_periods_detected(
        model_directoires: ModelDirectories
) -> pd.DataFrame:
    df = load_dataframe_from_csv(
        Path(model_directoires.data_dir / "filtered" / f"{model_directoires.filename}.csv"))

    return df


def check_if_sunny_cloudy_periods_exists(model_directories: ModelDirectories) -> bool:

    check_dir_sunny = Path(model_directories.data_dir / "filtered" / "sunny_periods")
    check_dir_cloudy = Path(model_directories.data_dir / "filtered" / "cloudy_periods")

    if check_dir_sunny.exists() and check_dir_cloudy.exists():
        return True
    else:
        return False


def initialize_dirs_for_base_dir(project_dir):

    log_dir = Path(os.path.join(project_dir, "logs"))
    plot_dir = Path(os.path.join(project_dir, "plots"))
    data_dir = Path(os.path.join(project_dir, "data"))

    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return log_dir, plot_dir, data_dir


def initialize_dirs_for_loading_dependencies(project_dir_path):

    log_dir_dependencies = Path(project_dir_path)
    log_dir_dependencies.mkdir(parents=True, exist_ok=True)

    return log_dir_dependencies


def create_calibrated_dataframe(
        model_parameters: ModelParameters,
        model_directories: ModelDirectories,
        calibration_method: str
) -> pd.DataFrame:

    df = model_parameters.df.copy()

    df = load_and_merge_calibrated_data_from_each_sensor(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        log_dir=model_directories.log_dir,
        filename=model_directories.filename,
        calibration_method=calibration_method,
        sensor_name_ref = model_parameters.sensor_name_ref
    )

    save_dir = Path(model_directories.data_dir / "filtered" / "calibrated" / model_directories.filename /
                    f"{calibration_method}.csv"
                    )

    save_dataframe_to_csv(
        df=df,
        output_path=save_dir
    )

    return df

def create_dataframe_with_sensor_values_and_poa(
        df_postprocess: pd.DataFrame,
        df_calibrated: pd.DataFrame,
        sensor_names: list[str],
        poa: pd.DataFrame,
        data_dir: Path,
        filename: str,
) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_org = load_dataframe_from_csv(Path(data_dir / "filtered" / f"{filename}.csv"))
    df_org.columns = [sanitize_filename(name) for name in df_org.columns]
    df_org["time"] = pd.to_datetime(df_org["time"])

    df_tmp_org = df_postprocess["time"].copy()
    df_tmp_postprocess_calibrated = df_postprocess["time"].copy()
    df_tmp_calibrated = df_calibrated["time"].copy()


    df_org_sensor_data_with_poa_global = df_postprocess["time"].copy()
    df_postprocess_calibrated_sensor_data_with_poa_global = df_postprocess["time"].copy()
    df_calibrated_sensor_data_with_poa_global = df_calibrated["time"].copy()

    for col in sensor_names:
        df_tmp_org = pd.merge(df_tmp_org, df_org[["time", col]], on="time", how="left")
        df_tmp_postprocess_calibrated = pd.merge(df_tmp_postprocess_calibrated, df_postprocess[["time", col]],
                                                 on="time", how="left")
        df_tmp_calibrated = pd.merge(df_tmp_calibrated, df_calibrated[["time", col]],
                                                 on="time", how="left")

    df_org_sensor_data_with_poa_global = pd.merge(
        df_tmp_org,
        poa[["time", "poa_global"]],
        on="time",
        how="left"
    )

    df_postprocess_calibrated_sensor_data_with_poa_global = pd.merge(
        df_tmp_postprocess_calibrated,
        poa[["time", "poa_global"]],
        on="time",
        how="left"
    )

    df_calibrated_sensor_data_with_poa_global = pd.merge(
        df_tmp_calibrated,
        poa[["time", "poa_global"]],
        on="time",
        how="left"
    )

    output_path_org = Path(data_dir) / "filtered" / f"df_org_sensor_data_with_poa_global.csv"
    output_path_postprocess_calibrated = Path(data_dir) / "filtered" / f"df_postprocess_calibrated_sensor_data_with_poa_global.csv"
    output_path_calibrated = Path(data_dir) / "filtered" / f"df_calibrated_sensor_data_with_poa_global.csv"

    save_dataframe_to_csv(
        df=df_org_sensor_data_with_poa_global,
        output_path=output_path_org,
        index=False,
        index_label=None
    )

    save_dataframe_to_csv(
        df=df_postprocess_calibrated_sensor_data_with_poa_global,
        output_path=output_path_postprocess_calibrated,
        index=False,
        index_label=None
    )

    save_dataframe_to_csv(
        df=df_calibrated_sensor_data_with_poa_global,
        output_path=output_path_calibrated,
        index=False,
        index_label=None
    )

    dfs_result = [
        df_org,
        df_postprocess_calibrated_sensor_data_with_poa_global,
        df_calibrated_sensor_data_with_poa_global,
        df_org_sensor_data_with_poa_global
    ]

    return dfs_result
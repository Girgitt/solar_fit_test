import math
import numpy as np
import pandas as pd

from argparse import ArgumentParser, Namespace
from datetime import datetime, time
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo

from pvtools.analysis.analyse_calibration import calibrate_by_linear_regression, calibrate_by_divided_linear_regression, \
    calibrate_by_polynominal_regression, calibrate_by_decision_tree_regression, calibrate_by_mlp_regression
from pvtools.config.params import ModelParameters, ClearSkyParameters
from pvtools.modeling.calibrate import linear_regression, divided_linear_regression, polynominal_regression, \
    decision_tree_regression, mlp_regression
from pvtools.solar_domain.clearsky import clear_sky, detect_clearsky_periods
from pvtools.solar_domain.determine_orientation import determine_system_azimuth_and_tilt
from pvtools.visualisation.plotter import plot_raw_data, plot_predicted_data, plot_poa_vs_reference, \
    plot_poa_reference_with_clearsky_periods, plot_raw_data_with_peaks
from pvtools.preprocess.preprocess_data import delete_night_period
from pvtools.io_file.reader import load_dataframe_from_csv

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

def check_if_csv_contains_timezone_info(file_path: str) -> None:
    file_path = Path(file_path)

    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])

    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize(ZoneInfo('Europe/Warsaw'))
        df.to_csv(file_path, index=False)
        print('Time does not contain timezone info. Timezone added and file updated')
    else:
        print('[INFO] Time already has timezone. No changes made')

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

def execute_function(model_parameters: ModelParameters, clear_sky_parameters: ClearSkyParameters) -> None:
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

    poa = clear_sky(
        clear_sky_parameters=clear_sky_parameters,
        show=False,
        save_dir_plot=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        save_dir_data=model_parameters.data_filename_dir
    )

    clearsky_periods = detect_clearsky_periods(
        poa=poa,
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        save_dir=model_parameters.data_filename_dir
    )

    determine_system_azimuth_and_tilt(
        clear_sky_parameters=clear_sky_parameters,
        df=model_parameters.df,
        sunny_mask=clearsky_periods,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        tilts=np.arange(0, 30, 1), # None
        azimuths=np.arange(170, 190, 1) # None
    )

    plot_raw_data(
        df=model_parameters.df,
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        filename="series_vs_time.png",
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        show=True,
    )

    # ----------------------------------- TEMPORARY PLOTTING FOR FILTERED DATA -----------------------------------------
    plot_raw_data(
        df=load_dataframe_from_csv(model_parameters.data_filename_dir.parent / "filtered" / Path(model_parameters.args.csv).name),
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        filename="series_vs_time_filtered.png",
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        show=True,
    )

    plot_predicted_data(
        calibration_method_dir=model_parameters.log_dir / Path(model_parameters.args.csv).stem,
        show=False,
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
    )

    plot_poa_vs_reference(
        poa_global=poa['poa_global'],
        sensor_reference=model_parameters.df[model_parameters.sensor_name_ref],
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        show=True,
    )

    plot_poa_reference_with_clearsky_periods(
        poa_global=poa['poa_global'],
        sensor_reference=model_parameters.df[model_parameters.sensor_name_ref],
        sunny=clearsky_periods,
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        show=True,
    )

    plot_raw_data_with_peaks(
        df=model_parameters.df,
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        peaks_dir= Path("data/interpolated") / Path(model_parameters.data_filename_dir).stem,
        filename="series_vs_time_with_peaks",
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        show=True
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


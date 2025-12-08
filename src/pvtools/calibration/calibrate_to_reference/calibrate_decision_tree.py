import os
import pandas as pd
import numpy as np
import logging

from pathlib import Path
from typing import TypeAlias, Literal

from pvtools.io_file.writer import save_true_and_predicted_data_to_csv
from pvtools.io_file.reader import decision_tree_regression_load_parameters
from pvtools.config.params import ModelData, ModelDirectories
from pvtools.calibration.calibrate_to_reference.calibration_utils import check_if_any_column_is_missing
from pvtools.calibration.calibrate_to_reference.validate_tree import _traverse_tree


log = logging.getLogger("calibrate")
Period_type: TypeAlias = Literal['sunny', 'cloudy']

def calibrate_by_decision_tree_regression(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        period_flag: bool = True  # if True - periods detected, else not
) -> None:

    df = model_data.df
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref
    load_params_dir = model_dirs.load_metrics_dir
    save_dir = model_dirs.log_dir
    filename = model_dirs.filename

    decision_tree_regression = "decision_tree_regression"
    calibration_method_dir_sunny = Path(os.path.join(load_params_dir, decision_tree_regression, "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(load_params_dir, decision_tree_regression, "cloudy"))

    log.debug(f"calibration_method_dir:{calibration_method_dir_sunny}")
    log.debug(f"calibration_method_dir:{calibration_method_dir_cloudy}")

    json_files_sunny = list(calibration_method_dir_sunny.glob("*.json"))
    json_files_cloudy = list(calibration_method_dir_cloudy.glob("*.json"))

    if len(json_files_sunny) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_sunny},"
            f" but found {len(json_files_sunny)}.")

    if len(json_files_cloudy) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_cloudy},"
            f" but found {len(json_files_cloudy)}.")

    time = df["time"]
    y_pred = pd.Series()

    if period_flag is True:

        y_true = df[sensor_name_ref]

        for i, json_file_dir_sunny in enumerate(json_files_sunny):
            params_sunny = decision_tree_regression_load_parameters(json_file_dir_sunny)

            for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
                params_cloudy = decision_tree_regression_load_parameters(json_file_dir_cloudy)

                log.debug(f"fitting json: {json_file_dir_sunny}")
                log.debug(f"fitting json: {json_file_dir_cloudy}")

                y_pred = decision_tree_regression_use_calibration_values(
                    df=df[["time", sensor_names[j], "if_sunny"]],
                    sensor_name=sensor_names[j],
                    params_sunny=params_sunny,
                    params_cloudy=params_cloudy
                )

            output_dir = Path(save_dir) / filename / decision_tree_regression
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)

    else:
        df["if_sunny"] = True

        for i, json_file_dir_sunny in enumerate(json_files_sunny):
            params_sunny = decision_tree_regression_load_parameters(json_file_dir_sunny)

            log.debug(f"fitting json: {json_file_dir_sunny}")

            if sensor_name_ref is not None:
                y_true = df[sensor_name_ref]
            else:
                y_true = None

            y_pred = decision_tree_regression_use_calibration_values(
                df=df[["time", sensor_names[i], "if_sunny"]],
                sensor_name=sensor_names[i],
                params_sunny=params_sunny,
                params_cloudy=None
            )

            output_dir = Path(save_dir) / filename / decision_tree_regression
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)


def decision_tree_regression_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny: dict,
        params_cloudy: dict | None = None,
) -> pd.Series:

    if params_cloudy is not None:
        if_sunny_col = "if_sunny"
    else:
        if_sunny_col = None

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time",
        if_sunny_col=if_sunny_col
    )

    x = df[sensor_name].to_numpy().flatten()
    is_sunny = df["if_sunny"].astype(bool).to_numpy()

    if not (params_sunny or params_cloudy):
        raise ValueError("At least params_sunny must contain a 'params' key with a tree structure.")

    y_pred = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        model = params_sunny if is_sunny[i] else params_cloudy
        y_pred[i] = _traverse_tree(model, x[i])

    return pd.Series(y_pred, index=df["time"])



import os
import sys
import pandas as pd
import numpy as np
import logging
import datetime

from pathlib import Path
from typing import TypeAlias, Literal

from urllib3.util.util import to_str

from pvtools.io_file.writer import save_true_and_predicted_data_to_csv
from pvtools.io_file.reader import (linear_regression_load_parameters, divided_linear_regression_load_parameters,
                                    polynominal_regression_load_parameters, mlp_load_parameters,
                                    decision_tree_regression_load_parameters)
from pvtools.calibration.validate_decision_tree import _traverse_tree
from pvtools.config.params import (DatatypeCoefficientsForMLPRegression, DatatypeMLPRegressionParameters,
                                   DatatypeScalersForMLPRegression, DatatypeCoefficientsForDividedLinearRegression,
                                   ModelData, ModelDirectories, ModelTimes, ClearSkyCalculatedValues)

log = logging.getLogger("calibrate")

Period_type: TypeAlias = Literal['sunny', 'cloudy']


def calibrate_by_linear_regression(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        period_flag: bool = True # if True - periods detected, else not
) -> None:

    df = model_data.df
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref
    load_params_dir = model_dirs.load_metrics_dir
    save_dir = model_dirs.log_dir
    filename = model_dirs.filename

    linear_regression = "linear_regression"
    calibration_method_dir_sunny = Path(os.path.join(load_params_dir, linear_regression, "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(load_params_dir, linear_regression, "cloudy"))

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
            params_sunny = linear_regression_load_parameters(json_file_dir_sunny)

            for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
                params_cloudy = linear_regression_load_parameters(json_file_dir_cloudy)

                log.debug(f"fitting json: {json_file_dir_sunny}")
                log.debug(f"fitting json: {json_file_dir_cloudy}")

                y_pred = linear_regression_use_calibration_values(
                    df=df[["time", sensor_names[j], "if_sunny"]],
                    sensor_name=sensor_names[j],
                    params_sunny=params_sunny,
                    params_cloudy=params_cloudy
                )

                output_dir = Path(save_dir) / filename / linear_regression
                file_stem = Path(json_file_dir_sunny).stem
                csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
                log.debug(f"csv_filename: {csv_filename}")
                save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None) # time passed as an index of series

    else:

        for i, json_file_dir_sunny in enumerate(json_files_sunny):
            params_sunny = linear_regression_load_parameters(json_file_dir_sunny)

            log.debug(f"fitting json: {json_file_dir_sunny}")

            if sensor_name_ref is not None:
                y_true = df[sensor_name_ref]
            else:
                y_true = None

            y_pred = linear_regression_use_calibration_values(
                df=df[["time", sensor_names[i]]],
                sensor_name=sensor_names[i],
                params_sunny=params_sunny,
                params_cloudy=None
            )

            output_dir = Path(save_dir) / filename / linear_regression
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=time)


def calibrate_by_fuzzy_linear_regression(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        clearsky_cal_val: ClearSkyCalculatedValues,
        period_flag: bool = True  # if True - periods detected, else not
) -> None:

    df = model_data.df
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref
    poa = clearsky_cal_val.poa
    load_params_dir = model_dirs.load_metrics_dir
    save_dir = model_dirs.log_dir
    filename = model_dirs.filename
    period_flag = period_flag

    linear_regression = "linear_regression"
    calibration_method_dir_sunny = Path(os.path.join(load_params_dir, linear_regression, "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(load_params_dir, linear_regression, "cloudy"))

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

    if period_flag is True:

        time = df["time"]
        y_true = df[sensor_name_ref]
        y_pred = pd.Series()

        left = df[["time", sensor_name_ref]]
        right = poa[["time", "poa_global"]]
        merged = left.merge(right, on="time", how="inner").sort_values("time")

        eps = 1e-6
        k_t = merged[sensor_name_ref] / (merged["poa_global"] + eps)

        for i, json_file_dir_sunny in enumerate(json_files_sunny):
            params_sunny = linear_regression_load_parameters(json_file_dir_sunny)

            for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
                params_cloudy = linear_regression_load_parameters(json_file_dir_cloudy)

                log.debug(f"fitting json: {json_file_dir_sunny}")
                log.debug(f"fitting json: {json_file_dir_cloudy}")

                y_pred = fuzzy_regression_use_calibration_values(
                    df=df[["time", sensor_names[j], "if_sunny"]],
                    sensor_name=sensor_names[j],
                    params_sunny=params_sunny,
                    params_cloudy=params_cloudy,
                    kt=k_t,  # uses k_t ramp 0.5→0.7 + smoothing
                    kt_col=None,
                    t0=0.50,
                    t1=0.70,
                    smooth_window=5
                )

            output_dir = Path(save_dir) / filename / "fuzzy_regression"
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)

    else:

        log.info(f"Cannot calibrate by fuzzy regression due to just one period type (sunny/cloudy)")
        sys.exit("SystemExit: No calibration possible - terminating the program")


def calibrate_by_divided_linear_regression(
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

    divided_linear_regression = "divided_linear_regression"
    calibration_method_dir_all = Path(os.path.join(load_params_dir, divided_linear_regression, "all"))
    calibration_method_dir_sunny = Path(os.path.join(load_params_dir, divided_linear_regression, "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(load_params_dir, divided_linear_regression, "cloudy"))

    log.debug(f"calibration_method_dir:{calibration_method_dir_all}")
    log.debug(f"calibration_method_dir:{calibration_method_dir_sunny}")
    log.debug(f"calibration_method_dir:{calibration_method_dir_cloudy}")

    json_files_all = list(calibration_method_dir_all.glob("*.json"))
    json_files_sunny = list(calibration_method_dir_sunny.glob("*.json"))
    json_files_cloudy = list(calibration_method_dir_cloudy.glob("*.json"))

    if len(json_files_all) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_all},"
            f" but found {len(json_files_all)}.")

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
    y_true = df[sensor_name_ref]

    for json_all, json_sunny, json_cloudy, sensor_name in (
            zip(json_files_all, json_files_sunny, json_files_cloudy, sensor_names)):

        params_all = divided_linear_regression_load_parameters(json_all)
        params_sunny = divided_linear_regression_load_parameters(json_sunny)
        params_cloudy = divided_linear_regression_load_parameters(json_cloudy)

        log.debug(f"fitting json all: {json_all}")
        log.debug(f"fitting json sunny: {json_sunny}")
        log.debug(f"fitting json cloudy: {json_cloudy}")

        if period_flag is True:
            y_pred = divided_linear_regression_use_calibration_values(
                df=df[["time", sensor_name, "if_sunny"]],
                sensor_name=sensor_name,
                params_sunny=params_sunny,
                params_cloudy=params_cloudy
            )

        elif period_flag is False:
            if sensor_name_ref is None:
                y_true = None

            y_pred = divided_linear_regression_use_calibration_values(
                df=df[["time", sensor_name, "if_sunny"]],
                sensor_name=sensor_name,
                params_sunny=params_sunny,
                params_cloudy=params_all
            )

        output_dir = Path(save_dir) / filename / divided_linear_regression

        file_stem_sunny = Path(json_sunny).stem
        file_stem_cloudy = Path(json_cloudy).stem

        for file in (file_stem_sunny, file_stem_cloudy):
            csv_filename = output_dir / f"{file}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)


def calibrate_by_divided_linear_regression_mean(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
        period_flag: bool = True  # if True - periods detected, else not
) -> None:

    df = model_data.df
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref
    load_params_dir = model_dirs.load_metrics_dir
    save_dir = model_dirs.log_dir
    filename = model_dirs.filename

    divided_linear_regression_mean = "divided_linear_regression_mean"
    calibration_method_dir_all = Path(os.path.join(load_params_dir, divided_linear_regression_mean, "all"))
    calibration_method_dir_sunny = Path(os.path.join(load_params_dir, divided_linear_regression_mean, "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(load_params_dir, divided_linear_regression_mean, "cloudy"))

    log.debug(f"calibration_method_dir:{calibration_method_dir_all}")
    log.debug(f"calibration_method_dir:{calibration_method_dir_sunny}")
    log.debug(f"calibration_method_dir:{calibration_method_dir_cloudy}")

    json_files_all = list(calibration_method_dir_all.glob("*.json"))
    json_files_sunny = list(calibration_method_dir_sunny.glob("*.json"))
    json_files_cloudy = list(calibration_method_dir_cloudy.glob("*.json"))

    if len(json_files_all) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_all},"
            f" but found {len(json_files_all)}.")

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

    if sensor_name_ref is not None:
        y_true = df[sensor_name_ref]
    else:
        y_true = None

    for json_all, json_sunny, json_cloudy, sensor_name in (
            zip(json_files_all, json_files_sunny, json_files_cloudy, sensor_names)):

        params_all = divided_linear_regression_load_parameters(json_all)
        params_sunny = divided_linear_regression_load_parameters(json_sunny)
        params_cloudy = divided_linear_regression_load_parameters(json_cloudy)

        log.debug(f"fitting json all: {json_all}")
        log.debug(f"fitting json sunny: {json_sunny}")
        log.debug(f"fitting json cloudy: {json_cloudy}")

        params_combined = select_calibration_parameters(
                params_all=params_all,
                params_sunny=params_sunny,
                params_cloudy=params_cloudy,
                df_time=time,
                frequency=model_times.divided_linear_regression_interval
        )

        if period_flag is True:
            y_pred = divided_linear_regression_use_calibration_values_mean(
                df=df[["time", sensor_name, "if_sunny"]],
                sensor_name=sensor_name,
                params_sunny=params_sunny
            )

        elif period_flag is False:
            df["if_sunny"] = True

            y_pred = divided_linear_regression_use_calibration_values_mean(
                df=df[["time", sensor_name, "if_sunny"]],
                sensor_name=sensor_name,
                params_sunny=params_combined
            )

        output_dir = Path(save_dir) / filename / divided_linear_regression_mean
        output_dir.mkdir(parents=True, exist_ok=True)

        file_stem_sunny = Path(json_sunny).stem
        csv_filename = output_dir / f"{file_stem_sunny}_all_predicted.csv"
        log.debug(f"csv_filename: {csv_filename}")
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=time)


def calibrate_by_polynominal_regression(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        period_flag: bool = False  # if True - periods detected, else not
) -> None:

    df = model_data.df
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref
    load_params_dir = model_dirs.load_metrics_dir
    save_dir = model_dirs.log_dir
    filename = model_dirs.filename

    polynominal_regression = "polynominal_regression"
    calibration_method_dir_sunny = Path(os.path.join(load_params_dir, polynominal_regression, "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(load_params_dir, polynominal_regression, "cloudy"))

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
            params_sunny = polynominal_regression_load_parameters(json_file_dir_sunny)

            for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
                params_cloudy = polynominal_regression_load_parameters(json_file_dir_cloudy)

                log.debug(f"fitting json: {json_file_dir_sunny}")
                log.debug(f"fitting json: {json_file_dir_cloudy}")

                y_pred = polynominal_regression_use_calibration_values(
                    df=df[["time", sensor_names[j], "if_sunny"]],
                    sensor_name=sensor_names[j],
                    params_sunny=params_sunny,
                    params_cloudy=params_cloudy
                )

            output_dir = Path(save_dir) / filename / polynominal_regression
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)

    else:
        df["if_sunny"] = True

        for i, json_file_dir_sunny in enumerate(json_files_sunny):
            params_sunny = polynominal_regression_load_parameters(json_file_dir_sunny)

            log.debug(f"fitting json: {json_file_dir_sunny}")

            if sensor_name_ref is not None:
                y_true = df[sensor_name_ref]
            else:
                y_true = None

            y_pred = polynominal_regression_use_calibration_values(
                df=df[["time", sensor_names[i], "if_sunny"]],
                sensor_name=sensor_names[i],
                params_sunny=params_sunny,
                params_cloudy=None
            )

            output_dir = Path(save_dir) / filename / polynominal_regression
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)


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


def calibrate_by_mlp_regression(
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

    mlp_regression = "mlp_regression"
    calibration_method_dir_sunny = Path(os.path.join(load_params_dir, mlp_regression, "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(load_params_dir, mlp_regression, "cloudy"))

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
            params_sunny = mlp_load_parameters(json_file_dir_sunny)

            for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
                params_cloudy = mlp_load_parameters(json_file_dir_cloudy)

                log.debug(f"fitting json: {json_file_dir_sunny}")
                log.debug(f"fitting json: {json_file_dir_cloudy}")

                y_pred = mlp_use_calibration_values(
                    df=df[["time", sensor_names[j], "if_sunny"]],
                    sensor_name=sensor_names[j],
                    params_sunny=params_sunny,
                    params_cloudy=params_cloudy
                )

            output_dir = Path(save_dir) / filename / mlp_regression
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)

    else:
        df["if_sunny"] = True

        for i, json_file_dir_sunny in enumerate(json_files_sunny):
            params_sunny = mlp_load_parameters(json_file_dir_sunny)

            log.debug(f"fitting json: {json_file_dir_sunny}")

            if sensor_name_ref is not None:
                y_true = df[sensor_name_ref]
            else:
                y_true = None

            y_pred = mlp_use_calibration_values(
                df=df[["time", sensor_names[i], "if_sunny"]],
                sensor_name=sensor_names[i],
                params_sunny=params_sunny,
                params_cloudy=None
            )

            output_dir = Path(save_dir) / filename / mlp_regression
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)


def linear_regression_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny: dict,
        params_cloudy: dict | None = None
) -> pd.Series:

    if params_cloudy is not None:
        if_sunny_col = "if_sunny"
    else:
        if_sunny_col = None

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time",
        if_sunny_col=if_sunny_col,
    )

    x = df[sensor_name]
    x.index = df["time"]

    if params_cloudy is not None:
        is_sunny = df["if_sunny"].astype(bool)
        is_sunny.index = df["time"]

        y_pred = pd.Series(index=df["time"], dtype=float)
        y_pred[is_sunny] = params_sunny["a"] * x[is_sunny] + params_sunny["b"]
        y_pred[~is_sunny] = params_cloudy["a"] * x[~is_sunny] + params_cloudy["b"]
    else:
        y_pred = params_sunny["a"] * x + params_sunny["b"]

    return y_pred


def fuzzy_regression_use_calibration_values(
    df: pd.DataFrame,
    sensor_name: str,
    params_sunny: dict,
    params_cloudy: dict | None = None,
    *,
    # choose ONE of the following to build weights:
    kt: np.ndarray | None = None,         # pass an array aligned to df.index
    kt_col: str | None = None,            # or name of a column in df with k_t
    use_mask_as_weight: bool = False,     # or derive soft weights from 'if_sunny'
    t0: float = 0.50,
    t1: float = 0.70,
    smooth_window: int = 5
) -> pd.Series:

    if params_cloudy is not None:
        if_sunny_col = "if_sunny"
    else:
        if_sunny_col = None

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time",
        if_sunny_col=if_sunny_col,
    )

    a_s, b_s = float(params_sunny["a"]), float(params_sunny["b"])
    a_c, b_c = float(params_cloudy["a"]), float(params_cloudy["b"])

    # feature vector (all rows; ensures shapes line up)
    x = np.asarray(df[sensor_name].to_numpy(), dtype=float).flatten()

    # per-regime predictions for ALL rows (avoids shape mismatch)
    y_s = a_s * x + b_s
    y_c = a_c * x + b_c

    # build weight vector w aligned to df rows
    if kt is not None:
        if len(kt) != len(df):
            raise ValueError("kt length must match df length.")
        w = _fuzzy_weight_from_kt(kt, t0=t0, t1=t1, smooth_window=smooth_window)

    elif kt_col is not None:
        if kt_col not in df.columns:
            raise KeyError(f"Missing clearness index column: {kt_col}")
        w = _fuzzy_weight_from_kt(
            df[kt_col].to_numpy(),
            t0=t0, t1=t1, smooth_window=smooth_window
        )

    elif use_mask_as_weight:
        if "if_sunny" not in df.columns:
            raise KeyError("Missing 'if_sunny' column required for mask-based weights.")
        # convert boolean mask to {0,1} and softly smooth to get fuzzy edges
        mask = df["if_sunny"].astype(bool).fillna(False).to_numpy().astype(float)
        w = _moving_average_1d(mask, smooth_window)
        w = np.clip(w, 0.0, 1.0)  # already in [0,1]; no ramp needed

    else:
        raise ValueError("Provide kt, kt_col, or set use_mask_as_weight=True.")

    # final blended prediction (shape == len(df))
    y_hat = w * y_s + (1.0 - w) * y_c

    y_hat = pd.Series(y_hat, index=df["time"])
    return y_hat


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average; preserves length; interpolates NaNs."""
    if window is None or window <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    nan = np.isnan(x)
    if nan.any():
        idx = np.arange(x.size)
        x[nan] = np.interp(idx[nan], idx[~nan], x[~nan]) if (~nan).any() else 0.0
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _fuzzy_weight_from_kt(
    k_t: np.ndarray,
    t0: float = 0.50,
    t1: float = 0.70,
    smooth_window: int = 5
) -> np.ndarray:
    """
    Sunny membership in [0,1] from k_t with a smoothed linear ramp:
      k_t <= t0 -> 0 (cloudy),  k_t >= t1 -> 1 (sunny)
    """
    k_t = np.asarray(k_t, dtype=float).flatten()
    k_t = np.clip(k_t, 0.0, 1.0)
    k_t_s = _moving_average_1d(k_t, smooth_window)
    eps = 1e-12
    w = (k_t_s - t0) / max(t1 - t0, eps)
    return np.clip(w, 0.0, 1.0)


def divided_linear_regression_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny:  list[DatatypeCoefficientsForDividedLinearRegression],
        params_cloudy: list[DatatypeCoefficientsForDividedLinearRegression] | None = None
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

    def build_intervals(
            param_list: list[DatatypeCoefficientsForDividedLinearRegression]
    ) -> list[tuple[pd.Timestamp, float, float]]:

        intervals = []
        for p in param_list:
            if all(k in p for k in ("hour", "a", "b")):
                hour = p["hour"]
                #if isinstance(hour, str):
                #    hour = pd.to_datetime(hour)
                intervals.append((hour, p["a"], p["b"]))

        return sorted(intervals, key=lambda x: x[0])

    intervals_sunny = build_intervals(params_sunny)

    if params_cloudy is not None:
        intervals_cloudy = build_intervals(params_cloudy)
    else:
        intervals_cloudy = None

    if not (intervals_sunny or intervals_cloudy):
        raise ValueError("At least params_sunny must contain valid (hour, a, b) entries.")

    x = df[sensor_name]
    x.index = df["time"]

    time = df["time"]
    time = pd.to_datetime(time)

    is_sunny = df["if_sunny"]
    is_sunny.index = df["time"]

    y_pred = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        current_time = time.iloc[i].time()
        current_params = intervals_sunny if is_sunny.iloc[i] else intervals_cloudy

        a, b = 0.0, 0.0
        for j, (t_start, a_j, b_j) in enumerate(current_params):
            t_end = (
                current_params[j + 1][0]
                if j + 1 < len(current_params)
                else pd.Timestamp.max
            )

            t_start = pd.to_datetime(t_start).time()
            t_end = pd.to_datetime(t_end).time()
            #current_time = current_time.time()

            if t_start <= current_time < t_end:
                a, b = a_j, b_j
                break

        y_pred[i] = a * x.iloc[i] + b

    #result = pd.Series(y_pred, index=df.index, name=f"{sensor_name}_calibrated")
    result = pd.Series(y_pred, index=df["time"])

    return result


def divided_linear_regression_use_calibration_values_mean(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny:  list[DatatypeCoefficientsForDividedLinearRegression],
) -> pd.Series:

    #df["if_sunny"] = True

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time"
    )

    def build_intervals(
            param_list: list[DatatypeCoefficientsForDividedLinearRegression]
    ) -> list[tuple[pd.Timestamp, float, float]]:

        intervals = []
        for p in param_list:
            if all(k in p for k in ("hour", "a", "b")):
                hour = p["hour"]
                intervals.append((hour, p["a"], p["b"]))

        return sorted(intervals, key=lambda x: x[0])

    intervals_sunny = build_intervals(params_sunny)

    x = df[sensor_name]
    x.index = df["time"]

    time = df["time"]
    time = pd.to_datetime(time)

    y_pred = np.empty_like(x, index=df["time"], dtype=float)

    for i in range(len(x)):
        current_time = time.iloc[i].time()

        a, b = 0.0, 0.0
        for j, (t_start, a_j, b_j) in enumerate(intervals_sunny):
            t_end = (
                intervals_sunny[j + 1][0]
                if j + 1 < len(intervals_sunny)
                else str("23:59")
            )

            t_start = datetime.datetime.strptime(t_start).time()
            t_end = datetime.datetime.strptime(t_end).time()

            if t_start <= current_time < t_end:
                a, b = a_j, b_j
                break

        y_pred[i] = a * x.iloc[i] + b

    result = pd.Series(y_pred, index=df["time"])
    return result


def polynominal_regression_use_calibration_values(
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

    x = df[sensor_name]
    x.index = df["time"]

    is_sunny = df["if_sunny"].astype(bool)
    is_sunny.index = df["time"]

    y_pred = pd.Series(index=df["time"], dtype=float)

    y_pred[is_sunny] = (
            params_sunny["a"] * x[is_sunny] ** 2
            + params_sunny["b"] * x[is_sunny]
            + params_sunny["c"]
    )

    if params_cloudy is not None:
        y_pred[~is_sunny] = (
                params_cloudy["a"] * x[~is_sunny] ** 2
                + params_cloudy["b"] * x[~is_sunny]
                + params_cloudy["c"]
        )

    return y_pred


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


def mlp_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny: DatatypeMLPRegressionParameters,
        params_cloudy: DatatypeMLPRegressionParameters | None = None,
        activation: str = 'relu'
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

    x = df[sensor_name].to_numpy().reshape(-1, 1)  # shape (n_samples, n_inputs)
    is_sunny = df["if_sunny"].astype(bool).to_numpy()
    y_pred = np.empty_like(x.flatten(), dtype=float)

    activation_sunny = params_sunny["scalers"].get("activation", activation)

    if params_cloudy is not None:
        activation_cloudy = params_cloudy["scalers"].get("activation", activation)

    if np.any(is_sunny):
        xs = _scale_in(x[is_sunny], params_sunny["scalers"])
        ys = _forward_pass(xs, params_sunny["coefficients"], activation_sunny).reshape(-1, 1)
        y_pred[is_sunny] = _inv_out(ys, params_sunny["scalers"]).ravel()

    if np.any(~is_sunny):
        xc = _scale_in(x[~is_sunny], params_cloudy["scalers"])
        yc = _forward_pass(xc, params_cloudy["coefficients"], activation_cloudy).reshape(-1, 1)
        y_pred[~is_sunny] = _inv_out(yc, params_cloudy["scalers"]).ravel()

    return pd.Series(y_pred, index=df["time"])

def _scale_in(
        x2d: np.ndarray,
        scaler: DatatypeScalersForMLPRegression
) -> np.ndarray:

    mean = scaler.get("x_scaler_mean", None)
    scale = scaler.get("x_scaler_scale", None)

    if mean is not None and scale is not None:
        mean = np.asarray(mean)
        scale = np.asarray(scale)
        scale = np.where(scale == 0, 1.0, scale)
        return (x2d - mean) / scale

    return x2d


def _inv_out(
        y: np.ndarray,
        scaler: DatatypeScalersForMLPRegression
) -> np.ndarray:

    mean = scaler.get("y_scaler_mean", None)
    scale = scaler.get("y_scaler_scale", None)

    if mean is not None and scale is not None:
        mean = np.asarray(mean)
        scale = np.asarray(scale)
        return y * scale + mean

    return y


def _forward_pass(
        x: np.ndarray,
        coeffs: DatatypeCoefficientsForMLPRegression,
        activation: str,
) -> np.ndarray:

    W1 = np.array(coeffs["layer_1_weights"])
    b1 = np.array(coeffs["layer_1_biases"])
    W2 = np.array(coeffs["layer_2_weights"])
    b2 = np.array(coeffs["layer_2_biases"])
    W3 = np.array(coeffs["output_weights"])
    b3 = np.array(coeffs["output_biases"])

    z1: np.ndarray = x @ W1 + b1
    a1 = _apply_activation(z1, activation)

    z2: np.ndarray = a1 @ W2 + b2
    a2 = _apply_activation(z2, activation)

    output = a2 @ W3 + b3

    return output.flatten()


def _apply_activation(
        z: np.ndarray,
        activation: str
) -> np.ndarray:

    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'tanh':
        return np.tanh(z)
    elif activation == 'identity':
        return z
    else:
        raise ValueError(f"Unsupported activation: {activation}")


def check_if_any_column_is_missing(
        df: pd.DataFrame,
        sensor_name: str,
        time_col: str,
        if_sunny_col: str = None
) -> None:

    if if_sunny_col is not None:
        required_cols = {time_col, if_sunny_col, sensor_name}
    else:
        required_cols = {time_col, sensor_name}

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def select_calibration_parameters(
        params_all: list[DatatypeCoefficientsForDividedLinearRegression],
        params_sunny: list[DatatypeCoefficientsForDividedLinearRegression],
        params_cloudy: list[DatatypeCoefficientsForDividedLinearRegression],
        df_time: pd.Series,
        frequency: str
) -> list[DatatypeCoefficientsForDividedLinearRegression]:

    log.info("Checking coverage for sunny parameters...")

    try:
        sunny_ok = check_if_params_contains_data_for_all_time_intervals(params_sunny, df_time, frequency)
    except ValueError as e:
        log.warning(f"Sunny params incomplete: {e}")
        sunny_ok = False

    if sunny_ok:
        log.info("Sunny parameters have full coverage. Using them.")
        return params_sunny

    log.info("Checking coverage for all parameters...")

    try:
        all_ok = check_if_params_contains_data_for_all_time_intervals(params_all, df_time, frequency)
    except ValueError as e:
        log.warning(f"All params incomplete: {e}")
        all_ok = False

    json_hours_sunny = sorted([item["hour"] for item in params_sunny])

    df_time = pd.to_datetime(df_time)
    start = pd.Timestamp(df_time.iloc[0]).time()
    end = pd.Timestamp(df_time.iloc[-1]).time()

    expected_times = date_range_only_hh_mm(start=start, end=end, freq=frequency)
    missing_from_sunny = sorted(set(expected_times) - set(json_hours_sunny))

    if all_ok:
        if missing_from_sunny:
            log.info(f"Filling {len(missing_from_sunny)} missing intervals from all params.")

            merged = params_sunny.copy()
            merged_hours = {p["hour"]: p for p in merged}

            for p in params_all:
                if p["hour"] in missing_from_sunny and p["hour"] not in merged_hours:
                    merged.append(p)

            log.info("Calibration params merged successfully.")

            return sorted(merged, key=lambda x: x["hour"])
        else:
            return params_all

    else:
        log.error("Cannot calibrate! Not enough data in both sunny and all parameter sets.")
        raise ValueError("Cannot calibrate! Missing intervals in both sunny and all parameter sets.")


def date_range_only_hh_mm(
        start: datetime.time,
        end: datetime.time,
        freq: str
):

    delta = pd.to_timedelta(freq)

    sample_date = datetime.date(2000, 1, 1)
    cur = datetime.datetime.combine(sample_date, start)
    stop = datetime.datetime.combine(sample_date, end)

    times = []
    while cur < stop:
        times.append(cur.time())
        cur += delta
    return times


def check_if_params_contains_data_for_all_time_intervals(
        coeffs,
        df_time: pd.Series,
        frequency: str
) -> bool:

    df_time = pd.to_datetime(df_time).dt.tz_localize(None).dt.tz_localize("Europe/Warsaw").dt.tz_convert("UTC")

    json_hours = sorted([item["hour"] for item in coeffs])

    day_start = pd.Timestamp(df_time.iloc[0]).strftime("%H:%M")
    day_end = pd.Timestamp(df_time.iloc[-1]).strftime("%H:%M")
    expected_times = pd.date_range(day_start, day_end, freq=frequency, inclusive="left").strftime("%H:%M").to_list()

    all_intervals_contained = set(json_hours).issuperset(expected_times)

    if not all_intervals_contained:
        raise ValueError(f"Not fully coverage for time intervals from calibrated matrics!")

    df_time = pd.to_datetime(df_time)

    df_start = df_time.iloc[0]
    df_end = df_time.iloc[-1]
    total_days = (df_end - df_start).days + 1

    counted_days = 0
    if len(coeffs) > 0 and "count_days" in coeffs[0]:
        counted_days = max(item.get("count_days", 0) for item in coeffs)

    missing_intervals = sorted(set(expected_times) - set(json_hours))

    log.debug(f"Expected intervals: {expected_times}")
    log.debug(f"JSON intervals: {json_hours}")
    log.debug(f"Missing intervals: {missing_intervals}")

    all_present = len(missing_intervals) == 0

    return all_present
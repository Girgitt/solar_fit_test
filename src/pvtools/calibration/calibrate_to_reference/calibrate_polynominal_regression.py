import os
import pandas as pd
import logging

from pathlib import Path
from typing import TypeAlias, Literal

from pvtools.io_file.writer import save_true_and_predicted_data_to_csv
from pvtools.io_file.reader import polynominal_regression_load_parameters
from pvtools.config.params import ModelData, ModelDirectories
from pvtools.calibration.calibrate_to_reference.calibration_utils import check_if_any_column_is_missing


log = logging.getLogger("calibrate")
Period_type: TypeAlias = Literal['sunny', 'cloudy']


def calibrate_by_polynominal_regression(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        period_flag: bool = False  # if True - periods detected, else not
) -> None:
    """
    Calibrate Polynominal Regression model.

    Loads metrics from .json files. Search for ``sunny`` and ``cloudy`` files containing metrics for that periods.
    If not found raise an Error.

    Based on input boolean parameter ``period_flag`` - calculates calibrated values:

    * if ``True`` calculation is made on both periods
    * if ``False`` calculation is made on only sunny period

    Saves calibrated sensor data to .csv file.

    Warning:
          To consider - in ``False`` case it should be calibrated by metrics taken from all period - not just sunny!
    """

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


def polynominal_regression_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny: dict,
        params_cloudy: dict | None = None,
) -> pd.Series:
    """
    Do a calculation of Polynominal Regression using calibration values.

    .. math::

        y = a_n x^n + a_{n-1} x^{n-1} + \\dots + a_2 x^2 + a_1 x + a_0

    where:

    * :math:`x` is the raw sensor value,
    * :math:`y` is the calibrated output,
    * :math:`a_0, a_1, \\dots, a_n` are the polynomial calibration coefficients,
    * :math:`n` is the degree of the polynomial.

    This formulation generalizes linear calibration (n = 1) and supports higher-order
    models when sensor behavior is nonlinear.
    """

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
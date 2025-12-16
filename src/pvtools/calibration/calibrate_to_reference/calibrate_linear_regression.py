import os
import sys
import pandas as pd
import numpy as np
import logging

from pathlib import Path
from typing import TypeAlias, Literal

from pvtools.io_file.writer import save_true_and_predicted_data_to_csv
from pvtools.io_file.reader import linear_regression_load_parameters
from pvtools.config.params import ModelData, ModelDirectories, ClearSkyCalculatedValues
from pvtools.calibration.calibrate_to_reference.calibration_utils import check_if_any_column_is_missing


log = logging.getLogger("calibrate")
Period_type: TypeAlias = Literal['sunny', 'cloudy']


def calibrate_by_linear_regression(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        period_flag: bool = True # if True - periods detected, else not
) -> None:
    """
    Calibrate Linear Regression model.

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
    """
    Calibrate Fuzzy Linear Regression model.

    Loads metrics from .json files. Search for ``sunny`` and ``cloudy`` files containing metrics for that periods.
    If not found raise an Error.

    Based on input boolean parameter ``period_flag`` - calculates calibrated values:

    * if ``True`` calculation is made on both periods
    * if ``False`` raise an Error - cannot fuzzy with no second period!

    Saves calibrated sensor data to .csv file.
    """

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


def linear_regression_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny: dict,
        params_cloudy: dict | None = None
) -> pd.Series:
    """
    Do a calculation of Linear Regression using calibration values.

    .. math::

            y = a * x + b
    """

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
    """
    Applies fuzzy linear regression blending between sunny and cloudy calibration models.

    Uses a soft weight vector based on clearness index or a boolean mask to combine
    two linear models:

    .. math::

        \\hat{y} = w \\cdot (a_s x + b_s) + (1 - w) \\cdot (a_c x + b_c)

    where:

    - :math:`x` is the sensor measurement,
    - :math:`(a_s, b_s)` are sunny calibration parameters,
    - :math:`(a_c, b_c)` are cloudy calibration parameters,
    - :math:`w \\in [0,1]` is a smooth blending weight based on sky conditions.

    Parameters are chosen based on :math:`k_t`, :math:`k_{t_{col}}`, or a boolean mask like ``if_sunny``
    """

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
        w = fuzzy_weight_from_kt(kt, t0=t0, t1=t1, smooth_window=smooth_window)

    elif kt_col is not None:
        if kt_col not in df.columns:
            raise KeyError(f"Missing clearness index column: {kt_col}")
        w = fuzzy_weight_from_kt(
            df[kt_col].to_numpy(),
            t0=t0, t1=t1, smooth_window=smooth_window
        )

    elif use_mask_as_weight:
        if "if_sunny" not in df.columns:
            raise KeyError("Missing 'if_sunny' column required for mask-based weights.")
        # convert boolean mask to {0,1} and softly smooth to get fuzzy edges
        mask = df["if_sunny"].astype(bool).fillna(False).to_numpy().astype(float)
        w = moving_average_1d(mask, smooth_window)
        w = np.clip(w, 0.0, 1.0)  # already in [0,1]; no ramp needed

    else:
        raise ValueError("Provide kt, kt_col, or set use_mask_as_weight=True.")

    # final blended prediction (shape == len(df))
    y_hat = w * y_s + (1.0 - w) * y_c

    y_hat = pd.Series(y_hat, index=df["time"])
    return y_hat


def moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    """
    Computes a centered 1D moving average with NaN interpolation.

    Smooths input values using a symmetric box filter of given window size.
    Missing values (NaNs) are linearly interpolated before smoothing.

    For a window size :math:`w`, the smoothed output at index :math:`i` is:

    .. math::

        y_i = \\frac{1}{w} \\sum_{j = i - w/2}^{i + w/2} x_j

    where the sum respects array boundaries using convolution mode `"same"`.
    """

    if window is None or window <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    nan = np.isnan(x)
    if nan.any():
        idx = np.arange(x.size)
        x[nan] = np.interp(idx[nan], idx[~nan], x[~nan]) if (~nan).any() else 0.0
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def fuzzy_weight_from_kt(
    k_t: np.ndarray,
    t0: float = 0.50,
    t1: float = 0.70,
    smooth_window: int = 5
) -> np.ndarray:
    """
    Converts clearness index :math:`k_t` into a fuzzy weight using a linear ramp.

    Smooths the input clearness index and maps it into a [0,1] range:

    - :math:`k_t \\leq t_0` → fully cloudy (weight = 0),
    - :math:`k_t \\geq t_1` → fully sunny (weight = 1),
    - Linear interpolation in between.

    The weight is defined as:

    .. math::

        w = \\text{clip}\\left(\\frac{k_t - t_0}{t_1 - t_0},\\ 0,\\ 1\\right)

    and smoothed using a moving average window.
    """

    k_t = np.asarray(k_t, dtype=float).flatten()
    k_t = np.clip(k_t, 0.0, 1.0)
    k_t_s = moving_average_1d(k_t, smooth_window)
    eps = 1e-12
    w = (k_t_s - t0) / max(t1 - t0, eps)
    return np.clip(w, 0.0, 1.0)
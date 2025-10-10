import os
import pandas as pd
import numpy as np
import logging

from pathlib import Path
from typing import TypeAlias, Literal

from pvtools.io_file.writer import save_true_and_predicted_data_to_csv
from pvtools.io_file.reader import (linear_regression_load_parameters, divided_linear_regression_load_parameters,
                                        polynominal_regression_load_parameters, decision_tree_regression_load_parameters, mlp_load_parameters)
from pvtools.calibration.validate_decision_tree import _traverse_tree

log = logging.getLogger("calibrate")

Period_type: TypeAlias = Literal['sunny', 'cloudy']


def calibrate_by_linear_regression(
        df: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str,
) -> None:

    calibration_method_dir = Path(os.path.join(log_dir, folder_data_name, "linear_regression", period))
    log.debug(f"calibration_method_dir:{calibration_method_dir}")

    json_files = list(calibration_method_dir.glob("*.json"))

    if len(json_files) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir}, but found {len(json_files)}.")

    for idx, json_file_dir in enumerate(json_files):
        log.debug(f"fitting json: {json_file_dir}")
        params = linear_regression_load_parameters(json_file_dir)
        time = df["time"]
        x = df[sensor_names[idx]].values
        y_true = df[sensor_name_ref]
        y_pred = linear_regression_use_calibration_values(x, params)

        output_dir = Path(calibration_method_dir)
        file_stem = Path(json_file_dir).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        log.debug(f"csv_filename: {csv_filename}")
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename, index=None,time=time)

def calibrate_by_fuzzy_regression(
        df: pd.DataFrame,
        poa: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str,
) -> None:

    df = df.copy()
    poa = poa.copy()

    calibration_method_dir_sunny = Path(os.path.join(log_dir, folder_data_name, "linear_regression", "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(log_dir, folder_data_name, "linear_regression", "cloudy"))
    log.debug(f"calibration_method_dir:{calibration_method_dir_sunny}")
    log.debug(f"calibration_method_dir:{calibration_method_dir_cloudy}")

    json_files_sunny = list(calibration_method_dir_sunny.glob("*.json"))
    json_files_cloudy = list(calibration_method_dir_cloudy.glob("*.json"))

    if len(json_files_sunny) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_sunny}, but found {len(json_files_sunny)}.")

    if len(json_files_cloudy) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_cloudy}, but found {len(json_files_cloudy)}.")

    time = df["time"]
    y_true = df[sensor_name_ref]

    for i, json_file_dir_sunny in enumerate(json_files_sunny):
        for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
            params_sunny = linear_regression_load_parameters(json_file_dir_sunny)
            params_cloudy = linear_regression_load_parameters(json_file_dir_cloudy)

            #x_sunny = df[df["if_sunny"] == True]
            #x_sunny = x_sunny["time", sensor_names[i]]

            #x_cloudy = df[df["if_sunny"] == False]
            #x_cloudy = x_cloudy["time", sensor_names[j]]

            left = df[["time", sensor_name_ref]]
            right = poa[["time", "poa_global"]]

            merged = left.merge(right, on="time", how="inner").sort_values("time")

            eps = 1e-6
            k_t = merged[sensor_name_ref] / (merged["poa_global"] + eps)

            y_pred = fuzzy_regression_use_calibration_values_df(
                df=df[["time", sensor_names[j], "if_sunny"]],
                sensor_col=sensor_names[j],
                params_sunny=params_sunny,
                params_cloudy=params_cloudy,
                kt=k_t, # uses k_t ramp 0.5→0.7 + smoothing
                kt_col=None,
                t0=0.50,
                t1=0.70,
                smooth_window=5
            )

        output_dir = Path(calibration_method_dir_sunny).parent
        file_stem = Path(json_file_dir_sunny).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        log.debug(f"csv_filename: {csv_filename}")
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename, index=None, time=time)


def calibrate_by_divided_linear_regression(
        df_sunny: pd.DataFrame,
        df_cloudy: pd.DataFrame,
        df_time: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:

    calibration_method_dir = Path(os.path.join(log_dir, folder_data_name, "divided_linear_regression", period))

    json_files = list(calibration_method_dir.glob("*.json"))

    if len(json_files) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir}, but found {len(json_files)}.")

    for idx, json_file_dir in enumerate(json_files):
        params = divided_linear_regression_load_parameters(json_file_dir)

        time = df["time"]
        x = df[sensor_names[idx]].values
        y_true = df[sensor_name_ref]
        y_pred = divided_linear_regression_use_calibration_values(x, df_time, params)

        output_dir = Path(calibration_method_dir)
        file_stem = Path(json_file_dir).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename, index=None,time=time)


def calibrate_by_polynominal_regression(
        df_sunny: pd.DataFrame,
        df_cloudy: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:

    calibration_method_dir = Path(os.path.join(log_dir, folder_data_name, "polynominal_regression", period))

    json_files = list(calibration_method_dir.glob("*.json"))

    if len(json_files) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir}, but found {len(json_files)}.")

    for idx, json_file_dir in enumerate(json_files):
        params = polynominal_regression_load_parameters(json_file_dir)

        time = df["time"]
        x = df[sensor_names[idx]].values
        y_true = df[sensor_name_ref]
        y_pred = polynominal_regression_use_calibration_values(x, params)

        output_dir = Path(calibration_method_dir)
        file_stem = Path(json_file_dir).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename, index=None,time=time)


def calibrate_by_decision_tree_regression(
        df_sunny: pd.DataFrame,
        df_cloudy: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:

    calibration_method_dir = Path(os.path.join(log_dir, folder_data_name, "decision_tree_regression", period))

    json_files = list(calibration_method_dir.glob("*.json"))

    if len(json_files) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir}, but found {len(json_files)}.")

    for idx, json_file_dir in enumerate(json_files):
        params = decision_tree_regression_load_parameters(json_file_dir)

        time = df["time"]
        x = df[sensor_names[idx]].values
        y_true = df[sensor_name_ref]
        y_pred = decision_tree_regression_use_calibration_values(x, params)

        output_dir = Path(calibration_method_dir)
        file_stem = Path(json_file_dir).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename, index=None,time=time)


def calibrate_by_mlp_regression(
        df_sunny: pd.DataFrame,
        df_cloudy: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:

    calibration_method_dir = Path(os.path.join(log_dir, folder_data_name, "mlp_regression", period))

    json_files = list(calibration_method_dir.glob("*.json"))

    if len(json_files) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir}, but found {len(json_files)}.")

    for idx, json_file_dir in enumerate(json_files):
        params = mlp_load_parameters(json_file_dir)

        time = df["time"]
        x = df[sensor_names[idx]].values
        y_true = df[sensor_name_ref]
        y_pred = mlp_use_calibration_values(x, params, activation="relu")

        output_dir = Path(calibration_method_dir)
        file_stem = Path(json_file_dir).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename, index=None,time=time)


def linear_regression_use_calibration_values(
        x: np.ndarray,
        params: dict
) -> np.ndarray:
    a = params["a"]
    b = params["b"]
    x = np.asarray(x).flatten()

    y_pred = a * x + b
    return y_pred

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

def fuzzy_regression_use_calibration_values_df(
    df: pd.DataFrame,
    sensor_col: str,
    params_sunny: dict,
    params_cloudy: dict,
    *,
    # choose ONE of the following to build weights:
    kt: np.ndarray | None = None,         # pass an array aligned to df.index
    kt_col: str | None = None,            # or name of a column in df with k_t
    use_mask_as_weight: bool = False,     # or derive soft weights from 'if_sunny'
    t0: float = 0.50,
    t1: float = 0.70,
    smooth_window: int = 5
) -> np.ndarray:
    """
    Fuzzy blend of two linear models on a single DataFrame.
    y_hat = w*(a_s*x + b_s) + (1-w)*(a_c*x + b_c), aligned to df.index.

    df must contain:
      - sensor_col (e.g., VEML7700 reading)
      - 'if_sunny' (bool) if use_mask_as_weight=True
      - optionally a clearness index column if kt_col is provided
    """
    if sensor_col not in df.columns:
        raise KeyError(f"Missing column: {sensor_col}")

    a_s, b_s = float(params_sunny["a"]), float(params_sunny["b"])
    a_c, b_c = float(params_cloudy["a"]), float(params_cloudy["b"])

    # feature vector (all rows; ensures shapes line up)
    x = np.asarray(df[sensor_col].to_numpy(), dtype=float).flatten()

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
    return y_hat

'''
def fuzzy_regression_use_calibration_values(
        x_sunny: np.ndarray,
        x_cloudy: np.ndarray,
        params_sunny: dict,
        params_cloudy: dict
) -> np.ndarray:
    a_sunny = params_sunny["a"]
    b_sunny = params_sunny["b"]

    a_cloudy = params_cloudy["a"]
    b_cloudy = params_cloudy["b"]

    x_sunny = np.asarray(x_sunny).flatten()
    x_cloudy = np.asarray(x_cloudy).flatten()

    y_pred = a * x + b
    return y_pred
'''

def divided_linear_regression_use_calibration_values(
        x: np.ndarray,
        time: np.ndarray,
        params: dict
) -> np.ndarray:
    intervals = sorted([
        (pd.to_datetime(c["hour"]), c["a"], c["b"])
        for c in params
        if all(k in c for k in ("hour", "a", "b"))
    ], key=lambda x: x[0])

    time = pd.to_datetime(time)

    y_pred = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        t = time[i]
        a, b = 0.0, 0.0

        for j in range(len(intervals)):
            t_start, a_j, b_j = intervals[j]
            t_end = intervals[j + 1][0] if j + 1 < len(intervals) else pd.Timestamp.max.tz_localize('Europe/Warsaw')

            if t_start <= t < t_end:
                a, b = a_j, b_j
                break

        y_pred[i] = a * x[i] + b

    return y_pred


def polynominal_regression_use_calibration_values(
        x: np.ndarray,
        params: dict
) -> np.ndarray:
    a = params["a"]
    b = params["b"]
    c = params["c"]

    x = np.asarray(x).flatten()
    y_pred = a * x ** 2 + b * x + c

    return y_pred


def decision_tree_regression_use_calibration_values(
        x: np.ndarray,
        params: dict
) -> np.ndarray:
    tree = params #params["params"]
    x = np.asarray(x).flatten()
    y_pred = np.array([_traverse_tree(tree, val) for val in x])

    return y_pred


def mlp_use_calibration_values(
        x: np.ndarray,
        params: dict,
        activation: str='relu'
) -> np.ndarray:
    W1 = np.array(params["layer_1_weights"])  # shape (n_inputs, 10)
    b1 = np.array(params["layer_1_biases"])  # (10,)
    W2 = np.array(params["layer_2_weights"])  # shape (10, 10)
    b2 = np.array(params["layer_2_biases"])  # (10,)
    W3 = np.array(params["output_weights"])  # (10, 1)
    b3 = np.array(params["output_biases"])  # (1,)

    x = np.atleast_2d(x).reshape(-1, W1.shape[0])  # x: shape (n_samples, n_inputs)

    # Forward pass through first hidden layer
    z1 = x @ W1 + b1  # shape: (n_samples, n_hidden)
    a1 = _apply_activation(z1, activation)

    # Forward pass through second hidden layer
    z2 = a1 @ W2 + b2
    a2 = _apply_activation(z2, activation) # shape: (h_hidden, n_hidden)

    # Output layer
    output = a2 @ W3 + b3 # shape: (n_samples, 1)

    return output.flatten()


def _apply_activation(z, activation) -> np.ndarray:
    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'tanh':
        return np.tanh(z)
    elif activation == 'identity':
        return z
    else:
        raise ValueError(f"Unsupported activation: {activation}")
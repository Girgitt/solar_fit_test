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
        sensor_name_ref: str,
        load_params_dir: Path,
        save_dir: Path,
        filename: str,
        period_flag: bool = False # if True - periods detected, else not
) -> None:

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

        for i, json_file_dir_sunny in enumerate(json_files_sunny):
            params_sunny = linear_regression_load_parameters(json_file_dir_sunny)

            for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
                params_cloudy = linear_regression_load_parameters(json_file_dir_cloudy)

                log.debug(f"fitting json: {json_file_dir_sunny}")
                log.debug(f"fitting json: {json_file_dir_cloudy}")

                time = df["time"]
                y_true = df[sensor_name_ref]

                y_pred = linear_regression_use_calibration_values(
                    df=df[["time", sensor_names[j], "if_sunny"]],
                    sensor_name=sensor_names[j],
                    params_sunny=params_sunny,
                    params_cloudy=params_cloudy
                )

            output_dir = Path(save_dir)
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=time)

    else:

        for i, json_file_dir_sunny in enumerate(json_files_sunny):
            params_sunny = linear_regression_load_parameters(json_file_dir_sunny)

            log.debug(f"fitting json: {json_file_dir_sunny}")

            time = df["time"]

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
        params_sunny = linear_regression_load_parameters(json_file_dir_sunny)

        for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
            params_cloudy = linear_regression_load_parameters(json_file_dir_cloudy)

            left = df[["time", sensor_name_ref]]
            right = poa[["time", "poa_global"]]

            merged = left.merge(right, on="time", how="inner").sort_values("time")

            eps = 1e-6
            k_t = merged[sensor_name_ref] / (merged["poa_global"] + eps)

            y_pred = fuzzy_regression_use_calibration_values_df(
                df=df[["time", sensor_names[j], "if_sunny"]],
                sensor_name=sensor_names[j],
                params_sunny=params_sunny,
                params_cloudy=params_cloudy,
                kt=k_t, # uses k_t ramp 0.5→0.7 + smoothing
                kt_col=None,
                t0=0.50,
                t1=0.70,
                smooth_window=5
            )

        output_dir = Path(calibration_method_dir_sunny).parent.parent / "fuzzy_regression"
        output_dir.mkdir(parents=True, exist_ok=True)

        file_stem = Path(json_file_dir_sunny).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        log.debug(f"csv_filename: {csv_filename}")
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=time)


def calibrate_by_divided_linear_regression(
        df: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:

    calibration_method_dir_sunny = Path(os.path.join(log_dir, folder_data_name, "divided_linear_regression", "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(log_dir, folder_data_name, "divided_linear_regression", "cloudy"))

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

    for i, json_file_dir_sunny in enumerate(json_files_sunny):
        params_sunny = divided_linear_regression_load_parameters(json_file_dir_sunny)

        for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
            params_cloudy = divided_linear_regression_load_parameters(json_file_dir_cloudy)

            log.debug(f"fitting json: {json_file_dir_sunny}")
            log.debug(f"fitting json: {json_file_dir_cloudy}")

            time = df["time"]
            y_true = df[sensor_name_ref]

            y_pred = divided_linear_regression_use_calibration_values(
                df=df[["time", sensor_names[j], "if_sunny"]],
                sensor_name=sensor_names[j],
                params_sunny=params_sunny,
                params_cloudy=params_cloudy
            )

        output_dir = Path(calibration_method_dir_sunny).parent
        file_stem = Path(json_file_dir_sunny).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        log.debug(f"csv_filename: {csv_filename}")
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=time)

def calibrate_by_polynominal_regression(
        df: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:

    calibration_method_dir_sunny = Path(os.path.join(log_dir, folder_data_name, "polynominal_regression", "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(log_dir, folder_data_name, "polynominal_regression", "cloudy"))

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

    for i, json_file_dir_sunny in enumerate(json_files_sunny):
        params_sunny = polynominal_regression_load_parameters(json_file_dir_sunny)

        for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
            params_cloudy = polynominal_regression_load_parameters(json_file_dir_cloudy)

            log.debug(f"fitting json: {json_file_dir_sunny}")
            log.debug(f"fitting json: {json_file_dir_cloudy}")

            time = df["time"]
            y_true = df[sensor_name_ref]

            y_pred = polynominal_regression_use_calibration_values(
                df=df[["time", sensor_names[j], "if_sunny"]],
                sensor_name=sensor_names[j],
                params_sunny=params_sunny,
                params_cloudy=params_cloudy
            )

        output_dir = Path(calibration_method_dir_sunny).parent
        file_stem = Path(json_file_dir_sunny).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        log.debug(f"csv_filename: {csv_filename}")
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=time)


def calibrate_by_decision_tree_regression(
        df: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:
    calibration_method_dir_sunny = Path(os.path.join(log_dir, folder_data_name, "decision_tree_regression", "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(log_dir, folder_data_name, "decision_tree_regression", "cloudy"))

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

    for i, json_file_dir_sunny in enumerate(json_files_sunny):
        params_sunny = decision_tree_regression_load_parameters(json_file_dir_sunny)

        for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
            params_cloudy = decision_tree_regression_load_parameters(json_file_dir_cloudy)

            log.debug(f"fitting json: {json_file_dir_sunny}")
            log.debug(f"fitting json: {json_file_dir_cloudy}")

            time = df["time"]
            y_true = df[sensor_name_ref]

            y_pred = decision_tree_regression_use_calibration_values(
                df=df[["time", sensor_names[j], "if_sunny"]],
                sensor_name=sensor_names[j],
                params_sunny=params_sunny,
                params_cloudy=params_cloudy
            )

        output_dir = Path(calibration_method_dir_sunny).parent
        file_stem = Path(json_file_dir_sunny).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        log.debug(f"csv_filename: {csv_filename}")
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=time)


def calibrate_by_mlp_regression(
        df: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:

    calibration_method_dir_sunny = Path(os.path.join(log_dir, folder_data_name, "mlp_regression", "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(log_dir, folder_data_name, "mlp_regression", "cloudy"))

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

    for i, json_file_dir_sunny in enumerate(json_files_sunny):
        params_sunny = mlp_load_parameters(json_file_dir_sunny)

        for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
            params_cloudy = mlp_load_parameters(json_file_dir_cloudy)

            log.debug(f"fitting json: {json_file_dir_sunny}")
            log.debug(f"fitting json: {json_file_dir_cloudy}")

            time = df["time"]
            y_true = df[sensor_name_ref]

            y_pred = mlp_use_calibration_values(
                df=df[["time", sensor_names[j], "if_sunny"]],
                sensor_name=sensor_names[j],
                params_sunny=params_sunny,
                params_cloudy=params_cloudy
            )

        output_dir = Path(calibration_method_dir_sunny).parent
        file_stem = Path(json_file_dir_sunny).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        log.debug(f"csv_filename: {csv_filename}")
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=time)


def linear_regression_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny: dict,
        params_cloudy: dict = None
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

    if params_cloudy is not None:
        is_sunny = df["if_sunny"].astype(bool)

        y_pred = pd.Series(index=df.index, dtype=float)
        y_pred[is_sunny] = params_sunny["a"] * x[is_sunny] + params_sunny["b"]
        y_pred[~is_sunny] = params_cloudy["a"] * x[~is_sunny] + params_cloudy["b"]
    else:
        y_pred = params_sunny["a"] * x + params_sunny["b"]

    return y_pred


def fuzzy_regression_use_calibration_values_df(
    df: pd.DataFrame,
    sensor_name: str,
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

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time",
        if_sunny_col="if_sunny"
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
        params_sunny: list[dict],
        params_cloudy: list[dict]
) -> pd.Series:

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time",
        if_sunny_col="if_sunny"
    )

    def build_intervals(param_list: list[dict]) -> list[tuple[pd.Timestamp, float, float]]:
        return sorted(
            [
                (p["hour"], p["a"], p["b"]) for p in param_list
                if all(k in p for k in ("hour", "a", "b"))
            ],
            key=lambda x: x[0],
        )

    intervals_sunny = build_intervals(params_sunny)
    intervals_cloudy = build_intervals(params_cloudy)

    if not intervals_sunny or not intervals_cloudy:
        raise ValueError("Both params_sunny and params_cloudy must contain valid (hour, a, b) entries.")

    x = df[sensor_name]
    time = df["time"]
    is_sunny = df["if_sunny"]
    y_pred = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        current_time = time.iloc[i]
        current_params = intervals_sunny if is_sunny.iloc[i] else intervals_cloudy

        a, b = 0.0, 0.0
        for j, (t_start, a_j, b_j) in enumerate(current_params):
            t_end = (
                current_params[j + 1][0]
                if j + 1 < len(current_params)
                else pd.Timestamp.max
            )
            if t_start <= current_time < t_end:
                a, b = a_j, b_j
                break

        y_pred[i] = a * x.iloc[i] + b

    result = pd.Series(y_pred, index=df.index, name=f"{sensor_name}_calibrated")

    return result


def polynominal_regression_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny: dict,
        params_cloudy: dict
) -> pd.Series:

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time",
        if_sunny_col="if_sunny"
    )

    x = df[sensor_name]
    is_sunny = df["if_sunny"].astype(bool)

    y_pred = pd.Series(index=df.index, dtype=float)

    y_pred[is_sunny] = (
            params_sunny["a"] * x[is_sunny] ** 2
            + params_sunny["b"] * x[is_sunny]
            + params_sunny["c"]
    )

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
        params_cloudy: dict
) -> pd.Series:

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time",
        if_sunny_col="if_sunny"
    )

    x = df[sensor_name].to_numpy().flatten()
    is_sunny = df["if_sunny"].astype(bool).to_numpy()

    if params_sunny is None or params_cloudy is None:
        raise ValueError("Both params_sunny and params_cloudy must contain a 'params' key with a tree structure.")

    y_pred = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        model = params_sunny if is_sunny[i] else params_cloudy
        y_pred[i] = _traverse_tree(model, x[i])

    return pd.Series(y_pred, index=df.index)


def mlp_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny: dict,
        params_cloudy: dict,
        activation: str='relu'
) -> pd.Series:

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time",
        if_sunny_col="if_sunny"
    )

    x = df[sensor_name].to_numpy().reshape(-1, 1)  # shape (n_samples, n_inputs)
    is_sunny = df["if_sunny"].astype(bool).to_numpy()
    y_pred = np.empty_like(x.flatten(), dtype=float)

    activation_sunny = params_sunny["scalers"].get("activation", activation)
    activation_cloudy = params_cloudy["scalers"].get("activation", activation)

    if np.any(is_sunny):
        xs = _scale_in(x[is_sunny], params_sunny["scalers"])
        ys = _forward_pass(xs, params_sunny["coefficients"], activation_sunny).reshape(-1, 1)
        y_pred[is_sunny] = _inv_out(ys, params_sunny["scalers"]).ravel()

    if np.any(~is_sunny):
        xc = _scale_in(x[~is_sunny], params_cloudy["scalers"])
        yc = _forward_pass(xc, params_cloudy["coefficients"], activation_cloudy).reshape(-1, 1)
        y_pred[~is_sunny] = _inv_out(yc, params_cloudy["scalers"]).ravel()

    return pd.Series(y_pred, index=df.index)

def _scale_in(
        x2d: np.ndarray,
        scaler: dict
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
        scaler: dict
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
        coeffs: dict,
        activation: str,
) -> np.ndarray:

    W1 = np.array(coeffs["layer_1_weights"])
    b1 = np.array(coeffs["layer_1_biases"])
    W2 = np.array(coeffs["layer_2_weights"])
    b2 = np.array(coeffs["layer_2_biases"])
    W3 = np.array(coeffs["output_weights"])
    b3 = np.array(coeffs["output_biases"])

    z1 = x @ W1 + b1
    a1 = _apply_activation(z1, activation)

    z2 = a1 @ W2 + b2
    a2 = _apply_activation(z2, activation)

    output = a2 @ W3 + b3

    return output.flatten()


def _apply_activation(
        z: float,
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
        if_sunny_col: str
) -> None:

    if if_sunny_col is not None:
        required_cols = {time_col, if_sunny_col, sensor_name}
    else:
        required_cols = {time_col, sensor_name}

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
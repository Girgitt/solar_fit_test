import inspect
import statistics
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import logging

from pathlib import Path
from typing import Dict, Any
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import _tree
from typing import TypeAlias, Literal

from pvtools.config.params import ModelData, ModelDirectories, ModelTimes
from pvtools.config.sensor_calibration_metrics import SensorCalibrationMetrics
from pvtools.io_file.writer import save_metrics_to_json, save_true_and_predicted_data_to_csv
from pvtools.preprocess.preprocess_data import sanitize_filename

log = logging.getLogger("calculate_calibration_parameters")

Period_type: TypeAlias = Literal['sunny', 'cloudy', 'all']

my_test_size=0.3
my_random_state=42


def linear_regression(
        df: pd.DataFrame,
        period: Period_type,
        model_data: ModelData,
        model_dirs: ModelDirectories,
) -> None:
    """
    Calculates Linear Regression coefficients.

    Saves coefficients ``a``, ``b`` and :class:`Sensor Calibration Metrics
    <pvtools.config.sensor_calibration_metrics.SensorCalibrationMetrics>` to the .json file.

    Saves predicted and true values of the test dataset to .csv file.
    """

    df = df.copy()

    log_dir = model_dirs.log_dir
    filename = model_dirs.filename
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    for idx, sensor_col in enumerate(sensor_names):
        time = df["time"]
        x = df[sensor_col].values.reshape(-1, 1)
        y = df[sensor_name_ref].values

        indices = np.arange(len(df))
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, indices, test_size=my_test_size, random_state=my_random_state
            )

        time_test = time.iloc[idx_test]

        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics = SensorCalibrationMetrics(y_test, y_pred)

        coefficients = {
            "a": float(model.coef_[0]),
            "b": float(model.intercept_)
        }

        function_name = inspect.currentframe().f_code.co_name
        column_name = sanitize_filename(sensor_col)
        filename = sanitize_filename(Path(filename).stem)

        json_metrics_filename = Path(log_dir) / filename / function_name / period / f"{column_name}.json"
        save_metrics_to_json(metrics, len(x), coefficients, json_metrics_filename)

        csv_filename = Path(log_dir) / filename / function_name / period / f"{column_name}_test_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_test, idx_test, time_test)


def divided_linear_regression(
        df: pd.DataFrame,
        period: Period_type,
        model_data: ModelData,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
) -> None:
    """
        Calculates Divided Linear Regression coefficients.

        Saves coefficients ``a``, ``b``, ``hour`` and :class:`Sensor Calibration Metrics
        <pvtools.config.sensor_calibration_metrics.SensorCalibrationMetrics>` to the .json file.

        Saves predicted and true values of the test dataset to .csv file.
    """

    df = df.copy()

    log_dir = model_dirs.log_dir
    filename = model_dirs.filename
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref

    min_samples = 10
    freq = model_times.divided_linear_regression_interval
    df["interval"] = df["time"].dt.floor(freq)

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    for sensor_col in sensor_names:
        metrics_list = []
        coefficients_list = []

        time_test_all = []
        y_test_all = []
        y_pred_all = []
        idx_test_all = []

        for idx, (interval_start, group) in enumerate(df.groupby("interval")):
            n = len(group)

            if n < min_samples:
                log.info(f"Skipping interval_start {interval_start}: only {n} samples (min={min_samples})")
                continue

            x = group[sensor_col].values.reshape(-1, 1)
            y = group[sensor_name_ref].values
            indices = group.index.values
            time = group["time"]

            x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
                x, y, indices, test_size=my_test_size, random_state=my_random_state
            )

            time_test = time.loc[idx_test]

            model_hour = LinearRegression()
            model_hour.fit(x_train, y_train)
            y_pred = model_hour.predict(x_test)

            time_test_all.append(time_test)
            y_test_all.append(y_test)
            y_pred_all.append(y_pred)
            idx_test_all.append(idx_test)

            metrics = SensorCalibrationMetrics(y_test, y_pred)
            metrics_list.append(metrics)

            coefficients_list.append({
                "hour": interval_start.isoformat(),
                "a": float(model_hour.coef_[0]),
                "b": float(model_hour.intercept_)
            })

        time_test_all = pd.concat(time_test_all)
        y_test_all = np.concatenate(y_test_all)
        y_pred_all = np.concatenate(y_pred_all)
        idx_test_all = np.concatenate(idx_test_all)

        y_true_all = np.concatenate([m.y_true for m in metrics_list])
        y_pred_all = np.concatenate([m.y_pred for m in metrics_list])
        avg_metrics = SensorCalibrationMetrics(y_true_all, y_pred_all)

        y_pred_all = pd.Series(y_pred_all)
        y_test_all = pd.Series(y_test_all)
        idx_test_all = pd.Series(idx_test_all)
        time_test_all = pd.Series(time_test_all)

        coefficients_mean = mean_coefficients_by_time(coefficients_list)

        function_name = inspect.currentframe().f_code.co_name
        column_name = sanitize_filename(sensor_col)
        filename = sanitize_filename(Path(filename).stem)
        json_filename = Path(log_dir) / filename / function_name  / period / f"{column_name}.json"
        save_metrics_to_json(avg_metrics, len(x), coefficients_list, json_filename)

        # MEAN VALUES ARE FOR TESTING
        json_filename_mean = Path(log_dir) / filename / f"{function_name}_mean" / period / f"{column_name}.json"
        save_metrics_to_json(avg_metrics, len(x), coefficients_mean, json_filename_mean)

        csv_filename = Path(log_dir) / filename / function_name / period / f"{column_name}_test_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_pred_all, csv_filename, y_test_all, idx_test_all, time_test_all)


def polynominal_regression(
        df: pd.DataFrame,
        period: Period_type,
        model_data: ModelData,
        model_dirs: ModelDirectories,
) -> None:
    """
        Calculates Polynominal Regression coefficients.

        Depending on degree of the model, saves coefficients ``a``, ``b`` etc.. and :class:`Sensor Calibration Metrics
        <pvtools.config.sensor_calibration_metrics.SensorCalibrationMetrics>` to the .json file.

        Saves predicted and true values of the test dataset to .csv file.
    """

    df = df.copy()

    log_dir = model_dirs.log_dir
    filename = model_dirs.filename
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref

    coefficients = []

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    for idx, sensor_col in enumerate(sensor_names):
        time = df["time"]
        x = df[sensor_col].values.reshape(-1, 1)
        y = df[sensor_name_ref].values

        indices = np.arange(len(df))
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, indices, test_size=my_test_size, random_state=my_random_state
            )

        time_test = time.iloc[idx_test]

        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x_train)
        x_test_poly = poly.transform(x_test)

        model = LinearRegression()
        model.fit(x_poly, y_train)
        y_pred = model.predict(x_test_poly)

        metrics = SensorCalibrationMetrics(y_test, y_pred)

        # only in case when degree=2
        coefficients = {
            "a": float(model.coef_[2]),
            "b": float(model.coef_[1]),
            "c": float(model.intercept_)
        }

        function_name = inspect.currentframe().f_code.co_name
        column_name = sanitize_filename(sensor_col)
        filename = sanitize_filename(Path(filename).stem)
        json_filename = Path(log_dir) / filename / function_name / period / f"{column_name}.json"
        save_metrics_to_json(metrics, len(x), coefficients, json_filename)

        csv_filename = Path(log_dir) / filename / function_name / period / f"{column_name}_test_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_test, idx_test, time_test)


def decision_tree_regression(
        df: pd.DataFrame,
        period: Period_type,
        model_data: ModelData,
        model_dirs: ModelDirectories,
) -> None:
    """
        Calculates Decision Tree Regression coefficients.

        Saves tree rules coefficients and :class:`Sensor Calibration Metrics
        <pvtools.config.sensor_calibration_metrics.SensorCalibrationMetrics>` to the .json file.

        Saves predicted and true values of the test dataset to .csv file.
    """

    df = df.copy()

    log_dir = model_dirs.log_dir
    filename = model_dirs.filename
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    for idx, sensor_col in enumerate(sensor_names):
        time = df["time"]
        x = df[sensor_col].values.reshape(-1, 1)
        y = df[sensor_name_ref].values

        indices = np.arange(len(df))
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, indices, test_size=my_test_size, random_state=my_random_state
            )

        time_test = time.iloc[idx_test]

        model = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics = SensorCalibrationMetrics(y_test, y_pred)

        coefficients = {
            "params": export_tree_as_rules(model)
        }

        function_name = inspect.currentframe().f_code.co_name
        column_name = sanitize_filename(sensor_col)
        filename = sanitize_filename(Path(filename).stem)
        json_filename = Path(log_dir) / filename / function_name / period / f"{column_name}.json"
        save_metrics_to_json(metrics, len(x), coefficients, json_filename)

        csv_filename = Path(log_dir) / filename / function_name / period / f"{column_name}_test_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_test, idx_test, time_test)


def mlp_regression(
        df: pd.DataFrame,
        period: Period_type,
        model_data: ModelData,
        model_dirs: ModelDirectories,
) -> None:
    """
        Calculates Multi Layer Perceptron coefficients.

        Saves weights and biases for every layer and :class:`Sensor Calibration Metrics
        <pvtools.config.sensor_calibration_metrics.SensorCalibrationMetrics>` to the .json file.

        Saves predicted and true values of the test dataset to .csv file.
        """

    df = df.copy()

    log_dir = model_dirs.log_dir
    filename = model_dirs.filename
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref

    coefficients = []

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    for idx, sensor_col in enumerate(sensor_names):
        time = df["time"]
        x = df[sensor_col].values.reshape(-1, 1)
        y = df[sensor_name_ref].values

        indices = np.arange(len(df))
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, indices, test_size=my_test_size, random_state=my_random_state, shuffle=True
            )

        time_test = time.iloc[idx_test]

        # scale x and y on train only (not the whole data)
        x_scaler = StandardScaler().fit(x_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        x_train_s = x_scaler.transform(x_train)
        x_test_s = x_scaler.transform(x_test)
        y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()

        model = MLPRegressor(
            loss='squared_error',
            hidden_layer_sizes=(10, 10),
            activation='relu',
            solver='adam',
            learning_rate_init=1e-3,
            alpha=1e-4,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.15,
            max_iter=5000,
            random_state=my_random_state
        )

        model.fit(x_train_s, y_train_s)

        y_pred_s = model.predict(x_test_s).reshape(-1, 1)
        y_pred = y_pred_s * y_scaler.scale_ + y_scaler.mean_
        y_pred = y_pred.ravel()

        metrics = SensorCalibrationMetrics(y_test, y_pred)

        if isinstance(model, MLPRegressor):
            coefficients = {
                "layer_1_weights": model.coefs_[0].tolist(),
                "layer_1_biases": model.intercepts_[0].tolist(),
                "layer_2_weights": model.coefs_[1].tolist(),
                "layer_2_biases": model.intercepts_[1].tolist(),
                "output_weights": model.coefs_[2].tolist(),
                "output_biases": model.intercepts_[2].tolist()
        }
            scalers = {
                "x_scaler_mean": x_scaler.mean_.tolist(),
                "x_scaler_scale": x_scaler.scale_.tolist(),
                "y_scaler_mean": y_scaler.mean_.tolist(),
                "y_scaler_scale": y_scaler.scale_.tolist(),
                "activation": model.activation
            }

        function_name = inspect.currentframe().f_code.co_name
        column_name = sanitize_filename(sensor_col)
        filename = sanitize_filename(Path(filename).stem)
        json_filename = Path(log_dir) / filename / function_name / period / f"{column_name}.json"
        save_metrics_to_json(metrics, len(x), coefficients, json_filename, scalers)

        csv_filename = Path(log_dir) / filename / function_name / period / f"{column_name}_test_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_test, idx_test, time_test)


def export_tree_as_rules(model: DecisionTreeRegressor) -> Dict[str, Any]:
    tree_ = model.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    value = tree_.value

    def recurse(node: int) -> Dict[str, Any]:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            return {
                "feature": int(feature[node]),
                "threshold": float(threshold[node]),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node])
            }
        else:
            return {
                "value": float(value[node][0][0])
            }

    return recurse(0)

def mean_coefficients_by_time(
        coeffs_list: list[dict]
) -> list[dict]:

    grouped = defaultdict(lambda: {"a": [], "b": []})

    for item in coeffs_list:
        time_of_day = datetime.fromisoformat(item["hour"]).strftime("%H:%M")
        grouped[time_of_day]["a"].append(item["a"])
        grouped[time_of_day]["b"].append(item["b"])

    mean_result = [
        {
            "hour": time,
            "a": statistics.mean(values["a"]),
            "b": statistics.mean(values["b"]),
            "count_days": len(values["a"]),
        }
        for time, values in sorted(grouped.items())
    ]

    return mean_result

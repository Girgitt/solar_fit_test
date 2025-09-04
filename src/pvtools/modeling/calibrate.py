import inspect
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import _tree

from pvtools.config.sensor_calibration_metrics import SensorCalibrationMetrics
from pvtools.io_file.writer import save_metrics_to_json, save_true_and_predicted_data_to_csv
from pvtools.preprocess.preprocess_data import sanitize_filename, normalize_values

'''
MAE does not indicate whether the model overestimates or underestimates values
MSE is particularly sensitive to large errors
RMSE same units as the predicted values
R2 correlation between two datasets
MAPE especially useful when you want to assess the accuracy of predictions in percentage
Bias shows whether the model regularly over- or under-predicts
'''

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

def linear_regression(
        df: pd.DataFrame,
        log_dir: Path,
        data_filename_dir: Path,
        sensor_names: list[str] = None,
        sensor_name_ref: str = None,
) -> None:
    df = df.copy()

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    for idx, sensor_col in enumerate(sensor_names):
        x = df[sensor_col].values.reshape(-1, 1)
        y = df[sensor_name_ref].values

        indices = np.arange(len(df))
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, indices, test_size=0.2, random_state=42
            )

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
        data_filename = sanitize_filename(Path(data_filename_dir).stem)

        json_metrics_filename = Path(log_dir) / data_filename / function_name / f"{column_name}.json"
        save_metrics_to_json(metrics, len(x), coefficients, json_metrics_filename)

        csv_filename = Path(log_dir) / data_filename / function_name / f"{column_name}_test_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_test, y_pred, csv_filename, idx_test)

def divided_linear_regression(
        df: pd.DataFrame,
        log_dir: Path,
        data_filename_dir: Path,
        sensor_names: list[str] = None,
        sensor_name_ref: str = None,
) -> None:
    df = df.copy()
    df["hour"] = df["time"].dt.floor("h")
    min_samples = 10

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    for sensor_col in sensor_names:
        metrics_list = []
        coefficients_list = []

        y_test_all_hours = []
        y_pred_all_hours = []
        idx_test_all_hours = []

        for idx, (hour, group) in enumerate(df.groupby("hour")):
            n = len(group)

            # skip too-small groups
            if n < min_samples:
                print(f"[INFO] Skipping hour {hour}: only {n} samples (< min_samples={min_samples})")
                continue

            x = group[sensor_col].values.reshape(-1, 1)
            y = group[sensor_name_ref].values

            indices = group.index.values
            x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
                x, y, indices, test_size=0.2, random_state=42
            )

            model_hour = LinearRegression()
            model_hour.fit(x_train, y_train)
            y_pred = model_hour.predict(x_test)

            y_test_all_hours.append(y_test)
            y_pred_all_hours.append(y_pred)
            idx_test_all_hours.append(idx_test)

            metrics = SensorCalibrationMetrics(y_test, y_pred)
            metrics_list.append(metrics)

            coefficients_list.append({
                "hour": hour.isoformat(),
                "a": float(model_hour.coef_[0]),
                "b": float(model_hour.intercept_)
            })

        y_test_all_hours = np.concatenate(y_test_all_hours)
        y_pred_all_hours = np.concatenate(y_pred_all_hours)
        idx_test_all_hours = np.concatenate(idx_test_all_hours)

        y_true_all = np.concatenate([m.y_true for m in metrics_list])
        y_pred_all = np.concatenate([m.y_pred for m in metrics_list])
        avg_metrics = SensorCalibrationMetrics(y_true_all, y_pred_all)

        function_name = inspect.currentframe().f_code.co_name
        column_name = sanitize_filename(sensor_col)
        data_filename = sanitize_filename(Path(data_filename_dir).stem)
        json_filename = Path(log_dir) / data_filename / function_name / f"{column_name}.json"
        save_metrics_to_json(avg_metrics, len(x), coefficients_list, json_filename)

        csv_filename = Path(log_dir) / data_filename / function_name / f"{column_name}_test_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_test_all_hours, y_pred_all_hours, csv_filename, idx_test_all_hours)

def polynominal_regression(
        df: pd.DataFrame,
        log_dir: Path,
        data_filename_dir: Path,
        sensor_names: list[str] = None,
        sensor_name_ref: str = None,
) -> None:
    df = df.copy()
    coefficients = []

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    for idx, sensor_col in enumerate(sensor_names):
        x = df[sensor_col].values.reshape(-1, 1)
        y = df[sensor_name_ref].values

        indices = np.arange(len(df))
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, indices, test_size=0.2, random_state=42
            )

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
        data_filename = sanitize_filename(Path(data_filename_dir).stem)
        json_filename = Path(log_dir) / data_filename / function_name / f"{column_name}.json"
        save_metrics_to_json(metrics, len(x), coefficients, json_filename)

        csv_filename = Path(log_dir) / data_filename / function_name / f"{column_name}_test_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_test, y_pred, csv_filename, idx_test)

def decision_tree_regression(
        df: pd.DataFrame,
        log_dir: Path,
        data_filename_dir: Path,
        sensor_names: list[str] = None,
        sensor_name_ref: str = None,
) -> None:
    df = df.copy()

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    for idx, sensor_col in enumerate(sensor_names):
        x = df[sensor_col].values.reshape(-1, 1)
        y = df[sensor_name_ref].values

        indices = np.arange(len(df))
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, indices, test_size=0.2, random_state=42
            )

        model = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics = SensorCalibrationMetrics(y_test, y_pred)

        coefficients = {
            "params": export_tree_as_rules(model)
        }

        function_name = inspect.currentframe().f_code.co_name
        column_name = sanitize_filename(sensor_col)
        data_filename = sanitize_filename(Path(data_filename_dir).stem)
        json_filename = Path(log_dir) / data_filename / function_name / f"{column_name}.json"
        save_metrics_to_json(metrics, len(x), coefficients, json_filename)

        csv_filename = Path(log_dir) / data_filename / function_name / f"{column_name}_test_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_test, y_pred, csv_filename, idx_test)

def mlp_regression(
        df: pd.DataFrame,
        log_dir: Path,
        data_filename_dir: Path,
        sensor_names: list[str] = None,
        sensor_name_ref: str = None,
) -> None:
    df = df.copy()
    coefficients = []

    df = normalize_values(df)

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    for idx, sensor_col in enumerate(sensor_names):
        x = df[sensor_col].values.reshape(-1, 1)
        y = df[sensor_name_ref].values

        indices = np.arange(len(df))
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, indices, test_size=0.2, random_state=42
            )

        model = MLPRegressor(
            loss='squared_error',
            hidden_layer_sizes=(10, 10),
            activation='relu'
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

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

        function_name = inspect.currentframe().f_code.co_name
        column_name = sanitize_filename(sensor_col)
        data_filename = sanitize_filename(Path(data_filename_dir).stem)
        json_filename = Path(log_dir) / data_filename / function_name / f"{column_name}.json"
        save_metrics_to_json(metrics, len(x), coefficients, json_filename)

        csv_filename = Path(log_dir) / data_filename / function_name / f"{column_name}_test_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_test, y_pred, csv_filename, idx_test)

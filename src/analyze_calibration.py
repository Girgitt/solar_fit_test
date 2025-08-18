import json

from model_params import *
from typing import Dict, List, Any
from save_functions import save_true_and_predicted_data_to_csv

def linear_regression_load_parameters(calibration_method_dir: Path) -> Dict[str, float]:
    with open(calibration_method_dir, 'r') as f:
        data = json.load(f)

    if "coefficients" not in data or not data["coefficients"]:
        raise ValueError("JSON file does not contain 'coefficients' list.")

    params = data["coefficients"]

    required_keys = ["a", "b"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing key '{key}' in coefficients.")

    return params

def divided_linear_regression_load_parameters(calibration_method_dir: Path) -> List[DatatypeCoefficientsForDividedLinearRegression]:
    with open(calibration_method_dir, 'r') as f:
        data = json.load(f)

    if "coefficients" not in data or not data["coefficients"]:
        raise ValueError("JSON file does not contain 'coefficients' list.")

    params = data["coefficients"][:]

    required_keys = ["hour", "a", "b"]
    for idx, c in enumerate(data["coefficients"]):
        for key in required_keys:
            if key not in c:
                raise ValueError(f"Missing key '{key}' in coefficients at index {idx}: {c}")

    return params

def polynominal_regression_load_parameters(calibration_method_dir: Path) -> Dict[str, float]:
    with open(calibration_method_dir, 'r') as f:
        data = json.load(f)

    if "coefficients" not in data or not data["coefficients"]:
        raise ValueError("JSON file does not contain 'coefficients' list.")

    params = data["coefficients"]

    required_keys = ["a", "b", "c"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing key '{key}' in coefficients.")

    return params

def decision_tree_regression_load_parameters(calibration_method_dir: Path) -> Dict[str, Any]:
    with open(calibration_method_dir, 'r') as f:
        data = json.load(f)

    if "coefficients" not in data or not data["coefficients"]:
        raise ValueError("JSON file does not contain 'coefficients' list.")

    params = data["coefficients"]

    if "params" not in params:
        raise ValueError("Missing 'params' key in 'coefficients'.")

        # Recursively validate the tree structure
    _validate_tree_structure(params["params"])

    return params["params"]

def mlp_load_parameters(calibration_method_dir: Path) -> DatatypeCoefficientsForMLPRegression:
    with open(calibration_method_dir, 'r') as f:
        data = json.load(f)

    if "coefficients" not in data or not data["coefficients"]:
        raise ValueError("JSON file does not contain 'coefficients' list.")

    params = data["coefficients"]

    required_keys = ["layer_1_weights", "layer_1_biases", "layer_2_weights", "layer_2_biases", "output_weights", "output_biases"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing key '{key}' in coefficients.")

    return params

def _validate_tree_structure(node: dict) -> None:
    if "value" in node:
        return  # it's a leaf, that's fine

    required_keys = ["feature", "threshold", "left", "right"]
    for key in required_keys:
        if key not in node:
            raise ValueError(f"Missing key '{key}' in internal node: {node}")

    # Recursively validate children
    _validate_tree_structure(node["left"])
    _validate_tree_structure(node["right"])

def _traverse_tree(node: dict, x_val: float) -> float:
    if "value" in node:
        return node["value"]

    threshold = node["threshold"]
    feature = node["feature"]  # always 0 in this case

    if x_val <= threshold:
        return _traverse_tree(node["left"], x_val)
    else:
        return _traverse_tree(node["right"], x_val)

def _apply_activation(z, activation) -> np.ndarray:
    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'tanh':
        return np.tanh(z)
    elif activation == 'identity':
        return z
    else:
        raise ValueError(f"Unsupported activation: {activation}")

def linear_regression_calculate_calibration_values(x: np.ndarray, params: dict) -> np.ndarray:
    a = params["a"]
    b = params["b"]
    x = np.asarray(x).flatten()

    y_pred = a * x + b
    return y_pred

def divided_linear_regression_calculate_calibration_values(x: np.ndarray, time: np.ndarray, params: dict) -> np.ndarray:
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
            t_end = intervals[j + 1][0] if j + 1 < len(intervals) else pd.Timestamp.max

            if t_start <= t < t_end:
                a, b = a_j, b_j
                break

        y_pred[i] = a * x[i] + b

    return y_pred

def polynominal_regression_calculate_calibration_values(x: np.ndarray, params: dict) -> np.ndarray:
    a = params["a"]
    b = params["b"]
    c = params["c"]

    x = np.asarray(x).flatten()
    y_pred = a * x ** 2 + b * x + c

    return y_pred

def decision_tree_regression_calculate_calibration_values(x: np.ndarray, params: dict) -> np.ndarray:
    tree = params #params["params"]
    x = np.asarray(x).flatten()
    y_pred = np.array([_traverse_tree(tree, val) for val in x])

    return y_pred

def mlp_calculate_calibration_values(x: np.ndarray, params: dict, activation: str='relu') -> np.ndarray:
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

def calibrate_by_linear_regression(
        df: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str,
) -> None:
    calibration_method_dir = log_dir / folder_data_name / "linear_regression"

    for idx, json_file_dir in enumerate(calibration_method_dir.glob("*.json")):
        params = linear_regression_load_parameters(json_file_dir)

        x = df[sensor_names[idx]].values
        y_true = df[sensor_name_ref]
        y_pred = linear_regression_calculate_calibration_values(x, params)

        output_dir = Path(json_file_dir).parent
        file_stem = Path(json_file_dir).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        print(csv_filename)
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename)

def calibrate_by_divided_linear_regression(
        df: pd.DataFrame,
        df_time: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:
    calibration_method_dir = log_dir / folder_data_name / "divided_linear_regression"

    for idx, json_file_dir in enumerate(calibration_method_dir.glob("*.json")):
        params = divided_linear_regression_load_parameters(json_file_dir)

        x = df[sensor_names[idx]].values
        y_true = df[sensor_name_ref]
        y_pred = divided_linear_regression_calculate_calibration_values(x, df_time.values, params)

        output_dir = Path(json_file_dir).parent
        file_stem = Path(json_file_dir).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename)

def calibrate_by_polynominal_regression(
        df: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:
    calibration_method_dir = log_dir / folder_data_name / "polynominal_regression"

    for idx, json_file_dir in enumerate(calibration_method_dir.glob("*.json")):
        params = polynominal_regression_load_parameters(json_file_dir)

        x = df[sensor_names[idx]].values
        y_true = df[sensor_name_ref]
        y_pred = polynominal_regression_calculate_calibration_values(x, params)

        output_dir = Path(json_file_dir).parent
        file_stem = Path(json_file_dir).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename)

def calibrate_by_decision_tree_regression(
        df: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:
    calibration_method_dir = log_dir / folder_data_name / "decision_tree_regression"

    for idx, json_file_dir in enumerate(calibration_method_dir.glob("*.json")):
        params = decision_tree_regression_load_parameters(json_file_dir)

        x = df[sensor_names[idx]].values
        y_true = df[sensor_name_ref]
        y_pred = decision_tree_regression_calculate_calibration_values(x, params)

        output_dir = Path(json_file_dir).parent
        file_stem = Path(json_file_dir).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename)

def calibrate_by_mlp_regression(
        df: pd.DataFrame,
        sensor_names: np.ndarray,
        sensor_name_ref: np.ndarray,
        log_dir: Path,
        folder_data_name: str
) -> None:

    calibration_method_dir = log_dir / folder_data_name / "mlp_regression"

    for idx, json_file_dir in enumerate(calibration_method_dir.glob("*.json")):
        params = mlp_load_parameters(json_file_dir)

        x = df[sensor_names[idx]].values
        y_true = df[sensor_name_ref]
        y_pred = mlp_calculate_calibration_values(x, params, activation="relu")

        output_dir = Path(json_file_dir).parent
        file_stem = Path(json_file_dir).stem
        csv_filename = output_dir / f"{file_stem}_all_true_vs_pred.csv"
        save_true_and_predicted_data_to_csv(y_true, y_pred, csv_filename)






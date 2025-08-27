from typing import Dict, Any

import json

from model_params import *
from validate_decision_tree import _validate_tree_structure

def load_true_and_predicted_data_for_all_methods(calibration_method_dirs: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    all_data = {}

    for method_dir in calibration_method_dirs.iterdir():
        if method_dir.is_dir():
            method_name = method_dir.name
            method_data = {}
            for csv_file in method_dir.glob("*all_true_vs_pred.csv"):
                sensor_name = csv_file.stem.replace("_all_true_vs_pred", "")
                method_data[sensor_name] = pd.read_csv(csv_file)
            all_data[method_name] = method_data

    return all_data

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
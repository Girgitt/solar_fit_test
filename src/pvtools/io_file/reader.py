import json
import pandas as pd

from typing import Dict, Any, List, TypeAlias, Literal
from pathlib import Path

from pvtools.config.params import DatatypeCoefficientsForMLPRegression, DatatypeCoefficientsForDividedLinearRegression
from pvtools.calibration.validate_decision_tree import _validate_tree_structure
from pvtools.config.params import ModelParameters
from pvtools.preprocess.preprocess_data import sanitize_filename

Period_type: TypeAlias = Literal['sunny', 'cloudy']

def load_and_merge_calibrated_data_from_each_sensor(
        df: pd.DataFrame,
        model_parameters: ModelParameters,
        period: Period_type
) -> pd.DataFrame:

    def create_dataframe_from_csv(
            calibration_name: str,
            col: list = None,
    ) -> list:

        for s_name in model_parameters.sensor_names:
            sanitized_name = sanitize_filename(s_name)
            tmp_df = load_dataframe_from_csv(Path(directory / calibration_name / period / f"{sanitized_name}_all_true_vs_pred.csv"))
            col.append(tmp_df['y_pred'].rename(sanitized_name))

        return col

    directory = Path(model_parameters.log_dir / model_parameters.filename)

    col = []
    df_calibrated = []
    col.append(df["time"])

    if model_parameters.args.calibration == "linear":
        df_calibrated = create_dataframe_from_csv("linear_regression", col)

    elif model_parameters.args.calibration == "divided_linear":
        df_calibrated = create_dataframe_from_csv("divided_linear_regression", col)

    elif model_parameters.args.calibration == "decision_tree":
        df_calibrated = create_dataframe_from_csv("decision_tree_regression", col)

    elif model_parameters.args.calibration == "poly":
        df_calibrated = create_dataframe_from_csv("polynominal_regression", col)

    elif model_parameters.args.calibration == "mlp":
        df_calibrated = create_dataframe_from_csv("mlp_regression", col)

    df_calibrated.append(df[model_parameters.sensor_name_ref].rename(
        sanitize_filename(model_parameters.sensor_name_ref)))

    result_df = pd.concat(df_calibrated, axis=1)

    if period == 'sunny':
        result_df['if_sunny'] = True
    elif period == 'cloudy':
        result_df['if_sunny'] = False

    return result_df


def load_dataframe_from_csv(load_path: Path = None) -> pd.DataFrame:
    load_path = Path(load_path)

    if load_path.suffix == "":
        load_path = load_path.with_suffix(".csv")
    elif load_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got '{load_path.suffix}' in path: {load_path}")

    return pd.read_csv(load_path)


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
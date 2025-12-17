import pandas as pd
import json
import csv

from typing import Dict, Any, List, TypeAlias, Literal, Optional
from pathlib import Path

from pvtools.config.params import DatatypeMLPRegressionParameters, DatatypeCoefficientsForDividedLinearRegression
from pvtools.calibration.calibrate_to_reference.validate_tree import _validate_tree_structure
from pvtools.preprocess.preprocess_data import sanitize_filename

Period_type: TypeAlias = Literal['sunny', 'cloudy']


def load_and_merge_calibrated_data_from_each_sensor(
        df: pd.DataFrame,
        sensor_names: list[str],
        calibration_directory: Path,
        sensor_name_ref: Optional[str] = None
) -> pd.DataFrame:

    """
    Combine per-sensor calibration outputs into a single time-aligned dataframe.
    """

    df = df.copy().reset_index(drop=True)

    merged_df = pd.DataFrame()

    for s_name in sensor_names:
        sanitized_name = sanitize_filename(s_name)
        csv_path = calibration_directory / f"{sanitized_name}_all_predicted.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"[ERROR] File not found: {csv_path}")

        tmp_df = pd.read_csv(csv_path)

        if "y_pred" not in tmp_df.columns or "time" not in tmp_df.columns:
            raise KeyError(f"[ERROR] File {csv_path} must contain 'y_pred' and 'time' columns")

        tmp_df["time"] = pd.to_datetime(tmp_df["time"])
        tmp_df.rename(columns={"y_pred": sanitized_name}, inplace=True)

        if merged_df.empty:
            merged_df = tmp_df[["time", sanitized_name]]
        else:
            merged_df = pd.merge(merged_df, tmp_df[["time", sanitized_name]], on="time", how="outer")

    if sensor_name_ref is not None:
        sanitized_ref = sanitize_filename(sensor_name_ref)
        if sensor_name_ref not in df.columns:
            raise KeyError(f"[ERROR] Reference sensor '{sensor_name_ref}' not found in original dataframe")
        merged_df[sanitized_ref] = df[sensor_name_ref].values

    merged_df.sort_values(by="time", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df

def load_dataframe_from_csv(load_path: Path = None) -> pd.DataFrame:

    """
    Read a CSV file into a DataFrame, enforcing the ``.csv`` suffix.
    """

    load_path = Path(load_path)

    if load_path.suffix == "":
        load_path = load_path.with_suffix(".csv")
    elif load_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got '{load_path.suffix}' in path: {load_path}")

    return pd.read_csv(load_path)


def load_true_and_predicted_data_for_all_methods(calibration_method_dirs: Path) -> Dict[str, Dict[str, pd.DataFrame]]:

    """
    Load true versus predicted datasets for every calibration method from disk.
    """

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


def linear_regression_load_parameters(calibration_method_dir: Path) -> dict[str, float]:
    """
    Loads Linear Regression coefficients from .json file and returns dictionary of name and its value.
    """

    with open(calibration_method_dir, 'r') as f:
        data = json.load(f)

    coeffs: str = "coefficients"
    if coeffs not in data or not data[coeffs]:
        raise ValueError(f"JSON file does not contain {coeffs} list.")

    params = data[coeffs]

    required_keys = ["a", "b"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing key '{key}' in {coeffs}")

    return params


def divided_linear_regression_load_parameters(
        calibration_method_dir: Path
) -> List[DatatypeCoefficientsForDividedLinearRegression]:
    """
    Loads Divided Linear Regression coefficients from .json file and returns list of
    DatatypeCoefficientsForDividedLinearRegression.
    """

    with open(calibration_method_dir, 'r') as f:
        data = json.load(f)

    coeffs: str = "coefficients"
    if coeffs not in data or not data[coeffs]:
        raise ValueError(f"JSON file does not contain {coeffs} list.")

    params = data[coeffs][:]

    required_keys = ["hour", "a", "b"]
    for idx, c in enumerate(data[coeffs]):
        for key in required_keys:
            if key not in c:
                raise ValueError(f"Missing key '{key}' in {coeffs} at index {idx}: {c}")

    return params


def polynominal_regression_load_parameters(calibration_method_dir: Path) -> dict[str, float]:
    """
    Loads Polynominal Regression coefficients from .json file and returns dictionary of name and its value.
    """

    with open(calibration_method_dir, 'r') as f:
        data = json.load(f)

    coeffs: str = "coefficients"
    if coeffs not in data or not data[coeffs]:
        raise ValueError(f"JSON file does not contain {coeffs} list.")

    params = data[coeffs]

    required_keys = ["a", "b", "c"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing key '{key}' in {coeffs}")

    return params


def decision_tree_regression_load_parameters(calibration_method_dir: Path) -> dict[str, Any]:
    """
    Loads Decision Tree Regression coefficients from .json file and returns dictionary of name and its value (Any).
    """

    with open(calibration_method_dir, 'r') as f:
        data = json.load(f)

    coeffs: str = "coefficients"
    if coeffs not in data or not data[coeffs]:
        raise ValueError(f"JSON file does not contain {coeffs} list.")

    params = data[coeffs]

    if "params" not in params:
        raise ValueError(f"Missing 'params' key in {coeffs}")

        # Recursively validate the tree structure
    _validate_tree_structure(params["params"])

    return params["params"]


def mlp_load_parameters(calibration_method_dir: Path) -> DatatypeMLPRegressionParameters:
    """
    Loads Multi Layer Perceptron coefficients from .json file and returns the DatatypeMLPRegressionParameters.
    """

    with open(calibration_method_dir, 'r') as f:
        data = json.load(f)

    coeffs: str = "coefficients"
    if coeffs not in data or not data[coeffs]:
        raise ValueError(f"JSON file does not contain {coeffs} list.")

    coeff_params = data.get(coeffs)

    required_keys = ["layer_1_weights", "layer_1_biases", "layer_2_weights", "layer_2_biases", "output_weights", "output_biases"]
    for key in required_keys:
        if key not in coeff_params:
            raise ValueError(f"Missing key '{key}' in {coeffs}")

    scalers: str = "scalers"
    if scalers not in data or not data[scalers]:
        raise ValueError(f"JSON file does not contain {scalers} list.")

    scaler_params = data.get(scalers)

    required_keys = ["x_scaler_mean", "x_scaler_scale", "y_scaler_mean", "y_scaler_scale", "activation"]
    for key in required_keys:
        if key not in scaler_params:
            raise ValueError(f"Missing key '{key}' in {scalers}")

    params: DatatypeMLPRegressionParameters = {
        "coefficients": coeff_params,
        "scalers": scaler_params
    }

    return params


def load_str_dict_from_csv(load_path: Path = None) -> dict[str, str]:

    """
    Restore a string-to-string mapping saved as a CSV with ``key`` and ``value`` columns.
    """

    load_path = Path(load_path)

    if load_path.suffix == "":
        load_path = load_path.with_suffix(".csv")
    elif load_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got '{load_path.suffix}' in path: {load_path}")

    with load_path.open(mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        result = {row['key']: row['value'] for row in reader}

    return result
import os
import pandas as pd
import numpy as np
import logging

from pathlib import Path
from typing import TypeAlias, Literal

from pvtools.io_file.writer import save_true_and_predicted_data_to_csv
from pvtools.io_file.reader import mlp_load_parameters
from pvtools.config.params import (DatatypeCoefficientsForMLPRegression, DatatypeMLPRegressionParameters,
                                   DatatypeScalersForMLPRegression, ModelData, ModelDirectories)
from pvtools.calibration.calibrate_to_reference.calibration_utils import check_if_any_column_is_missing


log = logging.getLogger("calibrate")
Period_type: TypeAlias = Literal['sunny', 'cloudy']


def calibrate_by_mlp_regression(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        period_flag: bool = True  # if True - periods detected, else not
) -> None:
    """
    Calibrate Multi Layer Perceptron model.

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

    mlp_regression = "mlp_regression"
    calibration_method_dir_sunny = Path(os.path.join(load_params_dir, mlp_regression, "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(load_params_dir, mlp_regression, "cloudy"))

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
            params_sunny = mlp_load_parameters(json_file_dir_sunny)

            for j, json_file_dir_cloudy in enumerate(json_files_cloudy):
                params_cloudy = mlp_load_parameters(json_file_dir_cloudy)

                log.debug(f"fitting json: {json_file_dir_sunny}")
                log.debug(f"fitting json: {json_file_dir_cloudy}")

                y_pred = mlp_use_calibration_values(
                    df=df[["time", sensor_names[j], "if_sunny"]],
                    sensor_name=sensor_names[j],
                    params_sunny=params_sunny,
                    params_cloudy=params_cloudy
                )

            output_dir = Path(save_dir) / filename / mlp_regression
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)

    else:
        df["if_sunny"] = True

        for i, json_file_dir_sunny in enumerate(json_files_sunny):
            params_sunny = mlp_load_parameters(json_file_dir_sunny)

            log.debug(f"fitting json: {json_file_dir_sunny}")

            if sensor_name_ref is not None:
                y_true = df[sensor_name_ref]
            else:
                y_true = None

            y_pred = mlp_use_calibration_values(
                df=df[["time", sensor_names[i], "if_sunny"]],
                sensor_name=sensor_names[i],
                params_sunny=params_sunny,
                params_cloudy=None
            )

            output_dir = Path(save_dir) / filename / mlp_regression
            file_stem = Path(json_file_dir_sunny).stem
            csv_filename = output_dir / f"{file_stem}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)


def mlp_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny: DatatypeMLPRegressionParameters,
        params_cloudy: DatatypeMLPRegressionParameters | None = None,
        activation: str = 'relu'
) -> pd.Series:
    """
    Do a calculation of Multi Layer Perceptron using calibration values.

    This function performs inference with a Multi-Layer Perceptron (MLP)
    regression model to transform raw sensor measurements into calibrated
    physical values. The calibration is applied conditionally, depending on
    atmospheric state (e.g., sunny vs. cloudy), allowing different neural
    network parameter sets to be used for distinct regimes.

    The workflow consists of:
    1. Validating required input columns.
    2. Splitting samples into regime-specific subsets (sunny / non-sunny).
    3. Applying input normalization consistent with model training.
    4. Executing the MLP forward pass for each subset.
    5. Applying inverse output scaling to recover physical units.
    6. Merging predictions back into a time-aligned output series.

    Mathematically, for each sample :math:`x(t)`, the calibrated output is:

    .. math::

        \\hat{y}(t) = g^{-1}\\left(
        f_\\theta\\left(g(x(t))\\right)
        \\right)

    where:

    * :math:`g(\\cdot)` denotes input standardization
    * :math:`f_\\theta(\\cdot)` is the MLP regression model
    * :math:`g^{-1}(\\cdot)` denotes inverse output scaling
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

    x = df[sensor_name].to_numpy().reshape(-1, 1)  # shape (n_samples, n_inputs)
    is_sunny = df["if_sunny"].astype(bool).to_numpy()
    y_pred = np.empty_like(x.flatten(), dtype=float)

    activation_sunny = params_sunny["scalers"].get("activation", activation)

    if params_cloudy is not None:
        activation_cloudy = params_cloudy["scalers"].get("activation", activation)

    if np.any(is_sunny):
        xs = scale_in(x[is_sunny], params_sunny["scalers"])
        ys = forward_pass(xs, params_sunny["coefficients"], activation_sunny).reshape(-1, 1)
        y_pred[is_sunny] = inv_out(ys, params_sunny["scalers"]).ravel()

    if np.any(~is_sunny):
        xc = scale_in(x[~is_sunny], params_cloudy["scalers"])
        yc = forward_pass(xc, params_cloudy["coefficients"], activation_cloudy).reshape(-1, 1)
        y_pred[~is_sunny] = inv_out(yc, params_cloudy["scalers"]).ravel()

    return pd.Series(y_pred, index=df["time"])

def scale_in(
        x2d: np.ndarray,
        scaler: DatatypeScalersForMLPRegression
) -> np.ndarray:
    """
    Apply affine input normalization consistent with MLP training.

    This function performs feature-wise standardization of input data prior to
    neural network inference. The transformation is defined as:

    .. math::

        x_\\text{scaled} = \\frac{x - \\mu_x}{\\sigma_x}

    where :math:`\\mu_x` and :math:`\\sigma_x` are the mean and scale parameters
    estimated during model training.

    If scaling parameters are not available, the input is passed through
    unchanged. Zero-valued scale factors are safely replaced to preserve
    numerical stability.
    """

    mean = scaler.get("x_scaler_mean", None)
    scale = scaler.get("x_scaler_scale", None)

    if mean is not None and scale is not None:
        mean = np.asarray(mean)
        scale = np.asarray(scale)
        scale = np.where(scale == 0, 1.0, scale)
        return (x2d - mean) / scale

    return x2d


def inv_out(
        y: np.ndarray,
        scaler: DatatypeScalersForMLPRegression
) -> np.ndarray:
    """
    Apply inverse affine transformation to MLP regression outputs.

    This function restores neural network outputs from normalized space back
    to physical units using the inverse of the training-time scaling:

    .. math::

        y = y_\\text{scaled} \\cdot \\sigma_y + \\mu_y

    where :math:`\\mu_y` and :math:`\\sigma_y` are the output mean and scale
    parameters learned during training.

    If no output scaling parameters are defined, the output is returned
    unchanged.
    """

    mean = scaler.get("y_scaler_mean", None)
    scale = scaler.get("y_scaler_scale", None)

    if mean is not None and scale is not None:
        mean = np.asarray(mean)
        scale = np.asarray(scale)
        return y * scale + mean

    return y


def forward_pass(
        x: np.ndarray,
        coeffs: DatatypeCoefficientsForMLPRegression,
        activation: str,
) -> np.ndarray:
    """
    Execute the forward pass of a fully connected MLP regression model.

    This function evaluates a fixed-architecture Multi-Layer Perceptron with
    two hidden layers and one linear output layer. The computation consists of
    successive affine transformations followed by element-wise nonlinear
    activations.

    The forward propagation is defined as:

    .. math::

        \\mathbf{z}^{(1)} = \\mathbf{X}\\mathbf{W}^{(1)} + \\mathbf{b}^{(1)}

    .. math::

        \\mathbf{a}^{(1)} = \\phi\\left(\\mathbf{z}^{(1)}\\right)

    .. math::

        \\mathbf{z}^{(2)} = \\mathbf{a}^{(1)}\\mathbf{W}^{(2)} + \\mathbf{b}^{(2)}

    .. math::

        \\mathbf{a}^{(2)} = \\phi\\left(\\mathbf{z}^{(2)}\\right)

    .. math::

        \\hat{\\mathbf{y}} = \\mathbf{a}^{(2)}\\mathbf{W}^{(3)} + \\mathbf{b}^{(3)}

    where :math:`\\phi(\\cdot)` is a nonlinear activation function and the output
    layer uses an identity activation suitable for regression tasks.

    The model represents a smooth, nonlinear approximation of the calibration
    function.
    """

    W1 = np.array(coeffs["layer_1_weights"])
    b1 = np.array(coeffs["layer_1_biases"])
    W2 = np.array(coeffs["layer_2_weights"])
    b2 = np.array(coeffs["layer_2_biases"])
    W3 = np.array(coeffs["output_weights"])
    b3 = np.array(coeffs["output_biases"])

    z1: np.ndarray = x @ W1 + b1
    a1 = apply_activation(z1, activation)

    z2: np.ndarray = a1 @ W2 + b2
    a2 = apply_activation(z2, activation)

    output = a2 @ W3 + b3

    return output.flatten()


def apply_activation(
        z: np.ndarray,
        activation: str
) -> np.ndarray:
    """
    Apply an element-wise activation function to a neural network layer.

    This function implements common nonlinear activation functions used in
    Multi-Layer Perceptrons, applied element-wise to pre-activation values.

    Supported activations include:

    .. math::

        \\mathrm{ReLU}(z) = \\max(0, z)

    .. math::

        \\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}

    .. math::

        \\mathrm{Identity}(z) = z

    The activation function determines the nonlinearity of the network and
    directly influences the smoothness and expressiveness of the resulting
    regression model.
    """

    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'tanh':
        return np.tanh(z)
    elif activation == 'identity':
        return z
    else:
        raise ValueError(f"Unsupported activation: {activation}")
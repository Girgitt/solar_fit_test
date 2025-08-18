from pathlib import Path

import numpy as np
import pandas as pd
import json

from sensor_calibration_metrics import SensorCalibrationMetrics

def save_metrics_to_json(
        metrics: SensorCalibrationMetrics,
        samples_count: int,
        coefficients_list: list[dict],
        filename_path: Path
) -> None:
    metrics_json = {
        "mse": float(metrics.mse),
        "mae": float(metrics.mae),
        "rmse": float(metrics.rmse),
        "r2": float(metrics.r2),
        "mape": float(metrics.mape),
        "max_error": float(metrics.max_error),
        "bias": float(metrics.bias),
        "n_samples": samples_count,
    }

    if coefficients_list is not None:
        metrics_json["coefficients"] = coefficients_list

    filename_path.parent.mkdir(parents=True, exist_ok=True)

    with open(filename_path, "w") as f:
        json.dump(metrics_json, f, indent=2)

def save_true_and_predicted_data_to_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    index: np.ndarray = None
) -> None:
    if index is None:
        index = np.arange(len(y_true))

    df_out = pd.DataFrame({
        "index": index,
        "y_true": y_true,
        "y_pred": y_pred
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
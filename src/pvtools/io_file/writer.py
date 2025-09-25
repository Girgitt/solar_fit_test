import numpy as np
import pandas as pd
import json

from pathlib import Path
from matplotlib.figure import Figure
from typing import List, Tuple, Optional

from pvtools.config.sensor_calibration_metrics import SensorCalibrationMetrics

REQUIRED_METRIC_FIELDS = ["mse", "mae", "rmse", "r2", "mape", "max_error", "bias"]

def save_metrics_to_json(
        metrics: SensorCalibrationMetrics,
        samples_count: int,
        coefficients_list: list[dict],
        filename_path: Path = None
) -> None:
    for attr in REQUIRED_METRIC_FIELDS:
        if not hasattr(metrics, attr):
            raise TypeError(f"metrics must have '{attr}' attribute")

    if not isinstance(samples_count, int):
        raise TypeError("samples_count must be an int")

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

    if filename_path is not None:
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

def save_dataframe_to_csv(
        df: pd.DataFrame,
        output_path: Path,
        index: Optional[bool] = False,
        index_label: Optional[str] = None
) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            output_path,
            index=index,
            index_label=index_label
        )

def save_figure(
        fig: Figure,
        save_dir: Path=None,
        filename: str=None,
) -> None:
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / filename

    print(f"[DEBUG] Saving: {output_path}")
    try:
        fig.savefig(output_path, dpi=300)
        print(f"[SUCCESS] Saved: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save {output_path}: {e}")

def save_predicted_data_figures(
        figures: List[Tuple[str, str, Figure]],
        save_dir: Path=None,
) -> None:
    print(f"[INFO] Saving figures to: {save_dir} (type: {type(save_dir)})")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for sensor_name, calibration_method, fig in figures:
        save_figure(fig, save_dir, f"{sensor_name}.png")
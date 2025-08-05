import os
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from matplotlib.figure import Figure
from typing import Dict, List, Tuple

from utils import load_true_and_predicted_data_for_all_methods

def plot_raw_data(
    df: pd.DataFrame,
    save_dir: Path=None,
    sensor_names: list[str] = None,
    sensor_name_ref: str=None,
    show: bool=True,
) -> None:
    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    # Plot 1: Raw input series over time
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["time"], df[sensor_name_ref], label="Power Reference (actual)", linewidth=0.9)
    for sensor_col in sensor_names:
        ax.plot(df["time"], df[sensor_col], label=f"Sensor: {sensor_col}", linewidth=0.9)
    ax.set_title("Raw input series over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power [W]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if show:
        fig.show()

    filename = f"series_vs_time.png"

    if fig is not None:
        save_raw_data_figure(filename, fig, save_dir)

def save_figure(
        fig: Figure,
        fig_path: Path=None,
) -> None:
    print(f"[DEBUG] Saving: {fig_path}")
    try:
        fig.savefig(fig_path, dpi=300)
        print(f"[SUCCESS] Saved: {fig_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save {fig_path}: {e}")

def save_raw_data_figure(
        filename: str,
        fig: Figure,
        save_dir: Path=None
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / filename

    save_figure(fig, fig_path)

def save_predicted_data_figures(
        figures: List[Tuple[str, str, Figure]],
        save_dir: Path=None,
) -> None:
    print(f"[INFO] Saving figures to: {save_dir} (type: {type(save_dir)})")

    save_dir.mkdir(parents=True, exist_ok=True)

    for sensor_name, calibration_method, fig in figures:
        filename = f"{sensor_name}.png"
        fig_path = save_dir / filename

        save_figure(fig, fig_path)

def subplot_predicted_data(
        data: Dict[str, pd.DataFrame],
        y_true: str,
        y_pred: str,
        calibration_method: str,
) -> List[Tuple[str, str, Figure]]:

    figures = []

    for sensor_name, df in data.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(
            data[sensor_name]["index"],
            data[sensor_name][y_true],
            label=y_true,
            linewidth=0.9)

        ax.plot(
            data[sensor_name]["index"],
            data[sensor_name][y_pred],
            label=y_pred,
            linewidth=0.9)

        ax.set_title(f"{sensor_name} prediciton by {calibration_method}")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Power [W]")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        figures.append((sensor_name, calibration_method, fig))

    return figures

def plot_predicted_data(
        calibration_method_dir: Path,
        show: bool = True,
        save_dir: Path = None,
) -> None:
    all_data = {}
    all_data = load_true_and_predicted_data_for_all_methods(calibration_method_dir)

    calibration_method_names = [name for name in os.listdir(calibration_method_dir)
                                if os.path.isdir(os.path.join(calibration_method_dir, name))]

    for calibration_method in calibration_method_names:
        figures = subplot_predicted_data(
            all_data[calibration_method],
            y_true="y_true",
            y_pred="y_pred",
            calibration_method=calibration_method,
        )

        save_predicted_data_figures(
            figures=figures,
            save_dir=save_dir / calibration_method,
        )

        if not show:
            for _, _, fig in figures:
                plt.close(fig)

    if show:
        plt.show()
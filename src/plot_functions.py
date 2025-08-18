import os
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional

from load_functions import load_true_and_predicted_data_for_all_methods

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
    ax.set_ylabel("Power (W/m²)")
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
        ax.set_ylabel("Power (W/m²)")
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

def plot_clear_sky(
    cs: pd.DataFrame,
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    cs.plot(ax=ax)
    ax.set_ylabel("Irradiance (W/m²)")
    ax.set_title("Clear‐sky irradiance (DNI, GHI, DHI)")
    ax.grid(True)
    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / "clear_sky.png"
        save_figure(fig, fig_path)

    if show:
        fig.show()
    return fig

def plot_poa_components(
    poa: pd.DataFrame,
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    poa[['poa_global', 'poa_direct', 'poa_diffuse', 'poa_ground_diffuse']].plot(ax=ax)
    ax.set_ylabel("Irradiance (W/m²)")
    ax.set_title("Plane‐of‐Array Irradiance (Perez model)")
    ax.legend(title="")
    ax.grid(True)
    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / "poa_components.png"
        save_figure(fig, fig_path)

    if show:
        fig.show()
    return fig


def plot_poa_vs_reference(
        poa_global: pd.Series,
        sensor_reference: pd.Series,
        save_dir: Optional[Path] = None,
        show: bool = True,
) -> Figure:
    if len(poa_global) == len(sensor_reference):
        sensor_copy = sensor_reference.copy()
        sensor_copy.index = poa_global.index
    else:
        raise ValueError("Incorret number of rows")

    df = pd.concat({"POA Global": poa_global, "Reference": sensor_copy}, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    df.plot(ax=ax, linewidth=0.9)
    ax.set_ylabel("Irradiance / Power (W/m²)")
    ax.set_title("POA Global vs Sensor Reference")
    ax.legend(title="")
    ax.grid(True)
    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / "poa_vs_reference.png"
        save_figure(fig, fig_path)

    if show:
        fig.show()

    return fig


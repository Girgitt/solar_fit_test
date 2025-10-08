import os
import matplotlib.pyplot as plt
import pandas as pd

from typing import List, Tuple, Optional
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Dict

from pvtools.io_file.reader import load_true_and_predicted_data_for_all_methods
from pvtools.io_file.writer import save_figure, save_predicted_data_figures
from pvtools.preprocess.preprocess_data import sanitize_filename


def plot_from_dataframe(
    df: pd.DataFrame,
    save_dir: Path=None,
    filename: str=None,
    sensor_names: list[str] = None,
    sensor_name_ref: str=None,
    show: bool=True,
    title: str = "Plot"
) -> tuple[Figure, Axes]:
    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    df = df.copy()
    if 'time' in df.columns:    
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        x = df['time']
    else:
        x = pd.to_datetime(df.index, errors='coerce')

    # Plot 1: Raw input series over time
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, df[sensor_name_ref], label="Power Reference (actual)", linewidth=0.9)
    for sensor_col in sensor_names:
        ax.plot(x, df[sensor_col], label=f"Sensor: {sensor_col}", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W/m²)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if show:
        fig.show()

    if save_dir is not None:
        save_figure(fig, save_dir, filename)

    return fig, ax


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
            data[sensor_name].index,
            data[sensor_name][y_true],
            label=y_true,
            linewidth=0.9)

        ax.plot(
            data[sensor_name].index,
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

    save_figure(fig, save_dir, "clear_sky_model.png")

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

    save_figure(fig, save_dir, "poa_components.png")

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
        save_figure(fig, save_dir, "poa_vs_reference.png")

    if show:
        fig.show()

    return fig


def plot_poa_reference_with_clearsky_periods(
        poa_global: pd.Series,
        sensor_reference: pd.Series,
        sunny: pd.Series,
        save_dir: Optional[Path] = None,
        show: bool = True
) -> Figure:
    poa_global = poa_global.copy()
    sensor_reference = sensor_reference.copy()
    sunny = sunny.copy()

    fig = plot_poa_vs_reference(
        poa_global=poa_global,
        sensor_reference=sensor_reference,
        save_dir=None,
        show=False
    )

    sensor_aligned = sensor_reference
    sensor_aligned.index = poa_global.index
    sunny_aligned = sunny.reindex(poa_global.index).fillna(False).astype(bool)

    ax = fig.axes[0]
    ax.scatter(
        poa_global.index[sunny_aligned],
        sensor_aligned[sunny_aligned],
        s=12,
        zorder=5,
        label="Clear-sky samples"
    )

    ax.set_title("POA Global vs Sensor Reference (clear-sky highlighted)")
    ax.legend(title="")
    fig.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, "poa_vs_reference_sunny.png")

    if show:
        fig.show()

    return fig

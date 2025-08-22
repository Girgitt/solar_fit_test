import os
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from typing import Dict, Optional

from save_functions import *
from load_functions import load_true_and_predicted_data_for_all_methods
from calibrate_methods import sanitize_filename

def plot_raw_data(
    df: pd.DataFrame,
    save_dir: Path=None,
    filename: str=None,
    sensor_names: list[str] = None,
    sensor_name_ref: str=None,
    show: bool=True,
) -> tuple[Figure, Axes]:
    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    if 'time' in df.columns:
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        x = df['time']
    else:
        x = pd.to_datetime(df.index, errors='coerce')

    # Plot 1: Raw input series over time
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, df[sensor_name_ref], label="Power Reference (actual)", linewidth=0.9)
    for sensor_col in sensor_names:
        ax.plot(x, df[sensor_col], label=f"Sensor: {sensor_col}", linewidth=0.9)
    ax.set_title("Raw input series over time")
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

def plot_raw_data_with_peaks(
    df: pd.DataFrame,
    save_dir: Path=None,
    peaks_dir: Path=None,
    filename = str,
    sensor_names: list[str] = None,
    sensor_name_ref: str=None,
    show: bool=True,
) -> None:
    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")
    if peaks_dir is None:
        raise ValueError("Parameter 'peaks_dir' must be a directory containing peaks CSV files.")

    peaks_dir = Path(peaks_dir)

    fig, ax = plot_raw_data(
        df=df,
        save_dir=None,
        filename=None,
        sensor_names=sensor_names,
        sensor_name_ref=sensor_name_ref,
        show=False,
    )

    sensors_all = list(sensor_names)
    if sensor_name_ref is not None and sensor_name_ref not in sensors_all:
        sensors_all.append(sensor_name_ref)

    line_colors = {}
    for ln in ax.get_lines():
        line_colors[ln.get_label()] = ln.get_color()

    for sensor_col in sensors_all:
        peaks_path = peaks_dir / f"{sanitize_filename(sensor_col)}_peaks.csv"
        if not peaks_path.exists():
            print(f"[WARN] Peaks CSV not found for '{sensor_col}': {peaks_path}")
            continue

        peaks_df = pd.read_csv(peaks_path, parse_dates=['time'])
        peaks_df['value'] = pd.to_numeric(peaks_df['value'], errors='coerce')
        peaks_df = peaks_df.dropna(subset=['time', 'value'])

        line_label = f"Sensor: {sensor_col}"
        line_color = line_colors.get(line_label, None)

        ax.scatter(
            peaks_df['time'],
            peaks_df['value'],
            s=15, # control the shape
            facecolors='white',
            edgecolors=line_color or 'black',
            linewidths=1.6,
            alpha=1.0,
            zorder=10,
            label=f"{sensor_col} peaks",
        )

    ax.set_title("Raw input series with detected peaks")
    fig.tight_layout()

    if show:
        fig.show()

    if save_dir is not None:
        save_figure(fig, save_dir, filename)

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

    save_figure(fig, save_dir, "poa_vs_reference.png")

    if show:
        fig.show()

    return fig


import matplotlib.pyplot as plt
import pandas as pd

from typing import List, Tuple, Optional
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Dict

from pvtools.io_file.writer import save_figure


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

    if sensor_name_ref is not None:
        ax.plot(x, df[sensor_name_ref], label="Power Reference (actual)", linewidth=0.9)

    for sensor_col in sensor_names:
        ax.plot(x, df[sensor_col], label=f"Sensor: {sensor_col}", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W/m²)")
    ax.legend(
        loc="upper right",
        fontsize=5
    )
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
    ax.legend(
        title="",
        loc="upper right",
        fontsize=5
    )
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
    ax.legend(
        title="",
        loc="upper right",
        fontsize=5
    )
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
    ax.legend(
        title="",
        loc="upper right",
        fontsize=5
    )
    ax.grid(True)
    fig.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, "poa_vs_reference.png")

    if show:
        fig.show()

    return fig


def plot_poa_reference_with_clearsky_periods(
        poa_global: pd.DataFrame,
        sensor_reference: pd.DataFrame,
        sunny: pd.Series,
        save_dir: Optional[Path] = None,
        show: bool = True
) -> Figure:

    poa_global = poa_global.copy()
    sensor_reference = sensor_reference.copy()
    sunny = sunny.copy()

    poa_global = poa_global.set_index("time")
    sensor_reference = sensor_reference.set_index("time")

    fig = plot_poa_vs_reference(
        poa_global=poa_global,
        sensor_reference=sensor_reference,
        save_dir=None,
        show=False
    )

    ax = fig.axes[0]
    ax.scatter(
        sunny.index[sunny],
        sensor_reference[sunny],
        s=12,
        zorder=5,
        label="Clear-sky samples"
    )

    ax.set_title("POA Global vs Sensor Reference (clear-sky highlighted)")
    ax.legend(title="")
    ax.legend(
        loc="upper right",
        fontsize=5
    )
    fig.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, "poa_vs_reference_sunny.png")

    if show:
        fig.show()

    return fig


def plot_sensors_calibrated_directly_to_poa(
        result_df: pd.DataFrame,
        title: str = "default plot",
        save_dir: Optional[Path] = None,
        filename: str = "default_filename",
        show: bool = False,
) -> None:

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(result_df.index, result_df["poa_global"], label="poa global", linewidth=1.5)
    ax.plot(result_df.index, result_df["sensor"], label="sensor", alpha=0.5)
    ax.plot(result_df.index, result_df["sensor_cal"], label="sensor calibrated", linewidth=1.2)

    # highlight clear-sky
    clear_idx = result_df.index[result_df["clearsky_mask"]]
    ax.scatter(clear_idx,
               result_df.loc[clear_idx, "sensor"],
               s=5,
               color="green",
               label="Clear-sky detected")

    ax.set_title(title)
    ax.set_ylabel("Irradiance W/m²")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid()
    plt.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, f"{filename}.png")

    if show:
        fig.show()


def plot_frequency_histogram(
        freqs,
        fft_mag,
        bins=100,
        title: str = "Frequency Histogram of Irradiance Signal",
        save_dir: Optional[Path] = None,
        filename: str = "default_filename",
        show: bool = False,
) -> None:

    fig, ax = plt.figure(figsize=(12, 5))

    fig.hist(freqs, weights=fft_mag, bins=bins, edgecolor='black')
    fig.xlabel("Frequency (Hz)")
    fig.ylabel("Magnitude (sum of FFT power)")
    fig.title(title)
    fig.grid()

    if save_dir is not None:
        save_figure(fig, save_dir, f"{filename}.png")

    if show:
        fig.show()

def plot_fft_spectrum(
        freqs,
        fft_mag,
        max_freq=None,
        title: str = "Frequency Spectrum (FFT)",
        save_dir: Optional[Path] = None,
        filename: str = "default_filename",
        show: bool = False,
) -> None:

    fig, ax = plt.figure(figsize=(12, 5))

    if max_freq:
        mask = freqs <= max_freq
        fig.plot(freqs[mask], fft_mag[mask])
    else:
        fig.plot(freqs, fft_mag)

    fig.xlabel("Frequency (Hz)")
    fig.ylabel("Magnitude")
    fig.title(title)
    fig.grid()

    if save_dir is not None:
        save_figure(fig, save_dir, f"{filename}.png")

    if show:
        fig.show()
        

def tmp_plot_check_masks(
        result_df: pd.DataFrame,
        title: str = "default plot",
        save_dir: Optional[Path] = None,
        filename: str = "default_filename",
        show: bool = False,
) -> None:

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(result_df.index, result_df["poa_global"], label="poa global", linewidth=1.5)
    ax.plot(result_df.index, result_df["sensor"], label="sensor", alpha=0.5)

    # highlight clear-sky
    clear_idx = result_df.index[result_df["mask"]]
    ax.scatter(clear_idx,
               result_df.loc[clear_idx, "sensor"],
               s=5,
               color="green",
               label="Clear-sky detected")

    ax.set_title(title)
    ax.set_ylabel("Irradiance W/m²")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid()
    plt.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, f"{filename}.png")

    if show:
        fig.show()

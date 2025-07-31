from typing import List

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

from pathlib import Path
import json

def plot_model_outputs(
    df: pd.DataFrame,
    model_id="tmp_v1",
    out_dir=Path("../plots"),
    prefix="linear",
    show=True,
    sensor_names: list[str] = None,
    sensor_name_ref='sensor_ref'
):
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    # Plot 1: Raw input series over time
    plt.figure(figsize=(9, 4))
    plt.plot(df["time"], df[sensor_name_ref], label="Power Reference (actual)", linewidth=0.9)
    for sensor_col in sensor_names:
        plt.plot(df["time"], df[sensor_col], label=f"Sensor: {sensor_col}", linewidth=0.9)
    plt.title("Raw input series over time")
    plt.xlabel("Time")
    plt.ylabel("Power [W]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = out_dir / f"{prefix}_series_vs_time_{model_id}_{timestamp}.png"
    plt.savefig(fname, dpi=300)
    print("Saved:", fname)
    if show:
        plt.show()
    plt.close()

    # For each sensor with prediction, plot comparison
    for sensor_col in sensor_names:
        pred_col = f"pred_reference_hourly__{sensor_col}"
        if pred_col not in df:
            continue

        df_sorted = df[df[pred_col].notna()].sort_values(sensor_col)

        # Plot 2: Prediction vs actual data (scatter + line)
        plt.figure(figsize=(6, 5))
        plt.scatter(df[sensor_col], df[sensor_name_ref], s=8, alpha=0.3, label="Actual reference")
        plt.plot(df_sorted[sensor_col], df_sorted[pred_col], linewidth=2, label="Hourly model prediction")
        plt.title(f"Hourly model fit: {sensor_col}")
        plt.xlabel(f"{sensor_col} [W]")
        plt.ylabel("Reference power [W]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fname = out_dir / f"{prefix}_fit_curve_{model_id}_{sensor_col}_{timestamp}.png"
        plt.savefig(fname, dpi=300)
        print("Saved:", fname)
        if show:
            plt.show()
        plt.close()

        # Plot 3: Predicted vs actual value over time
        plt.figure(figsize=(9, 4))
        plt.plot(df["time"], df[sensor_name_ref], label="Power Reference (actual)", linewidth=0.9)
        plt.plot(df["time"], df[pred_col], label="Hourly model output", linewidth=0.9)
        plt.title(f"Reference vs model: {sensor_col} over time")
        plt.xlabel("Time")
        plt.ylabel("Reference power [W]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fname = out_dir / f"{prefix}_reference_vs_fit_{model_id}_{sensor_col}_{timestamp}.png"
        plt.savefig(fname, dpi=300)
        print("Saved:", fname)
        if show:
            plt.show()
        plt.close()

def plot_weak_hourly_segments(
    df,
    weak_hours_path,
    model_data_path,
    output_dir="./plots/weak_hours",
    sensor_names=None,
    sensor_name_ref='default_sensor_ref'
):
    """
    For each weak hour listed in a JSON file, plot series for each sensor:
    - Raw sensor values
    - Reference (actual) values
    - Regression output (a*x + b)
    """

    if sensor_names is None:
        raise ValueError("Parameter 'sensor_names' must be a list of column names.")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    # Load weak hours
    with open(weak_hours_path, "r") as f:
        weak_hours = json.load(f)
    weak_hours = [pd.to_datetime(ts) for ts in weak_hours]

    # Load model data
    with open(model_data_path, "r") as f:
        model_data = json.load(f)

    # Index model data by (hour, sensor)
    model_dict = {}
    for entry in model_data:
        try:
            key = (pd.to_datetime(entry["hour"]), entry["sensor"])
            model_dict[key] = entry
        except Exception as e:
            print(f"[!] Skipped invalid model entry: {entry} -> {e}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for timestamp in weak_hours:
        hour_start = timestamp
        hour_end = timestamp + pd.Timedelta(hours=1)

        segment = df[(df["time"] >= hour_start) & (df["time"] < hour_end)].copy()
        if segment.empty:
            continue

        for sensor_col in sensor_names:
            key = (timestamp, sensor_col)
            model_entry = model_dict.get(key, None)

            if not model_entry:
                print(f"No model for hour {timestamp} and sensor {sensor_col}")
                continue

            a = model_entry["a"]
            b = model_entry["b"]
            segment["predicted"] = a * segment[sensor_col] + b

            # Plot all three series
            plt.figure(figsize=(10, 5))
            plt.plot(segment["time"], segment[sensor_col],
                     label=f"{sensor_col} (raw)", linewidth=0.9)
            plt.plot(segment["time"], segment[sensor_name_ref],
                     label=f"{sensor_name_ref} (actual)", linewidth=0.9)
            plt.plot(segment["time"], segment["predicted"],
                     label=f"Regression Output: y = {a:.2f}·x + {b:.1f}",
                     linewidth=1.4, linestyle="--")

            plt.title(f"Regression Fit – {timestamp.strftime('%Y-%m-%d %H:%M')} – {sensor_col}")
            plt.xlabel("Time")
            plt.ylabel("Power [W]")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            fname = output_dir / f"regression_fail_{timestamp.strftime('%Y%m%d_%H')}__{sensor_col.replace('@','_').replace(':','_')}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved: {fname}")

def plot_predictions_by_method(all_data: dict):
    all_methods = set()
    for sensor_data in all_data.values():
        all_methods.update(sensor_data.keys())

    for method in all_methods:
        plt.figure(figsize=(10, 6))
        plt.title(f"Predictions for method: {method}")
        plt.xlabel("Index")
        plt.ylabel("Value")

        for sensor, method_dict in all_data.items():
            if method in method_dict:
                df = method_dict[method]
                plt.plot(df.index, df["y_true"], label=f"{sensor} - y_true", linestyle="--")
                plt.plot(df.index, df["y_pred"], label=f"{sensor} - y_pred")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_bar_metrics(
        selected_metrics: List[str],
        df: pd.DataFrame,
        output_dir: Path,
        sensor_name: str
):
    df = df[selected_metrics]

    ax = df.plot(kind="bar", figsize=(10, 6), title=f"Comparison of {selected_metrics} for {sensor_name}")
    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Method")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    if selected_metrics == ["r2"]:
        plt.ylim([0.95, 1.05])

    plt.savefig(output_dir / f"{sensor_name}_{selected_metrics}_comparison.png", dpi=300)
    plt.show()
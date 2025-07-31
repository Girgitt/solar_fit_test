import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

from pathlib import Path

def plot_model_outputs(
    df: pd.DataFrame,
    model_id: str="tmp_v1",
    out_dir: Path=Path("../plots"),
    prefix: str="linear",
    show: bool=True,
    sensor_names: list[str] = None,
    sensor_name_ref: str='sensor_ref'
) -> None:
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




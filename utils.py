import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# HELPER: very-lightweight solar-elevation approximation
# ----------------------------------------------------------------------------
def solar_elevation(lat, lon, tz_offset, dt_local):
    n = dt_local.timetuple().tm_yday
    lt = dt_local.hour + dt_local.minute / 60 + dt_local.second / 3600  # local clock time
    B = math.radians((360 / 365) * (n - 81))
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)  # Eq. of Time [min]
    lstm = 15 * tz_offset
    tc = 4 * (lon - lstm) + eot  # Time-corr [min]
    lst = lt + tc / 60  # Local solar time
    omega = math.radians(15 * (lst - 12))  # Hour angle
    delta = math.radians(23.45 * math.sin(math.radians(360 * (284 + n) / 365)))
    phi = math.radians(lat)
    cos_z = (math.sin(phi) * math.sin(delta) +
             math.cos(phi) * math.cos(delta) * math.cos(omega))
    z = math.acos(max(-1, min(1, cos_z)))  # clamp
    return math.degrees(math.pi / 2 - z)

# ----------------------------------------------------------------------------
# MODEL STORAGE HELPERS
# ----------------------------------------------------------------------------
def save_model_to_json(model, poly, path):
    data = {
        "intercept": model.intercept_,
        "coefficients": model.coef_.tolist(),
        "features": poly.get_feature_names_out().tolist()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Model saved to {path}")

def load_model_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["intercept"], data["coefficients"], data["features"]

# ----------------------------------------------------------------------------
# CREATE ONE-HOUR TIME SLOTS AND FIT CORRECTIVE FUNCTION
# ----------------------------------------------------------------------------

def fit_hourly_models(df: pd.DataFrame, model_id: str, log_dir = './logs'):
    df = df.copy()
    df["hour"] = df["time"].dt.floor("H")

    hourly_models = []
    for hour, group in df.groupby("hour"):
        x = group["power_hi.common@sensor_1:VALUE"].values.reshape(-1, 1)
        y = group["power_reference.common@sensor_1:VALUE"].values

        if len(x) < 5:
            continue

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        model_hour = LinearRegression()
        model_hour.fit(x_train, y_train)
        y_pred = model_hour.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        hourly_models.append({
            "hour": hour.isoformat(),
            "a": float(model_hour.coef_[0]),
            "b": float(model_hour.intercept_),
            "r2": float(r2),
            "mae": float(mae),
            "n_samples": len(x)
        })

    hourly_path = log_dir / Path(f"{model_id}.json")
    with open(hourly_path, "w") as f:
        json.dump(hourly_models, f, indent=2)

    print(f"Hourly calibration models saved to {hourly_path}")

def execute_hourly_prediction(df: pd.DataFrame, model_id: str, log_dir = './logs') -> pd.DataFrame:
    hourly_path = log_dir / Path(f"{model_id}.json")
    with open(hourly_path, "r") as f:
        hourly_models = json.load(f)

    # Convert list to dictionary for quick access
    model_dict = {
        m["hour"]: {"a": m["a"], "b": m["b"]}
        for m in hourly_models
    }

    predictions = []
    for _, row in df.iterrows():
        hour = row["time"].floor("H").isoformat()
        model = model_dict.get(hour)
        if model:
            x = row["power_hi.common@sensor_1:VALUE"]
            y_pred = model["a"] * x + model["b"]
        else:
            y_pred = np.nan
        predictions.append(y_pred)

    df = df.copy()
    df["pred_reference_hourly"] = predictions
    return df

import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt

# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------

def plot_model_outputs(df, model_id="tmp_v1", out_dir=Path("./plots"), prefix="linear", show=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Raw input series over time
    plt.figure(figsize=(9, 4))
    plt.plot(df["time"], df["power_hi.common@sensor_1:VALUE"], label="Power HI sensor", linewidth=0.9)
    plt.plot(df["time"], df["power_reference.common@sensor_1:VALUE"], label="Power Reference (actual)", linewidth=0.9)
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

    if "pred_reference_hourly" in df:
        df_sorted = df[df["pred_reference_hourly"].notna()].sort_values("power_hi.common@sensor_1:VALUE")

        # Plot 2: Prediction vs actual data
        plt.figure(figsize=(6, 5))
        plt.scatter(df["power_hi.common@sensor_1:VALUE"],
                    df["power_reference.common@sensor_1:VALUE"],
                    s=8, alpha=0.3, label="Actual reference")
        plt.plot(df_sorted["power_hi.common@sensor_1:VALUE"],
                 df_sorted["pred_reference_hourly"],
                 linewidth=2, label="Hourly model prediction")
        plt.title("Hourly model fit: output vs input")
        plt.xlabel("Power HI sensor [W]")
        plt.ylabel("Reference power [W]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fname = out_dir / f"{prefix}_fit_curve_{model_id}_{timestamp}.png"
        plt.savefig(fname, dpi=300)
        print("Saved:", fname)
        if show:
            plt.show()
        plt.close()

        # Plot 3: Predicted vs actual value over time
        plt.figure(figsize=(9, 4))
        plt.plot(df["time"], df["power_reference.common@sensor_1:VALUE"], label="Power Reference (actual)", linewidth=0.9)
        plt.plot(df["time"], df["pred_reference_hourly"], label="Hourly model output", linewidth=0.9)
        plt.title("Reference sensor vs hourly model over time")
        plt.xlabel("Time")
        plt.ylabel("Reference power [W]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fname = out_dir / f"{prefix}_reference_vs_fit_{model_id}_{timestamp}.png"
        plt.savefig(fname, dpi=300)
        print("Saved:", fname)
        if show:
            plt.show()
        plt.close()


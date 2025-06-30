import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from pathlib import Path
from datetime import datetime, timedelta

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

def fit_hourly_models(df: pd.DataFrame, model_id: str, log_dir = './logs', sensor_value_1='power_hi.common@sensor_1:VALUE', sensor_value_ref='power_reference.common@sensor_1:VALUE'):
    df = df.copy()
    df["hour"] = df["time"].dt.floor("H")

    hourly_models = []
    for hour, group in df.groupby("hour"):
        x = group[sensor_value_1].values.reshape(-1, 1)
        y = group[sensor_value_ref].values

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

def execute_hourly_prediction(df: pd.DataFrame, model_id: str, log_dir = './logs', sensor_value_1='power_hi.common@sensor_1:VALUE') -> pd.DataFrame:
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
            x = row[sensor_value_1]
            y_pred = model["a"] * x + model["b"]
        else:
            y_pred = np.nan
        predictions.append(y_pred)

    df = df.copy()
    df["pred_reference_hourly"] = predictions
    return df

# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------

def plot_model_outputs(df, model_id="tmp_v1", out_dir=Path("./plots"), prefix="linear", show=True, sensor_value_1='power_hi.common@sensor_1:VALUE', sensor_value_ref='power_reference.common@sensor_1:VALUE'):
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Raw input series over time
    plt.figure(figsize=(9, 4))
    plt.plot(df["time"], df[sensor_value_ref], label="Power HI sensor", linewidth=0.9)
    plt.plot(df["time"], df[sensor_value_1], label="Power Reference (actual)", linewidth=0.9)
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
        df_sorted = df[df["pred_reference_hourly"].notna()].sort_values(sensor_value_1)

        # Plot 2: Prediction vs actual data
        plt.figure(figsize=(6, 5))
        plt.scatter(df[sensor_value_1],
                    df[sensor_value_ref],
                    s=8, alpha=0.3, label="Actual reference")
        plt.plot(df_sorted[sensor_value_1],
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
        plt.plot(df["time"], df[sensor_value_ref], label="Power Reference (actual)", linewidth=0.9)
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

def identify_and_export_weak_hours(metrics_df, min_r2=0.8, max_mae=50.0, model_id="default", output_dir="./logs"):
    """
    Identifies model hours with weak fit based on R² and MAE thresholds.
    Exports the list of weak timestamps to a JSON file.

    Parameters:
        metrics_df (pd.DataFrame): Must contain 'hour', 'r2', and 'mae' columns. 'hour' must be datetime or ISO strings.
        min_r2 (float): Minimum acceptable R² score.
        max_mae (float): Maximum acceptable MAE value.
        model_id (str): Model identifier used for naming the output file.
        output_dir (str or Path): Directory where the JSON file will be saved.

    Returns:
        List[str]: List of weak hours as ISO 8601 strings.
    """

    # Ensure datetime type for 'hour' column
    df = metrics_df.copy()
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")

    # Filter weak rows
    weak = df[(df["r2"] < min_r2) | (df["mae"] > max_mae)]

    # Convert to list of ISO strings
    weak_hours = [dt.isoformat() for dt in weak["hour"].dropna()]

    # Export to JSON
    output_path = Path(output_dir) / f"weak_hours_{model_id}.json"
    with open(output_path, "w") as f:
        json.dump(weak_hours, f, indent=2)

    print(f"Weak hours exported to {output_path}")

    return weak_hours


def group_and_export_models(metrics_df, output_path, tol_a=0.02, tol_b=2.0):
    """
    Groups similar regression models based on coefficient similarity, and exports each group
    with full timestamp information and averaged model metrics.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing at least ['hour', 'a', 'b', 'r2', 'mae', 'n_samples'].
                                   The 'hour' column must be parseable as datetime.
        output_path (str or Path): Path to the output JSON file.
        tol_a (float): Allowed difference in 'a' coefficient to consider two models similar.
        tol_b (float): Allowed difference in 'b' coefficient to consider two models similar.

    Returns:
        List[Dict]: The list of grouped model summaries that was saved to JSON.
    """

    # Ensure hour column is in datetime format
    metrics_df = metrics_df.copy()
    metrics_df["hour"] = pd.to_datetime(metrics_df["hour"], errors="coerce")

    # Convert each row to a dict
    records = metrics_df.sort_values("hour").to_dict(orient="records")

    used = set()
    grouped_output = []

    for group_id, base in enumerate(records):
        base_key = base["hour"]
        if base_key in used:
            continue

        group_rows = [base]
        used.add(base_key)

        for candidate in records:
            candidate_key = candidate["hour"]
            if candidate_key in used or candidate_key == base_key:
                continue

            if (
                abs(base["a"] - candidate["a"]) <= tol_a and
                abs(base["b"] - candidate["b"]) <= tol_b
            ):
                group_rows.append(candidate)
                used.add(candidate_key)

        # Compute average metrics for the group
        group_df = pd.DataFrame(group_rows)

        grouped_output.append({
            "group_id": group_id,
            "hours": [row["hour"].isoformat() for row in group_rows],
            "avg_a": group_df["a"].mean(),
            "avg_b": group_df["b"].mean(),
            "avg_r2": group_df["r2"].mean(),
            "avg_mae": group_df["mae"].mean(),
            "total_samples": int(group_df["n_samples"].sum())
        })

    with open(output_path, "w") as f:
        json.dump(grouped_output, f, indent=2)

    return grouped_output

def plot_weak_hourly_segments(df, weak_hours_path, model_data_path, output_dir="./plots/weak_hours", sensor_value_1='power_hi.common@sensor_1:VALUE', sensor_value_ref='power_reference.common@sensor_1:VALUE'):
    """
    For each weak hour listed in a JSON file, plot three series:
    - Raw power_hi sensor values
    - Reference (calibrated) power values
    - Regression output (a*x + b)

    Parameters:
        df (pd.DataFrame): Dataset with 'time', 'power_hi', 'power_reference'
        weak_hours_path (str or Path): Path to weak_hours__{model_id}.json
        model_data_path (str or Path): Path to data.json with regression coefficients
        output_dir (str or Path): Output directory for PNG plots
    """

    # Ensure datetime parsing
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    # Load weak hours
    with open(weak_hours_path, "r") as f:
        weak_hours = json.load(f)
    weak_hours = [pd.to_datetime(ts) for ts in weak_hours]

    # Load model data
    with open(model_data_path, "r") as f:
        model_data = json.load(f)

    # Index model_data by timestamp for fast lookup
    model_dict = {}
    for entry in model_data:
        if "hour" in entry:
            try:
                key = pd.to_datetime(entry["hour"])
                model_dict[key] = entry
            except Exception as e:
                print(f"[!] Skipped invalid timestamp: {entry['hour']} -> {e}")
        else:
            print(f"[!] Entry missing 'hour' field: {entry}")

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for timestamp in weak_hours:
        hour_start = timestamp
        hour_end = timestamp + pd.Timedelta(hours=1)

        # Filter data segment for the given hour
        segment = df[(df["time"] >= hour_start) & (df["time"] < hour_end)].copy()
        if segment.empty:
            continue

        timestamp_dt = pd.to_datetime(timestamp)

        # Find matching model coefficients
        model_entry = model_dict.get(timestamp_dt, None)
        if not model_entry:
            print(f"No model found for {timestamp}")
            continue

        a = model_entry["a"]
        b = model_entry["b"]

        # Predict values using regression
        segment["predicted"] = a * segment[sensor_value_1] + b

        # Plot all three series
        plt.figure(figsize=(10, 5))
        plt.plot(segment["time"], segment[sensor_value_1],
                 label="Power HI Sensor (raw)", linewidth=0.9)
        plt.plot(segment["time"], segment[sensor_value_1],
                 label="Reference Sensor (actual)", linewidth=0.9)
        plt.plot(segment["time"], segment["predicted"],
                 label=f"Regression Output: y = {a:.2f}·x + {b:.1f}", linewidth=1.4, linestyle="--")

        plt.title(f"Regression Fit – {timestamp.strftime('%Y-%m-%d %H:%M')}")
        plt.xlabel("Time")
        plt.ylabel("Power [W]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save to file
        fname = output_dir / f"regression_fail_{timestamp.strftime('%Y%m%d_%H')}.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved: {fname}")

def smooth_models(models_list_path, blend_minutes, time_delta, output_path="smoothed_models.json"):
    """
    Generates smooth transitions between models for every minute (or every `time_delta`),
    returning a list of smoothed models with blending.

    Parameters:
    - models_list: list of input models with keys ‘hour’, ‘a’, 'b'
    - blend_minutes: width of the transition zone (e.g., 20 means 10 minutes for each side)
    - time_delta: timedelta object (e.g. timedelta(minutes=5)) - time step
    - output_path: path to save the resulting JSON
    """

    with open(models_list_path, "r") as f:
        models_list = json.load(f)

    model_dict = {
        datetime.fromisoformat(m["hour"]): m for m in models_list
    }

    half_blend = blend_minutes // 2
    step = time_delta
    result = []

    sorted_hours = sorted(model_dict.keys())

    for i in range(len(sorted_hours) - 1):
        t_curr = sorted_hours[i]
        t_next = sorted_hours[i + 1]
        model_curr = model_dict[t_curr]
        model_next = model_dict[t_next]

        t = t_curr + timedelta(minutes=60 - half_blend)
        while t < t_curr + timedelta(hours=1):
            w = (t - (t_curr + timedelta(minutes=60 - half_blend))).total_seconds() / (half_blend * 60)
            a = (1 - w) * model_curr["a"] + w * model_next["a"]
            b = (1 - w) * model_curr["b"] + w * model_next["b"]
            result.append({
                "hour": t.isoformat(),
                "a": a,
                "b": b
            })
            t += step

        t = t_next
        end = t_next + timedelta(minutes=half_blend)
        while t < end:
            w = (t - t_next).total_seconds() / (half_blend * 60)
            a = (1 - w) * model_curr["a"] + w * model_next["a"]
            b = (1 - w) * model_curr["b"] + w * model_next["b"]
            result.append({
                "hour": t.isoformat(),
                "a": a,
                "b": b
            })
            t += step

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved in: {output_path}")
    return result







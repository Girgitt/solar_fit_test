import math

from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from plotting_functions import *
from model_params import *


def argument_parsing(parser):
    parser.add_argument("--action", choices=["update", "execute"], required=True,
                        help="Specify whether to 'update' (train/save) or 'execute' (load/apply) the model")
    parser.add_argument("--model_id", default="default",
                        help="Model identifier used for saving/loading coefficients")
    parser.add_argument("--csv", required=True,
                        help="Path to CSV file with input data")

    return parser.parse_args()

def print_available_data_columns(data_columns):
    print("Available data columns:")
    for i, col in enumerate(data_columns):
        print(f"{i}: {col}")

def select_available_data_columns_to_process(data_columns, df):
    sensor_values = [
        data_columns[0],
        #data_columns[3],
        #data_columns[4],
    ]
    sensor_value_ref = data_columns[1] #data_columns[6]
    df = df.dropna(subset=sensor_values + [sensor_value_ref])

    return sensor_values, sensor_value_ref, df


def update_function(model_parameters: ModelParameters):
    fit_hourly_models(
        df=model_parameters.df,
        model_id=model_parameters.args.model_id,
        log_dir=model_parameters.log_dir,
        sensor_values=model_parameters.sensor_values,
        sensor_value_ref=model_parameters.sensor_value_ref,
    )

def execute_function(model_parameters: ModelParameters):
    df = execute_hourly_prediction(
        df=model_parameters.df,
        model_id=model_parameters.args.model_id,
        log_dir=model_parameters.log_dir,
        sensor_values=model_parameters.sensor_values,
    )

    results = evaluate_predictions_per_sensor(
        df=df,
        model_id=model_parameters.args.model_id,
        sensor_values=model_parameters.sensor_values,
        sensor_value_ref=model_parameters.sensor_value_ref,
    )

    for sensor, res in results.items():
        df_metrics = res["df_metrics"]

        identify_and_export_weak_hours(
            df_metrics,
            min_r2=model_parameters.min_r2,
            max_mae=model_parameters.max_mae,
            model_id=f"{model_parameters.args.model_id}__{sensor.replace('@', '_').replace(':', '_')}"
        )

        group_and_export_models(
            df_metrics,
            output_path=f"{model_parameters.log_dir}/grouped_{model_parameters.args.model_id}__{sensor.replace('@', '_').replace(':', '_')}.json",
            tol_a=model_parameters.tolerance_min,
            tol_b=model_parameters.tolerance_max,
        )

    plot_weak_hourly_segments(
        df=model_parameters.df,
        weak_hours_path=f"{model_parameters.log_dir}/weak_hours_{model_parameters.args.model_id}.json",
        model_data_path=f"{model_parameters.log_dir}/{model_parameters.args.model_id}.json",
        output_dir=f"{model_parameters.plot_dir}/weak_hours",
        sensor_values=model_parameters.sensor_values,
        sensor_value_ref=model_parameters.sensor_value_ref,
    )

    smooth_models(models_list_path=f"{model_parameters.log_dir}/{model_parameters.args.model_id}.json",
                  blend_minutes=model_parameters.blend_minutes,
                  time_delta=timedelta(minutes=model_parameters.timedelta),
                  output_path=f"{model_parameters.log_dir}/smoothed_{model_parameters.args.model_id}.json")

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

def fit_hourly_models(
        df: pd.DataFrame,
        model_id: str,
        log_dir='./logs',
        sensor_values: list[str] = None,
        sensor_value_ref: str = 'default_sensor_ref'
):
    df = df.copy()
    df["hour"] = df["time"].dt.floor("H")

    if sensor_values is None:
        raise ValueError("Parameter 'sensor_values' must be a list of column names.")

    all_hourly_models = []

    for sensor_col in sensor_values:
        hourly_models = []

        for hour, group in df.groupby("hour"):
            x = group[sensor_col].values.reshape(-1, 1)
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
                "sensor": sensor_col,
                "a": float(model_hour.coef_[0]),
                "b": float(model_hour.intercept_),
                "r2": float(r2),
                "mae": float(mae),
                "n_samples": len(x)
            })

        all_hourly_models.extend(hourly_models)

    # Zapisywanie do pliku JSON
    hourly_path = Path(log_dir) / f"{model_id}.json"
    with open(hourly_path, "w") as f:
        json.dump(all_hourly_models, f, indent=2)

    print(f"Hourly calibration models saved to {hourly_path}")

def execute_hourly_prediction(
    df: pd.DataFrame,
    model_id: str,
    log_dir='./logs',
    sensor_values: list[str] = None
) -> pd.DataFrame:

    if sensor_values is None:
        raise ValueError("Parameter 'sensor_values' must be a list of column names.")

    # Load the model
    hourly_path = Path(log_dir) / f"{model_id}.json"
    with open(hourly_path, "r") as f:
        hourly_models = json.load(f)

    # Index models by (hour, sensor)
    model_dict = {
        (m["hour"], m["sensor"]): {"a": m["a"], "b": m["b"]}
        for m in hourly_models
    }

    df = df.copy()
    for sensor_col in sensor_values:
        predictions = []

        for _, row in df.iterrows():
            hour = row["time"].floor("h").isoformat()
            key = (hour, sensor_col)
            model = model_dict.get(key)

            if model:
                x = row[sensor_col]
                y_pred = model["a"] * x + model["b"]
            else:
                y_pred = np.nan

            predictions.append(y_pred)

        # Store predictions in a separate column
        pred_col = f"pred_reference_hourly__{sensor_col}"
        df[pred_col] = predictions

    return df

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

def evaluate_predictions_per_sensor(df, model_id, sensor_values, sensor_value_ref):
    results_per_sensor = {}

    for sensor in sensor_values:
        pred_col = f"pred_reference_hourly__{sensor}"

        if pred_col not in df.columns:
            print(f"[!] Skipping {sensor}: missing prediction column '{pred_col}'")
            continue

        mask = df[pred_col].notna()
        y_true = df.loc[mask, sensor_value_ref]
        y_pred = df.loc[mask, pred_col]

        if y_true.empty or y_pred.empty:
            print(f"[!] No valid data for sensor {sensor}")
            continue

        # Global metrics for the whole period
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"[{sensor}] R² = {r2:.4f}, MAE = {mae:.2f}")

        # Load hourly model data for this sensor
        model_path = Path(f"./logs/{model_id}.json")
        with open(model_path, "r") as f:
            model_data = json.load(f)

        # Filter model entries for this specific sensor
        sensor_model_data = [entry for entry in model_data if entry.get("sensor") == sensor]

        if not sensor_model_data:
            print(f"[!] No model data for sensor {sensor} in {model_id}.json")
            continue

        df_metrics = pd.DataFrame(sensor_model_data)
        df_metrics["hour"] = pd.to_datetime(df_metrics["hour"])
        df_metrics = df_metrics.sort_values("hour")

        # Store everything in the result dictionary
        results_per_sensor[sensor] = {
            "r2_global": r2,
            "mae_global": mae,
            "df_metrics": df_metrics
        }

    return results_per_sensor

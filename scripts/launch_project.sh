#!/usr/bin/env bash
# Runs your Python main using arguments sourced from a JSON file.
# Requires: jq
# Usage: ./run_from_json.sh path/to/config_execute.json
# If no path is given, defaults to ./config_execute.json

set -euo pipefail

CONFIG_JSON="${1:-config_execute.json}"

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: 'jq' is required (https://stedolan.github.io/jq/)."
  exit 1
fi

if [ ! -f "$CONFIG_JSON" ]; then
  echo "Error: JSON file not found: $CONFIG_JSON"
  exit 1
fi

### -------- Read launch section --------
PY_FILE=$(jq -r '.launch_python_script.python_file_directory' "$CONFIG_JSON")
ACTION=$(jq -r '.launch_python_script.action' "$CONFIG_JSON")
MODEL_ID=$(jq -r '.launch_python_script.model_id' "$CONFIG_JSON")
CSV_INPUT=$(jq -r '.launch_python_script.csv_input_file_directory' "$CONFIG_JSON")
CALIBRATION=$(jq -r '.launch_python_script.calibration_method' "$CONFIG_JSON")
SENSORS=$(jq -r '.launch_python_script.sensors_id' "$CONFIG_JSON")
REFERENCE=$(jq -r '.launch_python_script.reference_sensor_id // empty' "$CONFIG_JSON")
PROJECT_DIR=$(jq -r '.launch_python_script.project_directory // empty' "$CONFIG_JSON")
CAL_MET_DIR=$(jq -r '.launch_python_script.calibration_metrics_directory // empty' "$CONFIG_JSON")

### -------- Read clearsky parameters --------
LAT=$(jq -r '.clearsky_input_parameters.latitude' "$CONFIG_JSON")
LON=$(jq -r '.clearsky_input_parameters.longtitude' "$CONFIG_JSON")       # note: matches your parser flag name
TZ=$(jq -r '.clearsky_input_parameters.timezone' "$CONFIG_JSON")
ALT=$(jq -r '.clearsky_input_parameters.altitude' "$CONFIG_JSON")
NAME=$(jq -r '.clearsky_input_parameters.name' "$CONFIG_JSON")
FREQ=$(jq -r '.clearsky_input_parameters.frequency' "$CONFIG_JSON")
ALBEDO=$(jq -r '.clearsky_input_parameters.albedo' "$CONFIG_JSON")
TILT=$(jq -r '.clearsky_input_parameters.surface_tilt' "$CONFIG_JSON")
AZIMUTH=$(jq -r '.clearsky_input_parameters.surface_azimuth' "$CONFIG_JSON")

### -------- Read filtering parameters --------
ST_H=$(jq -r '.filtering_data_parameters.delete_night_period.start_hour' "$CONFIG_JSON")
ST_M=$(jq -r '.filtering_data_parameters.delete_night_period.start_minute' "$CONFIG_JSON")
END_H=$(jq -r '.filtering_data_parameters.delete_night_period.end_hour' "$CONFIG_JSON")
END_M=$(jq -r '.filtering_data_parameters.delete_night_period.end_minute' "$CONFIG_JSON")

### -------- Resolve CSV path (file or directory) --------
# Your JSON key is named "..._directory". If a directory is given, pick the first *.csv in it.
if [ -d "$CSV_INPUT" ]; then
  CSV=$(find "$CSV_INPUT" -maxdepth 1 -type f -name '*.csv' | head -n 1 || true)
  if [ -z "${CSV:-}" ]; then
    echo "Error: No CSV files found in directory: $CSV_INPUT"
    exit 1
  fi
else
  CSV="$CSV_INPUT"
fi

### -------- Prepare sensors list for --sensors (nargs='+') --------
# JSON has "0 1 2" as a space-separated string → turn it into an array so each becomes its own CLI arg.
read -r -a SENSORS_ARR <<< "$SENSORS"

### -------- Create optional output dirs if provided --------
[ -n "$PROJECT_DIR" ] && mkdir -p "$PROJECT_DIR"
[ -n "$CAL_MET_DIR" ] && mkdir -p "$CAL_MET_DIR"

### -------- Build command safely (array preserves spaces/quoting) --------
cmd=(
  python "$PY_FILE"
  --action="$ACTION"
  --model_id="$MODEL_ID"
  --csv="$CSV"
  --calibration="$CALIBRATION"
  --sensors "${SENSORS_ARR[@]}"
  --start_time_hour "$ST_H"
  --start_time_minute "$ST_M"
  --end_time_hour "$END_H"
  --end_time_minute "$END_M"
  --latitude "$LAT"
  --longtitude "$LON"
  --timezone="$TZ"
  --altitude "$ALT"
  --name="$NAME"
  --frequency="$FREQ"
  --albedo "$ALBEDO"
  --surface_tilt "$TILT"
  --surface_azimuth "$AZIMUTH"
)

# Optional flags (only if present in JSON)
[ -n "$PROJECT_DIR" ]   && cmd+=( --project_dir="$PROJECT_DIR" )
[ -n "$CAL_MET_DIR" ]   && cmd+=( --calibration_metrics_dir="$CAL_MET_DIR" )
[ -n "$REFERENCE" ]     && cmd+=( --reference "$REFERENCE" )

### -------- Show and run --------
echo "Running: ${cmd[*]}"
"${cmd[@]}"

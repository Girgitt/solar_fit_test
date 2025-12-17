import os
import pandas as pd
import numpy as np
import datetime
import logging

from pathlib import Path
from typing import TypeAlias, Literal

from pvtools.io_file.writer import save_true_and_predicted_data_to_csv
from pvtools.io_file.reader import divided_linear_regression_load_parameters
from pvtools.config.params import (DatatypeCoefficientsForDividedLinearRegression, ModelData, ModelDirectories,
                                   ModelTimes)
from pvtools.calibration.calibrate_to_reference.calibration_utils import check_if_any_column_is_missing


log = logging.getLogger("calibrate")
Period_type: TypeAlias = Literal['sunny', 'cloudy']


def calibrate_by_divided_linear_regression(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        period_flag: bool = True  # if True - periods detected, else not
) -> None:
    """
    Calibrate Divided Linear Regression model.

    Loads metrics from .json files. Search for ``sunny``, ``cloudy`` and ``all`` files containing metrics for that
    periods. If not found raise an Error.

    Based on input boolean parameter ``period_flag`` - calculates calibrated values:

    * if ``True`` calculation is made on both periods
    * if ``False`` calculation is made on sunny periods and all periods

    Saves calibrated sensor data to .csv file.

    Warning:
          To consider is it properly proceeded!!
    """

    df = model_data.df
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref
    load_params_dir = model_dirs.load_metrics_dir
    save_dir = model_dirs.log_dir
    filename = model_dirs.filename

    divided_linear_regression = "divided_linear_regression"
    calibration_method_dir_all = Path(os.path.join(load_params_dir, divided_linear_regression, "all"))
    calibration_method_dir_sunny = Path(os.path.join(load_params_dir, divided_linear_regression, "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(load_params_dir, divided_linear_regression, "cloudy"))

    log.debug(f"calibration_method_dir:{calibration_method_dir_all}")
    log.debug(f"calibration_method_dir:{calibration_method_dir_sunny}")
    log.debug(f"calibration_method_dir:{calibration_method_dir_cloudy}")

    json_files_all = list(calibration_method_dir_all.glob("*.json"))
    json_files_sunny = list(calibration_method_dir_sunny.glob("*.json"))
    json_files_cloudy = list(calibration_method_dir_cloudy.glob("*.json"))

    if len(json_files_all) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_all},"
            f" but found {len(json_files_all)}.")

    if len(json_files_sunny) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_sunny},"
            f" but found {len(json_files_sunny)}.")

    if len(json_files_cloudy) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_cloudy},"
            f" but found {len(json_files_cloudy)}.")

    time = df["time"]
    y_pred = pd.Series()
    y_true = df[sensor_name_ref]

    for json_all, json_sunny, json_cloudy, sensor_name in (
            zip(json_files_all, json_files_sunny, json_files_cloudy, sensor_names)):

        params_all = divided_linear_regression_load_parameters(json_all)
        params_sunny = divided_linear_regression_load_parameters(json_sunny)
        params_cloudy = divided_linear_regression_load_parameters(json_cloudy)

        log.debug(f"fitting json all: {json_all}")
        log.debug(f"fitting json sunny: {json_sunny}")
        log.debug(f"fitting json cloudy: {json_cloudy}")

        if period_flag is True:
            y_pred = divided_linear_regression_use_calibration_values(
                df=df[["time", sensor_name, "if_sunny"]],
                sensor_name=sensor_name,
                params_sunny=params_sunny,
                params_cloudy=params_cloudy
            )

        elif period_flag is False:
            if sensor_name_ref is None:
                y_true = None

            y_pred = divided_linear_regression_use_calibration_values(
                df=df[["time", sensor_name, "if_sunny"]],
                sensor_name=sensor_name,
                params_sunny=params_sunny,
                params_cloudy=params_all
            )

        output_dir = Path(save_dir) / filename / divided_linear_regression

        file_stem_sunny = Path(json_sunny).stem
        file_stem_cloudy = Path(json_cloudy).stem

        for file in (file_stem_sunny, file_stem_cloudy):
            csv_filename = output_dir / f"{file}_all_predicted.csv"
            log.debug(f"csv_filename: {csv_filename}")
            save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=None)


def calibrate_by_divided_linear_regression_mean(
        model_data: ModelData,
        model_dirs: ModelDirectories,
        model_times: ModelTimes,
        period_flag: bool = True  # if True - periods detected, else not
) -> None:
    """
    Calibrate Mean Divided Linear Regression model.

    Loads metrics from .json files. Search for ``sunny``, ``cloudy`` and ``all`` files containing metrics for that
    periods. If not found raise an Error.

    Based on input boolean parameter ``period_flag`` - calculates calibrated values:

    * if ``True`` calculation is made on both periods
    * if ``False`` calculation is made on sunny and all periods

    Saves calibrated sensor data to .csv file.

    Warning:
          To consider is it properly proceeded!!
    """


    df = model_data.df
    sensor_names = model_data.sensor_names
    sensor_name_ref = model_data.sensor_name_ref
    load_params_dir = model_dirs.load_metrics_dir
    save_dir = model_dirs.log_dir
    filename = model_dirs.filename

    divided_linear_regression_mean = "divided_linear_regression_mean"
    calibration_method_dir_all = Path(os.path.join(load_params_dir, divided_linear_regression_mean, "all"))
    calibration_method_dir_sunny = Path(os.path.join(load_params_dir, divided_linear_regression_mean, "sunny"))
    calibration_method_dir_cloudy = Path(os.path.join(load_params_dir, divided_linear_regression_mean, "cloudy"))

    log.debug(f"calibration_method_dir:{calibration_method_dir_all}")
    log.debug(f"calibration_method_dir:{calibration_method_dir_sunny}")
    log.debug(f"calibration_method_dir:{calibration_method_dir_cloudy}")

    json_files_all = list(calibration_method_dir_all.glob("*.json"))
    json_files_sunny = list(calibration_method_dir_sunny.glob("*.json"))
    json_files_cloudy = list(calibration_method_dir_cloudy.glob("*.json"))

    if len(json_files_all) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_all},"
            f" but found {len(json_files_all)}.")

    if len(json_files_sunny) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_sunny},"
            f" but found {len(json_files_sunny)}.")

    if len(json_files_cloudy) != len(sensor_names):
        raise RuntimeError(
            f"Expected {len(sensor_names)} .json files in {calibration_method_dir_cloudy},"
            f" but found {len(json_files_cloudy)}.")

    time = df["time"]
    y_pred = pd.Series()

    if sensor_name_ref is not None:
        y_true = df[sensor_name_ref]
    else:
        y_true = None

    for json_all, json_sunny, json_cloudy, sensor_name in (
            zip(json_files_all, json_files_sunny, json_files_cloudy, sensor_names)):

        params_all = divided_linear_regression_load_parameters(json_all)
        params_sunny = divided_linear_regression_load_parameters(json_sunny)
        params_cloudy = divided_linear_regression_load_parameters(json_cloudy)

        log.debug(f"fitting json all: {json_all}")
        log.debug(f"fitting json sunny: {json_sunny}")
        log.debug(f"fitting json cloudy: {json_cloudy}")

        params_combined = select_calibration_parameters(
                params_all=params_all,
                params_sunny=params_sunny,
                params_cloudy=params_cloudy,
                df_time=time,
                frequency=model_times.divided_linear_regression_interval
        )

        if period_flag is True:
            y_pred = divided_linear_regression_use_calibration_values_mean(
                df=df[["time", sensor_name, "if_sunny"]],
                sensor_name=sensor_name,
                params_sunny=params_sunny
            )

        elif period_flag is False:
            df["if_sunny"] = True

            y_pred = divided_linear_regression_use_calibration_values_mean(
                df=df[["time", sensor_name, "if_sunny"]],
                sensor_name=sensor_name,
                params_sunny=params_combined
            )

        output_dir = Path(save_dir) / filename / divided_linear_regression_mean
        output_dir.mkdir(parents=True, exist_ok=True)

        file_stem_sunny = Path(json_sunny).stem
        csv_filename = output_dir / f"{file_stem_sunny}_all_predicted.csv"
        log.debug(f"csv_filename: {csv_filename}")
        save_true_and_predicted_data_to_csv(y_pred, csv_filename, y_true, index=None, time=time)


def divided_linear_regression_use_calibration_values(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny:  list[DatatypeCoefficientsForDividedLinearRegression],
        params_cloudy: list[DatatypeCoefficientsForDividedLinearRegression] | None = None
) -> pd.Series:
    """
    Do a calculation of Divided Linear Regression using calibration values. Same calculation as for Linear Regression,
    but divided for equal, specified periods.

    .. math::

            y = a * x + b
    """

    if params_cloudy is not None:
        if_sunny_col = "if_sunny"
    else:
        if_sunny_col = None

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time",
        if_sunny_col=if_sunny_col
    )

    def build_intervals(
            param_list: list[DatatypeCoefficientsForDividedLinearRegression]
    ) -> list[tuple[pd.Timestamp, float, float]]:

        """
        Order divided regression coefficients by start hour.
        """

        """
        Generate ordered (time, a, b) tuples from divided regression coefficients.
        """

        """
        Prepare sorted interval tuples from divided regression parameters.
        """

        """
        Convert calibration parameter dictionaries into ordered time intervals.
        """

        intervals = []
        for p in param_list:
            if all(k in p for k in ("hour", "a", "b")):
                hour = p["hour"]
                #if isinstance(hour, str):
                #    hour = pd.to_datetime(hour)
                intervals.append((hour, p["a"], p["b"]))

        return sorted(intervals, key=lambda x: x[0])

    intervals_sunny = build_intervals(params_sunny)

    if params_cloudy is not None:
        intervals_cloudy = build_intervals(params_cloudy)
    else:
        intervals_cloudy = None

    if not (intervals_sunny or intervals_cloudy):
        raise ValueError("At least params_sunny must contain valid (hour, a, b) entries.")

    x = df[sensor_name]
    x.index = df["time"]

    time = df["time"]
    time = pd.to_datetime(time)

    is_sunny = df["if_sunny"]
    is_sunny.index = df["time"]

    y_pred = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        current_time = time.iloc[i].time()
        current_params = intervals_sunny if is_sunny.iloc[i] else intervals_cloudy

        a, b = 0.0, 0.0
        for j, (t_start, a_j, b_j) in enumerate(current_params):
            t_end = (
                current_params[j + 1][0]
                if j + 1 < len(current_params)
                else pd.Timestamp.max
            )

            t_start = pd.to_datetime(t_start).time()
            t_end = pd.to_datetime(t_end).time()
            #current_time = current_time.time()

            if t_start <= current_time < t_end:
                a, b = a_j, b_j
                break

        y_pred[i] = a * x.iloc[i] + b

    result = pd.Series(y_pred, index=df["time"])

    return result


def divided_linear_regression_use_calibration_values_mean(
        df: pd.DataFrame,
        sensor_name: str,
        params_sunny:  list[DatatypeCoefficientsForDividedLinearRegression],
) -> pd.Series:
    """
    Do a calculation of Divided Linear Regression using calibration values. Same calculation as for Linear Regression,
    but divided for equal, specified periods. Using mean values.

    .. math::

            y = a * x + b
    """

    #df["if_sunny"] = True

    check_if_any_column_is_missing(
        df=df,
        sensor_name=sensor_name,
        time_col="time"
    )

    def build_intervals(
            param_list: list[DatatypeCoefficientsForDividedLinearRegression]
    ) -> list[tuple[pd.Timestamp, float, float]]:

        """
        Convert divided regression parameters into sorted interval tuples.
        """

        intervals = []
        for p in param_list:
            if all(k in p for k in ("hour", "a", "b")):
                hour = p["hour"]
                intervals.append((hour, p["a"], p["b"]))

        return sorted(intervals, key=lambda x: x[0])

    intervals_sunny = build_intervals(params_sunny)

    x = df[sensor_name]
    x.index = df["time"]

    time = df["time"]
    time = pd.to_datetime(time)

    y_pred = np.empty_like(x, index=df["time"], dtype=float)

    for i in range(len(x)):
        current_time = time.iloc[i].time()

        a, b = 0.0, 0.0
        for j, (t_start, a_j, b_j) in enumerate(intervals_sunny):
            t_end = (
                intervals_sunny[j + 1][0]
                if j + 1 < len(intervals_sunny)
                else str("23:59")
            )

            t_start = datetime.datetime.strptime(t_start).time()
            t_end = datetime.datetime.strptime(t_end).time()

            if t_start <= current_time < t_end:
                a, b = a_j, b_j
                break

        y_pred[i] = a * x.iloc[i] + b

    result = pd.Series(y_pred, index=df["time"])
    return result


def select_calibration_parameters(
        params_all: list[DatatypeCoefficientsForDividedLinearRegression],
        params_sunny: list[DatatypeCoefficientsForDividedLinearRegression],
        params_cloudy: list[DatatypeCoefficientsForDividedLinearRegression],
        df_time: pd.Series,
        frequency: str
) -> list[DatatypeCoefficientsForDividedLinearRegression]:
    """
    Note:
        There are two types of periods. Let's call them:

        * ``irradiance periods`` - determines if there is ``sunny``, ``cloudy`` or ``all`` period. Describing whether
          it is a full sunlight metrics or not
        * ``time periods`` - specified time interval

    1. Search for ``sunny`` metrics for all time periods
    2. Search for ``all`` metrics for all time periods
    3. If some time periods are missing in ``all`` metrics function will fill data with ``sunny`` metrics.
    4. If there are not enough ``sunny`` and ``all`` metrics - missing some time intervals, program raise an Error.
    """

    log.info("Checking coverage for sunny parameters...")

    try:
        sunny_ok = check_if_params_contains_data_for_all_time_intervals(params_sunny, df_time, frequency)
    except ValueError as e:
        log.warning(f"Sunny params incomplete: {e}")
        sunny_ok = False

    if sunny_ok:
        log.info("Sunny parameters have full coverage. Using them.")
        return params_sunny

    log.info("Checking coverage for all parameters...")

    try:
        all_ok = check_if_params_contains_data_for_all_time_intervals(params_all, df_time, frequency)
    except ValueError as e:
        log.warning(f"All params incomplete: {e}")
        all_ok = False

    json_hours_sunny = sorted([item["hour"] for item in params_sunny])

    df_time = pd.to_datetime(df_time)
    start = pd.Timestamp(df_time.iloc[0]).time()
    end = pd.Timestamp(df_time.iloc[-1]).time()

    expected_times = date_range_only_hh_mm(start=start, end=end, freq=frequency)
    missing_from_sunny = sorted(set(expected_times) - set(json_hours_sunny))

    if all_ok:
        if missing_from_sunny:
            log.info(f"Filling {len(missing_from_sunny)} missing intervals from all params.")

            merged = params_sunny.copy()
            merged_hours = {p["hour"]: p for p in merged}

            for p in params_all:
                if p["hour"] in missing_from_sunny and p["hour"] not in merged_hours:
                    merged.append(p)

            log.info("Calibration params merged successfully.")

            return sorted(merged, key=lambda x: x["hour"])
        else:
            return params_all

    else:
        log.error("Cannot calibrate! Not enough data in both sunny and all parameter sets.")
        raise ValueError("Cannot calibrate! Missing intervals in both sunny and all parameter sets.")


def date_range_only_hh_mm(
        start: datetime.time,
        end: datetime.time,
        freq: str
) -> list[datetime.time]:
    """
    Generates a list of time values between two times at a given frequency.

    The function creates a time-only range (HH:MM) by stepping from `start`
    to `end` using a pandas-compatible frequency string (e.g. "5min", "15min").
    The date component is fixed internally and not relevant to the output.
    """

    delta = pd.to_timedelta(freq)

    sample_date = datetime.date(2000, 1, 1)
    cur = datetime.datetime.combine(sample_date, start)
    stop = datetime.datetime.combine(sample_date, end)

    times = []
    while cur < stop:
        times.append(cur.time())
        cur += delta

    return times


def check_if_params_contains_data_for_all_time_intervals(
        coeffs,
        df_time: pd.Series,
        frequency: str
) -> bool:
    """
    Checks whether given coefficients contain data from all time intervals or not.
    """

    df_time = pd.to_datetime(df_time).dt.tz_localize(None).dt.tz_localize("Europe/Warsaw").dt.tz_convert("UTC")

    json_hours = sorted([item["hour"] for item in coeffs])

    day_start = pd.Timestamp(df_time.iloc[0]).strftime("%H:%M")
    day_end = pd.Timestamp(df_time.iloc[-1]).strftime("%H:%M")
    expected_times = pd.date_range(day_start, day_end, freq=frequency, inclusive="left").strftime("%H:%M").to_list()

    all_intervals_contained = set(json_hours).issuperset(expected_times)

    if not all_intervals_contained:
        raise ValueError(f"Not fully coverage for time intervals from calibrated matrics!")

    df_time = pd.to_datetime(df_time)

    df_start = df_time.iloc[0]
    df_end = df_time.iloc[-1]
    total_days = (df_end - df_start).days + 1

    counted_days = 0
    if len(coeffs) > 0 and "count_days" in coeffs[0]:
        counted_days = max(item.get("count_days", 0) for item in coeffs)

    missing_intervals = sorted(set(expected_times) - set(json_hours))

    log.debug(f"Expected intervals: {expected_times}")
    log.debug(f"JSON intervals: {json_hours}")
    log.debug(f"Missing intervals: {missing_intervals}")

    all_present = len(missing_intervals) == 0

    return all_present
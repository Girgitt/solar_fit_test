import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression



def compute_residual_metrics(
        poa,
        poa_pred
):

    resid = np.abs(poa - poa_pred)
    resid_slope = np.abs(np.gradient(resid))
    resid_smooth = gaussian_filter1d(resid, sigma=10)

    return resid, resid_slope, resid_smooth


def clearsky_detection(
        resid,
        resid_slope,
        resid_smooth,
        resid_thr=40,
        slope_thr=8,
        smooth_thr=30
):

    clear = (
        (resid < resid_thr) &
        (resid_slope < slope_thr) &
        (resid_smooth < smooth_thr)
    )

    return clear


#--------------------------------- FREQUENCY METHOD ---------------------------------#

def lowfreq_calibration_pipeline(
        df: pd.DataFrame,
        poa: pd.DataFrame,
        sensor_col: str,
        poa_col: str = "poa_global",
        time_col: str = "time",
        sampling_sec: int = 5
) -> tuple[pd.DataFrame, float, float]:

    sensor = df[sensor_col]
    poa = poa[poa_col]
    time = df[time_col]

    sensor.index = time

    # Step 1: Extract low-frequency trend
    sensor_lowfreq = extract_low_frequency(
        sensor=sensor,
        window_sec=300,
        sampling_sec=sampling_sec,
        polyorder=3
    )

    #FIXME - instead of linear regression calibration later planned change for: 1. RANSAC 2. Gaussian Process
    a, b, poa_pred_lowfreq = calibrate_lowfreq(sensor_lowfreq, poa)

    a = float(a)
    b = 0.0

    # Step 3: Predict POA from low-frequency trend
    poa_pred = predict_poa(sensor_lowfreq, a, b)

    poa_pred.index = time

    # Step 4 (optional): detect clear-sky
    clear, resid = detect_clear_sky(poa, poa_pred)

    # Return everything as a DataFrame
    result = pd.DataFrame({
        "sensor_raw": sensor,
        "sensor_lowfreq": sensor_lowfreq,
        "poa": poa,
        "poa_pred": poa_pred,
        "residual": resid,
        "clear_sky": clear
    }).set_index(time)

    return result, a, b


def extract_low_frequency(
        sensor: pd.Series,
        window_sec: int = 300,
        sampling_sec: int = 5,
        polyorder: int = 3
) -> pd.Series:

    # Convert seconds → number of samples
    window_length = int(window_sec / sampling_sec)

    # window length must be odd
    if window_length % 2 == 0:
        window_length += 1

    # Apply low-frequency filter
    filtered = savgol_filter(sensor.values,
                             window_length=window_length,
                             polyorder=polyorder,
                             mode='interp')

    return pd.Series(filtered, index=sensor.index)


def calibrate_lowfreq(
        sensor_lowfreq: pd.Series,
        poa: pd.Series
) -> [float, float, np.ndarray]:

    x = sensor_lowfreq.values.reshape(-1, 1)
    y = poa.values

    model = LinearRegression()
    model.fit(x, y)

    a = model.coef_[0]
    b = model.intercept_

    return a, b, model.predict(x)


def predict_poa(
        sensor_lowfreq: pd.Series,
        a: float,
        b: float
) -> pd.Series:

    return a * sensor_lowfreq + b


def detect_clear_sky(
        poa: pd.Series,
        poa_pred: pd.Series,
        threshold: float = 40
) -> [pd.Series, pd.Series]:

    resid = np.abs(poa - poa_pred)

    return resid < threshold, resid





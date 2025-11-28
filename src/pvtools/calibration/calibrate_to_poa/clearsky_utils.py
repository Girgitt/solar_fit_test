import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, ShortTimeFFT
from scipy.signal.windows import gaussian
from sklearn.linear_model import LinearRegression

from pvtools.visualisation.plotter import plot_frequency_histogram, plot_fft_spectrum


#--------------------------------- Savitzky-Golay Filter ---------------------------------#

def clearsky_detection_by_frequency_method(
        sensor: pd.Series,
        poa_global: pd.Series,
        time: pd.Series,
        sampling_sec: int = 5
) -> tuple[pd.DataFrame, float, float]:

    sensor.index = time
    poa_global.index = time

    sensor_lowfreq = extract_low_frequency(
        sensor=sensor,
        window_sec=300,
        sampling_sec=sampling_sec,
        polyorder=3
    )

    a, b, poa_pred_lowfreq = calibrate_lowfreq(sensor_lowfreq, poa_global)

    a = float(a)
    b = 0.0

    poa_pred = predict_poa(sensor_lowfreq, a, b)

    poa_pred.index = time

    clear, resid = detect_clear_sky(poa_global, poa_pred)

    result = pd.DataFrame({
        "sensor": sensor,
        "sensor_lowfreq": sensor_lowfreq,
        "poa": poa_global,
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

    window_length = int(window_sec / sampling_sec)

    if window_length % 2 == 0:
        window_length += 1

    filtered = savgol_filter(
        x=sensor.values,
        window_length=window_length,
        polyorder=polyorder,
        mode='interp'
    )

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


#--------------------------------- FREQUENCY ANALYSIS ---------------------------------#

def prepare_signal(sensor: pd.Series):

    if not isinstance(sensor.index, pd.DatetimeIndex):
        raise ValueError("Sensor series must have DatetimeIndex")

    # compute sampling period
    deltas = sensor.index.to_series().diff().dropna().dt.total_seconds()
    dt = deltas.median()           # typical sampling interval
    fs = 1.0 / dt                  # sampling frequency (Hz)

    signal = sensor.values.astype(float)

    return signal, fs

def compute_fft(
        signal: np.ndarray,
        fs: float
):

    N = len(signal)

    fft_raw = np.fft.rfft(signal)
    fft_mag = np.abs(fft_raw) / N

    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    return freqs, fft_mag

def frequency_analysis(
        sensor: pd.Series,
        time: pd.Series,
        max_freq=None
):

    sensor.index = time

    signal, fs = prepare_signal(sensor)
    freqs, fft_mag = compute_fft(signal, fs)

    print(f"Sampling frequency: {fs:.4f} Hz")
    print(f"Sampling interval: {1/fs:.3f} seconds")
    print(f"Number of samples: {len(signal)}")

    plot_fft_spectrum(
        freqs,
        fft_mag,
        max_freq=max_freq
    )

    plot_frequency_histogram(
        freqs=freqs,
        fft_mag=fft_mag,
        bins=100,
        title="Frequency Histogram of Irradiance Signal",
        save_dir=x,
        filename=y,
        show=False
    )

    return freqs, fft_mag


def low_frequency_mask(
        sensor: pd.Series,
        sampling_sec: float = 60.0,
        low_freq_max: float = 0.002,
        window_sec: float = 3600,
        thershold: float = 0.8
) -> pd.Series:

    x = sensor.values.astype(float)
    n_samples = len(x)

    fs = 1.0 / sampling_sec
    win_len = int(window_sec / sampling_sec)
    if win_len < 8:
        win_len = 8
    if win_len % 2 == 0:
        win_len += 1

    g_std = 0.4 * win_len
    window = gaussian(win_len, std=g_std, sym=True)

    sft = ShortTimeFFT(
        win=window,
        hop=1,  # 1-sample hop → full resolution
        fs=fs,
        fft_mode='onesided'
    )

    sx = sft.stft(x)
    mag = np.abs(sx)

    f = sft.f
    t = sft.t(n=n_samples)

    low_band = f <= low_freq_max

    low_energy = mag[low_band, :].sum(axis=0)
    total_energy = mag.sum(axis=0)

    ratio = low_energy / (total_energy + 1e-12)

    local_mask = ratio > thershold

    T = sft.T

    sample_idx = np.round(t / T).astype(int)
    sample_idx = np.clip(sample_idx, 0, n_samples - 1)

    bool_mask = pd.Series(False, index=sample_idx)
    bool_mask.iloc[sample_idx] = local_mask[sample_idx]
    bool_mask = bool_mask[~bool_mask.index.duplicated(keep='first')]

    bool_mask.index = sensor.index

    return bool_mask


#--------------------------------- TWO MEDIANS METHOD ---------------------------------#

def two_medians_mask(
        sensor: pd.Series,
        time: pd.Series,
        short_window: str = "30min",
        long_window: str = "4h",
        rel_threshold: float = 0.05,
        min_run_length: int = 5
) -> pd.Series:

    sensor.index = time

    if not isinstance(sensor.index, pd.DatetimeIndex):
        raise TypeError("sensor.index must be a DatetimeIndex")

    med_short = sensor.rolling(short_window, center=True, min_periods=1).median()
    med_long = sensor.rolling(long_window, center=True, min_periods=1).median()

    eps = 1e-9
    rel_diff = np.abs(med_short - med_long) / (np.abs(med_long) + eps)

    raw_mask = rel_diff < rel_threshold

    mask = raw_mask.copy()
    values = mask.values

    start = None
    for i in range(len(values)):
        if values[i] and start is None:
            start = i
        if (not values[i] or i == len(values) - 1) and start is not None:
            end = i if not values[i] else i + 1
            run_length = end - start
            if run_length < min_run_length:
                values[start:end] = False
            start = None

    mask = pd.Series(values, index=sensor.index)

    return mask


#--------------------------------- DERIVATIVE METHOD ---------------------------------#

def derivative_df(
        sensor: pd.Series,
        poa_global: pd.Series,
        time: pd.Series,
        window_length: int = 30,
        polyorder: int = 1,
        delta: float = 1.0
) -> pd.DataFrame:

    x = sensor.values.astype(float)
    y = poa_global.values.astype(float)

    x_smooth = savgol_filter(
        x=x,
        window_length=window_length,
        polyorder=polyorder,
        deriv=0,
        delta=delta
    )

    x_d_dt = savgol_filter(
        x=x,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta
    )

    y_smooth = savgol_filter(
        x=y,
        window_length=window_length,
        polyorder=polyorder,
        deriv=0,
        delta=delta
    )

    y_d_dt = savgol_filter(
        x=y,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta
    )

    x_smooth = pd.Series(x_smooth, index=time)
    x_d_dt = pd.Series(x_d_dt, index=time)

    y_smooth = pd.Series(y_smooth, index=time)
    y_d_dt = pd.Series(y_d_dt, index=time)

    df = pd.DataFrame({
        "sensor": sensor,
        "sensor_smooth": x_smooth,
        "sensor_d_dt": x_d_dt,
        "poa_global": poa_global,
        "poa_global_smooth": y_smooth,
        "poa_global_d_dt": y_d_dt,
    })

    return df

def relative_derivative_mask(
        sensor: pd.Series,
        poa_global: pd.Series,
        time: pd.Series,
        window_length: int = 30,
        polyorder: int = 1,
        delta: float = 1.0
) -> pd.DataFrame:

    x = sensor.values.astype(float)
    y = poa_global.values.astype(float)
    
    x_clip = np.clip(x, 1.0, None)
    y_clip = np.clip(y, 1.0, None)

    log_x = np.log(x_clip)
    log_y = np.log(y_clip)

    log_sensor_smooth = savgol_filter(
        x=log_x,
        window_length=window_length,
        polyorder=polyorder,
        deriv=0,
        delta=delta
    )
    log_poa_global_smooth = savgol_filter(
        x=log_y,
        window_length=window_length,
        polyorder=polyorder,
        deriv=0,
        delta=delta
    )

    log_sensor_d_dt = savgol_filter(
        x=log_x,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta
    )

    log_poa_global_d_dt = savgol_filter(
        x=log_y,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta
    )

    tau = 0.001 # need to be adjusted

    shape_diff = np.abs(log_sensor_d_dt - log_poa_global_d_dt)
    shape_mask = (shape_diff < tau)

    log_sensor_smooth = pd.Series(log_sensor_smooth, index=time)
    log_poa_global_smooth = pd.Series(log_poa_global_smooth, index=time)
    log_sensor_d_dt = pd.Series(log_sensor_d_dt, index=time)
    log_poa_global_d_dt = pd.Series(log_poa_global_d_dt, index=time)
    shape_mask = pd.Series(shape_mask, index=time)

    df = pd.DataFrame({
        "sensor": sensor,
        "sensor_smooth": log_sensor_smooth,
        "sensor_d_dt": log_sensor_d_dt,
        "poa_global": poa_global,
        "poa_global_smooth": log_poa_global_smooth,
        "poa_global_d_dt": log_poa_global_d_dt,
        "mask": shape_mask
    })

    return df







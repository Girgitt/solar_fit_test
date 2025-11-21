from collections import OrderedDict

import numpy as np
import pandas as pd

from pvlib import tools
from pvlib.clearsky import _clearsky_get_threshold, _calc_stats, _line_length_windowed, _max_diff_windowed, \
    _clear_sample_index
from scipy.linalg import hankel
from pvlib_mod.tools_mod import _get_sample_intervals_mod


def detect_clearsky_mod(measured: pd.Series,
                    clearsky: pd.Series,
                    times: pd.Series | pd.DatetimeIndex = None,
                    infer_limits: bool = False,
                    window_length: int = 10,
                    mean_diff: int = 75,
                    max_diff: int = 75,
                    lower_line_length = -5,
                    upper_line_length: int = 10,
                    var_diff: float = 0.005,
                    slope_dev: int = 8,
                    max_iterations: int = 20,
                    return_components: bool = False
                    ):

    times = pd.to_datetime(times)

    if times is None:
        try:
            times = measured.index
        except AttributeError:
            raise ValueError("times is required when measured is not a Series")

    # be polite about returning the same type as was input
    ispandas = isinstance(measured, pd.Series)

    # for internal use, need a Series
    if not ispandas:
        meas = pd.Series(measured, index=times)
    else:
        meas = measured

    if not isinstance(clearsky, pd.Series):
        clear = pd.Series(clearsky, index=times)
    else:
        clear = clearsky

    sample_interval, samples_per_window = \
        _get_sample_intervals_mod(times, window_length)

    # if infer_limits, find threshold values using the sample interval
    if infer_limits:
        (
            window_length,
            mean_diff,
            max_diff,
            lower_line_length,
            upper_line_length,
            var_diff,
            slope_dev,
        ) = _clearsky_get_threshold(sample_interval)

        # recalculate samples_per_window using returned window_length
        sample_interval, samples_per_window = tools._get_sample_intervals(
            times, window_length
        )

    if samples_per_window < 3:
        raise ValueError(
            f"Samples per window of {samples_per_window}"
            " found. Each window must contain at least 3 data"
            " points."
            f" Window length of {window_length} found. Increase"
            f" window length to {3 * sample_interval} or longer."
        )

    # check that we have enough data to produce a nonempty hankel matrix
    if len(times) < samples_per_window:
        raise ValueError(f"times has only {len(times)} entries, but it must \
                           have at least {samples_per_window} entries")

    # generate matrix of integers for creating windows with indexing
    H = hankel(np.arange(samples_per_window),
               np.arange(samples_per_window-1, len(times)))

    # calculate measurement statistics
    meas_mean, meas_max, meas_slope_nstd, meas_slope = _calc_stats(
        meas, samples_per_window, sample_interval, H)
    meas_line_length = _line_length_windowed(
        meas, H, samples_per_window, sample_interval)

    # calculate clear sky statistics
    clear_mean, clear_max, _, clear_slope = _calc_stats(
        clear, samples_per_window, sample_interval, H)

    # find a scaling factor for the clear sky time series that minimizes the
    # RMSE between the clear times identified in the measured data and the
    # scaled clear sky time series. Optimization to determine the scaling
    # factor considers all identified clear times, which is different from [1]
    # where the scaling factor was determined from clear times on days with
    # at least 50% of the day being identified as clear.
    alpha = 1
    for iteration in range(max_iterations):
        scaled_clear = alpha * clear
        clear_line_length = _line_length_windowed(
            scaled_clear, H, samples_per_window, sample_interval)

        line_diff = meas_line_length - clear_line_length
        slope_max_diff = _max_diff_windowed(
            meas - scaled_clear, H, samples_per_window)
        # evaluate comparison criteria
        c1 = np.abs(meas_mean - alpha*clear_mean) < mean_diff
        c2 = np.abs(meas_max - alpha*clear_max) < max_diff
        c3 = (line_diff > lower_line_length) & (line_diff < upper_line_length)
        c4 = meas_slope_nstd < var_diff
        c5 = slope_max_diff < slope_dev
        c6 = (clear_mean != 0) & ~np.isnan(clear_mean)
        clear_windows = c1 & c2 & c3 & c4 & c5 & c6

        # create array to return
        clear_samples = np.full_like(meas, False, dtype='bool')
        # find the samples contained in any window classified as clear
        idx = _clear_sample_index(clear_windows, samples_per_window, 'center',
                                  H)
        clear_samples[idx] = True

        # find a new alpha
        previous_alpha = alpha
        clear_meas = meas[clear_samples]
        clear_clear = clear[clear_samples]

        # Compute arg min of MSE between model and observations
        C = (clear_clear**2).sum()
        if not (pd.isna(C) or C == 0):  # safety check
            # only update alpha if C is strictly positive
            alpha = (clear_meas * clear_clear).sum() / C
        if round(alpha*10000) == round(previous_alpha*10000):
            break
    else:
        import warnings
        warnings.warn('rescaling failed to converge after %s iterations'
                      % max_iterations, RuntimeWarning)

    # be polite about returning the same type as was input
    if ispandas:
        clear_samples = pd.Series(clear_samples, index=times)

    if return_components:
        components = OrderedDict()
        components['mean_diff_flag'] = c1
        components['max_diff_flag'] = c2
        components['line_length_flag'] = c3
        components['slope_nstd_flag'] = c4
        components['slope_max_flag'] = c5
        components['mean_nan_flag'] = c6
        components['windows'] = clear_windows

        components['mean_diff'] = np.abs(meas_mean - alpha * clear_mean)
        components['max_diff'] = np.abs(meas_max - alpha * clear_max)
        components['line_length'] = meas_line_length - clear_line_length
        components['slope_nstd'] = meas_slope_nstd
        components['slope_max'] = slope_max_diff

        return clear_samples, components, alpha
    else:
        return clear_samples
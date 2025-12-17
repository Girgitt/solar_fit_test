import numpy as np
import pandas as pd
import warnings

def _get_sample_intervals_mod(
        times: pd.DatetimeIndex,
        win_length: int
) -> [float, int]:

    """
    Estimate sampling interval and window length in samples for uneven time indices.
    """

    deltas = np.diff(times.values) / np.timedelta64(1, "m")

    if len(deltas) == 0:
        raise ValueError("DatetimeIndex must contain at least two timestamps.")

    sample_interval = float(np.median(deltas))

    if np.any(np.abs(deltas - sample_interval) > 1e-6):
        warnings.warn(
            "Non-uniform time intervals detected. Using median interval "
            f"{sample_interval:.4f} minutes for calculations.",
            UserWarning
        )

    samples_per_window = int(round(win_length / sample_interval))

    return sample_interval, samples_per_window
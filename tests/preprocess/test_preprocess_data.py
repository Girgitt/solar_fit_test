import pytest
import pandas as pd
import numpy as np

from datetime import time
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_float_dtype

from pvtools.preprocess.preprocess_data import (delete_night_period,
                                                average_measurements,
                                                preprocess_data,
                                                ensure_datetime,
                                                ensure_target_frequency_is_lower_than_measurements)

@pytest.fixture
def tz_series():
    return pd.date_range(
        "2025-07-21 04:59:00",
        periods=6,
        freq="1min",
        tz="Europe/Warsaw"
    )

@pytest.fixture
def df_simple(tz_series):
    return pd.DataFrame(
        {"time": tz_series,
         "irr": [0,10,20,30,40,50],
         "ref": [0.1,9.9,20.2,29.7,40.1,50.3]}
    )

@pytest.fixture
def tz_utc_series():
    return np.arange(
        start=1755488310,
        stop=1755488316,
        step=1,
        dtype=np.int32
    )

@pytest.fixture
def df_utc_simple(tz_utc_series):
    return pd.DataFrame(
        {"time": tz_utc_series,
         "irr": [0.,10.,20.,30.,40.,50.],
         "ref": [0.1,9.9,20.2,29.7,40.1,50.3]
         }
    )

def test_ensure_datetime_converts_strings():
    df = pd.DataFrame({"time": ["2025-07-21 04:59:00", "2025-07-21 05:00:00"], "x": [1, 2]})
    out = ensure_datetime(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(out["time"])
    assert out["time"].dt.tz is None

def test_ensure_datetime_preserves_datetime(df_simple):
    out = ensure_datetime(df_simple.copy())
    pd.testing.assert_frame_equal(out, df_simple)

def test_ensure_datetime_converts_utc_in_seconds_to_datetime_object(df_utc_simple):
    out = ensure_datetime(df_utc_simple)
    assert is_datetime64_any_dtype(out["time"])
    assert is_float_dtype(out["irr"])

def test_ensure_datetime_missing_time_keyerror():
    with pytest.raises(KeyError):
        ensure_datetime(pd.DataFrame({"x": [1, 2]}))

def test_target_freq_check_not_enough_samples_raises():
    df = pd.DataFrame({"time": pd.to_datetime(["2025-01-01"])})
    with pytest.raises(ValueError, match="Not enough samples"):
        ensure_target_frequency_is_lower_than_measurements(df.copy(), "1min")

def test_target_freq_check_logs_when_actual_is_coarser(capsys):
    df = pd.DataFrame({"time": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:02:00"])})
    ensure_target_frequency_is_lower_than_measurements(df, "1min")
    assert "Nothing to do." in capsys.readouterr().out

def test_target_freq_check_when_actual_is_finer():
    df = pd.DataFrame({"time": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:30"])})
    ensure_target_frequency_is_lower_than_measurements(df, "1min")

@pytest.mark.parametrize(
    "start,end,expected_hours",
    [
        (time(3, 0), time(18, 0), {5}), # overlaps
        (time(3, 0), time(3, 2),  {5}), # narrow window inside 5 o’clock
    ],
)
def test_delete_night_period_filters(df_simple, start, end, expected_hours):
    out = delete_night_period(df_simple.copy(), start=start, end=end)
    assert not out.empty
    out_hours = set(out["time"].dt.hour.unique())
    assert out_hours.issubset(expected_hours)

def test_delete_night_period_inclusive_bounds(df_simple):
    start = time(3, 0) # 3:00 GMT -> 5:00 UTC+2
    end = time(3, 2) # 3:02 GMT -> 5:02 UTC+2
    out = delete_night_period(df_simple.copy(), start=start, end=end)
    assert list(out["time"].dt.time.unique()) == [time(5, 0), time(5, 1), time(5, 2)] # UTC+2 time

def test_average_measurements_resamples_mean(df_simple):
    out = average_measurements(df_simple.copy(), "2min")
    assert "time" in out.columns
    assert len(out) == 3
    assert np.isclose(out.loc[0, "irr"], 5.0)

def test_preprocess_data_roundtrip(tmp_path, df_simple):
    save_path = tmp_path / "out.csv"
    out = preprocess_data(df_simple.copy(), save_dir=save_path, target_timedelta="2min")
    assert len(out) == 3
    expected_file = save_path.parent / "filtered" / save_path.name
    assert expected_file.exists()
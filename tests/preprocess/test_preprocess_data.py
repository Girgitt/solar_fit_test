import pytest
import pandas as pd
import numpy as np

from datetime import time
from zoneinfo import ZoneInfo

from pvtools.preprocess.preprocess_data import (normalize_values,
                                                sanitize_filename,
                                                delete_night_period,
                                                average_measurements,
                                                preprocess_data,
                                                ensure_dataframe_contains_valid_data,
                                                ensure_datetime_contains_timezone,
                                                check_if_target_frequency_is_lower_than_measurements)

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

def test_preprocess_data_roundtrip(tmp_path, df_simple):
    save_path = tmp_path / "out.csv"
    out = preprocess_data(df_simple.copy(), save_dir=save_path, target_timedelta="2min")
    assert len(out) == 3
    expected_file = save_path.parent / "filtered" / save_path.name
    assert expected_file.exists()

def test_ensure_dataframe_contains_valid_data_int_column_with_nan_drops_rows():
    df = pd.DataFrame({"time": [1, np.nan, 3]})
    out = ensure_dataframe_contains_valid_data(df)
    assert len(out) == 2
    assert out.index.equals(pd.RangeIndex(0, 2))
    assert out["time"].tolist() == [1, 3]

def test_ensure_dataframe_contains_valid_data_float_column_with_nan_drops_rows():
    df = pd.DataFrame({"value": [1.0, np.nan, 2.0, 3.5]})
    out = ensure_dataframe_contains_valid_data(df)
    assert len(out) == 3
    assert out["value"].tolist() == [1.0, 2.0, 3.5]

def test_ensure_dataframe_contains_valid_data_datetime_with_tz_with_nan_drops_rows():
    dt = pd.Series(pd.to_datetime(["2025-07-21T12:00:00Z", None, "2025-07-21T12:01:00Z"]))
    df = pd.DataFrame({"time": dt})
    out = ensure_dataframe_contains_valid_data(df)
    assert len(out) == 2
    assert out["time"].isna().sum() == 0
    assert out.index.equals(pd.RangeIndex(0, 2))

def test_ensure_dataframe_contains_valid_data_datetime_without_tz_with_nan_drops_rows():
    dt = pd.to_datetime(["2025-07-21 12:00:00", None, "2025-07-21 12:01:00"])
    df = pd.DataFrame({"time": dt})
    out = ensure_dataframe_contains_valid_data(df)
    assert len(out) == 2
    assert out["time"].isna().sum() == 0

def test_ensure_dataframe_contains_valid_data_string_column_with_nan_drops_rows():
    df = pd.DataFrame({"label": ["a", None, "c", "d"]})
    out = ensure_dataframe_contains_valid_data(df)
    assert len(out) == 3
    assert out["label"].tolist() == ["a", "c", "d"]

def test_ensure_dataframe_contains_valid_data_object_column_with_nan_drops_rows():
    df = pd.DataFrame({"obj": [{"x": 1}, np.nan, [1, 2], ("t", 2)]})
    out = ensure_dataframe_contains_valid_data(df)
    assert len(out) == 3
    assert not out["obj"].isna().any()

def test_ensure_dataframe_contains_valid_data_all_nan_column_results_in_empty_dataframe():
    df = pd.DataFrame({"col": [np.nan, np.nan]})
    out = ensure_dataframe_contains_valid_data(df)
    assert out.empty
    assert list(out.columns) == ["col"]

def test_ensure_dataframe_contains_valid_data_mixed_types_nan_in_one_column_drops_row():
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, np.nan, 3.3],
        }
    )
    out = ensure_dataframe_contains_valid_data(df)
    assert len(out) == 2
    assert out["int_col"].tolist() == [1, 3]
    assert out["float_col"].tolist() == [1.1, 3.3]

def test_ensure_dataframe_contains_valid_data_multiple_nans_in_same_row_drops_that_row_once():
    df = pd.DataFrame(
        {
            "a": [1, np.nan, 3],
            "b": [np.nan, 2, 3],
            "c": [5, 6, np.nan],
        }
    )
    out = ensure_dataframe_contains_valid_data(df)
    assert out.empty

def test_ensure_dataframe_contains_valid_data_all_rows_have_some_nan_results_empty_dataframe():
    df = pd.DataFrame(
        {
            "x": [1, np.nan],
            "y": [np.nan, 2],
        }
    )
    out = ensure_dataframe_contains_valid_data(df)
    assert out.empty
    assert list(out.columns) == ["x", "y"]

def test_ensure_dataframe_contains_valid_data_no_nan_returns_dataframe_unchanged():
    df = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "label": ["a", "b", "c"],
        }
    )
    out = ensure_dataframe_contains_valid_data(df)
    pd.testing.assert_frame_equal(out, df.reset_index(drop=True))

def test_ensure_dataframe_contains_valid_data_empty_dataframe_returns_empty_dataframe():
    df = pd.DataFrame(columns=["a", "b"])
    out = ensure_dataframe_contains_valid_data(df)
    assert out.empty
    assert list(out.columns) == ["a", "b"]

def test_ensure_dataframe_contains_valid_data_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError):
        ensure_dataframe_contains_valid_data([1, 2, 3])

def test_ensure_datetime_contains_timezone_numeric_int_seconds_are_converted_and_tz_applied():
    df = pd.DataFrame({"time": [1755488310, 1755488316]})
    out = ensure_datetime_contains_timezone(df, tz_name="Europe/Warsaw")

    expected = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(ZoneInfo("Europe/Warsaw"))
    pd.testing.assert_series_equal(out["time"], expected, check_names=False)

def test_ensure_datetime_contains_timezone_numeric_float_seconds_are_converted_and_tz_applied():
    df = pd.DataFrame({"time": [1755488310.0, 1755488316.0]})
    out = ensure_datetime_contains_timezone(df, tz_name="Europe/Warsaw")

    expected = (
        pd.to_datetime(df["time"], unit="s", utc=True)
        .dt.tz_convert(ZoneInfo("Europe/Warsaw"))
    )
    pd.testing.assert_series_equal(out["time"], expected, check_names=False)

def test_ensure_datetime_contains_timezone_naive_datetime_is_localized():
    dt = pd.to_datetime(["2025-07-21 12:00:00", "2025-07-21 12:01:00"])
    df = pd.DataFrame({"time": dt})
    out = ensure_datetime_contains_timezone(df, tz_name="Europe/Warsaw")

    assert isinstance(out['time'].dtype, pd.DatetimeTZDtype)
    assert str(out["time"].dt.tz) == "Europe/Warsaw"

def test_ensure_datetime_contains_timezone_already_tz_aware_datetime_is_preserved():
    dt = pd.date_range("2025-07-21 12:00:00+00:00", periods=2, freq="min", tz="UTC")
    df = pd.DataFrame({"time": dt})

    out = ensure_datetime_contains_timezone(df, tz_name="Europe/Warsaw")

    pd.testing.assert_series_equal(out["time"], df["time"], check_names=False)

def test_ensure_datetime_contains_timezone_string_of_epoch_seconds_is_cast_and_converted():
    df = pd.DataFrame({"time": ["1755488310", "1755488316"]})
    out = ensure_datetime_contains_timezone(df, tz_name="Europe/Warsaw")

    expected = pd.to_datetime(pd.Series([1755488310, 1755488316]), unit="s", utc=True).dt.tz_convert("Europe/Warsaw")

    pd.testing.assert_series_equal(
        out["time"].dt.tz_convert("UTC"),
        expected.dt.tz_convert("UTC"),
        check_names=False,
        check_freq=False,
    )

def test_ensure_datetime_contains_timezone_object_mixed_digit_strings_and_ints_are_converted():
    df = pd.DataFrame({"time": ["1755488310", 1755488316]})
    out = ensure_datetime_contains_timezone(df, tz_name="Europe/Warsaw")

    expected_ = pd.to_datetime([1755488310, 1755488316], unit="s", utc=True).tz_convert("Europe/Warsaw")
    expected = pd.Series(expected_, name="time", index=out.index)

    pd.testing.assert_series_equal(out["time"].dt.tz_convert("UTC"),
                                   expected.dt.tz_convert("UTC"),
                                   check_names=False
                                   )

def test_ensure_datetime_contains_timezone_invalid_object_values_raise_typeerror():
    df = pd.DataFrame({"time": ["not_a_number", "123abc"]})
    with pytest.raises(TypeError, match="Unsupported 'time' format"):
        ensure_datetime_contains_timezone(df, tz_name="Europe/Warsaw")

def test_ensure_datetime_contains_timezone_missing_time_column_raises_keyerror():
    df = pd.DataFrame({"t": [1, 2, 3]})
    with pytest.raises(KeyError, match="must contain a 'time'"):
        ensure_datetime_contains_timezone(df, tz_name="Europe/Warsaw")

@pytest.mark.parametrize(
    "start,end,expected_hours",
    [
        (time(3, 0), time(18, 0), {5}), # overlaps
        (time(3, 0), time(3, 2),  {5}), # narrow window inside 5 o’clock
    ]
)
def test_delete_night_period_filters(df_simple, start, end, expected_hours):
    out = delete_night_period(df_simple.copy(), start=start, end=end)
    assert not out.empty
    out_hours = set(out["time"].dt.hour.unique())
    assert out_hours.issubset(expected_hours)

def test_delete_night_period_inclusive_bounds(df_simple):
    start = time(3, 0)  # 3:00 GMT -> 5:00 UTC+2
    end = time(3, 2)  # 3:02 GMT -> 5:02 UTC+2
    out = delete_night_period(df_simple.copy(), start=start, end=end)
    assert list(out["time"].dt.time.unique()) == [time(5, 0), time(5, 1), time(5, 2)]  # UTC+2 time

def test_check_if_target_freq_check_not_enough_samples_raises():
    df = pd.DataFrame({"time": pd.to_datetime(["2025-01-01"])})
    with pytest.raises(ValueError, match="Not enough samples"):
        check_if_target_frequency_is_lower_than_measurements(df.copy(), "1min")

def test_check_if_target_freq_check_logs_when_actual_is_coarser(capsys):
    df = pd.DataFrame({"time": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:02:00"])})
    check_if_target_frequency_is_lower_than_measurements(df, "1min")
    assert "Nothing to do." in capsys.readouterr().out

def test_check_if_target_freq_check_when_actual_is_finer():
    df = pd.DataFrame({"time": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:30"])})
    check_if_target_frequency_is_lower_than_measurements(df, "1min")

def test_average_measurements_resamples_mean(df_simple):
    out = average_measurements(df_simple.copy(), "2min")
    assert "time" in out.columns
    assert len(out) == 4

def test_normalize_values_basic():
    df = pd.DataFrame({
        'A': [10, 20, 30],
        'B': [100, 200, 300],
        'C': ['x', 'y', 'z']
    })

    result = normalize_values(df)

    expected_A = [0.0, 0.5, 1.0]
    expected_B = [0.0, 0.5, 1.0]

    np.testing.assert_array_almost_equal(result['A'], expected_A)
    np.testing.assert_array_almost_equal(result['B'], expected_B)

    # Column 'C' should remain unchanged
    assert all(result['C'] == df['C'])

def test_normalize_values_invalid_inputs():
    invalid_inputs_df = [
        None,
        pd.Series([1, -2, 3]),
        np.array([[1, -1], [2, -2]]),
        "not a dataframe",
        123
    ]

    for invalid_df in invalid_inputs_df:
        with pytest.raises(TypeError, match="Expected 'df' to be a pandas DataFrame"):
            normalize_values(invalid_df)


def test_sanitize_filename():
    assert sanitize_filename("user@example.com") == "example_com"
    assert sanitize_filename("test@site.name!2023") == "site_name_2023"
    assert sanitize_filename("@admin@dev-site.com") == "dev-site_com"
    assert sanitize_filename("plain_string") == "plain_string"
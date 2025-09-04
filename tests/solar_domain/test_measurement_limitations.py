import pytest
import pandas as pd
import numpy as np

from pathlib import Path
from unittest.mock import patch

from pvtools.solar_domain.measurement_limitations import (limit_measured_irradiance_to_clear_sky_model,
                                                          remove_negative_measurements
                                                          )

@pytest.fixture()
def sensor_names():
    return ['sensor1', 'sensor2']

@pytest.fixture()
def mismatch_timestamps():
    return pd.date_range(start="2025-01-01 05:02:00+02:00", periods=4, freq="min")

@pytest.fixture
def timestamps():
    return pd.date_range(start="2025-01-01 05:00:00+02:00", periods=4, freq="min")

@pytest.fixture
def measured_df(timestamps, sensor_names):
    return pd.DataFrame({
        'time': timestamps,
        sensor_names[0]: [0.5, 1.5, 1.0, 2.0],
        sensor_names[1]: [0.6, 0.8, 1.2, 1.5],
        'sensor_ref': [0.4, 0.9, 1.3, 2.1]
    })

@pytest.fixture()
def negative_measured_df(sensor_names, timestamps):
    return pd.DataFrame({
        'time': timestamps,
        sensor_names[0]: [0.5, -1.0, 2.0, -0.3],
        sensor_names[1]: [-0.2, 0.0, 1.0, -5.0],
        'sensor_ref': [0.1, 0.2, 0.3, 0.4],
    })

@pytest.fixture
def clear_sky_df(timestamps):
    return pd.DataFrame({
        'time': timestamps,
        'poa_global': [1.0, 1.2, 1.1, 1.5],
        'poa_direct': [0.6, 0.7, 0.6, 0.9]
    })

@pytest.fixture
def clear_sky_df_with_mismatch_timestamps(timestamps):
    return pd.DataFrame({
        'time': mismatch_timestamps,
        'poa_global': [1.0, 1.2, 1.1, 1.5],
        'poa_direct': [0.6, 0.7, 0.6, 0.9]
    })

def test_limit_measured_irradiance_to_clear_sky_model_basic(
        measured_df,
        clear_sky_df,
        sensor_names,
):
    expected = measured_df.copy()
    sensor1 = sensor_names[0]
    sensor2 = sensor_names[1]

    expected[sensor1] = [0.5, 1.2, 1.0, 1.5]
    expected[sensor2] = [0.6, 0.8, 1.1, 1.5]

    result = limit_measured_irradiance_to_clear_sky_model(
        df=measured_df,
        clear_sky_df=clear_sky_df,
        poa_global_column_name='poa_global',
        save_dir=None
    )

    pd.testing.assert_series_equal(result[sensor1], expected[sensor1])
    pd.testing.assert_series_equal(result[sensor2], expected[sensor2])

def test_limit_measured_irradiance_to_clear_sky_model_with_no_time_column(
        measured_df,
        clear_sky_df,
):
    with pytest.raises(ValueError, match="'time' column needs to be provided!"):
        limit_measured_irradiance_to_clear_sky_model(
            df=measured_df.drop('time', axis='columns'),
            clear_sky_df=clear_sky_df,
            poa_global_column_name='poa_global',
            save_dir=None
        )

        with pytest.raises(ValueError, match="'time' column needs to be provided!"):
            limit_measured_irradiance_to_clear_sky_model(
                df=measured_df,
                clear_sky_df=clear_sky_df.drop('time', axis='columns'),
                poa_global_column_name='poa_global',
                save_dir=None
            )

def test_limit_measured_irradiance_to_clear_sky_model_with_mismatch_timestamps(
        measured_df,
        clear_sky_df_with_mismatch_timestamps,
        sensor_names,
):
    with pytest.raises(ValueError, match='Timestamps are mismatched!'):
        limit_measured_irradiance_to_clear_sky_model(
            df=measured_df,
            clear_sky_df=clear_sky_df_with_mismatch_timestamps,
            poa_global_column_name='poa_global',
        )

def test_limit_measured_irradiance_to_clear_sky_model_invalid_inputs(measured_df, clear_sky_df):
    invalid_inputs = [
        None,
        pd.Series([1, 2, 4]),
        np.array([[1, 5], [7, 1]]),
        "not a dataframe",
        44
    ]

    for invalid_df in invalid_inputs:
        with pytest.raises(TypeError, match="Expected 'df' and 'clear_sky_df' to be a pandas DataFrame"):
            limit_measured_irradiance_to_clear_sky_model(
                df=invalid_df,
                clear_sky_df=clear_sky_df,
                poa_global_column_name='poa_global',
                save_dir=None
            )

    for invalid_clear_sky_df in invalid_inputs:
        with pytest.raises(TypeError, match="Expected 'df' and 'clear_sky_df' to be a pandas DataFrame"):
            limit_measured_irradiance_to_clear_sky_model(
                df=measured_df,
                clear_sky_df=invalid_clear_sky_df,
                poa_global_column_name='poa_global',
                save_dir=None
            )

def test_remove_negative_measurements_basic(
        sensor_names,
        timestamps,
        negative_measured_df
):
    expected = pd.DataFrame({
        'time': timestamps,
        sensor_names[0]: [0.5, 0.0, 2.0, 0.0],
        sensor_names[1]: [0.0, 0.0, 1.0, 0.0],
        'sensor_ref': [0.1, 0.2, 0.3, 0.4],
    })

    result = remove_negative_measurements(negative_measured_df)
    pd.testing.assert_frame_equal(result, expected)

def test_remove_negative_measurements_no_negatives(measured_df):
    result = remove_negative_measurements(measured_df)
    pd.testing.assert_frame_equal(result, measured_df)

def test_remove_negative_measurements_empty_df():
    df = pd.DataFrame()
    result = remove_negative_measurements(df)

    assert result.empty

def test_remove_negative_measurements_invalid_inputs():
    invalid_inputs = [
        None,
        pd.Series([1, -2, 3]),
        np.array([[1, -1], [2, -2]]),
        "not a dataframe",
        123
    ]

    for invalid_df in invalid_inputs:
        with pytest.raises(TypeError, match="Expected 'df' to be a pandas DataFrame"):
            remove_negative_measurements(invalid_df)

@patch("pvtools.solar_domain.measurement_limitations.save_dataframe_to_csv")
def test_remove_negative_measurements_saves_on_change(
        mock_to_csv,
        negative_measured_df,
        tmp_path
):
    _ = remove_negative_measurements(negative_measured_df, save_dir=tmp_path / "output.csv")

    mock_to_csv.assert_called_once()

@patch("pvtools.solar_domain.measurement_limitations.save_dataframe_to_csv")
def test_remove_negative_measurements_skips_save_on_no_change(
        mock_to_csv,
        measured_df,
        tmp_path
):
    _ = remove_negative_measurements(measured_df, save_dir=tmp_path)

    mock_to_csv.assert_not_called()
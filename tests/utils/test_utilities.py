import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from pvtools.utils.utilities import select_available_data_columns_to_process

@pytest.fixture
def sample():
    data_columns = ["s0", "s1", "s2", "s3"]
    df = pd.DataFrame({
        "s0": [1.0,  None, 3.0, 4.0],
        "s1": [1.0,  2.0,  None, 4.0],
        "s2": [None, 2.0,  3.0, 4.0],
        "s3": [1.0,  2.0,  3.0, None],
    })
    return data_columns, df

def test_sensors_and_reference(sample):
    data_columns, df = sample
    sensors, ref, df_out = select_available_data_columns_to_process(
        data_columns, df, sensors_chosen=[0, 2], sensor_ref_chosen=1
    )
    assert sensors == ["s0", "s2"]
    assert ref == "s1"
    expected = df.dropna(subset=["s0", "s2", "s1"])
    assert_frame_equal(df_out, expected)

def test_no_reference(sample):
    data_columns, df = sample
    sensors, ref, df_out = select_available_data_columns_to_process(
        data_columns, df, sensors_chosen=[0, 2], sensor_ref_chosen=None
    )
    assert sensors == ["s0", "s2"]
    assert ref is None
    expected = df.dropna(subset=["s0", "s2"])
    assert_frame_equal(df_out, expected)

def test_no_sensors_raises(sample):
    data_columns, df = sample
    with pytest.raises(ValueError, match="cannot be empty"):
        select_available_data_columns_to_process(
            data_columns, df, sensors_chosen=[], sensor_ref_chosen=1
        )

def test_no_sensors_and_no_reference_raises(sample):
    data_columns, df = sample
    with pytest.raises(ValueError, match="cannot be empty"):
        select_available_data_columns_to_process(
            data_columns, df, sensors_chosen=[], sensor_ref_chosen=None
        )

def test_reference_included_in_sensors_raises(sample):
    data_columns, df = sample
    with pytest.raises(ValueError, match="overlap"):
        select_available_data_columns_to_process(
            data_columns, df, sensors_chosen=[0, 1], sensor_ref_chosen=1
        )

def test_two_or_more_references_raises(sample):
    data_columns, df = sample

    with pytest.raises(TypeError, match="int or None"):
        select_available_data_columns_to_process(
            data_columns, df, sensors_chosen=[0, 2], sensor_ref_chosen=[1, 3]
        )
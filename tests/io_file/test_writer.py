import pytest
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from types import SimpleNamespace

from pvtools.io_file.writer import (
    save_metrics_to_json,
    save_true_and_predicted_data_to_csv,
    save_dataframe_to_csv,
    save_figure,
    save_predicted_data_figures,
)

@pytest.fixture
def sample_metrics() -> SimpleNamespace:
    return SimpleNamespace(
        mse=1.2345,
        mae=0.5,
        rmse=1.1111,
        r2=0.9876,
        mape=2.5,
        max_error=3.3,
        bias=-0.05,
    )

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [0.1, 0.2, 0.3],
        }
    )


@pytest.fixture
def sample_arrays():
    y_true = np.array([10.0, 20.0, 30.0], dtype=float)
    y_pred = np.array([12.0, 19.5, 29.0], dtype=float)
    return y_true, y_pred

def test_save_metrics_to_json_with_coefficients(
        tmp_path: Path,
        sample_metrics: SimpleNamespace
):
    out_path = tmp_path / "metrics" / "result.json"
    coeffs = [{"hour": "2025-07-21T10:00:00", "a": 1.0, "b": 0.1}]

    save_metrics_to_json(
        metrics=sample_metrics,
        samples_count=123,
        coefficients_list=coeffs,
        filename_path=out_path,
    )

    assert out_path.exists(), "Metrics JSON file was not created."

    with open(out_path, "r") as f:
        data = json.load(f)

    for key in ["mse", "mae", "rmse", "r2", "mape", "max_error", "bias", "n_samples", "coefficients"]:
        assert key in data, f"Missing key '{key}' in metrics JSON."

    assert data["n_samples"] == 123
    assert isinstance(data["coefficients"], list)
    assert data["coefficients"] == coeffs

def test_save_metrics_to_json_without_coefficients(tmp_path: Path, sample_metrics: SimpleNamespace):
    out_path = tmp_path / "metrics" / "result_no_coeffs.json"

    save_metrics_to_json(
        metrics=sample_metrics,
        samples_count=3,
        coefficients_list=None,
        filename_path=out_path,
    )

    with open(out_path, "r") as f:
        data = json.load(f)

    assert out_path.exists()
    assert "coefficients" not in data
    assert data["n_samples"] == 3

def test_save_metrics_to_json_invalid_inputs_metrics():
    invalid_inputs = [
        None,
        pd.DataFrame(),
        "just string",
        124
    ]

    for metric in invalid_inputs:
        with pytest.raises(TypeError, match=f"metrics must have '.*' attribute"):
            save_metrics_to_json(
                metrics=metric,
                samples_count=1,
                coefficients_list=[],
                filename_path=None,
            )

def test_save_metrics_to_json_invalid_input_samples_count(sample_metrics):
    invalid_inputs = [
        None,
        pd.DataFrame(),
        "just string",
        [1, 5, 10]
    ]

    for sample_count in invalid_inputs:
        with pytest.raises(TypeError, match="samples_count must be an int"):
            save_metrics_to_json(
                metrics=sample_metrics,
                samples_count=sample_count,
                coefficients_list=[],
                filename_path=None,
            )

def test_save_true_and_predicted_data_to_csv_default_index(tmp_path: Path, sample_arrays):
    y_true, y_pred = sample_arrays
    out_path = tmp_path / "pred" / "y.csv"

    save_true_and_predicted_data_to_csv(y_true=y_true, y_pred=y_pred, output_path=out_path)

    assert out_path.exists(), "Output CSV was not created."
    df = pd.read_csv(out_path)

    assert list(df.columns) == ["index", "y_true", "y_pred"]
    assert df["index"].tolist() == [0, 1, 2]
    assert np.allclose(df["y_true"].values, y_true)
    assert np.allclose(df["y_pred"].values, y_pred)

def test_save_true_and_predicted_data_to_csv_custom_index(tmp_path: Path, sample_arrays):
    y_true, y_pred = sample_arrays
    custom_index = np.array([101, 105, 108])
    out_path = tmp_path / "pred_custom" / "y.csv"

    save_true_and_predicted_data_to_csv(
        y_true=y_true, y_pred=y_pred, output_path=out_path, index=custom_index
    )

    assert out_path.exists()
    df = pd.read_csv(out_path)

    assert df["index"].tolist() == custom_index.tolist()
    assert np.allclose(df["y_true"].values, y_true)
    assert np.allclose(df["y_pred"].values, y_pred)

def test_save_dataframe_to_csv_no_index(tmp_path: Path, sample_df: pd.DataFrame):
    out_path = tmp_path / "df" / "data.csv"
    save_dataframe_to_csv(df=sample_df, output_path=out_path, index=False)

    assert out_path.exists()
    df_loaded = pd.read_csv(out_path)

    assert list(df_loaded.columns) == ["a", "b"]
    pd.testing.assert_frame_equal(df_loaded, sample_df.reset_index(drop=True))

def test_save_dataframe_to_csv_with_index_and_label(tmp_path: Path, sample_df: pd.DataFrame):
    out_path = tmp_path / "df" / "data_with_index.csv"
    save_dataframe_to_csv(df=sample_df, output_path=out_path, index=True, index_label="row_id")

    assert out_path.exists()
    df_loaded = pd.read_csv(out_path)

    assert "row_id" in df_loaded.columns
    expect = sample_df.copy()
    expect = expect.reset_index().rename(columns={"index": "row_id"})
    pd.testing.assert_frame_equal(df_loaded, expect)

def test_save_figure_creates_png_file(tmp_path: Path):
    fig = plt.figure()
    try:
        save_dir = tmp_path / "figs"
        filename = "plot.png"

        save_figure(fig=fig, save_dir=save_dir, filename=filename)

        out_path = save_dir / filename
        assert out_path.exists(), "Figure file was not created."
        assert out_path.stat().st_size > 0, "Figure file seems empty."
    finally:
        plt.close(fig)


def test_save_predicted_data_figures_multiple(tmp_path: Path):
    figs = []
    f1 = plt.figure()
    f2 = plt.figure()
    figs.append(("sensor_A", "linear", f1))
    figs.append(("sensor_B", "mlp", f2))

    try:
        out_dir = tmp_path / "pred_figs"
        save_predicted_data_figures(figures=figs, save_dir=out_dir)

        for sensor_name in ["sensor_A", "sensor_B"]:
            p = out_dir / f"{sensor_name}.png"
            assert p.exists(), f"Expected saved figure for {sensor_name}."
            assert p.stat().st_size > 0
    finally:
        plt.close(f1)
        plt.close(f2)
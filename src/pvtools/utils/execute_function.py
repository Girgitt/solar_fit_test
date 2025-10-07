from pathlib import Path

from pvtools.calibration.calibrate import calibrate_by_linear_regression, calibrate_by_divided_linear_regression, \
    calibrate_by_polynominal_regression, calibrate_by_decision_tree_regression, calibrate_by_mlp_regression
from pvtools.config.params import ModelParameters, ClearSkyCalculatedValues
from pvtools.visualisation.plotter import plot_from_dataframe, plot_predicted_data, plot_poa_vs_reference, \
    plot_poa_reference_with_clearsky_periods
from pvtools.io_file.reader import load_dataframe_from_csv
from pvtools.utils.utilities import load_filtered_and_calculated_data_needed_for_execute_function, sanitize_filename
from pvtools.postprocess.postprocess_data import postprocess_data

def execute_function(
        model_parameters: ModelParameters,
        clearsky_calculated_values: ClearSkyCalculatedValues
) -> None:
    [df, poa, clearsky_periods] = load_filtered_and_calculated_data_needed_for_execute_function(model_parameters)

    model_parameters.df = df
    clearsky_calculated_values.poa = poa
    clearsky_calculated_values.clearsky_periods = clearsky_periods

    calibrate(model_parameters=model_parameters)

    postprocess_df = postprocess_data(
        model_parameters=model_parameters,
        clearsky_df=clearsky_calculated_values.poa,
        poa_global_name='poa_global'
    )

    model_parameters.df = postprocess_df

    plot(
        model_parameters=model_parameters,
        clearsky_calculated_values=clearsky_calculated_values
    )

def calibrate(model_parameters: ModelParameters) -> None:
    calibrate_by_linear_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=model_parameters.filename
    )

    calibrate_by_divided_linear_regression(
        df=model_parameters.df,
        df_time=model_parameters.df_time,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=model_parameters.filename
    )

    calibrate_by_polynominal_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=model_parameters.filename
    )

    calibrate_by_decision_tree_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=model_parameters.filename
    )

    calibrate_by_mlp_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=model_parameters.filename
    )

def plot(
        model_parameters: ModelParameters,
        clearsky_calculated_values: ClearSkyCalculatedValues,
) -> None:

    df_org = load_dataframe_from_csv(Path(model_parameters.data_dir / "filtered" / f"{model_parameters.filename}.csv"))
    df_org.columns = [sanitize_filename(name) for name in df_org.columns]

    df_filtered = load_dataframe_from_csv(Path(model_parameters.data_dir /
                                               "filtered" / "calibrated" /
                                               model_parameters.filename /
                                               f"{model_parameters.args.calibration}.csv"))

    plot_from_dataframe(
        df=df_org,
        save_dir=model_parameters.plot_dir / model_parameters.filename,
        filename="org_series_vs_time.png",
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        show=True,
        title="Original series vs. time"
    )

    plot_from_dataframe(
        df=df_filtered,
        save_dir=model_parameters.plot_dir / model_parameters.filename,
        filename=f"predicted_series_vs_time_{model_parameters.args.calibration}.png",
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        show=True,
        title="Predicted series vs. time"
    )

    plot_predicted_data(
        calibration_method_dir=model_parameters.log_dir / model_parameters.filename,
        show=False,
        save_dir=model_parameters.plot_dir / model_parameters.filename,
    )

    plot_poa_vs_reference(
        poa_global=clearsky_calculated_values.poa['poa_global'],
        sensor_reference=model_parameters.df[model_parameters.sensor_name_ref],
        save_dir=model_parameters.plot_dir / model_parameters.filename,
        show=True,
    )

    plot_poa_reference_with_clearsky_periods(
        poa_global=clearsky_calculated_values.poa['poa_global'],
        sensor_reference=model_parameters.df[model_parameters.sensor_name_ref],
        sunny=clearsky_calculated_values.clearsky_periods['if_sunny'],
        save_dir=model_parameters.plot_dir / model_parameters.filename,
        show=True,
    )
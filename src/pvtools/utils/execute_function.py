from pathlib import Path

from pvtools.calibration.calibrate import calibrate_by_linear_regression, calibrate_by_divided_linear_regression, \
    calibrate_by_polynominal_regression, calibrate_by_decision_tree_regression, calibrate_by_mlp_regression
from pvtools.config.params import ModelParameters, ClearSkyCalculatedValues
from pvtools.visualisation.plotter import plot_raw_data, plot_predicted_data, plot_poa_vs_reference, \
    plot_poa_reference_with_clearsky_periods, plot_raw_data_with_peaks
from pvtools.io_file.reader import load_dataframe_from_csv, load_calibrated_data
from pvtools.utils.utilities import load_filtered_and_calculated_data_needed_for_execute_function
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

    load_calibrated_data(model_parameters)

    postprocess_df = postprocess_data(
        df=model_parameters.df,
        clearsky_df=clearsky_calculated_values.poa,
        sensor_names=model_parameters.sensor_names,
        poa_global_name='poa_global',
        save_dir=model_parameters.data_dir,
        filename=model_parameters.filename,
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
        folder_data_name=Path(model_parameters.args.csv).stem
    )

    calibrate_by_divided_linear_regression(
        df=model_parameters.df,
        df_time=model_parameters.df_time,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=Path(model_parameters.args.csv).stem
    )

    calibrate_by_polynominal_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=Path(model_parameters.args.csv).stem
    )

    calibrate_by_decision_tree_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=Path(model_parameters.args.csv).stem
    )

    calibrate_by_mlp_regression(
        df=model_parameters.df,
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        log_dir=model_parameters.log_dir,
        folder_data_name=Path(model_parameters.args.csv).stem
    )

def plot(
        model_parameters: ModelParameters,
        clearsky_calculated_values: ClearSkyCalculatedValues,
) -> None:
    plot_raw_data(
        df=model_parameters.df,
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        filename="series_vs_time.png",
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        show=True,
    )

    # ----------------------------------- TEMPORARY PLOTTING FOR FILTERED DATA -----------------------------------------
    plot_raw_data(
        df=load_dataframe_from_csv(
            model_parameters.data_dir / "filtered" / model_parameters.filename),
        save_dir=model_parameters.plot_dir / model_parameters.filename,
        filename="series_vs_time_filtered.png",
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        show=True,
    )

    plot_predicted_data(
        calibration_method_dir=model_parameters.log_dir / Path(model_parameters.args.csv).stem,
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

    plot_raw_data_with_peaks(
        df=model_parameters.df,
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        peaks_dir=Path("data/interpolated") / model_parameters.filename,
        filename="series_vs_time_with_peaks",
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        show=True
    )
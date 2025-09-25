from pathlib import Path

from pvtools.analysis.analyse_calibration import calibrate_by_linear_regression, calibrate_by_divided_linear_regression, \
    calibrate_by_polynominal_regression, calibrate_by_decision_tree_regression, calibrate_by_mlp_regression
from pvtools.config.params import ModelParameters, ClearSkyCalculatedValues
from pvtools.visualisation.plotter import plot_raw_data, plot_predicted_data, plot_poa_vs_reference, \
    plot_poa_reference_with_clearsky_periods, plot_raw_data_with_peaks
from pvtools.io_file.reader import load_dataframe_from_csv

def execute_function(
        model_parameters: ModelParameters,
        clear_sky_calculated_values: ClearSkyCalculatedValues,
) -> None:
    calibrate(model_parameters=model_parameters)
    plot(
        model_parameters=model_parameters,
        clear_sky_calculated_values=clear_sky_calculated_values
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
        clear_sky_calculated_values: ClearSkyCalculatedValues,
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
            model_parameters.data_filename_dir.parent / "filtered" / Path(model_parameters.args.csv).name),
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        filename="series_vs_time_filtered.png",
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        show=True,
    )

    plot_predicted_data(
        calibration_method_dir=model_parameters.log_dir / Path(model_parameters.args.csv).stem,
        show=False,
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
    )

    plot_poa_vs_reference(
        poa_global=clear_sky_calculated_values.poa['poa_global'],
        sensor_reference=model_parameters.df[model_parameters.sensor_name_ref],
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        show=True,
    )

    plot_poa_reference_with_clearsky_periods(
        poa_global=clear_sky_calculated_values.poa['poa_global'],
        sensor_reference=model_parameters.df[model_parameters.sensor_name_ref],
        sunny=clear_sky_calculated_values.clearsky_periods['if_sunny'],
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        show=True,
    )

    plot_raw_data_with_peaks(
        df=model_parameters.df,
        save_dir=model_parameters.plot_dir / Path(model_parameters.args.csv).stem,
        peaks_dir=Path("data/interpolated") / Path(model_parameters.data_filename_dir).stem,
        filename="series_vs_time_with_peaks",
        sensor_names=model_parameters.sensor_names,
        sensor_name_ref=model_parameters.sensor_name_ref,
        show=True
    )
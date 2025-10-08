import pandas as pd

from pvtools.solar_domain.measurement_limitations import limit_sensors_irradiance_to_clear_sky_model, remove_negative_measurements
from pvtools.io_file.reader import load_calibrated_data
from pvtools.io_file.writer import save_dataframe_to_csv
from pvtools.config.params import ModelParameters


def postprocess_data(
        model_parameters: ModelParameters,
        clearsky_df: pd.DataFrame,
        poa_global_name: str = 'poa_global'
) -> pd.DataFrame:
    df = load_calibrated_data(model_parameters)

    limit_df = limit_sensors_irradiance_to_clear_sky_model(
        df=df,
        clearsky_df=clearsky_df,
        sensor_names=model_parameters.sensor_names,
        poa_global_name=poa_global_name
    )

    result_df = remove_negative_measurements(df=limit_df)

    save_dataframe_to_csv(
        df=result_df,
        output_path=model_parameters.data_dir / "filtered" / "calibrated" / model_parameters.filename / f"{model_parameters.args.calibration}.csv",
        index=False,
        index_label=None,
    )

    return result_df
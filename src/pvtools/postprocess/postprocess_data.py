import pandas as pd

from pathlib import Path

from pvtools.solar_domain.measurement_limitations import limit_sensors_irradiance_to_clear_sky_model, remove_negative_measurements

def postprocess_data(
        df: pd.DataFrame,
        clearsky_df: pd.DataFrame,
        sensor_names: list[str] = None,
        poa_global_name: str = 'poa_global',
        save_dir: Path = None,
        filename: str = None
) -> pd.DataFrame:
    limit_df = limit_sensors_irradiance_to_clear_sky_model(
        df=df,
        clearsky_df=clearsky_df,
        sensor_names=sensor_names,
        poa_global_name=poa_global_name,
        save_dir=save_dir,
        filename=filename
    )

    result_df = remove_negative_measurements(
        df=limit_df,
        save_dir=save_dir,
        filename=filename
    )

    return result_df
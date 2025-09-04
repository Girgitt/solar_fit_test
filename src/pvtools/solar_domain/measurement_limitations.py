import pandas as pd

from pathlib import Path

from pvtools.io_file.writer import save_dataframe_to_csv

def limit_measured_irradiance_to_clear_sky_model(
        df: pd.DataFrame,
        clear_sky_df: pd.DataFrame,
        poa_global_column_name: str = 'poa_global',
        save_dir: Path = None
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or not isinstance(clear_sky_df, pd.DataFrame):
        raise TypeError("Expected 'df' and 'clear_sky_df' to be a pandas DataFrame")

    if 'time' not in df.columns or 'time' not in clear_sky_df.columns:
        raise ValueError("'time' column needs to be provided!")

    mismatched_times = set(df['time']) - set(clear_sky_df['time'])
    if mismatched_times:
        raise ValueError("Timestamps are mismatched!")

    merged_df = df.merge(clear_sky_df[['time', poa_global_column_name]], on='time', how='left')

    result = df.copy()

    for col in df.columns:
        if col == 'time':
            continue
        result[col] = merged_df[[col, poa_global_column_name]].min(axis=1)

    #if save_dir is not None:


    return result

def remove_negative_measurements(
        df: pd.DataFrame,
        save_dir: Path = None
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected 'df' to be a pandas DataFrame")

    original_df = df.copy()
    df = df.copy()

    for col in df.columns:
        if col == 'time':
            continue
        df[col] = df[col].clip(lower=0)

    if_changed = df.equals(original_df)

    if save_dir is not None and not if_changed:
        save_dir = Path(save_dir)
        output_path = save_dir.parent / "filtered" / (save_dir.stem + save_dir.suffix)
        save_dataframe_to_csv(df, output_path, index=False)

    return df
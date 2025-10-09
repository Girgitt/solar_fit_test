import pandas as pd

from pathlib import Path
from typing import TypeAlias, Literal

from pvtools.io_file.reader import load_dataframe_from_csv
from pvtools.io_file.writer import save_dataframe_to_csv
from pvtools.preprocess.preprocess_data import sanitize_filename

Period_type: TypeAlias = Literal['sunny', 'cloudy']

def apply_mask_for_dataframe(
        data_filename: str,
        sensor_name_ref: str,
        period_type: Period_type,
        save_dir: Path = None,
) -> pd.DataFrame:
    df_data = load_dataframe_from_csv(Path(save_dir / "filtered" / f"{data_filename}.csv"))
    df_mask = load_dataframe_from_csv(Path(save_dir / "calculated_data" / data_filename /
                                           f"{sanitize_filename(sensor_name_ref)}_{period_type}_periods.csv"))

    df_data["time"] = pd.to_datetime(df_data["time"])
    df_mask["time"] = pd.to_datetime(df_mask["time"])
    df_merged = df_data.merge(df_mask, on="time")
    df_result = df_merged.loc[df_merged["if_sunny"], df_data.columns]

    if save_dir is not None:
        save_dir = Path(save_dir)
        output_path = Path(save_dir / "filtered" / f"{period_type}_periods" / (data_filename + ".csv"))
        save_dataframe_to_csv(df_result, output_path, index=False, index_label="time")

    return df_result


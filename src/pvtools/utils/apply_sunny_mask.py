import pandas as pd

from pathlib import Path

from pvtools.io_file.reader import load_dataframe_from_csv
from pvtools.io_file.writer import save_dataframe_to_csv
from pvtools.preprocess.preprocess_data import sanitize_filename

def apply_sunny_mask(
        data_dir: Path,
        data_filename: str,
        sensor_name_ref: str,
        save_dir: Path = None,
) -> pd.DataFrame:
    df_data = load_dataframe_from_csv(Path(data_dir.parent / "filtered" / f"{data_filename.stem}.csv"))
    df_mask = load_dataframe_from_csv(Path(data_dir.parent / "calculated_data" / data_filename.stem / f"{sanitize_filename(sensor_name_ref)}_sunny_periods.csv"))

    df_data["time"] = pd.to_datetime(df_data["time"])
    df_mask["time"] = pd.to_datetime(df_mask["time"])
    df_merged = df_data.merge(df_mask, on="time")
    df_sunny = df_merged.loc[df_merged["if_sunny"], df_data.columns]

    if save_dir is not None:
        save_dir = Path(save_dir)
        output_path = Path(save_dir / "filtered" / "sunny_periods" / data_filename.stem)
        save_dataframe_to_csv(df_sunny, output_path, index=False, index_label="time")

    return df_sunny


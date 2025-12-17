import pandas as pd


def check_if_any_column_is_missing(
        df: pd.DataFrame,
        sensor_name: str,
        time_col: str,
        if_sunny_col: str = None
) -> None:
    """
    Checks if any column is missing from the dataframe.
    """

    if if_sunny_col is not None:
        required_cols = {time_col, if_sunny_col, sensor_name}
    else:
        required_cols = {time_col, sensor_name}

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
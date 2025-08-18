from pathlib import Path
from typing import Dict

import pandas as pd

def load_true_and_predicted_data_for_all_methods(calibration_method_dirs: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    all_data = {}

    for method_dir in calibration_method_dirs.iterdir():
        if method_dir.is_dir():
            method_name = method_dir.name
            method_data = {}
            for csv_file in method_dir.glob("*all_true_vs_pred.csv"):
                sensor_name = csv_file.stem.replace("_all_true_vs_pred", "")
                method_data[sensor_name] = pd.read_csv(csv_file)
            all_data[method_name] = method_data

    return all_data
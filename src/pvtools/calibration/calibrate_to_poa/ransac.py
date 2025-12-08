import pandas as pd
import numpy as np

from sklearn.linear_model import RANSACRegressor, LinearRegression


def ransac_pipeline(
        sensor: pd.Series,
        poa_global: pd.Series,
        clearsky_mask: pd.Series,
        time: pd.Series
) -> pd.DataFrame:

    a, b, sensor_cal = robust_calibration(
        sensor=sensor,
        poa_global=poa_global,
        clearsky_mask=clearsky_mask
    )

    sensor_cal = pd.Series(sensor_cal)

    result = pd.DataFrame({
        "sensor": sensor,
        "poa_global": poa_global,
        "sensor_cal": sensor_cal,
        "mask": clearsky_mask
    })

    return result


def robust_calibration(
        sensor: pd.Series,
        poa_global: pd.Series,
        clearsky_mask: pd.Series
) -> tuple[float, float, np.ndarray]: #np.ndarray -> datatype of model.inlier_mask_

    sensor_clearsky = sensor[clearsky_mask]
    poa_global_clearsky = poa_global[clearsky_mask]

    x = sensor.values.astype(float).reshape(-1, 1)
    y = poa_global.values.astype(float)

    x_clearsky = sensor_clearsky.values.astype(float).reshape(-1, 1)
    y_clearsky = poa_global_clearsky.values.astype(float)

    base_model = LinearRegression(fit_intercept=False)
    model = RANSACRegressor(
        base_model,
        min_samples=0.7,        # require 40% inliers
        residual_threshold=20,  # 20 W/m2 tolerance
        max_trials=200
    )

    model.fit(x_clearsky, y_clearsky)

    #model.estimator_.intercept_ = 0.0

    y_pred = model.predict(x)

    a = model.estimator_.coef_[0]
    b = model.estimator_.intercept_

    return a, b, y_pred # model.inlier_mask_
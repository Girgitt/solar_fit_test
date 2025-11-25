import pandas as pd
import numpy as np

from sklearn.linear_model import RANSACRegressor, LinearRegression

from pvtools.calibration.calibrate_to_poa.clearsky_utils import compute_residual_metrics, clearsky_detection


def ransac_pipeline(
        sensor: pd.Series,
        poa_global: pd.Series,
        time: pd.Series,
) -> pd.DataFrame:

    a, b, poa_pred, inliers = robust_calibration(sensor, poa_global)

    resid, resid_slope, resid_smooth = compute_residual_metrics(poa_global, poa_pred)
    clear = clearsky_detection(
        resid=resid,
        resid_slope=resid_slope,
        resid_smooth=resid_smooth,
        resid_thr=40,
        slope_thr=8,
        smooth_thr=30
    )

    index = time

    sensor.index = index
    poa_pred = pd.Series(poa_pred, index=index)
    resid_slope = pd.Series(resid_slope, index=index)
    resid_smooth = pd.Series(resid_smooth, index=index)
    inliers = pd.Series(inliers, index=index)

    result = pd.DataFrame({
        "sensor": sensor,
        "poa": poa_global,
        "poa_pred": poa_pred,
        "residual": resid,
        "residual_slope": resid_slope,
        "residual_smooth": resid_smooth,
        "ransac_inlier": inliers,
        "clear_sky": clear
    })

    return result.set_index(time)


def robust_calibration(
        sensor: pd.Series,
        poa: pd.Series
) -> tuple[float, float, np.ndarray, np.ndarray]:

    x = sensor.values.reshape(-1, 1).astype(float)
    y = poa.values.astype(float)

    base_model = LinearRegression()
    model = RANSACRegressor(
        base_model,
        min_samples=0.4,        # require 40% inliers
        residual_threshold=20,  # 20 W/m2 tolerance
        max_trials=200
    )

    model.fit(x, y)
    y_pred = model.predict(x)

    a = model.estimator_.coef_[0]
    b = model.estimator_.intercept_

    return a, b, y_pred, model.inlier_mask_
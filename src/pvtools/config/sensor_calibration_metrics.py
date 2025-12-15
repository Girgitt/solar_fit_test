import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SensorCalibrationMetrics:
    """
    The class stores following metrics to evaluate model quality:

    * ``MAE``: Mean Absolute Error - does not indicate whether the model overestimates or underestimates values
    * ``MSE``: Mean Squeared Error - is particularly sensitive to large errors
    * ``RMSE``: Root Mean Squared Erorr - same units as the predicted values
    * ``R2``: correlation between two datasets
    * ``MAPE``: Mean Absolute Percentage Error - especially useful when you want to assess the accuracy of
      predictions in percentage
    * ``Bias``: shows whether the model regularly over - or under-predicts
    """

    def __init__(self, y_true: np.ndarray, y_pred:np.ndarray):
        self.y_true: np.ndarray = np.array(y_true)
        self.y_pred: np.ndarray = np.array(y_pred)

        self.mae: float = mean_absolute_error(self.y_true, self.y_pred)
        self.mse: float = mean_squared_error(self.y_true, self.y_pred)
        self.rmse: float = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        self.r2: float = r2_score(self.y_true, self.y_pred)
        self.mape: float = float(np.mean(np.abs((self.y_true - self.y_pred) / (self.y_true + 1e-8))) * 100)  # +1e-8 for safety
        self.max_error: float = float(np.max(np.abs(self.y_true - self.y_pred)))
        self.bias: float = float(np.mean(self.y_pred - self.y_true))
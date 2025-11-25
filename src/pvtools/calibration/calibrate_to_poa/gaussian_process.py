import torch
import gpytorch
import pandas as pd

from pvtools.calibration.calibrate_to_poa.clearsky_utils import compute_residual_metrics, clearsky_detection


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        # RBF kernel similar to your sklearn setup
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def gaussian_process_pipeline(
        sensor: pd.Series,
        poa_global: pd.Series,
        time: pd.Series,
):

    x, y = prepare_inputs(sensor, poa_global)
    gp, y_pred, sigma = fit_gp_model(x, y)

    resid, resid_slope, resid_smooth = compute_residual_metrics(
        poa_global.values,
        y_pred
    )

    clear = clearsky_detection(
        resid=resid,
        resid_slope=resid_slope,
        resid_smooth=resid_smooth,
        resid_thr=40,
        slope_thr=6,
        smooth_thr=60
    )

    result = pd.DataFrame({
        "time": time,
        "sensor": sensor.values,
        "poa": poa_global.values,
        "poa_pred": y_pred,
        "gp_sigma": sigma,
        "residual": resid,
        "residual_slope": resid_slope,
        "clear_sky": clear
    })

    return result.set_index("time")


def fit_gp_model(
        x: torch.Tensor,
        y: torch.Tensor,
        training_iter: int = 200
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = x.to(device)
    y = y.to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(x, y, likelihood).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

    # -------------------------
    # Switch to evaluation mode
    # -------------------------
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))
        y_pred = pred.mean.cpu().numpy()
        sigma = pred.stddev.cpu().numpy()

    return model, y_pred, sigma


def prepare_inputs(sensor: pd.Series, poa: pd.Series):

    x = torch.tensor(sensor.values.astype(float)).float().reshape(-1, 1)
    y = torch.tensor(poa.values.astype(float)).float()

    return x, y


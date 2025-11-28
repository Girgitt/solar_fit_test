import torch
import gpytorch
import pandas as pd
import numpy as np


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        # RBF kernel similar to your sklearn setup
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def gaussian_process_pipeline(
        sensor: pd.Series,
        poa_global: pd.Series,
        clearsky_mask: pd.Series,
        time: pd.Series
) -> pd.DataFrame:

    sensor_clearsky = sensor[clearsky_mask]
    poa_global_clearsky = poa_global[clearsky_mask]

    x, y = prepare_inputs(
        sensor=sensor,
        poa=poa_global,
    )

    x_cs, y_cs = prepare_inputs(
        sensor=sensor_clearsky,
        poa=poa_global_clearsky,
    )

    gp, y_pred, sigma = fit_gp_model(
        x=x,
        y=y,
        x_cs=x_cs,
        y_cs=y_cs,
        training_iter=10
    )

    y_pred = pd.Series(y_pred, index=sensor.index)

    result = pd.DataFrame({
        "sensor": sensor,
        "poa_global": poa_global,
        "sensor_cal": y_pred,
        "clearsky_mask": clearsky_mask,
        "gp_sigma": sigma
    })

    return result.set_index(time)


def fit_gp_model(
        x: torch.Tensor,
        y: torch.Tensor,
        x_cs: torch.Tensor,
        y_cs: torch.Tensor,
        training_iter: int = 20
) -> tuple[gpytorch.models.ExactGP, np.ndarray, np.ndarray]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = x.to(device)
    y = y.to(device)

    x_cs = x_cs.to(device)
    y_cs = y_cs.to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(x_cs, y_cs, likelihood).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(x_cs)
        loss = -mll(output, y_cs)
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


def prepare_inputs(
        sensor: pd.Series,
        poa: pd.Series
) -> tuple[torch.Tensor, torch.Tensor]:

    x = torch.tensor(sensor.values.astype(float)).float().reshape(-1, 1)
    y = torch.tensor(poa.values.astype(float)).float()

    return x, y


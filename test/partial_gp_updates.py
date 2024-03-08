import marimo

__generated_with = "0.3.1"
app = marimo.App()


@app.cell
def __():
    import torch
    import gpytorch
    import math
    from matplotlib import cm
    from matplotlib import pyplot as plt
    import numpy as np
    import copy
    return cm, copy, gpytorch, math, np, plt, torch


@app.cell
def __(torch):
    def franke(X, Y):
        term1 = .75*torch.exp(-((9*X - 2).pow(2) + (9*Y - 2).pow(2))/4)
        term2 = .75*torch.exp(-((9*X + 1).pow(2))/49 - (9*Y + 1)/10)
        term3 = .5*torch.exp(-((9*X - 7).pow(2) + (9*Y - 3).pow(2))/4)
        term4 = .2*torch.exp(-(9*X - 4).pow(2) - (9*Y - 7).pow(2))

        f = term1 + term2 + term3 - term4
        dfx = -2*(9*X - 2)*9/4 * term1 - 2*(9*X + 1)*9/49 * term2 + \
              -2*(9*X - 7)*9/4 * term3 + 2*(9*X - 4)*9 * term4
        dfy = -2*(9*Y - 2)*9/4 * term1 - 9/10 * term2 + \
              -2*(9*Y - 3)*9/4 * term3 + 2*(9*Y - 7)*9 * term4

        return f, dfx, dfy
    return franke,


@app.cell
def __(franke, torch):
    from torch import Tensor

    def make_dataset(train_num: int, train_missing: float) -> tuple[Tensor, Tensor]:
        _xv, _yv = torch.meshgrid(torch.linspace(0, 1, train_num), torch.linspace(0, 1, train_num), indexing="ij")
        train_x = torch.cat((
            _xv.contiguous().view(_xv.numel(), 1),
            _yv.contiguous().view(_yv.numel(), 1)),
            dim=1
        )

        # val_x = torch.linspace(0, 1, val_num, device=device)
        _f, _dfx, _dfy = franke(train_x[:, 0], train_x[:, 1])
        train_y = torch.stack([_f, _dfx, _dfy], -1).squeeze(1)

        # Randomly mask out some data
        if train_missing > 0:
            train_mask = torch.bernoulli(torch.full_like(train_y, train_missing)).to(torch.bool)
            train_y[train_mask] = torch.nan

        return train_x, train_y
    return Tensor, make_dataset


@app.cell
def __(make_dataset):
    train_num = 10
    train_missing = 0.1

    train_x_pdbg, train_y_pdbg = make_dataset(train_num, train_missing)

    train_x_pdbg, train_y_pdbg
    return train_missing, train_num, train_x_pdbg, train_y_pdbg


@app.cell
def __(franke, torch):
    _xv, _yv = torch.meshgrid(torch.linspace(0, 1, 10), torch.linspace(0, 1, 10), indexing="ij")
    train_x = torch.cat((
        _xv.contiguous().view(_xv.numel(), 1),
        _yv.contiguous().view(_yv.numel(), 1)),
        dim=1
    )

    _f, _dfx, _dfy = franke(train_x[:, 0], train_x[:, 1])
    train_y = torch.stack([_f, _dfx, _dfy], -1).squeeze(1)

    train_y += 0.05 * torch.randn(train_y.size()) # Add noise to both values and gradients
    return train_x, train_y


@app.cell
def __(copy, torch, train_y):
    train_y_partial = copy.deepcopy(train_y)
    # mask out x derivatives
    train_y_partial[:,2] = torch.nan
    train_y_partial
    return train_y_partial,


@app.cell
def __(gpytorch, train_x, train_y):
    class GPModelWithDerivatives(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMeanGrad()
            self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
            self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)  # Value + x-derivative + y-derivative
    model = GPModelWithDerivatives(train_x, train_y, likelihood)
    return GPModelWithDerivatives, likelihood, model


@app.cell
def __(gpytorch, likelihood, model, torch, train_x, train_y):
    # this is for running the notebook in our testing framework
    import os
    smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else 50


    def train_gp_model(model,likelihood,train_x,train_y):
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print("Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f" % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.squeeze()[0],
                model.covar_module.base_kernel.lengthscale.squeeze()[1],
                model.likelihood.noise.item()
            ))
            optimizer.step()

    train_gp_model(model,likelihood,train_x,train_y)
    return os, smoke_test, train_gp_model, training_iter


@app.cell
def __(
    GPModelWithDerivatives,
    gpytorch,
    likelihood,
    train_gp_model,
    train_x,
    train_y_partial,
):
    model_partial = GPModelWithDerivatives(train_x, train_y_partial, likelihood)
    likelihood_partial = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3) 

    # Make predictions
    with gpytorch.settings.observation_nan_policy('mask'):
        train_gp_model(model_partial,likelihood_partial,train_x,train_y_partial)
    return likelihood_partial, model_partial


@app.cell
def __(train_x, train_x_pdbg, train_y, train_y_pdbg):
    train_x_pdbg.shape, train_y_pdbg.shape, train_x.shape, train_y.shape
    return


@app.cell
def __(
    GPModelWithDerivatives,
    gpytorch,
    train_gp_model,
    train_x_pdbg,
    train_y_pdbg,
):
    likelihood_pdbg = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3) 
    model_pdbg = GPModelWithDerivatives(train_x_pdbg, train_y_pdbg, likelihood_pdbg)

    # Make predictions
    with gpytorch.settings.observation_nan_policy('mask'):
        train_gp_model(model_pdbg,likelihood_pdbg,train_x_pdbg,train_y_pdbg)
    return likelihood_pdbg, model_pdbg


@app.cell
def __(
    cm,
    franke,
    gpytorch,
    likelihood,
    likelihood_partial,
    likelihood_pdbg,
    model,
    model_partial,
    model_pdbg,
    plt,
    torch,
):
    # Set into eval mode
    model.eval()
    model_partial.eval()
    model_pdbg.eval()
    likelihood.eval()

    # Initialize plots
    fig, ax = plt.subplots(4, 3, figsize=(14, 10))

    # Test points
    n1, n2 = 50, 50
    xv, yv = torch.meshgrid(torch.linspace(0, 1, n1), torch.linspace(0, 1, n2), indexing="ij")
    f, dfx, dfy = franke(xv, yv)

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
        test_x = torch.stack([xv.reshape(n1*n2, 1), yv.reshape(n1*n2, 1)], -1).squeeze(1)
        predictions = likelihood(model(test_x))
        mean = predictions.mean

        with gpytorch.settings.observation_nan_policy('mask'):
            predictions_partial = likelihood_partial(model_partial(test_x))
            predictions_pdbg = likelihood_pdbg(model_pdbg(test_x))
            mean_partial = predictions_partial.mean
            mean_pdbg = predictions_pdbg.mean

    extent = (xv.min(), xv.max(), yv.max(), yv.min())
    ax[0, 0].imshow(f, extent=extent, cmap=cm.jet)
    ax[0, 0].set_title('True values')
    ax[0, 1].imshow(dfx, extent=extent, cmap=cm.jet)
    ax[0, 1].set_title('True x-derivatives')
    ax[0, 2].imshow(dfy, extent=extent, cmap=cm.jet)
    ax[0, 2].set_title('True y-derivatives')

    ax[1, 0].imshow(mean[:, 0].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    ax[1, 0].set_title('Predicted values')
    ax[1, 1].imshow(mean[:, 1].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    ax[1, 1].set_title('Predicted x-derivatives')
    ax[1, 2].imshow(mean[:, 2].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    ax[1, 2].set_title('Predicted y-derivatives')

    ax[2, 0].imshow(mean_partial[:, 0].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    ax[2, 0].set_title('Predicted values (partial obs)')
    ax[2, 1].imshow(mean_partial[:, 1].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    ax[2, 1].set_title('Predicted x-derivatives (partial obs)')
    ax[2, 2].imshow(mean_partial[:, 2].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    ax[2, 2].set_title('Predicted y-derivatives (partial obs)')

    ax[3, 0].imshow(mean_pdbg[:, 0].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    ax[3, 0].set_title('Predicted values (random masked)')
    ax[3, 1].imshow(mean_pdbg[:, 1].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    ax[3, 1].set_title('Predicted x-derivatives (random masked)')
    ax[3, 2].imshow(mean_pdbg[:, 2].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    ax[3, 2].set_title('Predicted y-derivatives (random masked)')
    return (
        ax,
        dfx,
        dfy,
        extent,
        f,
        fig,
        mean,
        mean_partial,
        mean_pdbg,
        n1,
        n2,
        predictions,
        predictions_partial,
        predictions_pdbg,
        test_x,
        xv,
        yv,
    )


@app.cell
def __(mean_partial):
    mean_partial
    return


@app.cell
def __(mean_pdbg):
    mean_pdbg
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

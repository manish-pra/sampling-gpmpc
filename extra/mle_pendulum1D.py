import torch
import gpytorch
import math
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import sys

# sample A and B matrices
workspace = "sampling-gpmpc"
sys.path.append(workspace)
import yaml
from src.environments.pendulum import Pendulum
from src.environments.pendulum1D import Pendulum as Pendulum1D


# 1) Load the config file
with open(workspace + "/params/" + "params_pendulum1D_samples" + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = 21
params["env"]["name"] = 0
print(params)

env_model = globals()[params["env"]["dynamics"]](params)


Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()
# x1 = torch.linspace(-3.14, 3.14, 7)
# u = torch.linspace(-8, 8, 11)
# X1, U = torch.meshgrid(x1, u)
# Dyn_gp_X_train = torch.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)])
# Dyn_gp_Y_train = {}
# y1, y2 = get_prior_data(Dyn_gp_X_train)
# Dyn_gp_Y_train["y1"] = y1
# Dyn_gp_Y_train["y2"] = y2


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(
            ard_num_dims=params["agent"]["g_dim"]["nx"] + params["agent"]["g_dim"]["nu"]
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks=3, noise_constraint=gpytorch.constraints.GreaterThan(0.0)
)
# model = ExactGPModel(train_x, train_y, likelihood)
Dyn_gp_model = GPModelWithDerivatives(
    Dyn_gp_X_train, Dyn_gp_Y_train[0, :, :], likelihood
)
Dyn_gp_model.covar_module.base_kernel.lengthscale = torch.tensor(
    params["agent"]["Dyn_gp_lengthscale"]["both"]
)

Dyn_gp_model.likelihood.noise = torch.tensor(params["agent"]["Dyn_gp_noise"])
Dyn_gp_model.covar_module.outputscale = torch.tensor(
    params["agent"]["Dyn_gp_outputscale"]["both"]
)
# model_1.covar_module.outputscale = torch.tensor(
#     params["agent"]["Dyn_gp_outputscale"]["both"]
# )
# likelihood = {}
# Dyn_gp_model = {}
# likelihood["y1"] = gpytorch.likelihoods.MultitaskGaussianLikelihood(
#     num_tasks=4, noise_constraint=gpytorch.constraints.GreaterThan(0.0)
# )  # Value + Derivative
# Dyn_gp_model["y1"] = GPModelWithDerivatives(
#     Dyn_gp_X_train, Dyn_gp_Y_train["y1"], likelihood["y1"]
# )
# Dyn_gp_model["y1"].likelihood.noise = torch.ones(1) * 0.00001
# Dyn_gp_model["y1"].likelihood.task_noises = (
#     torch.Tensor([1.28, 3.8, 3.8, 3.8]) * 0.00001
# )

# likelihood["y2"] = gpytorch.likelihoods.MultitaskGaussianLikelihood(
#     num_tasks=4, noise_constraint=gpytorch.constraints.GreaterThan(0.0)
# )  # Value + Derivative
# Dyn_gp_model["y2"] = GPModelWithDerivatives(
#     Dyn_gp_X_train, Dyn_gp_Y_train["y2"], likelihood["y2"]
# )
# Dyn_gp_model["y2"].likelihood.noise = torch.ones(1) * 0.00001
# Dyn_gp_model["y2"].likelihood.task_noises = (
#     torch.Tensor([1.28, 3.8, 3.8, 3.8]) * 0.00001
# )

# for params in model.parameters():
#     print(params)
# print("params:", model.covar_module.base_kernel.lengthscale, model.covar_module.outputscale, model.likelihood.noise, model.likelihood.task_noises)

# model.likelihood.task_noises=torch.Tensor([3.8, 1.27, 3.8])*0.00001
# model.likelihood.noise = 0.00001

# # model.covar_module.base_kernel.lengthscale = torch.Tensor([[0.252, 0.252]])
# # model.covar_module.outputscale = 1
# for params in model.parameters():
#     print(params)

# print("params:", model.covar_module.base_kernel.lengthscale, model.covar_module.outputscale, model.likelihood.noise, model.likelihood.task_noises)
# # this is for running the notebook in our testing framework
import os

smoke_test = "CI" in os.environ
training_iter = 2 if smoke_test else 50


for out in ["y1"]:
    # Find optimal model hyperparameters
    Dyn_gp_model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        Dyn_gp_model.parameters(), lr=0.05
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, Dyn_gp_model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = Dyn_gp_model(Dyn_gp_X_train)
        loss = -mll(output, Dyn_gp_Y_train[0, :, :])
        loss.backward()
        print(
            "Iter",
            i + 1,
            training_iter,
            "Loss:",
            loss.item(),
            "lengthscales:",
            Dyn_gp_model.covar_module.base_kernel.lengthscale,
            "noise:",
            Dyn_gp_model.likelihood.noise.item(),
            "outputscale:",
            Dyn_gp_model.covar_module.outputscale,
        )
        optimizer.step()


# # Set into eval mode
# Dyn_gp_model['y1'].eval()
# likelihood['y1'].eval()

# # Initialize plots
# fig, ax = plt.subplots(2, 3, figsize=(14, 10))

# # Test points
# n1, n2 = 20, 20
# xv, yv = torch.meshgrid(torch.linspace(0, 1, n1), torch.linspace(0, 1, n2), indexing="ij")
# f, dfx, dfy = franke(xv, yv)

# # Make predictions
# with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
#     test_x = torch.stack([xv.reshape(n1*n2, 1), yv.reshape(n1*n2, 1)], -1).squeeze(1)
#     predictions = likelihood(model(test_x))
#     mean = predictions.mean

# extent = (xv.min(), xv.max(), yv.max(), yv.min())
# ax[0, 0].imshow(f, extent=extent, cmap=cm.jet)
# ax[0, 0].set_title('True values')
# ax[0, 1].imshow(dfx, extent=extent, cmap=cm.jet)
# ax[0, 1].set_title('True x-derivatives')
# ax[0, 2].imshow(dfy, extent=extent, cmap=cm.jet)
# ax[0, 2].set_title('True y-derivatives')

# ax[1, 0].imshow(mean[:, 0].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
# ax[1, 0].set_title('Predicted values')
# ax[1, 1].imshow(mean[:, 1].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
# ax[1, 1].set_title('Predicted x-derivatives')
# ax[1, 2].imshow(mean[:, 2].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
# ax[1, 2].set_title('Predicted y-derivatives')

# plt.show()

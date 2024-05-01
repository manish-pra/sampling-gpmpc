import torch
import gpytorch
import math
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import sys

sys.path.insert(0, "/home/manish/work/MPC_Dyn/safe_gpmpc")

from src.environments.car_model import CarKinematicsModel
import argparse
import yaml
from src.GP_model import (
    BatchMultitaskGPModelWithDerivatives_fromParams,
    GPModelWithDerivatives,
    BatchMultitaskGPModelWithDerivatives,
)

workspace = "safe_gpmpc"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_car")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=40)  # initialized at origin
args = parser.parse_args()

# 1) Load the config file
with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

params["env"]["n_data_x"] = 11
params["env"]["n_data_u"] = 11
idx = 0
env_model = CarKinematicsModel(params)

data_X, data_Y = env_model.initial_training_data()
data_Y = data_Y + torch.randn_like(data_Y) * 0.0001
data_X = data_X.cuda()
data_Y = data_Y.cuda()
g_nx = params["agent"]["g_dim"]["nx"]
g_nu = params["agent"]["g_dim"]["nu"]
g_ny = params["agent"]["g_dim"]["ny"]
in_dim = g_nx + g_nu
batch_shape = torch.Size([1, 1])

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks=in_dim + 1, noise_constraint=gpytorch.constraints.GreaterThan(0.0)
)  # Value + Derivative
likelihood = likelihood.cuda()
model_i = GPModelWithDerivatives(data_X, data_Y[idx], likelihood)

# model_i = BatchMultitaskGPModelWithDerivatives(
#     data_X, data_Y[0], likelihood, batch_shape=batch_shape
# )

# model_i = BatchMultitaskGPModelWithDerivatives_fromParams(
#     data_X, data_Y[0], likelihood, params, batch_shape=batch_shape
# )
model_i.likelihood.noise = torch.Tensor([params["agent"]["Dyn_gp_noise"]]).cuda()
model_i.likelihood.task_noises = torch.Tensor(
    params["agent"]["Dyn_gp_task_noises"]["val"]
).cuda()
model_i.covar_module.base_kernel.lengthscale = torch.Tensor(
    params["agent"]["Dyn_gp_lengthscale"]["both"][idx]
).cuda()

# model_i.covar_module.outputscale = torch.Tensor(
#     params["agent"]["Dyn_gp_outputscale"]["y1"]
# ).cuda()

model_i = model_i.cuda()
import os

smoke_test = "CI" in os.environ
training_iter = 2 if smoke_test else 200


for out in ["y1"]:
    # Find optimal model hyperparameters
    model_i.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model_i.parameters(), lr=0.05
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_i).cuda()

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model_i(data_X)
        loss = -mll(output, data_Y[idx])
        loss.backward()
        print(
            "Iter",
            i + 1,
            training_iter,
            "Loss:",
            loss.item(),
            "lengthscales:",
            model_i.covar_module.base_kernel.lengthscale,
            "noise:",
            model_i.likelihood.noise.item(),
            "outputscale:",
            model_i.covar_module.outputscale,
            "task_noise:",
            model_i.likelihood.task_noises,
        )
        optimizer.step()


def pendulum_discrete_dyn(X1_k, X2_k, U_k):
    """_summary_

    Args:
        x (_type_): _description_
        u (_type_): _description_
    """
    m = 1
    l = 1
    g = 10
    dt = 0.01
    X1_kp1 = X1_k + X2_k * dt
    X2_kp1 = X2_k - g * np.sin(X1_k) * dt / l + U_k * dt / (l * l)
    return X1_kp1, X2_kp1


def get_prior_data(x_hat):
    l = 1
    g = 10
    dt = 0.01
    y1_fx, y2_fx = pendulum_discrete_dyn(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2])
    y1_ret = torch.zeros((x_hat.shape[0], 4))
    y2_ret = torch.zeros((x_hat.shape[0], 4))
    y1_ret[:, 0] = y1_fx
    y1_ret[:, 1] = torch.ones(x_hat.shape[0])
    y1_ret[:, 2] = torch.ones(x_hat.shape[0]) * dt

    y2_ret[:, 0] = y2_fx
    y2_ret[:, 1] = (-g * torch.cos(x_hat[:, 0]) / l) * dt
    y2_ret[:, 2] = torch.ones(x_hat.shape[0])
    y2_ret[:, 3] = torch.ones(x_hat.shape[0]) * dt / (l * l)
    # A = np.array([[0.0, 1.0],
    #               [g*np.cos(x_hat[0])/l,0.0]])
    # B = np.array([[0.0],
    #               [1/l]])
    return y1_ret, y2_ret


x1 = torch.linspace(-3.14, 3.14, 5)
x2 = torch.linspace(-10, 10, 5)
u = torch.linspace(-3, 3, 5)
X1, X2, U = torch.meshgrid(x1, x2, u)
Dyn_gp_X_train = torch.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)])
Dyn_gp_Y_train = {}
y1, y2 = get_prior_data(Dyn_gp_X_train)
Dyn_gp_Y_train["y1"] = y1
Dyn_gp_Y_train["y2"] = y2


# class GPModelWithDerivatives(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMeanGrad()
#         self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=3)
#         self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = {}
Dyn_gp_model = {}
likelihood["y1"] = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks=4, noise_constraint=gpytorch.constraints.GreaterThan(0.0)
)  # Value + Derivative
Dyn_gp_model["y1"] = GPModelWithDerivatives(
    Dyn_gp_X_train, Dyn_gp_Y_train["y1"], likelihood["y1"]
)
Dyn_gp_model["y1"].likelihood.noise = torch.ones(1) * 0.00001
Dyn_gp_model["y1"].likelihood.task_noises = (
    torch.Tensor([1.28, 3.8, 3.8, 3.8]) * 0.00001
)

likelihood["y2"] = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks=4, noise_constraint=gpytorch.constraints.GreaterThan(0.0)
)  # Value + Derivative
Dyn_gp_model["y2"] = GPModelWithDerivatives(
    Dyn_gp_X_train, Dyn_gp_Y_train["y2"], likelihood["y2"]
)
Dyn_gp_model["y2"].likelihood.noise = torch.ones(1) * 0.00001
Dyn_gp_model["y2"].likelihood.task_noises = (
    torch.Tensor([1.28, 3.8, 3.8, 3.8]) * 0.00001
)

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

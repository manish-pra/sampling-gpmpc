import gpytorch
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import yaml


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=params["agent"]["g_dim"]["nx"]
                + params["agent"]["g_dim"]["nu"]
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Get training data and update GP with the


# sample A and B matrices
workspace = "sampling-gpmpc"

sys.path.append(workspace)
from src.agent import Agent
from src.environments.pendulum import Pendulum as pendulum
from src.environments.pendulum1D import Pendulum as Pendulum1D

# 1) Load the config file
with open(workspace + "/params/" + "params_pendulum1D_samples" + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = 21
params["env"]["name"] = 0
print(params)

env_model = globals()[params["env"]["dynamics"]](params)

agent = Agent(params, env_model)


Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()
# Computation of C_D
# 1) Compute RKHS norm of the mean function


Dyn_gp_noise = 0.0
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(Dyn_gp_noise),
    # batch_shape=torch.Size([3, 1]),
)
gp_idx = 0
model_1 = ExactGPModel(Dyn_gp_X_train, Dyn_gp_Y_train[gp_idx, :, 0], likelihood)
model_1.covar_module.base_kernel.lengthscale = torch.tensor(
    params["agent"]["Dyn_gp_lengthscale"]["both"]
)
# model_1.covar_module.base_kernel.lengthscale = 5.2649
model_1.likelihood.noise = torch.tensor(params["agent"]["Dyn_gp_noise"])
model_1.covar_module.outputscale = torch.tensor(
    params["agent"]["Dyn_gp_outputscale"]["both"]
)

eval_covar_module = model_1.covar_module(Dyn_gp_X_train)
K_DD = eval_covar_module.to_dense()
# kernel = covar_module = gpytorch.kernels.ScaleKernel(
#     gpytorch.kernels.RBFKernel(ard_num_dims=3)
# )
y = Dyn_gp_Y_train[gp_idx, :, 0].reshape(-1, 1)
# norm = torch.matmul(y.t(), eval_covar_module.inv_matmul(y))
norm = torch.matmul(
    y.t(), torch.matmul(torch.inverse(K_DD + 1e-6 * torch.eye(K_DD.shape[0])), y)
)
print("RKHS norm of the mean function", norm)

print("lengthscale", model_1.covar_module.base_kernel.lengthscale)
print("noise", model_1.likelihood.noise)
print("outputscale", model_1.covar_module.outputscale)

# def compute_rkhs_norm(K_DD, Dyn_gp_Y_train):
#     """
#     Computes the RKHS norm of the GP's mean function.

#     Parameters:
#     K_DD (numpy.ndarray): Covariance matrix of shape (N, N)
#     Dyn_gp_Y_train (numpy.ndarray): Output data of shape (N, 1)

#     Returns:
#     float: RKHS norm of the mean function
#     """
#     # Ensure correct shapes
#     N = Dyn_gp_Y_train.shape[0]
#     assert K_DD.shape == (N, N), "K_DD must have shape (N, N)"

#     # Compute the RKHS norm ||m||_H = sqrt(y^T K^(-1) y)
#     K_inv = np.linalg.inv(K_DD)  # Invert the covariance matrix
#     norm_squared = Dyn_gp_Y_train.T @ K_inv @ Dyn_gp_Y_train
#     rkhs_norm = np.sqrt(norm_squared).item()

#     return rkhs_norm


# def rbf_kernel(x1, x2, length_scale):
#     """
#     Radial Basis Function (RBF) kernel with a vector length scale.
#     """
#     return np.exp(-np.sum(((x1 - x2) ** 2) / (2 * length_scale**2))) * 0.65


# def compute_rkhs_norm(K_DD, Dyn_gp_X_train, Dyn_gp_Y_train, length_scale):
#     """
#     Computes the RKHS norm of the GP's mean function using the RBF kernel.

#     Parameters:
#     K_DD (numpy.ndarray): Covariance matrix of shape (N, N)
#     Dyn_gp_X_train (numpy.ndarray): Input data of shape (N, 3)
#     Dyn_gp_Y_train (numpy.ndarray): Output data of shape (N, 1)
#     length_scale (numpy.ndarray): Length scale vector of shape (3,)

#     Returns:
#     float: RKHS norm of the mean function
#     """
#     # Ensure correct shapes
#     N = Dyn_gp_Y_train.shape[0]
#     assert K_DD.shape == (N, N), "K_DD must have shape (N, N)"
#     assert Dyn_gp_X_train.shape[0] == N, "Dyn_gp_X_train must have shape (N, 3)"
#     assert length_scale.shape == (3,), "length_scale must have shape (3,)"

#     # Compute alpha = K_DD^(-1) * y
#     K_inv = np.linalg.inv(K_DD)
#     alpha = K_inv @ Dyn_gp_Y_train

#     # Compute RKHS norm squared: sum_{i, j} alpha_i alpha_j k(x_i, x_j)
#     rkhs_norm_squared = 0
#     for i in range(N):
#         for j in range(N):
#             rkhs_norm_squared += (
#                 alpha[i]
#                 * alpha[j]
#                 * rbf_kernel(Dyn_gp_X_train[i], Dyn_gp_X_train[j], length_scale)
#             )

#     rkhs_norm = np.sqrt(rkhs_norm_squared).item()

#     return rkhs_norm

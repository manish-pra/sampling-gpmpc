import gpytorch
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import yaml
from helper import (
    compute_rkhs_norm,
    compute_small_ball_probability,
    compute_posterior_norm_diff,
)

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


params["env"]["n_data_x"] = 50  # 80
params["env"]["n_data_u"] = 90  # 100

env_model = globals()[params["env"]["dynamics"]](params)
Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()
true_function_norm, _, _ = compute_rkhs_norm(Dyn_gp_X_train, Dyn_gp_Y_train, params)
print(true_function_norm)

params["env"]["n_data_x"] = 5
params["env"]["n_data_u"] = 9

env_model = globals()[params["env"]["dynamics"]](params)
Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()
mean_norm, alpha, y = compute_rkhs_norm(Dyn_gp_X_train, Dyn_gp_Y_train, params)

kernel_norm_diff = compute_posterior_norm_diff(Dyn_gp_X_train, Dyn_gp_Y_train, params)
print(mean_norm, kernel_norm_diff)
y = Dyn_gp_Y_train[0, :, 0].reshape(-1, 1)
Cd = (
    true_function_norm
    + mean_norm
    - 2 * torch.matmul(y.t(), alpha)
    + torch.sum(torch.abs(alpha)) * params["agent"]["tight"]["w_bound"]
    + kernel_norm_diff / 2
)
posterior_norm_diff = compute_posterior_norm_diff(
    Dyn_gp_X_train, Dyn_gp_Y_train, params
)

params["common"]["use_cuda"] = False
env_model = globals()[params["env"]["dynamics"]](params)
Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()

B_phi = compute_small_ball_probability(Dyn_gp_X_train, Dyn_gp_Y_train, params)

B_phi = B_phi.cuda()
delta = torch.tensor([0.01]).cuda()  # safety with 90% probability (1-\delta)
Num_samples = torch.log(delta) / torch.log(1 - torch.exp(-Cd) * B_phi)

print(
    f"Number of dynamics samples for safety with {1-delta.item()} probability are {Num_samples.item()}"
)
# Computation of C_D
# 1) Compute RKHS norm of the mean function


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

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
    compute_multi_dim_small_ball_probability,
    compute_multi_dim_small_ball_probability_fixed_eps
)

# sample A and B matrices
workspace = "sampling-gpmpc"

sys.path.append(workspace)
from src.agent import Agent
from src.environments.pendulum import Pendulum as pendulum
from src.environments.pendulum1D import Pendulum as Pendulum1D
from src.environments.car_model import CarKinematicsModel as bicycle
from src.environments.car_model_residual import CarKinematicsModel as bicycle_Bdx

# 1) Load the config file
with open(workspace + "/params/" + "params_car_residual" + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = 21
params["env"]["name"] = 0
print(params)
# gp_idx, N_grid = 0, 6
# gp_idx, N_grid = 1, 6
# gp_idx, N_grid = 2, 4
gp_idx, N_grid = 0, 6
gp_idx, N_grid = 1, 6
gp_idx, N_grid = 2, 5

Cd_list = []
for gp_idx in range(params["agent"]["g_dim"]["ny"]):
    n_data_x = params["env"]["n_data_x"]
    n_data_u = params["env"]["n_data_u"]

    params["env"]["n_data_x"] *= 10  # 80
    params["env"]["n_data_u"] *= 10  # 100

    env_model = globals()[params["env"]["dynamics"]](params)
    Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()
    true_function_norm, _, _,_ = compute_rkhs_norm(
        Dyn_gp_X_train, Dyn_gp_Y_train, params, gp_idx
    )
    print(true_function_norm)

    params["env"]["n_data_x"] = n_data_x
    params["env"]["n_data_u"] = n_data_u

    env_model = globals()[params["env"]["dynamics"]](params)
    Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()
    mean_norm, alpha, y,_ = compute_rkhs_norm(Dyn_gp_X_train, Dyn_gp_Y_train, params, gp_idx)

    kernel_norm_diff = compute_posterior_norm_diff(
        Dyn_gp_X_train, Dyn_gp_Y_train, params, gp_idx
    )
    print("Mean norm and kernel diff", mean_norm, kernel_norm_diff)
    y = Dyn_gp_Y_train[gp_idx, :, 0].reshape(-1, 1)
    Cd = (
        (true_function_norm
        + mean_norm
        - 2 * torch.matmul(y.t(), alpha)
        + 2*torch.sum(torch.abs(alpha)) * params["agent"]["tight"]["w_bound"])/2
        + kernel_norm_diff / 2
    )
    Cd_list.append(Cd)
# posterior_norm_diff = compute_posterior_norm_diff(
#     Dyn_gp_X_train, Dyn_gp_Y_train, params, gp_idx
# )
print("The C_d constants", Cd_list)
params["common"]["use_cuda"] = False
env_model = globals()[params["env"]["dynamics"]](params)
Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()


N_samples_list = torch.Tensor([200, 2000, 20000, 200000, 2000000, 20000000, 120000000]).cuda()
N_samples_list = N_samples_list/2 
delta = torch.tensor([0.01]).cuda() 
Cd_tot = torch.sum(torch.stack(Cd_list))
prob_eps = (1- torch.exp( torch.log(delta)/N_samples_list))/torch.exp(-Cd_tot.cuda())
print(prob_eps)
epsilons = compute_multi_dim_small_ball_probability_fixed_eps(Dyn_gp_X_train, Dyn_gp_Y_train, params, N_grid=4, prob_eps=prob_eps.cpu())
torch.set_printoptions(precision=16) 
print(epsilons)
quit()
eB_phi_list = []
# for gp_idx, N_grid in zip([0,1,2], [6,6,4]):
#     eB_phi = compute_small_ball_probability(Dyn_gp_X_train, Dyn_gp_Y_train, params, N_grid, gp_idx)
#     eB_phi_list.append(eB_phi)
N_grid=4
eB_phi = compute_multi_dim_small_ball_probability(Dyn_gp_X_train, Dyn_gp_Y_train, params, N_grid, gp_idx)
eB_phi_list.append(eB_phi)
# print(eB_phi)
# B_phi = (
#     0.6
#     * torch.log(torch.tensor([1 / params["agent"]["tight"]["dyn_eps"]]))
#     ** params["agent"]["g_dim"]["nx"]
# )
# eB_phi = torch.exp(-B_phi)
# print(eB_phi)
Cd_tot = torch.sum(torch.stack(Cd_list))
eB_phi = torch.prod(torch.stack(eB_phi_list)).cuda()
delta = torch.tensor([0.01]).cuda()  # safety with 99% probability (1-\delta)
Num_samples = torch.log(delta) / torch.log(1 - torch.exp(-Cd_tot.cuda()) * eB_phi)

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

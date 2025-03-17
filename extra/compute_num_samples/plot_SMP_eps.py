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
    compute_epsilon_fix_small_ball_probability
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

n_data_x = params["env"]["n_data_x"]
n_data_u = params["env"]["n_data_u"]

params["env"]["n_data_x"] *= 10  # 80
params["env"]["n_data_u"] *= 10  # 100

env_model = globals()[params["env"]["dynamics"]](params)
Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()
true_function_norm, _, _, _ = compute_rkhs_norm(Dyn_gp_X_train, Dyn_gp_Y_train, params)
print(true_function_norm)

params["env"]["n_data_x"] = n_data_x
params["env"]["n_data_u"] = n_data_u

env_model = globals()[params["env"]["dynamics"]](params)
Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()
mean_norm, alpha, y, beta_data = compute_rkhs_norm(Dyn_gp_X_train, Dyn_gp_Y_train, params)
print("Beta", torch.sqrt(true_function_norm)+beta_data)

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


N_max = 8 # maximum such that matrix is still psd
repeat_idx = 3
plt.figure()
eps_orig = params["agent"]["tight"]["dyn_eps"]
eps_range = 0.0002
for eps_idx in range(-5, 6):
    B_phi_data = torch.zeros(repeat_idx, N_max)
    params["agent"]["tight"]["dyn_eps"] = eps_orig + eps_range*eps_idx
    for N in range(1, N_max+1):
        for i in range(repeat_idx):
            B_phi = compute_small_ball_probability(Dyn_gp_X_train, Dyn_gp_Y_train, params, N)
            B_phi_data[i, N-1] = B_phi

    # plot the B_phi for different N with repeat_idx as error bars 

    plt.errorbar(np.arange(1, N_max+1), B_phi_data.mean(dim=0).cpu().numpy(), yerr=B_phi_data.std(dim=0).cpu().numpy(), label=f"Epsilon {params['agent']['tight']['dyn_eps']}")
# plt.show()
plt.legend()
plt.title(f"P vs N for different epsilon")
plt.savefig("smb_no_jitter_diff_eps_ls.png")

plt.figure()
for prob in range(1, 10):
    prob = prob / 10
    eps_data = torch.zeros(repeat_idx, N_max)
    for N in range(1, N_max+1):
        for i in range(repeat_idx):
            eps = compute_epsilon_fix_small_ball_probability(Dyn_gp_X_train, Dyn_gp_Y_train, params, N, prob=prob)
            eps_data[i, N-1] = eps

    # plot the B_phi for different N with repeat_idx as error bars 

    plt.errorbar(np.arange(1, N_max+1), eps_data.mean(dim=0).cpu().numpy(), yerr=eps_data.std(dim=0).cpu().numpy(), label=f"Probability {prob}") 
# plt.show()
plt.legend()
plt.title(f"Epsilon vs N for different probability")
plt.savefig("eps_no_jitter_diff_p_ls.png")
quit()
epsilon = compute_epsilon_fix_small_ball_probability(Dyn_gp_X_train, Dyn_gp_Y_train, params, N_max, prob=0.9)
# Pass in different N and get 3 different p from 1 to 8


# Fix a p=0.95 and 3 different epsilon for different N's 


B_phi = B_phi.cuda()
delta = torch.tensor([0.01]).cuda()  # safety with 99% probability (1-\delta)
Num_samples = torch.log(delta) / torch.log(1 - torch.exp(-Cd) * B_phi)

print(
    f"Number of dynamics samples for safety with {1-delta.item()} probability are {Num_samples.item()}"
)



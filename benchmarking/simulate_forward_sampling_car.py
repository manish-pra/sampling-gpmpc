import argparse
import errno
import os, sys
import warnings

import matplotlib.pyplot as plt
import yaml

import dill as pickle
import numpy as np
import torch

import gpytorch
import copy

# NOTE: this file needs to be called from outside the root directory of the project, e.g.:
# python sampling-gpmpc/benchmarking/simulate_forward_sampling_car.py
workspace = "sampling-gpmpc"
sys.path.append(workspace)

from src.agent import Agent
from src.environments.car_model_residual import CarKinematicsModel as bicycle_Bdx

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_car_residual_fs")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=4)  # initialized at origin
parser.add_argument("-epistemic_idx", type=int, default=400)  # initialized at origin
args = parser.parse_args()

# 1) Load the config file
with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = args.i
params["env"]["name"] = args.env
print(params)

# random seed
if params["experiment"]["rnd_seed"]["use"]:
    torch.manual_seed(params["experiment"]["rnd_seed"]["value"])

# 2) Set the path and copy params from file
exp_name = params["experiment"]["name"]
env_load_path = (
    workspace
    + "/experiments/"
    + params["experiment"]["folder"]
    + "/env_"
    + str(args.env)
)

# torch.cuda.memory._record_memory_history(enabled=True)

save_path = env_load_path + "/" + args.param + "/"

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

print(args)
if args.i != -1:
    traj_iter = args.i

if not os.path.exists(save_path + str(traj_iter)):
    os.makedirs(save_path + str(traj_iter))

env_model = globals()[params["env"]["dynamics"]](params)

agent = Agent(params, env_model)

if params["agent"]["load_epistemic_rv"]:
    with open(save_path + str(traj_iter) + f"/data_epistemic_vector_{args.epistemic_idx}.pkl", "rb") as epistemic_rv_file:
        agent.epistimic_random_vector = pickle.load(epistemic_rv_file)

# get saved input trajectory
if params["agent"]["feedback"]["use"]:
    input_data_path = (
        f"{save_path}{str(args.i)}/data_feedback_1e-8.pkl"
    )
else:
    input_data_path = (
        f"{save_path}{str(args.i)}/data.pkl"
    )
with open(input_data_path, "rb") as input_data_file:
    input_gpmpc_data = pickle.load(input_data_file)

# 4) Set the initial state
# agent.update_current_state(np.array(params["env"]["start"]))

input_gpmpc_timestep = -1
input_gpmpc_input_traj = input_gpmpc_data["input_traj"][input_gpmpc_timestep]

# initialize initial condition for each sample
if agent.use_cuda:
    # torch.cuda.device(torch.device("cuda"))
    torch.set_default_device(torch.device("cuda"))
else:
    # torch.cuda.device(torch.device("cpu"))
    torch.set_default_device(torch.device("cpu"))

agent.update_current_state(np.array(params["env"]["start"]))
x_curr = agent.current_state[: agent.nx].reshape(agent.nx)

H = input_gpmpc_input_traj.shape[0]
K = np.array(params["optimizer"]["terminal_tightening"]["K"])
ns = params["agent"]["num_dyn_samples"]
x_h = np.tile(x_curr, (1, ns)) #np.zeros((1, agent.nx * params["agent"]["num_dyn_samples"]))
X_traj = torch.empty((ns, agent.nx, H + 1))
x_equi = np.array(params["env"]["goal_state"])
for H_idx in range(H):
    agent.train_hallucinated_dynGP(1, use_model_without_derivatives=params["env"]["use_model_without_derivatives"])
    agent.mpc_iteration(H_idx) # Hack for the sampling of epistemic uncertainty
    u_h = input_gpmpc_input_traj[H_idx].reshape(1, -1)
    if params["agent"]["feedback"]["use"]:
        batch_x_hat = agent.get_batch_x_hat_u_diff(x_h, -(x_equi-x_h.reshape(1, ns, -1))@K.T + np.tile(u_h[:, None,:], (ns,1)))
        # sample the gradients
        gp_val, y_grad, u_grad = agent.dyn_fg_jacobians(batch_x_hat, 1)
        
        # y_grad = y_grad - u_grad @ K
    else:
        batch_x_hat = agent.get_batch_x_hat(x_h, u_h)
        # sample the gradients
        gp_val, y_grad, u_grad = agent.dyn_fg_jacobians(batch_x_hat, 1)
    
    # fill in X_traj
    X_traj[:, :, H_idx] = batch_x_hat[:,0,0,:agent.nx] #torch.tensor(x_h)
    del batch_x_hat

    x_h = gp_val[:,:,0,0].reshape(1, -1)

X_traj[:, :, H_idx+1] = torch.tensor(gp_val[:,:,0,0])

X_traj = X_traj.detach().cpu().numpy()
if params["visu"]["show"]:
    plt.plot(X_traj[:,0,:].T, X_traj[:,1,:].T, linewidth=0.2)
    a_file = open(save_path + str(traj_iter) + "/data_feedback_1e-8.pkl", "rb")
    data_dict = pickle.load(a_file)
    true_state_traj = data_dict["true_state_traj"]
    state_traj = data_dict["state_traj"]
    a_file.close()
    from src.visu import Visualizer
    visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=agent)
    propagated_state = visu.propagate_true_dynamics(x_curr, input_gpmpc_input_traj)
    plt.plot(propagated_state[:,0], propagated_state[:,1], ls='--',color="black", label="propagated Trajectory", linewidth=0.5)
    plt.plot(state_traj[0][:,0], state_traj[0][:,1], color="black", label="SQP Trajectory", linewidth=0.5)
    plt.plot(true_state_traj[0][:,0], true_state_traj[0][:,1], color="red", label="True Trajectory", linewidth=0.5)
    plt.legend()
    plt.savefig("fs.png") # for debugging

if params["agent"]["save_data"]:
    print(f"Saving data for {args.epistemic_idx}")
    a_file = open(save_path + str(traj_iter) + f"/data_X_traj_{args.epistemic_idx}.pkl", "wb")
    pickle.dump(X_traj, a_file)
    a_file.close()

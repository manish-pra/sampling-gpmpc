import argparse
import errno
import os
import warnings

import matplotlib.pyplot as plt
import yaml

import dill as pickle

from src.DEMPC import DEMPC
from src.visu import Visualizer
from src.agent import Agent
from src.environments.pendulum import Pendulum
from src.environments.car_model import CarKinematicsModel
import numpy as np
import torch

import gpytorch
import copy

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

workspace = "safe_gpmpc"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_car")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=40)  # initialized at origin
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
    + "/"
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

if params["env"]["dynamics"] == "pendulum":
    env_model = Pendulum(params)
elif params["env"]["dynamics"] == "bicycle":
    env_model = CarKinematicsModel(params)
else:
    raise ValueError("Unknown dynamics model")

agent = Agent(params, env_model)

# get saved input trajectory
input_data_path = (
    "/home/amon/Repositories/safe_gpmpc/experiments/pendulum/env_0/params_pendulum/1/data.pkl"
    # "/home/amon/Repositories/safe_gpmpc/experiments/pendulum/env_0/params/_static/reachable_set_input.pkl"
)
with open(input_data_path, "rb") as input_data_file:
    input_gpmpc_data = pickle.load(input_data_file)

# 4) Set the initial state
# agent.update_current_state(np.array(params["env"]["start"]))

input_gpmpc_timestep = 0
input_gpmpc_input_traj = input_gpmpc_data["input_traj"][input_gpmpc_timestep]

# initialize initial condition for each sample
if agent.use_cuda:
    # torch.cuda.device(torch.device("cuda"))
    torch.set_default_device(torch.device("cuda"))
else:
    # torch.cuda.device(torch.device("cpu"))
    torch.set_default_device(torch.device("cpu"))


X_traj = torch.zeros(
    (agent.batch_shape[0], agent.batch_shape[1], 1, agent.nx + agent.nu)
)
X_traj[:, :, :, 0 : agent.nx] = torch.tile(
    torch.tensor(np.array(params["env"]["start"])),
    (agent.batch_shape[0], agent.batch_shape[1], 1, 1),
)
Y_traj = torch.zeros((agent.batch_shape[0], agent.batch_shape[1], 1, agent.nx + 1))

num_repeat = 100
X_traj_list = [
    [copy.deepcopy(X_traj) for i in range(params["optimizer"]["H"] + 1)]
    for i in range(num_repeat)
]
Y_traj_list = [
    [copy.deepcopy(Y_traj) for i in range(params["optimizer"]["H"])]
    for i in range(num_repeat)
]

sqrt_beta = params["agent"]["Dyn_gp_beta"]

for j in range(num_repeat):

    print(
        f"Repeats: {j}/{num_repeat}, Samples: {agent.batch_shape[0]*j}/{num_repeat*agent.batch_shape[0]}"
    )
    agent = Agent(params, env_model)

    # pre-condition with random sampled data points
    X_along_traj = (
        torch.tensor(
            input_gpmpc_data["state_traj"][0][0 : params["optimizer"]["H"], :].reshape(
                params["optimizer"]["H"], -1, 2
            )
        )
        .unsqueeze(2)
        .permute(1, 2, 0, 3)
        .tile(1, agent.batch_shape[1], 1, 1)
    )
    U_along_traj = torch.tensor(input_gpmpc_input_traj)
    U_along_traj_tile = torch.tile(
        U_along_traj, (agent.batch_shape[0], agent.batch_shape[1], 1, 1)
    )
    X_inp_random_list = []
    X_inp_opt = torch.cat((X_along_traj, U_along_traj_tile), dim=-1)
    X_inp_random_list.append(X_inp_opt)
    n_random_conditionings = 10
    for i in range(n_random_conditionings):

        X_inp_random = torch.randn_like(X_inp_opt) * 0.1 + X_inp_opt
        X_inp_random_list.append(X_inp_random)

    for i in range(params["optimizer"]["H"]):
        # train model on data points
        agent.train_hallucinated_dynGP(i)
        agent.model_i.eval()

        # get control input into right shape to stack with Y_perm
        U_single = torch.tensor(input_gpmpc_input_traj[i, :])
        U_tile = torch.tile(
            U_single, (agent.batch_shape[0], agent.batch_shape[1], 1, 1)
        )
        X_traj_list[j][i][:, :, :, agent.nx : agent.nx + agent.nu] = U_tile

        # sample functions from GP
        with torch.no_grad(), gpytorch.settings.observation_nan_policy(
            "mask"
        ), gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ), gpytorch.settings.cholesky_jitter(
            float_value=self.params["agent"]["Dyn_gp_jitter"],
            double_value=self.params["agent"]["Dyn_gp_jitter"],
            half_value=self.params["agent"]["Dyn_gp_jitter"],
        ):
            model_i_call = agent.model_i(X_traj_list[j][i])
            # TODO: truncate
            Y_traj_list[j][i] = model_i_call.sample(
                # base_samples=self.epistimic_random_vector[self.mpc_iter][sqp_iter]
            )
            Y_max = model_i_call.mean + sqrt_beta * torch.sqrt(model_i_call.variance)
            Y_min = model_i_call.mean - sqrt_beta * torch.sqrt(model_i_call.variance)
            Y_traj_list[j][i] = torch.max(Y_traj_list[j][i], Y_min)
            Y_traj_list[j][i] = torch.min(Y_traj_list[j][i], Y_max)

        # condition on sampled values
        agent.update_hallucinated_Dyn_dataset(X_traj_list[j][i], Y_traj_list[j][i])

        # duplicate Y values for batching in X
        Y_stack = torch.stack([Y_traj_list[j][i][:, :, :, 0]] * agent.nx, dim=-1)
        # permute dimensions such that output becomes input for X
        Y_perm = Y_stack.permute(0, 3, 2, 1)

        # stack U with Y_perm to propagate to next state
        X_traj_list[j][i + 1][:, :, :, 0 : agent.nx] = Y_perm
        # X_traj_list[i + 1] = torch.cat((Y_perm, U_tile), dim=-1)

# save trajectories
# flatten list
X_traj_list = [
    torch.cat([X_traj_list[j][i] for j in range(num_repeat)], dim=0)
    for i in range(params["optimizer"]["H"] + 1)
]
# X_flatten = torch.cat(X_traj_list)
Y_traj_list = [item for sublist in Y_traj_list for item in sublist]


with open(save_path + str(traj_iter) + "/X_traj_list.pkl", "wb") as f:
    pickle.dump(X_traj_list, f)

# plot trajectories
x_plot = np.array(
    [
        X_traj_list[i][:, 0, 0, 0 : agent.nx].detach().cpu().numpy()
        for i in range(params["optimizer"]["H"])
    ]
)
plt.plot(x_plot[:, :, 0], x_plot[:, :, 1])

plt.show()
# de_mpc = DEMPC(params, visu, agent)
# de_mpc.dempc_main()
# visu.save_data()
# dict_file = torch.cuda.memory._snapshot()
# pickle.dump(dict_file, open(save_path + str(traj_iter) + "/memory_snapshot_1.pickle", "wb"))
exit()

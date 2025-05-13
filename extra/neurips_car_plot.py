import argparse
import errno
import os
import warnings
import sys, os

import matplotlib.pyplot as plt
import yaml
import dill as pickle
workspace = "sampling-gpmpc"
sys.path.append(workspace)

from src.visu import Visualizer
from src.environments.pendulum import Pendulum as pendulum
from src.environments.car_model_residual import CarKinematicsModel as bicycle_Bdx
from src.environments.car_model import CarKinematicsModel as bicycle
from src.environments.pendulum1D import Pendulum as Pendulum1D
from src.environments.drone import Drone as drone
from src.agent import Agent
import numpy as np
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]


def load_data(file_name):
    a_file = open(save_path + str(traj_iter) + "/"+file_name, "rb")
    data_dict = pickle.load(a_file)
    solver_status = 0
    state_traj = data_dict["state_traj"]
    input_traj = data_dict["input_traj"]
    mean_state_traj = data_dict["mean_state_traj"]
    true_state_traj = data_dict["true_state_traj"]
    physical_state_traj = data_dict["physical_state_traj"]
    solver_cost = data_dict["solver_cost"]
    if "solver_Status" in data_dict:
        solver_status = data_dict["solver_Status"]
    tilde_eps_list, ci_list = None, None
    if "tilde_eps_list" in data_dict:
        tilde_eps_list = data_dict["tilde_eps_list"]
        ci_list = data_dict["ci_list"]
    a_file.close()
    return solver_cost, solver_status, np.vstack(physical_state_traj)[:,:6]

# read data
param_list = ["params_car_racing_sagedynx"]#, "params_car_racing_no_learning"]

save_path = workspace + "/experiments/car/env_0/params_car_racing_opt/"
traj_iter = 1
cost_opt, solver_status_opt, state_traj_opt = load_data("data_opt.pkl")

# 2,
i_max = 1
regret_dict = {}
for env in range(0, 1):
    for idx, param in enumerate(param_list):
        if param not in regret_dict:
            regret_dict[param] = []
        for traj_iter in range(8,8+i_max):
            # 2) Set the path and copy params from file
            exp_folder = "car"
            env_load_path = (
                workspace
                + "/experiments/"
                + exp_folder
                + "/env_"
                + str(env)
                + "/"
            )

            save_path = env_load_path + "/" + param + "/"

            cost_l1, solver_status_l1, state_traj_l1 = load_data("data_lap1.pkl")
            # Compute regret
            close_loop_regret = np.linalg.norm(state_traj_opt - state_traj_l1, axis=1)
            # close_loop_regret = np.array(cost_l1) - np.array(cost_opt)
            if idx!=1:
                cost_l2, solver_status_l2, state_traj_l2 = load_data("data_lap2.pkl")
                close_loop_regret_l2 = np.linalg.norm(state_traj_opt - state_traj_l2, axis=1)
                # close_loop_regret_l2 = np.array(cost_l2) - np.array(cost_opt)
                # merge two regrets
                close_loop_regret = np.concatenate((close_loop_regret, close_loop_regret_l2))

                cost_l3, solver_status_l3, state_traj_l3 = load_data("data_lap3.pkl")
                close_loop_regret_l3 = np.linalg.norm(state_traj_opt - state_traj_l3, axis=1)
                # close_loop_regret_l2 = np.array(cost_l2) - np.array(cost_opt)
                # merge two regrets
                close_loop_regret = np.concatenate((close_loop_regret, close_loop_regret_l3))

            cum_close_loop_regret = np.cumsum(close_loop_regret)/np.arange(1, len(close_loop_regret)+1)
            regret_dict[param].append(cum_close_loop_regret)


# take std dev and plot
for idx, param in enumerate(regret_dict):
    alpha = 0.6
    if idx==2:
        alpha = 0.2
    mean_regret = np.mean(np.vstack(regret_dict[param]), axis=0)
    std_regret = np.std(np.vstack(regret_dict[param]), axis=0)/np.sqrt(len(regret_dict[param]))
    plt.plot(mean_regret, label=param)
    plt.fill_between(
        np.arange(0, len(mean_regret)), mean_regret - std_regret, mean_regret + std_regret, alpha = alpha
    )
plt.legend()
plt.xlabel("Iteration")
plt.yscale("log")
plt.ylabel("CR/T")
plt.ylim(0.04, 5)
# plt.xscale("log")
# plt.ylabel("Cummulative Regret")
# plt.title("Cummulative Regret")
plt.savefig("cr_t_car.png")
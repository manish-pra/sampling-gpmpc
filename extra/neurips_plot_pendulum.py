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

workspace_plotting_utils = "extra"
sys.path.append(os.path.join(os.path.dirname(__file__),workspace_plotting_utils))
print("sys.path", sys.path)
from plotting_tools.plotting_utilities import *
TEXTWIDTH = 16
set_figure_params(serif=True, fontsize=14)
f, ax = plt.subplots(figsize=(cm2inches(12.0), cm2inches(6.0)))



def load_data(file_name):
    a_file = open(save_path + str(traj_iter) + "/"+file_name, "rb")
    data_dict = pickle.load(a_file)
    solver_status = 0
    state_traj = data_dict["state_traj"]
    input_traj = data_dict["input_traj"]
    mean_state_traj = data_dict["mean_state_traj"]
    true_state_traj = data_dict["true_state_traj"]
    physical_state_traj = data_dict["physical_state_traj"]
    solver_cost =0 
    if "solver_cost" in data_dict:
        solver_cost = data_dict["solver_cost"]
    if "solver_Status" in data_dict:
        solver_status = data_dict["solver_Status"]
    tilde_eps_list, ci_list = None, None
    if "tilde_eps_list" in data_dict:
        tilde_eps_list = data_dict["tilde_eps_list"]
        ci_list = data_dict["ci_list"]
    a_file.close()
    return solver_cost, solver_status, np.vstack(physical_state_traj)[:,:2]

# read data
param_list = ["params_pendulum_no_learning","params_pendulum_sagedynx"]#, "params_pendulum_sagedynx1", "params_pendulum_sagedynx2", "params_pendulum_sagedynx3"]
legend_list = ["No learning", "SAGE-DynX"]#, "SAGE-DynX2", "SAGE-DynX1", "SAGE-DynX4"]
color_list = ["tab:blue", "tab:green"]#, "green", "red", "purple"]

save_path = workspace + "/experiments/pendulum/env_0/params_pendulum_opt/"
traj_iter = 41
cost_opt, solver_status_opt, state_traj_opt = load_data("data.pkl")

# 2,
i_max = 11
regret_dict = {}
for env in range(0, 1):
    for idx, param in enumerate(param_list):
        if param not in regret_dict:
            regret_dict[param] = []
        for traj_iter in range(0,i_max):
            # 2) Set the path and copy params from file
            exp_folder = "pendulum"
            env_load_path = (
                workspace
                + "/experiments/"
                + exp_folder
                + "/env_"
                + str(env)
                + "/"
            )

            save_path = env_load_path + "/" + param + "/"

            cost_l1, solver_status_l1, state_traj_l1 = load_data("data.pkl")
            # min_shape = min(state_traj_l1.shape[0], state_traj_opt.shape[0])
            min_shape = 45
            state_traj_l1 = state_traj_l1[:min_shape]

            # Compute regret
            close_loop_regret = np.linalg.norm(state_traj_opt[:min_shape] - state_traj_l1[:min_shape], axis=1)
            # if idx!=0:
            #     cost_l2, solver_status_l2, state_traj_l2 = load_data("data.pkl")
            #     close_loop_regret_l2 = np.linalg.norm(state_traj_opt - state_traj_l2, axis=1)
            #     # merge two regrets
            #     close_loop_regret = np.concatenate((close_loop_regret, close_loop_regret_l2))
            # else:
            #     cost_l2, solver_status_l2, state_traj_l2 = load_data("data_lap1.pkl")
            #     close_loop_regret_l2 = np.linalg.norm(state_traj_opt - state_traj_l2[-1], axis=1)
            #     # merge two regrets
            #     close_loop_regret = np.concatenate((close_loop_regret, close_loop_regret_l2))

            cum_close_loop_regret = np.cumsum(close_loop_regret)/np.arange(1, len(close_loop_regret)+1)
            regret_dict[param].append(cum_close_loop_regret)


# take std dev and plot
for idx, param in enumerate(regret_dict):
    alpha = 0.6
    # if idx==2:
    #     alpha = 0.2
    mean_regret = np.mean(np.vstack(regret_dict[param]), axis=0)
    std_regret = np.std(np.vstack(regret_dict[param]), axis=0)/np.sqrt(len(regret_dict[param]))
    plt.plot(mean_regret, label=legend_list[idx], color=color_list[idx])
    plt.fill_between(
        np.arange(0, len(mean_regret)), mean_regret - std_regret, mean_regret + std_regret, alpha = alpha, color=color_list[idx]
    )


ax = f.axes
adapt_figure_size_from_axes(ax)
plt.ylabel(r"Cumm. Regret / Time", labelpad=-1)
plt.xlabel(r"Time")
plt.tight_layout(pad=1.0)

plt.legend(fontsize='small', labelspacing=0.2, handlelength=1)
plt.yscale("log")

# plt.legend()
# plt.xlabel("Iteration")
# plt.yscale("log")
# plt.ylabel("CR/T")
plt.ylim(0.08, 0.5)
plt.xlim(0, 45)
# plt.xscale("log")
# plt.ylabel("Cummulative Regret")
# plt.title("Cummulative Regret")
# plt.savefig("cr_t_pendulum.png")
plt.savefig("pendulum.pdf", dpi=600,transparent=True,format="pdf", bbox_inches="tight")
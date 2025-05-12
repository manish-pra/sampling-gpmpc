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



parser = argparse.ArgumentParser(description="A foo that bars")
# parser.add_argument("-param", default="params_pendulum1D_samples")  # params
# parser.add_argument("-param", default="params_car_samples")  # params
# parser.add_argument("-param", default="params_car_residual")  # params
# parser.add_argument("-param", default="params_pendulum_exploration")
parser.add_argument("-param", default="params_drone")

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=str, default="43_neurips")  # initialized at origin
args = parser.parse_args()

# 1) Load the config file
with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = args.i
params["env"]["name"] = args.env
print(params)

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


cost_algo, solver_status_algo, state_traj_algo = load_data("data.pkl")
cost_algo1, solver_status_algo1, state_traj_algo1 = load_data("data1.pkl")
cost_opt, solver_status_opt, state_traj_opt = load_data("data_opt.pkl")
cost_no_learning, solver_status_no_learn, state_traj_no_learning = load_data("data_no_learning.pkl")

# Plot closed loop cost
close_loop_regret = np.linalg.norm(state_traj_opt - state_traj_algo, axis=1)
close_loop_regret1 = np.linalg.norm(state_traj_opt - state_traj_algo1, axis=1)
plt.plot(close_loop_regret, label="close_loop_regret")
plt.plot(close_loop_regret1, label="close_loop_regret1")
cum_regret_over_time = np.cumsum(close_loop_regret)/np.arange(1, len(close_loop_regret)+1)
cum_regret_over_time1 = np.cumsum(close_loop_regret1)/np.arange(1, len(close_loop_regret1)+1)
plt.plot(cum_regret_over_time, label="cummulative regret")
plt.plot(cum_regret_over_time1, label="cummulative regret1")
plt.plot(np.zeros(len(close_loop_regret)), label="zero")
plt.legend()
plt.savefig("close_loop_regret.png")
a=1


# FUll exploration debugging
cost_full_expl, solver_status_full_expl = load_data("data_full_expl.pkl")
cost_full_expl_true, solver_status_full_expl_true = load_data("data_full_expl_true.pkl")

plt.plot(cost_full_expl, label="cost_full_expl")
plt.plot(cost_full_expl_true, label="cost_full_expl_true")
plt.ylim(-0.0008, 0.00)
plt.legend()
plt.savefig("cost_full_expl.png")
plt.close()

plt.plot(solver_status_full_expl, label="solver_status_full_expl")
plt.plot(solver_status_full_expl_true, label="solver_status_full_expl_true")
plt.ylim(-0.001, 0.001)
plt.savefig("cost_full_expl_status.png")
plt.legend()
plt.close()

# params["visu"]["show"] = True
# env_model = globals()[params["env"]["dynamics"]](params)

# plot cost difference
regret = np.array(cost_algo) - np.array(cost_opt)
regret1 = np.array(cost_algo1) - np.array(cost_opt)
regret_no_learning = (np.array(cost_no_learning) - np.array(cost_opt))[:50]
plt.plot(regret, label="regret")
plt.plot(regret1, label="regret1")
plt.plot(regret_no_learning, label="regret_no_learning")
cummulative_regret_over_time = np.cumsum(regret)/np.arange(1, len(regret)+1)
plt.plot(cummulative_regret_over_time, label="cummulative regret")
cum_no_learning = np.cumsum(regret_no_learning)/np.arange(1, len(regret_no_learning)+1)
plt.plot(cum_no_learning, label="cummulative regret_no_learning")
plt.plot(np.zeros(len(cummulative_regret_over_time)), label="zero")
plt.legend()
plt.savefig("cost.png")
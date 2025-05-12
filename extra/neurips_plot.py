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
    state_traj = data_dict["state_traj"]
    input_traj = data_dict["input_traj"]
    mean_state_traj = data_dict["mean_state_traj"]
    true_state_traj = data_dict["true_state_traj"]
    physical_state_traj = data_dict["physical_state_traj"]
    solver_cost = data_dict["solver_cost"]
    tilde_eps_list, ci_list = None, None
    if "tilde_eps_list" in data_dict:
        tilde_eps_list = data_dict["tilde_eps_list"]
        ci_list = data_dict["ci_list"]
    a_file.close()
    return solver_cost


cost_algo = load_data("data.pkl")
cost_algo1 = load_data("data1.pkl")
cost_opt = load_data("data_opt.pkl")
cost_no_learning = load_data("data_no_learning.pkl")

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
import argparse
import errno
import os
import warnings
from datetime import datetime

import casadi as ca
import gpytorch
import matplotlib.pyplot as plt
import torch
import yaml

from src.environment import ContiWorld
# from utils.ground_truth import GroundTruth
from src.DEMPC import DEMPC
# from utils.helper import TrainAndUpdateDensity, TrainAndUpdateConstraint, get_optimistic_intersection, SafelyExplore, UpdateSafeVisu, UpdateObjectiveVisu, get_frame_writer, save_data_plots, submodular_optimization, idxfromloc, get_pessimistic_union
# from src.utils.initializer import get_players_initialized
# from src.utils.plotting import plot_1D, plot_2D
from src.visu import Visualizer

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [12, 6]

workspace = "safe_gpmpc"

parser = argparse.ArgumentParser(description='A foo that bars')
parser.add_argument('-param', default="params")  # params

parser.add_argument('-env', type=int, default=0)
parser.add_argument('-i', type=int, default=3)  # initialized at origin
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

# set start and the goal location
# with open(env_load_path + "params_env.yaml") as file:
#     env_st_goal_pos = yaml.load(file, Loader=yaml.FullLoader)
# params["env"]["start_loc"] = env_st_goal_pos["start_loc"]
# params["env"]["goal_loc"] = env_st_goal_pos["goal_loc"]

# 3) Setup the environment. This class defines different environments eg: wall, forest, or a sample from GP.
env = ContiWorld(
    env_params=params["env"], common_params=params["common"], visu_params=params["visu"], env_dir=env_load_path, params=params)

print(args)
if args.i != -1:
    traj_iter = args.i

if not os.path.exists(save_path + str(traj_iter)):
    os.makedirs(save_path + str(traj_iter))

visu = Visualizer(params=params, path=save_path + str(traj_iter))

de_mpc = DEMPC(params, env, visu)
de_mpc.dempc_main()
visu.save_data()
exit()



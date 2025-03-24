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
# python sampling-gpmpc/benchmarking/simulate_true_reachable_set.py
workspace = "sampling-gpmpc"
sys.path.append(workspace)
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_car_residual_fs")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=4000)  # initialized at origin
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

input_gpmpc_data_list = []

# import glob
# file_list = glob.glob(f"{save_path}{str(args.i)}/*")
# for file in file_list:
#     with open(file, "rb") as input_data_file:
#         input_gpmpc_data = pickle.load(input_data_file)
#     input_gpmpc_data_list.append(input_gpmpc_data)

for i in range(1):
    input_data_path = input_data_path = f"{save_path}{str(args.i)}/data_X_traj_{i}.pkl"
    with open(input_data_path, "rb") as input_data_file:
        input_gpmpc_data = pickle.load(input_data_file)
    input_gpmpc_data_list.append(input_gpmpc_data)

print("Loaded input data")
X_traj = np.vstack(input_gpmpc_data_list)

# X_traj = X_traj[:151, :,:]
X_traj = X_traj[:3493, :,:]
from scipy.spatial import ConvexHull, convex_hull_plot_2d

hull_points = []
for i in range(1, X_traj.shape[2]):
    # pts_i = state_traj[0][i].reshape(-1, 2)
    pts_i = X_traj[:,:2,i]
    hull = ConvexHull(pts_i)
    if i - 1 < len(hull_points):
        hull_points[i - 1] = np.vstack(
            [hull_points[i - 1], hull.points[hull.vertices]]
        )
    else:
        hull_points.append(hull.points[hull.vertices])

a_file = open(save_path + str(traj_iter) + "/data_convex_hull_eps4e-4.pkl", "wb")
pickle.dump(hull_points, a_file)
a_file.close()

color = "powderblue" 
for i in range(X_traj.shape[2] - 2):
    hull = ConvexHull(np.concatenate([hull_points[i], hull_points[i + 1]]))
    stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
    plt.fill(
        hull.points[stack_vertices, 0],
        hull.points[stack_vertices, 1],
        alpha=1,
        color=color,
        lw=0,
    )

# plt.plot(X_traj[:,0,:].T, X_traj[:,1,:].T)
plt.savefig("fs_60.png")

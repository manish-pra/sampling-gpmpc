import sys, os
from scipy.spatial.distance import pdist, squareform
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from pathlib import Path
import argparse
import yaml
import torch

# NOTE: this file needs to be called from outside the root directory of the project, e.g.:
# python sampling-gpmpc/benchmarking/linearization_based_predictions.py
workspace = "sampling-gpmpc"
sys.path.append(workspace)

from src.environments.car_model_residual import CarKinematicsModel as bicycle_Bdx

workspace_plotting_utils = "extra/plotting_utilities"
sys.path.append(workspace_plotting_utils)

from plotting_utilities.plotting_utilities import *


parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_car_residual_fs")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=4)  # initialized at origin
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

print(args)
if args.i != -1:
    traj_iter = args.i

# Load trajectory data
a_file = open(save_path + str(traj_iter) + "/data_feedback_1e-8.pkl", "rb")
data_dict = pickle.load(a_file)
state_traj = data_dict["state_traj"]
a_file.close()

TEXTWIDTH = 16
set_figure_params(serif=True, fontsize=14)
# f = plt.figure(figsize=(TEXTWIDTH * 0.5 + 2.75, TEXTWIDTH * 0.5 * 1 / 2))
f = plt.figure(figsize=(cm2inches(12.0), cm2inches(4.0)))
ax = f.axes
plt.ylabel(r"$y$")
plt.xlabel(r"$x$")
plt.tight_layout(pad=0.0)

# Prepare the true reachable set
true_reachable_data_path = save_path + str(traj_iter)  + "/data_convex_hull.pkl"

with open(true_reachable_data_path, "rb") as input_data_file:
    true_reachable_hull_points = pickle.load(input_data_file)

color = "powderblue" 
for i in range(49):
    hull = ConvexHull(np.concatenate([true_reachable_hull_points[i], true_reachable_hull_points[i + 1]]))
    stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
    plt.fill(
        hull.points[stack_vertices, 0],
        hull.points[stack_vertices, 1],
        alpha=1,
        color=color,
        lw=0,
    )

def plot_ellipses(ax, x, y, eps_list):
    P = np.array(params["optimizer"]["terminal_tightening"]["P"])[:2,:2]
    # P*=10
    nH = len(eps_list) - 1 # not required on terminal state
    n_pts = 50
    ns_sub = x.shape[1] #int(x.shape[1]/4) + 1
    P_scaled = np.tile(P, (nH,1,1))/(eps_list[:nH,None, None]**2+1e-8)
    L = np.linalg.cholesky(P_scaled)
    t = np.linspace(0, 2 * np.pi, n_pts)
    z = np.vstack([np.cos(t), np.sin(t)])
    ell = np.linalg.inv(np.transpose(L, (0,2,1))) @ z

    all_ell = np.tile(ell, (ns_sub, 1, 1, 1))
    x_plt = all_ell[:,:,0,:] + x.T[:ns_sub,:nH,None]
    y_plt = all_ell[:,:,1,:] + y.T[:ns_sub,:nH,None]
    return plt.plot(x_plt.reshape(-1,n_pts).T, y_plt.reshape(-1,n_pts).T, color="blue", label="Terminal set", linewidth=0.2)


def plot_reachable_eps(ax, filepath, tilde_eps_list):
    with open(filepath, "rb") as input_data_file:
        reachable_hull_points = pickle.load(input_data_file)
    color = "lightcoral" 
    for i in range(49):
        hull = ConvexHull(np.concatenate([reachable_hull_points[i], reachable_hull_points[i + 1]]))
        stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
        plt.fill(
            hull.points[stack_vertices, 0],
            hull.points[stack_vertices, 1],
            alpha=1,
            color=color,
            lw=0,
        )

    max_rows = max(A.shape[0] for A in reachable_hull_points)
    # Pad each array to (10,2)
    padded_arrays = np.array([
        np.pad(A, ((0, max_rows - A.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        for A in reachable_hull_points
    ])
    plot_ellipses(ax, padded_arrays[:,:,0], padded_arrays[:,:,1], np.stack(tilde_eps_list)[:,-1])
    # for idx, xy in enumerate(reachable_hull_points):
    #     plot_ellipses(ax, xy[:,:,0], xy[:,1], [tilde_eps_list[idx][-1]])

eps = 2e-4

eps_data_path = save_path + str(traj_iter)  + f"/data_convex_hull_eps{eps:.0e}.pkl".replace("e-0", "e-")
params["agent"]["tight"]["dyn_eps"] = eps
params["optimizer"]["H"] = 50
tilde_eps_list, ci_list = bicycle_Bdx.get_reachable_set_ball(params, state_traj[0][:,3])
plot_reachable_eps(ax, eps_data_path, tilde_eps_list)

adapt_figure_size_from_axes(ax)
plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
# plt.xlim(-0.1, 1.45)
plt.ylim(-3, 15)
plt.savefig(
    f"eps{eps:.0e}.pdf",
    format="pdf",
    dpi=300,
    transparent=True,
)

# eps4e_4_data_path = save_path + str(traj_iter)  + "/data_convex_hull_eps4e-4.pkl"
# plot_reachable_eps(eps4e_4_data_path)

# eps2e_4_data_path = save_path + str(traj_iter)  + "/data_convex_hull_eps2e-4.pkl"
# plot_reachable_eps(eps2e_4_data_path)



# plt.savefig(save_path + str(traj_iter) + "/true_reachable_set.png")
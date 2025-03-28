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
parser.add_argument("-i", type=int, default=4441)  # initialized at origin
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
input_gpmpc_input_traj = data_dict["input_traj"][-1]
state_traj = data_dict["state_traj"]
true_state_traj = data_dict["true_state_traj"]
a_file.close()

TEXTWIDTH = 16
set_figure_params(serif=True, fontsize=14)
# f = plt.figure(figsize=(TEXTWIDTH * 0.5 + 2.75, TEXTWIDTH * 0.5 * 1 / 2))
f = plt.figure(figsize=(cm2inches(12.0), cm2inches(6.0)))
ax = f.axes
plt.ylabel(r"$y$")
plt.xlabel(r"$x$")
plt.tight_layout(pad=0.0)

def plot_ellipses(ax, x, y, eps_list):
    P = np.array(params["optimizer"]["terminal_tightening"]["P"])[:2,:2]
    # P*=10
    nH = len(eps_list) - 1 # not required on terminal state
    n_pts = 30
    ns_sub = x.shape[1] #int(x.shape[1]/4) + 1
    P_scaled = np.tile(P, (nH,1,1))/(eps_list[:nH,None, None]**2+1e-8)
    L = np.linalg.cholesky(P_scaled)
    t = np.linspace(0, 2 * np.pi, n_pts)
    z = np.vstack([np.cos(t), np.sin(t)])
    ell = np.linalg.inv(np.transpose(L, (0,2,1))) @ z

    all_ell = np.tile(ell, (ns_sub, 1, 1, 1))
    x_plt = all_ell[:,:,0,:] + x.T[:ns_sub,:nH,None]
    y_plt = all_ell[:,:,1,:] + y.T[:ns_sub,:nH,None]
    # return plt.plot(x_plt.reshape(-1,n_pts).T, y_plt.reshape(-1,n_pts).T, color="blue", label="Terminal set", linewidth=0.2)
    return x_plt, y_plt


def plot_reachable_eps(ax, filepath, tilde_eps_list, color):
    with open(filepath, "rb") as input_data_file:
        reachable_hull_points = pickle.load(input_data_file)
    # color = "lightcoral" 
    # for i in range(49):
    #     hull = ConvexHull(np.concatenate([reachable_hull_points[i], reachable_hull_points[i + 1]]))
    #     stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
    #     plt.fill(
    #         hull.points[stack_vertices, 0],
    #         hull.points[stack_vertices, 1],
    #         alpha=1,
    #         color=color,
    #         lw=0,
    #     )

    max_rows = max(A.shape[0] for A in reachable_hull_points)
    # Pad each array to (10,2)
    padded_arrays = np.array([
        np.pad(A, ((0, max_rows - A.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        for A in reachable_hull_points
    ])
    ell_x, ell_y = plot_ellipses(ax, padded_arrays[:,:,0], padded_arrays[:,:,1], np.stack(tilde_eps_list)[:,-1])

    for i in range(0,49):
        ell_pts_i = np.stack([ell_x[:,i,:].reshape(-1), ell_y[:,i,:].reshape(-1)]).T
        ell_pts_ip1 = np.stack([ell_x[:,i+1,:].reshape(-1), ell_y[:,i+1,:].reshape(-1)]).T
        hull = ConvexHull(np.concatenate([ell_pts_i[~np.isnan(ell_pts_i).any(axis=1)], ell_pts_ip1[~np.isnan(ell_pts_ip1).any(axis=1)]]))
        stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
        plt.fill(
            hull.points[stack_vertices, 0],
            hull.points[stack_vertices, 1],
            alpha=1,
            color=color,
            lw=0,
        )
    # for idx, xy in enumerate(reachable_hull_points):
    #     plot_ellipses(ax, xy[:,:,0], xy[:,1], [tilde_eps_list[idx][-1]])

# eps = 2e-4

# color_list = ["tab:orange","tab:blue", "tab:green"]
# color_list = ["lightsalmon","deepskyblue", "tab:orange", "lightgreen"]
color_list = ["lightsalmon","deepskyblue", "lightgreen"]
# eps_list = [8e-4, 4e-4, 2e-4]

# for eps, color in zip(eps_list, color_list):
#     eps_data_path = save_path + str(traj_iter)  + f"/data_convex_hull_eps{eps:.0e}.pkl".replace("e-0", "e-")
#     params["agent"]["tight"]["dyn_eps"] = eps
#     params["optimizer"]["H"] = 50
#     tilde_eps_list, ci_list = bicycle_Bdx.get_reachable_set_ball(params, state_traj[0][:,3])
#     plot_reachable_eps(ax, eps_data_path, tilde_eps_list, color)

r_1 = np.sqrt([4.50618781e-01])/np.sqrt([4.28592943e-01])
r_2 = np.sqrt([4.90922812e+00])/np.sqrt([4.28592943e-01])

# def get_eps_vec(N):
#     if N == 200:
#         eps = 0.0012467938969720 #0.0008862126722769
#     elif N == 2000:
#         eps = 0.0008861463373472
#     elif N == 20000:
#         eps = 0.0007138872931322
#     elif N == 200000:
#         eps = 0.0006026153249353
#     elif N == 2000000:
#         eps = 0.0005213836197142
#     elif N == 20000000:
#         eps = 0.0004596721185586
#     return np.array([eps/r_1, eps/np.ones(1), eps/r_2])

def get_eps_vec(N):
    if N == 200:
        eps = 0.0011142622935513
    elif N == 2000:
        eps = 0.0008404190593132
    elif N == 20000:
        eps = 0.0006954232571821
    elif N == 200000:
        eps = 0.0006009067950271
    elif N == 2000000:
        eps = 0.0005326789384778
    elif N == 20000000:
        eps = 0.0004763720895024
    return np.array([eps/r_1, eps/np.ones(1), eps/r_2])


# N_list = [200, 2000, 20000, 200000, 2000000, 20000000]
# N_list = [200, 2000, 20000, 20000000]
N_list = [200, 20000, 20000000]

for N, color in zip(N_list, color_list):
    eps_data_path = save_path + str(traj_iter)  + f"/data_convex_hull_dim_N{N}.pkl"
    # params["agent"]["tight"]["dyn_eps"] = get_eps_vec(N)
    params["optimizer"]["H"] = 50
    tilde_eps_list, ci_list = bicycle_Bdx.get_reachable_set_ball(params, state_traj[0][:,3], get_eps_vec(N))
    plot_reachable_eps(ax, eps_data_path, tilde_eps_list, color)


# Prepare the true reachable set
# true_reachable_data_path = save_path + str(traj_iter)  + "/data_convex_hull_N6e7_eta_e-7.pkl"
# true_reachable_data_path = save_path + str(traj_iter)  + "/data_convex_hull_dim_full.pkl"

# with open(true_reachable_data_path, "rb") as input_data_file:
#     true_reachable_hull_points = pickle.load(input_data_file)

# color = "black" 
# for i in range(49):
#     hull = ConvexHull(np.concatenate([true_reachable_hull_points[i], true_reachable_hull_points[i + 1]]))
#     stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
#     plt.fill(
#         hull.points[stack_vertices, 0],
#         hull.points[stack_vertices, 1],
#         alpha=1,
#         color=color,
#         lw=0,
#     )

# plt.plot(x_traj, y_traj, color="black", label="Trajectory", linewidth=0.4)
# from src.environments.car_model_residual import CarKinematicsModel as bicycle_Bdx
# from src.agent import Agent
# env_model = globals()[params["env"]["dynamics"]](params)
# agent = Agent(params, env_model)
# from src.visu import Visualizer
# visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=agent)
# agent.update_current_state(np.array(params["env"]["start"]))
# x_curr = agent.current_state[: agent.nx].reshape(agent.nx)
# propagated_state = visu.propagate_true_dynamics(x_curr, input_gpmpc_input_traj)
# plt.plot(propagated_state[:,0], propagated_state[:,1], ls='--',color="black", label="propagated Trajectory", linewidth=0.5)
plt.plot(true_state_traj[0][:,0], true_state_traj[0][:,1], ls='--',color="black", label="Trajectory", linewidth=0.8)

# plot lines of tracks with fixed start and end locations
w = 4.8
# y_coord = -0.62
# lw=1.25
# plt.plot([0, 7],[y_coord, y_coord], color="black", linewidth=lw)
# plt.plot([0, 7],[y_coord+w, y_coord+w], color="black", linewidth=lw)

# y_coord = 8.6
# plt.plot([13.5, 23.75],[y_coord, y_coord], color="black", linewidth=lw)
# plt.plot([13.5, 23.75],[y_coord+w, y_coord+w], color="black", linewidth=lw)

# y_coord = -0.62
# plt.plot([32.75, 42],[y_coord, y_coord], color="black", linewidth=lw)
# plt.plot([32.75, 42],[y_coord+w, y_coord+w], color="black", linewidth=lw)

w = 3.9
y_coord = -0.1
lw=1.25
plt.plot([0, 7.15],[y_coord, y_coord], color="black", linewidth=lw)
plt.plot([0, 7.15],[y_coord+w, y_coord+w], color="black", linewidth=lw)

y_coord = 9
plt.plot([14.2, 24.1],[y_coord, y_coord], color="black", linewidth=lw)
plt.plot([14.2, 24.1],[y_coord+w, y_coord+w], color="black", linewidth=lw)

y_coord = -0.1
plt.plot([33.25, 42],[y_coord, y_coord], color="black", linewidth=lw)
plt.plot([33.25, 42],[y_coord+w, y_coord+w], color="black", linewidth=lw)

import matplotlib.image as mpimg
# car_img = mpimg.imread("car_shorten.png") 
car_img = mpimg.imread("car_icon.png") 


adapt_figure_size_from_axes(ax)
# lx = 2.843
# ly = lx/2.0
# sx = 0.8
# sy = 2.7
# plt.imshow(car_img, extent=[sx-lx, sx, sy-ly, sy])  # (xmin, xmax, ymin, ymax)
lx = 2.843 
ly = lx/2.1
sx = 1.3
sy = 3.6
mx = 0.5
my = 2
plt.imshow(car_img, extent=[sx-lx-mx, sx, sy-ly-my, sy], zorder=10)  # (xmin, xmax, ymin, ymax)
plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.xlim(-2, 44)
plt.ylim(-3, 15)
plt.savefig(
    # f"eps{eps:.0e}.pdf",
    "overlapping_N2e2_2e4_2e7_true.pdf",
    format="pdf",
    dpi=300,
    transparent=True,
)
# eps4e_4_data_path = save_path + str(traj_iter)  + "/data_convex_hull_eps4e-4.pkl"
# plot_reachable_eps(eps4e_4_data_path)

# eps2e_4_data_path = save_path + str(traj_iter)  + "/data_convex_hull_eps2e-4.pkl"
# plot_reachable_eps(eps2e_4_data_path)



# plt.savefig(save_path + str(traj_iter) + "/true_reachable_set.png")
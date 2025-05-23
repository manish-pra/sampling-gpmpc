import argparse
import errno
import os
import warnings

import matplotlib.pyplot as plt
import yaml
import dill as pickle
import sys
workspace = "sampling-gpmpc"
sys.path.append(workspace)
import numpy as np

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from src.environments.pendulum import Pendulum as pendulum
from src.environments.car_model_residual import CarKinematicsModel as bicycle_Bdx
from src.environments.car_model import CarKinematicsModel as bicycle
from src.environments.car_racing import CarKinematicsModel as car_racing
from src.environments.pendulum1D import Pendulum as Pendulum1D
from src.environments.drone import Drone as drone
from src.agent import Agent

warnings.filterwarnings("ignore")
# plt.rcParams["figure.figsize"] = [12, 6]

workspace = "sampling-gpmpc"
sys.path.append(workspace)
from src.visu import Visualizer
parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_drone_sagedynx")
parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=43)  # initialized at origin

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

a_file = open(save_path + str(traj_iter) + "/data_lap1.pkl", "rb")
data_dict = pickle.load(a_file)
state_traj = data_dict["state_traj"]
input_traj = data_dict["input_traj"]
mean_state_traj = data_dict["mean_state_traj"]
true_state_traj = data_dict["true_state_traj"]
physical_state_traj = data_dict["physical_state_traj"]
tilde_eps_list, ci_list = None, None
a_file.close()


opt_a_file = open(save_path + str(traj_iter) + "/data_opt.pkl", "rb")
opt_data_dict = pickle.load(opt_a_file)
# state_traj = data_dict["state_traj"]
# input_traj = data_dict["input_traj"]
# mean_state_traj = data_dict["mean_state_traj"]
# true_state_traj = data_dict["true_state_traj"]
opt_physical_state_traj = opt_data_dict["physical_state_traj"]
# tilde_eps_list, ci_list = None, None
opt_a_file.close()


params["visu"]["show"] = True
env_model = globals()[params["env"]["dynamics"]](params)

agent = Agent(params, env_model)
visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=agent)
visu.tilde_eps_list = tilde_eps_list
visu.ci_list = ci_list

nx = params["agent"]["dim"]["nx"]


workspace_plotting_utils = "extra"
sys.path.append(os.path.join(os.path.dirname(__file__),workspace_plotting_utils))
print("sys.path", sys.path)
from plotting_tools.plotting_utilities import *
TEXTWIDTH = 16
set_figure_params(serif=True, fontsize=14)
f, ax = plt.subplots(figsize=(cm2inches(12.0), cm2inches(8.0)))
visu.f_handle["gp"] = f
visu.initialize_plot_handles(visu.f_handle["gp"])
# from matplotlib.ticker import AutoMinorLocator
# ax.xaxis.set_minor_locator(AutoMinorLocator(15))
# ax.grid(True, axis='x', which='major', linestyle='-', linewidth=0.75)
# ax.grid(True, axis='x', which='minor', linestyle=':', linewidth=0.5)

# # Function x**(1/2)
# def forward(x):
#     return x*6


# def inverse(x):
#     return x/6
# ax.set_xscale("function", functions=(forward, inverse))

def plot_reachable_set(ell_x, ell_y, color):
    H = ell_x.shape[1]
    for i in range(0,H-1):
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

# ax.set_box_aspect(0.5) 
# agent = Agent(params)
# visu.extract_data()

# physical_state_traj = np.vstack(visu.physical_state_traj)
# plt.plot(physical_state_traj[:,0], physical_state_traj[:,1])
# plt.show()
# load data)

idx_true_start = 0
idx_mean_start = 0
# (l,) = ax.plot([], [], "tab:orange")


plot_iter = 200
# for i in range(plot_iter, plot_iter + 1):
#     if params["agent"]["true_dyn_as_sample"]:
#         idx_true_start = 0
#         if params["agent"]["mean_as_dyn_sample"]:
#             idx_mean_start = nx
#         else:
#             # set mean to true for plotting
#             idx_mean_start = 0
#     elif params["agent"]["mean_as_dyn_sample"]:
#         # set true to mean for plotting
#         idx_true_start = 0
#         idx_mean_start = 0

#     true_state_traj = state_traj[i][:, idx_true_start: idx_true_start+nx]
#     mean_state_traj = state_traj[i][:, idx_mean_start: idx_mean_start+nx]

#     # true_state_traj_prop = visu.true_state_traj[-1]

#     visu.record_out(
#         physical_state_traj[i],
#         state_traj[i],
#         input_traj[i],
#         true_state_traj,
#         mean_state_traj, 
#     )
#     # print(true_state_traj[i])
#     temp_obj = visu.plot_receding_traj()

#     # visu.writer_gp.grab_frame()
#     # visu.remove_temp_objects(temp_obj)

# import matplotlib.image as mpimg
# car_img = mpimg.imread("car_shorten.png") 
# car_img = mpimg.imread("drone_icon.png") 
# lx = 0.2843 
# ly = lx
# sx = 1.3
# sy = 3.6
# mx = 0.05
# my = 0.2
# plt.imshow(car_img, extent=[sx-lx-mx, sx, sy-ly-my, sy])
# plt.gca().set_aspect('equal', adjustable='box')  # Maintain aspect ratio
# plt.imshow(car_img, extent=[sx-lx-mx, sx, sy-ly-my, sy])

plot_iter_list = [2, 100, 150, 200, 320, 370, 420]

for plot_iter in plot_iter_list:
    X = state_traj[plot_iter]

    # ell_x, ell_y = visu.get_ellipses_pts(
    #     X[:, :: nx],
    #     X[:, 1 :: nx]
    # )

    # plot_reachable_set(ell_x, ell_y, color="powderblue")    

    s_x, s_y = X[:, :: nx].T[:,:,np.newaxis], X[:, 1 :: nx].T[:,:,np.newaxis]
    plot_reachable_set(s_x, s_y, color="powderblue")    
    # pred_true_state = np.vstack(true_state_traj)
    ax.plot(X[:, 0 :: nx], X[:, 1 :: nx], linestyle="-")
    x_phy, y_phy = X[:, :: nx][:,0], X[:, 1 :: nx][:,0]
    plt.plot(x_phy[0], y_phy[0], marker='o',color="tab:blue", linewidth=0.5, markersize=4)
    # ax.plot(
    #     pred_true_state[:, 0],
    #     pred_true_state[:, 1],
    #     color="black",
    #     label="true",
    #     linestyle="-",
    # )

opt_physical_state_traj = np.vstack(opt_physical_state_traj)
plt.plot(opt_physical_state_traj[:,0], opt_physical_state_traj[:,1], color="tab:green", linewidth=1.6, label="Optimal")

# plot physical state upto plot iter
physical_state_traj = np.vstack(physical_state_traj)
closed_loop_color = "black"
closed_loop_lw = 1.6
plt.plot(physical_state_traj[:,0], physical_state_traj[:,1], "--", color=closed_loop_color, linewidth=closed_loop_lw, label="SAGE-DynX")
# plt.plot(state_traj[-1][:,0], state_traj[-1][:,1], "--", color=closed_loop_color, label="Trajectory", linewidth=closed_loop_lw)

# x_phy, y_phy = physical_state_traj[:plot_iter+1,:nx][:,0], physical_state_traj[:plot_iter+1,:nx][:,1]
# plt.plot(x_phy, y_phy,"--",color=closed_loop_color, linewidth=closed_loop_lw)


# plot mean state
# plt.plot(X[:, 0], X[:, 1], "tab:blue", linewidth=1.5)
# plt.plot(x_phy[-1], y_phy[-1], marker='o',color="tab:blue", linewidth=0.5, markersize=5)

# plt.plot(2.15, 2.3, marker='x',color="tab:blue", linewidth=0.5, markersize=5)
# plt.plot(3.12, 0.0, marker='x',color="tab:green", linewidth=0.5, markersize=5)
plt.legend(fontsize='small', labelspacing=0.2, handlelength=1)
plt.grid(False) 
ax = visu.f_handle["gp"].axes
adapt_figure_size_from_axes(ax)
plt.ylabel(r"$p_2[m]$")
plt.xlabel(r"$p_1[m]$")
# plt.xlim(2.0, 3.7)
plt.ylim(-6.0, 6.0)
plt.xlim(-6.0, 6.0)
plt.tight_layout(pad=0.0)
visu.f_handle["gp"].savefig("drone_setup.pdf", dpi=600,transparent=True,format="pdf", bbox_inches="tight")
# plt.savefig(
#     # f"eps{eps:.0e}.pdf",
#     "pendulum.pdf",
#     format="pdf",
#     dpi=300,
#     transparent=True,
# )
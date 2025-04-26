import argparse
import errno
import os
import warnings

import matplotlib.pyplot as plt
import yaml
import dill as pickle

import numpy as np
import sys
from scipy.spatial import ConvexHull, convex_hull_plot_2d

warnings.filterwarnings("ignore")
# plt.rcParams["figure.figsize"] = [12, 6]

workspace = "sampling-gpmpc"
sys.path.append(workspace)
from src.visu import Visualizer
parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_pendulum1D_samples")  # params
parser.add_argument("-env_model", type=str, default="pendulum")
parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=43)  # initialized at origin
parser.add_argument("-plot_koller", type=bool, default=False)
parser.add_argument("-plot_automatica", type=bool, default=False)

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

a_file = open(save_path + str(traj_iter) + "/data.pkl", "rb")
data_dict = pickle.load(a_file)
state_traj = data_dict["state_traj"]
input_traj = data_dict["input_traj"]
mean_state_traj = data_dict["mean_state_traj"]
true_state_traj = data_dict["true_state_traj"]
physical_state_traj = data_dict["physical_state_traj"]
tilde_eps_list, ci_list = None, None
if "tilde_eps_list" in data_dict:
    from src.environments.car_model_residual import CarKinematicsModel as bicycle_Bdx
    tilde_eps_list, ci_list = bicycle_Bdx.get_reachable_set_ball(params, state_traj[0][:,3])
    # tilde_eps_list = data_dict["tilde_eps_list"]
    # ci_list = data_dict["ci_list"]
a_file.close()



params["visu"]["show"] = True
visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=None)
visu.tilde_eps_list = tilde_eps_list
visu.ci_list = ci_list

nx = params["agent"]["dim"]["nx"]


workspace_plotting_utils = "extra"
sys.path.append(os.path.join(os.path.dirname(__file__),workspace_plotting_utils))
print("sys.path", sys.path)
from plotting_tools.plotting_utilities import *
TEXTWIDTH = 16
set_figure_params(serif=True, fontsize=14)
f, ax = plt.subplots(figsize=(cm2inches(12.0), cm2inches(6.0)))
visu.f_handle["gp"] = f
visu.initialize_plot_handles(visu.f_handle["gp"])
# from matplotlib.ticker import AutoMinorLocator
# ax.xaxis.set_minor_locator(AutoMinorLocator(15))
# ax.grid(True, axis='x', which='major', linestyle='-', linewidth=0.75)
# ax.grid(True, axis='x', which='minor', linestyle=':', linewidth=0.5)

# Function x**(1/2)
def forward(x):
    return x*6


def inverse(x):
    return x/6
ax.set_xscale("function", functions=(forward, inverse))

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


plot_iter = 2
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

X = state_traj[plot_iter]
eps_tightening = np.stack(tilde_eps_list)[:,-1]
ell_x, ell_y = visu.get_ellipses_pts(
    X[:, :: nx],
    X[:, 1 :: nx],
    eps_tightening
)

plot_reachable_set(ell_x, ell_y, color="powderblue")    

# pred_true_state = np.vstack(true_state_traj)

# ax.plot(
#     pred_true_state[:, 0],
#     pred_true_state[:, 1],
#     color="black",
#     label="true",
#     linestyle="-",
# )

# plot physical state upto plot iter
physical_state_traj = np.vstack(physical_state_traj)
closed_loop_color = "black"
closed_loop_lw = 1.6
plt.plot(physical_state_traj[:,0], physical_state_traj[:,1], "--", color=closed_loop_color, label="Trajectory", linewidth=closed_loop_lw)
plt.plot(state_traj[-1][:,0], state_traj[-1][:,1], "--", color=closed_loop_color, label="Trajectory", linewidth=closed_loop_lw)

x_phy, y_phy = physical_state_traj[:plot_iter+1,:nx][:,0], physical_state_traj[:plot_iter+1,:nx][:,1]
plt.plot(x_phy, y_phy,"--",color=closed_loop_color, linewidth=closed_loop_lw)


# plot mean state
plt.plot(X[:, 0], X[:, 1], "tab:blue", linewidth=1.5)
plt.plot(x_phy[-1], y_phy[-1], marker='o',color="tab:blue", linewidth=0.5, markersize=5)

# plt.plot(2.15, 2.3, marker='x',color="tab:blue", linewidth=0.5, markersize=5)
# plt.plot(3.12, 0.0, marker='x',color="tab:green", linewidth=0.5, markersize=5)

ax = visu.f_handle["gp"].axes
adapt_figure_size_from_axes(ax)
plt.ylabel(r"$\omega$")
plt.xlabel(r"$\theta$")
# plt.xlim(2.0, 3.7)
# plt.xlim(2.1, 3.6)
plt.xlim(2.1, 3.3)
plt.tight_layout(pad=0.0)
visu.f_handle["gp"].savefig("pendulum.pdf", dpi=600,transparent=True,format="pdf", bbox_inches="tight")
# plt.savefig(
#     # f"eps{eps:.0e}.pdf",
#     "pendulum.pdf",
#     format="pdf",
#     dpi=300,
#     transparent=True,
# )
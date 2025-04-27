import argparse
import errno
import os
import warnings

import matplotlib.pyplot as plt
import yaml
import dill as pickle
from src.visu import Visualizer
import numpy as np
import sys

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

workspace = "sampling-gpmpc"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_pendulum1D_samples")  # params
# parser.add_argument("-param", default="params_car_samples")  # params
# parser.add_argument("-param", default="params_car_residual")  # params
parser.add_argument("-env_model", type=str, default="pendulum")
parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=42)  # initialized at origin
parser.add_argument("-plot_koller", type=bool, default=True)
parser.add_argument("-plot_automatica", type=bool, default=True)

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

# Koller plot
if args.plot_koller:
    with open(save_path + str(traj_iter) + "/koller_ellipse_data.pkl", "rb") as pkl_file:
        koller_ellipse_list = pickle.load(pkl_file)
    with open(save_path + str(traj_iter) + "/koller_mean_data.pkl", "rb") as pkl_file:
        koller_mean_arr = np.array(pickle.load(pkl_file))
    with open(save_path + str(traj_iter) + "/koller_true_data.pkl", "rb") as pkl_file:
        koller_true_arr = np.array(pickle.load(pkl_file))

params["visu"]["show"] = True
visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=None)
visu.tilde_eps_list = tilde_eps_list
visu.ci_list = ci_list

nx = params["agent"]["dim"]["nx"]
ax = visu.f_handle["gp"].axes[0]
if args.plot_automatica:
    workspace_plotting_utils = "extra"
    sys.path.append(os.path.join(os.path.dirname(__file__),workspace_plotting_utils))
    print("sys.path", sys.path)
    from plotting_tools.plotting_utilities import *
    TEXTWIDTH = 16
    set_figure_params(serif=True, fontsize=14)
    f, ax = plt.subplots(figsize=(cm2inches(12.0), cm2inches(6.0)))
    visu.f_handle["gp"] = f


# agent = Agent(params)
# visu.extract_data()

# physical_state_traj = np.vstack(visu.physical_state_traj)
# plt.plot(physical_state_traj[:,0], physical_state_traj[:,1])
# plt.show()
# load data)

idx_true_start = 0
idx_mean_start = 0
# (l,) = ax.plot([], [], "tab:orange")
for i in range(0, len(state_traj)):
    if params["agent"]["true_dyn_as_sample"]:
        idx_true_start = 0
        if params["agent"]["mean_as_dyn_sample"]:
            idx_mean_start = nx
        else:
            # set mean to true for plotting
            idx_mean_start = 0
    elif params["agent"]["mean_as_dyn_sample"]:
        # set true to mean for plotting
        idx_true_start = 0
        idx_mean_start = 0

    true_state_traj = state_traj[i][:, idx_true_start: idx_true_start+nx]
    mean_state_traj = state_traj[i][:, idx_mean_start: idx_mean_start+nx]

    # true_state_traj_prop = visu.true_state_traj[-1]

    if not args.plot_automatica:
        visu.record_out(
            physical_state_traj[i],
            state_traj[i],
            input_traj[i],
            true_state_traj,
            mean_state_traj, 
        )
        # print(true_state_traj[i])
        temp_obj = visu.plot_receding_traj()

    if args.plot_koller:
        for j in range(len(koller_ellipse_list)):
            plt_obj = visu.plot_general_ellipsoid(koller_ellipse_list[j], color="tab:red", alpha=0.7)
        H_explode = 14
        ax.plot(koller_mean_arr[:H_explode,0,:H_explode], koller_mean_arr[:H_explode,1,:H_explode], "tab:blue", linewidth=1)
        # ax.plot(koller_true_arr[:,0,:], koller_true_arr[:,1,:], "tab:green", linewidth=0.5)
        plt.plot(true_state_traj[:,0], true_state_traj[:,1], ls='--',color="black", label="Trajectory", linewidth=0.8)
        if args.plot_automatica:
            # plt.plot(koller_mean_arr[:,0,:], koller_mean_arr[:,1,:], "tab:blue", linewidth=0.5)
            # plt.plot(koller_true_arr[:,0,:], koller_true_arr[:,1,:], "tab:green", linewidth=0.5)
            import matplotlib.image as mpimg
            # car_img = mpimg.imread("car_shorten.png") 
            car_img = mpimg.imread("car_icon.png") 
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
            # plt.xlim(-2, 10.9)
            # plt.ylim(-3, 9)
            ax = f.axes
            adapt_figure_size_from_axes(ax)
            w = 3.9
            y_coord = -0.1
            lw=1.25
            plt.plot([0, 7.15],[y_coord, y_coord], color="black", linewidth=lw, zorder=-1)
            plt.plot([0, 7.15],[y_coord+w, y_coord+w], color="black", linewidth=lw, zorder=-1)

            y_coord = 9
            plt.plot([14.2, 24.1],[y_coord, y_coord], color="black", linewidth=lw, zorder=-1)
            plt.plot([14.2, 24.1],[y_coord+w, y_coord+w], color="black", linewidth=lw, zorder=-1)

            y_coord = -0.1
            plt.plot([33.25, 42],[y_coord, y_coord], color="black", linewidth=lw, zorder=-1)
            plt.plot([33.25, 42],[y_coord+w, y_coord+w], color="black", linewidth=lw, zorder=-1)
            plt.ylabel(r"$y$")
            plt.xlabel(r"$x$")
            plt.tight_layout(pad=0.0)
            plt.savefig(
                # f"eps{eps:.0e}.pdf",
                "koller.pdf",
                format="pdf",
                dpi=300,
                transparent=True,
            )
        # print(np.mean(koller_ellipse_list[i]))
        # plt_obj = visu.plot_general_ellipsoid(ax, koller_ellipse_list[i])
    # temp_obj = visu.plot_receding_pendulum_traj()
    # temp_obj = visu.plot_receding_car_traj()
    # visu.plot_car(
    #     physical_state_traj[i][0],
    #     physical_state_traj[i][1],
    #     physical_state_traj[i][2],
    #     l,
    # )
    visu.writer_gp.grab_frame()
    visu.remove_temp_objects(temp_obj)
    # visu.remove_temp_objects(plt_obj)

# visu.f_handle["gp"].savefig(save_path + str(traj_iter) + "/prediction.png", dpi=600)
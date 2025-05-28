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
parser.add_argument("-param", default="params_pendulum")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=22)  # initialized at origin
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

path_traj_opt = f"{save_path}{str(args.i)}/data.pkl"
with open(path_traj_opt, "rb") as input_data_file:
    opt_data = pickle.load(input_data_file)

true_traj = opt_data["true_state_traj"]

for i in range(1):
    input_data_path = f"{save_path}{str(args.i)}/X_traj_list_{i}.pkl"
    with open(input_data_path, "rb") as input_data_file:
        input_gpmpc_data = pickle.load(input_data_file)
    input_gpmpc_data_list.append(input_gpmpc_data)

print("Loaded input data")

metadata = dict(
    title="Movie Test", artist="Matplotlib", comment="Movie support!"
)
import matplotlib.animation as manimation
writer = manimation.FFMpegWriter(
    fps=24, codec="libx264", metadata=metadata
)
fig, ax = plt.subplots(figsize=(12 / 2.4, 6 / 2.4))
writer.setup(fig,"video_sampling_finite.mp4", dpi=800) 
# X_traj = np.vstack(input_gpmpc_data_list)
ax.set_xlim(-0.03, 0.7)
ax.set_ylim(-0.01, 2.5)
# X_traj = np.vstack(input_gpmpc_data_list[0])

ax.plot([-1,-0.9],[-1,-0.9],lw=0.8, alpha=0.3, label=r"trajectory samples")
max_samples = 200
for n_sample in range(1,max_samples):
    remove_list = []
    x = [input_gpmpc_h[:n_sample,0,0,0] for input_gpmpc_h in input_gpmpc_data]
    y = [input_gpmpc_h[:n_sample,0,0,1] for input_gpmpc_h in input_gpmpc_data]
    for i in range(len(x)):
        x[i] = np.array(x[i])
        y[i] = np.array(y[i])
    remove_list.append(ax.plot(x,y, lw=0.8, alpha=0.2, color="tab:blue"))
    x_all = np.vstack(x)
    y_all = np.vstack(y)
    x_diff = np.abs(x_all - true_traj[0][:,[0]])
    y_diff = np.abs(y_all - true_traj[0][:,[1]])
    x_max = np.max(x_diff, axis=0)
    y_max = np.max(y_diff, axis=0)
    # samples_max_distance = np.max(np.maximum(x_diff, y_diff, out=x_diff), axis=0)
    samples_max_distance = np.sqrt(x_max**2 + y_max**2)
    closet_idx = np.argmin(samples_max_distance)
    print(np.min(np.max(np.maximum(x_diff, y_diff, out=x_diff), axis=0)), samples_max_distance[closet_idx], closet_idx)
    # np.diff(x_all - true_traj[0][:,[0]], axis=1)
    # closest_sample = np.argmin(np.linalg.norm(x_all - true_traj[0][:,0], axis=1))
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["text.usetex"] =True
    plt.rcParams["font.family"] = "serif"
    remove_list.append(ax.plot(true_traj[0][:,0], true_traj[0][:,1], color="black", lw=2, ls="--", label=r"$g^\mathrm{tr}$ trajectory"))
    remove_list.append(ax.plot(x_all[:,closet_idx], y_all[:,closet_idx], color="tab:orange",lw=1.2, label=r"$\epsilon$-close trajectory"))
    print(np.max([np.max(np.abs(x_all[:,closet_idx]-true_traj[0][:,0])), 
                  np.max(np.abs(y_all[:,closet_idx]-true_traj[0][:,1]))]))
    # plt.savefig("temp.png")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.legend()

    # Turn off all ticks
    # ax.tick_params(
    #     axis='both',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     left=False,
    #     right=False,
    #     labelbottom=False,
    #     labelleft=False
    # )
    plt.tight_layout(pad=0.2)
    writer.grab_frame()
    for t in remove_list:
        if type(t) is list:
            for tt in t:
                tt.remove()
        else:
            t.remove()
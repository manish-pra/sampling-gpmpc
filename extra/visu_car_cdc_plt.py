from plotting_utilities.plotting_utilities.utilities import *

import argparse
import errno
import os
import warnings
import sys

sys.path.insert(0, "/home/manish/work/MPC_Dyn/safe_gpmpc")
import matplotlib.pyplot as plt
import yaml
import dill as pickle
from src.visu import Visualizer
import numpy as np
import math

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]


filename = "car_trajectory"


workspace = "safe_gpmpc"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_car")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=40)  # initialized at origin
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
a_file.close()

params["visu"]["show"] = True
# visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=None)
# agent = Agent(params)
# visu.extract_data()

# physical_state_traj = np.vstack(visu.physical_state_traj)
# plt.plot(physical_state_traj[:,0], physical_state_traj[:,1])
# plt.show()
# load data)

fig_gp, ax = plt.subplots(figsize=(16 / 2.4, 3.4 / 2.4))
fig_gp.tight_layout(pad=0)
ax.set_aspect("equal", "box")
nx = params["agent"]["dim"]["nx"]
(l,) = ax.plot([], [], "tab:brown")


def plot_car(x, y, yaw, l):
    factor = 0.2
    l_f = 0.275 * factor
    l_r = 0.425 * factor
    W = 0.3 * factor
    outline = np.array(
        [[-l_r, l_f, l_f, -l_r, -l_r], [W / 2, W / 2, -W / 2, -W / 2, W / 2]]
    )

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

    outline = np.matmul(outline.T, Rot1).T

    outline[0, :] += x
    outline[1, :] += y
    plt.plot(
        np.array(outline[0, :]).flatten(),
        np.array(outline[1, :]).flatten(),
        "tab:brown",
        linewidth=2,
    )
    # l.set_data(np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten())


y_min = params["optimizer"]["x_min"][1]
y_max = params["optimizer"]["x_max"][1]
ax.add_line(plt.Line2D([-0.3, 2.4], [y_max, y_max], color="red", linestyle="--"))
ax.add_line(plt.Line2D([-0.3, 2.4], [y_min, y_min], color="red", linestyle="--"))

for ellipse in params["env"]["ellipses"]:
    x0 = params["env"]["ellipses"][ellipse][0]
    y0 = params["env"]["ellipses"][ellipse][1]
    a = params["env"]["ellipses"][ellipse][2]
    b = params["env"]["ellipses"][ellipse][3]
    f = params["env"]["ellipses"][ellipse][4]
    a = np.sqrt(a * f)  # radius on the x-axis
    b = np.sqrt(b * f)  # radius on the y-axis
    t = np.linspace(0, 2 * 3.14, 100)
    plt.plot(x0 + a * np.cos(t), y0 + b * np.sin(t), color="black")


state_traj = np.stack(state_traj)
ax.plot(state_traj[-1, :, ::4], state_traj[-1, :, 1::4], alpha=0.5, lw=1)
ax.plot(state_traj[20, :, ::4], state_traj[20, :, 1::4], alpha=0.5, lw=1)

true_state_traj = np.stack(true_state_traj)
ax.plot(
    true_state_traj[-1, :, ::4],
    true_state_traj[-1, :, 1::4],
    color="black",
    lw=1.5,
)
ax.plot(
    true_state_traj[20, :, ::4],
    true_state_traj[20, :, 1::4],
    color="black",
    lw=1.5,
)

physical_state_traj = np.stack(physical_state_traj)
ax.plot(physical_state_traj[:, 0], physical_state_traj[:, 1], color="tab:blue", lw=2)


plot_car(
    physical_state_traj[-1][0],
    physical_state_traj[-1][1],
    physical_state_traj[-1][2],
    l,
)

plot_car(
    physical_state_traj[20][0],
    physical_state_traj[20][1],
    physical_state_traj[20][2],
    l,
)


ax.set_yticklabels([])
ax.set_xticklabels([])
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0.0)

plt.savefig("figures/" + filename + ".pdf")

# for i in range(0, len(state_traj)):
#     mean_state_traj = state_traj[i][:, :nx]
#     visu.record_out(
#         physical_state_traj[i],
#         state_traj[i],
#         input_traj[i],
#         true_state_traj[i],
#         mean_state_traj,
#     )
#     # print(true_state_traj[i])
#     # temp_obj = visu.plot_receding_pendulum_traj()
#     temp_obj = visu.plot_receding_car_traj()
#     # visu.plot_car(
#     #     physical_state_traj[i][0],
#     #     physical_state_traj[i][1],
#     #     physical_state_traj[i][2],
#     #     l,
#     # )
#     visu.writer_gp.grab_frame()
#     visu.remove_temp_objects(temp_obj)

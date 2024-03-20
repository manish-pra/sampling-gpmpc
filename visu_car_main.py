import argparse
import errno
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import math

import yaml
import dill as pickle
from src.visu import Visualizer

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

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

visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=None)
visu.extract_data()
ax = visu.f_handle["gp"].axes[0]
(l,) = ax.plot([], [], "tab:orange")
(l1,) = ax.plot([], [], "black")

factor = 1.0
l_f = 0.029 * factor
l_r = 0.033 * factor

W = 0.03 * factor


def plot_car(x, y, yaw, l):

    outline = np.array(
        [[-l_r, l_f, l_f, -l_r, -l_r], [W / 2, W / 2, -W / 2, -W / 2, W / 2]]
    )

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

    outline = np.matmul(outline.T, Rot1).T

    outline[0, :] += x
    outline[1, :] += y

    # plt.plot(np.array(outline[0, :]).flatten(),
    #          np.array(outline[1, :]).flatten(), 'black')
    l.set_data(np.array(outline[0, :-1]).flatten(), np.array(outline[1, :-1]).flatten())
    l1.set_data(
        np.array(outline[0, -2:]).flatten(), np.array(outline[1, -2:]).flatten()
    )


# agent = Agent(params)
# visu.extract_data()

# physical_state_traj = np.vstack(visu.physical_state_traj)
# plt.plot(physical_state_traj[:,0], physical_state_traj[:,1])
# plt.show()
# load data)
nx = params["agent"]["dim"]["nx"]
for i in range(0, len(state_traj)):
    mean_state_traj = state_traj[i][:, :nx]
    visu.record_out(
        physical_state_traj[i],
        state_traj[i],
        input_traj[i],
        true_state_traj[i],
        mean_state_traj,
    )
    temp_obj = visu.plot_receding_pendulum_traj()
    visu.writer_gp.grab_frame()
    visu.remove_temp_objects(temp_obj)


visu = Visu(
    grid_V=env.VisuGrid,
    safe_boundary=env.get_safe_init()["Cx_X"],
    true_constraint_function=opt.true_constraint_function,
    true_objective_func=opt.true_density,
    opt_goal=opt.opt_goal,
    optimal_feasible_boundary=opt.optimal_feasible_boundary,
    params=params,
    path=data_load_path + str(traj_iter),
)

visu.extract_data()

players = get_players_initialized(env.get_safe_init(), params, env.VisuGrid)
num_iters = len(visu.player_train_pts)

if (
    args.param == "params_bicycle"
    or args.param == "params_2D"
    or "car" in args.param
    or params["agent"]["dynamics"] == "bicycle"
):
    factor = 1.0
    l_f = 0.029 * factor
    l_r = 0.033 * factor

    W = 0.03 * factor

    ax = visu.f_handle["gp"].axes[0]
    # l, = ax.plot([], [], 'tab:orange')
    (l,) = ax.plot([], [], "tab:brown")

    def plot_car(x, y, yaw, l):

        outline = np.array(
            [[-l_r, l_f, l_f, -l_r, -l_r], [W / 2, W / 2, -W / 2, -W / 2, W / 2]]
        )

        Rot1 = np.array(
            [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
        )

        outline = np.matmul(outline.T, Rot1).T

        outline[0, :] += x
        outline[1, :] += y

        # plt.plot(np.array(outline[0, :]).flatten(),
        #          np.array(outline[1, :]).flatten(), 'black')
        l.set_data(
            np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten()
        )  # , 'black')

else:
    factor = 1.0
    l_f = 0.030 * factor
    l_r = 0.030 * factor

    W = 0.06 * factor

    def plot_car(x, y, yaw, l):

        outline = np.array(
            [[-l_r, l_f, l_f, -l_r, -l_r], [W / 2, W / 2, -W / 2, -W / 2, W / 2]]
        )

        Rot1 = np.array(
            [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
        )

        outline = np.matmul(outline.T, Rot1).T

        outline[0, :] += x
        outline[1, :] += y

        # plt.plot(np.array(outline[0, :]).flatten(),
        #          np.array(outline[1, :]).flatten(), 'black')
        l.set_data(
            np.array(outline[0, :-1]).flatten(), np.array(outline[1, :-1]).flatten()
        )
        l1.set_data(
            np.array(outline[0, -2:]).flatten(), np.array(outline[1, -2:]).flatten()
        )


pose_torch = np.empty((0, 3))
vel_torch = np.empty(0)
input_torch = np.empty((0, 2))
time_torch = np.empty(0)
dt_torch = np.empty(0)
for iter in range(num_iters):
    len_traj = len(visu.state_traj[iter])
    time, pose = (
        visu.state_traj[iter][: int(len_traj / 2)][:, -1],
        visu.state_traj[iter][: int(len_traj / 2)][:, :3],
    )
    vel_torch = np.concatenate(
        (vel_torch, visu.state_traj[iter][: int(len_traj / 2)][:, 3])
    )
    input_torch = np.concatenate(
        (input_torch, visu.input_traj[iter][: int(len_traj / 2)][:, :2])
    )
    dt_torch = np.concatenate(
        (dt_torch, visu.input_traj[iter][: int(len_traj / 2)][:, 2])
    )
    pose_torch = np.concatenate((pose_torch, pose))
    time_torch = np.concatenate(
        (time_torch, time + time_torch[-1] if len(time_torch) > 0 else time)
    )


# Plot statistics
# b = plt.figure(3)
# pose_x vs pose_y, pose_x vs time, pose_y vs time, yaw vs time, vel vs time, acc vs time, steer vs time, dt vs time
fig, ax = plt.subplots(3, 3)
ax[0, 0].plot(pose_torch[:, 0], pose_torch[:, 1], color="tab:green", alpha=0.3)
ax[0, 0].plot(pose_torch[:, 0], pose_torch[:, 1], ".")
ax[0, 0].set_title("pose_x vs pose_y")

ax[0, 1].plot(time_torch, pose_torch[:, 0])
ax[0, 1].set_title("pose_x")

ax[0, 2].plot(time_torch, pose_torch[:, 1])
ax[0, 2].set_title("pose_y")

ax[1, 0].plot(time_torch, pose_torch[:, 2])
ax[1, 0].set_title("yaw")

ax[1, 1].plot(time_torch, vel_torch)
ax[1, 1].set_title("vel")

ax[1, 2].plot(time_torch, input_torch[:, 0])
ax[1, 2].set_title("acc")

ax[2, 0].plot(time_torch, input_torch[:, 1])
ax[2, 0].set_title("steer")

ax[2, 1].plot(time_torch, dt_torch)
ax[2, 1].set_title("dt")

fig.legend()
fig.show()
fig.savefig("pose_x.png")
# exit()

common_time = np.arange(0, time_torch[-1], 0.01)
intrep_pose = np.vstack(
    (
        np.interp(common_time, time_torch, pose_torch[:, 0]),
        np.interp(common_time, time_torch, pose_torch[:, 1]),
        np.interp(common_time, time_torch, pose_torch[:, 2]),
    )
).transpose()
cur_idx = 0
st_time = 0
for iter in range(0, num_iters):
    visu.n_iter = iter
    len_traj = len(visu.state_traj[iter])
    visu.UpdateIter(iter, -1)
    visu.traj = visu.state_traj[iter]
    current_goal = {}
    current_goal["Fx_X"] = visu.meas_traj[iter]
    # current_goal["Fx_Y"] = env.get_density_observation(torch.from_numpy(current_goal["Fx_X"]).float())[
    #         0].detach()
    visu.FxUpdate(0, current_goal, 0, 0, 0)
    visu.remove_temp_objects("Fx")
    visu.temp_objects["Fx"] = visu.plot_Fx(visu.f_handle["gp"])
    visu.CxVisuUpdate(
        visu.player_model[iter],
        visu.player_train_pts[iter]["loc"],
        visu.player_train_pts[iter],
        0,
    )
    visu.remove_temp_objects("Cx")
    visu.opti_path = visu.opti_path_list[iter]
    visu.utility_minimizer = visu.utility_minimizer_list[iter]

    visu.temp_objects["Cx"] = visu.plot_safe_GP(visu.f_handle["gp"])
    common_time = np.arange(0, visu.state_traj[iter][int(len_traj / 2)][-1], 0.01)
    intrep_pose = np.vstack(
        (
            np.interp(
                common_time,
                visu.state_traj[iter][: int(len_traj / 2)][:, -1],
                visu.state_traj[iter][: int(len_traj / 2)][:, 0],
            ),
            np.interp(
                common_time,
                visu.state_traj[iter][: int(len_traj / 2)][:, -1],
                visu.state_traj[iter][: int(len_traj / 2)][:, 1],
            ),
            np.interp(
                common_time,
                visu.state_traj[iter][: int(len_traj / 2)][:, -1],
                visu.state_traj[iter][: int(len_traj / 2)][:, 2],
            ),
        )
    ).transpose()
    for i in range(common_time.shape[0]):
        if i % 4 == 0:
            plot_car(intrep_pose[i][0], intrep_pose[i][1], intrep_pose[i][2], l)
            visu.writer_gp.grab_frame()
    # while (common_time[cur_idx]) < (st_time + visu.state_traj[iter][int(len(visu.state_traj[iter-1])/2)][-1]):
    #     print((common_time[cur_idx]), " ", st_time, " " , visu.state_traj[iter][int(len(visu.state_traj[iter])/2)][-1])
    #     cur_idx += 1
    #     plot_car(intrep_pose[cur_idx][0], intrep_pose[cur_idx][1], intrep_pose[cur_idx][2],l)
    #     visu.writer_gp.grab_frame()
    # st_time = st_time + visu.state_traj[iter][int(len(visu.state_traj[iter])/2)][-1]
    # visu.writer_dyn.grab_frame()
    # visu.f_handle["dyn"].savefig("temp1D.png")
    visu.f_handle["gp"].savefig("temp in prog2.png")
    # visu.plot_safe_GP(i, data_load_path + str(traj_iter) + "/")


# # plot the data
# visu.plot_safe_GP(f_handle=)

# # save the animations
# for i in range():
#     visu.save_animation(i)

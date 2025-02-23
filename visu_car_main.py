import argparse
import errno
import os
import warnings

import matplotlib.pyplot as plt
import yaml
import dill as pickle
from src.visu import Visualizer

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

workspace = "sampling-gpmpc"

parser = argparse.ArgumentParser(description="A foo that bars")
# parser.add_argument("-param", default="params_pendulum1D_samples")  # params
# parser.add_argument("-param", default="params_car_samples")  # params
parser.add_argument("-param", default="params_car_residual")  # params

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

a_file = open(save_path + str(traj_iter) + "/data.pkl", "rb")
data_dict = pickle.load(a_file)
state_traj = data_dict["state_traj"]
input_traj = data_dict["input_traj"]
mean_state_traj = data_dict["mean_state_traj"]
true_state_traj = data_dict["true_state_traj"]
physical_state_traj = data_dict["physical_state_traj"]
tilde_eps_list, ci_list = None, None
if "tilde_eps_list" in data_dict:
    tilde_eps_list = data_dict["tilde_eps_list"]
    ci_list = data_dict["ci_list"]
a_file.close()

params["visu"]["show"] = True
visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=None)
visu.tilde_eps_list = tilde_eps_list
visu.ci_list = ci_list
# agent = Agent(params)
# visu.extract_data()

# physical_state_traj = np.vstack(visu.physical_state_traj)
# plt.plot(physical_state_traj[:,0], physical_state_traj[:,1])
# plt.show()
# load data)
nx = params["agent"]["dim"]["nx"]
ax = visu.f_handle["gp"].axes[0]
(l,) = ax.plot([], [], "tab:orange")
for i in range(0, len(state_traj)):
    mean_state_traj = state_traj[i][:, :nx]
    visu.record_out(
        physical_state_traj[i],
        state_traj[i],
        input_traj[i],
        true_state_traj[i],
        mean_state_traj,
    )
    # print(true_state_traj[i])
    temp_obj = visu.plot_receding_traj()
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

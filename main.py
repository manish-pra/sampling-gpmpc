import argparse
import errno
import os
import warnings

import matplotlib.pyplot as plt
import yaml

from src.DEMPC import DEMPC
from src.visu import Visualizer
from src.agent import Agent
from src.environments.pendulum import Pendulum as pendulum
from src.environments.car_model_residual import CarKinematicsModel as bicycle_Bdx
from src.environments.car_model import CarKinematicsModel as bicycle
from src.environments.pendulum1D import Pendulum as Pendulum1D
import numpy as np
import torch

# torch.cuda.set_per_process_memory_fraction(0.99, "cuda:0")

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

workspace = "sampling-gpmpc"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_pendulum1D_samples")  # params
# parser.add_argument("-param", default="params_car_residual")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=43)  # initialized at origin
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
    + "/"
)

# torch.cuda.memory._record_memory_history(enabled=True)

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

env_model = globals()[params["env"]["dynamics"]](params)
# if params["env"]["dynamics"] == "pendulum":
#     env_model = Pendulum(params)
# elif params["env"]["dynamics"] == "bicycle":
#     env_model = CarKinematicsModel(params)
# else:
#     raise ValueError("Unknown dynamics model")

agent = Agent(params, env_model)
visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=agent)

# 4) Set the initial state
agent.update_current_state(np.array(params["env"]["start"]))


de_mpc = DEMPC(params, visu, agent)
de_mpc.dempc_main()
print(np.average(visu.solver_time[1:]), np.std(visu.solver_time[1:]))
visu.save_data()
# dict_file = torch.cuda.memory._snapshot()
# pickle.dump(dict_file, open(save_path + str(traj_iter) + "/memory_snapshot_1.pickle", "wb"))
exit()

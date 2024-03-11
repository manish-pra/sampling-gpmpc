import argparse
import errno
import os
import warnings

import matplotlib.pyplot as plt
import yaml

from src.environment import ContiWorld
from src.DEMPC import DEMPC
from src.visu import Visualizer
from src.agent import Agent
import numpy as np
import torch
import pickle

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

workspace = "safe_gpmpc"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=39)  # initialized at origin
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

# 3) Setup the environment. This class defines different environments
env = ContiWorld(
    env_params=params["env"],
    common_params=params["common"],
    visu_params=params["visu"],
    env_dir=env_load_path,
    params=params,
)

print(args)
if args.i != -1:
    traj_iter = args.i

if not os.path.exists(save_path + str(traj_iter)):
    os.makedirs(save_path + str(traj_iter))

agent = Agent(params)
visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=agent)

# 4) Set the initial state
agent.update_current_state(np.array(params["env"]["start"]))


de_mpc = DEMPC(params, env, visu, agent)
de_mpc.dempc_main()
visu.save_data()
# dict_file = torch.cuda.memory._snapshot()
# pickle.dump(dict_file, open(save_path + str(traj_iter) + "/memory_snapshot_1.pickle", "wb"))
exit()

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
parser.add_argument("-param", default="params_car_racing")  # params
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



params["visu"]["show"] = True
env_model = globals()[params["env"]["dynamics"]](params)

agent = Agent(params, env_model)
visu = Visualizer(params=params, path=save_path + str(traj_iter), agent=agent)


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

import matplotlib.image as mpimg
# car_img = mpimg.imread("car_shorten.png") 
car_img = mpimg.imread("car_icon.png") 


lx = 2.843 
ly = lx/2.1
sx = 0.3
sy = 8.4
mx = 0.5
my = 2
plt.imshow(car_img, extent=[sx-lx-mx, sx, sy-ly-my, sy], zorder=10)  # (xmin, xmax, ymin, ymax)


plt.legend(fontsize='small', labelspacing=0.2, handlelength=1)
plt.grid(False) 
ax = visu.f_handle["gp"].axes
adapt_figure_size_from_axes(ax)
plt.ylabel(r"$x[m]$")
plt.xlabel(r"$y[m]$")
# plt.xlim(2.0, 3.7)
# plt.ylim(-6.0, 6.0)
# plt.xlim(-6.0, 6.0)
plt.tight_layout(pad=0.0)
visu.f_handle["gp"].savefig("car_setup.pdf", dpi=600,transparent=True,format="pdf", bbox_inches="tight")
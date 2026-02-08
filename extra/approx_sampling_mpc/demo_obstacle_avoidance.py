#!/usr/bin/env python3
"""
Drone Obstacle Avoidance Demonstration
This script demonstrates a drone navigating from start to goal while avoiding circular obstacles.
"""

import argparse
import errno
import os
import warnings
import matplotlib.pyplot as plt
import yaml
import numpy as np
import torch

from src.DEMPC import DEMPC
from src.visu import Visualizer
from src.agent import Agent
from src.environments.drone import Drone as drone

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parse arguments
parser = argparse.ArgumentParser(description="Drone Obstacle Avoidance Demo")
parser.add_argument("-param", default="params_drone_obstacles_approx")
parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=1)
args = parser.parse_args()

# Load configuration
with open(os.path.join(SCRIPT_DIR, "params", args.param + ".yaml")) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

params["env"]["i"] = args.i
params["env"]["name"] = args.env

print("="*60)
print("DRONE OBSTACLE AVOIDANCE DEMONSTRATION")
print("="*60)
print(f"Start Position: {params['env']['start'][:2]}")
print(f"Goal Position: {params['env']['goal_state']}")
print(f"Number of Obstacles: {len(params['env']['obstacles'])}")
for obs_name, obs_params in params['env']['obstacles'].items():
    print(f"  {obs_name}: center=({obs_params[0]}, {obs_params[1]}), radius={obs_params[2]}")
print("="*60)

# Set random seed if specified
if params["experiment"]["rnd_seed"]["use"]:
    torch.manual_seed(params["experiment"]["rnd_seed"]["value"])

# Setup paths
exp_name = params["experiment"]["name"]
env_load_path = os.path.join(
    SCRIPT_DIR,
    "experiments",
    params["experiment"]["folder"],
    f"env_{args.env}"
)

save_path = os.path.join(env_load_path, args.param)

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

traj_iter = args.i
traj_path = os.path.join(save_path, str(traj_iter))
if not os.path.exists(traj_path):
    os.makedirs(traj_path)

# Initialize environment and agent
env_model = drone(params)
agent = Agent(params, env_model)

print(f"Training data shape: {agent.Dyn_gp_X_train.shape}, {agent.Dyn_gp_Y_train.shape}")

# Initialize visualizer
visu = Visualizer(params=params, path=traj_path, agent=agent)

# Set initial state
agent.update_current_state(np.array(params["env"]["start"]))

# Run MPC
print("\nStarting MPC controller...")
print(f"Using {params['agent']['num_dyn_samples']} samples for predictions")
de_mpc = DEMPC(params, visu, agent)
de_mpc.dempc_main()

# Print statistics
print("\n" + "="*60)
print("SIMULATION COMPLETE")
print("="*60)
print(f"Average solve time: {np.average(visu.solver_time[1:]):.4f} s")
print(f"Std solve time: {np.std(visu.solver_time[1:]):.4f} s")
print(f"Results saved to: {traj_path}")
print("="*60)

# Save data
visu.save_data()

# Generate animation
print("\nGenerating animation...")
import subprocess
data_file = os.path.join(traj_path, "data_obstacles.pkl")
params_file = os.path.join(SCRIPT_DIR, "params", args.param + ".yaml")
output_dir = traj_path
subprocess.run(["python3", os.path.join(SCRIPT_DIR, "create_animation.py"), data_file, params_file, output_dir])
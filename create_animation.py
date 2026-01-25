#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import yaml
import sys
import os

# Get command line arguments for paths
if len(sys.argv) > 1:
    data_path = sys.argv[1]
    params_path = sys.argv[2]
    output_dir = sys.argv[3]
else:
    # Default paths
    data_path = 'experiments/drone/env_0/params_drone_obstacles/1/data_obstacles.pkl'
    params_path = 'params/params_drone_obstacles.yaml'
    output_dir = 'experiments/drone/env_0/params_drone_obstacles/1'

# Load data
with open(data_path, 'rb') as f:
    data = pickle.load(f)

with open(params_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Extract trajectory
traj = np.array(data['physical_state_traj'])[:, :2]  # px, py
state_samples = np.array(data['state_traj'])  # (time, horizon, state_dim)
reference_traj = np.array(data['reference_cost'])  # Reference trajectory
nx = 6

# Detect number of samples from state_samples shape
if len(state_samples.shape) == 3:
    # state_samples shape: (time, horizon, state_dim)
    # state_dim could be nx*num_samples or just nx
    state_dim = state_samples.shape[2]
    if state_dim % nx == 0:
        num_samples = state_dim // nx
    else:
        num_samples = 1
else:
    num_samples = 1

print(f"Detected {num_samples} samples in data")

# Generate full heart reference path for visualization
s = np.linspace(0, 4 * np.pi, 500)
x_ref = 8 * np.sin(s)**3 / 1.5 + 1
y_ref = (10 * np.cos(s) - 5 * np.cos(2*s) - 2 * np.cos(3*s) - np.cos(4*s))/2 - 1 + 1
ref_path = np.vstack([x_ref, y_ref]).T

# Setup figure
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X Position (m)', fontsize=14)
ax.set_ylabel('Y Position (m)', fontsize=14)
ax.set_title('Drone Obstacle Avoidance with Prediction Samples', fontsize=16, fontweight='bold')

# Plot obstacles
for obs_name, obs_params in params['env']['obstacles'].items():
    cx, cy, r = obs_params[0], obs_params[1], obs_params[2]
    circle = Circle((cx, cy), r, color='red', alpha=0.3)
    ax.add_patch(circle)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), 'r-', linewidth=2)

# Plot start and goal
start = params['env']['start'][:2]
goal = params['env']['goal_state']
ax.plot(start[0], start[1], 'go', markersize=15, label='Start', zorder=10)

# Plot box constraints
x_min = params['optimizer']['x_min'][0]
x_max = params['optimizer']['x_max'][0]
y_min = params['optimizer']['x_min'][1]
y_max = params['optimizer']['x_max'][1]
ax.plot([x_min, x_max, x_max, x_min, x_min], 
        [y_min, y_min, y_max, y_max, y_min], 
        'k--', linewidth=2, alpha=0.7, label='Box Constraints')

# Plot reference trajectory (figure-8)
ax.plot(ref_path[:, 0], ref_path[:, 1], 'g--', alpha=0.4, linewidth=2, label='Reference Path')

# Plot full trajectory
ax.plot(traj[:, 0], traj[:, 1], 'b--', alpha=0.3, linewidth=1, label='Actual Path')

# Animated elements
drone, = ax.plot([], [], 'bo', markersize=12, zorder=10, label='Drone')
trail, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
# Create prediction lines based on detected number of samples
prediction_lines = [ax.plot([], [], 'c-', alpha=0.15, linewidth=0.5)[0] for _ in range(max(1, num_samples))]
prediction_points = [ax.plot([], [], 'c.', alpha=0.3, markersize=2)[0] for _ in range(max(1, num_samples))]

ax.legend(loc='upper left', fontsize=12)

def init():
    drone.set_data([], [])
    trail.set_data([], [])
    for line in prediction_lines:
        line.set_data([], [])
    for pts in prediction_points:
        pts.set_data([], [])
    return [drone, trail] + prediction_lines + prediction_points

def animate(i):
    # Update drone position
    drone.set_data([traj[i, 0]], [traj[i, 1]])
    # Update trail
    trail.set_data(traj[:i+1, 0], traj[:i+1, 1])
    
    # Update prediction samples
    if i < len(state_samples) and num_samples > 1:
        for s in range(num_samples):
            # Extract prediction for this sample
            pred_x = state_samples[i, :, s*nx + 0]  # px for sample s
            pred_y = state_samples[i, :, s*nx + 1]  # py for sample s
            
            # Plot prediction trajectory
            prediction_lines[s].set_data(pred_x, pred_y)
            prediction_points[s].set_data(pred_x, pred_y)
    elif i < len(state_samples) and num_samples == 1:
        # Single sample case - just show the mean prediction
        pred_x = state_samples[i, :, 0]  # px
        pred_y = state_samples[i, :, 1]  # py
        prediction_lines[0].set_data(pred_x, pred_y)
        prediction_points[0].set_data(pred_x, pred_y)
    
    return [drone, trail] + prediction_lines + prediction_points

# Create animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(traj), interval=50, blit=True)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save animation
print(f"Saving animation to {output_dir}...")
mp4_path = os.path.join(output_dir, 'obstacle_avoidance_with_predictions.mp4')
anim.save(mp4_path, writer='ffmpeg', fps=20, dpi=150)
print(f"Saved: {mp4_path}")

plt.close()
print("Done!")

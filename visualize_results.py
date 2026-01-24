#!/usr/bin/env python3
"""
Visualization script for obstacle avoidance results
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import yaml

def plot_obstacle_avoidance_results(data_file, params_file):
    """Plot the trajectory and obstacles"""
    
    # Load data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    with open(params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # Extract trajectory - use physical_state_traj
    X_traj = np.array(data['physical_state_traj'])  # Shape: (N, nx)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Trajectory with obstacles
    ax = axes[0]
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Drone Trajectory with Obstacles', fontsize=14, fontweight='bold')
    
    # Plot obstacles
    if 'obstacles' in params['env']:
        for obs_name, obs_params in params['env']['obstacles'].items():
            cx, cy, r = obs_params[0], obs_params[1], obs_params[2]
            circle = plt.Circle((cx, cy), r, color='red', alpha=0.3, label='Obstacle')
            ax.add_patch(circle)
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), 'r-', linewidth=2)
    
    # Plot trajectory
    ax.plot(X_traj[:, 0], X_traj[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax.plot(X_traj[0, 0], X_traj[0, 1], 'go', markersize=12, label='Start')
    ax.plot(X_traj[-1, 0], X_traj[-1, 1], 'r*', markersize=15, label='End')
    
    # Plot goal
    goal = params['env']['goal_state']
    ax.plot(goal[0], goal[1], 'b*', markersize=15, label='Goal')
    
    # Plot box constraints
    x_min, x_max = params['optimizer']['x_min'][0], params['optimizer']['x_max'][0]
    y_min, y_max = params['optimizer']['x_min'][1], params['optimizer']['x_max'][1]
    ax.plot([x_min, x_max, x_max, x_min, x_min], 
            [y_min, y_min, y_max, y_max, y_min], 
            'k--', alpha=0.5, linewidth=1.5, label='Bounds')
    
    ax.legend(loc='best')
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    
    # Plot 2: Distance to obstacles over time
    ax = axes[1]
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Distance to Obstacle (m)', fontsize=12)
    ax.set_title('Safety Margins', fontsize=14, fontweight='bold')
    
    if 'obstacles' in params['env']:
        for obs_name, obs_params in params['env']['obstacles'].items():
            cx, cy, r = obs_params[0], obs_params[1], obs_params[2]
            distances = np.sqrt((X_traj[:, 0] - cx)**2 + (X_traj[:, 1] - cy)**2) - r
            ax.plot(distances, linewidth=2, label=f'{obs_name}')
        
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Collision')
        ax.legend(loc='best')
    
    plt.tight_layout()
    return fig

def plot_control_inputs(data_file):
    """Plot control inputs over time"""
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    U_traj = np.array(data['input_traj'])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Control input 1
    axes[0].plot(U_traj[:, 0], 'b-', linewidth=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('u1 (N)', fontsize=12)
    axes[0].set_title('Control Inputs', fontsize=14, fontweight='bold')
    
    # Control input 2
    axes[1].plot(U_traj[:, 1], 'r-', linewidth=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('u2 (N)', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_velocities(data_file):
    """Plot velocity profiles"""
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    X_traj = np.array(data['physical_state_traj'])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Linear velocities
    axes[0].plot(X_traj[:, 3], 'b-', linewidth=2, label='vx')
    axes[0].plot(X_traj[:, 4], 'r-', linewidth=2, label='vy')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[0].set_title('Velocity Profiles', fontsize=14, fontweight='bold')
    axes[0].legend()
    
    # Angular velocity
    axes[1].plot(X_traj[:, 5], 'g-', linewidth=2, label='phi_dot')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize obstacle avoidance results")
    parser.add_argument("--data", default="experiments/drone/env_0/params_drone_obstacles/1/data_obstacles.pkl",
                       help="Path to data file")
    parser.add_argument("--params", default="params/params_drone_obstacles.yaml",
                       help="Path to params file")
    parser.add_argument("--save", action="store_true", help="Save figures")
    args = parser.parse_args()
    
    print("Loading data from:", args.data)
    print("Loading params from:", args.params)
    
    try:
        # Plot trajectory
        fig1 = plot_obstacle_avoidance_results(args.data, args.params)
        if args.save:
            fig1.savefig('obstacle_avoidance_trajectory.png', dpi=300, bbox_inches='tight')
            print("Saved: obstacle_avoidance_trajectory.png")
        
        # Plot controls
        fig2 = plot_control_inputs(args.data)
        if args.save:
            fig2.savefig('obstacle_avoidance_controls.png', dpi=300, bbox_inches='tight')
            print("Saved: obstacle_avoidance_controls.png")
        
        # Plot velocities
        fig3 = plot_velocities(args.data)
        if args.save:
            fig3.savefig('obstacle_avoidance_velocities.png', dpi=300, bbox_inches='tight')
            print("Saved: obstacle_avoidance_velocities.png")
        
        if not args.save:
            plt.show()
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the simulation first: python demo_obstacle_avoidance.py")

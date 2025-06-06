# Simple Car Model with GP-MPC

This directory contains implementations of a simple car model controlled using Gaussian Process Model Predictive Control (GP-MPC). The implementation includes both offline and online learning approaches.

## Overview

The system implements a car control system using:
- A simple bicycle model for car dynamics
- Gaussian Process (GP) for modeling uncertainties
- Model Predictive Control (MPC) for trajectory optimization
- Random Fourier Features (RFF) for efficient GP approximation

## Key Components

### Car Dynamics
- State space: [x, y, θ, v] (position, heading angle, velocity)
- Control inputs: [δ, a] (steering angle, acceleration)

### MPC Implementation Details
- **Objective Function**: 
  - State cost: Quadratic cost on state error with diagonal Q matrix [2, 36, 0.07, 0.005]
  - Control cost: Quadratic cost on control inputs with diagonal R matrix [2, 2]
  - Slack cost: Quadratic penalty on obstacle avoidance slack variables (λ = 1e4)
  - Cost is averaged over all GP function samples

- **Constraints**:
  - State bounds:
    - x: [0, 100] 
    - y: [0, 6] 
    - θ: [-1.14, 1.14] 
    - v: [-1, 15] 
  - Control bounds:
    - Steering angle δ: [-0.6, 0.6] 
    - Acceleration a: [-2, 2] m/s²
  - Obstacle avoidance: Elliptical constraints with slack variables
  - Initial state constraint
  - GP dynamics constraints for each sample

- **Solver**:
  - Uses IPOPT (Interior Point Optimizer)
  - Maximum iterations: 1000

### GP-MPC Implementation
Three variants are provided:
1. **exact MPC, no learning** (`exactMPC.py`)
   - Uses exact bicycle model dynamics without any learning

2. **Offline Learning** (`offline_car_simple.py`)
   - Trains GP model once using initial data

3. **Online Learning** (`online_car_simple.py`)
   - Trains GP model once using initial data + continuously updates GP model at each MPC step

### Running the Simulation

```python
# For exact MPC
python exactMPC.py

# For offline learning
python offline_car_simple.py

# For online learning
python online_car_simple.py
```

### Key Parameters

- `num_features`: Number of RFF features for GP approximation
- `num_samples`: Number of function samples
- `horizon`: MPC prediction horizon
- `num_trajectories`: Number of trajectories used for training data generation 
- `trajectory_length`: Number of steps in each training trajectory
- `dt`: Time step for discretization

### Output
The simulation provides:
- Vehicle trajectory visualization
- Control input plots
- Computation time analysis
- State plots 


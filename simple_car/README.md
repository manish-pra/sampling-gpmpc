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

### GP-MPC Implementation
Two variants are provided:

1. **Offline Learning** (`offline_car_simple.py`)
   - Trains GP model once using initial data

2. **Online Learning** (`online_car_simple.py`)
   - Trains GP model once using initial data + continuously updates GP model at each MPC step 



### Running the Simulation

```python
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


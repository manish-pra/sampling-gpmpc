# Approximate Sampling-based MPC

## Overview

This folder contains code for the approximate sampling-based Gaussian Process Model Predictive Control (GP-MPC) approach. Instead of solving a large-scale optimization problem with multiple sampled dynamics models, this method uses a single nominal dynamics model with conservatively tightened constraints.

## Key Features

- **Computational Efficiency**: Solves a single nominal MPC problem instead of handling multiple samples in the optimization
- **Conservative Constraint Tightening**: Ensures feasibility across all sampled dynamics by tightening state and obstacle constraints
- **Sample-based Tightening Computation**: Propagates sampled dynamics to compute appropriate constraint tightenings

## Method

1. **GP Sampling**: Sample multiple dynamics models from the learned GP posterior
2. **Propagate Samples**: Forward propagate all sampled dynamics using the nominal control sequence
3. **Compute Tightening**: Calculate maximum deviation across sampled trajectories $x^n_k$ and the mean predictions $x_k^{\mu}$ to determine constraint tightening as shown below (illustrated for box state constraints):

$$
\mathcal{X}_k^{\mathrm{tight}} = \mathcal{X} \ominus \Delta_k
$$

$$
\Delta_k = \max_{n=1,\dots,N} \left| x_k^{n} - x_k^{\mu} \right|
$$

4. **Solve Nominal MPC**: Solve the nominal MPC problem (mean dynamics model) with tightened constraints
5. **Iterate within SQP**: The above steps (1-4) are iterated in each SQP iteration



## Files

- `demo_obstacle_avoidance.py`: Main demonstration script for drone obstacle avoidance
- `create_animation.py`: Generates visualization videos showing all sample predictions

## Usage

Run the following command from within the folder `extra/approx_sampling_mpc`:

```bash
python demo_obstacle_avoidance.py -param params_drone_obstacles_approx
```
<p align="center">
  <video src="https://github.com/user-attachments/assets/f45fc828-ba8e-421f-8b87-5bb6491bc29b" width="600" controls></video>
</p>

## Parameters

Key parameters in `params_drone_obstacles_approx.yaml`:
- `num_dyn_samples`: Number of samples used in the MPC optimization (typically 1 for approximate method)
- `num_samples_tightening`: Number of samples used to compute constraint tightening (e.g., 50)
- `use_approx_tightening`: Enable approximate tightening computation

## Results

The method generates:
- Trajectory data with sample predictions
- Visualization videos showing uncertainty propagation
- Constraint tightening values per stage


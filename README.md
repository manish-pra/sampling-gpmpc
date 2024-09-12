
![iterative_gp_conditioning](https://github.com/user-attachments/assets/8c0ff769-a9e5-42f5-a49f-d1a9093c5323)

# Sampling-based Gaussian Process Model Predictive Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code for the paper "Towards safe and tractable Gaussian process-based MPC:
Efficient sampling within a sequential quadratic programming framework", accepted for publication at the 63rd IEEE Conference on Decision and Control (CDC 2024).

The code is licensed under the MIT license.

## Installation

1. Create new `workspace` folder. This is used as a base directory for the following steps.
2. Clone the main repository
    ```bash
        git clone https://github.com/manish-pra/sampling-gpmpc.git
    ```
3. Clone auxiliary repositories
    ```bash
        git clone https://github.com/manish-pra/safe-exploration-koller.git
        git clone https://github.com/befelix/plotting_utilities.git
    ```
3. Install dependencies
    1. Download and install [acados](https://docs.acados.org/installation/).
    2. Install Python requirements
        ```bash
            cd sampling-gpmpc
            pip install -r requirements.txt
        ```

## Usage

1. To run the code, run the following Python script from your base directory:

    ```
        python sampling-gpmpc/main.py -i $i -param $param_file
    ```
    where,
    ```
        $param_file: Name of the param file, either "params_pendulum" or "params_car"
        $i         : Experiment number, e.g. "1"
    ```

1. For visualizations/videos use the following script once the experiment is completed:

    ```
        python sampling-gpmpc/visu_main.py -i $i -param $param_file
    ```
    This will generate a video of the simulation in the folder corresponding to the `$param_file` and the instance `$i`.

## CDC 2024 Experiments

### Pendulum example

#### Instructions

To simulate different reachable set approximations, run the following scripts (here we choose `$i=1`):

1. Set experiment index and params file:
    ```bash
        param_file=params_pendulum
        i=1
    ```
2. Generate control input sequence and sampling-based predictions (`data.pkl`) by running main script:
    ```bash
        python sampling-gpmpc/main.py -i $i -param $param_file
    ```
3. Generate linearization-based predictions (`cautious_ellipse_data.pkl`, `cautious_ellipse_center_data.pkl`), robust tube-based predictions (`koller_ellipse_data.pkl`, `koller_ellipse_center_data.pkl`) and true reachable set (`X_traj_list_0.pkl`, ..., `X_traj_list_<num_files>.pkl`):
    ```bash
        python sampling-gpmpc/benchmarking/linearization_based_predictions.py -i $i -param $param_file
        python sampling-gpmpc/benchmarking/robust_tube_based_GPMPC_koller.py -i $i -param $param_file
        python sampling-gpmpc/benchmarking/simulate_true_reachable_set.py -i $i -param $param_file
    ```

    Important script parameters:
    - `simulate_true_reachable_set.py`: 
        - number of samples per repeat is given by `num_samples` in `params_pendulum.yaml`
        - number of repeats can be set by `num_repeat` in script file

4. Run plotting script:
    ```bash
        python sampling-gpmpc/extra/cdc_plt.py -i $i -param $param_file
    ```

#### Result

The final result should look similar to this:

![Sampling-based-predictions-pendulum](https://github.com/user-attachments/assets/d52d4d3f-1ecd-4f78-8cbb-864297662579)


### Car example

#### Instructions

1. Set experiment index and params file:
    ```bash
        param_file=params_car
        i=1
    ```
2. Run closed-loop simulation by running main script:
    ```bash
        python sampling-gpmpc/main.py -i $i -param $param_file
    ```
3. Generate video by running visualizer script:
    ```bash
        python sampling-gpmpc/visu_main.py -i $i -param $param_file
    ```

#### Result

The final result should look similar to this:

![CDC_car_video](https://github.com/user-attachments/assets/de8b05e0-bf04-4bf4-9dbc-d51210cc9bec)

## Citing us

If you use results from the paper or the code, please cite the following paper:

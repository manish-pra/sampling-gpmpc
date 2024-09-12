# Sampling-based Gaussian Process Model Predictive Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Sampling-based-predictions-pendulum](https://github.com/user-attachments/assets/d52d4d3f-1ecd-4f78-8cbb-864297662579)

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
    ```
3. Install dependencies
    1. Download and install [acados](https://docs.acados.org/installation/).
    2. Install Python requirements
        ```bash
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

![CDC_car_video](https://github.com/user-attachments/assets/de8b05e0-bf04-4bf4-9dbc-d51210cc9bec)

## Citing us

If you use results from the paper or the code, please cite the following paper:

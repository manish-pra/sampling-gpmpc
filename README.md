# Sampling based GPMPC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code for the "Safe Gaussian process-based MPC using efficient sampling within a sequential quadratic programming framework" work.

## Getting started

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


1. To run the code, use the following command

    ```
    python3 safe_gpmpc/main.py -i $i -param $param_i
    ```
    where,
    ```
    $param_i: Name of the param file (see params folder) to pick an algo and the env type 
    $i      : An integer to run multiple instances
    ```

1. For visualizations/videos use the following script once your experiment is completed

    ```
    python3 safe_gpmpc/visu_main.py -i $i -param $param_i
    ```
    This will generate a video of the simulation in the folder cooresponding to the param file and the instance i.
<!-- 1. To run the code, use the following command

    ```
    python3 safe_gpmpc/main.py -i $i -env $env_i -param $param_i
    ```
    where,

    ```
    $param_i: Name of the param file (see params folder) to pick an algo and the env type 
    $env_i  : An integer to pick an instance of the environment
    $i      : An integer to run multiple instances
    ```

    E.g., the following command runs SageMPC on the cluttered environment with env_0 and i=2 instance

    ``` 
    python3 safe_gpmpc/main.py -i 2 -env 0 -param "params_cluttered_car"
    ``` -->

<!-- 1. For visualizations/videos use the following script once your experiment is completed

    ```
    python3 safe_gpmpc/video.py
    ``` -->


![CDC_car_video](https://github.com/user-attachments/assets/de8b05e0-bf04-4bf4-9dbc-d51210cc9bec)



    

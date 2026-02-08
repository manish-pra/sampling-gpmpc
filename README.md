# Safe Guaranteed Dynamics Exploration with Probabilistic Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code for the paper "Safe Guaranteed Dynamics Exploration with Probabilistic Models".

The code is licensed under the MIT license.

## Installation

1. Create new `workspace` folder. This is used as a base directory for the following steps.
2. Place the main repository `sampling-gpmpc`
3. Install dependencies
    1. Download and install [acados](https://docs.acados.org/installation/) (does not need to be in `workspace`).
    2. Install [acados Python interface](https://docs.acados.org/python_interface/index.html).
    3. Install Python requirements
        ```bash
            cd sage-dynx
            pip install -r requirements.txt
        ```

## Usage

1. To run the code, run the following Python script from your base directory:

    ```
        python sage-dynx/main.py -param params_drone_sagedynx -i 3
        python sage-dynx/main.py -param  -i $i

    ```
    where,
    ```
        $param_file: Name of the param file, e.g. "params_drone_sagedynx" or "params_drone_opt"
        $i         : Experiment number, e.g. "1"
    ```
    see params folder for other param file names.

1. For visualizations/videos use the following script once the experiment is completed:

    ```
        python sage-dynx/visu_main.py -param params_drone_sagedynx -i 3
        python sage-dynx/visu_main.py -i $i -param $param_file
    ```
    This will generate a video of the simulation in the folder corresponding to the `$param_file` and the instance `$i`.


1. For reference experiment videos, please refer to the folder "experiment/drone/env_0" and the files named "video_gp.mp4" contained within.

<p align="center">
  <video src="https://github.com/user-attachments/assets/f45fc828-ba8e-421f-8b87-5bb6491bc29b" width="600" controls></video>
</p>


1. The above commands will automatically create subfolders within the experiment directory, named according to the parameter file, environment number, and experiment number. Running the visualization script will generate videos for these experiments in the corresponding folders.
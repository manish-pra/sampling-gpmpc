import argparse
import errno
import os, sys
import warnings

import matplotlib.pyplot as plt
import yaml

import dill as pickle
import numpy as np
import torch
import numpy.linalg as nLa
import gpytorch

# NOTE: this file needs to be called from outside the root directory of the project, e.g.:
# python sampling-gpmpc/benchmarking/linearization_based_predictions.py
workspace = "sampling-gpmpc"
sys.path.append(workspace)

from src.DEMPC import DEMPC
from src.visu import Visualizer
from src.agent import Agent
from src.environments.pendulum import Pendulum

from extra.zoro_code import generate_gp_funs
from src.GP_model import BatchMultitaskGPModelWithDerivatives_fromParams


def P_propagation(P, A, B, W):
    #  P_i+1 = A P A^T +  B*W*B^T
    return A @ P @ A.T + B @ W @ B.T


def extract_function_value_for_first_sample(y):
    return y[0, :, :, 0].T


def mean_fun_sum(y):
    with gpytorch.settings.fast_pred_var():
        return extract_function_value_for_first_sample(gp_model(y).mean).sum(dim=[0, 1])


if __name__ == "__main__":

    # get GP model from agent
    warnings.filterwarnings("ignore")
    plt.rcParams["figure.figsize"] = [12, 6]

    parser = argparse.ArgumentParser(description="A foo that bars")
    parser.add_argument("-param", default="params_pendulum")  # params

    parser.add_argument("-env", type=int, default=0)
    parser.add_argument("-i", type=int, default=999)  # initialized at origin
    args = parser.parse_args()

    # 1) Load the config file
    with open(workspace + "/params/" + args.param + ".yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    params["env"]["i"] = args.i
    params["env"]["name"] = args.env
    params["common"]["use_cuda"] = False
    print(params)

    # random seed
    if params["experiment"]["rnd_seed"]["use"]:
        torch.manual_seed(params["experiment"]["rnd_seed"]["value"])

    # 2) Set the path and copy params from file
    exp_name = params["experiment"]["name"]
    env_load_path = (
        workspace
        + "/experiments/"
        + params["experiment"]["folder"]
        + "/env_"
        + str(args.env)
        + "/"
    )

    save_path = env_load_path + "/" + args.param + "/"

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    print(args)
    if args.i != -1:
        traj_iter = args.i

    if not os.path.exists(save_path + str(traj_iter)):
        os.makedirs(save_path + str(traj_iter))

    save_path_iter = save_path + str(traj_iter)

    env_model = Pendulum(params)
    # TODO: abstract data generation from agent and just call the data generation function here
    agent = Agent(params, env_model)
    agent.train_hallucinated_dynGP(0)

    gp_model = agent.model_i
    gp_model.eval()

    pkl_file = open(save_path_iter + "/data.pkl", "rb")
    data_dict = pickle.load(pkl_file)
    state_traj = data_dict["state_traj"]
    input_traj = data_dict["input_traj"]
    pkl_file.close()
    x0 = state_traj[0][0, 0:2]
    U = input_traj[0]

    nx = params["agent"]["dim"]["nx"]
    nu = params["agent"]["dim"]["nu"]
    ny = params["agent"]["dim"]["ny"]
    n_inp = nx + nu

    # get sensitivities fun
    gp_sensitivities = generate_gp_funs(gp_model, B=None)

    # roll out the GP model
    y_current = torch.tile(
        torch.Tensor(np.hstack((x0, U[0])).reshape((1, n_inp))), dims=(1, ny, 1, 1)
    )

    N = params["optimizer"]["H"]

    x_mean = torch.zeros((N + 1, nx))
    x_mean[0, :] = torch.Tensor(x0)
    x_covar = torch.zeros((N + 1, nx, nx))
    x_covar[0, :, :] = torch.zeros((nx, nx))

    ellipse_list = []
    ellipse_center_list = []

    for i in range(N):

        inp_current = torch.tile(
            torch.Tensor(np.hstack((x_mean[i, :], U[i])).reshape((1, n_inp))),
            dims=(1, ny, 1, 1),
        )

        inp_current_autograd = torch.autograd.Variable(inp_current, requires_grad=True)

        # DERIVATIVE
        mean_dy = torch.autograd.functional.jacobian(mean_fun_sum, inp_current_autograd)

        with torch.no_grad(), gpytorch.settings.observation_nan_policy(
            "mask"
        ), gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ), gpytorch.settings.cholesky_jitter(
            float_value=agent.params["agent"]["Dyn_gp_jitter"],
            double_value=agent.params["agent"]["Dyn_gp_jitter"],
            half_value=agent.params["agent"]["Dyn_gp_jitter"],
        ):
            predictions = gp_model(
                inp_current_autograd
            )  # only model (we want to find true function)
            mean = extract_function_value_for_first_sample(predictions.mean)
            variance = extract_function_value_for_first_sample(predictions.variance)

        # dynamics
        x_mean[i + 1, :] = mean[0, :]
        x_covar[i + 1, :, :] = P_propagation(
            x_covar[i, :, :],
            mean_dy[0, :, 0, 0:nx],
            torch.eye(nx),
            torch.diag(variance[0, :]),
        )

        R = nLa.cholesky(x_covar[i + 1, :, :]).T
        # R = np.array(x_covar[i + 1, :, :]).T
        # checks spd inside the function
        t = np.linspace(0, 2 * np.pi, 100)
        z = [np.cos(t), np.sin(t)]
        ellipse = params["agent"]["Dyn_gp_beta"] * R @ z + x_mean[[i + 1], :].numpy().T
        ellipse_list.append(ellipse)
        ellipse_center_list.append(x_mean[[i + 1], :].numpy().T)
        plt.plot(ellipse[0, :], ellipse[1, :])

    with open(os.path.join(save_path_iter, "cautious_ellipse_data.pkl"), "wb") as a_file:
        pickle.dump(ellipse_list, a_file)
    with open(os.path.join(save_path_iter, "cautious_ellipse_center_data.pkl"), "wb") as a_file:
        pickle.dump(ellipse_center_list, a_file)

    # plt.show()

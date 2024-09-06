import os, sys
import torch
from easydict import EasyDict
import numpy.linalg as nLa
import matplotlib.pyplot as plt
import dill as pickle
import gpytorch
import numpy as np
import argparse
import yaml
from dataclasses import dataclass

# NOTE: this file needs to be called from outside the root directory of the project, e.g.:
# python sampling-gpmpc/benchmarking/linearization_based_predictions.py
workspace = "sampling-gpmpc"
sys.path.append(workspace)

from src.agent import Agent
from src.environments.pendulum import Pendulum

# clone https://github.com/manish-pra/safe-exploration-koller and add it to the path
# Add path to safe-exploration-koller
workspace_safe_exploration = "safe-exploration-koller"
sys.path.append(workspace_safe_exploration)

from safe_exploration.gp_reachability_pytorch import onestep_reachability
from safe_exploration.ssm_cem.gp_ssm_cem import GpCemSSM
from safe_exploration.environments.environments import InvertedPendulum, Environment

# save_path = "/home/manish/work/horrible/safe-exploration_cem/experiments"
# a_file = open(save_path + "/data.pkl", "rb")
# save_path = "/home/manish/work/MPC_Dyn/sampling-gpmpc/experiments/pendulum/env_0/params/401_sampling_mpc/data.pkl"
# save_path = "/home/manish/work/MPC_Dyn/sampling-gpmpc/experiments/pendulum/env_0/params_pendulum/22/data.pkl"
# save_path = "/home/amon/Repositories/sampling-gpmpc/experiments/pendulum/env_0/params_pendulum/22/data.pkl"

def extract_function_value_for_first_sample(y):
    return y[0, :, :, 0]

@dataclass
class mean_and_variance:
    mean: torch.Tensor
    variance: torch.Tensor


class GPModelWithDerivativesProjectedToFunctionValues(torch.nn.Module):
    def __init__(self, gp_model, batch_shape_tile=None):
        super().__init__()
        self.gp_model = gp_model
        if batch_shape_tile is None:
            self.tile_shape = torch.Size([1,1,1,1])
        else:
            self.tile_shape = torch.Size([batch_shape_tile[0], batch_shape_tile[1], 1, 1])

    def forward(self, x):
        # mean_x = extract_function_value_for_first_sample(self.gp_model.mean_module(x))
        # covar_x = extract_function_covariance_for_first_sample(self.gp_model.covar_module(x))
        x_tile = x.tile(self.tile_shape)
        full_dist = self.gp_model(x_tile)
        # mean_x = self.gp_model.mean_module(x_tile)
        # covar_x = self.gp_model.covar_module(x_tile)
        # full_dist = gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        full_dist_mean_proj = extract_function_value_for_first_sample(full_dist.mean)
        full_dist_variance_proj = extract_function_value_for_first_sample(full_dist.variance)
        return mean_and_variance(mean=full_dist_mean_proj, variance=full_dist_variance_proj)

if __name__ == "__main__":

    # get GP model from agent
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

    with open(os.path.join(save_path_iter, "data.pkl"), "rb") as pkl_file:
        data_dict = pickle.load(pkl_file)

    state_traj = data_dict["state_traj"]
    input_traj = torch.Tensor(data_dict["input_traj"])
    x0 = state_traj[0][0, 0:2]

    # only need single prediction
    params["agent"]["num_dyn_samples"] = 1

    env_model = Pendulum(params)
    # TODO: abstract data generation from agent and just call the data generation function here
    agent = Agent(params, env_model)

    # agent.train_hallucinated_dynGP(0, use_model_without_derivatives=True)
    # use derivative data to train GP, then need to project down again
    agent.train_hallucinated_dynGP(0)
    agent.model_i.eval()

    gp_model_orig = agent.model_i
    gp_model_orig.eval()

    gp_model = GPModelWithDerivativesProjectedToFunctionValues(agent.model_i, batch_shape_tile=agent.model_i.batch_shape)
    likelihood = agent.likelihood # NOTE: dimensions wrong, but not used in GpCemSSM

    env = InvertedPendulum(verbosity=0)
    env.n_s = 2
    env.n_u = 1

    conf = {
        "exact_gp_kernel": "rbf",
        "cem_ssm": "exact_gp",
        "exact_gp_training_iterations": 1000,
        "cem_beta_safety": params["agent"]["Dyn_gp_beta"],
        "device": "cpu",
    }

    conf = EasyDict(conf)

    n_s, n_u = env.n_s, env.n_u
    ssm = GpCemSSM(conf, env.n_s, env.n_u, model=gp_model, likelihood=likelihood)
    device = "cpu"

    a = torch.zeros((env.n_s, env.n_s), device=device)
    b = torch.zeros((env.n_s, env.n_u), device=device)
    k_fb_apply = torch.zeros((n_u, n_s))

    ps = torch.tensor([x0]).to(device)
    qs = None
    H = params["optimizer"]["H"]
    ellipse_list = []
    ellipse_center_list = []
    # fig, ax = plt.subplots()
    # iteratively compute it for the next steps
    for i in range(H):
        print(i)
        # TODO: CONTINUE NAN STUFF 
        if torch.any(torch.isnan(ps)) or (qs is not None and torch.any(torch.isnan(qs))):
            ellise = ellipse_list[-1]
            print("Nans in ps or qs")
        else:
            ps, qs, _ = onestep_reachability(
                ps,
                ssm,
                input_traj[0][i].reshape(-1, 1),
                torch.tensor(env.l_mu).to(device),
                torch.tensor(env.l_sigm).to(device),
                q_shape=qs,
                k_fb=k_fb_apply,
                c_safety=conf.cem_beta_safety,
                verbose=0,
                a=a,
                b=b,
            )
            print(ps, qs)

            # x_tile = torch.tile(torch.hstack((ps, input_traj[0][i].reshape(-1, 1))),(1,2,1,1))
            # pred_orig = gp_model_orig(x_tile)

            r = nLa.cholesky(qs).T
            r = r[:, :, 0]
            # checks spd inside the function
            t = np.linspace(0, 2 * np.pi, 100)
            z = [np.cos(t), np.sin(t)]
            ellipse = np.dot(r, z) + ps.numpy().T

        ellipse_list.append(ellipse)
        ellipse_center_list.append(ps.numpy().T)
        # ax.plot(ellipse[0, :], ellipse[1, :])

    # plt.show()

    with open(os.path.join(save_path_iter, "koller_ellipse_data.pkl"), "wb") as a_file:
        pickle.dump(ellipse_list, a_file)
    with open(os.path.join(save_path_iter, "koller_ellipse_center_data.pkl"), "wb") as a_file:
        pickle.dump(ellipse_center_list, a_file)
    # plt.xlim(-0.1, 1.45)
    # plt.ylim(-0.1, 2.7)
    # plt.show()

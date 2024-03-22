import argparse
import errno
import os
import warnings

import matplotlib.pyplot as plt
import yaml

import dill as pickle

from src.DEMPC import DEMPC
from src.visu import Visualizer
from src.agent import Agent
import numpy as np
import torch
import gpytorch
import copy

import plotting_utilities

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

workspace = "safe_gpmpc"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=40)  # initialized at origin
parser.add_argument("-repeat", type=int, default=1)  # initialized at origin
args = parser.parse_args()

# 1) Load the config file
with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = args.i
params["env"]["name"] = args.env
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

# torch.cuda.memory._record_memory_history(enabled=True)

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

# get saved input trajectory
input_data_path = (
    "/home/amon/Repositories/safe_gpmpc/experiments/pendulum/env_0/params/40/data.pkl"
    # "/home/amon/Repositories/safe_gpmpc/experiments/pendulum/env_0/params/_static/reachable_set_input.pkl"
)
with open(input_data_path, "rb") as input_data_file:
    input_gpmpc_data = pickle.load(input_data_file)

agent = Agent(params)


import torch
import gpytorch
import math
from matplotlib import pyplot as plt
import numpy as np

lb_plot, ub_plot = 0.0, 4.0
lb, ub = 0.5, 3.5
n = 3
n_hyper = 200

train_x = torch.linspace(lb, ub, n).unsqueeze(-1)
train_y = torch.stack(
    [
        torch.sin(2 * train_x) + torch.cos(train_x),
        # torch.nan(train_x.shape),
        -torch.sin(train_x) + 2 * torch.cos(2 * train_x),
    ],
    -1,
).squeeze(1)

train_x_hyper = torch.linspace(lb, ub, n_hyper).unsqueeze(-1)
train_y_hyper = torch.stack(
    [
        torch.sin(2 * train_x_hyper) + torch.cos(train_x_hyper),
        # torch.nan(train_x_hyper.shape),
        -torch.sin(train_x_hyper) + 2 * torch.cos(2 * train_x_hyper),
    ],
    -1,
).squeeze(1)

train_y_hyper += 0.08 * torch.randn(n_hyper, 2)


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks=2
)  # Value + Derivative
model = GPModelWithDerivatives(train_x_hyper, train_y_hyper, likelihood)

# this is for running the notebook in our testing framework
import os

smoke_test = "CI" in os.environ
training_iter = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# set noise to zero
model.likelihood.noise = torch.tensor([1e-4])
model.likelihood.task_noises = torch.tensor([1e-4, 1e-4])
# set requires_grad=False for likelihood noise
# model.likelihood.noise_covar.noise.requires_grad = False
for param in model.likelihood.parameters():
    param.requires_grad = False

# Use the adam optimizer
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    # model.parameters(),
    lr=0.1,
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x_hyper)
    loss = -mll(output, train_y_hyper)
    loss.backward()
    print(
        "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
        % (
            i + 1,
            training_iter,
            loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item(),
        )
    )
    optimizer.step()

# extract model hyperparameters in one line
state_dict = model.state_dict()

# new model
y_train_nod = train_y.clone()
y_train_nod[:, 1] = torch.tensor(float("nan"))


def new_model(train_x, train_y):
    model = GPModelWithDerivatives(train_x, train_y, likelihood)
    model.load_state_dict(state_dict)
    # meas_noise = 1e-3
    # model_nod.likelihood.noise = torch.tensor([meas_noise])
    # model_nod.likelihood.task_noises = torch.tensor([meas_noise, meas_noise])
    model.eval()
    return model


# Set into eval mode
likelihood.eval()

# sys.path.append("/home/manish/work/MPC_Dyn/safe_gpmpc")
import sys
from plotting_utilities.utilities import *

sys.path.append("/home/amon/Repositories/safe_gpmpc")


plot_GT = True
plot_sampling_MPC = False
plot_cautious_MPC = False
plot_safe_MPC = True
filename = "iterative_conditioning.pdf"  # "sam_uncertainity.pdf" "cautious_uncertainity.pdf" "safe_uncertainity.pdf"

TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=14)
# plt.figure(figsize=(TEXTWIDTH * 0.5 + 0.75, TEXTWIDTH * 0.5 * 1 / 2))

# set_figure_params(serif=True, fontsize=14)
# f = plt.figure(figsize=(TEXTWIDTH * 0.5 + 2.75, TEXTWIDTH * 0.5 * 1 / 2))
# f = plt.figure(figsize=(cm2inches(12.0), cm2inches(8.0)))
# f, ax = plt.subplots(1, 3, figsize=(3 * cm2inches(12.0), 3 * cm2inches(8.0)))
f, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH * 0.5 + 0.75, TEXTWIDTH * 0.5 * 1 / 2))

marker_symbols = ["*", "o", "x", "s", "D", "P", "v", "^", "<", ">", "1", "2", "3", "4"]

train_x_2 = torch.tensor([0.8, 1.8, 2.8]).unsqueeze(-1)
train_x_3 = torch.tensor([0.9, 1.9, 3.0]).unsqueeze(-1)
# train_x_arr_add = [train_x, train_x_2, train_x_3]
train_x_arr_add = [train_x.clone(), train_x_2.clone(), train_x_3.clone()]
train_y_arr_add = [train_y.clone(), train_y.clone(), train_y.clone()]
train_x_arr = train_x.clone()
train_y_arr = train_y.clone()
train_x_arr_all = []
train_y_arr_all = []
# loop over the axes
for i in range(3):
    # ax[0].ylabel(r"$z$")
    # ax[0].xlabel(r"$g^n(z)$")

    model_nod = new_model(train_x_arr, train_y_arr)
    # Make predictions
    with torch.no_grad(), gpytorch.settings.observation_nan_policy("mask"):
        test_x = torch.linspace(lb_plot, ub_plot, 1000)
        predictions = model_nod(test_x)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

        if i == 0:
            sample = predictions.sample()

    train_x_arr_all.append(train_x_arr.clone())
    train_y_arr_all.append(train_y_arr.clone())

    # condition model on new data
    # get values of sample at train_x_i, if it does not exist then find the closest value
    if i < 2:
        train_x_arr = torch.cat([train_x_arr, train_x_arr_add[i + 1]])
        train_y_arr_add[i + 1] = sample[
            np.searchsorted(test_x, train_x_arr_add[i + 1])
        ][:, 0, :]
        train_y_arr = torch.cat(
            [
                train_y_arr,
                train_y_arr_add[i + 1],
            ]
        )

    # Predictive mean as blue line
    ax[i].plot(test_x.numpy(), mean[:, 0].numpy(), "tab:blue")
    ax[i].plot(test_x.numpy(), sample[:, 0].numpy(), "tab:orange")
    # Shade in confidence
    ax[i].fill_between(
        test_x.numpy(),
        lower[:, 0].numpy(),
        upper[:, 0].numpy(),
        alpha=0.5,
        color="tab:blue",
    )
    # ax[i].legend(["Observed Values", "Mean", "Confidence"])
    ax[i].set_title(f"$j = {i+1}$")
    # remove tick labels
    ax[i].set_yticklabels([])
    ax[i].set_xticklabels([])
    # remove ticks
    ax[i].set_yticks([])
    ax[i].set_xticks([])

for i in range(3):
    # Plot training data as black stars
    for j in range(i + 1):
        # for j in range(2 - i, -1, -1):
        ax[i].plot(
            train_x_arr_add[j].detach().numpy(),
            train_y_arr_add[j][:, 0].detach().numpy(),
            f"k{marker_symbols[j]}",
        )


f.tight_layout(pad=0.0)
f.savefig(
    "/home/amon/Repositories/sampling-gpmpc/images/conditioning.pdf",
    format="pdf",
    dpi=300,
    transparent=True,
)
# plt.show()

# de_mpc = DEMPC(params, visu, agent)
# de_mpc.dempc_main()
# visu.save_data()
# dict_file = torch.cuda.memory._snapshot()
# pickle.dump(dict_file, open(save_path + str(traj_iter) + "/memory_snapshot_1.pickle", "wb"))
exit()

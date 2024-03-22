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

lb, ub = 0.0, 4
n = 3

train_x = torch.linspace(lb, ub, n).unsqueeze(-1)
train_y = torch.stack(
    [
        torch.sin(2 * train_x) + torch.cos(train_x),
        # torch.nan(train_x.shape),
        -torch.sin(train_x) + 2 * torch.cos(2 * train_x),
    ],
    -1,
).squeeze(1)

train_y += 0.05 * torch.randn(n, 2)


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
model = GPModelWithDerivatives(train_x, train_y, likelihood)

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
    output = model(train_x)
    loss = -mll(output, train_y)
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

model_nod = GPModelWithDerivatives(train_x, y_train_nod, likelihood)
model_nod.load_state_dict(state_dict)


# Set into eval mode
model_nod.train()
model_nod.eval()
likelihood.eval()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(12, 6))

# Make predictions
with torch.no_grad(), gpytorch.settings.max_cg_iterations(
    50
), gpytorch.settings.observation_nan_policy("mask"):
    test_x = torch.linspace(lb, ub, 500)
    predictions = likelihood(model_nod(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# Plot training data as black stars
y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), "k*")
# Predictive mean as blue line
y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), "b")
# Shade in confidence
y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
y1_ax.legend(["Observed Values", "Mean", "Confidence"])
y1_ax.set_title("Function values")

# Plot training data as black stars
y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), "k*")
# Predictive mean as blue line
y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), "b")
# Shade in confidence
y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
y2_ax.legend(["Observed Derivatives", "Mean", "Confidence"])
y2_ax.set_title("Derivatives")

plt.show()
# de_mpc = DEMPC(params, visu, agent)
# de_mpc.dempc_main()
# visu.save_data()
# dict_file = torch.cuda.memory._snapshot()
# pickle.dump(dict_file, open(save_path + str(traj_iter) + "/memory_snapshot_1.pickle", "wb"))
exit()

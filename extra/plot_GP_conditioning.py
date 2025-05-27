import argparse
import errno
import os, sys
import warnings

import matplotlib.pyplot as plt
import yaml

import dill as pickle

import numpy as np
import torch
import gpytorch
import copy

import plotting_utilities
from plotting_utilities import *

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

# # NOTE: this file needs to be called from outside the root directory of the project, e.g.:
# # python sampling-gpmpc/extra/plot_GP_conditioning.py
workspace = "sampling-gpmpc"
sys.path.append(workspace)

filename = "iterative_conditioning.pdf" 

import torch
import gpytorch
import math
from matplotlib import pyplot as plt
import numpy as np

# torch random seed
torch.manual_seed(0)

lb_plot, ub_plot = 0.0, 4.0
lb, ub = 0.5, 3.5
n = 4
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

train_y_hyper += 0.07 * torch.randn(n_hyper, 2)

test_x = torch.linspace(lb_plot, ub_plot, 1000)
test_y = torch.stack(
    [
        torch.sin(2 * test_x) + torch.cos(test_x),
        -torch.sin(test_x) + 2 * torch.cos(2 * test_x),
    ],
    -1,
).squeeze(1)


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
    return model


# Set into eval mode
model.eval()
likelihood.eval()

plot_GT = True
plot_sampling_MPC = False
plot_cautious_MPC = False
plot_safe_MPC = True

TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=10)
f, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH * 0.5 + 0.75, TEXTWIDTH * 0.25 * 1 / 2))

marker_symbols = ["*", "o", "x", "s", "D", "P", "v", "^", "<", ">", "1", "2", "3", "4"]

train_x_2 = torch.tensor([0.8, 1.8, 2.8]).unsqueeze(-1)
train_x_3 = torch.tensor([0.9, 1.9, 3.0]).unsqueeze(-1)
# train_x_arr_add = [train_x, train_x_2, train_x_3]
train_x_arr_add = [
    train_x.clone(),
    train_x_2.clone(),
    train_x_3.clone(),
    train_x_3.clone(),
]
train_y_arr_add = [train_y.clone(), train_y.clone(), train_y.clone()]
train_x_arr = train_x.clone()
train_y_arr = train_y.clone()
train_x_arr_all = []
train_y_arr_all = []

beta_fac = 1.5
# loop over the axes
data = {"test_x": test_x.numpy(), "test_y": test_y[:, 0].numpy()}
data["marker_symbols"] = marker_symbols

for i in range(3):
    model_nod = new_model(train_x_arr, train_y_arr)
    model_nod.covar_module.base_kernel.lengthscale = torch.tensor([[0.3]])
    model_nod.covar_module.outputscale = torch.tensor([0.5])
    model_nod.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.observation_nan_policy("mask"):
        predictions = model_nod(test_x)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

        mean_lower = beta_fac * (mean - lower)
        mean_upper = beta_fac * (upper - mean)

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

    # vlines at test_x_add
    if i < 2 or True:
        # if i < 2:
        for x in train_x_arr_add[i + 1]:
            ax[i].axvline(
                x.numpy().flatten(),
                color="k",
                linestyle="solid",
                alpha=0.3,
                linewidth=3,
            )
    # Predictive mean as blue line
    h_func = ax[i].plot(test_x.numpy(), test_y[:, 0].numpy(), "k--")
    h_mean = ax[i].plot(test_x.numpy(), mean[:, 0].numpy(), "tab:blue")
    h_samp = ax[i].plot(test_x.numpy(), sample[:, 0].numpy(), "tab:orange")
    data[i] = {
        "mean": mean[:, 0].numpy(),
        "sample": sample[:, 0].numpy(),
        "train_x_arr_add": train_x_arr_add,
        "train_y_arr_add": train_y_arr_add,
        "lcb": mean[:, 0].numpy() - mean_lower[:, 0].numpy(),
        "ucb": mean[:, 0].numpy() + mean_upper[:, 0].numpy(),
    }
    # Shade in confidence
    # h_conf = ax[i].fill_between(
    #     test_x.numpy(),
    #     lower[:, 0].numpy(),
    #     upper[:, 0].numpy(),
    #     alpha=0.5,
    #     color="tab:blue",
    # )
    h_conf = ax[i].fill_between(
        test_x.numpy(),
        mean[:, 0] - mean_lower[:, 0].numpy(),
        mean[:, 0] + mean_upper[:, 0].numpy(),
        alpha=0.5,
        color="tab:blue",
    )

    if i == 2:
        ax[i].legend(
            [h_func[0], h_samp[0], h_conf],
            # ["True", "Mean", "Sample"],
            [
                r"true function $g^{\mathrm{tr}}$",
                r"sampled function $g^n$",
                r"$\mathcal{GP}_{[\underline{g}, \overline{g}]}(0,k_{\mathrm{d}};\mathcal{D}^n_{0:j-1})$",
            ],
            # loc="lower left",
        )
    # ax[i].legend(["Observed Values", "Mean", "Confidence"])
    # set title with current marker_symbols

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

# with open("/home/manish/work/MPC_Dyn/slides_data.pickle", "wb") as handle:
#     pickle.dump(data, handle)

f.tight_layout(pad=0.5)
f.savefig(
    os.path.join(workspace, "figures", filename),
    format="pdf",
    dpi=600,
    transparent=True,
)
# plt.show()

exit()

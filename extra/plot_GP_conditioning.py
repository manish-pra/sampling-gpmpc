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
k_lengthscale = 0.3
k_outscale = 0.5
beta_fac = 1.5

file_format = "png"

# iterative conditioning plot
filename = "iterative_conditioning" 
plot_separate_figs = True  # Set to True to plot each iteration in a separate figure
plot_without_reconditioning = True

# sampling plot
n_samples = 100
filename_sampling = "gp_samples"

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

def train_hyperparams():
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

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=2
    )  # Value + Derivative
    model = GPModelWithDerivatives(train_x_hyper, train_y_hyper, likelihood)

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

    return model, likelihood, state_dict

def new_model(train_x, train_y, likelihood, state_dict):
    model = GPModelWithDerivatives(train_x, train_y, likelihood)
    model.load_state_dict(state_dict)
    return model

model, likelihood, state_dict = train_hyperparams()

train_x = torch.linspace(lb, ub, n).unsqueeze(-1)
train_y = torch.stack(
    [
        torch.sin(2 * train_x) + torch.cos(train_x),
        # torch.nan(train_x.shape),
        -torch.sin(train_x) + 2 * torch.cos(2 * train_x),
    ],
    -1,
).squeeze(1)

test_x = torch.linspace(lb_plot, ub_plot, 1000)
test_y = torch.stack(
    [
        torch.sin(2 * test_x) + torch.cos(test_x),
        -torch.sin(test_x) + 2 * torch.cos(2 * test_x),
    ],
    -1,
).squeeze(1)


TEXTWIDTH = 16
set_figure_params(serif=True, fontsize=10)
marker_symbols = ["*", "o", "x", "s", "D", "P", "v", "^", "<", ">", "1", "2", "3", "4"]

def plot_iterative_conditioning(plot_separate_figs=False):
    plot_settings = []
    plot_settings.append({
        "filename_suffix": "without_reconditioning",
        "plot_without_reconditioning": True,
        "plot_sample": [True, True, True],
        "plot_sample_from_data": [True, True, True],
        "plot_mean": False,
    })
    plot_settings.append({
        "filename_suffix": "with_reconditioning",
        "plot_without_reconditioning": False,
        "plot_sample": [True, True, True],
        "plot_sample_from_data": [True, True, True],
        "plot_mean": False,
    })

    for ps in plot_settings: 
        if plot_separate_figs:
            f_arr = []
            ax = []
            for i in range(3):
                f, ax_i = plt.subplots(1, 1, figsize=(TEXTWIDTH * 0.5 + 0.75, TEXTWIDTH * 0.25))
                f_arr.append(f)
                ax.append(ax_i)
        else:
            f, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH * 0.5 + 0.75, TEXTWIDTH * 0.25 * 1 / 2))
            f_arr = [f]

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
        train_x_arr_wo_recon = train_x.clone()
        train_y_arr_wo_recon = train_y.clone()

        # loop over the axes
        data = {"test_x": test_x.numpy(), "test_y": test_y[:, 0].numpy()}
        data["marker_symbols"] = marker_symbols

        for i in range(3):
            model_nod = new_model(train_x_arr, train_y_arr, likelihood, state_dict)
            model_nod.covar_module.base_kernel.lengthscale = torch.tensor([[k_lengthscale]])
            model_nod.covar_module.outputscale = torch.tensor([k_outscale])
            model_nod.eval()

            # Make predictions
            with torch.no_grad(), gpytorch.settings.observation_nan_policy("mask"):
                predictions = model_nod(test_x)
                mean = predictions.mean
                lower, upper = predictions.confidence_region()

                mean_lower = beta_fac * (mean - lower)
                mean_upper = beta_fac * (upper - mean)

                if (i == 0 or ps["plot_without_reconditioning"]) and i < 2:
                    # set seed again to get the same samples here with or without reconditioning
                    # torch.manual_seed(1234)
                    sample = predictions.sample()

            train_x_arr_all.append(train_x_arr_wo_recon.clone())
            train_y_arr_all.append(train_y_arr_wo_recon.clone())


            # sample from model without reconditioning
            # if ps["plot_without_reconditioning"]:
            model_wo_recon = new_model(
                train_x_arr_wo_recon, train_y_arr_wo_recon, likelihood, state_dict
            )
            model_wo_recon.covar_module.base_kernel.lengthscale = torch.tensor([[k_lengthscale]])
            model_wo_recon.covar_module.outputscale = torch.tensor([k_outscale])
            model_wo_recon.eval()

            with torch.no_grad(), gpytorch.settings.observation_nan_policy("mask"):
                predictions_wo_recon = model_wo_recon(test_x)

                # TODO: save base_samples 
                if i == 0 or True:
                    mean_wo_recon = predictions.mean
                    lower_wo_recon, upper_wo_recon = predictions.confidence_region()

                    mean_lower_wo_recon = mean_wo_recon - beta_fac * (mean_wo_recon - lower_wo_recon)
                    mean_upper_wo_recon = mean_wo_recon + beta_fac * (upper_wo_recon - mean_wo_recon)

                    sample_shape = torch.Size((n_samples,))
                    # base_samples = predictions.get_base_samples(sample_shape=sample_shape)
                    # base_samples = base_samples.permute((1,0,2))
                
                    sample_wo_recon_all = predictions_wo_recon.rsample(
                        sample_shape=sample_shape, 
                        # base_samples=base_samples
                    )

                    # find samples that are outside the confidence region
                    outside_confidence_pointwise = torch.logical_or(
                        sample_wo_recon_all < mean_lower_wo_recon.unsqueeze(0),
                        sample_wo_recon_all > mean_upper_wo_recon.unsqueeze(0),
                    )
                    outside_confidence = outside_confidence_pointwise.any(dim=1)

                    sample_wo_recon = sample_wo_recon_all
                    sample_wo_recon_outside = sample_wo_recon_all[outside_confidence[:, 0], :, :]
                    sample_wo_recon_inside = sample_wo_recon_all[~outside_confidence[:, 0], :, :]
                    # sample_wo_recon = sample_wo_recon_all[~outside_confidence[:, 0], :, :]
                    # base_samples = base_samples[~outside_confidence[:, 0], :, :]
                else:
                    sample_wo_recon = predictions_wo_recon.rsample(
                        sample_shape=torch.Size((n_samples,)), 
                        base_samples=base_samples
                    )

            # condition model on new data
            # get values of sample at train_x_i, if it does not exist then find the closest value
            if i < 2:
                train_y_arr_add[i + 1] = sample[
                    np.searchsorted(test_x, train_x_arr_add[i + 1])
                ][:, 0, :]

                train_x_arr_wo_recon = torch.cat([train_x_arr_wo_recon, train_x_arr_add[i + 1]])
                train_y_arr_wo_recon = torch.cat(
                    [
                        train_y_arr_wo_recon,
                        train_y_arr_add[i + 1],
                    ]
                )
                if not ps["plot_without_reconditioning"]:
                    train_x_arr = train_x_arr_wo_recon.clone()
                    train_y_arr = train_y_arr_wo_recon.clone()

            if ps["plot_sample_from_data"][i]:
                if torch.any(outside_confidence[:, 0]):
                    h_sample_wo_recon_outside = ax[i].plot(
                        test_x.numpy(),
                        sample_wo_recon_outside[:, :, 0].numpy().T,
                        "tab:red",
                        alpha=0.5,
                        linewidth=1,
                    )
                    h_sample_wo_recon_outside[0].set_label(
                        r"$\mathrm{Samples} \notin [\underline{g}, \overline{g}]$"
                    )

                if torch.any(~outside_confidence[:, 0]):
                    h_sample_wo_recon_inside = ax[i].plot(
                        test_x.numpy(),
                        sample_wo_recon_inside[:, :, 0].numpy().T,
                        "tab:blue",
                        alpha=0.5,
                        linewidth=1,
                    )
                    h_sample_wo_recon_inside[0].set_label(
                        r"$\mathrm{Samples} \in [\underline{g}, \overline{g}]$"
                    )

            # Predictive mean as blue line
            h_func = ax[i].plot(test_x.numpy(), test_y[:, 0].numpy(), "k--")

            if ps["plot_mean"]:
                h_mean = ax[i].plot(test_x.numpy(), mean[:, 0].numpy(), "tab:blue", label="Predictive mean")

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
                alpha=0.3,
                color="tab:blue",
                label= r"Confidence bounds $\underline{g}, \overline{g}$",
            )

        # for i in range(3):
            # Plot training data as black stars
            for j in range(i + 1):
                # for j in range(2 - i, -1, -1):
                h_data = ax[i].plot(
                    train_x_arr_add[j].detach().numpy(),
                    train_y_arr_add[j][:, 0].detach().numpy(),
                    f"k{marker_symbols[j]}",
                )

                for h in h_data:
                    h.set_zorder(10)

            # ax[i].legend(["Observed Values", "Mean", "Confidence"])
            # set title with current marker_symbols
            ax[i].legend(loc="upper right")

            ax[i].set_title(f"$j = {i+1}$")
            # remove tick labels
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])
            # remove ticks
            ax[i].set_yticks([])
            ax[i].set_xticks([])
            ax[i].set_xlim([lb_plot, ub_plot])
            ax[i].set_ylim([-3.2, 3.2])

            if plot_separate_figs:
                i_f = i
            else:
                i_f = 0
            f_i = f_arr[i_f]

            f_i.tight_layout(pad=0.5)
            filename_suffix = ps["filename_suffix"]
            f_i.savefig(
                os.path.join(workspace, "figures", f"{filename}_{i_f}_{filename_suffix}_1_samples.{file_format}"),
                format=file_format,
                dpi=600,
                transparent=False,
            )
            print(f"Figure saved to {os.path.join(workspace, 'figures', f'{filename}_{i_f}_{filename_suffix}.{file_format}')}")

            # vlines at test_x_add
            if i < 2 or True:
                # if i < 2:
                for x in train_x_arr_add[i + 1]:
                    h_vlines = ax[i].axvline(
                        x.numpy().flatten(),
                        color="k",
                        linestyle="solid",
                        alpha=0.3,
                        linewidth=3,
                    )

            f_i.tight_layout(pad=0.5)
            filename_suffix = ps["filename_suffix"]
            f_i.savefig(
                os.path.join(workspace, "figures", f"{filename}_{i_f}_{filename_suffix}_2_vlines.{file_format}"),
                format=file_format,
                dpi=600,
                transparent=False,
            )
            print(f"Figure saved to {os.path.join(workspace, 'figures', f'{filename}_{i_f}_{filename_suffix}.{file_format}')}")

            if ps["plot_sample"][i]:
                h_samp = ax[i].plot(test_x.numpy(), sample[:, 0].numpy(), "tab:orange")

            f_i.tight_layout(pad=0.5)
            filename_suffix = ps["filename_suffix"]
            f_i.savefig(
                os.path.join(workspace, "figures", f"{filename}_{i_f}_{filename_suffix}_3_final.{file_format}"),
                format=file_format,
                dpi=600,
                transparent=False,
            )
            print(f"Figure saved to {os.path.join(workspace, 'figures', f'{filename}_{i_f}_{filename_suffix}.{file_format}')}")

        # with open("/home/manish/work/MPC_Dyn/slides_data.pickle", "wb") as handle:
        #     pickle.dump(data, handle)

def plot_GP_samples():
    model = new_model(train_x, train_y, likelihood, state_dict)
    model.covar_module.base_kernel.lengthscale = torch.tensor([[k_lengthscale]])
    model.covar_module.outputscale = torch.tensor([k_outscale])
    model.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.observation_nan_policy("mask"):
        predictions = model(test_x)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

        mean_lower = mean - beta_fac * (mean - lower)
        mean_upper = mean + beta_fac * (upper - mean)

        sample = predictions.sample(sample_shape=torch.Size((n_samples,)))

    # find samples that are outside the confidence region
    outside_confidence_pointwise = torch.logical_or(
        sample < mean_lower.unsqueeze(0),
        sample > mean_upper.unsqueeze(0),
    )
    outside_confidence = outside_confidence_pointwise.any(dim=1)

    plot_settings = []
    plot_settings.append({
        "filename_suffix": "confidence",
        "plot_samples_in": False,
        "plot_samples_out": False,
        "color_samples_in": "tab:blue",
        "color_samples_out": "tab:red",
    })
    plot_settings.append({
        "filename_suffix": "samples",
        "plot_samples_in": True,
        "plot_samples_out": True,
        "color_samples_in": "tab:blue",
        "color_samples_out": "tab:blue",
    })
    plot_settings.append({
        "filename_suffix": "samples_outside",
        "plot_samples_in": True,
        "plot_samples_out": True,
        "color_samples_in": "tab:blue",
        "color_samples_out": "tab:red",
    })
    plot_settings.append({
        "filename_suffix": "samples_filtered",
        "plot_samples_in": True,
        "plot_samples_out": False,
        "color_samples_in": "tab:blue",
        "color_samples_out": "tab:red",
    })

    for ps in plot_settings:
        f, ax = plt.subplots(1, 1, figsize=(TEXTWIDTH * 0.5 + 0.75, TEXTWIDTH * 0.25))
        set_figure_params(serif=True, fontsize=10)
        if ps["plot_samples_out"]:
            h_samp_outside = ax.plot(test_x.numpy(), sample[outside_confidence[:, 0], :, 0].numpy().T, ps["color_samples_out"],alpha=0.5, linewidth=1)
            h_samp_outside[0].set_label(r"$\mathrm{Samples} \notin [\underline{g}, \overline{g}]$")
        if ps["plot_samples_in"]:
            h_samp_inside = ax.plot(test_x.numpy(), sample[~outside_confidence[:, 0], :, 0].numpy().T, ps["color_samples_in"],alpha=0.5, linewidth=1)
            h_samp_inside[0].set_label(r"$\mathrm{Samples} \in [\underline{g}, \overline{g}]$")

        h_func = ax.plot(test_x.numpy(), test_y[:, 0].numpy(), "k--", label="True function $g^{\\mathrm{tr}}$")
        h_mean = ax.plot(test_x.numpy(), mean[:, 0].numpy(), "tab:blue", label="Predictive mean")
        h_conf = ax.fill_between(
            test_x.numpy(),
            mean_lower[:, 0].numpy(),
            mean_upper[:, 0].numpy(),
            alpha=0.3,
            color="tab:blue",
            label=r"Confidence bounds $\underline{g}, \overline{g}$",
        )
        h_data = ax.plot(
            train_x.numpy(),
            train_y[:, 0].numpy(),
            "k",
            linestyle="",
            marker=marker_symbols[0],
            markersize=10,
            label="Training data",
        )

        # remove xticks and yticks
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlim([lb_plot, ub_plot])
        ax.set_ylim([-3.2, 3.2])
        ax.legend(loc="upper right")

        f.tight_layout(pad=0.5)

        f.savefig(
            os.path.join(workspace, "figures", f"{filename_sampling}_{ps['filename_suffix']}.{file_format}"),
            format=file_format,
            dpi=600,
            transparent=False,
        )

if __name__ == "__main__":
    plot_iterative_conditioning(plot_separate_figs=plot_separate_figs)
    # plot_GP_samples()
    # plt.show()  # Uncomment to display the plot interactively
import math
import torch
import gpytorch
import sys
from matplotlib import pyplot as plt

sys.path.append("../")
# from LBFGS import FullBatchLBFGS

import os
import urllib.request
from scipy.io import loadmat

dataset = "protein"
if not os.path.isfile(f"../{dataset}.mat"):
    print(f"Downloading '{dataset}' UCI dataset...")
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?export=download&id=1nRb8e7qooozXkNghC5eQS0JeywSXGX2S",
        f"../{dataset}.mat",
    )

data = torch.Tensor(loadmat(f"../{dataset}.mat")["data"])


import numpy as np

N = data.shape[0]
# make train/val/test
n_train = int(0.8 * N)
train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
test_x, test_y = data[n_train:, :-1], data[n_train:, -1]

# normalize features
mean = train_x.mean(dim=-2, keepdim=True)
std = train_x.std(dim=-2, keepdim=True) + 1e-6  # prevent dividing by 0
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

# normalize labels
mean, std = train_y.mean(), train_y.std()
train_y = (train_y - mean) / std
test_y = (test_y - mean) / std

# make continguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()

output_device = torch.device("cuda:0")

train_x, train_y = train_x.to(output_device), train_y.to(output_device)
test_x, test_y = test_x.to(output_device), test_y.to(output_device)


n_devices = torch.cuda.device_count()
print("Planning to run on {} GPUs.".format(n_devices))


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_devices):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices), output_device=output_device
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(
    train_x,
    train_y,
    n_devices,
    output_device,
    checkpoint_size,
    preconditioner_size,
    n_training_iter,
):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = ExactGPModel(train_x, train_y, likelihood, n_devices).to(output_device)
    model.train()
    likelihood.train()

    optimizer = torch.optim.LBFGS.FullBatchLBFGS(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.beta_features.checkpoint_kernel(
        checkpoint_size
    ), gpytorch.settings.max_preconditioner_size(preconditioner_size):

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            return loss

        loss = closure()
        loss.backward()

        for i in range(n_training_iter):
            options = {"closure": closure, "current_loss": loss, "max_ls": 10}
            loss, _, _, _, _, _, _, fail = optimizer.step(options)

            print(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    n_training_iter,
                    loss.item(),
                    model.covar_module.module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item(),
                )
            )

            if fail:
                print("Convergence reached!")
                break

    print(f"Finished training on {train_x.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


model, likelihood = train(
    train_x,
    train_y,
    n_devices=n_devices,
    output_device=output_device,
    checkpoint_size=10000,
    preconditioner_size=100,
    n_training_iter=20,
)

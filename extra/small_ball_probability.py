import gpytorch
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import yaml


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=params["agent"]["g_dim"]["nx"]
                + params["agent"]["g_dim"]["nu"]
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Get training data and update GP with the


# sample A and B matrices
workspace = "sampling-gpmpc"

sys.path.append(workspace)
from src.agent import Agent
from src.environments.pendulum import Pendulum
from src.environments.pendulum1D import Pendulum as Pendulum1D

# 1) Load the config file
with open(workspace + "/params/" + "params_pendulum1D_samples" + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = 21
params["env"]["name"] = 0
print(params)

env_model = globals()[params["env"]["dynamics"]](params)

# if params["env"]["dynamics"] == "pendulum":
#     env_model = Pendulum(params)

agent = Agent(params, env_model)


Dyn_gp_X_train, Dyn_gp_Y_train = env_model.initial_training_data()
# Computation of C_D
# 1) Compute RKHS norm of the mean function
#######################################

g_nx = params["agent"]["g_dim"]["nx"]
g_nu = params["agent"]["g_dim"]["nu"]
g_ny = params["agent"]["g_dim"]["ny"]

train_x = torch.ones((1, g_nx + g_nu)) * 20000
train_y = torch.zeros((1, g_ny))
Dyn_gp_noise = 0.0
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(Dyn_gp_noise)
)
# model = ExactGPModel(train_x, train_y, likelihood)
model = ExactGPModel(Dyn_gp_X_train, Dyn_gp_Y_train[0, :, 0], likelihood)
# model.covar_module.base_kernel.lengthscale = torch.tensor(
#     params["agent"]["Dyn_gp_lengthscale"]["both"]
# )
# model.likelihood.noise = 1.0e-6
# model.covar_module.outputscale = 0.1
model.covar_module.base_kernel.lengthscale = torch.tensor(
    params["agent"]["Dyn_gp_lengthscale"]["both"]
)
# model.covar_module.base_kernel.lengthscale = 5.2649
model.likelihood.noise = torch.tensor(params["agent"]["Dyn_gp_noise"])
model.covar_module.outputscale = torch.tensor(
    params["agent"]["Dyn_gp_outputscale"]["both"]
)


# Define the ranges
x_range = (-2.14, 2.14)
z_range = (-8, 8)

# Define the number of points in each dimension
num_points = 11  # You can adjust this number as needed

# Generate the linspace for each dimension
x = np.linspace(x_range[0], x_range[1], num_points)
z = np.linspace(z_range[0], z_range[1], num_points)

# Create a meshgrid for multi-dimensional space
X, Z = np.meshgrid(x, z, indexing="ij")

# Flatten the grid and stack the coordinates into a tensor of shape (-1, 3)

grid_points = torch.from_numpy(np.stack([X.flatten(), Z.flatten()], axis=-1)).cuda()
print(grid_points.shape)

# X = torch.linspace(-np.pi, np.pi, 100)


total_samples = 1000000
with torch.no_grad(), gpytorch.settings.observation_nan_policy(
    "mask"
), gpytorch.settings.fast_computations(
    covar_root_decomposition=False, log_prob=False, solves=False
), gpytorch.settings.cholesky_jitter(
    float_value=params["agent"]["Dyn_gp_jitter"],
    double_value=params["agent"]["Dyn_gp_jitter"],
    half_value=params["agent"]["Dyn_gp_jitter"],
):
    model.eval()
    pred = model(grid_points)
    samples = pred.sample(sample_shape=torch.Size([total_samples]))
sample_diff = samples - pred.mean
# plt.plot(X, samples.transpose(0, 1), lw=0.1)
# plt.savefig("samples.png")

eps = 0.01  # 0.7  #
in_samples = torch.logical_and(sample_diff > -eps, sample_diff < eps)
total_in_samples = torch.sum(torch.all(in_samples, dim=1))
print("in samples", total_in_samples)
print("Probability", total_in_samples / total_samples)


print("lengthscale", model.covar_module.base_kernel.lengthscale)
print("noise", model.likelihood.noise)
print("outputscale", model.covar_module.outputscale)


quit()


######################################1D############################
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


train_x = torch.ones((1, 1)) * 20000
train_y = torch.zeros((1, 1))
# train_x = torch.linspace(-np.pi, np.pi, 7).reshape(-1, 1)
# train_y = torch.zeros(7)
Dyn_gp_noise = 1.0e-6
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(Dyn_gp_noise)
)
model = ExactGPModel(train_x, train_y, likelihood)
model.covar_module.base_kernel.lengthscale = 5.2649
model.likelihood.noise = 1.0e-6
model.covar_module.outputscale = 0.65

model.eval()
X = torch.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
pred = model(X)

total_samples = 10000
samples = pred.sample(sample_shape=torch.Size([total_samples]))
sample_diff = samples - pred.mean
plt.plot(X, samples.transpose(0, 1), lw=0.1)
plt.savefig("samples.png")

eps = 0.1  # 0.003
in_samples = torch.logical_and(sample_diff > -eps, sample_diff < eps)
total_in_samples = torch.sum(torch.all(in_samples, dim=1))
print("in samples", total_in_samples)
print("Probability", total_in_samples / total_samples)
a = 1

print("lengthscale", model.covar_module.base_kernel.lengthscale)
print("noise", model.likelihood.noise)
print("outputscale", model.covar_module.outputscale)

import torch
import gpytorch
import math
from matplotlib import pyplot as plt
import numpy as np
from botorch.models import SingleTaskGP

lb, ub = 0.0, 5*math.pi
n = 50

train_x = torch.linspace(lb, ub, n).unsqueeze(-1)
train_y = torch.stack([
    torch.sin(2*train_x) + torch.cos(train_x),
    -torch.sin(train_x) + 2*torch.cos(2*train_x)
], -1).squeeze(1)

train_y += 0.05 * torch.randn(n, 2)

# model = SingleTaskGP(train_x.reshape(-1,1),train_y.reshape(-1,2))
# base_kernel = gpytorch.kernels.RBFKernel()
# model.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
# model.covar_module.base_kernel.lengthscale = 1
# model.likelihood.noise = 0.001
# likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
class GPModelWithDerivatives(gpytorch.models.ExactGP):
    """_summary_ The model has multiple parameters: The corresponding values can be seen at:
    - model.likelihood.task_noises
    - model.likelihood.noises
    - model.covar_module.outputscale
    - model.covar_module.base_kernel.lengthscale
    - model.mean_module.constant

    Args:
        gpytorch (_type_): _description_
    """
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad() #(prior=gpytorch.priors.NormalPrior(4.9132,0.01))
        self.base_kernel = gpytorch.kernels.RBFKernelGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        self.covar_module.base_kernel.lengthscale = torch.Tensor([[1.2241]])
        self.covar_module.outputscale = torch.Tensor([[2.4601]])
        # self.likelihood = likelihood  # it is unneccessary, it automatically set the value

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# cov_mat = torch.eye(2)*1
# prior_noise = gpytorch.priors.MultivariateNormalPrior(loc = torch.zeros(2),covariance_matrix=cov_mat)
# prior_noise = gpytorch.priors.NormalPrior(loc = torch.zeros(1), scale=torch.ones(1)*0.1)
constraint_noise = gpytorch.constraints.GreaterThan(0.0)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2,noise_constraint=constraint_noise)  # Value + Derivative
model = GPModelWithDerivatives(train_x, train_y, likelihood)
model.likelihood.noise = torch.ones(1)*0.002
model.likelihood.task_noises=torch.Tensor([3.8,1.27])*0.00001
# model.likelihood.task=torch.ones(2)*0.1
for params in model.parameters():
    print(params)

# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# for i in range(training_iter):
#     optimizer.zero_grad()
#     output = model(train_x)
#     loss = -mll(output, train_y)
#     loss.backward()
#     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#         i + 1, training_iter, loss.item(),
#         model.covar_module.base_kernel.lengthscale.item(),
#         model.likelihood.noise.item()
#     ))
#     print(model.likelihood.task_noises, model.covar_module.outputscale)
#     optimizer.step()


# # Set into eval mode
# # model.train()
model.eval()
likelihood.eval()

print(model.parameters)
for params in model.parameters():
    print(params)

print( model.likelihood.raw_task_noises, 
      model.likelihood.raw_noise,
      model.mean_module.constant,
      model.covar_module.base_kernel.raw_lengthscale,
      model.covar_module.raw_outputscale)

print("Set value", model.likelihood.task_noises, 
    model.likelihood.noise,
    model.mean_module.constant,
    model.covar_module.base_kernel.lengthscale,
    model.covar_module.outputscale)

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(12, 6))

# Make predictions
with torch.no_grad(), gpytorch.settings.max_cg_iterations(50):
    test_x = torch.linspace(lb, ub, 500)
    # predictions = likelihood(model(test_x))
    predictions = model(test_x)
    mean = predictions.mean
    stddev = torch.sqrt(predictions.stddev)
    lower = mean - 3*stddev
    upper = mean + 3*stddev
    # lower, upper = predictions.confidence_region()
    sample  = predictions.sample()
# Plot training data as black stars
y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
# Predictive mean as blue line
y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
# Shade in confidence
y1_ax.plot(test_x.numpy(), sample[:, 0].numpy(),label="sample")
y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
y1_ax.legend(['Observed Values', 'Mean', 'Confidence'])
y1_ax.set_title('Function values')

# Plot training data as black stars
y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
# Predictive mean as blue line
y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
y2_ax.plot(test_x.numpy(), sample[:, 1].numpy(),label="sample")
# Shade in confidence
y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
y2_ax.legend(['Observed Derivatives', 'Mean', 'Confidence'])
y2_ax.set_title('Derivatives')

plt.show()
# import numpy as np
# import cvxpy as cp

# # Given system matrices
# A = np.array([[0.9, 0.3], [-0.2, 0.8]])  # System matrix
# B = np.array([[0.1], [0.2]])  # Control input matrix

# # Disturbance ellipsoid: w^T Q^{-1} w <= 1
# Q = np.array([[0.01, 0], [0, 0.01]])  # Defines the disturbance shape

# # Define optimization variables
# n, m = B.shape  # System dimensions
# E = cp.Variable((n, n), symmetric=True)  # RPI shape matrix (E)
# Y = cp.Variable((m, n))  # New variable Y = K * E
# lmbda = cp.Variable()  # Slack variable
# alpha = cp.Variable()  # Contraction factor

# # Define LMI constraints using Schur Complement

# # 1. Contraction LMI
# LMI1 = cp.bmat([[(1 + alpha) * E, A @ E + B @ Y], [(A @ E + B @ Y).T, E]]) >> 0

# # 2. Disturbance containment LMI
# LMI2 = (
#     cp.bmat(
#         [
#             [E, A @ E + B @ Y, np.zeros((n, n))],
#             [(A @ E + B @ Y).T, E - lmbda * np.linalg.inv(Q), np.zeros((n, n))],
#             [np.zeros((n, n)), np.zeros((n, n)), lmbda * np.eye(n)],
#         ]
#     )
#     >> 0
# )

# # 3. Ensure positive definiteness
# constraints = [E >> 0, lmbda >= 0, alpha >= 0, LMI1, LMI2]

# # Objective: Maximize log(det(E)) for largest RPI set
# objective = cp.Maximize(cp.log_det(E))

# # Solve the LMI problem
# prob = cp.Problem(objective, constraints)
# prob.solve()

# # Recover the optimal feedback gain K
# K_opt = Y.value @ np.linalg.inv(E.value)

# # Print results
# print("Optimal Feedback Gain K:\n", K_opt)
# print("Computed Largest RPI Ellipsoid E:\n", E.value)

import gpytorch
import torch
import math

train_x = torch.linspace(0, 1, 100)

train_y = torch.stack(
    [
        torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    ],
    -1,
)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskGPModel(train_x, train_y, likelihood)

# this is for running the notebook in our testing framework
import os

smoke_test = "CI" in os.environ
training_iterations = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))
    optimizer.step()

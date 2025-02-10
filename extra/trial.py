import numpy as np
import cvxpy as cp

# Given system matrices
A = np.array([[0.9, 0.3], [-0.2, 0.8]])  # System matrix
B = np.array([[0.1], [0.2]])  # Control input matrix

# Disturbance ellipsoid: w^T Q^{-1} w <= 1
Q = np.array([[0.01, 0], [0, 0.01]])  # Defines the disturbance shape

# Define optimization variables
n, m = B.shape  # System dimensions
E = cp.Variable((n, n), symmetric=True)  # RPI shape matrix (E)
Y = cp.Variable((m, n))  # New variable Y = K * E
lmbda = cp.Variable()  # Slack variable
alpha = cp.Variable()  # Contraction factor

# Define LMI constraints using Schur Complement

# 1. Contraction LMI
LMI1 = cp.bmat([[(1 + alpha) * E, A @ E + B @ Y], [(A @ E + B @ Y).T, E]]) >> 0

# 2. Disturbance containment LMI
LMI2 = (
    cp.bmat(
        [
            [E, A @ E + B @ Y, np.zeros((n, n))],
            [(A @ E + B @ Y).T, E - lmbda * np.linalg.inv(Q), np.zeros((n, n))],
            [np.zeros((n, n)), np.zeros((n, n)), lmbda * np.eye(n)],
        ]
    )
    >> 0
)

# 3. Ensure positive definiteness
constraints = [E >> 0, lmbda >= 0, alpha >= 0, LMI1, LMI2]

# Objective: Maximize log(det(E)) for largest RPI set
objective = cp.Maximize(cp.log_det(E))

# Solve the LMI problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Recover the optimal feedback gain K
K_opt = Y.value @ np.linalg.inv(E.value)

# Print results
print("Optimal Feedback Gain K:\n", K_opt)
print("Computed Largest RPI Ellipsoid E:\n", E.value)

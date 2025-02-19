# import numpy as np
# import scipy.linalg
# import matplotlib.pyplot as plt
# from scipy.spatial import ConvexHull

# # Pendulum parameters
# g = 9.81  # Gravity (m/s^2)
# L = 1.0  # Length of pendulum (m)
# m = 1.0  # Mass of pendulum (kg)
# b = 0.1  # Damping coefficient

# # Linearize around the topmost position (theta = pi)
# A = np.array(
#     [[0, 1], [g / L, -b / (m * L**2)]]
# )  # Sign flipped due to linearization at theta = pi
# B = np.array([[0], [1 / (m * L**2)]])
# Q = np.array([[10, 0], [0, 1]])  # Penalize angle deviation  # Penalize angular velocity
# R = np.array([[0.1]])  # Penalize control effort

# dt = 0.015  # Time step for integration

# # Solve the Continuous-time Algebraic Riccati Equation (CARE)
# P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# # Compute the Continuous LQR gain K
# K = np.linalg.inv(R) @ B.T @ P

# # Compute the ellipsoid set using level sets of the Lyapunov function
# P_inv = np.linalg.inv(P)  # Inverse of P defines the shape of the ellipsoid

# theta_vals = np.linspace(-0.5, 0.5, 100)
# omega_vals = np.linspace(-1, 1, 100)
# X, Y = np.meshgrid(theta_vals, omega_vals)
# Z = np.zeros_like(X)

# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         x = np.array([[X[i, j]], [Y[i, j]]])
#         Z[i, j] = x.T @ P @ x  # Quadratic form defining the ellipsoid

# # Plot the Maximum Positive Invariant Set and the Ellipsoid
# # plt.figure(figsize=(6, 6))
# plt.contour(
#     X, Y, Z, levels=[1], colors="r", label="Ellipsoid Boundary"
# )  # Level set at 1
# plt.xlabel("Theta (rad) - Relative to π")
# plt.ylabel("Angular Velocity (rad/s)")
# plt.title("Ellipsoid Set Around Top Position")
# plt.legend()
# plt.grid()
# plt.show()

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Pendulum parameters
g = 9.81  # Gravity (m/s^2)
L = 1.0  # Length of pendulum (m)
m = 1.0  # Mass of pendulum (kg)
b = 0.0  # Damping coefficient

# Linearize around the topmost position (theta = pi)
A = np.array(
    [[0, 1], [g / L, -b / (m * L**2)]]
)  # Sign flipped due to linearization at theta = pi
B = np.array([[0], [1 / (m * L**2)]])
Q = np.array([[10, 0], [0, 1]])  # Penalize angle deviation  # Penalize angular velocity
R = np.array([[0.1]])  # Penalize control effort

dt = 0.01  # Time step for integration

# Solve the Continuous-time Algebraic Riccati Equation (CARE)
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# Compute the Continuous LQR gain K
K = np.linalg.inv(R) @ B.T @ P


# Function to compute one-step forward dynamics under LQR control
def step_dynamics(x):
    u = -K @ x  # LQR control law
    return A @ x + B @ u  # Continuous dynamics


# Define grid of initial conditions around (pi, 0)
theta_vals = np.linspace(-0.5, 0.5, 50)  # Small perturbations from pi
omega_vals = np.linspace(-1, 1, 50)
mpi_states = []

# Check if state remains within bounds over multiple steps
for theta in theta_vals:
    for omega in omega_vals:
        x = np.array([[theta], [omega]])
        stable = True
        for _ in range(100):  # Simulate forward in time
            x = x + dt * step_dynamics(x)  # Euler step for stability check
            if np.abs(x[0]) > 0.5 or np.abs(x[1]) > 1:  # If it escapes bounds
                stable = False
                break
        if stable:
            mpi_states.append([theta, omega])

# Convert to array for plotting
mpi_states = np.array(mpi_states)

# Plot the Maximum Positive Invariant Set
plt.figure(figsize=(6, 6))
plt.scatter(mpi_states[:, 0], mpi_states[:, 1], s=5, label="MPI Set")
plt.xlabel("Theta (rad) - Relative to π")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Maximum Positive Invariant Set Around Top Position")
plt.legend()
plt.grid()
plt.show()

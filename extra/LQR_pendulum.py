import numpy as np
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt

# Pendulum parameters
g = 9.81  # Gravity (m/s^2)
L = 1.0  # Length of pendulum (m)
m = 1.0  # Mass of pendulum (kg)
b = 0.1  # Damping coefficient

# Continuous-time state-space matrices
A = np.array(
    [[0, 1], [g / L, -b / (m * L**2)]]
)  # Change sign for new coordinate system
B = np.array([[0], [1 / (m * L**2)]])
Q = np.array([[10, 0], [0, 1]])  # Penalize angle deviation  # Penalize angular velocity
R = np.array([[0.1]])  # Penalize control effort

dt = 0.015  # Time step for discretization

# Discretize the system using zero-order hold (ZOH)
system = scipy.signal.cont2discrete((A, B, np.eye(2), 0), dt, method="zoh")
A_d, B_d, _, _ = system[:4]
# A_d = A
# B_d = B

# Solve the Discrete-time Algebraic Riccati Equation (DARE)
P = scipy.linalg.solve_discrete_are(A_d, B_d, Q, R)

# Compute the Discrete LQR gain K
K = np.linalg.inv(R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)
print(K, P)

# Simulation parameters
T = 5  # Simulation duration
time = np.arange(0, T, dt)

# New goal position (inverted position) at theta = pi
goal_position = np.pi

# Initial state (perturbation from bottom position)
x = np.array(
    [
        [0.2],
        [0],
    ]  # Initial angle (radians) from bottom (positive for small perturbation)
)  # Initial angular velocity

# Store results
x_history = []
u_history = []

# Run simulation
for t in time:
    # LQR control law with goal at pi
    u = -K @ (
        x - np.array([[goal_position], [0]])
    )  # Control law shifted by the goal position
    x = A_d @ x + B_d @ u  # Discrete-time state update
    x_history.append(x.flatten())
    u_history.append(u.flatten())

# Convert history to arrays
x_history = np.array(x_history)
u_history = np.array(u_history)

# Plot results
plt.figure(figsize=(10, 5))

# Plot state (angle and angular velocity)
plt.subplot(2, 1, 1)
plt.plot(time, x_history[:, 0], label="Theta (rad)")
plt.plot(time, x_history[:, 1], label="Angular velocity (rad/s)")
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.legend()
plt.grid()

# Plot control input (torque)
plt.subplot(2, 1, 2)
plt.plot(time, u_history, label="Control Input (Torque)")
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# import numpy as np
# import scipy.linalg
# import scipy.signal
# import matplotlib.pyplot as plt

# # Pendulum parameters
# g = 9.81  # Gravity (m/s^2)
# L = 1.0  # Length of pendulum (m)
# m = 1.0  # Mass of pendulum (kg)
# b = 0.1  # Damping coefficient

# # Continuous-time state-space matrices
# A = np.array([[0, 1], [g / L, -b / (m * L**2)]])
# B = np.array([[0], [1 / (m * L**2)]])
# Q = np.array([[10, 0], [0, 1]])  # Penalize angle deviation  # Penalize angular velocity
# R = np.array([[0.1]])  # Penalize control effort

# # Solve the Continuous-time Algebraic Riccati Equation (CARE)
# P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# # Compute the LQR gain K
# K = np.linalg.inv(R) @ B.T @ P

# # Simulate the system
# dt = 0.01  # Time step
# T = 5  # Simulation duration
# time = np.arange(0, T, dt)

# # Initial state (perturbation from upright position)
# x = np.array([[0.2], [0]])  # Initial angle (radians)  # Initial angular velocity

# # Store results
# x_history = []
# u_history = []

# # Run simulation
# for t in time:
#     u = -K @ x  # LQR control law
#     dx = A @ x + B @ u  # State derivative
#     x = x + dx * dt  # Euler integration
#     x_history.append(x.flatten())
#     u_history.append(u.flatten())

# # Convert history to arrays
# x_history = np.array(x_history)
# u_history = np.array(u_history)

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.subplot(2, 1, 1)
# plt.plot(time, x_history[:, 0], label="Theta (rad)")
# plt.plot(time, x_history[:, 1], label="Angular velocity (rad/s)")
# plt.xlabel("Time (s)")
# plt.ylabel("State")
# plt.legend()
# plt.grid()

# plt.subplot(2, 1, 2)
# plt.plot(time, u_history, label="Control Input (Torque)")
# plt.xlabel("Time (s)")
# plt.ylabel("Torque (Nm)")
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()

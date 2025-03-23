import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import scipy.spatial as sp


# L = 0.965
# Horizon = 10
# eps = 0.004
# shrinkage = np.power(L, Horizon-1)*eps + eps*2*np.sum([np.power(L, i) for i in range(Horizon-1)])
n = 4
m = 2
rho = 0.992
print("rho", rho)
E = cp.Variable((n, n), PSD=True)
Y = cp.Variable((m, n))
bar_w_2 = cp.Variable()

lmbda = 0.000001
obj = -cp.log_det(E)  # + 1 * cp.sum(c_x_2)  # + bar_w_2 / (1 - rho)

# obj = -cp.trace(E)
constraints = []
# constraints += [E >> np.diag(np.ones(n))]
# PSD
constraints += [E >> 0]

def range_float(start, end, step):
    while start < end:
        yield round(start, 2)
        start += step

dt = 0.06
lr = 1.738
lf = 1.105

delta_min = -0.6
delta_max = 0.6
theta_min = -1.0
theta_max = 1.0
v_min = 14
v_max = 16.4
d_delta = 0.1
d_theta = 0.1
d_v = 0.5

for delta in range_float(delta_min, delta_max, d_delta):
    for theta in range_float(theta_min, theta_max, d_theta):
        for v in range_float(v_min, v_max, d_v):  # Lipschitz constant is sensitive to v
            beta_in = (lr * np.tan(delta)) / (lf + lr)
            beta = np.arctan(beta_in)

            term = ((lr / (np.cos(delta) ** 2)) / (lf + lr)) / (1 + beta_in**2)

            A = np.array(
                [
                    [
                        1,
                        0,
                        -v * np.sin(theta + beta) * dt,
                        np.cos(theta + beta) * dt,
                    ],
                    [
                        0,
                        1,
                        v * np.cos(theta + beta) * dt,
                        np.sin(theta + beta) * dt
                    ],
                    [
                        0,
                        0,
                        1,
                        np.sin(beta) * dt / lr,

                    ],
                    [0, 0, 0, 1],
                ]
            )
            B = np.array([[-v * np.sin(theta + beta) * dt * term, 0], 
                          [v * np.cos(theta + beta) * dt * term, 0],
                          [v * np.cos(beta) * dt * term / lr,0],
                          [0, dt]])
            
            E_bmat = cp.bmat([[rho**2 * E, (A @ E + B @ Y).T], [(A @ E + B @ Y), E]])
            constraints += [E_bmat >> 0]


# state constraints
# Ax_i = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]])
# bx_i = np.array([[20.0], [-5.4], [13.0], [-13.0],[theta_max], [theta_min],[v_max], [v_min]])
Ax_i = np.array([ [0, 0, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]])
bx_i = np.array([[theta_max], [theta_min],[v_max], [v_min]])
for i, A_i in enumerate(Ax_i):
    constraints += [cp.quad_form(A_i, E) <= bx_i[i] ** 2]


# input constraints
Au = np.array([[1, 0], [-1,0], [0, 1], [0, -1]])
bu = np.array([[delta_max], [delta_min], [2], [-2]])
for Au_i, bu_i in zip(Au, bu):
    Au_i = Au_i.reshape(2, 1)
    bu_i = bu_i.reshape(1, 1)
    E_bmat = cp.bmat([[bu_i**2, Au_i.T @ Y], [(Y.T @ Au_i), E]])
    constraints += [E_bmat >> 0]

# # state tightenings
# for i in range(nx):
#     x_bmat = cp.bmat(
#         [
#             [cp.reshape(c_x_2[i], (1, 1)), Ax_i[i, :].reshape(1, -1) @ E],
#             [E.T @ Ax_i[i, :].reshape(1, -1).T, E],
#         ]
#     )
#     constraints += [x_bmat >> 0]

# Au_i = np.array([[1], [-1]])
# # bx_i = np.array([[dtheta + x_1], [dtheta - x_1], [domega], [domega]])
# for i in range(nu):
#     u_bmat = cp.bmat(
#         [
#             [cp.reshape(c_u_2[i], (1, 1)), Au_i[i, :].reshape(1, -1) @ Y],
#             [Y.T @ Au_i[i, :].reshape(1, -1).T, E],
#         ]
#     )
#     constraints += [u_bmat >> 0]


# W_vertices = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * 1
# for i in range(W_vertices.shape[0]):
#     w_bmat = cp.bmat(
#         [
#             [cp.reshape(bar_w_2, (1, 1)), W_vertices[i, :].reshape(1, -1)],
#             [W_vertices[i, :].reshape(1, -1).T, E],
#         ]
#     )
#     constraints += [w_bmat >> 0]
# for i in range(4):
#     bmat = cp.bmat(
#         [
#             [cp.reshape(bx_i[i] ** 2, (1, 1)), cp.reshape(Ax_i[i], (1, -1))],
#             [cp.reshape(Ax_i[i], (1, -1)).T, E],
#         ]
#     )
#     constraints += [bmat >> 0]


# define optimization problem
prob = cp.Problem(cp.Minimize(obj), constraints)

# solve optimization problem
prob.solve()

# print results
print("Optimal value: ", prob.value)
print("Optimal var: ", E.value)
P = np.linalg.inv(np.array(E.value))
print("Optimal P: ", P)
K = np.array(Y.value) @ P
print("Optimal K: ", K)

from numpy import linalg as LA

def transform_matrix(J, P):
    # Compute P^{1/2} using matrix square root (eigendecomposition)
    eigvals, eigvecs = np.linalg.eigh(P)  # P must be symmetric positive definite
    P_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    P_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

    # Compute P^{-1/2} J P^{1/2}
    # transformed_J = P_sqrt @ J
    # transformed_J = P_inv_sqrt @ J @ P_sqrt
    transformed_J = P_sqrt @ J @ P_inv_sqrt
    # transformed_J = P_sqrt @ J.T @ P @ J @ P_inv_sqrt
    return transformed_J



norm2_list = []
val_list = []
max_val = -1

for delta in range_float(delta_min, delta_max, d_delta):
    for theta in range_float(theta_min, theta_max, d_theta):
        for v in range_float(v_min, v_max, d_v):  # Lipschitz constant is sensitive to v
            beta_in = (lr * np.tan(delta)) / (lf + lr)
            beta = np.arctan(beta_in)

            term = ((lr / (np.cos(delta) ** 2)) / (lf + lr)) / (1 + beta_in**2)

            A = np.array(
                [
                    [
                        1,
                        0,
                        -v * np.sin(theta + beta) * dt,
                        np.cos(theta + beta) * dt,
                    ],
                    [
                        0,
                        1,
                        v * np.cos(theta + beta) * dt,
                        np.sin(theta + beta) * dt
                    ],
                    [
                        0,
                        0,
                        1,
                        np.sin(beta) * dt / lr,

                    ],
                    [0, 0, 0, 1],
                ]
            )
            B = np.array([[-v * np.sin(theta + beta) * dt * term, 0], 
                          [v * np.cos(theta + beta) * dt * term, 0],
                          [v * np.cos(beta) * dt * term / lr,0],
                          [0, dt]])
            # P = solve_discrete_are(A, B, Q, R)  # Discrete Algebraic Riccati Equation
            # K = -np.linalg.inv(R) @ B.T @ P # 2x4
            # # print(K, P)
            # K = np.array([[-8.70325634e-05, -5.80935002e-04, -8.59414487e-01,
            #             2.81945619e-03],
            #         [-2.19896176e-03,  3.74071269e-04, -5.67735217e-04,
            #             -2.80473875e-01]])
            J = A + B @ K
            transformed_J = transform_matrix(J, P)
            norm2 = LA.norm(transformed_J, ord=2)
            norm2_list.append(norm2)
            val_list.append([delta, theta, v, P, K])
            max_round = np.max(np.sum(transformed_J, axis=1))
            if max_val < max_round:
                max_val = max_round
print(norm2_list)
max_norm2 = max(norm2_list)
print("values at max",val_list[np.argmax(norm2_list)])
print(max_norm2)
print("Max", max_val)


quit()
# plot result
_, ax = plt.subplots(1, 1, figsize=(5, 4))
# plt.plot(ax, fill=False, alpha=0.8, edgecolor="black", linewidth=2, label="X")

L = np.linalg.cholesky(P)
t = np.linspace(0, 2 * np.pi, 200)
z = np.vstack([np.cos(t), np.sin(t)])
ell = np.linalg.inv(L.T) @ z

ax.plot(ell[0, :] + x_1, ell[1, :] + x_2, color="blue", label="P")


# # plot result
# # _, ax = plt.subplots(1, 1, figsize=(5, 4))
# # plt.plot(ax, fill=False, alpha=0.8, edgecolor="black", linewidth=2, label="X")
# w_bar = 0.001
# L = np.linalg.cholesky(P * (1 - rho) / w_bar)
# t = np.linspace(0, 2 * np.pi, 200)
# z = np.vstack([np.cos(t), np.sin(t)])
# ell = np.linalg.inv(L.T) @ z

# ax.plot(ell[0, :], ell[1, :], color="red", label="P-tight")


# you can use the provided xlim and ylim properties to set the plot limits
# ax.set_xlim(X.xlim)
# ax.set_ylim(X.ylim)


# Find intersection points (vertices) by solving Ax = b
vertices = []
for i in range(len(Ax_i)):
    for j in range(i + 1, len(Ax_i)):
        try:
            vertex = np.linalg.solve(
                np.array([Ax_i[i], Ax_i[j]]), np.array([bx_i[i], bx_i[j]])
            )
            if np.all(
                Ax_i @ vertex <= bx_i + 1e-5
            ):  # Check if the point satisfies all inequalities
                vertices.append(vertex)
        except np.linalg.LinAlgError:
            continue  # Skip parallel lines

vertices = np.array(vertices)


# Sort vertices in a counterclockwise order for correct polygon plotting
def sort_vertices_ccw(points):
    center = np.mean(points, axis=0)  # Compute centroid
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles, 0).reshape(-1)]


arr = sort_vertices_ccw(vertices)
plt.plot(
    np.vstack([arr[:, 0], arr[0, 0]]) + x_1,
    np.vstack([arr[:, 1], arr[0, 1]]) + x_2,
    "k-",
)
# plot lines from equation
for i in range(len(Au)):
    x = np.linspace(-1, 1, 100)
    y = (bu[i] / Au[i] - K[0, 0] * x) / K[0, 1]
    plt.plot(x + x_1, y + x_2, "k--")

plt.legend(loc="lower left")
plt.grid(True)
# plt.savefig(f"temp_ellipse_{ns}2_u0.6.png")
plt.show()


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

# import numpy as np
# import scipy.linalg
# import matplotlib.pyplot as plt
# from scipy.spatial import ConvexHull

# # Pendulum parameters
# g = 9.81  # Gravity (m/s^2)
# L = 1.0  # Length of pendulum (m)
# m = 1.0  # Mass of pendulum (kg)
# b = 0.0  # Damping coefficient

# # Linearize around the topmost position (theta = pi)
# A = np.array(
#     [[0, 1], [g / L, -b / (m * L**2)]]
# )  # Sign flipped due to linearization at theta = pi
# B = np.array([[0], [1 / (m * L**2)]])
# Q = np.array([[10, 0], [0, 1]])  # Penalize angle deviation  # Penalize angular velocity
# R = np.array([[0.1]])  # Penalize control effort

# dt = 0.01  # Time step for integration

# # Solve the Continuous-time Algebraic Riccati Equation (CARE)
# P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# # Compute the Continuous LQR gain K
# K = np.linalg.inv(R) @ B.T @ P


# # Function to compute one-step forward dynamics under LQR control
# def step_dynamics(x):
#     u = -K @ x  # LQR control law
#     return A @ x + B @ u  # Continuous dynamics


# # Define grid of initial conditions around (pi, 0)
# theta_vals = np.linspace(-0.5, 0.5, 50)  # Small perturbations from pi
# omega_vals = np.linspace(-1, 1, 50)
# mpi_states = []

# # Check if state remains within bounds over multiple steps
# for theta in theta_vals:
#     for omega in omega_vals:
#         x = np.array([[theta], [omega]])
#         stable = True
#         for _ in range(100):  # Simulate forward in time
#             x = x + dt * step_dynamics(x)  # Euler step for stability check
#             if np.abs(x[0]) > 0.5 or np.abs(x[1]) > 1:  # If it escapes bounds
#                 stable = False
#                 break
#         if stable:
#             mpi_states.append([theta, omega])

# # Convert to array for plotting
# mpi_states = np.array(mpi_states)

# # Plot the Maximum Positive Invariant Set
# plt.figure(figsize=(6, 6))
# plt.scatter(mpi_states[:, 0], mpi_states[:, 1], s=5, label="MPI Set")
# plt.xlabel("Theta (rad) - Relative to π")
# plt.ylabel("Angular Velocity (rad/s)")
# plt.title("Maximum Positive Invariant Set Around Top Position")
# plt.legend()
# plt.grid()
# plt.show()

def transform_matrix(J, P):
    # Compute P^{1/2} using matrix square root (eigendecomposition)
    eigvals, eigvecs = np.linalg.eigh(P)  # P must be symmetric positive definite
    P_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    P_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

    # Compute P^{-1/2} J P^{1/2}
    # transformed_J = P_sqrt @ J
    # transformed_J = P_inv_sqrt @ J @ P_sqrt
    transformed_J = P_sqrt @ J @ P_inv_sqrt
    # transformed_J = P_sqrt @ J.T @ P @ J @ P_inv_sqrt
    return transformed_J


from scipy.linalg import solve_discrete_are
from numpy import linalg as LA

Q = np.array([[10, 0], [0, 1]])  # State cost matrix
R = np.array([[1]])  # Control cost matrix
# # Pendulum example
dt = 0.015
l = 10
g = 9.81
theta = np.pi
norm2_list = []
max_val = -1
for i in range(100):
    # P = np.array([[5.42156267, 1.8713373], [1.8713373, 0.80592194]])
    # K = np.array([[-13.81818248, -4.44151409]])
    # P = np.array([[4.07070798, 1.35050928], [1.35050928, 0.60804891]])
    # K = np.array([[-12.10018314, -3.99232037]])
    # P = np.array([[2.61719978, 0.82286531], [0.82286531, 0.41871497]])
    # K = np.array([[-9.68932421, -3.17352989]])
    # rho=0.95
    # P = np.array([[51.15935795, 11.49689237], [11.49689237, 3.34257094]])
    # K = np.array([[-29.26571774, -10.56473448]])
    # P = np.array([[34.92096505,  7.76312014], [ 7.76312014,  1.98764905]])
    # K = np.array([[-23.66438105,  -7.58198189]])
    # P = np.array([[56.35692934, 12.3262689], [12.3262689, 3.27775006]])
    # K = np.array([[-32.82044444, -10.40084854]])
    # P = np.array([[220.09074475, 25.18699789], [25.18699789, 5.311497]])
    # K = np.array([[-30.01183042, -12.3192521]])
    B = np.array([[0], [1]]) * dt
    A = np.array([[1, dt], [-g * np.cos(theta * i / 50) * dt / l, 1]])
    # P = solve_discrete_are(A, B, Q, R)  # Discrete Algebraic Riccati Equation
    # K = -np.linalg.inv(R) @ B.T @ P
    # print(K, P)
    J = A + B @ K
    transformed_J = transform_matrix(J, P)
    # transformed_J = np.linalg.sqrtm(J)
    max_round = np.max(np.sum(transformed_J, axis=1))
    if max_val < max_round:
        max_val = max_round
    norm2 = LA.norm(transformed_J, ord=2)
    norm2_list.append(norm2)
# print(norm2_list)
max_norm2 = max(norm2_list)
print(max_norm2)
print("Max", max_val)
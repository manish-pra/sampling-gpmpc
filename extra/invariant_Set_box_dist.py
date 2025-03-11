import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import scipy.spatial as sp

# sample A and B matrices
workspace = "sampling-gpmpc"

sys.path.append(workspace)
from src.agent import Agent
from src.environments.pendulum import Pendulum as pendulum
from src.environments.pendulum1D import Pendulum as Pendulum1D

# 1) Load the config file
with open(workspace + "/params/" + "params_pendulum1D_invariant" + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = 20
params["env"]["name"] = 0
print(params)

env_model = globals()[params["env"]["dynamics"]](params)

agent = Agent(params, env_model)

# 4) Set the initial state ---> NOT REQUIRED FOR INVARIANT SET COMPUTATION
agent.update_current_state(np.array(params["env"]["start"]))
nx = params["agent"]["dim"]["nx"]
nu = params["agent"]["dim"]["nu"]
H = params["optimizer"]["H"]
ns = params["agent"]["num_dyn_samples"]
rand_x = np.random.uniform(-1, 1, nx * H * ns).reshape(H, nx, ns)
# rand_x[:, 0, :] = rand_x[:, 0, :] * 0.2
# rand_x[:, 1, :] = rand_x[:, 1, :] * 0.1
x_1 = 0
x_2 = 0
u = 0
dtheta = 0.5  # 0.5
domega = 0.3  # 0.3
du = 1.0  # 0.3
x_h = np.zeros_like(rand_x)
x_h[:, 0, :] = rand_x[:, 0, :] * dtheta + x_1
x_h[:, 1, :] = rand_x[:, 1, :] * domega + x_2

# plt.plot(x_h[:,0,:].reshape(-1), x_h[:,1,:].reshape(-1),".")
# plt.savefig("temp.png")

x_h = x_h.transpose(0, 2, 1).reshape(H, -1)
rand_u = np.random.uniform(-1, 1, nu * H * ns).reshape(H, nu, ns) * du
u_h = np.zeros_like(rand_u)
u_h[:, 0, :] = rand_u[:, 0, :] + u
u_h = u_h.reshape(H, -1)
agent.train_hallucinated_dynGP(sqp_iter=0)
batch_x_hat = agent.get_batch_x_hat_u_diff(x_h, u_h)
agent.mpc_iter = 0
gp_val, y_grad, u_grad = agent.dyn_fg_jacobians(batch_x_hat, 0)

n = params["agent"]["dim"]["nx"]
m = params["agent"]["dim"]["nu"]
rho = 0.9999
E = cp.Variable((n, n), PSD=True)
Y = cp.Variable((m, n))
bar_w_2 = cp.Variable()
nx = 4
c_x_2 = cp.Variable((nx, 1))
nu = 2
c_u_2 = cp.Variable((nu, 1))
# lmbda = cp.Variable()  # Slack variable
# lmbda = 0.000001
obj = (50 * cp.sum(c_x_2) + cp.sum(c_u_2) + (nx + nu) * bar_w_2) / (
    2 * (1 - rho)
) * 0.5 - 40 * cp.log_det(
    E
)  # + 1 * cp.sum(c_x_2)  # + bar_w_2 / (1 - rho)

# obj = -cp.trace(E)
constraints = []
# constraints += [E >> np.diag(np.ones(n))]
constraints += [E >> 0]

for i in range(ns):
    for j in range(H):
        A = y_grad[i, :, j, :]
        B = u_grad[i, :, j, :]
        E_bmat = cp.bmat([[rho**2 * E, (A @ E + B @ Y).T], [(A @ E + B @ Y), E]])
        constraints += [E_bmat >> 0]

# Q = np.array([[0.01, 0], [0, 0.01]])
# constraints.append(lmbda >= 0)
# state constraints
Ax_i = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
bx_i = np.array([[dtheta + x_1], [dtheta - x_1], [domega], [domega]])
# for i, A_i in enumerate(Ax_i):
#     constraints += [cp.quad_form(A_i, E) <= (bx_i[i]) ** 2]

# state tightenings
for i in range(nx):
    x_bmat = cp.bmat(
        [
            [cp.reshape(c_x_2[i], (1, 1)), Ax_i[i, :].reshape(1, -1) @ E],
            [E.T @ Ax_i[i, :].reshape(1, -1).T, E],
        ]
    )
    constraints += [x_bmat >> 0]

Au_i = np.array([[1], [-1]])
# bx_i = np.array([[dtheta + x_1], [dtheta - x_1], [domega], [domega]])
for i in range(nu):
    u_bmat = cp.bmat(
        [
            [cp.reshape(c_u_2[i], (1, 1)), Au_i[i, :].reshape(1, -1) @ Y],
            [Y.T @ Au_i[i, :].reshape(1, -1).T, E],
        ]
    )
    constraints += [u_bmat >> 0]


W_vertices = (
    np.array(
        [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [np.sqrt(2) / 2, np.sqrt(2) / 2],
            [-np.sqrt(2) / 2, np.sqrt(2) / 2],
            [np.sqrt(2) / 2, -np.sqrt(2) / 2],
            [-np.sqrt(2) / 2, -np.sqrt(2) / 2],
        ]
    )
    * params["agent"]["tight"]["w_bound"]
)
for i in range(W_vertices.shape[0]):
    w_bmat = cp.bmat(
        [
            [cp.reshape(bar_w_2, (1, 1)), W_vertices[i, :].reshape(1, -1)],
            [W_vertices[i, :].reshape(1, -1).T, E],
        ]
    )
    constraints += [w_bmat >> 0]
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
prob.solve(solver="SCS")

# print results
print("Optimal value: ", prob.value)
print("Optimal var: ", E.value)
P = np.linalg.inv(np.array(E.value))
print("Optimal P: ", P)
K = np.array(Y.value) @ P
print("Optimal K: ", K)
print("Optimal c_x_2: ", c_x_2.value)
print("Optimal c_u_2: ", c_u_2.value)
print("Optimal bar_w_2: ", bar_w_2.value)


# compute delta
delta = np.sqrt(bar_w_2.value) / (1 - rho)

# compute tightening of state constraints
x_tight = delta * np.sqrt(c_x_2.value)

# compute tightening of input constraints
u_tight = delta * np.sqrt(c_u_2.value)

print("delta: ", delta)
print("x_tight: ", x_tight)
print("u_tight: ", u_tight)


# plot result
_, ax = plt.subplots(1, 1, figsize=(5, 4))
# plt.plot(ax, fill=False, alpha=0.8, edgecolor="black", linewidth=2, label="X")

L = np.linalg.cholesky(P)
t = np.linspace(0, 2 * np.pi, 200)
z = np.vstack([np.cos(t), np.sin(t)])
ell = np.linalg.inv(L.T) @ z

ax.plot(ell[0, :] + x_1, ell[1, :] + x_2, color="blue", label="P-level set=1")


# plot result
# _, ax = plt.subplots(1, 1, figsize=(5, 4))
# plt.plot(ax, fill=False, alpha=0.8, edgecolor="black", linewidth=2, label="X")
# w_bar = 0.001
L = np.linalg.cholesky(P / delta)
t = np.linspace(0, 2 * np.pi, 200)
z = np.vstack([np.cos(t), np.sin(t)])
ell = np.linalg.inv(L.T) @ z

ax.plot(ell[0, :] + x_1, ell[1, :] + x_2, color="red", label="Terminal set")


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
plt.plot(np.vstack([arr[:, 0], arr[0, 0]]), np.vstack([arr[:, 1], arr[0, 1]]), "k-")

plt.legend(loc="lower left")
plt.grid(True)
# plt.savefig(f"temp_ellipse_{ns}_bigger.png")
plt.show()

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

# sample A and B matrices
workspace = "sampling-gpmpc"
workspace = "sampling-gpmpc"
sys.path.append(workspace)
from src.agent import Agent
from src.environments.pendulum import Pendulum

# 1) Load the config file
with open(workspace + "/params/" + "params_pendulum" + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = 20
params["env"]["name"] = 0
print(params)

if params["env"]["dynamics"] == "pendulum":
    env_model = Pendulum(params)

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
x_1 = 3.14
x_2 = 0
u = 0
dtheta = 0.2
domega = 0.1
x_h = np.zeros_like(rand_x)
x_h[:, 0, :] = rand_x[:, 0, :] * dtheta + x_1
x_h[:, 1, :] = rand_x[:, 1, :] * domega + x_2

# plt.plot(x_h[:,0,:].reshape(-1), x_h[:,1,:].reshape(-1),".")
# plt.savefig("temp.png")

x_h = x_h.transpose(0, 2, 1).reshape(H, -1)
rand_u = np.random.uniform(-1, 1, nu * H * ns).reshape(H, nu, ns) * 0.1
u_h = np.zeros_like(rand_u)
u_h[:, 0, :] = rand_u[:, 0, :] + u
u_h = u_h.reshape(H, -1)
agent.train_hallucinated_dynGP(sqp_iter=0)
batch_x_hat = agent.get_batch_x_hat_u_diff(x_h, u_h)
agent.mpc_iter = 0
gp_val, y_grad, u_grad = agent.dyn_fg_jacobians(batch_x_hat, 0)


# define optimization variable
# E = cp.Variable((2, 2), symmetric=True)
# P = cp.Variable((2, 2), PSD=True)
# P_inv = cp.Variable((2, 2), PSD=True)

# # define objective
# obj = -cp.log_det(P_inv)

# # define constraints
# constraints = [
#     P_inv >> 0,
#     P >> 0,
#     P * P_inv == np.identity(2),
# ]  # The operator >> denotes a definitness constraint
# for i in range(ns):
#     for j in range(H):
#         A = y_grad[i, :, j, :]
#         B = u_grad[i, :, j, :]
#         K = np.ones((1, 2)) * (-2)
#         constraints += [cp.quad_form(A + B * K, P) <= P]

# P = cp.Variable((2, 2), PSD=True)

# # define objective
# obj = cp.trace(P)

# # define constraints
# constraints = [
#     P >> 0,
#     P >> np.identity(2),
# ]  # The operator >> denotes a definitness constraint
# for i in range(ns):
#     for j in range(H):
#         A = y_grad[i, :, j, :].transpose(0, 1)
#         B = u_grad[i, :, j, :]
#         K = np.ones((1, 2)) * (0.0)
#         constraints += [(A + B * K).transpose(0, 1) * P * (A + B * K) << P]

n = 2
m = 1
rho = 0.99
E = cp.Variable((n, n), PSD=True)
Y = cp.Variable((m, n))
obj = -cp.log_det(E)
# obj = cp.trace(E)
constraints = []
# constraints += [E >> np.diag(np.ones(n))]
constraints += [E >> 0]

for i in range(ns):
    for j in range(H):
        A = y_grad[i, :, j, :]
        B = u_grad[i, :, j, :]
        E_bmat = cp.bmat([[rho**2 * E, (A @ E + B @ Y).T], [(A @ E + B @ Y), E]])
        constraints += [E_bmat >> 0]

Ax_i = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
bx_i = np.array([[dtheta], [dtheta], [domega], [domega]])
for i, A_i in enumerate(Ax_i):
    constraints += [cp.quad_form(A_i, E) <= bx_i[i] ** 2]
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


# plot result
_, ax = plt.subplots(1, 1, figsize=(5, 4))
# plt.plot(ax, fill=False, alpha=0.8, edgecolor="black", linewidth=2, label="X")

L = np.linalg.cholesky(P)
t = np.linspace(0, 2 * np.pi, 200)
z = np.vstack([np.cos(t), np.sin(t)])
ell = np.linalg.inv(L.T) @ z

ax.plot(ell[0, :], ell[1, :], color="blue", label="P")

# you can use the provided xlim and ylim properties to set the plot limits
# ax.set_xlim(X.xlim)
# ax.set_ylim(X.ylim)

plt.legend(loc="lower left")
plt.grid(True)
plt.savefig("temp_ellipse.png")

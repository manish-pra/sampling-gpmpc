import torch
import casadi as ca


def get_prior_data(xu):
    dt = 0.015
    nx = 2  # phi, v
    nu = 1  # delta
    ny = 3  # phi, v, delta
    lf = 1.105 * 0.01
    lr = 1.738 * 0.01
    phi, v, delta = xu[:, 0], xu[:, 1], xu[:, 2]
    g_xu = unknown_dyn(xu)  # phi, v, delta
    y_ret = torch.zeros((xu.shape[0], ny, 1 + nx + nu))

    y_ret[:, 0, 0] = g_xu[:, 0]
    y_ret[:, 1, 0] = g_xu[:, 1]
    y_ret[:, 2, 0] = g_xu[:, 2]
    # derivative of dx w.r.t. phi, v, delta
    beta_in = (lr * torch.tan(delta)) / (lf + lr)
    beta = torch.atan(beta_in)
    y_ret[:, 0, 1] = -v * torch.sin(phi + beta) * dt
    y_ret[:, 0, 2] = torch.cos(phi + beta) * dt

    term = ((lr / (torch.cos(delta) ** 2)) / (lf + lr)) / (1 + beta_in**2)
    y_ret[:, 0, 3] = -v * torch.sin(phi + beta) * dt * term

    # derivative of dy w.r.t. phi, v, delta
    y_ret[:, 1, 1] = v * torch.cos(phi + beta) * dt
    y_ret[:, 1, 2] = torch.sin(phi + beta) * dt
    y_ret[:, 1, 3] = v * torch.cos(phi + beta) * dt * term

    # derivative of dphi w.r.t. phi, v, delta
    # y_ret[:, 2, 0] is zeros
    y_ret[:, 2, 2] = torch.sin(beta) * dt / lr
    y_ret[:, 2, 3] = v * torch.cos(beta) * dt * term / lr
    return y_ret


# x = ca.SX.sym("x")
# y = ca.SX.sym("y")
# phi = ca.SX.sym("phi")
# v = ca.SX.sym("v")

# x = ca.vertcat(x, y, phi, v)

# a = ca.SX.sym("a")
# delta = ca.SX.sym("delta")
# # lf = 1.105 * 0.01
# # lr = 1.738 * 0.01
# lf = ca.SX.sym("lf")
# lr = ca.SX.sym("lr")
# b = ca.atan(lr * ca.tan(delta) / (lf + lr))
# f_expl = ca.vertcat(v * ca.cos(phi + b), v * ca.sin(phi + b), v * ca.sin(b) / lr, a)


def unknown_dyn(xu):
    """_summary_"""
    Phi_k, V_k, delta_k = xu[:, [0]], xu[:, [1]], xu[:, [2]]
    lf = 1.105 * 0.01
    lr = 1.738 * 0.01
    dt = 0.015
    beta = torch.atan(torch.tan(delta_k) * lr / (lr + lf))
    dX_kp1 = V_k * torch.cos(Phi_k + beta) * dt
    dY_kp1 = V_k * torch.sin(Phi_k + beta) * dt
    Phi_kp1 = V_k * torch.sin(beta) * dt / lr
    state_kp1 = torch.hstack([dX_kp1, dY_kp1, Phi_kp1])
    return state_kp1


def discrete_dyn(xu):
    """_summary_"""

    g_xu = unknown_dyn(xu[:, [2, 3, 4]])  # phi, v, delta
    X_k, Y_k, Phi_k, V_k = xu[:, [0]], xu[:, [1]], xu[:, [2]], xu[:, [3]]
    acc_k = xu[:, 5]
    # lf = 1.105 * 0.01
    # lr = 1.738 * 0.01
    dt = 0.015
    # beta = torch.atan(torch.tan(delta_k) * lr / (lr + lf))
    X_kp1 = X_k + g_xu[:, [0]]
    Y_kp1 = Y_k + g_xu[:, [1]]
    Phi_kp1 = Phi_k + g_xu[:, [2]]  # + V_k * torch.sin(beta) * dt / lr
    V_kp1 = V_k + acc_k * dt
    state_kp1 = torch.stack([X_kp1, Y_kp1, Phi_kp1, V_kp1])
    return state_kp1


# def get_true_gradient(x_hat):
#     l = 1
#     g = 10
#     # A = np.array([[0.0, 1.0],
#     #               [-g*np.cos(x_hat[0])/l,0.0]])
#     # B = np.array([[0.0],
#     #               [1/l]])
#     ret = torch.zeros((2, x_hat.shape[0], 3))
#     ret[0, :, 1] = torch.ones(x_hat.shape[0])
#     ret[1, :, 0] = -g * torch.cos(x_hat[:, 0]) / l
#     ret[1, :, 2] = torch.ones(x_hat.shape[0]) / l

#     val = self.pendulum_dyn(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2])
#     return torch.hstack([val[:, 0].reshape(-1, 1), ret[0, :, :]]), torch.hstack(
#         [val[:, 1].reshape(-1, 1), ret[1, :, :]]

n_data_xp = 3
n_data_yp = 3
n_data_phi = 5
n_data_v = 3
n_data_u = 5

# x_p = torch.linspace(-1, 5, n_data_xp)
# y_p = torch.linspace(-1, 1, n_data_yp)
phi = torch.linspace(-1, 1, n_data_phi)
v_k = torch.linspace(-4, 10, n_data_v)
delta = torch.linspace(-0.5, 0.5, n_data_u)  # steering angle +/-30 degrees
acc = torch.linspace(-1, 1, n_data_u)
# v_k can also be removed from here
phik, Vk, deltak = torch.meshgrid(phi, v_k, delta)

Dyn_gp_X_train = torch.hstack(
    [phik.reshape(-1, 1), Vk.reshape(-1, 1), deltak.reshape(-1, 1)]
)

# Dyn_gp_Y_train = unknown_dyn(Dyn_gp_X_train)

Dyn_gp_Y_train = get_prior_data(Dyn_gp_X_train)
print(Dyn_gp_Y_train)

# # dynamics
# Fx = (Cm1 - Cm2 * vx) * T - Cd * vx * vx - Croll


# ar = atan((-vy + lr * omega) / vx)
# af_atan = atan((-vy - lf * omega) / vx)
# vel_threshold = 0.3
# condition = fabs(vx) < vel_threshold
# ar_interpolation_val = substitute(ar, vx, vel_threshold)
# af_atan_interpolation_val = substitute(af_atan, vx, vel_threshold)
# ar_interpolated = if_else(
#     condition,
#     ar_interpolation_val * sin(vx * pi / (2 * vel_threshold)),
#     ar,
# )
# af_atan_interpolated = if_else(
#     condition,
#     af_atan_interpolation_val * sin(vx * pi / (2 * vel_threshold)),
#     af_atan,
# )
# # interpolation interlude is over, nominal dynamics continue from here.
# Fr = Dr * sin(Cr * atan(Br * ar_interpolated))
# Ff = Df * sin(Cf * atan(Bf * (delta + af_atan_interpolated)))
# f_expl = vertcat(
#     vx * cos(yaw) - vy * sin(yaw),
#     vx * sin(yaw) + vy * cos(yaw),
#     omega,
#     1 / m * (Fx - Ff * sin(delta) + m * vy * omega),
#     1 / m * (Fr + Ff * cos(delta) - m * vx * omega),
#     1 / I * (Ff * lf * cos(delta) - Fr * lr),
#     dT,
#     ddelta,
#     dtheta,
# )
# # cost
# eC = sin(phi_d) * (xp - xd - grad_xd * (theta - theta_hat)) - cos(phi_d) * (
#     yp - yd - grad_yd * (theta - theta_hat)
# )
# eL = -cos(phi_d) * (xp - xd - grad_xd * (theta - theta_hat)) - sin(phi_d) * (
#     yp - yd - grad_yd * (theta - theta_hat)
# )
# c_eC = eC * eC * Q1
# c_eL = eL * eL * Q2
# c_theta = -q * theta
# c_dT = dT * dT * R1
# c_ddelta = ddelta * ddelta * R2
# c_dtheta = dtheta * dtheta * R3
# model.cost_expr_ext_cost = c_eC + c_eL + c_theta + c_dT + c_ddelta + c_dtheta
# # nonlinear track constraints
# radius_sq = (xp - xd) * (xp - xd) + (yp - yd) * (yp - yd)
# constraint.expr = vertcat(radius_sq, vx)

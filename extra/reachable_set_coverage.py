import dill as pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import torch


def get_minmax_theta_dot(X):
    # state_traj = true_gpmpc_data["state_traj"]
    theta_loc = np.linspace(0.0, 2.0, 100)
    theta_dot_list = []
    for i in range(X.shape[1] // 2):
        theta_dot_loc = np.interp(
            theta_loc,
            X[:, 2 * i],
            X[:, 2 * i + 1],
            right=np.nan,
        )
        theta_dot_list.append(theta_dot_loc)
    theta_dot_list = np.vstack(theta_dot_list)
    min_theta_dot = np.nanmin(theta_dot_list, axis=0)
    max_theta_dot = np.nanmax(theta_dot_list, axis=0)

    return theta_loc, min_theta_dot, max_theta_dot


# from plotting_utilities.plotting_utilities.utilities import *
from pathlib import Path

sys.path.append("/home/amon/Repositories/safe_gpmpc")

path_traj_true = "/home/amon/Repositories/safe_gpmpc/experiments/pendulum/env_0/params/40/X_traj_list.pkl"
path_traj_opt = (
    "/home/amon/Repositories/safe_gpmpc/experiments/pendulum/env_0/params/40/data.pkl"
    # "/home/amon/Repositories/safe_gpmpc/experiments/pendulum/env_0/params/_static/reachable_set_input.pkl"
)

nx = 2
nu = 1
H = 71

# get true trajectories
with open(path_traj_true, "rb") as file:
    traj_true = pickle.load(file)

XU_true = torch.stack([traj_true[i][:, 0, 0, :] for i in range(len(traj_true))])
X_true = XU_true[:, :, :nx]
U_true = XU_true[:, :, nx:]
# X_tmp = torch.permute(X_true, (0, 2, 1))
# X_tmp = torch.permute(X_true, (1, 2, 0))
# X_true_alt = np.array(torch.reshape(X_tmp, (H, -1)))
X_true_alt = torch.cat(
    [
        torch.stack((X_true[:, i, 0], X_true[:, i, 1]), dim=1)
        for i in range(XU_true.shape[1])
    ],
    dim=1,
)

theta_loc_true, min_theta_dot_true, max_theta_dot_true = get_minmax_theta_dot(
    X_true_alt
)

# get optimized trajectories
with open(path_traj_opt, "rb") as file:
    input_gpmpc_data = pickle.load(file)

X = input_gpmpc_data["state_traj"][0]
U = input_gpmpc_data["input_traj"][0]
theta_loc, min_theta_dot, max_theta_dot = get_minmax_theta_dot(
    input_gpmpc_data["state_traj"][0]
)

hull_vol_opt_true_ratio = np.zeros((H,))
# calculate coverage
for i in range(1, H):
    # opt
    pts_i = X[i].reshape(-1, 2)
    hull = ConvexHull(pts_i)
    hull_vol = hull.volume

    # true
    pts_i_true = X_true_alt[i].reshape(-1, 2)
    hull_true = ConvexHull(pts_i_true)
    hull_vol_true = hull_true.volume

    hull_vol_opt_true_ratio[i] = hull_vol / hull_vol_true


cover_opt = max_theta_dot - min_theta_dot
cover_true = max_theta_dot_true - min_theta_dot_true


plot_GT = True
plot_sampling_MPC = False
plot_cautious_MPC = False
plot_safe_MPC = True
filename = "safe_uncertainity.pdf"  # "sam_uncertainity.pdf" "cautious_uncertainity.pdf" "safe_uncertainity.pdf"

TEXTWIDTH = 16
# set_figure_params(serif=True, fontsize=14)
# f = plt.figure(figsize=(TEXTWIDTH * 0.5 + 2.75, TEXTWIDTH * 0.5 * 1 / 2))
# f = plt.figure(figsize=(cm2inches(12.0), cm2inches(8.0)))
f = plt.figure()
ax = f.axes
plt.ylabel(r"$\theta$")
plt.xlabel(r"$\dot{\theta}$")
plt.tight_layout(pad=0.0)
# plt.plot(x1_smpc, x2_smpc, alpha=0.7)
plt.xlim(-0.1, 1.45)

plt.fill_between(
    theta_loc_true, min_theta_dot_true, max_theta_dot_true, alpha=0.5, color="tab:red"
)
plt.fill_between(theta_loc, min_theta_dot, max_theta_dot, alpha=0.5, color="tab:blue")
# plt.plot(x1_true, x2_true, color="black")
plt.plot(X_true_alt[:, 0::nx], X_true_alt[:, 1::nx], alpha=0.7, color="tab:red")
plt.plot(X[:, 0::nx], X[:, 1::nx], alpha=0.7, color="tab:blue")
plt.show()
exit()
a = 1

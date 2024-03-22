import dill as pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from plotting_utilities.utilities import *
from pathlib import Path

# sys.path.append("/home/manish/work/MPC_Dyn/safe_gpmpc")
sys.path.append("/home/amon/Repositories/safe_gpmpc")

from src.visu import propagate_true_dynamics

plot_GT = True
plot_sampling_MPC = False
plot_cautious_MPC = False
plot_safe_MPC = True
filename = "safe_uncertainity.pdf"  # "sam_uncertainity.pdf" "cautious_uncertainity.pdf" "safe_uncertainity.pdf"

TEXTWIDTH = 16
set_figure_params(serif=True, fontsize=14)
# f = plt.figure(figsize=(TEXTWIDTH * 0.5 + 2.75, TEXTWIDTH * 0.5 * 1 / 2))
f = plt.figure(figsize=(cm2inches(12.0), cm2inches(8.0)))
ax = f.axes
plt.ylabel(r"$\theta$")
plt.xlabel(r"$\dot{\theta}$")
plt.tight_layout(pad=0.0)
# plt.plot(x1_smpc, x2_smpc, alpha=0.7)
plt.xlim(-0.1, 1.45)

GT_data_path = (
    "/home/amon/Repositories/safe_gpmpc/experiments/pendulum/env_0/params/40/data.pkl"
)
sampling_data_path = (
    "/home/amon/Repositories/safe_gpmpc/experiments/pendulum/env_0/params/40/data.pkl"
)

if plot_GT:
    # Load GT_uncertainity data
    a_file = open(GT_data_path, "rb")
    true_gpmpc_data = pickle.load(a_file)
    a_file.close()
    x_init, U = (
        true_gpmpc_data["state_traj"][0][0, :2],
        true_gpmpc_data["input_traj"][0],
    )
    x1_true, x2_true = propagate_true_dynamics(x_init, U)
    H = 71
    state_traj = true_gpmpc_data["state_traj"]
    # theta_loc = np.tile(np.linspace(0.0, 2.0, 100), (300, 1))
    # theta_loc = np.linspace(-0.1, 1.5, 100)
    theta_loc = np.linspace(0.0, 2.0, 100)
    theta_dot_list = []
    for i in range(300):
        theta_dot_loc = np.interp(
            theta_loc,
            state_traj[0][:, 2 * i],
            state_traj[0][:, 2 * i + 1],
            right=np.nan,
        )
        theta_dot_list.append(theta_dot_loc)
    theta_dot_list = np.vstack(theta_dot_list)
    min_theta_dot = np.nanmin(theta_dot_list, axis=0)
    max_theta_dot = np.nanmax(theta_dot_list, axis=0)

    plt.fill_between(
        theta_loc, min_theta_dot, max_theta_dot, alpha=0.5, color="tab:blue"
    )
    plt.plot(x1_true, x2_true, color="black")


if plot_sampling_MPC:
    a_file = open(sampling_data_path, "rb")
    sampling_gpmpc_data = pickle.load(a_file)
    a_file.close()
    x1_smpc = sampling_gpmpc_data["state_traj"][0][:, ::2]
    x2_smpc = sampling_gpmpc_data["state_traj"][0][:, 1::2]

    # Plot convex hull
    state_traj = sampling_gpmpc_data["state_traj"]
    pts_i = state_traj[0][0].reshape(-1, 2)
    plt.plot(pts_i[:, 0], pts_i[:, 1], ".", alpha=0.5, color="tab:blue")
    for i in range(1, H):
        pts_i = state_traj[0][i].reshape(-1, 2)
        hull = ConvexHull(pts_i)
        # plt.plot(pts_i[:, 0], pts_i[:, 1], ".", alpha=0.5, color="tab:blue")
        # plt.plot(
        #     pts_i[hull.vertices, 0],
        #     pts_i[hull.vertices, 1],
        #     alpha=0.7,
        #     color="tab:green",
        #     lw=1.5,
        # )
        stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
        plt.plot(
            pts_i[stack_vertices, 0],
            pts_i[stack_vertices, 1],
            alpha=0.7,
            color="tab:green",
            lw=1.5,
        )

if plot_cautious_MPC:
    ellipse_list_path = "/home/manish/work/MPC_Dyn/safe_gpmpc/experiments/pendulum/env_0/params/40123/ellipse_data.pkl"
    a_file = open(ellipse_list_path, "rb")
    ellipse_list = pickle.load(a_file)
    a_file.close()
    for ellipse in ellipse_list:
        plt.plot(ellipse[0, :], ellipse[1, :], lw=1.5, alpha=0.7)

if plot_safe_MPC:
    ellipse_list_path = (
        "/home/manish/work/horrible/safe-exploration_cem/koller_ellipse_data.pkl"
    )
    a_file = open(ellipse_list_path, "rb")
    ellipse_list = pickle.load(a_file)
    a_file.close()
    for ellipse in ellipse_list:
        plt.plot(ellipse[0, :], ellipse[1, :], lw=1.5, alpha=0.7, color="tab:red")


plt.plot([-0.1, 2.2], [2.5, 2.5], color="red", linestyle="--")
plt.xlim(-0.1, 1.45)
plt.ylim(-0.1, 2.7)
fname = Path().resolve().joinpath("figures")
fname.mkdir(exist_ok=True)
adapt_figure_size_from_axes(ax)
plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.savefig(
    str(fname.joinpath(filename)),
    format="pdf",
    dpi=300,
    transparent=True,
)
# plt.savefig("sam_uncertainity.png")

# plt.ylabel("theta_dot")
# plt.xlabel("theta")
# plt.grid()
# plt.savefig("uncertainity_convex_hull.png")
# a = 1


# Load sampling_gpmpc data

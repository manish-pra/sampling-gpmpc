# sys.path.append("/home/manish/work/MPC_Dyn/safe_gpmpc")
import sys, os
sys.path.append("sampling-gpmpc")

import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

# from plotting_utilities.plotting_utilities.utilities import *
from pathlib import Path

# from src.visu import propagate_true_dynamics

plot_GT = False
plot_GT_sampling = True
plot_sampling_MPC = True
plot_cautious_MPC = False
plot_safe_MPC = False
i = 22
filename = f"safe_uncertainity_{i}.pdf"  # "sam_uncertainity.pdf" "cautious_uncertainity.pdf" "safe_uncertainity.pdf"


TEXTWIDTH = 16
# set_figure_params(serif=True, fontsize=14)
f = plt.figure(figsize=(TEXTWIDTH * 0.5 + 2.75, TEXTWIDTH * 0.5 * 1 / 2))
# f = plt.figure(figsize=(cm2inches(12.0), cm2inches(8.0)))
ax = f.axes
plt.ylabel(r"$\theta$")
plt.xlabel(r"$\dot{\theta}$")
plt.tight_layout(pad=0.0)
# plt.plot(x1_smpc, x2_smpc, alpha=0.7)
plt.xlim(-0.1, 1.45)

prefix_X_traj_list = "X_traj_list"
GT_data_path = f"sampling-gpmpc/experiments/pendulum/env_0/params_pendulum_{i}/{i}/"
GT_sampling_data_path = f"sampling-gpmpc/experiments/pendulum/env_0/params_pendulum_{i}/{i}/"
sampling_data_path = f"sampling-gpmpc/experiments/pendulum/env_0/params_pendulum_{i}/{i}/data.pkl"

H = 31

if plot_GT:
    # TODO: add compatibility with multiple-files for X_traj_list (see below)
    # Load GT_uncertainity data
    a_file = open(GT_data_path, "rb")
    true_gpmpc_data = pickle.load(a_file)
    a_file.close()
    true_gpmpc_data_numpy = np.array([np.array(x.cpu()) for x in true_gpmpc_data])
    H_GT, N_samples, nx, _, nxu = true_gpmpc_data_numpy.shape
    assert H_GT == H

    theta_loc = np.linspace(0.0, 2.0, 100)
    theta_dot_list = []
    for i in range(N_samples):
        theta_dot_loc = np.interp(
            theta_loc,
            true_gpmpc_data_numpy[:, i, 0, 0, 0],
            true_gpmpc_data_numpy[:, i, 0, 0, 1],
            right=np.nan,
        )
        theta_dot_list.append(theta_dot_loc)
    theta_dot_list = np.vstack(theta_dot_list)
    min_theta_dot = np.nanmin(theta_dot_list, axis=0)
    max_theta_dot = np.nanmax(theta_dot_list, axis=0)

    plt.fill_between(
        theta_loc, min_theta_dot, max_theta_dot, alpha=0.5, color="tab:blue"
    )
    # plt.plot(x1_true, x2_true, color="black")

if plot_sampling_MPC:
    a_file = open(sampling_data_path, "rb")
    sampling_gpmpc_data = pickle.load(a_file)
    a_file.close()

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

if plot_GT_sampling:
    # find all files with name prefix_X_traj_list_i in directory (i is integer) of GT_sampling_data_path
    all_files_at_GT_sampling_data_path = os.listdir(GT_sampling_data_path)

    hull_points = []
    for file in all_files_at_GT_sampling_data_path:
        if not file.startswith(prefix_X_traj_list):
            continue

        traj_iter = file.split("_")[-1].split(".")[0]

        a_file = open(os.path.join(GT_sampling_data_path, file), "rb")
        sampling_gpmpc_data = pickle.load(a_file)
        a_file.close()
        sampling_gpmpc_data_np = np.array(
            [np.array(x.cpu()) for x in sampling_gpmpc_data]
        )
        H_GT, N_samples, nx, _, nxu = sampling_gpmpc_data_np.shape
        assert H_GT == H

        # Plot convex hull
        state_traj = sampling_gpmpc_data_np[:, :, 0, 0, 0:2]
        # plt.plot(pts_i[:, 0], pts_i[:, 1], ".", alpha=0.5, color="tab:blue")
        for i in range(1, H):
            # pts_i = state_traj[0][i].reshape(-1, 2)
            pts_i = state_traj[i]
            hull = ConvexHull(pts_i)
            if i - 1 < len(hull_points):
                hull_points[i - 1] = np.vstack(
                    [hull_points[i - 1], hull.points[hull.vertices]]
                )
            else:
                hull_points.append(hull.points[hull.vertices])

    for i in range(1, H):
        hull = ConvexHull(hull_points[i - 1])
        stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
        plt.plot(
            hull.points[stack_vertices, 0],
            hull.points[stack_vertices, 1],
            alpha=0.7,
            color="tab:red",
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
# adapt_figure_size_from_axes(ax)
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

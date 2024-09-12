# sys.path.append("/home/manish/work/MPC_Dyn/safe_gpmpc")
import sys, os
from scipy.spatial.distance import pdist, squareform
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from pathlib import Path
import argparse
import yaml
import torch

# NOTE: this file needs to be called from outside the root directory of the project, e.g.:
# python sampling-gpmpc/benchmarking/linearization_based_predictions.py
workspace = "sampling-gpmpc"
sys.path.append(workspace)

workspace_plotting_utils = "plotting_utilities"
sys.path.append(workspace_plotting_utils)

from plotting_utilities.utilities import *

plot_GT = False
plot_GT_sampling = True
plot_sampling_MPC = True
plot_cautious_MPC = True
plot_safe_MPC = True

plot_cautious_mean = False
plot_safe_mean = False


if __name__ == "__main__":

    # get GP model from agent
    plt.rcParams["figure.figsize"] = [12, 6]

    parser = argparse.ArgumentParser(description="A foo that bars")
    parser.add_argument("-param", default="params_pendulum")  # params

    parser.add_argument("-env", type=int, default=0)
    parser.add_argument("-i", type=int, default=999)  # initialized at origin
    args = parser.parse_args()

    # 1) Load the config file
    with open(workspace + "/params/" + args.param + ".yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    params["env"]["i"] = args.i
    params["env"]["name"] = args.env
    params["common"]["use_cuda"] = False
    print(params)

    # random seed
    if params["experiment"]["rnd_seed"]["use"]:
        torch.manual_seed(params["experiment"]["rnd_seed"]["value"])

    # 2) Set the path and copy params from file
    exp_name = params["experiment"]["name"]
    env_load_path = (
        workspace
        + "/experiments/"
        + params["experiment"]["folder"]
        + "/env_"
        + str(args.env)
        + "/"
    )

    save_path = env_load_path + "/" + args.param + "/"

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    print(args)
    if args.i != -1:
        traj_iter = args.i

    if not os.path.exists(save_path + str(traj_iter)):
        os.makedirs(save_path + str(traj_iter))

    save_path_iter = save_path + str(traj_iter)

    with open(os.path.join(save_path_iter, "data.pkl"), "rb") as pkl_file:
        data_dict = pickle.load(pkl_file)

    prefix_X_traj_list = "X_traj_list"
    GT_data_path = os.path.join(
        workspace, f"experiments/pendulum/env_0/params_pendulum/{args.i}/"
    )
    GT_sampling_data_path = os.path.join(
        workspace, f"experiments/pendulum/env_0/params_pendulum/{args.i}/"
    )
    sampling_data_path = os.path.join(
        workspace, f"experiments/pendulum/env_0/params_pendulum/{args.i}/data.pkl"
    )

    H = 31
    color = "powderblue"  # "lightblue"

    TEXTWIDTH = 16
    set_figure_params(serif=True, fontsize=14)

    create_plots = []
    if plot_cautious_MPC:
        create_plots.append("cautious")
    if plot_safe_MPC:
        create_plots.append("safe")
    if plot_sampling_MPC:
        create_plots.append("sampling")

    for plot_name in create_plots:

        # f = plt.figure(figsize=(TEXTWIDTH * 0.5 + 2.75, TEXTWIDTH * 0.5 * 1 / 2))
        f = plt.figure(figsize=(cm2inches(12.0), cm2inches(8.0)))
        ax = f.axes
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$\omega$")
        plt.tight_layout(pad=0.0)
        # plt.plot(x1_smpc, x2_smpc, alpha=0.7)

        if plot_GT:
            # TODO: add compatibility with multiple-files for X_traj_list (see below)
            # Load GT_uncertainity data
            with open(GT_data_path, "rb") as a_file:
                true_gpmpc_data = pickle.load(a_file)

            true_gpmpc_data_numpy = np.array(
                [np.array(x.cpu()) for x in true_gpmpc_data]
            )
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

        if plot_GT_sampling:
            # find all files with name prefix_X_traj_list_i in directory (i is integer) of GT_sampling_data_path
            all_files_at_GT_sampling_data_path = os.listdir(GT_sampling_data_path)
            hull_points = []

            for file in all_files_at_GT_sampling_data_path:
                if not file.startswith(prefix_X_traj_list):
                    continue

                traj_iter = file.split("_")[-1].split(".")[0]

                with open(os.path.join(GT_sampling_data_path, file), "rb") as a_file:
                    sampling_gpmpc_data = pickle.load(a_file)

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

            hull_points.insert(0, np.array([[0, 0]]))
            for i in range(H - 1):
                hull = ConvexHull(np.concatenate([hull_points[i], hull_points[i + 1]]))
                stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
                plt.fill(
                    hull.points[stack_vertices, 0],
                    hull.points[stack_vertices, 1],
                    alpha=1,
                    color=color,
                    lw=0,
                )
            plt.fill(
                np.array([-100, -100]),
                alpha=1,
                color=color,
                lw=0,
                label="True uncertainty",
            )

        if plot_name == "sampling":

            with open(sampling_data_path, "rb") as a_file:
                sampling_gpmpc_data = pickle.load(a_file)
            # Plot convex hull
            state_traj = sampling_gpmpc_data["state_traj"]
            pts_i = state_traj[0][0].reshape(-1, 2)
            # plt.plot(pts_i[:, 0], pts_i[:, 1], ".", alpha=0.5, color=color)
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

            filename = f"sam_uncertainty_{args.i}.pdf"  # "sam_uncertainity.pdf" "cautious_uncertainity.pdf" "safe_uncertainity.pdf"

        if plot_name == "cautious":
            # ellipse_list_path = "/home/manish/work/MPC_Dyn/sampling-gpmpc/experiments/pendulum/env_0/params_pendulum/999/ellipse_data.pkl"
            ellipse_list_path = "/home/amon/Repositories/sampling-gpmpc/experiments/pendulum/env_0/params_pendulum/22/cautious_ellipse_data.pkl"
            with open(ellipse_list_path, "rb") as a_file:
                ellipse_list = pickle.load(a_file)
            for ellipse in ellipse_list:
                plt.plot(
                    ellipse[0, :], ellipse[1, :], lw=1.5, alpha=0.7, color="tab:orange"
                )

            if plot_cautious_mean:
                ellipse_mean_list_path = "/home/amon/Repositories/sampling-gpmpc/experiments/pendulum/env_0/params_pendulum/22/cautious_ellipse_center_data.pkl"
                with open(ellipse_mean_list_path, "rb") as a_file:
                    ellipse_center_list = pickle.load(a_file)

                ellipse_center_np = np.array(ellipse_center_list)[:, :, 0]
                plt.plot(
                    ellipse_center_np[:, 0],
                    ellipse_center_np[:, 1],
                    lw=1.5,
                    alpha=0.7,
                    color="tab:orange",
                    linestyle="-",
                )
            filename = f"cautious_uncertainty_{args.i}.pdf"  # "sam_uncertainity.pdf" "cautious_uncertainity.pdf" "safe_uncertainity.pdf"

        if plot_name == "safe":
            # ellipse_list_path = (
            #     "/home/manish/work/horrible/safe-exploration_cem/koller_ellipse_data.pkl"
            # )
            ellipse_list_path = "/home/amon/Repositories/sampling-gpmpc/experiments/pendulum/env_0/params_pendulum/22/koller_ellipse_data.pkl"
            with open(ellipse_list_path, "rb") as a_file:
                ellipse_list = pickle.load(a_file)
            for ellipse in ellipse_list:
                plt.plot(
                    ellipse[0, :], ellipse[1, :], lw=1.5, alpha=0.7, color="tab:red"
                )

            if plot_safe_mean:
                ellipse_mean_list_path = "/home/amon/Repositories/sampling-gpmpc/experiments/pendulum/env_0/params_pendulum/22/koller_ellipse_center_data.pkl"
                with open(ellipse_mean_list_path, "rb") as a_file:
                    ellipse_center_list = pickle.load(a_file)

                ellipse_center_np = np.array(ellipse_center_list)[:, :, 0]
                plt.plot(
                    ellipse_center_np[:, 0],
                    ellipse_center_np[:, 1],
                    lw=1.5,
                    alpha=0.7,
                    color="tab:red",
                    linestyle="-",
                )
            filename = f"safe_uncertainty_{args.i}.pdf"  # "sam_uncertainity.pdf" "cautious_uncertainity.pdf" "safe_uncertainity.pdf"

        with open(sampling_data_path, "rb") as a_file:
            sampling_gpmpc_data = pickle.load(a_file)

        x1_true = sampling_gpmpc_data["true_state_traj"][0][:, 0]
        x2_true = sampling_gpmpc_data["true_state_traj"][0][:, 1]
        plt.plot(x1_true, x2_true, color="black", label="True dynamics")
        plt.plot(
            [-0.1, 2.2], [2.5, 2.5], color="black", linestyle="--", label="Constraint"
        )
        plt.xlim(-0.1, 0.8)
        plt.ylim(-0.2, 2.7)
        # plt.grid()

        if plot_name == "cautious":
            plt.legend()

        # adapt_figure_size_from_axes(ax)
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")

        plt.savefig(
            os.path.join(workspace, "figures", filename),
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

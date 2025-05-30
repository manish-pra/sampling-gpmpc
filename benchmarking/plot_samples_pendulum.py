import argparse
import errno
import os, sys
import warnings

import matplotlib.pyplot as plt
import yaml
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import dill as pickle
import numpy as np
import torch

import gpytorch
import copy

# NOTE: this file needs to be called from outside the root directory of the project, e.g.:
# python sampling-gpmpc/benchmarking/simulate_true_reachable_set.py
workspace = "sampling-gpmpc"
sys.path.append(workspace)
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]


parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_pendulum")  # params

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=22)  # initialized at origin
args = parser.parse_args()

# 1) Load the config file
with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = args.i
params["env"]["name"] = args.env
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

input_gpmpc_data_list = []

path_traj_opt = f"{save_path}{str(args.i)}/data.pkl"
with open(path_traj_opt, "rb") as input_data_file:
    opt_data = pickle.load(input_data_file)

true_traj = opt_data["true_state_traj"]

for i in range(1):
    input_data_path = f"{save_path}{str(args.i)}/X_traj_list_{i}.pkl"
    with open(input_data_path, "rb") as input_data_file:
        input_gpmpc_data = pickle.load(input_data_file)
    input_gpmpc_data_list.append(input_gpmpc_data)


metadata = dict(
    title="Movie Test", artist="Matplotlib", comment="Movie support!"
)
import matplotlib.animation as manimation
writer = manimation.FFMpegWriter(
    fps=24, codec="libx264", metadata=metadata
)
fig, ax = plt.subplots(figsize=(12 / 2.4, 6 / 2.4))
writer.setup(fig,"video_sampling_finite_200_legend.mp4", dpi=800) 
# X_traj = np.vstack(input_gpmpc_data_list)
ax.set_xlim(-0.05, 0.8)
ax.set_ylim(-0.01, 2.8)
# X_traj = np.vstack(input_gpmpc_data_list[0])

ax.plot([-1,-0.9],[-1,-0.9],lw=0.8, alpha=0.4, label=r"Trajectory samples")
max_samples = 200

print("Loaded input data")
GT_sampling_data_path = os.path.join(
    workspace, f"experiments/pendulum/env_0/params_pendulum/223/"
)
H = 31
color = "powderblue"
color_finite_reachable_set = "palegreen"
all_files_at_GT_sampling_data_path = os.listdir(GT_sampling_data_path)
hull_points = []
prefix_X_traj_list = "X_traj_list"
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
        zorder=1,
    )
plt.fill(
    np.array([-100, -100]),
    alpha=1,
    color=color,
    lw=0,
    label="True reachable set",
    zorder=0.01,
)
plt.fill(
    np.array([-100, -100]),
    alpha=1,
    color=color_finite_reachable_set,
    lw=0,
    label="Finite-sample reachable set",
    zorder=0.001,
)

def get_eps_vec(n_sample):
    C = 0.025
    eps = C / np.log(n_sample)
    # eps = 0.0002
    return np.array([eps, eps])

def get_reachable_set_ball(params, eps_vec=None):
    H = params["optimizer"]["H"]

    # computation of tightenings
    P = np.array(params["optimizer"]["terminal_tightening"]["P"])
    # P *=10

    L = params["agent"]["tight"]["Lipschitz"]
    dyn_eps = params["agent"]["tight"]["dyn_eps"]
    w_bound = params["agent"]["tight"]["w_bound"]
    var_eps = (dyn_eps + w_bound)
    if eps_vec is not None:
        # np.dot(np.sqrt(np.diag(P[:3][:3])),np.array([8e-4,9e-4,3e-4]))
        # B_d_norm = (np.dot(np.sqrt(np.diag(P[:3][:3])),np.array([3.65e-4,4e-4,1.35e-4]))/var_eps)*V_k
        B_d_norm = (np.dot(np.sqrt(np.diag(P[:2][:2])),eps_vec)/var_eps)
    else:
        B_d_norm = np.sum(np.sqrt(np.diag(P[:2][:2])))
    P_inv = np.linalg.inv(P)
    K = np.array(params["optimizer"]["terminal_tightening"]["K"])
    B_eps_0 = 0
    tightenings = np.sqrt(np.diag(P_inv))*B_eps_0
    u_tight = np.sqrt(np.diag(K@P_inv@K.T))*B_eps_0
    tilde_eps_list = []
    tilde_eps_list.append(np.concatenate([tightenings.tolist(), u_tight.tolist(), [B_eps_0]]))
    ci_list = []
    for stage in range(1, H + 1):
        B_eps_k = var_eps*B_d_norm* np.sum(np.power(L, np.arange(0, stage)))  
        # arange has inbuild -1 in [sstart, end-1]
        # box constraints tightenings
        tightenings = np.sqrt(np.diag(P_inv))*B_eps_k
        u_tight = np.sqrt(np.diag(K@P_inv@K.T))*B_eps_k
        print(f"u_tight_{stage} = {u_tight}")
        tilde_eps_list.append(np.concatenate([tightenings.tolist(), u_tight.tolist(), [B_eps_k]]))
        ci_list.append(B_eps_k)
        print(f"tilde_eps_{stage} = {tilde_eps_list[-1]}")
    # quit()
    return tilde_eps_list, ci_list

def plot_ellipses(ax, x, y, eps_list):
    P = np.array(params["optimizer"]["terminal_tightening"]["P"])[:2,:2]
    # P*=10
    nH = len(eps_list) - 1 # not required on terminal state
    n_pts = 30
    ns_sub = x.shape[1] #int(x.shape[1]/4) + 1
    P_scaled = np.tile(P, (nH,1,1))/(eps_list[:nH,None, None]**2+1e-8)
    L = np.linalg.cholesky(P_scaled)
    t = np.linspace(0, 2 * np.pi, n_pts)
    z = np.vstack([np.cos(t), np.sin(t)])
    ell = np.linalg.inv(np.transpose(L, (0,2,1))) @ z

    all_ell = np.tile(ell, (ns_sub, 1, 1, 1))
    x_plt = all_ell[:,:,0,:] + x.T[:ns_sub,:nH,None]
    y_plt = all_ell[:,:,1,:] + y.T[:ns_sub,:nH,None]
    # return plt.plot(x_plt.reshape(-1,n_pts).T, y_plt.reshape(-1,n_pts).T, color="blue", label="Terminal set", linewidth=0.2)
    return x_plt, y_plt


def plot_reachable_eps(ax, x, y, ci_list,remove_list, color):
    # with open(filepath, "rb") as input_data_file:
    #     reachable_hull_points = pickle.load(input_data_file)
    reachable_hull_points = [np.stack([x[i], y[i]], axis=-1) for i in range(len(x))]
    # color = "lightcoral" 
    # for i in range(len(x)-1):
    #     hull = ConvexHull(np.concatenate([reachable_hull_points[i], reachable_hull_points[i + 1]]))
    #     stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
    #     plt.fill(
    #         hull.points[stack_vertices, 0],
    #         hull.points[stack_vertices, 1],
    #         alpha=1,
    #         color=color,
    #         lw=0,
    #     )

    max_rows = max(A.shape[0] for A in reachable_hull_points)
    # Pad each array to (10,2)
    padded_arrays = np.array([
        np.pad(A, ((0, max_rows - A.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        for A in reachable_hull_points
    ])
    ell_x, ell_y = plot_ellipses(ax, padded_arrays[:,:,0], padded_arrays[:,:,1], ci_list)

    # for i in range(0,len(x)-1):
    #     ell_pts_i = np.stack([ell_x[:,i,:].reshape(-1), ell_y[:,i,:].reshape(-1)]).T
    #     remove_list.append(ax.plot(ell_pts_i[:,0], ell_pts_i[:,1], color=color, lw=0.2, alpha=0.5))
    for i in range(0,len(x)-2):
        ell_pts_i = np.stack([ell_x[:,i,:].reshape(-1), ell_y[:,i,:].reshape(-1)]).T
        ell_pts_ip1 = np.stack([ell_x[:,i+1,:].reshape(-1), ell_y[:,i+1,:].reshape(-1)]).T
        hull = ConvexHull(np.concatenate([ell_pts_i[~np.isnan(ell_pts_i).any(axis=1)], ell_pts_ip1[~np.isnan(ell_pts_ip1).any(axis=1)]]))
        stack_vertices = np.hstack([hull.vertices, hull.vertices[0]])
        remove_list.append(plt.fill(
            hull.points[stack_vertices, 0],
            hull.points[stack_vertices, 1],
            alpha=1,
            color=color,
            lw=0,
            zorder=0.2,
        ))

    return remove_list

for n_sample in range(3,max_samples):
    remove_list = []
    x = [input_gpmpc_h[:n_sample,0,0,0] for input_gpmpc_h in input_gpmpc_data]
    y = [input_gpmpc_h[:n_sample,0,0,1] for input_gpmpc_h in input_gpmpc_data]
    for i in range(len(x)):
        x[i] = np.array(x[i])
        y[i] = np.array(y[i])
    
    # Plot the finte sample reachable set
    tilde_eps_list, ci_list = get_reachable_set_ball(params, get_eps_vec(n_sample))
    ci_list = np.stack(tilde_eps_list)[:,-1]
    remove_list = plot_reachable_eps(ax, x,y, ci_list, remove_list, color=color_finite_reachable_set)
    
    # Plot the samples
    remove_list.append(ax.plot(x,y, lw=0.8, alpha=0.2, color="tab:blue"))
    x_all = np.vstack(x)
    y_all = np.vstack(y)
    x_diff = np.abs(x_all - true_traj[0][:,[0]])
    y_diff = np.abs(y_all - true_traj[0][:,[1]])
    x_max = np.max(x_diff, axis=0)
    y_max = np.max(y_diff, axis=0)
    # samples_max_distance = np.max(np.maximum(x_diff, y_diff, out=x_diff), axis=0)
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["text.usetex"] =True
    plt.rcParams["font.family"] = "serif"
    remove_list.append(ax.plot(true_traj[0][:,0], true_traj[0][:,1], color="black", lw=2, ls="--", label=r"True trajectory"))
    if False:
        samples_max_distance = np.sqrt(x_max**2 + y_max**2)
        closet_idx = np.argmin(samples_max_distance)
        print(np.min(np.max(np.maximum(x_diff, y_diff, out=x_diff), axis=0)), samples_max_distance[closet_idx], closet_idx)
        # np.diff(x_all - true_traj[0][:,[0]], axis=1)
        # closest_sample = np.argmin(np.linalg.norm(x_all - true_traj[0][:,0], axis=1))

        remove_list.append(ax.plot(x_all[:,closet_idx], y_all[:,closet_idx], color="tab:orange",lw=1.2, label=r"$\epsilon$-close trajectory"))
        print(np.max([np.max(np.abs(x_all[:,closet_idx]-true_traj[0][:,0])), 
                    np.max(np.abs(y_all[:,closet_idx]-true_traj[0][:,1]))]))
    # plt.savefig("temp.png")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 3, 1, 2]  # New order of legend items
    # order = [0, 1, 2,3,4]
    plt.legend([handles[i] for i in order], [labels[i] for i in order],
            #    ncol=2,
                labelspacing=0.4,
                # handlelength=1,
                handletextpad=0.5,
                borderpad=0.3
                )
    # plt.legend(ncol=2)

    # Turn off all ticks
    # ax.tick_params(
    #     axis='both',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     left=False,
    #     right=False,
    #     labelbottom=False,
    #     labelleft=False
    # )
    plt.tight_layout(pad=0.2)
    writer.grab_frame()
    for t in remove_list:
        if type(t) is list:
            for tt in t:
                tt.remove()
        else:
            t.remove()
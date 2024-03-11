import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import dill as pickle


class Visualizer:
    def __init__(self, params, path, agent):
        self.params = params
        self.state_traj = []
        self.input_traj = []
        self.mean_state_traj = []
        self.true_state_traj = []
        self.physical_state_traj = []
        self.save_path = path
        self.agent = agent
        if self.params["visu"]["show"]:
            self.initialize_plot_handles(path)

    def initialize_plot_handles(self, path):
        fig_gp, ax = plt.subplots(figsize=(16 / 2.4, 16 / 2.4))
        # fig_gp.tight_layout(pad=0)
        ax.grid(which="both", axis="both")
        ax.minorticks_on()
        ax.set_xlabel("theta")
        ax.set_ylabel("theta_dot")
        ax.add_line(plt.Line2D([-0.3, 2.4], [2.5, 2.5], color="red", linestyle="--"))
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_aspect('equal', 'box')
        ax.set_xlim(-0.3, 2.4)
        ax.set_ylim(-3, 3)
        fig_dyn, ax2 = plt.subplots()  # plt.subplots(2,2)

        # ax2.set_aspect('equal', 'box')
        self.f_handle = {}
        self.f_handle["gp"] = fig_gp
        self.f_handle["dyn"] = fig_dyn
        # self.plot_contour_env("dyn")

        # Move it to visu
        self.writer_gp = self.get_frame_writer()
        self.writer_dyn = self.get_frame_writer()
        self.writer_dyn.setup(fig_dyn, path + "/video_dyn.mp4", dpi=200)
        self.writer_gp.setup(fig_gp, path + "/video_gp.mp4", dpi=200)

    def get_frame_writer(self):
        # FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(
            title="Movie Test", artist="Matplotlib", comment="Movie support!"
        )
        writer = manimation.FFMpegWriter(
            fps=10, codec="libx264", metadata=metadata
        )  # libx264 (good quality), mpeg4
        return writer

    def pendulum_discrete_dyn(self, X1_k, X2_k, U_k):
        """_summary_

        Args:
            x (_type_): _description_
            u (_type_): _description_
        """
        m = 1
        l = 1
        g = 10
        dt = self.params["optimizer"]["dt"]
        X1_kp1 = X1_k + X2_k * dt
        X2_kp1 = X2_k - g * np.sin(X1_k) * dt / l + U_k * dt / (l * l)
        return X1_kp1, X2_kp1

    def propagate_true_dynamics(self, x_init, U):
        x1_list = []
        x2_list = []
        X1_k = x_init[0]
        X2_k = x_init[1]
        x1_list.append(X1_k.item())
        x2_list.append(X2_k.item())
        for ele in range(U.shape[0]):
            X1_kp1, X2_kp1 = self.pendulum_discrete_dyn(X1_k, X2_k, U[ele])
            x1_list.append(X1_kp1.item())
            x2_list.append(X2_kp1.item())
            X1_k = X1_kp1.copy()
            X2_k = X2_kp1.copy()
        return x1_list, x2_list

    def propagate_mean_dyn(self, x_init, U):
        self.agent.Dyn_gp_model["y1"].eval()
        self.agent.Dyn_gp_model["y2"].eval()
        x1_list = []
        x2_list = []
        X1_k = x_init[0]
        X2_k = x_init[1]
        x1_list.append(X1_k.item())
        x2_list.append(X2_k.item())
        for ele in range(U.shape[0]):
            y1 = self.agent.Dyn_gp_model["y1"](
                torch.Tensor([[X1_k, X2_k, U[ele]]]).cuda()
            ).mean.detach()[0]
            y2 = self.agent.Dyn_gp_model["y2"](
                torch.Tensor([[X1_k, X2_k, U[ele]]]).cuda()
            ).mean.detach()[0]
            X1_kp1, X2_kp1 = y1[0].cpu(), y2[0].cpu()
            del y1, y2
            x1_list.append(X1_kp1.item())
            x2_list.append(X2_kp1.item())
            X1_k = X1_kp1.clone()
            X2_k = X2_kp1.clone()
        return x1_list, x2_list

    def plot_receding_pendulum_traj(self):
        rm = []
        ax = self.f_handle["gp"].axes[0]
        physical_state_traj = np.vstack(self.physical_state_traj)
        ax.plot(
            physical_state_traj[:, 0],
            physical_state_traj[:, 1],
            color="tab:blue",
            label="real",
            linestyle="-",
        )
        X = self.state_traj[-1]
        U = self.input_traj[-1]
        rm.append(ax.plot(X[:, 0::2], X[:, 1::2], linestyle="-"))
        pred_true_state = np.vstack(self.true_state_traj[-1])
        rm.append(
            ax.plot(
                pred_true_state[:, 0],
                pred_true_state[:, 1],
                color="black",
                label="true",
                linestyle="-",
            )
        )
        pred_mean_state = np.vstack(self.mean_state_traj[-1])
        rm.append(
            ax.plot(
                pred_mean_state[:, 0],
                pred_mean_state[:, 1],
                color="black",
                label="mean",
                linestyle="--",
            )
        )
        return rm

    def remove_temp_objects(self, temp_obj):
        for t in temp_obj:
            if type(t) is list:
                for tt in t:
                    tt.remove()
            else:
                t.remove()

    def plot_pendulum_traj(self, X, U):
        plt.close()
        plt.plot(
            X[:, 0::2], X[:, 1::2]
        )  # , label = [i for i in range(self.params["agent"]["num_dyn_samples"])])
        # plt.legend([i for i in range(self.params["agent"]["num_dyn_samples"])])
        plt.xlabel("theta")
        plt.ylabel("theta_dot")
        x1_true, x2_true = self.propagate_true_dynamics(X[0, 0:2], U)
        plt.plot(x1_true, x2_true, color="black", label="true", linestyle="--")
        x1_mean, x2_mean = self.propagate_mean_dyn(X[0, 0:2], U)
        print("x1_mean", x1_mean, x2_mean)
        plt.plot(x1_mean, x2_mean, color="black", label="mean", linestyle="-.")
        plt.legend()
        plt.grid()
        plt.savefig("pendulum.png")

    def record_out(self, x_curr, X, U, pred_true_state, pred_mean_state):
        self.physical_state_traj.append(x_curr)
        self.state_traj.append(X)
        self.input_traj.append(U)
        self.true_state_traj.append(pred_true_state)
        self.mean_state_traj.append(pred_mean_state)

    def record(self, x_curr, X, U):
        self.physical_state_traj.append(x_curr)
        self.state_traj.append(X)
        self.input_traj.append(U)
        x1_true, x2_true = self.propagate_true_dynamics(X[0, 0:2], U)
        # x1_mean, x2_mean = self.propagate_mean_dyn(X[0,0:2], U)
        self.true_state_traj.append(torch.Tensor([x1_true, x2_true]).transpose(0, 1))
        # self.mean_state_traj.append(torch.Tensor([x1_mean, x2_mean]).transpose(0,1))

    def save_data(self):
        data_dict = {}
        data_dict["state_traj"] = self.state_traj
        data_dict["input_traj"] = self.input_traj
        data_dict["mean_state_traj"] = self.mean_state_traj
        data_dict["true_state_traj"] = self.true_state_traj
        data_dict["physical_state_traj"] = self.physical_state_traj
        a_file = open(self.save_path + "/data.pkl", "wb")
        # data_dict["meas_traj"] = self.meas_traj
        # data_dict["player_train_pts"] = self.player_train_pts
        # data_dict["player_model"] = self.player_model
        # data_dict["opti_path_list"] = self.opti_path_list
        # data_dict["utility_minimizer_list"] = self.utility_minimizer_list
        # data_dict["num_safe_nodes_list"] = self.num_safe_nodes_list
        # data_dict["iteration_time"] = self.iteration_time
        # a_file = open(self.save_path + "/data.pkl", "wb")
        pickle.dump(data_dict, a_file)
        a_file.close()

    def extract_data(self):
        a_file = open(self.save_path + "/data.pkl", "rb")
        data_dict = pickle.load(a_file)
        self.state_traj = data_dict["state_traj"]
        self.input_traj = data_dict["input_traj"]
        self.mean_state_traj = data_dict["mean_state_traj"]
        self.true_state_traj = data_dict["true_state_traj"]
        self.physical_state_traj = data_dict["physical_state_traj"]
        a_file.close()

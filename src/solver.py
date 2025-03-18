import timeit

import casadi as ca
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from acados_template import AcadosOcpSolver, AcadosSimSolver

from src.utils.ocp import export_dempc_ocp
from src.utils.termcolor import bcolors


# The class below is an optimizer class,
# it takes in GP function, x_g and rest are parameters
class DEMPC_solver(object):
    def __init__(self, params) -> None:
        self.params = params
        self.ocp = export_dempc_ocp(params)
        self.name_prefix = (
            "env_" + str(params["env"]["name"]) + "_i_" + str(params["env"]["i"]) + "_"
        )
        self.ocp_solver = AcadosOcpSolver(
            self.ocp, json_file=self.name_prefix + "acados_ocp_sempc.json"
        )
        self.ocp_solver.store_iterate(self.name_prefix + "ocp_initialization.json")

        self.H = params["optimizer"]["H"]
        self.max_sqp_iter = params["optimizer"]["SEMPC"]["max_sqp_iter"]
        self.tol_nlp = params["optimizer"]["SEMPC"]["tol_nlp"]
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.pos_dim = 1

        self.x_h = np.zeros((self.H, self.nx * self.params["agent"]["num_dyn_samples"]))
        self.u_h = np.zeros((self.H, self.nu))  # u_dim

        # computation of tightenings
        L = self.params["agent"]["tight"]["Lipschitz"]
        dyn_eps = self.params["agent"]["tight"]["dyn_eps"]
        w_bound = self.params["agent"]["tight"]["w_bound"]
        var_eps = dyn_eps + w_bound
        P_inv = np.linalg.inv(self.params["optimizer"]["terminal_tightening"]["P"])
        K = np.array(self.params["optimizer"]["terminal_tightening"]["K"])
        tilde_eps_0 = 0
        tightenings = np.sqrt(np.diag(P_inv)*tilde_eps_0)
        u_tight = np.sqrt(np.diag(K@P_inv@K.T)*tilde_eps_0)
        self.tilde_eps_list = []
        self.tilde_eps_list.append(np.concatenate([tightenings.tolist(), u_tight.tolist(), [tilde_eps_0]]))
        self.ci_list = []
        tilde_eps_i = 0
        for stage in range(1, self.H + 1):
            c_i = np.power(L, stage - 1) * var_eps + 2 * dyn_eps * np.sum(
                np.power(L, np.arange(0, stage - 1))
            )  # arange has inbuild -1 in [sstart, end-1]
            if stage == self.H:
                self.tilde_eps_list.append([c_i]*(self.nx+self.nu+1))
                self.ci_list.append(c_i)
            else:
                tilde_eps_i += c_i
                # box constraints tightenings
                tightenings = np.sqrt(np.diag(P_inv)*tilde_eps_i)
                u_tight = np.sqrt(np.diag(K@P_inv@K.T)*tilde_eps_i)
                print(f"u_tight_{stage} = {u_tight}")
                self.tilde_eps_list.append(np.concatenate([tightenings.tolist(), u_tight.tolist(), [tilde_eps_i]]))
                self.ci_list.append(c_i)
            print(f"tilde_eps_{stage} = {self.tilde_eps_list[-1]}")

    def solve(self, player, plot_pendulum=False):
        w = np.ones(self.H + 1) * self.params["optimizer"]["w"]
        xg = np.ones((self.H + 1, self.pos_dim)) * player.get_next_to_go_loc()
        K = np.array(self.params["optimizer"]["terminal_tightening"]["K"])
        x_equi = np.array(self.params["env"]["goal_state"])
        ns = self.params["agent"]["num_dyn_samples"]
        for sqp_iter in range(self.max_sqp_iter):
            x_h_old = self.x_h.copy()
            u_h_old = self.u_h.copy()
            for stage in range(self.H):
                # current stage values
                self.x_h[stage, :] = self.ocp_solver.get(stage, "x")
                self.u_h[stage, :] = self.ocp_solver.get(stage, "u")

            x_h_e = self.ocp_solver.get(self.H, "x")

            x_diff = np.linalg.norm(self.x_h - x_h_old) / (
                np.linalg.norm(x_h_old) + 1e-6
            )
            u_diff = np.linalg.norm(self.u_h - u_h_old) / (
                np.linalg.norm(u_h_old) + 1e-6
            )
            print(f"x_diff = {x_diff}, u_diff = {u_diff}")

            if (
                sqp_iter >= 1
                and status == 0
                and x_diff < self.tol_nlp
                and u_diff < self.tol_nlp
            ):
                print("Converged")
                break

            # create model with updated data
            player.train_hallucinated_dynGP(sqp_iter)
            if self.params["agent"]["feedback"]:
                batch_x_hat = player.get_batch_x_hat_u_diff(self.x_h, (x_equi-self.x_h.reshape(self.H, ns, -1))@K.T + np.tile(self.u_h[:, None,:], (ns,1)))
                # sample the gradients
                gp_val, y_grad, u_grad = player.dyn_fg_jacobians(batch_x_hat, sqp_iter)
                
                y_grad = y_grad - u_grad @ K
            else:
                batch_x_hat = player.get_batch_x_hat(self.x_h, self.u_h)
                # sample the gradients
                gp_val, y_grad, u_grad = player.dyn_fg_jacobians(batch_x_hat, sqp_iter)

            del batch_x_hat

            for stage in range(self.H):
                p_lin = np.empty(0)
                for i in range(self.params["agent"]["num_dyn_samples"]):
                    p_lin = np.concatenate(
                        [
                            p_lin,
                            y_grad[i, :, stage, :].reshape(-1),
                            u_grad[i, :, stage, :].reshape(-1),
                            self.x_h[stage, i * self.nx : self.nx * (i + 1)],
                            gp_val[i, :, stage, :].reshape(-1),
                        ]
                    )

                # # computation of tightenings
                # if stage == 0:
                #     tilde_eps_i = 0
                # else:
                #     # i = stage-1
                #     c_i = np.power(L, stage - 1) * var_eps + 2 * dyn_eps * np.sum(
                #         np.power(L, np.arange(0, stage - 1))
                #     )  # arange has inbuild -1 in [sstart, end-1]

                #     tilde_eps_i += c_i
                #     print(f"tilde_eps_{stage} = {tilde_eps_i}")
                p_lin = np.hstack(
                    [
                        p_lin,
                        self.u_h[stage],
                        xg[stage],
                        w[stage],
                        self.tilde_eps_list[stage],
                    ]
                )
                self.ocp_solver.set(stage, "p", p_lin)

            residuals = self.ocp_solver.get_residuals(recompute=True)
            print("residuals (before solve)", residuals)

            t_0 = timeit.default_timer()
            status = self.ocp_solver.solve()
            t_1 = timeit.default_timer()
            print("Time taken for QP solve", t_1 - t_0)
            # self.ocp_solver.print_statistics()
            # print("statistics", self.ocp_solver.get_stats("statistics"))
            print("cost", self.ocp_solver.get_cost())
            residuals = self.ocp_solver.get_residuals(recompute=True)
            print("residuals (after solve)", residuals)

            if status != 0:
                print(
                    bcolors.FAIL
                    + f"acados returned status {status} in closed loop solve"
                )
                break

            if plot_pendulum:
                self.plot_iterates_pendulum(sqp_iter, player, self.x_h, x_h_e, self.u_h)

        return status

    def get_solution(self):
        nx = self.ocp_solver.acados_ocp.model.x.size()[0]
        nu = self.ocp_solver.acados_ocp.model.u.size()[0]
        X = np.zeros((self.H + 1, nx))
        U = np.zeros((self.H, nu))
        Sl = np.zeros((self.H + 1))

        # get data
        for i in range(self.H):
            X[i, :] = self.ocp_solver.get(i, "x")
            U[i, :] = self.ocp_solver.get(i, "u")
            # Sl[i] = self.ocp_solver.get(i, "sl")

        X[self.H, :] = self.ocp_solver.get(self.H, "x")
        return X, U, Sl

    def shift_solution(self, X, U, Sl):
        for i in range(self.H - 1):
            self.ocp_solver.set(i, "x", X[i + 1, :])
            self.ocp_solver.set(i, "u", U[i + 1, :])
        self.ocp_solver.set(self.H - 1, "x", X[self.H, :])

    def initialize_solution(self, X, U, Sl):
        for i in range(self.H - 1):
            self.ocp_solver.set(i, "x", X[i, :])
            self.ocp_solver.set(i, "u", U[i, :])
        self.ocp_solver.set(self.H, "x", X[self.H, :])

    def get_and_shift_solution(self):
        X, U, Sl = self.get_solution()
        self.shift_solution(X, U, Sl)
        return X, U, Sl

    def get_solver_status():
        return None

    def plot_iterates_pendulum(
        self, sqp_iter, player, x_h, x_h_e, u_h, n_supersample=10
    ):
        n_test_points = (self.H - 1) * n_supersample + 1
        X_input = torch.zeros(
            (
                self.params["agent"]["num_dyn_samples"],
                self.nx,
                n_test_points,
                self.nx + self.nu,
            )
        )
        X_input_orig = torch.zeros(
            (
                self.params["agent"]["num_dyn_samples"],
                self.nx,
                self.H,
                self.nx + self.nu,
            )
        )
        x_input_interp = []
        x_input_len = []
        x_input_arr = []
        for s in range(self.params["agent"]["num_dyn_samples"]):
            x_input = np.vstack((x_h[:, 2 * s], x_h[:, 2 * s + 1], u_h.flatten())).T
            x_input_arr.append(x_input)
            # supersample points between points in x_input
            x_input_interp_s = np.zeros((n_test_points, self.nx + self.nu))
            x_input_len_s = np.zeros(n_test_points)
            for i in range(self.H - 1):
                x_input_len_s[1 + i * n_supersample : 1 + (i + 1) * n_supersample] = (
                    np.linalg.norm(
                        x_input[i + 1, 0 : self.nx] - x_input[i, 0 : self.nx]
                    )
                    / n_supersample
                )
                x_input_interp_s[i * n_supersample : (i + 1) * n_supersample, :] = (
                    np.linspace(
                        x_input[i, :],
                        x_input[i + 1, :],
                        n_supersample,
                        endpoint=False,
                    )
                )
            x_input_interp_s[-1, :] = x_input[-1, :]

            X_input_orig[s, :, :, :] = torch.tensor(x_input)
            X_input[s, :, :, :] = torch.tensor(x_input_interp_s)
            x_input_interp.append(x_input_interp_s)
            x_input_len.append(x_input_len_s)

        with torch.no_grad(), gpytorch.settings.observation_nan_policy(
            "mask"
        ), gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ), gpytorch.settings.cholesky_jitter(
            float_value=self.params["agent"]["Dyn_gp_jitter"],
            double_value=self.params["agent"]["Dyn_gp_jitter"],
            half_value=self.params["agent"]["Dyn_gp_jitter"],
        ):
            predictions = player.model_i(X_input)
            mean = predictions.mean.detach().numpy()
            # var = predictions.variance.detach().numpy()
            lower, upper = predictions.confidence_region()
            lower = lower.detach().numpy()
            upper = upper.detach().numpy()
            # samples = predictions.sample().detach().numpy()
            # samples = player.model_i_samples

        predictions_call = player.model_i_call
        mean_call = predictions_call.mean.detach().numpy()
        # var = predictions.variance.detach().numpy()
        lower_call, upper_call = predictions_call.confidence_region()
        lower_call = lower_call.detach().numpy()
        upper_call = upper_call.detach().numpy()
        # samples = predictions.sample().detach().numpy()
        samples_call = player.model_i_samples.detach().numpy()

        # plot GP model
        h_plot_arr = []
        fig, ax = plt.subplots(1, 3)
        for s in range(self.params["agent"]["num_dyn_samples"]):
            x_input_plot = np.cumsum(x_input_len[s])
            x_input_plot /= x_input_plot[-1] + 1e-10
            h_plot = ax[0].plot(x_input_plot, mean[s, 0, :, 0], label=f"mean {s}")
            h_plot_arr.append(h_plot)
            # ax[0].fill_between(x_input_plot, mean[s,0,:,0]-2*np.sqrt(var[s,0,:,0]), mean[s,0,:,0]+2*np.sqrt(var[s,0,:,0]), alpha=0.2, color=h_plot[0].get_color())
            ax[0].fill_between(
                x_input_plot,
                lower[s, 0, :, 0],
                upper[s, 0, :, 0],
                alpha=0.2,
                color=h_plot[0].get_color(),
            )
            # ax[0].plot(x_input_plot, samples[s,0,:,0], label=f"sample {s}")
            ax[0].plot(
                x_input_plot[0::n_supersample],
                samples_call[s, 0, :, 0],
                label=f"sample {s}",
                color=h_plot[0].get_color(),
                linestyle="None",
                marker="x",
            )
            # vertical line
            # for x_vline in x_input_plot[0::n_supersample]:
            #     ax[0].axvline(x=x_vline, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

            ax[1].plot(
                np.hstack((x_h[:, s * self.nx], x_h_e[s * self.nx])),
                np.hstack((x_h[:, s * self.nx + 1], x_h_e[s * self.nx + 1])),
                "-d",
                label="x_h",
                color=h_plot[0].get_color(),
            )

        # horizontal line at theta_dot
        ax[1].axhline(
            y=self.params["optimizer"]["x_max"][1],
            color="k",
            linestyle="-",
            linewidth=0.5,
            alpha=0.3,
        )

        # ax[0].legend()
        ax[0].set_title("GP models")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("g(x)")
        ax[0].set_xlim([0.0, 1.0])
        ax[0].set_ylim([-0.1, 1.8])

        # ax[1].legend()
        ax[1].set_title("Trajectories")
        ax[1].set_xlabel("theta")
        ax[1].set_ylabel("theta_dot")
        ax[1].set_xlim([-0.2, 2.2])
        ax[1].set_ylim([-0.2, 2.7])

        # ax[2].plot(u_h)
        ax[2].stairs(u_h.flatten(), np.linspace(0, self.H, self.H + 1))

        # plt.show()
        plt.savefig(f"pendulum_{sqp_iter}.png", dpi=600)
        plt.close()

        # 3d scatter plot of all input points
        # X_input_orig_past.append(X_input_orig)
        # for s in range(self.params["agent"]["num_dyn_samples"]):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     for i,X in enumerate(X_input_orig_past):
        #         # print(f"i={i}, alpha={1.0-0.9**(i+1)}")
        #         ax.scatter(X[s,0,:,0],X[s,0,:,1],X[s,0,:,2], alpha=1/len(X_input_orig_past)*(i+1), marker='x')
        #     ax.set_xlabel('theta_dot')
        #     ax.set_ylabel('theta')
        #     ax.set_zlabel('u')
        #     # plt.show()
        #     fig.savefig(f"pendulum_scatter_s{s}_i{sqp_iter}.png", dpi=600)
        #     plt.close()

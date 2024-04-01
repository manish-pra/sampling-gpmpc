import timeit

import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt
from acados_template import AcadosOcpSolver, AcadosSimSolver

from src.utils.ocp import export_dempc_ocp


# The class below is an optimizer class,
# it takes in GP function, x_g and rest are parameters
class DEMPC_solver(object):
    def __init__(self, params) -> None:
        self.params = params
        ocp = export_dempc_ocp(params)
        self.name_prefix = (
            "env_" + str(params["env"]["name"]) + "_i_" + str(params["env"]["i"]) + "_"
        )
        self.ocp_solver = AcadosOcpSolver(
            ocp, json_file=self.name_prefix + "acados_ocp_sempc.json"
        )
        self.ocp_solver.store_iterate(self.name_prefix + "ocp_initialization.json")

        self.H = params["optimizer"]["H"]
        self.max_sqp_iter = params["optimizer"]["SEMPC"]["max_sqp_iter"]
        self.tol_nlp = params["optimizer"]["SEMPC"]["tol_nlp"]
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.pos_dim = 1

    def initilization(self, sqp_iter, x_h, u_h):
        for stage in range(self.H):
            # current stage values
            x_h[stage, :] = self.ocp_solver.get(stage, "x")
            u_h[stage, :] = self.ocp_solver.get(stage, "u")
        x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
        if sqp_iter == 0:
            x_h_old = x_h.copy()
            u_h_old = u_h.copy()
            if (
                self.params["algo"]["type"] == "ret_expander"
                or self.params["algo"]["type"] == "MPC_expander"
            ):
                u_h_old[:, -self.x_dim :] = x_h_old[:-1, : self.x_dim].copy()
            # initialize the first SQP iteration.
            for stage in range(self.H):
                if stage < (self.H - self.Hm):
                    # current stage values
                    x_init = x_h_old[stage + self.Hm, :].copy()
                    u_init = u_h_old[stage + self.Hm, :].copy()
                    x_init[-1] = (
                        x_h_old[stage + self.Hm, -1] - x_h_old[self.Hm, -1]
                    ).copy()
                    self.ocp_solver.set(stage, "x", x_init)
                    self.ocp_solver.set(stage, "u", u_init)
                    x_h[stage, :] = x_init.copy()
                    u_h[stage, :] = u_init.copy()
                    half_time = x_init[-1].copy()
                else:
                    dt = (1.0 - half_time) / self.Hm
                    x_init = x_h_old[self.H, :].copy()  # reached the final state
                    x_init[-1] = half_time + dt * (stage - self.Hm)
                    z_init = x_init[0 : self.x_dim]
                    if (
                        self.params["algo"]["type"] == "ret_expander"
                        or self.params["algo"]["type"] == "MPC_expander"
                    ):
                        u_init = np.concatenate([np.array([0.0, 0.0, dt]), z_init])
                    else:
                        u_init = np.array([0.0, 0.0, dt])
                    self.ocp_solver.set(stage, "x", x_init)
                    self.ocp_solver.set(stage, "u", u_init)
                    x_h[stage, :] = x_init.copy()
                    u_h[stage, :] = u_init.copy()
            self.ocp_solver.set(self.H, "x", x_init)
            x_init[-1] = half_time + dt * (self.H - self.Hm)
            x_h[self.H, :] = x_init.copy()
        return x_h, u_h

    def solve(self, player):
        # self.ocp_solver.store_iterate(self.name_prefix + 'ocp_initialization.json', overwrite=True)
        x_h = np.zeros((self.H, self.nx * self.params["agent"]["num_dyn_samples"]))
        u_h = np.zeros((self.H, self.nu))  # u_dim
        # w = 1e-3*np.ones(self.H+1)
        # w[int(self.H/2)] = self.params["optimizer"]["w"]
        w = np.ones(self.H + 1) * self.params["optimizer"]["w"]
        xg = np.ones((self.H + 1, self.pos_dim)) * player.get_next_to_go_loc()

        for sqp_iter in range(self.max_sqp_iter):
            self.ocp_solver.options_set("rti_phase", 1)
            # x_h, u_h = self.initilization(sqp_iter, x_h, u_h)
            for stage in range(self.H):
                # current stage values
                x_h[stage, :] = self.ocp_solver.get(stage, "x")
                u_h[stage, :] = self.ocp_solver.get(stage, "u")
            # x_h[self.H, :] = self.ocp_solver.get(self.H, "x") --> not needed, no corresponding u

            # create model with updated data
            player.train_hallucinated_dynGP(sqp_iter)
            batch_x_hat = player.get_batch_x_hat(x_h, u_h)
            # sample the gradients
            gp_val, y_grad, u_grad = player.dyn_fg_jacobians(batch_x_hat, sqp_iter)
            del batch_x_hat
            # gp_val, gp_grad = player.get_gp_sensitivities(np.hstack([x_h, u_h]), "mean", 0)
            # gp_val, gp_grad = player.get_true_gradient(np.hstack([x_h,u_h]))
            for stage in range(self.H):
                p_lin = np.empty(0)
                # TODO (manish) remove num_dyn for loop --> use dim manipulation
                for i in range(self.params["agent"]["num_dyn_samples"]):
                    p_lin = np.concatenate(
                        [
                            p_lin,
                            y_grad[i, :, stage, :].reshape(-1),
                            u_grad[i, :, stage, :].reshape(-1),
                            x_h[stage, i * self.nx : self.nx * (i + 1)],
                            gp_val[i, :, stage, :].reshape(-1),
                        ]
                    )
                p_lin = np.hstack([p_lin, u_h[stage], xg[stage], w[stage]])
                self.ocp_solver.set(stage, "p", p_lin)
            status = self.ocp_solver.solve()

            self.ocp_solver.options_set("rti_phase", 2)
            t_0 = timeit.default_timer()
            status = self.ocp_solver.solve()
            t_1 = timeit.default_timer()
            print("Time taken for SQP iteration", t_1 - t_0)
            # self.ocp_solver.print_statistics()
            print("cost", self.ocp_solver.get_cost())
            residuals = self.ocp_solver.get_residuals()

            # print("statistics", self.ocp_solver.get_stats("statistics"))
            if max(residuals) < self.tol_nlp:
                print("Residual less than tol", max(residuals), " ", self.tol_nlp)
                break
            # if self.ocp_solver.status != 0:
            #     print("acados returned status {} in closed loop solve".format(
            #         self.ocp_solver.status))
            #     self.ocp_solver.reset()
            #     self.ocp_solver.load_iterate(self.name_prefix + 'ocp_initialization.json')

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

    def get_solver_status():
        return None

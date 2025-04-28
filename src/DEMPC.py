# This is the algorithm file. It will be responsible to call environement,
# collect measurement, setup of MPC problem, call model, solver, etc.
import numpy as np
import torch

from src.solver import DEMPC_solver
from src.utils.initializer import get_players_initialized
from src.utils.termcolor import bcolors
import timeit


class DEMPC:
    def __init__(self, params, visu, agent) -> None:
        self.dempc_solver = DEMPC_solver(params)
        self.visu = visu
        self.params = params
        self.iter = -1
        self.data = {}
        self.flag_reached_xt_goal = False
        self.H = self.params["optimizer"]["H"]
        self.n_order = params["optimizer"]["order"]
        self.nx = self.params["agent"]["dim"]["nx"]
        self.agent = agent
        self.visu.tilde_eps_list = self.dempc_solver.tilde_eps_list
        self.visu.ci_list = self.dempc_solver.ci_list

    def dempc_main(self):
        """_summary_ Responsible for initialization, logic for when to collect sample vs explore"""
        # while not self.agent.infeasible:
        run = True
        # self.agent.feasible = True
        while run:
            run = self.receding_horizon()
            print("while loop")
        # print("Number of samples", self.agent.Cx_X_train.shape)

    def receding_horizon(self):
        print("Receding Horizon")
        nx = self.params["agent"]["dim"]["nx"]
        for i in range(self.params["common"]["num_MPC_itrs"]):
            self.agent.mpc_iteration(i)
            torch.cuda.empty_cache()
            x_curr = self.agent.current_state[: self.nx].reshape(self.nx)
            if torch.is_tensor(x_curr):
                x_curr = x_curr.numpy()
            st_curr = np.array(
                x_curr.tolist() * self.params["agent"]["num_dyn_samples"]
            )
            # if i == 0:
            #     x_hist, u_hist = self.agent.env_model.traj_initialize(x_curr)
            #     self.dempc_solver.initialize_solution(
            #         np.tile(x_hist, (1, self.params["agent"]["num_dyn_samples"])),
            #         u_hist,
            #         0,
            #     )
            X, U = self.one_step_planner(st_curr)

            # check for uncertainity and online learn the data
            # Initialize the next state at the point nearest to the goal
            # propagate the trajectory
            X_true_traj = self.agent.env_model.propagate_true_dynamics(X[0, 0:nx], U)
            # self.agent.get_posterior_uncertainity(X_true_traj, U)
            # let's jump twice
            jump_idx = 0
            state_input = torch.hstack(
                [torch.from_numpy(X_true_traj[jump_idx][: self.nx]), U[jump_idx]]
            ).reshape(1, -1)

            if self.params["common"]["active_learning"]:
                if i % 2 == 0:
                    Y_data = self.agent.env_model.get_prior_data(state_input)
                    if self.params["common"]["use_cuda"]:
                        self.agent.online_learnt_datapoints(state_input.cuda(), Y_data)
                    else:
                        self.agent.online_learnt_datapoints(state_input, Y_data)

            state_kp1 = self.agent.env_model.discrete_dyn(state_input)
            self.agent.update_current_state(state_kp1)
            # propagate the agent to the next state
            # forward sampling and reject dynamics c_i away from the projection
            if self.params["common"]["dynamics_rejection"]:
                self.agent.prepare_dynamics_set(X, U, state_kp1)

            print(
                bcolors.green + "Reached:",
                i,
                " ",
                state_input,
                " ",
                bcolors.ENDC,
            )
        return False

    def one_step_planner(self, st_curr):
        """_summary_: Plans going and coming back all in one trajectory plan
        Input: current location, end location, dyn, etc.
        Process: Solve the NLP and simulate the system until the measurement collection point
        Output: trajectory
        """
        print(bcolors.OKCYAN + "Solving Constrints" + bcolors.ENDC)
        self.dempc_solver.ocp_solver.set(0, "lbx", st_curr)
        self.dempc_solver.ocp_solver.set(0, "ubx", st_curr)

        # set objective as per desired goal
        t_0 = timeit.default_timer()
        solver_status = self.dempc_solver.solve(self.agent)
        t_1 = timeit.default_timer()
        dt = t_1 - t_0
        print("Time to solve", dt)

        if self.params["agent"]["shift_soln"]:
            X, U, Sl = self.dempc_solver.get_and_shift_solution()
        else:
            X, U, Sl = self.dempc_solver.get_solution()
        #
        self.visu.record(st_curr, X, U, dt,record_gp_model=False)
        print("X", X)
        print("U", U)

        # self.visu.plot_pendulum_traj(X,U)
        return torch.from_numpy(X).float(), torch.from_numpy(U).float()

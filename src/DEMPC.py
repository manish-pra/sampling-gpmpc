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
        self.x_dim = params["optimizer"]["x_dim"]
        self.state_dim = self.x_dim
        self.agent = agent

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
        for i in range(self.params["common"]["num_MPC_itrs"]):
            self.agent.mpc_iteration(i)
            torch.cuda.empty_cache()
            x_curr = self.agent.current_state[: self.state_dim].reshape(self.state_dim)
            if torch.is_tensor(x_curr):
                x_curr = x_curr.numpy()
            st_curr = np.array(
                x_curr.tolist() * self.params["agent"]["num_dyn_samples"]
            )
            X, U = self.one_step_planner(st_curr)
            X1_kp1, X2_kp1 = self.agent.pendulum_discrete_dyn(X[0][0], X[0][1], U[0])
            self.agent.update_current_state(torch.Tensor([X1_kp1, X2_kp1]))
            # propagate the agent to the next state
            print(
                bcolors.green + "Reached:",
                i,
                " ",
                X1_kp1,
                " ",
                X2_kp1,
                " ",
                U[0],
                bcolors.ENDC,
            )
        return False

    def one_step_planner(self, st_curr):
        """_summary_: Plans going and coming back all in one trajectory plan
        Input: current location, end location, dyn, etc.
        Process: Solve the NLP and simulate the system until the measurement collection point
        Output: trajectory
        """
        # questions:

        # self.visu.UpdateIter(self.iter, -1)
        print(bcolors.OKCYAN + "Solving Constrints" + bcolors.ENDC)
        self.dempc_solver.ocp_solver.set(0, "lbx", st_curr)
        self.dempc_solver.ocp_solver.set(0, "ubx", st_curr)

        # set objective as per desired goal
        t_0 = timeit.default_timer()
        self.dempc_solver.solve(self.agent)
        t_1 = timeit.default_timer()
        print("Time to solve", t_1 - t_0)
        X, U, Sl = self.dempc_solver.get_solution()
        # self.visu.Dyn_gp_model = self.agent.Dyn_gp_model
        self.visu.record(st_curr, X, U)
        # print(X,U)

        # self.visu.plot_pendulum_traj(X,U)
        return torch.from_numpy(X).float(), torch.from_numpy(U).float()

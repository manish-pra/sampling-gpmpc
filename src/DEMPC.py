# This is the algorithm file. It will be responsible to call environement,
# collect measurement, setup of MPC problem, call model, solver, etc.
import numpy as np
import torch

from src.solver import DEMPC_solver
from src.utils.initializer import get_players_initialized
from src.utils.termcolor import bcolors
import timeit

class DEMPC():
    def __init__(self, params, env, visu, agent) -> None:
        self.dempc_solver = DEMPC_solver(params)
        self.env = env
        self.visu = visu
        self.params = params
        self.iter = -1
        self.data = {}
        self.flag_reached_xt_goal = False
        self.flag_new_goal = True
        self.pl_idx = 0  # currently single player, so does not matter
        self.H = self.params["optimizer"]["H"]
        self.Hm = self.params["optimizer"]["Hm"]
        self.n_order = params["optimizer"]["order"]
        self.x_dim = params["optimizer"]["x_dim"]
        self.eps = params["common"]["epsilon"]
        self.q_th = params["common"]["constraint"]
        self.prev_goal_dist = 100
        self.goal_in_pessi = False
        self.state_dim = self.x_dim
        self.agent = agent
        

    def dempc_main(self):
        """_summary_ Responsible for initialization, logic for when to collect sample vs explore
        """
        # while not self.agent.infeasible:
        run = True
        # self.agent.feasible = True
        while run:
            run = self.receding_horizon()
            print("while loop")
        # print("Number of samples", self.agent.Cx_X_train.shape)

    def receding_horizon(self):
        print("Receding Horizon")
        for i in range(100):
            torch.cuda.empty_cache()
            x_curr = self.agent.current_state[:self.state_dim].reshape(self.state_dim)
            if torch.is_tensor(x_curr):
                x_curr = x_curr.numpy()
            st_curr = np.array(x_curr.tolist()*self.params["agent"]["num_dyn_samples"])
            X, U = self.one_step_planner(st_curr)
            X1_kp1, X2_kp1 = self.agent.pendulum_discrete_dyn(X[0][0], X[0][1], U[0])
            self.agent.update_current_state(torch.Tensor([X1_kp1, X2_kp1]))
            # propagate the agent to the next state
            print(bcolors.green + "Reached:", i, " ", X1_kp1," ", X2_kp1 , " ", U[0],  bcolors.ENDC)
        return False

    def receding_horizon_old(self, player):
        # diff = (player.planned_measure_loc[0] -
        #         player.current_location[0][0]).numpy()
        diff = np.array([100])
        temp_iter = 0
        while np.abs(diff.item()) > 1.0e-3 and temp_iter < 50:
            self.iter += 1
            temp_iter += 1
            self.visu.UpdateIter(self.iter, -1)
            print(bcolors.OKCYAN + "Solving Constrints" + bcolors.ENDC)

            # Write in MPC style to reach the goal. The main loop is outside
            x_curr = self.agent.current_location[0][0].reshape(
                1).numpy()
            x_origin = self.agent.origin[0].reshape(1).numpy()
            self.dempc_solver.ocp_solver.set(0, "lbx", x_curr)
            self.dempc_solver.ocp_solver.set(0, "ubx", x_curr)
            self.dempc_solver.ocp_solver.set(self.H, "lbx", x_origin)
            self.dempc_solver.ocp_solver.set(self.H, "ubx", x_origin)

            # warmstart
            # if self.flag_new_goal:
            #     optim.setwarmstartparam(
            #         player.obj_optim.getx(), player.obj_optim.getu())
            # else:
            #     optim.setwarmstartparam(
            #         player.optim_getx, player.optim_getu)

            # set objective as per desired goal
            self.dempc_solver.solve(self.agent)
            X, U, Sl = self.oracle_solver.get_solution()

            # integrator
            self.env.integrator.set("x", x_curr)
            self.env.integrator.set("u", U[0])
            self.env.integrator.solve()
            x_next = self.env.integrator.get("x")
            self.agent.update_current_location(
                torch.Tensor([x_next.item(), -2.0]).reshape(-1, 2))
            diff = X[int(self.H/2)] - x_curr
            print(x_curr, " ", diff)
            # self.visu.UpdateIter(self.iter, -1)
            # self.visu.UpdateSafeVisu(0, self.players, self.env)
            # # pt_se_dyn = self.visu.plot_SE_traj(
            # #     optim, player, fig_dyn, pt_se_dyn)
            # self.visu.writer_gp.grab_frame()
            # self.visu.writer_dyn.grab_frame()
            # self.visu.f_handle["dyn"].savefig("temp1D.png")
            # self.visu.f_handle["gp"].savefig(
            #     str(self.iter) + 'temp in prog2.png')
            # print(self.iter, " ", diff, " cost ",
            #       self.dempc_solver.ocp_solver.get_cost())

        # set current location as the location to be measured
        self.agent.safe_meas_loc = player.current_location
        goal_dist = (
            player.planned_measure_loc[0] - player.current_location[0][0]).numpy()
        if np.abs(goal_dist.item()) < 1.0e-2:
            self.flag_reached_xt_goal = True
        self.prev
        # apply this input to your environment

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
        # if self.params["algo"]["type"] == "MPC_Xn":
        #     pass
        #     # st_lb = np.zeros(self.state_dim+1)
        #     # st_ub = np.zeros(self.state_dim+1)
        #     # st_lb[:self.x_dim] = -np.ones(self.x_dim)*100
        #     # st_ub[:self.x_dim] = np.ones(self.x_dim)*100
        #     # st_lb[-1] = 1.0
        #     # st_ub[-1] = 1.0
        #     # self.dempc_solver.ocp_solver.set(self.H, "lbx", st_lb)
        #     # self.dempc_solver.ocp_solver.set(self.H, "ubx", st_ub)
        # elif self.params["algo"]["type"] == "MPC_V0":
        #     st_lb = np.zeros(self.state_dim+1)
        #     st_ub = np.zeros(self.state_dim+1)
        #     # st_lb[:self.x_dim] = -np.ones(self.x_dim)*100
        #     # st_ub[:self.x_dim] = np.ones(self.x_dim)*100
        #     st_lb[:self.state_dim] = np.array(self.params["optimizer"]["x_min"])
        #     st_ub[:self.state_dim] = np.array(self.params["optimizer"]["x_max"])
        #     if self.params["agent"]["dynamics"] == "robot":
        #         st_lb[3] = 0.0
        #         st_ub[3] = 0.0
        #         st_lb[4] = 0.0
        #         st_ub[4] = 0.0
        #     elif self.params["agent"]["dynamics"] == "unicycle" or self.params["agent"]["dynamics"] == "bicycle":
        #         st_lb[3] = 0.0
        #         st_ub[3] = 0.0
        #     elif self.params["agent"]["dynamics"] == "int":
        #         st_lb[2] = 0.0
        #         st_ub[2] = 0.0
        #         st_lb[3] = 0.0
        #         st_ub[3] = 0.0
        #     # st_ub[2] = 6.28
        #     # st_lb[2] = -6.28
        #     st_ub[-1] = 1.0
        #     # self.dempc_solver.ocp_solver.set(self.Hm, "lbx", st_lb)
        #     # self.dempc_solver.ocp_solver.set(self.Hm, "ubx", st_ub)
        #     st_lb[-1] = 1.0
        #     self.dempc_solver.ocp_solver.set(self.H, "lbx", st_lb)
        #     self.dempc_solver.ocp_solver.set(self.H, "ubx", st_ub)
        # else:
        #     st_origin = np.zeros(self.state_dim+1)
        #     st_origin[:self.x_dim] = np.ones(self.x_dim)*x_origin
        #     st_origin[-1] = 1.0
        #     self.dempc_solver.ocp_solver.set(self.H, "lbx", st_origin)
        #     self.dempc_solver.ocp_solver.set(self.H, "ubx", st_origin)

        # set objective as per desired goal
        t_0 = timeit.default_timer()
        self.dempc_solver.solve(self.agent)
        t_1 = timeit.default_timer()
        print("Time to solve", t_1-t_0)
        X, U, Sl = self.dempc_solver.get_solution()
        # self.visu.Dyn_gp_model = self.agent.Dyn_gp_model
        self.visu.record(st_curr, X, U)
        # print(X,U)
        
        # self.visu.plot_pendulum_traj(X,U)
        return torch.from_numpy(X).float(), torch.from_numpy(U).float()
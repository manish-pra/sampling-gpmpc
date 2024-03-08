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

        # val= 2*self.agent.Cx_beta*2*torch.sqrt(self.agent.Cx_model(torch.from_numpy(X[self.Hm, :self.x_dim]).reshape(-1,2).float()).variance).detach().item()
        # # print("slack", Sl, "uncertainity", X[self.Hm], val)#, "z-x",np.linalg.norm(X[:-1,0:2] - U[:,3:5]))
        # # self.visu.record(X, U, X[self.Hm], self.pl_idx, self.players)
        self.visu.record(X, U, self.agent.get_next_to_go_loc(), self.pl_idx, self.players)


        # Environement simulation
        # x_curr = X[0]
        # for i in range(self.Hm):
        #     self.env.integrator.set("x", x_curr)
        #     self.env.integrator.set("u", U[i])
        #     self.env.integrator.solve()
        #     x_curr = self.env.integrator.get("x")
        #     if self.x_dim == 1:
        #         x_curr = np.hstack([x_curr[:self.x_dim].item(), -2.0])
        #     self.agent.update_current_state(x_curr)
        self.agent.safe_meas_loc = X[self.Hm][:self.x_dim]
        if self.params["algo"]["type"] == "ret" or self.params["algo"]["type"] == "ret_expander":
            self.agent.update_current_state(X[self.H])
            if self.goal_in_pessi:
                # if np.linalg.norm(self.visu.utility_minimizer-self.agent.safe_meas_loc) < 0.025:
                self.agent.update_current_state(X[self.Hm])
            else:
                pass
            
        # assert np.isclose(x_curr,X[self.Hm]).all()
            # self.visu.UpdateIter(self.iter+i, -1)
            # self.visu.UpdateSafeVisu(0, self.players, self.env)
            # self.visu.writer_gp.grab_frame()
            # self.visu.writer_dyn.grab_frame()
            # self.visu.f_handle["dyn"].savefig("temp1D.png")
            # self.visu.f_handle["gp"].savefig(
            #     str(self.iter) + 'temp in prog2.png')
        
        # set current location as the location to be measured
        
        goal_dist = np.linalg.norm(self.agent.planned_measure_loc -
                                   self.agent.current_location)
        if np.abs(goal_dist) < 1.0e-2:
            self.flag_reached_xt_goal = True
        # if np.abs(self.prev_goal_dist - goal_dist) < 1e-2:
        #     # set infeasibility flag to true and ask for a new goal
        #     self.agent.infeasible = True
        # self.prev_goal_dist = goal_dist
        # apply this input to your environment

    def not_reached_and_prob_feasible(self):
        """_summary_ The agent safely explores and either reach goal or remove it from safe set
        (not reached_xt_goal) and (not player.infeasible)
        """
        # while not self.flag_reached_xt_goal and (not self.agent.infeasible):
            # this while loops ensures we collect measurement only at constraint and not all along
            # the path
            # self.receding_horizon(self.agent)
        self.one_step_planner()
        # if self.flag_reached_xt_goal:
        #     self.visu.UpdateIter(self.iter, -1)
        #     self.visu.UpdateSafeVisu(0, self.players, self.env)
        #     self.visu.writer_gp.grab_frame()
        #     self.visu.writer_dyn.grab_frame()
        #     self.visu.f_handle["dyn"].savefig("temp1D.png")
        #     self.visu.f_handle["gp"].savefig('temp in prog2.png')
        #     return None
        # collect measurement at the current location
        # if problem is infeasible then also return
        if not self.goal_in_pessi:
            print("Uncertainity at meas_loc", self.agent.get_width_at_curr_loc())
            TrainAndUpdateConstraint(self.agent.safe_meas_loc, self.pl_idx, self.players, self.params, self.env)
            print("Uncertainity at meas_loc", self.agent.get_width_at_curr_loc())
        if self.params["visu"]["show"]:
            self.visu.UpdateIter(self.iter, -1)
            self.visu.UpdateSafeVisu(0, self.players, self.env)
            self.visu.writer_gp.grab_frame()
            self.visu.writer_dyn.grab_frame()
            # self.visu.f_handle["dyn"].savefig("temp1D.png")
            # self.visu.f_handle["gp"].savefig('temp in prog2.png')

    def goal_reached_or_prob_infeasible(self):
        self.iter += 1
        self.visu.UpdateIter(self.iter, -1)
        print(bcolors.OKGREEN + "Solving Objective" + bcolors.ENDC)
        # Fx_model = self.players[0].Fx_model.eval()

        # get new goal
        xi_star = self.oracle()
        new_goal = True
        dist = np.linalg.norm(self.agent.planned_measure_loc -
                              self.agent.current_location)
        if np.abs(dist) > 1.0e-3:
            self.flag_reached_xt_goal = False
        self.agent.infeasible = False
        self.visu.UpdateIter(self.iter, -1)
        self.visu.UpdateObjectiveVisu(0, self.players, self.env, 0)
        # self.visu.UpdateDynVisu(0, self.players)
        # reached_xt_goal = False
        # if LB(player.planned_measure_loc.reshape(-1, 2)) >= params["common"]["constraint"]:
        #     reached_xt_goal = True
        #     reached_pt = player.planned_measure_loc.reshape(-1, 2)
        #     player.update_current_location(reached_pt)
        # player.infeasible = False
        # pt_fx_dyn = self.visu.plot_Fx_traj(player, fig_dyn, pt_fx_dyn)

        self.visu.writer_gp.grab_frame()
        self.visu.writer_dyn.grab_frame()
        self.visu.f_handle["dyn"].savefig("temp1D.png")
        self.visu.f_handle["gp"].savefig('temp in prog2.png')

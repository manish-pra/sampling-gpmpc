# This is the algorithm file. It will be responsible to call environement,
# collect measurement, setup of MPC problem, call model, solver, etc.
import os

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.solver import DEMPC_solver
# from src.utils.helper import (TrainAndUpdateConstraint, TrainAndUpdateDensity,
                            #   get_frame_writer, oracle)
from src.utils.initializer import get_players_initialized
from src.utils.termcolor import bcolors


class DEMPC():
    def __init__(self, params, env, visu) -> None:
        # self.oracle_solver = Oracle_solver(params)
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
        self.dempc_initialization()
    
    def get_optimistic_path(self, node, goal_node, init_node):
        # If there doesn't exists a safe path then re-evaluate the goal
        try:
            print("init_node", init_node.item(), "goal_node", goal_node.item())
            t1 = self.players[self.pl_idx].get_optimistic_path(node.item(), goal_node.item())
            t2 = self.players[self.pl_idx].get_optimistic_path(goal_node.item(), init_node.item())
            opti_path = t1 + t2
        except: # change of utility minimizer location
            list_opti_node = list(self.players[self.pl_idx].optimistic_graph.nodes())
            val_optimistic_graph = self.env.get_true_objective_func()[list_opti_node]
            self.players[self.pl_idx].get_utility_minimizer = self.players[self.pl_idx].grid_V[list_opti_node[val_optimistic_graph.argmin().item()]].numpy()
            goal_node = self.players[self.pl_idx].get_idx_from_grid(torch.from_numpy(self.players[self.pl_idx].get_utility_minimizer))
            self.visu.utility_minimizer = self.players[self.pl_idx].get_utility_minimizer
            print("init_node", init_node.item(), "curr_node", node.item(), "goal_node", goal_node.item())
            t1 = self.players[self.pl_idx].get_optimistic_path(node.item(), goal_node.item())
            t2 = self.players[self.pl_idx].get_optimistic_path(goal_node.item(), init_node.item())
            opti_path = t1 + t2
        return opti_path, goal_node
    
    def set_next_goal(self):
        # 2) Set goal based on objective and strategy
        if self.params["algo"]["objective"] == "SE":
            if self.params["algo"]["strategy"] == "SEpessi":
                w, xi_star = self.players[self.pl_idx].uncertainity_sampling(const_set="pessi")
                self.players[self.pl_idx].set_maximizer_goal(xi_star)
            elif self.params["algo"]["strategy"] == "SEopti":
                w, xi_star = self.players[self.pl_idx].uncertainity_sampling(const_set="opti")
                self.players[self.pl_idx].set_maximizer_goal(xi_star)
            elif self.params["algo"]["strategy"] == "goose":
                # Set the x_g which can be used in the distance cost function
                xi_star = np.array(self.params["env"]["goal_loc"])
                self.players[self.pl_idx].set_maximizer_goal(xi_star)       
                w=100     
            elif self.params["algo"]["strategy"] == "optiTraj":
                xi_star = self.oracle()
                self.players[self.pl_idx].set_maximizer_goal(xi_star)   
                w=100
            self.visu.num_safe_nodes = self.players[self.pl_idx].num_safe_nodes
            self.visu.opti_path = None
            self.visu.utility_minimizer = None

        elif self.params["algo"]["objective"] == "GO":
            V_lower_Cx, V_upper_Cx = self.players[self.pl_idx].get_Cx_bounds(self.players[self.pl_idx].grid_V)
            self.visu.num_safe_nodes = len(V_lower_Cx[V_lower_Cx>0])
            init_node = self.players[self.pl_idx].get_idx_from_grid(self.players[self.pl_idx].origin)
            curr_node = self.players[self.pl_idx].get_idx_from_grid(torch.from_numpy(self.players[self.pl_idx].current_location))
            # self.players[self.pl_idx].update_pessimistic_graph(V_lower_Cx, init_node, self.q_th, Lc=0)
            # curr_node = self.players[self.pl_idx].get_nearest_pessi_idx(torch.from_numpy(self.players[self.pl_idx].current_location))
            # intersect_pessi_opti =  torch.max(V_upper_Cx-self.eps, V_lower_Cx+0.04)
            intersect_pessi_opti = V_upper_Cx - self.eps - 0.1
            self.players[self.pl_idx].update_optimistic_graph(intersect_pessi_opti, init_node, self.q_th,  curr_node, Lc=0)
            curr_node = self.players[self.pl_idx].get_nearest_opti_idx(torch.from_numpy(self.players[self.pl_idx].current_location))
            goal_node = self.players[self.pl_idx].get_idx_from_grid(torch.from_numpy(self.players[self.pl_idx].get_utility_minimizer))
            self.visu.utility_minimizer = self.players[self.pl_idx].get_utility_minimizer
            if self.params["algo"]["type"] == "MPC_V0":
                opti_path, goal_node = self.get_optimistic_path(curr_node, goal_node, init_node)
            else:
                opti_path, goal_node = self.get_optimistic_path(init_node, goal_node, init_node)
            self.visu.opti_path = self.players[self.pl_idx].grid_V[opti_path]
            # if goal is within the pessimitic set set xi_star as goal directly
            if V_lower_Cx[goal_node] >=0:
                xi_star = self.players[self.pl_idx].grid_V[goal_node.item()].numpy()
                self.goal_in_pessi = True
                self.players[self.pl_idx].goal_in_pessi = True
            else:
                pessi_value = V_lower_Cx[opti_path]
                if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
                    pessi_value = self.enlarge_pessi_set(pessi_value)
                idx_out_pessi = np.where(pessi_value < self.q_th)[0][0].item()
                xi_star = self.players[self.pl_idx].grid_V[opti_path[idx_out_pessi]].numpy()
            self.players[self.pl_idx].set_maximizer_goal(xi_star)
            w=100

        if self.params["visu"]["show"]: 
            self.visu.UpdateIter(self.iter, -1)
            self.visu.UpdateObjectiveVisu(0, self.players, self.env, 0)
            self.visu.writer_gp.grab_frame()
            self.visu.writer_dyn.grab_frame()
            self.visu.f_handle["dyn"].savefig("temp1D.png")
            self.visu.f_handle["gp"].savefig('temp in prog2.png')
        print(bcolors.green + "Goal:", xi_star , " uncertainity:", w, bcolors.ENDC)
        return w
    
    def enlarge_pessi_set(self, pessi_value):
        # Make it more efficient by only using matrix instead of vectors
        max_idx = pessi_value.argmax().item()
        cur_max_val = pessi_value[max_idx]
        cur_max_idx = max_idx
        for i in range(0,len(pessi_value)):
            if pessi_value[i] > cur_max_val-self.params["common"]["Lc"]*self.params["visu"]["step_size"]*np.abs(i-cur_max_idx):
                cur_max_idx = i
                cur_max_val = pessi_value[i]
            else:
                pessi_value[i] = cur_max_val-self.params["common"]["Lc"]*self.params["visu"]["step_size"]*np.abs(i-cur_max_idx)
        return pessi_value


    def dempc_main(self):
        """_summary_ Responsible for initialization, logic for when to collect sample vs explore
        """
        # if self.params["algo"]["strategy"] == "SEpessi" or "SEopti" or "goose":
        w=100
        # while not self.players[self.pl_idx].infeasible:
        running_condition_true= True
        self.players[self.pl_idx].feasible = True
        while running_condition_true:
            self.not_reached_and_prob_feasible()

            if w < self.params["common"]["epsilon"]:
                self.players[self.pl_idx].feasible = False
            else:
                w = self.set_next_goal()

            if self.params["algo"]["objective"] == "GO":
                running_condition_true = (not np.linalg.norm(self.visu.utility_minimizer-self.players[self.pl_idx].current_location) < 0.025)
            elif self.params["algo"]["objective"] == "SE":
                running_condition_true = self.players[self.pl_idx].feasible
            else:
                raise NameError("Objective is not clear")
        print("Number of samples", self.players[self.pl_idx].Cx_X_train.shape)
        # while (self.max_density_sigma > self.params["algo"]["eps_density_thresh"]) and self.iter < self.params["algo"]["n_iter"]:
        #     # while (not self.flag_reached_xt_goal) and (not self.players[self.pl_idx].infeasible):
        #     #     # recursively run MPC until the goal is reached (also collect constraint measurement meanwhile)
        #     self.not_reached_and_prob_feasible()

        #     # Collect density measurement if your goal is reached
        #     if self.flag_reached_xt_goal:
        #         # player.planned_measure_loc[0], player.current_location[0][0], self.players[self.pl_idx].safe_meas_loc
        #         TrainAndUpdateDensity(
        #             self.players[self.pl_idx].safe_meas_loc, self.pl_idx, self.players, self.params, self.env)

        #     # decide on a new goal
        #     # or player.infeasible:
        #     # if self.flag_reached_xt_goal or self.players[self.pl_idx].infeasible:
        #     self.goal_reached_or_prob_infeasible()
        # print("Max density uncertainity", self.max_density_sigma, " iter: ", self.iter)

    # def dempc_main(self):
    #     """_summary_ Responsible for initialization, logic for when to collect sample vs explore
    #     """
    #     while (self.max_density_sigma > self.params["algo"]["eps_density_thresh"]) and self.iter < self.params["algo"]["n_iter"]:
    #         while (not self.flag_reached_xt_goal) and (not self.players[self.pl_idx].infeasible):
    #             # recursively run MPC until the goal is reached (also collect constraint measurement meanwhile)
    #             self.not_reached_and_prob_feasible()

    #         # Collect density measurement if your goal is reached
    #         if self.flag_reached_xt_goal:
    #             # player.planned_measure_loc[0], player.current_location[0][0], self.players[self.pl_idx].safe_meas_loc
    #             TrainAndUpdateDensity(
    #                 self.players[self.pl_idx].safe_meas_loc, self.pl_idx, self.players, self.params, self.env)

    #         # decide on a new goal
    #         # or player.infeasible:
    #         if self.flag_reached_xt_goal or self.players[self.pl_idx].infeasible:
    #             self.goal_reached_or_prob_infeasible()
    #     print("Max density uncertainity",
    #           self.max_density_sigma, " iter: ", self.iter)

    def dempc_initialization(self):
        """_summary_ Everything before the looping for gp-measurements
        """
        # 1) Initialize players to safe location in the environment
        # print("initialized location", self.env.get_safe_init())
        # TODO: Remove dependence of player on visu grid
        self.players = get_players_initialized(self.params)

        for it, player in enumerate(self.players):
            # player.update_Cx_gp_with_current_data()
            # player.update_Fx_gp_with_current_data()
            # player.save_posterior_normalization_const()
            # init = self.env.get_safe_init()["Cx_X"][it].reshape(-1, 2).numpy()
            # state = np.zeros(self.state_dim+1)
            # state[:self.x_dim] = init
            player.update_current_state(np.array(self.params["env"]["start"]))

        # associate_dict = {}
        # associate_dict[0] = []
        # for idx in range(self.params["env"]["n_players"]):
        #     associate_dict[0].append(idx)

        # # 3) Set termination criteria
        # self.max_density_sigma = sum(
        #     [player.max_density_sigma for player in self.players])
        # self.data["sum_max_density_sigma"] = []
        # self.data["sum_max_density_sigma"].append(self.max_density_sigma)
        # print(self.iter, self.max_density_sigma)

    def oracle(self):
        """_summary_ Setup and solve the oracle MPC problem

        Args:
            players (_type_): _description_
            init_safe (_type_): _description_

        Returns:
            _type_: Goal location for the agent
        """
        # x_curr = self.players[self.pl_idx].current_location[0][0].reshape(
        #     1).numpy()
        # x_origin = self.players[self.pl_idx].origin[:self.x_dim].numpy()
        # lbx = np.zeros(self.state_dim+1)
        # # lbx[:self.x_dim] = np.ones(self.x_dim)*x_origin
        # lbx[:self.x_dim] = np.ones(self.x_dim)*self.players[self.pl_idx].current_location.reshape(-1)
        # ubx = lbx.copy()
        # self.oracle_solver.ocp_solver.set(0, "lbx", lbx.copy())
        # self.oracle_solver.ocp_solver.set(0, "ubx", ubx.copy())
        # # self.oracle_solver.ocp_solver.set(self.H, "lbx", lbx.copy() - 1)
        # # self.oracle_solver.ocp_solver.set(self.H, "ubx", ubx.copy() + 1)

        # Write in MPC style to reach the goal. The main loop is outside
        # reachability constraint
        x_curr = self.players[self.pl_idx].current_state[:self.x_dim*2].reshape(4)
        x_origin = self.players[self.pl_idx].origin[:self.x_dim].numpy()
        if torch.is_tensor(x_curr):
            x_curr = x_curr.numpy()
        st_curr = np.zeros(self.state_dim+1)
        st_curr[:self.x_dim*2] = np.ones(self.x_dim*2)*x_curr
        self.oracle_solver.ocp_solver.set(0, "lbx", st_curr)
        self.oracle_solver.ocp_solver.set(0, "ubx", st_curr)
        
        # returnability constraint
        st_origin = np.zeros(self.state_dim+1)
        st_origin[:self.x_dim] = np.ones(self.x_dim)*x_origin
        st_origin[-1] = 1.0
        self.oracle_solver.ocp_solver.set(self.H, "lbx", st_origin)
        self.oracle_solver.ocp_solver.set(self.H, "ubx", st_origin)

        self.oracle_solver.solve(self.players[self.pl_idx])
        # highest uncertainity location in safe
        X, U = self.oracle_solver.get_solution()
        # print(X, U)
        self.players[self.pl_idx].update_oracle_XU(X, U)
        xi_star = X[int(self.H/2), :self.x_dim]
        if self.x_dim == 1:
            xi_star = torch.Tensor([xi_star.item(), -2.0]).reshape(2)
        # self.players[self.pl_idx].set_maximizer_goal(xi_star)
        # print(bcolors.green + "Goal:", xi_star ,  bcolors.ENDC)

        # plt.plot(X[:,0],X[:,1])
        # plt.savefig("temp.png")
        # save status to the player also its goal location, may be ignote the dict concept now
        # associate_dict, pessi_associate_dict, acq_density = oracle(
        #     self, players, init_safe, self.params)
        # return associate_dict, pessi_associate_dict, acq_density
        return xi_star

    def receding_horizon(self, player):
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
            x_curr = self.players[self.pl_idx].current_location[0][0].reshape(
                1).numpy()
            x_origin = self.players[self.pl_idx].origin[0].reshape(1).numpy()
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
            self.dempc_solver.solve(self.players[self.pl_idx])
            X, U, Sl = self.oracle_solver.get_solution()

            # integrator
            self.env.integrator.set("x", x_curr)
            self.env.integrator.set("u", U[0])
            self.env.integrator.solve()
            x_next = self.env.integrator.get("x")
            self.players[self.pl_idx].update_current_location(
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
        self.players[self.pl_idx].safe_meas_loc = player.current_location
        goal_dist = (
            player.planned_measure_loc[0] - player.current_location[0][0]).numpy()
        if np.abs(goal_dist.item()) < 1.0e-2:
            self.flag_reached_xt_goal = True
        self.prev
        # apply this input to your environment

    def one_step_planner(self):
        """_summary_: Plans going and coming back all in one trajectory plan
        Input: current location, end location, dyn, etc.
        Process: Solve the NLP and simulate the system until the measurement collection point
        Output: trajectory
        """
        # questions:

        # self.visu.UpdateIter(self.iter, -1)
        print(bcolors.OKCYAN + "Solving Constrints" + bcolors.ENDC)

        # Write in MPC style to reach the goal. The main loop is outside
        x_curr = self.players[self.pl_idx].current_state[:self.state_dim].reshape(self.state_dim)
        # x_origin = self.players[self.pl_idx].origin[:self.x_dim].numpy()
        if torch.is_tensor(x_curr):
            x_curr = x_curr.numpy()
        # st_curr = np.zeros(self.state_dim)
        st_curr = np.array(x_curr.tolist()*self.params["agent"]["num_dyn_samples"]) #(self.state_dim)*x_curr.tolist()
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
        self.dempc_solver.solve(self.players[self.pl_idx])
        X, U, Sl = self.dempc_solver.get_solution()
        print(X,U)
        self.visu.Dyn_gp_model = self.players[self.pl_idx].Dyn_gp_model
        self.visu.plot_pendulum_traj(X,U)
        val= 2*self.players[self.pl_idx].Cx_beta*2*torch.sqrt(self.players[self.pl_idx].Cx_model(torch.from_numpy(X[self.Hm, :self.x_dim]).reshape(-1,2).float()).variance).detach().item()
        # print("slack", Sl, "uncertainity", X[self.Hm], val)#, "z-x",np.linalg.norm(X[:-1,0:2] - U[:,3:5]))
        # self.visu.record(X, U, X[self.Hm], self.pl_idx, self.players)
        self.visu.record(X, U, self.players[self.pl_idx].get_next_to_go_loc(), self.pl_idx, self.players)

        # Environement simulation
        # x_curr = X[0]
        # for i in range(self.Hm):
        #     self.env.integrator.set("x", x_curr)
        #     self.env.integrator.set("u", U[i])
        #     self.env.integrator.solve()
        #     x_curr = self.env.integrator.get("x")
        #     if self.x_dim == 1:
        #         x_curr = np.hstack([x_curr[:self.x_dim].item(), -2.0])
        #     self.players[self.pl_idx].update_current_state(x_curr)
        self.players[self.pl_idx].safe_meas_loc = X[self.Hm][:self.x_dim]
        if self.params["algo"]["type"] == "ret" or self.params["algo"]["type"] == "ret_expander":
            self.players[self.pl_idx].update_current_state(X[self.H])
            if self.goal_in_pessi:
                # if np.linalg.norm(self.visu.utility_minimizer-self.players[self.pl_idx].safe_meas_loc) < 0.025:
                self.players[self.pl_idx].update_current_state(X[self.Hm])
        else:
            self.players[self.pl_idx].update_current_state(X[self.Hm])
        # assert np.isclose(x_curr,X[self.Hm]).all()
            # self.visu.UpdateIter(self.iter+i, -1)
            # self.visu.UpdateSafeVisu(0, self.players, self.env)
            # self.visu.writer_gp.grab_frame()
            # self.visu.writer_dyn.grab_frame()
            # self.visu.f_handle["dyn"].savefig("temp1D.png")
            # self.visu.f_handle["gp"].savefig(
            #     str(self.iter) + 'temp in prog2.png')
        print(bcolors.green + "Reached:", x_curr ,  bcolors.ENDC)
        # set current location as the location to be measured
        
        goal_dist = np.linalg.norm(self.players[self.pl_idx].planned_measure_loc -
                                   self.players[self.pl_idx].current_location)
        if np.abs(goal_dist) < 1.0e-2:
            self.flag_reached_xt_goal = True
        # if np.abs(self.prev_goal_dist - goal_dist) < 1e-2:
        #     # set infeasibility flag to true and ask for a new goal
        #     self.players[self.pl_idx].infeasible = True
        # self.prev_goal_dist = goal_dist
        # apply this input to your environment

    def not_reached_and_prob_feasible(self):
        """_summary_ The agent safely explores and either reach goal or remove it from safe set
        (not reached_xt_goal) and (not player.infeasible)
        """
        # while not self.flag_reached_xt_goal and (not self.players[self.pl_idx].infeasible):
            # this while loops ensures we collect measurement only at constraint and not all along
            # the path
            # self.receding_horizon(self.players[self.pl_idx])
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
            print("Uncertainity at meas_loc", self.players[self.pl_idx].get_width_at_curr_loc())
            TrainAndUpdateConstraint(self.players[self.pl_idx].safe_meas_loc, self.pl_idx, self.players, self.params, self.env)
            print("Uncertainity at meas_loc", self.players[self.pl_idx].get_width_at_curr_loc())
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
        dist = np.linalg.norm(self.players[self.pl_idx].planned_measure_loc -
                              self.players[self.pl_idx].current_location)
        if np.abs(dist) > 1.0e-3:
            self.flag_reached_xt_goal = False
        self.players[self.pl_idx].infeasible = False
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

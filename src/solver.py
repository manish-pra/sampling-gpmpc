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
        ocp = export_dempc_ocp(params)
        self.name_prefix = params["algo"]["type"] + '_env_' + str(params["env"]["name"]) + '_i_' + str(params["env"]["i"]) + '_'
        self.ocp_solver = AcadosOcpSolver(
            ocp, json_file= self.name_prefix + 'acados_ocp_sempc.json')
        self.ocp_solver.store_iterate(self.name_prefix + 'ocp_initialization.json')
        
        self.H = params["optimizer"]["H"]
        self.Hm = params["optimizer"]["Hm"]
        self.max_sqp_iter = params["optimizer"]["SEMPC"]["max_sqp_iter"]
        self.tol_nlp = params["optimizer"]["SEMPC"]["tol_nlp"]
        self.nx = ocp.model.x.size()[0]
        self.nu = ocp.model.u.size()[0]
        self.eps = params["common"]["epsilon"]
        self.n_order = params["optimizer"]["order"]
        self.x_dim = params["optimizer"]["x_dim"]
        self.pos_dim = 1
        self.params = params
        self.state_dim = self.n_order*self.x_dim

    def initilization(self, sqp_iter, x_h, u_h):
        for stage in range(self.H):
            # current stage values
            x_h[stage, :] = self.ocp_solver.get(stage, "x") 
            u_h[stage, :] = self.ocp_solver.get(stage, "u")
        x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
        if sqp_iter == 0:
            x_h_old = x_h.copy()
            u_h_old = u_h.copy()
            if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
                u_h_old[:, -self.x_dim:] = x_h_old[:-1, :self.x_dim].copy()
            # initialize the first SQP iteration.
            for stage in range(self.H):
                if stage < (self.H - self.Hm):
                    # current stage values
                    x_init = x_h_old[stage + self.Hm, :].copy() 
                    u_init = u_h_old[stage + self.Hm, :].copy()
                    x_init[-1] = (x_h_old[stage + self.Hm, -1] - x_h_old[self.Hm, -1]).copy() 
                    self.ocp_solver.set(stage, "x", x_init)
                    self.ocp_solver.set(stage, "u", u_init)
                    x_h[stage, :] = x_init.copy()
                    u_h[stage, :] = u_init.copy()
                    half_time = x_init[-1].copy() 
                else:
                    dt = (1.0-half_time)/self.Hm
                    x_init = x_h_old[self.H, :].copy() # reached the final state
                    x_init[-1] = half_time + dt*(stage-self.Hm)
                    z_init = x_init[0:self.x_dim]
                    if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
                        u_init = np.concatenate([np.array([0.0,0.0, dt]), z_init])
                    else:
                        u_init = np.array([0.0,0.0, dt])
                    self.ocp_solver.set(stage, "x", x_init)
                    self.ocp_solver.set(stage, "u", u_init)
                    x_h[stage, :] = x_init.copy()
                    u_h[stage, :] = u_init.copy()
            self.ocp_solver.set(self.H, "x", x_init)
            x_init[-1] = half_time + dt*(self.H-self.Hm)
            x_h[self.H, :] = x_init.copy() 
            # x0 = np.zeros(self.state_dim)
            # x0[:self.x_dim] = np.ones(self.x_dim)*0.72
            # x0=np.concatenate([x0, np.array([0.0])])
            # x_init=x0.copy()
            # # x_init = self.ocp_solver.get(0, "x")
            # u_init = self.ocp_solver.get(0, "u")
            # Ts = 1/200
            # # MPC controller
            # x_init = np.array([0.72,0.72,0.0,0.0, 0.0])
            # u_init = np.array([-0.2,-0.2, Ts])
            #     x_h[stage, :] = x_init
            #     u_h[stage, :] = u_init
            # x_h[self.H, :] = x_init
            # self.ocp_solver.set(self.H, "x", x_init)    
        return x_h, u_h    

    def path_init(self,path):
        split_path = np.zeros((self.H+1,self.x_dim))
        interp_h = np.arange(self.Hm)
        path_step = np.linspace(0,self.Hm,path.shape[0])
        x_pos = np.interp(interp_h, path_step, path.numpy()[:,0])
        y_pos = np.interp(interp_h, path_step, path.numpy()[:,1])
        split_path[:self.Hm,0], split_path[:self.Hm,1] = x_pos, y_pos
        split_path[self.Hm:,:] = np.ones_like(split_path[self.Hm:,:])*path[-1].numpy()
        # split the path into horizons
        for stage in range(self.H+1):
            x_init = self.ocp_solver.get(stage, "x")
            x_init[:self.x_dim] = split_path[stage]
            self.ocp_solver.set(stage, "x", x_init)

    def solve(self, player):
        x_h = np.zeros((self.H, (self.pos_dim+1)*self.params["agent"]["num_dyn_samples"]))
        u_h = np.zeros((self.H, self.pos_dim)) # u_dim
        w = 1e-3*np.ones(self.H+1)
        # we = 1e-8*np.ones(self.H+1)
        # we[int(self.H-1)] = 10000
        # w[:int(self.Hm)] = 1e-1*np.ones(self.Hm)
        w[int(self.H - 1)] = self.params["optimizer"]["w"]
        # cw = 1e+3*np.ones(self.H+1)
        # if not player.goal_in_pessi:
        #     cw[int(self.Hm)] = 1
        xg = np.ones((self.H+1, self.pos_dim))*player.get_next_to_go_loc()
        # x_origin = player.origin[:self.pos_dim].numpy()
        x_terminal = np.zeros(self.pos_dim)
        # x_terminal[:self.pos_dim] = np.ones(self.pos_dim)*x_origin
        for sqp_iter in range(self.max_sqp_iter):
            self.ocp_solver.options_set('rti_phase', 1)
            # if self.params["algo"]["type"] == "ret" or self.params["algo"]["type"] == "ret_expander":
            #     if player.goal_in_pessi:
            #         x_h, u_h = self.initilization(sqp_iter, x_h, u_h)
            #     else:
            #         for stage in range(self.H):
            #             # current stage values
            #             x_h[stage, :] = self.ocp_solver.get(stage, "x") 
            #             u_h[stage, :] = self.ocp_solver.get(stage, "u")
            #         x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
            # else:
            #    pass
            # x_h, u_h = self.initilization(sqp_iter, x_h, u_h)
            for stage in range(self.H):
                # current stage values
                x_h[stage, :] = self.ocp_solver.get(stage, "x") 
                u_h[stage, :] = self.ocp_solver.get(stage, "u")
            # x_h[self.H, :] = self.ocp_solver.get(self.H, "x")

            player.train_hallucinated_dynGP(sqp_iter)
            batch_x_hat = player.get_batch_x_hat(x_h, u_h)
            gp_val, y_grad, u_grad = player.get_batch_gp_sensitivities(batch_x_hat)
            
            # gp_val, gp_grad = player.get_gp_sensitivities(np.hstack([x_h, u_h]), "mean", 0)
            # gp_val, gp_grad = player.get_true_gradient(np.hstack([x_h,u_h]))
            for stage in range(self.H):
                p_lin = np.empty(0)
                for i in range(self.params["agent"]["num_dyn_samples"]):
                    p_lin = np.concatenate([p_lin,
                                            y_grad['y1'][i,stage,:].reshape(-1),
                                            y_grad['y2'][i,stage,:].reshape(-1),
                                            u_grad[i,stage,:].reshape(-1),
                                            x_h[stage,i*2: 2*(i+1)], gp_val[i,stage,:].reshape(-1)])
                p_lin = np.hstack([p_lin, u_h[stage], xg[stage], w[stage]])
                self.ocp_solver.set(stage, "p", p_lin)
                # self.ocp_solver.set(stage, "p", np.hstack(
                #     (gp_grad[:,stage,:][:,:2].transpose().reshape(-1), gp_grad[:,stage,:][:,-1].reshape(-1), x_h[stage], 
                #      u_h[stage], gp_val[:,stage], xg[stage], w[stage])))
            status = self.ocp_solver.solve()

            self.ocp_solver.options_set('rti_phase', 2)
            t_0 = timeit.default_timer()
            status = self.ocp_solver.solve()
            t_1 = timeit.default_timer()
            print("Time taken for SQP iteration", t_1-t_0)
            # self.ocp_solver.print_statistics()
            print("cost", self.ocp_solver.get_cost())
            residuals = self.ocp_solver.get_residuals()
            
            X, U, Sl = self.get_solution()
            # print("X", X, "U", U)
            # for stage in range(self.H):
            #     print(stage, " constraint ", self.constraint(LB_cz_val[stage], LB_cz_grad[stage], U[stage,3:5], X[stage,0:4], u_h[stage,-self.x_dim:], x_h[stage, :self.state_dim], self.params["common"]["Lc"]))
            if sqp_iter==(self.max_sqp_iter-1):
                if self.params["visu"]["show"]:
                    plt.figure(2)
                    if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
                        plt.plot(X[:,0],X[:,1], color="tab:green") # state
                        plt.plot(U[:,3],U[:,4], color="tab:blue") # l(x)
                    else:
                        plt.plot(X[:,0],X[:,1], color="tab:green")
                    plt.xlim(self.params["env"]["start"],self.params["env"]["start"] + self.params["visu"]["step_size"]*self.params["env"]["shape"]["x"])
                    plt.ylim(self.params["env"]["start"],self.params["env"]["start"] + self.params["visu"]["step_size"]*self.params["env"]["shape"]["y"])
                    # plt.axes().set_aspect('equal')
                    plt.savefig("temp.png")
            # print("statistics", self.ocp_solver.get_stats("statistics"))
            if max(residuals) < self.tol_nlp:
                print("Residual less than tol", max(
                    residuals), " ", self.tol_nlp)
                break
            # if self.ocp_solver.status != 0:
            #     print("acados returned status {} in closed loop solve".format(
            #         self.ocp_solver.status))
            #     self.ocp_solver.reset()
            #     self.ocp_solver.load_iterate(self.name_prefix + 'ocp_initialization.json')

    def constraint(self, lb_cz_lin, lb_cz_grad, model_z, model_x, z_lin, x_lin, Lc):
        x_dim = self.x_dim
        tol = 1e-5
        ret = lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin) - (Lc/(ca.norm_2(x_lin[:x_dim] - z_lin)+tol))*((x_lin[:x_dim] - z_lin).T@(model_x-x_lin)[:x_dim]) - (Lc/(ca.norm_2(x_lin[:x_dim] - z_lin)+tol))*((z_lin-x_lin[:x_dim]).T@(model_z-z_lin)) - Lc*ca.norm_2(x_lin[:x_dim] - z_lin)
        # ret = lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin) - 2*Lc*(x_lin[:x_dim] - z_lin).T@(model_x-x_lin)[:x_dim] - 2*Lc*(z_lin-x_lin[:x_dim]).T@(model_z-z_lin) - Lc*(x_lin[:x_dim] - z_lin).T@(x_lin[:x_dim] - z_lin)
        return ret, lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin)
    
    def model_ss(self, model_x):
        val = (model_x - model.f_expl_expr[:-1])

    def get_solution(self):
        X = np.zeros((self.H+1, self.nx))
        U = np.zeros((self.H, self.nu))
        Sl = np.zeros((self.H+1))

        # get data
        for i in range(self.H):
            X[i, :] = self.ocp_solver.get(i, "x")
            U[i, :] = self.ocp_solver.get(i, "u")
            # Sl[i] = self.ocp_solver.get(i, "sl")

        X[self.H, :] = self.ocp_solver.get(self.H, "x")
        return X, U, Sl

    def get_solver_status():
        return None
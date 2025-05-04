import torch
import numpy as np
import scipy.linalg
import scipy.signal
import casadi as ca

class Pendulum(object):
    def __init__(self, params):
        self.params = params
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.g_ny = self.params["agent"]["g_dim"]["ny"]
        self.pad_g = [0, 1, 2, 3]  # 0, self.g_nx + self.g_nu :
        if self.params["common"]["use_cuda"] and torch.cuda.is_available():
            self.use_cuda = True
            self.torch_device = torch.device("cuda")
            torch.set_default_device(self.torch_device)
        else:
            self.use_cuda = False
            self.torch_device = torch.device("cpu")
            torch.set_default_device(self.torch_device)

        self.B_d = torch.eye(self.nx, self.g_ny, device=self.torch_device)

    def initial_training_data(self):
        # keep low output scale, TODO: check if variance on gradient output can be controlled Dyn_gp_task_noises
        n_data_x = self.params["env"]["n_data_x"]
        n_data_u = self.params["env"]["n_data_u"]

        if self.params["env"]["prior_dyn_meas"]:
            x1 = torch.linspace(
                self.params["optimizer"]["x_min"][0],
                self.params["optimizer"]["x_max"][0],
                n_data_x,
            )
            x2 = torch.linspace(
                self.params["optimizer"]["x_min"][1],
                self.params["optimizer"]["x_max"][1],
                n_data_x,
            )
            u = torch.linspace(
                self.params["optimizer"]["u_min"][0],
                self.params["optimizer"]["u_max"][0],
                n_data_u,
            )
            X1, X2, U = torch.meshgrid(x1, x2, u)
            Dyn_gp_X_train = torch.hstack(
                [X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)]
            )
            # y1, y2 = self.get_prior_data(Dyn_gp_X_train)
            # Dyn_gp_Y_train = torch.stack((y1, y2), dim=0)
            Dyn_gp_Y_train = self.get_prior_data(Dyn_gp_X_train)
        else:
            Dyn_gp_X_train = torch.rand(1, self.in_dim)
            Dyn_gp_Y_train = torch.rand(2, 1, 1 + self.in_dim)

        if not self.params["env"]["train_data_has_derivatives"]:
            Dyn_gp_Y_train[:, :, 1:] = torch.nan

        return Dyn_gp_X_train, Dyn_gp_Y_train

    def get_prior_data(self, x_hat):
        l = self.params["env"]["params"]["l"]
        g = self.params["env"]["params"]["g"]
        dt = self.params["optimizer"]["dt"]
        g_xu = self.unknown_dyn(x_hat)
        y1_fx, y2_fx = g_xu[:, 0], g_xu[:, 1]
        # y1_ret = torch.zeros((x_hat.shape[0], 4))
        # y2_ret = torch.zeros((x_hat.shape[0], 4))
        # y1_ret[:, 0] = y1_fx
        # y1_ret[:, 1] = torch.ones(x_hat.shape[0])
        # y1_ret[:, 2] = torch.ones(x_hat.shape[0]) * dt

        # y2_ret[:, 0] = y2_fx
        # y2_ret[:, 1] = (-g * torch.cos(x_hat[:, 0]) / l) * dt
        # y2_ret[:, 2] = torch.ones(x_hat.shape[0])
        # y2_ret[:, 3] = torch.ones(x_hat.shape[0]) * dt / (l * l)
        # return y1_ret, y2_ret
        ny = 2
        y_ret = torch.zeros((ny, x_hat.shape[0], 4))
        # y2_ret = torch.zeros((x_hat.shape[0], 4))
        y_ret[0, :, 0] = y1_fx
        y_ret[0, :, 1] = torch.ones(x_hat.shape[0])
        y_ret[0, :, 2] = torch.ones(x_hat.shape[0]) * dt

        y_ret[1, :, 0] = y2_fx
        y_ret[1, :, 1] = (-g * torch.cos(x_hat[:, 0]) / l) * dt
        y_ret[1, :, 2] = torch.ones(x_hat.shape[0])
        y_ret[1, :, 3] = torch.ones(x_hat.shape[0]) * dt / (l * l)
        return y_ret

    def continous_dyn(self, X1, X2, U):
        """_summary_

        Args:
            x (_type_): _description_
            u (_type_): _description_
        """
        m = self.params["env"]["params"]["m"]
        l = self.params["env"]["params"]["l"]
        g = self.params["env"]["params"]["g"]
        X1dot = X2.clone()
        X2dot = -g * torch.sin(X1) / l + U / l
        train_data_y = torch.hstack([X1dot.reshape(-1, 1), X2dot.reshape(-1, 1)])
        return train_data_y

    def get_true_gradient(self, x_hat):
        l = self.params["env"]["params"]["l"]
        g = self.params["env"]["params"]["g"]
        ret = torch.zeros((2, x_hat.shape[0], 3))
        ret[0, :, 1] = torch.ones(x_hat.shape[0])
        ret[1, :, 0] = -g * torch.cos(x_hat[:, 0]) / l
        ret[1, :, 2] = torch.ones(x_hat.shape[0]) / l

        val = self.pendulum_dyn(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2])
        return torch.hstack([val[:, 0].reshape(-1, 1), ret[0, :, :]]), torch.hstack(
            [val[:, 1].reshape(-1, 1), ret[1, :, :]]
        )

    def discrete_dyn(self, xu):
        return self.unknown_dyn(xu)

    def unknown_dyn(self, xu):
        m = self.params["env"]["params"]["m"]
        l = self.params["env"]["params"]["l"]
        g = self.params["env"]["params"]["g"]
        X1_k, X2_k, U_k = xu[:, [0]], xu[:, [1]], xu[:, [2]]
        dt = self.params["optimizer"]["dt"]
        X1_kp1 = X1_k + X2_k * dt
        X2_kp1 = X2_k - g * torch.sin(X1_k) * dt / l + U_k * dt / (l * l)
        state_kp1 = torch.hstack([X1_kp1, X2_kp1])
        return state_kp1

    def get_f_known_jacobian(self, xu):
        ns = xu.shape[0]
        nH = xu.shape[2]
        # dimension is ns, ny, H, nx+nu
        df_dxu_grad = torch.zeros(
            (ns, self.nx, nH, 1 + self.nx + self.nu), device=xu.device
        )
        return df_dxu_grad

    def get_g_xu_hat(self, xu_hat):
        return xu_hat

    def LQR_controller(self):
        g = self.params["env"]["params"]["g"]
        m = self.params["env"]["params"]["m"]
        l = self.params["env"]["params"]["l"]
        b = 0

        R = np.diag(np.array(self.params["optimizer"]["Qu"]))
        Qx = np.diag(np.array(self.params["optimizer"]["Qx"]))

        # Continuous-time state-space matrices
        A = np.array([[0, 1], [-g / l, 0]])  # Change sign for new coordinate system
        B = np.array([[0], [1]])

        dt = self.params["optimizer"]["dt"]
        # Discretize the system using zero-order hold (ZOH)
        system = scipy.signal.cont2discrete((A, B, np.eye(2), 0), dt, method="zoh")
        A_d, B_d, _, _ = system[:4]

        # Solve the Discrete-time Algebraic Riccati Equation (DARE)
        P = scipy.linalg.solve_discrete_are(A_d, B_d, Qx, R)

        # Compute the Discrete LQR gain K
        K = np.linalg.inv(R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)
        print(K, P)

        return K, P, A_d, B_d

    def traj_initialize(self, x_curr):
        # Store results
        x_history = []
        u_history = []
        x = x_curr
        K, P, A_d, B_d = self.LQR_controller()
        # Run simulation
        for _ in range(self.params["optimizer"]["H"]):
            u = -K @ x  # LQR control law
            x = A_d @ x + B_d @ u  # Discrete-time state update
            x_history.append(x.flatten())
            u_history.append(u.flatten())
        return x_history, u_history

    def transform_sensitivity(self, dg_dxu_grad, xu_hat):
        return dg_dxu_grad

    def propagate_true_dynamics(self, x_init, U):
        state_list = []
        state_list.append(x_init)
        for ele in range(U.shape[0]):
            state_input = (
                torch.from_numpy(np.hstack([state_list[-1], U[ele]]))
                .reshape(1, -1)
                .float()
            )
            state_kp1 = self.discrete_dyn(state_input)
            state_list.append(state_kp1.reshape(-1))
        return np.stack(state_list)

    def feature_theta(self, state, control):
        theta, omega = state[0], state[1]
        return ca.vertcat(theta,omega)
    
    def feature_omega(self, state, control):
        theta, omega = state[0], state[1]
        alpha = control[0]  
        return ca.vertcat(omega, ca.sin(theta), alpha)

    def get_gt_weights(self):
        dt = self.params["optimizer"]["dt"]
        g = self.params["env"]["params"]["g"]
        l = self.params["env"]["params"]["l"]
        tr_weight = [[1.0, dt],[1.0,-g*dt/l, dt]]
        return tr_weight

    # General function to compute weighted gradient
    def compute_weighted_grad(feature_func, state, weights, control=None):
        """
        feature_func: callable(state) or callable(state, control)
        state: casadi MX vector of states (6,)
        weights: list or array of weights (same size as output of feature_func)
        control: optional, casadi MX vector (2,)
        """
        if control is not None:
            features = feature_func(state, control)
        else:
            features = feature_func(state)
            
        # Weighted sum of features
        weighted_sum = ca.dot(weights, features)  # sum_i w_i * feature_i
        
        # Compute gradient
        grad = ca.gradient(weighted_sum, state)
        return grad
    
    def BLR_features_casadi(self):
        import casadi as ca
        theta = ca.MX.sym("theta")
        omega = ca.MX.sym("omega")
        alpha = ca.MX.sym("alpa")
        state = ca.vertcat(theta, omega)
        control = ca.vertcat(alpha)

        # Build CasADi functions for fast evaluation
        f_theta = ca.Function('f_theta', [state, control], [self.feature_theta(state, control)])
        f_omega = ca.Function('f_omega', [state, control], [self.feature_omega(state, control)])
        
        # compute jacobian as well for theses features
        f_theta_jac = ca.Function('f_theta_jac', [state, control], [ca.jacobian(self.feature_theta(state, control), state)])
        f_omega_jac = ca.Function('f_omega_jac', [state, control], [ca.jacobian(self.feature_omega(state, control), state)])

        # compute jacobian as well for theses features
        f_theta_u_jac = ca.Function('f_theta_ujac', [state, control], [ca.jacobian(self.feature_theta(state, control), control)])
        f_omega_u_jac = ca.Function('f_omega_ujac', [state, control], [ca.jacobian(self.feature_omega(state, control), control)])

        # Suppose batch size = (50, 1, 30) --> total 50*1*30 = 1500 points
        batch_size = self.params["agent"]["num_dyn_samples"]
        batch_1 = 1
        batch_2 = self.params["optimizer"]["H"]
        total_samples = batch_size * batch_1 * batch_2

        # Map f_theta over total_samples
        f_theta_batch = f_theta.map(total_samples, 'serial')
        f_omega_batch = f_omega.map(total_samples, 'serial')

        f_theta_jac_batch = f_theta_jac.map(total_samples , 'serial')
        f_omega_jac_batch = f_omega_jac.map(total_samples, 'serial')
        f_theta_u_jac_batch = f_theta_u_jac.map(total_samples, 'serial')
        f_omega_u_jac_batch = f_omega_u_jac.map(total_samples, 'serial')

        f_list = [f_theta, f_omega]
        f_jac_list = [f_theta_jac, f_omega_jac]
        f_ujac_list = [f_theta_u_jac, f_omega_u_jac]
        f_batch_list = [f_theta_batch, f_omega_batch]
        f_jac_batch_list = [f_theta_jac_batch, f_omega_jac_batch]
        f_ujac_batch_list = [f_theta_u_jac_batch, f_omega_u_jac_batch]

        return f_list, f_jac_list, f_ujac_list, f_batch_list, f_jac_batch_list, f_ujac_batch_list
    
    def BLR_features(self, X):    
        theta = X[:, [0]]
        omega = X[:, [1]]
        alpha = X[:, [2]]
        # theta, vel, alpha

        f1 = np.hstack([theta, omega])
        f2 = np.hstack([omega, np.sin(theta), alpha])
        return f1, f2
        # return np.vstack([f1, f2])


    def BLR_features_test(self, X):   
        if X.ndim == 2:
            # X = X.reshape(X.shape[0], 1, X.shape[1], 1)
            X = X[:, np.newaxis, np.newaxis,:]
        theta = X[:,0:1,:, 0:1]
        omega = X[:,0:1,:, 1:2]
        alpha = X[:,0:1,:, 2:3]
        # theta, vel, alpha

        # Compute f1 and f2
        f1 = np.concatenate([theta, omega], axis=-1)                 # (50,2,30,2)
        f2 = np.concatenate([omega, np.sin(theta), alpha], axis=-1)   # (50,2,30,3)


        # Determine max feature size
        max_dim = max(f1.shape[-1], f2.shape[-1])
        Phi = np.zeros((X.shape[0], self.g_ny, X.shape[2], max_dim))
        Phi[:,[0],:,:f1.shape[-1]] = f1
        Phi[:,[1],:,:f2.shape[-1]] = f2
        return Phi, [f1[:,0,0,:], f2[:,0,0,:]]

    def BLR_features_grad(self, X):
        theta = X[:,0:1,:, 0:1]
        omega = X[:,0:1,:, 1:2]
        alpha = X[:,0:1,:, 2:3]
        # theta, vel, alpha

        f1_grad = np.concatenate([
            np.ones_like(theta),
            np.ones_like(omega)
        ], axis=-1)
        f2_grad = np.concatenate([
            np.ones_like(omega),
            np.cos(theta),
            np.ones_like(alpha)
        ], axis=-1)

        # Determine max feature size
        max_dim = max(f1_grad.shape[-1], f2_grad.shape[-1])
        Phi_grad = np.zeros((X.shape[0], self.g_ny, X.shape[2], max_dim))
        Phi_grad[:,[0],:,:f1_grad.shape[-1]] = f1_grad
        Phi_grad[:,[1],:,:f2_grad.shape[-1]] = f2_grad

        return Phi_grad

    def ocp_handler(self, func_name, *args, **kwargs):
        """
        Dynamically dispatch OCP function by name.
        
        Args:
            func_name (str): Name of the function to call, e.g., 'cost', 'constraint', etc.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass.
        
        Returns:
            Result of the called function.
        """
        if not hasattr(self, func_name):
            raise ValueError(f"Unknown OCP function requested: {func_name}")

        func = getattr(self, func_name)
        return func(*args, **kwargs)

    def const_expr(self, model_x, num_dyn):
        const_expr = []
        nx = self.params["agent"]["dim"]["nx"]
        for i in range(num_dyn):
            xf = np.array(self.params["env"]["terminate_state"])
            xf_dim = xf.shape[0]
            expr = (
                (model_x[nx * i : nx * (i + 1)][:xf_dim] - xf).T
                @ np.array(self.params["optimizer"]["terminal_tightening"]["P"])
                @ (model_x[nx * i : nx * (i + 1)][:xf_dim] - xf)
            )
            const_expr = ca.vertcat(const_expr, expr)
        return const_expr
    
    def const_value(self, num_dyn):
        lh = np.empty((0,), dtype=np.float64)
        uh = np.empty((0,), dtype=np.float64)
        delta = self.params["optimizer"]["terminal_tightening"]["delta"]
        lh_e = np.hstack([[0] * num_dyn])
        uh_e = np.hstack([[delta] * num_dyn])
        return lh, uh, lh_e, uh_e

    def initialize_plot_handles(self, fig_gp, fig_dyn=None):
        import matplotlib.pyplot as plt
        # if self.params["env"]["dynamics"] == "bicycle":
        #     fig_gp, ax = plt.subplots(figsize=(30 / 2.4, 3.375 / 2.4))
        # elif "endulum" in self.params["env"]["dynamics"]:
        #     fig_gp, ax = plt.subplots(figsize=(8 / 2.4, 8 / 2.4))
        # fig_gp.tight_layout(pad=0)
        ax = fig_gp.axes[0]
        ax.grid(which="both", axis="both")
        ax.minorticks_on()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        y_min = self.params["optimizer"]["x_min"][1]
        y_max = self.params["optimizer"]["x_max"][1]
        x_min = self.params["optimizer"]["x_min"][0]
        x_max = self.params["optimizer"]["x_max"][0]
        if self.params["env"]["dynamics"] == "bicycle":
            x_max = self.params["optimizer"]["x_max"][0]
            y_ref = self.params["env"]["goal_state"][1]

            ax.add_line(
                plt.Line2D([x_min, x_max], [y_max, y_max], color="red", linestyle="--")
            )
            ax.add_line(
                plt.Line2D([x_min, x_max], [y_min, y_min], color="red", linestyle="--")
            )
            ax.add_line(
                plt.Line2D(
                    [x_min, x_max],
                    [y_ref, y_ref],
                    color="cyan",
                    linestyle=(0, (5, 5)),
                    lw=2,
                )
            )
            # ellipse = Ellipse(xy=(1, 0), width=1.414, height=1,
            #                 edgecolor='r', fc='None', lw=2)
            # ax.add_patch(ellipse)
            if self.params["env"]["ellipses"]:
                for ellipse in self.params["env"]["ellipses"]:
                    x0 = self.params["env"]["ellipses"][ellipse][0]
                    y0 = self.params["env"]["ellipses"][ellipse][1]
                    a_sq = self.params["env"]["ellipses"][ellipse][2]
                    b_sq = self.params["env"]["ellipses"][ellipse][3]
                    f = self.params["env"]["ellipses"][ellipse][4]
                    # u = 1.0  # x-position of the center
                    # v = 0.1  # y-position of the center
                    # f = 0.01
                    a = np.sqrt(a_sq * f)  # radius on the x-axis
                    b = np.sqrt(b_sq * f)  # radius on the y-axis
                    t = np.linspace(0, 2 * np.pi, 100)
                    f2 = 0.5  # plot 2 ellipses, 1 for ego, 1 for other
                    # plt.plot(x0 + a * np.cos(t), y0 + b * np.sin(t))
                    plt.plot(
                        x0 + f2 * a * np.cos(t),
                        y0 + f2 * b * np.sin(t),
                        "black",
                        alpha=0.5,
                    )
                    # plot constarint ellipse
                    plt.plot(x0 + a * np.cos(t), y0 + b * np.sin(t), "gray", alpha=0.5)
                    self.plot_car_stationary(x0, y0, 0, plt)
            # plt.grid(color="lightgray", linestyle="--")
            ax.set_aspect("equal", "box")
            ax.set_xlim(x_min, x_max - 10)
            relax = 0
            ax.set_ylim(y_min - relax, y_max + relax)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.xticks([])
            plt.yticks([])
            plt.xlim([-2.14, 70 + relax])
            plt.tight_layout(pad=0.3)

        elif "endulum" in self.params["env"]["dynamics"]:
            ax.add_line(
                plt.Line2D([x_min, x_max], [y_max, y_max], color="red", linestyle="--")
            )
            ax.add_line(
                plt.Line2D([x_max, x_max], [y_min, y_max], color="red", linestyle="--")
            )
            ax.set_aspect("equal", "box")
            relax = 0.3
            ax.set_xlim(x_min - relax, x_max + relax)
            ax.set_ylim(y_min - relax, y_max + relax)

            if "P" in self.params["optimizer"]["terminal_tightening"]:
                xf = np.array(self.params["env"]["start"])
                P = np.array(self.params["optimizer"]["terminal_tightening"]["P"])
                delta = self.params["optimizer"]["terminal_tightening"]["delta"]
                L = np.linalg.cholesky(P / delta)
                t = np.linspace(0, 2 * np.pi, 200)
                z = np.vstack([np.cos(t), np.sin(t)])
                ell = np.linalg.inv(L.T) @ z

                ax.plot(
                    ell[0, :] + xf[0],
                    ell[1, :] + xf[1],
                    color="red",
                    label="Terminal set",
                )

        # # ax.set_yticklabels([])
        # # ax.set_xticklabels([])
        # # ax.set_xticks([])
        # # ax.set_yticks([])

        # fig_dyn, ax2 = plt.subplots()  # plt.subplots(2,2)

        # # ax2.set_aspect('equal', 'box')
        # self.f_handle = {}
        # self.f_handle["gp"] = fig_gp
        # self.f_handle["dyn"] = fig_dyn
        # # self.plot_contour_env("dyn")

        # # Move it to visu
        # self.writer_gp = self.get_frame_writer()
        # self.writer_dyn = self.get_frame_writer()
        # self.writer_dyn.setup(fig_dyn, path + "/video_dyn.mp4", dpi=200)
        # self.writer_gp.setup(fig_gp, path + "/video_gp.mp4", dpi=300) 
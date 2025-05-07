import torch
import numpy as np
import scipy.linalg
import scipy.signal
import casadi as ca

class Drone(object):
    def __init__(self, params):
        self.params = params
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.g_ny = self.params["agent"]["g_dim"]["ny"]
        self.g_nx = self.params["agent"]["g_dim"]["nx"]
        self.g_nu = self.params["agent"]["g_dim"]["nu"]
        self.pad_g = [0, 1, 2, 3]  # 0, self.g_nx + self.g_nu :
        self.datax_idx = [0,1,2,3,4,5]
        self.datau_idx = [0,1]
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
            mesh_list = []
            for idx in self.datax_idx:
                mesh_list.append(torch.linspace(
                self.params["optimizer"]["x_min"][idx],
                self.params["optimizer"]["x_max"][idx],
                n_data_x,
            ))
            for idx in self.datau_idx:
                mesh_list.append(torch.linspace(
                self.params["optimizer"]["u_min"][idx],
                self.params["optimizer"]["u_max"][idx],
                n_data_u,
            ))
            XU_grid = torch.meshgrid(mesh_list)

            Dyn_gp_X_train = torch.hstack(
                [XU.reshape(-1,1) for XU in XU_grid]
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
        y_ret = torch.zeros((self.g_ny, x_hat.shape[0], 1 + self.g_nx + self.g_nu))
        y_ret[:, :, 0] = g_xu.transpose(0, 1)
        return y_ret

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
        d = self.params["env"]["params"]["d"]
        J = self.params["env"]["params"]["J"]
        px_k, py_k, phi_k, vx_k, vy_k, phidot_k = xu[:, [0]], xu[:, [1]], xu[:, [2]], xu[:, [3]], xu[:, [4]], xu[:, [5]] 
        u1_k, u2_k = xu[:, [6]], xu[:, [7]]
        dt = self.params["optimizer"]["dt"]
        px_kp1 = px_k + (vx_k * torch.cos(phi_k) - vy_k*torch.sin(phi_k))* dt
        py_kp1 = py_k + (vx_k * torch.sin(phi_k) + vy_k*torch.cos(phi_k))* dt
        phi_kp1 = phi_k + phidot_k * dt
        vx_kp1 = vx_k + (vy_k*phidot_k - g * torch.sin(phi_k) + torch.cos(phi_k)*d)* dt
        vy_kp1 = vy_k + (-vx_k*phidot_k - g * torch.cos(phi_k) + u1_k/m + u2_k/m - torch.sin(phi_k)*d)* dt
        phidot_kp1 = phidot_k + (u1_k - u2_k)*l/J*dt
        state_kp1 = torch.hstack([px_kp1, py_kp1, phi_kp1, vx_kp1, vy_kp1, phidot_kp1])
        return state_kp1
    
    def get_gt_weights(self):
        dt = self.params["optimizer"]["dt"]
        m = self.params["env"]["params"]["m"]
        l = self.params["env"]["params"]["l"]
        g = self.params["env"]["params"]["g"]
        d = self.params["env"]["params"]["d"]
        J = self.params["env"]["params"]["J"]

        tr_weight = [[1.0, dt, -dt],
                     [1.0, dt, dt],
                     [1.0, dt],
                     [1.0, dt, -g*dt, d*dt],
                     [1.0, -dt, -g*dt,  -d*dt, dt/m, dt/m],
                     [1.0, dt*l/J, -dt*l/J]]
        return tr_weight

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

    def BLR_features_casadi(self):
        import casadi as ca

        # === 1. Define CasADi symbols ===
        px = ca.MX.sym("px")
        py = ca.MX.sym("py")
        phi = ca.MX.sym("phi")
        vx = ca.MX.sym("vx")
        vy = ca.MX.sym("vy")
        phidot = ca.MX.sym("phidot")
        u1 = ca.MX.sym("u1")
        u2 = ca.MX.sym("u2")

        state = ca.vertcat(px, py, phi, vx, vy, phidot)
        control = ca.vertcat(u1, u2)

        # === 2. Build CasADi feature functions ===
        f_px = ca.Function('f_px', [state, control], [self.feature_px(state, control)])
        f_py = ca.Function('f_py', [state, control], [self.feature_py(state, control)])
        f_phi = ca.Function('f_phi', [state, control], [self.feature_phi(state, control)])
        f_vx = ca.Function('f_vx', [state, control], [self.feature_vx(state, control)])
        f_vy = ca.Function('f_vy', [state, control], [self.feature_vy(state, control)])
        f_phidot = ca.Function('f_phidot', [state, control], [self.feature_phidot(state, control)])

        # === 3. Compute Jacobians (w.r.t. state) ===
        f_px_jac = ca.Function('f_px_jac', [state, control], [ca.densify(ca.jacobian(self.feature_px(state, control), state))])
        f_py_jac = ca.Function('f_py_jac', [state, control], [ca.densify(ca.jacobian(self.feature_py(state, control), state))])
        f_phi_jac = ca.Function('f_phi_jac', [state, control], [ca.densify(ca.jacobian(self.feature_phi(state, control), state))])
        f_vx_jac = ca.Function('f_vx_jac', [state, control], [ca.densify(ca.jacobian(self.feature_vx(state, control), state))])
        f_vy_jac = ca.Function('f_vy_jac', [state, control], [ca.densify(ca.jacobian(self.feature_vy(state, control), state))])
        f_phidot_jac = ca.Function('f_phidot_jac', [state, control], [ca.densify(ca.jacobian(self.feature_phidot(state, control), state))])

        # === 4. Compute Jacobians (w.r.t. control) ===
        f_px_u_jac = ca.Function('f_px_ujac', [state, control], [ca.densify(ca.jacobian(self.feature_px(state, control), control))])
        f_py_u_jac = ca.Function('f_py_ujac', [state, control], [ca.densify(ca.jacobian(self.feature_py(state, control), control))])
        f_phi_u_jac = ca.Function('f_phi_ujac', [state, control], [ca.densify(ca.jacobian(self.feature_phi(state, control), control))])
        f_vx_u_jac = ca.Function('f_vx_ujac', [state, control], [ca.densify(ca.jacobian(self.feature_vx(state, control), control))])
        f_vy_u_jac = ca.Function('f_vy_ujac', [state, control], [ca.densify(ca.jacobian(self.feature_vy(state, control), control))])
        f_phidot_u_jac = ca.Function('f_phidot_ujac', [state, control], [ca.densify(ca.jacobian(self.feature_phidot(state, control), control))])

        # === 5. Setup batch mapping ===
        batch_size = self.params["agent"]["num_dyn_samples"]
        batch_1 = 1
        batch_2 = self.params["optimizer"]["H"]
        total_samples = batch_size * batch_1 * batch_2

        # Map feature evaluations
        f_px_batch = f_px.map(total_samples, 'serial')
        f_py_batch = f_py.map(total_samples, 'serial')
        f_phi_batch = f_phi.map(total_samples, 'serial')
        f_vx_batch = f_vx.map(total_samples, 'serial')
        f_vy_batch = f_vy.map(total_samples, 'serial')
        f_phidot_batch = f_phidot.map(total_samples, 'serial')

        # Map feature jacobians w.r.t. state
        f_px_jac_batch = f_px_jac.map(total_samples, 'serial')
        f_py_jac_batch = f_py_jac.map(total_samples, 'serial')
        f_phi_jac_batch = f_phi_jac.map(total_samples, 'serial')
        f_vx_jac_batch = f_vx_jac.map(total_samples, 'serial')
        f_vy_jac_batch = f_vy_jac.map(total_samples, 'serial')
        f_phidot_jac_batch = f_phidot_jac.map(total_samples, 'serial')

        # Map feature jacobians w.r.t. control
        f_px_u_jac_batch = f_px_u_jac.map(total_samples, 'serial')
        f_py_u_jac_batch = f_py_u_jac.map(total_samples, 'serial')
        f_phi_u_jac_batch = f_phi_u_jac.map(total_samples, 'serial')
        f_vx_u_jac_batch = f_vx_u_jac.map(total_samples, 'serial')
        f_vy_u_jac_batch = f_vy_u_jac.map(total_samples, 'serial')
        f_phidot_u_jac_batch = f_phidot_u_jac.map(total_samples, 'serial')

        # === 6. Return everything organized ===
        f_list = [f_px, f_py, f_phi, f_vx, f_vy, f_phidot]
        f_jac_list = [f_px_jac, f_py_jac, f_phi_jac, f_vx_jac, f_vy_jac, f_phidot_jac]
        f_ujac_list = [f_px_u_jac, f_py_u_jac, f_phi_u_jac, f_vx_u_jac, f_vy_u_jac, f_phidot_u_jac]
        f_batch_list = [f_px_batch, f_py_batch, f_phi_batch, f_vx_batch, f_vy_batch, f_phidot_batch]
        f_jac_batch_list = [f_px_jac_batch, f_py_jac_batch, f_phi_jac_batch, f_vx_jac_batch, f_vy_jac_batch, f_phidot_jac_batch]
        f_ujac_batch_list = [f_px_u_jac_batch, f_py_u_jac_batch, f_phi_u_jac_batch, f_vx_u_jac_batch, f_vy_u_jac_batch, f_phidot_u_jac_batch]

        return f_list, f_jac_list, f_ujac_list, f_batch_list, f_jac_batch_list, f_ujac_batch_list


    def BLR_features(self, X):    
        theta = X[:, [0]]
        omega = X[:, [1]]
        alpha = X[:, [2]]
        # theta, vel, alpha

        f1 = np.hstack([theta,omega])
        f2 = np.hstack([omega, np.sin(theta),alpha])
        return f1, f2
        # return np.vstack([f1, f2])

    def feature_px(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        return ca.vertcat(px, vx * ca.cos(phi), vy * ca.sin(phi))

    def feature_py(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        return ca.vertcat(py, vx * ca.sin(phi), vy * ca.cos(phi))

    def feature_phi(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        return ca.vertcat(phi, phidot)

    def feature_vx(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        return ca.vertcat(vx, vy * phidot, ca.sin(phi), ca.cos(phi))

    def feature_vy(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        u1, u2 = control[0], control[1]
        return ca.vertcat(vy, vx * phidot, ca.cos(phi), ca.sin(phi), u1, u2)

    def feature_phidot(self, state, control):
        px, py, phi, vx, vy, phidot = state[0], state[1], state[2], state[3], state[4], state[5]
        u1, u2 = control[0], control[1]
        return ca.vertcat(phidot, u1, u2)

    def BLR_features_test(self, X):
        if X.ndim == 2:
            # X = X.reshape(X.shape[0], 1, X.shape[1], 1)
            X = X[:, np.newaxis, np.newaxis,:]
        # X shape: (batch_size, 2, horizon, 8)
        px = X[:, 0:1, :, 0:1]
        py = X[:, 0:1, :, 1:2]
        phi = X[:, 0:1, :, 2:3]
        vx = X[:, 0:1, :, 3:4]
        vy = X[:, 0:1, :, 4:5]
        phidot = X[:, 0:1, :, 5:6]
        u1 = X[:, 0:1, :, 6:7]
        u2 = X[:, 0:1, :, 7:8]

        f_px = np.concatenate([px, vx*np.cos(phi), vy*np.sin(phi)], axis=-1)
        f_py = np.concatenate([py, vx*np.sin(phi), vy*np.cos(phi)], axis=-1)
        f_phi = np.concatenate([phi, phidot], axis=-1)
        f_vx = np.concatenate([vx, vy*phidot, np.sin(phi), np.cos(phi)], axis=-1)
        f_vy = np.concatenate([vy, vx*phidot, np.sin(phi), np.cos(phi), u1, u2], axis=-1)
        f_phidot = np.concatenate([phidot, u1, u2], axis=-1)

        feature_list = [f_px, f_py, f_phi, f_vx, f_vy, f_phidot]

        # Find maximum feature dimension across all outputs
        max_dim = max([f.shape[-1] for f in feature_list])

        Phi = np.zeros((X.shape[0], self.g_ny, X.shape[2], max_dim))  # (batch, 6, horizon, max_dim)

        for idx, f in enumerate(feature_list):
            Phi[:, [idx], :, :f.shape[-1]] = f
        
        return Phi, [feature[:,0,0,:] for feature in feature_list]
        # return Phi, feature_list

    def BLR_features_grad(self, X):
        # X shape: (batch_size, 2, horizon, 8)
        px = X[:, 0:1, :, 0:1]
        py = X[:, 0:1, :, 1:2]
        phi = X[:, 0:1, :, 2:3]
        vx = X[:, 0:1, :, 3:4]
        vy = X[:, 0:1, :, 4:5]
        phidot = X[:, 0:1, :, 5:6]
        u1 = X[:, 0:1, :, 6:7]
        u2 = X[:, 0:1, :, 7:8]

        ### Now: correct addition rule and product rule

        ## f_px = [px, vx*cos(phi), vy*sin(phi)]
        f_px_grad = np.concatenate([
            np.ones_like(px),                                    # d(px)/d(px)
            vx * (-np.sin(phi)) + np.cos(phi),                   # d(vx*cos(phi))/d(vx,phi)
            vy * np.cos(phi) + np.sin(phi),                      # d(vy*sin(phi))/d(vy,phi)
        ], axis=-1)

        ## f_py = [py, vx*sin(phi), vy*cos(phi)]
        f_py_grad = np.concatenate([
            np.ones_like(py),                                    # d(py)/d(py)
            vx * np.cos(phi) + np.sin(phi),                      # d(vx*sin(phi))/d(vx,phi)
            vy * (-np.sin(phi)) + np.cos(phi),                   # d(vy*cos(phi))/d(vy,phi)
        ], axis=-1)

        ## f_phi = [phi, phidot]
        f_phi_grad = np.concatenate([
            np.ones_like(phi),                                   # d(phi)/d(phi)
            np.ones_like(phidot),                                # d(phidot)/d(phidot)
        ], axis=-1)

        ## f_vx = [vx, vy*phidot, sin(phi), cos(phi)]
        f_vx_grad = np.concatenate([
            np.ones_like(vx),                                    # d(vx)/d(vx)
            vy + phidot,                                                  # d(vy*phidot)/d(vy)                                    # d(vy*phidot)/d(phidot)
            np.cos(phi),                                         # d(sin(phi))/d(phi)
            -np.sin(phi),                                        # d(cos(phi))/d(phi)
        ], axis=-1)

        ## f_vy = [vy, vx*phidot, sin(phi), cos(phi), u1, u2]
        f_vy_grad = np.concatenate([
            np.ones_like(vy),                                    # d(vy)/d(vy)
            vx + phidot,                                               # d(vx*phidot)/d(vx)                                              # d(vx*phidot)/d(phidot)
            np.cos(phi),                                         # d(sin(phi))/d(phi)
            -np.sin(phi),                                        # d(cos(phi))/d(phi)
            np.ones_like(u1),                                    # d(u1)/d(u1)
            np.ones_like(u2),                                    # d(u2)/d(u2)
        ], axis=-1)

        ## f_phidot = [phidot, u1, u2]
        f_phidot_grad = np.concatenate([
            np.ones_like(phidot),                                # d(phidot)/d(phidot)
            np.ones_like(u1),                                    # d(u1)/d(u1)
            np.ones_like(u2),                                    # d(u2)/d(u2)
        ], axis=-1)

        # Collect gradients
        grad_list = [f_px_grad, f_py_grad, f_phi_grad, f_vx_grad, f_vy_grad, f_phidot_grad]

        # Find maximum feature dimension
        max_dim = max(f.shape[-1] for f in grad_list)
        Phi_grad = np.zeros((X.shape[0], self.g_ny, X.shape[2], max_dim))

        for idx, f_grad in enumerate(grad_list):
            Phi_grad[:, [idx], :, :f_grad.shape[-1]] = f_grad

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
        v_dim=3
        for i in range(num_dyn):
            xf = np.array(self.params["env"]["terminate_state"])
            xf_dim = xf.shape[0]
            expr = (
                (model_x[nx * i : nx * (i + 1)][v_dim:v_dim+xf_dim] - xf).T
                @ np.array(self.params["optimizer"]["terminal_tightening"]["P"])
                @ (model_x[nx * i : nx * (i + 1)][v_dim:v_dim+xf_dim] - xf)
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

    def cost_expr(self, model_x, model_u, ns, p, we, optimizer_str):
        pos_dim = 1
        nx = self.params["agent"]["dim"]["nx"]
        nu = self.params["agent"]["dim"]["nu"]
        Qu = np.diag(np.array(self.params[optimizer_str]["Qu"]))
        xg = np.array(self.params["env"]["goal_state"])
        xg_dim = xg.shape[0]
        w = self.params[optimizer_str]["w"]
        Qx = np.diag(np.array(self.params[optimizer_str]["Qx"]))

        # # cost
        # if self.params["optimizer"]["cost"] == "mean":
        #     ns = 1
        # else:
        #     ns = self.params["agent"]["num_dyn_samples"]
        expr = 0
        expr_e=0
        v_max = np.array([10,10])
        for i in range(ns):
            expr += (
                (model_x[nx * i : nx * (i + 1)][:xg_dim] - p).T
                @ Qx
                @ (model_x[nx * i : nx * (i + 1)][:xg_dim] - p)
                # + (model_x[nx * i : nx * (i + 1)][3:3+xg_dim] - v_max).T
                # @ (Qx/50)
                # @ (model_x[nx * i : nx * (i + 1)][3:3+xg_dim] - v_max) 
            )
            expr_e += (
                (model_x[nx * i : nx * (i + 1)][:xg_dim] - we).T
                @ Qx
                @ (model_x[nx * i : nx * (i + 1)][:xg_dim] - we)
            )
        cost_expr_ext_cost = expr / ns + model_u.T @ (Qu) @ model_u
        cost_expr_ext_cost_e = expr_e / ns
        return cost_expr_ext_cost, cost_expr_ext_cost_e
    
    def path_generator(self, st, length=None):
        # Generate values for t from 0 to 2π
        if length is None:
            length = self.params["optimizer"]["H"] + 1
        s = np.linspace(0, 4 * np.pi, 1000)
        t = s[st:st+length] #np.linspace(st + 0, st + 2 * np.pi/100*length, )

        # # Parametric equations for heart
        # x = 16 * np.sin(t)**3
        # y = 10 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)

        # Parametric equations for heart
        x = 8 * np.sin(t)**3 / 1.5 + 1
        y = (10 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))/2 - 1 + 1

        traj = np.vstack([x,y]).T
        return traj

    def initialize_plot_handles(self, fig_gp, fig_dyn=None):
        import matplotlib.pyplot as plt
        ax = fig_gp.axes[0]
        ax.set_xlim(-8,8)
        ax.set_ylim(-6, 6)

        ax.grid(which="both", axis="both")
        ax.minorticks_on()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        y_min = self.params["optimizer"]["x_min"][1]
        y_max = self.params["optimizer"]["x_max"][1]
        x_min = self.params["optimizer"]["x_min"][0]
        x_max = self.params["optimizer"]["x_max"][0]

        tracking_path = self.path_generator(0, 500)
        ax.plot(
            tracking_path[:, 0],
            tracking_path[:, 1],
            color="blue",
            label="Tracking path",
        )

        y_min = self.params["optimizer"]["x_min"][1]
        y_max = self.params["optimizer"]["x_max"][1]
        x_min = self.params["optimizer"]["x_min"][0]
        x_max = self.params["optimizer"]["x_max"][0]

        ax.add_line(
            plt.Line2D([x_min, x_max], [y_max, y_max], color="red", linestyle="--")
        )
        ax.add_line(
            plt.Line2D([x_max, x_max], [y_min, y_max], color="red", linestyle="--")
        )
        ax.add_line(
            plt.Line2D([x_min, x_min], [y_min, y_max], color="red", linestyle="--")
        )
        ax.add_line(
            plt.Line2D([x_min, x_max], [y_min, y_min], color="red", linestyle="--")
        )

        if "P" in self.params["optimizer"]["terminal_tightening"]:
            xf = np.array(self.params["env"]["terminate_state"])
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
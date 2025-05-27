import torch
import numpy as np
import scipy.linalg
import scipy.signal
import casadi as ca

class CarKinematicsModel(object):
    def __init__(self, params):
        self.params = params
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.g_ny = self.params["agent"]["g_dim"]["ny"]
        self.g_nx = self.params["agent"]["g_dim"]["nx"]
        self.g_nu = self.params["agent"]["g_dim"]["nu"]
        self.pad_g = [0, 3, 4, 5]  # 0, self.g_nx + self.g_nu :
        self.g_idx_inputs = [2, 3, 4]
        self.datax_idx = [0,1,2,3]
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
        # need more training data for decent result
        # keep low output scale, TODO: check if variance on gradient output can be controlled Dyn_gp_task_noises

        n_data_x = self.params["env"]["n_data_x"]
        n_data_u = self.params["env"]["n_data_u"]

        if self.params["env"]["prior_dyn_meas"]:
            # phi_min = self.params["optimizer"]["x_min"][2]
            # phi_max = self.params["optimizer"]["x_max"][2]
            # v_min = self.params["optimizer"]["x_min"][3]
            # v_max = self.params["optimizer"]["x_max"][3]
            # delta_min = self.params["optimizer"]["u_min"][0]
            # delta_max = self.params["optimizer"]["u_max"][0]
            # dphi = (phi_max - phi_min) / (n_data_x)  # removed -1; we want n gaps
            # dv = (v_max - v_min) / (n_data_x)
            # ddelta = (delta_max - delta_min) / (n_data_u)
            # phi = torch.linspace(phi_min + dphi / 2, phi_max - dphi / 2, n_data_x)
            # v = torch.linspace(v_min + dv / 2, v_max - dv / 2, n_data_x)
            # delta = torch.linspace(
            #     delta_min + ddelta / 2, delta_max - ddelta / 2, n_data_u
            # )
            # Phi, V, Delta = torch.meshgrid(phi, v, delta)
            # Dyn_gp_X_train = torch.hstack(
            #     [Phi.reshape(-1, 1), V.reshape(-1, 1), Delta.reshape(-1, 1)]
            # )
            # Dyn_gp_Y_train = self.get_prior_data(Dyn_gp_X_train)
            # # Dyn_gp_Y_train = torch.stack((y1, y2), dim=0)
            mesh_list = []
            for idx in self.datax_idx:
                mesh_list.append(torch.linspace(
                self.params["optimizer"]["data_x_min"][idx],
                self.params["optimizer"]["data_x_max"][idx],
                n_data_x,
            ))
            for idx in self.datau_idx:
                mesh_list.append(torch.linspace(
                self.params["optimizer"]["data_u_min"][idx],
                self.params["optimizer"]["data_u_max"][idx],
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
        g_xu = self.unknown_dyn(x_hat)
        y_ret = torch.zeros((self.g_ny, x_hat.shape[0], 1 + self.g_nx + self.g_nu))
        y_ret[:, :, 0] = g_xu.transpose(0, 1)
        return y_ret

    def get_prior_data_withgrad(self, xu):
        # NOTE: xu already needs to be filtered to only contain modeled inputs
        assert xu.shape[1] == 3

        dt = self.params["optimizer"]["dt"]
        nx = 2  # phi, v
        nu = 1  # delta
        ny = 3  # phi, v, delta
        lf = self.params["env"]["params"]["lf"]
        lr = self.params["env"]["params"]["lr"]
        # lf = 1.105 * 0.01
        # lr = 1.738 * 0.01
        phi, v, delta = xu[:, 0], xu[:, 1], xu[:, 2]
        g_xu = self.unknown_dyn(xu)  # phi, v, delta
        y_ret = torch.zeros((ny, xu.shape[0], 1 + nx + nu))

        y_ret[0, :, 0] = g_xu[:, 0]
        y_ret[1, :, 0] = g_xu[:, 1]
        y_ret[2, :, 0] = g_xu[:, 2]
        # derivative of dx w.r.t. phi, v, delta
        beta_in = (lr * torch.tan(delta)) / (lf + lr)
        beta = torch.atan(beta_in)
        y_ret[0, :, 1] = -v * torch.sin(phi + beta) * dt
        y_ret[0, :, 2] = torch.cos(phi + beta) * dt

        term = ((lr / (torch.cos(delta) ** 2)) / (lf + lr)) / (1 + beta_in**2)
        y_ret[0, :, 3] = -v * torch.sin(phi + beta) * dt * term

        # derivative of dy w.r.t. phi, v, delta
        y_ret[1, :, 1] = v * torch.cos(phi + beta) * dt
        y_ret[1, :, 2] = torch.sin(phi + beta) * dt
        y_ret[1, :, 3] = v * torch.cos(phi + beta) * dt * term

        # derivative of dphi w.r.t. phi, v, delta
        # y_ret[2,:, 0] is zeros
        y_ret[2, :, 2] = torch.sin(beta) * dt / lr
        y_ret[2, :, 3] = v * torch.cos(beta) * dt * term / lr
        return y_ret

    def get_f_known_jacobian(self, xu):
        # NOTE: xu has to be in shape used by optimizer
        assert len(xu.shape) == 4
        assert xu.shape[1] == self.nx
        assert xu.shape[3] == self.nx + self.nu

        f_val = self.known_dyn(xu)
        ns = xu.shape[0]
        nH = xu.shape[2]
        dt = self.params["optimizer"]["dt"]
        # dimension is ns, ny, H, nx+nu
        df_dxu_grad = torch.zeros(
            (ns, self.nx, nH, 1 + self.nx + self.nu), device=xu.device
        )
        # set function value on the 1st coordinate
        df_dxu_grad[:, :, :, 0] = f_val

        # set the derivative of the function w.r.t. state X
        # dX_kpi/dX_k = 1 , dX_kpi/dY_k = 0, dX_kpi/dPhi_k = 0, dX_kpi/dV_k = 0
        df_dxu_grad[:, 0, :, 1] = torch.ones((ns, nH))
        # dY_kpi/dY_k = 1 , dY_kpi/dX_k = 0, dY_kpi/dPhi_k = 0, dY_kpi/dV_k = 0
        df_dxu_grad[:, 1, :, 2] = torch.ones((ns, nH))
        # dPhi_kpi/dPhi_k, dPhi_kpi/dX_k = 0, dPhi_kpi/dY_k = 0, dPhi_kpi/dV_k = 0
        df_dxu_grad[:, 2, :, 3] = torch.ones((ns, nH))
        # dV_kpi/dV_k = 1 , dV_kpi/dX_k = 0, dV_kpi/dPhi_k = 0, dV_kpi/dY_k = 0
        df_dxu_grad[:, 3, :, 4] = torch.ones((ns, nH))

        # set the derivative of the function w.r.t. control input
        df_dxu_grad[:, 3, :, 6] = dt * torch.ones((ns, nH))  # dV_kpi/acc_k = dt
        return df_dxu_grad

    def get_g_xu_hat(self, xu_hat):
        # arg 1 truncated to self.g_ny -> all the same since tiles
        for i in range(xu_hat.shape[1] - 1):
            assert torch.all(xu_hat[:, i, :, 0] == xu_hat[:, i + 1, :, 0])

        return xu_hat[:, 0 : self.g_ny, :, self.g_idx_inputs]

    def known_dyn(self, xu):
        """_summary_"""
        # NOTE: xu has to be in shape used by optimizer
        assert len(xu.shape) == 4
        assert xu.shape[1] == self.nx
        assert xu.shape[3] == self.nx + self.nu

        X_k, Y_k, Phi_k, V_k = (
            xu[:, [0], :, 0],
            xu[:, [0], :, 1],
            xu[:, [0], :, 2],
            xu[:, [0], :, 3],
        )
        delta_k, acc_k = xu[:, [0], :, 4], xu[:, [0], :, 5]
        lf = self.params["env"]["params"]["lf"]
        lr = self.params["env"]["params"]["lr"]
        dt = self.params["optimizer"]["dt"]
        X_kp1 = X_k
        Y_kp1 = Y_k
        Phi_kp1 = Phi_k
        V_kp1 = V_k + acc_k * dt
        state_kp1 = torch.cat([X_kp1, Y_kp1, Phi_kp1, V_kp1], 1)
        return state_kp1

    def unknown_dyn(self, xu):
        lf = self.params["env"]["params"]["lf"]
        lr = self.params["env"]["params"]["lr"]
        dt = self.params["optimizer"]["dt"]
        c = self.params["env"]["params"]["c"]

        px_k, py_k, phi_k, vx_k = xu[:, [0]], xu[:, [1]], xu[:, [2]], xu[:, [3]]
        delta_k, ax_k = xu[:, [4]], xu[:, [5]]
        beta = torch.atan(torch.tan(delta_k) * lr / (lr + lf))
        px_kp1 = px_k +  vx_k * torch.cos(phi_k + beta) * dt
        py_kp1 = py_k + vx_k * torch.sin(phi_k + beta) * dt
        phi_kp1 = phi_k +  vx_k * torch.sin(beta) * dt / lr
        vx_kp1 = vx_k + ax_k* dt - c*vx_k*vx_k
        state_kp1 = torch.hstack([px_kp1, py_kp1, phi_kp1, vx_kp1])
        return state_kp1

    # def unknown_dyn(self, xu):
    #     """_summary_"""
    #     # NOTE: xu already needs to be filtered to only contain modeled inputs
    #     assert xu.shape[1] == len(self.g_idx_inputs)

    #     Phi_k, V_k, delta_k = xu[:, [0]], xu[:, [1]], xu[:, [2]]
    #     lf = self.params["env"]["params"]["lf"]
    #     lr = self.params["env"]["params"]["lr"]
    #     dt = self.params["optimizer"]["dt"]
    #     beta = torch.atan(torch.tan(delta_k) * lr / (lr + lf))
    #     dX_kp1 = V_k * torch.cos(Phi_k + beta) * dt
    #     dY_kp1 = V_k * torch.sin(Phi_k + beta) * dt
    #     Phi_kp1 = V_k * torch.sin(beta) * dt / lr
    #     state_kp1 = torch.hstack([dX_kp1, dY_kp1, Phi_kp1])
    #     return state_kp1

    def discrete_dyn(self, xu):
        return self.unknown_dyn(xu)

    # def discrete_dyn(self, xu):
    #     """_summary_"""
    #     # NOTE: takes only single xu
    #     assert xu.shape[1] == self.nx + self.nu

    #     f_xu = self.known_dyn(xu.tile((1, self.nx, 1, 1)))[0, :, :]
    #     g_xu = self.unknown_dyn(xu[:, self.g_idx_inputs]).transpose(0, 1)
    #     B_d = torch.eye(self.nx, self.g_ny).to(device="cpu", dtype=g_xu.dtype)
    #     return f_xu + torch.matmul(B_d, g_xu)

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

    def transform_sensitivity(self, dg_dxu_grad, xu_hat):
        return dg_dxu_grad


    def get_gt_weights(self):
        dt = self.params["optimizer"]["dt"]
        lf = self.params["env"]["params"]["lf"]
        lr = self.params["env"]["params"]["lr"]
        c = self.params["env"]["params"]["c"]

        tr_weight = [[1.0, dt],
                     [1.0, dt],
                     [1.0, dt/lr],
                     [1.0, dt, -c]]
        return tr_weight

    def feature_px(self, state, control):
        px, py, phi, vx = state[0], state[1], state[2], state[3]
        delta, ax = control[0], control[1]
        lf = self.params["env"]["params"]["lf"]
        lr = self.params["env"]["params"]["lr"]
        beta = ca.atan(ca.tan(delta) * lr / (lr + lf))
        return ca.vertcat(px, vx * ca.cos(phi + beta))

    def feature_py(self, state, control):
        px, py, phi, vx = state[0], state[1], state[2], state[3]
        delta, ax = control[0], control[1]
        lf = self.params["env"]["params"]["lf"]
        lr = self.params["env"]["params"]["lr"]
        beta = ca.atan(ca.tan(delta) * lr / (lr + lf))
        return ca.vertcat(py, vx * ca.sin(phi + beta))

    def feature_phi(self, state, control):
        px, py, phi, vx = state[0], state[1], state[2], state[3]
        delta, ax = control[0], control[1]
        lf = self.params["env"]["params"]["lf"]
        lr = self.params["env"]["params"]["lr"]
        beta = ca.atan(ca.tan(delta) * lr / (lr + lf))
        return ca.vertcat(phi, vx * ca.sin(beta))

    def feature_vx(self, state, control):
        px, py, phi, vx = state[0], state[1], state[2], state[3]
        delta, ax = control[0], control[1]
        return ca.vertcat(vx, ax, vx*vx)
    
    def BLR_features_casadi(self):
        import casadi as ca

        # Define the state and control variables
        px = ca.MX.sym("px")
        py = ca.MX.sym("py")
        phi = ca.MX.sym("phi")
        vx = ca.MX.sym("vx")
        delta = ca.MX.sym("delta")
        ax = ca.MX.sym("ax")

        # Define the state and control vector
        state = ca.vertcat(px, py, phi, vx)
        control = ca.vertcat(delta, ax)

        # Define the feature functions
        f_px = ca.Function('f_px', [state, control], [self.feature_px(state, control)])
        f_py = ca.Function('f_py', [state, control], [self.feature_py(state, control)])
        f_phi = ca.Function('f_phi', [state, control], [self.feature_phi(state, control)])
        f_vx = ca.Function('f_vx', [state, control], [self.feature_vx(state, control)])

        # === 3. Compute Jacobians (w.r.t. state) ===
        f_px_jac = ca.Function('f_px_jac', [state, control], [ca.densify(ca.jacobian(self.feature_px(state, control), state))])
        f_py_jac = ca.Function('f_py_jac', [state, control], [ca.densify(ca.jacobian(self.feature_py(state, control), state))])
        f_phi_jac = ca.Function('f_phi_jac', [state, control], [ca.densify(ca.jacobian(self.feature_phi(state, control), state))])
        f_vx_jac = ca.Function('f_vx_jac', [state, control], [ca.densify(ca.jacobian(self.feature_vx(state, control), state))])

        # === 4. Compute Jacobians (w.r.t. control) ===
        f_px_u_jac = ca.Function('f_px_u_jac', [state, control], [ca.densify(ca.jacobian(self.feature_px(state, control), control))])
        f_py_u_jac = ca.Function('f_py_u_jac', [state, control], [ca.densify(ca.jacobian(self.feature_py(state, control), control))])
        f_phi_u_jac = ca.Function('f_phi_u_jac', [state, control], [ca.densify(ca.jacobian(self.feature_phi(state, control), control))])
        f_vx_u_jac = ca.Function('f_vx_u_jac', [state, control], [ca.densify(ca.jacobian(self.feature_vx(state, control), control))])

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


        # Map feature jacobians w.r.t. state
        f_px_jac_batch = f_px_jac.map(total_samples, 'serial')
        f_py_jac_batch = f_py_jac.map(total_samples, 'serial')
        f_phi_jac_batch = f_phi_jac.map(total_samples, 'serial')
        f_vx_jac_batch = f_vx_jac.map(total_samples, 'serial')


        # Map feature jacobians w.r.t. control
        f_px_u_jac_batch = f_px_u_jac.map(total_samples, 'serial')
        f_py_u_jac_batch = f_py_u_jac.map(total_samples, 'serial')
        f_phi_u_jac_batch = f_phi_u_jac.map(total_samples, 'serial')
        f_vx_u_jac_batch = f_vx_u_jac.map(total_samples, 'serial')

        # === 6. Return everything organized ===
        f_list = [f_px, f_py, f_phi, f_vx]
        f_jac_list = [f_px_jac, f_py_jac, f_phi_jac, f_vx_jac]
        f_u_jac_list = [f_px_u_jac, f_py_u_jac, f_phi_u_jac, f_vx_u_jac]
        f_batch_list = [f_px_batch, f_py_batch, f_phi_batch, f_vx_batch]
        f_jac_batch_list = [f_px_jac_batch, f_py_jac_batch, f_phi_jac_batch, f_vx_jac_batch]
        f_u_jac_batch_list = [f_px_u_jac_batch, f_py_u_jac_batch, f_phi_u_jac_batch, f_vx_u_jac_batch]

        return f_list, f_jac_list, f_u_jac_list, f_batch_list, f_jac_batch_list, f_u_jac_batch_list

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
        px_cord=0
        for i in range(num_dyn):
            P1_outer = np.array(self.params["env"]["track"]["P1"])
            dim_track = P1_outer.shape[0]
            expr = (
                (model_x[nx * i : nx * (i + 1)][px_cord:px_cord+dim_track]).T
                @ np.array(self.params["env"]["track"]["P1"])
                @ (model_x[nx * i : nx * (i + 1)][px_cord:px_cord+dim_track])
            )
            const_expr = ca.vertcat(const_expr, expr)

        for i in range(num_dyn):
            P2_inner = np.array(self.params["env"]["track"]["P2"])
            dim_track = P2_inner.shape[0]
            expr = (
                (model_x[nx * i : nx * (i + 1)][px_cord:px_cord+dim_track]).T
                @ np.array(self.params["env"]["track"]["P2"])
                @ (model_x[nx * i : nx * (i + 1)][px_cord:px_cord+dim_track])
            )
            const_expr = ca.vertcat(const_expr, expr)

        # add 2 ellipse constraints with P
        # v_dim=3
        # for i in range(num_dyn):
        #     xf = np.array(self.params["env"]["terminate_state"])
        #     xf_dim = xf.shape[0]
        #     expr = (
        #         (model_x[nx * i : nx * (i + 1)][v_dim:v_dim+xf_dim] - xf).T
        #         @ np.array(self.params["optimizer"]["terminal_tightening"]["P"])
        #         @ (model_x[nx * i : nx * (i + 1)][v_dim:v_dim+xf_dim] - xf)
        #     )
        #     const_expr = ca.vertcat(const_expr, expr)
        return const_expr 
    
    def const_value(self, num_dyn):
        d1_outer = np.array(self.params["env"]["track"]["d1"])
        d2_inner = np.array(self.params["env"]["track"]["d2"])
        lh = np.hstack([[0] * num_dyn, [d2_inner] * num_dyn])
        uh = np.hstack([[d1_outer] * num_dyn, [1.0e4] * num_dyn])
        # lh = np.hstack([[0] * num_dyn])
        # uh = np.hstack([[d1_outer] * num_dyn])
        # delta = self.params["optimizer"]["terminal_tightening"]["delta"]
        # lh_e = np.hstack([[0] * num_dyn])
        # uh_e = np.hstack([[delta] * num_dyn])
        lh_e = np.empty((0,), dtype=np.float64)
        uh_e = np.empty((0,), dtype=np.float64)
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
        v_max = 20.0
        for i in range(ns):
            expr += (
                (model_x[nx * i : nx * (i + 1)][:xg_dim] - p).T
                @ Qx
                @ (model_x[nx * i : nx * (i + 1)][:xg_dim] - p)
                # + 0.1*(model_x[nx * i : nx * (i + 1)][3:] - v_max)**2
            )
            # expr_e += (
            #     (model_x[nx * i : nx * (i + 1)][:xg_dim] - we).T
            #     @ Qx
            #     @ (model_x[nx * i : nx * (i + 1)][:xg_dim] - we)
            # )
        cost_expr_ext_cost = expr / ns + model_u.T @ (Qu) @ model_u
        cost_expr_ext_cost_e = expr_e / ns
        return cost_expr_ext_cost, cost_expr_ext_cost_e
    

    def cost_expr_variance(self, model_x, model_u, ns, p, p_var, optimizer_str):
        pos_dim = 1
        nx = self.params["agent"]["dim"]["nx"]
        nu = self.params["agent"]["dim"]["nu"]
        Qu = np.diag(np.array(self.params[optimizer_str]["Qu"]))
        xg = np.array(self.params["env"]["goal_state"])
        xg_dim = xg.shape[0]
        w = self.params[optimizer_str]["w"]
        Qx = np.diag(np.array(self.params[optimizer_str]["Qx"]))

        f_list = [self.feature_px, self.feature_py, self.feature_phi, self.feature_vx]

        expr = 0
        expr_e=0
        cost = 0
        t_sh = 0
        f_shape = 0
        for feature in f_list:
            t_sh += f_shape
            for i in range(ns):
                f_val = feature(model_x[nx * i : nx * (i + 1)], model_u)
                f_shape = f_val.shape[0]
                expr += -f_val.T @ ca.diag(p_var[t_sh:t_sh+f_shape]) @ f_val


        cost_expr_ext_cost = expr / ns #+ model_u.T @ (Qu) @ model_u
        cost_expr_ext_cost_e = expr_e / ns
        return cost_expr_ext_cost, cost_expr_ext_cost_e



    def path_generator(self, st, length=None):
        # Generate values for t from 0 to 2π
        if length is None:
            length = self.params["optimizer"]["H"] + 1

        s = np.linspace(0, 4 * np.pi, 1000)
        theta = s[st:st+length] #np.linspace(st + 0, st + 2 * np.pi/100*length, )
        unit_circle = np.stack([np.sin(theta), np.cos(theta)])  # Shape: (2, N)
        P = np.array(self.params["env"]["track"]["Pc"]),
        d = np.array(self.params["env"]["track"]["dc"])
        # Transform unit circle to match ellipse shape
        L = np.linalg.cholesky(np.linalg.inv(P / d))  # such that x = L @ u gives the ellipse
        # if c is None:
        # c = np.zeros((2,))
        ellipse = (L @ unit_circle).T 

        return ellipse

    def initialize_plot_handles(self, fig_gp, fig_dyn=None):
        import matplotlib.pyplot as plt
        ax = fig_gp.axes[0]        
        ax.set_xlim(-42,42)
        ax.set_ylim(-14, 14)

        ax.grid(which="both", axis="both")
        # ax.minorticks_on()
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
            color="tab:blue",
            linestyle="--",
            # label="Tracking path",
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

        def ellipse_points(P, d, c =None, num_points=300):
            """Generate points on the ellipse defined by (x - c)^T P (x - c) = d."""
            theta = np.linspace(0, 2 * np.pi, num_points)
            unit_circle = np.stack([np.cos(theta), np.sin(theta)])  # Shape: (2, N)

            # Transform unit circle to match ellipse shape
            L = np.linalg.cholesky(np.linalg.inv(P/d))  # such that x = L @ u gives the ellipse
            if c is None:
                c = np.zeros((2,))
            ellipse = (L @ unit_circle).T + c
            return ellipse

        ax.plot(
            *ellipse_points(
                np.array(self.params["env"]["track"]["P1"]),
                np.array(self.params["env"]["track"]["d1"]),
            ).T,
            color="black",
            # alpha=0.5,
        )
        ax.plot(
            *ellipse_points(
                np.array(self.params["env"]["track"]["P2"]),
                np.array(self.params["env"]["track"]["d2"]),
            ).T,
            color="black",
            # alpha=0.5,
        )

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
        # fig_gp.savefig("car_racing.png")
        # quit()

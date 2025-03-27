import torch
import numpy as np


class CarKinematicsModel(object):
    def __init__(self, params):
        self.params = params
        self.H = self.params["optimizer"]["H"]
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.g_ny = self.params["agent"]["g_dim"]["ny"]
        self.g_nx = self.params["agent"]["g_dim"]["nx"]
        self.g_nu = self.params["agent"]["g_dim"]["nu"]
        self.pad_vg = [0, 1, 3]
        self.pad_g = [0, 3, 4, 5]  # 0, !self.g_nx + self.g_nu, included v (4) as well
        self.g_idx_inputs = [2, 4]
        if self.params["common"]["use_cuda"] and torch.cuda.is_available():
            self.use_cuda = True
            self.torch_device = torch.device("cuda")
            torch.set_default_device(self.torch_device)
        else:
            self.use_cuda = False
            self.torch_device = torch.device("cpu")
            torch.set_default_device(self.torch_device)

        self.B_d = torch.eye(self.nx, self.g_ny, device=self.torch_device)
        self.has_nominal_model = True

    def initial_training_data(self):
        # need more training data for decent result
        # keep low output scale, TODO: check if variance on gradient output can be controlled Dyn_gp_task_noises

        n_data_x = self.params["env"]["n_data_x"]
        n_data_u = self.params["env"]["n_data_u"]

        if self.params["env"]["prior_dyn_meas"]:
            phi_min = self.params["optimizer"]["x_min"][2]
            phi_max = self.params["optimizer"]["x_max"][2]
            delta_min = self.params["optimizer"]["u_min"][0]
            delta_max = self.params["optimizer"]["u_max"][0]
            dphi = 0 #(phi_max - phi_min) / (n_data_x)  # removed -1; we want n gaps

            ddelta = 0 #(delta_max - delta_min) / (n_data_u)
            phi = torch.linspace(phi_min + dphi / 2, phi_max - dphi / 2, n_data_x)

            delta = torch.linspace(
                delta_min + ddelta / 2, delta_max - ddelta / 2, n_data_u
            )
            Phi, Delta = torch.meshgrid(phi, delta)
            Dyn_gp_X_train = torch.hstack([Phi.reshape(-1, 1), Delta.reshape(-1, 1)])
            Dyn_gp_Y_train = self.get_prior_data(Dyn_gp_X_train)
            # Dyn_gp_Y_train = torch.stack((y1, y2), dim=0)
        else:
            Dyn_gp_X_train = torch.rand(1, self.in_dim)
            Dyn_gp_Y_train = torch.rand(2, 1, 1 + self.in_dim)

        if not self.params["env"]["train_data_has_derivatives"]:
            Dyn_gp_Y_train[:, :, 1:] = torch.nan

        return Dyn_gp_X_train, Dyn_gp_Y_train

    def get_prior_data(self, xu):
        # NOTE: xu already needs to be filtered to only contain modeled inputs
        assert xu.shape[1] == self.g_nx + self.g_nu

        dt = self.params["optimizer"]["dt"]
        g_nx = self.g_nx  # phi
        g_nu = self.g_nu  # delta
        g_ny = self.g_ny  # phi, v, delta
        lf = self.params["env"]["params"]["lf"]
        lr = self.params["env"]["params"]["lr"]
        # lf = 1.105 * 0.01
        # lr = 1.738 * 0.01
        phi, delta = xu[:, 0], xu[:, 1]
        g_xu = self.unknown_dyn(xu)  # phi, v, delta
        y_ret = torch.zeros((g_ny, xu.shape[0], 1 + g_nx + g_nu))

        y_ret[0, :, 0] = g_xu[:, 0]
        y_ret[1, :, 0] = g_xu[:, 1]
        y_ret[2, :, 0] = g_xu[:, 2]
        # derivative of dx w.r.t. phi, v, delta
        beta_in = (lr * torch.tan(delta)) / (lf + lr)
        beta = torch.atan(beta_in)
        y_ret[0, :, 1] = -torch.sin(phi + beta) * dt
        # y_ret[0, :, 2] = torch.cos(phi + beta) * dt

        term = ((lr / (torch.cos(delta) ** 2)) / (lf + lr)) / (1 + beta_in**2)
        y_ret[0, :, 2] = -torch.sin(phi + beta) * dt * term

        # derivative of dy w.r.t. phi, v, delta
        y_ret[1, :, 1] = torch.cos(phi + beta) * dt
        # y_ret[1, :, 2] = torch.sin(phi + beta) * dt
        y_ret[1, :, 2] = torch.cos(phi + beta) * dt * term

        # derivative of dphi w.r.t. phi, v, delta
        # y_ret[2,:, 0] is zeros
        # y_ret[2, :, 2] = torch.sin(beta) * dt / lr
        y_ret[2, :, 2] = torch.cos(beta) * dt * term / lr
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
        """_summary_"""
        # NOTE: xu already needs to be filtered to only contain modeled inputs
        assert xu.shape[1] == len(self.g_idx_inputs)

        Phi_k, delta_k = xu[:, [0]], xu[:, [1]]
        lf = self.params["env"]["params"]["lf"]
        lr = self.params["env"]["params"]["lr"]
        dt = self.params["optimizer"]["dt"]
        beta = torch.atan(torch.tan(delta_k) * lr / (lr + lf))
        dX_kp1 = torch.cos(Phi_k + beta) * dt
        dY_kp1 = torch.sin(Phi_k + beta) * dt
        Phi_kp1 = torch.sin(beta) * dt / lr
        state_kp1 = torch.hstack([dX_kp1, dY_kp1, Phi_kp1])
        # V_state_kp1 = torch.hstack([V_k * dX_kp1, V_k * dY_kp1, V_k * Phi_kp1])
        return state_kp1

    def unknown_dyn_Bd_fun(self, xu):
        B_d = xu[:, 3] * torch.eye(self.nx, self.g_ny).to(device=self.torch_device, dtype=xu.dtype)
        return B_d

    def discrete_dyn(self, xu):
        """_summary_"""
        # NOTE: takes only single xu
        assert xu.shape[1] == self.nx + self.nu

        f_xu = self.known_dyn(xu.tile((1, self.nx, 1, 1)))[0, :, :]
        g_xu = self.unknown_dyn(xu[:, self.g_idx_inputs]).transpose(0, 1)
        B_d_xu = self.unknown_dyn_Bd_fun(xu)
        return f_xu + torch.matmul(B_d_xu, g_xu) 

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
        ns = dg_dxu_grad.shape[0]
        nH = dg_dxu_grad.shape[2]
        gnew_dim = 4 # 1 + dg_dxu_grad.shape[3]

        v_dg_dxu_grad = torch.zeros(
            (ns, self.g_ny, nH, gnew_dim), device=dg_dxu_grad.device
        )
        # tranform gradient from g(theta, delta) to g(theta, v, delta) form 
        v_dg_dxu_grad[:, :, :, self.pad_vg] = (
            xu_hat[:, 0:3, :, [3]] * dg_dxu_grad
        )  # multiply with V_k
        v_dg_dxu_grad[:, :, :, 2] = dg_dxu_grad[:, :, :, 0]  # set grad_v
        return v_dg_dxu_grad

    @staticmethod
    def get_reachable_set_ball(params, V_k, eps_vec=None):
        H = params["optimizer"]["H"]
        assert V_k.shape[0] == H + 1
        # computation of tightenings
        P = np.array(params["optimizer"]["terminal_tightening"]["P"])
        # P *=10
        print(f"P = {P}", "V_k = ", V_k)
        L = params["agent"]["tight"]["Lipschitz"]
        dyn_eps = params["agent"]["tight"]["dyn_eps"]
        w_bound = params["agent"]["tight"]["w_bound"]
        var_eps = (dyn_eps + w_bound)
        if eps_vec is not None:
            # np.dot(np.sqrt(np.diag(P[:3][:3])),np.array([8e-4,9e-4,3e-4]))
            # B_d_norm = (np.dot(np.sqrt(np.diag(P[:3][:3])),np.array([3.65e-4,4e-4,1.35e-4]))/var_eps)*V_k
            B_d_norm = (np.dot(np.sqrt(np.diag(P[:3][:3])),eps_vec)/var_eps)*V_k
        else:
            B_d_norm = np.sum(np.sqrt(np.diag(P[:3][:3])))*V_k
        P_inv = np.linalg.inv(P)
        K = np.array(params["optimizer"]["terminal_tightening"]["K"])
        B_eps_0 = 0
        tightenings = np.sqrt(np.diag(P_inv))*B_eps_0
        u_tight = np.sqrt(np.diag(K@P_inv@K.T))*B_eps_0
        tilde_eps_list = []
        tilde_eps_list.append(np.concatenate([tightenings.tolist(), u_tight.tolist(), [B_eps_0]]))
        ci_list = []
        for stage in range(1, H + 1):
            B_eps_k = var_eps*B_d_norm[stage-1] * np.sum(np.power(L, np.arange(0, stage)))  
            # arange has inbuild -1 in [sstart, end-1]
            # box constraints tightenings
            tightenings = np.sqrt(np.diag(P_inv))*B_eps_k
            u_tight = np.sqrt(np.diag(K@P_inv@K.T))*B_eps_k
            print(f"u_tight_{stage} = {u_tight}")
            tilde_eps_list.append(np.concatenate([tightenings.tolist(), u_tight.tolist(), [B_eps_k]]))
            ci_list.append(B_eps_k)
            print(f"tilde_eps_{stage} = {tilde_eps_list[-1]}")
        # quit()
        return tilde_eps_list, ci_list

    def get_mpc_tightenings(self):
        # computation of tightenings
        L = self.params["agent"]["tight"]["Lipschitz"]
        dyn_eps = self.params["agent"]["tight"]["dyn_eps"]
        w_bound = self.params["agent"]["tight"]["w_bound"]
        B_d_norm = np.sqrt(self.params["optimizer"]["terminal_tightening"]["P"][1][1])
        var_eps = (dyn_eps + w_bound)*B_d_norm
        P_inv = np.linalg.inv(self.params["optimizer"]["terminal_tightening"]["P"])
        K = np.array(self.params["optimizer"]["terminal_tightening"]["K"])
        tilde_eps_0 = 0
        tightenings = np.sqrt(np.diag(P_inv)*tilde_eps_0)
        u_tight = np.sqrt(np.diag(K@P_inv@K.T)*tilde_eps_0)
        self.tilde_eps_list = []
        self.tilde_eps_list.append(np.concatenate([tightenings.tolist(), u_tight.tolist(), [tilde_eps_0]]))
        self.ci_list = []
        tilde_eps_i = 0
        for stage in range(1, self.H + 1):
            c_i = np.power(L, stage - 1) * var_eps + 2 * dyn_eps *B_d_norm* np.sum(
                np.power(L, np.arange(0, stage - 1))
            )  # arange has inbuild -1 in [sstart, end-1]
            if stage == self.H:
                self.tilde_eps_list.append([c_i]*(self.nx+self.nu+1))
                self.ci_list.append(c_i)
            else:
                tilde_eps_i += c_i
                # box constraints tightenings
                tightenings = np.sqrt(np.diag(P_inv))*tilde_eps_i
                u_tight = np.sqrt(np.diag(K@P_inv@K.T))*tilde_eps_i
                print(f"u_tight_{stage} = {u_tight}")
                self.tilde_eps_list.append(np.concatenate([tightenings.tolist(), u_tight.tolist(), [tilde_eps_i]]))
                self.ci_list.append(c_i)
            print(f"tilde_eps_{stage} = {self.tilde_eps_list[-1]}")
        # quit()
        return self.tilde_eps_list, self.ci_list
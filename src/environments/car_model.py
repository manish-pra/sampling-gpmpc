import torch
import numpy as np


class CarKinematicsModel(object):
    def __init__(self, params):
        self.params = params
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.g_ny = self.params["agent"]["g_dim"]["ny"]
        self.pad_g = [0, 3, 4, 5]  # 0, self.g_nx + self.g_nu :

    def initial_training_data(self):
        # Initialize model
        x1 = torch.linspace(-3.14, 3.14, 11)
        x2 = torch.linspace(-10, 10, 11)
        u = torch.linspace(-30, 30, 11)
        X1, X2, U = torch.meshgrid(x1, x2, u)

        # need more training data for decent result
        # keep low output scale, TODO: check if variance on gradient output can be controlled Dyn_gp_task_noises

        n_data_x = self.params["env"]["n_data_x"]
        n_data_u = self.params["env"]["n_data_u"]

        if self.params["env"]["prior_dyn_meas"]:
            phi = torch.linspace(-1.14, 1.14, n_data_x)
            v = torch.linspace(0, 15, n_data_x)
            delta = torch.linspace(-0.6, 0.6, n_data_u)
            Phi, V, Delta = torch.meshgrid(phi, v, delta)
            Dyn_gp_X_train = torch.hstack(
                [Phi.reshape(-1, 1), V.reshape(-1, 1), Delta.reshape(-1, 1)]
            )
            Dyn_gp_Y_train = self.get_prior_data(Dyn_gp_X_train)
            # Dyn_gp_Y_train = torch.stack((y1, y2), dim=0)
        else:
            Dyn_gp_X_train = torch.rand(1, self.in_dim)
            Dyn_gp_Y_train = torch.rand(2, 1, 1 + self.in_dim)

        if not self.params["env"]["train_data_has_derivatives"]:
            Dyn_gp_Y_train[:, :, 1:] = torch.nan

        return Dyn_gp_X_train, Dyn_gp_Y_train

    def get_prior_data(self, xu):
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
        return xu_hat[:, : self.g_ny, :, [2, 3, 4]]

    def known_dyn(self, xu):
        """_summary_"""
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
        Phi_k, V_k, delta_k = xu[:, [0]], xu[:, [1]], xu[:, [2]]
        lf = self.params["env"]["params"]["lf"]
        lr = self.params["env"]["params"]["lr"]
        dt = self.params["optimizer"]["dt"]
        beta = torch.atan(torch.tan(delta_k) * lr / (lr + lf))
        dX_kp1 = V_k * torch.cos(Phi_k + beta) * dt
        dY_kp1 = V_k * torch.sin(Phi_k + beta) * dt
        Phi_kp1 = V_k * torch.sin(beta) * dt / lr
        state_kp1 = torch.hstack([dX_kp1, dY_kp1, Phi_kp1])
        return state_kp1

    def discrete_dyn(self, xu):
        """_summary_"""
        f_xu = self.known_dyn(xu.reshape(1, 1, 1, -1)).reshape(-1, 1)
        g_xu = self.unknown_dyn(xu[:, [2, 3, 4]])  # phi, v, delta
        B_d = torch.eye(self.nx, self.g_ny).to(device="cpu", dtype=torch.float32)
        state_kp1 = f_xu + torch.matmul(B_d, g_xu.reshape(3, 1))
        # X_k, Y_k, Phi_k, V_k = xu[:, [0]], xu[:, [1]], xu[:, [2]], xu[:, [3]]
        # acc_k = xu[:, 5]
        # lf = self.params["env"]["params"]["lf"]
        # lr = self.params["env"]["params"]["lr"]
        # dt = self.params["optimizer"]["dt"]
        # # beta = torch.atan(torch.tan(delta_k) * lr / (lr + lf))
        # X_kp1 = X_k + g_xu[:, [0]]
        # Y_kp1 = Y_k + g_xu[:, [1]]
        # Phi_kp1 = Phi_k + g_xu[:, [2]]  # + V_k * torch.sin(beta) * dt / lr
        # V_kp1 = V_k + acc_k * dt
        # state_kp1 = torch.stack([X_kp1, Y_kp1, Phi_kp1, V_kp1])
        return state_kp1

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

    def continous_dyn(self, X1, X2, U):
        """_summary_

        Args:
            x (_type_): _description_
            u (_type_): _description_
        """
        m = 1
        l = 1
        g = 10
        X1dot = X2.clone()
        X2dot = -g * torch.sin(X1) / l + U / l
        train_data_y = torch.hstack([X1dot.reshape(-1, 1), X2dot.reshape(-1, 1)])
        return train_data_y

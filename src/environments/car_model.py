import torch
import numpy as np


class CarKinematicsModel(object):
    def __init__(self, params):
        self.params = params
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]

    def initial_training_data(self):
        # Initialize model
        x1 = torch.linspace(-3.14, 3.14, 11)
        x2 = torch.linspace(-10, 10, 11)
        u = torch.linspace(-30, 30, 11)
        X1, X2, U = torch.meshgrid(x1, x2, u)

        if self.params["agent"]["train_data_has_derivatives"]:
            # need more training data for decent result
            n_data_x = 3
            n_data_u = 5
        else:
            # need more training data for decent result
            # keep low output scale, TODO: check if variance on gradient output can be controlled
            n_data_x = 5
            n_data_u = 9

        if self.params["agent"]["prior_dyn_meas"]:
            x1 = torch.linspace(-2.14, 2.14, n_data_x)
            # x1 = torch.linspace(-0.57,1.14,5)
            x2 = torch.linspace(-2.5, 2.5, n_data_x)
            u = torch.linspace(-8, 8, n_data_u)
            X1, X2, U = torch.meshgrid(x1, x2, u)
            Dyn_gp_X_train = torch.hstack(
                [X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)]
            )
            Dyn_gp_Y_train = self.get_prior_data(Dyn_gp_X_train)
            # Dyn_gp_Y_train = torch.stack((y1, y2), dim=0)
        else:
            Dyn_gp_X_train = torch.rand(1, self.in_dim)
            Dyn_gp_Y_train = torch.rand(2, 1, 1 + self.in_dim)

        if not self.params["agent"]["train_data_has_derivatives"]:
            Dyn_gp_Y_train[:, :, 1:] = torch.nan

        return Dyn_gp_X_train, Dyn_gp_Y_train

    def get_prior_data(self, xu):
        dt = 0.015
        nx = 2  # phi, v
        nu = 1  # delta
        ny = 3  # phi, v, delta
        lf = 1.105 * 0.01
        lr = 1.738 * 0.01
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

    def known_dyn(self, xu):
        """_summary_"""
        X_k, Y_k, Phi_k, V_k = (
            xu[:, [0], :, 0],
            xu[:, [0], :, 1],
            xu[:, [0], :, 2],
            xu[:, [0], :, 3],
        )
        delta_k, acc_k = xu[:, [0], :, 4], xu[:, [0], :, 5]
        lf = 1.105 * 0.01
        lr = 1.738 * 0.01
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
        lf = 1.105 * 0.01
        lr = 1.738 * 0.01
        dt = 0.015
        beta = torch.atan(torch.tan(delta_k) * lr / (lr + lf))
        dX_kp1 = V_k * torch.cos(Phi_k + beta) * dt
        dY_kp1 = V_k * torch.sin(Phi_k + beta) * dt
        Phi_kp1 = V_k * torch.sin(beta) * dt / lr
        state_kp1 = torch.hstack([dX_kp1, dY_kp1, Phi_kp1])
        return state_kp1

    def discrete_dyn(self, xu):
        """_summary_"""

        g_xu = self.unknown_dyn(xu[:, [2, 3, 4]])  # phi, v, delta
        X_k, Y_k, Phi_k, V_k = xu[:, [0]], xu[:, [1]], xu[:, [2]], xu[:, [3]]
        acc_k = xu[:, 5]
        # lf = 1.105 * 0.01
        # lr = 1.738 * 0.01
        dt = 0.015
        # beta = torch.atan(torch.tan(delta_k) * lr / (lr + lf))
        X_kp1 = X_k + g_xu[:, [0]]
        Y_kp1 = Y_k + g_xu[:, [1]]
        Phi_kp1 = Phi_k + g_xu[:, [2]]  # + V_k * torch.sin(beta) * dt / lr
        V_kp1 = V_k + acc_k * dt
        state_kp1 = torch.stack([X_kp1, Y_kp1, Phi_kp1, V_kp1])
        return state_kp1

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

    def get_true_gradient(self, x_hat):
        l = 1
        g = 10
        # A = np.array([[0.0, 1.0],
        #               [-g*np.cos(x_hat[0])/l,0.0]])
        # B = np.array([[0.0],
        #               [1/l]])
        ret = torch.zeros((2, x_hat.shape[0], 3))
        ret[0, :, 1] = torch.ones(x_hat.shape[0])
        ret[1, :, 0] = -g * torch.cos(x_hat[:, 0]) / l
        ret[1, :, 2] = torch.ones(x_hat.shape[0]) / l

        val = self.pendulum_dyn(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2])
        return torch.hstack([val[:, 0].reshape(-1, 1), ret[0, :, :]]), torch.hstack(
            [val[:, 1].reshape(-1, 1), ret[1, :, :]]
        )

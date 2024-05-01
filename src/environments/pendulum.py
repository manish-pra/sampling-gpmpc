import torch
import numpy as np


class Pendulum(object):
    def __init__(self, params):
        self.params = params
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.g_ny = self.params["agent"]["g_dim"]["ny"]
        self.pad_g = [0, 1, 2, 3]  # 0, self.g_nx + self.g_nu :

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
            x1 = torch.linspace(-2.14, 2.14, n_data_x)
            # x1 = torch.linspace(-0.57,1.14,5)
            x2 = torch.linspace(-2.5, 2.5, n_data_x)
            u = torch.linspace(-8, 8, n_data_u)
            X1, X2, U = torch.meshgrid(x1, x2, u)
            Dyn_gp_X_train = torch.hstack(
                [X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)]
            )
            y1, y2 = self.get_prior_data(Dyn_gp_X_train)
            Dyn_gp_Y_train = torch.stack((y1, y2), dim=0)
        else:
            Dyn_gp_X_train = torch.rand(1, self.in_dim)
            Dyn_gp_Y_train = torch.rand(2, 1, 1 + self.in_dim)

        if not self.params["env"]["train_data_has_derivatives"]:
            Dyn_gp_Y_train[:, :, 1:] = torch.nan

        return Dyn_gp_X_train, Dyn_gp_Y_train

    def get_prior_data(self, x_hat):
        l = 1
        g = 10
        dt = self.params["optimizer"]["dt"]
        g_xu = self.unknown_dyn(x_hat)
        y1_fx, y2_fx = g_xu[:, 0], g_xu[:, 1]
        y1_ret = torch.zeros((x_hat.shape[0], 4))
        y2_ret = torch.zeros((x_hat.shape[0], 4))
        y1_ret[:, 0] = y1_fx
        y1_ret[:, 1] = torch.ones(x_hat.shape[0])
        y1_ret[:, 2] = torch.ones(x_hat.shape[0]) * dt

        y2_ret[:, 0] = y2_fx
        y2_ret[:, 1] = (-g * torch.cos(x_hat[:, 0]) / l) * dt
        y2_ret[:, 2] = torch.ones(x_hat.shape[0])
        y2_ret[:, 3] = torch.ones(x_hat.shape[0]) * dt / (l * l)
        # A = np.array([[0.0, 1.0],
        #               [-g*np.cos(x_hat[0])/l,0.0]])
        # B = np.array([[0.0],
        #               [1/l]])
        return y1_ret, y2_ret

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

    def discrete_dyn(self, xu):
        return self.unknown_dyn(xu)

    def unknown_dyn(self, xu):
        m = 1
        l = 1
        g = 10
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

    # def propagate_true_dynamics(self, x_init, U):
    #     x1_list = []
    #     x2_list = []
    #     X1_k = x_init[0]
    #     X2_k = x_init[1]
    #     x1_list.append(X1_k.item())
    #     x2_list.append(X2_k.item())
    #     for ele in range(U.shape[0]):
    #         X1_kp1, X2_kp1 = self.discrete_dyn(X1_k, X2_k, U[ele])
    #         x1_list.append(X1_kp1.item())
    #         x2_list.append(X2_kp1.item())
    #         X1_k = X1_kp1.copy()
    #         X2_k = X2_kp1.copy()
    #     return x1_list, x2_list

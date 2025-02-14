import torch
import numpy as np


class Pendulum(object):
    def __init__(self, params):
        self.params = params
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.g_ny = self.params["agent"]["g_dim"]["ny"]
        self.g_nx = self.params["agent"]["g_dim"]["nx"]
        self.g_nu = self.params["agent"]["g_dim"]["nu"]
        self.pad_g = [0, 1, 3]  # 0, self.g_nx + self.g_nu :
        self.g_idx_inputs = [0, 2]
        if self.params["common"]["use_cuda"] and torch.cuda.is_available():
            self.use_cuda = True
            self.torch_device = torch.device("cuda")
            torch.set_default_device(self.torch_device)
        else:
            self.use_cuda = False
            self.torch_device = torch.device("cpu")
            torch.set_default_device(self.torch_device)

        self.B_d = torch.tensor([0.0, 1.0], device=self.torch_device).reshape(
            self.nx, self.g_ny
        )

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
            u = torch.linspace(
                self.params["optimizer"]["u_min"][0],
                self.params["optimizer"]["u_max"][0],
                n_data_u,
            )
            X1, U = torch.meshgrid(x1, u)
            Dyn_gp_X_train = torch.hstack([X1.reshape(-1, 1), U.reshape(-1, 1)])
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
        y1_fx = g_xu[:, 0]
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
        y_ret = torch.zeros((self.g_ny, x_hat.shape[0], 1 + self.g_nx + self.g_nu))
        # y2_ret = torch.zeros((x_hat.shape[0], 4))
        y_ret[0, :, 0] = (
            y1_fx
            # + torch.randn(x_hat.shape[0]) * self.params["agent"]["tight"]["w_bound"]*0.1
        )
        y_ret[0, :, 1] = (-g * torch.cos(x_hat[:, 0]) / l) * dt
        y_ret[0, :, 2] = torch.ones(x_hat.shape[0]) * dt / (l * l)

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
        """_summary_"""
        # NOTE: takes only single xu
        assert xu.shape[1] == self.nx + self.nu

        f_xu = self.known_dyn(xu.tile((1, self.nx, 1, 1)))[0, :, :]
        g_xu = self.unknown_dyn(xu[:, self.g_idx_inputs]).transpose(0, 1)
        B_d = torch.tensor([0.0, 1.0], device="cpu", dtype=g_xu.dtype).reshape(
            self.nx, self.g_ny
        )
        return f_xu + torch.matmul(B_d, g_xu)

    def unknown_dyn(self, xu):
        assert xu.shape[1] == len(self.g_idx_inputs)
        l = self.params["env"]["params"]["l"]
        g = self.params["env"]["params"]["g"]
        X1_k, U_k = xu[:, [0]], xu[:, [1]]
        dt = self.params["optimizer"]["dt"]
        dX2_kp1 = -g * torch.sin(X1_k) * dt / l + U_k * dt / (l * l)
        state_kp1 = torch.hstack([dX2_kp1])
        return state_kp1

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
        # dtheta_kpi/dtheta_k = 1 , dtheta_kpi/domega_k = dt
        df_dxu_grad[:, 0, :, 1] = torch.ones((ns, nH))
        df_dxu_grad[:, 0, :, 2] = dt * torch.ones((ns, nH))
        # domega_kpi/domega_k = 1 , domega_kpi/dtheta_k = 0
        df_dxu_grad[:, 1, :, 2] = torch.ones((ns, nH))

        # set the derivative of the function w.r.t. control input
        # df_dxu_grad[:, 3, :, 6] = dt * torch.ones((ns, nH))  # dV_kpi/acc_k = dt
        return df_dxu_grad

    def get_g_xu_hat(self, xu_hat):
        # arg 1 truncated to self.g_ny -> all the same since tiles
        for i in range(xu_hat.shape[1] - 1):
            assert torch.all(xu_hat[:, i, :, 0] == xu_hat[:, i + 1, :, 0])

        return xu_hat[:, 0 : self.g_ny, :, self.g_idx_inputs]

    def known_dyn(self, xu):
        """_summary_"""
        # NOTE: xu has to be in shape used by optimizer
        assert len(xu.shape) == 4  # ns, nx, H, nx+nu
        assert xu.shape[1] == self.nx
        assert xu.shape[3] == self.nx + self.nu

        theta_k, omega_k, alpha_k = (
            xu[:, [0], :, 0],
            xu[:, [0], :, 1],
            xu[:, [0], :, 2],
        )
        dt = self.params["optimizer"]["dt"]
        theta_kp1 = theta_k + omega_k * dt
        omega_kp1 = omega_k
        state_kp1 = torch.cat([theta_kp1, omega_kp1], 1)
        return state_kp1

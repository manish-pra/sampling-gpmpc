import torch


class Pendulum(object):
    def __init__(self, params):
        self.params = params
        self.dt = 0.1
        self.state = np.array([0.0, 0.0])
        self.state_dim = 2
        self.action_dim = 1
        self.max_action = 2.0
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.length = 0.5
        self.m = 1.0
        self.g = 9.8
        self.viewer = None

    def initial_training_data(self):
        # Initialize model
        x1 = torch.linspace(-3.14, 3.14, 11)
        x2 = torch.linspace(-10, 10, 11)
        u = torch.linspace(-30, 30, 11)
        X1, X2, U = torch.meshgrid(x1, x2, u)
        self.Dyn_gp_X_range = torch.hstack(
            [X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)]
        )

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
            self.Dyn_gp_X_train = torch.hstack(
                [X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)]
            )
            y1, y2 = self.get_prior_data(self.Dyn_gp_X_train)
            self.Dyn_gp_Y_train = torch.stack((y1, y2), dim=0)
        else:
            self.Dyn_gp_X_train = torch.rand(1, self.in_dim)
            self.Dyn_gp_Y_train = torch.rand(2, 1, 1 + self.in_dim)

        if not self.params["agent"]["train_data_has_derivatives"]:
            self.Dyn_gp_Y_train[:, :, 1:] = torch.nan

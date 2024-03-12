from copy import copy
from dataclasses import dataclass

import gpytorch
import numpy as np

import torch
from gpytorch.kernels import (
    RBFKernel,
    ScaleKernel,
)
from src.GP_model import BatchMultitaskGPModelWithDerivatives, GPModelWithDerivatives
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, params) -> None:
        self.my_key = 0
        self.params = params
        self.nx = self.params["agent"]["dim"]["nx"]
        self.nu = self.params["agent"]["dim"]["nu"]
        self.ny = self.params["agent"]["dim"]["ny"]
        self.ns = self.params["agent"]["num_dyn_samples"]

        # TODO: (Manish) replace self.in_dim --> self.nx + self.nu
        self.in_dim = self.nx + self.nu
        self.batch_shape = torch.Size(
            [self.params["agent"]["num_dyn_samples"], self.ny]
        )
        self.mean_shift_val = params["agent"]["mean_shift_val"]
        self.converged = False
        self.x_dim = params["optimizer"]["x_dim"]

        if self.params["common"]["use_cuda"] and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        self.Hallcinated_X_train = None
        self.Hallcinated_Y_train = None
        self.model_i = None

        self.Hallcinated_X_train = torch.empty(self.ns, self.ny, 0, self.in_dim)
        self.Hallcinated_Y_train = torch.empty(
            self.ns, self.ny, 0, self.in_dim + 1
        )  # NOTE(amon): added+1 to in_dim
        if self.use_cuda:
            self.Hallcinated_X_train = self.Hallcinated_X_train.cuda()
            self.Hallcinated_Y_train = self.Hallcinated_Y_train.cuda()

        self.initial_training_data()
        self.real_data_batch()
        self.planned_measure_loc = np.array([2])
        self.epistimic_random_vector = self.random_vector_within_bounds()

    def initial_training_data(self):
        # Initialize model
        x1 = torch.linspace(-3.14, 3.14, 11)
        x2 = torch.linspace(-10, 10, 11)
        u = torch.linspace(-30, 30, 11)
        X1, X2, U = torch.meshgrid(x1, x2, u)
        self.Dyn_gp_X_range = torch.hstack(
            [X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)]
        )

        n_data_x = 7
        n_data_u = 7
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

    def random_vector_within_bounds(self):
        # generate a normally distributed weight vector within bounds by continous respampling
        H = self.params["optimizer"]["H"]
        n_dyn = self.params["agent"]["num_dyn_samples"]
        beta = self.params["agent"]["Dyn_gp_beta"]
        n_mpc = self.params["common"]["num_MPC_itrs"]
        n_itrs = self.params["optimizer"]["SEMPC"]["max_sqp_iter"]
        # ret_itrs = torch.normal(0, 1, size=(2, n_dyn, 2, H, 4))
        # ret_itrs = torch.normal(
        #     torch.zeros(2, n_dyn, 2, H, 4), torch.ones(2, n_dyn, 2, H, 4)
        # )
        ret_mpc_iters = torch.empty(n_mpc, n_itrs, n_dyn, self.ny, H, self.in_dim + 1)
        for j in range(self.params["common"]["num_MPC_itrs"]):
            ret_itrs = torch.empty(n_itrs, n_dyn, self.ny, H, self.in_dim + 1)
            for i in range(self.params["optimizer"]["SEMPC"]["max_sqp_iter"]):
                ret = torch.empty(0, self.ny, H, self.in_dim + 1)
                while True:
                    w = torch.normal(0, 1, size=(1, self.ny, H, self.in_dim + 1))
                    if torch.all(w >= -beta) and torch.all(w <= beta):
                        ret = torch.cat([ret, w], dim=0)
                    if ret.shape[0] == n_dyn:
                        break
                ret_itrs[i, :, :, :, :] = ret
            ret_mpc_iters[j, :, :, :, :, :] = ret_itrs
        return ret_mpc_iters.cuda()

    def update_current_location(self, loc):
        self.current_location = loc

    def update_current_state(self, state):
        self.current_state = state
        self.update_current_location(state[: self.x_dim])

    def update_hallucinated_Dyn_dataset(self, newX, newY):
        self.Hallcinated_X_train = torch.cat([self.Hallcinated_X_train, newX], 2)
        self.Hallcinated_Y_train = torch.cat([self.Hallcinated_Y_train, newY], 2)

    def real_data_batch(self):
        n_pnts, n_dims = self.Dyn_gp_X_train.shape
        self.Dyn_gp_X_train_batch = torch.tile(
            self.Dyn_gp_X_train, dims=(self.ns, self.ny, 1, 1)
        )
        self.Dyn_gp_Y_train_batch = torch.tile(
            self.Dyn_gp_Y_train, dims=(self.ns, 1, 1, 1)
        )
        if self.use_cuda:
            self.Dyn_gp_X_train_batch = self.Dyn_gp_X_train_batch.cuda()
            self.Dyn_gp_Y_train_batch = self.Dyn_gp_Y_train_batch.cuda()

    def train_hallucinated_dynGP(self, sqp_iter):
        n_sample = self.ns
        if self.model_i is not None:
            del self.model_i
        data_X = torch.concat(
            [self.Dyn_gp_X_train_batch, self.Hallcinated_X_train], dim=2
        )

        data_Y = torch.concat(
            [self.Dyn_gp_Y_train_batch, self.Hallcinated_Y_train], dim=2
        )
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.in_dim + 1,
            noise_constraint=gpytorch.constraints.GreaterThan(0.0),
            batch_shape=self.batch_shape,
        )  # Value + Derivative
        self.model_i = BatchMultitaskGPModelWithDerivatives(
            data_X, data_Y, likelihood, self.batch_shape
        )
        self.model_i.likelihood.noise = torch.tile(
            torch.Tensor([self.params["agent"]["Dyn_gp_noise"]]),
            dims=(self.batch_shape[0], self.batch_shape[1], 1),
        )
        self.model_i.likelihood.task_noises = torch.tile(
            torch.Tensor(self.params["agent"]["Dyn_gp_task_noises"]["val"])
            * self.params["agent"]["Dyn_gp_task_noises"]["multiplier"],
            dims=(self.batch_shape[0], self.batch_shape[1], 1),
        )
        self.model_i.covar_module.base_kernel.lengthscale = torch.tile(
            torch.Tensor(self.params["agent"]["Dyn_gp_lengthscale"]["both"]),
            dims=(self.batch_shape[0], 1, 1, 1),
        )
        self.model_i.covar_module.outputscale = torch.tile(
            torch.Tensor(self.params["agent"]["Dyn_gp_outputscale"]["both"]),
            dims=(self.batch_shape[0], 1),
        )
        if self.use_cuda:
            self.model_i = self.model_i.cuda()

        del data_X
        del data_Y
        del likelihood
        if sqp_iter == 0:
            if self.Hallcinated_X_train is not None:
                del self.Hallcinated_X_train
                del self.Hallcinated_Y_train

            self.Hallcinated_X_train = torch.empty(n_sample, self.ny, 0, self.in_dim)
            self.Hallcinated_Y_train = torch.empty(
                n_sample, self.ny, 0, 1 + self.in_dim
            )
            if self.use_cuda:
                self.Hallcinated_X_train = self.Hallcinated_X_train.cuda()
                self.Hallcinated_Y_train = self.Hallcinated_Y_train.cuda()

    def get_next_to_go_loc(self):
        return self.planned_measure_loc

    def pendulum_dyn(self, X1, X2, U):
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

    def pendulum_discrete_dyn(self, X1_k, X2_k, U_k):
        """_summary_

        Args:
            x (_type_): _description_
            u (_type_): _description_
        """
        m = 1
        l = 1
        g = 10
        dt = self.params["optimizer"]["dt"]
        X1_kp1 = X1_k + X2_k * dt
        X2_kp1 = X2_k - g * np.sin(X1_k) * dt / l + U_k * dt / (l * l)
        return X1_kp1, X2_kp1

    def visu(self):
        plt.close()
        U_k = torch.linspace(-0.3, 0.3, 100)
        X1_k = torch.Tensor([0.0])
        X2_k = torch.Tensor([0.0])
        x1_list = []
        x2_list = []
        x1_list.append(X1_k.item())
        x2_list.append(X2_k.item())
        for i in range(100):
            X1_kp1, X2_kp1 = self.pendulum_discrete_dyn(X1_k, X2_k, U_k[i])
            x1_list.append(X1_kp1.item())
            x2_list.append(X2_kp1.item())
            X1_k = X1_kp1.clone()
            X2_k = X2_kp1.clone()

        plt.plot(x1_list, x2_list)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid()
        plt.savefig("pendulum.png")
        plt.close()
        plt.plot(x1_list)
        plt.plot(x2_list)
        plt.plot(U_k.numpy())
        plt.legend(["x1", "x2", "u"])
        plt.ylim(-3, 3)
        plt.grid()
        plt.savefig("pendulum2.png")
        exit()
        pass

    def get_prior_data(self, x_hat):
        l = 1
        g = 10
        dt = self.params["optimizer"]["dt"]
        y1_fx, y2_fx = self.pendulum_discrete_dyn(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2])
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

    def get_batch_x_hat(self, x_h, u_h):
        x_h = torch.from_numpy(x_h).float()
        u_h = torch.from_numpy(u_h).float()
        x_h_batch = (
            x_h.transpose(0, 1)
            .view(
                self.params["agent"]["num_dyn_samples"],
                self.x_dim,
                self.params["optimizer"]["H"],
            )
            .transpose(1, 2)
        )
        u_h_batch = (
            torch.ones(
                self.params["agent"]["num_dyn_samples"],
                self.params["optimizer"]["H"],
                1,
            )
            * u_h
        )
        ret = torch.cat([x_h_batch, u_h_batch], 2)
        ret_allout = torch.stack([ret] * 2, dim=1)
        if self.use_cuda:
            ret_allout = ret_allout.cuda()
        return ret_allout

    def mpc_iteration(self, i):
        self.mpc_iter = i

    def get_batch_gp_sensitivities(self, x_hat, sqp_iter):
        """_summaary_ Derivatives are obtained by sampling from the GP directly. Record those derivatives.

        Args:
            x_hat (_type_): states to evaluate the GP and its gradients
            sample_idx (_type_): _description_

        Returns:
            _type_: in numpy format
        """

        with gpytorch.settings.fast_pred_var(), torch.no_grad(), gpytorch.settings.max_cg_iterations(
            50
        ), gpytorch.settings.observation_nan_policy(
            "mask"
        ):
            self.model_i.eval()
            model_i_call = self.model_i(x_hat)
            y_sample = model_i_call.sample(
                base_samples=self.epistimic_random_vector[self.mpc_iter][sqp_iter]
            )

            idx_overwrite = 0
            if self.params["agent"]["true_dyn_as_sample"]:
                # overwrite next sample with true dynamics
                y_sample_true_1, y_sample_true_2 = self.get_prior_data(
                    x_hat[idx_overwrite, 0, :, :].cpu()
                )
                y_sample_true = torch.stack([y_sample_true_1, y_sample_true_2], dim=0)
                y_sample[idx_overwrite, :, :, :] = y_sample_true
                idx_overwrite += 1

            if self.params["agent"]["mean_as_dyn_sample"]:
                # overwrite next sample with mean
                y_sample[[idx_overwrite], :, :, :] = model_i_call.mean[
                    [idx_overwrite], :, :, :
                ]
                idx_overwrite += 1

        assert torch.all(x_hat[:, 0, :, :] == x_hat[:, 1, :, :])
        self.update_hallucinated_Dyn_dataset(x_hat, y_sample)

        gp_val = y_sample[:, :, :, [0]].cpu().numpy()
        y_grad = y_sample[:, :, :, 1 : 1 + self.nx].cpu().numpy()
        u_grad = y_sample[:, :, :, 1 + self.nx : 1 + self.nx + self.nu].cpu().numpy()
        del y_sample
        del x_hat
        return gp_val, y_grad, u_grad  # y, dy/dx1, dy/dx2, dy/du

    def scale_with_beta(self, lower, upper, beta):
        temp = lower * (1 + beta) / 2 + upper * (1 - beta) / 2
        upper = upper * (1 + beta) / 2 + lower * (1 - beta) / 2
        lower = temp
        return lower, upper


if __name__ == "__main__":
    agent = Agent()

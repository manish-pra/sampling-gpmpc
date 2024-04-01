from copy import copy
from dataclasses import dataclass

import gpytorch
import numpy as np

import torch
from gpytorch.kernels import (
    RBFKernel,
    ScaleKernel,
)
from src.GP_model import BatchMultitaskGPModelWithDerivatives_fromParams
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, params, env_model) -> None:
        self.my_key = 0
        self.params = params
        self.env_model = env_model
        self.g_nx = self.params["agent"]["g_dim"]["nx"]
        self.g_nu = self.params["agent"]["g_dim"]["nu"]
        self.g_ny = self.params["agent"]["g_dim"]["ny"]
        self.ns = self.params["agent"]["num_dyn_samples"]
        self.nx = params["agent"]["dim"]["nx"]
        self.nu = params["agent"]["dim"]["nu"]

        # TODO: (Manish) replace self.in_dim --> self.g_nx + self.g_nu
        self.in_dim = self.g_nx + self.g_nu
        self.batch_shape = torch.Size(
            [self.params["agent"]["num_dyn_samples"], self.g_ny]
        )
        self.mean_shift_val = params["agent"]["mean_shift_val"]
        self.converged = False

        if self.params["common"]["use_cuda"] and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        self.Hallcinated_X_train = None
        self.Hallcinated_Y_train = None
        self.model_i = None

        self.Hallcinated_X_train = torch.empty(self.ns, self.g_ny, 0, self.in_dim)
        self.Hallcinated_Y_train = torch.empty(
            self.ns, self.g_ny, 0, self.in_dim + 1
        )  # NOTE(amon): added+1 to in_dim
        if self.use_cuda:
            self.Hallcinated_X_train = self.Hallcinated_X_train.cuda()
            self.Hallcinated_Y_train = self.Hallcinated_Y_train.cuda()

        self.Dyn_gp_X_train, self.Dyn_gp_Y_train = env_model.initial_training_data()
        self.real_data_batch()
        self.planned_measure_loc = np.array([2])
        self.epistimic_random_vector = self.random_vector_within_bounds()

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
        ret_mpc_iters = torch.empty(n_mpc, n_itrs, n_dyn, self.g_ny, H, self.in_dim + 1)
        for j in range(self.params["common"]["num_MPC_itrs"]):
            ret_itrs = torch.empty(n_itrs, n_dyn, self.g_ny, H, self.in_dim + 1)
            for i in range(self.params["optimizer"]["SEMPC"]["max_sqp_iter"]):
                ret = torch.empty(0, self.g_ny, H, self.in_dim + 1)
                while True:
                    w = torch.normal(0, 1, size=(1, self.g_ny, H, self.in_dim + 1))
                    if torch.all(w >= -beta) and torch.all(w <= beta):
                        ret = torch.cat([ret, w], dim=0)
                    if ret.shape[0] == n_dyn:
                        break
                ret_itrs[i, :, :, :, :] = ret
            ret_mpc_iters[j, :, :, :, :, :] = ret_itrs

        if self.use_cuda:
            ret_mpc_iters = ret_mpc_iters.cuda()
        return ret_mpc_iters

    def update_current_location(self, loc):
        self.current_location = loc

    def update_current_state(self, state):
        self.current_state = state
        self.update_current_location(state[: self.nx])

    def update_hallucinated_Dyn_dataset(self, newX, newY):
        self.Hallcinated_X_train = torch.cat([self.Hallcinated_X_train, newX], 2)
        self.Hallcinated_Y_train = torch.cat([self.Hallcinated_Y_train, newY], 2)

    def real_data_batch(self):
        n_pnts, n_dims = self.Dyn_gp_X_train.shape
        self.Dyn_gp_X_train_batch = torch.tile(
            self.Dyn_gp_X_train, dims=(self.ns, self.g_ny, 1, 1)
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

        data_X, data_Y = self.concatenate_real_hallucinated_data()

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.in_dim + 1,
            noise_constraint=gpytorch.constraints.GreaterThan(0.0),
            batch_shape=self.batch_shape,
        )  # Value + Derivative

        self.model_i = BatchMultitaskGPModelWithDerivatives_fromParams(
            data_X, data_Y, likelihood, self.params, batch_shape=self.batch_shape
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

            self.Hallcinated_X_train = torch.empty(n_sample, self.g_ny, 0, self.in_dim)
            self.Hallcinated_Y_train = torch.empty(
                n_sample, self.g_ny, 0, 1 + self.in_dim
            )
            if self.use_cuda:
                self.Hallcinated_X_train = self.Hallcinated_X_train.cuda()
                self.Hallcinated_Y_train = self.Hallcinated_Y_train.cuda()

    def concatenate_real_hallucinated_data(self):
        data_X = torch.concat(
            [self.Dyn_gp_X_train_batch, self.Hallcinated_X_train], dim=2
        )
        data_Y = torch.concat(
            [self.Dyn_gp_Y_train_batch, self.Hallcinated_Y_train], dim=2
        )
        return data_X, data_Y

    def get_next_to_go_loc(self):
        return self.planned_measure_loc

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

    def get_batch_x_hat(self, x_h, u_h):
        x_h = torch.from_numpy(x_h).float()
        u_h = torch.from_numpy(u_h).float()
        x_h_batch = (
            x_h.transpose(0, 1)
            .view(
                self.params["agent"]["num_dyn_samples"],
                self.nx,
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
        ret_allout = torch.stack([ret] * self.nx, dim=1)
        if self.use_cuda:
            ret_allout = ret_allout.cuda()
        return ret_allout

    def mpc_iteration(self, i):
        self.mpc_iter = i

    def dyn_fg_jacobians(self, xu_hat, sqp_iter):
        # get dynamics value and Jacobians
        ns = xu_hat.shape[0]
        nH = xu_hat.shape[2]
        df_dxu_grad = self.env_model.get_f_known_jacobian(xu_hat)
        g_xu_hat = self.env_model.get_g_xu_hat(xu_hat)
        # g_xu_hat = xu_hat[:, : self.g_ny, :, [2, 3, 4]]  # phi, v, delta
        dg_dxu_grad = self.get_batch_gp_sensitivities(g_xu_hat, sqp_iter)
        if False:  # for debugging the multiplication with B_d logic
            ch_pad_dg_dxu_grad = torch.zeros_like(df_dxu_grad, device=xu_hat.device)
            ch_pad_dg_dxu_grad[:, : self.g_ny, :, [0, 3, 4, 5]] = dg_dxu_grad
            y_sample = df_dxu_grad + ch_pad_dg_dxu_grad
        else:
            pad_dg_dxu_grad = torch.zeros(
                ns, self.g_ny, nH, 1 + self.nx + self.nu, device=xu_hat.device
            )
            pad_dg_dxu_grad[:, :, :, self.env_model.pad_g] = dg_dxu_grad
            B_d = torch.eye(self.nx, self.g_ny, device=xu_hat.device)
            y_sample = df_dxu_grad + torch.matmul(
                B_d, pad_dg_dxu_grad.transpose(1, 2)
            ).transpose(1, 2)
        gp_val = y_sample[:, :, :, [0]].cpu().numpy()
        y_grad = y_sample[:, :, :, 1 : 1 + self.nx].cpu().numpy()
        u_grad = y_sample[:, :, :, 1 + self.nx : 1 + self.nx + self.nu].cpu().numpy()
        del y_sample
        del xu_hat
        return gp_val, y_grad, u_grad  # y, dy/dx, dy/du

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
                y_sample_true = self.env_model.get_prior_data(
                    x_hat[idx_overwrite, 0, :, :].cpu()
                )
                # y_sample_true = torch.stack([y_sample_true_1, y_sample_true_2], dim=0)
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
        del x_hat
        return y_sample

    def scale_with_beta(self, lower, upper, beta):
        temp = lower * (1 + beta) / 2 + upper * (1 - beta) / 2
        upper = upper * (1 + beta) / 2 + lower * (1 - beta) / 2
        lower = temp
        return lower, upper


if __name__ == "__main__":
    agent = Agent()

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
            self.torch_device = torch.device("cuda")
            torch.set_default_device(self.torch_device)
        else:
            self.use_cuda = False
            self.torch_device = torch.device("cpu")
            torch.set_default_device(self.torch_device)

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
                    w = torch.normal(
                        0,
                        1,
                        size=(1, self.g_ny, H, self.in_dim + 1),
                        device=self.torch_device,
                    )
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
        # eliminate similar data points
        # newX, newY = self.eliminate_similar_data(newX, newY)

        # calculate distance between new data and existing data
        # self.model_i.covar_module(self.Hallcinated_X_train)
        # self.model_i.

        # eliminate_radius = 1e-3
        # dist = torch.zeros((newX.shape[2], self.Hallcinated_X_train.shape[2]))
        # for i in range(newX.shape[2]):
        #     dist[i,:]
        min_distance = 1e-3
        X_cond, Y_cond = self.concatenate_real_hallucinated_data()
        # X_all = torch.cat([newX, X_cond], 2)
        # X_all = torch.cat([newX, self.Hallcinated_X_train], 2)
        # diag = (
        #     torch.eye(X_all.shape[2], newX.shape[2])
        #     .unsqueeze(-1)
        #     .unsqueeze(0)
        #     .unsqueeze(0)
        # )

        # if self.use_cuda:
        #     diag = diag.cuda()

        # dist = newX[:, :, None, :, :] - X_all[:, dist:, :, None, :] + diag
        dist = newX[:, :, None, :, :] - X_cond[:, :, :, None, :]
        dist_norm = torch.linalg.vector_norm(dist, dim=-1)
        filter_these_out = torch.any(dist_norm <= min_distance, dim=2)

        # set filtered out points to nan
        filter_these_out_x = filter_these_out.unsqueeze(-1).tile(
            1, 1, 1, (self.nx + self.nu)
        )
        filter_these_out_y = filter_these_out.unsqueeze(-1).tile(
            1, 1, 1, (self.nx + self.nu + 1)
        )
        newX_filter = newX.clone()
        newY_filter = newY.clone()
        # newX_filter[filter_these_out_x] = torch.nan
        newY_filter[filter_these_out_y] = torch.nan

        # check if point should be filtered for all samples
        # filter_these_out_all = torch.all(filter_these_out, dim=0, keepdim=True).tile(self.ns, 1, 1, 1)
        filter_these_out_all = torch.all(filter_these_out, dim=0)
        filter_these_out_all = torch.any(filter_these_out_all, dim=0)
        # filter_these_out_all_x = filter_these_out_all.tile(1,1,1,(self.nx + self.nu))
        # filter_these_out_all_y = filter_these_out_all.tile(1,1,1,(self.nx + self.nu + 1))

        # remove from newX
        newX_filter_all = newX_filter[:, :, filter_these_out_all == False, :]
        newY_filter_all = newY_filter[:, :, filter_these_out_all == False, :]

        # filter_these_out = torch.any(filter_these_out == True, dim=1)
        # filter_these_out = torch.any(filter_these_out == True, dim=0)
        # self.Hallcinated_X_train = torch.cat(
        #     [self.Hallcinated_X_train, newX[:, :, filter_these_out == False, :]], 2
        # )
        # self.Hallcinated_Y_train = torch.cat(
        #     [self.Hallcinated_Y_train, newY[:, :, filter_these_out == False, :]], 2
        # )
        # self.Hallcinated_X_train = torch.cat(
        #     [self.Hallcinated_X_train, newX[:, :, :, :]], 2
        # )
        # self.Hallcinated_Y_train = torch.cat(
        #     [self.Hallcinated_Y_train, newY[:, :, :, :]], 2
        # )
        self.Hallcinated_X_train = torch.cat(
            [self.Hallcinated_X_train, newX_filter_all], 2
        )
        self.Hallcinated_Y_train = torch.cat(
            [self.Hallcinated_Y_train, newY_filter_all], 2
        )

        print(
            f"Filtered out {torch.sum(filter_these_out_all)} points, still nan: {torch.sum(torch.isnan(newY_filter_all[:,0,:,0]),dim=1)}"
        )

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

        with torch.no_grad(), gpytorch.settings.observation_nan_policy(
            "mask"
        ), gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ), gpytorch.settings.cholesky_jitter(
            float_value=1e-6, double_value=1e-6, half_value=1e-6
        ):
            self.model_i.eval()
            self.model_i_call = self.model_i(x_hat)
            y_sample_orig = self.model_i_call.sample(
                base_samples=self.epistimic_random_vector[self.mpc_iter][sqp_iter]
            )

            beta_confidence = self.params["agent"]["Dyn_gp_beta"] * torch.sqrt(
                self.model_i_call.variance
            )
            beta_confidence_norm = torch.linalg.norm(beta_confidence)
            mean_diff_norm = torch.linalg.norm(y_sample_orig - self.model_i_call.mean)
            mean_norm = torch.linalg.norm(self.model_i_call.mean)
            print(f"beta_confidence_rel: {beta_confidence_norm / (mean_norm+1e-6)}")
            beta_confidence_rel_all = np.array(
                [
                    torch.linalg.norm(beta_confidence[s, :, :, :])
                    / (torch.linalg.norm(self.model_i_call.mean[s, :, :, :]) + 1e-6)
                    for s in range(self.ns)
                ]
            )
            # print(f"beta_confidence_rel_all: {beta_confidence_rel_all}")
            print(f": {beta_confidence_norm / (mean_norm+1e-6)}")
            print(f"mean_diff_orig: {torch.linalg.norm(mean_diff_norm)}")

            variance_numerically_zero = self.model_i_call.variance <= 1.001e-6
            variance_numerically_zero_all_outputs = torch.all(
                variance_numerically_zero, dim=-1, keepdim=True
            ).tile(1, 1, 1, self.nx + self.nu + 1)
            variance_numerically_zero_num = torch.zeros_like(self.model_i_call.variance)
            variance_numerically_zero_num[
                variance_numerically_zero_all_outputs == True
            ] = 1
            set_sample_to_mean = [False] * self.ns
            # for s in range(self.ns):
            # if beta_confidence_rel_all[s] < 5e-3:
            # y_sample_orig[s,:,:,:] = self.model_i_call.mean[s,:,:,:]
            y_sample_orig = (
                variance_numerically_zero_num * self.model_i_call.mean
                + (1 - variance_numerically_zero_num) * y_sample_orig
            )
            # set_sample_to_mean[s] = True
            # print(f"Set sample to mean: {set_sample_to_mean}")

            # check that sampled dynamics are within bounds
            y_max = self.model_i_call.mean + self.params["agent"][
                "Dyn_gp_beta"
            ] * torch.sqrt(self.model_i_call.variance)
            y_min = self.model_i_call.mean - self.params["agent"][
                "Dyn_gp_beta"
            ] * torch.sqrt(self.model_i_call.variance)
            y_sample = torch.max(y_sample_orig, y_min)
            y_sample = torch.min(y_sample, y_max)
            # y_sample = y_sample_orig

            self.model_i_samples = y_sample

            if not torch.allclose(y_sample, y_sample_orig):
                print(
                    f"y_sample truncated! Reldiff: {torch.norm(y_sample - y_sample_orig)/(torch.norm(y_sample_orig)+1e-6)}"
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
                y_sample[[idx_overwrite], :, :, :] = self.model_i_call.mean[
                    [idx_overwrite], :, :, :
                ]
                idx_overwrite += 1

        assert torch.all(x_hat[:, 0, :, :] == x_hat[:, 1, :, :])

        # if not set_sample_to_mean:
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

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

torch.set_default_dtype(torch.float64)


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

    def get_min_dist_train_data(self):
        min_dist = np.zeros((self.params["agent"]["num_dyn_samples"],))
        for s in range(self.params["agent"]["num_dyn_samples"]):
            train_inputs_i = self.model_i.train_inputs[0][s, 0, :, :]
            train_targets_i = self.model_i.train_targets[s, 0, :, :]
            train_targets_i_nan = torch.all(torch.isnan(train_targets_i), dim=1)
            train_inputs_i_diff = (
                train_inputs_i[None, train_targets_i_nan == False, :]
                - train_inputs_i[train_targets_i_nan == False, None, :]
            )
            train_inputs_i_diff_norm = torch.linalg.vector_norm(
                train_inputs_i_diff, dim=-1
            )
            train_inputs_i_diff_norm_plus_diag = train_inputs_i_diff_norm + torch.eye(
                train_inputs_i_diff_norm.shape[0]
            )
            min_dist[s] = torch.min(train_inputs_i_diff_norm_plus_diag)
        return min_dist

    def update_current_location(self, loc):
        self.current_location = loc

    def update_current_state(self, state):
        self.current_state = state
        self.update_current_location(state[: self.nx])

    def update_hallucinated_Dyn_dataset(self, newX, newY):
        # eliminate similar data points
        min_distance = self.params["agent"]["Dyn_gp_min_data_dist"]
        X_cond, Y_cond = self.concatenate_real_hallucinated_data()

        # dist = newX[:, :, None, :, :] - X_all[:, dist:, :, None, :] + diag
        dist = newX[:, :, None, :, :] - X_cond[:, :, :, None, :]
        dist_norm = torch.linalg.vector_norm(dist, dim=-1)
        filter_these_out = torch.any(dist_norm <= min_distance, dim=2)

        # set filtered out points to nan
        filter_these_out_x = filter_these_out.unsqueeze(-1).tile(
            1, 1, 1, (self.nx + self.nu)
        )
        filter_these_out_y = filter_these_out.unsqueeze(-1).tile(
            1, 1, 1, (self.g_nx + self.g_nu + 1)
        )
        newX_filter = newX.clone()
        newY_filter = newY.clone()
        newY_filter[filter_these_out_y] = torch.nan

        # check if point should be filtered for all samples
        filter_these_out_all = torch.all(filter_these_out, dim=0)
        filter_these_out_all = torch.any(filter_these_out_all, dim=0)

        # remove from newX
        newX_filter_all = newX_filter[:, :, filter_these_out_all == False, :]
        newY_filter_all = newY_filter[:, :, filter_these_out_all == False, :]

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

    def train_hallucinated_dynGP(self, sqp_iter, use_model_without_derivatives=False):
        n_sample = self.ns
        if self.model_i is not None:
            del self.model_i

        if use_model_without_derivatives:
            # just use real data, this is for debugging only
            data_X, data_Y = (
                self.Dyn_gp_X_train_batch,
                self.Dyn_gp_Y_train_batch[:, :, :, [0]],
            )
        else:
            data_X, data_Y = self.concatenate_real_hallucinated_data()

        if use_model_without_derivatives:
            num_tasks = 1
        else:
            num_tasks = self.in_dim + 1

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks,
            noise_constraint=gpytorch.constraints.GreaterThan(0.0),
            batch_shape=self.batch_shape,
        )  # Value + Derivative
        self.model_i = BatchMultitaskGPModelWithDerivatives_fromParams(
            data_X,
            data_Y,
            likelihood,
            self.params,
            batch_shape=self.batch_shape,
            use_grad=not use_model_without_derivatives,
        )
        self.model_i.eval()
        likelihood.eval()
        # self.model_i(data_X[:,:,[0],:])
        # likelihood(self.model_i(data_X[:,:,[0],:]))

        self.likelihood = likelihood

        if self.use_cuda:
            self.model_i = self.model_i.cuda()

        del data_X
        del data_Y
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

    def train_forward_sampling_dynGP(self):

        if self.model_i is not None:
            del self.model_i

        data_X = torch.concat(
            [
                self.Dyn_gp_X_train_batch,
                self.FS_X_train_batch,
                self.Hallcinated_X_train,
            ],
            dim=2,
        )
        data_Y = torch.concat(
            [
                self.Dyn_gp_Y_train_batch,
                self.FS_Y_train_batch,
                self.Hallcinated_Y_train,
            ],
            dim=2,
        )
        num_tasks = self.in_dim + 1

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks,
            noise_constraint=gpytorch.constraints.GreaterThan(0.0),
            batch_shape=self.batch_shape,
        )  # Value + Derivative
        self.model_i = BatchMultitaskGPModelWithDerivatives_fromParams(
            data_X,
            data_Y,
            likelihood,
            self.params,
            batch_shape=self.batch_shape,
        )
        self.model_i.eval()
        likelihood.eval()
        # self.model_i(data_X[:,:,[0],:])
        # likelihood(self.model_i(data_X[:,:,[0],:]))

        self.likelihood = likelihood

        if self.use_cuda:
            self.model_i = self.model_i.cuda()

        del data_X
        del data_Y

    def prepare_dynamics_set(self, X_soln, U_soln, X_kp1):

        # TODO: update the dynamic model_i with the converged dynamics (perhaps not required?)
        n_sample = self.ns
        L = self.params["agent"]["tight"]["Lipschitz"]
        dyn_eps = self.params["agent"]["tight"]["dyn_eps"]
        w_bound = self.params["agent"]["tight"]["w_bound"]
        var_eps = dyn_eps + w_bound

        # Initialize forward sampling dataset
        self.FS_X_train_batch = torch.empty(n_sample, self.g_ny, 0, self.in_dim)
        self.FS_Y_train_batch = torch.empty(n_sample, self.g_ny, 0, 1 + self.in_dim)

        # Reshaping and load on cuda
        X_soln = X_soln.reshape(X_soln.shape[0], n_sample, self.nx).cuda()
        X_kp1 = X_kp1.transpose(0, 1).cuda()
        U_soln = U_soln.cuda()

        # check  x_{1|k} - x(k+1) < var_eps
        diff = X_soln[1, :, :] - X_kp1

        samples_left = torch.prod(torch.abs(diff) - var_eps < 0, dim=1)
        print("Samples remaininng in N{k+1} are ", torch.sum(samples_left))

        # ingredients to propatate the state
        xu_init = torch.cat([X_kp1, U_soln[[1]]], dim=-1)
        xu_hat = torch.tile(xu_init, dims=(n_sample, self.nx, 1, 1))
        # g_xu_hat = self.env_model.get_g_xu_hat(xu_hat)

        # Forward sampling of each of the dynamic model_i
        for i in range(1, X_soln.shape[0] - 1):

            # 1. Sample the dynamics
            with torch.no_grad(), gpytorch.settings.observation_nan_policy(
                "mask"
            ), gpytorch.settings.fast_computations(
                covar_root_decomposition=False, log_prob=False, solves=False
            ), gpytorch.settings.cholesky_jitter(
                float_value=self.params["agent"]["Dyn_gp_jitter"],
                double_value=self.params["agent"]["Dyn_gp_jitter"],
                half_value=self.params["agent"]["Dyn_gp_jitter"],
            ):
                g_xu_hat = self.env_model.get_g_xu_hat(xu_hat)
                model_i_call = self.model_i(g_xu_hat)
                Y_sample = model_i_call.sample(
                    # base_samples=agent.epistimic_random_vector[agent.mpc_iter][sqp_iter]
                )

                # check  x_{i+1|k} - x_{i|k+1} < var_eps
                g_val = Y_sample[:, :, :].squeeze()[
                    :, : self.g_ny
                ]  # get the function values
                # Get state x_{i|k+1} using the sampled dynamics g^n
                f_val = self.env_model.known_dyn(xu_hat).squeeze()
                x_next = f_val + torch.matmul(self.env_model.B_d, g_val.t()).t()
                diff = X_soln[i + 1, :, :] - x_next
                c_i = np.power(L, i) * var_eps + 2 * dyn_eps * np.sum(
                    np.power(L, np.arange(0, i))
                )  # arange has inbuild -1 in [start, end-1]
                N_kp1 = torch.prod(torch.abs(diff) - c_i < 0, dim=1)
                samples_left = samples_left * N_kp1
                print("Samples remaininng in N{k+1} are ", torch.sum(samples_left))

                if i == X_soln.shape[0] - 2:
                    break
                # 2. Update the hallucinated dataset with the propagated data
                self.FS_X_train_batch = torch.cat(
                    [self.FS_X_train_batch, g_xu_hat], dim=2
                )
                Y_sample[:, :, :, 1:] = torch.nan
                self.FS_Y_train_batch = torch.cat(
                    [self.FS_Y_train_batch, Y_sample], dim=2
                )
                # Train the model_i with the next state: x_{i|k+1} and u[i+1] --> x[i+1|k+1]
                self.train_forward_sampling_dynGP()
                # reshape the inputs for the GP
                xu_hat = torch.cat(
                    [
                        torch.stack([x_next] * self.nx, dim=1)[:, :, None, :],
                        torch.tile(U_soln[[i + 1]], dims=(n_sample, self.nx, 1, 1)),
                    ],
                    dim=-1,
                )

        # 3. Throw away the data for the rejected samples
        if torch.sum(samples_left) > 0:
            num_samples_replaced = torch.sum(samples_left == 0).item()
            remaining_idx = torch.arange(n_sample)[samples_left > 0].cpu().numpy()
            self.Hallcinated_X_train[samples_left == 0, :, :, :] = (
                self.Hallcinated_X_train[
                    np.random.choice(remaining_idx, num_samples_replaced).tolist(),
                    :,
                    :,
                    :,
                ]
            )
            self.Hallcinated_Y_train[samples_left == 0, :, :, :] = (
                self.Hallcinated_Y_train[
                    np.random.choice(remaining_idx, num_samples_replaced).tolist(),
                    :,
                    :,
                    :,
                ]
            )

        # 4. Restore the previous model_i
        self.train_hallucinated_dynGP(
            sqp_iter=self.params["optimizer"]["SEMPC"]["max_sqp_iter"]
        )

        return

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

    def get_batch_x_hat_u_diff(self, x_h, u_h):
        x_h = torch.tensor(x_h)
        u_h = torch.tensor(u_h)
        x_h_batch = (
            x_h.transpose(0, 1)
            .view(
                self.params["agent"]["num_dyn_samples"],
                self.nx,
                self.params["optimizer"]["H"],
            )
            .transpose(1, 2)
        )
        u_h_batch = u_h.transpose(0, 1).view(
            self.params["agent"]["num_dyn_samples"],
            self.params["optimizer"]["H"],
            self.nu,
        )
        ret = torch.cat([x_h_batch, u_h_batch], 2)
        ret_allout = torch.stack([ret] * self.nx, dim=1)
        if self.use_cuda:
            ret_allout = ret_allout.cuda()
        return ret_allout

    def get_batch_x_hat(self, x_h, u_h):
        x_h = torch.tensor(x_h)
        u_h = torch.tensor(u_h)
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
        ns = xu_hat.shape[0]
        nH = xu_hat.shape[2]

        # get dynamics value and Jacobians
        df_dxu_grad = self.env_model.get_f_known_jacobian(xu_hat)
        dg_dxu_grad = self.get_batch_gp_sensitivities(xu_hat, sqp_iter)

        if False:  # for debugging the multiplication with B_d logic
            ch_pad_dg_dxu_grad = torch.zeros_like(df_dxu_grad, device=xu_hat.device)
            ch_pad_dg_dxu_grad[:, : self.g_ny, :, [0, 3, 4, 5]] = dg_dxu_grad
            y_sample = df_dxu_grad + ch_pad_dg_dxu_grad
        else:
            pad_dg_dxu_grad = torch.zeros(
                ns, self.g_ny, nH, 1 + self.nx + self.nu, device=xu_hat.device
            )
            v_dg_dxu_grad = torch.zeros(
                (ns, self.g_ny, nH, 1 + dg_dxu_grad.shape[3]), device=xu_hat.device
            )
            v_dg_dxu_grad[:, :, :, self.env_model.pad_vg] = (
                xu_hat[:, 0:3, :, [3]] * dg_dxu_grad
            )  # multiply with V_k
            v_dg_dxu_grad[:, :, :, 2] = dg_dxu_grad[:, :, :, 0]  # set grad_v
            pad_dg_dxu_grad[:, :, :, self.env_model.pad_g] = v_dg_dxu_grad
            # pad_dg_dxu_grad[:, :, :, self.env_model.pad_g] = dg_dxu_grad
            # B_d = torch.eye(self.nx, self.g_ny, device=xu_hat.device)
            y_sample = df_dxu_grad + torch.matmul(
                self.env_model.B_d, pad_dg_dxu_grad.transpose(1, 2)
            ).transpose(1, 2)
        gp_val = y_sample[:, :, :, [0]].cpu().numpy()
        y_grad = y_sample[:, :, :, 1 : 1 + self.nx].cpu().numpy()
        u_grad = y_sample[:, :, :, 1 + self.nx : 1 + self.nx + self.nu].cpu().numpy()

        if torch.any(torch.isnan(y_sample)) or torch.any(torch.isinf(y_sample)):
            print("Nan/inf in y_sample")

        del y_sample
        del xu_hat
        return gp_val, y_grad, u_grad  # y, dy/dx, dy/du

    def get_batch_gp_sensitivities(self, xu_hat, sqp_iter):
        """_summaary_ Derivatives are obtained by sampling from the GP directly. Record those derivatives.

        Args:
            xu_hat (_type_): states to evaluate the GP and its gradients
            sample_idx (_type_): _description_

        Returns:
            _type_: in numpy format
        """

        # filter inputs from full x,u to gp inputs
        g_xu_hat = self.env_model.get_g_xu_hat(xu_hat)
        for i in range(g_xu_hat.shape[1] - 1):
            assert torch.all(g_xu_hat[:, i + 1, :, :] == g_xu_hat[:, i, :, :])

        y_sample = self.sample_gp(
            g_xu_hat, base_samples=self.epistimic_random_vector[self.mpc_iter][sqp_iter]
        )
        # y_sample = self.sample_gp(g_xu_hat)

        idx_overwrite = 0
        if self.params["agent"]["true_dyn_as_sample"]:
            # overwrite next sample with true dynamics
            y_sample[idx_overwrite, :, :, :] = self.env_model.get_prior_data(
                g_xu_hat[idx_overwrite, 0, :, :]
            )
            idx_overwrite += 1

        if self.params["agent"]["mean_as_dyn_sample"]:
            # overwrite next sample with mean
            y_sample[[idx_overwrite], :, :, :] = self.model_i_call.mean[
                [idx_overwrite], :, :, :
            ]
            idx_overwrite += 1

        self.update_hallucinated_Dyn_dataset(g_xu_hat, y_sample)
        return y_sample

    def sample_gp(self, x_input, base_samples=None):
        with torch.no_grad(), gpytorch.settings.observation_nan_policy(
            "mask"
        ), gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ), gpytorch.settings.cholesky_jitter(
            float_value=self.params["agent"]["Dyn_gp_jitter"],
            double_value=self.params["agent"]["Dyn_gp_jitter"],
            half_value=self.params["agent"]["Dyn_gp_jitter"],
        ):
            self.model_i.eval()
            self.model_i_call = self.model_i(x_input)
            y_sample = self.model_i_call.sample(base_samples=base_samples)
            y_train = self.model_i.train_targets
            x_train = self.model_i.train_inputs[0]

            # check if variance is numerically zero
            if self.params["agent"]["Dyn_gp_variance_is_zero"] >= 0.0:
                variance_numerically_zero = (
                    self.model_i_call.variance
                    <= self.params["agent"]["Dyn_gp_variance_is_zero"]
                )
                variance_numerically_zero_all_outputs = torch.all(
                    variance_numerically_zero, dim=-1, keepdim=True
                ).tile(1, 1, 1, self.g_nx + self.g_nu + 1)
                variance_numerically_zero_num = torch.zeros_like(
                    self.model_i_call.variance
                )
                variance_numerically_zero_num[
                    variance_numerically_zero_all_outputs == True
                ] = 1
                y_sample = (
                    variance_numerically_zero_num * self.model_i_call.mean
                    + (1 - variance_numerically_zero_num) * y_sample
                )

            # find too close points and overwrite with closest y-value
            if self.params["agent"]["Dyn_gp_min_data_dist"] >= 0.0:
                min_distance = self.params["agent"]["Dyn_gp_min_data_dist"]
                dist = x_input[:, :, None, :, :] - x_train[:, :, :, None, :]
                y_train_isnan = (
                    torch.any(torch.isnan(y_train), dim=3)
                    .unsqueeze(-1)
                    .tile(1, 1, 1, self.params["optimizer"]["H"])
                )
                dist_norm = torch.linalg.vector_norm(dist, dim=-1)
                dist_norm[y_train_isnan] = torch.tensor(float("inf"))
                dist_too_small = (
                    torch.any(dist_norm <= min_distance, dim=2)
                    .unsqueeze(-1)
                    .tile(1, 1, 1, self.g_nx + self.g_nu + 1)
                )
                min_dist_input, min_dist_input_index = torch.min(dist_norm, dim=2)
                # create tuple of tensors with indices of closest training data point
                # Assuming y_train and min_dist_input_index are already defined and have the shapes mentioned above
                A, B, C, D = y_train.shape
                E = min_dist_input_index.shape[2]
                # Create indices for the first and second dimensions
                i1 = torch.arange(A).view(A, 1, 1).expand(A, B, E)  # Shape: (A, B, E)
                i2 = torch.arange(B).view(1, B, 1).expand(A, B, E)  # Shape: (A, B, E)
                # Use these indices to gather the elements from y_train
                y_sample_closest_train = y_train[i1, i2, min_dist_input_index, :]

                y_sample = torch.where(
                    dist_too_small,
                    y_sample_closest_train,
                    y_sample,
                )

                assert not torch.any(torch.isnan(y_sample))

            # check that sampled dynamics are within bounds
            y_max = self.model_i_call.mean + self.params["agent"][
                "Dyn_gp_beta"
            ] * torch.sqrt(self.model_i_call.variance)
            y_min = self.model_i_call.mean - self.params["agent"][
                "Dyn_gp_beta"
            ] * torch.sqrt(self.model_i_call.variance)
            y_sample = torch.max(y_sample, y_min)
            y_sample = torch.min(y_sample, y_max)

            self.model_i_samples = y_sample

            # debugging
            if False:
                # print min and max variance
                print(
                    f"Min variance: {torch.tensor([torch.min(self.model_i_call.variance)])}, Max variance: {torch.tensor([torch.max(self.model_i_call.variance)])}"
                )
                if self.params["agent"]["Dyn_gp_min_data_dist"] >= 0.0:
                    print(
                        f"Replaced samples with training data: {np.array(torch.count_nonzero(dist_too_small[:,0,:,0],dim=1).detach().cpu())}"
                    )
                if self.params["agent"]["Dyn_gp_variance_is_zero"] >= 0.0:
                    print(
                        f"Variance numerically zero (all outputs): {torch.sum(variance_numerically_zero_num[:,:,:,0], dim=(1,2), dtype=torch.int32)}"
                    )
                print(
                    f"y_sample truncated! Reldiff: {torch.norm(y_sample - y_sample)/(torch.norm(y_sample)+1e-6)}"
                )

            return y_sample


if __name__ == "__main__":
    agent = Agent()

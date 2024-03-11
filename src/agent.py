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
        self.env_dim = params["common"]["dim"]
        self.in_dim = len(self.params["optimizer"]["x_min"]) + len(
            self.params["optimizer"]["u_min"]
        )
        self.out_dim = 1  # len(self.params["optimizer"]["x_min"])
        self.batch_shape = torch.Size([self.params["agent"]["num_dyn_samples"], 2])
        # self.Fx_X_train = X_train.reshape(-1, self.env_dim)
        # self.Cx_X_train = X_train.reshape(-1, self.env_dim)
        # self.Fx_Y_train = Fx_Y_train.reshape(-1, 1)
        # self.Cx_Y_train = Cx_Y_train.reshape(-1, 1)
        self.mean_shift_val = params["agent"]["mean_shift_val"]
        self.converged = False
        self.x_dim = params["optimizer"]["x_dim"]
        if (
            self.params["agent"]["true_dyn_as_sample"]
            or self.params["agent"]["mean_as_dyn_sample"]
        ):
            self.eff_dyn_samples = self.params["agent"]["num_dyn_samples"] - 1
        else:
            self.eff_dyn_samples = self.params["agent"]["num_dyn_samples"]

        if self.params["common"]["use_cuda"] and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        self.Hallcinated_X_train = None
        self.Hallcinated_Y_train = None
        self.model_i = None

        self.Hallcinated_X_train = torch.empty(self.eff_dyn_samples, 2, 0, self.in_dim)
        self.Hallcinated_Y_train = torch.empty(
            self.eff_dyn_samples, 2, 0, self.in_dim + 1
        )  # NOTE(amon): added+1 to in_dim
        # self.Hallcinated_Y_train['y1'] = torch.empty(self.eff_dyn_samples,0,1+self.in_dim)
        # self.Hallcinated_Y_train['y2'] = torch.empty(self.eff_dyn_samples,0,1+self.in_dim)
        if self.use_cuda:
            self.Hallcinated_X_train = self.Hallcinated_X_train.cuda()
            self.Hallcinated_Y_train = self.Hallcinated_Y_train.cuda()
            # self.Hallcinated_Y_train['y1'] = self.Hallcinated_Y_train['y1'].cuda()
            # self.Hallcinated_Y_train['y2'] = self.Hallcinated_Y_train['y2'].cuda()

        # Initialize model
        x1 = torch.linspace(-3.14, 3.14, 11)
        x2 = torch.linspace(-10, 10, 11)
        u = torch.linspace(-30, 30, 11)
        X1, X2, U = torch.meshgrid(x1, x2, u)
        self.Dyn_gp_X_range = torch.hstack(
            [X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)]
        )
        if params["agent"]["prior_dyn_meas"]:
            x1 = torch.linspace(-2.14, 2.14, 3)
            # x1 = torch.linspace(-0.57,1.14,5)
            x2 = torch.linspace(-2.5, 2.5, 3)
            u = torch.linspace(-8, 8, 5)
            X1, X2, U = torch.meshgrid(x1, x2, u)
            self.Dyn_gp_X_train = torch.hstack(
                [X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)]
            )
            # self.Dyn_gp_Y_train = {}
            y1, y2 = self.get_prior_data(self.Dyn_gp_X_train)
            self.Dyn_gp_Y_train = torch.stack((y1, y2), dim=0)
            # self.Dyn_gp_Y_train['y1'] = y1
            # self.Dyn_gp_Y_train['y2'] = y2
        else:
            # self.Dyn_gp_X_train = torch.from_numpy(np.hstack([0.0, 0.0, 0.0])).reshape(-1,3)
            # self.Dyn_gp_Y_train = torch.from_numpy(np.hstack([0.0, 0.0])).reshape(-1,2)
            self.Dyn_gp_X_train = torch.rand(1, self.in_dim)
            self.Dyn_gp_Y_train = torch.rand(2, 1, 1 + self.in_dim)
            # self.Dyn_gp_Y_train = {}
            # self.Dyn_gp_Y_train['y1'] = torch.rand(1, 1 + self.in_dim)
            # self.Dyn_gp_Y_train['y2'] = torch.rand(1, 1 + self.in_dim)
        self.Dyn_gp_model = self.__update_Dyn()
        self.real_data_batch()
        self.planned_measure_loc = np.array([2])

    def update_current_location(self, loc):
        self.current_location = loc

    def update_current_state(self, state):
        self.current_state = state
        self.update_current_location(state[: self.x_dim])

    def fit_i_model(self, data_sa, labels_s_next):
        Fx_X = data_sa.reshape(-1, 3)
        Fx_Y = labels_s_next.reshape(-1, 2)
        model_i = SingleTaskGP(
            Fx_X,
            Fx_Y,
            covar_module=ScaleKernel(
                base_kernel=RBFKernel(),
            ),
        )
        model_i.covar_module.base_kernel.lengthscale = self.Dyn_gp_lengthscale
        model_i.likelihood.noise = self.Dyn_gp_noise
        return model_i

    def __update_Dyn(self):
        # likelihood = {}
        # self.Dyn_gp_model = {}
        # Fx_Y_train = self.__mean_corrected(self.Fx_Y_train)
        # self.Dyn_gp_model = GPModelWithDerivatives(self.Dyn_gp_X_train, self.Dyn_gp_Y_train)
        # raise RuntimeError("Shapes are not broadcastable for mul operation")
        # RuntimeError: Shapes are not broadcastable for mul operation

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=4,
            noise_constraint=gpytorch.constraints.GreaterThan(0.0),
            batch_shape=self.batch_shape,
        )
        self.Dyn_gp_model = BatchMultitaskGPModelWithDerivatives(
            self.Dyn_gp_X_train,
            self.Dyn_gp_Y_train,
            likelihood,
            batch_shape=self.batch_shape,
        )
        # likelihood['y1'] = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=4,noise_constraint=gpytorch.constraints.GreaterThan(0.0))  # Value + Derivative
        # self.Dyn_gp_model['y1'] = GPModelWithDerivatives(self.Dyn_gp_X_train, self.Dyn_gp_Y_train['y1'], likelihood['y1'])
        self.Dyn_gp_model.likelihood.noise = torch.tile(
            torch.Tensor([self.params["agent"]["Dyn_gp_noise"]]),
            dims=(self.batch_shape[0], self.batch_shape[1], 1),
        )
        # self.Dyn_gp_model.likelihood.noise = torch.ones(self.batch_shape)*self.Dyn_gp_noise
        self.Dyn_gp_model.likelihood.task_noises = torch.tile(
            torch.Tensor(self.params["agent"]["Dyn_gp_task_noises"]["val"])
            * self.params["agent"]["Dyn_gp_task_noises"]["multiplier"],
            dims=(self.batch_shape[0], self.batch_shape[1], 1),
        )
        # self.Dyn_gp_model.likelihood.task_noises=torch.Tensor([3.8,1.27, 3.8,1.27])*0.00001

        # likelihood['y2'] = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=4,noise_constraint=gpytorch.constraints.GreaterThan(0.0))  # Value + Derivative
        # self.Dyn_gp_model['y2'] = GPModelWithDerivatives(self.Dyn_gp_X_train, self.Dyn_gp_Y_train['y2'], likelihood['y2'])
        # self.Dyn_gp_model['y2'].likelihood.noise = torch.ones(1)*self.Dyn_gp_noise
        # self.Dyn_gp_model['y2'].likelihood.task_noises=torch.Tensor([3.8,1.27, 3.8,1.27])*0.00001

        # TODO: (amon) check if multiple length scales can be set, differently for y1, y2
        self.Dyn_gp_model.covar_module.base_kernel.lengthscale = torch.tile(
            torch.Tensor(self.params["agent"]["Dyn_gp_lengthscale"]["both"]),
            dims=(self.batch_shape[0], 1, 1, 1),
        )
        self.Dyn_gp_model.covar_module.outputscale = torch.tile(
            torch.Tensor([self.params["agent"]["Dyn_gp_outputscale"]["both"]]),
            dims=(self.batch_shape[0], 1),
        )
        # self.Dyn_gp_model['y1'].covar_module.base_kernel.lengthscale = torch.Tensor(self.params["agent"]["Dyn_gp_lengthscale"]["y1"])
        # self.Dyn_gp_model['y1'].covar_module.outputscale = torch.Tensor([self.params["agent"]["Dyn_gp_outputscale"]["y1"]])
        # self.Dyn_gp_model['y2'].covar_module.base_kernel.lengthscale = torch.Tensor(self.params["agent"]["Dyn_gp_lengthscale"]["y2"])
        # self.Dyn_gp_model['y2'].covar_module.outputscale = torch.Tensor([self.params["agent"]["Dyn_gp_outputscale"]["y2"]])
        # mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit_gpytorch_model(mll)
        if self.use_cuda:
            # for out in ['y1','y2']:
            self.Dyn_gp_model = self.Dyn_gp_model.cuda()

        return self.Dyn_gp_model

    # def __update_Dyn(self):
    #     # Fx_Y_train = self.__mean_corrected(self.Fx_Y_train)
    #     self.Dyn_gp_model = SingleTaskGP(self.Dyn_gp_X_train, self.Dyn_gp_Y_train)
    #     self.Dyn_gp_model.covar_module.base_kernel.lengthscale = self.Dyn_gp_lengthscale
    #     self.Dyn_gp_model.likelihood.noise = self.Dyn_gp_noise
    #     # mll = ExactMarginalLogLikelihood(model.likelihood, model)
    #     # fit_gpytorch_model(mll)
    #     return self.Dyn_gp_model

    def update_Dyn_gp(self, newX, newY):
        self.__update_Dyn_set(newX, newY)
        self.__update_Dyn()
        return self.Dyn_gp_model

    def update_Dyn_gp_with_current_data(self):
        self.__update_Dyn()
        return self.Dyn_gp_model

    def __update_Dyn_set(self, newX, newY):
        newX = newX.reshape(-1, self.in_dim)
        newY = newY.reshape(-1, 1)
        self.Dyn_gp_X_train = torch.cat([self.Dyn_gp_X_train, newX]).reshape(
            -1, self.in_dim
        )
        self.Dyn_gp_Y_train = torch.cat([self.Dyn_gp_Y_train, newY]).reshape(-1, 1)
        if self.use_cuda:
            self.Dyn_gp_X_train = self.Dyn_gp_X_train.cuda()
            self.Dyn_gp_Y_train = self.Dyn_gp_Y_train.cuda()

    # gp_val, y_grad, u_grad,
    def update_hallucinated_Dyn_dataset(self, newX, newY):
        # if self.params["agent"]["true_dyn_as_sample"] or self.params["agent"]["mean_dyn_as_sample"]:
        #     newX = newX[1:]
        #     for out in ['y1','y2']:
        #         newY[out] = newY[out][1:]
        # newX = newX.reshape(-1, ,self.in_dim)
        # newY['y1'] = newY['y1'].reshape(-1, 1 + self.in_dim)
        # newY['y2'] = newY['y2'].reshape(-1, 1 + self.in_dim)
        self.Hallcinated_X_train = torch.cat([self.Hallcinated_X_train, newX], 2)
        self.Hallcinated_Y_train = torch.cat([self.Hallcinated_Y_train, newY], 2)

    def real_data_batch(self):
        n_pnts, n_dims = self.Dyn_gp_X_train.shape
        self.Dyn_gp_X_train_batch = torch.tile(
            self.Dyn_gp_X_train, dims=(self.eff_dyn_samples, 2, 1, 1)
        )
        self.Dyn_gp_Y_train_batch = torch.tile(
            self.Dyn_gp_Y_train, dims=(self.eff_dyn_samples, 1, 1, 1)
        )
        # self.Dyn_gp_Y_train_batch = torch.stack([self.Dyn_gp_Y_train]*self.eff_dyn_samples, dim=0)
        # for out in ['y1','y2']:
        #     a, b = self.Dyn_gp_Y_train[out].shape
        #     self.Dyn_gp_Y_train_batch[out] = torch.ones((self.eff_dyn_samples, a, b))*self.Dyn_gp_Y_train[out]
        if self.use_cuda:
            self.Dyn_gp_X_train_batch = self.Dyn_gp_X_train_batch.cuda()
            self.Dyn_gp_Y_train_batch = self.Dyn_gp_Y_train_batch.cuda()
            # self.Dyn_gp_Y_train_batch['y1'] = self.Dyn_gp_Y_train_batch['y1'].cuda()
            # self.Dyn_gp_Y_train_batch['y2'] = self.Dyn_gp_Y_train_batch['y2'].cuda()

    def train_hallucinated_dynGP(self, sqp_iter):
        n_sample = self.eff_dyn_samples
        if self.model_i is not None:
            # del self.model_i['y1']
            # del self.model_i['y2']
            del self.model_i
        # likelihood = {}
        # self.model_i = {}
        data_X = torch.concat(
            [self.Dyn_gp_X_train_batch, self.Hallcinated_X_train], dim=2
        )
        # data_X = torch.stack([data_X]*2, dim=1)

        # for out in ['y1','y2']:
        data_Y = torch.concat(
            [self.Dyn_gp_Y_train_batch, self.Hallcinated_Y_train], dim=2
        )
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=4,
            noise_constraint=gpytorch.constraints.GreaterThan(0.0),
            batch_shape=self.batch_shape,
        )  # Value + Derivative
        # model_i[out] = GPModelWithDerivatives(data_X, data_Y, likelihood[out])
        # model_i[out] = BatchIndependentMultitaskGPModelWithDerivatives(data_X, data_Y, likelihood[out],1)
        self.model_i = BatchMultitaskGPModelWithDerivatives(
            data_X, data_Y, likelihood, self.batch_shape
        )
        self.model_i.likelihood.noise = torch.tile(
            torch.Tensor([self.params["agent"]["Dyn_gp_noise"]]),
            dims=(self.batch_shape[0], self.batch_shape[1], 1),
        )
        # self.model_i.likelihood.task_noises= torch.ones(self.batch_shape)*torch.Tensor([3.8,1.27,1.27,1.27])*0.0000003
        self.model_i.likelihood.task_noises = torch.tile(
            torch.Tensor(self.params["agent"]["Dyn_gp_task_noises"]["val"])
            * self.params["agent"]["Dyn_gp_task_noises"]["multiplier"],
            dims=(self.batch_shape[0], self.batch_shape[1], 1),
        )
        # self.model_i.covar_module.base_kernel.lengthscale = torch.ones(self.batch_shape)*torch.Tensor(self.params["agent"]["Dyn_gp_lengthscale"]["both"])
        self.model_i.covar_module.base_kernel.lengthscale = torch.tile(
            torch.Tensor(self.params["agent"]["Dyn_gp_lengthscale"]["both"]),
            dims=(self.batch_shape[0], 1, 1, 1),
        )
        # self.model_i.covar_module.outputscale = torch.ones(self.batch_shape)*torch.Tensor([self.params["agent"]["Dyn_gp_outputscale"][out]])
        self.model_i.covar_module.outputscale = torch.tile(
            torch.Tensor(self.params["agent"]["Dyn_gp_outputscale"]["both"]),
            dims=(self.batch_shape[0], 1),
        )
        # model_i.covar_module.task_covar_module.var
        # model_i.covar_module.lengthscale =  torch.Tensor([[1.2241]]) #torch.Tensor([[self.Dyn_gp_lengthscale]])
        # model_i.covar_module.outputscale = torch.Tensor([[2.4601]]) #torch.Tensor([[self.Dyn_gp_outputscale]])
        # model_i = self.fit_i_model(self.Dyn_gp_X_range, sample_i)
        # model_i.eval()
        # model_i(torch.rand(5,2)).sample()
        if self.use_cuda:
            self.model_i = self.model_i.cuda()

        del data_X
        del data_Y
        del likelihood
        if sqp_iter == 0:
            if self.Hallcinated_X_train is not None:
                del self.Hallcinated_X_train
                # del self.Hallcinated_Y_train['y1']
                # del self.Hallcinated_Y_train['y2']
                del self.Hallcinated_Y_train

            self.Hallcinated_X_train = torch.empty(n_sample, 2, 0, self.in_dim)
            self.Hallcinated_Y_train = torch.empty(n_sample, 2, 0, 1 + self.in_dim)
            # self.Hallcinated_Y_train['y1'] = torch.empty(n_sample,0,1+self.in_dim)
            # self.Hallcinated_Y_train['y2'] = torch.empty(n_sample,0,1+self.in_dim)
            if self.use_cuda:
                self.Hallcinated_X_train = self.Hallcinated_X_train.cuda()
                self.Hallcinated_Y_train = self.Hallcinated_Y_train.cuda()
                # self.Hallcinated_Y_train['y1'] = self.Hallcinated_Y_train['y1'].cuda()
                # self.Hallcinated_Y_train['y2'] = self.Hallcinated_Y_train['y2'].cuda()

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
        ret[1, :, 0] = -g * torch.cos(x_hat[:, 0]) / l
        ret[0, :, 1] = torch.ones(x_hat.shape[0])
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

    def get_batch_gp_sensitivities(self, x_hat):
        """_summaary_ Derivatives are obtained by sampling from the GP directly. Record those derivatives.

        Args:
            x_hat (_type_): states to evaluate the GP and its gradients
            sample_idx (_type_): _description_

        Returns:
            _type_: in numpy format
        """

        y_mean_dyn = None
        batch_idx = 1
        if self.params["agent"]["true_dyn_as_sample"]:
            y_mean_dyn = {}
            y_mean_dyn["y1"], y_mean_dyn["y2"] = self.get_true_gradient(x_hat[0].cpu())
            if self.use_cuda:
                y_mean_dyn["y1"] = y_mean_dyn["y1"].cuda()
                y_mean_dyn["y2"] = y_mean_dyn["y2"].cuda()
        elif self.params["agent"]["mean_as_dyn_sample"]:
            y_mean_dyn = {}
            for out in ["y1", "y2"]:
                with gpytorch.settings.fast_pred_var(), torch.no_grad(), gpytorch.settings.max_cg_iterations(
                    50
                ):
                    self.Dyn_gp_model[out].eval()
                    y_mean_dyn[out] = self.Dyn_gp_model[out](x_hat[0]).mean.detach()
        else:
            batch_idx = 0

        # y_sample = {}
        # for out in ['y1','y2']:
        with gpytorch.settings.fast_pred_var(), torch.no_grad(), gpytorch.settings.max_cg_iterations(
            50
        ):
            self.model_i.eval()
            # likelihood(self.Dyn_model_list[self.sample_idx](x_hat)) for sampling with noise
            # lower, upper = self.model_i[out](x_hat[batch_idx:]).confidence_region()
            # lower, upper = self.scale_with_beta(lower, upper, 16)
            # while True:
            #     sample = self.model_i[out](x_hat[batch_idx:]).sample()
            #     if (sample>=lower).all() and (sample<=upper).all():
            #         break
            #     del sample
            # y_sample[out] = sample
            # del lower, upper
            y_sample = self.model_i(x_hat[batch_idx:]).sample()

            # mean = self.Dyn_model_list[self.sample_idx-1][out](x_hat).mean
            # y_sample[out] = sample
        self.update_hallucinated_Dyn_dataset(x_hat[batch_idx:], y_sample)

        if y_mean_dyn is not None:
            # TODO: (amon) support includion of mean dyn again
            y_sample = torch.cat([y_mean_dyn[out].unsqueeze(0), y_sample[out]], 0)
        # gp_val = torch.cat([y_sample['y1'][:,:,0].unsqueeze(2),y_sample['y2'][:,:,0].unsqueeze(2)],2).cpu().numpy()
        # gp_val = torch.cat([y_sample[:,0,:,0].unsqueeze(1).unsqueeze(3),y_sample[:,1,:,0].unsqueeze(1).unsqueeze(3)],1).cpu().numpy()
        gp_val = y_sample[:, :, :, [0]].cpu().numpy()
        # y_grad = {}
        # y_grad['y1'] = y_sample['y1'][:,:,1:3].cpu().numpy()
        # y_grad['y2'] = y_sample['y2'][:,:,1:3].cpu().numpy()
        y_grad = y_sample[:, :, :, 1:3].cpu().numpy()
        # u_grad = torch.cat([y_sample['y1'][:,:,-1].unsqueeze(2),y_sample['y2'][:,:,-1].unsqueeze(2)],2).cpu().numpy()
        # u_grad = torch.cat([y_sample[:,0,:,-1].unsqueeze(1).unsqueeze(3),y_sample[:,1,:,-1].unsqueeze(1).unsqueeze(3)],1).cpu().numpy()
        u_grad = y_sample[:, :, :, [-1]].cpu().numpy()
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

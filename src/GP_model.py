import torch
import gpytorch
import math


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    """_summary_ The model has multiple parameters: The corresponding values can be seen at:
    - model.likelihood.task_noises
    - model.likelihood.noises
    - model.covar_module.outputscale
    - model.covar_module.base_kernel.lengthscale
    - model.mean_module.constant

    Args:
        gpytorch (_type_): _description_
    """

    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = (
            gpytorch.means.ConstantMeanGrad()
        )  # (prior=gpytorch.priors.NormalPrior(4.9132,0.01))
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=3)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

        # self.mean_module = gpytorch.means.MultitaskMean(
        #     gpytorch.means.ConstantMean(), num_tasks=2
        # )
        # self.covar_module = gpytorch.kernels.LCMKernel(
        #     [
        #         gpytorch.kernels.RBFKernelGrad(active_dims=[0, 1]),
        #         gpytorch.kernels.RBFKernelGrad(active_dims=[2, 3]),
        #     ],
        #     num_tasks=6,
        #     rank=1,
        # )
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.RBFKernelGrad(), num_tasks=2, rank=1
        # )
        # self.covar_module.base_kernel.lengthscale = torch.Tensor([[1.2241]])
        # self.covar_module.outputscale = torch.Tensor([[2.4601]])
        # self.likelihood = likelihood  # it is unneccessary, it automatically set the value

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class BatchMultitaskGPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_shape):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad(
            batch_shape=batch_shape
        )  # (prior=gpytorch.priors.NormalPrior(4.9132,0.01))
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(
            ard_num_dims=3, batch_shape=batch_shape
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel, batch_shape=batch_shape
        )
        self.batch_shape = batch_shape

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class BatchMultitaskGPModelWithDerivatives_fromParams(
    BatchMultitaskGPModelWithDerivatives
):
    def __init__(self, train_x, train_y, likelihood, params, batch_shape=None):

        if batch_shape is None:
            batch_shape = torch.Size(
                [params["agent"]["num_dyn_samples"], params["agent"]["dim"]["ny"]]
            )

        super().__init__(
            train_x,
            train_y,
            likelihood,
            batch_shape=batch_shape,
        )

        self.likelihood.noise = torch.tile(
            torch.Tensor([params["agent"]["Dyn_gp_noise"]]),
            dims=(batch_shape[0], batch_shape[1], 1),
        )
        self.likelihood.task_noises = torch.tile(
            torch.Tensor(params["agent"]["Dyn_gp_task_noises"]["val"])
            * params["agent"]["Dyn_gp_task_noises"]["multiplier"],
            dims=(batch_shape[0], batch_shape[1], 1),
        )
        self.covar_module.base_kernel.lengthscale = torch.tile(
            torch.Tensor(params["agent"]["Dyn_gp_lengthscale"]["both"]),
            dims=(batch_shape[0], 1, 1, 1),
        )
        self.covar_module.outputscale = torch.tile(
            torch.Tensor(params["agent"]["Dyn_gp_outputscale"]["both"]),
            dims=(batch_shape[0], 1),
        )

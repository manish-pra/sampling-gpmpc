import gpytorch
import torch
import numpy as np


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, params):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=params["agent"]["g_dim"]["nx"]
                + params["agent"]["g_dim"]["nu"]
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def rbf_kernel(x1, x2, length_scale):
    """
    Radial Basis Function (RBF) kernel with a vector length scale.
    """
    return torch.exp(-torch.sum(((x1 - x2) ** 2) / (2 * length_scale**2))) * 0.5


def compute_kernel(x1, x2, length_scale):
    # Ensure length_scale is properly broadcasted
    scaled_diff = (x1[:, None, :] - x2[None, :, :]) ** 2 / (2 * length_scale**2)
    return 0.5 * torch.exp(-torch.sum(scaled_diff, dim=-1))


def compute_rkhs_norm(Dyn_gp_X_train, Dyn_gp_Y_train, params):
    Dyn_gp_noise = 0.0
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(Dyn_gp_noise),
        # batch_shape=torch.Size([3, 1]),
    )
    gp_idx = 0
    model_1 = ExactGPModel(
        Dyn_gp_X_train, Dyn_gp_Y_train[gp_idx, :, 0], likelihood, params
    )
    model_1.covar_module.base_kernel.lengthscale = torch.tensor(
        params["agent"]["Dyn_gp_lengthscale"]["both"]
    )
    model_1.likelihood.noise = torch.tensor(params["agent"]["Dyn_gp_noise"])
    model_1.covar_module.outputscale = torch.tensor(
        params["agent"]["Dyn_gp_outputscale"]["both"]
    )
    # model_1.eval()
    # pred = model_1(Dyn_gp_X_train)
    # K_DD = pred.covariance_matrix
    eval_covar_module = model_1.covar_module(Dyn_gp_X_train)
    K_DD = eval_covar_module.to_dense()
    # K_DD = compute_kernel(
    #     Dyn_gp_X_train,
    #     Dyn_gp_X_train,
    #     torch.tensor(params["agent"]["Dyn_gp_lengthscale"]["both"]),
    # )
    # kernel = covar_module = gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.RBFKernel(ard_num_dims=3)
    # )

    # alpha =  Dyn_gp_Y_train
    y = Dyn_gp_Y_train[gp_idx, :, 0].reshape(-1, 1)
    # norm = torch.matmul(y.t(), eval_covar_module.inv_matmul(y))
    lambda_sq = params["agent"]["Dyn_gp_noise"]
    K_DD_inv = torch.inverse(K_DD + lambda_sq * torch.eye(K_DD.shape[0]))
    alpha = torch.matmul(K_DD_inv, y)
    norm = torch.matmul(y.t(), alpha)
    print("RKHS norm of the mean function", norm)

    print("lengthscale", model_1.covar_module.base_kernel.lengthscale)
    print("noise", model_1.likelihood.noise)
    print("outputscale", model_1.covar_module.outputscale)
    # alpha = eval_covar_module.inv_matmul(y)
    return norm, alpha, y


def compute_posterior_norm_diff(Dyn_gp_X_train, Dyn_gp_Y_train, params):
    Dyn_gp_noise = 0.0
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(Dyn_gp_noise),
        # batch_shape=torch.Size([3, 1]),
    )
    gp_idx = 0
    model_1 = ExactGPModel(
        Dyn_gp_X_train, Dyn_gp_Y_train[gp_idx, :, 0], likelihood, params
    )
    model_1.covar_module.base_kernel.lengthscale = torch.tensor(
        params["agent"]["Dyn_gp_lengthscale"]["both"]
    )
    model_1.likelihood.noise = torch.tensor(params["agent"]["Dyn_gp_noise"])
    model_1.covar_module.outputscale = torch.tensor(
        params["agent"]["Dyn_gp_outputscale"]["both"]
    )
    model_1.eval()
    pred = model_1(Dyn_gp_X_train)
    y = Dyn_gp_Y_train[gp_idx, :, 0].reshape(-1)
    diff = torch.abs(pred.mean - y)
    out = (
        torch.sum((diff + params["agent"]["tight"]["w_bound"]) ** 2)
        / params["agent"]["Dyn_gp_noise"]
    )
    return out


def compute_small_ball_probability(Dyn_gp_X_train, Dyn_gp_Y_train, params):

    # Computation of C_D
    # 1) Compute RKHS norm of the mean function
    #######################################

    g_nx = params["agent"]["g_dim"]["nx"]
    g_nu = params["agent"]["g_dim"]["nu"]
    g_ny = params["agent"]["g_dim"]["ny"]

    train_x = torch.ones((1, g_nx + g_nu)) * 20000
    train_y = torch.zeros((1, g_ny))
    Dyn_gp_noise = 0.0
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(Dyn_gp_noise)
    )
    # model = ExactGPModel(train_x, train_y, likelihood)
    model = ExactGPModel(Dyn_gp_X_train, Dyn_gp_Y_train[0, :, 0], likelihood, params)
    # model.covar_module.base_kernel.lengthscale = torch.tensor(
    #     params["agent"]["Dyn_gp_lengthscale"]["both"]
    # )
    # model.likelihood.noise = 1.0e-6
    # model.covar_module.outputscale = 0.1
    model.covar_module.base_kernel.lengthscale = torch.tensor(
        params["agent"]["Dyn_gp_lengthscale"]["both"]
    )
    # model.covar_module.base_kernel.lengthscale = 5.2649
    model.likelihood.noise = torch.tensor(params["agent"]["Dyn_gp_noise"])
    model.covar_module.outputscale = torch.tensor(
        params["agent"]["Dyn_gp_outputscale"]["both"]
    )

    # Define the ranges
    x_range = (params["optimizer"]["x_min"][0], params["optimizer"]["x_max"][0])
    z_range = (params["optimizer"]["u_min"], params["optimizer"]["u_max"])

    # Define the number of points in each dimension
    num_points = 11  # You can adjust this number as needed

    # Generate the linspace for each dimension
    x = np.linspace(x_range[0], x_range[1], num_points)
    z = np.linspace(z_range[0], z_range[1], num_points)

    # Create a meshgrid for multi-dimensional space
    X, Z = np.meshgrid(x, z, indexing="ij")

    # Flatten the grid and stack the coordinates into a tensor of shape (-1, 3)

    grid_points = torch.from_numpy(np.stack([X.flatten(), Z.flatten()], axis=-1))
    if params["common"]["use_cuda"] and torch.cuda.is_available():
        grid_points = grid_points.cuda()
    print(grid_points.shape)

    # X = torch.linspace(-np.pi, np.pi, 100)

    total_samples = 100000
    with torch.no_grad(), gpytorch.settings.observation_nan_policy(
        "mask"
    ), gpytorch.settings.fast_computations(
        covar_root_decomposition=False, log_prob=False, solves=False
    ), gpytorch.settings.cholesky_jitter(
        float_value=params["agent"]["Dyn_gp_jitter"],
        double_value=params["agent"]["Dyn_gp_jitter"],
        half_value=params["agent"]["Dyn_gp_jitter"],
    ):
        model.eval()
        model_call = model(grid_points)
        samples = model_call.sample(sample_shape=torch.Size([total_samples]))

    assert not torch.any(torch.isnan(samples))
    sample_diff = samples - model_call.mean
    # plt.plot(X, samples.transpose(0, 1), lw=0.1)
    # plt.savefig("samples.png")

    eps = params["agent"]["tight"]["dyn_eps"]
    in_samples = torch.logical_and(sample_diff > -eps, sample_diff < eps)
    total_in_samples = torch.sum(torch.all(in_samples, dim=1))
    print("in samples", total_in_samples)
    print("Probability", total_in_samples / total_samples)

    print("lengthscale", model.covar_module.base_kernel.lengthscale)
    print("noise", model.likelihood.noise)
    print("outputscale", model.covar_module.outputscale)
    return total_in_samples / total_samples

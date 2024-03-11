import os
from datetime import datetime

# import pickle
import dill as pickle
import matplotlib.pyplot as plt
import torch
import yaml
from acados_template import AcadosOcpSolver, AcadosSimSolver
from botorch.models import SingleTaskGP
from gpytorch.kernels import (
    LinearKernel,
    MaternKernel,
    PiecewisePolynomialKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
)
import numpy as np

# from src.utils.ocp import export_sim


class ContiWorld:
    """[summary] ls_Fx: lengthscale of"""

    # def __init__(self, env_params, common_params, env_dir) -> None:
    #     self.env_dim = common_params["dim"]
    #     self.N = env_params["shape"]["x"]

    def __init__(self, env_params, common_params, visu_params, env_dir, params) -> None:
        self.env_dim = common_params["dim"]
        self.Nx = env_params["shape"]["x"]
        self.Ny = env_params["shape"]["y"]
        self.Dyn_gp_beta = env_params["Dyn_gp_beta"]
        self.Dyn_gp_lengthscale = env_params["Dyn_gp_lengthscale"]
        self.Dyn_gp_noise = env_params["Dyn_gp_noise"]
        self.n_players = env_params["n_players"]
        self.VisuGrid = grid(
            env_params["shape"], visu_params["step_size"], env_params["start"]
        )
        if env_params["cov_module"] == "Sq_exp":
            self.Cx_covar_module = ScaleKernel(
                base_kernel=RBFKernel(),
            )  # ard_num_dims=self.env_dim
            self.Fx_covar_module = ScaleKernel(
                base_kernel=RBFKernel(),
            )  # ard_num_dims=self.env_dim
        elif env_params["cov_module"] == "Matern":
            self.Cx_covar_module = ScaleKernel(
                base_kernel=MaternKernel(nu=2.5),
            )  # ard_num_dims=self.env_dim
            self.Fx_covar_module = ScaleKernel(
                base_kernel=MaternKernel(nu=2.5),
            )  # ard_num_dims=self.env_dim
        else:
            self.Cx_covar_module = ScaleKernel(
                base_kernel=PiecewisePolynomialKernel()
            )  # ard_num_dims=self.env_dim
            self.Fx_covar_module = ScaleKernel(
                base_kernel=PiecewisePolynomialKernel()
            )  # ard_num_dims=self.env_dim
        self.env_dir = env_dir
        env_file_path = env_dir + env_params["env_file_name"]
        self.constraint = common_params["constraint"]
        self.epsilon = common_params["epsilon"]
        self.params = params
        self.env_data = {}
        # Update GP with x,u, x_next
        if env_params["generate"] == True:
            global SingleTaskGP
            self.__Cx = self.true_constraint_sampling()
            self.__Fx = self.true_density_sampling()
            a, b = self.__Cx, self.__Fx
            self.__init_safe = {}
            init = self.__get_safe_init()
            self.__init_safe["loc"] = init[0]
            self.__init_safe["idx"] = init[1]
            self.env_data["Cx"] = self.__Cx
            self.env_data["Fx"] = self.__Fx
            self.env_data["Cx_model"] = self.Cx_model_cont
            self.env_data["Fx_model"] = self.Fx_model_cont
            self.env_data["init_safe"] = self.__init_safe
            a_file = open(env_file_path, "wb")
            pickle.dump(self.env_data, a_file)
            a_file.close()
            self.params["env"]["start_loc"] = init[0][0].tolist()
            self.params["env"]["goal_loc"] = self.VisuGrid[
                torch.randint(0, self.VisuGrid.shape[0], (1,)).item()
            ].tolist()
            self.plot2D()
            with open(
                self.env_dir + "/params_env.yaml", "w", encoding="utf-8"
            ) as yaml_file:
                dump = yaml.dump(
                    self.params["env"],
                    default_flow_style=False,
                    allow_unicode=True,
                    encoding=None,
                )
                yaml_file.write(dump)

        # elif env_params["generate"] == False:
        #     k = open(env_file_path, "rb")
        #     self.env_data = pickle.load(k)
        #     k.close()
        #     self.env_data["init_safe"]['loc'][0] = torch.Tensor(env_params["start_loc"])
        #     self.Cx_model_cont = self.env_data["Cx_model"]
        #     self.Fx_model_cont = self.env_data["Fx_model"]
        #     self.__Cx = self.Cx_model_cont.posterior(
        #         self.VisuGrid).mvn.mean.detach()
        #     # print("Lipschitz", self.get_true_lipschitz())
        #     # self.__Fx = self.Fx_model_cont.posterior(
        #     #     self.VisuGrid).mvn.mean.detach()
        #     self.__Fx = torch.norm(self.VisuGrid - torch.Tensor(env_params["goal_loc"]),2, dim=1)
        #     init = {}
        #     init['loc'] = []
        #     init['idx'] = []
        #     for init_agent in self.env_data["init_safe"]['loc']:
        #         idx = torch.abs((self.VisuGrid - init_agent)
        #                         [:, 0]).argmin()
        #         init['loc'].append(self.VisuGrid[idx])
        #         init['idx'].append(idx)
        #     init['idx'] = torch.stack(init['idx']).reshape(-1)
        #     self.__init_safe = init
        #     if self.Ny != 1:
        #         self.__init_safe = self.env_data["init_safe"]
        #     a = 1
        #     self.plot2D()

        # self.state = self.__init_safe
        # # sim = export_sim(params, 'sim_env')
        # # self.integrator = AcadosSimSolver(sim, json_file='acados_ocp_env.json')
        # # self.test_integrator(10000)
        # a = 1
        # # self.plot1D()

    def test_integrator(self, iter):
        x_curr = torch.Tensor([0.72, 0.0]).numpy()
        U = torch.zeros(1).numpy()
        x_record = []
        for i in range(iter):
            self.integrator.set("x", x_curr)
            self.integrator.set("u", U)
            self.integrator.solve()
            x_curr = self.integrator.get("x")
            x_record.append(x_curr[0])
            print(x_curr)
        plt.close()
        plt.plot(x_record)
        plt.show()
        plt.savefig("integrator.png")

    def get_true_safety_func(self):
        return self.__Cx

    def get_true_objective_func(self):
        return self.__Fx

    def plot2D(self):
        f, ax = plt.subplots()
        ax.set_aspect("equal", "box")
        x = self.VisuGrid
        posterior_mean = (
            self.Cx_model_cont.posterior(x)
            .mean.detach()
            .numpy()
            .reshape(self.Nx, self.Ny)
        )
        self.x = self.VisuGrid.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[0]
        self.y = self.VisuGrid.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[1]
        CS = ax.contour(
            self.x.numpy(),
            self.y.numpy(),
            posterior_mean,
            np.array([self.constraint, self.constraint + self.epsilon]),
        )
        ax.plot(
            self.params["env"]["start_loc"][0],
            self.params["env"]["start_loc"][1],
            "*",
            color="tab:red",
            mew=2,
        )
        ax.plot(
            self.params["env"]["goal_loc"][0],
            self.params["env"]["goal_loc"][1],
            "*",
            color="tab:green",
            mew=2,
        )
        # plt.plot(observed_pred.mean.detach().numpy())
        # lower, upper = observed_posterior.mvn.confidence_region()
        # lower = lower*(1+self.Cx_beta)/2 + upper*(1-self.Cx_beta)/2
        # upper = upper*(1+self.Cx_beta)/2 + lower*(1-self.Cx_beta)/2
        # ax.plot(x[:, 0].numpy(), self.__Cx, color="tab:orange")
        # ax.fill_between(x[:, 0].numpy(), lower.detach().numpy(),
        #                 upper.detach().numpy(), alpha=0.5, color="tab:blue")
        # ax.plot(x[:, 0].numpy(),
        #         observed_posterior.mean.detach().numpy(), color="tab:blue", label="Cx-mean")

        # for init_idx in self.__init_safe["idx"]:
        #     ax.plot(x[init_idx, 0], observed_posterior.mean.detach().numpy()
        #             [init_idx], "*", color="red", mew=2)
        # ax.axhline(y=self.constraint, linestyle='--', color="k")

        # Fx_observed_posterior = self.Fx_model_cont.posterior(x)
        # # ax.plot(observed_pred.mean.detach().numpy())
        # Fx_lower, Fx_upper = Fx_observed_posterior.mvn.confidence_region()
        # Fx_lower = Fx_lower
        # Fx_upper = Fx_upper
        # temp = Fx_lower*(1+self.Fx_beta)/2 + Fx_upper*(1-self.Fx_beta)/2
        # Fx_upper = Fx_upper*(1+self.Fx_beta)/2 + Fx_lower*(1-self.Fx_beta)/2
        # Fx_lower = temp
        # ax.plot(x[:, 0].numpy(), self.__Fx, color="tab:orange")
        # ax.fill_between(x[:, 0].numpy(), Fx_lower.detach().numpy(),
        #                 Fx_upper.detach().numpy(), alpha=0.5, color="tab:purple")
        # ax.plot(x[:, 0].numpy(),
        #         Fx_observed_posterior.mean.detach().numpy(), color="tab:purple", label="Fx-mean")
        # plt.show()
        plt.savefig(self.env_dir + "env.png")

    def plot1D(self):
        f, ax = plt.subplots()
        x = self.VisuGrid
        observed_posterior = self.Cx_model_cont.posterior(x)
        # plt.plot(observed_pred.mean.detach().numpy())
        lower, upper = observed_posterior.mvn.confidence_region()
        lower = lower * (1 + self.Cx_beta) / 2 + upper * (1 - self.Cx_beta) / 2
        upper = upper * (1 + self.Cx_beta) / 2 + lower * (1 - self.Cx_beta) / 2
        ax.plot(x[:, 0].numpy(), self.__Cx, color="tab:orange")
        ax.fill_between(
            x[:, 0].numpy(),
            lower.detach().numpy(),
            upper.detach().numpy(),
            alpha=0.5,
            color="tab:blue",
        )
        ax.plot(
            x[:, 0].numpy(),
            observed_posterior.mean.detach().numpy(),
            color="tab:blue",
            label="Cx-mean",
        )

        for init_idx in self.__init_safe["idx"]:
            ax.plot(
                x[init_idx, 0],
                observed_posterior.mean.detach().numpy()[init_idx],
                "*",
                color="red",
                mew=2,
            )
        ax.axhline(y=self.constraint, linestyle="--", color="k")

        Fx_observed_posterior = self.Fx_model_cont.posterior(x)
        # ax.plot(observed_pred.mean.detach().numpy())
        Fx_lower, Fx_upper = Fx_observed_posterior.mvn.confidence_region()
        Fx_lower = Fx_lower
        Fx_upper = Fx_upper
        temp = Fx_lower * (1 + self.Fx_beta) / 2 + Fx_upper * (1 - self.Fx_beta) / 2
        Fx_upper = Fx_upper * (1 + self.Fx_beta) / 2 + Fx_lower * (1 - self.Fx_beta) / 2
        Fx_lower = temp
        ax.plot(x[:, 0].numpy(), self.__Fx, color="tab:orange")
        ax.fill_between(
            x[:, 0].numpy(),
            Fx_lower.detach().numpy(),
            Fx_upper.detach().numpy(),
            alpha=0.5,
            color="tab:purple",
        )
        ax.plot(
            x[:, 0].numpy(),
            Fx_observed_posterior.mean.detach().numpy(),
            color="tab:purple",
            label="Fx-mean",
        )
        plt.show()
        plt.savefig(self.env_dir + "env.png")

    def propagate(self, input):
        self.state = self.state + input
        return self.state

    def get_disk_constraint_observation(self, disc_nodes):
        noise = torch.normal(
            mean=torch.zeros(len(disc_nodes), 1),
            std=torch.ones(len(disc_nodes), 1) * self.Fx_noise,
        )
        obs_Cx = [self.__Cx[node] + noise[idx] for idx, node in enumerate(disc_nodes)]
        disc_pts = self.grid_V[disc_nodes]
        return torch.stack(obs_Cx).reshape(-1, 1), disc_pts

    def get_disk_density_observation(self, disc_nodes):
        noise = torch.normal(
            mean=torch.zeros(len(disc_nodes), 1),
            std=torch.ones(len(disc_nodes), 1) * self.Fx_noise,
        )
        obs_Fx = [self.__Fx[node] + noise[idx] for idx, node in enumerate(disc_nodes)]
        disc_pts = self.grid_V[disc_nodes]
        return torch.stack(obs_Fx).reshape(-1, 1), disc_pts

    def true_density_sampling(self):
        # torch.Tensor([0]).reshape(-1, 1)
        self.Fx_X = (torch.rand(2) * 10).reshape(-1, self.env_dim)
        self.Fx_Y = torch.zeros(self.Fx_X.shape[0], 1)
        self.Fx_model = SingleTaskGP(
            self.Fx_X, self.Fx_Y, covar_module=self.Fx_covar_module
        )
        self.Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale
        self.Fx_model.likelihood.noise = self.Fx_noise
        density = self.Fx_model.posterior(self.VisuGrid).sample().reshape(-1, 1)
        if density.min() > -3:
            meas_density = density + 3
        else:
            meas_density = density + density.min()
        self.Fx_model_cont = SingleTaskGP(
            self.VisuGrid, meas_density, covar_module=self.Fx_covar_module
        )
        return meas_density
        # TODO update GP with this sample, to get continuous mean and variance model

    def true_constraint_sampling(self):
        # torch.Tensor([0]).reshape(-1, 1)
        self.Cx_X = (torch.rand(6) * 10).reshape(-1, self.env_dim)
        self.Cx_Y = torch.zeros(self.Cx_X.shape[0], 1)
        # self.Cx_model = SingleTaskGP(
        #     self.Cx_X, self.Cx_Y, covar_module=self.Cx_covar_module)
        self.Cx_model = SingleTaskGP(
            self.Cx_X, self.Cx_Y, covar_module=self.Cx_covar_module
        )
        self.Cx_model.covar_module.base_kernel.lengthscale = self.Cx_lengthscale
        self.Cx_model.likelihood.noise = self.Cx_noise
        sample = self.Cx_model.posterior(self.VisuGrid).sample().reshape(-1, 1)
        # sample = self.Cx_model.posterior(self.VisuGrid).sample().reshape(-1, 1)
        print(torch.mean(sample[sample > 0.1]))
        ret = sample - torch.mean(sample) + 0.2
        self.Cx_model_cont = SingleTaskGP(
            self.VisuGrid, ret, covar_module=self.Cx_covar_module
        )
        # plt.plot(torch.diff(ret)/0.12)
        # plt.plot(ret)
        # plt.show()
        return ret

    def get_multi_density_observation(self, sets):
        train = {}
        locs = torch.vstack(sets)
        train["Fx_Y"] = self.Fx_model_cont.posterior(locs.reshape(-1, 2)).mean
        # TODO may be change it such that it returens a list of 1D tensor
        return train["Fx_Y"]

    def get_multi_constraint_observation(self, sets):
        train = {}
        locs = torch.vstack(sets)
        train["Cx_Y"] = self.Cx_model_cont.posterior(locs.reshape(-1, 2)).mean
        # TODO may be change it such that it returens a list of 1D tensor
        return train["Cx_Y"]

    def get_density_observation(self, loc):
        obs_Fx = []
        noise = torch.normal(
            mean=torch.zeros(2, 1), std=torch.ones(2, 1) * self.Fx_noise
        )
        obs_Fx.append(self.Fx_model_cont.posterior(loc.reshape(-1, 2)).mean + noise[0])
        return torch.stack(obs_Fx).reshape(-1, 1)

    def get_constraint_observation(self, loc):
        obs_Cx = []
        noise = torch.normal(
            mean=torch.zeros(2, 1), std=torch.ones(2, 1) * self.Cx_noise
        )
        # self.Cx_model_cont.posterior(loc.reshape(-1,2)).mean
        obs_Cx.append(self.Cx_model_cont.posterior(loc).mean + noise[0])
        return torch.stack(obs_Cx).reshape(-1, 1)

    def get_true_lipschitz(self):  # get_true_safety_func
        tol = 10
        i = 0
        prev_max = 0
        while tol > 0.001:
            i = i + 2
            step_size = self.params["visu"]["step_size"] / i
            shape = {
                key: self.params["env"]["shape"][key] * i
                for key in self.params["env"]["shape"]
            }
            self.LcGrid = grid(shape, step_size, self.params["env"]["start"])
            Cx = self.Cx_model_cont.posterior(self.LcGrid).mvn.mean.detach()
            tr_constraint = Cx.reshape(self.Nx * i, self.Ny * i)
            Lc = torch.diff(tr_constraint.transpose(0, 1)).max() / step_size
            Lc1 = torch.diff(tr_constraint).max() / step_size
            max = torch.max(Lc, Lc1)
            tol = torch.abs(max - prev_max)
            prev_max = max.clone()
            print(Lc, Lc1)
        return max

    def __get_safe_init(self):
        opt_set = self.__Cx - self.epsilon * 1.5 > self.constraint
        p = opt_set.view(-1) * 1
        for i in range(2):
            p_forward = torch.cat([p[1:], torch.zeros(1)], dim=0)
            p_backward = torch.cat([torch.zeros(1), p[:-1]], dim=0)
            p = p_forward * p_backward
        dist = p / torch.sum(p)
        # number of places to put agents is less than number of agents
        if torch.sum(p) + 1 < self.n_players:
            raise "Non valid init"
        idx = dist.multinomial(self.n_players, replacement=False)
        return [loc.reshape(-1) for loc in self.VisuGrid[idx]], idx

    def get_safe_init(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        init_xy = {}
        init_xy["Cx_X"] = self.__init_safe["loc"]
        init_xy["Fx_X"] = self.__init_safe["loc"]
        init_xy["Cx_Y"] = self.get_multi_constraint_observation(init_xy["Cx_X"])
        init_xy["Fx_Y"] = self.get_multi_density_observation(init_xy["Fx_X"])
        return init_xy

    def step(self, action_u):
        self.state = torch.matmul(self.A, self.state) + torch.matmul(self.B, action_u)
        return self.state


def nodes_to_states(nodes, world_shape, step_size):
    """Convert node numbers to physical states.
    Parameters
    ----------
    nodes: np.array
        Node indices of the grid world
    world_shape: tuple
        The size of the grid_world
    step_size: np.array
        The step size of the grid world
    Returns
    -------
    states: np.array
        The states in physical coordinates
    """
    nodes = torch.as_tensor(nodes)
    step_size = torch.as_tensor(step_size)
    return (
        torch.vstack(((nodes // world_shape["y"]), (nodes % world_shape["y"]))).T
        * step_size
    )


def grid(world_shape, step_size, start_loc):
    """
    Creates grids of coordinates and indices of state space
    Parameters
    ----------
    world_shape: tuple
        Size of the grid world (rows, columns)
    step_size: tuple
        Phyiscal step size in the grid world
    Returns
    -------
    states_ind: np.array
        (n*m) x 2 array containing the indices of the states
    states_coord: np.array
        (n*m) x 2 array containing the coordinates of the states
    """
    nodes = torch.arange(0, world_shape["x"] * world_shape["y"])
    return nodes_to_states(nodes, world_shape, step_size) + np.array(start_loc)


if __name__ == "__main__":
    workspace = "safe-mpc"
    with open(workspace + "/params/params_env.yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    if params["env"]["generate"]:
        for i in range(0, 10):
            env_load_path = (
                workspace
                + "/experiments/"
                + params["experiment"]["folder"]
                + "/env_"
                + str(i)
                + "/"
            )
            # save_path = env_load_path + "/" + args.param + "/"
            # save_path = workspace + "/experiments/" + datetime.today().strftime('%d-%m-%y') + \
            #     datetime.today().strftime(
            #         '-%A')[0:4] + "/environments/env_" + str(i) + "/"
            # save_path = workspace + \
            #     "/experiments/gorilla11/environments/env_" + str(i) + "/"
            if not os.path.exists(env_load_path):
                os.makedirs(env_load_path)
            env = ContiWorld(
                env_params=params["env"],
                common_params=params["common"],
                visu_params=params["visu"],
                env_dir=env_load_path,
                params=params,
            )
    else:
        exp_name = params["experiment"]["name"]
        env_load_path = (
            workspace + "/experiments/16-07-22-Sat/environments/env_" + str(0) + "/"
        )
        save_path = env_load_path + "/"

        env = ContiWorld(
            env_params=params["env"],
            common_params=params["common"],
            visu_params=params["visu"],
            env_dir=save_path,
            params=params,
        )
    # env = ContiWorld()
    # for i in range(10):
    #     print(env.step(torch.FloatTensor([[1], [1]])))

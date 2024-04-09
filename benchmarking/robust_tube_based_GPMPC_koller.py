import sys
import torch

# clone https://github.com/manish-pra/safe-exploration-koller and add it to the path
# Add path to safe-exploration-koller
sys.path.append("/home/manish/work/safe-exploration-koller/")
from easydict import EasyDict
from safe_exploration.gp_reachability_pytorch import onestep_reachability
from safe_exploration.ssm_cem.gp_ssm_cem import GpCemSSM
from safe_exploration.environments.environments import InvertedPendulum, Environment
import numpy as np
import numpy.linalg as nLa
import matplotlib.pyplot as plt
import dill as pickle
import gpytorch

# save_path = "/home/manish/work/horrible/safe-exploration_cem/experiments"
# a_file = open(save_path + "/data.pkl", "rb")
save_path = "/home/manish/work/MPC_Dyn/safe_gpmpc/experiments/pendulum/env_0/params/401_sampling_mpc/data.pkl"
a_file = open(save_path, "rb")
data_dict = pickle.load(a_file)
state_traj = data_dict["state_traj"]
input_traj = torch.Tensor(data_dict["input_traj"])
a_file.close()
x0 = state_traj[0][0, 0:2]

env = InvertedPendulum(verbosity=0)
env.n_s = 2
env.n_u = 1

conf = {
    "exact_gp_kernel": "rbf",
    "cem_ssm": "exact_gp",
    "exact_gp_training_iterations": 1000,
    "cem_beta_safety": 2.5,
    "device": "cpu",
}

conf = EasyDict(conf)
x1 = torch.linspace(-2.14, 2.14, 5)
x2 = torch.linspace(-2.5, 2.5, 5)
u = torch.linspace(-8, 8, 5)
X1, X2, U = torch.meshgrid(x1, x2, u)
X_train = torch.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1), U.reshape(-1, 1)])


def pendulum_discrete_dyn(X1_k, X2_k, U_k):
    l = 1
    g = 10
    dt = 0.015
    X1_kp1 = X1_k + X2_k * dt
    X2_kp1 = X2_k - g * torch.sin(X1_k) * dt / l + U_k * dt / (l * l)
    return torch.vstack((X1_kp1, X2_kp1)).transpose(0, 1)


Y_train = pendulum_discrete_dyn(X_train[:, 0], X_train[:, 1], X_train[:, 2])

n_s, n_u = env.n_s, env.n_u
ssm = GpCemSSM(conf, env.n_s, env.n_u)
device = "cpu"

x_train = torch.tensor(X_train).to(device)
y_train = torch.tensor(Y_train).to(device)
ssm.update_model(x_train, y_train, opt_hyp=True)
# ssm._likelihood.noise = 7.0e-9
# ssm._model.likelihood.constraints = gpytorch.constraints.GreaterThan(0.0)
# ssm._likelihood.noise_covar = gpytorch.constraints.GreaterThan(0.0)
ssm._model.likelihood.noise = 7.0e-9
ssm._model.likelihood.noise_covar.noise = 7.0e-9
ssm._model._kernel.outputscale = 0.1040
ssm._model._kernel._modules["base_kernel"].lengthscale = torch.Tensor(
    [5.2649, 4.5967, 7.0177]
)
# ssm._model._kernel._modules['base_kernel'].lengthscale = torch.tensor([0.1, 0.1])
# ssm._model._kernel.lengthscale = torch.tensor([0.1, 0.1])
x_test = torch.tensor([[8.0]]).to(device)
states = torch.zeros((x_test.shape[0], env.n_s)).to(device)
actions = torch.tensor(x_test).to(device)

a = torch.zeros((env.n_s, env.n_s), device=device)
b = torch.zeros((env.n_s, env.n_u), device=device)

ps, qs, _ = onestep_reachability(
    states,
    ssm,
    actions,
    torch.tensor(env.l_mu).to(device),
    torch.tensor(env.l_sigm).to(device),
    c_safety=conf.cem_beta_safety,
    verbose=0,
    a=a,
    b=b,
)
print(ps, qs)
H = 30

k_fb_apply = torch.zeros((n_u, n_s))
# k_fb_apply = torch.reshape(k_fb_0, (-1, n_u, n_s))
ellipse_list = []
# iteratively compute it for the next steps
for i in range(H):
    print(i)
    ps, qs, _ = onestep_reachability(
        ps,
        ssm,
        input_traj[0][i].reshape(-1, 1),
        torch.tensor(env.l_mu).to(device),
        torch.tensor(env.l_sigm).to(device),
        q_shape=qs,
        k_fb=k_fb_apply,
        c_safety=conf.cem_beta_safety,
        verbose=0,
        a=a,
        b=b,
    )
    print(ps, qs)

    r = nLa.cholesky(qs).T
    r = r[:, :, 0]
    # checks spd inside the function
    t = np.linspace(0, 2 * np.pi, 100)
    z = [np.cos(t), np.sin(t)]
    ellipse = np.dot(r, z) + ps.numpy().T
    ellipse_list.append(ellipse)
    plt.plot(ellipse[0, :], ellipse[1, :])
# a_file = open("koller_ellipse_data.pkl", "wb")
# pickle.dump(ellipse_list, a_file)
plt.xlim(-0.1, 1.45)
plt.ylim(-0.1, 2.7)
plt.show()

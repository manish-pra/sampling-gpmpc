# from utils import *
from plotting_utilities.plotting_utilities.utilities import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from pathlib import Path
from botorch.models import SingleTaskGP
from gpytorch.kernels import (
    LinearKernel,
    MaternKernel,
    PiecewisePolynomialKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
)

# Load the pytorch model 3025750798213835061 8740660194874727738
seed = 13048364859988300057  # torch.random.seed() #8740660194874727738 #torch.random.seed() 14864543189012277679
print(seed)
torch.random.manual_seed(seed)

model1_F_x = torch.Tensor([[-2.5], [-1], [4]])
model1_F_y = torch.Tensor([[0.2], [0.4], [1]]) - 0.1
model2_F_x = torch.Tensor([[-1.5], [3], [2.5]])
model2_F_y = torch.Tensor([[0.2], [0.4], [1]]) - 0.1
model3_F_x = torch.Tensor([[-0.5], [1.4], [0.4]])
model3_F_y = torch.Tensor([[1.2], [0.4], [1]]) + 0.8

model1_F_x_2 = torch.Tensor([[-2], [-0.45], [3]])
model1_F_y_2 = torch.Tensor([[0.5], [0.5], [1.5]]) - 0.1
model2_F_x_2 = torch.Tensor([[-2], [3.5], [4]])
model2_F_y_2 = torch.Tensor([[1], [0], [0.5]]) - 0.1


model2_F_x_3 = torch.Tensor([[-2], [-0.5], [1]]) - 0.5
model2_F_y_3 = torch.Tensor([[1.6], [1], [1.2]]) - 0.1


def torch_model(Fx_X, Fx_Y):
    # Fx_X = torch.tensor([[0.2], [1], [ 2.1458], [-3.1922], [-3.7438]])
    # Fx_Y = torch.tensor([[1.7], [1.4], [1.9969], [0.767],[0.6552]]) - 0.1
    # Model 1
    Fx_model = SingleTaskGP(
        Fx_X,
        Fx_Y,
        covar_module=ScaleKernel(
            base_kernel=RBFKernel(),
        ),
    )
    Fx_model.covar_module.base_kernel.lengthscale = 0.88
    Fx_model.likelihood.noise = 0.0001
    return Fx_model
    # density = Fx_model.posterior(grid_V).sample().reshape(-1)
    # if density.min() > -3:
    #     return density + 3
    # else:
    #     return density + density.min()


Fx_X_init = torch.tensor([[0.2], [1], [2.1458], [-3.1922], [-3.7438]])
Fx_Y_init = torch.tensor([[1.7], [1.4], [1.9969], [0.767], [0.6552]]) - 0.1

Fx_X = torch.vstack([Fx_X_init, model2_F_x, model2_F_x_2])
Fx_Y = torch.vstack([Fx_Y_init, model2_F_y, model2_F_y_2])
# Fx_X = torch.vstack([Fx_X_init, model2_F_x])
# Fx_Y = torch.vstack([Fx_Y_init, model2_F_y])
Fx_model = torch_model(Fx_X, Fx_Y)
# Fx_model = torch_model(Fx_X_init, Fx_Y_init)
Lc = 0.75

k = 1.2
qx = 0.41


def tr(x):
    return Fx_model.posterior(torch.Tensor(x).reshape(-1, 1)).sample().reshape(-1)
    # return np.exp(-(x+1.5)**2) + 1.7 * np.exp(-(x-1.5)**2)


x = np.linspace(-4.2, 4.1, 200)
tr_x = Fx_model.posterior(torch.Tensor(x).reshape(-1, 1)).sample().reshape(-1)
tr_x = Fx_model.posterior(torch.Tensor(x).reshape(-1, 1)).sample().reshape(-1)
tr_x = tr_x + 0.015 * torch.cat([torch.zeros(182), torch.cumsum(torch.ones(18), 0)])


def f(x):
    return (
        Fx_model.posterior(torch.Tensor(x).reshape(-1, 1))
        .mean.reshape(-1)
        .detach()
        .numpy()
    )


def sig(x):
    std = (
        Fx_model.posterior(torch.Tensor(x).reshape(-1, 1)).variance.reshape(-1).detach()
    )
    return 2.5 * std.sqrt().numpy()


# def get_slope():

# def lb(x):
#     return max(f(x)-sig(x),)
# V_lower_Cx_Lc = get_Lc_lb(f(x)-sig(x), x)


def get_cross(func, th=qx):
    leq_zero = func <= 0.5
    geq_zero = func >= 0.5
    up_cross = x[:-1][np.logical_and(leq_zero[:-1], geq_zero[1:])]
    down_cross = x[:-1][np.logical_and(geq_zero[:-1], leq_zero[1:])] + 0.04
    return up_cross, down_cross


def Lipschitz_lines(st_loc, dir="right"):
    pessi_st = f(st_loc) - sig(st_loc)
    delta = pessi_st - qx
    if dir == "right":
        return [st_loc, st_loc + Lc * delta], [pessi_st, qx]
    else:
        return [st_loc, st_loc - Lc * delta], [pessi_st, qx]


TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=24)
plt.figure(figsize=(TEXTWIDTH * 0.5 + 0.75, TEXTWIDTH * 0.5 * 1 / 2))

tr_x = torch.Tensor(
    [
        0.1470,
        0.2023,
        0.2446,
        0.2989,
        0.3366,
        0.3775,
        0.4124,
        0.4420,
        0.4849,
        0.5113,
        0.5450,
        0.5745,
        0.5994,
        0.6123,
        0.6432,
        0.6519,
        0.6728,
        0.6858,
        0.6898,
        0.6965,
        0.6939,
        0.6915,
        0.6910,
        0.6814,
        0.6585,
        0.6439,
        0.6228,
        0.6016,
        0.5733,
        0.5421,
        0.5096,
        0.4648,
        0.4315,
        0.3875,
        0.3396,
        0.2935,
        0.2442,
        0.2042,
        0.1632,
        0.1073,
        0.0628,
        0.0187,
        -0.0242,
        -0.0640,
        -0.1006,
        -0.1271,
        -0.1636,
        -0.1843,
        -0.2069,
        -0.2215,
        -0.2348,
        -0.2350,
        -0.2366,
        -0.2396,
        -0.2328,
        -0.2179,
        -0.2099,
        -0.1876,
        -0.1668,
        -0.1422,
        -0.1135,
        -0.0771,
        -0.0498,
        -0.0080,
        0.0308,
        0.0626,
        0.1087,
        0.1515,
        0.1959,
        0.2420,
        0.2840,
        0.3387,
        0.3880,
        0.4380,
        0.4896,
        0.5455,
        0.5925,
        0.6479,
        0.7048,
        0.7606,
        0.8107,
        0.8736,
        0.9301,
        0.9843,
        1.0375,
        1.0922,
        1.1405,
        1.2057,
        1.2507,
        1.2990,
        1.3431,
        1.3875,
        1.4257,
        1.4582,
        1.4907,
        1.5258,
        1.5500,
        1.5660,
        1.5860,
        1.5970,
        1.6050,
        1.6107,
        1.6111,
        1.6033,
        1.6002,
        1.5878,
        1.5713,
        1.5567,
        1.5453,
        1.5271,
        1.5079,
        1.4862,
        1.4654,
        1.4425,
        1.4262,
        1.4134,
        1.3872,
        1.3742,
        1.3614,
        1.3496,
        1.3411,
        1.3423,
        1.3392,
        1.3441,
        1.3377,
        1.3470,
        1.3534,
        1.3629,
        1.3803,
        1.3855,
        1.4034,
        1.4169,
        1.4490,
        1.4623,
        1.4767,
        1.5012,
        1.5204,
        1.5409,
        1.5655,
        1.5876,
        1.6066,
        1.6354,
        1.6525,
        1.6743,
        1.6918,
        1.7141,
        1.7311,
        1.7463,
        1.7673,
        1.7930,
        1.7996,
        1.8232,
        1.8343,
        1.8536,
        1.8676,
        1.8808,
        1.8902,
        1.8949,
        1.9061,
        1.9064,
        1.9073,
        1.9145,
        1.9153,
        1.9155,
        1.9080,
        1.8985,
        1.8955,
        1.8707,
        1.8572,
        1.8389,
        1.8108,
        1.7843,
        1.7570,
        1.7217,
        1.6848,
        1.6503,
        1.6056,
        1.5591,
        1.5078,
        1.4527,
        1.4065,
        1.3437,
        1.2957,
        1.2450,
        1.1905,
        1.1333,
        1.0789,
        1.0188,
        0.9558,
        0.8938,
        0.8323,
        0.7655,
        0.7005,
        0.6297,
        0.5717,
        0.5030,
        0.4427,
        0.3816,
        0.3194,
        0.2625,
    ]
)
# plt.plot(x, tr_x, linewidth=2.5,label=r'$g^{tr}$')

hi = 0.175
nbar = 4.35
hi_exp = (nbar - 3) * hi
hi_p = (nbar - 5) * hi
hi_pl = (nbar - 4) * hi
hi_0 = (nbar - 3) * hi
hi_o = 0
st_loc = np.array([2.1458])
dist = 0.02
Lc_lw = 2
Lc_color = "tab:orange"
# x_line, y_line = Lipschitz_lines(st_loc + dist,"right")
# plt.plot(x_line, y_line, '--',  color=Lc_color, linewidth=Lc_lw,label=r'$L_c$')
# x_line, y_line = Lipschitz_lines(st_loc - dist,"left")
# plt.plot(x_line, y_line, '--' , color=Lc_color, linewidth=Lc_lw)
p2_loc = np.array([0.2])
# x_line, y_line = Lipschitz_lines(p2_loc + dist,"right")
# plt.plot(x_line, y_line, '--', color=Lc_color, linewidth=Lc_lw)
# x_line, y_line = Lipschitz_lines(p2_loc - dist,"left")
# plt.plot(x_line, y_line, '--', color=Lc_color, linewidth=Lc_lw)
# plt.plot(x,V_lower_Cx_Lc,'--',color="tab:orange", label=r'$L_q$', linewidth=Lc_lw)

up_cross, down_cross = get_cross(f(x) - sig(x), th=qx)
idx = 1
rect4 = patches.Rectangle(
    (up_cross[idx], hi_p),
    down_cross[idx] - up_cross[idx],
    hi,
    linewidth=1,
    edgecolor="None",
    facecolor="tab:green",
    alpha=0.65,
    label=r"$\mathcal{R}_T(\mathbb{X}_n,\mathcal{S}^{p}_n)$",
)
# plt.gca().add_patch(rect4)


# plt.fill_between(x, V_lower_Cx_Lc, f(x)+sig(x), alpha=0.3,label=r'$\left[l_{n}(x),u_n(x)\right]$', color="tab:blue")
plt.fill_between(x, f(x) - sig(x), f(x) + sig(x), alpha=0.35, color="tab:blue")
# plt.plot(x, np.full_like(x, qx), '--k')
plt.plot(x, f(x) + sig(x), color="tab:blue")
plt.plot(x, f(x) - sig(x), color="tab:blue")

# plt.plot(0.5, 0.5, marker='$\U0001F601$', ms=20, c='green')
# plt.text(-2.75, 0.7, r'$q(x)\geq 0$')
# plt.text(-1.9, 0.6, '$q(\mathbf{x})\geq 0$')
plt.ylabel(r"$g(x,a)$")
plt.xlabel(r"$\mathcal{X}\times\mathcal{A}$")


rect1 = patches.Rectangle(
    (st_loc - 0.1 / 2, hi_0),
    0.1,
    hi,
    linewidth=0.5,
    edgecolor="None",
    facecolor="k",
    alpha=0.5,
    label=r"$\mathbb{X}_n$",
)

# hide_all_ticks(plt.gca())
# up_cross, down_cross = get_cross(tr(x)-0.05, th=qx)
# rect2 = patches.Rectangle((up_cross[0], hi_eps), down_cross[0] - up_cross[0], hi, linewidth=1,
#                           edgecolor='None', facecolor='tab:orange', alpha=0.5, label=r'$\overline{R}_\epsilon(X_0)$')
# up_cross, down_cross = get_cross(tr(x), th=qx)
# rect3 = patches.Rectangle((up_cross[0], hi_0), down_cross[0] - up_cross[0], hi, linewidth=1,
#                           edgecolor='None', facecolor='tab:purple', alpha=0.5, label=r'$\overline{R}_0(X_0)$')


x_right, y_right = Lipschitz_lines(st_loc + dist, "right")
x_left, y_left = Lipschitz_lines(p2_loc - dist, "left")
rect6 = patches.Rectangle(
    (x_left[1], hi_pl),
    x_right[1] - x_left[1],
    hi,
    linewidth=1,
    edgecolor="None",
    facecolor="tab:orange",
    alpha=0.65,
    label=r"$\mathcal{R}_T(\mathbb{X}_n,\mathcal{S}^{{p\!}_L}_n)$",
)
# rect6 = patches.Rectangle((up_cross[1], hi_p), down_cross[1] - up_cross[1], hi, linewidth=1,
#                           edgecolor='None', facecolor='tab:green', alpha=0.5)
up_cross, down_cross = get_cross(f(x) + sig(x), th=qx)
# rect5 = patches.Rectangle((up_cross[0], hi_o), down_cross[0] - up_cross[0], hi, linewidth=1,
#                           edgecolor='None', facecolor='tab:red', alpha=0.5, label=r'$S^{o}_n$')

up_cross, down_cross = get_cross(f(x) - sig(x), th=qx)
idx = 1
rect3 = patches.Rectangle(
    (x_left[1], hi_exp),
    -x_left[1] + up_cross[idx],
    hi,
    linewidth=1,
    edgecolor="None",
    facecolor="tab:purple",
    alpha=0.65,
    label=r"$\mathcal{G}_n(\mathbb{X}_n,\mathcal{S}^{{p\!}_L}_n)$",
)
idx = 1
rect2 = patches.Rectangle(
    (x_right[1], hi_exp),
    -x_right[1] + down_cross[idx],
    hi,
    linewidth=1,
    edgecolor="None",
    facecolor="tab:purple",
    alpha=0.65,
)

# plt.ylim((0, 2.05))
plt.xlim((-4.2, 4.1))
#
# # Aded the patch to the Axes


# plt.gca().add_patch(rect2)
# plt.gca().add_patch(rect3)


# # plt.gca().add_patch(rect5)
# plt.gca().add_patch(rect6)
# plt.gca().add_patch(rect1)

plt.plot(Fx_X_init, Fx_Y_init, "x", color="black", ms=18, mew=4)

# plt.plot(model1_F_x, model1_F_y, 'x', color='tab:green', ms=14, mew=3)
plt.plot(model2_F_x, model2_F_y, "x", color="tab:orange", ms=14, mew=3)
# plt.plot(model3_F_x, model3_F_y, 'x', color='tab:purple', ms=14, mew=3)

plt.plot(model2_F_x_2, model2_F_y_2, "+", color="orange", ms=16, mew=3)
plt.plot(model2_F_x_3, model2_F_y_3, "*", color="orange", ms=16, mew=3)

plt.legend(
    frameon=False,
    ncol=4,
    prop={"size": 20},
    columnspacing=0.6,
    loc="lower left",
    bbox_to_anchor=(-0.02, -0.067),
)
hide_spines(plt.gca())
hide_all_ticks(plt.gca())
plt.tight_layout(pad=0.0)
fname = Path().resolve().joinpath("figures")
fname.mkdir(exist_ok=True)
plt.savefig(
    str(fname.joinpath("gp_orig_22.png")), format="png", dpi=300, transparent=True
)
plt.show()

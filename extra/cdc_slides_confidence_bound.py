# from utils import *
from plotting_utilities.plotting_utilities.utilities import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from pathlib import Path


def f(x):
    return np.exp(-((x + 1.5) ** 2)) + 1.7 * np.exp(-((x - 1.5) ** 2))


def tr(x):
    return (
        np.exp(-((x + 1.5) ** 2))
        + 1.8 * np.exp(-((x - 1.5) ** 2))
        + np.cos(x * 10) / 10
    )


def sig(x):
    return 0.25


def get_cross(func, th=0.5):
    leq_zero = func <= 0.5
    geq_zero = func >= 0.5
    up_cross = x[:-1][np.logical_and(leq_zero[:-1], geq_zero[1:])]
    down_cross = x[:-1][np.logical_and(geq_zero[:-1], leq_zero[1:])]
    return up_cross, down_cross


TEXTWIDTH = 16
set_figure_params(serif=True, fontsize=24)
plt.figure(figsize=(TEXTWIDTH * 0.5, TEXTWIDTH * 0.5 * 1 / 2))
x = np.linspace(-4, 4, 200)

plt.plot(x, f(x), linewidth=1.5, label=r"$\mu(z)$")
plt.plot(x, tr(x), "--k", linewidth=2.5, label=r"$g^{\mathrm{tr}}(z)$")
plt.fill_between(
    x,
    f(x) - sig(x),
    f(x) + sig(x),
    alpha=0.5,
    label=r"$\left[\underline{g}(z),\bar{g}(z)\right]$",
)
# plt.plot(x, np.full_like(x, 0.5), "--k")
# plt.plot(0.5, 0.5, marker='$\U0001F601$', ms=20, c='green')
# plt.text(0.8, 0.6, r"$q(x)\geq 0$")
# plt.text(-1.9, 0.6, '$q(\mathbf{x})\geq 0$')
plt.ylabel(r"$g(z)$")
plt.xlabel(r"$\mathcal{Z}$")

# hi = 0.075
# nbar = 5
# hi_p = (nbar - 4) * hi
# hi_eps = (nbar - 3) * hi
# hi_0 = (nbar - 4) * hi
# hi_o = 0
# rect1 = patches.Rectangle(
#     (-1.5, (nbar - 3) * hi),
#     0.1,
#     hi,
#     linewidth=0.5,
#     edgecolor="None",
#     facecolor="k",
#     alpha=0.5,
#     label="$X_0$",
# )


# # hide_all_ticks(plt.gca())
# up_cross, down_cross = get_cross(tr(x) - 0.05, th=0.5)
# rect2 = patches.Rectangle(
#     (up_cross[0], hi_eps),
#     down_cross[0] - up_cross[0],
#     hi,
#     linewidth=1,
#     edgecolor="None",
#     facecolor="tab:orange",
#     alpha=0.5,
#     label=r"$\overline{R}_\epsilon(X_0)$",
# )
# up_cross, down_cross = get_cross(tr(x), th=0.5)
# rect3 = patches.Rectangle(
#     (up_cross[0], hi_0),
#     down_cross[0] - up_cross[0],
#     hi,
#     linewidth=1,
#     edgecolor="None",
#     facecolor="tab:purple",
#     alpha=0.5,
#     label=r"$\overline{R}_0(X_0)$",
# )
# up_cross, down_cross = get_cross(f(x) - sig(x), th=0.5)
# rect4 = patches.Rectangle(
#     (up_cross[0], hi_p),
#     down_cross[0] - up_cross[0],
#     hi,
#     linewidth=1,
#     edgecolor="None",
#     facecolor="tab:green",
#     alpha=0.5,
#     label=r"$\mathcal{S}^{p}_n$",
# )
# rect6 = patches.Rectangle(
#     (up_cross[1], hi_p),
#     down_cross[1] - up_cross[1],
#     hi,
#     linewidth=1,
#     edgecolor="None",
#     facecolor="tab:green",
#     alpha=0.5,
# )
# up_cross, down_cross = get_cross(f(x) + sig(x), th=0.5)
# rect5 = patches.Rectangle(
#     (up_cross[0], hi_o),
#     down_cross[0] - up_cross[0],
#     hi,
#     linewidth=1,
#     edgecolor="None",
#     facecolor="tab:orange",
#     alpha=0.5,
#     label=r"$\mathcal{S}^{o}_n$",
# )


plt.ylim((0, 2.05))
plt.xlim((-3.5, 3.5))
#
# # Aded the patch to the Axes
# plt.gca().add_patch(rect1)
# plt.gca().add_patch(rect2)
# plt.gca().add_patch(rect3)
# plt.gca().add_patch(rect4)
# plt.gca().add_patch(rect5)
# plt.gca().add_patch(rect6)
plt.legend(frameon=False, ncol=2, prop={"size": 24}, columnspacing=0.6)
hide_spines(plt.gca())
hide_all_ticks(plt.gca())
plt.tight_layout(pad=0.0)
fname = Path().resolve().joinpath("figures")
fname.mkdir(exist_ok=True)
plt.savefig(
    str(fname.joinpath("cdc_slides_confidence_bound.pdf")),
    format="pdf",
    dpi=300,
    transparent=True,
)
# plt.show()

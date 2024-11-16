import dill as pickle
import matplotlib.pyplot as plt

with open("/home/manish/work/MPC_Dyn/slides_data.pickle", "rb") as handle:
    data = pickle.load(handle)

a = 1


from plotting_utilities.plotting_utilities import *


format = "png"
y_lim = [-2.6, 2.6]
# Plot 1
# ===============================================================================
TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=10)
f, ax = plt.subplots(
    1, 1, figsize=(TEXTWIDTH * 0.5 / 2 + 0.25, TEXTWIDTH * 0.25 * 1 / 2)
)

i = 0
h_func = ax.plot(data["test_x"], data["test_y"], "k--")
j = 0
marker_sym = data["marker_symbols"][j]
ax.plot(
    data[i]["train_x_arr_add"][j].detach().numpy(),
    data[i]["train_y_arr_add"][j][:, 0].detach().numpy(),
    f"k{marker_sym}",
)

# remove tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
# remove ticks
ax.set_yticks([])
ax.set_xticks([])

ax.set_ylim(y_lim)

f.tight_layout(pad=0.5)
f.savefig(
    f"/home/manish/work/MPC_Dyn/figures/slides_fig{i}.{format}",
    format=format,
    dpi=300,
    transparent=True,
)


# ===============================================================================
# Plot 2

TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=10)
f, ax = plt.subplots(
    1, 1, figsize=(TEXTWIDTH * 0.5 / 2 + 0.25, TEXTWIDTH * 0.25 * 1 / 2)
)

i = 0
h_func = ax.plot(data["test_x"], data["test_y"], "k--")

h_mean = ax.plot(data["test_x"], data[i]["mean"], "tab:blue")

h_conf = ax.fill_between(
    data["test_x"],
    data[i]["lcb"],
    data[i]["ucb"],
    alpha=0.5,
    color="tab:blue",
)


j = 0
marker_sym = data["marker_symbols"][j]
ax.plot(
    data[i]["train_x_arr_add"][j].detach().numpy(),
    data[i]["train_y_arr_add"][j][:, 0].detach().numpy(),
    f"k{marker_sym}",
)

# remove tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
# remove ticks
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylim(y_lim)

f.tight_layout(pad=0.5)
f.savefig(
    f"/home/manish/work/MPC_Dyn/figures/slides_fig1.{format}",
    format=format,
    dpi=300,
    transparent=True,
)


# ===============================================================================
# Plot 3

TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=10)
f, ax = plt.subplots(
    1, 1, figsize=(TEXTWIDTH * 0.5 / 2 + 0.25, TEXTWIDTH * 0.25 * 1 / 2)
)

i = 0

for x in data[i]["train_x_arr_add"][i + 1]:
    ax.axvline(
        x.numpy().flatten(),
        color="k",
        linestyle="solid",
        alpha=0.3,
        linewidth=3,
    )

h_func = ax.plot(data["test_x"], data["test_y"], "k--")

h_mean = ax.plot(data["test_x"], data[i]["mean"], "tab:blue")

h_conf = ax.fill_between(
    data["test_x"],
    data[i]["lcb"],
    data[i]["ucb"],
    alpha=0.5,
    color="tab:blue",
)

j = 0
marker_sym = data["marker_symbols"][j]
ax.plot(
    data[i]["train_x_arr_add"][j].detach().numpy(),
    data[i]["train_y_arr_add"][j][:, 0].detach().numpy(),
    f"k{marker_sym}",
)

# remove tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
# remove ticks
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylim(y_lim)

f.tight_layout(pad=0.5)
f.savefig(
    f"/home/manish/work/MPC_Dyn/figures/slides_fig2.{format}",
    format=format,
    dpi=300,
    transparent=True,
)


# ===============================================================================
# Plot 4

TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=10)
f, ax = plt.subplots(
    1, 1, figsize=(TEXTWIDTH * 0.5 / 2 + 0.25, TEXTWIDTH * 0.25 * 1 / 2)
)

i = 0

for x in data[i]["train_x_arr_add"][i + 1]:
    ax.axvline(
        x.numpy().flatten(),
        color="k",
        linestyle="solid",
        alpha=0.3,
        linewidth=3,
    )

h_func = ax.plot(data["test_x"], data["test_y"], "k--")

h_mean = ax.plot(data["test_x"], data[i]["mean"], "tab:blue")
# h_samp = ax.plot(data["test_x"], data[i]["sample"], "tab:orange")

h_conf = ax.fill_between(
    data["test_x"],
    data[i]["lcb"],
    data[i]["ucb"],
    alpha=0.5,
    color="tab:blue",
)

j = 0
marker_sym = data["marker_symbols"][j]
ax.plot(
    data[i]["train_x_arr_add"][j].detach().numpy(),
    data[i]["train_y_arr_add"][j][:, 0].detach().numpy(),
    f"k{marker_sym}",
)

j = 1
marker_sym = data["marker_symbols"][j]
ax.plot(
    data[i]["train_x_arr_add"][j].detach().numpy(),
    data[i]["train_y_arr_add"][j][:, 0].detach().numpy(),
    f"k{marker_sym}",
)

# remove tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
# remove ticks
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylim(y_lim)

f.tight_layout(pad=0.5)
f.savefig(
    f"/home/manish/work/MPC_Dyn/figures/slides_fig3.{format}",
    format=format,
    dpi=300,
    transparent=True,
)


# ===============================================================================
# Plot 5

TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=10)
f, ax = plt.subplots(
    1, 1, figsize=(TEXTWIDTH * 0.5 / 2 + 0.25, TEXTWIDTH * 0.25 * 1 / 2)
)

i = 1

h_func = ax.plot(data["test_x"], data["test_y"], "k--")

h_mean = ax.plot(data["test_x"], data[i]["mean"], "tab:blue")
# h_samp = ax.plot(data["test_x"], data[i]["sample"], "tab:orange")

h_conf = ax.fill_between(
    data["test_x"],
    data[i]["lcb"],
    data[i]["ucb"],
    alpha=0.5,
    color="tab:blue",
)

j = 0
marker_sym = data["marker_symbols"][j]
ax.plot(
    data[0]["train_x_arr_add"][j].detach().numpy(),
    data[0]["train_y_arr_add"][j][:, 0].detach().numpy(),
    f"k{marker_sym}",
)

j = 1
marker_sym = data["marker_symbols"][j]
ax.plot(
    data[0]["train_x_arr_add"][j].detach().numpy(),
    data[0]["train_y_arr_add"][j][:, 0].detach().numpy(),
    f"k{marker_sym}",
)

# remove tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
# remove ticks
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylim(y_lim)

f.tight_layout(pad=0.5)
f.savefig(
    f"/home/manish/work/MPC_Dyn/figures/slides_fig4.{format}",
    format=format,
    dpi=300,
    transparent=True,
)

# ===============================================================================
# Plot 6
fig_num = 5

TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=10)
f, ax = plt.subplots(
    1, 1, figsize=(TEXTWIDTH * 0.5 / 2 + 0.25, TEXTWIDTH * 0.25 * 1 / 2)
)

i = 1

for x in data[i]["train_x_arr_add"][i + 1]:
    ax.axvline(
        x.numpy().flatten(),
        color="k",
        linestyle="solid",
        alpha=0.3,
        linewidth=3,
    )

h_func = ax.plot(data["test_x"], data["test_y"], "k--")

h_mean = ax.plot(data["test_x"], data[i]["mean"], "tab:blue")
# h_samp = ax.plot(data["test_x"], data[i]["sample"], "tab:orange")

h_conf = ax.fill_between(
    data["test_x"],
    data[i]["lcb"],
    data[i]["ucb"],
    alpha=0.5,
    color="tab:blue",
)

j = 0
marker_sym = data["marker_symbols"][j]
ax.plot(
    data[0]["train_x_arr_add"][j].detach().numpy(),
    data[0]["train_y_arr_add"][j][:, 0].detach().numpy(),
    f"k{marker_sym}",
)

j = 1
marker_sym = data["marker_symbols"][j]
ax.plot(
    data[0]["train_x_arr_add"][j].detach().numpy(),
    data[0]["train_y_arr_add"][j][:, 0].detach().numpy(),
    f"k{marker_sym}",
)

# remove tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
# remove ticks
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylim(y_lim)

f.tight_layout(pad=0.5)
f.savefig(
    f"/home/manish/work/MPC_Dyn/figures/slides_fig{fig_num}.{format}",
    format=format,
    dpi=300,
    transparent=True,
)


# ===============================================================================
# Plot 7
fig_num = 6

TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=10)
f, ax = plt.subplots(
    1, 1, figsize=(TEXTWIDTH * 0.5 / 2 + 0.25, TEXTWIDTH * 0.25 * 1 / 2)
)

i = 1

for x in data[i]["train_x_arr_add"][i + 1]:
    ax.axvline(
        x.numpy().flatten(),
        color="k",
        linestyle="solid",
        alpha=0.3,
        linewidth=3,
    )

h_func = ax.plot(data["test_x"], data["test_y"], "k--")

h_mean = ax.plot(data["test_x"], data[i]["mean"], "tab:blue")
# h_samp = ax.plot(data["test_x"], data[i]["sample"], "tab:orange")

h_conf = ax.fill_between(
    data["test_x"],
    data[i]["lcb"],
    data[i]["ucb"],
    alpha=0.5,
    color="tab:blue",
)

for j in range(3):
    marker_sym = data["marker_symbols"][j]
    ax.plot(
        data[0]["train_x_arr_add"][j].detach().numpy(),
        data[0]["train_y_arr_add"][j][:, 0].detach().numpy(),
        f"k{marker_sym}",
    )

# remove tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
# remove ticks
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylim(y_lim)

f.tight_layout(pad=0.5)
f.savefig(
    f"/home/manish/work/MPC_Dyn/figures/slides_fig{fig_num}.{format}",
    format=format,
    dpi=300,
    transparent=True,
)


# ===============================================================================
# Plot 8
fig_num = 7

TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=10)
f, ax = plt.subplots(
    1, 1, figsize=(TEXTWIDTH * 0.5 / 2 + 0.25, TEXTWIDTH * 0.25 * 1 / 2)
)

i = 2

# for x in data[i]["train_x_arr_add"][i + 1]:
#     ax.axvline(
#         x.numpy().flatten(),
#         color="k",
#         linestyle="solid",
#         alpha=0.3,
#         linewidth=3,
#     )

h_func = ax.plot(data["test_x"], data["test_y"], "k--")

h_mean = ax.plot(data["test_x"], data[i]["mean"], "tab:blue")
# h_samp = ax.plot(data["test_x"], data[i]["sample"], "tab:orange")

h_conf = ax.fill_between(
    data["test_x"],
    data[i]["lcb"],
    data[i]["ucb"],
    alpha=0.5,
    color="tab:blue",
)

for j in range(3):
    marker_sym = data["marker_symbols"][j]
    ax.plot(
        data[0]["train_x_arr_add"][j].detach().numpy(),
        data[0]["train_y_arr_add"][j][:, 0].detach().numpy(),
        f"k{marker_sym}",
    )

# remove tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
# remove ticks
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylim(y_lim)

f.tight_layout(pad=0.5)
f.savefig(
    f"/home/manish/work/MPC_Dyn/figures/slides_fig{fig_num}.{format}",
    format=format,
    dpi=300,
    transparent=True,
)


# ===============================================================================
# Plot 9
fig_num = 8

TEXTWIDTH = 16

set_figure_params(serif=True, fontsize=10)
f, ax = plt.subplots(
    1, 1, figsize=(TEXTWIDTH * 0.5 / 2 + 0.25, TEXTWIDTH * 0.25 * 1 / 2)
)

i = 2

h_func = ax.plot(data["test_x"], data["test_y"], "k--")

h_mean = ax.plot(data["test_x"], data[i]["mean"], "tab:blue")
h_samp = ax.plot(data["test_x"], data[i]["sample"], "tab:orange")

h_conf = ax.fill_between(
    data["test_x"],
    data[i]["lcb"],
    data[i]["ucb"],
    alpha=0.5,
    color="tab:blue",
)

for j in range(3):
    marker_sym = data["marker_symbols"][j]
    ax.plot(
        data[0]["train_x_arr_add"][j].detach().numpy(),
        data[0]["train_y_arr_add"][j][:, 0].detach().numpy(),
        f"k{marker_sym}",
    )

# remove tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
# remove ticks
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylim(y_lim)

f.tight_layout(pad=0.5)
f.savefig(
    f"/home/manish/work/MPC_Dyn/figures/slides_fig{fig_num}.{format}",
    format=format,
    dpi=300,
    transparent=True,
)

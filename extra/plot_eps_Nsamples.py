import matplotlib.pyplot as plt
import numpy as np
import sys, os
workspace_plotting_utils = "extra/plotting_tools"
sys.path.append(workspace_plotting_utils)

from plotting_tools.plotting_utilities import *

TEXTWIDTH = 16
# set_figure_params(serif=True, fontsize=14)
# f = plt.figure(figsize=(TEXTWIDTH * 0.5 + 2.75, TEXTWIDTH * 0.5 * 1 / 2))
f = plt.figure(figsize=(cm2inches(12.0), cm2inches(6.0)))
ax = f.axes
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.usetex"] =True
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times"]
# set fontsize
plt.rcParams["font.size"] = 14
# plt.rc('text', usetex=False)
# Data
eps = np.array([0.0011142622935513, 0.0008404190593132, 0.0006954232571821,
                0.0006009067950271, 0.0005326789384778, 0.0004763720895024])
samples = [200, 2000, 20000, 200000, 2000000, 20000000]

# Scale epsilon for pretty y-axis values
eps_scaled = eps * 1e3  # Now in units of 10^-3

# Plot
# plt.figure(figsize=(8, 5))
plt.plot(samples, eps_scaled, marker='o', linestyle='-', color='royalblue', label=r'$\epsilon$')

# Log scale for x-axis
plt.xscale('log')

# Labels and title
plt.xlabel('Number of Samples (log scale)')
plt.ylabel(r'$\epsilon \ ( \times 10^{-3} )$')
# plt.ylabel('$ϵ$')
# plt.title(r'Convergence of $\epsilon$ with Increasing Number of Samples', fontsize=14)
adapt_figure_size_from_axes(ax)
# Grid and layout
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tight_layout(pad=0.0)

# Show plot
plt.savefig(
    # f"eps{eps:.0e}.pdf",
    "epsilon_vs_Num_samples.pdf",
    format="pdf",
    dpi=300,
    transparent=True,
)
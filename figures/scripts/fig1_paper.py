# =============================================================================
# Farid Rafie | UWM Materials Science & Engineering | 2025–2026
#
# Research code accompanying:
# Crystallography-Informed Gaussian-Process Surrogate Modeling of Low-Energy Nitrogen Implantation in Diamond
#
# Author / maintainer:
#   Seyed Faridedin Rafie (Farid Rafie)
#   Materials Science and Engineering Department, University of Wisconsin-Milwaukee, Milwaukee, WI 53211, USA
#
# Script: fig1_paper.py
# Purpose: Generate the molecular-dynamics campaign and dataset schematic panels.
#
# Research note:
#   This script belongs to the reproducibility package for my doctoral
#   research on simulation-guided low-energy nitrogen implantation in diamond.
#   Numerical settings used for manuscript results are intentionally kept
#   explicit in the script rather than hidden behind external configuration.
# =============================================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# ============================================================
# 1) Paths
# ============================================================

HERE = Path(__file__).resolve().parent

OUT_BASE = HERE.parent / "rendered" / "Figure01_panel_b_energy_tilt_campaign"


# ============================================================
# 2) Data
# ============================================================

tilts = np.array([0, 3, 7, 10, 15, 18.43, 20, 22, 25, 30], dtype=float)
energies = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], dtype=float)

X, Y = np.meshgrid(tilts, energies)

# for coloring by energy
norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
cmap = plt.get_cmap("viridis")


# ============================================================
# 3) Figure style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # Portable repository default; manuscript typography may use Arial
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
})


# ============================================================
# 4) Figure and axes
# ============================================================

fig = plt.figure(figsize=(8.8, 4.4))
ax = fig.add_axes([0.09, 0.14, 0.60, 0.76])   # left main plot

# annotation area to the right
ax_text = fig.add_axes([0.73, 0.14, 0.23, 0.76])
ax_text.axis("off")


# ============================================================
# 5) Grid lines
# ============================================================

for y in energies:
    ax.hlines(
        y,
        xmin=tilts.min(),
        xmax=tilts.max(),
        colors="#BFBFBF",
        linestyles="--",
        linewidth=0.8,
        zorder=1,
    )

for x in tilts:
    ax.vlines(
        x,
        ymin=energies.min(),
        ymax=energies.max(),
        colors="#D3D3D3",
        linestyles="--",
        linewidth=0.7,
        zorder=1,
    )


# ============================================================
# 6) Scatter points
# ============================================================

for e in energies:
    mask = np.isclose(Y, e)
    ax.scatter(
        X[mask],
        Y[mask],
        s=68,
        c=[cmap(norm(e))],
        edgecolors="#303030",
        linewidths=0.6,
        zorder=3,
    )


# ============================================================
# 7) Axes formatting
# ============================================================

ax.set_xlim(-1.5, 31.5)
ax.set_ylim(0.2, 5.25)

ax.set_xticks(tilts)
ax.set_xticklabels(["0", "3", "7", "10", "15", "18.43", "20", "22", "25", "30"])
ax.set_yticks(energies)

ax.set_xlabel(r"Tilt angle, $\theta$ (°)")
ax.set_ylabel(r"Implantation energy, $E$ (keV)")
ax.set_title("Energy–tilt simulation campaign", pad=12, weight="bold")

# panel label
ax.text(
    -0.18, 1.05, "(b)",
    transform=ax.transAxes,
    fontsize=17,
    fontweight="bold",
    va="bottom",
    ha="left",
)

# ticks on all sides
ax.tick_params(direction="out", top=False, right=False)

# spine styling
for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(1.0)


# ============================================================
# 8) Right-side summary text
# ============================================================

# dashed separator
ax_text.plot(
    [0.02, 0.02], [0.0, 1.0],
    transform=ax_text.transAxes,
    color="#9E9E9E",
    linestyle=(0, (4, 3)),
    linewidth=1.2,
)

summary_lines = [
    ("100 conditions", True),
    ("", False),
    ("10 replicates/", False),
    ("condition", False),
    ("", False),
    ("1,000 replicate", True),
    ("trajectories", False),
    ("", False),
    ("10,000", True),
    ("implantation", False),
    ("events", False),
]

y0 = 0.86
dy = 0.08
for line, bold in summary_lines:
    ax_text.text(
        0.14,
        y0,
        line,
        transform=ax_text.transAxes,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold" if bold else "normal",
        color="black",
    )
    y0 -= dy


# ============================================================
# 9) Save high-quality outputs
# ============================================================

fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(OUT_BASE.with_name(OUT_BASE.name + "_600dpi.png"), dpi=600, bbox_inches="tight")

plt.close(fig)

print("Saved:")
print(OUT_BASE.with_suffix(".pdf"))
print(OUT_BASE.with_suffix(".svg"))
print(OUT_BASE.with_name(OUT_BASE.name + "_600dpi.png"))
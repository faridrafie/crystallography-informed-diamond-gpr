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
# Script: Fig0_paper.py
# Purpose: Generate the manuscript workflow overview figure.
#
# Research note:
#   This script belongs to the reproducibility package for my doctoral
#   research on simulation-guided low-energy nitrogen implantation in diamond.
#   Numerical settings used for manuscript results are intentionally kept
#   explicit in the script rather than hidden behind external configuration.
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

# ============================================================
# Global style
# ============================================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 8.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.5)
ax.axis("off")

# ============================================================
# Colors
# ============================================================
blue_fill   = "#EAF4FF"
blue_edge   = "#2C6DA4"

green_fill  = "#EEF8EC"
green_edge  = "#3A8A3A"

orange_fill = "#FFF1DE"
orange_edge = "#C76A16"

purple_fill = "#F1EAFE"
purple_edge = "#6941A5"

gray_fill   = "#F4F4F4"
gray_edge   = "#444444"

arrow_color = "#333333"


# ============================================================
# Helper functions
# ============================================================
def rounded_box(x, y, w, h, facecolor, edgecolor,
                title, lines=None,
                title_size=11, text_size=9.5,
                linewidth=1.6):
    
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=linewidth,
        facecolor=facecolor,
        edgecolor=edgecolor
    )
    ax.add_patch(box)

    ax.text(
        x + w/2, y + h - 0.28,
        title,
        ha="center", va="top",
        fontsize=title_size,
        fontweight="bold"
    )

    if lines:
        ax.text(
            x + w/2,
            y + h/2 - 0.12,
            "\n".join(lines),
            ha="center", va="center",
            fontsize=text_size,
            linespacing=1.35
        )

    return box


def diamond(xc, yc, w, h, facecolor, edgecolor,
            title, lines=None,
            title_size=11, text_size=9):
    
    pts = [
        (xc, yc + h/2),
        (xc + w/2, yc),
        (xc, yc - h/2),
        (xc - w/2, yc)
    ]

    poly = Polygon(
        pts,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6
    )
    ax.add_patch(poly)

    ax.text(
        xc, yc + 0.23,
        title,
        ha="center", va="center",
        fontsize=title_size,
        fontweight="bold"
    )

    if lines:
        ax.text(
            xc, yc - 0.18,
            "\n".join(lines),
            ha="center", va="top",
            fontsize=text_size,
            linespacing=1.25
        )

    return poly


def arrow(x1, y1, x2, y2, style="-|>", lw=1.7):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=arrow_color,
        connectionstyle="arc3"
    )
    ax.add_patch(a)


# ============================================================
# Title
# ============================================================
ax.text(
    7, 8.0,
    "Surrogate-modeling workflow",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold"
)


# ============================================================
# 1. MD dataset
# ============================================================
rounded_box(
    0.25, 4.15, 2.05, 2.45,
    blue_fill, blue_edge,
    "MD implantation\ndataset",
    [
        "Energy: 0.5–5.0 keV",
        "Tilt: 0–30°",
        "100 conditions",
        "10 replicates / condition"
    ],
    title_size=11.5,
    text_size=9.5
)


# ============================================================
# 2. Condition-averaged responses
# ============================================================
rounded_box(
    2.65, 4.05, 2.35, 2.65,
    green_fill, green_edge,
    "Condition-averaged\nresponses",
    [
        "Median implantation depth",
        "Shallow fraction (0–5 nm)",
        "Near-surface sp² fraction",
        "(0–2 nm)",
        "Vacancy count"
    ],
    title_size=11.2,
    text_size=9.2
)


# ============================================================
# 3A. Model A
# ============================================================
rounded_box(
    5.45, 5.45, 1.9, 1.15,
    orange_fill, orange_edge,
    "Model A inputs",
    ["[E, θ]"],
    title_size=10.8,
    text_size=10
)


# ============================================================
# 3B. Model B
# ============================================================
rounded_box(
    5.45, 3.75, 1.9, 1.15,
    orange_fill, orange_edge,
    "Model B inputs",
    ["[E, Atlas descriptors]"],
    title_size=10.8,
    text_size=9.3
)


# ============================================================
# 4. Gaussian-process regression
# ============================================================
rounded_box(
    7.75, 4.55, 2.15, 1.55,
    purple_fill, purple_edge,
    "Gaussian-process\nregression",
    ["Surrogate models"],
    title_size=11,
    text_size=9.5
)


# ============================================================
# Validation diamond
# ============================================================
diamond(
    8.82, 2.45,
    2.5, 2.0,
    "#F5F0FF", purple_edge,
    "Validation",
    [
        "Crossed energy–angle",
        "holdout",
        "LOAO / LOEO (depth)",
        "PCHIP benchmark"
    ],
    title_size=11,
    text_size=8.8
)


# ============================================================
# 5. Outputs
# ============================================================
rounded_box(
    10.25, 4.2, 2.25, 2.2,
    blue_fill, blue_edge,
    "Outputs",
    [
        "Continuous energy–tilt",
        "response maps",
        "Prediction uncertainty",
        "Model A vs Model B",
        "comparison"
    ],
    title_size=11.2,
    text_size=9.2
)


# ============================================================
# Interpretation diamond
# ============================================================
diamond(
    13.15, 5.3,
    1.7, 1.7,
    gray_fill, gray_edge,
    "Interpretation",
    [
        "Response-dependent",
        "performance"
    ],
    title_size=10.5,
    text_size=8.7
)


# ============================================================
# Arrows
# ============================================================

# Dataset -> Responses
arrow(2.30, 5.35, 2.65, 5.35)

# Responses -> split point
arrow(5.00, 5.35, 5.28, 5.35)

# Split to Model A
arrow(5.28, 5.35, 5.28, 6.02, style="-")
arrow(5.28, 6.02, 5.45, 6.02)

# Split to Model B
arrow(5.28, 5.35, 5.28, 4.32, style="-")
arrow(5.28, 4.32, 5.45, 4.32)

# Model A -> GPR
arrow(7.35, 6.02, 7.55, 6.02, style="-")
arrow(7.55, 6.02, 7.55, 5.50, style="-")
arrow(7.55, 5.50, 7.75, 5.50)

# Model B -> GPR
arrow(7.35, 4.32, 7.55, 4.32, style="-")
arrow(7.55, 4.32, 7.55, 5.05, style="-")
arrow(7.55, 5.05, 7.75, 5.05)

# GPR -> Outputs
arrow(9.90, 5.32, 10.25, 5.32)

# Outputs -> Interpretation
arrow(12.50, 5.3, 12.30 + 0.35, 5.3)

# Validation <-> GPR
arrow(8.82, 3.45, 8.82, 4.55)
arrow(9.03, 4.55, 9.03, 3.45)


# ============================================================
# Save
# ============================================================
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE.parent / "rendered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "surrogate_modeling_workflow.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "surrogate_modeling_workflow.pdf",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "surrogate_modeling_workflow.svg",
    bbox_inches="tight"
)

plt.close(fig)
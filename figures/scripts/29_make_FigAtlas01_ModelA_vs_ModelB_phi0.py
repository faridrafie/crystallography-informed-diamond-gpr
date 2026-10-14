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
# Script: 29_make_FigAtlas01_ModelA_vs_ModelB_phi0.py
# Purpose: Generate the fixed-azimuth aggregate Model A versus Model B comparison figure.
#
# Research note:
#   This script belongs to the reproducibility package for my doctoral
#   research on simulation-guided low-energy nitrogen implantation in diamond.
#   Numerical settings used for manuscript results are intentionally kept
#   explicit in the script rather than hidden behind external configuration.
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import (
    apply_paper_style,
    add_panel_label,
    save_paper_figure,
    DOUBLE_COLUMN_IN,
    FONT_ANNOTATION,
)


# ============================================================
# 1. Frozen manuscript style
# ============================================================

apply_paper_style()


# ============================================================
# 2. Paths
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

SUMMARY_PATH = (
    ROOT
    / "analysis"
    / "atlas_model"
    / "phi0_modelA_vs_modelB_summary.csv"
)

RENDER_DIR = HERE.parent / "rendered"
SOURCE_DIR = HERE.parent / "source_data"
OUTPUT_BASE = RENDER_DIR / "Figure_Atlas01_ModelA_vs_ModelB_Phi0"
OUTPUT_AUDIT = SOURCE_DIR / "Figure_Atlas01_ModelA_vs_ModelB_Phi0_Audit.csv"


# ============================================================
# 3. Atlas visual identity
# ============================================================

MODEL_A_COLOR = "#6F6F6F"
MODEL_B_COLOR = "#7B3294"

WORSE_COLOR = "#B33A1B"


# ============================================================
# 4. Display definitions
# ============================================================

TARGET_INFO = {

    "median_depth": {
        "panel": "(a)",
        "title": "Median implantation depth",
        "ylabel": "Prediction error (nm)",
        "digits": 3,
    },

    "shallow_fraction": {
        "panel": "(b)",
        "title": "Shallow fraction (0–5 nm)",
        "ylabel": "Prediction error",
        "digits": 3,
    },

    "near_surface_sp2": {
        "panel": "(c)",
        "title": "Near-surface sp² fraction (0–2 nm)",
        "ylabel": "Prediction error",
        "digits": 4,
    },

    "vacancy_count": {
        "panel": "(d)",
        "title": "Vacancy count",
        "ylabel": "Prediction error (count)",
        "digits": 2,
    },
}


TARGET_ORDER = [
    "median_depth",
    "shallow_fraction",
    "near_surface_sp2",
    "vacancy_count",
]


# ============================================================
# 5. Load frozen φ=0 comparison
# ============================================================

df = pd.read_csv(
    SUMMARY_PATH
)


required_columns = [
    "target_label",
    "n_test",

    "modelA_MAE",
    "modelB_MAE",

    "modelA_RMSE",
    "modelB_RMSE",

    "modelA_R2",
    "modelB_R2",

    "modelB_MAE_reduction_percent",
    "modelB_RMSE_reduction_percent",

    "modelA_point_wins",
    "modelB_point_wins",

    "modelA_physical_violations",
    "modelB_physical_violations",
]


missing = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


if len(df) != 4:
    raise ValueError(
        f"Expected exactly 4 targets; "
        f"found {len(df)}."
    )


if set(df["target_label"]) != set(TARGET_ORDER):
    raise ValueError(
        "Unexpected target labels in frozen summary."
    )


if not np.all(
    df["n_test"].to_numpy() == 64
):
    raise ValueError(
        "All targets must use the same 64-point crossed holdout."
    )


# deterministic order
df = (
    df
    .set_index("target_label")
    .loc[TARGET_ORDER]
    .reset_index()
)


print("\nQC PASSED")
print(
    "Frozen φ=0 Model A vs Model B benchmark:"
)
print(
    "4 targets × 64 common crossed-holdout conditions."
)


# ============================================================
# 6. Save exact audit table
# ============================================================

df.to_csv(
    OUTPUT_AUDIT,
    index=False,
)


# ============================================================
# 7. Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        DOUBLE_COLUMN_IN,
        5.85,
    ),
)


axes = axes.ravel()

bar_width = 0.34

x = np.array(
    [
        0.0,
        1.0,
    ]
)


for ax, (_, row) in zip(
    axes,
    df.iterrows(),
):

    target_key = row[
        "target_label"
    ]

    info = TARGET_INFO[
        target_key
    ]


    # ========================================================
    # Metric values
    # ========================================================

    modelA_values = np.array(
        [
            row["modelA_MAE"],
            row["modelA_RMSE"],
        ],
        dtype=float,
    )


    modelB_values = np.array(
        [
            row["modelB_MAE"],
            row["modelB_RMSE"],
        ],
        dtype=float,
    )


    # ========================================================
    # Bars
    # ========================================================

    bars_A = ax.bar(
        x - bar_width / 2,
        modelA_values,

        width=bar_width,

        color=MODEL_A_COLOR,

        edgecolor="black",

        linewidth=0.5,

        label="Model A",

        zorder=3,
    )


    bars_B = ax.bar(
        x + bar_width / 2,
        modelB_values,

        width=bar_width,

        color=MODEL_B_COLOR,

        edgecolor="black",

        linewidth=0.5,

        label="Model B",

        zorder=3,
    )


    # ========================================================
    # Axes
    # ========================================================

    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        [
            "MAE",
            "RMSE",
        ]
    )


    ax.set_ylabel(
        info["ylabel"]
    )


    ax.set_title(
        info["title"],
        pad=6,
    )


    add_panel_label(
        ax,
        info["panel"],
    )


    ax.grid(
        axis="y",

        linestyle="--",

        linewidth=0.45,

        alpha=0.18,

        zorder=0,
    )


    # ========================================================
    # Y limits
    # ========================================================

    ymax = float(
        max(
            np.max(modelA_values),
            np.max(modelB_values),
        )
    )


    ax.set_ylim(
        0.0,
        ymax * 1.36,
    )


    y_offset = (
        ymax
        *
        0.035
    )


    # ========================================================
    # Values above bars
    # ========================================================

    digits = info[
        "digits"
    ]


    for bar, value in zip(
        bars_A,
        modelA_values,
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            value
            +
            y_offset,

            f"{value:.{digits}f}",

            ha="center",
            va="bottom",

            fontsize=7.2,

            color=MODEL_A_COLOR,
        )


    for bar, value in zip(
        bars_B,
        modelB_values,
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            value
            +
            y_offset,

            f"{value:.{digits}f}",

            ha="center",
            va="bottom",

            fontsize=7.2,

            color=MODEL_B_COLOR,
        )


    # ========================================================
    # MAE change
    # ========================================================

    mae_reduction = float(
        row[
            "modelB_MAE_reduction_percent"
        ]
    )


    if mae_reduction > 0:

        change_text = (
            f"B MAE: "
            f"{mae_reduction:.1f}% lower"
        )

        change_color = (
            MODEL_B_COLOR
        )

    elif mae_reduction < 0:

        change_text = (
            f"B MAE: "
            f"{abs(mae_reduction):.1f}% higher"
        )

        change_color = (
            WORSE_COLOR
        )

    else:

        change_text = (
            "B MAE: unchanged"
        )

        change_color = (
            MODEL_A_COLOR
        )


    # ========================================================
    # Combined annotation:
    # R² + MAE change
    # ========================================================

    ax.text(
        0.04,
        0.94,

        (
            f"$R^2$ (A) = "
            f"{row['modelA_R2']:.3f}"
            "\n"
            f"$R^2$ (B) = "
            f"{row['modelB_R2']:.3f}"
        ),

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontsize=FONT_ANNOTATION,

        bbox={
            "facecolor":
                "white",

            "edgecolor":
                "none",

            "alpha":
                0.88,

            "pad":
                2.0,
        },

        zorder=5,
    )


    ax.text(
        0.04,
        0.74,

        change_text,

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontsize=7.8,

        fontweight="bold",

        color=change_color,

        zorder=5,
    )


# ============================================================
# 8. Shared legend
# ============================================================

handles = [

    plt.Rectangle(
        (0, 0),
        1,
        1,

        facecolor=MODEL_A_COLOR,

        edgecolor="black",

        linewidth=0.5,
    ),

    plt.Rectangle(
        (0, 0),
        1,
        1,

        facecolor=MODEL_B_COLOR,

        edgecolor="black",

        linewidth=0.5,
    ),
]


labels = [
    "Model A: E + raw tilt",
    "Model B: E + Atlas geometry",
]


fig.legend(
    handles,
    labels,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.995,
    ),

    ncol=2,

    frameon=False,

    columnspacing=1.7,

    handlelength=1.3,

    handletextpad=0.55,
)


# ============================================================
# 9. Layout
# ============================================================

fig.subplots_adjust(
    left=0.095,
    right=0.985,

    bottom=0.09,
    top=0.895,

    wspace=0.30,
    hspace=0.36,
)


# ============================================================
# 10. Save
# ============================================================

save_paper_figure(
    fig,
    OUTPUT_BASE,
)


# ============================================================
# 11. Console summary
# ============================================================

print("\n")
print("=" * 150)

print(
    "FIGURE ATLAS-01 — FINAL MODEL A vs MODEL B "
    "ON FROZEN φ=0 BENCHMARK"
)

print("=" * 150)


print(
    df[
        [
            "target_label",

            "modelA_MAE",
            "modelB_MAE",

            "modelB_MAE_reduction_percent",

            "modelA_RMSE",
            "modelB_RMSE",

            "modelB_RMSE_reduction_percent",

            "modelA_R2",
            "modelB_R2",

            "modelA_point_wins",
            "modelB_point_wins",

            "modelA_physical_violations",
            "modelB_physical_violations",
        ]
    ].to_string(
        index=False
    )
)


print("\nSaved:")
print(
    OUTPUT_AUDIT
)


print("\nInterpretation:")

print(
    "Model A uses implantation energy + raw tilt."
)

print(
    "Model B replaces the raw orientation coordinate "
    "with the frozen crystallographic Atlas representation."
)

print(
    "No feature modification, target-driven retuning, "
    "or metric recomputation was performed by this figure."
)


plt.close(
    fig
)
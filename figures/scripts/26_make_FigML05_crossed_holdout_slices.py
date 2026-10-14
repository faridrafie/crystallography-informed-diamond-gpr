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
# Script: 26_make_FigML05_crossed_holdout_slices.py
# Purpose: Generate representative crossed-holdout response slices.
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

from matplotlib.lines import Line2D

from paper_plot_style import (
    apply_paper_style,
    add_panel_label,
    save_paper_figure,
    DOUBLE_COLUMN_IN,
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

DATA_DIR = (
    ROOT
    / "analysis"
    / "baseline"
    / "model_comparison"
)

RENDER_DIR = HERE.parent / "rendered"
SOURCE_DIR = HERE.parent / "source_data"
OUTPUT_BASE = RENDER_DIR / "Figure_ML05_CrossedHoldout_Response_Slices"
OUTPUT_DATA = SOURCE_DIR / "Figure_ML05_CrossedHoldout_Response_Slices_Data.csv"


# ============================================================
# 3. Representative energies
#
# Low / middle / high interior energies.
# Fixed before inspecting target-specific performance.
# ============================================================

SLICE_ENERGIES = [
    1.5,
    3.0,
    4.5,
]


# ============================================================
# 4. Energy color identity
#
# Color encodes ENERGY.
# Marker shape encodes METHOD.
# ============================================================

ENERGY_COLORS = {
    1.5: "#0072B2",
    3.0: "#009E73",
    4.5: "#D55E00",
}


# ============================================================
# 5. Target definitions
# ============================================================

TARGETS = [

    {
        "panel": "(a)",
        "key": "depth",

        "title":
            "Median implantation depth",

        "file":
            "depth_gpr_vs_pchip_crossed_predictions.csv",

        "truth":
            "true_depth_nm",

        "gpr":
            "gpr_prediction_nm",

        "pchip":
            "pchip_prediction_nm",

        "gpr_std":
            "gpr_prediction_std_nm",

        "ylabel":
            "Median depth (nm)",
    },

    {
        "panel": "(b)",
        "key": "shallow",

        "title":
            "Shallow fraction (0–5 nm)",

        "file":
            "shallow_gpr_vs_pchip_crossed_predictions.csv",

        "truth":
            "true_shallow_fraction",

        "gpr":
            "gpr_prediction",

        "pchip":
            "pchip_prediction",

        "gpr_std":
            "gpr_prediction_std",

        "ylabel":
            "Shallow fraction",
    },

    {
        "panel": "(c)",
        "key": "sp2",

        "title":
            "Near-surface sp² fraction (0–2 nm)",

        "file":
            "sp2_gpr_vs_pchip_crossed_predictions.csv",

        "truth":
            "true_sp2_fraction",

        "gpr":
            "gpr_prediction",

        "pchip":
            "pchip_prediction",

        "gpr_std":
            "gpr_prediction_std",

        "ylabel":
            "Near-surface sp² fraction",
    },

    {
        "panel": "(d)",
        "key": "vacancy",

        "title":
            "Vacancy count",

        "file":
            "vacancy_gpr_vs_pchip_crossed_predictions.csv",

        "truth":
            "true_vac_count",

        "gpr":
            "gpr_prediction",

        "pchip":
            "pchip_prediction",

        "gpr_std":
            "gpr_prediction_std",

        "ylabel":
            "Vacancy count",
    },
]


# ============================================================
# 6. Load + QC
# ============================================================

loaded = {}

reference_conditions = None


for target in TARGETS:

    path = (
        DATA_DIR
        /
        target["file"]
    )


    if not path.exists():

        raise FileNotFoundError(
            path
        )


    df = pd.read_csv(
        path
    )


    if len(df) != 64:

        raise ValueError(
            f"{target['file']} contains "
            f"{len(df)} rows instead of 64."
        )


    required_columns = [

        "held_energy_keV",
        "held_angle_deg",

        target["truth"],
        target["gpr"],
        target["pchip"],
        target["gpr_std"],
    ]


    missing = [

        column

        for column
        in required_columns

        if column
        not in df.columns
    ]


    if missing:

        raise ValueError(
            f"Missing columns in "
            f"{target['file']}: {missing}"
        )


    conditions = (
        df[
            [
                "held_energy_keV",
                "held_angle_deg",
            ]
        ]
        .sort_values(
            [
                "held_energy_keV",
                "held_angle_deg",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    if reference_conditions is None:

        reference_conditions = (
            conditions.copy()
        )

    else:

        if not conditions.equals(
            reference_conditions
        ):

            raise ValueError(
                "The four target files do not "
                "contain identical crossed-holdout conditions."
            )


    available_energies = np.sort(
        df[
            "held_energy_keV"
        ].unique()
    )


    for energy in SLICE_ENERGIES:

        if not np.any(
            np.isclose(
                available_energies,
                energy,
            )
        ):

            raise ValueError(
                f"{energy} keV is absent from "
                f"the common crossed-holdout set."
            )


    loaded[
        target["file"]
    ] = df


print("\nQC PASSED")

print(
    "All four targets use the same "
    "64 common crossed-holdout conditions."
)

print(
    "Representative energies:",
    SLICE_ENERGIES,
)


# ============================================================
# 7. Build clean plotting / audit table
# ============================================================

plot_rows = []


for target in TARGETS:

    df = loaded[
        target["file"]
    ]


    subset = df[
        df[
            "held_energy_keV"
        ].isin(
            SLICE_ENERGIES
        )
    ].copy()


    # 3 energies × 8 common interior angles.
    if len(subset) != 24:

        raise ValueError(
            f"Expected 24 selected crossed predictions "
            f"for {target['key']}; "
            f"found {len(subset)}."
        )


    for _, row in subset.iterrows():

        plot_rows.append(
            {
                "target":
                    target["key"],

                "energy_keV":
                    float(
                        row[
                            "held_energy_keV"
                        ]
                    ),

                "tilt_deg":
                    float(
                        row[
                            "held_angle_deg"
                        ]
                    ),

                "md_truth":
                    float(
                        row[
                            target["truth"]
                        ]
                    ),

                "gpr_prediction":
                    float(
                        row[
                            target["gpr"]
                        ]
                    ),

                "gpr_prediction_std":
                    float(
                        row[
                            target["gpr_std"]
                        ]
                    ),

                "pchip_prediction":
                    float(
                        row[
                            target["pchip"]
                        ]
                    ),
            }
        )


plot_df = pd.DataFrame(
    plot_rows
)


plot_df = (
    plot_df
    .sort_values(
        [
            "target",
            "energy_keV",
            "tilt_deg",
        ]
    )
    .reset_index(
        drop=True
    )
)


plot_df.to_csv(
    OUTPUT_DATA,
    index=False,
)


# ============================================================
# 8. Figure geometry
# ============================================================

fig, axes = plt.subplots(
    2,
    2,

    figsize=(
        DOUBLE_COLUMN_IN,
        5.75,
    ),
)


axes = axes.ravel()


# ============================================================
# 9. Method offsets
#
# Important:
# Methods are horizontally separated only for visibility.
#
# These offsets do NOT alter the actual held-out angle.
# ============================================================

X_OFFSET_MD = -0.32
X_OFFSET_GPR = 0.00
X_OFFSET_PCHIP = 0.32


# ============================================================
# 10. Plot panels
# ============================================================

for ax, target in zip(
    axes,
    TARGETS,
):

    target_df = plot_df[
        plot_df[
            "target"
        ]
        ==
        target["key"]
    ]


    all_y = []


    # ========================================================
    # Three representative energies
    # ========================================================

    for energy in SLICE_ENERGIES:

        color = (
            ENERGY_COLORS[
                energy
            ]
        )


        sub = (
            target_df[
                np.isclose(
                    target_df[
                        "energy_keV"
                    ],
                    energy,
                )
            ]
            .sort_values(
                "tilt_deg"
            )
        )


        x = sub[
            "tilt_deg"
        ].to_numpy(
            dtype=float
        )


        truth = sub[
            "md_truth"
        ].to_numpy(
            dtype=float
        )


        gpr = sub[
            "gpr_prediction"
        ].to_numpy(
            dtype=float
        )


        gpr_std = sub[
            "gpr_prediction_std"
        ].to_numpy(
            dtype=float
        )


        pchip = sub[
            "pchip_prediction"
        ].to_numpy(
            dtype=float
        )


        # ====================================================
        # MD truth
        #
        # Open circles.
        # ====================================================

        ax.scatter(
            x + X_OFFSET_MD,
            truth,

            s=25,

            marker="o",

            facecolor="white",

            edgecolor=color,

            linewidth=1.0,

            zorder=5,
        )


        # ====================================================
        # GPR crossed-holdout prediction
        #
        # Filled circles + ±1 predictive SD.
        #
        # Every point comes from an independent
        # crossed-holdout fold.
        # ====================================================

        ax.errorbar(
            x + X_OFFSET_GPR,
            gpr,

            yerr=gpr_std,

            fmt="o",

            markersize=4.8,

            markerfacecolor=color,

            markeredgecolor="black",

            markeredgewidth=0.45,

            ecolor=color,

            elinewidth=0.75,

            capsize=1.7,

            capthick=0.65,

            alpha=0.62,

            zorder=4,
        )


        # ====================================================
        # PCHIP crossed-holdout prediction
        #
        # Filled squares.
        # ====================================================

        ax.scatter(
            x + X_OFFSET_PCHIP,
            pchip,

            s=27,

            marker="s",

            facecolor=color,

            edgecolor="black",

            linewidth=0.45,

            alpha=0.86,

            zorder=4,
        )


        # ====================================================
        # Y-range bookkeeping
        # ====================================================

        all_y.extend(
            truth.tolist()
        )


        all_y.extend(
            gpr.tolist()
        )


        all_y.extend(
            pchip.tolist()
        )


        all_y.extend(
            (
                gpr
                -
                gpr_std
            ).tolist()
        )


        all_y.extend(
            (
                gpr
                +
                gpr_std
            ).tolist()
        )


    # ========================================================
    # Axes
    # ========================================================

    ax.set_xlim(
        2.0,
        26.0,
    )


    # --------------------------------------------------------
    # Reduced tick labels for manuscript readability.
    #
    # Predictions at 18.43° and 22° remain plotted at their
    # exact positions but are intentionally not separately
    # labelled on the axis.
    # --------------------------------------------------------

    ax.set_xticks(
        [
            3,
            7,
            10,
            15,
            20,
            25,
        ]
    )


    ax.set_xticklabels(
        [
            "3",
            "7",
            "10",
            "15",
            "20",
            "25",
        ]
    )


    ax.set_xlabel(
        "Held-out tilt angle (°)"
    )


    ax.set_ylabel(
        target["ylabel"]
    )


    ax.set_title(
        target["title"],
        pad=6,
    )


    add_panel_label(
        ax,
        target["panel"],
    )


    # ========================================================
    # Very subtle horizontal guides
    # ========================================================

    ax.grid(
        axis="y",

        linestyle="--",

        linewidth=0.45,

        alpha=0.16,

        zorder=0,
    )


    # ========================================================
    # Clean data-driven y limits
    # ========================================================

    all_y = np.asarray(
        all_y,
        dtype=float,
    )


    ymin = float(
        np.nanmin(
            all_y
        )
    )


    ymax = float(
        np.nanmax(
            all_y
        )
    )


    span = (
        ymax
        -
        ymin
    )


    if span <= 0:

        span = 1.0


    lower = (
        ymin
        -
        0.07
        *
        span
    )


    upper = (
        ymax
        +
        0.08
        *
        span
    )


    # All four physical targets are non-negative.
    lower = max(
        0.0,
        lower,
    )


    ax.set_ylim(
        lower,
        upper,
    )


# ============================================================
# 11. Method legend
#
# Shape = model / truth type
# ============================================================

METHOD_HANDLES = [

    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="None",

        markerfacecolor="white",

        markeredgecolor="black",

        markeredgewidth=1.0,

        markersize=5.5,

        label="MD truth",
    ),


    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="None",

        markerfacecolor="black",

        markeredgecolor="black",

        markersize=5.0,

        label="GPR",
    ),


    Line2D(
        [0],
        [0],

        marker="s",

        linestyle="None",

        markerfacecolor="black",

        markeredgecolor="black",

        markersize=5.0,

        label="PCHIP",
    ),


    Line2D(
        [0],
        [0],

        marker="_",

        linestyle="-",

        color="black",

        linewidth=0.75,

        markersize=7,

        alpha=0.60,

        label="GPR ±1 predictive SD",
    ),
]


# ============================================================
# 12. Energy legend
#
# Color = implantation energy
# ============================================================

ENERGY_HANDLES = [

    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="None",

        markerfacecolor=
            ENERGY_COLORS[
                1.5
            ],

        markeredgecolor="none",

        markersize=6,

        label="1.5 keV",
    ),


    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="None",

        markerfacecolor=
            ENERGY_COLORS[
                3.0
            ],

        markeredgecolor="none",

        markersize=6,

        label="3.0 keV",
    ),


    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="None",

        markerfacecolor=
            ENERGY_COLORS[
                4.5
            ],

        markeredgecolor="none",

        markersize=6,

        label="4.5 keV",
    ),
]


# ============================================================
# 13. Shared legends
# ============================================================

legend_methods = fig.legend(
    handles=METHOD_HANDLES,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.995,
    ),

    ncol=4,

    frameon=False,

    handlelength=1.4,

    columnspacing=1.45,

    handletextpad=0.50,
)


fig.add_artist(
    legend_methods
)


fig.legend(
    handles=ENERGY_HANDLES,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.957,
    ),

    ncol=3,

    frameon=False,

    handlelength=1.0,

    columnspacing=1.65,

    handletextpad=0.50,
)


# ============================================================
# 14. Layout
# ============================================================

fig.subplots_adjust(
    left=0.09,
    right=0.985,

    bottom=0.09,
    top=0.865,

    wspace=0.27,
    hspace=0.34,
)


# ============================================================
# 15. Save
# ============================================================

save_paper_figure(
    fig,
    OUTPUT_BASE,
)


# ============================================================
# 16. Console summary
# ============================================================

shown_angles = sorted(
    plot_df[
        "tilt_deg"
    ].unique()
)


print("\n")
print("=" * 120)

print(
    "FIGURE ML-05 — FINAL MARKER-ONLY "
    "CROSSED-HOLDOUT RESPONSE SLICES"
)

print("=" * 120)


print(
    "Shown conditions per target:",
    24,
)


print(
    "Representative energies:",
    SLICE_ENERGIES,
)


print(
    "All held-out angles:",
    shown_angles,
)


print(
    "\nDisplayed x-axis tick labels:"
)

print(
    [
        3,
        7,
        10,
        15,
        20,
        25,
    ]
)


print("\nSaved:")

print(
    OUTPUT_DATA
)


print("\nIMPORTANT:")

print(
    "Every GPR and PCHIP marker is an "
    "independent crossed-holdout prediction."
)

print(
    "No prediction lines connect independent folds."
)

print(
    "GPR error bars show the stored "
    "±1 predictive standard deviation."
)

print(
    "The 18.43° and 22° conditions remain plotted "
    "at their exact x positions but are not separately "
    "labelled on the x axis."
)


plt.close(
    fig
)
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
# Script: 27_make_FigML06_GPR_PCHIP_error_advantage_maps.py
# Purpose: Generate condition-resolved GPR-versus-PCHIP absolute-error advantage maps.
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

from matplotlib.colors import (
    LinearSegmentedColormap,
    TwoSlopeNorm,
)
from matplotlib.ticker import MaxNLocator

from paper_plot_style import (
    apply_paper_style,
    add_panel_label,
    save_paper_figure,
    DOUBLE_COLUMN_IN,
    COLORS,
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
OUTPUT_BASE = RENDER_DIR / "Figure_ML06_GPR_PCHIP_Error_Advantage_Maps"
OUTPUT_LONG = SOURCE_DIR / "Figure_ML06_GPR_PCHIP_Error_Advantage_Data.csv"
OUTPUT_SUMMARY = SOURCE_DIR / "Figure_ML06_GPR_PCHIP_Error_Advantage_Summary.csv"


# ============================================================
# 3. Target definitions
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

        "cbar_label":
            "Δ|error| (nm)",
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

        "cbar_label":
            "Δ|error|",
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

        "cbar_label":
            "Δ|error|",
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

        "cbar_label":
            "Δ|error| (count)",
    },
]


# ============================================================
# 4. Diverging colormap
#
# Negative  -> PCHIP lower absolute error -> orange
# Zero      -> equal absolute error       -> white
# Positive  -> GPR lower absolute error   -> blue
# ============================================================

ERROR_CMAP = LinearSegmentedColormap.from_list(
    "PCHIP_to_GPR",
    [
        COLORS["pchip"],
        "#FFFFFF",
        COLORS["gpr"],
    ],
    N=256,
)


# ============================================================
# 5. Load and validate frozen benchmark files
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
            f"{len(df)} rows; expected 64."
        )


    required = [
        "held_energy_keV",
        "held_angle_deg",
        target["truth"],
        target["gpr"],
        target["pchip"],
    ]


    missing = [
        column
        for column in required
        if column not in df.columns
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
                "contain identical 64-point "
                "crossed-holdout conditions."
            )


    loaded[
        target["file"]
    ] = df


# ============================================================
# 6. Common 8 × 8 benchmark grid
# ============================================================

ENERGIES = np.sort(
    reference_conditions[
        "held_energy_keV"
    ].unique()
)


ANGLES = np.sort(
    reference_conditions[
        "held_angle_deg"
    ].unique()
)


if len(ENERGIES) != 8:

    raise ValueError(
        f"Expected 8 common energies; "
        f"found {len(ENERGIES)}."
    )


if len(ANGLES) != 8:

    raise ValueError(
        f"Expected 8 common angles; "
        f"found {len(ANGLES)}."
    )


if (
    len(ENERGIES)
    *
    len(ANGLES)
    != 64
):

    raise ValueError(
        "Benchmark grid is not 8 × 8."
    )


print("\nQC PASSED")

print(
    "Common crossed-holdout grid: "
    "8 energies × 8 angles = 64 conditions"
)

print(
    "Energies:",
    ENERGIES
)

print(
    "Angles:",
    ANGLES
)


# ============================================================
# 7. Build long-form error-advantage table
# ============================================================

long_rows = []


for target in TARGETS:

    df = loaded[
        target["file"]
    ].copy()


    truth = df[
        target["truth"]
    ].to_numpy(
        dtype=float
    )


    gpr = df[
        target["gpr"]
    ].to_numpy(
        dtype=float
    )


    pchip = df[
        target["pchip"]
    ].to_numpy(
        dtype=float
    )


    gpr_abs_error = np.abs(
        truth
        -
        gpr
    )


    pchip_abs_error = np.abs(
        truth
        -
        pchip
    )


    # ========================================================
    # Positive:
    #     GPR lower absolute error
    #
    # Negative:
    #     PCHIP lower absolute error
    # ========================================================

    delta_abs_error = (
        pchip_abs_error
        -
        gpr_abs_error
    )


    for i in range(
        len(df)
    ):

        long_rows.append(
            {
                "target":
                    target["key"],

                "energy_keV":
                    float(
                        df.iloc[i][
                            "held_energy_keV"
                        ]
                    ),

                "tilt_deg":
                    float(
                        df.iloc[i][
                            "held_angle_deg"
                        ]
                    ),

                "md_truth":
                    float(
                        truth[i]
                    ),

                "gpr_prediction":
                    float(
                        gpr[i]
                    ),

                "pchip_prediction":
                    float(
                        pchip[i]
                    ),

                "gpr_abs_error":
                    float(
                        gpr_abs_error[i]
                    ),

                "pchip_abs_error":
                    float(
                        pchip_abs_error[i]
                    ),

                "delta_abs_error_pchip_minus_gpr":
                    float(
                        delta_abs_error[i]
                    ),
            }
        )


long_df = pd.DataFrame(
    long_rows
)


long_df = (
    long_df
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


long_df.to_csv(
    OUTPUT_LONG,
    index=False,
)


# ============================================================
# 8. Summary statistics
# ============================================================

summary_rows = []

TOLERANCE = 1e-12


for target in TARGETS:

    sub = long_df[
        long_df[
            "target"
        ]
        ==
        target["key"]
    ].copy()


    delta = sub[
        "delta_abs_error_pchip_minus_gpr"
    ].to_numpy(
        dtype=float
    )


    gpr_wins = int(
        np.sum(
            delta
            >
            TOLERANCE
        )
    )


    pchip_wins = int(
        np.sum(
            delta
            <
            -TOLERANCE
        )
    )


    ties = int(
        np.sum(
            np.abs(
                delta
            )
            <=
            TOLERANCE
        )
    )


    gpr_best_idx = (
        sub[
            "delta_abs_error_pchip_minus_gpr"
        ]
        .idxmax()
    )


    gpr_best = (
        sub.loc[
            gpr_best_idx
        ]
    )


    pchip_best_idx = (
        sub[
            "delta_abs_error_pchip_minus_gpr"
        ]
        .idxmin()
    )


    pchip_best = (
        sub.loc[
            pchip_best_idx
        ]
    )


    summary_rows.append(
        {
            "target":
                target["key"],

            "n":
                len(sub),

            "gpr_point_wins":
                gpr_wins,

            "pchip_point_wins":
                pchip_wins,

            "ties":
                ties,

            "mean_delta_abs_error":
                float(
                    np.mean(
                        delta
                    )
                ),

            "median_delta_abs_error":
                float(
                    np.median(
                        delta
                    )
                ),

            "largest_gpr_advantage":
                float(
                    gpr_best[
                        "delta_abs_error_pchip_minus_gpr"
                    ]
                ),

            "largest_gpr_advantage_energy_keV":
                float(
                    gpr_best[
                        "energy_keV"
                    ]
                ),

            "largest_gpr_advantage_tilt_deg":
                float(
                    gpr_best[
                        "tilt_deg"
                    ]
                ),

            "largest_pchip_advantage":
                float(
                    pchip_best[
                        "delta_abs_error_pchip_minus_gpr"
                    ]
                ),

            "largest_pchip_advantage_energy_keV":
                float(
                    pchip_best[
                        "energy_keV"
                    ]
                ),

            "largest_pchip_advantage_tilt_deg":
                float(
                    pchip_best[
                        "tilt_deg"
                    ]
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ============================================================
# 9. Helper: exact 8 × 8 matrix
# ============================================================

def build_matrix(
    target_key,
):

    sub = long_df[
        long_df[
            "target"
        ]
        ==
        target_key
    ]


    pivot = (
        sub
        .pivot(
            index="energy_keV",
            columns="tilt_deg",
            values=
                "delta_abs_error_pchip_minus_gpr",
        )
        .reindex(
            index=ENERGIES,
            columns=ANGLES,
        )
    )


    if pivot.isna().any().any():

        raise ValueError(
            f"Missing cell in "
            f"{target_key} advantage matrix."
        )


    if pivot.shape != (
        8,
        8,
    ):

        raise ValueError(
            f"{target_key} matrix has shape "
            f"{pivot.shape}; expected (8, 8)."
        )


    return pivot.to_numpy(
        dtype=float
    )


# ============================================================
# 10. Figure
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


for ax, target in zip(
    axes,
    TARGETS,
):

    Z = build_matrix(
        target["key"]
    )


    # ========================================================
    # Symmetric target-specific scale
    # ========================================================

    abs_max = float(
        np.max(
            np.abs(
                Z
            )
        )
    )


    if abs_max <= 0:

        abs_max = 1.0


    norm = TwoSlopeNorm(
        vmin=-abs_max,
        vcenter=0.0,
        vmax=abs_max,
    )


    # ========================================================
    # Discrete 8 × 8 heatmap
    #
    # No interpolation or smoothing.
    # ========================================================

    image = ax.imshow(
        Z,

        origin="lower",

        cmap=ERROR_CMAP,

        norm=norm,

        interpolation="nearest",

        aspect="equal",

        zorder=1,
    )


    # ========================================================
    # Cell boundaries
    # ========================================================

    ax.set_xticks(
        np.arange(
            len(
                ANGLES
            )
        )
        -
        0.5,
        minor=True,
    )


    ax.set_yticks(
        np.arange(
            len(
                ENERGIES
            )
        )
        -
        0.5,
        minor=True,
    )


    ax.grid(
        which="minor",

        color="white",

        linestyle="-",

        linewidth=0.50,

        alpha=0.55,

        zorder=3,
    )


    ax.tick_params(
        which="minor",

        bottom=False,
        left=False,
        top=False,
        right=False,
    )


    # ========================================================
    # X labels
    # ========================================================

    ax.set_xticks(
        np.arange(
            len(
                ANGLES
            )
        )
    )


    angle_labels = []


    for angle in ANGLES:

        if np.isclose(
            angle,
            18.43,
        ):

            angle_labels.append(
                "18.43"
            )

        else:

            angle_labels.append(
                f"{angle:g}"
            )


    ax.set_xticklabels(
        angle_labels,

        rotation=25,

        ha="right",

        rotation_mode="anchor",
    )


    # ========================================================
    # Y labels
    # ========================================================

    ax.set_yticks(
        np.arange(
            len(
                ENERGIES
            )
        )
    )


    ax.set_yticklabels(
        [
            f"{energy:g}"
            for energy
            in ENERGIES
        ]
    )


    # ========================================================
    # Labels
    # ========================================================

    ax.set_xlabel(
        "Held-out tilt angle (°)"
    )


    ax.set_ylabel(
        "Held-out energy (keV)"
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
    # Colorbar
    # ========================================================

    cbar = fig.colorbar(
        image,

        ax=ax,

        fraction=0.044,

        pad=0.028,
    )


    cbar.set_label(
        target[
            "cbar_label"
        ],

        fontsize=8.5,
    )


    cbar.ax.tick_params(
        labelsize=7.5,
        width=0.7,
        length=3.0,
    )


    cbar.outline.set_linewidth(
        0.7
    )


    # --------------------------------------------------------
    # Symmetric ticks centered on zero
    # --------------------------------------------------------

    cbar.set_ticks(
        np.linspace(
            -abs_max,
            abs_max,
            5,
        )
    )


# ============================================================
# 11. Layout
# ============================================================

fig.subplots_adjust(
    left=0.085,
    right=0.965,

    bottom=0.105,
    top=0.95,

    wspace=0.38,
    hspace=0.38,
)


# ============================================================
# 12. Save
# ============================================================

save_paper_figure(
    fig,
    OUTPUT_BASE,
)


# ============================================================
# 13. Console summary
# ============================================================

print("\n")
print("=" * 150)

print(
    "FIGURE ML-06 — FINAL GPR vs PCHIP "
    "CROSSED-HOLDOUT ERROR-ADVANTAGE MAPS"
)

print("=" * 150)


print(
    summary_df.to_string(
        index=False
    )
)


print("\nSaved:")

print(
    OUTPUT_LONG
)

print(
    OUTPUT_SUMMARY
)


print("\nDefinition:")

print(
    "Δ|error| = "
    "|PCHIP error| - |GPR error|"
)


print(
    "Positive values -> GPR lower absolute error."
)

print(
    "Negative values -> PCHIP lower absolute error."
)


print(
    "\nNo interpolation, smoothing, "
    "retraining, or post-hoc transformation "
    "was applied."
)


plt.close(
    fig
)
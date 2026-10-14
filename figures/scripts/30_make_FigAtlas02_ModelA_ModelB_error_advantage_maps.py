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
# Script: 30_make_FigAtlas02_ModelA_ModelB_error_advantage_maps.py
# Purpose: Generate condition-resolved Model A versus Model B absolute-error advantage maps.
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
from matplotlib.patches import Patch

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

MODEL_DIR = (
    ROOT
    / "analysis"
    / "atlas_model"
)

DEPTH_PATH = (
    MODEL_DIR
    / "depth_modelA_vs_modelB_crossed64_predictions.csv"
)

REMAINING_PATH = (
    MODEL_DIR
    / "remaining_targets_modelA_vs_modelB_crossed64_predictions.csv"
)


RENDER_DIR = HERE.parent / "rendered"
SOURCE_DIR = HERE.parent / "source_data"
OUTPUT_BASE = RENDER_DIR / "Figure_Atlas02_ModelA_ModelB_Error_Advantage_Maps"
OUTPUT_LONG = SOURCE_DIR / "Figure_Atlas02_ModelA_ModelB_Error_Advantage_Data.csv"
OUTPUT_SUMMARY = SOURCE_DIR / "Figure_Atlas02_ModelA_ModelB_Error_Advantage_Summary.csv"


# ============================================================
# 3. Visual identity
#
# Negative Δ|e|:
#   Model A has lower absolute error
#
# Positive Δ|e|:
#   Model B has lower absolute error
# ============================================================

MODEL_A_ADVANTAGE_COLOR = "#C65D13"
NEUTRAL_COLOR = "#FFFFFF"
MODEL_B_ADVANTAGE_COLOR = "#7B3294"


ADVANTAGE_CMAP = LinearSegmentedColormap.from_list(
    "ModelA_to_ModelB",
    [
        MODEL_A_ADVANTAGE_COLOR,
        NEUTRAL_COLOR,
        MODEL_B_ADVANTAGE_COLOR,
    ],
    N=256,
)


# ============================================================
# 4. Target display definitions
# ============================================================

TARGETS = [

    {
        "panel": "(a)",
        "key": "median_depth",
        "title": "Median implantation depth",
        "unit_label": "Δ|error| (nm)",
    },

    {
        "panel": "(b)",
        "key": "shallow_fraction",
        "title": "Shallow fraction (0–5 nm)",
        "unit_label": "Δ|error|",
    },

    {
        "panel": "(c)",
        "key": "near_surface_sp2",
        "title": "Near-surface sp² fraction (0–2 nm)",
        "unit_label": "Δ|error|",
    },

    {
        "panel": "(d)",
        "key": "vacancy_count",
        "title": "Vacancy count",
        "unit_label": "Δ|error| (count)",
    },
]


TARGET_ORDER = [
    "median_depth",
    "shallow_fraction",
    "near_surface_sp2",
    "vacancy_count",
]


# ============================================================
# 5. Load frozen crossed-holdout predictions
# ============================================================

depth_df = pd.read_csv(
    DEPTH_PATH
)

remaining_df = pd.read_csv(
    REMAINING_PATH
)


# ============================================================
# 6. QC — depth
# ============================================================

required_common_columns = [
    "held_energy",
    "held_angle",
    "y_true",

    "modelA_pred",
    "modelA_abs_error",

    "modelB_pred",
    "modelB_abs_error",
]


missing_depth = [
    col
    for col in required_common_columns
    if col not in depth_df.columns
]


if missing_depth:
    raise ValueError(
        f"Missing columns in depth file: "
        f"{missing_depth}"
    )


if len(depth_df) != 64:
    raise ValueError(
        f"Expected 64 depth predictions; "
        f"found {len(depth_df)}."
    )


# ============================================================
# 7. QC — remaining targets
# ============================================================

required_remaining_columns = (
    [
        "target_label",
    ]
    +
    required_common_columns
)


missing_remaining = [
    col
    for col in required_remaining_columns
    if col not in remaining_df.columns
]


if missing_remaining:
    raise ValueError(
        f"Missing columns in remaining-target file: "
        f"{missing_remaining}"
    )


if len(remaining_df) != 192:
    raise ValueError(
        f"Expected 192 remaining-target predictions; "
        f"found {len(remaining_df)}."
    )


remaining_target_labels = set(
    remaining_df[
        "target_label"
    ].unique()
)


expected_remaining_labels = {
    "shallow_fraction",
    "near_surface_sp2",
    "vacancy_count",
}


if (
    remaining_target_labels
    !=
    expected_remaining_labels
):

    raise ValueError(
        "Unexpected target labels in "
        "remaining-target prediction file."
    )


# ============================================================
# 8. Build common long-form table
# ============================================================

long_rows = []


# ------------------------------------------------------------
# Depth
# ------------------------------------------------------------

for _, row in depth_df.iterrows():

    delta_abs_error = (
        float(
            row[
                "modelA_abs_error"
            ]
        )
        -
        float(
            row[
                "modelB_abs_error"
            ]
        )
    )


    long_rows.append(
        {
            "target":
                "median_depth",

            "energy_keV":
                float(
                    row[
                        "held_energy"
                    ]
                ),

            "tilt_deg":
                float(
                    row[
                        "held_angle"
                    ]
                ),

            "md_truth":
                float(
                    row[
                        "y_true"
                    ]
                ),

            "modelA_prediction":
                float(
                    row[
                        "modelA_pred"
                    ]
                ),

            "modelB_prediction":
                float(
                    row[
                        "modelB_pred"
                    ]
                ),

            "modelA_abs_error":
                float(
                    row[
                        "modelA_abs_error"
                    ]
                ),

            "modelB_abs_error":
                float(
                    row[
                        "modelB_abs_error"
                    ]
                ),

            "delta_abs_error_A_minus_B":
                delta_abs_error,
        }
    )


# ------------------------------------------------------------
# Remaining three targets
# ------------------------------------------------------------

for _, row in remaining_df.iterrows():

    delta_abs_error = (
        float(
            row[
                "modelA_abs_error"
            ]
        )
        -
        float(
            row[
                "modelB_abs_error"
            ]
        )
    )


    long_rows.append(
        {
            "target":
                str(
                    row[
                        "target_label"
                    ]
                ),

            "energy_keV":
                float(
                    row[
                        "held_energy"
                    ]
                ),

            "tilt_deg":
                float(
                    row[
                        "held_angle"
                    ]
                ),

            "md_truth":
                float(
                    row[
                        "y_true"
                    ]
                ),

            "modelA_prediction":
                float(
                    row[
                        "modelA_pred"
                    ]
                ),

            "modelB_prediction":
                float(
                    row[
                        "modelB_pred"
                    ]
                ),

            "modelA_abs_error":
                float(
                    row[
                        "modelA_abs_error"
                    ]
                ),

            "modelB_abs_error":
                float(
                    row[
                        "modelB_abs_error"
                    ]
                ),

            "delta_abs_error_A_minus_B":
                delta_abs_error,
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


# ============================================================
# 9. Global QC
# ============================================================

for target_key in TARGET_ORDER:

    sub = long_df[
        long_df[
            "target"
        ]
        ==
        target_key
    ]


    if len(sub) != 64:

        raise ValueError(
            f"{target_key}: "
            f"expected 64 conditions, "
            f"found {len(sub)}."
        )


# ============================================================
# 10. Common 8 × 8 coordinates
# ============================================================

reference = long_df[
    long_df[
        "target"
    ]
    ==
    "median_depth"
]


ENERGIES = np.sort(
    reference[
        "energy_keV"
    ].unique()
)


ANGLES = np.sort(
    reference[
        "tilt_deg"
    ].unique()
)


if len(ENERGIES) != 8:
    raise ValueError(
        f"Expected 8 energies; "
        f"found {len(ENERGIES)}."
    )


if len(ANGLES) != 8:
    raise ValueError(
        f"Expected 8 angles; "
        f"found {len(ANGLES)}."
    )


# ------------------------------------------------------------
# Confirm every target contains identical coordinates
# ------------------------------------------------------------

reference_coords = (
    reference[
        [
            "energy_keV",
            "tilt_deg",
        ]
    ]
    .sort_values(
        [
            "energy_keV",
            "tilt_deg",
        ]
    )
    .reset_index(
        drop=True
    )
)


for target_key in TARGET_ORDER:

    sub_coords = (
        long_df[
            long_df[
                "target"
            ]
            ==
            target_key
        ][
            [
                "energy_keV",
                "tilt_deg",
            ]
        ]
        .sort_values(
            [
                "energy_keV",
                "tilt_deg",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    if not sub_coords.equals(
        reference_coords
    ):

        raise ValueError(
            f"{target_key} does not use "
            f"the identical 64-condition grid."
        )


print("\nQC PASSED")

print(
    "Atlas Model A/B comparison:"
)

print(
    "4 targets × 64 identical crossed-holdout conditions."
)

print(
    "Grid:"
)

print(
    f"{len(ENERGIES)} energies × "
    f"{len(ANGLES)} angles"
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
# 11. Save exact long-form data
# ============================================================

long_df.to_csv(
    OUTPUT_LONG,
    index=False,
)


# ============================================================
# 12. Build summary statistics
# ============================================================

summary_rows = []

TOL = 1e-12


for target_key in TARGET_ORDER:

    sub = long_df[
        long_df[
            "target"
        ]
        ==
        target_key
    ].copy()


    delta = sub[
        "delta_abs_error_A_minus_B"
    ].to_numpy(
        dtype=float
    )


    modelB_wins = int(
        np.sum(
            delta
            >
            TOL
        )
    )


    modelA_wins = int(
        np.sum(
            delta
            <
            -TOL
        )
    )


    ties = int(
        np.sum(
            np.abs(
                delta
            )
            <=
            TOL
        )
    )


    # --------------------------------------------------------
    # Largest Atlas / Model B advantage
    # --------------------------------------------------------

    best_B_idx = (
        sub[
            "delta_abs_error_A_minus_B"
        ]
        .idxmax()
    )


    best_B = sub.loc[
        best_B_idx
    ]


    # --------------------------------------------------------
    # Largest Model A advantage
    # --------------------------------------------------------

    best_A_idx = (
        sub[
            "delta_abs_error_A_minus_B"
        ]
        .idxmin()
    )


    best_A = sub.loc[
        best_A_idx
    ]


    summary_rows.append(
        {
            "target":
                target_key,

            "n":
                len(sub),

            "modelA_point_wins":
                modelA_wins,

            "modelB_point_wins":
                modelB_wins,

            "ties":
                ties,

            "mean_delta_abs_error_A_minus_B":
                float(
                    np.mean(
                        delta
                    )
                ),

            "median_delta_abs_error_A_minus_B":
                float(
                    np.median(
                        delta
                    )
                ),

            "largest_modelB_advantage":
                float(
                    best_B[
                        "delta_abs_error_A_minus_B"
                    ]
                ),

            "largest_modelB_advantage_energy_keV":
                float(
                    best_B[
                        "energy_keV"
                    ]
                ),

            "largest_modelB_advantage_tilt_deg":
                float(
                    best_B[
                        "tilt_deg"
                    ]
                ),

            "largest_modelA_advantage":
                float(
                    best_A[
                        "delta_abs_error_A_minus_B"
                    ]
                ),

            "largest_modelA_advantage_energy_keV":
                float(
                    best_A[
                        "energy_keV"
                    ]
                ),

            "largest_modelA_advantage_tilt_deg":
                float(
                    best_A[
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
# 13. Helper: build exact discrete matrix
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
                "delta_abs_error_A_minus_B",
        )
        .reindex(
            index=ENERGIES,
            columns=ANGLES,
        )
    )


    if pivot.isna().any().any():

        raise ValueError(
            f"Missing cell in "
            f"{target_key} matrix."
        )


    if pivot.shape != (
        8,
        8,
    ):

        raise ValueError(
            f"{target_key} matrix shape "
            f"is {pivot.shape}; "
            f"expected (8, 8)."
        )


    return pivot.to_numpy(
        dtype=float
    )


# ============================================================
# 14. Figure
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


for ax, target in zip(
    axes,
    TARGETS,
):

    Z = build_matrix(
        target[
            "key"
        ]
    )


    # ========================================================
    # Symmetric full-range scale
    #
    # No percentile clipping:
    # full frozen benchmark range is shown.
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
    # Exact discrete heatmap
    # ========================================================

    image = ax.imshow(
        Z,

        origin="lower",

        cmap=ADVANTAGE_CMAP,

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
        top=False,

        left=False,
        right=False,
    )


    # ========================================================
    # X-axis
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
    # Y-axis
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


    ax.set_xlabel(
        "Held-out tilt angle (°)"
    )


    ax.set_ylabel(
        "Held-out energy (keV)"
    )


    ax.set_title(
        target[
            "title"
        ],

        pad=6,
    )


    add_panel_label(
        ax,
        target[
            "panel"
        ],
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
            "unit_label"
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


    cbar.set_ticks(
        np.linspace(
            -abs_max,
            abs_max,
            5,
        )
    )


# ============================================================
# 15. Shared color-meaning legend
# ============================================================

legend_handles = [

    Patch(
        facecolor=
            MODEL_A_ADVANTAGE_COLOR,

        edgecolor="black",

        linewidth=0.4,

        label=
            "Model A lower |error|",
    ),


    Patch(
        facecolor=
            MODEL_B_ADVANTAGE_COLOR,

        edgecolor="black",

        linewidth=0.4,

        label=
            "Model B lower |error|",
    ),
]


fig.legend(
    handles=legend_handles,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.997,
    ),

    ncol=2,

    frameon=False,

    columnspacing=1.7,

    handlelength=1.25,

    handletextpad=0.55,
)


# ============================================================
# 16. Layout
# ============================================================

fig.subplots_adjust(
    left=0.085,
    right=0.965,

    bottom=0.105,
    top=0.895,

    wspace=0.38,
    hspace=0.38,
)


# ============================================================
# 17. Save
# ============================================================

save_paper_figure(
    fig,
    OUTPUT_BASE,
)


# ============================================================
# 18. Console summary
# ============================================================

print("\n")
print("=" * 155)

print(
    "FIGURE ATLAS-02 — MODEL A vs MODEL B "
    "CROSSED-HOLDOUT ERROR-ADVANTAGE MAPS"
)

print("=" * 155)


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
    "|Model A error| - |Model B error|"
)


print(
    "Positive / purple -> Model B has lower absolute error."
)

print(
    "Negative / orange -> Model A has lower absolute error."
)


print(
    "\nNo smoothing, interpolation, "
    "retraining, clipping, or target-driven "
    "post-processing was used."
)


plt.close(
    fig
)
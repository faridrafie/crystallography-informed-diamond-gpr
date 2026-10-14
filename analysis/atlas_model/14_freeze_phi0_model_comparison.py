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
# Script: 14_freeze_phi0_model_comparison.py
# Purpose: Freeze the fixed-azimuth Model A versus Model B comparison tables used in the manuscript.
#
# Research note:
#   This script belongs to the reproducibility package for my doctoral
#   research on simulation-guided low-energy nitrogen implantation in diamond.
#   Numerical settings used for manuscript results are intentionally kept
#   explicit in the script rather than hidden behind external configuration.
# =============================================================================

from pathlib import Path

import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]

DEPTH_PATH = (
    SCRIPT_DIR
    / "depth_modelA_vs_modelB_crossed64_summary.csv"
)

REMAINING_PATH = (
    SCRIPT_DIR
    / "remaining_targets_modelA_vs_modelB_crossed64_summary.csv"
)

OUTPUT_PATH = (
    SCRIPT_DIR
    / "phi0_modelA_vs_modelB_summary.csv"
)

PROTOCOL_PATH = (
    SCRIPT_DIR
    / "atlas_model_v1_frozen_protocol.txt"
)


# ============================================================
# 2. Load
# ============================================================

depth_df = pd.read_csv(
    DEPTH_PATH
)

remaining_df = pd.read_csv(
    REMAINING_PATH
)


if len(depth_df) != 1:
    raise ValueError(
        "Expected exactly one depth-summary row."
    )


if len(remaining_df) != 3:
    raise ValueError(
        "Expected exactly three remaining-target rows."
    )


# ============================================================
# 3. Standardize depth row
# ============================================================

depth_row = depth_df.iloc[0]


depth_standard = pd.DataFrame(
    [
        {
            "target_label":
                "median_depth",

            "target_column":
                "p1_depth_median_nm",

            "n_test":
                int(
                    depth_row["n_test"]
                ),

            "modelA_MAE":
                depth_row["modelA_MAE"],

            "modelB_MAE":
                depth_row["modelB_MAE"],

            "modelA_RMSE":
                depth_row["modelA_RMSE"],

            "modelB_RMSE":
                depth_row["modelB_RMSE"],

            "modelA_R2":
                depth_row["modelA_R2"],

            "modelB_R2":
                depth_row["modelB_R2"],

            "modelB_MAE_reduction_percent":
                depth_row[
                    "modelB_MAE_reduction_percent"
                ],

            "modelB_RMSE_reduction_percent":
                depth_row[
                    "modelB_RMSE_reduction_percent"
                ],

            "modelA_point_wins":
                depth_row[
                    "modelA_point_wins"
                ],

            "modelB_point_wins":
                depth_row[
                    "modelB_point_wins"
                ],

            "ties":
                depth_row[
                    "ties"
                ],

            "modelA_physical_violations":
                0,

            "modelB_physical_violations":
                0,

            "modelA_convergence_warnings":
                depth_row[
                    "modelA_convergence_warnings"
                ],

            "modelB_convergence_warnings":
                depth_row[
                    "modelB_convergence_warnings"
                ],
        }
    ]
)


# ============================================================
# 4. Combine all four frozen targets
# ============================================================

summary_df = pd.concat(
    [
        depth_standard,
        remaining_df,
    ],
    ignore_index=True,
)


EXPECTED_TARGETS = {
    "median_depth",
    "shallow_fraction",
    "near_surface_sp2",
    "vacancy_count",
}


if set(
    summary_df["target_label"]
) != EXPECTED_TARGETS:

    raise ValueError(
        "Unexpected target set."
    )


if not (
    summary_df["n_test"]
    == 64
).all():

    raise ValueError(
        "Every target must contain exactly "
        "64 common crossed-holdout predictions."
    )


# ============================================================
# 5. Add metric winners
# ============================================================

summary_df[
    "MAE_winner"
] = summary_df.apply(
    lambda row:
        "Model B"
        if row["modelB_MAE"]
        < row["modelA_MAE"]
        else "Model A",
    axis=1,
)


summary_df[
    "RMSE_winner"
] = summary_df.apply(
    lambda row:
        "Model B"
        if row["modelB_RMSE"]
        < row["modelA_RMSE"]
        else "Model A",
    axis=1,
)


summary_df[
    "R2_winner"
] = summary_df.apply(
    lambda row:
        "Model B"
        if row["modelB_R2"]
        > row["modelA_R2"]
        else "Model A",
    axis=1,
)


# ============================================================
# 6. Save frozen summary
# ============================================================

summary_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# 7. Freeze Model-B protocol
# ============================================================

protocol = """
ATLAS MODEL V1 — FROZEN BEFORE AZIMUTH EVALUATION
=================================================

Purpose
-------
Evaluate whether a crystallographic representation
can replace raw orientation coordinates and support
generalization beyond the original phi = 0 MD slice.

MODEL A
-------
Inputs:
    E
    theta

MODEL B
-------
Inputs:
    E

    4 fixed axis-distance features:
        d_[001]
        d_[103]
        d_[113]
        d_[102]

    11 fixed plane-family-distance features:
        d_{100}
        d_{110}
        d_{201}
        d_{210}
        d_{211}
        d_{221}
        d_{301}
        d_{310}
        d_{311}
        d_{320}
        d_{321}

Raw theta and raw phi are not supplied to Model B.

Important:
    d_[001] is geometrically identical to tilt theta.
    Therefore Model B retains tilt information through
    the crystallographic representation.

ATLAS FEATURE NORMALIZATION
---------------------------
The 15 Atlas features are normalized using fixed
geometry-only statistics from:

    theta = 0 to 30 deg
    phi   = 0 to 45 deg

on the pre-defined 0.25-deg orientation grid.

No MD response value is used in this normalization.

ENERGY NORMALIZATION
--------------------
Energy is standardized independently inside each
training fold.

GPR ARCHITECTURE
----------------
Kernel:

    Constant
    x
    Grouped RBF
    +
    White noise

Grouped RBF contains two optimized characteristic
length scales:

    l_E        : energy
    l_geometry : entire crystallographic representation

GPR settings:

    normalize_y = True
    optimizer restarts = 8
    random state = 42

PHI=0 QUALIFICATION BENCHMARK
-----------------------------
Common interior crossed energy-angle holdout:

    held energies:
        1.0 to 4.5 keV

    held angles:
        3, 7, 10, 15, 18.43, 20, 22, 25 deg

For each target point (E*, theta*):

    remove all conditions with E = E*
    AND all conditions with theta = theta*

Training conditions per fold:
    81

Test conditions:
    64 total per target

TARGETS
-------
    median implantation depth
    shallow fraction
    near-surface sp2 fraction
    vacancy count

FREEZE RULE
-----------
The Atlas representation, geometry scaler, grouped GPR
architecture, and phi=0 qualification protocol were fixed
before response-specific predictive comparisons.

MD response values were not used to select or redefine Atlas
axis families, plane families, or geometry-domain normalization
statistics.
"""


with open(
    PROTOCOL_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        protocol.strip()
        + "\n"
    )


# ============================================================
# 8. Print
# ============================================================

print("\n")
print("=" * 120)
print("FROZEN PHI=0 MODEL A vs MODEL B SUMMARY")
print("=" * 120)

print(
    summary_df[
        [
            "target_label",
            "modelA_MAE",
            "modelB_MAE",
            "MAE_winner",
            "modelA_RMSE",
            "modelB_RMSE",
            "RMSE_winner",
            "modelA_R2",
            "modelB_R2",
            "R2_winner",
            "modelA_point_wins",
            "modelB_point_wins",
        ]
    ].to_string(
        index=False
    )
)


print("\nSaved:")
print(
    OUTPUT_PATH.name
)
print(
    PROTOCOL_PATH.name
)
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
# Script: 24_make_FigML04_GPR_uncertainty_maps.py
# Purpose: Generate full-domain GPR predictive-standard-deviation maps.
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
from matplotlib.ticker import MaxNLocator

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

SOURCE_DIR = HERE.parent / "source_data"
RENDER_DIR = HERE.parent / "rendered"

# Use the frozen ML-03 source grid shipped with the repository.
# Therefore ML-04 can be generated directly without first running ML-03.
GRID_PATH = SOURCE_DIR / "Figure_ML03_GPR_Response_Surface_Grid.csv"

MD_PATH = (
    ROOT
    / "data"
    / "master_condition_table_core.csv"
)

OUTPUT_BASE = RENDER_DIR / "Figure_ML04_GPR_Predictive_Uncertainty"
QC_OUTPUT = SOURCE_DIR / "Figure_ML04_GPR_Predictive_Uncertainty_QC.csv"


# ============================================================
# 3. Target definitions
# ============================================================

TARGETS = [

    {
        "panel": "(a)",
        "key": "depth",
        "title": "Median implantation depth",
        "std_col": "depth_std",
        "cbar_label": "Predictive SD (nm)",
        "digits": 2,
    },

    {
        "panel": "(b)",
        "key": "shallow",
        "title": "Shallow fraction (0–5 nm)",
        "std_col": "shallow_std",
        "cbar_label": "Predictive SD",
        "digits": 3,
    },

    {
        "panel": "(c)",
        "key": "sp2",
        "title": "Near-surface sp² fraction (0–2 nm)",
        "std_col": "sp2_std",
        "cbar_label": "Predictive SD",
        "digits": 4,
    },

    {
        "panel": "(d)",
        "key": "vacancy",
        "title": "Vacancy count",
        "std_col": "vacancy_std",
        "cbar_label": "Predictive SD (count)",
        "digits": 1,
    },
]


# ============================================================
# 4. Load saved response-grid predictions
# ============================================================

grid_df = pd.read_csv(
    GRID_PATH
)

md_df = pd.read_csv(
    MD_PATH
)


required_grid_columns = [
    "energy_keV",
    "tilt_deg",
] + [
    target["std_col"]
    for target in TARGETS
]


missing = [
    col
    for col in required_grid_columns
    if col not in grid_df.columns
]


if missing:

    raise ValueError(
        f"Missing columns in response grid: {missing}"
    )


if len(md_df) != 100:

    raise ValueError(
        f"Expected 100 MD conditions, "
        f"found {len(md_df)}."
    )


if md_df[
    [
        "E",
        "A",
    ]
].duplicated().any():

    raise ValueError(
        "Duplicate MD E-angle conditions detected."
    )


# ============================================================
# 5. Recover exact dense grid
# ============================================================

energy_values = np.sort(
    grid_df[
        "energy_keV"
    ].unique()
)


angle_values = np.sort(
    grid_df[
        "tilt_deg"
    ].unique()
)


N_ENERGY = len(
    energy_values
)

N_ANGLE = len(
    angle_values
)


expected_rows = (
    N_ENERGY
    *
    N_ANGLE
)


if len(grid_df) != expected_rows:

    raise ValueError(
        f"Grid size mismatch: "
        f"{len(grid_df)} rows found, "
        f"{expected_rows} expected."
    )


# Make sure row order is deterministic.
grid_df = (
    grid_df
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


ANGLE_GRID, ENERGY_GRID = np.meshgrid(
    angle_values,
    energy_values,
)


print("\nQC PASSED")

print(
    f"Dense grid: "
    f"{N_ENERGY} energies × "
    f"{N_ANGLE} angles = "
    f"{expected_rows} points"
)

print(
    f"Energy range: "
    f"{energy_values.min():.2f}–"
    f"{energy_values.max():.2f} keV"
)

print(
    f"Tilt range: "
    f"{angle_values.min():.2f}–"
    f"{angle_values.max():.2f}°"
)


# ============================================================
# 6. Helper: sparse uncertainty contours
# ============================================================

def get_contour_levels(
    Z,
    nbins=4,
):

    zmin = float(
        np.nanmin(
            Z
        )
    )

    zmax = float(
        np.nanmax(
            Z
        )
    )


    locator = MaxNLocator(
        nbins=nbins
    )


    levels = locator.tick_values(
        zmin,
        zmax,
    )


    levels = levels[
        (levels > zmin)
        &
        (levels < zmax)
    ]


    if len(levels) < 2:

        levels = np.linspace(
            zmin,
            zmax,
            5,
        )[1:-1]


    return levels


# ============================================================
# 7. Prepare uncertainty surfaces + QC
# ============================================================

uncertainty_surfaces = {}

qc_rows = []


for target in TARGETS:

    values = grid_df[
        target[
            "std_col"
        ]
    ].to_numpy(
        dtype=float
    )


    if np.any(
        ~np.isfinite(
            values
        )
    ):

        raise ValueError(
            f"Non-finite uncertainty values "
            f"for {target['key']}."
        )


    if np.any(
        values < 0
    ):

        raise ValueError(
            f"Negative predictive SD detected "
            f"for {target['key']}."
        )


    Z = values.reshape(
        N_ENERGY,
        N_ANGLE,
    )


    uncertainty_surfaces[
        target["key"]
    ] = Z


    # --------------------------------------------------------
    # Locate highest uncertainty
    # --------------------------------------------------------

    max_index = int(
        np.argmax(
            values
        )
    )


    max_row = grid_df.iloc[
        max_index
    ]


    qc_rows.append(
        {
            "target":
                target["key"],

            "std_min":
                float(
                    np.min(
                        values
                    )
                ),

            "std_mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "std_median":
                float(
                    np.median(
                        values
                    )
                ),

            "std_max":
                float(
                    np.max(
                        values
                    )
                ),

            "max_uncertainty_energy_keV":
                float(
                    max_row[
                        "energy_keV"
                    ]
                ),

            "max_uncertainty_tilt_deg":
                float(
                    max_row[
                        "tilt_deg"
                    ]
                ),
        }
    )


qc_df = pd.DataFrame(
    qc_rows
)


qc_df.to_csv(
    QC_OUTPUT,
    index=False,
)


# ============================================================
# 8. Figure
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

    Z = uncertainty_surfaces[
        target["key"]
    ]


    # ========================================================
    # Smooth predictive-uncertainty field
    # ========================================================

    contourf = ax.contourf(
        ANGLE_GRID,
        ENERGY_GRID,
        Z,

        levels=80,

        cmap="magma",

        antialiased=True,
    )


    # ========================================================
    # Very sparse contour guides
    #
    # No numeric labels:
    # colorbar already carries the quantitative scale.
    # ========================================================

    contour_levels = get_contour_levels(
        Z,
        nbins=4,
    )


    ax.contour(
        ANGLE_GRID,
        ENERGY_GRID,
        Z,

        levels=contour_levels,

        colors="black",

        linewidths=0.45,

        alpha=0.35,

        zorder=2,
    )


    # ========================================================
    # Original MD sampling locations
    # ========================================================

    ax.scatter(
        md_df["A"],
        md_df["E"],

        s=5,

        facecolors="white",

        edgecolors="black",

        linewidths=0.25,

        alpha=0.55,

        zorder=4,
    )


    # ========================================================
    # Axes — identical to Figure ML-03
    # ========================================================

    ax.set_xlim(
        0,
        30,
    )


    ax.set_ylim(
        0.5,
        5.0,
    )


    ax.set_xticks(
        [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
        ]
    )


    ax.set_yticks(
        [
            0.5,
            1,
            2,
            3,
            4,
            5,
        ]
    )


    ax.set_xlabel(
        "Tilt angle (°)"
    )


    ax.set_ylabel(
        "Implantation energy (keV)"
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
        contourf,

        ax=ax,

        fraction=0.046,

        pad=0.025,
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


    # Control number of colorbar ticks.
    cbar.locator = MaxNLocator(
        nbins=6
    )

    cbar.update_ticks()


# ============================================================
# 9. Layout — matched to Figure ML-03
# ============================================================

fig.subplots_adjust(
    left=0.085,
    right=0.965,

    bottom=0.09,
    top=0.95,

    wspace=0.38,
    hspace=0.38,
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
print("=" * 110)
print("FIGURE ML-04 — GPR PREDICTIVE UNCERTAINTY QC")
print("=" * 110)


print(
    qc_df.to_string(
        index=False
    )
)


print("\nSaved:")
print(
    QC_OUTPUT
)


print(
    "\nNo GPR models were retrained."
)

print(
    "All uncertainty values were read directly from:"
)

print(
    GRID_PATH
)


plt.close(
    fig
)
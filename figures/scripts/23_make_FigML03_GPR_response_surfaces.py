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
# Script: 23_make_FigML03_GPR_response_surfaces.py
# Purpose: Generate continuous GPR response maps across the energy–tilt design space.
#
# Research note:
#   This script belongs to the reproducibility package for my doctoral
#   research on simulation-guided low-energy nitrogen implantation in diamond.
#   Numerical settings used for manuscript results are intentionally kept
#   explicit in the script rather than hidden behind external configuration.
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)
from sklearn.exceptions import ConvergenceWarning

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

DATA_PATH = (
    ROOT
    / "data"
    / "master_condition_table_core.csv"
)

RENDER_DIR = HERE.parent / "rendered"
SOURCE_DIR = HERE.parent / "source_data"

OUTPUT_BASE = RENDER_DIR / "Figure_ML03_GPR_Response_Surfaces"
GRID_OUTPUT = SOURCE_DIR / "Figure_ML03_GPR_Response_Surface_Grid.csv"
DIAGNOSTIC_OUTPUT = SOURCE_DIR / "Figure_ML03_GPR_FullFit_Diagnostics.csv"
QC_OUTPUT = SOURCE_DIR / "Figure_ML03_GPR_Response_Surface_QC.csv"


# ============================================================
# 3. Targets
# ============================================================

TARGETS = [
    {
        "panel": "(a)",
        "key": "depth",
        "title": "Median implantation depth",
        "column": "p1_depth_median_nm",
        "cbar_label": "Median depth (nm)",
        "contour_decimals": 1,
    },

    {
        "panel": "(b)",
        "key": "shallow",
        "title": "Shallow fraction (0–5 nm)",
        "column": "p1_frac_0_5nm",
        "cbar_label": "Shallow fraction",
        "contour_decimals": 1,
    },

    {
        "panel": "(c)",
        "key": "sp2",
        "title": "Near-surface sp² fraction (0–2 nm)",
        "column": "p2_C_sp2_frac_0_2nm_trimmed_surface",
        "cbar_label": "Near-surface sp² fraction",
        "contour_decimals": 3,
    },

    {
        "panel": "(d)",
        "key": "vacancy",
        "title": "Vacancy count",
        "column": "p2_vac_count",
        "cbar_label": "Vacancy count",
        "contour_decimals": 0,
    },
]


# ============================================================
# 4. SAME GPR architecture used in crossed holdout
# ============================================================

def make_gpr():

    kernel = (
        ConstantKernel(
            constant_value=1.0,
            constant_value_bounds=(1e-2, 1e2),
        )
        *
        RBF(
            length_scale=[1.0, 1.0],
            length_scale_bounds=(1e-2, 1e2),
        )
        +
        WhiteKernel(
            noise_level=1e-2,
            noise_level_bounds=(1e-6, 1e1),
        )
    )

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=42,
    )


# ============================================================
# 5. Load condition-level MD dataset
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


required_columns = (
    [
        "E",
        "A",
    ]
    +
    [
        target["column"]
        for target in TARGETS
    ]
)


missing = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


if len(df) != 100:

    raise ValueError(
        f"Expected 100 condition-level rows, "
        f"found {len(df)}."
    )


if df[
    [
        "E",
        "A",
    ]
].duplicated().any():

    raise ValueError(
        "Duplicate E-angle conditions detected."
    )


print("\nQC PASSED")
print(
    "Condition-level dataset: "
    "100 unique E-angle conditions"
)

print(
    f"Energy range: "
    f"{df['E'].min():.2f}–"
    f"{df['E'].max():.2f} keV"
)

print(
    f"Tilt range: "
    f"{df['A'].min():.2f}–"
    f"{df['A'].max():.2f}°"
)


# ============================================================
# 6. Input matrix + full-data scaling
# ============================================================

X = df[
    [
        "E",
        "A",
    ]
].to_numpy(
    dtype=float
)


scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# 7. Dense E-theta design-space grid
# ============================================================

N_ENERGY = 201
N_ANGLE = 241


energy_grid = np.linspace(
    0.5,
    5.0,
    N_ENERGY,
)


angle_grid = np.linspace(
    0.0,
    30.0,
    N_ANGLE,
)


ANGLE_GRID, ENERGY_GRID = np.meshgrid(
    angle_grid,
    energy_grid,
)


X_grid = np.column_stack(
    (
        ENERGY_GRID.ravel(),
        ANGLE_GRID.ravel(),
    )
)


X_grid_scaled = scaler.transform(
    X_grid
)


# ============================================================
# 8. Numerical output containers
# ============================================================

surface_data = pd.DataFrame(
    {
        "energy_keV":
            ENERGY_GRID.ravel(),

        "tilt_deg":
            ANGLE_GRID.ravel(),
    }
)


prediction_surfaces = {}

diagnostic_rows = []

qc_rows = []


# ============================================================
# 9. Full-data GPR fit for each target
# ============================================================

for target in TARGETS:

    print("\n" + "=" * 90)

    print(
        f"FULL-DATA GPR: "
        f"{target['title']}"
    )

    print("=" * 90)


    y = df[
        target["column"]
    ].to_numpy(
        dtype=float
    )


    model = make_gpr()


    with warnings.catch_warnings(
        record=True
    ) as caught:

        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        model.fit(
            X_scaled,
            y,
        )


    n_warnings = sum(
        issubclass(
            warning.category,
            ConvergenceWarning,
        )
        for warning
        in caught
    )


    pred_mean, pred_std = model.predict(
        X_grid_scaled,
        return_std=True,
    )


    mean_surface = pred_mean.reshape(
        ENERGY_GRID.shape
    )


    std_surface = pred_std.reshape(
        ENERGY_GRID.shape
    )


    prediction_surfaces[
        target["key"]
    ] = {
        "mean":
            mean_surface,

        "std":
            std_surface,
    }


    # --------------------------------------------------------
    # RAW values are saved.
    #
    # No clipping or physical truncation is applied here.
    # --------------------------------------------------------

    surface_data[
        f"{target['key']}_mean"
    ] = pred_mean


    surface_data[
        f"{target['key']}_std"
    ] = pred_std


    optimized_kernel = (
        model.kernel_
    )


    # --------------------------------------------------------
    # Physical-bound QC
    # --------------------------------------------------------

    if target["key"] in [
        "shallow",
        "sp2",
    ]:

        outside_count = int(
            np.sum(
                (pred_mean < 0.0)
                |
                (pred_mean > 1.0)
            )
        )

    elif target["key"] in [
        "depth",
        "vacancy",
    ]:

        outside_count = int(
            np.sum(
                pred_mean < 0.0
            )
        )

    else:

        outside_count = 0


    outside_fraction = (
        outside_count
        /
        len(pred_mean)
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "Optimized kernel:"
    )

    print(
        optimized_kernel
    )


    print(
        f"Prediction mean range: "
        f"{pred_mean.min():.6g} to "
        f"{pred_mean.max():.6g}"
    )


    print(
        f"Prediction std range : "
        f"{pred_std.min():.6g} to "
        f"{pred_std.max():.6g}"
    )


    print(
        f"Physical-bound violations: "
        f"{outside_count} / "
        f"{len(pred_mean)} "
        f"({100 * outside_fraction:.2f}%)"
    )


    print(
        f"Convergence warnings : "
        f"{n_warnings}"
    )


    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    diagnostic_rows.append(
        {
            "target":
                target["key"],

            "optimized_kernel":
                str(
                    optimized_kernel
                ),

            "prediction_mean_min":
                float(
                    np.min(
                        pred_mean
                    )
                ),

            "prediction_mean_max":
                float(
                    np.max(
                        pred_mean
                    )
                ),

            "prediction_std_min":
                float(
                    np.min(
                        pred_std
                    )
                ),

            "prediction_std_max":
                float(
                    np.max(
                        pred_std
                    )
                ),

            "convergence_warnings":
                n_warnings,
        }
    )


    qc_rows.append(
        {
            "target":
                target["key"],

            "grid_points":
                len(
                    pred_mean
                ),

            "raw_prediction_min":
                float(
                    np.min(
                        pred_mean
                    )
                ),

            "raw_prediction_max":
                float(
                    np.max(
                        pred_mean
                    )
                ),

            "physical_bound_violations":
                outside_count,

            "physical_bound_violation_fraction":
                outside_fraction,
        }
    )


# ============================================================
# 10. Save RAW numerical surfaces BEFORE visualization
# ============================================================

surface_data.to_csv(
    GRID_OUTPUT,
    index=False,
)


diagnostic_df = pd.DataFrame(
    diagnostic_rows
)


diagnostic_df.to_csv(
    DIAGNOSTIC_OUTPUT,
    index=False,
)


qc_df = pd.DataFrame(
    qc_rows
)


qc_df.to_csv(
    QC_OUTPUT,
    index=False,
)


# ============================================================
# 11. Helper: clean contour levels
# ============================================================

def clean_contour_levels(
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


    # Keep only levels actually inside the data range.
    levels = levels[
        (levels > zmin)
        &
        (levels < zmax)
    ]


    # Fallback if automatic locator creates too few levels.
    if len(levels) < 2:

        levels = np.linspace(
            zmin,
            zmax,
            5,
        )[1:-1]


    return levels


# ============================================================
# 12. Figure
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

    # --------------------------------------------------------
    # RAW model surface
    # --------------------------------------------------------

    Z_raw = prediction_surfaces[
        target["key"]
    ]["mean"]


    # --------------------------------------------------------
    # DISPLAY-ONLY physical clipping
    #
    # IMPORTANT:
    # Raw model values remain unchanged in CSV outputs.
    #
    # Standard Gaussian-process regression is unbounded.
    # Shallow fraction therefore exhibits a small overshoot
    # above 1.0.
    #
    # For the response-map visualization only, bounded
    # fractional quantities are displayed inside [0, 1].
    # --------------------------------------------------------

    if target["key"] in [
        "shallow",
        "sp2",
    ]:

        Z_display = np.clip(
            Z_raw,
            0.0,
            1.0,
        )

    else:

        Z_display = (
            Z_raw.copy()
        )


    # ========================================================
    # Filled surface
    # ========================================================

    if target["key"] == "shallow":

        # Physical fraction range.
        filled_levels = np.linspace(
            0.0,
            1.0,
            81,
        )


        contourf = ax.contourf(
            ANGLE_GRID,
            ENERGY_GRID,
            Z_display,

            levels=filled_levels,

            cmap="viridis",

            vmin=0.0,
            vmax=1.0,

            antialiased=True,
        )


        contour_levels = np.array(
            [
                0.2,
                0.4,
                0.6,
                0.8,
            ]
        )


    else:

        contourf = ax.contourf(
            ANGLE_GRID,
            ENERGY_GRID,
            Z_display,

            levels=80,

            cmap="viridis",

            antialiased=True,
        )


        contour_levels = (
            clean_contour_levels(
                Z_display,
                nbins=4,
            )
        )


    # ========================================================
    # Sparse contour guides
    # ========================================================

    contours = ax.contour(
        ANGLE_GRID,
        ENERGY_GRID,
        Z_display,

        levels=contour_levels,

        colors="black",

        linewidths=0.50,

        alpha=0.48,

        zorder=2,
    )


    ax.clabel(
        contours,

        inline=True,

        inline_spacing=4,

        fontsize=6.5,

        fmt=(
            f"%.{target['contour_decimals']}f"
        ),
    )


    # ========================================================
    # Original 100 MD conditions
    #
    # Deliberately subtle:
    # show sampling density without dominating the surface.
    # ========================================================

    ax.scatter(
        df["A"],
        df["E"],

        s=5,

        facecolors="white",

        edgecolors="black",

        linewidths=0.25,

        alpha=0.55,

        zorder=4,
    )


    # ========================================================
    # Axes
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
        target["cbar_label"],
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


    # Explicit physical scale for shallow fraction.
    if target["key"] == "shallow":

        cbar.set_ticks(
            [
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ]
        )


# ============================================================
# 13. Layout
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
# 14. Save
# ============================================================

save_paper_figure(
    fig,
    OUTPUT_BASE,
)


# ============================================================
# 15. Final console summary
# ============================================================

print("\n")
print("=" * 100)
print("FIGURE ML-03 RESPONSE-SURFACE QC")
print("=" * 100)

print(
    qc_df.to_string(
        index=False
    )
)


print("\nSaved:")
print(
    GRID_OUTPUT
)

print(
    DIAGNOSTIC_OUTPUT
)

print(
    QC_OUTPUT
)


print(
    "\nDense grid points:",
    len(
        surface_data
    ),
)


print(
    "\nIMPORTANT:"
)

print(
    "Figure display uses physical clipping only "
    "for bounded fractional quantities."
)

print(
    "All CSV prediction values remain RAW "
    "unmodified GPR outputs."
)


plt.close(
    fig
)
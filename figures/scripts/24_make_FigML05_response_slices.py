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
# Script: 24_make_FigML05_response_slices.py
# Purpose: Generate full-fit angular response slices for representative implantation energies.
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

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.interpolate import PchipInterpolator

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

DATA_PATH = (
    ROOT
    / "data"
    / "master_condition_table_core.csv"
)

RENDER_DIR = HERE.parent / "rendered"
SOURCE_DIR = HERE.parent / "source_data"
OUTPUT_BASE = RENDER_DIR / "Figure_ML05_Response_Slices"
CURVE_OUTPUT = SOURCE_DIR / "Figure_ML05_Response_Slices_Curves.csv"
DIAGNOSTIC_OUTPUT = SOURCE_DIR / "Figure_ML05_Response_Slices_Diagnostics.csv"


# ============================================================
# 3. Representative energies
#
# Pre-defined low / middle / high interior energies.
# These were not selected from target outcomes.
# ============================================================

SLICE_ENERGIES = [
    1.5,
    3.0,
    4.5,
]


# ============================================================
# 4. Energy colors
#
# Color encodes ENERGY.
# Line style / marker encodes METHOD.
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

        "column":
            "p1_depth_median_nm",

        "ylabel":
            "Median depth (nm)",
    },

    {
        "panel": "(b)",
        "key": "shallow",

        "title":
            "Shallow fraction (0–5 nm)",

        "column":
            "p1_frac_0_5nm",

        "ylabel":
            "Shallow fraction",
    },

    {
        "panel": "(c)",
        "key": "sp2",

        "title":
            "Near-surface sp² fraction (0–2 nm)",

        "column":
            "p2_C_sp2_frac_0_2nm_trimmed_surface",

        "ylabel":
            "Near-surface sp² fraction",
    },

    {
        "panel": "(d)",
        "key": "vacancy",

        "title":
            "Vacancy count",

        "column":
            "p2_vac_count",

        "ylabel":
            "Vacancy count",
    },
]


# ============================================================
# 6. SAME GPR architecture used in crossed-holdout benchmark
#    and Figure ML-03.
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
# 7. Load MD condition-level dataset
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
    column
    for column in required_columns
    if column not in df.columns
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


df = (
    df
    .sort_values(
        [
            "E",
            "A",
        ]
    )
    .reset_index(
        drop=True
    )
)


energies_all = np.sort(
    df["E"].unique()
)


angles_all = np.sort(
    df["A"].unique()
)


for energy in SLICE_ENERGIES:

    if not np.any(
        np.isclose(
            energies_all,
            energy,
        )
    ):

        raise ValueError(
            f"Requested slice energy "
            f"{energy} keV is not an "
            f"original MD energy level."
        )


print("\nQC PASSED")

print(
    "Condition-level dataset: "
    "100 unique E-angle conditions"
)

print(
    "Representative energies:",
    SLICE_ENERGIES,
)


# ============================================================
# 8. Full-data input scaling
#
# SAME scaler concept as Figure ML-03.
# One scaler is fit on all 100 condition-level inputs.
# ============================================================

X_train = df[
    [
        "E",
        "A",
    ]
].to_numpy(
    dtype=float
)


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


# ============================================================
# 9. Dense angular grid for slice curves
# ============================================================

ANGLE_FINE = np.linspace(
    0.0,
    30.0,
    301,
)


# ============================================================
# 10. PCHIP slice
#
# At the chosen energies (which are original energy nodes),
# the two possible sequential interpolation routes collapse
# to the same angle-wise PCHIP curve.
#
# Therefore we interpolate target vs tilt directly at each
# fixed original energy.
# ============================================================

def pchip_at_fixed_energy(
    data,
    target_column,
    fixed_energy,
    angle_grid,
):

    sub = (
        data[
            np.isclose(
                data["E"],
                fixed_energy,
            )
        ]
        .sort_values(
            "A"
        )
    )


    x = sub[
        "A"
    ].to_numpy(
        dtype=float
    )


    y = sub[
        target_column
    ].to_numpy(
        dtype=float
    )


    interpolator = PchipInterpolator(
        x,
        y,
        extrapolate=False,
    )


    prediction = interpolator(
        angle_grid
    )


    return prediction


# ============================================================
# 11. MD slice helper
# ============================================================

def get_md_slice(
    data,
    target_column,
    fixed_energy,
):

    sub = (
        data[
            np.isclose(
                data["E"],
                fixed_energy,
            )
        ]
        .sort_values(
            "A"
        )
    )


    return (
        sub[
            "A"
        ].to_numpy(
            dtype=float
        ),

        sub[
            target_column
        ].to_numpy(
            dtype=float
        ),
    )


# ============================================================
# 12. Fit one full-data GPR per target
# ============================================================

curve_rows = []

diagnostic_rows = []


for target in TARGETS:

    print("\n" + "=" * 90)

    print(
        f"FULL-DATA GPR: "
        f"{target['title']}"
    )

    print("=" * 90)


    y_train = df[
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
            X_train_scaled,
            y_train,
        )


    n_warnings = sum(
        issubclass(
            warning.category,
            ConvergenceWarning,
        )
        for warning
        in caught
    )


    print(
        "Optimized kernel:"
    )

    print(
        model.kernel_
    )


    print(
        f"Convergence warnings: "
        f"{n_warnings}"
    )


    diagnostic_rows.append(
        {
            "target":
                target["key"],

            "optimized_kernel":
                str(
                    model.kernel_
                ),

            "convergence_warnings":
                n_warnings,
        }
    )


    # --------------------------------------------------------
    # Predict three representative energy slices
    # --------------------------------------------------------

    for energy in SLICE_ENERGIES:

        X_pred = np.column_stack(
            (
                np.full(
                    len(
                        ANGLE_FINE
                    ),
                    energy,
                    dtype=float,
                ),

                ANGLE_FINE,
            )
        )


        X_pred_scaled = scaler.transform(
            X_pred
        )


        gpr_mean, gpr_std = model.predict(
            X_pred_scaled,
            return_std=True,
        )


        pchip_mean = pchip_at_fixed_energy(
            data=df,

            target_column=
                target["column"],

            fixed_energy=
                energy,

            angle_grid=
                ANGLE_FINE,
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # NO PHYSICAL CLIPPING HERE.
        #
        # These are raw GPR and PCHIP outputs because this
        # figure directly compares model responses.
        # ----------------------------------------------------

        for (
            angle,
            mean,
            std,
            pchip,
        ) in zip(
            ANGLE_FINE,
            gpr_mean,
            gpr_std,
            pchip_mean,
        ):

            curve_rows.append(
                {
                    "target":
                        target["key"],

                    "energy_keV":
                        energy,

                    "tilt_deg":
                        angle,

                    "gpr_mean_raw":
                        mean,

                    "gpr_std":
                        std,

                    "pchip_mean_raw":
                        pchip,
                }
            )


# ============================================================
# 13. Save curve data BEFORE plotting
# ============================================================

curve_df = pd.DataFrame(
    curve_rows
)


curve_df.to_csv(
    CURVE_OUTPUT,
    index=False,
)


diagnostic_df = pd.DataFrame(
    diagnostic_rows
)


diagnostic_df.to_csv(
    DIAGNOSTIC_OUTPUT,
    index=False,
)


# ============================================================
# 14. Figure
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

    target_key = (
        target["key"]
    )


    # ========================================================
    # Three energy slices
    # ========================================================

    for energy in SLICE_ENERGIES:

        color = (
            ENERGY_COLORS[
                energy
            ]
        )


        sub_curve = (
            curve_df[
                (
                    curve_df[
                        "target"
                    ]
                    ==
                    target_key
                )
                &
                (
                    np.isclose(
                        curve_df[
                            "energy_keV"
                        ],
                        energy,
                    )
                )
            ]
            .sort_values(
                "tilt_deg"
            )
        )


        x = sub_curve[
            "tilt_deg"
        ].to_numpy(
            dtype=float
        )


        gpr_mean = sub_curve[
            "gpr_mean_raw"
        ].to_numpy(
            dtype=float
        )


        gpr_std = sub_curve[
            "gpr_std"
        ].to_numpy(
            dtype=float
        )


        pchip_mean = sub_curve[
            "pchip_mean_raw"
        ].to_numpy(
            dtype=float
        )


        x_md, y_md = get_md_slice(
            data=df,

            target_column=
                target["column"],

            fixed_energy=
                energy,
        )


        # ====================================================
        # GPR predictive ±1 SD
        # ====================================================

        ax.fill_between(
            x,

            gpr_mean - gpr_std,
            gpr_mean + gpr_std,

            facecolor=color,

            alpha=0.10,

            edgecolor="none",

            zorder=1,
        )


        # ====================================================
        # PCHIP
        # ====================================================

        ax.plot(
            x,
            pchip_mean,

            color=color,

            linestyle="--",

            linewidth=1.35,

            zorder=2,
        )


        # ====================================================
        # GPR
        # ====================================================

        ax.plot(
            x,
            gpr_mean,

            color=color,

            linestyle="-",

            linewidth=1.65,

            zorder=3,
        )


        # ====================================================
        # MD condition means
        # ====================================================

        ax.scatter(
            x_md,
            y_md,

            s=18,

            facecolor="white",

            edgecolor=color,

            linewidth=0.9,

            zorder=4,
        )


    # ========================================================
    # Axes
    # ========================================================

    ax.set_xlim(
        0,
        30,
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


    ax.set_xlabel(
        "Tilt angle (°)"
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

        alpha=0.22,

        zorder=0,
    )


    # ========================================================
    # Target-specific Y limits
    #
    # NOTE:
    # No hard [0,1] clipping for shallow fraction.
    # Raw GPR overshoot remains visible.
    # ========================================================

    all_values = []


    for energy in SLICE_ENERGIES:

        sub_curve = curve_df[
            (
                curve_df[
                    "target"
                ]
                ==
                target_key
            )
            &
            (
                np.isclose(
                    curve_df[
                        "energy_keV"
                    ],
                    energy,
                )
            )
        ]


        gpr_mean = sub_curve[
            "gpr_mean_raw"
        ].to_numpy(
            dtype=float
        )


        gpr_std = sub_curve[
            "gpr_std"
        ].to_numpy(
            dtype=float
        )


        pchip_mean = sub_curve[
            "pchip_mean_raw"
        ].to_numpy(
            dtype=float
        )


        all_values.extend(
            (
                gpr_mean
                -
                gpr_std
            ).tolist()
        )


        all_values.extend(
            (
                gpr_mean
                +
                gpr_std
            ).tolist()
        )


        all_values.extend(
            pchip_mean.tolist()
        )


        _, md_values = get_md_slice(
            data=df,

            target_column=
                target["column"],

            fixed_energy=
                energy,
        )


        all_values.extend(
            md_values.tolist()
        )


    all_values = np.asarray(
        all_values,
        dtype=float,
    )


    ymin = float(
        np.nanmin(
            all_values
        )
    )


    ymax = float(
        np.nanmax(
            all_values
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
        0.06
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


    # Physical quantities are non-negative.
    lower = max(
        0.0,
        lower,
    )


    ax.set_ylim(
        lower,
        upper,
    )


# ============================================================
# 15. Shared legend
#
# Two logical rows:
#
# Row 1 = method / uncertainty encoding
# Row 2 = energy colors
# ============================================================

METHOD_HANDLES = [

    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="None",

        markerfacecolor="white",

        markeredgecolor="black",

        markeredgewidth=0.9,

        markersize=5,

        label="MD",
    ),


    Line2D(
        [0],
        [0],

        color="black",

        linewidth=1.65,

        linestyle="-",

        label="GPR",
    ),


    Line2D(
        [0],
        [0],

        color="black",

        linewidth=1.35,

        linestyle="--",

        label="PCHIP",
    ),


    Patch(
        facecolor="gray",

        alpha=0.14,

        edgecolor="none",

        label="GPR ±1 predictive SD",
    ),
]


ENERGY_HANDLES = [

    Line2D(
        [0],
        [0],

        color=ENERGY_COLORS[
            1.5
        ],

        linewidth=2.2,

        label="1.5 keV",
    ),


    Line2D(
        [0],
        [0],

        color=ENERGY_COLORS[
            3.0
        ],

        linewidth=2.2,

        label="3.0 keV",
    ),


    Line2D(
        [0],
        [0],

        color=ENERGY_COLORS[
            4.5
        ],

        linewidth=2.2,

        label="4.5 keV",
    ),
]


# ------------------------------------------------------------
# Legend row 1
# ------------------------------------------------------------

legend_methods = fig.legend(
    handles=METHOD_HANDLES,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.995,
    ),

    ncol=4,

    frameon=False,

    handlelength=2.0,

    columnspacing=1.5,

    handletextpad=0.55,
)


fig.add_artist(
    legend_methods
)


# ------------------------------------------------------------
# Legend row 2
# ------------------------------------------------------------

fig.legend(
    handles=ENERGY_HANDLES,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.957,
    ),

    ncol=3,

    frameon=False,

    handlelength=2.4,

    columnspacing=1.7,

    handletextpad=0.55,
)


# ============================================================
# 16. Layout
#
# No explanatory sentence at bottom.
# That belongs in the manuscript caption.
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
print("=" * 110)
print("FIGURE ML-05 — REPRESENTATIVE RESPONSE SLICES")
print("=" * 110)


print(
    diagnostic_df.to_string(
        index=False
    )
)


print("\nSaved:")
print(
    CURVE_OUTPUT
)

print(
    DIAGNOSTIC_OUTPUT
)


print(
    "\nIMPORTANT:"
)

print(
    "No physical clipping was applied "
    "to GPR or PCHIP predictions."
)

print(
    "All displayed curves are raw model outputs."
)

print(
    "Each target uses one full-data GPR fit "
    "for all three representative energies."
)


plt.close(
    fig
)
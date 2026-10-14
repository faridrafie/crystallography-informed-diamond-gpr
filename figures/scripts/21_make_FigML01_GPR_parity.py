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
# Script: 21_make_FigML01_GPR_parity.py
# Purpose: Generate the crossed-holdout GPR parity figure.
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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from paper_plot_style import (
    apply_paper_style,
    add_panel_label,
    save_paper_figure,
    COLORS,
    DOUBLE_COLUMN_IN,
    FONT_ANNOTATION,
)


# ============================================================
# 1. Apply frozen manuscript style
# ============================================================

apply_paper_style()


# ============================================================
# 2. Paths
# ============================================================

HERE = Path(__file__).resolve().parent

ROOT = (
    HERE
    .parent
    .parent
)

DATA_DIR = (
    ROOT
    / "analysis"
    / "baseline"
    / "model_comparison"
)

RENDER_DIR = HERE.parent / "rendered"
SOURCE_DIR = HERE.parent / "source_data"

OUTPUT_BASE = RENDER_DIR / "Figure_ML01_GPR_CrossedHoldout_Parity"
METRIC_OUTPUT = SOURCE_DIR / "Figure_ML01_GPR_CrossedHoldout_Parity_metrics.csv"


# ============================================================
# 3. Frozen target definitions
# ============================================================

TARGETS = [

    {
        "panel": "(a)",
        "title": "Median implantation depth",

        "file":
            "depth_gpr_vs_pchip_crossed_predictions.csv",

        "truth":
            "true_depth_nm",

        "prediction":
            "gpr_prediction_nm",

        "xlabel":
            "MD ground truth (nm)",

        "ylabel":
            "GPR prediction (nm)",

        "metric_digits":
            3,
    },

    {
        "panel": "(b)",
        "title": "Shallow fraction (0–5 nm)",

        "file":
            "shallow_gpr_vs_pchip_crossed_predictions.csv",

        "truth":
            "true_shallow_fraction",

        "prediction":
            "gpr_prediction",

        "xlabel":
            "MD ground truth",

        "ylabel":
            "GPR prediction",

        "metric_digits":
            3,
    },

    {
        "panel": "(c)",
        "title": r"Near-surface sp² fraction (0–2 nm)",

        "file":
            "sp2_gpr_vs_pchip_crossed_predictions.csv",

        "truth":
            "true_sp2_fraction",

        "prediction":
            "gpr_prediction",

        "xlabel":
            r"MD ground truth",

        "ylabel":
            r"GPR prediction",

        "metric_digits":
            4,
    },

    {
        "panel": "(d)",
        "title": "Vacancy count",

        "file":
            "vacancy_gpr_vs_pchip_crossed_predictions.csv",

        "truth":
            "true_vac_count",

        "prediction":
            "gpr_prediction",

        "xlabel":
            "MD ground truth (count)",

        "ylabel":
            "GPR prediction (count)",

        "metric_digits":
            2,
    },
]


# ============================================================
# 4. Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


    r2 = r2_score(
        y_true,
        y_pred,
    )


    return mae, rmse, r2


# ============================================================
# 5. Shared x/y limits and identical ticks
# ============================================================

def make_shared_parity_scale(
    y_true,
    y_pred,
):

    values = np.concatenate(
        [
            np.asarray(
                y_true,
                dtype=float,
            ),

            np.asarray(
                y_pred,
                dtype=float,
            ),
        ]
    )


    vmin = float(
        np.min(
            values
        )
    )

    vmax = float(
        np.max(
            values
        )
    )


    span = (
        vmax
        -
        vmin
    )


    if span <= 0:

        span = 1.0


    lower = (
        vmin
        -
        0.06
        *
        span
    )

    upper = (
        vmax
        +
        0.06
        *
        span
    )


    # All four physical quantities are non-negative.
    lower = max(
        0.0,
        lower,
    )


    locator = MaxNLocator(
        nbins=5,
        min_n_ticks=4,
    )


    ticks = locator.tick_values(
        lower,
        upper,
    )


    ticks = ticks[
        ticks >= 0
    ]


    # Use complete tick interval as final limits.
    lower_final = float(
        ticks[0]
    )

    upper_final = float(
        ticks[-1]
    )


    return (
        lower_final,
        upper_final,
        ticks,
    )


# ============================================================
# 6. Load and validate all four datasets
# ============================================================

data = {}

reference_conditions = None


for target in TARGETS:

    file_path = (
        DATA_DIR
        /
        target["file"]
    )


    if not file_path.exists():

        raise FileNotFoundError(
            file_path
        )


    df = pd.read_csv(
        file_path
    )


    if len(df) != 64:

        raise ValueError(
            f"{target['file']} contains "
            f"{len(df)} rows; expected exactly 64."
        )


    required = [

        "held_energy_keV",
        "held_angle_deg",

        target[
            "truth"
        ],

        target[
            "prediction"
        ],
    ]


    missing = [

        column

        for column
        in required

        if column
        not in df.columns
    ]


    if missing:

        raise ValueError(
            f"Missing columns in "
            f"{target['file']}: "
            f"{missing}"
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
                "The four targets do not use "
                "the exact same crossed-holdout conditions."
            )


    data[
        target["file"]
    ] = df


print("\nQC PASSED")
print(
    "All four targets use the same "
    "64 crossed energy-angle holdout conditions."
)


# ============================================================
# 7. Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,

    figsize=(
        DOUBLE_COLUMN_IN,
        6.20,
    ),
)


axes = axes.ravel()


metric_rows = []


for ax, target in zip(
    axes,
    TARGETS,
):

    df = data[
        target["file"]
    ]


    y_true = df[
        target[
            "truth"
        ]
    ].to_numpy(
        dtype=float
    )


    y_pred = df[
        target[
            "prediction"
        ]
    ].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae, rmse, r2 = (
        calculate_metrics(
            y_true,
            y_pred,
        )
    )


    # --------------------------------------------------------
    # Shared scale
    # --------------------------------------------------------

    lower, upper, ticks = (
        make_shared_parity_scale(
            y_true,
            y_pred,
        )
    )


    # --------------------------------------------------------
    # Identity line
    # --------------------------------------------------------

    ax.plot(
        [
            lower,
            upper,
        ],

        [
            lower,
            upper,
        ],

        linestyle="--",

        color=COLORS[
            "reference"
        ],

        linewidth=1.0,

        zorder=1,
    )


    # --------------------------------------------------------
    # GPR predictions
    # --------------------------------------------------------

    ax.scatter(
        y_true,
        y_pred,

        s=24,

        facecolor=COLORS[
            "gpr"
        ],

        edgecolor="black",

        linewidth=0.35,

        alpha=0.85,

        zorder=3,
    )


    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    ax.set_xlim(
        lower,
        upper,
    )

    ax.set_ylim(
        lower,
        upper,
    )


    ax.set_xticks(
        ticks
    )

    ax.set_yticks(
        ticks
    )


    # True square parity geometry
    ax.set_aspect(
        "equal",
        adjustable="box",
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        target[
            "xlabel"
        ]
    )

    ax.set_ylabel(
        target[
            "ylabel"
        ]
    )


    ax.set_title(
        target[
            "title"
        ],
        pad=5,
        fontweight="normal",
    )


    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    add_panel_label(
        ax,
        target[
            "panel"
        ],
    )


    # --------------------------------------------------------
    # Metrics annotation
    # --------------------------------------------------------

    digits = target[
        "metric_digits"
    ]


    metric_text = (

        rf"$R^2$ = {r2:.3f}"
        "\n"

        +
        f"MAE = {mae:.{digits}f}"
        "\n"

        +
        f"RMSE = {rmse:.{digits}f}"
    )


    ax.text(
        0.05,
        0.95,

        metric_text,

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontsize=FONT_ANNOTATION,

        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 2.0,
        },

        zorder=5,
    )


    # --------------------------------------------------------
    # Minor visual cleanup
    # --------------------------------------------------------

    ax.tick_params(
        which="both",
        top=True,
        right=True,
    )


    # --------------------------------------------------------
    # Save numerical audit
    # --------------------------------------------------------

    metric_rows.append(
        {

            "panel":
                target[
                    "panel"
                ],

            "target":
                target[
                    "title"
                ],

            "n_test":
                len(
                    df
                ),

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2,
        }
    )


# ============================================================
# 8. Consistent spacing
# ============================================================

fig.subplots_adjust(
    left=0.105,
    right=0.985,

    bottom=0.085,
    top=0.955,

    wspace=0.30,
    hspace=0.32,
)


# ============================================================
# 9. Save figure + numerical audit
# ============================================================

metric_df = pd.DataFrame(
    metric_rows
)


metric_df.to_csv(
    METRIC_OUTPUT,
    index=False,
)


save_paper_figure(
    fig,
    OUTPUT_BASE,
)


print(
    "\nMetrics:"
)

print(
    metric_df.to_string(
        index=False
    )
)


print(
    "\nMetric table saved:"
)

print(
    METRIC_OUTPUT
)


plt.close(
    fig
)
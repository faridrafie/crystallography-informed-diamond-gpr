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
# Script: 28_make_FigML07_GPR_generalization_stress_test.py
# Purpose: Generate the LOAO/LOEO median-depth generalization stress-test figure.
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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from paper_plot_style import (
    apply_paper_style,
    add_panel_label,
    save_paper_figure,
    DOUBLE_COLUMN_IN,
    COLORS,
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

LOAO_PRED_PATH = (
    ROOT
    / "analysis"
    / "baseline"
    / "gpr_loao"
    / "gpr_depth_loao_predictions.csv"
)

LOAO_SUMMARY_PATH = (
    ROOT
    / "analysis"
    / "baseline"
    / "gpr_loao"
    / "gpr_depth_loao_summary.csv"
)

LOEO_PRED_PATH = (
    ROOT
    / "analysis"
    / "baseline"
    / "gpr_loeo"
    / "gpr_depth_loeo_predictions.csv"
)

LOEO_SUMMARY_PATH = (
    ROOT
    / "analysis"
    / "baseline"
    / "gpr_loeo"
    / "gpr_depth_loeo_summary.csv"
)

RENDER_DIR = HERE.parent / "rendered"
SOURCE_DIR = HERE.parent / "source_data"
OUTPUT_BASE = RENDER_DIR / "Figure_ML07_GPR_Generalization_Stress_Test"
OUTPUT_SUMMARY = SOURCE_DIR / "Figure_ML07_GPR_Generalization_Stress_Test_Summary.csv"
OUTPUT_FOLD_DATA = SOURCE_DIR / "Figure_ML07_GPR_Generalization_Fold_Metrics.csv"


# ============================================================
# 3. Load existing frozen outputs
# ============================================================

loao_pred = pd.read_csv(
    LOAO_PRED_PATH
)

loao_summary = pd.read_csv(
    LOAO_SUMMARY_PATH
)

loeo_pred = pd.read_csv(
    LOEO_PRED_PATH
)

loeo_summary = pd.read_csv(
    LOEO_SUMMARY_PATH
)


# ============================================================
# 4. QC
# ============================================================

if len(loao_pred) != 100:
    raise ValueError(
        f"LOAO predictions: expected 100 rows, "
        f"found {len(loao_pred)}."
    )

if len(loeo_pred) != 100:
    raise ValueError(
        f"LOEO predictions: expected 100 rows, "
        f"found {len(loeo_pred)}."
    )

if len(loao_summary) != 10:
    raise ValueError(
        f"LOAO summary: expected 10 folds, "
        f"found {len(loao_summary)}."
    )

if len(loeo_summary) != 10:
    raise ValueError(
        f"LOEO summary: expected 10 folds, "
        f"found {len(loeo_summary)}."
    )


required_loao_pred = [
    "E",
    "A",
    "true_depth",
    "predicted_depth",
    "prediction_std",
    "held_angle_deg",
]

required_loeo_pred = [
    "E",
    "A",
    "true_depth",
    "predicted_depth",
    "prediction_std",
    "held_energy_keV",
]

required_loao_summary = [
    "held_angle_deg",
    "MAE_nm",
    "RMSE_nm",
    "R2",
]

required_loeo_summary = [
    "held_energy_keV",
    "MAE_nm",
    "RMSE_nm",
    "R2",
]


for col in required_loao_pred:
    if col not in loao_pred.columns:
        raise ValueError(
            f"Missing LOAO prediction column: {col}"
        )

for col in required_loeo_pred:
    if col not in loeo_pred.columns:
        raise ValueError(
            f"Missing LOEO prediction column: {col}"
        )

for col in required_loao_summary:
    if col not in loao_summary.columns:
        raise ValueError(
            f"Missing LOAO summary column: {col}"
        )

for col in required_loeo_summary:
    if col not in loeo_summary.columns:
        raise ValueError(
            f"Missing LOEO summary column: {col}"
        )


print("\nQC PASSED")
print("LOAO: 10 folds / 100 predictions")
print("LOEO: 10 folds / 100 predictions")


# ============================================================
# 5. Sort folds
# ============================================================

loao_summary = (
    loao_summary
    .sort_values(
        "held_angle_deg"
    )
    .reset_index(
        drop=True
    )
)

loeo_summary = (
    loeo_summary
    .sort_values(
        "held_energy_keV"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 6. Define interior interpolation vs edge extrapolation
#
# LOAO:
#   0° and 30° are edge holdouts.
#
# LOEO:
#   0.5 and 5.0 keV are edge holdouts.
# ============================================================

loao_min = float(
    loao_summary[
        "held_angle_deg"
    ].min()
)

loao_max = float(
    loao_summary[
        "held_angle_deg"
    ].max()
)

loeo_min = float(
    loeo_summary[
        "held_energy_keV"
    ].min()
)

loeo_max = float(
    loeo_summary[
        "held_energy_keV"
    ].max()
)


loao_pred["regime"] = np.where(
    np.isclose(
        loao_pred["held_angle_deg"],
        loao_min,
    )
    |
    np.isclose(
        loao_pred["held_angle_deg"],
        loao_max,
    ),
    "edge_extrapolation",
    "interior_interpolation",
)


loeo_pred["regime"] = np.where(
    np.isclose(
        loeo_pred["held_energy_keV"],
        loeo_min,
    )
    |
    np.isclose(
        loeo_pred["held_energy_keV"],
        loeo_max,
    ),
    "edge_extrapolation",
    "interior_interpolation",
)


# ============================================================
# 7. Pooled metric helper
# ============================================================

def pooled_metrics(df):

    y_true = df[
        "true_depth"
    ].to_numpy(
        dtype=float
    )

    y_pred = df[
        "predicted_depth"
    ].to_numpy(
        dtype=float
    )

    y_std = df[
        "prediction_std"
    ].to_numpy(
        dtype=float
    )


    residual = (
        y_true
        -
        y_pred
    )

    abs_error = np.abs(
        residual
    )


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


    within_1sigma = np.mean(
        abs_error
        <=
        y_std
    )

    within_2sigma = np.mean(
        abs_error
        <=
        2.0
        *
        y_std
    )


    return {
        "n":
            len(df),

        "MAE_nm":
            float(mae),

        "RMSE_nm":
            float(rmse),

        "R2":
            float(r2),

        "fraction_within_1sigma":
            float(within_1sigma),

        "fraction_within_2sigma":
            float(within_2sigma),
    }


# ============================================================
# 8. Build pooled audit table
# ============================================================

summary_rows = []


for protocol_name, pred_df in [
    ("LOAO", loao_pred),
    ("LOEO", loeo_pred),
]:

    # --------------------------------------------------------
    # All
    # --------------------------------------------------------

    all_stats = pooled_metrics(
        pred_df
    )

    summary_rows.append(
        {
            "protocol":
                protocol_name,

            "regime":
                "all",

            **all_stats,
        }
    )


    # --------------------------------------------------------
    # Interior
    # --------------------------------------------------------

    interior_df = pred_df[
        pred_df["regime"]
        ==
        "interior_interpolation"
    ]

    interior_stats = pooled_metrics(
        interior_df
    )

    summary_rows.append(
        {
            "protocol":
                protocol_name,

            "regime":
                "interior_interpolation",

            **interior_stats,
        }
    )


    # --------------------------------------------------------
    # Edge
    # --------------------------------------------------------

    edge_df = pred_df[
        pred_df["regime"]
        ==
        "edge_extrapolation"
    ]

    edge_stats = pooled_metrics(
        edge_df
    )

    summary_rows.append(
        {
            "protocol":
                protocol_name,

            "regime":
                "edge_extrapolation",

            **edge_stats,
        }
    )


audit_df = pd.DataFrame(
    summary_rows
)


audit_df.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ============================================================
# 9. Combined fold-level table
# ============================================================

loao_fold = loao_summary[
    [
        "held_angle_deg",
        "MAE_nm",
        "RMSE_nm",
        "R2",
    ]
].copy()


loao_fold["protocol"] = "LOAO"


loao_fold = loao_fold.rename(
    columns={
        "held_angle_deg":
            "held_coordinate",
    }
)


loeo_fold = loeo_summary[
    [
        "held_energy_keV",
        "MAE_nm",
        "RMSE_nm",
        "R2",
    ]
].copy()


loeo_fold["protocol"] = "LOEO"


loeo_fold = loeo_fold.rename(
    columns={
        "held_energy_keV":
            "held_coordinate",
    }
)


fold_df = pd.concat(
    [
        loao_fold,
        loeo_fold,
    ],
    ignore_index=True,
)


fold_df.to_csv(
    OUTPUT_FOLD_DATA,
    index=False,
)


# ============================================================
# 10. Retrieve audit rows
# ============================================================

def get_audit_row(
    protocol,
    regime,
):

    row = audit_df[
        (
            audit_df["protocol"]
            ==
            protocol
        )
        &
        (
            audit_df["regime"]
            ==
            regime
        )
    ]


    if len(row) != 1:
        raise ValueError(
            f"Audit row not unique: "
            f"{protocol}, {regime}"
        )


    return row.iloc[0]


loao_all = get_audit_row(
    "LOAO",
    "all",
)

loao_interior = get_audit_row(
    "LOAO",
    "interior_interpolation",
)

loao_edge = get_audit_row(
    "LOAO",
    "edge_extrapolation",
)


loeo_all = get_audit_row(
    "LOEO",
    "all",
)

loeo_interior = get_audit_row(
    "LOEO",
    "interior_interpolation",
)

loeo_edge = get_audit_row(
    "LOEO",
    "edge_extrapolation",
)


# ============================================================
# 11. Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,

    figsize=(
        DOUBLE_COLUMN_IN,
        3.15,
    ),
)


# ============================================================
# 12. Panel A — Leave-one-angle-out
# ============================================================

ax = axes[0]


angles = loao_summary[
    "held_angle_deg"
].to_numpy(
    dtype=float
)

mae = loao_summary[
    "MAE_nm"
].to_numpy(
    dtype=float
)

rmse = loao_summary[
    "RMSE_nm"
].to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# Edge extrapolation shading
# ------------------------------------------------------------

ax.axvspan(
    -0.8,
    1.5,

    facecolor="#E6E6E6",

    edgecolor="none",

    alpha=0.65,

    zorder=0,
)


ax.axvspan(
    28.5,
    30.8,

    facecolor="#E6E6E6",

    edgecolor="none",

    alpha=0.65,

    zorder=0,
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

ax.plot(
    angles,
    mae,

    marker="o",

    markersize=4.6,

    linewidth=1.35,

    color=COLORS["gpr"],

    markerfacecolor="white",

    markeredgecolor=COLORS["gpr"],

    markeredgewidth=0.9,

    label="MAE",

    zorder=3,
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

ax.plot(
    angles,
    rmse,

    marker="s",

    markersize=4.4,

    linewidth=1.35,

    linestyle="--",

    color=COLORS["pchip"],

    markerfacecolor="white",

    markeredgecolor=COLORS["pchip"],

    markeredgewidth=0.9,

    label="RMSE",

    zorder=3,
)


# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

ax.set_xlim(
    -0.5,
    30.5,
)


ax.set_xticks(
    [
        0,
        3,
        7,
        10,
        15,
        20,
        25,
        30,
    ]
)


ax.set_xlabel(
    "Held-out tilt angle (°)"
)


ax.set_ylabel(
    "Depth prediction error (nm)"
)


ax.set_title(
    "Leave-one-angle-out",
    pad=6,
)


add_panel_label(
    ax,
    "(a)",
)


ax.grid(
    axis="y",

    linestyle="--",

    linewidth=0.45,

    alpha=0.18,

    zorder=0,
)


# ------------------------------------------------------------
# Interior / edge performance annotation
# ------------------------------------------------------------

loao_text = (
    f"Interior pooled MAE = "
    f"{loao_interior['MAE_nm']:.3f} nm"
    "\n"
    f"Edge pooled MAE = "
    f"{loao_edge['MAE_nm']:.3f} nm"
)


ax.text(
    0.97,
    0.94,

    loao_text,

    transform=ax.transAxes,

    ha="right",
    va="top",

    fontsize=FONT_ANNOTATION,

    bbox={
        "facecolor":
            "white",

        "edgecolor":
            "none",

        "alpha":
            0.86,

        "pad":
            2.0,
    },

    zorder=5,
)


# ============================================================
# 13. Panel B — Leave-one-energy-out
# ============================================================

ax = axes[1]


energies = loeo_summary[
    "held_energy_keV"
].to_numpy(
    dtype=float
)

mae = loeo_summary[
    "MAE_nm"
].to_numpy(
    dtype=float
)

rmse = loeo_summary[
    "RMSE_nm"
].to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# Edge extrapolation shading
# ------------------------------------------------------------

ax.axvspan(
    0.35,
    0.72,

    facecolor="#E6E6E6",

    edgecolor="none",

    alpha=0.65,

    zorder=0,
)


ax.axvspan(
    4.78,
    5.15,

    facecolor="#E6E6E6",

    edgecolor="none",

    alpha=0.65,

    zorder=0,
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

ax.plot(
    energies,
    mae,

    marker="o",

    markersize=4.6,

    linewidth=1.35,

    color=COLORS["gpr"],

    markerfacecolor="white",

    markeredgecolor=COLORS["gpr"],

    markeredgewidth=0.9,

    label="MAE",

    zorder=3,
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

ax.plot(
    energies,
    rmse,

    marker="s",

    markersize=4.4,

    linewidth=1.35,

    linestyle="--",

    color=COLORS["pchip"],

    markerfacecolor="white",

    markeredgecolor=COLORS["pchip"],

    markeredgewidth=0.9,

    label="RMSE",

    zorder=3,
)


# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

ax.set_xlim(
    0.35,
    5.15,
)


ax.set_xticks(
    np.arange(
        0.5,
        5.01,
        0.5,
    )
)


ax.set_xlabel(
    "Held-out energy (keV)"
)


ax.set_ylabel(
    "Depth prediction error (nm)"
)


ax.set_title(
    "Leave-one-energy-out",
    pad=6,
)


add_panel_label(
    ax,
    "(b)",
)


ax.grid(
    axis="y",

    linestyle="--",

    linewidth=0.45,

    alpha=0.18,

    zorder=0,
)


# ------------------------------------------------------------
# Interior / edge performance annotation
# ------------------------------------------------------------

loeo_text = (
    f"Interior pooled MAE = "
    f"{loeo_interior['MAE_nm']:.3f} nm"
    "\n"
    f"Edge pooled MAE = "
    f"{loeo_edge['MAE_nm']:.3f} nm"
)


ax.text(
    0.04,
    0.94,

    loeo_text,

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
            0.86,

        "pad":
            2.0,
    },

    zorder=5,
)


# ============================================================
# 14. Shared legend
#
# Deliberately pushed upward so it does not interfere with
# panel labels (a) and (b).
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()


fig.legend(
    handles,
    labels,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        1.045,
    ),

    ncol=2,

    frameon=False,

    columnspacing=1.8,

    handlelength=2.25,

    handletextpad=0.55,

    borderaxespad=0.0,
)


# ============================================================
# 15. Layout
#
# Extra top margin reserved for legend.
# No explanatory sentence is placed inside the figure.
# Edge shading will be explained in the manuscript caption.
# ============================================================

fig.subplots_adjust(
    left=0.085,
    right=0.985,

    bottom=0.145,
    top=0.825,

    wspace=0.26,
)


# ============================================================
# 16. Save
# ============================================================

save_paper_figure(
    fig,
    OUTPUT_BASE,
)


# ============================================================
# 17. Console summary
# ============================================================

print("\n")
print("=" * 125)

print(
    "FIGURE ML-07 — FINAL GPR "
    "GENERALIZATION STRESS TEST"
)

print("=" * 125)


print(
    audit_df.to_string(
        index=False
    )
)


print("\nSaved:")

print(
    OUTPUT_SUMMARY
)

print(
    OUTPUT_FOLD_DATA
)


print("\nKey results:")

print(
    f"LOAO interior pooled MAE = "
    f"{loao_interior['MAE_nm']:.3f} nm"
)

print(
    f"LOAO edge pooled MAE     = "
    f"{loao_edge['MAE_nm']:.3f} nm"
)

print(
    f"LOEO interior pooled MAE = "
    f"{loeo_interior['MAE_nm']:.3f} nm"
)

print(
    f"LOEO edge pooled MAE     = "
    f"{loeo_edge['MAE_nm']:.3f} nm"
)


print(
    "\nGray regions correspond to "
    "edge-coordinate extrapolation folds."
)

print(
    "No new model fitting or prediction "
    "was performed by this plotting script."
)


plt.close(
    fig
)
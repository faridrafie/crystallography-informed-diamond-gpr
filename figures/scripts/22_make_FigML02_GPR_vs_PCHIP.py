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
# Script: 22_make_FigML02_GPR_vs_PCHIP.py
# Purpose: Generate the aggregate GPR-versus-PCHIP benchmark figure.
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
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from paper_plot_style import (
    apply_paper_style,
    add_panel_label,
    save_paper_figure,
    COLORS,
    DOUBLE_COLUMN_IN,
)

# ============================================================
# 1) Global paper style
# ============================================================

apply_paper_style()

# ============================================================
# 2) Paths
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

DATA_DIR = ROOT / "analysis" / "baseline" / "model_comparison"

RENDER_DIR = HERE.parent / "rendered"
SOURCE_DIR = HERE.parent / "source_data"
OUTPUT_BASE = RENDER_DIR / "Figure_ML02_GPR_vs_PCHIP_CrossedHoldout"
SUMMARY_OUTPUT = SOURCE_DIR / "Figure_ML02_GPR_vs_PCHIP_metrics.csv"

# ============================================================
# 3) Target definitions
# ============================================================

TARGETS = [
    {
        "panel": "(a)",
        "title": "Median implantation depth",
        "short_label": "Depth",
        "file": "depth_gpr_vs_pchip_crossed_predictions.csv",
        "truth_col": "true_depth_nm",
        "gpr_col": "gpr_prediction_nm",
        "pchip_col": "pchip_prediction_nm",
        "ylabel": "Prediction error (nm)",
        "digits": 3,
        "is_sp2": False,
    },
    {
        "panel": "(b)",
        "title": "Shallow fraction (0–5 nm)",
        "short_label": "Shallow fraction",
        "file": "shallow_gpr_vs_pchip_crossed_predictions.csv",
        "truth_col": "true_shallow_fraction",
        "gpr_col": "gpr_prediction",
        "pchip_col": "pchip_prediction",
        "ylabel": "Prediction error",
        "digits": 3,
        "is_sp2": False,
    },
    {
        "panel": "(c)",
        "title": "Near-surface sp² fraction (0–2 nm)",
        "short_label": "Near-surface sp²",
        "file": "sp2_gpr_vs_pchip_crossed_predictions.csv",
        "truth_col": "true_sp2_fraction",
        "gpr_col": "gpr_prediction",
        "pchip_col": "pchip_prediction",
        "ylabel": "Prediction error",
        "digits": 4,
        "is_sp2": True,
    },
    {
        "panel": "(d)",
        "title": "Vacancy count",
        "short_label": "Vacancy count",
        "file": "vacancy_gpr_vs_pchip_crossed_predictions.csv",
        "truth_col": "true_vac_count",
        "gpr_col": "gpr_prediction",
        "pchip_col": "pchip_prediction",
        "ylabel": "Prediction error (count)",
        "digits": 2,
        "is_sp2": False,
    },
]

# ============================================================
# 4) Helper
# ============================================================

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def add_value_labels(ax, bars, values, digits, y_offset):
    for bar, val in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        ax.text(
            x,
            y + y_offset,
            f"{val:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

# ============================================================
# 5) Load and compute
# ============================================================

summary_rows = []

for target in TARGETS:
    path = DATA_DIR / target["file"]
    df = pd.read_csv(path)

    y_true = df[target["truth_col"]].to_numpy(dtype=float)
    y_gpr = df[target["gpr_col"]].to_numpy(dtype=float)
    y_pchip = df[target["pchip_col"]].to_numpy(dtype=float)

    gpr_mae, gpr_rmse, gpr_r2 = compute_metrics(y_true, y_gpr)
    pchip_mae, pchip_rmse, pchip_r2 = compute_metrics(y_true, y_pchip)

    abs_gpr = np.abs(y_true - y_gpr)
    abs_pchip = np.abs(y_true - y_pchip)

    gpr_wins = int(np.sum(abs_gpr < abs_pchip))
    pchip_wins = int(np.sum(abs_pchip < abs_gpr))
    ties = int(np.sum(np.isclose(abs_gpr, abs_pchip, rtol=1e-12, atol=1e-14)))

    summary_rows.append(
        {
            "panel": target["panel"],
            "target": target["short_label"],
            "GPR_MAE": gpr_mae,
            "PCHIP_MAE": pchip_mae,
            "GPR_RMSE": gpr_rmse,
            "PCHIP_RMSE": pchip_rmse,
            "GPR_R2": gpr_r2,
            "PCHIP_R2": pchip_r2,
            "GPR_point_wins": gpr_wins,
            "PCHIP_point_wins": pchip_wins,
            "ties": ties,
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_OUTPUT, index=False)

# ============================================================
# 6) Plot
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(DOUBLE_COLUMN_IN, 5.8),
)

axes = axes.ravel()

bar_width = 0.34
x = np.arange(2)
metric_labels = ["MAE", "RMSE"]

legend_handles = None
legend_labels = None

for ax, target, row in zip(axes, TARGETS, summary_rows):
    gpr_vals = np.array([row["GPR_MAE"], row["GPR_RMSE"]], dtype=float)
    pchip_vals = np.array([row["PCHIP_MAE"], row["PCHIP_RMSE"]], dtype=float)

    max_val = max(np.max(gpr_vals), np.max(pchip_vals))
    y_top = max_val * 1.28
    if y_top == 0:
        y_top = 1.0
    y_offset = y_top * 0.025

    bars_gpr = ax.bar(
        x - bar_width / 2,
        gpr_vals,
        width=bar_width,
        color=COLORS["gpr"],
        edgecolor="black",
        linewidth=0.5,
        label="GPR",
        zorder=3,
    )

    bars_pchip = ax.bar(
        x + bar_width / 2,
        pchip_vals,
        width=bar_width,
        color=COLORS["pchip"],
        edgecolor="black",
        linewidth=0.5,
        label="PCHIP",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, y_top)
    ax.set_ylabel(target["ylabel"])
    ax.set_title(target["title"], pad=6)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)

    add_panel_label(ax, target["panel"])

    add_value_labels(
        ax=ax,
        bars=bars_gpr,
        values=gpr_vals,
        digits=target["digits"],
        y_offset=y_offset,
    )

    add_value_labels(
        ax=ax,
        bars=bars_pchip,
        values=pchip_vals,
        digits=target["digits"],
        y_offset=y_offset,
    )

    annotation = (
        f"$R^2$ (GPR) = {row['GPR_R2']:.3f}\n"
        f"$R^2$ (PCHIP) = {row['PCHIP_R2']:.3f}"
    )

    ax.text(
        0.04,
        0.94,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            pad=2.0,
        ),
    )

    if target["is_sp2"]:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, -3))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
        ax.yaxis.get_offset_text().set_fontsize(8)

    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

# ============================================================
# 7) Global legend and layout
# ============================================================

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995),
    handlelength=1.2,
    columnspacing=1.5,
)

fig.text(
    0.5,
    0.02,
    "Common 64-point interior crossed energy–angle holdout",
    ha="center",
    va="bottom",
    fontsize=8.5,
)

fig.subplots_adjust(
    left=0.095,
    right=0.985,
    bottom=0.08,
    top=0.89,
    wspace=0.30,
    hspace=0.38,
)

# ============================================================
# 8) Save
# ============================================================

save_paper_figure(fig, OUTPUT_BASE)

print("\n" + "=" * 120)
print("FIGURE ML-02 SUMMARY")
print("=" * 120)
print(summary_df.to_string(index=False))
print("\nSaved:")
print(SUMMARY_OUTPUT)

plt.close(fig)
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
# Script: 10_compare_gpr_pchip_depth.py
# Purpose: Condition-level comparison of GPR and PCHIP for median implantation depth.
#
# Research note:
#   This script belongs to the reproducibility package for my doctoral
#   research on simulation-guided low-energy nitrogen implantation in diamond.
#   Numerical settings used for manuscript results are intentionally kept
#   explicit in the script rather than hidden behind external configuration.
# =============================================================================

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent

gpr_path = BASELINE_DIR / "gpr_crossed_holdout" / "gpr_depth_crossed_predictions.csv"
pchip_path = BASELINE_DIR / "pchip_crossed_holdout" / "pchip_depth_crossed_predictions.csv"


# --------------------------------------------------
# 2. Load prediction files
# --------------------------------------------------

gpr_df = pd.read_csv(gpr_path)
pchip_df = pd.read_csv(pchip_path)


print("\nInput files")
print("-----------")

print(
    f"GPR predictions   : {len(gpr_df)}"
)

print(
    f"PCHIP predictions : {len(pchip_df)}"
)


# --------------------------------------------------
# 3. Keep only columns needed for comparison
# --------------------------------------------------

gpr_compare = gpr_df[
    [
        "held_energy_keV",
        "held_angle_deg",
        "true_depth_nm",
        "predicted_depth_nm",
        "prediction_std_nm",
    ]
].copy()

gpr_compare = gpr_compare.rename(
    columns={
        "true_depth_nm":
            "true_depth_gpr_nm",

        "predicted_depth_nm":
            "gpr_prediction_nm",

        "prediction_std_nm":
            "gpr_prediction_std_nm",
    }
)


pchip_compare = pchip_df[
    [
        "held_energy_keV",
        "held_angle_deg",
        "true_depth_nm",
        "predicted_depth_nm",
        "pred_E_then_A_nm",
        "pred_A_then_E_nm",
    ]
].copy()

pchip_compare = pchip_compare.rename(
    columns={
        "true_depth_nm":
            "true_depth_pchip_nm",

        "predicted_depth_nm":
            "pchip_prediction_nm",
    }
)


# --------------------------------------------------
# 4. Merge on EXACT same crossed targets
# --------------------------------------------------

comparison = pd.merge(
    gpr_compare,
    pchip_compare,
    on=[
        "held_energy_keV",
        "held_angle_deg",
    ],
    how="inner",
)


print("\nCommon crossed targets")
print("----------------------")

print(
    f"Number of common targets: "
    f"{len(comparison)}"
)


# --------------------------------------------------
# 5. Sanity checks
# --------------------------------------------------

if len(comparison) != 64:

    raise ValueError(
        "Expected exactly 64 common interior "
        "crossed targets."
    )


true_difference = np.abs(
    comparison["true_depth_gpr_nm"]
    -
    comparison["true_depth_pchip_nm"]
)

max_true_difference = true_difference.max()


print(
    f"Max difference between stored "
    f"true values: {max_true_difference:.12f} nm"
)


if max_true_difference > 1e-10:

    raise ValueError(
        "GPR and PCHIP files do not contain "
        "identical true target values."
    )


# One common truth column
comparison["true_depth_nm"] = (
    comparison["true_depth_gpr_nm"]
)


# --------------------------------------------------
# 6. Calculate errors for each model
# --------------------------------------------------

comparison["gpr_residual_nm"] = (
    comparison["true_depth_nm"]
    -
    comparison["gpr_prediction_nm"]
)

comparison["pchip_residual_nm"] = (
    comparison["true_depth_nm"]
    -
    comparison["pchip_prediction_nm"]
)


comparison["gpr_abs_error_nm"] = np.abs(
    comparison["gpr_residual_nm"]
)

comparison["pchip_abs_error_nm"] = np.abs(
    comparison["pchip_residual_nm"]
)


# Positive value:
# PCHIP had larger error, so GPR won.
comparison["error_advantage_gpr_nm"] = (
    comparison["pchip_abs_error_nm"]
    -
    comparison["gpr_abs_error_nm"]
)


# --------------------------------------------------
# 7. GPR metrics
# --------------------------------------------------

gpr_mae = mean_absolute_error(
    comparison["true_depth_nm"],
    comparison["gpr_prediction_nm"],
)

gpr_rmse = mean_squared_error(
    comparison["true_depth_nm"],
    comparison["gpr_prediction_nm"],
) ** 0.5

gpr_r2 = r2_score(
    comparison["true_depth_nm"],
    comparison["gpr_prediction_nm"],
)


# --------------------------------------------------
# 8. PCHIP metrics
# --------------------------------------------------

pchip_mae = mean_absolute_error(
    comparison["true_depth_nm"],
    comparison["pchip_prediction_nm"],
)

pchip_rmse = mean_squared_error(
    comparison["true_depth_nm"],
    comparison["pchip_prediction_nm"],
) ** 0.5

pchip_r2 = r2_score(
    comparison["true_depth_nm"],
    comparison["pchip_prediction_nm"],
)


# --------------------------------------------------
# 9. Relative improvement of GPR over PCHIP
# --------------------------------------------------

mae_improvement_percent = (
    (pchip_mae - gpr_mae)
    / pchip_mae
    * 100.0
)

rmse_improvement_percent = (
    (pchip_rmse - gpr_rmse)
    / pchip_rmse
    * 100.0
)


# --------------------------------------------------
# 10. Count point-by-point winners
# --------------------------------------------------

tolerance = 1e-12

gpr_wins = (
    comparison["gpr_abs_error_nm"]
    <
    comparison["pchip_abs_error_nm"] - tolerance
).sum()

pchip_wins = (
    comparison["pchip_abs_error_nm"]
    <
    comparison["gpr_abs_error_nm"] - tolerance
).sum()

ties = (
    np.abs(
        comparison["gpr_abs_error_nm"]
        -
        comparison["pchip_abs_error_nm"]
    )
    <= tolerance
).sum()


# --------------------------------------------------
# 11. Build summary table
# --------------------------------------------------

summary_df = pd.DataFrame(
    {
        "Model": [
            "GPR",
            "PCHIP",
        ],

        "N": [
            len(comparison),
            len(comparison),
        ],

        "MAE_nm": [
            gpr_mae,
            pchip_mae,
        ],

        "RMSE_nm": [
            gpr_rmse,
            pchip_rmse,
        ],

        "R2": [
            gpr_r2,
            pchip_r2,
        ],
    }
)


# --------------------------------------------------
# 12. Print final benchmark
# --------------------------------------------------

print("\n")
print("==========================================")
print("DEPTH: GPR vs PCHIP")
print("Interior Crossed Energy-Angle Holdout")
print("==========================================")

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\nGPR improvement over PCHIP")
print("--------------------------")

print(
    f"MAE reduction : "
    f"{mae_improvement_percent:.2f} %"
)

print(
    f"RMSE reduction: "
    f"{rmse_improvement_percent:.2f} %"
)


print("\nPoint-by-point comparison")
print("-------------------------")

print(
    f"GPR lower absolute error   : "
    f"{gpr_wins} / {len(comparison)}"
)

print(
    f"PCHIP lower absolute error : "
    f"{pchip_wins} / {len(comparison)}"
)

print(
    f"Ties                       : "
    f"{ties}"
)


# --------------------------------------------------
# 13. Show where GPR helps most
# --------------------------------------------------

print("\nLargest GPR advantages")
print("----------------------")

gpr_best = comparison.sort_values(
    "error_advantage_gpr_nm",
    ascending=False,
).head(10)

print(
    gpr_best[
        [
            "held_energy_keV",
            "held_angle_deg",
            "true_depth_nm",
            "gpr_prediction_nm",
            "pchip_prediction_nm",
            "gpr_abs_error_nm",
            "pchip_abs_error_nm",
            "error_advantage_gpr_nm",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# 14. Show where PCHIP helps most
# --------------------------------------------------

print("\nLargest PCHIP advantages")
print("------------------------")

pchip_best = comparison.sort_values(
    "error_advantage_gpr_nm",
    ascending=True,
).head(10)

print(
    pchip_best[
        [
            "held_energy_keV",
            "held_angle_deg",
            "true_depth_nm",
            "gpr_prediction_nm",
            "pchip_prediction_nm",
            "gpr_abs_error_nm",
            "pchip_abs_error_nm",
            "error_advantage_gpr_nm",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# 15. Clean columns before saving
# --------------------------------------------------

comparison = comparison.drop(
    columns=[
        "true_depth_gpr_nm",
        "true_depth_pchip_nm",
    ]
)


comparison = comparison.sort_values(
    [
        "held_energy_keV",
        "held_angle_deg",
    ]
)


# --------------------------------------------------
# 16. Save benchmark outputs
# --------------------------------------------------

comparison.to_csv(
    SCRIPT_DIR / "depth_gpr_vs_pchip_crossed_predictions.csv",
    index=False,
)

summary_df.to_csv(
    SCRIPT_DIR / "depth_gpr_vs_pchip_crossed_summary.csv",
    index=False,
)


print("\nSaved:")
print(
    "depth_gpr_vs_pchip_crossed_predictions.csv"
)

print(
    "depth_gpr_vs_pchip_crossed_summary.csv"
)
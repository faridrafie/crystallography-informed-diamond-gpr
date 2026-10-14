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
# Script: 19_compare_gpr_pchip_vacancy.py
# Purpose: Condition-level comparison of GPR and PCHIP for vacancy count.
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

gpr_path = BASELINE_DIR / "gpr_crossed_vacancy" / "gpr_vacancy_crossed_predictions.csv"
pchip_path = BASELINE_DIR / "pchip_crossed_vacancy" / "pchip_vacancy_crossed_predictions.csv"


# --------------------------------------------------
# 2. Load predictions
# --------------------------------------------------

gpr_df = pd.read_csv(gpr_path)
pchip_df = pd.read_csv(pchip_path)


print("\nInput files")
print("-----------")

print(f"GPR predictions   : {len(gpr_df)}")
print(f"PCHIP predictions : {len(pchip_df)}")


# --------------------------------------------------
# 3. Prepare GPR table
# --------------------------------------------------

gpr_compare = gpr_df[
    [
        "held_energy_keV",
        "held_angle_deg",
        "true_vac_count",
        "predicted_vac_count",
        "prediction_std",
    ]
].copy()


gpr_compare = gpr_compare.rename(
    columns={
        "true_vac_count":
            "true_gpr",

        "predicted_vac_count":
            "gpr_prediction",

        "prediction_std":
            "gpr_prediction_std",
    }
)


# --------------------------------------------------
# 4. Prepare PCHIP table
# --------------------------------------------------

pchip_compare = pchip_df[
    [
        "held_energy_keV",
        "held_angle_deg",
        "true_vac_count",
        "predicted_vac_count",
        "pred_E_then_A",
        "pred_A_then_E",
    ]
].copy()


pchip_compare = pchip_compare.rename(
    columns={
        "true_vac_count":
            "true_pchip",

        "predicted_vac_count":
            "pchip_prediction",
    }
)


# --------------------------------------------------
# 5. Merge exact same crossed targets
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
# 6. Sanity checks
# --------------------------------------------------

if len(comparison) != 64:

    raise ValueError(
        "Expected exactly 64 common interior "
        "crossed targets."
    )


true_difference = np.abs(
    comparison["true_gpr"]
    -
    comparison["true_pchip"]
)

max_true_difference = true_difference.max()


print(
    f"Max difference between stored "
    f"true values: "
    f"{max_true_difference:.12f}"
)


if max_true_difference > 1e-10:

    raise ValueError(
        "True target values are not identical "
        "between GPR and PCHIP files."
    )


comparison["true_vac_count"] = (
    comparison["true_gpr"]
)


# --------------------------------------------------
# 7. Calculate errors
# --------------------------------------------------

comparison["gpr_residual"] = (
    comparison["true_vac_count"]
    -
    comparison["gpr_prediction"]
)

comparison["pchip_residual"] = (
    comparison["true_vac_count"]
    -
    comparison["pchip_prediction"]
)


comparison["gpr_abs_error"] = np.abs(
    comparison["gpr_residual"]
)

comparison["pchip_abs_error"] = np.abs(
    comparison["pchip_residual"]
)


# Positive = GPR better
comparison["error_advantage_gpr"] = (
    comparison["pchip_abs_error"]
    -
    comparison["gpr_abs_error"]
)


# --------------------------------------------------
# 8. GPR metrics
# --------------------------------------------------

gpr_mae = mean_absolute_error(
    comparison["true_vac_count"],
    comparison["gpr_prediction"],
)

gpr_rmse = mean_squared_error(
    comparison["true_vac_count"],
    comparison["gpr_prediction"],
) ** 0.5

gpr_r2 = r2_score(
    comparison["true_vac_count"],
    comparison["gpr_prediction"],
)


# --------------------------------------------------
# 9. PCHIP metrics
# --------------------------------------------------

pchip_mae = mean_absolute_error(
    comparison["true_vac_count"],
    comparison["pchip_prediction"],
)

pchip_rmse = mean_squared_error(
    comparison["true_vac_count"],
    comparison["pchip_prediction"],
) ** 0.5

pchip_r2 = r2_score(
    comparison["true_vac_count"],
    comparison["pchip_prediction"],
)


# --------------------------------------------------
# 10. Relative improvement of GPR
#
# Positive = GPR better
# Negative = PCHIP better
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
# 11. Point-by-point winners
# --------------------------------------------------

tolerance = 1e-12


gpr_wins = (
    comparison["gpr_abs_error"]
    <
    comparison["pchip_abs_error"]
    - tolerance
).sum()


pchip_wins = (
    comparison["pchip_abs_error"]
    <
    comparison["gpr_abs_error"]
    - tolerance
).sum()


ties = (
    np.abs(
        comparison["gpr_abs_error"]
        -
        comparison["pchip_abs_error"]
    )
    <= tolerance
).sum()


# --------------------------------------------------
# 12. Physical lower-bound audit
# --------------------------------------------------

gpr_below_zero = (
    comparison["gpr_prediction"] < 0.0
).sum()

pchip_below_zero = (
    comparison["pchip_prediction"] < 0.0
).sum()


# --------------------------------------------------
# 13. Build summary table
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

        "MAE": [
            gpr_mae,
            pchip_mae,
        ],

        "RMSE": [
            gpr_rmse,
            pchip_rmse,
        ],

        "R2": [
            gpr_r2,
            pchip_r2,
        ],

        "Below_Zero": [
            gpr_below_zero,
            pchip_below_zero,
        ],
    }
)


# --------------------------------------------------
# 14. Print benchmark
# --------------------------------------------------

print("\n")
print("==========================================")
print("VACANCY COUNT: GPR vs PCHIP")
print("Interior Crossed Energy-Angle Holdout")
print("==========================================")

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print("\nGPR relative improvement over PCHIP")
print("-----------------------------------")

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


print("\nPhysical lower-bound audit")
print("--------------------------")

print(
    f"GPR below zero   : "
    f"{gpr_below_zero} / {len(comparison)}"
)

print(
    f"PCHIP below zero : "
    f"{pchip_below_zero} / {len(comparison)}"
)


# --------------------------------------------------
# 15. Largest GPR advantages
# --------------------------------------------------

print("\nLargest GPR advantages")
print("----------------------")

gpr_best = comparison.sort_values(
    "error_advantage_gpr",
    ascending=False,
).head(10)


print(
    gpr_best[
        [
            "held_energy_keV",
            "held_angle_deg",
            "true_vac_count",
            "gpr_prediction",
            "pchip_prediction",
            "gpr_abs_error",
            "pchip_abs_error",
            "error_advantage_gpr",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# 16. Largest PCHIP advantages
# --------------------------------------------------

print("\nLargest PCHIP advantages")
print("------------------------")

pchip_best = comparison.sort_values(
    "error_advantage_gpr",
    ascending=True,
).head(10)


print(
    pchip_best[
        [
            "held_energy_keV",
            "held_angle_deg",
            "true_vac_count",
            "gpr_prediction",
            "pchip_prediction",
            "gpr_abs_error",
            "pchip_abs_error",
            "error_advantage_gpr",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# 17. Clean and save
# --------------------------------------------------

comparison = comparison.drop(
    columns=[
        "true_gpr",
        "true_pchip",
    ]
)


comparison = comparison.sort_values(
    [
        "held_energy_keV",
        "held_angle_deg",
    ]
)


comparison.to_csv(
    SCRIPT_DIR / "vacancy_gpr_vs_pchip_crossed_predictions.csv",
    index=False,
)

summary_df.to_csv(
    SCRIPT_DIR / "vacancy_gpr_vs_pchip_crossed_summary.csv",
    index=False,
)


print("\nSaved:")
print(
    "vacancy_gpr_vs_pchip_crossed_predictions.csv"
)

print(
    "vacancy_gpr_vs_pchip_crossed_summary.csv"
)
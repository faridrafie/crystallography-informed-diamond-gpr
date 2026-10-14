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
# Script: 09_pchip_depth_crossed_holdout.py
# Purpose: Sequential two-path PCHIP benchmark for median implantation depth.
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

from scipy.interpolate import PchipInterpolator

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

data_path = PROJECT_ROOT / "data" / "master_condition_table_core.csv"

df = pd.read_csv(data_path)

target_col = "p1_depth_median_nm"


# --------------------------------------------------
# 2. Get grid coordinates
# --------------------------------------------------

energies = np.sort(df["E"].unique())
angles = np.sort(df["A"].unique())

print("\nAll energies:")
print(energies)

print("\nAll angles:")
print(angles)


# --------------------------------------------------
# 3. Define INTERIOR crossed targets only
#
# PCHIP is being used as an interpolation baseline.
# Therefore min/max energy and min/max angle are not
# used as test targets in the primary comparison.
# --------------------------------------------------

interior_energies = energies[1:-1]
interior_angles = angles[1:-1]

print("\nInterior energies used for crossed benchmark:")
print(interior_energies)

print("\nInterior angles used for crossed benchmark:")
print(interior_angles)

print(
    "\nNumber of crossed targets:",
    len(interior_energies) * len(interior_angles)
)


# --------------------------------------------------
# 4. Store predictions
# --------------------------------------------------

prediction_rows = []


# --------------------------------------------------
# 5. Crossed Energy-Angle Holdout
# --------------------------------------------------

for held_energy in interior_energies:

    for held_angle in interior_angles:

        print("\n========================================")
        print(
            f"Target: E={held_energy} keV, "
            f"A={held_angle} deg"
        )
        print("========================================")


        # ------------------------------------------
        # 5a. Training data
        #
        # Remove complete held-energy row
        # AND complete held-angle column.
        # ------------------------------------------

        train_mask = (
            (df["E"] != held_energy)
            &
            (df["A"] != held_angle)
        )

        train_df = df.loc[train_mask].copy()


        # ------------------------------------------
        # 5b. True target condition
        # ------------------------------------------

        test_mask = (
            (df["E"] == held_energy)
            &
            (df["A"] == held_angle)
        )

        test_df = df.loc[test_mask]

        true_value = test_df[target_col].iloc[0]


        # ==========================================
        # PATH 1:
        # ENERGY first -> ANGLE second
        # ==========================================

        energy_first_values = []

        remaining_angles = np.sort(
            train_df["A"].unique()
        )

        for angle_value in remaining_angles:

            subset = train_df[
                train_df["A"] == angle_value
            ].sort_values("E")

            x_energy = subset["E"].to_numpy()
            y_depth = subset[target_col].to_numpy()

            energy_interp = PchipInterpolator(
                x_energy,
                y_depth,
                extrapolate=False,
            )

            depth_at_held_energy = float(
                energy_interp(held_energy)
            )

            energy_first_values.append(
                depth_at_held_energy
            )


        energy_first_values = np.array(
            energy_first_values
        )

        angle_interp_second = PchipInterpolator(
            remaining_angles,
            energy_first_values,
            extrapolate=False,
        )

        pred_energy_then_angle = float(
            angle_interp_second(held_angle)
        )


        # ==========================================
        # PATH 2:
        # ANGLE first -> ENERGY second
        # ==========================================

        angle_first_values = []

        remaining_energies = np.sort(
            train_df["E"].unique()
        )

        for energy_value in remaining_energies:

            subset = train_df[
                train_df["E"] == energy_value
            ].sort_values("A")

            x_angle = subset["A"].to_numpy()
            y_depth = subset[target_col].to_numpy()

            angle_interp = PchipInterpolator(
                x_angle,
                y_depth,
                extrapolate=False,
            )

            depth_at_held_angle = float(
                angle_interp(held_angle)
            )

            angle_first_values.append(
                depth_at_held_angle
            )


        angle_first_values = np.array(
            angle_first_values
        )

        energy_interp_second = PchipInterpolator(
            remaining_energies,
            angle_first_values,
            extrapolate=False,
        )

        pred_angle_then_energy = float(
            energy_interp_second(held_energy)
        )


        # ==========================================
        # FINAL TWO-PATH PCHIP PREDICTION
        # ==========================================

        predicted_value = (
            pred_energy_then_angle
            +
            pred_angle_then_energy
        ) / 2.0


        # ------------------------------------------
        # 5c. Error
        # ------------------------------------------

        residual = (
            true_value
            - predicted_value
        )

        abs_error = abs(
            residual
        )


        # ------------------------------------------
        # 5d. Store
        # ------------------------------------------

        prediction_rows.append(
            {
                "held_energy_keV": held_energy,
                "held_angle_deg": held_angle,
                "true_depth_nm": true_value,

                "pred_E_then_A_nm":
                    pred_energy_then_angle,

                "pred_A_then_E_nm":
                    pred_angle_then_energy,

                "predicted_depth_nm":
                    predicted_value,

                "residual_nm":
                    residual,

                "abs_error_nm":
                    abs_error,
            }
        )


        # ------------------------------------------
        # 5e. Print prediction
        # ------------------------------------------

        print(
            f"True depth       : "
            f"{true_value:.4f} nm"
        )

        print(
            f"E -> A prediction: "
            f"{pred_energy_then_angle:.4f} nm"
        )

        print(
            f"A -> E prediction: "
            f"{pred_angle_then_energy:.4f} nm"
        )

        print(
            f"Two-path average : "
            f"{predicted_value:.4f} nm"
        )

        print(
            f"Absolute error   : "
            f"{abs_error:.4f} nm"
        )


# --------------------------------------------------
# 6. Combine predictions
# --------------------------------------------------

predictions_df = pd.DataFrame(
    prediction_rows
)


# --------------------------------------------------
# 7. Overall metrics
# --------------------------------------------------

mae = mean_absolute_error(
    predictions_df["true_depth_nm"],
    predictions_df["predicted_depth_nm"],
)

rmse = mean_squared_error(
    predictions_df["true_depth_nm"],
    predictions_df["predicted_depth_nm"],
) ** 0.5

r2 = r2_score(
    predictions_df["true_depth_nm"],
    predictions_df["predicted_depth_nm"],
)


# --------------------------------------------------
# 8. Difference between two interpolation paths
# --------------------------------------------------

predictions_df["path_difference_nm"] = abs(
    predictions_df["pred_E_then_A_nm"]
    -
    predictions_df["pred_A_then_E_nm"]
)

mean_path_difference = (
    predictions_df["path_difference_nm"].mean()
)

max_path_difference = (
    predictions_df["path_difference_nm"].max()
)


# --------------------------------------------------
# 9. Print overall results
# --------------------------------------------------

print("\n\n========================================")
print("PCHIP Crossed Energy-Angle Holdout")
print("========================================")

print(
    f"Number of predictions: "
    f"{len(predictions_df)}"
)

print(f"MAE : {mae:.4f} nm")
print(f"RMSE: {rmse:.4f} nm")
print(f"R2  : {r2:.4f}")


print("\nTwo-Path Consistency")
print("--------------------")

print(
    f"Mean |E->A - A->E| : "
    f"{mean_path_difference:.4f} nm"
)

print(
    f"Max  |E->A - A->E| : "
    f"{max_path_difference:.4f} nm"
)


# --------------------------------------------------
# 10. Worst predictions
# --------------------------------------------------

print("\nWorst 10 Predictions")
print("--------------------")

worst = predictions_df.sort_values(
    "abs_error_nm",
    ascending=False,
).head(10)

print(
    worst[
        [
            "held_energy_keV",
            "held_angle_deg",
            "true_depth_nm",
            "predicted_depth_nm",
            "pred_E_then_A_nm",
            "pred_A_then_E_nm",
            "abs_error_nm",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# 11. Save
# --------------------------------------------------

predictions_df.to_csv(
    SCRIPT_DIR / "pchip_depth_crossed_predictions.csv",
    index=False,
)
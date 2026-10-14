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
# Script: 12_pchip_shallow_crossed_holdout.py
# Purpose: Sequential two-path PCHIP benchmark for the shallow implanted fraction.
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

target_col = "p1_frac_0_5nm"


# --------------------------------------------------
# 2. Get grid coordinates
# --------------------------------------------------

energies = np.sort(
    df["E"].unique()
)

angles = np.sort(
    df["A"].unique()
)


print("\nTarget:")
print(target_col)

print("\nObserved target range:")
print(
    f"Minimum = {df[target_col].min():.6f}"
)
print(
    f"Maximum = {df[target_col].max():.6f}"
)


# --------------------------------------------------
# 3. Interior targets only
# --------------------------------------------------

interior_energies = energies[1:-1]
interior_angles = angles[1:-1]


print("\nInterior energies:")
print(interior_energies)

print("\nInterior angles:")
print(interior_angles)

print(
    "\nNumber of crossed targets:",
    len(interior_energies)
    * len(interior_angles)
)


# --------------------------------------------------
# 4. Storage
# --------------------------------------------------

prediction_rows = []


# --------------------------------------------------
# 5. Crossed Energy-Angle Holdout
# --------------------------------------------------

for held_energy in interior_energies:

    for held_angle in interior_angles:


        # ------------------------------------------
        # 5a. Training data
        # ------------------------------------------

        train_mask = (
            (df["E"] != held_energy)
            &
            (df["A"] != held_angle)
        )

        train_df = df.loc[
            train_mask
        ].copy()


        # ------------------------------------------
        # 5b. True target
        # ------------------------------------------

        test_mask = (
            (df["E"] == held_energy)
            &
            (df["A"] == held_angle)
        )

        test_df = df.loc[
            test_mask
        ]

        true_value = (
            test_df[target_col].iloc[0]
        )


        # ==========================================
        # PATH 1:
        # Energy -> Angle
        # ==========================================

        remaining_angles = np.sort(
            train_df["A"].unique()
        )

        energy_first_values = []


        for angle_value in remaining_angles:

            subset = train_df[
                train_df["A"] == angle_value
            ].sort_values("E")

            x_energy = (
                subset["E"].to_numpy()
            )

            y_value = (
                subset[target_col].to_numpy()
            )


            interp_energy = PchipInterpolator(
                x_energy,
                y_value,
                extrapolate=False,
            )


            value_at_energy = float(
                interp_energy(
                    held_energy
                )
            )


            energy_first_values.append(
                value_at_energy
            )


        energy_first_values = np.array(
            energy_first_values
        )


        interp_angle_second = (
            PchipInterpolator(
                remaining_angles,
                energy_first_values,
                extrapolate=False,
            )
        )


        pred_E_then_A = float(
            interp_angle_second(
                held_angle
            )
        )


        # ==========================================
        # PATH 2:
        # Angle -> Energy
        # ==========================================

        remaining_energies = np.sort(
            train_df["E"].unique()
        )

        angle_first_values = []


        for energy_value in remaining_energies:

            subset = train_df[
                train_df["E"] == energy_value
            ].sort_values("A")


            x_angle = (
                subset["A"].to_numpy()
            )

            y_value = (
                subset[target_col].to_numpy()
            )


            interp_angle = PchipInterpolator(
                x_angle,
                y_value,
                extrapolate=False,
            )


            value_at_angle = float(
                interp_angle(
                    held_angle
                )
            )


            angle_first_values.append(
                value_at_angle
            )


        angle_first_values = np.array(
            angle_first_values
        )


        interp_energy_second = (
            PchipInterpolator(
                remaining_energies,
                angle_first_values,
                extrapolate=False,
            )
        )


        pred_A_then_E = float(
            interp_energy_second(
                held_energy
            )
        )


        # ==========================================
        # Final two-path prediction
        # ==========================================

        predicted_value = (
            pred_E_then_A
            +
            pred_A_then_E
        ) / 2.0


        residual = (
            true_value
            - predicted_value
        )


        outside_bounds = (
            predicted_value < 0.0
            or
            predicted_value > 1.0
        )


        prediction_rows.append(
            {
                "held_energy_keV":
                    held_energy,

                "held_angle_deg":
                    held_angle,

                "true_shallow_fraction":
                    true_value,

                "pred_E_then_A":
                    pred_E_then_A,

                "pred_A_then_E":
                    pred_A_then_E,

                "predicted_shallow_fraction":
                    predicted_value,

                "residual":
                    residual,

                "abs_error":
                    abs(residual),

                "outside_0_1":
                    outside_bounds,
            }
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
    predictions_df[
        "true_shallow_fraction"
    ],
    predictions_df[
        "predicted_shallow_fraction"
    ],
)

rmse = mean_squared_error(
    predictions_df[
        "true_shallow_fraction"
    ],
    predictions_df[
        "predicted_shallow_fraction"
    ],
) ** 0.5

r2 = r2_score(
    predictions_df[
        "true_shallow_fraction"
    ],
    predictions_df[
        "predicted_shallow_fraction"
    ],
)


# --------------------------------------------------
# 8. Two-path consistency
# --------------------------------------------------

predictions_df[
    "path_difference"
] = np.abs(
    predictions_df[
        "pred_E_then_A"
    ]
    -
    predictions_df[
        "pred_A_then_E"
    ]
)


mean_path_difference = (
    predictions_df[
        "path_difference"
    ].mean()
)

max_path_difference = (
    predictions_df[
        "path_difference"
    ].max()
)


# --------------------------------------------------
# 9. Physical-bound audit
# --------------------------------------------------

n_outside = (
    predictions_df[
        "outside_0_1"
    ].sum()
)

pred_min = (
    predictions_df[
        "predicted_shallow_fraction"
    ].min()
)

pred_max = (
    predictions_df[
        "predicted_shallow_fraction"
    ].max()
)


# --------------------------------------------------
# 10. Print results
# --------------------------------------------------

print("\n")
print("========================================")
print("PCHIP SHALLOW-FRACTION CROSSED HOLDOUT")
print("========================================")

print(
    f"Number of predictions: "
    f"{len(predictions_df)}"
)

print(f"MAE : {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R2  : {r2:.6f}")


print("\nPrediction-range audit")
print("----------------------")

print(
    f"Minimum prediction: "
    f"{pred_min:.6f}"
)

print(
    f"Maximum prediction: "
    f"{pred_max:.6f}"
)

print(
    f"Predictions outside [0,1]: "
    f"{n_outside} / {len(predictions_df)}"
)


print("\nTwo-Path Consistency")
print("--------------------")

print(
    f"Mean |E->A - A->E| : "
    f"{mean_path_difference:.6f}"
)

print(
    f"Max  |E->A - A->E| : "
    f"{max_path_difference:.6f}"
)


# --------------------------------------------------
# 11. Worst predictions
# --------------------------------------------------

print("\nWorst 10 Predictions")
print("--------------------")

worst = predictions_df.sort_values(
    "abs_error",
    ascending=False,
).head(10)


print(
    worst[
        [
            "held_energy_keV",
            "held_angle_deg",
            "true_shallow_fraction",
            "predicted_shallow_fraction",
            "pred_E_then_A",
            "pred_A_then_E",
            "abs_error",
            "outside_0_1",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# 12. Save
# --------------------------------------------------

predictions_df.to_csv(
    SCRIPT_DIR / "pchip_shallow_crossed_predictions.csv",
    index=False,
)
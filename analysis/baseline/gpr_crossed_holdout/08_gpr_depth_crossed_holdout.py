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
# Script: 08_gpr_depth_crossed_holdout.py
# Purpose: Crossed energy–angle holdout validation for median implantation depth.
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

from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)

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

X = df[["E", "A"]]

y = df["p1_depth_median_nm"]


# --------------------------------------------------
# 2. Get all energies and angles
# --------------------------------------------------

energies = np.sort(
    df["E"].unique()
)

angles = np.sort(
    df["A"].unique()
)

print("\nEnergies:")
print(energies)

print("\nAngles:")
print(angles)


# --------------------------------------------------
# 3. Container for all crossed predictions
# --------------------------------------------------

prediction_rows = []


# --------------------------------------------------
# 4. Crossed Energy-Angle Holdout
# --------------------------------------------------

for held_energy in energies:

    for held_angle in angles:

        print("\n========================================")
        print(
            f"Target: E={held_energy} keV, "
            f"A={held_angle} deg"
        )
        print("========================================")


        # ------------------------------------------
        # 4a. Training set
        #
        # Remove EVERY condition with:
        #   held energy
        # OR
        #   held angle
        # ------------------------------------------

        train_mask = (
            (df["E"] != held_energy)
            &
            (df["A"] != held_angle)
        )


        # ------------------------------------------
        # 4b. Single test condition
        # ------------------------------------------

        test_mask = (
            (df["E"] == held_energy)
            &
            (df["A"] == held_angle)
        )


        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]

        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]


        # ------------------------------------------
        # 4c. Sanity check
        # ------------------------------------------

        print(
            f"Training conditions: {len(X_train)}"
        )

        print(
            f"Test conditions: {len(X_test)}"
        )


        # ------------------------------------------
        # 4d. Scale TRAINING data only
        # ------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )


        # ------------------------------------------
        # 4e. Fresh GPR kernel for this fold
        # ------------------------------------------

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


        # ------------------------------------------
        # 4f. Create GPR
        # ------------------------------------------

        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=8,
            random_state=42,
        )


        # ------------------------------------------
        # 4g. Train
        # ------------------------------------------

        model.fit(
            X_train_scaled,
            y_train,
        )


        # ------------------------------------------
        # 4h. Predict the crossed held-out condition
        # ------------------------------------------

        y_pred, y_std = model.predict(
            X_test_scaled,
            return_std=True,
        )


        true_value = y_test.iloc[0]
        pred_value = y_pred[0]
        std_value = y_std[0]

        residual = (
            true_value
            - pred_value
        )


        # ------------------------------------------
        # 4i. Store result
        # ------------------------------------------

        prediction_rows.append(
            {
                "held_energy_keV": held_energy,
                "held_angle_deg": held_angle,
                "true_depth_nm": true_value,
                "predicted_depth_nm": pred_value,
                "prediction_std_nm": std_value,
                "residual_nm": residual,
                "abs_error_nm": abs(residual),
                "optimized_kernel": str(
                    model.kernel_
                ),
            }
        )


        # ------------------------------------------
        # 4j. Print this prediction
        # ------------------------------------------

        print(
            f"True depth      : "
            f"{true_value:.4f} nm"
        )

        print(
            f"Predicted depth : "
            f"{pred_value:.4f} nm"
        )

        print(
            f"Prediction std  : "
            f"{std_value:.4f} nm"
        )

        print(
            f"Residual        : "
            f"{residual:.4f} nm"
        )


# --------------------------------------------------
# 5. Combine all 100 crossed predictions
# --------------------------------------------------

predictions_df = pd.DataFrame(
    prediction_rows
)


# --------------------------------------------------
# 6. Overall crossed-holdout metrics
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
# 7. Print overall result
# --------------------------------------------------

print("\n\n========================================")
print("GPR Crossed Energy-Angle Holdout")
print("========================================")

print(
    f"Number of predictions: "
    f"{len(predictions_df)}"
)

print(f"MAE : {mae:.4f} nm")
print(f"RMSE: {rmse:.4f} nm")
print(f"R2  : {r2:.4f}")


# --------------------------------------------------
# 8. Show worst predictions
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
            "prediction_std_nm",
            "abs_error_nm",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# 9. Save all predictions
# --------------------------------------------------

predictions_df.to_csv(
    SCRIPT_DIR / "gpr_depth_crossed_predictions.csv",
    index=False,
)
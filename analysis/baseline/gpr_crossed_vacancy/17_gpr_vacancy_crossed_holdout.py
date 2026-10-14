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
# Script: 17_gpr_vacancy_crossed_holdout.py
# Purpose: Crossed energy–angle holdout validation for vacancy count.
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

target_col = "p2_vac_count"

y = df[target_col]


# --------------------------------------------------
# 2. Inspect target
# --------------------------------------------------

print("\nTarget:")
print(target_col)

print("\nObserved target range:")
print(f"Minimum = {y.min():.6f}")
print(f"Maximum = {y.max():.6f}")


# --------------------------------------------------
# 3. Grid coordinates
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
# 4. Storage
# --------------------------------------------------

prediction_rows = []


# --------------------------------------------------
# 5. Crossed Energy-Angle Holdout
# --------------------------------------------------

for held_energy in energies:

    for held_angle in angles:


        # ------------------------------------------
        # 5a. Training set
        #
        # Remove entire held-energy row
        # AND entire held-angle column
        # ------------------------------------------

        train_mask = (
            (df["E"] != held_energy)
            &
            (df["A"] != held_angle)
        )


        # ------------------------------------------
        # 5b. Single crossed test condition
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
        # 5c. Sanity checks
        # ------------------------------------------

        if len(X_train) != 81:

            raise ValueError(
                f"Expected 81 training conditions, "
                f"found {len(X_train)}"
            )

        if len(X_test) != 1:

            raise ValueError(
                f"Expected 1 test condition, "
                f"found {len(X_test)}"
            )


        # ------------------------------------------
        # 5d. Scale TRAINING inputs only
        # ------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )


        # ------------------------------------------
        # 5e. Fresh ARD-RBF kernel
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
        # 5f. Create model
        # ------------------------------------------

        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=8,
            random_state=42,
        )


        # ------------------------------------------
        # 5g. Train
        # ------------------------------------------

        model.fit(
            X_train_scaled,
            y_train,
        )


        # ------------------------------------------
        # 5h. Predict
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
        # 5i. Physical lower-bound audit
        # ------------------------------------------

        below_zero = (
            pred_value < 0.0
        )


        # ------------------------------------------
        # 5j. Store
        # ------------------------------------------

        prediction_rows.append(
            {
                "held_energy_keV":
                    held_energy,

                "held_angle_deg":
                    held_angle,

                "true_vac_count":
                    true_value,

                "predicted_vac_count":
                    pred_value,

                "prediction_std":
                    std_value,

                "residual":
                    residual,

                "abs_error":
                    abs(residual),

                "below_zero":
                    below_zero,

                "optimized_kernel":
                    str(model.kernel_),
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
    predictions_df["true_vac_count"],
    predictions_df["predicted_vac_count"],
)

rmse = mean_squared_error(
    predictions_df["true_vac_count"],
    predictions_df["predicted_vac_count"],
) ** 0.5

r2 = r2_score(
    predictions_df["true_vac_count"],
    predictions_df["predicted_vac_count"],
)


# --------------------------------------------------
# 8. Physical-bound audit
# --------------------------------------------------

n_below_zero = (
    predictions_df["below_zero"].sum()
)

pred_min = (
    predictions_df[
        "predicted_vac_count"
    ].min()
)

pred_max = (
    predictions_df[
        "predicted_vac_count"
    ].max()
)


# --------------------------------------------------
# 9. Print overall result
# --------------------------------------------------

print("\n")
print("========================================")
print("GPR VACANCY-COUNT CROSSED HOLDOUT")
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
    f"Predictions below zero: "
    f"{n_below_zero} / {len(predictions_df)}"
)


# --------------------------------------------------
# 10. Worst predictions
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
            "true_vac_count",
            "predicted_vac_count",
            "prediction_std",
            "abs_error",
            "below_zero",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# 11. Negative predictions
# --------------------------------------------------

negative_df = predictions_df[
    predictions_df["below_zero"]
]


if len(negative_df) > 0:

    print("\nPredictions Below Physical Lower Bound")
    print("--------------------------------------")

    print(
        negative_df[
            [
                "held_energy_keV",
                "held_angle_deg",
                "true_vac_count",
                "predicted_vac_count",
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "\nNo predictions below "
        "the physical lower bound of zero."
    )


# --------------------------------------------------
# 12. Save
# --------------------------------------------------

predictions_df.to_csv(
    SCRIPT_DIR / "gpr_vacancy_crossed_predictions.csv",
    index=False,
)
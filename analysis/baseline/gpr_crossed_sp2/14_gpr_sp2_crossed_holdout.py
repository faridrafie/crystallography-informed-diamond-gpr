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
# Script: 14_gpr_sp2_crossed_holdout.py
# Purpose: Crossed energy–angle holdout validation for the near-surface sp² fraction.
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

target_col = "p2_C_sp2_frac_0_2nm_trimmed_surface"

y = df[target_col]


# --------------------------------------------------
# 2. Inspect target
# --------------------------------------------------

print("\nTarget:")
print(target_col)

print("\nObserved target range:")
print(f"Minimum = {y.min():.8f}")
print(f"Maximum = {y.max():.8f}")


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
        # Remove complete energy row
        # AND complete angle column
        # ------------------------------------------

        train_mask = (
            (df["E"] != held_energy)
            &
            (df["A"] != held_angle)
        )


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
        # Sanity checks
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
        # Scale training inputs only
        # ------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )


        # ------------------------------------------
        # Fresh GPR kernel
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


        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=8,
            random_state=42,
        )


        # ------------------------------------------
        # Train
        # ------------------------------------------

        model.fit(
            X_train_scaled,
            y_train,
        )


        # ------------------------------------------
        # Predict
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


        outside_bounds = (
            pred_value < 0.0
            or
            pred_value > 1.0
        )


        prediction_rows.append(
            {
                "held_energy_keV":
                    held_energy,

                "held_angle_deg":
                    held_angle,

                "true_sp2_fraction":
                    true_value,

                "predicted_sp2_fraction":
                    pred_value,

                "prediction_std":
                    std_value,

                "residual":
                    residual,

                "abs_error":
                    abs(residual),

                "outside_0_1":
                    outside_bounds,

                "optimized_kernel":
                    str(model.kernel_),
            }
        )


# --------------------------------------------------
# 6. Combine
# --------------------------------------------------

predictions_df = pd.DataFrame(
    prediction_rows
)


# --------------------------------------------------
# 7. Metrics
# --------------------------------------------------

mae = mean_absolute_error(
    predictions_df["true_sp2_fraction"],
    predictions_df["predicted_sp2_fraction"],
)

rmse = mean_squared_error(
    predictions_df["true_sp2_fraction"],
    predictions_df["predicted_sp2_fraction"],
) ** 0.5

r2 = r2_score(
    predictions_df["true_sp2_fraction"],
    predictions_df["predicted_sp2_fraction"],
)


# --------------------------------------------------
# 8. Physical-bound audit
# --------------------------------------------------

n_outside = (
    predictions_df["outside_0_1"].sum()
)

pred_min = (
    predictions_df[
        "predicted_sp2_fraction"
    ].min()
)

pred_max = (
    predictions_df[
        "predicted_sp2_fraction"
    ].max()
)


# --------------------------------------------------
# 9. Print
# --------------------------------------------------

print("\n")
print("========================================")
print("GPR SP2-FRACTION CROSSED HOLDOUT")
print("========================================")

print(
    f"Number of predictions: "
    f"{len(predictions_df)}"
)

print(f"MAE : {mae:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"R2  : {r2:.6f}")


print("\nPrediction-range audit")
print("----------------------")

print(
    f"Minimum prediction: "
    f"{pred_min:.8f}"
)

print(
    f"Maximum prediction: "
    f"{pred_max:.8f}"
)

print(
    f"Predictions outside [0,1]: "
    f"{n_outside} / {len(predictions_df)}"
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
            "true_sp2_fraction",
            "predicted_sp2_fraction",
            "prediction_std",
            "abs_error",
            "outside_0_1",
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# 11. Nonphysical predictions
# --------------------------------------------------

outside_df = predictions_df[
    predictions_df["outside_0_1"]
]

if len(outside_df) > 0:

    print("\nPredictions Outside Physical Bounds")
    print("-----------------------------------")

    print(
        outside_df[
            [
                "held_energy_keV",
                "held_angle_deg",
                "true_sp2_fraction",
                "predicted_sp2_fraction",
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "\nNo predictions outside "
        "the physical [0,1] interval."
    )


# --------------------------------------------------
# 12. Save
# --------------------------------------------------

predictions_df.to_csv(
    SCRIPT_DIR / "gpr_sp2_crossed_predictions.csv",
    index=False,
)
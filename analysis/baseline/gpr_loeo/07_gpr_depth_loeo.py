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
# Script: 07_gpr_depth_loeo.py
# Purpose: Leave-one-energy-out validation for the median implantation-depth GPR surrogate.
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
# 2. Get all implantation energies
# --------------------------------------------------

energies = np.sort(
    df["E"].unique()
)

print("\nEnergies used for LOEO:")
print(energies)


# --------------------------------------------------
# 3. Containers for results
# --------------------------------------------------

summary_rows = []
prediction_rows = []


# --------------------------------------------------
# 4. Leave one energy out at a time
# --------------------------------------------------

for held_energy in energies:

    print("\n====================================")
    print(f"Held-out energy: {held_energy} keV")
    print("====================================")


    # ----------------------------------------------
    # 4a. Define train and test rows
    # ----------------------------------------------

    test_mask = df["E"] == held_energy
    train_mask = ~test_mask

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]


    # ----------------------------------------------
    # 4b. Scale using TRAINING data only
    # ----------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # ----------------------------------------------
    # 4c. Define a fresh kernel for this fold
    # ----------------------------------------------

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


    # ----------------------------------------------
    # 4d. Create GPR
    # ----------------------------------------------

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=42,
    )


    # ----------------------------------------------
    # 4e. Train on the remaining 9 energies
    # ----------------------------------------------

    model.fit(
        X_train_scaled,
        y_train,
    )


    # ----------------------------------------------
    # 4f. Predict the completely unseen energy
    # ----------------------------------------------

    y_pred, y_std = model.predict(
        X_test_scaled,
        return_std=True,
    )


    # ----------------------------------------------
    # 4g. Evaluate this held-out energy
    # ----------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    rmse = mean_squared_error(
        y_test,
        y_pred,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred,
    )


    # ----------------------------------------------
    # 4h. Save fold-level summary
    # ----------------------------------------------

    summary_rows.append(
        {
            "held_energy_keV": held_energy,
            "MAE_nm": mae,
            "RMSE_nm": rmse,
            "R2": r2,
            "optimized_kernel": str(model.kernel_),
        }
    )


    # ----------------------------------------------
    # 4i. Save condition-level predictions
    # ----------------------------------------------

    fold_results = X_test.copy()

    fold_results["true_depth"] = y_test
    fold_results["predicted_depth"] = y_pred
    fold_results["prediction_std"] = y_std

    fold_results["residual"] = (
        fold_results["true_depth"]
        - fold_results["predicted_depth"]
    )

    fold_results["held_energy_keV"] = held_energy

    prediction_rows.append(
        fold_results
    )


    # ----------------------------------------------
    # 4j. Print this fold
    # ----------------------------------------------

    print("\nOptimized Kernel:")
    print(model.kernel_)

    print("\nPerformance:")
    print(f"MAE : {mae:.4f} nm")
    print(f"RMSE: {rmse:.4f} nm")
    print(f"R2  : {r2:.4f}")


# --------------------------------------------------
# 5. Combine all folds
# --------------------------------------------------

summary_df = pd.DataFrame(
    summary_rows
)

predictions_df = pd.concat(
    prediction_rows,
    ignore_index=True,
)


# --------------------------------------------------
# 6. Overall pooled LOEO metrics
# --------------------------------------------------

overall_mae = mean_absolute_error(
    predictions_df["true_depth"],
    predictions_df["predicted_depth"],
)

overall_rmse = mean_squared_error(
    predictions_df["true_depth"],
    predictions_df["predicted_depth"],
) ** 0.5

overall_r2 = r2_score(
    predictions_df["true_depth"],
    predictions_df["predicted_depth"],
)


# --------------------------------------------------
# 7. Mean per-energy metrics
# --------------------------------------------------

mean_fold_mae = summary_df["MAE_nm"].mean()
mean_fold_rmse = summary_df["RMSE_nm"].mean()
mean_fold_r2 = summary_df["R2"].mean()


# --------------------------------------------------
# 8. Print overall results
# --------------------------------------------------

print("\n\n====================================")
print("Overall GPR LOEO Performance")
print("====================================")


print("\nPooled metrics")
print("--------------")

print(f"MAE : {overall_mae:.4f} nm")
print(f"RMSE: {overall_rmse:.4f} nm")
print(f"R2  : {overall_r2:.4f}")


print("\nMean per-energy metrics")
print("-----------------------")

print(f"Mean MAE : {mean_fold_mae:.4f} nm")
print(f"Mean RMSE: {mean_fold_rmse:.4f} nm")
print(f"Mean R2  : {mean_fold_r2:.4f}")


print("\nPer-Energy Summary")
print("------------------")

print(
    summary_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# 9. Save outputs
# --------------------------------------------------

summary_df.to_csv(
    SCRIPT_DIR / "gpr_depth_loeo_summary.csv",
    index=False,
)

predictions_df.to_csv(
    SCRIPT_DIR / "gpr_depth_loeo_predictions.csv",
    index=False,
)
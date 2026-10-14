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
# Script: 13_compare_modelA_modelB_remaining_targets.py
# Purpose: Compare the raw-tilt and crystallographic-Atlas GPR representations for shallow fraction, sp² fraction, and vacancy count.
#
# Research note:
#   This script belongs to the reproducibility package for my doctoral
#   research on simulation-guided low-energy nitrogen implantation in diamond.
#   Numerical settings used for manuscript results are intentionally kept
#   explicit in the script rather than hidden behind external configuration.
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel,
    Hyperparameter,
    ConstantKernel,
    WhiteKernel,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

FEATURE_DIR = PROJECT_ROOT / "atlas"

DATA_PATH = (
    FEATURE_DIR
    / "frozen_atlas_condition_features_scaled.csv"
)

MANIFEST_PATH = (
    FEATURE_DIR
    / "frozen_atlas_feature_manifest.csv"
)

SCALER_PATH = (
    FEATURE_DIR
    / "frozen_atlas_geometry_scaler.csv"
)


# ============================================================
# 2. Frozen targets
# ============================================================

TARGETS = {
    "shallow_fraction":
        "p1_frac_0_5nm",

    "near_surface_sp2":
        "p2_C_sp2_frac_0_2nm_trimmed_surface",

    "vacancy_count":
        "p2_vac_count",
}


# ============================================================
# 3. Same common crossed-64 benchmark
# ============================================================

HELD_ENERGIES = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
]

HELD_ANGLES = [
    3.0,
    7.0,
    10.0,
    15.0,
    18.43,
    20.0,
    22.0,
    25.0,
]

N_RESTARTS = 8
RANDOM_STATE = 42


# ============================================================
# 4. SAME grouped RBF used for depth
# ============================================================

class GroupedRBF(Kernel):

    def __init__(
        self,
        length_scale_energy=1.0,
        length_scale_geometry=1.0,
        length_scale_energy_bounds=(1e-2, 1e2),
        length_scale_geometry_bounds=(1e-2, 1e2),
    ):

        self.length_scale_energy = length_scale_energy
        self.length_scale_geometry = length_scale_geometry

        self.length_scale_energy_bounds = (
            length_scale_energy_bounds
        )

        self.length_scale_geometry_bounds = (
            length_scale_geometry_bounds
        )


    @property
    def hyperparameter_length_scale_energy(self):

        return Hyperparameter(
            "length_scale_energy",
            "numeric",
            self.length_scale_energy_bounds,
        )


    @property
    def hyperparameter_length_scale_geometry(self):

        return Hyperparameter(
            "length_scale_geometry",
            "numeric",
            self.length_scale_geometry_bounds,
        )


    def __call__(
        self,
        X,
        Y=None,
        eval_gradient=False,
    ):

        X = np.atleast_2d(
            X
        ).astype(float)


        if Y is None:

            Y_array = X

        else:

            if eval_gradient:

                raise ValueError(
                    "Gradient can only be evaluated "
                    "when Y is None."
                )

            Y_array = np.atleast_2d(
                Y
            ).astype(float)


        # Energy contribution
        dE2 = (
            X[:, None, 0]
            -
            Y_array[None, :, 0]
        ) ** 2


        # Orientation / crystallographic contribution
        dG2 = np.sum(
            (
                X[:, None, 1:]
                -
                Y_array[None, :, 1:]
            ) ** 2,
            axis=2,
        )


        lE = float(
            self.length_scale_energy
        )

        lG = float(
            self.length_scale_geometry
        )


        K = np.exp(
            -0.5
            * (
                dE2 / lE**2
                +
                dG2 / lG**2
            )
        )


        if eval_gradient:

            grad_energy = (
                K
                * dE2
                / lE**2
            )

            grad_geometry = (
                K
                * dG2
                / lG**2
            )


            gradient = np.dstack(
                (
                    grad_energy,
                    grad_geometry,
                )
            )


            return K, gradient


        return K


    def diag(
        self,
        X,
    ):

        return np.ones(
            np.atleast_2d(
                X
            ).shape[0]
        )


    def is_stationary(
        self,
    ):

        return True


    def __repr__(
        self,
    ):

        return (
            "GroupedRBF("
            f"length_scale_energy="
            f"{self.length_scale_energy:.4g}, "
            f"length_scale_geometry="
            f"{self.length_scale_geometry:.4g}"
            ")"
        )


# ============================================================
# 5. SAME GPR factory used for depth
# ============================================================

def make_gpr():

    kernel = (
        ConstantKernel(
            1.0,
            (
                1e-2,
                1e2,
            ),
        )
        *
        GroupedRBF(
            length_scale_energy=1.0,
            length_scale_geometry=1.0,
            length_scale_energy_bounds=(
                1e-2,
                1e2,
            ),
            length_scale_geometry_bounds=(
                1e-2,
                1e2,
            ),
        )
        +
        WhiteKernel(
            noise_level=1e-2,
            noise_level_bounds=(
                1e-6,
                1.0,
            ),
        )
    )


    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=N_RESTARTS,
        random_state=RANDOM_STATE,
    )


# ============================================================
# 6. Load frozen data
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

manifest_df = pd.read_csv(
    MANIFEST_PATH
)

scaler_df = pd.read_csv(
    SCALER_PATH
)


if len(df) != 100:

    raise ValueError(
        f"Expected 100 conditions, found {len(df)}."
    )


if df[
    [
        "E",
        "A",
    ]
].duplicated().any():

    raise ValueError(
        "Duplicate E-A conditions found."
    )


# ============================================================
# 7. Frozen Atlas feature order
# ============================================================

atlas_raw_features = (
    manifest_df[
        "feature_name"
    ]
    .tolist()
)


atlas_z_features = [
    feature.replace(
        "_deg",
        "_z",
    )
    for feature
    in atlas_raw_features
]


if len(atlas_z_features) != 15:

    raise ValueError(
        "Expected 15 scaled Atlas features."
    )


# ============================================================
# 8. Model-A theta geometry scaling
#
# Same fixed target-free geometry scaling as depth.
# ============================================================

d001_scaler = scaler_df[
    scaler_df[
        "feature_name"
    ]
    ==
    "d_axis_0_0_1_deg"
].iloc[0]


THETA_GEOM_MEAN = float(
    d001_scaler[
        "geometry_mean"
    ]
)

THETA_GEOM_STD = float(
    d001_scaler[
        "geometry_std"
    ]
)


df = df.copy()


df[
    "theta_geometry_z"
] = (
    (
        df["A"]
        -
        THETA_GEOM_MEAN
    )
    /
    THETA_GEOM_STD
)


# ============================================================
# 9. Metrics helper
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    return {
        "MAE":
            mean_absolute_error(
                y_true,
                y_pred,
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            ),

        "R2":
            r2_score(
                y_true,
                y_pred,
            ),
    }


# ============================================================
# 10. Containers
# ============================================================

all_prediction_rows = []
summary_rows = []


# ============================================================
# 11. Run each target
# ============================================================

for target_label, target_column in TARGETS.items():

    print("\n\n")
    print("=" * 110)
    print(
        f"TARGET: {target_label}"
    )
    print(
        f"COLUMN: {target_column}"
    )
    print("=" * 110)


    if target_column not in df.columns:

        raise ValueError(
            f"Missing target:\n{target_column}"
        )


    target_rows = []

    fold_id = 0


    # ========================================================
    # 12. Same crossed holdout
    # ========================================================

    for held_energy in HELD_ENERGIES:

        for held_angle in HELD_ANGLES:

            fold_id += 1


            train_mask = (
                ~np.isclose(
                    df["E"],
                    held_energy,
                )
                &
                ~np.isclose(
                    df["A"],
                    held_angle,
                )
            )


            test_mask = (
                np.isclose(
                    df["E"],
                    held_energy,
                )
                &
                np.isclose(
                    df["A"],
                    held_angle,
                )
            )


            train_df = df[
                train_mask
            ].copy()


            test_df = df[
                test_mask
            ].copy()


            if len(train_df) != 81:

                raise ValueError(
                    f"{target_label}, fold {fold_id}: "
                    f"expected 81 training rows, "
                    f"found {len(train_df)}."
                )


            if len(test_df) != 1:

                raise ValueError(
                    f"{target_label}, fold {fold_id}: "
                    f"expected 1 test row, "
                    f"found {len(test_df)}."
                )


            # =================================================
            # 12a. Energy scaling fit on training fold only
            # =================================================

            E_mean = float(
                train_df[
                    "E"
                ].mean()
            )


            E_std = float(
                train_df[
                    "E"
                ].std(
                    ddof=0
                )
            )


            E_train_z = (
                (
                    train_df["E"]
                    -
                    E_mean
                )
                /
                E_std
            ).to_numpy()


            E_test_z = (
                (
                    test_df["E"]
                    -
                    E_mean
                )
                /
                E_std
            ).to_numpy()


            # =================================================
            # 12b. Model A = E + theta
            # =================================================

            X_A_train = np.column_stack(
                (
                    E_train_z,
                    train_df[
                        "theta_geometry_z"
                    ].to_numpy(),
                )
            )


            X_A_test = np.column_stack(
                (
                    E_test_z,
                    test_df[
                        "theta_geometry_z"
                    ].to_numpy(),
                )
            )


            # =================================================
            # 12c. Model B = E + frozen Atlas representation
            # =================================================

            X_B_train = np.column_stack(
                (
                    E_train_z,
                    train_df[
                        atlas_z_features
                    ].to_numpy(
                        dtype=float
                    ),
                )
            )


            X_B_test = np.column_stack(
                (
                    E_test_z,
                    test_df[
                        atlas_z_features
                    ].to_numpy(
                        dtype=float
                    ),
                )
            )


            y_train = train_df[
                target_column
            ].to_numpy(
                dtype=float
            )


            y_true = float(
                test_df[
                    target_column
                ].iloc[0]
            )


            # =================================================
            # 12d. Model A
            # =================================================

            model_A = make_gpr()


            with warnings.catch_warnings(
                record=True
            ) as caught_A:

                warnings.simplefilter(
                    "always",
                    ConvergenceWarning,
                )

                model_A.fit(
                    X_A_train,
                    y_train,
                )


            warnings_A = sum(
                issubclass(
                    warning.category,
                    ConvergenceWarning,
                )
                for warning
                in caught_A
            )


            pred_A, std_A = model_A.predict(
                X_A_test,
                return_std=True,
            )


            pred_A = float(
                pred_A[0]
            )

            std_A = float(
                std_A[0]
            )


            # =================================================
            # 12e. Model B
            # =================================================

            model_B = make_gpr()


            with warnings.catch_warnings(
                record=True
            ) as caught_B:

                warnings.simplefilter(
                    "always",
                    ConvergenceWarning,
                )

                model_B.fit(
                    X_B_train,
                    y_train,
                )


            warnings_B = sum(
                issubclass(
                    warning.category,
                    ConvergenceWarning,
                )
                for warning
                in caught_B
            )


            pred_B, std_B = model_B.predict(
                X_B_test,
                return_std=True,
            )


            pred_B = float(
                pred_B[0]
            )

            std_B = float(
                std_B[0]
            )


            # =================================================
            # 12f. Kernel diagnostics
            # =================================================

            kernel_A = model_A.kernel_
            kernel_B = model_B.kernel_


            group_A = (
                kernel_A
                .k1
                .k2
            )

            group_B = (
                kernel_B
                .k1
                .k2
            )


            row = {
                "target_label":
                    target_label,

                "target_column":
                    target_column,

                "fold_id":
                    fold_id,

                "held_energy":
                    held_energy,

                "held_angle":
                    held_angle,

                "y_true":
                    y_true,

                "modelA_pred":
                    pred_A,

                "modelA_std":
                    std_A,

                "modelA_error":
                    y_true - pred_A,

                "modelA_abs_error":
                    abs(
                        y_true
                        -
                        pred_A
                    ),

                "modelB_pred":
                    pred_B,

                "modelB_std":
                    std_B,

                "modelB_error":
                    y_true - pred_B,

                "modelB_abs_error":
                    abs(
                        y_true
                        -
                        pred_B
                    ),

                "modelA_l_energy":
                    float(
                        group_A
                        .length_scale_energy
                    ),

                "modelA_l_geometry":
                    float(
                        group_A
                        .length_scale_geometry
                    ),

                "modelB_l_energy":
                    float(
                        group_B
                        .length_scale_energy
                    ),

                "modelB_l_geometry":
                    float(
                        group_B
                        .length_scale_geometry
                    ),

                "modelA_noise":
                    float(
                        kernel_A
                        .k2
                        .noise_level
                    ),

                "modelB_noise":
                    float(
                        kernel_B
                        .k2
                        .noise_level
                    ),

                "modelA_convergence_warnings":
                    warnings_A,

                "modelB_convergence_warnings":
                    warnings_B,
            }


            target_rows.append(
                row
            )

            all_prediction_rows.append(
                row
            )


            print(
                f"{target_label:>18s} | "
                f"Fold {fold_id:02d}/64 | "
                f"E={held_energy:>3.1f} "
                f"A={held_angle:>5.2f} | "
                f"|err| A="
                f"{abs(y_true-pred_A):.6f} "
                f"B="
                f"{abs(y_true-pred_B):.6f}"
            )


    # ========================================================
    # 13. Target-level metrics
    # ========================================================

    pred_df = pd.DataFrame(
        target_rows
    )


    y_true_array = pred_df[
        "y_true"
    ].to_numpy()


    y_A = pred_df[
        "modelA_pred"
    ].to_numpy()


    y_B = pred_df[
        "modelB_pred"
    ].to_numpy()


    metrics_A = calculate_metrics(
        y_true_array,
        y_A,
    )


    metrics_B = calculate_metrics(
        y_true_array,
        y_B,
    )


    A_abs = pred_df[
        "modelA_abs_error"
    ].to_numpy()


    B_abs = pred_df[
        "modelB_abs_error"
    ].to_numpy()


    A_wins = int(
        np.sum(
            A_abs
            <
            B_abs
        )
    )


    B_wins = int(
        np.sum(
            B_abs
            <
            A_abs
        )
    )


    ties = int(
        np.sum(
            np.isclose(
                A_abs,
                B_abs,
                atol=1e-12,
                rtol=0.0,
            )
        )
    )


    mae_reduction = (
        100.0
        * (
            metrics_A["MAE"]
            -
            metrics_B["MAE"]
        )
        /
        metrics_A["MAE"]
    )


    rmse_reduction = (
        100.0
        * (
            metrics_A["RMSE"]
            -
            metrics_B["RMSE"]
        )
        /
        metrics_A["RMSE"]
    )


    # ========================================================
    # 14. Physical-bound checks
    # ========================================================

    modelA_physical_violations = 0
    modelB_physical_violations = 0


    if target_label in [
        "shallow_fraction",
        "near_surface_sp2",
    ]:

        modelA_physical_violations = int(
            np.sum(
                (y_A < 0.0)
                |
                (y_A > 1.0)
            )
        )


        modelB_physical_violations = int(
            np.sum(
                (y_B < 0.0)
                |
                (y_B > 1.0)
            )
        )


    elif target_label == "vacancy_count":

        modelA_physical_violations = int(
            np.sum(
                y_A < 0.0
            )
        )


        modelB_physical_violations = int(
            np.sum(
                y_B < 0.0
            )
        )


    warning_A_total = int(
        pred_df[
            "modelA_convergence_warnings"
        ].sum()
    )


    warning_B_total = int(
        pred_df[
            "modelB_convergence_warnings"
        ].sum()
    )


    summary_rows.append(
        {
            "target_label":
                target_label,

            "target_column":
                target_column,

            "n_test":
                64,

            "modelA_MAE":
                metrics_A["MAE"],

            "modelB_MAE":
                metrics_B["MAE"],

            "modelA_RMSE":
                metrics_A["RMSE"],

            "modelB_RMSE":
                metrics_B["RMSE"],

            "modelA_R2":
                metrics_A["R2"],

            "modelB_R2":
                metrics_B["R2"],

            "modelB_MAE_reduction_percent":
                mae_reduction,

            "modelB_RMSE_reduction_percent":
                rmse_reduction,

            "modelA_point_wins":
                A_wins,

            "modelB_point_wins":
                B_wins,

            "ties":
                ties,

            "modelA_physical_violations":
                modelA_physical_violations,

            "modelB_physical_violations":
                modelB_physical_violations,

            "modelA_convergence_warnings":
                warning_A_total,

            "modelB_convergence_warnings":
                warning_B_total,
        }
    )


    # ========================================================
    # 15. Print target summary
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        f"{target_label.upper()}: MODEL A vs MODEL B"
    )
    print("=" * 100)


    print("\nModel A: E + theta")
    print("------------------")

    print(
        f"MAE  = {metrics_A['MAE']:.8f}"
    )

    print(
        f"RMSE = {metrics_A['RMSE']:.8f}"
    )

    print(
        f"R2   = {metrics_A['R2']:.8f}"
    )


    print("\nModel B: E + frozen Atlas")
    print("-------------------------")

    print(
        f"MAE  = {metrics_B['MAE']:.8f}"
    )

    print(
        f"RMSE = {metrics_B['RMSE']:.8f}"
    )

    print(
        f"R2   = {metrics_B['R2']:.8f}"
    )


    print("\nChange A -> B")
    print("-------------")

    print(
        f"MAE reduction  = "
        f"{mae_reduction:.2f}%"
    )

    print(
        f"RMSE reduction = "
        f"{rmse_reduction:.2f}%"
    )


    print("\nPoint wins")
    print("----------")

    print(
        f"Model A wins = {A_wins}"
    )

    print(
        f"Model B wins = {B_wins}"
    )

    print(
        f"Ties         = {ties}"
    )


    print("\nPhysical violations")
    print("-------------------")

    print(
        f"Model A = "
        f"{modelA_physical_violations}"
    )

    print(
        f"Model B = "
        f"{modelB_physical_violations}"
    )


    print("\nConvergence warnings")
    print("--------------------")

    print(
        f"Model A = "
        f"{warning_A_total}"
    )

    print(
        f"Model B = "
        f"{warning_B_total}"
    )


# ============================================================
# 16. Build final summary
# ============================================================

all_predictions_df = pd.DataFrame(
    all_prediction_rows
)

summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 17. Save
# ============================================================

all_predictions_df.to_csv(
    SCRIPT_DIR
    / "remaining_targets_modelA_vs_modelB_crossed64_predictions.csv",
    index=False,
)


summary_df.to_csv(
    SCRIPT_DIR
    / "remaining_targets_modelA_vs_modelB_crossed64_summary.csv",
    index=False,
)


print("\n\n")
print("=" * 120)
print("ALL REMAINING TARGETS — FINAL SUMMARY")
print("=" * 120)

print(
    summary_df[
        [
            "target_label",
            "modelA_MAE",
            "modelB_MAE",
            "modelA_RMSE",
            "modelB_RMSE",
            "modelA_R2",
            "modelB_R2",
            "modelA_point_wins",
            "modelB_point_wins",
            "modelA_physical_violations",
            "modelB_physical_violations",
        ]
    ].to_string(
        index=False
    )
)


print("\nSaved:")
print(
    "remaining_targets_modelA_vs_modelB_crossed64_predictions.csv"
)

print(
    "remaining_targets_modelA_vs_modelB_crossed64_summary.csv"
)
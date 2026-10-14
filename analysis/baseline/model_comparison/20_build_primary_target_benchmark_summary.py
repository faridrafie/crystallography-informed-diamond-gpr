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
# Script: 20_build_primary_target_benchmark_summary.py
# Purpose: Build the four-response benchmark summary used in manuscript reporting.
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


# --------------------------------------------------
# 1. Working directory
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# 2. Define the four primary targets
#
# summary_file:
#     model-level metrics already produced
#
# prediction_file:
#     common 64-point condition-level comparison
#
# prediction columns are specified explicitly because
# depth uses "_nm" names while the other targets do not.
# --------------------------------------------------

targets = {
    "Median depth": {
        "summary_file":
            "depth_gpr_vs_pchip_crossed_summary.csv",

        "prediction_file":
            "depth_gpr_vs_pchip_crossed_predictions.csv",

        "gpr_prediction_col":
            "gpr_prediction_nm",

        "pchip_prediction_col":
            "pchip_prediction_nm",

        "gpr_abs_error_col":
            "gpr_abs_error_nm",

        "pchip_abs_error_col":
            "pchip_abs_error_nm",

        "physical_lower":
            0.0,

        "physical_upper":
            None,
    },

    "Shallow fraction": {
        "summary_file":
            "shallow_gpr_vs_pchip_crossed_summary.csv",

        "prediction_file":
            "shallow_gpr_vs_pchip_crossed_predictions.csv",

        "gpr_prediction_col":
            "gpr_prediction",

        "pchip_prediction_col":
            "pchip_prediction",

        "gpr_abs_error_col":
            "gpr_abs_error",

        "pchip_abs_error_col":
            "pchip_abs_error",

        "physical_lower":
            0.0,

        "physical_upper":
            1.0,
    },

    "Near-surface sp2 fraction": {
        "summary_file":
            "sp2_gpr_vs_pchip_crossed_summary.csv",

        "prediction_file":
            "sp2_gpr_vs_pchip_crossed_predictions.csv",

        "gpr_prediction_col":
            "gpr_prediction",

        "pchip_prediction_col":
            "pchip_prediction",

        "gpr_abs_error_col":
            "gpr_abs_error",

        "pchip_abs_error_col":
            "pchip_abs_error",

        "physical_lower":
            0.0,

        "physical_upper":
            1.0,
    },

    "Vacancy count": {
        "summary_file":
            "vacancy_gpr_vs_pchip_crossed_summary.csv",

        "prediction_file":
            "vacancy_gpr_vs_pchip_crossed_predictions.csv",

        "gpr_prediction_col":
            "gpr_prediction",

        "pchip_prediction_col":
            "pchip_prediction",

        "gpr_abs_error_col":
            "gpr_abs_error",

        "pchip_abs_error_col":
            "pchip_abs_error",

        "physical_lower":
            0.0,

        "physical_upper":
            None,
    },
}


# --------------------------------------------------
# 3. Helper function:
#    normalize metric column names
#
# Depth uses MAE_nm / RMSE_nm.
# Other summaries use MAE / RMSE.
# --------------------------------------------------

def normalize_summary_columns(df):

    rename_map = {}

    if "MAE_nm" in df.columns:
        rename_map["MAE_nm"] = "MAE"

    if "RMSE_nm" in df.columns:
        rename_map["RMSE_nm"] = "RMSE"

    df = df.rename(
        columns=rename_map
    )

    required_columns = [
        "Model",
        "N",
        "MAE",
        "RMSE",
        "R2",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required summary columns: "
            f"{missing}"
        )

    return df


# --------------------------------------------------
# 4. Helper function:
#    count physically invalid predictions
# --------------------------------------------------

def count_physical_violations(
    values,
    lower_bound=None,
    upper_bound=None,
):

    violation_mask = pd.Series(
        False,
        index=values.index,
    )

    if lower_bound is not None:

        violation_mask = (
            violation_mask
            |
            (values < lower_bound)
        )

    if upper_bound is not None:

        violation_mask = (
            violation_mask
            |
            (values > upper_bound)
        )

    return int(
        violation_mask.sum()
    )


# --------------------------------------------------
# 5. Containers
# --------------------------------------------------

final_rows = []
long_rows = []


# --------------------------------------------------
# 6. Process each target
# --------------------------------------------------

for target_name, info in targets.items():

    print("\n")
    print("=" * 70)
    print(target_name)
    print("=" * 70)


    # ----------------------------------------------
    # 6a. Load summary and prediction files
    # ----------------------------------------------

    summary_path = (
        BASE_DIR
        / info["summary_file"]
    )

    prediction_path = (
        BASE_DIR
        / info["prediction_file"]
    )


    if not summary_path.exists():

        raise FileNotFoundError(
            f"Missing file: {summary_path}"
        )


    if not prediction_path.exists():

        raise FileNotFoundError(
            f"Missing file: {prediction_path}"
        )


    summary_df = pd.read_csv(
        summary_path
    )

    predictions_df = pd.read_csv(
        prediction_path
    )


    summary_df = (
        normalize_summary_columns(
            summary_df
        )
    )


    # ----------------------------------------------
    # 6b. Sanity checks
    # ----------------------------------------------

    if len(predictions_df) != 64:

        raise ValueError(
            f"{target_name}: expected 64 "
            f"common predictions, found "
            f"{len(predictions_df)}"
        )


    expected_models = {
        "GPR",
        "PCHIP",
    }

    found_models = set(
        summary_df["Model"]
    )


    if found_models != expected_models:

        raise ValueError(
            f"{target_name}: expected models "
            f"{expected_models}, found "
            f"{found_models}"
        )


    if not (
        summary_df["N"] == 64
    ).all():

        raise ValueError(
            f"{target_name}: summary does not "
            "contain N=64 for both models."
        )


    # ----------------------------------------------
    # 6c. Extract GPR and PCHIP metrics
    # ----------------------------------------------

    gpr_row = summary_df[
        summary_df["Model"] == "GPR"
    ].iloc[0]

    pchip_row = summary_df[
        summary_df["Model"] == "PCHIP"
    ].iloc[0]


    gpr_mae = float(
        gpr_row["MAE"]
    )

    gpr_rmse = float(
        gpr_row["RMSE"]
    )

    gpr_r2 = float(
        gpr_row["R2"]
    )


    pchip_mae = float(
        pchip_row["MAE"]
    )

    pchip_rmse = float(
        pchip_row["RMSE"]
    )

    pchip_r2 = float(
        pchip_row["R2"]
    )


    # ----------------------------------------------
    # 6d. Relative GPR improvement
    #
    # Positive:
    #     GPR has lower error
    #
    # Negative:
    #     PCHIP has lower error
    # ----------------------------------------------

    mae_reduction_pct = (
        (pchip_mae - gpr_mae)
        /
        pchip_mae
        *
        100.0
    )


    rmse_reduction_pct = (
        (pchip_rmse - gpr_rmse)
        /
        pchip_rmse
        *
        100.0
    )


    r2_gain = (
        gpr_r2
        -
        pchip_r2
    )


    # ----------------------------------------------
    # 6e. Point-by-point winners
    # ----------------------------------------------

    gpr_error_col = (
        info["gpr_abs_error_col"]
    )

    pchip_error_col = (
        info["pchip_abs_error_col"]
    )


    tolerance = 1e-12


    gpr_wins = int(
        (
            predictions_df[
                gpr_error_col
            ]
            <
            predictions_df[
                pchip_error_col
            ]
            - tolerance
        ).sum()
    )


    pchip_wins = int(
        (
            predictions_df[
                pchip_error_col
            ]
            <
            predictions_df[
                gpr_error_col
            ]
            - tolerance
        ).sum()
    )


    ties = int(
        (
            np.abs(
                predictions_df[
                    gpr_error_col
                ]
                -
                predictions_df[
                    pchip_error_col
                ]
            )
            <= tolerance
        ).sum()
    )


    if (
        gpr_wins
        + pchip_wins
        + ties
        != 64
    ):

        raise ValueError(
            f"{target_name}: pointwise winner "
            "counts do not sum to 64."
        )


    # ----------------------------------------------
    # 6f. Physical-bound audit
    # ----------------------------------------------

    gpr_prediction_col = (
        info["gpr_prediction_col"]
    )

    pchip_prediction_col = (
        info["pchip_prediction_col"]
    )


    lower_bound = (
        info["physical_lower"]
    )

    upper_bound = (
        info["physical_upper"]
    )


    gpr_violations = (
        count_physical_violations(
            predictions_df[
                gpr_prediction_col
            ],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    )


    pchip_violations = (
        count_physical_violations(
            predictions_df[
                pchip_prediction_col
            ],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    )


    # ----------------------------------------------
    # 6g. Metric winners
    # ----------------------------------------------

    if gpr_mae < pchip_mae:

        mae_winner = "GPR"

    elif pchip_mae < gpr_mae:

        mae_winner = "PCHIP"

    else:

        mae_winner = "Tie"


    if gpr_rmse < pchip_rmse:

        rmse_winner = "GPR"

    elif pchip_rmse < gpr_rmse:

        rmse_winner = "PCHIP"

    else:

        rmse_winner = "Tie"


    if gpr_r2 > pchip_r2:

        r2_winner = "GPR"

    elif pchip_r2 > gpr_r2:

        r2_winner = "PCHIP"

    else:

        r2_winner = "Tie"


    # ----------------------------------------------
    # 6h. Final wide row
    # ----------------------------------------------

    final_rows.append(
        {
            "Target":
                target_name,

            "N":
                64,

            "GPR_MAE":
                gpr_mae,

            "PCHIP_MAE":
                pchip_mae,

            "MAE_Winner":
                mae_winner,

            "GPR_MAE_Reduction_pct":
                mae_reduction_pct,

            "GPR_RMSE":
                gpr_rmse,

            "PCHIP_RMSE":
                pchip_rmse,

            "RMSE_Winner":
                rmse_winner,

            "GPR_RMSE_Reduction_pct":
                rmse_reduction_pct,

            "GPR_R2":
                gpr_r2,

            "PCHIP_R2":
                pchip_r2,

            "R2_Winner":
                r2_winner,

            "GPR_R2_Gain":
                r2_gain,

            "GPR_Point_Wins":
                gpr_wins,

            "PCHIP_Point_Wins":
                pchip_wins,

            "Ties":
                ties,

            "GPR_Physical_Violations":
                gpr_violations,

            "PCHIP_Physical_Violations":
                pchip_violations,
        }
    )


    # ----------------------------------------------
    # 6i. Long-format rows
    # ----------------------------------------------

    long_rows.append(
        {
            "Target":
                target_name,

            "Model":
                "GPR",

            "N":
                64,

            "MAE":
                gpr_mae,

            "RMSE":
                gpr_rmse,

            "R2":
                gpr_r2,

            "Physical_Violations":
                gpr_violations,
        }
    )


    long_rows.append(
        {
            "Target":
                target_name,

            "Model":
                "PCHIP",

            "N":
                64,

            "MAE":
                pchip_mae,

            "RMSE":
                pchip_rmse,

            "R2":
                pchip_r2,

            "Physical_Violations":
                pchip_violations,
        }
    )


    # ----------------------------------------------
    # 6j. Print per-target result
    # ----------------------------------------------

    print(
        f"GPR   : "
        f"MAE={gpr_mae:.8f}, "
        f"RMSE={gpr_rmse:.8f}, "
        f"R2={gpr_r2:.6f}"
    )

    print(
        f"PCHIP : "
        f"MAE={pchip_mae:.8f}, "
        f"RMSE={pchip_rmse:.8f}, "
        f"R2={pchip_r2:.6f}"
    )

    print(
        f"Point wins: "
        f"GPR={gpr_wins}, "
        f"PCHIP={pchip_wins}, "
        f"Ties={ties}"
    )


# --------------------------------------------------
# 7. Build final tables
# --------------------------------------------------

final_df = pd.DataFrame(
    final_rows
)

long_df = pd.DataFrame(
    long_rows
)


# --------------------------------------------------
# 8. Overall consistency checks
# --------------------------------------------------

n_targets = len(
    final_df
)


gpr_mae_wins = int(
    (
        final_df[
            "MAE_Winner"
        ]
        == "GPR"
    ).sum()
)


gpr_rmse_wins = int(
    (
        final_df[
            "RMSE_Winner"
        ]
        == "GPR"
    ).sum()
)


gpr_r2_wins = int(
    (
        final_df[
            "R2_Winner"
        ]
        == "GPR"
    ).sum()
)


# --------------------------------------------------
# 9. Print final primary-target benchmark
# --------------------------------------------------

print("\n\n")
print("=" * 100)
print("FINAL PRIMARY-TARGET BENCHMARK")
print("Common 64-point Interior Crossed Energy-Angle Holdout")
print("=" * 100)


display_columns = [
    "Target",
    "GPR_MAE",
    "PCHIP_MAE",
    "MAE_Winner",
    "GPR_RMSE",
    "PCHIP_RMSE",
    "RMSE_Winner",
    "GPR_R2",
    "PCHIP_R2",
    "R2_Winner",
    "GPR_Point_Wins",
    "PCHIP_Point_Wins",
]


print(
    final_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# --------------------------------------------------
# 10. Print aggregate interpretation
# --------------------------------------------------

print("\n")
print("=" * 70)
print("BENCHMARK INTERPRETATION")
print("=" * 70)


print(
    f"GPR lower MAE for "
    f"{gpr_mae_wins} / {n_targets} "
    f"primary targets."
)

print(
    f"GPR lower RMSE for "
    f"{gpr_rmse_wins} / {n_targets} "
    f"primary targets."
)

print(
    f"GPR higher R2 for "
    f"{gpr_r2_wins} / {n_targets} "
    f"primary targets."
)


# --------------------------------------------------
# 11. Scientifically safe automated conclusion
# --------------------------------------------------

if (
    gpr_rmse_wins == n_targets
    and
    gpr_r2_wins == n_targets
):

    print(
        "\nResult:"
    )

    print(
        "GPR achieved lower RMSE and higher R2 "
        "than PCHIP for all primary targets."
    )


if gpr_mae_wins != n_targets:

    print(
        "\nImportant limitation:"
    )

    print(
        "GPR did NOT achieve lower MAE for "
        "every primary target."
    )

    print(
        "Therefore, do not make an unqualified "
        "claim that GPR reduced every definition "
        "of prediction error across all outputs."
    )


# --------------------------------------------------
# 12. Save final benchmark tables
# --------------------------------------------------

final_output = (
    "primary_target_benchmark_summary.csv"
)

long_output = (
    "primary_target_benchmark_long.csv"
)


final_df.to_csv(
    final_output,
    index=False,
)

long_df.to_csv(
    long_output,
    index=False,
)


# --------------------------------------------------
# 13. Save protocol / provenance text
# --------------------------------------------------

protocol_text = """
PRIMARY TARGET BENCHMARK — FROZEN BASELINE
==========================================

Dataset
-------
Condition-level MD dataset:
master_condition_table_core.csv

Grid:
10 energies x 10 angles = 100 conditions.

Primary outputs
---------------
1. Median implantation depth
2. Fraction of implanted nitrogen within 0–5 nm
3. Near-surface sp2 fraction within 0–2 nm
4. Vacancy count

Validation protocol
-------------------
Common interior crossed energy-angle holdout.

Interior test energies:
1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5 keV

Interior test angles:
3, 7, 10, 15, 18.43, 20, 22, 25 degrees

Number of common test targets:
8 x 8 = 64

For each crossed target (E*, A*):
- remove every condition with E = E*
- remove every condition with A = A*
- training set therefore contains 81 conditions
- predict only the crossed target (E*, A*)

GPR
---
Inputs:
Energy and angle.

Input preprocessing:
StandardScaler fit on training data only.

Kernel:
ConstantKernel * ARD RBF + WhiteKernel

normalize_y:
True

Optimizer restarts:
8

PCHIP
-----
Deterministic two-path interpolation.

Path 1:
Energy -> Angle

Path 2:
Angle -> Energy

Final prediction:
average of both paths.

PCHIP is evaluated only on interior targets.
No extrapolation is allowed.

Comparison rule
---------------
GPR and PCHIP are compared only on the exact same
64 interior crossed targets.

No clipping of predictions is used.

Interpretation
--------------
MAE, RMSE, and R2 are reported separately.

A statement that one model "consistently reduces
prediction error" is only justified for a named
metric if that metric improves across all outputs.

The current benchmark should therefore distinguish
MAE from RMSE rather than using an unqualified
generic prediction-error claim.
""".strip()


protocol_output = (
    "primary_target_benchmark_protocol.txt"
)


with open(
    protocol_output,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        protocol_text
    )


# --------------------------------------------------
# 14. Final confirmation
# --------------------------------------------------

print("\n")
print("=" * 70)
print("SAVED")
print("=" * 70)

print(final_output)
print(long_output)
print(protocol_output)
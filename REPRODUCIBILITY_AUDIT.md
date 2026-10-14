# Reproducibility audit — Release Candidate 1

**Audit date:** 2026-10-14  
**Maintainer:** Seyed Faridedin Rafie (Farid Rafie)  
**Affiliation:** Materials Science and Engineering Department, University of Wisconsin-Milwaukee, Milwaukee, WI 53211, USA

This file records the final repository-level checks performed before the first public GitHub/Zenodo release candidate.

## Data integrity

- 100 unique condition-level energy–tilt rows.
- 1,000 replicate-level rows.
- Exactly 10 replicate trajectories per energy–tilt condition.
- 10 energy levels and 10 tilt levels.
- Replicate-to-condition aggregation independently recomputed for all four primary surrogate targets; maximum discrepancies were at floating-point precision (≤ 1.8 × 10⁻15 for depth and ≤ 1.1 × 10⁻16 for fractional targets; vacancy count difference = 0).

## Crossed-holdout benchmark

The common 64-condition interior benchmark was independently recomputed from the frozen prediction files. Key GPR values were verified as:

- median depth: MAE 0.477472 nm, RMSE 0.644283 nm, R² 0.944225;
- shallow fraction: MAE 0.046851, RMSE 0.063234, R² 0.947084;
- near-surface sp²: MAE 0.001368735, RMSE 0.001748770, R² 0.722099;
- vacancy count: MAE 9.569631, RMSE 12.298427, R² 0.973935.

PCHIP benchmark values and pointwise comparisons were also checked against the manuscript/SI source tables.

## Coordinate-wise generalization

Median-depth LOAO/LOEO pooled MAEs were independently recomputed:

- LOAO interior: 0.490008 nm;
- LOAO edge: 1.392478 nm;
- LOEO interior: 0.256330 nm;
- LOEO edge: 0.621053 nm.

## Crystallographic Atlas

- 15 frozen descriptors: four axis-distance and eleven plane-family-distance features.
- All primary campaign rows have φ = 0° and `atlas_theta_deg == A`.
- Geometry-standardized z-scores were independently recomputed from the frozen scaler; maximum discrepancy across all descriptors was < 7 × 10⁻16.
- Frozen Model A / Model B prediction files reproduce the manuscript/SI metrics, including Model-B vacancy MAE = 6.840085 and R² = 0.985746.

## Fractional-output audit

Standard GPR is unconstrained. The common 64-condition shallow-fraction benchmark contains 11 GPR predictions outside [0,1]. These values are intentionally **not clipped** for quantitative validation. Display-only clipping is used for the shallow-fraction continuous response surface.

## Code and figure checks

- All Python files compile under Python 3.13.5.
- Repository smoke test passes (`quick_check.py`).
- Independent numerical audit passes (`verification/verify_reported_results.py`).
- Figure scripts were executed after path cleanup and write rendered figures to `figures/rendered/` and numerical figure tables to `figures/source_data/`.
- ML-04 now consumes the frozen ML-03 source grid directly and does not require ML-03 to be run immediately beforehand.
- The previous orphan response-slice kernel CSV was removed.

## Environment used for the audit

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- scikit-learn 1.8.0
- Matplotlib 3.10.8

## Computational note

A complete Model A / Model B refit is intentionally not part of the fast audit because it requires many Gaussian-process hyperparameter optimizations. The full scripts are supplied, and the frozen fold-level predictions are independently checked against the reported manuscript and supplementary metrics.

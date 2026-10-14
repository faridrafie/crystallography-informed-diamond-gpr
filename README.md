# Crystallography-Informed Gaussian-Process Surrogate Modeling of Low-Energy Nitrogen Implantation in Diamond
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23376495.svg)](https://doi.org/10.5281/zenodo.23376495)

**Research code and processed data by Seyed Faridedin Rafie (Farid Rafie)**  
Materials Science and Engineering Department, University of Wisconsin-Milwaukee, Milwaukee, WI 53211, USA  
**Research period:** 2025–2026

This repository is the reproducibility package for the manuscript analysis. It contains the processed molecular-dynamics tables used by the surrogate models, the frozen crystallographic Atlas representation, validation outputs, and the scripts used to reproduce the reported quantitative results and figures.

> **Scope.** The manuscript and this repository use the primary fixed-azimuth campaign at **φ = 0°**. Azimuth-transfer analyses are outside the scope of the present manuscript and repository.

## Coordinate convention

The crystallographic coordinate system is defined by `z || [001]`, `x || [100]`, and `y || [010]`. Polar tilt `θ` is measured from `[001]`; azimuth `φ` is measured from `[100]` toward `[010]`. The primary MD campaign uses `φ = 0°`.

## Dataset used by the manuscript

- 10 implantation energies: 0.5–5.0 keV in 0.5 keV increments;
- 10 polar tilt angles: 0°, 3°, 7°, 10°, 15°, 18.43°, 20°, 22°, 25°, and 30°;
- 100 unique energy–tilt conditions;
- 10 independent replicate trajectories per condition;
- 1,000 replicate trajectories and 10,000 implantation events in the underlying MD campaign.

The four condition-level surrogate targets are:

1. median implantation depth (`p1_depth_median_nm`);
2. shallow implanted fraction within 0–5 nm (`p1_frac_0_5nm`);
3. near-surface sp² carbon fraction within 0–2 nm (`p2_C_sp2_frac_0_2nm_trimmed_surface`);
4. vacancy count (`p2_vac_count`).

The machine-learning models use **100 condition-level rows**. The included 1,000-row replicate table supports aggregation and provenance checks; replicate variability is not supplied to GPR as observation-specific uncertainty.

## Important validation distinction: 100 GPR predictions vs the common 64-point benchmark

The base crossed-holdout GPR scripts retain predictions for **all 100 energy–tilt conditions**, including boundary conditions. GPR can return predictions at those edge conditions.

The manuscript's direct **GPR–PCHIP benchmark is intentionally restricted to 64 common interior conditions**:

- energies 1.0–4.5 keV;
- tilts 3°, 7°, 10°, 15°, 18.43°, 20°, 22°, and 25°.

For each target `(E*, θ*)`, all training rows sharing either `E = E*` or `θ = θ*` are removed. PCHIP extrapolation is disabled, so only the common 64 interior conditions are used for manuscript-reported GPR–PCHIP MAE, RMSE, and R². This distinction is deliberate, not a discrepancy between scripts and the paper.

LOAO and LOEO are used in the manuscript specifically for **median implantation depth** to test generalization to an entirely missing tilt or energy coordinate.

## Fractional-output convention

Standard GPR is unconstrained and therefore can return values slightly outside the physical `[0,1]` interval for fractional targets. **No clipping is applied to validation predictions or validation metrics.** For the shallow-fraction continuous response surface only, the plotted mean is limited to `[0,1]` for physical visualization; the underlying validation calculations remain unmodified.

## Repository layout

```text
crystallography-informed-diamond-gpr/
├── data/                 # Processed MD tables + focused data dictionary
├── atlas/                # Frozen crystallographic descriptors and geometry scaler
├── analysis/
│   ├── baseline/         # GPR, PCHIP, crossed holdout, LOAO, LOEO
│   └── atlas_model/      # Model A vs Model B tests
├── figures/
│   ├── scripts/          # Figure-generation scripts
│   ├── source_data/      # Numerical source/audit data generated or consumed by figures
│   └── rendered/         # Convenience copies of rendered figures
├── verification/         # Independent checks of reported manuscript/SI values
├── CITATION.cff
├── PROJECT_INFO.md
├── LICENSE               # MIT license for code
├── DATA_LICENSE.md       # CC BY 4.0 notice for processed data
├── requirements.txt
├── PYTHON_VERSION.txt
└── quick_check.py
```

## Core files

- `data/master_condition_table_core.csv` — 100 condition-level rows used by GPR/PCHIP.
- `data/master_replicate_table_core.csv` — 1,000 processed replicate rows.
- `data/data_dictionary.csv` — definitions for manuscript-relevant inputs, targets, variability fields, post-processing parameters, and Atlas descriptors.
- `atlas/frozen_atlas_condition_features_scaled.csv` — frozen Model-B table containing the standardized Atlas descriptors.
- `atlas/frozen_atlas_feature_manifest.csv` — manifest of four axial and eleven plane-family features.
- `atlas/frozen_atlas_geometry_scaler.csv` — fixed geometry-domain scaling parameters.

The processed tables retain additional post-processing columns for data provenance. The surrogate calculations reported in the manuscript use only the inputs and targets identified above and in `data/data_dictionary.csv`.

## Software environment

The release candidate was audited with **Python 3.13.5** and the exact package versions pinned in `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run the fast structure check:

```bash
python quick_check.py
```

Run the independent numerical audit of manuscript/SI values:

```bash
python verification/verify_reported_results.py
```

## Reproducing the analysis

A practical order is:

1. run the four crossed-holdout GPR scripts under `analysis/baseline/gpr_crossed_*`;
2. run the four matching PCHIP scripts under `analysis/baseline/pchip_crossed_*`;
3. run the four comparison scripts in `analysis/baseline/model_comparison/`, then `20_build_primary_target_benchmark_summary.py`;
4. run `analysis/baseline/gpr_loao/05_gpr_depth_loao.py` and `analysis/baseline/gpr_loeo/07_gpr_depth_loeo.py`;
5. run Model A / Model B scripts in `analysis/atlas_model/` if a full crystallographic refit is desired;
6. run figure scripts from `figures/scripts/`.

The Model A / Model B crossed-holdout refits are substantially more computationally expensive than the lightweight verification checks. Frozen fold-level prediction files and summary tables are therefore included so that the reported metrics can be audited immediately with `verification/verify_reported_results.py`, while the full fitting scripts remain available for complete recomputation.

All analysis scripts resolve repository paths from their own file locations; they do not require a particular working directory.

### Figure / script / source-data map

| Manuscript item | Repository script | Principal numerical source |
|---|---|---|
| Figure 1, workflow overview | `figures/scripts/Fig0_paper.py` | schematic of repository workflow |
| Figure 2b, energy–tilt campaign | `figures/scripts/fig1_paper.py` | fixed 10 × 10 campaign definition |
| Figure 3, GPR parity | `figures/scripts/21_make_FigML01_GPR_parity.py` | `analysis/baseline/model_comparison/*_predictions.csv` |
| Figure 4, GPR vs PCHIP | `figures/scripts/22_make_FigML02_GPR_vs_PCHIP.py` | same common 64-point benchmark files |
| Figure 5, LOAO/LOEO | `figures/scripts/28_make_FigML07_GPR_generalization_stress_test.py` | `analysis/baseline/gpr_loao/`, `gpr_loeo/` |
| Figure 6, response landscapes | `figures/scripts/23_make_FigML03_GPR_response_surfaces.py` | `data/master_condition_table_core.csv` |
| Figure 7, frozen Atlas representation | `figures/scripts/31_make_FigAtlas03_Frozen_Representation.py` | `atlas/frozen_atlas_condition_features_scaled.csv` |
| Figure 8, Model A vs B | `figures/scripts/29_make_FigAtlas01_ModelA_vs_ModelB_phi0.py` | `analysis/atlas_model/phi0_modelA_vs_modelB_summary.csv` |
| Figure S1, uncertainty maps | `figures/scripts/24_make_FigML04_GPR_uncertainty_maps.py` | frozen ML-03 source grid in `figures/source_data/` |
| Figure S2, full-fit slices | `figures/scripts/24_make_FigML05_response_slices.py` | condition-level dataset |
| Figure S3, crossed-holdout slices | `figures/scripts/26_make_FigML05_crossed_holdout_slices.py` | common 64-point benchmark files |
| Figure S4, GPR–PCHIP local advantage | `figures/scripts/27_make_FigML06_GPR_PCHIP_error_advantage_maps.py` | common 64-point benchmark files |
| Figure S5, Model A–B local advantage | `figures/scripts/30_make_FigAtlas02_ModelA_ModelB_error_advantage_maps.py` | frozen Model A/B prediction files |

Figure 2 is a composite scientific schematic. Panel (b) is generated programmatically by `fig1_paper.py`; panels (a), (c), and (d) are explanatory schematics assembled for the manuscript and are not treated as independent numerical analyses.

Generated figures are written to `figures/rendered/`; numerical figure audit/source tables are written to `figures/source_data/`.

## Atlas representation

Model A uses implantation energy plus raw polar tilt. Model B replaces the raw orientation coordinate with a frozen 15-feature crystallographic representation: four axis-distance and eleven plane-family-distance descriptors. Raw `θ` and `φ` are not separately supplied to Model B, although the `[001]` axis-distance descriptor is geometrically equivalent to tilt.

The descriptor set and geometry-domain scaler were defined independently of MD response targets. The fixed geometry scaler spans `0° ≤ θ ≤ 30°` and `0° ≤ φ ≤ 45°` on a 0.25° reference grid. Energy remains standardized within each training fold.

## Data scope and raw trajectories

This package is designed to reproduce the surrogate-modeling results from the processed data used by the manuscript. The multi-hundred-gigabyte raw atomistic trajectory archive is not required to reproduce the GPR, PCHIP, structured validation, Atlas comparisons, or quantitative figure results supplied here.

## Figure typography

The repository uses a portable Matplotlib font configuration for reproducibility. The manuscript production environment may use Arial; font substitution affects typography only and does not alter numerical calculations or plotted data.

## Citation and archival release

Citation metadata are provided in `CITATION.cff`. The final public release can be archived through Zenodo to obtain a persistent DOI; the DOI should then be added to the manuscript Data/Code Availability statements and to the final repository metadata.

## License

Python code is released under the MIT License (`LICENSE`). Processed numerical data and figure source-data are released under CC BY 4.0 (`DATA_LICENSE.md`).

## Contact

**Seyed Faridedin Rafie (Farid Rafie)**  
Materials Science and Engineering Department  
University of Wisconsin-Milwaukee  
Milwaukee, WI 53211, USA

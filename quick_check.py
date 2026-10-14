# =============================================================================
# Farid Rafie | UWM Materials Science & Engineering | 2025–2026
#
# Research code accompanying:
# Crystallography-Informed Gaussian-Process Surrogate Modeling of
# Low-Energy Nitrogen Implantation in Diamond
#
# Author / maintainer:
#   Seyed Faridedin Rafie (Farid Rafie)
#   Materials Science and Engineering Department
#   University of Wisconsin-Milwaukee
#   Milwaukee, WI 53211, USA
#
# Script: quick_check.py
# Purpose: Fast structural integrity check for the public reproducibility package.
# =============================================================================
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
cond=pd.read_csv(ROOT/'data'/'master_condition_table_core.csv')
rep=pd.read_csv(ROOT/'data'/'master_replicate_table_core.csv')
atlas=pd.read_csv(ROOT/'atlas'/'frozen_atlas_condition_features_scaled.csv')

assert len(cond)==100, f'Expected 100 condition rows, found {len(cond)}'
assert len(rep)==1000, f'Expected 1000 replicate rows, found {len(rep)}'
assert len(atlas)==100, f'Expected 100 Atlas rows, found {len(atlas)}'
assert cond[['E','A']].drop_duplicates().shape[0]==100
assert cond['E'].nunique()==10 and cond['A'].nunique()==10
assert rep.groupby(['E','A']).size().eq(10).all()
assert np.allclose(atlas['atlas_phi_deg'],0.0)
assert np.allclose(atlas['atlas_theta_deg'],atlas['A'])

required=['p1_depth_median_nm','p1_frac_0_5nm','p2_C_sp2_frac_0_2nm_trimmed_surface','p2_vac_count']
for c in required:
    assert c in cond and c in rep, f'Missing primary target {c}'

print('Repository quick check passed.')
print('  condition rows :',len(cond))
print('  replicate rows :',len(rep))
print('  energy levels  :',cond['E'].nunique())
print('  tilt levels    :',cond['A'].nunique())
print('  Atlas rows     :',len(atlas))

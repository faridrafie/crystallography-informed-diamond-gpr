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
# Script: verify_reported_results.py
# Purpose: Independently audit the processed data and manuscript-reported metrics.
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
TOL = 5e-4  # supports comparison to the rounded manuscript/SI values

def metrics(y, p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return dict(MAE=mean_absolute_error(y,p),RMSE=float(np.sqrt(mean_squared_error(y,p))),R2=r2_score(y,p))

def assert_close(label, value, expected, tol=TOL):
    if not np.isfinite(value) or abs(value-expected)>tol:
        raise AssertionError(f"{label}: {value:.12g} != {expected:.12g} (tol={tol})")
    print(f"PASS  {label:<44s} {value:.9g}")

# Dataset structure and replicate aggregation.
cond=pd.read_csv(ROOT/'data'/'master_condition_table_core.csv')
rep=pd.read_csv(ROOT/'data'/'master_replicate_table_core.csv')
assert len(cond)==100 and len(rep)==1000
assert cond[['E','A']].drop_duplicates().shape[0]==100
assert rep.groupby(['E','A']).size().eq(10).all()
for col in ['p1_depth_median_nm','p1_frac_0_5nm','p2_C_sp2_frac_0_2nm_trimmed_surface','p2_vac_count']:
    agg=rep.groupby(['E','A'])[col].mean().sort_index()
    ref=cond.set_index(['E','A'])[col].sort_index()
    delta=float(np.max(np.abs(agg.to_numpy()-ref.to_numpy())))
    assert_close(f'aggregation {col}',delta,0.0,tol=1e-12)

# Common 64-point GPR/PCHIP benchmarks.
mc=ROOT/'analysis'/'baseline'/'model_comparison'
bench={
'depth':('depth_gpr_vs_pchip_crossed_predictions.csv','true_depth_nm','gpr_prediction_nm','pchip_prediction_nm'),
'shallow':('shallow_gpr_vs_pchip_crossed_predictions.csv','true_shallow_fraction','gpr_prediction','pchip_prediction'),
'sp2':('sp2_gpr_vs_pchip_crossed_predictions.csv','true_sp2_fraction','gpr_prediction','pchip_prediction'),
'vacancy':('vacancy_gpr_vs_pchip_crossed_predictions.csv','true_vac_count','gpr_prediction','pchip_prediction'),
}
expected=pd.read_csv(ROOT/'verification'/'expected_reported_metrics.csv')
for key,(fn,t,g,p) in bench.items():
    d=pd.read_csv(mc/fn)
    assert len(d)==64
    for model,col in [('GPR',g),('PCHIP',p)]:
        m=metrics(d[t],d[col])
        for metric,value in m.items():
            row=expected[(expected.result_group==f'crossed_{key}')&(expected.model==model)&(expected.metric==metric)]
            if len(row): assert_close(f'crossed {key} {model} {metric}',value,float(row.expected_value.iloc[0]))

# LOAO/LOEO pooled checks from prediction files.
def pooled_subset(df, hold_col, subset):
    vals=sorted(df[hold_col].unique())
    edge={vals[0],vals[-1]}
    mask=df[hold_col].isin(edge) if subset=='edge' else ~df[hold_col].isin(edge)
    x=df[mask]
    return metrics(x['true_depth'],x['predicted_depth'])
loao=pd.read_csv(ROOT/'analysis'/'baseline'/'gpr_loao'/'gpr_depth_loao_predictions.csv')
loeo=pd.read_csv(ROOT/'analysis'/'baseline'/'gpr_loeo'/'gpr_depth_loeo_predictions.csv')
for group,df,col in [('LOAO',loao,'held_angle_deg'),('LOEO',loeo,'held_energy_keV')]:
    for subset in ['interior','edge']:
        val=pooled_subset(df,col,subset)['MAE']
        exp=float(expected[(expected.result_group==f'{group}_{subset}')&(expected.metric=='MAE')].expected_value.iloc[0])
        assert_close(f'{group} {subset} MAE',val,exp,tol=5e-4)

# Frozen Atlas scaler consistency.
atlas=pd.read_csv(ROOT/'atlas'/'frozen_atlas_condition_features_scaled.csv')
scaler=pd.read_csv(ROOT/'atlas'/'frozen_atlas_geometry_scaler.csv')
for _,r in scaler.iterrows():
    raw=r['feature_name']; z=raw.replace('_deg','_z')
    calc=(atlas[raw]-r['geometry_mean'])/r['geometry_std']
    assert_close(f'Atlas scaler {raw}',float(np.max(np.abs(calc-atlas[z]))),0.0,tol=1e-12)
assert np.allclose(atlas['atlas_phi_deg'],0.0)
assert np.allclose(atlas['atlas_theta_deg'],atlas['A'])

# Model A vs B frozen comparison.
ab=pd.read_csv(ROOT/'analysis'/'atlas_model'/'phi0_modelA_vs_modelB_summary.csv').set_index('target_label')
mapkey={'median_depth':'atlas_depth','shallow_fraction':'atlas_shallow','near_surface_sp2':'atlas_sp2','vacancy_count':'atlas_vacancy'}
for target,group in mapkey.items():
    exp_rows=expected[(expected.result_group==group)&(expected.model=='Model B')]
    for _,er in exp_rows.iterrows():
        col={'MAE':'modelB_MAE','RMSE':'modelB_RMSE','R2':'modelB_R2'}[er.metric]
        assert_close(f'{target} Model B {er.metric}',float(ab.loc[target,col]),float(er.expected_value),tol=5e-10)

# Fractional-output audit: validation remains unclipped.
sh=pd.read_csv(mc/'shallow_gpr_vs_pchip_crossed_predictions.csv')
outside=int(((sh['gpr_prediction']<0)|(sh['gpr_prediction']>1)).sum())
print(f'INFO  shallow GPR predictions outside [0,1] on common 64 set: {outside}')
print('\nAll repository verification checks passed.')

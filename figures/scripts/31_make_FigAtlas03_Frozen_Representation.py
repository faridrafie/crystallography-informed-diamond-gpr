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
# Script: 31_make_FigAtlas03_Frozen_Representation.py
# Purpose: Generate the frozen crystallographic Atlas representation heat map.
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
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# This standalone repository script intentionally uses Matplotlib defaults.
# The manuscript version used the shared paper_plot_style.py typography.

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "atlas" / "frozen_atlas_condition_features_scaled.csv"
OUTPUT_BASE = Path(__file__).resolve().parent.parent / "rendered" / "Figure_Atlas03_Frozen_Crystallographic_Representation"
OUTPUT_DATA = Path(__file__).resolve().parent.parent / "source_data" / "Figure_Atlas03_Frozen_Crystallographic_Representation_Data.csv"

AXIS_FEATURES = [
    "d_axis_0_0_1_z", "d_axis_1_0_3_z", "d_axis_1_1_3_z", "d_axis_1_0_2_z"
]
PLANE_FEATURES = [
    "d_plane_1_0_0_z", "d_plane_1_1_0_z", "d_plane_2_0_1_z", "d_plane_2_1_0_z",
    "d_plane_2_1_1_z", "d_plane_2_2_1_z", "d_plane_3_0_1_z", "d_plane_3_1_0_z",
    "d_plane_3_1_1_z", "d_plane_3_2_0_z", "d_plane_3_2_1_z"
]
FEATURES = AXIS_FEATURES + PLANE_FEATURES
LABELS = [
    r"Axis $[001]$", r"Axis $[103]$", r"Axis $[113]$", r"Axis $[102]$",
    r"Plane $\{100\}$", r"Plane $\{110\}$", r"Plane $\{201\}$", r"Plane $\{210\}$",
    r"Plane $\{211\}$", r"Plane $\{221\}$", r"Plane $\{301\}$", r"Plane $\{310\}$",
    r"Plane $\{311\}$", r"Plane $\{320\}$", r"Plane $\{321\}$"
]

df = pd.read_csv(FEATURE_PATH)
required = ["E","A","atlas_theta_deg","atlas_phi_deg"] + FEATURES
missing = [c for c in required if c not in df.columns]
if missing: raise ValueError(f"Missing columns: {missing}")
if len(df) != 100: raise ValueError(f"Expected 100 condition rows, found {len(df)}")
if not np.allclose(df["atlas_phi_deg"].to_numpy(float), 0.0): raise ValueError("Expected phi=0 primary campaign")
if not np.allclose(df["atlas_theta_deg"].to_numpy(float), df["A"].to_numpy(float)): raise ValueError("Atlas theta != raw tilt")
for feat in FEATURES:
    spread = df.groupby("A")[feat].agg(lambda x: float(np.max(x)-np.min(x))).max()
    if spread > 1e-10: raise ValueError(f"{feat} varies with energy at fixed orientation")

odf = df[["A","atlas_theta_deg","atlas_phi_deg"]+FEATURES].groupby("A",as_index=False,sort=True).first()
angles = odf["A"].to_numpy(float)
expected = np.array([0,3,7,10,15,18.43,20,22,25,30],float)
if not np.allclose(angles,expected): raise ValueError(f"Unexpected angles: {angles}")
Z = odf[FEATURES].to_numpy(float).T

audit = pd.DataFrame(Z.T,columns=FEATURES)
audit.insert(0,"phi_deg",odf["atlas_phi_deg"].to_numpy(float))
audit.insert(0,"theta_deg",angles)
OUTPUT_DATA.parent.mkdir(parents=True,exist_ok=True)
audit.to_csv(OUTPUT_DATA,index=False)

m=float(np.max(np.abs(Z))) or 1.0
norm=TwoSlopeNorm(vmin=-m,vcenter=0,vmax=m)
fig,ax=plt.subplots(figsize=(7.1,4.55))
im=ax.imshow(Z,cmap="PuOr",norm=norm,origin="upper",interpolation="nearest",aspect="auto")
ax.set_xticks(np.arange(len(angles)), ["18.43" if np.isclose(a,18.43) else f"{a:g}" for a in angles])
ax.set_yticks(np.arange(len(FEATURES)), LABELS)
ax.set_xlabel(r"Tilt angle, $\theta$ (°)")
ax.set_title(r"Frozen Atlas representation at $\phi=0^\circ$")
ax.axhline(3.5,color="black",linewidth=1.15)
ax.set_xticks(np.arange(len(angles))-0.5,minor=True)
ax.set_yticks(np.arange(len(FEATURES))-0.5,minor=True)
ax.grid(which="minor",color="white",linewidth=0.5,alpha=0.5)
ax.tick_params(which="minor",bottom=False,left=False)
cbar=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.025)
cbar.set_label("Geometry-standardized descriptor, z")
cbar.set_ticks(np.linspace(-m,m,5))
fig.subplots_adjust(left=0.205,right=0.94,bottom=0.14,top=0.91)
OUTPUT_BASE.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUTPUT_BASE.with_suffix(".pdf"),bbox_inches="tight")
fig.savefig(OUTPUT_BASE.with_suffix(".svg"),bbox_inches="tight")
fig.savefig(OUTPUT_BASE.with_name(OUTPUT_BASE.name+"_600dpi.png"),dpi=600,bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_DATA}")

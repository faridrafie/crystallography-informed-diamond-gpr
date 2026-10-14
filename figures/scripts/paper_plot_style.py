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
# Script: paper_plot_style.py
# Purpose: Central plotting style used to keep manuscript figures visually consistent.
#
# Research note:
#   This script belongs to the reproducibility package for my doctoral
#   research on simulation-guided low-energy nitrogen implantation in diamond.
#   Numerical settings used for manuscript results are intentionally kept
#   explicit in the script rather than hidden behind external configuration.
# =============================================================================

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# MASTER PUBLICATION STYLE
# ============================================================

MM_TO_INCH = 1.0 / 25.4

# Approximate journal-style widths
SINGLE_COLUMN_MM = 89.0
DOUBLE_COLUMN_MM = 180.0

SINGLE_COLUMN_IN = SINGLE_COLUMN_MM * MM_TO_INCH
DOUBLE_COLUMN_IN = DOUBLE_COLUMN_MM * MM_TO_INCH


# ============================================================
# Fixed visual identity
# ============================================================

COLORS = {
    "md": "#111111",
    "gpr": "#0072B2",
    "pchip": "#D55E00",
    "reference": "#555555",
    "neutral": "#777777",
}


# ============================================================
# Typography / line hierarchy
# ============================================================

FONT_FAMILY = "DejaVu Sans"

FONT_AXIS = 9.0
FONT_TICK = 8.0
FONT_LEGEND = 8.0
FONT_ANNOTATION = 8.0
FONT_PANEL = 10.0
FONT_TITLE = 9.0

AXIS_LINEWIDTH = 0.8
TICK_LINEWIDTH = 0.8
DATA_LINEWIDTH = 1.2
REFERENCE_LINEWIDTH = 1.0


def apply_paper_style():
    """
    Apply the frozen publication style used for every figure
    in the manuscript and supplementary information.
    """

    # --------------------------------------------------------
    # Require the repository plotting font explicitly rather than silently switching fonts.
    # --------------------------------------------------------

    try:
        font_path = font_manager.findfont(
            FONT_FAMILY,
            fallback_to_default=False,
        )

        print(f"Using font: {FONT_FAMILY}")
        print(f"Font file: {font_path}")

    except Exception as exc:
        raise RuntimeError(
            f"{FONT_FAMILY} was not found by Matplotlib. "
            "Install the required plotting font before regenerating repository figures."
        ) from exc


    mpl.rcParams.update({

        # ----------------------------------------------------
        # Font
        # ----------------------------------------------------

        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY],

        "font.size": FONT_TICK,

        "axes.labelsize": FONT_AXIS,
        "axes.titlesize": FONT_TITLE,

        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,

        "legend.fontsize": FONT_LEGEND,


        # ----------------------------------------------------
        # Math text — visually consistent with the repository font
        # ----------------------------------------------------

        "mathtext.fontset": "custom",
        "mathtext.rm": "DejaVu Sans",
        "mathtext.it": "DejaVu Sans:italic",
        "mathtext.bf": "DejaVu Sans:bold",


        # ----------------------------------------------------
        # Axes
        # ----------------------------------------------------

        "axes.linewidth": AXIS_LINEWIDTH,

        "axes.spines.top": True,
        "axes.spines.right": True,

        "axes.grid": False,


        # ----------------------------------------------------
        # Ticks
        # ----------------------------------------------------

        "xtick.direction": "in",
        "ytick.direction": "in",

        "xtick.top": True,
        "ytick.right": True,

        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,

        "xtick.major.width": TICK_LINEWIDTH,
        "ytick.major.width": TICK_LINEWIDTH,

        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,


        # ----------------------------------------------------
        # Lines
        # ----------------------------------------------------

        "lines.linewidth": DATA_LINEWIDTH,


        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        "legend.frameon": False,


        # ----------------------------------------------------
        # Vector output
        # ----------------------------------------------------

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Keep SVG text editable
        "svg.fonttype": "none",


        # ----------------------------------------------------
        # Saving
        # ----------------------------------------------------

        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",

        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def add_panel_label(
    ax,
    label,
    x=-0.16,
    y=1.06,
):
    """
    Uniform manuscript panel label: (a), (b), ...
    """

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=FONT_PANEL,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def save_paper_figure(
    fig,
    output_base,
):
    """
    Save every final figure in:
      PDF
      SVG
      PNG 600 dpi
    """

    output_base = Path(output_base)

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    pdf_path = output_base.with_suffix(
        ".pdf"
    )

    svg_path = output_base.with_suffix(
        ".svg"
    )

    png_path = (
        output_base.parent
        /
        f"{output_base.name}_600dpi.png"
    )


    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )


    print("\nSaved:")
    print(pdf_path)
    print(svg_path)
    print(png_path)
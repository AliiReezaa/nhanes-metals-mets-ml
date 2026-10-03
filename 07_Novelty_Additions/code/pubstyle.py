"""
Shared publication-figure style for the manuscript figures (Q1-journal standard).

Design rules encoded here (applied identically to every figure so the figure set
reads as one system):
  * Okabe-Ito colourblind-safe palette. The four models that are statistically
    indistinguishable from the top model get saturated, maximally-separable hues;
    every other model is rendered in a neutral grey ramp with thinner strokes.
    This is both a legibility decision and an honest one -- it puts the visual
    emphasis on the tied top tier rather than implying a false ranking.
  * Liberation Sans (metrically identical to Arial, the most commonly required
    journal figure font).
  * No titles baked into the image. Captions live in the manuscript text, which
    is what journals ask for; a title inside the artwork duplicates the caption
    and is the single most common figure-formatting rejection.
  * Top/right spines removed, hairline axes, no chartjunk, no gridlines except
    where reading a value off the axis is the point of the panel.
  * Every figure written at 600 dpi PNG *and* vector PDF.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ----------------------------------------------------------------------
# Palette (Okabe & Ito 2008, colourblind-safe)
# ----------------------------------------------------------------------
OKABE_BLUE    = "#0072B2"
OKABE_VERMIL  = "#D55E00"
OKABE_GREEN   = "#009E73"
OKABE_ORANGE  = "#E69F00"
OKABE_SKY     = "#56B4E9"
OKABE_PURPLE  = "#CC79A7"
OKABE_YELLOW  = "#F0E442"
BLACK         = "#000000"

# Greys for the non-tied models
GREY_RAMP = ["#4D4D4D", "#666666", "#808080", "#999999", "#B3B3B3", "#C6C6C6", "#D9D9D9"]

# The four models statistically indistinguishable from the top model
# (08_evaluate.py paired bootstrap, 95% level). Order = descending test AUC.
TIER1_MODELS = ["CatBoost", "LightGBM", "XGBoost", "Gradient Boosting"]
TIER1_COLORS = {
    "CatBoost":          OKABE_BLUE,
    "LightGBM":          OKABE_VERMIL,
    "XGBoost":           OKABE_GREEN,
    "Gradient Boosting": OKABE_ORANGE,
}

# Figure widths (inches). Most journals: single column ~3.5 in, double ~7.1 in.
W_SINGLE = 3.46
def shap_cmap():
    """Colourblind-safe replacement for SHAP's default red-to-blue feature-value ramp.

    SHAP ships a magenta/blue diverging map that is hard to separate with deuteranopia or
    protanopia, and it does not match the Okabe-Ito palette the rest of the figure set
    uses. This runs Okabe-Ito blue (low) through a neutral mid-tone to Okabe-Ito orange
    (high), which keeps the same low/high reading for every kind of colour vision and in
    greyscale, where it reads as dark-to-light-to-dark by luminance.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "okabe_shap", [OKABE_BLUE, "#B9B9B9", OKABE_ORANGE], N=256)


W_DOUBLE = 7.09

DPI_RASTER = 600


def model_style(name: str, rank: int) -> dict:
    """Line style for a model. Tier-1 models get saturated colour + heavier
    stroke; everything else falls back to the grey ramp in rank order."""
    if name in TIER1_COLORS:
        return {"color": TIER1_COLORS[name], "lw": 1.7, "zorder": 3, "alpha": 1.0}
    grey_idx = min(max(rank - len(TIER1_MODELS), 0), len(GREY_RAMP) - 1)
    return {"color": GREY_RAMP[grey_idx], "lw": 1.0, "zorder": 2, "alpha": 0.95}


def apply_style() -> None:
    """Global rcParams. Call once at import time in each figure script."""
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.solid_capstyle": "round",
        "figure.dpi": 120,
        "savefig.dpi": DPI_RASTER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType, not Type-3 (journal requirement)
        "ps.fonttype": 42,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
    })


def panel_label(ax, letter: str, dx: float = -0.13, dy: float = 1.04) -> None:
    """Bold (A)/(B)/... panel label in the standard top-left position."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


def save(fig, path_noext, formats=("png", "pdf")) -> list:
    """Write a figure to every requested format. Returns the paths written."""
    written = []
    for fmt in formats:
        p = f"{path_noext}.{fmt}"
        fig.savefig(p, format=fmt)
        written.append(p)
    plt.close(fig)
    return written

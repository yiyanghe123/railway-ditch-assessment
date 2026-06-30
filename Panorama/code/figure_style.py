"""Shared figure style for the panorama pipeline, UNIFIED with the LiDAR
experiment figures (same fonts, sizes, grid, and colour design) so the two
experiments look identical in the thesis.

Palette / rcParams mirror the LiDAR cross-section figures
(``LiDAR/.../plot_lidar_gnss_cross_sections.py`` and ``step07_full_analysis.py``):
  * DejaVu Sans, font 9 / title 10 / label 9 / legend 8 / ticks 8;
  * top & right spines off, grid colour #d8dde6;
  * dark-slate profile line, red-dashed RUK datum, red star for the detected
    ditch, light-red ditch-search zone, light-blue secondary zone;
  * vector PDF + 300 dpi PNG, editable text (pdf.fonttype 42).

Use:
    from figure_style import apply_style, savefig, panel_label, PALETTE, CM
"""
from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- unified colour palette (identical hues to the LiDAR figures) ----
PALETTE = {
    "profile":        "#1f2937",   # main profile / lower-envelope line (dark slate)
    "profile_2":      "#94a3b8",   # secondary / median profile
    "points":         "#c5ccd6",   # raw points (light grey)
    "ruk":            "#dc2626",   # rail / RUK datum line (red, dashed)
    "ditch":          "#ef4444",   # detected ditch marker (red star)
    "search_zone":    "#fee2e2",   # ditch-search window shading (light red)
    "outer_zone":     "#dbeafe",   # secondary zone shading (light blue)
    "grid":           "#d8dde6",
    "edge":           "#333333",
}

FULL_WIDTH_CM = 17.0
HALF_WIDTH_CM = 8.8       # matches the LiDAR cross-section figure width


def CM(cm: float) -> float:
    """centimetres -> inches (for figsize)."""
    return cm / 2.54


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["edge"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def panel_label(ax, text: str, dx: float = -0.02, dy: float = 1.04) -> None:
    """Bold panel tag, e.g. (a), in axis-fraction coordinates."""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")


def savefig(fig, path_noext, formats=("pdf", "png")) -> None:
    """Save a figure as vector PDF + high-res PNG (publication deliverables)."""
    path_noext = Path(path_noext)
    path_noext.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(path_noext.with_suffix(f".{ext}"))
    plt.close(fig)

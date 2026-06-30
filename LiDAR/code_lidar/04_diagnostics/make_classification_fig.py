"""
Generate a classification-result strip figure from ditch_metrics_final.csv.

Three panels per side (left + right), showing along-track (S):
  Panel 1 — Shape class   (V_HEALTHY / U_MODERATE / FLAT_SILTED / NA)
  Panel 2 — Content class (Roelens 2018 5-class)
  Panel 3 — TDOK priority (Ö / Å / M / V / NA)

Usage:
    python make_classification_fig.py <tile_output_dir>
e.g. python make_classification_fig.py output_630_415
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Colour maps (keyed on class label) ─────────────────
SHAPE_COLORS = {
    "V_HEALTHY":   "#2ecc71",  # green
    "U_MODERATE":  "#f1c40f",  # yellow
    "FLAT_SILTED": "#e74c3c",  # red
    "NA":          "#dcdcdc",  # light grey
}

CONTENT_COLORS = {
    "DRY_CLEAN":        "#27ae60",  # healthy green
    "DRY_VEGETATED":    "#f4d03f",  # yellow-green (dry but vegetated)
    "WET_CLEAR":        "#3498db",  # clear water blue
    "WET_PARTIAL_VEG":  "#d35400",  # orange (wet + partial veg)
    "WET_DENSE_VEG":    "#7b2d26",  # dark red-brown (worst)
    "NA":               "#dcdcdc",
}

PRIORITY_COLORS = {
    "Ö":  "#2ecc71",   # opportunistic (healthy)
    "Å":  "#f1c40f",   # 3-year action
    "M":  "#e67e22",   # 3-month action
    "V":  "#e74c3c",   # 2-week action (critical)
    "NA": "#dcdcdc",
}

SHAPE_ORDER    = ["V_HEALTHY", "U_MODERATE", "FLAT_SILTED", "NA"]
CONTENT_ORDER  = ["DRY_CLEAN", "DRY_VEGETATED", "WET_CLEAR",
                  "WET_PARTIAL_VEG", "WET_DENSE_VEG", "NA"]
PRIORITY_ORDER = ["Ö", "Å", "M", "V", "NA"]


def draw_strip(ax, s_values, class_values, colour_map, label_order, row_y, row_h):
    """Draw a horizontal strip for one side's class values along S."""
    for cls, colour in colour_map.items():
        mask = class_values == cls
        if not mask.any():
            continue
        s = s_values[mask]
        # draw rectangles 1 m wide (or use default width)
        ax.bar(s, height=row_h, width=1.0, bottom=row_y,
               color=colour, edgecolor="none", align="center")


def build_figure(csv_path, out_path):
    df = pd.read_csv(csv_path, keep_default_na=False)
    s = pd.to_numeric(df["s"], errors="coerce").values

    # Replace empty strings with "NA" for plotting
    def _clean(col):
        return df[col].replace({"": "NA", "nan": "NA"}).values

    left_shape    = _clean("left_shape_class")
    right_shape   = _clean("right_shape_class")
    left_content  = _clean("left_content_class")
    right_content = _clean("right_content_class")
    left_prio     = _clean("left_tdok_priority")
    right_prio    = _clean("right_tdok_priority")

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"hspace": 0.35},
    )

    # Row layout within each axis: left on top (y=1), right on bottom (y=0)
    row_h = 0.9

    # ── Panel 1: Shape class ───────────────────────────
    ax = axes[0]
    draw_strip(ax, s, left_shape,  SHAPE_COLORS, SHAPE_ORDER, row_y=1.05, row_h=row_h)
    draw_strip(ax, s, right_shape, SHAPE_COLORS, SHAPE_ORDER, row_y=0.05, row_h=row_h)
    ax.set_ylim(-0.1, 2.1)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Right", "Left"])
    ax.set_title("Shape class", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    # Legend
    patches = [Patch(facecolor=SHAPE_COLORS[k], label=k)
               for k in SHAPE_ORDER]
    ax.legend(handles=patches, loc="upper right", ncol=4, fontsize=8,
              framealpha=0.9)

    # ── Panel 2: Content class ─────────────────────────
    ax = axes[1]
    draw_strip(ax, s, left_content,  CONTENT_COLORS, CONTENT_ORDER, row_y=1.05, row_h=row_h)
    draw_strip(ax, s, right_content, CONTENT_COLORS, CONTENT_ORDER, row_y=0.05, row_h=row_h)
    ax.set_ylim(-0.1, 2.1)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Right", "Left"])
    ax.set_title("Content class", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    patches = [Patch(facecolor=CONTENT_COLORS[k], label=k)
               for k in CONTENT_ORDER]
    ax.legend(handles=patches, loc="upper right", ncol=6, fontsize=7,
              framealpha=0.9)

    # ── Panel 3: TDOK priority ─────────────────────────
    ax = axes[2]
    draw_strip(ax, s, left_prio,  PRIORITY_COLORS, PRIORITY_ORDER, row_y=1.05, row_h=row_h)
    draw_strip(ax, s, right_prio, PRIORITY_COLORS, PRIORITY_ORDER, row_y=0.05, row_h=row_h)
    ax.set_ylim(-0.1, 2.1)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Right", "Left"])
    ax.set_xlabel("S — along-track distance (m)", fontsize=10)
    ax.set_title("TDOK 2015:0155 maintenance priority",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    patches = [Patch(facecolor=PRIORITY_COLORS[k], label=f"Prio {k}")
               for k in PRIORITY_ORDER]
    ax.legend(handles=patches, loc="upper right", ncol=5, fontsize=8,
              framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        tile_dir = "output_630_415"
    else:
        tile_dir = sys.argv[1]
    base = os.environ.get("LIDAR_BASE", os.getcwd())
    csv_path = os.path.join(base, tile_dir, "ditch_metrics_final.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)
    out_path = os.path.join(base, tile_dir, "Fig_classification_strip.png")
    build_figure(csv_path, out_path)

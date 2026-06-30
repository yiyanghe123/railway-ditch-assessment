"""
Step 08 - plan-view overview maps for the LiDAR ditch output.

The script reads the per-section output from Step 07 and draws three
publication-style plan-view figures:

  - ditch presence along the track
  - ditch depth below RUK
  - a combined presence/depth overview
"""

from __future__ import annotations

import json
import os

import laspy
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch


BASE_DIR = os.environ.get("LIDAR_BASE", os.getcwd())
BASE_OUT_DIR = os.environ.get("LIDAR_OUT_DIR", os.path.join(BASE_DIR, "output"))
TILE_ID = os.environ.get("LIDAR_TILE_ID", os.path.basename(BASE_OUT_DIR).replace("output_", ""))
STEP03_META = os.path.join(BASE_OUT_DIR, "step03_ST_metadata.json")
REFERENCE_CSV = os.path.join(BASE_OUT_DIR, "rail_reference_framework_step06.csv")
if not os.path.exists(REFERENCE_CSV):
    legacy_reference_csv = os.path.join(BASE_OUT_DIR, "rail_reference_framework_step08.csv")
    if os.path.exists(legacy_reference_csv):
        REFERENCE_CSV = legacy_reference_csv
DITCH_CSV = os.path.join(BASE_OUT_DIR, "ditch_metrics_final.csv")
LAZ_FILE = os.environ.get("LIDAR_LAZ_FILE", os.path.join(BASE_DIR, "data", "laz", "tile.laz"))
OUT_DIR = os.path.join(BASE_OUT_DIR, "step08_overview")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DPI = 300
DITCH_OFFSET = 4.0
PRESENCE_COLOURS = {
    "ditch": "#1f77b4",
    "no_ditch": "#d9d9d9",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_figure(fig: plt.Figure, stem: str) -> None:
    png = os.path.join(OUT_DIR, f"{stem}.png")
    pdf = os.path.join(OUT_DIR, f"{stem}.pdf")
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  Saved {os.path.basename(png)} and {os.path.basename(pdf)}")


def make_coloured_strip(ax, x, y, values, cmap, vmin, vmax, lw=6, alpha=0.9):
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=plt.Normalize(vmin, vmax),
        linewidths=lw,
        alpha=alpha,
        capstyle="round",
    )
    lc.set_array(values[:-1])
    ax.add_collection(lc)
    return lc


def make_presence_strip(ax, x, y, exists, lw=6, alpha=0.9):
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [
        PRESENCE_COLOURS["ditch"] if e else PRESENCE_COLOURS["no_ditch"]
        for e in exists[:-1]
    ]
    lc = LineCollection(segments, colors=colors, linewidths=lw, alpha=alpha, capstyle="round")
    ax.add_collection(lc)
    return lc


def make_depth_strip(ax, x, y, values, exists, cmap="viridis", vmin=0.0, vmax=1.5, lw=6, alpha=0.9):
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin, vmax)
    colors = []
    for value, ok in zip(values[:-1], exists[:-1]):
        if ok and np.isfinite(value):
            colors.append(cmap_obj(norm(value)))
        else:
            colors.append(PRESENCE_COLOURS["no_ditch"])
    lc = LineCollection(segments, colors=colors, linewidths=lw, alpha=alpha, capstyle="round")
    ax.add_collection(lc)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    return sm


def chainage_marks(s_vals: np.ndarray, n_marks: int = 7) -> list[float]:
    s_min = float(np.nanmin(s_vals))
    s_max = float(np.nanmax(s_vals))
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
        return []
    raw = np.linspace(s_min, s_max, n_marks)
    return [round(v / 100.0) * 100.0 for v in raw]


def setup_track_strip_axes(ax, title, s_vals):
    ax.set_aspect("auto")
    ax.set_xlabel("Chainage S (m)")
    ax.set_ylabel("Side")
    ax.set_title(title, fontweight="bold")
    ax.plot(s_vals, np.zeros_like(s_vals), color="#222222", lw=1.4, alpha=0.85, zorder=5)
    ax.set_yticks([-DITCH_OFFSET, 0.0, DITCH_OFFSET])
    ax.set_yticklabels(["Right", "Track", "Left"])
    ax.set_ylim(-DITCH_OFFSET * 1.9, DITCH_OFFSET * 1.9)

    for s_mark in sorted(set(chainage_marks(s_vals))):
        idx = int(np.argmin(np.abs(s_vals - s_mark)))
        ax.plot(s_vals[idx], 0.0, "k|", ms=8, mew=1.2, zorder=6)
        ax.annotate(
            f"S={s_mark:.0f} m",
            (s_vals[idx], 0.0),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=7,
            color="#222222",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )
    ax.margins(x=0.02, y=0.12)


print("=" * 60)
print("STEP 08 - Overview maps")
print("=" * 60)

with open(STEP03_META, "r", encoding="utf-8") as f:
    meta3 = json.load(f)

ref = pd.read_csv(REFERENCE_CSV).sort_values("s").reset_index(drop=True)
ditch = pd.read_csv(DITCH_CSV).sort_values("s").reset_index(drop=True)

las = laspy.open(LAZ_FILE)
x0_laz = las.header.x_min
y0_laz = las.header.y_min

e_center = ref["X_center_refined_local"].values + x0_laz
n_center = ref["Y_center_refined_local"].values + y0_laz
s_vals = ref["s"].values

df = ref[["s"]].copy()
df["E"] = e_center
df["N"] = n_center
ditch_sub = ditch[[
    "s",
    "left_ditch_exists",
    "left_depth",
    "right_ditch_exists",
    "right_depth",
]].rename(columns={
    "left_ditch_exists": "l_exists",
    "left_depth": "l_depth",
    "right_ditch_exists": "r_exists",
    "right_depth": "r_depth",
})
df = df.merge(ditch_sub, on="s", how="left")

print(f"Tile: {TILE_ID}")
print(f"Sections: {len(df)}")
print(f"E range: [{df['E'].min():.1f}, {df['E'].max():.1f}]")
print(f"N range: [{df['N'].min():.1f}, {df['N'].max():.1f}]")

dE = np.gradient(e_center)
dN = np.gradient(n_center)
track_len = np.sqrt(dE ** 2 + dN ** 2)
tE = dE / np.maximum(track_len, 1e-6)
tN = dN / np.maximum(track_len, 1e-6)
nE = -tN
nN = tE

e_left = e_center + nE * DITCH_OFFSET
n_left = n_center + nN * DITCH_OFFSET
e_right = e_center - nE * DITCH_OFFSET
n_right = n_center - nN * DITCH_OFFSET

plot_x_center = s_vals
plot_y_center = np.zeros_like(s_vals)
plot_x_left = s_vals
plot_y_left = np.full_like(s_vals, DITCH_OFFSET, dtype=float)
plot_x_right = s_vals
plot_y_right = np.full_like(s_vals, -DITCH_OFFSET, dtype=float)

l_exists_arr = df["l_exists"].fillna(0).astype(int).values.astype(bool)
r_exists_arr = df["r_exists"].fillna(0).astype(int).values.astype(bool)
l_depth = df["l_depth"].values.copy()
r_depth = df["r_depth"].values.copy()
l_depth_plot = np.asarray(l_depth, dtype=float)
r_depth_plot = np.asarray(r_depth, dtype=float)

legend_patches = [
    Patch(facecolor=PRESENCE_COLOURS["ditch"], edgecolor="#666666", label="Ditch detected"),
    Patch(facecolor=PRESENCE_COLOURS["no_ditch"], edgecolor="#666666", label="No ditch"),
]

print("\nGenerating ditch presence map...")
fig11, ax11 = plt.subplots(figsize=(10.5, 3.8))
setup_track_strip_axes(ax11, "Ditch presence along track", s_vals)
make_presence_strip(ax11, plot_x_left, plot_y_left, l_exists_arr, lw=9)
make_presence_strip(ax11, plot_x_right, plot_y_right, r_exists_arr, lw=9)
ax11.legend(handles=legend_patches, loc="best", framealpha=0.95)
save_figure(fig11, "Fig11_presence_map")
plt.close(fig11)

print("Generating ditch depth map...")
fig12, ax12 = plt.subplots(figsize=(10.5, 3.8))
setup_track_strip_axes(ax12, "Ditch depth below RUK", s_vals)
make_depth_strip(ax12, plot_x_left, plot_y_left, l_depth_plot, l_exists_arr, cmap="viridis", vmin=0, vmax=1.5, lw=9)
sm12 = make_depth_strip(ax12, plot_x_right, plot_y_right, r_depth_plot, r_exists_arr, cmap="viridis", vmin=0, vmax=1.5, lw=9)
fig12.colorbar(sm12, ax=ax12, shrink=0.75, label="Depth below RUK (m)", pad=0.02)
save_figure(fig12, "Fig12_depth_map")
plt.close(fig12)

print("Generating combined overview...")
fig14, (ax14a, ax14b) = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
setup_track_strip_axes(ax14a, "A. Ditch presence", s_vals)
make_presence_strip(ax14a, plot_x_left, plot_y_left, l_exists_arr, lw=8)
make_presence_strip(ax14a, plot_x_right, plot_y_right, r_exists_arr, lw=8)
ax14a.legend(handles=legend_patches, loc="best", framealpha=0.95)

setup_track_strip_axes(ax14b, "B. Ditch depth below RUK", s_vals)
make_depth_strip(ax14b, plot_x_left, plot_y_left, l_depth_plot, l_exists_arr, cmap="viridis", vmin=0, vmax=1.5, lw=8)
sm14 = make_depth_strip(ax14b, plot_x_right, plot_y_right, r_depth_plot, r_exists_arr, cmap="viridis", vmin=0, vmax=1.5, lw=8)
fig14.colorbar(sm14, ax=ax14b, shrink=0.72, label="Depth below RUK (m)", pad=0.02)
fig14.tight_layout()
save_figure(fig14, "Fig14_combined_dashboard")
plt.close(fig14)

print("\n" + "=" * 60)
print("STEP 08 COMPLETE")
print("=" * 60)
print(f"Output: {OUT_DIR}")

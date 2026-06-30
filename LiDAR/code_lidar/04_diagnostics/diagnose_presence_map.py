"""
Plan-view ditch presence + depth map, built DIRECTLY from step07's
output (ditch_metrics_final.csv).  Reflects the latest algorithm fixes.

Outputs (to output_630_415/):
  Fig_presence_map_latest.png     — binary L/R ditch presence along track
  Fig_depth_map_latest.png        — L/R depth color-coded along track
"""
import os, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
import laspy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ───── Paths ────────────────────────────────────────────────
BASE_DIR      = os.environ.get("LIDAR_BASE", os.getcwd())
OUT_DIR       = os.environ.get("LIDAR_OUT_DIR", os.path.join(BASE_DIR, "output"))
DITCH_CSV     = os.path.join(OUT_DIR, "ditch_metrics_final.csv")
REFERENCE_CSV = os.path.join(OUT_DIR, "rail_reference_framework_step06.csv")
STEP03_META   = os.path.join(OUT_DIR, "step03_ST_metadata.json")
LAZ_FILE = os.environ.get("LIDAR_LAZ_FILE", os.path.join(BASE_DIR, "data", "laz", "tile.laz"))

DITCH_OFFSET = 4.0
STRIP_WIDTH  = 3.0

plt.rcParams.update({"figure.facecolor": "white", "font.size": 10})


# ───── Load ─────────────────────────────────────────────────
with open(STEP03_META) as f:
    meta = json.load(f)
angle_rad = np.radians(float(meta["pca_track_azimuth_deg"]))
mean_x    = float(meta["mean_x_local"])
mean_y    = float(meta["mean_y_local"])

with laspy.open(LAZ_FILE) as lf:
    x0 = float(lf.header.x_min); y0 = float(lf.header.y_min)

ref = pd.read_csv(REFERENCE_CSV).sort_values("s").reset_index(drop=True)
df  = pd.read_csv(DITCH_CSV).sort_values("s").reset_index(drop=True)

t_col = "T_center_refined" if "T_center_refined" in ref.columns else "T_center"
d_col = "dT_dS_refined"    if "dT_dS_refined"    in ref.columns else "dT_dS"

# Align ref rows to df by nearest-s match
ref_s_sorted = ref["s"].values
merged = df.copy()
match_idx = np.searchsorted(ref_s_sorted, df["s"].values)
match_idx = np.clip(match_idx, 0, len(ref)-1)
merged["T_center_used"] = ref.iloc[match_idx][t_col].values
merged["dT_dS_used"]    = ref.iloc[match_idx][d_col].values

# ───── Centreline UTM positions for each section ───────────
cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
def ST_to_UTM(S, T):
    dx =  S*cos_a - T*sin_a
    dy =  S*sin_a + T*cos_a
    return x0 + mean_x + dx, y0 + mean_y + dy

E_center, N_center = ST_to_UTM(
    merged["s"].values, merged["T_center_used"].values
)

# Unit normal vector (perpendicular to track tangent) in UTM
dtds = merged["dT_dS_used"].fillna(0.0).values
tangent_S = np.ones_like(dtds)
tangent_T = dtds
ts_norm = np.sqrt(tangent_S**2 + tangent_T**2)
tangent_S /= ts_norm; tangent_T /= ts_norm
# normal in (S, T): (-dtds, 1) normalised
normal_S = -tangent_T
normal_T =  tangent_S
# normal in UTM: rotate by same angle
nE = normal_S * cos_a - normal_T * sin_a
nN = normal_S * sin_a + normal_T * cos_a
# normalise in UTM
norm = np.sqrt(nE**2 + nN**2)
nE /= norm; nN /= norm

# Left / right ditch strip positions
E_left  = E_center + nE * DITCH_OFFSET
N_left  = N_center + nN * DITCH_OFFSET
E_right = E_center - nE * DITCH_OFFSET
N_right = N_center - nN * DITCH_OFFSET


# ───── Figure 1: Presence map (binary) ──────────────────────
def build_segments(E_arr, N_arr):
    """Pairs of consecutive points for LineCollection."""
    pts = np.column_stack([E_arr, N_arr])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    return segs

left_exists  = merged["left_ditch_exists"].values.astype(bool)
right_exists = merged["right_ditch_exists"].values.astype(bool)

fig, ax = plt.subplots(figsize=(16, 5))
# Track centreline (thin grey)
ax.plot(E_center, N_center, "-", color="gray", lw=0.8, alpha=0.6,
        label="Track centreline")
# Left side
segs_L = build_segments(E_left, N_left)
colors_L = np.where(left_exists[:-1], "#1f77b4", "#e0e0e0")
lc_L = LineCollection(segs_L, colors=colors_L, lw=STRIP_WIDTH*2, zorder=3,
                      capstyle="butt")
ax.add_collection(lc_L)
# Right side
segs_R = build_segments(E_right, N_right)
colors_R = np.where(right_exists[:-1], "#1f77b4", "#e0e0e0")
lc_R = LineCollection(segs_R, colors=colors_R, lw=STRIP_WIDTH*2, zorder=3,
                      capstyle="butt")
ax.add_collection(lc_R)

# S markers every 100 m
for s_mark in range(int(merged["s"].min()//100)*100,
                    int(merged["s"].max()//100)*100 + 1, 100):
    idx = int(np.argmin(np.abs(merged["s"].values - s_mark)))
    ax.annotate(f"{s_mark:+d}",
                xy=(E_center[idx], N_center[idx]),
                xytext=(0, -15), textcoords="offset points",
                fontsize=7, ha="center", color="black")
    ax.plot(E_center[idx], N_center[idx], "k|", ms=10)

# Side labels (Left / Right)
i_end = np.argmax(merged["s"].values)   # highest S end
i_start = np.argmin(merged["s"].values)
ax.annotate("Left",
            xy=(E_left[i_start], N_left[i_start]), fontsize=11,
            color="navy", weight="bold")
ax.annotate("Right",
            xy=(E_right[i_start], N_right[i_start]), fontsize=11,
            color="navy", weight="bold")

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend(
    handles=[
        Patch(color="#1f77b4", label="Ditch detected"),
        Patch(color="#e0e0e0", label="No ditch"),
    ],
    title="Detection", loc="upper left", fontsize=9,
)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "Fig_presence_map_latest.png")
plt.savefig(out1, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")


# ───── Figure 2: Depth map (continuous) ────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(E_center, N_center, "-", color="gray", lw=0.8, alpha=0.5,
        label="Track")

# Color normalization: 0 - 1.5 m, above clipped
norm_depth = Normalize(vmin=0.0, vmax=1.5)
cmap = plt.cm.plasma

# Left depth
d_L = merged["left_depth"].values.copy()
d_L[~left_exists] = np.nan
# Plot segment by segment so missing = gray line
for i in range(len(merged)-1):
    if np.isnan(d_L[i]):
        col = "#e8e8e8"
    else:
        col = cmap(norm_depth(d_L[i]))
    ax.plot([E_left[i], E_left[i+1]], [N_left[i], N_left[i+1]],
            "-", color=col, lw=STRIP_WIDTH*2, solid_capstyle="butt")

# Right depth
d_R = merged["right_depth"].values.copy()
d_R[~right_exists] = np.nan
for i in range(len(merged)-1):
    if np.isnan(d_R[i]):
        col = "#e8e8e8"
    else:
        col = cmap(norm_depth(d_R[i]))
    ax.plot([E_right[i], E_right[i+1]], [N_right[i], N_right[i+1]],
            "-", color=col, lw=STRIP_WIDTH*2, solid_capstyle="butt")

# S markers
for s_mark in range(int(merged["s"].min()//100)*100,
                    int(merged["s"].max()//100)*100 + 1, 100):
    idx = int(np.argmin(np.abs(merged["s"].values - s_mark)))
    ax.annotate(f"{s_mark:+d}",
                xy=(E_center[idx], N_center[idx]),
                xytext=(0, -15), textcoords="offset points",
                fontsize=7, ha="center")
    ax.plot(E_center[idx], N_center[idx], "k|", ms=10)

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")
ax.grid(alpha=0.3)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_depth)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Depth below RUK (m)", fontsize=9)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "Fig_depth_map_latest.png")
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

print()
print("Note: any cached images in output_630_415/step08_overview/ may be stale")
print("      if step07 was re-run since. This script always reflects the latest CSV.")

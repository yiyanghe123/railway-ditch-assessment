"""
Diagnostic: render 30 cross-sections across the tile to eyeball whether
the current ditch detection is correct.

Reads:
  - output_630_415/ditch_metrics_final.csv         (step07 detection output)
  - output_630_415/rail_reference_framework_step06.csv
  - output_630_415/step03_ST_metadata.json
  - the LAZ point cloud for the tile

Writes:
  - output_630_415/Fig_30_sections_diagnostic.png  (5 cols x 6 rows grid)

Samples 30 S-positions uniformly across the tile, rebuilds each cross-
section's lower envelope (same formula as step07), and overlays the
detected ditch spans + search zones + rail positions + Z_ref so the
user can judge visually which detections are right / wrong.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import laspy
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.signal import savgol_filter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ═══════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════
BASE_DIR = os.environ.get("LIDAR_BASE", os.getcwd())
OUT_DIR = os.environ.get("LIDAR_OUT_DIR", os.path.join(BASE_DIR, "output"))
LAZ_FILE = os.environ.get("LIDAR_LAZ_FILE", os.path.join(BASE_DIR, "data", "laz", "tile.laz"))
DITCH_CSV     = os.path.join(OUT_DIR, "ditch_metrics_final.csv")
REFERENCE_CSV = os.path.join(OUT_DIR, "rail_reference_framework_step06.csv")
STEP03_META   = os.path.join(OUT_DIR, "step03_ST_metadata.json")

# ═══════════════════════════════════════════════════════════
# Constants — match step07_full_analysis.py
# ═══════════════════════════════════════════════════════════
GROUND_Z_RANGE          = 8.0
ENV_V_RANGE             = 7.0
ENV_BIN_W               = 0.05
ENV_Q                   = 5
ENV_MIN_POINTS_PER_BIN  = 3
MORPH_KERNEL_SIZE       = 3
MORPH_POST_SMOOTH_WIN   = 5
LOCAL_SECTION_HALF_U    = 0.75
ROUGH_S_PRESELECT_HALF  = 2.0
ROUGH_T_PRESELECT_HALF  = 8.0
SIDE_SEARCH_MIN         = 0.50
SIDE_SEARCH_MAX         = 8.00

N_SECTIONS = 30
NROWS, NCOLS = 6, 5


# ═══════════════════════════════════════════════════════════
# Helpers (copied from step07)
# ═══════════════════════════════════════════════════════════
def to_ST(xv, yv, mean_x, mean_y, angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    S = (xv - mean_x) * cos_a + (yv - mean_y) * sin_a
    T = -(xv - mean_x) * sin_a + (yv - mean_y) * cos_a
    return S, T


def project_to_local_frame(S_vals, T_vals, s0, t0, dtds):
    tangent = np.array([1.0, dtds], dtype=float)
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-dtds, 1.0], dtype=float)
    normal /= np.linalg.norm(normal)
    dS = S_vals - s0
    dT = T_vals - t0
    u = dS * tangent[0] + dT * tangent[1]
    v = dS * normal[0] + dT * normal[1]
    return u, v


def sorted_slice_bounds(sorted_vals, lo, hi):
    i0 = int(np.searchsorted(sorted_vals, lo, side="left"))
    i1 = int(np.searchsorted(sorted_vals, hi, side="right"))
    return i0, i1


def make_savgol(y, max_win=21, poly=3):
    y = np.asarray(y)
    if len(y) < 5:
        return y.copy()
    win = min(len(y), max_win)
    if win % 2 == 0:
        win -= 1
    win = max(win, 5)
    poly = min(poly, win - 2)
    return savgol_filter(y, window_length=win, polyorder=poly)


def build_lower_envelope(v_vals, z_vals, v_range=7.0, bin_w=0.05, q=5,
                         min_points=3):
    """Match step07's build_lower_envelope exactly (min-of-flanks fill)."""
    bins = np.arange(-v_range, v_range + bin_w, bin_w)
    n_bins = len(bins) - 1
    grid_v = np.array([(bins[j] + bins[j + 1]) / 2.0 for j in range(n_bins)])
    grid_z = np.full(n_bins, np.nan)
    for j in range(n_bins):
        m = (v_vals >= bins[j]) & (v_vals < bins[j + 1])
        if m.sum() >= min_points:
            grid_z[j] = np.percentile(z_vals[m], q)
    valid = ~np.isnan(grid_z)
    if valid.sum() < 5:
        idx = np.where(valid)[0]
        return grid_v[idx], grid_z[idx]
    first_valid = np.argmax(valid)
    last_valid = len(valid) - 1 - np.argmax(valid[::-1])
    sl = slice(first_valid, last_valid + 1)
    gv = grid_v[sl].copy()
    gz = grid_z[sl].copy()
    # min-of-flanks gap fill
    valid_inner = ~np.isnan(gz)
    if valid_inner.sum() >= 2 and (~valid_inner).sum() > 0:
        gz_s = pd.Series(gz)
        left_fill = gz_s.ffill().values
        right_fill = gz_s.bfill().values
        left_fill = np.where(np.isnan(left_fill), right_fill, left_fill)
        right_fill = np.where(np.isnan(right_fill), left_fill, right_fill)
        gap_mask = ~valid_inner
        gz[gap_mask] = np.minimum(left_fill[gap_mask], right_fill[gap_mask])
    if len(gz) < 5:
        return gv, gz
    ez = minimum_filter1d(gz, size=MORPH_KERNEL_SIZE, mode="nearest")
    ez = maximum_filter1d(ez, size=MORPH_KERNEL_SIZE, mode="nearest")
    if len(ez) >= 5:
        ez = make_savgol(ez, max_win=MORPH_POST_SMOOTH_WIN, poly=2)
    return gv, ez


def resolve_reference_columns(ref_df):
    """Match step07 field names."""
    cols = ref_df.columns
    t_col = "T_center_refined" if "T_center_refined" in cols else "T_center"
    d_col = "dT_dS_refined"    if "dT_dS_refined"    in cols else "dT_dS"
    zref_col = "Z_ref_smooth"  if "Z_ref_smooth"     in cols else "Z_ref"
    return t_col, d_col, zref_col


# ═══════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════
print("Loading step03 metadata...")
with open(STEP03_META, "r", encoding="utf-8") as f:
    meta = json.load(f)
angle_rad = np.radians(float(meta["pca_track_azimuth_deg"]))
mean_x = float(meta["mean_x_local"])
mean_y = float(meta["mean_y_local"])
print(f"  PCA azimuth: {np.degrees(angle_rad):.2f} deg")

print(f"\nLoading LAZ: {LAZ_FILE}")
las = laspy.read(LAZ_FILE)
x0 = float(las.header.x_min)
y0 = float(las.header.y_min)

X = (np.asarray(las.x, dtype=np.float64) - x0).astype(np.float32)
Y = (np.asarray(las.y, dtype=np.float64) - y0).astype(np.float32)
Z = np.asarray(las.z, dtype=np.float32)
I = np.asarray(las.intensity, dtype=np.float32)
print(f"  {len(X):,} points loaded")

# Ground subset (1st pct + GROUND_Z_RANGE)
z1 = float(np.percentile(Z, 1))
gmask = (Z >= z1) & (Z <= z1 + GROUND_Z_RANGE)
X = X[gmask]; Y = Y[gmask]; Z = Z[gmask]; I = I[gmask]
print(f"  {len(X):,} ground points (z in [{z1:.2f}, {z1 + GROUND_Z_RANGE:.2f}])")

print("\nComputing ST...")
Sg, Tg = to_ST(X, Y, mean_x, mean_y, angle_rad)
Sg = Sg.astype(np.float32)
Tg = Tg.astype(np.float32)
order = np.argsort(Sg)
Sg = Sg[order]; Tg = Tg[order]
Zg = Z[order]; Ig = I[order]
del X, Y, Z, I, las, gmask

print("\nLoading ditch + reference CSVs...")
df_ditch = pd.read_csv(DITCH_CSV)
ref = pd.read_csv(REFERENCE_CSV)
t_col, d_col, zref_col = resolve_reference_columns(ref)
print(f"  df_ditch: {len(df_ditch)} rows, ref columns: t={t_col}, d={d_col}, zref={zref_col}")

# Sample 30 S-positions uniformly
S_min = float(df_ditch["s"].min())
S_max = float(df_ditch["s"].max())
# Drop a few metres from each end to avoid edge artefacts
s_samples = np.linspace(S_min + 5.0, S_max - 5.0, N_SECTIONS)
print(f"\nSampling {N_SECTIONS} sections from S = {s_samples[0]:.0f} to {s_samples[-1]:.0f} m")


# ═══════════════════════════════════════════════════════════
# Render
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(4.5 * NCOLS, 3.0 * NROWS))
axes = np.atleast_2d(axes).ravel()

for idx, s_target in enumerate(s_samples):
    ax = axes[idx]

    # Nearest 1m section in ditch CSV
    row_1m_idx = int(np.argmin(np.abs(df_ditch["s"].values - s_target)))
    row_1m = df_ditch.iloc[row_1m_idx]
    s0 = float(row_1m["s"])

    # Matching reference row (for T_center + dT/dS + Z_ref)
    ref_idx = int(np.argmin(np.abs(ref["s"].values - s0)))
    ref_row = ref.iloc[ref_idx]
    s_ref = float(ref_row["s"])
    t_ref = float(ref_row[t_col])
    d_ref = float(ref_row[d_col])

    # Extract corridor (±2 m along S, ±8 m across T)
    i0r, i1r = sorted_slice_bounds(
        Sg, s_ref - ROUGH_S_PRESELECT_HALF, s_ref + ROUGH_S_PRESELECT_HALF
    )
    S_sub = Sg[i0r:i1r]; T_sub = Tg[i0r:i1r]
    Z_sub = Zg[i0r:i1r]; I_sub = Ig[i0r:i1r]

    tm = np.abs(T_sub - t_ref) <= ROUGH_T_PRESELECT_HALF
    S_sub = S_sub[tm]; T_sub = T_sub[tm]
    Z_sub = Z_sub[tm]; I_sub = I_sub[tm]

    u, v = project_to_local_frame(S_sub, T_sub, s_ref, t_ref, d_ref)
    sm = (np.abs(u) <= LOCAL_SECTION_HALF_U) & (np.abs(v) <= ENV_V_RANGE)
    v_sec = v[sm]; z_sec = Z_sub[sm]; i_sec = I_sub[sm]

    if len(v_sec) < 50:
        ax.text(0.5, 0.5, f"S={s0:.0f} m\nno section data",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        continue

    # Envelope (for overlay)
    env_v, env_z = build_lower_envelope(
        v_sec, z_sec, v_range=ENV_V_RANGE, bin_w=ENV_BIN_W, q=ENV_Q,
        min_points=ENV_MIN_POINTS_PER_BIN,
    )

    z_ref   = float(row_1m["Z_ref"]) if not pd.isna(row_1m["Z_ref"]) else np.nan
    half_g  = float(row_1m.get("half_gauge_prior", 0.718))

    # Thin the scatter to ~3000 points / subplot
    step_plot = max(1, len(v_sec) // 3000)
    ax.scatter(v_sec[::step_plot], z_sec[::step_plot],
               c=i_sec[::step_plot], cmap="plasma_r",
               s=0.25, alpha=0.35, vmin=0, vmax=10000)

    if len(env_v):
        ax.plot(env_v, env_z, color="darkred", lw=1.3, zorder=5)
    if not np.isnan(z_ref):
        ax.axhline(z_ref, color="cyan", lw=0.8, ls="--", zorder=4)

    # Rail and search zones
    ax.axvline(-half_g, color="lime", lw=0.9, ls="--")
    ax.axvline(+half_g, color="lime", lw=0.9, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.axvspan(-SIDE_SEARCH_MAX, -SIDE_SEARCH_MIN, alpha=0.05, color="cyan")
    ax.axvspan(+SIDE_SEARCH_MIN, +SIDE_SEARCH_MAX, alpha=0.05, color="lime")

    # Detected ditch spans
    l_w = row_1m.get("left_width", np.nan)
    r_w = row_1m.get("right_width", np.nan)
    if row_1m.get("left_ditch_exists", 0) == 1 and not pd.isna(l_w):
        ax.axvspan(
            -float(row_1m["left_span_x1"]),
            -float(row_1m["left_span_x0"]),
            alpha=0.30, color="steelblue"
        )
    if row_1m.get("right_ditch_exists", 0) == 1 and not pd.isna(r_w):
        ax.axvspan(
            float(row_1m["right_span_x0"]),
            float(row_1m["right_span_x1"]),
            alpha=0.30, color="tomato"
        )

    # ── Compact title ──
    l_d    = row_1m.get("left_depth",      np.nan)
    r_d    = row_1m.get("right_depth",     np.nan)
    l_conf = float(row_1m.get("left_confidence",  0.0))
    r_conf = float(row_1m.get("right_confidence", 0.0))
    l_src  = str(row_1m.get("left_pass_source",  "none"))
    r_src  = str(row_1m.get("right_pass_source", "none"))
    l_wat  = "W" if row_1m.get("left_water_flag",  0) else "."
    r_wat  = "W" if row_1m.get("right_water_flag", 0) else "."

    ld_s = "--" if pd.isna(l_d) else f"{l_d:.2f}"
    rd_s = "--" if pd.isna(r_d) else f"{r_d:.2f}"

    # Shorten pass source labels
    short = {
        "pass1":              "P1",
        "pass2_cutting":      "P2c",
        "pass2_embankment":   "P2e",
        "pass3_toe":          "P3",
        "pass4_gap":          "P4",
        "pass4_gap_redetect": "P4r",
        "gap_interpolated":   "Gi",
        "position_outlier":   "Out",
        "none":               "--",
    }
    l_src_s = short.get(l_src, l_src[:4])
    r_src_s = short.get(r_src, r_src[:4])

    ax.set_title(f"S = {s0:.0f} m", fontsize=8)
    ax.set_xlim(-ENV_V_RANGE, ENV_V_RANGE)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.15)


plt.tight_layout()

out_path = os.path.join(OUT_DIR, "Fig_30_sections_diagnostic.png")
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")

import os
import json
import laspy
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# USER SETTINGS
# =========================================================
BASE_DIR = os.environ.get("LIDAR_BASE", os.getcwd())
LAZ_FILE = os.environ.get("LIDAR_LAZ_FILE", os.path.join(BASE_DIR, "data", "laz", "tile.laz"))
OUT_DIR = os.environ.get("LIDAR_OUT_DIR", os.path.join(BASE_DIR, "output"))
os.makedirs(OUT_DIR, exist_ok=True)

# from Step 1 / Step 2 (updated to match Step 2 best-set output)
GROUND_Z_RANGE = 8.0
SAT_MAX = 0.25
RGB_MEAN_MIN = 35.0
RGB_MEAN_MAX = 130.0

SCAN_ANGLE_MAX = 20   # consistent with Step 2

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =========================================================
# HELPERS
# =========================================================
def fmt_n(x):
    return f"{int(x):,}"


def save_show(fig, out_path):
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved: {os.path.basename(out_path)}")


def hsv_saturation(R, G, B):
    """HSV saturation: 0 = achromatic (gray), 1 = fully saturated."""
    R32 = np.asarray(R, dtype=np.float32)
    G32 = np.asarray(G, dtype=np.float32)
    B32 = np.asarray(B, dtype=np.float32)
    cmax = np.maximum(R32, G32)
    np.maximum(cmax, B32, out=cmax)
    cmin = np.minimum(R32, G32)
    np.minimum(cmin, B32, out=cmin)
    sat = np.where(cmax > 0, 1.0 - cmin / cmax, np.float32(0.0))
    del cmin
    rgb_mean = (R32 + G32 + B32) / np.float32(3.0)
    return sat, rgb_mean


def scan_angle_degrees(las):
    """Return scan angle in degrees for both LAS 1.2 and LAS 1.4 encodings."""
    if hasattr(las, "scan_angle_rank"):
        vals = np.asarray(las.scan_angle_rank, dtype=np.float32)
    elif hasattr(las, "scan_angle"):
        vals = np.asarray(las.scan_angle, dtype=np.float32)
    else:
        return np.zeros(len(las.x), dtype=np.float32)
    if np.nanmax(np.abs(vals)) > 180.0:
        vals = vals * np.float32(0.006)
    return vals


def compute_pca_track_azimuth(xg, yg):
    coords = np.column_stack([xg, yg])
    mean_xy = coords.mean(axis=0)
    centered = coords - mean_xy
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eig(cov)
    main_axis = evecs[:, np.argmax(evals)]
    if main_axis[0] < 0:
        main_axis = -main_axis
    angle_deg = float(np.degrees(np.arctan2(main_axis[1], main_axis[0])))
    return angle_deg, main_axis, mean_xy


def to_ST(xv, yv, mean_x, mean_y, angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    S =  (xv - mean_x) * cos_a + (yv - mean_y) * sin_a
    T = -(xv - mean_x) * sin_a + (yv - mean_y) * cos_a
    return S, T


# =========================================================
# LOAD DATA
# =========================================================
print("=" * 60)
print("STEP 3 — PCA track direction + formal S-T coordinate system")
print("=" * 60)

las = laspy.read(LAZ_FILE)

# Memory-conservative loading (see step06/step07 for rationale).
_x64 = np.asarray(las.x)
x0 = float(_x64.min())
xl = (_x64 - x0).astype(np.float32); del _x64

_y64 = np.asarray(las.y)
y0 = float(_y64.min())
yl = (_y64 - y0).astype(np.float32); del _y64

z = np.asarray(las.z, dtype=np.float32)
scan_angle = scan_angle_degrees(las)

if not all(hasattr(las, ch) for ch in ["red", "green", "blue"]):
    raise RuntimeError("RGB fields are required.")

R_raw = np.asarray(las.red)
G_raw = np.asarray(las.green)
B_raw = np.asarray(las.blue)

if R_raw.max() > 256:
    print("RGB: 16-bit detected, dividing by 256")
    R = R_raw.astype(np.float32) / np.float32(256.0)
    G = G_raw.astype(np.float32) / np.float32(256.0)
    B = B_raw.astype(np.float32) / np.float32(256.0)
else:
    R = R_raw.astype(np.float32)
    G = G_raw.astype(np.float32)
    B = B_raw.astype(np.float32)
del R_raw, G_raw, B_raw

saturation, rgb_mean = hsv_saturation(R, G, B)

# =========================================================
# STEP 3.1 — GROUND FILTER
# =========================================================
print("\n[STEP 3.1] Ground-layer filtering (+ scan angle constraint)...")
z_floor = np.percentile(z, 1)
ground = (
    (z >= z_floor) & (z <= z_floor + GROUND_Z_RANGE) &
    (np.abs(scan_angle) <= SCAN_ANGLE_MAX)
)

xg = xl[ground]
yg = yl[ground]
zg = z[ground]

print(f"z_floor (P1):   {z_floor:.2f} m")
print(f"Ground points:  {fmt_n(ground.sum())} ({ground.mean()*100:.1f}%)")

# =========================================================
# STEP 3.2 — BEST CORRIDOR CANDIDATES FROM STEP 2
# =========================================================
print("\n[STEP 3.2] Applying best corridor parameters from Step 2...")
corridor = (
    ground &
    (saturation <= SAT_MAX) &
    (rgb_mean >= RGB_MEAN_MIN) &
    (rgb_mean <= RGB_MEAN_MAX)
)

xc = xl[corridor]
yc = yl[corridor]
zc = z[corridor]

print(f"Corridor candidates: {fmt_n(corridor.sum())} ({corridor.mean()*100:.1f}% of all points)")
print(f"Using thresholds: saturation<{SAT_MAX}, {RGB_MEAN_MIN}<rgb_mean<{RGB_MEAN_MAX}")

# =========================================================
# STEP 3.3 — PCA ON CORRIDOR POINTS (not all ground)
# =========================================================
# Using corridor points gives a more track-aligned direction
# because it excludes vegetation, ditches, and far terrain that
# would bias the PCA axis (Arastounia 2015: use geometric
# properties of the track itself for direction estimation).
print("\n[STEP 3.3] PCA track azimuth on corridor candidates...")
angle_deg, main_axis, mean_xy = compute_pca_track_azimuth(xc, yc)
angle_rad = np.radians(angle_deg)
mean_x, mean_y = float(mean_xy[0]), float(mean_xy[1])

# Also compute PCA on all ground points for comparison
angle_deg_ground, _, _ = compute_pca_track_azimuth(xg, yg)
angle_diff = abs(angle_deg - angle_deg_ground)

print(f"PCA azimuth (corridor): {angle_deg:.2f}°")
print(f"PCA azimuth (ground):   {angle_deg_ground:.2f}°  (for comparison)")
print(f"Difference:             {angle_diff:.2f}°")
print(f"Main axis:   ({main_axis[0]:.4f}, {main_axis[1]:.4f})")
print(f"Mean XY:     ({mean_x:.2f}, {mean_y:.2f})")

# =========================================================
# STEP 3.4 — BUILD FORMAL ST COORDINATES
# =========================================================
print("\n[STEP 3.4] Building formal S-T coordinates...")

Sg, Tg = to_ST(xg, yg, mean_x, mean_y, angle_rad)
Sc, Tc = to_ST(xc, yc, mean_x, mean_y, angle_rad)

print(f"Ground S range:    {Sg.min():.1f} – {Sg.max():.1f} m")
print(f"Ground T range:    {Tg.min():.1f} – {Tg.max():.1f} m")
print(f"Corridor S range:  {Sc.min():.1f} – {Sc.max():.1f} m")
print(f"Corridor T range:  {Tc.min():.1f} – {Tc.max():.1f} m")

# ── Curvature diagnostic ─────────────────────────────────
# Estimate track curvature by binning corridor S and tracking
# the median T.  If T varies a lot, the track is curved and
# the global PCA S-T frame is only an approximation.
print("\n[STEP 3.4b] Curvature diagnostic...")
s_bins_diag = np.arange(np.floor(Sc.min()), np.ceil(Sc.max()), 10.0)
t_medians = []
for sb in s_bins_diag:
    m = (Sc >= sb) & (Sc < sb + 10.0)
    if m.sum() >= 20:
        t_medians.append(float(np.median(Tc[m])))
    else:
        t_medians.append(np.nan)
t_medians = np.asarray(t_medians)
valid = ~np.isnan(t_medians)

if valid.sum() >= 3:
    t_range = float(np.nanmax(t_medians) - np.nanmin(t_medians))
    s_span = float(Sc.max() - Sc.min())
    curvature_ratio = t_range / s_span if s_span > 0 else 0.0

    print(f"  Corridor T median range: {t_range:.1f} m over {s_span:.0f} m of S")
    print(f"  Curvature ratio (T_range / S_span): {curvature_ratio:.3f}")

    if curvature_ratio > 0.10:
        print(f"  WARNING: High curvature detected (ratio > 0.10).")
        print(f"           Global PCA S-T is a coarse approximation.")
        print(f"           Step 05 curved centreline + Step 08/09 local tangent")
        print(f"           frames will compensate for this in downstream analysis.")
    elif curvature_ratio > 0.03:
        print(f"  NOTE: Moderate curvature. S-T system is acceptable;")
        print(f"        local frames in Step 08/09 will handle the correction.")
    else:
        print(f"  Track is approximately straight. Global PCA S-T is well-suited.")
else:
    t_range = np.nan
    curvature_ratio = np.nan
    print("  Insufficient corridor data for curvature estimate.")

# =========================================================
# FIGURE 1 — GROUND LAYER + PCA ARROW
# =========================================================
idx_g_plot = np.random.choice(len(xg), min(200_000, len(xg)), replace=False)

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(xg[idx_g_plot], yg[idx_g_plot], c=zg[idx_g_plot],
                cmap='terrain', s=0.3, alpha=0.5)
plt.colorbar(sc, ax=ax, label='Z (m)')

L = 120
ax.annotate(
    "", xy=(mean_x + main_axis[0] * L, mean_y + main_axis[1] * L),
    xytext=(mean_x - main_axis[0] * L, mean_y - main_axis[1] * L),
    arrowprops=dict(arrowstyle='->', color='red', lw=2.5)
)

ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_aspect('equal')

save_show(fig, os.path.join(OUT_DIR, "13_step3_ground_pca_direction.png"))

# =========================================================
# FIGURE 2 — FORMAL ST VIEW OF GROUND
# =========================================================
fig, ax = plt.subplots(figsize=(14, 6))
sc = ax.scatter(Sg[idx_g_plot], Tg[idx_g_plot], c=zg[idx_g_plot],
                cmap='terrain', s=0.3, alpha=0.5)
plt.colorbar(sc, ax=ax, label='Z (m)')
ax.set_xlabel("S (m)")
ax.set_ylabel("T (m)")

save_show(fig, os.path.join(OUT_DIR, "14_step3_ground_ST.png"))

# =========================================================
# FIGURE 3 — BEST CORRIDOR IN XY
# =========================================================
fig, ax = plt.subplots(figsize=(12, 6))

idx_bg = np.random.choice(len(xg), min(200_000, len(xg)), replace=False)
ax.scatter(xg[idx_bg], yg[idx_bg], c='lightgray', s=0.2, alpha=0.10, label='ground layer')

idx_c_plot = np.random.choice(len(xc), min(120_000, len(xc)), replace=False)
ax.scatter(xc[idx_c_plot], yc[idx_c_plot], c='navy', s=0.3, alpha=0.6, label='best corridor candidates')

ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_aspect('equal')
ax.legend(markerscale=8)

save_show(fig, os.path.join(OUT_DIR, "15_step3_best_corridor_XY.png"))

# =========================================================
# FIGURE 4 — BEST CORRIDOR IN ST
# =========================================================
fig, ax = plt.subplots(figsize=(14, 6))

idx_bg2 = np.random.choice(len(Sg), min(200_000, len(Sg)), replace=False)
ax.scatter(Sg[idx_bg2], Tg[idx_bg2], c='lightgray', s=0.2, alpha=0.08, label='ground layer')

idx_c_plot2 = np.random.choice(len(Sc), min(120_000, len(Sc)), replace=False)
ax.scatter(Sc[idx_c_plot2], Tc[idx_c_plot2], c='navy', s=0.3, alpha=0.5, label='best corridor candidates')

ax.set_xlabel("S (m)")
ax.set_ylabel("T (m)")
ax.legend(markerscale=8)

save_show(fig, os.path.join(OUT_DIR, "16_step3_best_corridor_ST.png"))

# =========================================================
# FIGURE 5 — DIAGNOSTIC CROSS-SECTIONS AT 25/50/75% OF S
# These are only for verifying that ST is sensible.
# =========================================================
print("\n[STEP 3.5] Diagnostic slices in formal S-T coordinates...")

s_positions = [
    float(np.percentile(Sg, 25)),
    float(np.percentile(Sg, 50)),
    float(np.percentile(Sg, 75)),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 14))

Zg = z[ground]
del z  # free full array

for ax, s0 in zip(axes, s_positions):
    slab = (Sg > s0 - 0.5) & (Sg < s0 + 0.5)
    n_pts = int(slab.sum())

    if n_pts < 10:
        ax.set_title(f"S={s0:.0f} m: insufficient points ({n_pts})")
        continue

    T_sl = Tg[slab]
    Z_sl = Zg[slab]

    order = np.argsort(T_sl)
    T_sl = T_sl[order]
    Z_sl = Z_sl[order]

    ax.scatter(T_sl, Z_sl, c=Z_sl, cmap='terrain', s=1.2, alpha=0.7)
    ax.set_xlabel("T (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"S = {s0:.0f} m", fontsize=10)
    ax.axvline(0, color='red', ls='--', lw=1.0, alpha=0.7)

save_show(fig, os.path.join(OUT_DIR, "17_step3_ST_diagnostic_slices.png"))

# =========================================================
# SAVE STEP 3 METADATA FOR NEXT STEPS
# =========================================================
meta = {
    "laz_file": LAZ_FILE,
    "ground_z_range": GROUND_Z_RANGE,
    "sat_max": SAT_MAX,
    "rgb_mean_min": RGB_MEAN_MIN,
    "rgb_mean_max": RGB_MEAN_MAX,
    "scan_angle_max": SCAN_ANGLE_MAX,
    "pca_track_azimuth_deg": angle_deg,
    "pca_source": "corridor",
    "pca_azimuth_ground_deg": angle_deg_ground,
    "pca_azimuth_diff_deg": angle_diff,
    "main_axis_x": float(main_axis[0]),
    "main_axis_y": float(main_axis[1]),
    "mean_x_local": mean_x,
    "mean_y_local": mean_y,
    "ground_s_min": float(Sg.min()),
    "ground_s_max": float(Sg.max()),
    "ground_t_min": float(Tg.min()),
    "ground_t_max": float(Tg.max()),
    "corridor_candidate_count": int(corridor.sum()),
    "corridor_candidate_ratio_percent": float(corridor.mean() * 100.0),
    "corridor_s_min": float(Sc.min()),
    "corridor_s_max": float(Sc.max()),
    "corridor_t_min": float(Tc.min()),
    "corridor_t_max": float(Tc.max()),
    "curvature_t_range_m": float(t_range) if not np.isnan(t_range) else None,
    "curvature_ratio": float(curvature_ratio) if not np.isnan(curvature_ratio) else None,
}

meta_path = os.path.join(OUT_DIR, "step03_ST_metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\nMetadata saved: {meta_path}")

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("STEP 3 SUMMARY — FORMAL ST COORDINATE SYSTEM")
print("=" * 60)
print(f"PCA source:          corridor points (not all ground)")
print(f"PCA azimuth:         {angle_deg:.2f}° (corridor)  vs  {angle_deg_ground:.2f}° (ground)")
print(f"Main axis:           ({main_axis[0]:.4f}, {main_axis[1]:.4f})")
print(f"Mean XY local:       ({mean_x:.2f}, {mean_y:.2f})")
print(f"Ground S range:      {Sg.min():.1f} – {Sg.max():.1f} m")
print(f"Ground T range:      {Tg.min():.1f} – {Tg.max():.1f} m")
print(f"Corridor candidates: {fmt_n(corridor.sum())} ({corridor.mean()*100:.1f}%)")
print(f"Corridor S range:    {Sc.min():.1f} – {Sc.max():.1f} m")
print(f"Corridor T range:    {Tc.min():.1f} – {Tc.max():.1f} m")
if not np.isnan(curvature_ratio):
    print(f"Curvature ratio:     {curvature_ratio:.3f} (T_range={t_range:.1f}m)")

print("\nInterpretation:")
print("  1. PCA is computed on corridor candidates (track-aligned, excludes far terrain).")
print("  2. Scan angle filter applied for consistency with Step 2.")
print("  3. Curvature diagnostic quantifies how much the track bends in this tile.")
print("  4. For curved tracks: Step 05 centreline + Step 08/09 local tangent frames")
print("     compensate for the global PCA approximation.")
print("  5. Next step: track the curved centreline along S using the corridor candidates.")

print("\nOutput figures:")
print("  13_step3_ground_pca_direction.png")
print("  14_step3_ground_ST.png")
print("  15_step3_best_corridor_XY.png")
print("  16_step3_best_corridor_ST.png")
print("  17_step3_ST_diagnostic_slices.png")
print("  step03_ST_metadata.json")

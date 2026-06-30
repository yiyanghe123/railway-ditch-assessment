import os
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

GROUND_Z_RANGE = 8.0
EXG_VEG_THRESHOLD = 20.0
SAT_MAX_DIAG = 0.30
RGB_MEAN_MIN_DIAG = 0.22
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# =========================================================
# HELPERS
# =========================================================
def fmt_n(x):
    return f"{int(x):,}"


def percentile_report(arr, name, plist=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    print(f"\n{name} percentile summary:")
    for p in plist:
        print(f"  P{p:>2}: {np.percentile(arr, p):.2f}")


def save_show(fig, out_path):
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved: {os.path.basename(out_path)}")


def hsv_saturation(R, G, B):
    """HSV saturation: 0 = achromatic (gray), 1 = fully saturated.

    Standard HSV definition: S = 1 - min(R,G,B) / max(R,G,B).
    Low saturation identifies achromatic surfaces such as rail ballast and
    steel rails where R ≈ G ≈ B (Šašak et al., 2023; Velesaca et al., 2020).
    """
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


# =========================================================
# STEP 1 - READ BASIC FILE INFO
# =========================================================
print("=" * 60)
print("STEP 1 - Data diagnostics for geometry, vegetation, and corridor cues")
print("=" * 60)

las = laspy.read(LAZ_FILE)

# Memory-conservative loading: 82M-point tiles give ~631 MB per float64
# array. Cast each coordinate to float32 immediately after the min/max
# scalars are captured so that at most one full float64 array is alive.
_x64 = np.asarray(las.x)
x_min_raw = float(_x64.min()); x_max_raw = float(_x64.max())
xl = (_x64 - x_min_raw).astype(np.float32); del _x64

_y64 = np.asarray(las.y)
y_min_raw = float(_y64.min()); y_max_raw = float(_y64.max())
yl = (_y64 - y_min_raw).astype(np.float32); del _y64

z = np.asarray(las.z, dtype=np.float32)
intensity = np.asarray(las.intensity)

n_pts = len(xl)
x_span = x_max_raw - x_min_raw
y_span = y_max_raw - y_min_raw
print(f"Total points:     {fmt_n(n_pts)}")
print(f"X range:          {x_min_raw:.2f} - {x_max_raw:.2f}  (span {x_span:.1f} m)")
print(f"Y range:          {y_min_raw:.2f} - {y_max_raw:.2f}  (span {y_span:.1f} m)")
print(f"Z range:          {z.min():.2f} - {z.max():.2f}  (span {float(z.max() - z.min()):.1f} m)")
print(f"Available fields: {list(las.point_format.dimension_names)}")

# xl, yl are already float32 offset-subtracted (no extra cast needed)
x0, y0 = x_min_raw, y_min_raw


# =========================================================
# STEP 2 - RGB / ExG / Grayness
# =========================================================
print("\n" + "=" * 60)
print("STEP 2 - Checking RGB / ExG / grayness")
print("=" * 60)

has_rgb = all(hasattr(las, ch) for ch in ["red", "green", "blue"])
if not has_rgb:
    raise RuntimeError("RGB fields are required because later steps use ExG and HSV saturation.")

R_raw = np.asarray(las.red)
G_raw = np.asarray(las.green)
B_raw = np.asarray(las.blue)

print("RGB fields are available.")
print(f"Red   range: {R_raw.min()} - {R_raw.max()}")
print(f"Green range: {G_raw.min()} - {G_raw.max()}")
print(f"Blue  range: {B_raw.min()} - {B_raw.max()}")

if R_raw.max() > 256:
    print("Detected 16-bit RGB (0-65535); dividing by 256 before ExG and saturation.")
    R = R_raw.astype(np.float32) / 256.0
    G = G_raw.astype(np.float32) / 256.0
    B = B_raw.astype(np.float32) / 256.0
    rgb_bit_depth = "16-bit"
else:
    print("Detected 8-bit RGB (0-255); using values directly.")
    R = R_raw.astype(np.float32)
    G = G_raw.astype(np.float32)
    B = B_raw.astype(np.float32)
    rgb_bit_depth = "8-bit"
del R_raw, G_raw, B_raw

ExG = (2.0 * G - R - B).astype(np.float32)
saturation, rgb_mean = hsv_saturation(R, G, B)

print(f"ExG range:        {ExG.min():.1f} - {ExG.max():.1f}")
print(f"Saturation range: {saturation.min():.3f} - {saturation.max():.3f}")
print(f"RGB mean range:   {rgb_mean.min():.3f} - {rgb_mean.max():.3f}")

percentile_report(ExG, "ExG")
percentile_report(saturation, "HSV Saturation")
percentile_report(rgb_mean, "RGB mean")

veg_mask = ExG > EXG_VEG_THRESHOLD
print(f"\nPoints with ExG > {EXG_VEG_THRESHOLD:.0f} (vegetation candidates): "
      f"{fmt_n(veg_mask.sum())}  ({veg_mask.mean() * 100:.1f}%)")


# =========================================================
# STEP 3 - GROUND LAYER DIAGNOSTICS
# =========================================================
print("\n" + "=" * 60)
print("STEP 3 - Ground-layer diagnostics")
print("=" * 60)

z_floor = np.percentile(z, 1)
ground = (z >= z_floor) & (z <= z_floor + GROUND_Z_RANGE)

print(f"z_floor (P1):   {z_floor:.2f} m")
print(f"GROUND_Z_RANGE: {GROUND_Z_RANGE:.1f} m")
print(f"Ground-layer points: {fmt_n(ground.sum())}  ({ground.mean() * 100:.1f}%)")

ExGg = ExG[ground]
satg = saturation[ground]
rgbmg = rgb_mean[ground]
zg = z[ground]

print(f"\nGround-layer ExG > {EXG_VEG_THRESHOLD:.0f}: "
      f"{fmt_n((ExGg > EXG_VEG_THRESHOLD).sum())}  "
      f"({(ExGg > EXG_VEG_THRESHOLD).mean() * 100:.1f}%)")

gray_corridor_diag = ground & (saturation <= SAT_MAX_DIAG) & (rgb_mean > RGB_MEAN_MIN_DIAG)
print(f"Ground-layer gray-corridor candidates (saturation<={SAT_MAX_DIAG}, rgb_mean>{RGB_MEAN_MIN_DIAG}): "
      f"{fmt_n(gray_corridor_diag.sum())}  ({gray_corridor_diag.mean() * 100:.1f}%)")


# =========================================================
# STEP 4 - INTENSITY: KEEP ONLY AS DIAGNOSTIC
# =========================================================
print("\n" + "=" * 60)
print("STEP 4 - Intensity diagnostics only; not used for primary identification")
print("=" * 60)

print(f"Intensity range: {intensity.min()} - {intensity.max()}")
percentile_report(intensity, "Intensity")

thr_p1 = np.percentile(intensity, 1)
thr_p5 = np.percentile(intensity, 5)
thr_p10 = np.percentile(intensity, 10)

print("\nLow-intensity diagnostic thresholds:")
print(f"  P1  = {thr_p1:.1f}")
print(f"  P5  = {thr_p5:.1f}")
print(f"  P10 = {thr_p10:.1f}")

low_p5 = intensity <= thr_p5
low_p5_ground = ground & (intensity <= thr_p5)

print(f"\nLowest 5% intensity points: {fmt_n(low_p5.sum())} ({low_p5.mean() * 100:.1f}%)")
print(f"Lowest 5% intensity points within ground layer: {fmt_n(low_p5_ground.sum())} "
      f"({low_p5_ground.sum() / max(1, ground.sum()) * 100:.1f}% of ground)")


# =========================================================
# FIGURE 1 - INTENSITY HISTOGRAMS (DIAGNOSTIC ONLY)
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(intensity, bins=500, color="steelblue", edgecolor="none", log=True)
axes[0].set_xlabel("Intensity")
axes[0].set_ylabel("Count (log)")
axes[0].set_title("Intensity histogram — full range")

mask_low = intensity <= thr_p10
axes[1].hist(intensity[mask_low], bins=300, color="steelblue", edgecolor="none", log=True)
axes[1].axvline(thr_p1, color="red", linestyle="--", linewidth=1.5, label=f"P1 = {thr_p1:.1f}")
axes[1].axvline(thr_p5, color="orange", linestyle="--", linewidth=1.5, label=f"P5 = {thr_p5:.1f}")
axes[1].axvline(thr_p10, color="green", linestyle="--", linewidth=1.5, label=f"P10 = {thr_p10:.1f}")
axes[1].set_xlabel("Intensity")
axes[1].set_ylabel("Count (log)")
axes[1].set_title("Intensity histogram — low-end zoom (≤ P10)")
axes[1].legend()

save_show(fig, os.path.join(OUT_DIR, "01_intensity_histograms.png"))


# =========================================================
# FIGURE 2 - TOP VIEW (ALL POINTS)
# =========================================================
print("\n" + "=" * 60)
print("STEP 5 - Plan-view diagnostics")
print("=" * 60)

N_all = min(200_000, n_pts)
idx_all = np.random.choice(n_pts, N_all, replace=False)

fig, ax = plt.subplots(figsize=(14, 5))
sc = ax.scatter(xl[idx_all], yl[idx_all], c=z[idx_all], cmap="terrain", s=0.3, alpha=0.6)
plt.colorbar(sc, ax=ax, label="Z (m)")
ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_aspect("equal")
save_show(fig, os.path.join(OUT_DIR, "02_topview_all.png"))


# =========================================================
# FIGURE 3 - TOP VIEW (GROUND ONLY)
# =========================================================
N_g = min(200_000, int(ground.sum()))
idx_ground_all = np.where(ground)[0]
idx_g = np.random.choice(idx_ground_all, N_g, replace=False)

fig, ax = plt.subplots(figsize=(14, 5))
sc = ax.scatter(xl[idx_g], yl[idx_g], c=z[idx_g], cmap="terrain", s=0.3, alpha=0.5)
plt.colorbar(sc, ax=ax, label="Z (m)")
ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_aspect("equal")
save_show(fig, os.path.join(OUT_DIR, "03_topview_ground_only.png"))


# =========================================================
# FIGURE 4 - GROUND LAYER COLOURED BY ExG
# =========================================================
fig, ax = plt.subplots(figsize=(14, 5))
sc = ax.scatter(
    xl[idx_g], yl[idx_g],
    c=ExG[idx_g], cmap="RdYlGn", s=0.3, alpha=0.6,
    vmin=-50, vmax=50
)
plt.colorbar(sc, ax=ax, label="ExG")
ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_aspect("equal")
save_show(fig, os.path.join(OUT_DIR, "04_ground_topview_exg.png"))


# =========================================================
# FIGURE 5 - GROUND LAYER COLOURED BY GRAYNESS
# =========================================================
fig, ax = plt.subplots(figsize=(14, 5))
sc = ax.scatter(
    xl[idx_g], yl[idx_g],
    c=saturation[idx_g], cmap="viridis_r", s=0.3, alpha=0.6,
    vmin=0.0, vmax=0.6
)
plt.colorbar(sc, ax=ax, label="HSV Saturation")
ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_aspect("equal")
save_show(fig, os.path.join(OUT_DIR, "05_ground_topview_saturation.png"))


# =========================================================
# FIGURE 6 - DIAGNOSTIC GRAY-CORRIDOR CANDIDATES
# =========================================================
fig, ax = plt.subplots(figsize=(14, 5))

# background = ground layer
bg_idx = np.random.choice(idx_ground_all, min(200_000, len(idx_ground_all)), replace=False)
ax.scatter(xl[bg_idx], yl[bg_idx], c="lightgray", s=0.2, alpha=0.15, label="ground layer")

# foreground = gray corridor candidates
idx_corr = np.where(gray_corridor_diag)[0]
if len(idx_corr) > 0:
    idx_corr_plot = np.random.choice(idx_corr, min(120_000, len(idx_corr)), replace=False)
    ax.scatter(xl[idx_corr_plot], yl[idx_corr_plot], c="navy", s=0.3, alpha=0.6,
               label="gray-corridor candidates")

ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.legend(markerscale=8)
ax.set_aspect("equal")
save_show(fig, os.path.join(OUT_DIR, "06_gray_corridor_candidates.png"))


# =========================================================
# FIGURE 7 - LOW-INTENSITY SPATIAL DISTRIBUTION
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

# all points background
ax = axes[0]
ax.scatter(xl[idx_all], yl[idx_all], c="lightgray", s=0.2, alpha=0.15, label="all points")
idx_low = np.where(low_p5)[0]
if len(idx_low) > 0:
    idx_low_plot = np.random.choice(idx_low, min(80_000, len(idx_low)), replace=False)
    ax.scatter(xl[idx_low_plot], yl[idx_low_plot], c="red", s=0.3, alpha=0.6,
               label="lowest 5% intensity")
ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_title("Spatial distribution of lowest 5% intensity")
ax.legend(markerscale=8)
ax.set_aspect("equal")

# ground background
ax = axes[1]
ax.scatter(xl[bg_idx], yl[bg_idx], c="lightgray", s=0.2, alpha=0.15, label="ground layer")
idx_low_g = np.where(low_p5_ground)[0]
if len(idx_low_g) > 0:
    idx_low_g_plot = np.random.choice(idx_low_g, min(80_000, len(idx_low_g)), replace=False)
    ax.scatter(xl[idx_low_g_plot], yl[idx_low_g_plot], c="blue", s=0.3, alpha=0.6,
               label="ground ∩ lowest 5% intensity")
ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_title("Low-intensity points within ground layer")
ax.legend(markerscale=8)
ax.set_aspect("equal")

save_show(fig, os.path.join(OUT_DIR, "07_low_intensity_spatial_distribution.png"))


# =========================================================
# FIGURE 8 - DIAGNOSTIC SLICE (ROUGH ONLY)
# NOTE:
# This is not a true rail-normal section yet.
# It is only a rough visual slice through tile midpoint.
# =========================================================
print("\n" + "=" * 60)
print("STEP 6 - Diagnostic slice for a quick terrain check only")
print("=" * 60)

y_mid = yl.mean()
# Compute mask without allocating a full float64 temp array
slice_mask = ground.copy()
slice_mask &= (yl > y_mid - 0.1)
slice_mask &= (yl < y_mid + 0.1)
print(f"Diagnostic slice points: {fmt_n(slice_mask.sum())}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# by Z
ax = axes[0]
ax.scatter(
    xl[slice_mask], z[slice_mask], c=z[slice_mask],
    cmap="terrain", s=1, alpha=0.5
)
ax.set_xlabel("X local (m)")
ax.set_ylabel("Z (m)")
ax.set_title("Diagnostic slice — coloured by Z")

# by ExG
ax = axes[1]
sc = ax.scatter(
    xl[slice_mask], z[slice_mask], c=ExG[slice_mask],
    cmap="RdYlGn", s=1, alpha=0.5, vmin=-50, vmax=50
)
plt.colorbar(sc, ax=ax, label="ExG")
ax.set_xlabel("X local (m)")
ax.set_ylabel("Z (m)")
ax.set_title("Diagnostic slice — coloured by ExG")

save_show(fig, os.path.join(OUT_DIR, "08_diagnostic_slice_ground.png"))


# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("STEP 1 SUMMARY - DATA DIAGNOSTICS FOR NEW WORKFLOW")
print("=" * 60)
print(f"File:                 {os.path.basename(LAZ_FILE)}")
print(f"Total points:         {fmt_n(n_pts)}")
print(f"Extent:               {x_span:.1f} m x {y_span:.1f} m")
print(f"Z range:              {z.min():.2f} - {z.max():.2f} m")
print(f"RGB available:        yes")
print(f"RGB bit depth:        {rgb_bit_depth}")
print(f"ExG range:            {ExG.min():.1f} - {ExG.max():.1f}")
print(f"Saturation range:     {saturation.min():.3f} - {saturation.max():.3f}")
print(f"Vegetation-candidate ratio: {(veg_mask.mean() * 100):.1f}%  (ExG > {EXG_VEG_THRESHOLD:.0f})")
print(f"Ground-layer ratio:   {ground.mean() * 100:.1f}%")
print(f"Gray-corridor candidate ratio: {gray_corridor_diag.mean() * 100:.1f}%  "
      f"(saturation<={SAT_MAX_DIAG}, rgb_mean>{RGB_MEAN_MIN_DIAG})")
print(f"Intensity range:      {intensity.min()} - {intensity.max()}")
print(f"Intensity P1/P5/P50/P95: "
      f"{np.percentile(intensity, 1):.1f} / "
      f"{np.percentile(intensity, 5):.1f} / "
      f"{np.percentile(intensity, 50):.1f} / "
      f"{np.percentile(intensity, 95):.1f}")

print("\nMethod notes:")
print("  1. Step 1 no longer attempts to identify the track directly from intensity.")
print("  2. Intensity is retained only as an auxiliary diagnostic field.")
print("  3. The railway-detection route should use:")
print("       ground layer -> PCA main direction -> S-T coordinates -> gray corridor -> centreline")
print("  4. ExG is retained for later ditch vegetation-cover analysis (DVCI), not for railway positioning.")
print("  5. The next step should verify whether the gray corridor stably follows the railway corridor.")

print(f"\nOutput directory: {OUT_DIR}")

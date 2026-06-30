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
META_FILE = os.path.join(OUT_DIR, "step03_ST_metadata.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ground filter (from Step 1)
GROUND_Z_RANGE = 8.0

# scan angle filter — ONLY for corridor, NOT for ground
# Ground keeps all scan angles to preserve ditch/slope terrain data.
# Corridor uses strict scan angle for quality ridge tracking.
SCAN_ANGLE_MAX_CORRIDOR = 20

# diagnostic settings
SECTION_HALF_THICKNESS = 0.5
CURVATURE_BIN_SIZE = 10.0
DENSITY_BIN_SIZE = 10.0

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


def to_ST(xv, yv, mean_x, mean_y, angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    S =  (xv - mean_x) * cos_a + (yv - mean_y) * sin_a
    T = -(xv - mean_x) * sin_a + (yv - mean_y) * cos_a
    return S, T


# =========================================================
# LOAD STEP 3 METADATA
# =========================================================
print("=" * 60)
print("STEP 4 — Standardised S-T projection and data preparation")
print("=" * 60)

if not os.path.exists(META_FILE):
    raise FileNotFoundError(
        f"Step 3 metadata was not found: {META_FILE}\n"
        f"Run step03 before this script."
    )

with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

angle_deg = float(meta["pca_track_azimuth_deg"])
angle_rad = np.radians(angle_deg)
mean_x = float(meta["mean_x_local"])
mean_y = float(meta["mean_y_local"])

# corridor parameters from step03 metadata (calibrated in step02)
SAT_MAX = float(meta["sat_max"])
RGB_MEAN_MIN = float(meta["rgb_mean_min"])
RGB_MEAN_MAX = float(meta["rgb_mean_max"])

# curvature diagnostics
curvature_ratio = float(meta.get("curvature_ratio", 0.0))
curvature_t_range = float(meta.get("curvature_t_range_m", 0.0))
pca_source = meta.get("pca_source", "unknown")

print(f"Loaded metadata from: {META_FILE}")
print(f"PCA azimuth:  {angle_deg:.2f}° (source: {pca_source})")
print(f"Mean XY:      ({mean_x:.2f}, {mean_y:.2f})")
print(f"Corridor:     sat≤{SAT_MAX}, {RGB_MEAN_MIN}≤rgb_mean≤{RGB_MEAN_MAX}")
print(f"Scan angle:   corridor only ≤{SCAN_ANGLE_MAX_CORRIDOR}° (ground: no filter)")
print(f"Curvature:    ratio={curvature_ratio:.4f}, T_range={curvature_t_range:.1f} m")

if curvature_ratio > 0.10:
    print("  ⚠ HIGH CURVATURE — single PCA direction is a rough approximation.")
    print("    Step 05 curved centreline fitting will be critical.")
elif curvature_ratio > 0.05:
    print("  ⚡ MODERATE CURVATURE — some cross-track drift expected in S-T view.")

# =========================================================
# LOAD DATA
# =========================================================
print("\n[STEP 4.1] Loading LAZ and building masks...")

las = laspy.read(LAZ_FILE)

# Memory-conservative loading (see step06/step07 for rationale).
_x64 = np.asarray(las.x)
x0 = float(_x64.min())
xl = (_x64 - x0).astype(np.float32); del _x64

_y64 = np.asarray(las.y)
y0 = float(_y64.min())
yl = (_y64 - y0).astype(np.float32); del _y64

z = np.asarray(las.z, dtype=np.float32)

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

# scan angle
scan_angle = np.abs(scan_angle_degrees(las))

saturation, rgb_mean = hsv_saturation(R, G, B)
ExG = np.float32(2.0) * G - R - B

# =========================================================
# STEP 4.1 — GROUND + CORRIDOR MASKS
# =========================================================
z_floor = np.percentile(z, 1)

# Ground: NO scan_angle filter — preserves ditch/slope terrain for step07
ground = (z >= z_floor) & (z <= z_floor + GROUND_Z_RANGE)

# Corridor: ground + RGB corridor + scan_angle filter for quality
corridor = (
    ground &
    (saturation <= SAT_MAX) &
    (rgb_mean >= RGB_MEAN_MIN) &
    (rgb_mean <= RGB_MEAN_MAX) &
    (scan_angle <= SCAN_ANGLE_MAX_CORRIDOR)
)

print(f"Total points:     {fmt_n(len(xl))}")
print(f"Ground points:    {fmt_n(ground.sum())} ({ground.mean()*100:.2f}%)  [no scan_angle filter]")
print(f"Corridor points:  {fmt_n(corridor.sum())} ({corridor.mean()*100:.2f}%)  [scan_angle ≤{SCAN_ANGLE_MAX_CORRIDOR}°]")

# =========================================================
# STEP 4.2 — PROJECT TO FORMAL S-T
# =========================================================
print("\n[STEP 4.2] Projecting to formal S-T coordinates...")

# ground arrays
Sg, Tg = to_ST(xl[ground], yl[ground], mean_x, mean_y, angle_rad)
Zg = z[ground]
ExGg = ExG[ground]

# corridor arrays
Sc, Tc = to_ST(xl[corridor], yl[corridor], mean_x, mean_y, angle_rad)
Zc = z[corridor]
ExGc = ExG[corridor]

print(f"Ground S range:   [{Sg.min():.1f}, {Sg.max():.1f}] m  (span {Sg.max()-Sg.min():.1f} m)")
print(f"Ground T range:   [{Tg.min():.1f}, {Tg.max():.1f}] m")
print(f"Corridor S range: [{Sc.min():.1f}, {Sc.max():.1f}] m")
print(f"Corridor T range: [{Tc.min():.1f}, {Tc.max():.1f}] m")

# =========================================================
# STEP 4.3 — CORRIDOR T-MEDIAN TRACKING (curvature preview)
# =========================================================
print("\n[STEP 4.3] Tracking corridor T-median along S (curvature preview)...")

s_bins = np.arange(np.floor(Sc.min()), np.ceil(Sc.max()), CURVATURE_BIN_SIZE)
t_medians = []
t_counts = []
t_widths = []   # IQR of T per bin (corridor width diagnostic)
s_centers = []

for sb in s_bins:
    m = (Sc >= sb) & (Sc < sb + CURVATURE_BIN_SIZE)
    cnt = int(m.sum())
    sc_val = float(sb + CURVATURE_BIN_SIZE / 2)
    s_centers.append(sc_val)
    t_counts.append(cnt)
    if cnt >= 10:
        t_medians.append(float(np.median(Tc[m])))
        t_q25, t_q75 = np.percentile(Tc[m], [25, 75])
        t_widths.append(float(t_q75 - t_q25))
    else:
        t_medians.append(np.nan)
        t_widths.append(np.nan)

s_centers = np.array(s_centers)
t_medians = np.array(t_medians)
t_counts = np.array(t_counts)
t_widths = np.array(t_widths)

valid = ~np.isnan(t_medians)
print(f"Corridor bins:    {valid.sum()} valid / {len(s_centers)} total")
if valid.sum() >= 2:
    t_range = float(np.nanmax(t_medians) - np.nanmin(t_medians))
    print(f"T-median range:   {t_range:.1f} m (corridor center drift)")
    print(f"Corridor width:   median IQR = {np.nanmedian(t_widths):.2f} m, "
          f"max IQR = {np.nanmax(t_widths):.2f} m")

# =========================================================
# STEP 4.4 — GROUND POINT DENSITY ALONG S
# =========================================================
print("\n[STEP 4.4] Computing ground point density along S...")

d_bins = np.arange(np.floor(Sg.min()), np.ceil(Sg.max()), DENSITY_BIN_SIZE)
d_centers = d_bins + DENSITY_BIN_SIZE / 2
d_counts = np.array([int(((Sg >= sb) & (Sg < sb + DENSITY_BIN_SIZE)).sum()) for sb in d_bins])

# approximate area per bin: bin_length * T_range
t_span = Tg.max() - Tg.min()
d_density = d_counts / (DENSITY_BIN_SIZE * t_span) if t_span > 0 else d_counts

sparse_threshold = np.percentile(d_counts[d_counts > 0], 10) if (d_counts > 0).sum() > 5 else 10
n_sparse = int((d_counts < sparse_threshold).sum())
print(f"Ground bins:      {len(d_bins)} ({DENSITY_BIN_SIZE}m each)")
print(f"Points/bin:       median={np.median(d_counts):.0f}, min={d_counts.min()}, max={d_counts.max()}")
print(f"Sparse bins (<{sparse_threshold:.0f} pts): {n_sparse}")

# =========================================================
# STEP 4.5 — SAVE NPZ FOR STEP 05
# =========================================================
print("\n[STEP 4.5] Saving projected data for downstream steps...")

npz_path = os.path.join(OUT_DIR, "step04_ST_projected.npz")
np.savez_compressed(
    npz_path,
    # ground arrays (NO scan_angle filter — full terrain)
    Sg=Sg, Tg=Tg, Zg=Zg, ExGg=ExGg,
    # corridor arrays (WITH scan_angle filter — quality data)
    Sc=Sc, Tc=Tc, Zc=Zc, ExGc=ExGc,
    # corridor T-median tracking (for step05 initial guess)
    curvature_s=s_centers[valid],
    curvature_t_median=t_medians[valid],
    curvature_t_counts=t_counts[valid],
    # ground local XY (for back-projection in step05)
    xl_ground=xl[ground],
    yl_ground=yl[ground],
)

npz_size_mb = os.path.getsize(npz_path) / 1e6
print(f"Saved: {npz_path} ({npz_size_mb:.1f} MB)")
print(f"  Ground arrays:   {len(Sg):,} points  (no scan_angle filter)")
print(f"  Corridor arrays: {len(Sc):,} points  (scan_angle ≤{SCAN_ANGLE_MAX_CORRIDOR}°)")

# =========================================================
# FIGURE 1 — FORMAL ST OVERVIEW WITH T-MEDIAN
# =========================================================
print("\n[STEP 4.6] Generating diagnostic figures...")

n_g_plot = min(200_000, len(Sg))
n_c_plot = min(120_000, len(Sc))
idx_g_plot = np.random.choice(len(Sg), n_g_plot, replace=False) if len(Sg) > 0 else np.array([], dtype=int)
idx_c_plot = np.random.choice(len(Sc), n_c_plot, replace=False) if len(Sc) > 0 else np.array([], dtype=int)

fig, ax = plt.subplots(figsize=(14, 6))
if len(idx_g_plot) > 0:
    ax.scatter(Sg[idx_g_plot], Tg[idx_g_plot], c="lightgray", s=0.2, alpha=0.08, label="ground layer")
if len(idx_c_plot) > 0:
    ax.scatter(Sc[idx_c_plot], Tc[idx_c_plot], c="navy", s=0.3, alpha=0.5, label="corridor candidates")
if valid.sum() >= 2:
    ax.plot(s_centers[valid], t_medians[valid], "o-", color="red", ms=3, lw=1.5,
            alpha=0.8, label=f"corridor T-median (drift={t_range:.1f}m)")
ax.set_xlabel("S (m)")
ax.set_ylabel("T (m)")
ax.legend(markerscale=8)
ax.grid(alpha=0.3)

save_show(fig, os.path.join(OUT_DIR, "18_step4_formal_ST_overview.png"))

# =========================================================
# FIGURE 2 — CORRIDOR CURVATURE + WIDTH + DENSITY
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

# 2a: corridor T-median drift
ax = axes[0]
if valid.sum() >= 2:
    ax.plot(s_centers[valid], t_medians[valid], "o-", color="red", ms=4, lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5, label="T=0 (PCA origin)")
    ax.fill_between(s_centers[valid], t_medians[valid], 0, alpha=0.15, color="red")
ax.set_ylabel("Corridor T-median (m)")
ax.set_title("Corridor centre drift along S", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 2b: corridor T-width (IQR)
ax = axes[1]
valid_w = ~np.isnan(t_widths)
if valid_w.sum() >= 2:
    ax.bar(s_centers[valid_w], t_widths[valid_w], width=CURVATURE_BIN_SIZE * 0.8,
           alpha=0.7, color="orange")
    median_w = np.nanmedian(t_widths)
    ax.axhline(median_w, color="red", ls="--", lw=1.0,
               label=f"median IQR = {median_w:.2f} m")
    # flag abnormally wide bins (>2x median)
    wide = valid_w & (t_widths > 2.0 * median_w)
    if wide.sum() > 0:
        ax.bar(s_centers[wide], t_widths[wide], width=CURVATURE_BIN_SIZE * 0.8,
               alpha=0.7, color="red", label=f"wide bins (>{2*median_w:.1f}m): {wide.sum()}")
ax.set_ylabel("Corridor T-width IQR (m)")
ax.set_title("Corridor width (IQR) along S", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 2c: corridor point density
ax = axes[2]
ax.bar(s_centers, t_counts, width=CURVATURE_BIN_SIZE * 0.8, alpha=0.7, color="steelblue")
ax.set_xlabel("S (m)")
ax.set_ylabel("Corridor points / bin")
ax.set_title("Corridor point density along S")
ax.grid(alpha=0.3)

save_show(fig, os.path.join(OUT_DIR, "18b_step4_corridor_diagnostics.png"))

# =========================================================
# FIGURE 3 — GROUND POINT DENSITY ALONG S
# =========================================================
fig, ax = plt.subplots(figsize=(14, 4))
colors = np.where(d_counts < sparse_threshold, "red", "steelblue")
ax.bar(d_centers, d_counts, width=DENSITY_BIN_SIZE * 0.8, alpha=0.7, color=colors)
ax.axhline(sparse_threshold, color="red", ls="--", lw=1.0, alpha=0.7,
           label=f"sparse threshold = {sparse_threshold:.0f} pts")
ax.set_xlabel("S (m)")
ax.set_ylabel("Ground points / bin")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

save_show(fig, os.path.join(OUT_DIR, "18c_step4_ground_density.png"))

# =========================================================
# STEP 4.7 — SELECT THREE STANDARD S POSITIONS
# =========================================================
print("\n[STEP 4.7] Selecting standard S positions for diagnostic sections...")

s_positions = {
    "25%": float(np.percentile(Sg, 25)),
    "50%": float(np.percentile(Sg, 50)),
    "75%": float(np.percentile(Sg, 75)),
}

for k, v in s_positions.items():
    if valid.sum() >= 2:
        t_center = float(np.interp(v, s_centers[valid], t_medians[valid]))
    else:
        t_center = 0.0
    print(f"  {k}: S = {v:.1f} m, corridor T-median ≈ {t_center:.1f} m")

# =========================================================
# FIGURE 4 — THREE Z-COLOURED STANDARD SECTIONS
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

for ax, (label, s0) in zip(axes, s_positions.items()):
    slab_g = (Sg > s0 - SECTION_HALF_THICKNESS) & (Sg < s0 + SECTION_HALF_THICKNESS)
    slab_c = (Sc > s0 - SECTION_HALF_THICKNESS) & (Sc < s0 + SECTION_HALF_THICKNESS)

    n_g = int(slab_g.sum())
    n_c = int(slab_c.sum())

    if n_g < 10:
        ax.set_title(f"{label}: insufficient points ({n_g})")
        continue

    order_g = np.argsort(Tg[slab_g])
    T_sl = Tg[slab_g][order_g]
    Z_sl = Zg[slab_g][order_g]

    ax.scatter(T_sl, Z_sl, c=Z_sl, cmap="terrain", s=1.0, alpha=0.35, label="ground slice")

    if n_c > 0:
        order_c = np.argsort(Tc[slab_c])
        T_sc = Tc[slab_c][order_c]
        Z_sc = Zc[slab_c][order_c]
        ax.scatter(T_sc, Z_sc, c="navy", s=1.2, alpha=0.7, label="corridor slice")

    # mark corridor T-median at this S position
    if valid.sum() >= 2:
        t_center = float(np.interp(s0, s_centers[valid], t_medians[valid]))
        ax.axvline(t_center, color="red", ls="-", lw=1.5, alpha=0.7,
                   label=f"corridor centre T={t_center:.1f}")

    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.4, label="T=0 (PCA origin)")
    ax.set_xlabel("T (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"S = {s0:.0f} m ({label})", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

save_show(fig, os.path.join(OUT_DIR, "19_step4_ST_sections_Z.png"))

# =========================================================
# FIGURE 5 — THREE ExG-COLOURED STANDARD SECTIONS
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

for ax, (label, s0) in zip(axes, s_positions.items()):
    slab_g = (Sg > s0 - SECTION_HALF_THICKNESS) & (Sg < s0 + SECTION_HALF_THICKNESS)
    slab_c = (Sc > s0 - SECTION_HALF_THICKNESS) & (Sc < s0 + SECTION_HALF_THICKNESS)

    n_g = int(slab_g.sum())
    n_c = int(slab_c.sum())

    if n_g < 10:
        ax.set_title(f"{label}: insufficient points ({n_g})")
        continue

    order_g = np.argsort(Tg[slab_g])
    T_sl = Tg[slab_g][order_g]
    Z_sl = Zg[slab_g][order_g]
    E_sl = ExGg[slab_g][order_g]

    sc = ax.scatter(T_sl, Z_sl, c=E_sl, cmap="RdYlGn", s=1.0, alpha=0.45, vmin=-50, vmax=50)
    plt.colorbar(sc, ax=ax, label="ExG")

    if n_c > 0:
        order_c = np.argsort(Tc[slab_c])
        T_sc = Tc[slab_c][order_c]
        Z_sc = Zc[slab_c][order_c]
        ax.scatter(T_sc, Z_sc, c="black", s=1.0, alpha=0.6, label="corridor slice")

    if valid.sum() >= 2:
        t_center = float(np.interp(s0, s_centers[valid], t_medians[valid]))
        ax.axvline(t_center, color="red", ls="-", lw=1.5, alpha=0.7,
                   label=f"corridor centre T={t_center:.1f}")

    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.4, label="T=0 (PCA origin)")
    ax.set_xlabel("T (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"S = {s0:.0f} m ({label})  [ExG]", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

save_show(fig, os.path.join(OUT_DIR, "20_step4_ST_sections_ExG.png"))

# =========================================================
# FIGURE 6 — XY LOCATIONS OF THE THREE SECTIONS
# =========================================================
fig, ax = plt.subplots(figsize=(12, 6))

idx_bg_xy = np.random.choice(np.where(ground)[0], min(200_000, ground.sum()), replace=False)
ax.scatter(xl[idx_bg_xy], yl[idx_bg_xy], c="lightgray", s=0.2, alpha=0.08, label="ground layer")

idx_c_xy = np.random.choice(np.where(corridor)[0], min(120_000, corridor.sum()), replace=False)
ax.scatter(xl[idx_c_xy], yl[idx_c_xy], c="navy", s=0.3, alpha=0.4, label="corridor")

cos_a = np.cos(angle_rad)
sin_a = np.sin(angle_rad)

colors = ["red", "orange", "green"]
for (label, s0), col in zip(s_positions.items(), colors):
    T_line = np.linspace(-120, 120, 200)
    S_line = np.full_like(T_line, s0)

    X_line = mean_x + S_line * cos_a - T_line * sin_a
    Y_line = mean_y + S_line * sin_a + T_line * cos_a

    ax.plot(X_line, Y_line, color=col, lw=2.0, label=f"{label} section (S={s0:.0f})")

ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_aspect("equal")
ax.legend(markerscale=8)

save_show(fig, os.path.join(OUT_DIR, "21_step4_section_locations_XY.png"))

# =========================================================
# SAVE STEP 4 METADATA
# =========================================================
step4_meta = {
    "source_meta_file": META_FILE,
    "pca_track_azimuth_deg": angle_deg,
    "mean_x_local": mean_x,
    "mean_y_local": mean_y,
    "sat_max": SAT_MAX,
    "rgb_mean_min": RGB_MEAN_MIN,
    "rgb_mean_max": RGB_MEAN_MAX,
    "scan_angle_max_corridor": SCAN_ANGLE_MAX_CORRIDOR,
    "scan_angle_ground": "none (all angles kept)",
    "ground_count": int(ground.sum()),
    "corridor_count": int(corridor.sum()),
    "ground_s_range": [float(Sg.min()), float(Sg.max())],
    "ground_t_range": [float(Tg.min()), float(Tg.max())],
    "corridor_s_range": [float(Sc.min()), float(Sc.max())],
    "corridor_t_range": [float(Tc.min()), float(Tc.max())],
    "corridor_width_iqr_median": float(np.nanmedian(t_widths)) if valid.sum() >= 2 else None,
    "corridor_width_iqr_max": float(np.nanmax(t_widths)) if valid.sum() >= 2 else None,
    "curvature_ratio": curvature_ratio,
    "curvature_t_range_m": curvature_t_range,
    "ground_sparse_bins": n_sparse,
    "npz_file": npz_path,
}

meta_out = os.path.join(OUT_DIR, "step04_ST_metadata.json")
with open(meta_out, "w", encoding="utf-8") as f:
    json.dump(step4_meta, f, indent=2, ensure_ascii=False)

print(f"\nStep 4 metadata saved: {meta_out}")

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("STEP 4 SUMMARY — S-T PROJECTION & DATA PREPARATION")
print("=" * 60)
print(f"PCA azimuth:          {angle_deg:.2f}° (source: {pca_source})")
print(f"Corridor params:      sat≤{SAT_MAX}, {RGB_MEAN_MIN}≤rgb≤{RGB_MEAN_MAX}")
print(f"Ground points:        {fmt_n(ground.sum())} ({ground.mean()*100:.2f}%)  [no scan_angle filter]")
print(f"Corridor points:      {fmt_n(corridor.sum())} ({corridor.mean()*100:.2f}%)  [scan_angle ≤{SCAN_ANGLE_MAX_CORRIDOR}°]")
print(f"Curvature ratio:      {curvature_ratio:.4f} (T range = {curvature_t_range:.1f} m)")
if valid.sum() >= 2:
    print(f"Corridor width (IQR): median={np.nanmedian(t_widths):.2f} m, max={np.nanmax(t_widths):.2f} m")
print(f"Ground sparse bins:   {n_sparse} / {len(d_bins)}")
print(f"NPZ saved:            {npz_path} ({npz_size_mb:.1f} MB)")

print("\nDesign notes:")
print("  1. Ground has NO scan_angle filter — preserves full terrain (ditches, slopes)")
print("     for downstream ditch analysis in step07.")
print("  2. Corridor has scan_angle ≤ 20° — quality control for ridge tracking in step05.")
print("  3. Corridor width IQR per S-bin: normal ballast should be ~2-5 m.")
print("     Abnormally wide bins may indicate misclassified non-ballast points.")
print("  4. Ground density plot (18c) flags sparse regions that may produce")
print("     unreliable cross-sections in step07.")

print("\nOutput files:")
print("  18_step4_formal_ST_overview.png     — S-T overview with T-median tracking")
print("  18b_step4_corridor_diagnostics.png  — Corridor drift + width + density")
print("  18c_step4_ground_density.png        — Ground point density along S")
print("  19_step4_ST_sections_Z.png          — Z-coloured diagnostic sections")
print("  20_step4_ST_sections_ExG.png        — ExG-coloured diagnostic sections")
print("  21_step4_section_locations_XY.png   — Section locations in XY")
print("  step04_ST_metadata.json             — Step 4 metadata")
print("  step04_ST_projected.npz             — Projected data for step05")

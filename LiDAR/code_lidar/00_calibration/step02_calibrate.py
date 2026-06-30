import os
import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

# =========================================================
# USER SETTINGS
# =========================================================
BASE_DIR = os.environ.get("LIDAR_BASE", os.getcwd())
LAZ_FILE = os.environ.get("LIDAR_LAZ_FILE", os.path.join(BASE_DIR, "data", "laz", "tile.laz"))
OUT_DIR = os.environ.get("LIDAR_OUT_DIR", os.path.join(BASE_DIR, "output"))
os.makedirs(OUT_DIR, exist_ok=True)

GROUND_Z_RANGE = 8.0
RANDOM_SEED = 42

# =========================================================
# TEST SETS — Phase 1: RGB-only baselines
# Phase 2 sets are auto-generated after Phase 1 by bootstrapping
# the intensity distribution from the best RGB corridor.
#
# Yang & Fang (2014, IEEE JSTARS) show intensity is a key
# discriminator between rail/ballast/vegetation — but the
# absolute values are sensor-dependent, so we derive them
# from the data rather than hardcoding.
# =========================================================
RGB_ONLY_SETS = [
    (0.25, 35, 130),   # Set 1: relatively wide  (sat_max, rgb_min, rgb_max)
    (0.23, 38, 125),   # Set 2
    (0.215, 40, 120),  # Set 3
    (0.195, 42, 115),  # Set 4: balanced candidate
    (0.175, 45, 110),  # Set 5: strict
    (0.155, 47, 105),  # Set 6: stricter
]

# ExG upper bound to exclude green vegetation
EXG_CORRIDOR_MAX = 15.0

# Intensity bootstrap percentiles (applied to RGB-corridor points)
INTENSITY_BOOTSTRAP_LO_PCTL = 5    # exclude low outliers (water/shadow)
INTENSITY_BOOTSTRAP_HI_PCTL = 95   # exclude high outliers (specular)

# Scan angle filter — large angles have worse geometric accuracy
# and distorted intensity (Arastounia 2015)
SCAN_ANGLE_MAX = 20

# Local ridge-tracking settings
S_BIN_SIZE = 5.0
S_HALF_WIN = 6.0
T_BIN_SIZE = 0.10
LOCAL_MIN_POINTS = 80
T_PEAK_MIN_REL_HEIGHT = 0.15
T_PEAK_MIN_DIST_M = 1.5
MAX_JUMP = 4.0
MIN_VALID_BINS = 20

# Edge diagnostic — MLS tile edges may have degraded quality due to
# GNSS/IMU trajectory drift (not inherently one-sided; depends on
# acquisition conditions).  Check BOTH ends symmetrically.
EDGE_DIAG_SPAN = 80.0   # metres from each tile edge for quality check

np.random.seed(RANDOM_SEED)


# =========================================================
# HELPERS
# =========================================================
def fmt_n(x):
    return f"{int(x):,}"


def save_show(fig, out_path):
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
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
    """Return scan angle in degrees for LAS 1.2 and LAS 1.4 style files."""
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


def ST_to_XY(Sv, Tv, mean_x, mean_y, angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    X = mean_x + Sv * cos_a - Tv * sin_a
    Y = mean_y + Sv * sin_a + Tv * cos_a
    return X, Y


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


def track_local_ridge(S_vals, T_vals, Z_vals, label=""):
    """
    Track a local T ridge along S by:
      1. binning along S
      2. computing T histograms in each bin
      3. choosing local peaks
      4. linking by nearest-previous-peak rule
    """
    if len(S_vals) < 1000:
        return None

    s_min = float(np.percentile(S_vals, 1))
    s_max = float(np.percentile(S_vals, 99))
    s_samples = np.arange(s_min, s_max + S_BIN_SIZE, S_BIN_SIZE)

    ridge_s = []
    ridge_t = []
    ridge_n = []
    ridge_zmean = []

    prev_t = None

    for s0 in s_samples:
        win = np.abs(S_vals - s0) < S_HALF_WIN
        n_win = int(win.sum())
        if n_win < LOCAL_MIN_POINTS:
            continue

        T_win = T_vals[win]
        Z_win = Z_vals[win]

        t_lo, t_hi = np.percentile(T_win, [1, 99])
        bins = np.arange(t_lo, t_hi + T_BIN_SIZE, T_BIN_SIZE)
        if len(bins) < 6:
            continue

        hist, edges = np.histogram(T_win, bins=bins)
        ctrs = (edges[:-1] + edges[1:]) / 2
        hist_sm = gaussian_filter1d(hist.astype(float), sigma=2)

        min_dist_bins = max(1, int(T_PEAK_MIN_DIST_M / T_BIN_SIZE))
        peaks, _ = find_peaks(
            hist_sm,
            height=hist_sm.max() * T_PEAK_MIN_REL_HEIGHT,
            distance=min_dist_bins
        )

        if len(peaks) == 0:
            continue

        peak_ts = ctrs[peaks]
        peak_hs = hist_sm[peaks]

        peak_info = []
        for tp, th in zip(peak_ts, peak_hs):
            near = np.abs(T_win - tp) < 1.5
            if near.sum() < 20:
                z_mean = np.nan
            else:
                z_mean = float(np.mean(Z_win[near]))
            peak_info.append((tp, th, z_mean))

        if prev_t is None:
            chosen = max(
                peak_info,
                key=lambda x: ((-999 if np.isnan(x[2]) else x[2]), x[1])
            )
        else:
            dists = [abs(tp - prev_t) for tp, _, _ in peak_info]
            idx = int(np.argmin(dists))
            chosen = peak_info[idx]

        tp, th, z_mean = chosen

        ridge_s.append(float(s0))
        ridge_t.append(float(tp))
        ridge_n.append(n_win)
        ridge_zmean.append(z_mean)
        prev_t = tp

    if len(ridge_s) < MIN_VALID_BINS:
        return None

    ridge_s = np.asarray(ridge_s)
    ridge_t = np.asarray(ridge_t)
    ridge_n = np.asarray(ridge_n)
    ridge_zmean = np.asarray(ridge_zmean)

    # isolated jump cleanup
    if len(ridge_t) >= 3:
        dt_fwd = np.abs(np.diff(ridge_t, append=ridge_t[-1]))
        dt_bwd = np.abs(np.diff(ridge_t, prepend=ridge_t[0]))
        bad = (dt_fwd > MAX_JUMP) & (dt_bwd > MAX_JUMP)
        ridge_s = ridge_s[~bad]
        ridge_t = ridge_t[~bad]
        ridge_n = ridge_n[~bad]
        ridge_zmean = ridge_zmean[~bad]

    if len(ridge_s) < MIN_VALID_BINS:
        return None

    ridge_t_sm = make_savgol(ridge_t, max_win=21, poly=3)

    s_span = ridge_s.max() - ridge_s.min()
    dt = np.abs(np.diff(ridge_t_sm))
    jump_ratio = float((dt > MAX_JUMP).mean()) if len(dt) else 0.0
    mad_step = float(np.median(dt)) if len(dt) else np.nan
    coverage = float(len(ridge_s))
    mean_count = float(np.mean(ridge_n))

    # edge diagnostics — check BOTH ends (MLS quality can degrade
    # at either tile edge depending on GNSS/IMU conditions)
    def _edge_mad(mask):
        if mask.sum() >= 3:
            dt_e = np.abs(np.diff(ridge_t_sm[mask]))
            return float(np.median(dt_e)) if len(dt_e) else np.nan
        return np.nan

    left_edge_mask = ridge_s <= (ridge_s.min() + EDGE_DIAG_SPAN)
    right_edge_mask = ridge_s >= (ridge_s.max() - EDGE_DIAG_SPAN)

    left_edge_mad = _edge_mad(left_edge_mask)
    right_edge_mad = _edge_mad(right_edge_mask)

    # worst-edge MAD: use the worse of the two ends
    edge_mads = [m for m in (left_edge_mad, right_edge_mad) if not np.isnan(m)]
    worst_edge_mad = float(max(edge_mads)) if edge_mads else np.nan

    return {
        "label": label,
        "ridge_s": ridge_s,
        "ridge_t": ridge_t,
        "ridge_t_sm": ridge_t_sm,
        "ridge_n": ridge_n,
        "ridge_zmean": ridge_zmean,
        "s_span": s_span,
        "jump_ratio": jump_ratio,
        "mad_step": mad_step,
        "coverage": coverage,
        "mean_count": mean_count,
        "left_edge_mad": left_edge_mad,
        "right_edge_mad": right_edge_mad,
        "worst_edge_mad": worst_edge_mad,
    }


# =========================================================
# LOAD DATA
# =========================================================
print("=" * 60)
print("STEP 2 — Gray corridor calibration (local-ridge version, 6-set)")
print("=" * 60)

las = laspy.read(LAZ_FILE)

# Memory-conservative loading (see step06/step07 for rationale): cast
# coordinates to float32 right after the offset subtraction so at most
# one full float64 array is alive at a time.
_x64 = np.asarray(las.x)
x0 = float(_x64.min())
xl = (_x64 - x0).astype(np.float32); del _x64

_y64 = np.asarray(las.y)
y0 = float(_y64.min())
yl = (_y64 - y0).astype(np.float32); del _y64

z = np.asarray(las.z, dtype=np.float32)
intensity = np.asarray(las.intensity)
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

ExG = np.float32(2.0) * G - R - B

saturation, rgb_mean = hsv_saturation(R, G, B)

print(f"  Intensity range: {intensity.min()} – {intensity.max()}")
print(f"  Scan angle range: {scan_angle.min()} – {scan_angle.max()}")
print(f"  ExG range: {ExG.min():.1f} – {ExG.max():.1f}")

# =========================================================
# STEP 2.1 — GROUND FILTER
# =========================================================
print("\n[STEP 2.1] Ground-layer filtering (+ scan angle constraint)...")
z_floor = np.percentile(z, 1)
ground = (
    (z >= z_floor) & (z <= z_floor + GROUND_Z_RANGE) &
    (np.abs(scan_angle) <= SCAN_ANGLE_MAX)
)

xg = xl[ground]
yg = yl[ground]
zg = z[ground]

print(f"z_floor (P1):        {z_floor:.2f} m")
print(f"scan_angle max:      {SCAN_ANGLE_MAX}°")
print(f"Ground points:       {fmt_n(ground.sum())} ({ground.mean() * 100:.1f}%)")

# =========================================================
# STEP 2.2 — PCA / PROVISIONAL ST
# =========================================================
print("\n[STEP 2.2] PCA track azimuth from ground layer...")
angle_deg, main_axis, mean_xy = compute_pca_track_azimuth(xg, yg)
angle_rad = np.radians(angle_deg)
mean_x, mean_y = mean_xy[0], mean_xy[1]

Sg, Tg = to_ST(xg, yg, mean_x, mean_y, angle_rad)

print(f"PCA azimuth: {angle_deg:.2f}°")
print(f"Main axis:   ({main_axis[0]:.4f}, {main_axis[1]:.4f})")
print(f"S range:     {Sg.min():.1f} – {Sg.max():.1f} m")
print(f"T range:     {Tg.min():.1f} – {Tg.max():.1f} m")

# =========================================================
# FIGURE 1 — PCA OVERVIEW
# =========================================================
idx_g_plot = np.random.choice(len(xg), min(200_000, len(xg)), replace=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
sc = ax.scatter(xg[idx_g_plot], yg[idx_g_plot], c=zg[idx_g_plot],
                cmap="terrain", s=0.3, alpha=0.5)
plt.colorbar(sc, ax=ax, label="Z (m)")
cx, cy = mean_x, mean_y
L = 120
ax.annotate(
    "", xy=(cx + main_axis[0] * L, cy + main_axis[1] * L),
    xytext=(cx - main_axis[0] * L, cy - main_axis[1] * L),
    arrowprops=dict(arrowstyle='->', color='red', lw=2.5)
)
ax.set_title("Ground-layer PCA direction", fontsize=10)
ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_aspect("equal")

ax = axes[1]
sc = ax.scatter(Sg[idx_g_plot], Tg[idx_g_plot], c=zg[idx_g_plot],
                cmap="terrain", s=0.3, alpha=0.5)
plt.colorbar(sc, ax=ax, label="Z (m)")
ax.set_title("Ground points in provisional S-T coordinates")
ax.set_xlabel("S (m)")
ax.set_ylabel("T (m)")

save_show(fig, os.path.join(OUT_DIR, "09_step2_pca_ground_ST.png"))

# =========================================================
# STEP 2.3 — PHASE 1: RGB-ONLY SETS
# =========================================================
print("\n[STEP 2.3] Phase 1 — Testing RGB-only corridor sets...")

results_phase1 = []

for i, (smax, rmin, rmax) in enumerate(RGB_ONLY_SETS):
    mask = ground & (saturation <= smax) & (rgb_mean >= rmin) & (rgb_mean <= rmax)
    n = int(mask.sum())
    ratio = 100 * mask.mean()

    Xc = xl[mask]
    Yc = yl[mask]
    Zc = z[mask]

    if len(Xc) > 0:
        Sc, Tc = to_ST(Xc, Yc, mean_x, mean_y, angle_rad)
        ridge = track_local_ridge(Sc, Tc, Zc, label=f"Set {i+1}")
    else:
        Sc = Tc = np.array([])
        ridge = None

    results_phase1.append({
        "set_id": i + 1,
        "smax": smax,
        "rmin": rmin,
        "rmax": rmax,
        "imin": None,
        "imax": None,
        "exg_max": None,
        "n": n,
        "ratio": ratio,
        "Xc": Xc,
        "Yc": Yc,
        "Sc": Sc,
        "Tc": Tc,
        "Zc": Zc,
        "ridge": ridge,
    })

    print(f"  Set {i+1}: sat<{smax}, {rmin}<rgb<{rmax} → "
          f"n={fmt_n(n)}, ridge={'OK' if ridge else 'NONE'}")

# =========================================================
# STEP 2.3b — BOOTSTRAP INTENSITY RANGE FROM BEST RGB SET
# =========================================================
print("\n[STEP 2.3b] Bootstrapping intensity range from RGB corridor...")

# Use the widest RGB set that still produces a valid ridge
bootstrap_set = None
for res in results_phase1:
    if res["ridge"] is not None:
        bootstrap_set = res
        break

if bootstrap_set is not None:
    # Get intensity values of RGB-corridor points
    bootstrap_mask = (
        ground &
        (saturation <= bootstrap_set["smax"]) &
        (rgb_mean >= bootstrap_set["rmin"]) &
        (rgb_mean <= bootstrap_set["rmax"])
    )
    corridor_intensities = intensity[bootstrap_mask]

    I_lo = float(np.percentile(corridor_intensities, INTENSITY_BOOTSTRAP_LO_PCTL))
    I_hi = float(np.percentile(corridor_intensities, INTENSITY_BOOTSTRAP_HI_PCTL))

    print(f"  Bootstrap from Set {bootstrap_set['set_id']} "
          f"(n={fmt_n(bootstrap_set['n'])} corridor pts)")
    print(f"  Corridor intensity P{INTENSITY_BOOTSTRAP_LO_PCTL}: {I_lo:.0f}")
    print(f"  Corridor intensity P{INTENSITY_BOOTSTRAP_HI_PCTL}: {I_hi:.0f}")
    print(f"  → Intensity band: [{I_lo:.0f}, {I_hi:.0f}]")
else:
    I_lo, I_hi = None, None
    print("  WARNING: no valid RGB ridge found, skipping intensity bootstrap")

# =========================================================
# STEP 2.3c — PHASE 2: RGB + INTENSITY + ExG SETS
# =========================================================
print("\n[STEP 2.3c] Phase 2 — Testing enhanced sets (RGB + intensity + ExG)...")

results_phase2 = []

if I_lo is not None:
    for i, (smax, rmin, rmax) in enumerate(RGB_ONLY_SETS):
        mask = (
            ground &
            (saturation <= smax) & (rgb_mean >= rmin) & (rgb_mean <= rmax) &
            (intensity >= I_lo) & (intensity <= I_hi) &
            (ExG <= EXG_CORRIDOR_MAX)
        )
        n = int(mask.sum())
        ratio = 100 * mask.mean()

        Xc = xl[mask]
        Yc = yl[mask]
        Zc = z[mask]

        if len(Xc) > 0:
            Sc, Tc = to_ST(Xc, Yc, mean_x, mean_y, angle_rad)
            ridge = track_local_ridge(Sc, Tc, Zc,
                                      label=f"Set {len(RGB_ONLY_SETS)+i+1}")
        else:
            Sc = Tc = np.array([])
            ridge = None

        results_phase2.append({
            "set_id": len(RGB_ONLY_SETS) + i + 1,
            "smax": smax,
            "rmin": rmin,
            "rmax": rmax,
            "imin": I_lo,
            "imax": I_hi,
            "exg_max": EXG_CORRIDOR_MAX,
            "n": n,
            "ratio": ratio,
            "Xc": Xc,
            "Yc": Yc,
            "Sc": Sc,
            "Tc": Tc,
            "Zc": Zc,
            "ridge": ridge,
        })

        print(f"  Set {len(RGB_ONLY_SETS)+i+1}: sat<{smax}, {rmin}<rgb<{rmax}, "
              f"I=[{I_lo:.0f},{I_hi:.0f}], ExG<{EXG_CORRIDOR_MAX} → "
              f"n={fmt_n(n)}, ridge={'OK' if ridge else 'NONE'}")

# Combine all results for downstream figures/selection
results = results_phase1 + results_phase2

# =========================================================
# FIGURE 2 — XY VIEW OF ALL SETS
# =========================================================
n_sets = len(results)
n_cols = 2
n_rows = int(np.ceil(n_sets / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5.5 * n_rows))
axes = axes.ravel()

for i, res in enumerate(results):
    ax = axes[i]
    ax.scatter(xg[idx_g_plot], yg[idx_g_plot], c="lightgray", s=0.2, alpha=0.12)

    if len(res["Xc"]) > 0:
        idx_plot = np.random.choice(len(res["Xc"]), min(120_000, len(res["Xc"])), replace=False)
        ax.scatter(res["Xc"][idx_plot], res["Yc"][idx_plot], c="navy", s=0.3, alpha=0.6)

    extra = ""
    ax.set_title(f"Set {res['set_id']}", fontsize=10)
    ax.set_xlabel("X local (m)")
    ax.set_ylabel("Y local (m)")
    ax.set_aspect("equal")

# hide unused subplots
for j in range(n_sets, len(axes)):
    axes[j].set_visible(False)

save_show(fig, os.path.join(OUT_DIR, "10_step2_gray_corridor_sets_xy.png"))

# =========================================================
# FIGURE 3 — LOCAL RIDGE COMPARISON IN ST
# =========================================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
axes = axes.ravel()

for i, res in enumerate(results):
    ax = axes[i]
    idx_bg = np.random.choice(len(Sg), min(120_000, len(Sg)), replace=False)
    ax.scatter(Sg[idx_bg], Tg[idx_bg], c="lightgray", s=0.2, alpha=0.08)

    ridge = res["ridge"]
    if ridge is not None:
        ax.plot(ridge["ridge_s"], ridge["ridge_t"], ".", ms=4, alpha=0.5,
                color="navy", label="raw local ridge")
        ax.plot(ridge["ridge_s"], ridge["ridge_t_sm"], "-", lw=2.0,
                color="red", label="smoothed ridge")

        pass

    ax.set_title(f"Set {res['set_id']}", fontsize=10)
    ax.set_xlabel("S (m)")
    ax.set_ylabel("T (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

# hide unused subplots
for j in range(n_sets, len(axes)):
    axes[j].set_visible(False)

save_show(fig, os.path.join(OUT_DIR, "11_step2_local_ridge_comparison_ST.png"))

# =========================================================
# FIGURE 4 — BOTH-EDGE ZOOM FOR RIDGE BEHAVIOUR
# =========================================================
s_global_min = np.percentile(Sg, 1)
s_global_max = np.percentile(Sg, 99)

edges = {
    "left":  (s_global_min, s_global_min + EDGE_DIAG_SPAN),
    "right": (s_global_max - EDGE_DIAG_SPAN, s_global_max),
}

for edge_name, (s_lo, s_hi) in edges.items():
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.ravel()

    bg_mask = (Sg >= s_lo) & (Sg <= s_hi)
    bg_idx = np.where(bg_mask)[0]

    for i, res in enumerate(results):
        ax = axes[i]

        if len(bg_idx) > 0:
            bg_plot = np.random.choice(bg_idx, min(60_000, len(bg_idx)), replace=False)
            ax.scatter(Sg[bg_plot], Tg[bg_plot], c="lightgray", s=0.25, alpha=0.10)

        if len(res["Sc"]) > 0:
            c_mask = (res["Sc"] >= s_lo) & (res["Sc"] <= s_hi)
            c_idx = np.where(c_mask)[0]
            if len(c_idx) > 0:
                c_plot = np.random.choice(c_idx, min(50_000, len(c_idx)), replace=False)
                ax.scatter(res["Sc"][c_plot], res["Tc"][c_plot], c="navy", s=0.3, alpha=0.45)

        ridge = res["ridge"]
        if ridge is not None:
            r_mask = (ridge["ridge_s"] >= s_lo) & (ridge["ridge_s"] <= s_hi)
            if r_mask.sum() > 0:
                ax.plot(ridge["ridge_s"][r_mask], ridge["ridge_t"][r_mask], ".",
                        ms=5, alpha=0.55, color="black", label="raw ridge")
                ax.plot(ridge["ridge_s"][r_mask], ridge["ridge_t_sm"][r_mask], "-",
                        lw=2.5, color="red", label="smoothed ridge")

        ax.set_title(f"Set {res['set_id']} — {edge_name} edge", fontsize=10)
        ax.set_xlabel("S (m)")
        ax.set_ylabel("T (m)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for j in range(n_sets, len(axes)):
        axes[j].set_visible(False)

    save_show(fig, os.path.join(OUT_DIR, f"11b_step2_{edge_name}_edge_ridge_zoom.png"))

# =========================================================
# STEP 2.4 — SELECT BEST SET BY RIDGE QUALITY
# =========================================================
print("\n[STEP 2.4] Selecting best set by local-ridge quality...")

best = None
best_score = -np.inf

for res in results:
    ridge = res["ridge"]
    if ridge is None:
        continue

    edge_penalty = 0.0 if np.isnan(ridge["worst_edge_mad"]) else ridge["worst_edge_mad"]

    # Heuristic:
    #   more coverage is good
    #   smaller jump ratio is good
    #   smaller MAD step is good
    #   smaller candidate ratio is good
    #   smaller worst-edge MAD is good (penalise the worse end)
    #   bonus for using intensity+ExG (more robust across datasets)
    enhanced_bonus = 15.0 if res.get("imin") is not None else 0.0

    score = (
        2.0 * ridge["coverage"]
        - 80.0 * ridge["jump_ratio"]
        - 10.0 * ridge["mad_step"]
        - 0.5 * res["ratio"]
        - 8.0 * edge_penalty
        + enhanced_bonus
    )

    if score > best_score:
        best_score = score
        best = res

if best is None:
    raise RuntimeError("No test set produced a valid local ridge.")

best_ridge = best["ridge"]

extra_label = ""
if best.get("imin") is not None:
    extra_label = f", I=[{best['imin']:.0f},{best['imax']:.0f}], ExG<{best['exg_max']}"
print(f"Selected set: sat<{best['smax']}, {best['rmin']}<rgb<{best['rmax']}{extra_label}")
print(f"Candidate ratio: {best['ratio']:.1f}% of all points")
print(f"Local ridge coverage: {int(best_ridge['coverage'])} S-bins")
print(f"Local ridge S span:   {best_ridge['s_span']:.1f} m")
print(f"Local ridge MAD step: {best_ridge['mad_step']:.2f} m")
print(f"Left-edge MAD step:   {best_ridge['left_edge_mad']:.2f} m")
print(f"Local ridge jump ratio (> {MAX_JUMP:.1f} m): {best_ridge['jump_ratio']:.2f}")

# =========================================================
# FIGURE 5 — BEST SET PREVIEW
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# XY preview
ax = axes[0]
ax.scatter(xg[idx_g_plot], yg[idx_g_plot], c="lightgray", s=0.2, alpha=0.10, label="ground")

idx_best_xy = np.random.choice(len(best["Xc"]), min(120_000, len(best["Xc"])), replace=False)
ax.scatter(best["Xc"][idx_best_xy], best["Yc"][idx_best_xy], c="navy", s=0.3, alpha=0.6,
           label="best-set candidates")

S_line = best_ridge["ridge_s"]
T_line = best_ridge["ridge_t_sm"]
X_line, Y_line = ST_to_XY(S_line, T_line, mean_x, mean_y, angle_rad)

ax.plot(X_line, Y_line, color="red", lw=2.5, label="local ridge preview")
ax.set_title("Best set (XY)", fontsize=10)
ax.set_xlabel("X local (m)")
ax.set_ylabel("Y local (m)")
ax.set_aspect("equal")
ax.legend(markerscale=8)

# ST preview
ax = axes[1]
idx_bg2 = np.random.choice(len(Sg), min(150_000, len(Sg)), replace=False)
ax.scatter(Sg[idx_bg2], Tg[idx_bg2], c="lightgray", s=0.2, alpha=0.08, label="ground")

idx_best_st = np.random.choice(len(best["Sc"]), min(120_000, len(best["Sc"])), replace=False)
ax.scatter(best["Sc"][idx_best_st], best["Tc"][idx_best_st], c="navy", s=0.3, alpha=0.5,
           label="best-set candidates")
ax.plot(best_ridge["ridge_s"], best_ridge["ridge_t"], ".", ms=4, alpha=0.5,
        color="black", label="raw local ridge")
ax.plot(best_ridge["ridge_s"], best_ridge["ridge_t_sm"], "-", lw=2.5,
        color="red", label="smoothed ridge")
ax.set_title("Best set (S-T)", fontsize=10)
ax.set_xlabel("S (m)")
ax.set_ylabel("T (m)")
ax.legend(markerscale=8)

save_show(fig, os.path.join(OUT_DIR, "12_step2_best_set_local_ridge_preview.png"))

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("STEP 2 SUMMARY — LOCAL-RIDGE GRAY CORRIDOR CALIBRATION")
print("=" * 60)

for res in results:
    ridge = res["ridge"]
    tag = " [+I+ExG]" if res.get("imin") is not None else ""
    if ridge is None:
        print(
            f"Set {res['set_id']}{tag}: sat<{res['smax']}, "
            f"{res['rmin']}<rgb<{res['rmax']} | "
            f"n={fmt_n(res['n'])} ({res['ratio']:.1f}%) | ridge = NONE"
        )
    else:
        print(
            f"Set {res['set_id']}{tag}: sat<{res['smax']}, "
            f"{res['rmin']}<rgb<{res['rmax']} | "
            f"n={fmt_n(res['n'])} ({res['ratio']:.1f}%) | "
            f"coverage={int(ridge['coverage'])} | "
            f"S span={ridge['s_span']:.1f} m | "
            f"MAD={ridge['mad_step']:.2f} | "
            f"edge MAD: L={ridge['left_edge_mad']:.2f} R={ridge['right_edge_mad']:.2f} | "
            f"jump ratio={ridge['jump_ratio']:.2f}"
        )

print("\nRecommended provisional set:")
print(f"  saturation < {best['smax']}")
print(f"  {best['rmin']} < rgb_mean < {best['rmax']}")
if best.get("imin") is not None:
    print(f"  intensity:  [{best['imin']:.0f}, {best['imax']:.0f}]  (bootstrapped)")
    print(f"  ExG max:    {best['exg_max']}")
else:
    print(f"  intensity:  not used")
print(f"  local ridge coverage   = {int(best_ridge['coverage'])}")
print(f"  local ridge S span     = {best_ridge['s_span']:.1f} m")
print(f"  local ridge MAD step   = {best_ridge['mad_step']:.2f} m")
print(f"  edge MAD: left={best_ridge['left_edge_mad']:.2f} m, "
      f"right={best_ridge['right_edge_mad']:.2f} m")
print(f"  local ridge jump ratio = {best_ridge['jump_ratio']:.2f}")

print("\nInterpretation:")
print("  1. Phase 1 tests RGB-only sets; Phase 2 adds intensity + ExG exclusion.")
print("  2. Intensity range is auto-bootstrapped from Phase 1 corridor points.")
print("  3. Enhanced sets are preferred (bonus in scoring) when ridge quality is similar,")
print("     because intensity is light-independent (Yang & Fang 2014).")
print("  4. The best set should produce a narrow candidate band and a smooth local T ridge.")
print("  5. Next step: use the best set in the dedicated curved-centreline tracking script.")

print("\nOutput figures:")
print("  09_step2_pca_ground_ST.png")
print("  10_step2_gray_corridor_sets_xy.png")
print("  11_step2_local_ridge_comparison_ST.png")
print("  11b_step2_left_edge_ridge_zoom.png")
print("  12_step2_best_set_local_ridge_preview.png")

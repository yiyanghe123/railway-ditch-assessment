import os
import json
import warnings

import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline

# =========================================================
# USER SETTINGS
# =========================================================
BASE_DIR = os.environ.get("LIDAR_BASE", os.getcwd())
LAZ_FILE = os.environ.get("LIDAR_LAZ_FILE", os.path.join(BASE_DIR, "data", "laz", "tile.laz"))
OUT_DIR = os.environ.get("LIDAR_OUT_DIR", os.path.join(BASE_DIR, "output"))
STEP03_META = os.path.join(OUT_DIR, "step03_ST_metadata.json")
STEP05_META = os.path.join(OUT_DIR, "step05_centreline_metadata.json")
STEP05_CSV = os.path.join(OUT_DIR, "rail_centreline_step05.csv")
STEP05_DENSE_CSV = os.path.join(OUT_DIR, "rail_centreline_dense_step05.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# FIXED INPUTS FROM STEP 1/2/3/5
# =========================================================
GROUND_Z_RANGE = 8.0

# ── Relative height filter (tree/catenary removal) ───────
# After establishing local reference Z, exclude points more than
# this height above the local ballast/rail level.
# Rationale: catenary ~5.5m above rail, trees can be 3-15m.
# We keep generous margin to include embankment slopes but remove
# overhead objects.
REL_HEIGHT_ABOVE_MAX = 3.0   # m above local upper envelope median

# local section geometry
CHAINAGE_STEP = 1.0
LOCAL_SECTION_HALF_U = 0.75
LOCAL_SEARCH_HALF_U = 1.00
LOCAL_SUPPORT_V_HALF = 2.0
SECTION_T_RANGE = 8.0

# memory-safe preselection
LOCAL_S_PRESELECT = 3.0  # m

# engineering prior
GAUGE_PRIOR = 1.435
HALF_GAUGE_PRIOR = GAUGE_PRIOR / 2.0

# rail support — relaxed from 15° to 20° (consistent with step02-04)
SCAN_ANGLE_MAX = 20
# ── Adaptive support scan-angle (sensor-geometry robustness) ─────────────
# The ≤20° support filter assumes near-nadir ground returns.  The 2025 Cirrus
# tiles (point format 7) are wide-FOV: ground-return |scan_angle| has a median
# of ~50°, so only ~20% of ground sits within 20° and essentially zero points
# survive at the rail offset → n_support / rail_support_score / gauge_measured
# all collapse to 0 (Z_ref is unaffected; it comes from the P90 envelope, not
# support).  When too few ground points fall within SCAN_ANGLE_MAX we relax the
# support angle to the tile's own P95, so the gauge cross-check works on any
# scanner geometry.  Set SUPPORT_AUTO_SCAN_ANGLE=0 to force the fixed 20°.
SUPPORT_AUTO_SCAN_ANGLE     = os.environ.get("SUPPORT_AUTO_SCAN_ANGLE", "1") == "1"
SUPPORT_SCAN_ANGLE_MIN_FRAC = 0.50   # relax if fewer ground pts than this are within SCAN_ANGLE_MAX
SUPPORT_V_HALF = 0.20
SUPPORT_MIN_POINTS = 10
SUPPORT_SCORE_NORM = 40.0
SUPPORT_STRONG_SCORE = 0.25
SUPPORT_WEAK_SCORE = 0.10

# intensity is not a core criterion
USE_INTENSITY_FILTER = False
RAIL_INTENSITY_MAX = 6000

# ── Z_ref: RUK (Rälens Underkant, rail bottom) per TDOK 2015:0155 §10.13 ──
# TDOK 2015:0155 §10.13 specifies ditch depth thresholds measured below RUK:
#   - Priority Å (3-year action):  depth < 1.0 m below RUK
#   - Post-intervention target:     depth ≥ 1.3 m below RUK
#
# We detect the rail top (ROK, Rälens Överkant) from the upper envelope P90
# peaks, then subtract the nominal rail height to obtain RUK. This aligns
# all downstream depth calculations with the Swedish regulatory thresholds.
#
# Fallback: ballast shoulder (v ∈ [0.90, 1.40]) when no rail peak detected.
RAIL_HEIGHT_M = 0.16     # nominal rail height (BV50 ≈ 0.152 m, UIC60 ≈ 0.172 m)
ZREF_INNER = 0.90        # ballast shoulder inner (fallback only)
ZREF_OUTER = 1.40        # ballast shoulder outer (fallback only)
ZREF_MIN_POINTS = 12
ZREF_OK_POINTS = 20

# edge handling
EDGE_BUFFER_M = 10.0

# lower envelope / section diagnostics
ENV_BIN_W = 0.03
ENV_Q = 5

# ── Upper-envelope: raised from P70 to P90 (literature) ──
RAIL_ENV_BIN_W = 0.03
RAIL_ENV_Q = 90           # P90 — captures actual rail crown better
RAIL_ENV_SMOOTH_WIN = 11
RAIL_ENV_V_RANGE = 2.2

# =========================================================
# RAIL-PEAK REFINEMENT SETTINGS
# =========================================================
RAIL_SEARCH_HALF_W = 0.38
RAIL_PEAK_MIN_PROM = 0.008    # lowered slightly — P90 makes peaks more prominent
RAIL_PEAK_MAX_DIST = 0.28     # m from v_prior
RAIL_SCORE_MIN = 0.35

RAIL_MAX_CENTER_SHIFT_BOTH = 0.20
RAIL_MAX_CENTER_SHIFT_SINGLE = 0.10
RAIL_SINGLE_SIDE_SHRINK = 0.45

# ── Rail head width validation (UIC60: 72mm head width) ────
RAIL_HEAD_HALF_W = 0.036      # half of 72mm rail head
RAIL_HEAD_PLATEAU_RATIO = 0.50  # min fraction of head bins that must be elevated
RAIL_HEAD_DROP_THRESHOLD = 0.010  # m — max drop from peak Z within head width
RAIL_WIDTH_BONUS = 0.15       # score bonus for passing width validation

# Pair quality checks
RAIL_Z_DIFF_MAX = 0.06        # m
RAIL_GAUGE_TOL = 0.10         # m — relaxed from 0.08 for paired search
RAIL_BOTH_MIN_SCORE = 0.40    # relaxed slightly — paired search is more reliable

# gauge measurement
GAUGE_MEAS_BIN_W = 0.03
GAUGE_MEAS_SIGMA = 0.5
GAUGE_MIN = 1.30
GAUGE_MAX = 1.60
GAUGE_MIN_DIST_BINS = int(0.5 / GAUGE_MEAS_BIN_W)

# continuity / smoothing
REFINE_MEDIAN_WIN = 9
REFINE_SMOOTH_MAX_WIN = 31
REFINE_SMOOTH_POLY = 2
MAX_INTERP_GAP_FOR_REFINE = 8

# ── Weighted spline Z_ref smoothing ──────────────────────
# Weights by source quality (replaces hard RANSAC threshold)
ZREF_WEIGHT_BOTH_RAILS = 1.0
ZREF_WEIGHT_SINGLE_RAIL = 0.6
ZREF_WEIGHT_SHOULDER = 0.3
ZREF_WEIGHT_NONE = 0.05       # near-zero but keeps spline defined
ZREF_SPLINE_SMOOTH_FACTOR = 0.002  # s = n * factor (lower = tighter fit)

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
    S = (xv - mean_x) * cos_a + (yv - mean_y) * sin_a
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


def interp_centerline(s_query, s_nodes, t_nodes, dtds_nodes):
    t0 = float(np.interp(s_query, s_nodes, t_nodes))
    d0 = float(np.interp(s_query, s_nodes, dtds_nodes))
    return t0, d0


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


def compute_rail_support(v_vals, sa_vals, intensity_vals, v_prior,
                         scan_angle_max=SCAN_ANGLE_MAX):
    if USE_INTENSITY_FILTER:
        m = (
            (np.abs(v_vals - v_prior) <= SUPPORT_V_HALF) &
            (np.abs(sa_vals) <= scan_angle_max) &
            (intensity_vals <= RAIL_INTENSITY_MAX)
        )
    else:
        m = (
            (np.abs(v_vals - v_prior) <= SUPPORT_V_HALF) &
            (np.abs(sa_vals) <= scan_angle_max)
        )
    n = int(m.sum())
    score = min(1.0, n / SUPPORT_SCORE_NORM)
    return n, float(score)


def estimate_zref_shoulder(v_vals, z_vals, side="left"):
    """Fallback Z_ref from ballast shoulder (used when rail peaks not detected)."""
    if side == "left":
        m = (v_vals >= -ZREF_OUTER) & (v_vals <= -ZREF_INNER)
    else:
        m = (v_vals >= +ZREF_INNER) & (v_vals <= +ZREF_OUTER)
    n = int(m.sum())
    if n < ZREF_MIN_POINTS:
        return np.nan, n
    return float(np.median(z_vals[m])), n


def classify_reference_mode(n_left, n_right):
    left_ok = n_left >= SUPPORT_MIN_POINTS
    right_ok = n_right >= SUPPORT_MIN_POINTS
    if left_ok and right_ok:
        return "supported_prior"
    if left_ok or right_ok:
        return "mixed"
    return "prior_only"


def classify_support_strength(score_left, score_right):
    max_score = max(score_left, score_right)
    mean_score = float(np.mean([score_left, score_right]))
    if (score_left >= SUPPORT_STRONG_SCORE) and (score_right >= SUPPORT_STRONG_SCORE):
        return "strong_both"
    if max_score >= SUPPORT_STRONG_SCORE:
        return "strong_one_side"
    if max_score >= SUPPORT_WEAK_SCORE or mean_score >= SUPPORT_WEAK_SCORE:
        return "weak"
    return "none"


def build_lower_envelope(v_vals, z_vals, v_range=4.0, bin_w=0.03, q=5):
    bins = np.arange(-v_range, v_range + bin_w, bin_w)
    env_v, env_z = [], []
    for j in range(len(bins) - 1):
        m = (v_vals >= bins[j]) & (v_vals < bins[j + 1])
        if m.sum() >= 3:
            env_v.append((bins[j] + bins[j + 1]) / 2.0)
            env_z.append(np.percentile(z_vals[m], q))
    env_v = np.asarray(env_v)
    env_z = np.asarray(env_z)
    if len(env_z) >= 5:
        env_z = make_savgol(env_z, max_win=15, poly=3)
    return env_v, env_z


def build_quantile_envelope(v_vals, z_vals, v_range=2.2, bin_w=0.03, q=90, smooth_win=11):
    bins = np.arange(-v_range, v_range + bin_w, bin_w)
    env_v, env_z = [], []
    for j in range(len(bins) - 1):
        m = (v_vals >= bins[j]) & (v_vals < bins[j + 1])
        if m.sum() >= 3:
            env_v.append((bins[j] + bins[j + 1]) / 2.0)
            env_z.append(np.percentile(z_vals[m], q))
    env_v = np.asarray(env_v)
    env_z = np.asarray(env_z)
    if len(env_z) >= 5:
        env_z = make_savgol(env_z, max_win=smooth_win, poly=2)
    return env_v, env_z


def measure_gauge_per_section(v_sup_sec):
    if len(v_sup_sec) < 10:
        return np.nan
    v_min, v_max = v_sup_sec.min(), v_sup_sec.max()
    if (v_max - v_min) < 1.0:
        return np.nan
    bins = np.arange(v_min, v_max + GAUGE_MEAS_BIN_W, GAUGE_MEAS_BIN_W)
    if len(bins) < 4:
        return np.nan
    h, e = np.histogram(v_sup_sec, bins=bins)
    c = (e[:-1] + e[1:]) / 2.0
    h_sm = gaussian_filter1d(h.astype(float), sigma=GAUGE_MEAS_SIGMA)
    pks, _ = find_peaks(h_sm, height=max(1, h_sm.max() * 0.05), distance=GAUGE_MIN_DIST_BINS)
    if len(pks) < 2:
        return np.nan
    peak_ts = c[pks]
    best, best_sc = None, np.inf
    for a in range(len(peak_ts)):
        for b in range(a + 1, len(peak_ts)):
            gap = peak_ts[b] - peak_ts[a]
            if GAUGE_MIN <= gap <= GAUGE_MAX:
                sc = abs(gap - GAUGE_PRIOR)
                if sc < best_sc:
                    best_sc = sc
                    best = gap
    return float(best) if best is not None else np.nan


# =========================================================
# RAIL-PEAK DETECTION + CENTER REFINEMENT
# =========================================================
def check_rail_head_width(env_v, env_z, v_peak, z_peak,
                          head_half_w=RAIL_HEAD_HALF_W,
                          drop_thresh=RAIL_HEAD_DROP_THRESHOLD,
                          plateau_ratio=RAIL_HEAD_PLATEAU_RATIO):
    """
    Validate that a peak has UIC60-like rail head plateau (~72mm wide).
    A real rail crown stays elevated within ±36mm of the peak centre.
    A noise spike drops off quickly.
    Returns True if the plateau shape is confirmed.
    """
    head_mask = (env_v >= v_peak - head_half_w) & (env_v <= v_peak + head_half_w)
    n_head = int(head_mask.sum())
    if n_head < 2:
        return False
    z_head = env_z[head_mask]
    elevated = (z_peak - z_head) <= drop_thresh
    return float(elevated.sum()) / n_head >= plateau_ratio


def find_rail_peak_in_window(env_v, env_z, v_prior,
                             search_half_w=RAIL_SEARCH_HALF_W,
                             peak_min_prom=RAIL_PEAK_MIN_PROM,
                             peak_max_dist=RAIL_PEAK_MAX_DIST):
    out = {
        "valid": False,
        "v_obs": np.nan,
        "z_obs": np.nan,
        "prominence": np.nan,
        "score": 0.0,
        "width_ok": False,
    }
    if len(env_v) < 5:
        return out
    m = (env_v >= v_prior - search_half_w) & (env_v <= v_prior + search_half_w)
    if m.sum() < 5:
        return out
    v_w = env_v[m]
    z_w = env_z[m]
    order = np.argsort(v_w)
    v_w = v_w[order]
    z_w = z_w[order]
    try:
        pks, props = find_peaks(z_w, prominence=peak_min_prom)
    except Exception:
        return out
    candidates = []
    if len(pks) > 0:
        prominences = props.get("prominences", np.full(len(pks), np.nan))
        for idx_pk, prom in zip(pks, prominences):
            vp = float(v_w[idx_pk])
            zp = float(z_w[idx_pk])
            dist = abs(vp - v_prior)
            if dist <= peak_max_dist:
                closeness = max(0.0, 1.0 - dist / peak_max_dist)
                prom_term = min(1.0, prom / max(peak_min_prom, 1e-6))
                # width validation: UIC60 rail head plateau check
                w_ok = check_rail_head_width(env_v, env_z, vp, zp)
                width_bonus = RAIL_WIDTH_BONUS if w_ok else 0.0
                score = 0.55 * closeness + 0.25 * prom_term + 0.20 * (1.0 if w_ok else 0.0)
                candidates.append((score + width_bonus, vp, zp, prom, w_ok))
    if len(candidates) == 0:
        return out
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    out["valid"] = bool(best[0] >= RAIL_SCORE_MIN)
    out["v_obs"] = best[1]
    out["z_obs"] = best[2]
    out["prominence"] = best[3]
    out["score"] = float(best[0])
    out["width_ok"] = bool(best[4])
    return out


def paired_gauge_search(env_v, env_z,
                        gauge_prior=GAUGE_PRIOR,
                        gauge_tol=RAIL_GAUGE_TOL,
                        z_diff_max=RAIL_Z_DIFF_MAX,
                        peak_min_prom=RAIL_PEAK_MIN_PROM):
    """
    Paired gauge-constrained rail search (Arastounia 2018 inspired).
    Instead of finding left/right peaks independently then checking gauge,
    slide a gauge-width window across the envelope and find the position
    where combined left+right peak quality is maximised at ~1.435m separation.

    Returns (best_left_v, best_left_z, best_right_v, best_right_z,
             best_score, gauge_obs) or None if no valid pair found.
    """
    if len(env_v) < 10:
        return None
    order = np.argsort(env_v)
    ev = env_v[order]
    ez = env_z[order]

    # find all peaks in the full envelope
    try:
        pks, props = find_peaks(ez, prominence=peak_min_prom * 0.5)
    except Exception:
        return None
    if len(pks) < 2:
        return None

    prominences = props.get("prominences", np.full(len(pks), 0.0))
    peak_vs = ev[pks]
    peak_zs = ez[pks]

    # try all pairs within gauge tolerance
    best = None
    best_score = -1.0
    for i in range(len(pks)):
        for j in range(i + 1, len(pks)):
            gap = peak_vs[j] - peak_vs[i]
            if abs(gap - gauge_prior) > gauge_tol:
                continue
            z_diff = abs(peak_zs[j] - peak_zs[i])
            if z_diff > z_diff_max:
                continue

            # gauge closeness term
            gauge_closeness = 1.0 - abs(gap - gauge_prior) / gauge_tol
            # prominence terms
            prom_L = min(1.0, prominences[i] / max(peak_min_prom, 1e-6))
            prom_R = min(1.0, prominences[j] / max(peak_min_prom, 1e-6))
            # height terms (higher peaks = more likely rail)
            z_mean = 0.5 * (peak_zs[i] + peak_zs[j])
            z_range = ez.max() - ez.min() if ez.max() > ez.min() else 1.0
            height_term = min(1.0, (z_mean - ez.min()) / z_range)
            # width validation bonus
            w_L = check_rail_head_width(ev, ez, peak_vs[i], peak_zs[i])
            w_R = check_rail_head_width(ev, ez, peak_vs[j], peak_zs[j])
            width_bonus = 0.15 * (int(w_L) + int(w_R))

            score = (0.35 * gauge_closeness +
                     0.20 * (prom_L + prom_R) / 2.0 +
                     0.15 * height_term +
                     0.10 * (1.0 - z_diff / z_diff_max) +
                     width_bonus)

            if score > best_score:
                best_score = score
                best = (float(peak_vs[i]), float(peak_zs[i]),
                        float(peak_vs[j]), float(peak_zs[j]),
                        float(score), float(gap), w_L, w_R)

    return best


def refine_center_from_rail_peaks(env_v, env_z,
                                  left_peak, right_peak,
                                  half_gauge=HALF_GAUGE_PRIOR,
                                  gauge_prior=GAUGE_PRIOR,
                                  gauge_tol=RAIL_GAUGE_TOL,
                                  z_diff_max=RAIL_Z_DIFF_MAX):
    """
    Improved: first try paired gauge-constrained search on the full envelope.
    If that fails, fall back to independent peak results.
    """
    # ── Try paired gauge-constrained search first ──
    pair = paired_gauge_search(env_v, env_z, gauge_prior, gauge_tol, z_diff_max)
    if pair is not None:
        vL, zL, vR, zR, pair_score, gauge_obs, wL, wR = pair
        center_shift_raw = 0.5 * (vL + vR)
        z_lr_diff = abs(zR - zL)

        # update left/right peak info for Z_ref computation
        left_peak["valid"] = True
        left_peak["v_obs"] = vL
        left_peak["z_obs"] = zL
        left_peak["score"] = pair_score
        left_peak["width_ok"] = wL

        right_peak["valid"] = True
        right_peak["v_obs"] = vR
        right_peak["z_obs"] = zR
        right_peak["score"] = pair_score
        right_peak["width_ok"] = wR

        both_width = wL and wR
        confidence = "high" if (both_width and abs(gauge_obs - gauge_prior) <= 0.05) else "medium"

        center_shift = float(np.clip(center_shift_raw,
                                     -RAIL_MAX_CENTER_SHIFT_BOTH,
                                     RAIL_MAX_CENTER_SHIFT_BOTH))
        return {
            "center_shift_raw": float(center_shift_raw),
            "center_shift": center_shift,
            "rail_detect_mode": "both",
            "rail_confidence": confidence,
            "gauge_obs": gauge_obs,
            "z_lr_diff": z_lr_diff,
        }

    # ── Fallback: use independent peak results ──
    left_ok = bool(left_peak["valid"])
    right_ok = bool(right_peak["valid"])

    if left_ok and right_ok:
        gauge_obs = right_peak["v_obs"] - left_peak["v_obs"]
        z_lr_diff = abs(right_peak["z_obs"] - left_peak["z_obs"])
        center_shift_raw = 0.5 * (left_peak["v_obs"] + right_peak["v_obs"])
        center_shift = float(np.clip(center_shift_raw, -0.12, 0.12))
        return {
            "center_shift_raw": float(center_shift_raw),
            "center_shift": center_shift,
            "rail_detect_mode": "both",
            "rail_confidence": "medium",
            "gauge_obs": gauge_obs,
            "z_lr_diff": z_lr_diff,
        }

    if left_ok:
        center_shift_raw = left_peak["v_obs"] + half_gauge
        center_shift = float(np.clip(RAIL_SINGLE_SIDE_SHRINK * center_shift_raw,
                                     -RAIL_MAX_CENTER_SHIFT_SINGLE,
                                     RAIL_MAX_CENTER_SHIFT_SINGLE))
        return {
            "center_shift_raw": float(center_shift_raw),
            "center_shift": center_shift,
            "rail_detect_mode": "left_only",
            "rail_confidence": "medium" if left_peak["score"] >= 0.55 else "low",
            "gauge_obs": np.nan,
            "z_lr_diff": np.nan,
        }

    if right_ok:
        center_shift_raw = right_peak["v_obs"] - half_gauge
        center_shift = float(np.clip(RAIL_SINGLE_SIDE_SHRINK * center_shift_raw,
                                     -RAIL_MAX_CENTER_SHIFT_SINGLE,
                                     RAIL_MAX_CENTER_SHIFT_SINGLE))
        return {
            "center_shift_raw": float(center_shift_raw),
            "center_shift": center_shift,
            "rail_detect_mode": "right_only",
            "rail_confidence": "medium" if right_peak["score"] >= 0.55 else "low",
            "gauge_obs": np.nan,
            "z_lr_diff": np.nan,
        }

    return {
        "center_shift_raw": 0.0,
        "center_shift": 0.0,
        "rail_detect_mode": "none",
        "rail_confidence": "low",
        "gauge_obs": np.nan,
        "z_lr_diff": np.nan,
    }


def compute_zref_from_rails(left_peak, right_peak):
    """
    Z_ref = RUK (Rälens Underkant, rail bottom) per TDOK 2015:0155 §10.13.

    Procedure: the upper-envelope peak gives the rail top (ROK). We subtract
    the nominal rail height (RAIL_HEIGHT_M) to obtain RUK, which is the
    reference datum used by TDOK 2015:0155 for ditch depth thresholds
    (1.0 m / 1.3 m below RUK).

    Primary: mean of left/right rail-top Z − RAIL_HEIGHT_M.
    Returns (zref_ruk, source) where source is "both_rails", "left_rail",
    "right_rail", or "none".
    """
    left_valid = left_peak["valid"] and not np.isnan(left_peak["z_obs"])
    right_valid = right_peak["valid"] and not np.isnan(right_peak["z_obs"])

    if left_valid and right_valid:
        rok = float(np.mean([left_peak["z_obs"], right_peak["z_obs"]]))
        return rok - RAIL_HEIGHT_M, "both_rails"
    if left_valid:
        return float(left_peak["z_obs"]) - RAIL_HEIGHT_M, "left_rail"
    if right_valid:
        return float(right_peak["z_obs"]) - RAIL_HEIGHT_M, "right_rail"
    return np.nan, "none"


def classify_zref_quality(zref_valid, zref_source):
    if not zref_valid:
        return "missing"
    if zref_source in ("both_rails", "both_shoulders"):
        return "ok"
    return "weak"


def compute_usable_for_ditch(zref_valid, zref_quality):
    return int(bool(zref_valid) and zref_quality in {"ok", "weak"})


def robust_smooth_refined_center(s_vals, t_raw, mode_vals):
    s_vals = np.asarray(s_vals)
    t_raw = np.asarray(t_raw, dtype=float)
    mode_vals = np.asarray(mode_vals)

    qual = np.zeros_like(t_raw, dtype=float)
    qual[mode_vals == "both"] = 1.0
    qual[(mode_vals == "left_only") | (mode_vals == "right_only")] = 0.45
    qual[mode_vals == "none"] = 0.0

    t_use = t_raw.copy()
    t_use[qual == 0.0] = np.nan

    ser = pd.Series(t_use)
    t_interp = ser.interpolate(
        limit=MAX_INTERP_GAP_FOR_REFINE,
        limit_direction="both"
    ).to_numpy(copy=True)

    miss = np.isnan(t_interp)
    if miss.any():
        t_interp[miss] = t_raw[miss]

    t_med = median_filter(t_interp, size=REFINE_MEDIAN_WIN, mode="nearest")
    t_smooth = make_savgol(t_med, max_win=REFINE_SMOOTH_MAX_WIN, poly=REFINE_SMOOTH_POLY)
    return t_interp, t_med, t_smooth


# =========================================================
# LOAD PREVIOUS-STEP METADATA
# =========================================================
print("=" * 60)
print("STEP 06 — Reference framework (paired rail search + weighted spline Z_ref)")
print("=" * 60)

for req in [STEP03_META, STEP05_META, STEP05_CSV]:
    if not os.path.exists(req):
        raise FileNotFoundError(f"Missing required input: {req}")

with open(STEP03_META, "r", encoding="utf-8") as f:
    meta3 = json.load(f)

with open(STEP05_META, "r", encoding="utf-8") as f:
    meta5 = json.load(f)

cl_df = pd.read_csv(STEP05_CSV)

# try to load dense centreline for smoother interpolation
if os.path.exists(STEP05_DENSE_CSV):
    cl_dense_df = pd.read_csv(STEP05_DENSE_CSV)
    print(f"Loaded dense centreline: {STEP05_DENSE_CSV} ({len(cl_dense_df)} pts)")
    use_dense = True
else:
    cl_dense_df = None
    use_dense = False
    print("Dense centreline not found, using node-level centreline.")

angle_deg = float(meta3["pca_track_azimuth_deg"])
angle_rad = np.radians(angle_deg)
mean_x = float(meta3["mean_x_local"])
mean_y = float(meta3["mean_y_local"])

SAT_MAX = float(meta3["sat_max"])
RGB_MEAN_MIN = float(meta3["rgb_mean_min"])
RGB_MEAN_MAX = float(meta3["rgb_mean_max"])

print(f"PCA azimuth: {angle_deg:.2f}°")
print(f"Mean XY local: ({mean_x:.2f}, {mean_y:.2f})")
print(f"Corridor parameters: sat<{SAT_MAX}, {RGB_MEAN_MIN}<rgb_mean<{RGB_MEAN_MAX}")

# centreline arrays — prefer dense if available
if use_dense:
    cl_s = cl_dense_df["S"].values.astype(float)
    cl_t_sm = cl_dense_df["T_center"].values.astype(float)
else:
    cl_s = cl_df["S"].values.astype(float)
    cl_t_sm = cl_df["T_center_spline"].values.astype(float)

cl_t_raw = cl_df["T_center_raw"].values.astype(float)
cl_s_raw = cl_df["S"].values.astype(float)

if len(cl_s) < 5:
    raise RuntimeError("Step 5 centreline has too few nodes.")

cl_dtds = np.gradient(cl_t_sm, cl_s)

print(f"Centreline: {len(cl_s)} points, S=[{cl_s.min():.1f}, {cl_s.max():.1f}]")

# =========================================================
# READ LAZ
# =========================================================
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("\nReading LAZ...")
import gc
las = laspy.read(LAZ_FILE)

# Memory-conservative loading: Cirrus tiles are 60–80M points per file.
# A single float64 array of 80M pts is ~640 MB; full-cloud float64 for
# x,y,z costs ~1.9 GB sustained. We cast coordinates to float32 right
# after the offset subtraction (UTM → local frame preserves precision),
# and z / intensity / scan_angle directly to float32.
_x64 = np.asarray(las.x)
_y64 = np.asarray(las.y)
x0 = float(_x64.min())
y0 = float(_y64.min())
x = (_x64 - x0).astype(np.float32); del _x64
y = (_y64 - y0).astype(np.float32); del _y64
z = np.asarray(las.z, dtype=np.float32)
intensity = np.asarray(las.intensity)
gc.collect()

scan_angle = scan_angle_degrees(las)

if not all(hasattr(las, ch) for ch in ["red", "green", "blue"]):
    raise RuntimeError("RGB fields are required.")

R_raw = np.asarray(las.red)
G_raw = np.asarray(las.green)
B_raw = np.asarray(las.blue)

if R_raw.max() > 256:
    print("  RGB: 16-bit detected, dividing by 256")
    R = R_raw.astype(np.float32) / np.float32(256.0)
    G = G_raw.astype(np.float32) / np.float32(256.0)
    B = B_raw.astype(np.float32) / np.float32(256.0)
else:
    R = R_raw.astype(np.float32)
    G = G_raw.astype(np.float32)
    B = B_raw.astype(np.float32)
del R_raw, G_raw, B_raw
gc.collect()

# x, y are already offset-subtracted float32; keep xl, yl aliases for
# downstream compatibility but they now point at the same memory.
xl, yl = x, y

saturation, rgb_mean_arr = hsv_saturation(R, G, B)
# R, G, B no longer needed after saturation + rgb_mean computation
del R, G, B
gc.collect()

print(f"  Total points:    {fmt_n(len(x))}")
print(f"  Scan angle max:  {SCAN_ANGLE_MAX}° (relaxed from 15°)")

# =========================================================
# GROUND + CORRIDOR
# =========================================================
z_floor = np.percentile(z, 1)
gnd = (z >= z_floor) & (z <= z_floor + GROUND_Z_RANGE)
print(f"  Ground layer: {fmt_n(gnd.sum())} pts ({gnd.mean() * 100:.1f}%)")

S_all, T_all = to_ST(xl, yl, mean_x, mean_y, angle_rad)
S_all = S_all.astype(np.float32)
T_all = T_all.astype(np.float32)
del xl, yl, x, y  # free full coordinate arrays (x, y are aliases of xl, yl)
gc.collect()

corridor_mask = (
    gnd &
    (saturation <= SAT_MAX) &
    (rgb_mean_arr >= RGB_MEAN_MIN) &
    (rgb_mean_arr <= RGB_MEAN_MAX)
)

print(f"  Corridor candidates: {fmt_n(corridor_mask.sum())} ({corridor_mask.mean() * 100:.2f}%)")

# pre-sort ground points by S
S_gnd = S_all[gnd]
T_gnd = T_all[gnd]
Z_gnd = z[gnd].astype(np.float32)
I_gnd = intensity[gnd]

sort_idx = np.argsort(S_gnd)
S_gnd = S_gnd[sort_idx]
T_gnd = T_gnd[sort_idx]
Z_gnd = Z_gnd[sort_idx]
I_gnd = I_gnd[sort_idx]
del sort_idx

# support points (scan angle filter — adaptive to sensor geometry)
EFF_SCAN_ANGLE_MAX = float(SCAN_ANGLE_MAX)
if SUPPORT_AUTO_SCAN_ANGLE:
    _frac_in = float((np.abs(scan_angle[gnd]) <= SCAN_ANGLE_MAX).mean()) if gnd.sum() > 0 else 1.0
    if _frac_in < SUPPORT_SCAN_ANGLE_MIN_FRAC:
        EFF_SCAN_ANGLE_MAX = float(np.percentile(np.abs(scan_angle[gnd]), 95))
        print(f"  [SUPPORT] Wide-FOV scanner: only {_frac_in*100:.0f}% of ground within "
              f"{SCAN_ANGLE_MAX}° → relaxing support scan-angle to P95={EFF_SCAN_ANGLE_MAX:.0f}°")
rail_support_mask = gnd & (np.abs(scan_angle) <= EFF_SCAN_ANGLE_MAX)
S_sup = S_all[rail_support_mask]
T_sup = T_all[rail_support_mask]
I_sup = intensity[rail_support_mask]
SA_sup = scan_angle[rail_support_mask]

sort_sup = np.argsort(S_sup)
S_sup = S_sup[sort_sup]
T_sup = T_sup[sort_sup]
I_sup = I_sup[sort_sup]
SA_sup = SA_sup[sort_sup]
del sort_sup

print(f"  Scan_angle support pts: {fmt_n(rail_support_mask.sum())} (≤{EFF_SCAN_ANGLE_MAX:.0f}°)")

# corridor for figures
S_corr = S_all[corridor_mask]
T_corr = T_all[corridor_mask]
del S_all, T_all  # free full arrays

# =========================================================
# BUILD PER-METRE REFERENCE FRAMEWORK
# =========================================================
print("\nBuilding per-metre reference framework...")
print(f"  Z_ref source: RAIL TOP (P{RAIL_ENV_Q} upper envelope)")
print(f"  Fallback: ballast shoulder (v ∈ [{ZREF_INNER}, {ZREF_OUTER}] m)")
print(f"  Relative height filter: Z < local_ref + {REL_HEIGHT_ABOVE_MAX} m")

chainage_values = np.arange(
    np.ceil(cl_s.min()),
    np.floor(cl_s.max()) + CHAINAGE_STEP,
    CHAINAGE_STEP
)

records = []
n_sections = len(chainage_values)
s_global_min = float(chainage_values.min())
s_global_max = float(chainage_values.max())

for idx_c, s_c in enumerate(chainage_values):
    if idx_c % 100 == 0:
        print(f"  Progress: {idx_c}/{n_sections}  S={s_c:.0f}m", end="\r")

    t0, d0 = interp_centerline(s_c, cl_s, cl_t_sm, cl_dtds)

    lo = int(np.searchsorted(S_gnd, s_c - LOCAL_S_PRESELECT, side="left"))
    hi = int(np.searchsorted(S_gnd, s_c + LOCAL_S_PRESELECT, side="right"))
    if hi <= lo:
        continue

    S_sub = S_gnd[lo:hi]
    T_sub = T_gnd[lo:hi]
    Z_sub = Z_gnd[lo:hi]

    rough = np.abs(T_sub - t0) <= (SECTION_T_RANGE + 2.0)
    if rough.sum() < 30:
        continue

    S_sub = S_sub[rough]
    T_sub = T_sub[rough]
    Z_sub = Z_sub[rough]

    u_all, v_all = project_to_local_frame(S_sub, T_sub, s_c, t0, d0)
    sec_mask = (np.abs(u_all) <= LOCAL_SEARCH_HALF_U) & (np.abs(v_all) <= SECTION_T_RANGE)
    if sec_mask.sum() < 30:
        continue

    v_sec = v_all[sec_mask]
    z_sec = Z_sub[sec_mask]

    # ── Relative height filter ────────────────────────────
    # Compute a rough local reference Z (median of points near centreline)
    # then remove points far above it (trees, catenary, poles).
    near_centre = np.abs(v_sec) < 2.0
    if near_centre.sum() >= 10:
        local_z_ref_rough = float(np.percentile(z_sec[near_centre], 80))
        height_mask = z_sec < (local_z_ref_rough + REL_HEIGHT_ABOVE_MAX)
        v_sec = v_sec[height_mask]
        z_sec = z_sec[height_mask]

    if len(v_sec) < 30:
        continue

    # ── Rail-top search: P90 upper-quantile envelope ──────
    env_v_r, env_z_r = build_quantile_envelope(
        v_sec, z_sec,
        v_range=RAIL_ENV_V_RANGE,
        bin_w=RAIL_ENV_BIN_W,
        q=RAIL_ENV_Q,
        smooth_win=RAIL_ENV_SMOOTH_WIN,
    )

    vL_prior = -HALF_GAUGE_PRIOR
    vR_prior = +HALF_GAUGE_PRIOR

    left_peak = find_rail_peak_in_window(env_v_r, env_z_r, vL_prior)
    right_peak = find_rail_peak_in_window(env_v_r, env_z_r, vR_prior)

    refine_info = refine_center_from_rail_peaks(
        env_v_r, env_z_r,
        left_peak, right_peak,
        half_gauge=HALF_GAUGE_PRIOR,
        gauge_prior=GAUGE_PRIOR,
        gauge_tol=RAIL_GAUGE_TOL,
        z_diff_max=RAIL_Z_DIFF_MAX,
    )
    t_refined_raw = t0 + refine_info["center_shift"]

    # ── Z_ref: RUK per TDOK 2015:0155 §10.13 ────────────
    # Primary: rail-peak path already returns (rail_top - RAIL_HEIGHT_M).
    # Fallback: ballast-shoulder detection picks points at ∈ [0.90, 1.40] m
    # outboard of centreline. In well-tamped Swedish mainline ballast the
    # shoulder P75 is close to rail-top level (observed on tile 630:
    # shoulder ≈ rail_top within ±1 cm), so we apply the same RAIL_HEIGHT_M
    # subtraction on the shoulder fallback path for consistency. This keeps
    # all Z_ref values on the same RUK datum regardless of which source
    # contributed them.
    zref_rail, zref_rail_source = compute_zref_from_rails(left_peak, right_peak)

    # fallback: ballast shoulder (shift to RUK by subtracting RAIL_HEIGHT_M)
    _zref_sh_l_raw, nZL = estimate_zref_shoulder(v_sec, z_sec, side="left")
    _zref_sh_r_raw, nZR = estimate_zref_shoulder(v_sec, z_sec, side="right")
    zref_shoulder_left = _zref_sh_l_raw - RAIL_HEIGHT_M if not np.isnan(_zref_sh_l_raw) else np.nan
    zref_shoulder_right = _zref_sh_r_raw - RAIL_HEIGHT_M if not np.isnan(_zref_sh_r_raw) else np.nan

    if not np.isnan(zref_rail):
        # primary: rail top − RAIL_HEIGHT_M (already RUK-adjusted)
        zref = zref_rail
        zref_source = zref_rail_source
    else:
        # fallback: ballast shoulder − RAIL_HEIGHT_M
        if np.isnan(zref_shoulder_left) and np.isnan(zref_shoulder_right):
            zref = np.nan
            zref_source = "none"
        elif np.isnan(zref_shoulder_left):
            zref = zref_shoulder_right
            zref_source = "right_shoulder"
        elif np.isnan(zref_shoulder_right):
            zref = zref_shoulder_left
            zref_source = "left_shoulder"
        else:
            zref = float(np.median([zref_shoulder_left, zref_shoulder_right]))
            zref_source = "both_shoulders"

    zref_valid = int(not np.isnan(zref))
    zref_quality = classify_zref_quality(bool(zref_valid), zref_source)
    near_edge = int((s_c - s_global_min) < EDGE_BUFFER_M or (s_global_max - s_c) < EDGE_BUFFER_M)
    usable_for_ditch = compute_usable_for_ditch(zref_valid, zref_quality)

    # support points
    lo_s = int(np.searchsorted(S_sup, s_c - LOCAL_S_PRESELECT, side="left"))
    hi_s = int(np.searchsorted(S_sup, s_c + LOCAL_S_PRESELECT, side="right"))

    if hi_s > lo_s:
        S_sup_sub = S_sup[lo_s:hi_s]
        T_sup_sub = T_sup[lo_s:hi_s]
        I_sup_sub = I_sup[lo_s:hi_s]
        SA_sup_sub = SA_sup[lo_s:hi_s]
        u_sup, v_sup = project_to_local_frame(S_sup_sub, T_sup_sub, s_c, t0, d0)
        sup_mask = (np.abs(u_sup) <= LOCAL_SEARCH_HALF_U) & (np.abs(v_sup) <= LOCAL_SUPPORT_V_HALF)
        v_sup_sec = v_sup[sup_mask]
        i_sup_sec = I_sup_sub[sup_mask]
        sa_sup_sec = SA_sup_sub[sup_mask]
    else:
        v_sup_sec = np.array([])
        i_sup_sec = np.array([])
        sa_sup_sec = np.array([])

    n_left, score_left = compute_rail_support(v_sup_sec, sa_sup_sec, i_sup_sec, vL_prior, EFF_SCAN_ANGLE_MAX)
    n_right, score_right = compute_rail_support(v_sup_sec, sa_sup_sec, i_sup_sec, vR_prior, EFF_SCAN_ANGLE_MAX)
    support_score = float(np.mean([score_left, score_right]))
    reference_mode = classify_reference_mode(n_left, n_right)
    support_strength = classify_support_strength(score_left, score_right)

    gauge_measured = np.nan
    if len(v_sup_sec) >= 10:
        gauge_measured = measure_gauge_per_section(v_sup_sec)

    gauge_residual_mm = np.nan
    gauge_valid_local = 0
    if not np.isnan(gauge_measured):
        gauge_residual_mm = (gauge_measured - GAUGE_PRIOR) * 1000.0
        gauge_valid_local = int(GAUGE_MIN <= gauge_measured <= GAUGE_MAX)

    T_left_prior = t0 + vL_prior
    T_right_prior = t0 + vR_prior
    Xc, Yc = ST_to_XY(np.array([s_c]), np.array([t0]), mean_x, mean_y, angle_rad)

    records.append({
        "s": float(s_c),
        "T_center_init": float(t0),
        "T_center": float(t0),
        "dT_dS": float(d0),
        "X_center_local": float(Xc[0]),
        "Y_center_local": float(Yc[0]),

        "gauge_prior": float(GAUGE_PRIOR),
        "half_gauge_prior": float(HALF_GAUGE_PRIOR),
        "T_left_prior": float(T_left_prior),
        "T_right_prior": float(T_right_prior),

        "v_left_obs": left_peak["v_obs"],
        "z_left_obs": left_peak["z_obs"],
        "left_peak_score": left_peak["score"],
        "left_peak_prominence": left_peak["prominence"],
        "left_width_ok": int(left_peak.get("width_ok", False)),

        "v_right_obs": right_peak["v_obs"],
        "z_right_obs": right_peak["z_obs"],
        "right_peak_score": right_peak["score"],
        "right_peak_prominence": right_peak["prominence"],
        "right_width_ok": int(right_peak.get("width_ok", False)),

        "rail_detect_mode": refine_info["rail_detect_mode"],
        "rail_confidence": refine_info["rail_confidence"],
        "gauge_obs_from_peaks": refine_info["gauge_obs"],
        "rail_z_diff": refine_info["z_lr_diff"],
        "center_shift_raw": float(refine_info["center_shift_raw"]),
        "center_shift_clipped": float(refine_info["center_shift"]),
        "T_center_refined_raw": float(t_refined_raw),

        "Z_ref": zref,
        "Z_ref_rail": zref_rail,
        "Z_ref_shoulder_left": zref_shoulder_left,
        "Z_ref_shoulder_right": zref_shoulder_right,
        "zref_valid": zref_valid,
        "zref_quality": zref_quality,
        "zref_source": zref_source,

        "rail_support_left": score_left,
        "rail_support_right": score_right,
        "rail_support_score": support_score,
        "n_support_left": int(n_left),
        "n_support_right": int(n_right),
        "support_strength": support_strength,

        "gauge_measured": gauge_measured,
        "gauge_residual_mm": gauge_residual_mm,
        "gauge_valid_local": gauge_valid_local,

        "reference_mode": reference_mode,
        "usable_for_ditch": usable_for_ditch,
        "n_section_pts": int(len(v_sec)),
        "n_zref_left": int(nZL),
        "n_zref_right": int(nZR),
        "near_edge": near_edge,
    })

print(f"\n  Done: {len(records)} sections built")

df = pd.DataFrame(records)
if len(df) == 0:
    raise RuntimeError("No reference sections were built.")

# =========================================================
# WEIGHTED SPLINE Z_REF SMOOTHING
# =========================================================
print("\n[Post] Weighted spline Z_ref smoothing...")
print("  Strategy: fit spline ONLY on rail-top sections, interpolate shoulder gaps")

zref_raw = df["Z_ref"].values.copy()
zref_sources = df["zref_source"].values
s_vals_arr = df["s"].values.astype(float)

# separate rail-top vs shoulder/none
is_rail = np.array([s in ("both_rails", "left_rail", "right_rail") for s in zref_sources])
is_shoulder = np.array([s in ("both_shoulders", "left_shoulder", "right_shoulder") for s in zref_sources])

# weights among rail-top sections only
weight_map = {
    "both_rails": ZREF_WEIGHT_BOTH_RAILS,
    "left_rail": ZREF_WEIGHT_SINGLE_RAIL,
    "right_rail": ZREF_WEIGHT_SINGLE_RAIL,
}

# fit spline only on rail-top derived Z_ref values
rail_mask = is_rail & ~np.isnan(zref_raw)
n_rail_fit = int(rail_mask.sum())

if n_rail_fit >= 20:
    s_rail = s_vals_arr[rail_mask]
    z_rail = zref_raw[rail_mask]
    w_rail = np.array([weight_map.get(zref_sources[i], 0.5)
                       for i in range(len(zref_sources)) if rail_mask[i]])

    try:
        spline = UnivariateSpline(
            s_rail, z_rail,
            w=w_rail,
            k=3,
            s=n_rail_fit * ZREF_SPLINE_SMOOTH_FACTOR
        )
        zref_spline = spline(s_vals_arr)
    except Exception as e:
        print(f"  WARNING: Spline fit failed ({e}), falling back to SavGol")
        zref_interp = pd.Series(zref_raw).interpolate(limit_direction="both").values
        zref_spline = make_savgol(zref_interp, max_win=31, poly=2)
else:
    print(f"  WARNING: Only {n_rail_fit} rail-top sections, using all data")
    zref_interp = pd.Series(zref_raw).interpolate(limit_direction="both").values
    still_nan = np.isnan(zref_interp)
    if still_nan.any():
        zref_interp[still_nan] = np.nanmedian(zref_raw)
    zref_spline = make_savgol(zref_interp, max_win=31, poly=2)

# compute residuals only against rail-top sections (fair comparison)
zref_residual = zref_raw - zref_spline
rail_resid = np.abs(zref_residual[rail_mask])
resid_95 = float(np.nanpercentile(rail_resid, 95)) if len(rail_resid) > 0 else np.nan
n_large_resid = int((rail_resid > 0.15).sum())

# for shoulder sections, override Z_ref with spline value (rail-top estimate)
zref_final = zref_raw.copy()
zref_final[is_shoulder] = zref_spline[is_shoulder]
zref_final[np.isnan(zref_final)] = zref_spline[np.isnan(zref_final)]

df["Z_ref_smooth"] = zref_spline
df["Z_ref_residual"] = zref_raw - zref_spline
df["zref_weight"] = np.where(is_rail, 1.0, np.where(is_shoulder, 0.3, 0.05))

print(f"  Weighted spline fit: n_rail={n_rail_fit}, smooth_factor={ZREF_SPLINE_SMOOTH_FACTOR}")
print(f"  Weight distribution: both_rails={ZREF_WEIGHT_BOTH_RAILS}, "
      f"single_rail={ZREF_WEIGHT_SINGLE_RAIL}, shoulder={ZREF_WEIGHT_SHOULDER}")
print(f"  Residual P95: {resid_95:.4f} m")
print(f"  Large residuals (>0.15m): {n_large_resid}")

# =========================================================
# POST-SMOOTHING: CENTRELINE REFINEMENT
# =========================================================
t_ref_interp, t_ref_med, t_ref_smooth = robust_smooth_refined_center(
    df["s"].values,
    df["T_center_refined_raw"].values,
    df["rail_detect_mode"].values,
)

df["T_center_refined_interp"] = t_ref_interp
df["T_center_refined_med"] = t_ref_med
df["T_center_refined"] = t_ref_smooth
df["center_shift"] = df["T_center_refined"] - df["T_center_init"]
df["center_shift"] = np.clip(df["center_shift"].values, -0.25, 0.25)
df["T_center_refined"] = df["T_center_init"] + df["center_shift"]
df["dT_dS_refined"] = np.gradient(df["T_center_refined"].values, df["s"].values)

df["T_left_prior_refined"] = df["T_center_refined"] - HALF_GAUGE_PRIOR
df["T_right_prior_refined"] = df["T_center_refined"] + HALF_GAUGE_PRIOR

X_ref, Y_ref = ST_to_XY(df["s"].values, df["T_center_refined"].values, mean_x, mean_y, angle_rad)
df["X_center_refined_local"] = X_ref
df["Y_center_refined_local"] = Y_ref

csv_path = os.path.join(OUT_DIR, "rail_reference_framework_step06.csv")
df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"  Sections built: {len(df)}")
print(f"  S range: {df['s'].min():.1f} to {df['s'].max():.1f} m")
print(f"  CSV saved: {csv_path}")

# =========================================================
# STATISTICS
# =========================================================
n_rail_src = (df["zref_source"].isin(["both_rails", "left_rail", "right_rail"])).sum()
n_shoulder_src = (df["zref_source"].isin(["both_shoulders", "left_shoulder", "right_shoulder"])).sum()
n_none_src = (df["zref_source"] == "none").sum()

print(f"\n  Z_ref source: rail_top={n_rail_src} ({n_rail_src/len(df)*100:.1f}%), "
      f"shoulder={n_shoulder_src} ({n_shoulder_src/len(df)*100:.1f}%), "
      f"none={n_none_src}")

# =========================================================
# FIGURES
# =========================================================
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
ax = axes[0]
idx_corr = np.random.choice(len(S_corr), min(200000, len(S_corr)), replace=False)
ax.scatter(S_corr[idx_corr], T_corr[idx_corr], c="lightgray", s=0.2, alpha=0.20,
           label="corridor candidates")
ax.plot(cl_s_raw, cl_t_raw, "b.", ms=4, alpha=0.35, label="raw ridge (step05)")
ax.plot(cl_s, cl_t_sm, color="red", lw=2.0, label="centreline (step05)")
ax.plot(df["s"], df["T_center_refined"], color="magenta", lw=2.3, alpha=0.9,
        label="refined centreline")
ax.plot(df["s"], df["T_left_prior_refined"], "g--", lw=1.2, alpha=0.8, label="rail priors")
ax.plot(df["s"], df["T_right_prior_refined"], "g--", lw=1.2, alpha=0.8)
ax.set_ylabel("T (m)")
ax.set_title("Centreline + rail priors", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
support_counts = df["n_support_left"].values + df["n_support_right"].values
ax.bar(df["s"], support_counts, width=CHAINAGE_STEP * 0.8, alpha=0.7, color="steelblue")
ax.set_xlabel("S (m)")
ax.set_ylabel("support pts / section")
ax.set_title("Support counts", fontsize=10)
ax.grid(alpha=0.3)
save_show(fig, os.path.join(OUT_DIR, "B1_step06_centreline_and_priors.png"))

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
ax = axes[0]
ax.plot(df["s"], df["T_center_init"], color="red", lw=1.8, label="T_center_init")
ax.plot(df["s"], df["T_center_refined"], color="magenta", lw=2.2, label="T_center_refined")
ax.set_ylabel("T (m)")
ax.set_title("Initial vs refined centreline")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(df["s"], df["center_shift"], color="black", lw=1.5, label="center_shift")
ax.axhline(0.10, color="gray", lw=1, ls=":")
ax.axhline(-0.10, color="gray", lw=1, ls=":", label="±0.10 m")
ax.axhline(0.20, color="orange", lw=1, ls="--")
ax.axhline(-0.20, color="orange", lw=1, ls="--", label="±0.20 m")
ax.set_xlabel("S (m)")
ax.set_ylabel("Shift (m)")
ax.set_title("Centreline refinement shift")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
save_show(fig, os.path.join(OUT_DIR, "B1b_step06_centreline_refinement.png"))

# Z_ref profile with rail-top vs shoulder distinction
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
ax = axes[0]
ax.plot(df["s"], df["rail_support_left"], ".", ms=2.5, alpha=0.7, label="left support")
ax.plot(df["s"], df["rail_support_right"], ".", ms=2.5, alpha=0.7, label="right support")
ax.plot(df["s"], df["rail_support_score"], "k-", lw=1.2, alpha=0.8, label="mean support")
ax.axhline(SUPPORT_STRONG_SCORE, color="orange", lw=1, ls="--", alpha=0.6,
           label=f"strong {SUPPORT_STRONG_SCORE:.2f}")
ax.set_ylabel("Support score")
ax.set_title("Rail support", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

ax = axes[1]
# color Z_ref by source
rail_src = df["zref_source"].isin(["both_rails", "left_rail", "right_rail"])
shoulder_src = df["zref_source"].isin(["both_shoulders", "left_shoulder", "right_shoulder"])
none_src = df["zref_source"] == "none"

if rail_src.sum() > 0:
    ax.plot(df.loc[rail_src, "s"], df.loc[rail_src, "Z_ref"], ".",
            color="blue", ms=3, alpha=0.7, label=f"Z_ref rail top ({rail_src.sum()})")
if shoulder_src.sum() > 0:
    ax.plot(df.loc[shoulder_src, "s"], df.loc[shoulder_src, "Z_ref"], ".",
            color="orange", ms=3, alpha=0.7, label=f"Z_ref shoulder ({shoulder_src.sum()})")
ax.plot(df["s"], df["Z_ref_smooth"], "k-", lw=1.8, alpha=0.8, label="Z_ref smooth")
ax.set_xlabel("S (m)")
ax.set_ylabel("Z_ref (m)")
ax.set_title("Reference height (Z_ref)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
save_show(fig, os.path.join(OUT_DIR, "B2_step06_zref_profile.png"))

# diagnostic cross-sections
s_test = [
    float(np.percentile(df["s"], 25)),
    float(np.percentile(df["s"], 50)),
    float(np.percentile(df["s"], 75)),
]
fig, axes = plt.subplots(3, 1, figsize=(14, 15))
for ax, s_pos in zip(axes, s_test):
    row = df.iloc[(df["s"] - s_pos).abs().argmin()]
    t0, d0 = interp_centerline(s_pos, cl_s, cl_t_sm, cl_dtds)
    zref_s = float(np.interp(s_pos, df["s"], df["Z_ref_smooth"].values))

    lo2 = int(np.searchsorted(S_gnd, s_pos - LOCAL_S_PRESELECT, side="left"))
    hi2 = int(np.searchsorted(S_gnd, s_pos + LOCAL_S_PRESELECT, side="right"))

    S_sl = S_gnd[lo2:hi2]
    T_sl = T_gnd[lo2:hi2]
    Z_sl = Z_gnd[lo2:hi2]
    I_sl = I_gnd[lo2:hi2]

    rough = np.abs(T_sl - t0) <= (SECTION_T_RANGE + 2.0)
    S_sl, T_sl, Z_sl, I_sl = S_sl[rough], T_sl[rough], Z_sl[rough], I_sl[rough]

    u_all2, v_all2 = project_to_local_frame(S_sl, T_sl, s_pos, t0, d0)
    slab = np.abs(u_all2) <= LOCAL_SECTION_HALF_U
    if slab.sum() < 20:
        ax.set_title(f"S={s_pos:.0f} m: insufficient points")
        continue

    v_slice = v_all2[slab]
    z_slice = Z_sl[slab]
    i_slice = I_sl[slab]

    env_v, env_z = build_lower_envelope(v_slice, z_slice, v_range=4.0, bin_w=ENV_BIN_W, q=ENV_Q)
    env_v_r, env_z_r = build_quantile_envelope(v_slice, z_slice,
                                                v_range=RAIL_ENV_V_RANGE,
                                                bin_w=RAIL_ENV_BIN_W,
                                                q=RAIL_ENV_Q,
                                                smooth_win=RAIL_ENV_SMOOTH_WIN)

    zoom = np.abs(v_slice) < 4.0
    sc = ax.scatter(v_slice[zoom], z_slice[zoom], c=i_slice[zoom],
                    cmap="plasma_r", s=0.4, alpha=0.25, vmin=0, vmax=10000)
    plt.colorbar(sc, ax=ax, label="Intensity")

    if len(env_v):
        ax.plot(env_v, env_z, color="darkred", lw=2.5, label="P5 envelope")
    if len(env_v_r):
        ax.plot(env_v_r, env_z_r, color="cyan", lw=2.0, alpha=0.8, label=f"P{RAIL_ENV_Q} envelope")

    ax.axhline(zref_s, color="lime", lw=1.5, ls="--", alpha=0.9, label=f"Z_ref={zref_s:.3f}m ({row['zref_source']})")
    ax.axvline(-HALF_GAUGE_PRIOR, color="lime", lw=2.2, ls="--", alpha=0.6)
    ax.axvline(+HALF_GAUGE_PRIOR, color="lime", lw=2.2, ls="--", alpha=0.6, label="rail priors")
    ax.axvline(0, color="gray", lw=1, ls=":", alpha=0.7, label="centreline")
    if not np.isnan(row["v_left_obs"]):
        ax.axvline(row["v_left_obs"], color="blue", lw=2, alpha=0.85, label="left rail peak")
    if not np.isnan(row["v_right_obs"]):
        ax.axvline(row["v_right_obs"], color="orange", lw=2, alpha=0.85, label="right rail peak")

    ax.set_xlim(-4, 4)
    ax.set_xlabel("Local cross-track v (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"S = {s_pos:.0f} m", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
save_show(fig, os.path.join(OUT_DIR, "B3_step06_reference_sections.png"))

# gauge check
if df["gauge_measured"].notna().any():
    gauge_valid = df["gauge_measured"].dropna().values
    in_tol = float(((gauge_valid >= 1.430) & (gauge_valid <= 1.440)).mean() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    n_bins = min(40, max(5, len(gauge_valid) // 2 + 1))
    ax.hist(gauge_valid, bins=n_bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(GAUGE_PRIOR, color="red", lw=2.5, ls="--", label=f"Prior {GAUGE_PRIOR:.4f}m")
    ax.axvline(gauge_valid.mean(), color="black", lw=2, label=f"Mean {gauge_valid.mean():.4f}m")
    ax.set_xlabel("Measured gauge (m)")
    ax.set_ylabel("Count")
    ax.set_title("Gauge distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df.loc[df["gauge_measured"].notna(), "s"],
            df.loc[df["gauge_measured"].notna(), "gauge_measured"],
            ".", color="steelblue", ms=5, alpha=0.7)
    ax.axhline(GAUGE_PRIOR, color="red", lw=2, ls="--", label=f"Prior {GAUGE_PRIOR:.4f}m")
    ax.set_xlabel("S (m)")
    ax.set_ylabel("Gauge (m)")
    ax.set_title("Gauge along S")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_show(fig, os.path.join(OUT_DIR, "B4_step06_gauge_check.png"))
else:
    print("\nNo valid gauge measurements. Skipping B4.")

# quality diagnostics
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
ax = axes[0]
qmap = {"ok": 2, "weak": 1, "missing": 0}
ax.plot(df["s"], [qmap[v] for v in df["zref_quality"]], "k.", ms=3)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["missing", "weak", "ok"])
ax.set_ylabel("zref quality")
ax.set_title("Reference usability diagnostics")
ax.grid(alpha=0.3)

ax = axes[1]
# color-coded zref source
src_map = {"both_rails": 4, "left_rail": 3, "right_rail": 3,
           "both_shoulders": 2, "left_shoulder": 1, "right_shoulder": 1, "none": 0}
ax.plot(df["s"], [src_map.get(v, 0) for v in df["zref_source"]], "b.", ms=3)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(["none", "1 shoulder", "2 shoulders", "1 rail", "2 rails"])
ax.set_ylabel("Z_ref source")
ax.grid(alpha=0.3)

ax = axes[2]
ax.bar(df["s"], df["usable_for_ditch"], width=0.9, color="seagreen", alpha=0.75)
ax.set_ylim(-0.05, 1.15)
ax.set_ylabel("usable")
ax.set_xlabel("S (m)")
ax.grid(alpha=0.3)
save_show(fig, os.path.join(OUT_DIR, "B5_step06_reference_quality.png"))

# rail peak diagnostics
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
ax = axes[0]
ax.plot(df["s"], df["left_peak_score"], ".", ms=3, alpha=0.7, label="left peak score")
ax.plot(df["s"], df["right_peak_score"], ".", ms=3, alpha=0.7, label="right peak score")
ax.axhline(RAIL_SCORE_MIN, color="gray", ls="--", lw=1, alpha=0.7, label=f"min={RAIL_SCORE_MIN:.2f}")
ax.set_ylabel("Peak score")
ax.set_title("Rail-peak diagnostics", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
mode_map = {"none": 0, "left_only": 1, "right_only": 2, "both": 3}
ax.plot(df["s"], [mode_map[v] for v in df["rail_detect_mode"]], "m.", ms=3)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["none", "left_only", "right_only", "both"])
ax.set_ylabel("detect mode")
ax.grid(alpha=0.3)

ax = axes[2]
cmap2 = {"low": 0, "medium": 1, "high": 2}
ax.plot(df["s"], [cmap2[v] for v in df["rail_confidence"]], "k.", ms=3)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["low", "medium", "high"])
ax.set_ylabel("rail conf.")
ax.set_xlabel("S (m)")
ax.grid(alpha=0.3)
save_show(fig, os.path.join(OUT_DIR, "B6_step06_rail_peak_diagnostics.png"))

# =========================================================
# SAVE METADATA
# =========================================================
step6_meta = {
    "source_step03_meta": STEP03_META,
    "source_step05_meta": STEP05_META,
    "source_step05_csv": STEP05_CSV,
    "output_csv": csv_path,
    "pca_track_azimuth_deg": angle_deg,
    "mean_x_local": mean_x,
    "mean_y_local": mean_y,
    "sat_max": SAT_MAX,
    "rgb_mean_min": RGB_MEAN_MIN,
    "rgb_mean_max": RGB_MEAN_MAX,
    "upper_envelope_percentile": RAIL_ENV_Q,
    "scan_angle_max": SCAN_ANGLE_MAX,
    "rel_height_above_max": REL_HEIGHT_ABOVE_MAX,
    "zref_definition": "RUK (rail bottom, TDOK 2015:0155 §10.13) = rail_top − RAIL_HEIGHT_M",
    "zref_rail_height_m": RAIL_HEIGHT_M,
    "zref_source_semantics": "both_rails/left_rail/right_rail: rail top − RAIL_HEIGHT_M; "
                             "both_shoulders/…_shoulder: shoulder P75 − RAIL_HEIGHT_M "
                             "(shoulder ≈ rail_top on tile 630)",
    "centreline_node_count": int(len(cl_s)),
    "reference_section_count": int(len(df)),
    "s_min": float(df["s"].min()),
    "s_max": float(df["s"].max()),
    "zref_valid_ratio_percent": float(df["zref_valid"].mean() * 100.0),
    "zref_ok_ratio_percent": float((df["zref_quality"] == "ok").mean() * 100.0),
    "usable_for_ditch_ratio_percent": float(df["usable_for_ditch"].mean() * 100.0),
    "zref_source_rail_count": int(n_rail_src),
    "zref_source_shoulder_count": int(n_shoulder_src),
    "zref_source_none_count": int(n_none_src),
    "zref_smoothing": "weighted_spline",
    "zref_spline_smooth_factor": ZREF_SPLINE_SMOOTH_FACTOR,
    "zref_residual_p95_m": resid_95,
    "zref_large_residuals": n_large_resid,
    "mean_support_score": float(df["rail_support_score"].mean()),
    "rail_detect_both_count": int((df["rail_detect_mode"] == "both").sum()),
    "rail_detect_left_only_count": int((df["rail_detect_mode"] == "left_only").sum()),
    "rail_detect_right_only_count": int((df["rail_detect_mode"] == "right_only").sum()),
    "rail_detect_none_count": int((df["rail_detect_mode"] == "none").sum()),
    "rail_confidence_high_count": int((df["rail_confidence"] == "high").sum()),
    "rail_confidence_medium_count": int((df["rail_confidence"] == "medium").sum()),
    "rail_confidence_low_count": int((df["rail_confidence"] == "low").sum()),
    "mean_abs_center_shift_m": float(np.mean(np.abs(df["center_shift"]))),
    "max_abs_center_shift_m": float(np.max(np.abs(df["center_shift"]))),
}

meta_out = os.path.join(OUT_DIR, "step06_reference_metadata.json")
with open(meta_out, "w", encoding="utf-8") as f:
    json.dump(step6_meta, f, indent=2, ensure_ascii=False)

print(f"\nStep 6 metadata saved: {meta_out}")

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("STEP 06 SUMMARY — PAIRED RAIL SEARCH + WEIGHTED SPLINE Z_REF")
print("=" * 60)
print(f"Per-metre sections:    {len(df)}")
print(f"S range:               {df['s'].min():.1f} to {df['s'].max():.1f} m")
print(f"Upper envelope:        P{RAIL_ENV_Q}")
print(f"Scan angle max:        {SCAN_ANGLE_MAX}°")
print(f"Relative height filt:  Z < local + {REL_HEIGHT_ABOVE_MAX} m")
print(f"Z_ref definition:      Rail top (primary), shoulder (fallback)")
print(f"Z_ref from rail top:   {n_rail_src} ({n_rail_src/len(df)*100:.1f}%)")
print(f"Z_ref from shoulder:   {n_shoulder_src} ({n_shoulder_src/len(df)*100:.1f}%)")
print(f"Z_ref none:            {n_none_src}")
print(f"Z_ref residual P95:    {resid_95:.4f} m")
print(f"Z_ref large residuals: {n_large_resid} (>0.15m)")
print(f"Z_ref valid:           {df['zref_valid'].mean() * 100:.1f}%")
print(f"Usable for ditch:      {df['usable_for_ditch'].mean() * 100:.1f}%")
print(f"Mean support score:    {df['rail_support_score'].mean():.3f}")
print(f"Mean |center shift|:   {np.mean(np.abs(df['center_shift'])):.3f} m")

print("\nRail detect mode:")
print(df["rail_detect_mode"].value_counts(dropna=False).to_string())
print("\nRail confidence:")
print(df["rail_confidence"].value_counts(dropna=False).to_string())

n_width_left = int(df["left_width_ok"].sum())
n_width_right = int(df["right_width_ok"].sum())
print(f"\nRail head width validation:")
print(f"  Left rail plateau OK:  {n_width_left}/{len(df)} ({n_width_left/len(df)*100:.1f}%)")
print(f"  Right rail plateau OK: {n_width_right}/{len(df)} ({n_width_right/len(df)*100:.1f}%)")

print("\nKey improvements over v3:")
print(f"  1. Z_ref = rail top height (P{RAIL_ENV_Q}), matching Trafikverket standard")
print(f"  2. Upper envelope raised from P70 → P{RAIL_ENV_Q} for better rail crown capture")
print(f"  3. Scan angle relaxed from 15° → {SCAN_ANGLE_MAX}° (more support points)")
print(f"  4. Relative height filter removes trees/catenary (>{REL_HEIGHT_ABOVE_MAX}m above local ref)")
print(f"  5. Weighted spline Z_ref smoothing (replaces hard RANSAC)")
print(f"  6. Uses step05 dense centreline (500pts) for smoother interpolation")
print(f"  7. Z_ref source tracked: rail_top primary, shoulder fallback")
print(f"  8. Rail head width validation (UIC60 ~72mm plateau check)")
print(f"  9. Paired gauge-constrained search (Arastounia 2018)")
print(f" 10. Source-weighted spline: both_rails={ZREF_WEIGHT_BOTH_RAILS}, "
      f"single={ZREF_WEIGHT_SINGLE_RAIL}, shoulder={ZREF_WEIGHT_SHOULDER}")

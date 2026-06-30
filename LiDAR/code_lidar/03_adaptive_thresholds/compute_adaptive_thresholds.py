"""
Adaptive ditch-detection threshold computer (v2).

This script derives ALL 59 thresholds used by step07 from data + literature,
producing a single JSON file (adaptive_thresholds_v2.json) that step07
consumes at startup.  Each threshold's value, rule, source, reference, and
sample size are written to the JSON for full provenance.

Two modes of operation
======================

  COLD START  (no prior ditch_metrics_final.csv)
  ─────────────────────────────────────────────
      python compute_adaptive_thresholds_v2.py --tile <tile-id>
        Stage 1   σ_noise per 100 m sliding window from flat sides
        Stage 2   Cat B (instrument) thresholds  ← σ-driven
        Stage 3   Cat C (geometric) thresholds   ← LITERATURE FALLBACK
        Stage 4   Cat D (classification)         ← LITERATURE FALLBACK
        Stage 5   Cat E (procedural)             ← derived from B/C
        Stage 6   Cat A / F / G                  ← literature fixed
        Writes    adaptive_thresholds_v2.json
        Source    "literature_fallback" tagged on Cat C/D fields

  BOOTSTRAP  (after step07 has run at least once)
  ──────────────────────────────────────────────
      python compute_adaptive_thresholds_v2.py --tile <tile-id> --bootstrap
        Stage 1-2 same as cold start
        Stage 3   Cat C re-derived from confirmed slopes (P5/P95)
        Stage 4   Cat D re-derived from shape/content (P25/P75/P85)
        Stage 5   Cat E re-derived using observed lateral scatter (P95)
        Stage 6   Cat A / F / G unchanged
        Writes    adaptive_thresholds_v2.json   (overwrites, but bumps
                                                  bootstrap_iter)
        Source    "bootstrap_percentile" tagged with n_samples

Convergence check
=================
After each bootstrap, the script compares the new values against the previous
iteration (saved as adaptive_thresholds_v2.PREV.json) and prints a
table of |Δ|/value.  If max relative change < 5 %, the run is declared
converged.

Output files (in --out-dir)
=====================================================
  adaptive_thresholds_v2.json            current thresholds + provenance
  adaptive_thresholds_v2.PREV.json       previous iteration (auto-rotated)
  adaptive_thresholds_v2_diagnostic.png  6-panel: σ(s), widths, slopes,
                                          flatness, DVCI, asymmetry

References
==========
Pukelsheim, F. (1994). The three sigma rule. Am. Statistician 48(2), 88–91.
Lehmann, R. (2013). 3σ-rule for outlier detection. J. Surveying Eng. 139(4).
Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
Cui, Y. et al. (2021). Terrain-adaptive LiDAR filtering. ISPRS.
Yan, X. et al. (2024). Density-based multilevel terrain noise removal.
Roelens, J. et al. (2018). Drainage ditch extraction from airborne LiDAR.
   ISPRS J. Photogramm. Remote Sens. 146, 409–420.
Cazorzi, F. et al. (2013). Drainage network detection. Hydrol. Process. 27(4).
Bailly, J.-S. et al. (2008). Agrarian linear features from LiDAR. IJRS 29(12).
Sofia, G. et al. (2011). Channel network statistical thresholds. HESS 15.
Soille, P. (2004). Morphological Image Analysis (2nd ed.). Springer.
Knighton, A. D. (1981). Asymmetry of channel cross-sections. ESPL 6(6).
Meyer, G. E. & Neto, J. C. (2008). Excess Green index. Comput. Electron.
   Agric. 63(2).
Trafikverket (2015). TDOK 2015:0155 §10.13, §8.4.1, Bilaga 2.
Rail Baltica (2025). RBDG-MAN-016 — Hydraulic, Drainage and Culverts.
Virtanen, P. et al. (2020). SciPy 1.0. Nature Methods 17(3).
Habib, A. (2021). Mobile LiDAR cross-track accuracy. (~7 cm vegetated)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from dataclasses import asdict, replace
from datetime import datetime
from typing import Optional

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d
from scipy.signal import savgol_filter

from adaptive_thresholds_schema import (
    Category,
    Source,
    THRESHOLD_REGISTRY,
    ThresholdSpec,
    by_category,
    fresh_registry,
    save_registry,
    load_registry,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ════════════════════════════════════════════════════════════════════════
# Config — must match step07's envelope-building EXACTLY
# (otherwise σ_noise estimated here doesn't apply to step07's signal)
# ════════════════════════════════════════════════════════════════════════
GROUND_Z_RANGE = 8.0
ENV_V_RANGE    = 16.0
ENV_BIN_W      = 0.05
ENV_Q          = 5
ENV_MIN_PTS    = 3
MORPH_KERNEL   = 3
MORPH_POST_WIN = 5
LOCAL_SECTION_HALF_U   = 0.75
ROUGH_S_PRESELECT_HALF = 2.0
ROUGH_T_PRESELECT_HALF = 18.0
SIDE_SEARCH_MIN        = 0.50
SIDE_SEARCH_MAX        = 15.50

# Stage 1 — σ_noise estimation
WINDOW_LEN_M       = 100.0    # length of each sliding window along S
WINDOW_STRIDE_M    = 20.0     # stride between successive windows
SECTIONS_PER_WIN   = 50       # how many sections to sample inside each window
FLAT_PERCENTILE    = 30       # bottom-N % of relief is "flat"
MIN_FLAT_SIDES_WIN = 8        # need at least this many flat sides to estimate σ
SIGMA_SMOOTH_WIN   = 7        # Sav-Gol smoothing window for σ(s)

# Stage 2-5 — adaptive gates and clamps.
# The old fixed 0.05 m local-depression floor is intentionally not used.
# The candidate gates now come from flat-profile false depressions:
# Q3(false local depressions) + 1.5 IQR, compared with sigma_noise
# multiples.  This replaces the old fixed 0.05 m / 0.02 m floors.
FALSE_DEPRESSION_FLAT_PERCENTILE = 15
MIN_FALSE_DEPRESSION_SAMPLES     = 30
WIDTH_LO_FLOOR     = 0.20     # m, design floor (TDOK Bilaga 2)
WIDTH_HI_CEIL      = 6.00     # m, design ceiling (TDOK Bilaga 2)

# ── Decouple DETECTION gates from the noise / false-depression scale ──────
# Cat B sets DITCH_MIN_PROM / DITCH_MIN_DEPTH = max(k·σ, false_depression).  On
# rough embankment tiles the false-depression estimate climbs to ~0.37 m (tile
# 7302_816), which makes DITCH_MIN_PROM = 0.37 m — far too high a prominence bar:
# find_peaks then misses the real (deep but gently-banked, water-damped) ditch
# and the detector drifts to the fallback surface passes, starving the
# longitudinal candidate path of material.  "Too shallow to be a maintained
# ditch" is a CLASSIFICATION question (SHALLOW), not a detection gate.  We cap
# the noise-derived gates at a physical ceiling.  The cap only bites when the
# noise estimate is anomalously high; clean tiles (e.g. 7309_809, gate ~0.10 m)
# stay below the ceiling and are byte-for-byte unchanged → regression-safe.
# Set DECOUPLE_NOISE_GATES=0 to restore the legacy coupling.
DECOUPLE_NOISE_GATES    = os.environ.get("DECOUPLE_NOISE_GATES", "1") == "1"
NOISE_GATE_PROM_CEIL_M  = 0.12   # max DITCH_MIN_PROM the noise scale may set
NOISE_GATE_DEPTH_CEIL_M = 0.15   # max DITCH_MIN_DEPTH the noise scale may set


# ════════════════════════════════════════════════════════════════════════
# Coordinate / envelope helpers (mirror step07 EXACTLY)
# ════════════════════════════════════════════════════════════════════════
def to_ST(xv, yv, mean_x, mean_y, angle_rad):
    cos_a = np.cos(angle_rad); sin_a = np.sin(angle_rad)
    S =  (xv - mean_x) * cos_a + (yv - mean_y) * sin_a
    T = -(xv - mean_x) * sin_a + (yv - mean_y) * cos_a
    return S, T


def project_to_local_frame(S_vals, T_vals, s0, t0, dtds):
    tangent = np.array([1.0, dtds]); tangent /= np.linalg.norm(tangent)
    normal  = np.array([-dtds, 1.0]); normal  /= np.linalg.norm(normal)
    dS = S_vals - s0; dT = T_vals - t0
    u = dS * tangent[0] + dT * tangent[1]
    v = dS * normal[0]  + dT * normal[1]
    return u, v


def sorted_slice_bounds(sv, lo, hi):
    return (int(np.searchsorted(sv, lo, side="left")),
            int(np.searchsorted(sv, hi, side="right")))


def build_lower_envelope(v_vals, z_vals,
                         v_range=ENV_V_RANGE, bin_w=ENV_BIN_W,
                         q=ENV_Q, min_pts=ENV_MIN_PTS):
    """Identical to step07 — must match for σ to mean anything."""
    bins = np.arange(-v_range, v_range + bin_w, bin_w)
    n_bins = len(bins) - 1
    grid_v = np.array([(bins[j]+bins[j+1])/2.0 for j in range(n_bins)])
    grid_z = np.full(n_bins, np.nan)
    for j in range(n_bins):
        m = (v_vals >= bins[j]) & (v_vals < bins[j+1])
        if m.sum() >= min_pts:
            grid_z[j] = np.percentile(z_vals[m], q)
    valid = ~np.isnan(grid_z)
    if valid.sum() < 5:
        idx = np.where(valid)[0]
        return grid_v[idx], grid_z[idx]
    fv = np.argmax(valid); lv = len(valid) - 1 - np.argmax(valid[::-1])
    sl = slice(fv, lv+1)
    gv = grid_v[sl].copy(); gz = grid_z[sl].copy()
    valid_inner = ~np.isnan(gz)
    if valid_inner.sum() >= 2 and (~valid_inner).sum() > 0:
        s = pd.Series(gz)
        lf = s.ffill().values; rf = s.bfill().values
        lf = np.where(np.isnan(lf), rf, lf)
        rf = np.where(np.isnan(rf), lf, rf)
        gz[~valid_inner] = np.minimum(lf[~valid_inner], rf[~valid_inner])
    if len(gz) < 5: return gv, gz
    ez = minimum_filter1d(gz, size=MORPH_KERNEL, mode="nearest")
    ez = maximum_filter1d(ez, size=MORPH_KERNEL, mode="nearest")
    if len(ez) >= 5:
        win = MORPH_POST_WIN
        if win % 2 == 0: win -= 1
        if win >= 5 and win <= len(ez):
            ez = savgol_filter(ez, window_length=win, polyorder=2)
    return gv, ez


def max_false_depression_prominence(z_profile: np.ndarray) -> float:
    """
    Estimate the strongest local false depression in a profile that is
    assumed to be non-ditch terrain.

    The purpose is not to measure the macro-scale ditch relief.  Before the
    residual is measured, a one-metre median trend is subtracted so that
    gentle side slopes and broad terrain changes do not inflate the threshold.
    The remaining positive residuals represent roughness, stones,
    interpolation artefacts, and envelope noise.
    """
    z = np.asarray(z_profile, dtype=float)
    if z.size < 5 or np.all(~np.isfinite(z)):
        return 0.0
    # ENV_BIN_W is 0.05 m in the matching pipeline, so 21 bins is about 1 m.
    # It removes broad ditch/embankment form but keeps local false dips.
    win = max(5, int(round(1.0 / ENV_BIN_W)))
    if win % 2 == 0:
        win += 1
    trend = median_filter(z, size=win, mode="nearest")
    residual_depression = trend - z
    residual_depression = residual_depression[np.isfinite(residual_depression)]
    if residual_depression.size == 0:
        return 0.0
    return float(max(0.0, np.nanmax(residual_depression)))


# ════════════════════════════════════════════════════════════════════════
# STAGE 1 — σ_noise per sliding 100 m window along S
# ════════════════════════════════════════════════════════════════════════
def estimate_sigma_per_window(Sg, Tg, Zg, ref, t_col, d_col,
                              window_len=WINDOW_LEN_M,
                              stride=WINDOW_STRIDE_M,
                              n_sections=SECTIONS_PER_WIN,
                              flat_pctl=FLAT_PERCENTILE):
    """
    For each sliding window, sample sections, build envelope on each side,
    compute σ via MAD-of-first-differences on the bottom-`flat_pctl` % of
    relief (= "flat sides").

    Returns
    -------
    centers : 1-D array of window centre S values (m)
    sigma_local : 1-D array, σ_noise per window (m)        — possibly NaN
    sigma_smooth : Sav-Gol smoothed sigma_local (m)
    sigma_global : float, median of sigma_local (NaN-safe)
    n_per_window : 1-D array, # flat sides per window
    flat_cutoff_m : float, relief threshold used
    false_depth_threshold_m : float, robust upper-outlier threshold from
        local false depressions on the flattest side profiles
    false_depth_samples : 1-D array, false-depression samples used
    flat_slope_threshold_deg : float, P90 median absolute slope of the
        flattest side profiles, used as top-bank flat-slope criterion
    flat_slope_samples_deg : 1-D array, slope samples used
    """
    s_col = "s"
    s_min = float(ref[s_col].min())
    s_max = float(ref[s_col].max())
    centers = np.arange(s_min + window_len/2.0,
                        s_max - window_len/2.0,
                        stride)
    if len(centers) == 0:
        # Degenerate: tile shorter than one window — fall back to single value
        centers = np.array([0.5 * (s_min + s_max)])

    sigma_local  = np.full(len(centers), np.nan)
    n_per_window = np.zeros(len(centers), dtype=int)

    # First pass: collect (relief, σ) pairs over all windows so we can pick
    # one consistent flat_cutoff.  This avoids each window having a different
    # definition of "flat" which would make σ(s) jagged.
    all_pairs = []   # (window_idx, relief, sigma, max_false_depression, median_abs_slope_deg)

    for wi, sc in enumerate(centers):
        win_lo = sc - window_len/2.0
        win_hi = sc + window_len/2.0
        in_win = ref[(ref[s_col] >= win_lo) & (ref[s_col] <= win_hi)]
        if len(in_win) < 10:
            continue

        sample_idx = np.linspace(0, len(in_win)-1,
                                 min(n_sections, len(in_win)),
                                 dtype=int)
        for ii in sample_idx:
            r = in_win.iloc[ii]
            s0 = float(r[s_col]); t0 = float(r[t_col]); d0 = float(r[d_col])

            i0, i1 = sorted_slice_bounds(Sg, s0 - ROUGH_S_PRESELECT_HALF,
                                              s0 + ROUGH_S_PRESELECT_HALF)
            if i1 <= i0: continue
            S_sub, T_sub, Z_sub = Sg[i0:i1], Tg[i0:i1], Zg[i0:i1]
            tm = np.abs(T_sub - t0) <= ROUGH_T_PRESELECT_HALF
            if tm.sum() < 50: continue
            S_sub, T_sub, Z_sub = S_sub[tm], T_sub[tm], Z_sub[tm]
            u, v = project_to_local_frame(S_sub, T_sub, s0, t0, d0)
            sm = (np.abs(u) <= LOCAL_SECTION_HALF_U) & (np.abs(v) <= ENV_V_RANGE)
            if sm.sum() < 50: continue
            env_v, env_z = build_lower_envelope(v[sm], Z_sub[sm])
            if len(env_v) < 20: continue

            for sign, mask_fn in [(-1, lambda v: v < 0),
                                  (+1, lambda v: v > 0)]:
                ms = mask_fn(env_v)
                if ms.sum() < 10: continue
                x_side = (-env_v[ms] if sign < 0 else env_v[ms])
                z_side = env_z[ms]
                ord_ = np.argsort(x_side); x_side = x_side[ord_]; z_side = z_side[ord_]
                keep = (x_side >= SIDE_SEARCH_MIN) & (x_side <= SIDE_SEARCH_MAX)
                if keep.sum() < 10: continue
                z_use = z_side[keep]
                relief = float(z_use.max() - z_use.min())
                diffs = np.diff(z_use)
                if len(diffs) < 5: continue
                mad = float(np.median(np.abs(diffs - np.median(diffs))))
                sigma = mad / 0.6745 / np.sqrt(2.0)
                false_dep = max_false_depression_prominence(z_use)
                slopes_deg = np.degrees(np.arctan(np.gradient(z_use, x_side[keep])))
                med_abs_slope_deg = float(np.nanmedian(np.abs(slopes_deg)))
                all_pairs.append((wi, relief, sigma, false_dep, med_abs_slope_deg))

    if len(all_pairs) < 20:
        raise RuntimeError(
            f"Only {len(all_pairs)} (window, relief, σ) samples — "
            "tile too short or framework too sparse.  Reduce WINDOW_LEN_M."
        )

    arr = np.array(all_pairs)
    relief_all = arr[:, 1]
    sigma_all  = arr[:, 2]
    false_dep_all = arr[:, 3]
    flat_cutoff_m = float(np.percentile(relief_all, flat_pctl))
    flat_mask = relief_all <= flat_cutoff_m

    false_relief_cutoff_m = float(np.percentile(
        relief_all, FALSE_DEPRESSION_FLAT_PERCENTILE
    ))
    false_mask = relief_all <= false_relief_cutoff_m
    false_depth_samples = false_dep_all[false_mask]
    false_depth_samples = false_depth_samples[np.isfinite(false_depth_samples)]
    false_depth_samples = false_depth_samples[false_depth_samples >= 0]
    flat_slope_samples_deg = arr[:, 4][false_mask]
    flat_slope_samples_deg = flat_slope_samples_deg[np.isfinite(flat_slope_samples_deg)]
    flat_slope_samples_deg = flat_slope_samples_deg[flat_slope_samples_deg >= 0]
    if len(flat_slope_samples_deg) >= MIN_FALSE_DEPRESSION_SAMPLES:
        flat_slope_threshold_deg = float(np.clip(
            np.percentile(flat_slope_samples_deg, 90),
            3.0,
            15.0,
        ))
    else:
        flat_slope_threshold_deg = float("nan")
    if len(false_depth_samples) >= MIN_FALSE_DEPRESSION_SAMPLES:
        q1, q3 = np.percentile(false_depth_samples, [25, 75])
        false_depth_threshold_m = float(q3 + 1.5 * (q3 - q1))
    else:
        false_depth_threshold_m = float("nan")

    # Aggregate per window
    for wi in range(len(centers)):
        win_mask = (arr[:, 0].astype(int) == wi) & flat_mask
        if win_mask.sum() >= MIN_FLAT_SIDES_WIN:
            sigma_local[wi] = float(np.median(sigma_all[win_mask]))
            n_per_window[wi] = int(win_mask.sum())

    # Smooth σ(s) with Sav-Gol (preserves trend, removes per-window jitter)
    sigma_smooth = sigma_local.copy()
    valid = ~np.isnan(sigma_local)
    if valid.sum() >= max(SIGMA_SMOOTH_WIN, 5):
        win = SIGMA_SMOOTH_WIN
        if win % 2 == 0: win -= 1
        if win >= 5 and valid.sum() >= win:
            sigma_smooth[valid] = savgol_filter(
                sigma_local[valid], window_length=win,
                polyorder=2, mode="nearest"
            )

    sigma_global = float(np.nanmedian(sigma_local))
    return (centers, sigma_local, sigma_smooth, sigma_global,
            n_per_window, flat_cutoff_m, len(arr),
            false_relief_cutoff_m, false_depth_threshold_m,
            false_depth_samples, flat_slope_threshold_deg,
            flat_slope_samples_deg)


# ════════════════════════════════════════════════════════════════════════
# STAGE 2 — Cat B (instrument-noise) thresholds
# ════════════════════════════════════════════════════════════════════════
def compute_cat_b(specs: dict[str, ThresholdSpec],
                  sigma_global: float,
                  false_depth_threshold_m: float,
                  false_depth_n: int,
                  bootstrap_iter: int = 0) -> None:
    """In-place fill values for Cat B specs from σ_noise."""
    sigma_depth = 6.0 * sigma_global
    sigma_prom = 1.5 * sigma_global
    sigma_outer_rise = 3.0 * sigma_global
    if np.isfinite(false_depth_threshold_m):
        min_depth = max(sigma_depth, false_depth_threshold_m)
        min_prom = max(sigma_prom, false_depth_threshold_m)
        min_outer_rise = max(sigma_outer_rise, 0.5 * false_depth_threshold_m)
        depth_source = Source.FALSE_DEPRESSION_IQR
        depth_n = int(false_depth_n)
    else:
        min_depth = sigma_depth
        min_prom = sigma_prom
        min_outer_rise = sigma_outer_rise
        depth_source = Source.SIGMA_NOISE
        depth_n = None

    # Cap the noise-derived detection gates at a physical ceiling so an
    # anomalously high false-depression scale on rough tiles cannot raise the
    # prominence / depth bar past the point where real ditches are still
    # detectable (see DECOUPLE_NOISE_GATES note above).  Regression-safe: clean
    # tiles already sit below the ceilings.
    if DECOUPLE_NOISE_GATES:
        min_prom = min(min_prom, NOISE_GATE_PROM_CEIL_M)
        min_depth = min(min_depth, NOISE_GATE_DEPTH_CEIL_M)

    rules = {
        "DITCH_MIN_DEPTH":      min_depth,
        "DITCH_MIN_PROM":       min_prom,
        "SHAPE_OUTER_RISE_MIN": min_outer_rise,
    }
    for name, val in rules.items():
        s = specs[name]
        s.value = float(val)
        s.source = Source.SIGMA_NOISE
        s.bootstrap_iter = bootstrap_iter
        s.n_samples = None  # σ has its own n_samples in meta

    specs["DITCH_MIN_DEPTH"].source = depth_source
    specs["DITCH_MIN_DEPTH"].n_samples = depth_n
    specs["DITCH_MIN_PROM"].source = depth_source
    specs["DITCH_MIN_PROM"].n_samples = depth_n
    specs["SHAPE_OUTER_RISE_MIN"].source = depth_source
    specs["SHAPE_OUTER_RISE_MIN"].n_samples = depth_n

    # WATER_DENSITY_RATIO and WATER_INTENSITY_MAX stay literature unless
    # we have prior data — handled in bootstrap.

    # DITCH_MIN_AREA derives from MIN_WIDTH × MIN_DEPTH (computed AFTER C)


# ════════════════════════════════════════════════════════════════════════
# STAGE 3 — Cat C (geometric) thresholds — bootstrap if prior data
# ════════════════════════════════════════════════════════════════════════
def compute_cat_c(specs: dict[str, ThresholdSpec],
                  prior_df: Optional[pd.DataFrame],
                  bootstrap_iter: int,
                  false_depth_threshold_m: float = float("nan"),
                  flat_slope_threshold_deg: float = float("nan"),
                  flat_slope_n: int = 0) -> dict:
    """Bootstrap geometric thresholds from confirmed detections."""
    diagnostics = {}
    if prior_df is None or bootstrap_iter == 0:
        # Cold start: keep literature defaults
        for name in ("DITCH_MIN_WIDTH", "DITCH_MAX_WIDTH",
                     "TOP_BANK_MIN_WIDTH", "TOP_BANK_MAX_WIDTH",
                     "TOP_BANK_SEARCH_MAX_M", "TOP_BANK_RECOVERY_FRAC",
                     "TOP_BANK_FLAT_SLOPE_DEG",
                     "SLOPE_ANOMALY_INNER_MIN", "SLOPE_ANOMALY_INNER_MAX",
                     "SLOPE_ANOMALY_OUTER_MIN", "SLOPE_ANOMALY_OUTER_MAX",
                     "EFFECTIVE_INNER_SLOPE_MIN"):
            specs[name].source = Source.LITERATURE_FALLBACK
            specs[name].bootstrap_iter = 0
        return diagnostics

    # ── Width bootstrap ──────────────────────────────────────────────
    confirmed = prior_df[
        (prior_df.get("left_ditch_level", "") == "confirmed") |
        (prior_df.get("right_ditch_level", "") == "confirmed")
    ]
    widths = pd.concat([
        prior_df.loc[prior_df["left_ditch_level"] == "confirmed", "left_width"],
        prior_df.loc[prior_df["right_ditch_level"] == "confirmed", "right_width"],
    ]).dropna().values

    if len(widths) >= 30:
        w_p1  = float(np.percentile(widths, 1))
        w_p99 = float(np.percentile(widths, 99))
        top_w_min = float(max(WIDTH_LO_FLOOR, w_p1 - 0.20))
        top_w_max = float(min(8.0, w_p99 + 0.50))
        specs["DITCH_MIN_WIDTH"].value = float(max(WIDTH_LO_FLOOR, w_p1 - 0.05))
        specs["DITCH_MIN_WIDTH"].source = Source.BOOTSTRAP
        specs["DITCH_MIN_WIDTH"].n_samples = int(len(widths))
        specs["DITCH_MIN_WIDTH"].bootstrap_iter = bootstrap_iter
        specs["DITCH_MAX_WIDTH"].value = float(min(WIDTH_HI_CEIL, w_p99 + 0.5))
        specs["DITCH_MAX_WIDTH"].source = Source.BOOTSTRAP
        specs["DITCH_MAX_WIDTH"].n_samples = int(len(widths))
        specs["DITCH_MAX_WIDTH"].bootstrap_iter = bootstrap_iter
        specs["TOP_BANK_MIN_WIDTH"].value = top_w_min
        specs["TOP_BANK_MIN_WIDTH"].source = Source.BOOTSTRAP
        specs["TOP_BANK_MIN_WIDTH"].n_samples = int(len(widths))
        specs["TOP_BANK_MIN_WIDTH"].bootstrap_iter = bootstrap_iter
        specs["TOP_BANK_MAX_WIDTH"].value = top_w_max
        specs["TOP_BANK_MAX_WIDTH"].source = Source.BOOTSTRAP
        specs["TOP_BANK_MAX_WIDTH"].n_samples = int(len(widths))
        specs["TOP_BANK_MAX_WIDTH"].bootstrap_iter = bootstrap_iter
        specs["TOP_BANK_SEARCH_MAX_M"].value = float(np.clip(0.5 * top_w_max + 0.25, 1.5, 4.0))
        specs["TOP_BANK_SEARCH_MAX_M"].source = Source.DERIVED
        specs["TOP_BANK_SEARCH_MAX_M"].n_samples = int(len(widths))
        specs["TOP_BANK_SEARCH_MAX_M"].bootstrap_iter = bootstrap_iter
        diagnostics["widths"] = widths
        diagnostics["width_p1"]  = w_p1
        diagnostics["width_p99"] = w_p99
        diagnostics["top_width_min"] = top_w_min
        diagnostics["top_width_max"] = top_w_max

    # ── Top-of-bank recovery / flatness rules ───────────────────────
    # Recovery fraction: if a remaining unrecovered height is no larger
    # than the tile's false-depression scale, it is not meaningful to keep
    # searching for a higher bank top.  This turns the 0.80 implementation
    # constant into a data-derived ratio tied to observed ditch depth.
    depths = pd.concat([
        prior_df.loc[prior_df["left_ditch_level"] == "confirmed", "left_depth"],
        prior_df.loc[prior_df["right_ditch_level"] == "confirmed", "right_depth"],
    ]).dropna().values
    depths = depths[np.isfinite(depths) & (depths > 0)]
    if len(depths) >= 30 and np.isfinite(false_depth_threshold_m):
        med_depth = float(np.median(depths))
        recovery_frac = float(np.clip(
            1.0 - false_depth_threshold_m / max(med_depth, 1e-6),
            0.70,
            0.90,
        ))
        specs["TOP_BANK_RECOVERY_FRAC"].value = recovery_frac
        specs["TOP_BANK_RECOVERY_FRAC"].source = Source.DERIVED
        specs["TOP_BANK_RECOVERY_FRAC"].n_samples = int(len(depths))
        specs["TOP_BANK_RECOVERY_FRAC"].bootstrap_iter = bootstrap_iter
        diagnostics["top_bank_recovery_depths"] = depths
        diagnostics["top_bank_recovery_median_depth"] = med_depth

    # Flat-slope angle: measured directly from the flattest side profiles
    # sampled in Stage 1.  P90 keeps the criterion permissive enough for
    # rough ballast/vegetated terrain while still being tile-specific.
    if np.isfinite(flat_slope_threshold_deg):
        specs["TOP_BANK_FLAT_SLOPE_DEG"].value = float(flat_slope_threshold_deg)
        specs["TOP_BANK_FLAT_SLOPE_DEG"].source = Source.BOOTSTRAP
        specs["TOP_BANK_FLAT_SLOPE_DEG"].n_samples = int(flat_slope_n)
        specs["TOP_BANK_FLAT_SLOPE_DEG"].bootstrap_iter = bootstrap_iter
        diagnostics["top_bank_flat_slope_threshold_deg"] = float(flat_slope_threshold_deg)

    # ── Inner / outer / effective slope bootstrap ────────────────────
    inner = pd.concat([
        prior_df.loc[prior_df["left_ditch_level"] == "confirmed",  "left_inner_slope_deg"],
        prior_df.loc[prior_df["right_ditch_level"] == "confirmed", "right_inner_slope_deg"],
    ]).dropna().values
    outer = pd.concat([
        prior_df.loc[prior_df["left_ditch_level"] == "confirmed",  "left_outer_slope_deg"],
        prior_df.loc[prior_df["right_ditch_level"] == "confirmed", "right_outer_slope_deg"],
    ]).dropna().values

    if len(inner) >= 30:
        specs["SLOPE_ANOMALY_INNER_MIN"].value = float(max(3.0, np.percentile(inner, 5)))
        specs["SLOPE_ANOMALY_INNER_MIN"].source = Source.BOOTSTRAP
        specs["SLOPE_ANOMALY_INNER_MIN"].n_samples = int(len(inner))
        specs["SLOPE_ANOMALY_INNER_MIN"].bootstrap_iter = bootstrap_iter
        specs["SLOPE_ANOMALY_INNER_MAX"].value = float(min(75.0, np.percentile(inner, 95)))
        specs["SLOPE_ANOMALY_INNER_MAX"].source = Source.BOOTSTRAP
        specs["SLOPE_ANOMALY_INNER_MAX"].n_samples = int(len(inner))
        specs["SLOPE_ANOMALY_INNER_MAX"].bootstrap_iter = bootstrap_iter
        diagnostics["inner_slopes"] = inner

    if len(outer) >= 30:
        specs["SLOPE_ANOMALY_OUTER_MIN"].value = float(max(3.0, np.percentile(outer, 5)))
        specs["SLOPE_ANOMALY_OUTER_MIN"].source = Source.BOOTSTRAP
        specs["SLOPE_ANOMALY_OUTER_MIN"].n_samples = int(len(outer))
        specs["SLOPE_ANOMALY_OUTER_MIN"].bootstrap_iter = bootstrap_iter
        specs["SLOPE_ANOMALY_OUTER_MAX"].value = float(min(75.0, np.percentile(outer, 95)))
        specs["SLOPE_ANOMALY_OUTER_MAX"].source = Source.BOOTSTRAP
        specs["SLOPE_ANOMALY_OUTER_MAX"].n_samples = int(len(outer))
        specs["SLOPE_ANOMALY_OUTER_MAX"].bootstrap_iter = bootstrap_iter
        diagnostics["outer_slopes"] = outer

    # Effective inner slope only meaningful for far-track ditches
    far = prior_df[
        ((prior_df["left_ditch_level"]  == "confirmed") &
         (prior_df["left_bottom_x"]  > specs["EFFECTIVE_SLOPE_MIN_BOTTOM_X"].value)) |
        ((prior_df["right_ditch_level"] == "confirmed") &
         (prior_df["right_bottom_x"] > specs["EFFECTIVE_SLOPE_MIN_BOTTOM_X"].value))
    ]
    eff_inners = pd.concat([
        far.loc[far["left_ditch_level"]  == "confirmed", "left_inner_slope_deg"],
        far.loc[far["right_ditch_level"] == "confirmed", "right_inner_slope_deg"],
    ]).dropna().values
    if len(eff_inners) >= 20:
        specs["EFFECTIVE_INNER_SLOPE_MIN"].value = float(max(5.0, np.percentile(eff_inners, 10)))
        specs["EFFECTIVE_INNER_SLOPE_MIN"].source = Source.BOOTSTRAP
        specs["EFFECTIVE_INNER_SLOPE_MIN"].n_samples = int(len(eff_inners))
        specs["EFFECTIVE_INNER_SLOPE_MIN"].bootstrap_iter = bootstrap_iter
        diagnostics["eff_inner_slopes"] = eff_inners

    return diagnostics


# ════════════════════════════════════════════════════════════════════════
# STAGE 4 — Cat D (classification) thresholds — bootstrap shape/content
# ════════════════════════════════════════════════════════════════════════
def compute_cat_d(specs: dict[str, ThresholdSpec],
                  prior_df: Optional[pd.DataFrame],
                  bootstrap_iter: int) -> dict:
    diagnostics = {}
    if prior_df is None or bootstrap_iter == 0:
        for name in ("SHAPE_FLATNESS_U_MIN", "SHAPE_FLATNESS_FLAT_MIN",
                     "SHAPE_DEPTH_DEFICIT_MIN",
                     "CONTENT_DVCI_PARTIAL", "CONTENT_DVCI_DENSE"):
            specs[name].source = Source.LITERATURE_FALLBACK
            specs[name].bootstrap_iter = 0
        return diagnostics

    flatness = pd.concat([
        prior_df.loc[prior_df["left_ditch_level"] == "confirmed",  "left_flatness"],
        prior_df.loc[prior_df["right_ditch_level"] == "confirmed", "right_flatness"],
    ]).dropna().values

    if len(flatness) >= 30:
        specs["SHAPE_FLATNESS_U_MIN"].value     = float(np.percentile(flatness, 25))
        specs["SHAPE_FLATNESS_FLAT_MIN"].value  = float(np.percentile(flatness, 75))
        for n in ("SHAPE_FLATNESS_U_MIN", "SHAPE_FLATNESS_FLAT_MIN"):
            specs[n].source = Source.BOOTSTRAP
            specs[n].n_samples = int(len(flatness))
            specs[n].bootstrap_iter = bootstrap_iter
        diagnostics["flatness"] = flatness

    deficits_all = pd.concat([
        prior_df.loc[prior_df["left_ditch_level"] == "confirmed",  "left_depth_deficit"],
        prior_df.loc[prior_df["right_ditch_level"] == "confirmed", "right_depth_deficit"],
    ]).dropna().values
    # Only positive deficits represent silting candidates (negative = ditch is
    # actually deeper than ideal-V model, common for shallow-slope wide ditches).
    deficits = deficits_all[deficits_all > 0]
    if len(deficits) >= 20:
        specs["SHAPE_DEPTH_DEFICIT_MIN"].value = float(np.percentile(deficits, 75))
        specs["SHAPE_DEPTH_DEFICIT_MIN"].source = Source.BOOTSTRAP
        specs["SHAPE_DEPTH_DEFICIT_MIN"].n_samples = int(len(deficits))
        specs["SHAPE_DEPTH_DEFICIT_MIN"].bootstrap_iter = bootstrap_iter
        diagnostics["deficits"] = deficits

    dvci = pd.concat([
        prior_df.loc[prior_df["left_ditch_level"] == "confirmed",  "left_dvci"],
        prior_df.loc[prior_df["right_ditch_level"] == "confirmed", "right_dvci"],
    ]).dropna().values
    if len(dvci) >= 30:
        specs["CONTENT_DVCI_PARTIAL"].value = float(np.percentile(dvci, 50))
        specs["CONTENT_DVCI_DENSE"].value   = float(np.percentile(dvci, 85))
        for n in ("CONTENT_DVCI_PARTIAL", "CONTENT_DVCI_DENSE"):
            specs[n].source = Source.BOOTSTRAP
            specs[n].n_samples = int(len(dvci))
            specs[n].bootstrap_iter = bootstrap_iter
        diagnostics["dvci"] = dvci

    return diagnostics


# ════════════════════════════════════════════════════════════════════════
# STAGE 5 — Cat E (procedural) — derive from B/C, bootstrap continuity
# ════════════════════════════════════════════════════════════════════════
def compute_cat_e(specs: dict[str, ThresholdSpec],
                  prior_df: Optional[pd.DataFrame],
                  bootstrap_iter: int) -> dict:
    """Derive gap thresholds from Cat B; bootstrap POSITION_OUTLIER_TOL_M."""
    diagnostics = {}

    # Always-derived thresholds (from Cat B/C)
    specs["GAP_RELAXED_MIN_DEPTH"].value = float(0.5 * specs["DITCH_MIN_DEPTH"].value)
    specs["GAP_RELAXED_MIN_DEPTH"].source = Source.DERIVED
    specs["GAP_RELAXED_MIN_DEPTH"].bootstrap_iter = bootstrap_iter
    specs["GAP_RELAXED_MIN_PROM"].value  = float(0.5 * specs["DITCH_MIN_PROM"].value)
    specs["GAP_RELAXED_MIN_PROM"].source = Source.DERIVED
    specs["GAP_RELAXED_MIN_PROM"].bootstrap_iter = bootstrap_iter

    # POSITION_OUTLIER_TOL_M from observed lateral scatter
    if prior_df is not None and bootstrap_iter >= 1:
        delta_pos = []
        for side in ("left", "right"):
            col = f"{side}_bottom_x"
            lvl = f"{side}_ditch_level"
            sub = prior_df[prior_df[lvl] == "confirmed"][col].values
            if len(sub) >= 10:
                # rolling median of ±3 neighbours
                win = specs["POSITION_OUTLIER_WIN_HALF"].value
                roll_med = pd.Series(sub).rolling(
                    window=2*int(win)+1, center=True, min_periods=2
                ).median().values
                d = np.abs(sub - roll_med)
                delta_pos.append(d[~np.isnan(d)])
        if delta_pos:
            delta_pos = np.concatenate(delta_pos)
            if len(delta_pos) >= 30:
                p95 = float(np.percentile(delta_pos, 95))
                # clamp to engineering envelope
                tol = float(np.clip(p95, 0.3, 1.5))
                specs["POSITION_OUTLIER_TOL_M"].value = tol
                specs["POSITION_OUTLIER_TOL_M"].source = Source.BOOTSTRAP
                specs["POSITION_OUTLIER_TOL_M"].n_samples = int(len(delta_pos))
                specs["POSITION_OUTLIER_TOL_M"].bootstrap_iter = bootstrap_iter
                diagnostics["delta_pos"] = delta_pos

    # GAP_POSITION_TOLERANCE = max(3.0, 4 × POSITION_OUTLIER_TOL_M)
    pot = specs["POSITION_OUTLIER_TOL_M"].value
    specs["GAP_POSITION_TOLERANCE"].value = float(max(3.0, 4.0 * pot))
    specs["GAP_POSITION_TOLERANCE"].source = Source.DERIVED
    specs["GAP_POSITION_TOLERANCE"].bootstrap_iter = bootstrap_iter

    # DITCH_MIN_AREA from B + C
    specs["DITCH_MIN_AREA"].value = float(max(
        0.5 * specs["DITCH_MIN_WIDTH"].value * specs["DITCH_MIN_DEPTH"].value,
        0.005,
    ))
    specs["DITCH_MIN_AREA"].source = Source.DERIVED
    specs["DITCH_MIN_AREA"].bootstrap_iter = bootstrap_iter

    # DEPTH_VALID_MIN = DITCH_MIN_DEPTH (kept in sync by design)
    specs["DEPTH_VALID_MIN"].value = float(specs["DITCH_MIN_DEPTH"].value)
    specs["DEPTH_VALID_MIN"].source = Source.DERIVED
    specs["DEPTH_VALID_MIN"].bootstrap_iter = bootstrap_iter

    # Detection slope gates are no longer independent fixed thresholds.
    # They are broad sanity envelopes derived from the adaptive slope anomaly
    # bands.  This lets abnormal slopes survive detection so they can still be
    # labelled SLOPE_ISSUE, while removing the redundant hard-coded slope set.
    derived_slope_rules = {
        "DITCH_INNER_SLOPE_MIN_DEG": max(3.0, specs["SLOPE_ANOMALY_INNER_MIN"].value - 3.0),
        "DITCH_INNER_SLOPE_MAX_DEG": min(80.0, specs["SLOPE_ANOMALY_INNER_MAX"].value + 30.0),
        "DITCH_OUTER_SLOPE_MIN_DEG": max(3.0, specs["SLOPE_ANOMALY_OUTER_MIN"].value - 3.0),
        "DITCH_OUTER_SLOPE_MAX_DEG": min(80.0, specs["SLOPE_ANOMALY_OUTER_MAX"].value + 30.0),
    }
    sample_source = {
        "DITCH_INNER_SLOPE_MIN_DEG": "SLOPE_ANOMALY_INNER_MIN",
        "DITCH_INNER_SLOPE_MAX_DEG": "SLOPE_ANOMALY_INNER_MAX",
        "DITCH_OUTER_SLOPE_MIN_DEG": "SLOPE_ANOMALY_OUTER_MIN",
        "DITCH_OUTER_SLOPE_MAX_DEG": "SLOPE_ANOMALY_OUTER_MAX",
    }
    for name, value in derived_slope_rules.items():
        specs[name].value = float(value)
        specs[name].source = Source.DERIVED
        specs[name].bootstrap_iter = bootstrap_iter
        specs[name].n_samples = specs[sample_source[name]].n_samples

    return diagnostics


# ════════════════════════════════════════════════════════════════════════
# STAGE 6 — Cat A / F / G — fixed (just confirms literature defaults)
# ════════════════════════════════════════════════════════════════════════
def confirm_fixed_categories(specs: dict[str, ThresholdSpec]) -> None:
    """Cat A, F, G are set at registry-creation time.  Just sanity-check
    that none of them got accidentally NaN/None."""
    for cat in (Category.REGULATORY, Category.SEARCH, Category.SANITY):
        for name, s in by_category(cat).items():
            if name in specs and specs[name].value is None:
                # fall back to registry default
                specs[name].value = THRESHOLD_REGISTRY[name].value


# ════════════════════════════════════════════════════════════════════════
# Convergence check
# ════════════════════════════════════════════════════════════════════════
def check_convergence(specs_new: dict[str, ThresholdSpec],
                      prev_path: str,
                      tol: float = 0.05) -> tuple[bool, list[tuple[str, float, float, float]]]:
    """Compare new vs previous, return (converged?, change_table).

    change_table rows: (name, prev_value, new_value, rel_change)
    """
    if not os.path.exists(prev_path):
        return False, []

    try:
        prev_specs, _ = load_registry(prev_path)
    except Exception:
        return False, []

    rows = []
    max_change = 0.0
    for name, ns in specs_new.items():
        if name not in prev_specs: continue
        ps = prev_specs[name]
        if ps.value in (None, 0): continue
        if ns.value in (None,):    continue
        rel = abs(ns.value - ps.value) / abs(ps.value)
        rows.append((name, float(ps.value), float(ns.value), float(rel)))
        max_change = max(max_change, rel)
    rows.sort(key=lambda r: -r[3])
    converged = max_change < tol
    return converged, rows


# ════════════════════════════════════════════════════════════════════════
# Diagnostic plot
# ════════════════════════════════════════════════════════════════════════
def make_diagnostic_plot(out_path: str,
                         centers, sigma_local, sigma_smooth, sigma_global,
                         flat_cutoff_m, n_per_window,
                         cat_c_diag: dict, cat_d_diag: dict, cat_e_diag: dict,
                         specs: dict[str, ThresholdSpec],
                         bootstrap_iter: int):
    fig = plt.figure(figsize=(18, 11))

    # (1) σ(s) along S
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(centers, sigma_local * 1000, ".", color="lightgray",
             ms=4, label="raw σ per window")
    ax1.plot(centers, sigma_smooth * 1000, "-", color="navy",
             lw=2.0, label="smoothed σ(s)")
    ax1.axhline(sigma_global * 1000, color="red", lw=1.5, ls="--",
                label=f"global median = {sigma_global*1000:.2f} mm")
    ax1.set_xlabel("S (m)")
    ax1.set_ylabel("σ_noise  (mm)")
    ax1.set_title(f"(1) σ_noise(s) — {WINDOW_LEN_M:.0f} m windows, "
                  f"{WINDOW_STRIDE_M:.0f} m stride")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(alpha=0.3)

    # (2) flat-side count per window
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(centers, n_per_window, width=WINDOW_STRIDE_M*0.8,
            color="steelblue", alpha=0.7)
    ax2.axhline(MIN_FLAT_SIDES_WIN, color="red", ls="--", lw=1.0,
                label=f"min required = {MIN_FLAT_SIDES_WIN}")
    ax2.set_xlabel("S (m)")
    ax2.set_ylabel("# flat sides used")
    ax2.set_title(f"(2) Flat-side count  (relief ≤ {flat_cutoff_m:.2f} m)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # (3) widths (if bootstrapped)
    ax3 = fig.add_subplot(2, 3, 3)
    if "widths" in cat_c_diag:
        widths = cat_c_diag["widths"]
        ax3.hist(widths, bins=30, color="seagreen", edgecolor="k")
        wlo = specs["DITCH_MIN_WIDTH"].value
        whi = specs["DITCH_MAX_WIDTH"].value
        ax3.axvline(wlo, color="purple", lw=2, ls="--", label=f"min = {wlo:.2f} m")
        ax3.axvline(whi, color="purple", lw=2, ls="--", label=f"max = {whi:.2f} m")
        ax3.set_title(f"(3) Width bootstrap  (n = {len(widths)})")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Cold start — no width bootstrap\n"
                            f"using fallback [{specs['DITCH_MIN_WIDTH'].value:.2f}, "
                            f"{specs['DITCH_MAX_WIDTH'].value:.2f}] m",
                  ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("(3) Width bootstrap (skipped)")
    ax3.set_xlabel("Detected width (m)")
    ax3.set_ylabel("count")
    ax3.grid(alpha=0.3)

    # (4) inner / outer slopes
    ax4 = fig.add_subplot(2, 3, 4)
    if "inner_slopes" in cat_c_diag:
        ax4.hist(cat_c_diag["inner_slopes"], bins=30, color="coral",
                 alpha=0.6, label=f"inner (n={len(cat_c_diag['inner_slopes'])})")
        si_lo = specs["SLOPE_ANOMALY_INNER_MIN"].value
        si_hi = specs["SLOPE_ANOMALY_INNER_MAX"].value
        ax4.axvline(si_lo, color="red", lw=2, ls="--", label=f"P5 = {si_lo:.1f}°")
        ax4.axvline(si_hi, color="red", lw=2, ls="--", label=f"P95 = {si_hi:.1f}°")
    if "outer_slopes" in cat_c_diag:
        ax4.hist(cat_c_diag["outer_slopes"], bins=30, color="skyblue",
                 alpha=0.4, label=f"outer (n={len(cat_c_diag['outer_slopes'])})")
    ax4.set_xlabel("Slope (°)")
    ax4.set_ylabel("count")
    ax4.set_title("(4) Slope distribution + anomaly bounds")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    # (5) flatness / DVCI
    ax5 = fig.add_subplot(2, 3, 5)
    if "flatness" in cat_d_diag:
        ax5.hist(cat_d_diag["flatness"], bins=20, color="goldenrod",
                 alpha=0.7, label=f"flatness (n={len(cat_d_diag['flatness'])})")
        ax5.axvline(specs["SHAPE_FLATNESS_U_MIN"].value, color="red",
                    lw=2, ls="--", label=f"U≥{specs['SHAPE_FLATNESS_U_MIN'].value:.2f}")
        ax5.axvline(specs["SHAPE_FLATNESS_FLAT_MIN"].value, color="darkred",
                    lw=2, ls="--", label=f"FLAT≥{specs['SHAPE_FLATNESS_FLAT_MIN'].value:.2f}")
    if "dvci" in cat_d_diag:
        ax5b = ax5.twinx()
        ax5b.hist(cat_d_diag["dvci"], bins=20, color="green",
                  alpha=0.3, label=f"DVCI (n={len(cat_d_diag['dvci'])})")
        ax5b.axvline(specs["CONTENT_DVCI_PARTIAL"].value, color="green",
                     lw=1.5, ls=":")
        ax5b.axvline(specs["CONTENT_DVCI_DENSE"].value, color="darkgreen",
                     lw=1.5, ls=":")
        ax5b.set_ylabel("DVCI count", color="green")
    ax5.set_xlabel("flatness (left axis) / DVCI (right axis)")
    ax5.set_ylabel("flatness count")
    ax5.set_title("(5) Shape + content distributions")
    ax5.legend(fontsize=7, loc="upper right")
    ax5.grid(alpha=0.3)

    # (6) lateral scatter
    ax6 = fig.add_subplot(2, 3, 6)
    if "delta_pos" in cat_e_diag:
        dp = cat_e_diag["delta_pos"]
        ax6.hist(dp, bins=30, color="indianred", edgecolor="k")
        tol = specs["POSITION_OUTLIER_TOL_M"].value
        ax6.axvline(tol, color="red", lw=2, ls="--",
                    label=f"P95 = {tol:.2f} m")
        ax6.set_title(f"(6) Lateral scatter |bottom_x − rolling_median| (n={len(dp)})")
    else:
        ax6.text(0.5, 0.5, "Cold start — no scatter bootstrap\n"
                            f"using fallback {specs['POSITION_OUTLIER_TOL_M'].value:.2f} m",
                  ha="center", va="center", transform=ax6.transAxes)
        ax6.set_title("(6) Lateral scatter (skipped)")
    ax6.set_xlabel("|Δposition| (m)")
    ax6.set_ylabel("count")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic: {out_path}")


# ════════════════════════════════════════════════════════════════════════
# CLI driver
# ════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Adaptive ditch threshold computer v2 (per-window σ + bootstrap).",
    )
    p.add_argument("--tile",     default="tile",
                   help="Tile id, used for default paths.")
    p.add_argument("--out-dir",  default=None,
                   help="Output directory (must contain step03/04/08 metadata).")
    p.add_argument("--laz",      default=None,
                   help="LAZ file (default = inferred from tile id).")
    p.add_argument("--ref-csv",  default=None,
                   help="Rail-reference CSV. Default prefers OUT_DIR/rail_reference_framework_step06.csv and falls back to the legacy step08 name.")
    p.add_argument("--prior-csv", default=None,
                   help="Prior ditch_metrics_final.csv for bootstrap "
                        "(default = OUT_DIR/ditch_metrics_final.csv).")
    p.add_argument("--bootstrap", action="store_true",
                   help="Enable bootstrap mode (uses --prior-csv to derive Cat C/D/E).")
    p.add_argument("--bootstrap-iter", type=int, default=None,
                   help="Override the bootstrap iteration counter (default: auto-detect from PREV).")
    p.add_argument("--force", action="store_true",
                   help="Override the locked flag in adaptive_thresholds_v2.json. "
                        "Required if the JSON has been LOCKED at a particular "
                        "iteration (single-iteration bootstrap policy).")
    return p.parse_args()


def infer_laz_path(tile_id: str) -> str:
    """Resolve the LAZ path from LIDAR_LAZ_ROOT using the tile id."""
    env_laz = os.environ.get("LIDAR_LAZ_FILE")
    if env_laz:
        return env_laz
    base_dir = os.environ.get("LIDAR_BASE", os.getcwd())
    laz_root = os.environ.get("LIDAR_LAZ_ROOT", os.path.join(base_dir, "data", "laz"))
    matches = sorted(glob.glob(os.path.join(laz_root, "**", f"*_{tile_id}.laz"), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No LAZ file matching tile {tile_id!r} under {laz_root}")
    return matches[0]


def main():
    args = parse_args()
    base_dir = os.environ.get("LIDAR_BASE", os.getcwd())
    out_dir = args.out_dir or os.environ.get("LIDAR_OUT_DIR") or os.path.join(base_dir, f"output_{args.tile}")
    os.makedirs(out_dir, exist_ok=True)
    laz_file = args.laz or infer_laz_path(args.tile)
    ref_csv = args.ref_csv or os.path.join(out_dir, "rail_reference_framework_step06.csv")
    if args.ref_csv is None and not os.path.exists(ref_csv):
        legacy_ref_csv = os.path.join(out_dir, "rail_reference_framework_step08.csv")
        if os.path.exists(legacy_ref_csv):
            ref_csv = legacy_ref_csv
    prior_csv = args.prior_csv or os.path.join(out_dir, "ditch_metrics_final.csv")

    json_path = os.path.join(out_dir, "adaptive_thresholds_v2.json")
    prev_path = os.path.join(out_dir, "adaptive_thresholds_v2.PREV.json")
    diag_path = os.path.join(out_dir, "adaptive_thresholds_v2_diagnostic.png")

    # Auto-detect bootstrap iter
    bootstrap_iter = args.bootstrap_iter
    if bootstrap_iter is None:
        if args.bootstrap and os.path.exists(json_path):
            try:
                _, prev_meta = load_registry(json_path)
                bootstrap_iter = int(prev_meta.get("bootstrap_iter", 0)) + 1
            except Exception:
                bootstrap_iter = 1
        elif args.bootstrap:
            bootstrap_iter = 1
        else:
            bootstrap_iter = 0

    # ── Locked-flag check ───────────────────────────────────────────
    if os.path.exists(json_path):
        try:
            _, _existing_meta = load_registry(json_path)
        except Exception:
            _existing_meta = {}
        if _existing_meta.get("locked", False) and not args.force:
            locking_decision = _existing_meta.get(
                "locking_decision", "(no rationale recorded)")
            print(
                "═" * 70 + "\n"
                f"  REFUSING TO RUN — adaptive_thresholds_v2.json is LOCKED.\n"
                f"  Locked at iter = {_existing_meta.get('bootstrap_iter')}\n"
                f"  Locked at UTC : {_existing_meta.get('locked_at_utc')}\n"
                f"  Reason        : {locking_decision}\n"
                "  Pass --force to override.\n"
                + "═" * 70)
            sys.exit(2)

    print("=" * 70)
    print(f"Adaptive thresholds v2  — tile {args.tile}, "
          f"{'BOOTSTRAP' if args.bootstrap else 'COLD START'} "
          f"(iter = {bootstrap_iter})")
    print("=" * 70)
    print(f"  LAZ        {laz_file}")
    print(f"  Reference  {ref_csv}")
    if args.bootstrap:
        print(f"  Prior      {prior_csv}")

    # ── Rotate previous JSON ────────────────────────────────────────
    if os.path.exists(json_path):
        shutil.copy2(json_path, prev_path)
        print(f"  Rotated previous JSON → {os.path.basename(prev_path)}")

    # ── Load LAZ + framework ────────────────────────────────────────
    print("\n[Load] LAZ + framework + (optional) prior CSV...")
    with open(os.path.join(out_dir, "step03_ST_metadata.json")) as f:
        meta = json.load(f)
    angle_rad = np.radians(float(meta["pca_track_azimuth_deg"]))
    mean_x = float(meta["mean_x_local"]); mean_y = float(meta["mean_y_local"])

    las = laspy.read(laz_file)
    x0 = float(las.header.x_min); y0 = float(las.header.y_min)
    X = (np.asarray(las.x, dtype=np.float64) - x0).astype(np.float32)
    Y = (np.asarray(las.y, dtype=np.float64) - y0).astype(np.float32)
    Z = np.asarray(las.z, dtype=np.float32); del las
    z1 = float(np.percentile(Z, 1))
    m = (Z >= z1) & (Z <= z1 + GROUND_Z_RANGE)
    X, Y, Z = X[m], Y[m], Z[m]
    Sg, Tg = to_ST(X, Y, mean_x, mean_y, angle_rad)
    o = np.argsort(Sg)
    Sg = Sg[o].astype(np.float32)
    Tg = Tg[o].astype(np.float32)
    Zg = Z[o]
    del X, Y, Z
    print(f"  Ground points: {len(Sg):,}")

    ref = pd.read_csv(ref_csv)
    t_col = "T_center_refined" if "T_center_refined" in ref.columns else "T_center"
    d_col = "dT_dS_refined"    if "dT_dS_refined"    in ref.columns else "dT_dS"
    ref = ref.sort_values("s").reset_index(drop=True)
    print(f"  Sections (framework):  {len(ref)}")

    prior_df = None
    if args.bootstrap and os.path.exists(prior_csv):
        prior_df = pd.read_csv(prior_csv)
        n_conf_l = int((prior_df.get("left_ditch_level", "")  == "confirmed").sum())
        n_conf_r = int((prior_df.get("right_ditch_level", "") == "confirmed").sum())
        print(f"  Prior detections: confirmed L={n_conf_l}, R={n_conf_r}")

    # ── Stage 1: σ_noise per window ─────────────────────────────────
    print("\n[Stage 1] σ_noise per 100 m sliding window...")
    (centers, sigma_local, sigma_smooth, sigma_global,
     n_per_window, flat_cutoff_m, n_pairs,
     false_relief_cutoff_m, false_depth_threshold_m,
     false_depth_samples, flat_slope_threshold_deg,
     flat_slope_samples_deg) = estimate_sigma_per_window(
        Sg, Tg, Zg, ref, t_col, d_col,
    )
    n_valid_windows = int(np.sum(~np.isnan(sigma_local)))
    print(f"  Windows total          {len(centers)}")
    print(f"  Windows with σ valid   {n_valid_windows}")
    print(f"  Total (relief, σ) pts  {n_pairs}")
    print(f"  Flat cutoff (P{FLAT_PERCENTILE} of relief)  {flat_cutoff_m:.3f} m")
    print(f"  False-depression flat cutoff "
          f"(P{FALSE_DEPRESSION_FLAT_PERCENTILE} of relief)  "
          f"{false_relief_cutoff_m:.3f} m")
    print(f"  False-depression samples  {len(false_depth_samples)}")
    print(f"  False-depression threshold (Q3 + 1.5 IQR)  "
          f"{false_depth_threshold_m:.4f} m")
    print(f"  Flat-slope samples  {len(flat_slope_samples_deg)}")
    print(f"  Top-bank flat-slope threshold (P90)  "
          f"{flat_slope_threshold_deg:.2f} deg")
    print(f"  σ_global (median)      {sigma_global*1000:.2f} mm")
    print(f"  σ range (smoothed)     "
          f"{np.nanmin(sigma_smooth)*1000:.2f} – {np.nanmax(sigma_smooth)*1000:.2f} mm")

    # ── Build a fresh registry ─────────────────────────────────────
    specs = fresh_registry()

    # ── Stage 2: Cat B ─────────────────────────────────────────────
    print("\n[Stage 2] Cat B — instrument-noise thresholds")
    compute_cat_b(
        specs,
        sigma_global,
        false_depth_threshold_m,
        len(false_depth_samples),
        bootstrap_iter,
    )
    for n in ("DITCH_MIN_DEPTH", "DITCH_MIN_PROM", "SHAPE_OUTER_RISE_MIN"):
        print(f"  {n:30s}  {specs[n].value:.4f}  ({specs[n].rule})")

    # ── Stage 3: Cat C ─────────────────────────────────────────────
    print("\n[Stage 3] Cat C — geometric thresholds")
    cat_c_diag = compute_cat_c(
        specs, prior_df, bootstrap_iter,
        false_depth_threshold_m=false_depth_threshold_m,
        flat_slope_threshold_deg=flat_slope_threshold_deg,
        flat_slope_n=len(flat_slope_samples_deg),
    )
    for n in ("DITCH_MIN_WIDTH", "DITCH_MAX_WIDTH",
              "TOP_BANK_MIN_WIDTH", "TOP_BANK_MAX_WIDTH",
              "TOP_BANK_SEARCH_MAX_M", "TOP_BANK_RECOVERY_FRAC",
              "TOP_BANK_FLAT_SLOPE_DEG",
              "SLOPE_ANOMALY_INNER_MIN", "SLOPE_ANOMALY_INNER_MAX",
              "SLOPE_ANOMALY_OUTER_MIN", "SLOPE_ANOMALY_OUTER_MAX",
              "EFFECTIVE_INNER_SLOPE_MIN"):
        s = specs[n]
        ns = "" if s.n_samples is None else f"  n={s.n_samples}"
        print(f"  {n:30s}  {s.value:.3f}  ({s.source.value}{ns})")

    # ── Stage 4: Cat D ─────────────────────────────────────────────
    print("\n[Stage 4] Cat D — classification thresholds")
    cat_d_diag = compute_cat_d(specs, prior_df, bootstrap_iter)
    for n in ("SHAPE_FLATNESS_U_MIN", "SHAPE_FLATNESS_FLAT_MIN",
              "SHAPE_DEPTH_DEFICIT_MIN",
              "CONTENT_DVCI_PARTIAL", "CONTENT_DVCI_DENSE"):
        s = specs[n]
        ns = "" if s.n_samples is None else f"  n={s.n_samples}"
        print(f"  {n:30s}  {s.value:.3f}  ({s.source.value}{ns})")

    # ── Stage 5: Cat E ─────────────────────────────────────────────
    print("\n[Stage 5] Cat E — procedural / derived")
    cat_e_diag = compute_cat_e(specs, prior_df, bootstrap_iter)
    for n in ("GAP_RELAXED_MIN_DEPTH", "GAP_RELAXED_MIN_PROM",
              "GAP_POSITION_TOLERANCE", "POSITION_OUTLIER_TOL_M",
              "DITCH_MIN_AREA"):
        s = specs[n]
        ns = "" if s.n_samples is None else f"  n={s.n_samples}"
        print(f"  {n:30s}  {s.value:.4f}  ({s.source.value}{ns})")

    # ── Stage 6: Cat A / F / G ─────────────────────────────────────
    confirm_fixed_categories(specs)

    # ── Convergence check ───────────────────────────────────────────
    print("\n[Convergence]")
    converged, change_table = check_convergence(specs, prev_path, tol=0.05)
    if not change_table:
        print("  (no previous run to compare against — skipped)")
    else:
        print(f"  {'name':30s}  {'prev':>10s}  {'new':>10s}  {'|Δ|/value':>10s}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}")
        for nm, pv, nv, rel in change_table[:15]:
            print(f"  {nm:30s}  {pv:>10.4f}  {nv:>10.4f}  {rel*100:>9.2f}%")
        if len(change_table) > 15:
            print(f"  ... ({len(change_table)-15} more)")
        max_change = change_table[0][3]
        print(f"  Max relative change:  {max_change*100:.2f}%  → "
              f"{'CONVERGED ✓' if converged else 'NOT YET CONVERGED'}")

    # ── Write JSON ──────────────────────────────────────────────────
    meta_out = {
        "version": "v2",
        "tile_id": args.tile,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "bootstrap_iter": bootstrap_iter,
        "mode": "bootstrap" if args.bootstrap else "cold_start",
        "sigma_global_m": sigma_global,
        "sigma_per_window": {
            "S_centers_m": centers.tolist(),
            "sigma_local_m": [None if np.isnan(v) else float(v) for v in sigma_local],
            "sigma_smooth_m": [None if np.isnan(v) else float(v) for v in sigma_smooth],
            "n_per_window": n_per_window.tolist(),
            "window_len_m": WINDOW_LEN_M,
            "stride_m": WINDOW_STRIDE_M,
        },
        "stage1_diagnostics": {
            "n_total_pairs": n_pairs,
            "flat_cutoff_relief_m": flat_cutoff_m,
            "flat_percentile": FLAT_PERCENTILE,
            "n_valid_windows": n_valid_windows,
            "false_depression_flat_percentile": FALSE_DEPRESSION_FLAT_PERCENTILE,
            "false_depression_flat_cutoff_relief_m": false_relief_cutoff_m,
            "false_depression_threshold_m": (
                None if not np.isfinite(false_depth_threshold_m)
                else float(false_depth_threshold_m)
            ),
            "false_depression_n_samples": int(len(false_depth_samples)),
            "top_bank_flat_slope_threshold_deg": (
                None if not np.isfinite(flat_slope_threshold_deg)
                else float(flat_slope_threshold_deg)
            ),
            "top_bank_flat_slope_n_samples": int(len(flat_slope_samples_deg)),
        },
        "convergence": {
            "converged": bool(converged),
            "max_rel_change": float(change_table[0][3]) if change_table else None,
        },
        "config": {
            "WINDOW_LEN_M": WINDOW_LEN_M,
            "WINDOW_STRIDE_M": WINDOW_STRIDE_M,
            "SECTIONS_PER_WIN": SECTIONS_PER_WIN,
            "FLAT_PERCENTILE": FLAT_PERCENTILE,
            "MIN_FLAT_SIDES_WIN": MIN_FLAT_SIDES_WIN,
            "FALSE_DEPRESSION_FLAT_PERCENTILE": FALSE_DEPRESSION_FLAT_PERCENTILE,
            "MIN_FALSE_DEPRESSION_SAMPLES": MIN_FALSE_DEPRESSION_SAMPLES,
            "WIDTH_LO_FLOOR": WIDTH_LO_FLOOR,
            "WIDTH_HI_CEIL": WIDTH_HI_CEIL,
        },
        "references": [
            "Pukelsheim 1994 (3-σ rule)",
            "Lehmann 2013 (3σ outlier detection)",
            "Tukey 1977 (EDA percentile bootstrap)",
            "Cui et al. 2021 (terrain-adaptive LiDAR)",
            "Yan et al. 2024 (density-based noise removal)",
            "Roelens et al. 2018 (drainage ditch from LiDAR)",
            "Cazorzi et al. 2013 (drainage network)",
            "Bailly et al. 2008 (LiDAR linear features)",
            "Sofia et al. 2011 (channel statistical thresholds)",
            "Soille 2004 (morphological closing)",
            "Knighton 1981 (channel asymmetry A1)",
            "Meyer & Neto 2008 (Excess Green index)",
            "Trafikverket TDOK 2015:0155 §10.13, Bilaga 2",
            "Rail Baltica RBDG-MAN-016 (Hydraulic Drainage Culverts)",
            "Virtanen et al. 2020 (SciPy peak_prominences)",
            "Habib 2021 (MLS lateral accuracy)",
        ],
    }
    save_registry(specs, json_path, meta_out)
    print(f"\nSaved JSON: {json_path}")

    # ── Diagnostic plot ─────────────────────────────────────────────
    make_diagnostic_plot(diag_path,
                         centers, sigma_local, sigma_smooth, sigma_global,
                         flat_cutoff_m, n_per_window,
                         cat_c_diag, cat_d_diag, cat_e_diag,
                         specs, bootstrap_iter)

    # ── Summary table fixed vs adaptive ─────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY  — fixed vs adaptive (key thresholds)")
    print("=" * 70)
    print(f"  {'name':28s}  {'fixed':>8s}  {'adaptive':>10s}  {'source':<24s}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*10}  {'-'*24}")
    for n in ("DITCH_MIN_DEPTH", "DITCH_MIN_PROM",
              "DITCH_MIN_WIDTH", "DITCH_MAX_WIDTH",
              "TOP_BANK_MIN_WIDTH", "TOP_BANK_MAX_WIDTH",
              "TOP_BANK_SEARCH_MAX_M", "TOP_BANK_RECOVERY_FRAC",
              "TOP_BANK_FLAT_SLOPE_DEG",
              "DITCH_MIN_AREA", "SHAPE_OUTER_RISE_MIN",
              "SLOPE_ANOMALY_INNER_MIN", "SLOPE_ANOMALY_INNER_MAX",
              "POSITION_OUTLIER_TOL_M",
              "GAP_RELAXED_MIN_DEPTH", "GAP_RELAXED_MIN_PROM"):
        fix = THRESHOLD_REGISTRY[n].value
        new = specs[n].value
        src = specs[n].source.value
        print(f"  {n:28s}  {fix:>8.3f}  {new:>10.4f}  {src:<24s}")
    print(f"\n  σ_global  = {sigma_global*1000:.2f} mm")
    print(f"  6 σ      = {6*sigma_global*1000:.2f} mm  ({6*sigma_global:.4f} m)")
    print(f"  12 σ     = {12*sigma_global*1000:.2f} mm ({12*sigma_global:.4f} m)")
    print()


if __name__ == "__main__":
    main()

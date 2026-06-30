"""Profile-based panorama ditch candidate detection.

This step upgrades the panorama experiment from "take the deepest point in a
search band" to a profile-shape detector.  It works only on the DA3-derived
visible-surface profiles and does not use LiDAR to choose the panorama
candidate.  LiDAR is joined at the end only for reporting comparison metrics.

The output is deliberately split into two products:

* a metric-depth path, which is strict and should only be interpreted where the
  visible surface is suitable for depth evaluation;
* a corridor-presence path, which is more tolerant and records ditch-side
  structure even when vegetation, water or a boundary surface makes the metric
  bottom depth uncertain.

The thresholds are estimated from the current tile:

* local maxima in the visible-surface depth profile are used as raw candidates;
* the raw candidate distributions define depth, prominence, width, rim and
  lateral-position scales;
* the final path is selected by dynamic programming so that the chosen ditch
  candidate is both locally ditch-like and longitudinally coherent.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, peak_prominences, peak_widths
except Exception as exc:  # pragma: no cover - this should not happen in the thesis env
    raise RuntimeError("scipy is required for profile-based ditch detection") from exc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass(frozen=True)
class Config:
    profile_stack_csv: Path = Path(os.environ.get("PANO_PROFILE_STACK_CSV", ""))
    target_station_csv: Path = Path(os.environ.get("PANO_TARGET_STATION_CSV", ""))
    out_dir: Path = Path(os.environ.get("PANO_DITCH_OUT_DIR", "output_profile_ditch_likeness"))
    fig_dir: Path = Path(os.environ.get("PANO_DITCH_FIG_DIR", "figures_profile_ditch_likeness"))
    profile_depth_col: str = "visible_surface_depth_m"
    metric_acceptance_quantile: float = 0.60
    corridor_acceptance_quantile: float = 0.50
    min_metric_boundary_margin_m: float = 0.40
    strong_monotonic_corr: float = 0.88


CFG = Config()


def ensure_dirs() -> None:
    CFG.out_dir.mkdir(parents=True, exist_ok=True)
    CFG.fig_dir.mkdir(parents=True, exist_ok=True)


def robust_scale(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1.0
    q25, q75 = np.nanpercentile(vals, [25, 75])
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 1e-9:
        return float(iqr / 1.349)
    mad = np.nanmedian(np.abs(vals - np.nanmedian(vals)))
    if np.isfinite(mad) and mad > 1e-9:
        return float(1.4826 * mad)
    spread = np.nanmax(vals) - np.nanmin(vals)
    return float(spread) if np.isfinite(spread) and spread > 1e-9 else 1.0


def percentile_rank(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    ranks = x.rank(pct=True)
    return ranks.fillna(0.0).clip(0.0, 1.0)


def triangular_score(values: pd.Series, centre: float, scale: float) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(scale) or scale <= 0:
        scale = robust_scale(x.to_numpy(dtype=float))
    score = 1.0 - (x - centre).abs() / (2.0 * scale)
    return score.fillna(0.0).clip(0.0, 1.0)


def exp_distance_score(delta: pd.Series | np.ndarray, scale: float) -> pd.Series:
    x = pd.to_numeric(pd.Series(delta), errors="coerce")
    if not np.isfinite(scale) or scale <= 0:
        scale = robust_scale(x.to_numpy(dtype=float))
    score = np.exp(-x.abs() / max(scale, 1e-9))
    return pd.Series(score).fillna(0.0).clip(0.0, 1.0)


def metric_visibility_ok(label: str) -> bool:
    return str(label) in {
        "bottom_visible_candidate",
        "near_track_bottom_visible_candidate",
    }


def candidate_source_score(source: str) -> float:
    """Prefer real profile maxima for metric depth, keep fallbacks diagnostic."""
    source = str(source)
    if source == "local_maximum":
        return 1.0
    if source == "slope_break":
        return 0.45
    if source == "profile_maximum":
        return 0.18
    return 0.25


def profile_trend_score(corr: pd.Series) -> pd.Series:
    """Penalise profiles dominated by one monotonic slope across the search span."""
    x = pd.to_numeric(corr, errors="coerce").abs()
    score = (CFG.strong_monotonic_corr - x) / CFG.strong_monotonic_corr
    return score.fillna(1.0).clip(0.05, 1.0)


def boundary_margin_score(margin_m: pd.Series) -> pd.Series:
    """Penalise candidates selected at the edge of the finite side profile."""
    x = pd.to_numeric(margin_m, errors="coerce")
    score = x / CFG.min_metric_boundary_margin_m
    return score.fillna(0.0).clip(0.0, 1.0)


def local_quality_from_visibility(label: str, metric_ready: bool, quality: float) -> float:
    if not np.isfinite(quality):
        quality = 0.0
    if bool(metric_ready):
        return float(np.clip(quality, 0.0, 1.0))
    if label == "no_clear_visible_depression":
        return float(0.35 * np.clip(quality, 0.0, 1.0))
    if label == "boundary_surface_uncertain":
        return float(0.20 * np.clip(quality, 0.0, 1.0))
    return float(0.15 * np.clip(quality, 0.0, 1.0))


def corridor_context_from_presence(presence: str, label: str) -> float:
    """Score qualitative ditch-corridor evidence without requiring metric depth.

    This is intentionally separate from ``local_quality_from_visibility``.  A
    vegetated or boundary-dominated crop may be a poor metric depth observation
    while still being strong evidence that the side-ditch corridor is visible.
    """
    presence = str(presence)
    label = str(label)
    if presence == "exposed_ditch_bottom_candidate":
        return 1.0
    if presence == "ditch_corridor_visible_metric_uncertain":
        return 0.82
    if presence == "side_depression_context_visible":
        return 0.62
    if label in {"vegetation_surface_likely", "boundary_surface_uncertain", "monotonic_slope_surface"}:
        return 0.58
    if label == "no_clear_visible_depression":
        return 0.45
    return 0.15


def finite_gradient_strength(t: np.ndarray, y: np.ndarray, centre_idx: int, support_idx: np.ndarray) -> float:
    """Local slope-change magnitude around a candidate point.

    The value is a profile-shape cue for corridor presence.  It helps retain a
    side-ditch corridor when the bottom itself is hidden and the depth profile is
    dominated by a slope break rather than by a clean local maximum.
    """
    if len(support_idx) < 5:
        return 0.0
    local_pos = np.where(support_idx == centre_idx)[0]
    if len(local_pos) == 0:
        return 0.0
    k = int(local_pos[0])
    if k <= 0 or k >= len(support_idx) - 1:
        return 0.0
    left = support_idx[max(0, k - 2) : k + 1]
    right = support_idx[k : min(len(support_idx), k + 3)]
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left_run = float(t[left[-1]] - t[left[0]])
    right_run = float(t[right[-1]] - t[right[0]])
    if abs(left_run) < 1e-9 or abs(right_run) < 1e-9:
        return 0.0
    left_slope = float((y[left[-1]] - y[left[0]]) / left_run)
    right_slope = float((y[right[-1]] - y[right[0]]) / right_run)
    return abs(right_slope - left_slope)


def local_candidate_geometry(
    t: np.ndarray,
    depth: np.ndarray,
    support_idx: np.ndarray,
    centre_idx: int,
    bin_width: float,
) -> dict:
    """Compute robust local candidate geometry without requiring a true peak."""
    local_pos = np.where(support_idx == centre_idx)[0]
    if len(local_pos) == 0:
        return {
            "prominence": 0.0,
            "width": bin_width,
            "left_idx": centre_idx,
            "right_idx": centre_idx,
        }
    k = int(local_pos[0])
    lo = max(0, k - 5)
    hi = min(len(support_idx), k + 6)
    win = support_idx[lo:hi]
    y = depth[win]
    if len(win) == 0 or not np.isfinite(y).any():
        return {
            "prominence": 0.0,
            "width": bin_width,
            "left_idx": centre_idx,
            "right_idx": centre_idx,
        }
    centre_depth = float(depth[centre_idx])
    local_low = float(np.nanmin(y))
    local_prom = max(0.0, centre_depth - local_low)
    local_prom = max(local_prom, 0.5 * (float(np.nanmax(y)) - local_low))
    half_level = centre_depth - 0.5 * local_prom
    above = np.where(y >= half_level)[0]
    if len(above) == 0:
        left_idx = right_idx = centre_idx
    else:
        left_idx = int(win[int(above[0])])
        right_idx = int(win[int(above[-1])])
    width = max(bin_width, float(abs(t[right_idx] - t[left_idx]) + bin_width))
    return {
        "prominence": float(local_prom),
        "width": float(width),
        "left_idx": left_idx,
        "right_idx": right_idx,
    }


def extract_candidates_for_profile(group: pd.DataFrame) -> list[dict]:
    group = group.sort_values("t_m").copy()
    t = pd.to_numeric(group["t_m"], errors="coerce").to_numpy(dtype=float)
    depth = pd.to_numeric(group[CFG.profile_depth_col], errors="coerce").to_numpy(dtype=float)
    n_views = pd.to_numeric(group.get("n_fused_views", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
    n_points = pd.to_numeric(group.get("n_points", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
    # Candidate generation intentionally uses the full finite side profile.
    # The primary/near-track flags are kept only as provenance labels below;
    # the detector itself learns candidate quality from the tile distribution.
    search = np.isfinite(t) & np.isfinite(depth) & (n_views > 0) & (n_points > 0)
    if search.sum() < 3:
        return []

    idx_all = np.where(search)[0]
    y = depth[idx_all]
    if len(idx_all) >= 5 and np.nanstd(y) > 1e-9:
        profile_trend_corr = float(np.corrcoef(t[idx_all], y)[0, 1])
    else:
        profile_trend_corr = 0.0
    finite_left_t = float(t[idx_all[0]])
    finite_right_t = float(t[idx_all[-1]])
    true_peaks, _ = find_peaks(y)
    peak_sources: dict[int, str] = {int(p): "local_maximum" for p in true_peaks}
    if len(true_peaks) == 0:
        fallback = int(np.nanargmax(y))
        true_peaks = np.array([fallback], dtype=int)
        peak_sources[fallback] = "profile_maximum"

    # Add a slope-break candidate in addition to strict local maxima.  This is
    # important for vegetated or boundary-dominated ditches where the visible
    # surface may not form a clean bowl but the corridor still appears as a
    # stable break in the side profile.
    if len(idx_all) >= 5:
        gradients = np.gradient(y, t[idx_all])
        curvature = np.abs(np.gradient(gradients, t[idx_all]))
        if np.isfinite(curvature).any():
            slope_break = int(np.nanargmax(curvature))
            peak_sources.setdefault(slope_break, "slope_break")

    if len(t) > 1:
        bin_width = float(np.nanmedian(np.diff(np.sort(np.unique(t[np.isfinite(t)])))))
    else:
        bin_width = 0.2

    first = group.iloc[0]
    rows: list[dict] = []
    for peak_local, source in sorted(peak_sources.items()):
        peak_idx = int(idx_all[peak_local])
        if source == "local_maximum":
            prominence_arr, left_bases, right_bases = peak_prominences(y, np.array([peak_local], dtype=int))
            width_arr, _height, _left_ips, _right_ips = peak_widths(y, np.array([peak_local], dtype=int), rel_height=0.5)
            prominence = float(prominence_arr[0])
            width = float(width_arr[0] * bin_width)
            left_idx = int(idx_all[int(left_bases[0])])
            right_idx = int(idx_all[int(right_bases[0])])
        else:
            geom = local_candidate_geometry(t, depth, idx_all, peak_idx, bin_width)
            prominence = float(geom["prominence"])
            width = float(geom["width"])
            left_idx = int(geom["left_idx"])
            right_idx = int(geom["right_idx"])
        in_primary = bool(group.iloc[peak_idx].get("in_automatic_ditch_window", False))
        in_near = bool(group.iloc[peak_idx].get("in_near_track_fallback_window", False))
        if in_primary:
            zone = "primary"
        elif in_near:
            zone = "near_track_fallback"
        else:
            zone = "profile_context"
        left_depth = float(depth[left_idx]) if np.isfinite(depth[left_idx]) else np.nan
        right_depth = float(depth[right_idx]) if np.isfinite(depth[right_idx]) else np.nan
        rim_recovery = float(prominence / max(abs(depth[peak_idx]), 1e-6))
        boundary_margin = float(min(t[peak_idx] - finite_left_t, finite_right_t - t[peak_idx]))
        rows.append(
            {
                "filename": first["filename"],
                "frame_no": int(first["frame_no"]),
                "side": first["side"],
                "S_cam": float(first["S_cam"]),
                "visibility_class": first.get("visibility_class", ""),
                "ditch_presence_class": first.get("ditch_presence_class", ""),
                "metric_exclusion_reason": first.get("metric_exclusion_reason", ""),
                "metric_evaluation_eligible_base": bool(first.get("metric_evaluation_eligible", False)),
                "combined_metric_quality_score": float(first.get("combined_metric_quality_score", np.nan)),
                "candidate_zone": zone,
                "candidate_source": source,
                "candidate_T_m": float(t[peak_idx]),
                "candidate_depth_m": float(depth[peak_idx]),
                "candidate_prominence_m": prominence,
                "candidate_width_m": width,
                "candidate_boundary_margin_m": boundary_margin,
                "profile_trend_corr": profile_trend_corr,
                "left_rim_T_m": float(t[left_idx]),
                "right_rim_T_m": float(t[right_idx]),
                "left_rim_depth_m": left_depth,
                "right_rim_depth_m": right_depth,
                "rim_recovery_ratio": rim_recovery,
                "slope_break_strength": finite_gradient_strength(t, depth, peak_idx, idx_all),
                "n_fused_views_at_candidate": int(n_views[peak_idx]) if np.isfinite(n_views[peak_idx]) else 0,
                "n_points_at_candidate": int(n_points[peak_idx]) if np.isfinite(n_points[peak_idx]) else 0,
            }
        )
    return rows


def build_raw_candidates(profile_stack: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["filename", "frame_no", "side", "S_cam"]
    for _, group in profile_stack.groupby(keys, dropna=False):
        rows.extend(extract_candidates_for_profile(group))
    return pd.DataFrame(rows)


def derive_adaptive_specs(cand: pd.DataFrame) -> dict:
    if cand.empty:
        return {}
    depth = pd.to_numeric(cand["candidate_depth_m"], errors="coerce")
    prom = pd.to_numeric(cand["candidate_prominence_m"], errors="coerce")
    width = pd.to_numeric(cand["candidate_width_m"], errors="coerce")
    rim = pd.to_numeric(cand["rim_recovery_ratio"], errors="coerce")
    t = pd.to_numeric(cand["candidate_T_m"], errors="coerce")
    views = pd.to_numeric(cand["n_fused_views_at_candidate"], errors="coerce")
    quality = pd.to_numeric(cand["combined_metric_quality_score"], errors="coerce")

    seed_core = (
        cand["candidate_source"].eq("local_maximum")
        & pd.to_numeric(cand["candidate_boundary_margin_m"], errors="coerce").ge(CFG.min_metric_boundary_margin_m)
        & pd.to_numeric(cand["profile_trend_corr"], errors="coerce").abs().le(CFG.strong_monotonic_corr)
    )
    seed = (
        depth.ge(depth.quantile(0.50))
        & prom.ge(prom.quantile(0.50))
        & rim.ge(rim.quantile(0.35))
        & quality.ge(quality.quantile(0.35))
        & seed_core
    )
    seed_cand = cand[seed].copy()
    seed_rule = "high_quality_local_maxima"
    if len(seed_cand) < max(5, int(0.15 * len(cand))):
        seed_cand = cand[seed_core].copy()
        seed_rule = "local_maxima_with_boundary_and_trend_checks"
    if len(seed_cand) < max(5, int(0.10 * len(cand))):
        seed_cand = cand[cand["candidate_source"].eq("local_maximum")].copy()
        seed_rule = "all_local_maxima"
    if len(seed_cand) < 5:
        seed_cand = cand.copy()
        seed_rule = "all_candidates_fallback"

    specs = {
        "n_raw_candidates": int(len(cand)),
        "n_seed_candidates": int(len(seed_cand)),
        "seed_rule": seed_rule,
        "depth_q50_m": float(depth.quantile(0.50)),
        "prominence_q50_m": float(prom.quantile(0.50)),
        "rim_recovery_q35": float(rim.quantile(0.35)),
        "quality_q35": float(quality.quantile(0.35)),
        "width_median_m": float(pd.to_numeric(seed_cand["candidate_width_m"], errors="coerce").median()),
        "width_scale_m": robust_scale(pd.to_numeric(seed_cand["candidate_width_m"], errors="coerce").to_numpy(dtype=float)),
        "t_median_m": float(pd.to_numeric(seed_cand["candidate_T_m"], errors="coerce").median()),
        "t_scale_m": robust_scale(pd.to_numeric(seed_cand["candidate_T_m"], errors="coerce").to_numpy(dtype=float)),
        "depth_scale_m": robust_scale(depth.to_numpy(dtype=float)),
        "metric_quality_floor": float(quality.quantile(0.35)),
        "view_support_floor": int(max(2, math.floor(pd.to_numeric(seed_cand["n_fused_views_at_candidate"], errors="coerce").quantile(0.25)))),
        "score_acceptance_q60": math.nan,
    }
    return specs


def add_seeded_lateral_corridor(cand: pd.DataFrame, target: pd.DataFrame, specs: dict) -> pd.DataFrame:
    """Add a panorama-only lateral prior learned from high-quality target rows.

    The seed rows come from the DA3 panorama pipeline itself, not from LiDAR.
    They define where the visible ditch-side depression tends to lie for each
    side, and the dynamic path tracker then uses this as a soft corridor rather
    than as a hard gate.
    """
    out = cand.copy()
    out["seed_corridor_T_m"] = np.nan
    out["seed_corridor_delta_m"] = np.nan
    out["seed_corridor_score"] = 0.0
    side_specs: dict[str, dict] = {}

    seed_base = (
        target["has_metric_reference"].eq(True)
        & pd.to_numeric(target["combined_metric_quality_score"], errors="coerce").ge(float(specs.get("metric_quality_floor", 0.0)))
        & target["visibility_class"].map(metric_visibility_ok)
        & target["visible_surface_T_m"].notna()
        & target["S_cam"].notna()
    )
    for side, cand_side in out.groupby("side"):
        seed = target[seed_base & target["side"].eq(side)].sort_values("S_cam").copy()
        idx = cand_side.index
        if len(seed) >= 2:
            s_seed = seed["S_cam"].to_numpy(dtype=float)
            t_seed = seed["visible_surface_T_m"].to_numpy(dtype=float)
            order = np.argsort(s_seed)
            s_seed = s_seed[order]
            t_seed = t_seed[order]
            s_cand = out.loc[idx, "S_cam"].to_numpy(dtype=float)
            prior_t = np.interp(s_cand, s_seed, t_seed)
            seed_step = np.diff(t_seed)
            scale = robust_scale(seed_step if len(seed_step) else t_seed)
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = robust_scale(t_seed)
            source = "metric_visible_seed_interpolation"
        else:
            prior_t = np.full(len(idx), float(specs["t_median_m"]))
            scale = float(specs["t_scale_m"])
            source = "candidate_distribution_fallback"

        delta = out.loc[idx, "candidate_T_m"].to_numpy(dtype=float) - prior_t
        out.loc[idx, "seed_corridor_T_m"] = prior_t
        out.loc[idx, "seed_corridor_delta_m"] = delta
        out.loc[idx, "seed_corridor_score"] = exp_distance_score(delta, scale).to_numpy(dtype=float)
        side_specs[str(side)] = {
            "source": source,
            "n_seed_rows": int(len(seed)),
            "corridor_scale_m": float(scale),
            "seed_T_median_m": float(seed["visible_surface_T_m"].median()) if len(seed) else float(specs["t_median_m"]),
        }
    specs["side_lateral_corridors"] = side_specs
    return out


def score_candidates(cand: pd.DataFrame, target: pd.DataFrame, specs: dict) -> pd.DataFrame:
    if cand.empty:
        return cand
    out = add_seeded_lateral_corridor(cand, target, specs)
    meta_cols = [
        "filename",
        "side",
        "has_metric_reference",
        "reference_quality_score",
        "visible_surface_metric_candidate",
    ]
    available_meta = [c for c in meta_cols if c in target.columns]
    if len(available_meta) > 2:
        meta = target[available_meta].drop_duplicates(subset=["filename", "side"], keep="first")
        out = out.merge(meta, on=["filename", "side"], how="left")
    else:
        out["has_metric_reference"] = False
        out["reference_quality_score"] = np.nan
        out["visible_surface_metric_candidate"] = False

    out["metric_visibility_ok"] = out["visibility_class"].map(metric_visibility_ok)
    out["metric_reference_ok"] = (
        out["has_metric_reference"].fillna(False).astype(bool)
        & pd.to_numeric(out["combined_metric_quality_score"], errors="coerce").ge(float(specs.get("metric_quality_floor", 0.0)))
    )
    out["depth_score"] = percentile_rank(out["candidate_depth_m"])
    out["prominence_score"] = percentile_rank(out["candidate_prominence_m"])
    out["rim_score"] = percentile_rank(out["rim_recovery_ratio"])
    out["view_score"] = percentile_rank(out["n_fused_views_at_candidate"])
    out["width_score"] = triangular_score(out["candidate_width_m"], specs["width_median_m"], specs["width_scale_m"])
    out["lateral_prior_score"] = triangular_score(out["candidate_T_m"], specs["t_median_m"], specs["t_scale_m"])
    out["candidate_source_score"] = out["candidate_source"].map(candidate_source_score).fillna(0.25).clip(0.0, 1.0)
    out["candidate_boundary_score"] = boundary_margin_score(out["candidate_boundary_margin_m"])
    out["profile_trend_score"] = profile_trend_score(out["profile_trend_corr"])
    out["visibility_quality_score"] = [
        local_quality_from_visibility(label, metric_ready, quality)
        for label, metric_ready, quality in zip(
            out["visibility_class"],
            out["metric_reference_ok"] & out["metric_visibility_ok"],
            out["combined_metric_quality_score"],
        )
    ]
    out["corridor_context_score"] = [
        corridor_context_from_presence(presence, label)
        for presence, label in zip(out["ditch_presence_class"], out["visibility_class"])
    ]
    out["slope_break_score"] = percentile_rank(out["slope_break_strength"])
    out["profile_shape_score"] = (
        0.27 * out["depth_score"]
        + 0.29 * out["prominence_score"]
        + 0.20 * out["width_score"]
        + 0.14 * out["rim_score"]
        + 0.10 * out["view_score"]
    ) * out["candidate_source_score"] * out["candidate_boundary_score"] * out["profile_trend_score"]
    out["corridor_shape_score"] = (
        0.18 * out["depth_score"]
        + 0.19 * out["prominence_score"]
        + 0.16 * out["width_score"]
        + 0.12 * out["rim_score"]
        + 0.15 * out["slope_break_score"]
        + 0.15 * out["view_score"]
        + 0.05 * out["candidate_source_score"]
    )
    out["local_ditch_likeness"] = (
        out["profile_shape_score"].clip(0.0, 1.0)
        * out["visibility_quality_score"].clip(0.0, 1.0)
        * out["seed_corridor_score"].clip(0.0, 1.0)
    )
    out["corridor_presence_score"] = (
        0.45 * out["corridor_shape_score"].clip(0.0, 1.0)
        + 0.25 * out["corridor_context_score"].clip(0.0, 1.0)
        + 0.15 * out["seed_corridor_score"].clip(0.0, 1.0)
        + 0.15 * out["lateral_prior_score"].clip(0.0, 1.0)
    )
    metric_pool = out[
        out["metric_reference_ok"].astype(bool)
        & out["metric_visibility_ok"].astype(bool)
        & out["candidate_source"].eq("local_maximum")
        & out["candidate_boundary_margin_m"].ge(CFG.min_metric_boundary_margin_m)
        & out["profile_trend_corr"].abs().le(CFG.strong_monotonic_corr)
    ]
    if len(metric_pool) >= 5:
        metric_q = float(metric_pool["local_ditch_likeness"].quantile(CFG.metric_acceptance_quantile))
        specs["metric_score_pool"] = "reference_visible_local_maxima"
        specs["metric_score_pool_size"] = int(len(metric_pool))
    else:
        metric_q = float(out["local_ditch_likeness"].quantile(CFG.metric_acceptance_quantile)) if len(out) else math.nan
        specs["metric_score_pool"] = "all_candidates_fallback"
        specs["metric_score_pool_size"] = int(len(out))
    corridor_q = float(out["corridor_presence_score"].quantile(CFG.corridor_acceptance_quantile)) if len(out) else math.nan
    specs["metric_score_acceptance_q"] = metric_q
    specs["metric_score_acceptance_quantile"] = float(CFG.metric_acceptance_quantile)
    specs["corridor_score_acceptance_q"] = corridor_q
    specs["corridor_score_acceptance_quantile"] = float(CFG.corridor_acceptance_quantile)
    out["adaptive_local_candidate"] = out["local_ditch_likeness"].ge(metric_q)
    out["adaptive_corridor_candidate"] = out["corridor_presence_score"].ge(corridor_q)
    return out


def track_side(cand_side: pd.DataFrame, specs: dict, score_col: str, path_kind: str) -> pd.DataFrame:
    if cand_side.empty:
        return pd.DataFrame()
    stations = sorted(cand_side["S_cam"].unique())
    t_scale = max(float(specs.get("t_scale_m", 1.0)), 1e-6)
    d_scale = max(float(specs.get("depth_scale_m", 1.0)), 1e-6)
    if path_kind == "metric":
        score_floor = float(specs.get("metric_score_acceptance_q", math.nan))
    else:
        score_floor = float(specs.get("corridor_score_acceptance_q", math.nan))
    if not np.isfinite(score_floor):
        q = CFG.metric_acceptance_quantile if path_kind == "metric" else CFG.corridor_acceptance_quantile
        score_floor = float(cand_side[score_col].quantile(q))

    layers: list[pd.DataFrame] = []
    for s in stations:
        layer = cand_side[cand_side["S_cam"].eq(s)].copy().reset_index(drop=True)
        if layer.empty:
            continue
        layers.append(layer)
    if not layers:
        return pd.DataFrame()

    scores: list[np.ndarray] = []
    back: list[np.ndarray] = []
    scores.append(layers[0][score_col].to_numpy(dtype=float))
    back.append(np.full(len(layers[0]), -1, dtype=int))
    for i in range(1, len(layers)):
        prev = layers[i - 1]
        cur = layers[i]
        prev_scores = scores[-1]
        cur_scores = np.full(len(cur), -np.inf, dtype=float)
        cur_back = np.full(len(cur), -1, dtype=int)
        for j, row in cur.iterrows():
            dt = np.abs(prev["candidate_T_m"].to_numpy(dtype=float) - float(row["candidate_T_m"])) / t_scale
            dd = np.abs(prev["candidate_depth_m"].to_numpy(dtype=float) - float(row["candidate_depth_m"])) / d_scale
            transition = np.exp(-dt) * np.exp(-0.5 * dd)
            value = prev_scores + transition
            k = int(np.nanargmax(value))
            cur_scores[j] = float(row[score_col]) + float(value[k])
            cur_back[j] = k
        scores.append(cur_scores)
        back.append(cur_back)

    idx = int(np.nanargmax(scores[-1]))
    chosen: list[pd.Series] = []
    for i in range(len(layers) - 1, -1, -1):
        row = layers[i].iloc[idx].copy()
        chosen.append(row)
        idx = int(back[i][idx])
        if idx < 0 and i > 0:
            idx = int(np.nanargmax(scores[i - 1]))
    chosen.reverse()
    path = pd.DataFrame(chosen)
    path["path_kind"] = path_kind
    path["path_score_column"] = score_col
    path["path_selected"] = True
    if path_kind == "metric":
        path["path_adaptive_confirmed"] = path[score_col].ge(score_floor)
        path["path_score_floor"] = score_floor
        path["path_metric_eligible"] = (
            path["path_adaptive_confirmed"]
            & path["adaptive_local_candidate"].astype(bool)
            & path["candidate_source"].eq("local_maximum")
            & path["candidate_boundary_margin_m"].ge(CFG.min_metric_boundary_margin_m)
            & path["profile_trend_corr"].abs().le(CFG.strong_monotonic_corr)
            & path["metric_reference_ok"].astype(bool)
            & path["metric_visibility_ok"].astype(bool)
            & path["n_fused_views_at_candidate"].ge(int(specs.get("view_support_floor", 2)))
        )
        path["path_corridor_confirmed"] = False
    else:
        path["path_corridor_confirmed"] = path[score_col].ge(score_floor) & path["adaptive_corridor_candidate"].astype(bool)
        path["path_score_floor"] = score_floor
        path["path_adaptive_confirmed"] = path["path_corridor_confirmed"]
        path["path_metric_eligible"] = (
            path["path_corridor_confirmed"]
            & path["adaptive_local_candidate"].astype(bool)
            & path["candidate_source"].eq("local_maximum")
            & path["candidate_boundary_margin_m"].ge(CFG.min_metric_boundary_margin_m)
            & path["profile_trend_corr"].abs().le(CFG.strong_monotonic_corr)
            & path["metric_reference_ok"].astype(bool)
            & path["metric_visibility_ok"].astype(bool)
            & path["n_fused_views_at_candidate"].ge(int(specs.get("view_support_floor", 2)))
        )
    return path


def track_paths(cand: pd.DataFrame, specs: dict, score_col: str, path_kind: str) -> pd.DataFrame:
    paths: list[pd.DataFrame] = []
    for side, sub in cand.groupby("side"):
        tracked = track_side(sub.sort_values("S_cam").copy(), specs, score_col, path_kind)
        if not tracked.empty:
            tracked["side"] = side
            paths.append(tracked)
    return pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()


def join_lidar_for_reporting(path: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    if path.empty:
        return path
    cols = ["filename", "side", "lidar_depth_m", "lidar_bottom_T_m", "lidar_priority"]
    meta = target[[c for c in cols if c in target.columns]].copy()
    out = path.merge(meta, on=["filename", "side"], how="left")
    # Robust to the independent variant where no LiDAR depth is available.
    if "lidar_depth_m" in out.columns:
        out["panorama_minus_lidar_depth_m"] = out["candidate_depth_m"] - out["lidar_depth_m"]
    else:
        out["lidar_depth_m"] = math.nan
        out["panorama_minus_lidar_depth_m"] = math.nan
    return out


def metric_summary(df: pd.DataFrame, mask: pd.Series) -> dict:
    sub = df[mask & df["panorama_minus_lidar_depth_m"].notna()].copy()
    diff = sub["panorama_minus_lidar_depth_m"].to_numpy(dtype=float)
    return {
        "n": int(len(diff)),
        "bias_m": float(np.mean(diff)) if len(diff) else math.nan,
        "rmse_m": float(np.sqrt(np.mean(diff**2))) if len(diff) else math.nan,
        "mae_m": float(np.mean(np.abs(diff))) if len(diff) else math.nan,
    }


def plot_path(path: pd.DataFrame, filename: str, title: str) -> None:
    if path.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for side, color in [("left", "#1476B8"), ("right", "#D95F02")]:
        sub = path[path["side"].eq(side)].sort_values("S_cam")
        if sub.empty:
            continue
        axes[0].plot(sub["S_cam"], sub["candidate_T_m"], "-o", ms=3, lw=1.4, color=color, label=side)
        axes[1].plot(sub["S_cam"], sub["candidate_depth_m"], "-o", ms=3, lw=1.4, color=color, label=side)
    axes[0].set_ylabel("Selected candidate T (m)")
    axes[1].set_ylabel("Visible-surface depth (m)")
    axes[1].set_xlabel("Track chainage S (m)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CFG.fig_dir / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    print("Profile-based panorama ditch-likeness detection")
    ensure_dirs()
    profiles = pd.read_csv(CFG.profile_stack_csv)
    target = pd.read_csv(CFG.target_station_csv)
    raw = build_raw_candidates(profiles)
    specs = derive_adaptive_specs(raw)
    cand = score_candidates(raw, target, specs)
    metric_path = track_paths(cand, specs, "local_ditch_likeness", "metric")
    corridor_path = track_paths(cand, specs, "corridor_presence_score", "corridor")
    metric_path = join_lidar_for_reporting(metric_path, target)
    corridor_path = join_lidar_for_reporting(corridor_path, target)
    path = pd.concat([metric_path, corridor_path], ignore_index=True) if not metric_path.empty or not corridor_path.empty else pd.DataFrame()

    raw_csv = CFG.out_dir / "visible_ditch_raw_candidates.csv"
    scored_csv = CFG.out_dir / "visible_ditch_scored_candidates.csv"
    metric_path_csv = CFG.out_dir / "visible_ditch_metric_path.csv"
    corridor_path_csv = CFG.out_dir / "visible_ditch_presence_path.csv"
    path_csv = CFG.out_dir / "visible_ditch_selected_path.csv"
    raw.to_csv(raw_csv, index=False, float_format="%.6f")
    cand.to_csv(scored_csv, index=False, float_format="%.6f")
    metric_path.to_csv(metric_path_csv, index=False, float_format="%.6f")
    corridor_path.to_csv(corridor_path_csv, index=False, float_format="%.6f")
    path.to_csv(path_csv, index=False, float_format="%.6f")
    plot_path(metric_path, "visible_ditch_metric_path.png", "Metric visible-surface ditch-depth path")
    plot_path(corridor_path, "visible_ditch_presence_path.png", "Ditch-corridor presence path")

    metric_final_mask = (
        metric_path["path_selected"].eq(True) & metric_path["path_adaptive_confirmed"].eq(True)
        if not metric_path.empty
        else pd.Series(dtype=bool)
    )
    metric_mask = metric_path["path_metric_eligible"].eq(True) if not metric_path.empty else pd.Series(dtype=bool)
    corridor_mask = (
        corridor_path["path_corridor_confirmed"].eq(True)
        if not corridor_path.empty
        else pd.Series(dtype=bool)
    )
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "adaptive_specs": specs,
        "n_raw_candidates": int(len(raw)),
        "n_scored_candidates": int(len(cand)),
        "n_metric_path_rows": int(len(metric_path)),
        "n_corridor_path_rows": int(len(corridor_path)),
        "n_combined_path_rows": int(len(path)),
        "candidate_zone_counts": cand["candidate_zone"].value_counts(dropna=False).to_dict() if not cand.empty else {},
        "candidate_source_counts": cand["candidate_source"].value_counts(dropna=False).to_dict() if not cand.empty else {},
        "metric_path_visibility_counts": metric_path["visibility_class"].value_counts(dropna=False).to_dict() if not metric_path.empty else {},
        "corridor_path_visibility_counts": corridor_path["visibility_class"].value_counts(dropna=False).to_dict() if not corridor_path.empty else {},
        "corridor_path_presence_counts": corridor_path["ditch_presence_class"].value_counts(dropna=False).to_dict() if not corridor_path.empty else {},
        "path_metrics_against_lidar": {
            "metric_path_selected": metric_summary(metric_path, metric_path["path_selected"].eq(True)) if not metric_path.empty else {},
            "metric_path_adaptive_confirmed": metric_summary(metric_path, metric_final_mask) if not metric_path.empty else {},
            "metric_path_metric_eligible": metric_summary(metric_path, metric_mask) if not metric_path.empty else {},
            "corridor_path_selected": metric_summary(corridor_path, corridor_path["path_selected"].eq(True)) if not corridor_path.empty else {},
            "corridor_path_confirmed": metric_summary(corridor_path, corridor_mask) if not corridor_path.empty else {},
        },
        "outputs": {
            "raw_candidates": str(raw_csv),
            "scored_candidates": str(scored_csv),
            "metric_path": str(metric_path_csv),
            "corridor_path": str(corridor_path_csv),
            "combined_path": str(path_csv),
            "figures": str(CFG.fig_dir),
        },
        "interpretation": (
            "Candidates are selected from panorama-derived visible-surface profiles only. "
            "The metric path is strict and should be used for visible-surface depth evaluation. "
            "The corridor path is tolerant and records ditch-side structure even when the bottom "
            "is hidden by vegetation, water or boundary surfaces. Both confirmation floors are "
            "tile-derived score quantiles, not fixed metric depth thresholds. LiDAR is joined "
            "after selection for reporting; it is not used by either path tracker."
        ),
    }
    summary_json = CFG.out_dir / "visible_ditch_candidate_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


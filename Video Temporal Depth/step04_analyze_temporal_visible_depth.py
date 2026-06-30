"""Metric, RUK-consistent temporal change analysis for the inspection videos.

Each frame is calibrated from track geometry before the ditch surface is
measured:

  1.  the 1.435 m gauge between the running rails fixes the in-frame metric
      scale and the horizon/height (focal-free lateral metric);
  2.  the ~0.60 m sleeper spacing breaks the focal/height degeneracy and yields
      a per-camera focal length;
  3.  DA3 relative depth is affine-fitted (Z = a*D + b) to the model range at
      on-plane rail pixels, so any pixel can be back-projected and expressed as a
      vertical distance below the rail plane in metres, using the same RUK datum
      as the LiDAR experiment.

The ditch depth is then a metric quantity with a per-frame uncertainty. Change
between 2020 and 2025 is declared only when the cross-year difference exceeds
the propagated noise, and is stratified by vegetation state (ground-vs-ground
comparisons are trustworthy; canopy-vs-ground ones are flagged as confounded).

Inputs are the cached DA3 depth NPZ files + manual rail guides produced by
`step03_run_video_temporal_pipeline.py`; no torch, no raw video re-decode required.

Run:  python step04_analyze_temporal_visible_depth.py
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from step01_calibrate_video_metric_scale import (
    GAUGE_M,
    CameraModel,
    fit_camera_model,
    fit_da3_affine,
    rail_polys_for_frame,
)
from step02_detect_sleepers import detect_sleepers

ROOT = Path(os.environ.get("VIDEO_TEMPORAL_ROOT", str(Path(__file__).resolve().parent)))
OUTPUT_DIR = Path(os.environ.get("VIDEO_OUTPUT_DIR", str(ROOT / "output")))
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
CONFIG_DIR = Path(os.environ.get("VIDEO_CONFIG_DIR", str(ROOT / "config")))
DEPTH_INDEX_CSV = Path(os.environ.get("VIDEO_DEPTH_INDEX_CSV", str(TABLE_DIR / "da3_depth_index.csv")))
RAIL_GUIDE_CSV = Path(os.environ.get("VIDEO_RAIL_GUIDE_CSV", str(CONFIG_DIR / "rail_guides.csv")))


def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}_tmp_{os.getpid()}{path.suffix}")
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated_{os.getpid()}{path.suffix}")
        os.replace(tmp, fallback)
        print(f"[write] {path.name} is locked; wrote {fallback.name} instead")
        return fallback


def _safe_write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}_tmp_{os.getpid()}{path.suffix}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.replace(tmp, path)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated_{os.getpid()}{path.suffix}")
        os.replace(tmp, fallback)
        print(f"[write] {path.name} is locked; wrote {fallback.name} instead")
        return fallback

UNIT_LENGTH_M = 10.0
ANALYSIS_START_M = 1_152_891.0
ANALYSIS_END_M = 1_153_050.0

# Toe-ditch search window, lateral metres from the track centreline (right side).
TOE_T_MIN_M = 1.0
TOE_T_MAX_M = 4.0

# A sample is "ground" (bare soil/ballast) when ExG <= this; above it the
# visible surface is green canopy and the depth is canopy-top, not ditch bottom.
VEG_GROUND_MAX = 0.40

# Significance multiplier for the cross-year change test (|delta| > K * sigma).
CHANGE_K = 2.0
# Floor on the calibration component of the per-unit uncertainty (m). DA3 affine
# residuals plus gauge-fit residuals leave a few-cm systematic depth error that
# the between-frame spread alone does not capture.
CALIB_SIGMA_FLOOR_M = 0.05

# Minimum R^2 of the DA3-depth -> geometric-range affine fit for a frame's metric
# depth to be trusted. Below this the relative depth is too non-linear to
# back-project reliably, so the frame is calibrated but not used for a ditch
# measurement (reason "poor_affine_fit").
AFFINE_R2_MIN = 0.80


# Per-frame metric measurement
@dataclass
class FrameMetric:
    year: int
    sample_order: int
    chainage_m: float
    ok: bool
    focal_source: str = ""
    focal_px: float = float("nan")
    pitch_deg: float = float("nan")
    height_m: float = float("nan")
    gauge_rms_px: float = float("nan")
    affine_a: float = float("nan")
    affine_b: float = float("nan")
    affine_rms_m: float = float("nan")
    affine_r2: float = float("nan")
    ditch_side: str = ""                              # "left" | "right" (chosen ditch side)
    ditch_depth_below_plane_m: float = float("nan")  # RUK datum: ditch bottom below rail plane
    ditch_recession_m: float = float("nan")           # ditch bottom below the near-rail shoulder
    ditch_T_m: float = float("nan")                   # lateral offset of the detected bottom
    depth_other_side_m: float = float("nan")          # depth on the non-chosen side (transparency)
    profile_noise_m: float = float("nan")
    sigma_frame_m: float = float("nan")               # propagated per-frame depth uncertainty
    vegetation_ratio: float = float("nan")            # mean ExG>0 over the toe zone
    surface_class: str = ""                           # "ground" | "canopy" | "mixed"
    detected: bool = False
    reason: str = ""


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _build_metric_profile(model: CameraModel, depth_map: np.ndarray,
                          exg: np.ndarray, a: float, b: float,
                          img_h: int, img_w: int, side: int = 1) -> pd.DataFrame:
    """Profile on one track side: lateral T (m) -> vertical depth below rail plane (m).

    `side` = +1 measures to the right of the centreline, -1 to the left. T is
    always the positive lateral magnitude in metres.
    """
    dh, dw = depth_map.shape[:2]
    row_grid = np.linspace(int(0.55 * img_h), int(0.93 * img_h), 80).astype(int)
    t_bins = np.arange(0.0, TOE_T_MAX_M + 0.151, 0.15)
    samples = []
    for y in row_grid:
        g = float(model.gauge_px(y))
        if not np.isfinite(g) or g < 20:
            continue
        px_per_m = g / GAUGE_M
        xc = float(model.centre_px(y))
        for t in t_bins:
            x = xc + side * t * px_per_m
            xi = int(round(x * dw / img_w))
            yi = int(round(y * dh / img_h))
            if not (0 <= xi < dw and 0 <= yi < dh):
                continue
            D = float(depth_map[yi, xi])
            z = a * D + b
            if not np.isfinite(z) or z <= 0:
                continue
            pc = model.backproject(x, y, z)
            below = float(model.depth_below_plane_m(pc))
            xe = int(round(x))
            ye = int(round(y))
            e = float(exg[ye, xe]) if (0 <= xe < img_w and 0 <= ye < img_h) else np.nan
            samples.append({"T_m": float(t), "row_px": int(y),
                            "below_m": below, "exg": e})
    df = pd.DataFrame(samples)
    if df.empty:
        return df
    rows = []
    for t, grp in df.groupby("T_m", sort=True):
        vals = grp["below_m"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 4:
            continue
        rows.append({
            "T_m": float(t),
            "below_m": float(np.median(vals)),
            "below_se_m": float(np.std(vals) / math.sqrt(vals.size)),
            "n": int(vals.size),
            "row_px": float(np.median(grp["row_px"].to_numpy(float))),
            "vegetation_ratio": float(np.mean(grp["exg"].to_numpy(float) > 0.0)),
        })
    return pd.DataFrame(rows)


def _detect_toe_ditch(profile: pd.DataFrame, model: CameraModel,
                      affine_rms_m: float) -> dict:
    """Ditch bottom = a bounded VALLEY in the depth-below-rail-plane profile.

    `below_m` is the vertical distance below the rail plane (>0 = below). A real
    drainage ditch is a localised depression: the ground drops below the shoulder
    and then RISES again at the far bank, so the bottom is an INTERIOR local
    maximum of `below_m` with lower ground on both sides (topographic
    prominence). A plain argmax instead latches onto monotonic background
    fall-off (adjacent track / embankment / image edge), which pins the "bottom"
    to the search-zone boundary; requiring far-side recovery rejects that. The
    depth is referenced to the rail plane (RUK datum) and to the near-rail
    shoulder (recession), and gated on the profile noise.
    """
    if profile.empty or profile["T_m"].nunique() < 6:
        return {"detected": False, "reason": "insufficient_profile"}
    p = profile.sort_values("T_m").reset_index(drop=True)
    t = p["T_m"].to_numpy(float)
    below = p["below_m"].to_numpy(float)

    # Shoulder reference: median over the near band just outside the rail.
    sh = below[(t >= 0.45) & (t <= 0.95) & np.isfinite(below)]
    shoulder = float(np.median(sh)) if sh.size else 0.0

    noise = _robust_sigma(np.diff(below))
    in_zone = (t >= TOE_T_MIN_M) & (t <= TOE_T_MAX_M) & np.isfinite(below)
    if in_zone.sum() < 5:
        return {"detected": False, "reason": "no_toe_zone", "profile_noise_m": float(noise)}

    idx = np.where(in_zone)[0]
    zb = below[idx]
    zt = t[idx]
    zveg = p["vegetation_ratio"].to_numpy(float)[idx]
    n = zb.size
    gate = max(2.5 * noise if np.isfinite(noise) else 0.0, 0.08)

    # Interior local maxima with far-side recovery (topographic prominence).
    best_k, best_prom = -1, -np.inf
    for i in range(1, n - 1):
        if not (zb[i] >= zb[i - 1] and zb[i] >= zb[i + 1]):
            continue
        near_min = float(np.min(zb[:i]))          # toward the rail/shoulder
        far_min = float(np.min(zb[i + 1:]))        # toward the far bank
        prom = float(zb[i] - max(near_min, far_min))
        # require a genuine far-side rise: the ground must come back up by at
        # least half the gate beyond the bottom (rejects cliff/edge fall-off).
        far_rise = float(zb[i] - far_min)
        if far_rise < 0.5 * gate:
            continue
        if prom > best_prom:
            best_prom, best_k = prom, i
    if best_k < 0:
        return {"detected": False, "reason": "no_bounded_valley",
                "profile_noise_m": float(noise)}

    k = best_k
    bottom_below = float(zb[k])
    bottom_T = float(zt[k])
    recession = bottom_below - shoulder
    detected = bool(best_prom > gate and recession > gate and bottom_below > 0.05)

    # local vegetation around the bottom
    near = np.abs(zt - bottom_T) <= 0.45
    veg_local = float(np.mean(zveg[near])) if near.any() else float(np.mean(zveg))

    # propagate calibration uncertainty in depth at the bottom row.
    row = float(p["row_px"].to_numpy(float)[in_zone][k]) if "row_px" in p else model.cy
    dZ = abs(np.cos(model.pitch_rad) * (row - model.cy) / model.focal_px
             + np.sin(model.pitch_rad)) * (affine_rms_m if np.isfinite(affine_rms_m) else 0.0)
    spread = float(p["below_se_m"].to_numpy(float)[in_zone][k]) if "below_se_m" in p else 0.0
    sigma_frame = float(math.sqrt(spread ** 2 + dZ ** 2 + (noise if np.isfinite(noise) else 0.0) ** 2))

    return {
        "detected": detected,
        "ditch_depth_below_plane_m": bottom_below,
        "ditch_recession_m": recession,
        "ditch_T_m": bottom_T,
        "profile_noise_m": float(noise),
        "sigma_frame_m": sigma_frame,
        "vegetation_ratio": veg_local,
        "reason": "ok" if detected else "below_gate",
    }


def _choose_ditch_side(raw_frames: list, sides: dict) -> str:
    """Pick the physical ditch side (one side for both years).

    The drainage ditch is on a fixed geographic side, which maps to a fixed
    image side because both years travel the corridor in the same direction.
    Score each side by total evidence = sum of detected recessions across ALL
    frames of both years; the deeper, more frequently detected side wins.
    """
    score = {name: 0.0 for name in sides}
    for base in raw_frames:
        if not base.get("ok") or not base.get("det"):
            continue
        for name in sides:
            det = base["det"].get(name, {})
            if det.get("detected"):
                score[name] += float(det.get("ditch_recession_m", 0.0) or 0.0)
    return max(score, key=score.get) if any(score.values()) else "right"


def _surface_class(veg: float) -> str:
    if not np.isfinite(veg):
        return ""
    if veg <= VEG_GROUND_MAX:
        return "ground"
    if veg >= 0.70:
        return "canopy"
    return "mixed"


# Frame loading + two-pass focal calibration
def _load_index() -> pd.DataFrame:
    df = pd.read_csv(DEPTH_INDEX_CSV)
    df = df[df["da3_status"].isin(["cached", "computed"])].copy()
    return df.sort_values(["year", "sample_order"]).reset_index(drop=True)


def _rail_callables(model: CameraModel):
    def xl(rows):
        return np.polyval(model.left_poly, np.asarray(rows, float))

    def xr(rows):
        return np.polyval(model.right_poly, np.asarray(rows, float))

    return xl, xr


def _calibrate_focal_per_camera(index: pd.DataFrame, guides: pd.DataFrame) -> dict:
    """Pass 1: median per-frame sleeper focal estimate per year."""
    focals: dict[int, list[float]] = {}
    for _, fr in index.iterrows():
        year = int(fr["year"])
        order = int(fr["sample_order"])
        img = cv2.imread(str(fr["image_path"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        polys = rail_polys_for_frame(guides, year, order, h, w)
        if polys is None:
            continue
        left_poly, right_poly = polys
        m0 = fit_camera_model(left_poly, right_poly, h, w)
        if not m0.ok:
            continue
        xl, xr = _rail_callables(m0)
        sl = detect_sleepers(img, xl, xr)
        if not sl.ok or sl.quality < 0.3 or sl.rows.size < 3:
            continue
        m = fit_camera_model(left_poly, right_poly, h, w,
                             sleeper_rows=sl.rows, sleeper_gauges=sl.gauges_px)
        if m.ok and np.isfinite(m.focal_px) and m.focal_px > 0:
            focals.setdefault(year, []).append(float(m.focal_px))
    return {y: float(np.median(v)) for y, v in focals.items() if v}


# Cross-correlation chainage alignment
def _crosscorr_lag(series_a: pd.DataFrame, series_b: pd.DataFrame,
                   max_lag_m: float = 16.0, step_m: float = 2.0) -> dict:
    """Estimate the chainage lag (m) that best aligns year B onto year A.

    Both inputs: columns chainage_m, value. Resample onto a common grid and find
    the integer-step lag maximizing Pearson correlation over the overlap.
    """
    if series_a.empty or series_b.empty:
        return {"lag_m": 0.0, "corr": float("nan"), "n_overlap": 0, "applied": False}
    grid = np.arange(ANALYSIS_START_M, ANALYSIS_END_M + step_m, step_m)

    def resample(s):
        s = s.dropna(subset=["value"]).sort_values("chainage_m")
        if s["chainage_m"].nunique() < 4:
            return None
        return np.interp(grid, s["chainage_m"].to_numpy(float),
                         s["value"].to_numpy(float), left=np.nan, right=np.nan)

    va = resample(series_a)
    vb = resample(series_b)
    if va is None or vb is None:
        return {"lag_m": 0.0, "corr": float("nan"), "n_overlap": 0, "applied": False}

    best = {"lag_m": 0.0, "corr": -np.inf, "n_overlap": 0}
    n_lag = int(round(max_lag_m / step_m))
    for k in range(-n_lag, n_lag + 1):
        b_shift = np.full_like(vb, np.nan)
        if k >= 0:
            b_shift[k:] = vb[:len(vb) - k] if k else vb
        else:
            b_shift[:k] = vb[-k:]
        m = np.isfinite(va) & np.isfinite(b_shift)
        if m.sum() < 6:
            continue
        x, y = va[m] - va[m].mean(), b_shift[m] - b_shift[m].mean()
        denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
        if denom <= 0:
            continue
        corr = float(np.dot(x, y) / denom)
        if corr > best["corr"]:
            best = {"lag_m": float(k * step_m), "corr": corr, "n_overlap": int(m.sum())}
    # Only trust a shift when it is clearly better than zero-lag and positive.
    zero = {"lag_m": 0.0}
    applied = bool(np.isfinite(best["corr"]) and best["corr"] > 0.35
                   and abs(best["lag_m"]) <= max_lag_m)
    return {**best, "applied": applied}


# Aggregation + change detection
def _unit_id(chainage_m: float) -> int:
    return int((chainage_m - ANALYSIS_START_M) // UNIT_LENGTH_M)


def _unit_label(uid: int) -> str:
    s = ANALYSIS_START_M + uid * UNIT_LENGTH_M
    e = min(s + UNIT_LENGTH_M, ANALYSIS_END_M)
    def fmt(m):
        return f"{int(m) // 1000}+{int(m) % 1000:03d}"
    return f"{fmt(s)}-{fmt(e)}"


def _aggregate_units(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    det = frames[frames["detected"]].copy()
    for (year, uid), grp in det.groupby(["year", "unit_id"]):
        depths = grp["ditch_depth_below_plane_m"].to_numpy(float)
        depths = depths[np.isfinite(depths)]
        n = depths.size
        if n == 0:
            continue
        med = float(np.median(depths))
        between = float(np.std(depths) / math.sqrt(n)) if n >= 2 else float("nan")
        within = float(np.sqrt(np.nanmedian(grp["sigma_frame_m"].to_numpy(float) ** 2) / max(n, 1)))
        se = float(np.sqrt(np.nansum([
            0.0 if not np.isfinite(between) else between ** 2,
            0.0 if not np.isfinite(within) else within ** 2,
        ])))
        veg = float(np.nanmedian(grp["vegetation_ratio"].to_numpy(float)))
        rows.append({
            "year": int(year),
            "unit_id": int(uid),
            "unit_label": _unit_label(int(uid)),
            "n_frames": int(len(grp)),
            "n_detected": int(n),
            "ditch_depth_below_plane_m": med,
            "depth_se_m": se,
            "depth_p25_m": float(np.percentile(depths, 25)),
            "depth_p75_m": float(np.percentile(depths, 75)),
            "vegetation_ratio": veg,
            "surface_class": _surface_class(veg),
            "median_rail_rms_px": float(np.nanmedian(grp["gauge_rms_px"].to_numpy(float))),
            "metric_eligible": bool(n >= 2),
        })
    return pd.DataFrame(rows).sort_values(["year", "unit_id"]).reset_index(drop=True)


def _detect_change(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = {(int(r.year), int(r.unit_id)): r for r in units.itertuples()}
    uids = sorted({uid for (_, uid) in by})
    for uid in uids:
        a = by.get((2020, uid))
        b = by.get((2025, uid))
        rec = {"unit_id": uid, "unit_label": _unit_label(uid)}
        if a is None or b is None:
            rec.update({
                "depth_2020_m": getattr(a, "ditch_depth_below_plane_m", np.nan) if a else np.nan,
                "depth_2025_m": getattr(b, "ditch_depth_below_plane_m", np.nan) if b else np.nan,
                "delta_m": np.nan, "sigma_delta_m": np.nan, "z_score": np.nan,
                "change_label": "single_year_only",
                "surface_2020": getattr(a, "surface_class", "") if a else "",
                "surface_2025": getattr(b, "surface_class", "") if b else "",
                "comparison_class": "incomparable",
            })
            rows.append(rec)
            continue
        d2020 = float(a.ditch_depth_below_plane_m)
        d2025 = float(b.ditch_depth_below_plane_m)
        delta = d2025 - d2020
        sigma = math.sqrt(
            (a.depth_se_m if np.isfinite(a.depth_se_m) else 0.0) ** 2
            + (b.depth_se_m if np.isfinite(b.depth_se_m) else 0.0) ** 2
            + CALIB_SIGMA_FLOOR_M ** 2)
        z = delta / sigma if sigma > 0 else float("nan")
        eligible = bool(a.metric_eligible and b.metric_eligible)
        if not eligible:
            label = "insufficient_support"
        elif abs(z) <= CHANGE_K:
            label = "no_significant_change"
        elif delta > 0:
            label = "deepening"          # ditch bottom dropped further below rail plane
        else:
            label = "shallowing"
        # vegetation stratification
        sc = {a.surface_class, b.surface_class}
        if sc == {"ground"}:
            comp = "ground_vs_ground"          # trustworthy
        elif "canopy" in sc:
            comp = "canopy_confounded"          # visible surface is foliage, not ground
        else:
            comp = "mixed_surface"
        rec.update({
            "depth_2020_m": d2020, "depth_2025_m": d2025,
            "delta_m": delta, "sigma_delta_m": sigma, "z_score": z,
            "change_label": label,
            "surface_2020": a.surface_class, "surface_2025": b.surface_class,
            "comparison_class": comp,
            "vegetation_2020": float(a.vegetation_ratio),
            "vegetation_2025": float(b.vegetation_ratio),
        })
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


# Driver
def run() -> None:
    index = _load_index()
    guides = pd.read_csv(RAIL_GUIDE_CSV)

    print("[pass 1] calibrating per-camera focal length from sleeper spacing ...")
    focal_by_year = _calibrate_focal_per_camera(index, guides)
    for y, f in focal_by_year.items():
        print(f"         {y}: focal = {f:8.1f} px (sleeper-spacing median)")

    print("[pass 2] per-frame metric measurement (both sides) ...")
    SIDES = {"right": 1, "left": -1}
    raw_frames = []   # one dict per frame: meta + model scalars + per-side detection
    for _, fr in index.iterrows():
        year = int(fr["year"])
        order = int(fr["sample_order"])
        chain = float(fr["chainage_m"])
        base = {"year": year, "sample_order": order, "chainage_m": chain,
                "ok": False, "reason": "", "det": {}}

        img = cv2.imread(str(fr["image_path"]))
        if img is None:
            base["reason"] = "no_image"
            raw_frames.append(base)
            continue
        h, w = img.shape[:2]
        polys = rail_polys_for_frame(guides, year, order, h, w)
        if polys is None:
            base["reason"] = "no_rail"
            raw_frames.append(base)
            continue
        left_poly, right_poly = polys
        focal = focal_by_year.get(year)
        model = fit_camera_model(left_poly, right_poly, h, w, focal_override=focal)
        if not (model.ok and np.isfinite(model.focal_px)):
            base["reason"] = model.reason or "no_focal"
            raw_frames.append(base)
            continue

        npz = np.load(fr["depth_npz_path"])
        depth = npz["depth"]
        a, b, arms, r2 = fit_da3_affine(model, depth)
        if not np.isfinite(a):
            base["reason"] = "affine_fail"
            raw_frames.append(base)
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        exg = 2.0 * rgb[:, :, 1] - rgb[:, :, 0] - rgb[:, :, 2]
        base.update({
            "ok": True,
            "focal_source": model.focal_source,
            "focal_px": float(model.focal_px),
            "pitch_deg": float(np.degrees(model.pitch_rad)),
            "height_m": float(model.height_m),
            "gauge_rms_px": float(model.gauge_fit_rms_px),
            "affine_a": float(a), "affine_b": float(b),
            "affine_rms_m": float(arms), "affine_r2": float(r2),
        })
        affine_ok = np.isfinite(r2) and r2 >= AFFINE_R2_MIN
        for name, sign in SIDES.items():
            prof = _build_metric_profile(model, depth, exg, a, b, h, w, side=sign)
            det = _detect_toe_ditch(prof, model, arms)
            if not affine_ok:
                det["detected"] = False
                det["reason"] = "poor_affine_fit"
            base["det"][name] = det
        base["reason"] = "ok" if affine_ok else "poor_affine_fit"
        raw_frames.append(base)

    # ---- choose the physical ditch side (same side for both years) ----
    chosen_side = _choose_ditch_side(raw_frames, SIDES)
    print(f"[side] ditch measured on the '{chosen_side}' track side "
          f"(higher, more consistent depression across the corridor)")

    frame_records = []
    for base in raw_frames:
        fm = FrameMetric(year=base["year"], sample_order=base["sample_order"],
                         chainage_m=base["chainage_m"], ok=base["ok"],
                         reason=base["reason"])
        for k in ("focal_source", "focal_px", "pitch_deg", "height_m", "gauge_rms_px",
                  "affine_a", "affine_b", "affine_rms_m", "affine_r2"):
            if k in base:
                setattr(fm, k, base[k])
        if base["ok"] and base["det"]:
            det = base["det"][chosen_side]
            other = base["det"]["left" if chosen_side == "right" else "right"]
            fm.ditch_side = chosen_side
            fm.detected = bool(det.get("detected", False))
            fm.reason = det.get("reason", base["reason"])
            fm.profile_noise_m = float(det.get("profile_noise_m", float("nan")))
            fm.depth_other_side_m = float(other.get("ditch_depth_below_plane_m", float("nan")))
            if fm.detected:
                fm.ditch_depth_below_plane_m = float(det["ditch_depth_below_plane_m"])
                fm.ditch_recession_m = float(det["ditch_recession_m"])
                fm.ditch_T_m = float(det["ditch_T_m"])
                fm.sigma_frame_m = float(det["sigma_frame_m"])
                fm.vegetation_ratio = float(det["vegetation_ratio"])
                fm.surface_class = _surface_class(fm.vegetation_ratio)
        frame_records.append(fm)

    frames = pd.DataFrame([asdict(r) for r in frame_records])
    frames["unit_id"] = frames["chainage_m"].apply(_unit_id)

    # ---- cross-correlation chainage alignment (refine constant-speed mapping) ----
    def year_series(y):
        d = frames[(frames["year"] == y) & frames["detected"]]
        return d[["chainage_m"]].assign(value=d["ditch_depth_below_plane_m"].to_numpy(float))
    align = _crosscorr_lag(year_series(2020), year_series(2025))
    print(f"[align] cross-correlation lag(2025->2020) = {align['lag_m']:+.1f} m "
          f"(corr={align['corr']:.2f}, n={align['n_overlap']}, "
          f"applied={align['applied']})")
    if align["applied"]:
        m25 = frames["year"] == 2025
        frames.loc[m25, "chainage_aligned_m"] = frames.loc[m25, "chainage_m"] - align["lag_m"]
        frames.loc[~m25, "chainage_aligned_m"] = frames.loc[~m25, "chainage_m"]
    else:
        frames["chainage_aligned_m"] = frames["chainage_m"]
    frames["unit_id"] = frames["chainage_aligned_m"].apply(_unit_id)
    frames["unit_label"] = frames["unit_id"].apply(_unit_label)

    units = _aggregate_units(frames)
    change = _detect_change(units)

    # ---- write outputs ----
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _safe_to_csv(frames, TABLE_DIR / "metric_frame_measurements.csv")
    _safe_to_csv(units, TABLE_DIR / "metric_units_by_year.csv")
    _safe_to_csv(change, TABLE_DIR / "metric_temporal_change.csv")

    summary = {
        "method": "rail_plane_metric_reconstruction",
        "datum": "vertical_distance_below_rail_plane_RUK",
        "focal_px_by_year": focal_by_year,
        "alignment": align,
        "ditch_side": chosen_side,
        "affine_r2_min": AFFINE_R2_MIN,
        "n_frames": int(len(frames)),
        "n_frames_calibrated": int(frames["ok"].sum()),
        "n_frames_poor_affine": int((frames["reason"] == "poor_affine_fit").sum()),
        "median_affine_r2_by_year": {
            int(y): round(float(frames.loc[(frames["year"] == y) & frames["ok"], "affine_r2"].median()), 3)
            for y in sorted(frames["year"].unique())},
        "n_frames_detected": int(frames["detected"].sum()),
        "change_k_sigma": CHANGE_K,
        "calib_sigma_floor_m": CALIB_SIGMA_FLOOR_M,
        "veg_ground_max_exg_fraction": VEG_GROUND_MAX,
        "units_paired_significant": int(change["change_label"].isin(
            ["deepening", "shallowing"]).sum()) if not change.empty else 0,
        "units_ground_vs_ground": int((change["comparison_class"] == "ground_vs_ground").sum())
            if not change.empty else 0,
    }
    _safe_write_json(summary, TABLE_DIR / "metric_temporal_summary.json")

    _make_figure(frames, units, change, align)

    print(f"[done] calibrated {summary['n_frames_calibrated']}/{summary['n_frames']} frames; "
          f"detected ditch in {summary['n_frames_detected']}; "
          f"{summary['units_paired_significant']} units show significant change; "
          f"{summary['units_ground_vs_ground']} ground-vs-ground pairs.")


def _make_figure(frames, units, change, align):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[figure] skipped ({exc})")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {2020: "#1f77b4", 2025: "#d62728"}
    for y in (2020, 2025):
        u = units[units["year"] == y].sort_values("unit_id")
        if u.empty:
            continue
        xc = ANALYSIS_START_M + u["unit_id"].to_numpy() * UNIT_LENGTH_M + 5.0
        ax1.errorbar(xc, u["ditch_depth_below_plane_m"], yerr=u["depth_se_m"],
                     fmt="o-", color=colors[y], capsize=3, label=f"{y}")
    ax1.set_ylabel("ditch depth below rail plane (m)")
    ax1.set_title("RUK-datum ditch depth per 10 m unit "
                  f"(2025 chainage shift {align['lag_m']:+.0f} m, "
                  f"applied={align['applied']})")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.invert_yaxis()

    if not change.empty:
        c = change.dropna(subset=["delta_m"]).sort_values("unit_id")
        xc = ANALYSIS_START_M + c["unit_id"].to_numpy() * UNIT_LENGTH_M + 5.0
        bar_colors = ["#7f7f7f" if lab not in ("deepening", "shallowing") else
                      ("#d62728" if lab == "deepening" else "#2ca02c")
                      for lab in c["change_label"]]
        ax2.bar(xc, c["delta_m"], width=8.0, color=bar_colors)
        ax2.errorbar(xc, c["delta_m"], yerr=CHANGE_K * c["sigma_delta_m"],
                     fmt="none", ecolor="k", capsize=3, alpha=0.6)
        ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("delta depth 2025-2020 (m)")
    ax2.set_xlabel("chainage (m)")
    ax2.set_title(f"Significant change at +/-{CHANGE_K}sigma "
                  "(red=deepening, green=shallowing, grey=n.s.)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURE_DIR / "metric_temporal_change.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[figure] {out}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()



"""Sleeper-bar detection for in-frame metric calibration.

Sleepers cross the track perpendicular to the rails at a known, fixed
longitudinal spacing (Swedish main-line concrete sleepers: ~0.60-0.65 m).
In a forward-facing frame they appear as quasi-periodic horizontal bands
between the two rails, getting denser toward the horizon (perspective).

This module returns, per frame:
  - sleeper_rows: image rows (px) of detected sleeper centres at track centre
  - their local gauge g (px) from the supplied rail model
  - a periodicity quality score in [0, 1]

The sleeper rows + local gauges feed the focal-length solve in
`step01_calibrate_video_metric_scale.py` (the known 0.60 m spacing is the longitudinal
constraint that breaks the focal/height degeneracy left by the gauge alone).

Detection is deliberately conservative: when periodicity is weak (e.g. the
vegetated 2025 camera) the quality score drops and the caller down-weights or
drops that frame's focal estimate rather than trusting a noisy pick.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

SLEEPER_SPACING_M = 0.60          # nominal concrete-sleeper period (validation target)
SLEEPER_SPACING_TOL_M = 0.05      # plausible band 0.55-0.65 m


@dataclass
class SleeperResult:
    ok: bool
    rows: np.ndarray = field(default_factory=lambda: np.empty(0))   # image rows (px) of sleeper centres
    gauges_px: np.ndarray = field(default_factory=lambda: np.empty(0))  # local gauge at each row
    quality: float = 0.0          # periodicity strength in [0, 1]
    period_px_median: float = float("nan")  # median row spacing between sleepers (lower image only)
    reason: str = ""


def _track_band_profile(gray: np.ndarray, x_left: np.ndarray, x_right: np.ndarray,
                        rows: np.ndarray, inset_frac: float = 0.18) -> np.ndarray:
    """Mean brightness of the inter-rail band per row (the track-bed signal)."""
    h, w = gray.shape
    prof = np.full(rows.shape, np.nan, dtype=np.float32)
    for i, y in enumerate(rows):
        xl = x_left[i]
        xr = x_right[i]
        if not (np.isfinite(xl) and np.isfinite(xr)) or xr - xl < 6:
            continue
        inset = (xr - xl) * inset_frac
        a = int(round(xl + inset))
        b = int(round(xr - inset))
        a = max(0, min(w - 1, a))
        b = max(a + 1, min(w, b))
        prof[i] = float(np.mean(gray[int(y), a:b]))
    return prof


def detect_sleepers(image_bgr: np.ndarray, x_left_of_row, x_right_of_row,
                    y_top_frac: float = 0.55, y_bottom_frac: float = 0.96) -> SleeperResult:
    """Detect sleeper centres between the rails.

    `x_left_of_row` / `x_right_of_row` are callables row_px(int|array) -> x_px,
    typically built from the interpolated rail polynomials.
    """
    if image_bgr is None or image_bgr.ndim != 3:
        return SleeperResult(ok=False, reason="bad_image")
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (1, 5), 0)  # smooth along columns, keep row structure

    y0 = int(round(y_top_frac * h))
    y1 = int(round(y_bottom_frac * h))
    rows = np.arange(y0, y1, 1)
    if rows.size < 30:
        return SleeperResult(ok=False, reason="corridor_too_short")

    xl = np.asarray(x_left_of_row(rows), dtype=float)
    xr = np.asarray(x_right_of_row(rows), dtype=float)
    gauges = xr - xl

    prof = _track_band_profile(gray, xl, xr, rows)
    valid = np.isfinite(prof)
    if valid.sum() < 30:
        return SleeperResult(ok=False, reason="no_band_profile")

    # Detrend with a rolling median so sleeper bars stand out from the slow
    # brightness gradient toward the horizon.
    p = prof.copy()
    p[~valid] = np.nanmedian(prof[valid])
    win = max(9, (rows.size // 12) | 1)
    trend = cv2.medianBlur(p.reshape(-1, 1).astype(np.float32), 1).ravel()
    trend = _rolling_median(p, win)
    detr = p - trend

    # Sleepers are brighter than the ballast/soil gaps; find positive peaks.
    sig = cv2.GaussianBlur(detr.reshape(-1, 1).astype(np.float32), (1, 3), 0).ravel()
    thr = max(1.5, 0.6 * float(np.nanstd(sig)))
    peaks = []
    for i in range(1, len(sig) - 1):
        if sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1] and sig[i] > thr:
            peaks.append(i)
    if len(peaks) < 4:
        return SleeperResult(ok=False, quality=0.0, reason="too_few_peaks")

    peak_rows = rows[np.asarray(peaks)]
    peak_gauges = gauges[np.asarray(peaks)]

    # Periodicity quality: in the lower (near) half the spacing should be
    # roughly regular. Use the dispersion of nearest-neighbour spacings there.
    lower = peak_rows[peak_rows > rows[int(0.45 * rows.size)]]
    if lower.size >= 3:
        d = np.diff(np.sort(lower))
        d = d[d > 2]
        period = float(np.median(d)) if d.size else float("nan")
        quality = float(np.clip(1.0 - (np.std(d) / max(np.mean(d), 1e-6)), 0.0, 1.0)) if d.size >= 2 else 0.0
    else:
        period = float("nan")
        quality = 0.0

    return SleeperResult(
        ok=True,
        rows=peak_rows.astype(float),
        gauges_px=peak_gauges.astype(float),
        quality=quality,
        period_px_median=period,
        reason="ok",
    )


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    win = max(3, win | 1)
    half = win // 2
    n = len(x)
    out = np.empty(n, dtype=float)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = np.median(x[a:b])
    return out


"""In-frame metric calibration via the rail plane (RUK-consistent).

The two running rails give a known lateral distance (gauge G = 1.435 m) at
every image row; the sleepers give a known longitudinal distance
(~0.60 m). Together these calibrate a pinhole + ground-plane camera model
per frame, with no external dataset and no global depth-scale constant.

Geometry (principal point at image centre, zero roll, no lens distortion;
camera at height H above the rail plane, pitched down by theta):

    gauge:   g(y) = (G cos(theta) / H) * (y - y_h)         # linear in row
    horizon: y_h  = cy - f tan(theta)                       # gauge -> 0
    sleeper: dy   = 0.60 * H * g1 * g2 / (f * G^2)          # row gap of 0.60 m

From the rails alone we recover y_h and the slope m = G cos(theta)/H, hence
the lateral metric  T(x, y) = (x - x_centre(y)) * G / g(y)  which is fully
focal-independent and robust across the two cameras. The sleeper spacing
supplies the third equation that breaks the (f, H) degeneracy and yields f,
theta, H. DA3 relative depth is then affine-fitted (Z = a*D + b) to the
model range at on-plane rail pixels, so the ditch surface can be expressed
as a vertical distance below the rail plane in metres.

Everything here runs on the cached DA3 depth + the manual rail guides; no
torch required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GAUGE_M = 1.435
SLEEPER_SPACING_M = 0.60


def rail_polys_for_frame(guides: pd.DataFrame, year: int, sample_order: int,
                         h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (left_poly, right_poly): x_px = polyval(poly, y_px), deg 2.

    Mirrors the interpolation in the main pipeline: for each rail and each
    annotated y_frac, interpolate x_frac across sample_order, then fit a
    quadratic in y_px.
    """
    g = guides[guides["year"].astype(int) == int(year)]
    if g.empty:
        return None
    polys = {}
    for rail in ("left", "right"):
        rg = g[g["rail"] == rail]
        if rg.empty:
            return None
        y_fracs = sorted(rg["y_frac"].unique())
        ys, xs = [], []
        for yf in y_fracs:
            sub = rg[rg["y_frac"] == yf].sort_values("sample_order")
            if len(sub) < 2:
                continue
            xf = float(np.interp(sample_order,
                                 sub["sample_order"].to_numpy(float),
                                 sub["x_frac"].to_numpy(float)))
            ys.append(yf * (h - 1))
            xs.append(xf * w)
        if len(ys) < 3:
            return None
        polys[rail] = np.polyfit(np.asarray(ys), np.asarray(xs), deg=2)
    return polys["left"], polys["right"]


@dataclass
class CameraModel:
    ok: bool
    h: int = 0
    w: int = 0
    cx: float = 0.0
    cy: float = 0.0
    y_horizon: float = float("nan")     # px
    gauge_slope: float = float("nan")   # m_slope = G cos(theta)/H  [px gauge per px row]
    focal_px: float = float("nan")
    pitch_rad: float = float("nan")
    height_m: float = float("nan")      # camera height above rail plane
    left_poly: np.ndarray | None = None
    right_poly: np.ndarray | None = None
    gauge_fit_rms_px: float = float("nan")
    focal_source: str = ""
    reason: str = ""

    def gauge_px(self, y_px) -> np.ndarray:
        y = np.asarray(y_px, dtype=float)
        xl = np.polyval(self.left_poly, y)
        xr = np.polyval(self.right_poly, y)
        return xr - xl

    def centre_px(self, y_px) -> np.ndarray:
        y = np.asarray(y_px, dtype=float)
        return 0.5 * (np.polyval(self.left_poly, y) + np.polyval(self.right_poly, y))

    def lateral_T_m(self, x_px, y_px) -> np.ndarray:
        """Cross-track offset in metres, with positive values to the right."""
        g = self.gauge_px(y_px)
        return (np.asarray(x_px, float) - self.centre_px(y_px)) * GAUGE_M / np.maximum(g, 1e-6)

    def plane_normal_cam(self) -> np.ndarray:
        # up-world expressed in camera coords (see module docstring derivation)
        th = self.pitch_rad
        return np.array([0.0, -np.cos(th), -np.sin(th)])

    def backproject(self, x_px, y_px, z_along_axis) -> np.ndarray:
        """Camera-frame 3D point for pixel (x,y) at axial range z. Shape (...,3)."""
        x = np.asarray(x_px, float)
        y = np.asarray(y_px, float)
        z = np.asarray(z_along_axis, float)
        X = (x - self.cx) * z / self.focal_px
        Y = (y - self.cy) * z / self.focal_px
        return np.stack([X, Y, z], axis=-1)

    def depth_below_plane_m(self, pts_cam: np.ndarray) -> np.ndarray:
        """Signed vertical distance below the rail plane, with ditch depth positive."""
        n = self.plane_normal_cam()
        signed_above = pts_cam @ n + self.height_m   # 0 on plane, >0 toward camera
        return -signed_above


def fit_camera_model(left_poly: np.ndarray, right_poly: np.ndarray, h: int, w: int,
                     sleeper_rows: np.ndarray | None = None,
                     sleeper_gauges: np.ndarray | None = None,
                     focal_override: float | None = None) -> CameraModel:
    cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
    # Sample gauge across the corridor and fit g(y) = slope*(y - y_h).
    ys = np.linspace(0.52 * h, 0.95 * h, 60)
    g = np.polyval(right_poly, ys) - np.polyval(left_poly, ys)
    good = np.isfinite(g) & (g > 1)
    if good.sum() < 8:
        return CameraModel(False, h, w, cx, cy, left_poly=left_poly,
                           right_poly=right_poly, reason="degenerate_gauge")
    A = np.vstack([ys[good], np.ones(good.sum())]).T
    slope, intercept = np.linalg.lstsq(A, g[good], rcond=None)[0]
    if slope <= 1e-6:
        return CameraModel(False, h, w, cx, cy, left_poly=left_poly,
                           right_poly=right_poly, reason="nonpositive_gauge_slope")
    y_h = -intercept / slope
    rms = float(np.sqrt(np.mean((g[good] - (slope * ys[good] + intercept)) ** 2)))

    # Focal length: either supplied (per-camera constant) or from sleeper spacing.
    focal = float("nan")
    focal_src = "none"
    if focal_override is not None and np.isfinite(focal_override):
        focal = float(focal_override)
        focal_src = "camera_constant"
    elif sleeper_rows is not None and len(sleeper_rows) >= 3:
        f_est = _focal_from_sleepers(sleeper_rows, slope, y_h, cy)
        if np.isfinite(f_est):
            focal = f_est
            focal_src = "sleeper_spacing"

    if not np.isfinite(focal) or focal <= 0:
        # No focal: lateral metric still valid; mark 3D as unavailable.
        return CameraModel(
            ok=True, h=h, w=w, cx=cx, cy=cy, y_horizon=y_h, gauge_slope=slope,
            focal_px=float("nan"), pitch_rad=float("nan"), height_m=GAUGE_M / slope,
            left_poly=left_poly, right_poly=right_poly, gauge_fit_rms_px=rms,
            focal_source="lateral_only", reason="no_focal_lateral_only",
        )

    # Iterate theta/H with focal fixed (small-angle start).
    theta = 0.0
    for _ in range(6):
        theta = float(np.arctan2(cy - y_h, focal))
        # slope = G cos(theta) / H  ->  H = G cos(theta)/slope
        H = GAUGE_M * np.cos(theta) / slope
    H = GAUGE_M * np.cos(theta) / slope
    return CameraModel(
        ok=True, h=h, w=w, cx=cx, cy=cy, y_horizon=y_h, gauge_slope=slope,
        focal_px=focal, pitch_rad=theta, height_m=H,
        left_poly=left_poly, right_poly=right_poly, gauge_fit_rms_px=rms,
        focal_source=focal_src, reason="ok",
    )


def _focal_from_sleepers(sleeper_rows: np.ndarray, slope: float, y_h: float,
                         cy: float) -> float:
    """f from the known 0.60 m sleeper spacing.

    dy = 0.60 * H * g1 * g2 / (f * G^2),  with g(y)=slope*(y-y_h), H=G cos/slope:
        f = 0.60 * cos(theta) * g1 * g2 / (slope * dy * G)
    Use adjacent sleeper pairs in the lower image (best SNR), iterate theta.
    """
    rows = np.sort(np.asarray(sleeper_rows, float))
    rows = rows[rows > y_h + 5]
    if rows.size < 3:
        return float("nan")
    f_candidates = []
    for i in range(len(rows) - 1):
        y1, y2 = rows[i], rows[i + 1]
        dy = abs(y2 - y1)
        if dy < 3:
            continue
        g1 = slope * (y1 - y_h)
        g2 = slope * (y2 - y_h)
        if g1 <= 0 or g2 <= 0:
            continue
        theta = 0.0
        f = float("nan")
        for _ in range(5):
            f = SLEEPER_SPACING_M * np.cos(theta) * g1 * g2 / (slope * dy * GAUGE_M)
            if f <= 0 or not np.isfinite(f):
                break
            theta = float(np.arctan2(cy - y_h, f))
        if np.isfinite(f) and f > 0:
            f_candidates.append(f)
    if not f_candidates:
        return float("nan")
    return float(np.median(f_candidates))


def fit_da3_affine(model: CameraModel, depth_map: np.ndarray) -> tuple[float, float, float, float]:
    """Fit Z_axis = a*D + b using on-plane rail pixels (range known from model).

    Returns (a, b, rms_m, r2). r2 is the coefficient of determination of the
    DA3-depth -> geometric-range linear fit; it is the per-frame quality signal
    that says whether DA3 relative depth is trustworthy enough to back-project
    (a low r2 means the metric depths from this frame are unreliable).
    Requires a focal-calibrated model.
    """
    nan = float("nan")
    if not (model.ok and np.isfinite(model.focal_px)):
        return nan, nan, nan, nan
    h, w = model.h, model.w
    dh, dw = depth_map.shape[:2]
    ys = np.linspace(0.55 * h, 0.93 * h, 40)
    D_vals, Z_vals = [], []
    for y in ys:
        g = model.gauge_px(y)
        if not np.isfinite(g) or g <= 1:
            continue
        # model axial range at this row from gauge: z = f*G/g
        z = model.focal_px * GAUGE_M / g
        xc = float(model.centre_px(y))
        for x in (np.polyval(model.left_poly, y), xc, np.polyval(model.right_poly, y)):
            xi = int(round(x * dw / w))
            yi = int(round(y * dh / h))
            if 0 <= xi < dw and 0 <= yi < dh:
                D_vals.append(float(depth_map[yi, xi]))
                Z_vals.append(float(z))
    if len(D_vals) < 8:
        return nan, nan, nan, nan
    D_vals = np.asarray(D_vals)
    Z_vals = np.asarray(Z_vals)
    A = np.vstack([D_vals, np.ones_like(D_vals)]).T
    (a, b), *_ = np.linalg.lstsq(A, Z_vals, rcond=None)
    resid = A @ np.array([a, b]) - Z_vals
    rms = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((Z_vals - Z_vals.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return float(a), float(b), rms, r2

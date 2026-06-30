"""Multi-view Cirrus panorama + Depth Anything 3 experiment.

For each target panorama and each track side, neighbouring panoramas are
reprojected into the same side-looking virtual pinhole camera.  The virtual
crops, their intrinsics, and their world-to-camera extrinsics are passed to
Depth Anything 3 as a multi-view group.

The resulting depth maps are back-projected to a metric point cloud, transformed
to the target track frame, and reduced to a visible-surface cross-section
profile.  A narrow ballast-shoulder reference band is used only to correct the
vertical datum; the ditch search band is never used for scale or datum fitting.

The method estimates visible surface depth.  If the ditch bottom is hidden by
vegetation, the output is the depth of the visible vegetation/ground surface,
not a recovered hidden bottom.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

if os.environ.get("PANO_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["PANO_HF_HOME"])

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# publication figure style (shared); figure_style.py sits next to this script
try:
    from figure_style import apply_style, savefig, panel_label, PALETTE, CM
    _PUB_STYLE = True
except Exception:
    _PUB_STYLE = False
import torch

# OpenCV is used only for deterministic image warping and filtering here.
# Keeping its own thread pool and OpenCL backend disabled avoids occasional
# binary-level conflicts with the PyTorch/CUDA stack during long DA3 runs.
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass(frozen=True)
class Config:
    pano_csv: Path = Path(os.environ.get("PANO_PANO_CSV", ""))
    sparmitt_csv: Path = Path(os.environ.get("PANO_SPARMITT_CSV", ""))
    lidar_csv: Path = Path(os.environ.get("PANO_LIDAR_CSV", ""))
    st_meta_json: Path = Path(os.environ.get("PANO_ST_META_JSON", ""))
    model_dir: Path = Path(os.environ.get("PANO_DA3_MODEL_DIR", ""))
    # Optional HuggingFace repo id (e.g. "depth-anything/DA3-LARGE-1.1").  When
    # set it OVERRIDES model_dir and is passed straight to from_pretrained, which
    # downloads + caches it on first use.  DA3-LARGE-1.1 / DA3-GIANT-1.1 are
    # drop-in any-view replacements for DA3-SMALL (identical multi-view API, no
    # other code change needed).  Note: DA3METRIC-LARGE is NOT drop-in (different
    # output convention metric = focal*net/300, monocular) and would need rework.
    model_id: str = os.environ.get("PANO_DA3_MODEL_ID", "").strip()
    out_dir: Path = Path(os.environ.get("PANO_MULTIVIEW_OUT_DIR", "output_multiview"))
    fig_dir: Path = Path(os.environ.get("PANO_MULTIVIEW_FIG_DIR", "figures_multiview"))
    camera_height_m: float = 2.30
    datum_shift_limit_m: float = 2.50
    view_size_px: int = 1024
    process_res: int = 504
    # Rail-inclusive virtual camera.  The earlier ditch-focused view used a
    # narrower 70 x 60 degree crop aimed near T=4.8 m.  This wider view anchors
    # closer to the ballast shoulder, so the rail/ballast reference remains in
    # the foreground while the ditch search band is still inside the crop.
    hfov_deg: float = 110.0
    vfov_deg: float = 82.0
    pitch_deg: float = -35.0
    target_anchor_t_m: float = 2.6
    # Multiple side-looking anchors are used to cover near-track, mid-side and
    # far-side ditch candidates.  Each anchor is inferred as a separate DA3
    # multi-view group, then all back-projected points are fused into one side
    # profile.  target_anchor_t_m is retained as a single-anchor fallback, while
    # lateral_anchor_t_values_m drives the current multi-anchor run.
    lateral_anchor_t_values_m: str = os.environ.get("PANO_LATERAL_ANCHORS_M", "2.0,4.5,7.2,10.5,13.5")
    target_anchor_depth_below_rail_m: float = 0.30
    min_view_pitch_deg: float = -45.0
    max_view_pitch_deg: float = -8.0
    preferred_half_window: int = 2
    min_half_window: int = 0
    side_t_min_m: float = float(os.environ.get("PANO_SIDE_T_MIN_M", "0.8"))
    side_t_max_m: float = float(os.environ.get("PANO_SIDE_T_MAX_M", "15.5"))
    along_half_m: float = 6.0
    bin_width_m: float = 0.20
    min_pixels_per_bin: int = 80
    primary_ditch_search_t_min_m: float = float(os.environ.get("PANO_PRIMARY_SEARCH_T_MIN_M", "2.0"))
    # Restrict the visible-surface ditch search to the realistic near-track ditch
    # corridor.  Validation against the corrected LiDAR reference (out_7309_809)
    # showed that the wider 15.5 m band (matching the LiDAR side-ditch search) let the
    # depth argmax lock onto far cutting/embankment
    # slopes 5+ m beyond the true ditch, overshooting depth (809 near-ditch
    # RMSE 0.58 -> 0.30 m, bias +0.42 -> +0.07 m when narrowed to 7 m).  Ditches
    # farther than this (e.g. tile 7302_816 at ~12 m) are not recoverable from the
    # panorama visible surface regardless of band, so 7 m is a safe default.
    primary_ditch_search_t_max_m: float = float(os.environ.get("PANO_PRIMARY_SEARCH_T_MAX_M", "7.0"))
    near_track_search_t_min_m: float = float(os.environ.get("PANO_NEAR_TRACK_SEARCH_T_MIN_M", "1.2"))
    near_track_search_t_max_m: float = float(os.environ.get("PANO_NEAR_TRACK_SEARCH_T_MAX_M", "2.0"))
    lidar_guided_half_width_m: float = 0.30
    shoulder_t_min_m: float = 1.4
    shoulder_t_max_m: float = 2.2
    shoulder_along_half_m: float = 4.0
    min_shoulder_points_per_view: int = 250
    min_reference_quality_score: float = 0.60
    rail_crop_pitch_deg: float = -22.0
    rail_expected_gauge_m: float = 1.435
    rail_gauge_tolerance_m: float = 0.15
    vegetation_exg_threshold: float = 25.0
    max_depth_m: float = 30.0
    diag_count: int = 6
    side_frame_tag: str = "lidar_axis"


CFG = Config()
# Independent variant switch.  PANO_USE_LIDAR=0 runs the panorama pipeline
# without ever reading the LiDAR ditch reference (depth/datum/sides stay
# panorama-only); the default keeps the LiDAR comparison for reporting.
USE_LIDAR = os.environ.get("PANO_USE_LIDAR", "1").strip().lower() not in ("0", "false", "no", "")
LIDAR_ST_AXIS_EN: np.ndarray | None = None
LOCAL_FRAME_FLIP_COUNT = 0


def lateral_anchor_values() -> list[float]:
    values: list[float] = []
    for item in str(CFG.lateral_anchor_t_values_m).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise ValueError(f"Invalid PANO_LATERAL_ANCHORS_M value: {item!r}") from exc
    if not values:
        return [float(CFG.target_anchor_t_m)]
    lo = float(CFG.side_t_min_m)
    hi = float(CFG.side_t_max_m)
    return sorted({float(np.clip(v, lo, hi)) for v in values})


def anchor_tag(anchor_t_m: float) -> str:
    return f"t{anchor_t_m:.1f}".replace(".", "p").replace("-", "m")


def ensure_dirs() -> dict[str, Path]:
    paths = {
        "crops": CFG.out_dir / "crops",
        "rail_crops": CFG.out_dir / "rail_crops",
        "depths": CFG.out_dir / "depths",
        "profiles": CFG.out_dir / "profiles",
        "figures": CFG.fig_dir,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def wrap_angle_deg(angle: float | np.ndarray):
    return (angle + 360.0) % 360.0


def virtual_intrinsics(width: int, height: int) -> np.ndarray:
    fx = (width * 0.5) / math.tan(math.radians(CFG.hfov_deg) * 0.5)
    fy = (height * 0.5) / math.tan(math.radians(CFG.vfov_deg) * 0.5)
    return np.array(
        [[fx, 0.0, (width - 1) * 0.5], [0.0, fy, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def camera_basis(bearing_deg: float, pitch_deg: float):
    """Return camera basis in world ENU: right, down, forward."""
    b = math.radians(bearing_deg)
    p = math.radians(pitch_deg)
    cp = math.cos(p)
    forward = np.array([cp * math.sin(b), cp * math.cos(b), math.sin(p)], dtype=np.float64)
    right = np.array([math.cos(b), -math.sin(b), 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    down = -up
    return right, down, forward / np.linalg.norm(forward)


def c2w_from_pose(center_enu: np.ndarray, bearing_deg: float, pitch_deg: float) -> np.ndarray:
    right, down, forward = camera_basis(bearing_deg, pitch_deg)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = center_enu
    return c2w


def w2c_from_pose(center_enu: np.ndarray, bearing_deg: float, pitch_deg: float) -> np.ndarray:
    return np.linalg.inv(c2w_from_pose(center_enu, bearing_deg, pitch_deg)).astype(np.float32)


def perspective_ray_grid(width: int, height: int, bearing_deg: float, pitch_deg: float) -> np.ndarray:
    K = virtual_intrinsics(width, height)
    K_inv = np.linalg.inv(K)
    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows)
    pix = np.stack([cc, rr, np.ones_like(cc)], axis=-1)
    rays_cam = pix @ K_inv.T
    rays_cam /= np.linalg.norm(rays_cam, axis=2, keepdims=True)
    right, down, forward = camera_basis(bearing_deg, pitch_deg)
    rot = np.stack([right, down, forward], axis=1)
    rays_world = rays_cam @ rot.T
    rays_world /= np.linalg.norm(rays_world, axis=2, keepdims=True)
    return rays_world


def rays_to_equirect_maps(rays: np.ndarray, pano_width: int, pano_height: int, beta0_deg: float):
    east = rays[:, :, 0]
    north = rays[:, :, 1]
    up = np.clip(rays[:, :, 2], -1.0, 1.0)
    bearing = wrap_angle_deg(np.degrees(np.arctan2(east, north)))
    pitch = np.degrees(np.arcsin(up))
    map_x = ((bearing - beta0_deg) % 360.0) / 360.0 * pano_width
    map_y = (90.0 - pitch) / 180.0 * pano_height
    return map_x.astype(np.float32), np.clip(map_y, 0.0, pano_height - 1.0).astype(np.float32)


def reproject_perspective(pano_rgb: np.ndarray, beta0_deg: float, bearing_deg: float, pitch_deg: float):
    rays = perspective_ray_grid(CFG.view_size_px, CFG.view_size_px, bearing_deg, pitch_deg)
    map_x, map_y = rays_to_equirect_maps(rays, pano_rgb.shape[1], pano_rgb.shape[0], beta0_deg)
    crop = cv2.remap(
        pano_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return crop


def load_lidar_st_axis() -> np.ndarray | None:
    """Load the LiDAR/ST S-axis so panorama left/right matches LiDAR output."""
    if str(CFG.st_meta_json) in {"", "."}:
        return None
    if not CFG.st_meta_json.exists():
        print(f"  warning: ST metadata not found; using sparmitt file direction for sides: {CFG.st_meta_json}")
        return None
    with open(CFG.st_meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    candidates = [
        ("main_axis_x", "main_axis_y"),
        ("axis_east", "axis_north"),
        ("pca_axis_east", "pca_axis_north"),
    ]
    for key_e, key_n in candidates:
        if key_e in meta and key_n in meta:
            axis = np.array([float(meta[key_e]), float(meta[key_n])], dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm > 0:
                return axis / norm
    print(f"  warning: ST metadata has no recognised S-axis fields; using sparmitt file direction: {CFG.st_meta_json}")
    return None


def local_track_frame(sp: pd.DataFrame, e_cam: float, n_cam: float, half_window: int = 5):
    global LOCAL_FRAME_FLIP_COUNT
    dist = np.hypot(sp["East"].to_numpy() - e_cam, sp["North"].to_numpy() - n_cam)
    idx = int(np.argmin(dist))
    i0 = max(0, idx - half_window)
    i1 = min(len(sp) - 1, idx + half_window)
    tx = float(sp["East"].iloc[i1] - sp["East"].iloc[i0])
    ty = float(sp["North"].iloc[i1] - sp["North"].iloc[i0])
    norm = math.hypot(tx, ty)
    tx /= norm
    ty /= norm
    if LIDAR_ST_AXIS_EN is not None and (tx * LIDAR_ST_AXIS_EN[0] + ty * LIDAR_ST_AXIS_EN[1]) < 0.0:
        tx = -tx
        ty = -ty
        LOCAL_FRAME_FLIP_COUNT += 1
    track_bearing = wrap_angle_deg(math.degrees(math.atan2(tx, ty)))
    rx, ry = ty, -tx
    return float(track_bearing), tx, ty, rx, ry


def load_da3_model():
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if CFG.model_id:
        target = CFG.model_id
    elif CFG.model_dir != Path("."):
        target = str(CFG.model_dir)
    else:
        raise RuntimeError("PANO_DA3_MODEL_DIR is required. Run through run_panorama.py.")
    print(f"  loading DA3 from: {target}")
    model = DepthAnything3.from_pretrained(target).to(device).eval()
    print(f"  DA3 loaded on {device}")
    return model


def group_indices(target_idx: int, n: int, half_window: int) -> list[int]:
    lo = max(0, target_idx - half_window)
    hi = min(n - 1, target_idx + half_window)
    return list(range(lo, hi + 1))


def target_side_anchor(target: pd.Series, sp: pd.DataFrame, side: str, anchor_t_m: float | None = None) -> np.ndarray:
    e0 = float(target["East"])
    n0 = float(target["North"])
    z0 = float(target["Height"])
    _bearing, _tx, _ty, rx, ry = local_track_frame(sp, e0, n0)
    side_sign = 1.0 if side == "right" else -1.0
    anchor_t = CFG.target_anchor_t_m if anchor_t_m is None else float(anchor_t_m)
    signed_t = side_sign * anchor_t
    z_anchor = z0 - CFG.camera_height_m - CFG.target_anchor_depth_below_rail_m
    return np.array([e0 + signed_t * rx, n0 + signed_t * ry, z_anchor], dtype=np.float64)


def look_at_bearing_pitch(camera_center: np.ndarray, target_point: np.ndarray) -> tuple[float, float]:
    delta = target_point - camera_center
    horizontal = max(0.1, math.hypot(float(delta[0]), float(delta[1])))
    bearing = wrap_angle_deg(math.degrees(math.atan2(float(delta[0]), float(delta[1]))))
    pitch = math.degrees(math.atan2(float(delta[2]), horizontal))
    pitch = float(np.clip(pitch, CFG.min_view_pitch_deg, CFG.max_view_pitch_deg))
    return float(bearing), pitch


def prepare_multiview_group(
    panos: pd.DataFrame,
    sp: pd.DataFrame,
    target_idx: int,
    side: str,
    half_window: int,
    paths: dict[str, Path],
    anchor_t_m: float | None = None,
):
    idxs = group_indices(target_idx, len(panos), half_window)
    target = panos.iloc[target_idx]
    e0 = float(target["East"])
    n0 = float(target["North"])
    z0 = float(target["Height"])
    anchor = target_side_anchor(target, sp, side, anchor_t_m)
    anchor_name = anchor_tag(CFG.target_anchor_t_m if anchor_t_m is None else float(anchor_t_m))

    crop_paths: list[str] = []
    extrinsics_local: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    c2w_global: list[np.ndarray] = []
    source_indices: list[int] = []

    for idx in idxs:
        row = panos.iloc[idx]
        e = float(row["East"])
        n = float(row["North"])
        z = float(row["Height"])
        center_global = np.array([e, n, z], dtype=np.float64)
        view_bearing, view_pitch = look_at_bearing_pitch(center_global, anchor)
        beta0 = wrap_angle_deg(float(row["Pan_world_deg"]) + 180.0)

        pano_bgr = cv2.imread(str(row["jpg_path"]))
        if pano_bgr is None:
            continue
        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
        crop = reproject_perspective(pano_rgb, beta0, view_bearing, view_pitch)
        crop_path = (
            paths["crops"]
            / f"{row['Filename']}_{CFG.side_frame_tag}_{side}_{anchor_name}_target{target_idx:03d}_hw{half_window}.jpg"
        )
        cv2.imwrite(
            str(crop_path),
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )

        center_local = np.array([e - e0, n - n0, z - z0], dtype=np.float64)
        crop_paths.append(str(crop_path))
        intrinsics.append(virtual_intrinsics(CFG.view_size_px, CFG.view_size_px))
        extrinsics_local.append(w2c_from_pose(center_local, view_bearing, view_pitch))
        c2w_global.append(c2w_from_pose(center_global, view_bearing, view_pitch))
        source_indices.append(idx)

    return crop_paths, np.asarray(extrinsics_local), np.asarray(intrinsics), c2w_global, source_indices


def run_da3_group(model, crop_paths: list[str], extrinsics: np.ndarray, intrinsics: np.ndarray):
    with torch.inference_mode():
        return model.inference(
            image=crop_paths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=True,
            process_res=CFG.process_res,
            process_res_method="upper_bound_resize",
            ref_view_strategy="middle",
        )


def infer_with_retry(model, panos, sp, target_idx, side, paths, anchor_t_m: float | None = None):
    last_error = None
    for half_window in range(CFG.preferred_half_window, CFG.min_half_window - 1, -1):
        crop_paths, extrinsics, intrinsics, c2w_global, source_indices = prepare_multiview_group(
            panos, sp, target_idx, side, half_window, paths, anchor_t_m=anchor_t_m
        )
        if len(crop_paths) == 0:
            continue
        try:
            pred = run_da3_group(model, crop_paths, extrinsics, intrinsics)
            return pred, crop_paths, c2w_global, source_indices, half_window
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            last_error = exc
            print(f"    OOM with half_window={half_window}; retrying smaller group")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                last_error = exc
                print(f"    OOM with half_window={half_window}; retrying smaller group")
            else:
                raise
    raise RuntimeError(f"DA3 inference failed for target {target_idx} {side}: {last_error}")


def infer_or_load_cached(model, panos, sp, target_idx, target_name: str, side: str, paths, anchor_t_m: float | None = None):
    anchor_name = anchor_tag(CFG.target_anchor_t_m if anchor_t_m is None else float(anchor_t_m))
    depth_path = paths["depths"] / f"{target_name}_{CFG.side_frame_tag}_{side}_{anchor_name}_multiview_depths.npz"
    if depth_path.exists():
        crop_paths, _, _, c2w_global, source_indices = prepare_multiview_group(
            panos, sp, target_idx, side, CFG.preferred_half_window, paths, anchor_t_m=anchor_t_m
        )
        cached = np.load(depth_path, allow_pickle=True)

        class CachedPrediction:
            pass

        pred = CachedPrediction()
        pred.depth = cached["depth"]
        pred.intrinsics = cached["intrinsics"]
        return pred, crop_paths, c2w_global, source_indices, CFG.preferred_half_window, depth_path

    pred, crop_paths, c2w_global, source_indices, half_window_used = infer_with_retry(
        model, panos, sp, target_idx, side, paths, anchor_t_m=anchor_t_m
    )
    np.savez_compressed(
        depth_path,
        depth=pred.depth.astype(np.float32),
        intrinsics=pred.intrinsics.astype(np.float32),
        source_indices=np.asarray(source_indices, dtype=np.int32),
        crop_paths=np.asarray(crop_paths),
    )
    return pred, crop_paths, c2w_global, source_indices, half_window_used, depth_path


def backproject_depths(depths: np.ndarray, intrinsics: np.ndarray, c2w_global: list[np.ndarray]):
    points = []
    n_views, h, w = depths.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([cols, rows, np.ones_like(cols)], axis=-1).reshape(-1, 3)
    for i in range(n_views):
        d = depths[i].reshape(-1)
        valid = np.isfinite(d) & (d > 0.0) & (d < CFG.max_depth_m)
        if valid.sum() == 0:
            continue
        K_inv = np.linalg.inv(intrinsics[i])
        rays = (K_inv @ pix[valid].T).T
        xyz_cam = rays * d[valid, None]
        xyz_h = np.concatenate([xyz_cam, np.ones((xyz_cam.shape[0], 1))], axis=1)
        xyz_world = (c2w_global[i] @ xyz_h.T).T[:, :3]
        points.append(xyz_world)
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(points, axis=0)


def backproject_depths_by_view(depths: np.ndarray, intrinsics: np.ndarray, c2w_global: list[np.ndarray]):
    points_by_view: list[np.ndarray] = []
    n_views, h, w = depths.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([cols, rows, np.ones_like(cols)], axis=-1).reshape(-1, 3)
    for i in range(n_views):
        d = depths[i].reshape(-1)
        valid = np.isfinite(d) & (d > 0.0) & (d < CFG.max_depth_m)
        if valid.sum() == 0:
            points_by_view.append(np.zeros((0, 3), dtype=np.float64))
            continue
        K_inv = np.linalg.inv(intrinsics[i])
        rays = (K_inv @ pix[valid].T).T
        xyz_cam = rays * d[valid, None]
        xyz_h = np.concatenate([xyz_cam, np.ones((xyz_cam.shape[0], 1))], axis=1)
        xyz_world = (c2w_global[i] @ xyz_h.T).T[:, :3]
        points_by_view.append(xyz_world)
    return points_by_view


def _reference_quality_score(
    shoulder_points: int,
    shoulder_iqr_m: float,
    datum_shift_m: float,
    n_reference_views: int,
    valid_bins: int,
):
    support_score = min(1.0, shoulder_points / 8000.0)
    if np.isfinite(shoulder_iqr_m):
        spread_score = float(np.clip((0.45 - shoulder_iqr_m) / 0.35, 0.0, 1.0))
    else:
        spread_score = 0.0
    shift_score = float(np.clip((CFG.datum_shift_limit_m - abs(datum_shift_m)) / CFG.datum_shift_limit_m, 0.0, 1.0))
    view_score = min(1.0, n_reference_views / 3.0)
    bin_score = min(1.0, valid_bins / 20.0)
    return float(0.30 * support_score + 0.25 * spread_score + 0.20 * shift_score + 0.15 * view_score + 0.10 * bin_score)


def _profile_arrays_from_points(
    points: np.ndarray,
    target: pd.Series,
    sp: pd.DataFrame,
    side: str,
    datum_shift: float,
):
    e0 = float(target["East"])
    n0 = float(target["North"])
    z0 = float(target["Height"])
    z_rail = z0 - CFG.camera_height_m
    track_bearing, tx, ty, rx, ry = local_track_frame(sp, e0, n0)
    side_sign = 1.0 if side == "right" else -1.0
    d_e = points[:, 0] - e0
    d_n = points[:, 1] - n0
    along = d_e * tx + d_n * ty
    side_t = side_sign * (d_e * rx + d_n * ry)
    z = points[:, 2]
    z_corr = z + datum_shift
    zone = (
        (side_t >= CFG.side_t_min_m)
        & (side_t <= CFG.side_t_max_m)
        & (np.abs(along) <= CFG.along_half_m)
        & np.isfinite(z_corr)
    )
    bins = np.arange(CFG.side_t_min_m, CFG.side_t_max_m + CFG.bin_width_m, CFG.bin_width_m)
    centres = 0.5 * (bins[:-1] + bins[1:])
    z_profile = np.full(len(centres), np.nan)
    n_profile = np.zeros(len(centres), dtype=int)
    if zone.sum() > 0:
        idx = np.digitize(side_t[zone], bins) - 1
        zz = z_corr[zone]
        for j in range(len(centres)):
            m = idx == j
            if m.sum() >= CFG.min_pixels_per_bin:
                # Lower visible surface, robust against vegetation outliers above the ground.
                z_profile[j] = float(np.nanpercentile(zz[m], 25))
                n_profile[j] = int(m.sum())
    return centres, z_profile, n_profile, track_bearing, z_rail, int(zone.sum())


def _band_candidate(
    depth_profile: np.ndarray,
    centres: np.ndarray,
    valid: np.ndarray,
    t_min: float,
    t_max: float,
    zone_name: str,
) -> dict:
    search = valid & (centres >= t_min) & (centres <= t_max)
    if not search.any():
        return {
            "zone": zone_name,
            "depth": np.nan,
            "t": np.nan,
            "prominence": np.nan,
            "trend": np.nan,
            "peak_at_boundary": False,
            "n_bins": 0,
            "idx": -1,
        }

    search_idx = np.where(search)[0]
    best_idx = int(search_idx[np.nanargmax(depth_profile[search])])
    search_depth = depth_profile[search]
    local_floor = float(np.nanpercentile(search_depth, 25))
    prominence = float(depth_profile[best_idx] - local_floor)
    valid_depth = np.isfinite(search_depth)
    if valid_depth.sum() >= 5:
        x = centres[search][valid_depth]
        y = search_depth[valid_depth]
        trend = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-6 else 0.0
    else:
        trend = np.nan
    peak_at_boundary = bool(
        centres[best_idx] <= t_min + CFG.bin_width_m
        or centres[best_idx] >= t_max - CFG.bin_width_m
    )
    return {
        "zone": zone_name,
        "depth": float(depth_profile[best_idx]),
        "t": float(centres[best_idx]),
        "prominence": prominence,
        "trend": trend,
        "peak_at_boundary": peak_at_boundary,
        "n_bins": int(search.sum()),
        "idx": best_idx,
    }


def select_visible_depression(depth_profile: np.ndarray, centres: np.ndarray, valid: np.ndarray) -> dict:
    primary = _band_candidate(
        depth_profile,
        centres,
        valid,
        CFG.primary_ditch_search_t_min_m,
        CFG.primary_ditch_search_t_max_m,
        "primary",
    )
    near = _band_candidate(
        depth_profile,
        centres,
        valid,
        CFG.near_track_search_t_min_m,
        CFG.near_track_search_t_max_m,
        "near_track_fallback",
    )

    def diagnostic_score(item: dict) -> float:
        if not np.isfinite(item["depth"]):
            return -np.inf
        # Step 4 is no longer a hard ditch detector.  It chooses a diagnostic
        # visible-surface bin for plotting and fallback reporting.  The final
        # ditch decision is made later by the profile-path detector.
        score = float(item["depth"])
        if np.isfinite(item["prominence"]):
            score += float(item["prominence"])
        if item["peak_at_boundary"]:
            score -= abs(score)
        return score

    primary_score = diagnostic_score(primary)
    near_score = diagnostic_score(near)
    if near_score > primary_score:
        chosen = near
        fallback_used = True
    else:
        chosen = primary
        fallback_used = False

    return {
        "visible_surface_depth_m": chosen["depth"],
        "visible_surface_T_m": chosen["t"],
        "visible_surface_search_zone": chosen["zone"],
        "visible_surface_search_zone_used_fallback": bool(fallback_used),
        "selected_profile_local_prominence_m": chosen["prominence"],
        "selected_profile_trend_corr": chosen["trend"],
        "selected_peak_at_search_boundary": bool(chosen["peak_at_boundary"]),
        "primary_visible_surface_depth_m": primary["depth"],
        "primary_visible_surface_T_m": primary["t"],
        "primary_profile_local_prominence_m": primary["prominence"],
        "near_track_visible_surface_depth_m": near["depth"],
        "near_track_visible_surface_T_m": near["t"],
        "near_track_profile_local_prominence_m": near["prominence"],
    }


def profile_from_multiview_points(
    points_by_view: list[np.ndarray],
    target: pd.Series,
    sp: pd.DataFrame,
    side: str,
    target_ditch_t_m: float | None = None,
):
    if not points_by_view or sum(len(p) for p in points_by_view) == 0:
        return empty_profile()
    e0 = float(target["East"])
    n0 = float(target["North"])
    z0 = float(target["Height"])
    z_rail = z0 - CFG.camera_height_m
    track_bearing, tx, ty, rx, ry = local_track_frame(sp, e0, n0)
    side_sign = 1.0 if side == "right" else -1.0

    shoulder_medians: list[float] = []
    shoulder_points_total = 0
    for points in points_by_view:
        if len(points) == 0:
            continue
        d_e = points[:, 0] - e0
        d_n = points[:, 1] - n0
        along = d_e * tx + d_n * ty
        side_t = side_sign * (d_e * rx + d_n * ry)
        z = points[:, 2]
        shoulder = (
            (side_t >= CFG.shoulder_t_min_m)
            & (side_t <= CFG.shoulder_t_max_m)
            & (np.abs(along) <= CFG.shoulder_along_half_m)
            & np.isfinite(z)
        )
        if shoulder.sum() >= CFG.min_shoulder_points_per_view:
            shoulder_medians.append(float(np.nanmedian(z[shoulder])))
            shoulder_points_total += int(shoulder.sum())

    if shoulder_medians:
        shoulder_z = float(np.nanmedian(shoulder_medians))
        shoulder_iqr = float(np.nanpercentile(shoulder_medians, 75) - np.nanpercentile(shoulder_medians, 25))
        candidate_shift = float(z_rail - shoulder_z)
        if abs(candidate_shift) <= CFG.datum_shift_limit_m:
            datum_shift = candidate_shift
            datum_shift_status = "accepted"
        else:
            datum_shift = 0.0
            datum_shift_status = "rejected_large_shift"
    else:
        shoulder_z = np.nan
        shoulder_iqr = np.nan
        datum_shift = 0.0
        datum_shift_status = "insufficient_shoulder_points"

    view_profiles: list[np.ndarray] = []
    view_counts: list[np.ndarray] = []
    zone_points_total = 0
    centres = None
    for points in points_by_view:
        if len(points) == 0:
            continue
        c, z_prof, n_prof, _bearing, _zrail, zone_points = _profile_arrays_from_points(points, target, sp, side, datum_shift)
        centres = c
        zone_points_total += zone_points
        if np.isfinite(z_prof).any():
            view_profiles.append(z_prof)
            view_counts.append(n_prof)

    if centres is None:
        return empty_profile()
    if view_profiles:
        stack = np.vstack(view_profiles)
        z_profile = np.full(stack.shape[1], np.nan, dtype=np.float64)
        for j in range(stack.shape[1]):
            col = stack[:, j]
            if np.isfinite(col).any():
                z_profile[j] = float(np.nanmedian(col))
        n_profile = np.sum(np.vstack(view_counts), axis=0).astype(int)
        n_profile[~np.isfinite(z_profile)] = 0
        n_fused_views = np.sum(np.isfinite(stack), axis=0)
    else:
        z_profile = np.full(len(centres), np.nan)
        n_profile = np.zeros(len(centres), dtype=int)
        n_fused_views = np.zeros(len(centres), dtype=int)

    depth_profile = z_rail - z_profile
    valid = np.isfinite(depth_profile)
    if valid.any():
        best_idx = int(np.nanargmax(depth_profile))
        diagnostic_max_depth = float(depth_profile[best_idx])
        diagnostic_max_t = float(centres[best_idx])
    else:
        diagnostic_max_depth = np.nan
        diagnostic_max_t = np.nan

    visible = select_visible_depression(depth_profile, centres, valid)

    if target_ditch_t_m is not None and np.isfinite(target_ditch_t_m):
        guided = valid & (np.abs(centres - float(target_ditch_t_m)) <= CFG.lidar_guided_half_width_m)
        if guided.any():
            lidar_guided_depth = float(np.nanmedian(depth_profile[guided]))
            lidar_guided_t = float(np.nanmedian(centres[guided]))
        else:
            lidar_guided_depth = float(np.interp(float(target_ditch_t_m), centres[valid], depth_profile[valid]))
            lidar_guided_t = float(target_ditch_t_m)
    else:
        lidar_guided_depth = np.nan
        lidar_guided_t = np.nan

    reference_quality_score = _reference_quality_score(
        shoulder_points_total,
        shoulder_iqr,
        datum_shift,
        len(shoulder_medians),
        int(valid.sum()),
    )
    has_metric_reference = datum_shift_status == "accepted" and reference_quality_score >= CFG.min_reference_quality_score

    return {
        "track_bearing_deg": track_bearing,
        "z_rail": z_rail,
        "shoulder_points": int(shoulder_points_total),
        "n_reference_views": int(len(shoulder_medians)),
        "shoulder_z_before_shift": shoulder_z,
        "shoulder_z_iqr_m": shoulder_iqr,
        "datum_shift_m": datum_shift,
        "datum_shift_status": datum_shift_status,
        "reference_quality_score": reference_quality_score,
        "has_metric_reference": has_metric_reference,
        "zone_points": int(zone_points_total),
        "t_centres": centres,
        "z_profile": z_profile,
        "depth_profile": depth_profile,
        "n_profile": n_profile,
        "n_fused_views": n_fused_views,
        "visible_surface_depth_m": visible["visible_surface_depth_m"],
        "visible_surface_T_m": visible["visible_surface_T_m"],
        "visible_surface_search_zone": visible["visible_surface_search_zone"],
        "visible_surface_search_zone_used_fallback": visible["visible_surface_search_zone_used_fallback"],
        "selected_profile_local_prominence_m": visible["selected_profile_local_prominence_m"],
        "selected_profile_trend_corr": visible["selected_profile_trend_corr"],
        "selected_peak_at_search_boundary": visible["selected_peak_at_search_boundary"],
        "primary_visible_surface_depth_m": visible["primary_visible_surface_depth_m"],
        "primary_visible_surface_T_m": visible["primary_visible_surface_T_m"],
        "primary_profile_local_prominence_m": visible["primary_profile_local_prominence_m"],
        "near_track_visible_surface_depth_m": visible["near_track_visible_surface_depth_m"],
        "near_track_visible_surface_T_m": visible["near_track_visible_surface_T_m"],
        "near_track_profile_local_prominence_m": visible["near_track_profile_local_prominence_m"],
        "diagnostic_max_depth_m": diagnostic_max_depth,
        "diagnostic_max_T_m": diagnostic_max_t,
        "lidar_guided_visible_surface_depth_m": lidar_guided_depth,
        "lidar_guided_visible_surface_T_m": lidar_guided_t,
        "valid_bins": int(valid.sum()),
    }


def _project_world_points_to_crop(points_enu: np.ndarray, c2w: np.ndarray, K: np.ndarray):
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([points_enu, np.ones((len(points_enu), 1))], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    uv = np.full((len(points_enu), 2), np.nan, dtype=np.float64)
    valid = z > 0.1
    if valid.any():
        pix = (K @ pts_cam[valid].T).T
        uv[valid, 0] = pix[:, 0] / pix[:, 2]
        uv[valid, 1] = pix[:, 1] / pix[:, 2]
    return uv, valid


def _rail_world_points(target: pd.Series, sp: pd.DataFrame, gauge_m: float, forward_sign: float):
    e0 = float(target["East"])
    n0 = float(target["North"])
    z_rail = float(target["Height"]) - CFG.camera_height_m
    _bearing, tx, ty, rx, ry = local_track_frame(sp, e0, n0)
    distances = np.linspace(4.0, 35.0, 80)
    lines = []
    for t in (-0.5 * gauge_m, 0.5 * gauge_m):
        e = e0 + forward_sign * distances * tx + t * rx
        n = n0 + forward_sign * distances * ty + t * ry
        z = np.full_like(e, z_rail)
        lines.append(np.column_stack([e, n, z]))
    return lines


def _edge_support_for_projected_lines(edge_img: np.ndarray, projected_lines: list[np.ndarray]):
    if edge_img.size == 0:
        return 0.0, 0
    dilated = cv2.dilate(edge_img, np.ones((5, 5), np.uint8), iterations=1)
    h, w = edge_img.shape[:2]
    supports = []
    n_valid = 0
    for uv in projected_lines:
        ok = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if ok.sum() < 10:
            continue
        xs = np.round(uv[ok, 0]).astype(int)
        ys = np.round(uv[ok, 1]).astype(int)
        supports.append(float(np.mean(dilated[ys, xs] > 0)))
        n_valid += int(ok.sum())
    if not supports:
        return 0.0, n_valid
    return float(np.mean(supports)), n_valid


def rail_gauge_quality(target: pd.Series, sp: pd.DataFrame, paths: dict[str, Path]):
    pano_bgr = cv2.imread(str(target["jpg_path"]))
    if pano_bgr is None:
        return {
            "rail_gauge_quality_score": 0.0,
            "rail_gauge_status": "missing_image",
            "rail_gauge_est_m": np.nan,
            "rail_gauge_error_m": np.nan,
            "rail_edge_support": 0.0,
            "rail_crop_path": "",
        }
    pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
    beta0 = wrap_angle_deg(float(target["Pan_world_deg"]) + 180.0)
    track_bearing, _tx, _ty, _rx, _ry = local_track_frame(sp, float(target["East"]), float(target["North"]))

    best = None
    for forward_sign, bearing in ((1.0, track_bearing), (-1.0, wrap_angle_deg(track_bearing + 180.0))):
        crop = reproject_perspective(pano_rgb, beta0, bearing, CFG.rail_crop_pitch_deg)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(gray, 60, 160)
        c2w = c2w_from_pose(
            np.array([float(target["East"]), float(target["North"]), float(target["Height"])], dtype=np.float64),
            bearing,
            CFG.rail_crop_pitch_deg,
        )
        K = virtual_intrinsics(CFG.view_size_px, CFG.view_size_px)
        scores = []
        for gauge in np.linspace(1.10, 1.80, 29):
            projected = []
            for line in _rail_world_points(target, sp, float(gauge), forward_sign):
                uv, _valid = _project_world_points_to_crop(line, c2w, K)
                projected.append(uv)
            support, n_valid = _edge_support_for_projected_lines(edges, projected)
            error = abs(float(gauge) - CFG.rail_expected_gauge_m)
            gauge_score = max(0.0, 1.0 - error / CFG.rail_gauge_tolerance_m)
            score = 0.65 * support + 0.35 * gauge_score
            scores.append((score, support, n_valid, float(gauge), projected, crop, bearing, forward_sign))
        candidate = max(scores, key=lambda item: item[0])
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        status = "no_projection"
        quality = 0.0
        support = 0.0
        n_valid = 0
        gauge_est = np.nan
        projected = []
        crop = pano_rgb
    else:
        quality, support, n_valid, gauge_est, projected, crop, _bearing, _forward_sign = best
        error = abs(gauge_est - CFG.rail_expected_gauge_m)
        if n_valid < 40:
            status = "insufficient_projected_rail"
        elif support < 0.08:
            status = "weak_edge_support"
        elif error > CFG.rail_gauge_tolerance_m:
            status = "gauge_mismatch"
        else:
            status = "accepted"

    rail_overlay = crop.copy()
    colors = [(255, 70, 70), (70, 255, 70)]
    for line_idx, uv in enumerate(projected):
        ok = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        pts = np.round(uv[ok]).astype(int)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < CFG.view_size_px) & (pts[:, 1] >= 0) & (pts[:, 1] < CFG.view_size_px)]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(rail_overlay, tuple(p0), tuple(p1), colors[line_idx % len(colors)], 2, cv2.LINE_AA)
    rail_crop_path = paths["rail_crops"] / f"{target['Filename']}_rail_gauge_qc.jpg"
    cv2.imwrite(str(rail_crop_path), cv2.cvtColor(rail_overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
    return {
        "rail_gauge_quality_score": float(quality),
        "rail_gauge_status": status,
        "rail_gauge_est_m": float(gauge_est) if np.isfinite(gauge_est) else np.nan,
        "rail_gauge_error_m": float(abs(gauge_est - CFG.rail_expected_gauge_m)) if np.isfinite(gauge_est) else np.nan,
        "rail_edge_support": float(support),
        "rail_projected_samples": int(n_valid),
        "rail_crop_path": str(rail_crop_path),
    }


def classify_visibility(profile: dict, crop_path: str):
    crop_bgr = cv2.imread(str(crop_path))
    vegetation_ratio = np.nan
    if crop_bgr is not None:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w = crop_rgb.shape[:2]
        # The lower central part of the side-looking crop contains ballast,
        # shoulder and ditch foreground.  It is only used as a visibility cue,
        # not as a metric reference.
        roi = crop_rgb[int(0.48 * h) : int(0.88 * h), int(0.12 * w) : int(0.88 * w)]
        exg = 2.0 * roi[:, :, 1] - roi[:, :, 0] - roi[:, :, 2]
        vegetation_ratio = float(np.mean(exg > CFG.vegetation_exg_threshold))

    depth = profile["depth_profile"]
    centres = profile["t_centres"]
    local_prominence = profile.get("selected_profile_local_prominence_m", np.nan)
    trend = profile.get("selected_profile_trend_corr", np.nan)

    fused_views_at_peak = 0
    if np.isfinite(profile["visible_surface_T_m"]):
        idx = int(np.nanargmin(np.abs(centres - profile["visible_surface_T_m"])))
        if idx < len(profile["n_fused_views"]):
            fused_views_at_peak = int(profile["n_fused_views"][idx])
    peak_at_boundary = bool(profile.get("selected_peak_at_search_boundary", False))
    monotonic_surface = bool(np.isfinite(trend) and abs(trend) >= 0.88)
    near_track_fallback = bool(profile.get("visible_surface_search_zone_used_fallback", False))
    has_diagnostic_surface = bool(
        np.isfinite(profile.get("visible_surface_depth_m", np.nan))
        and np.isfinite(profile.get("visible_surface_T_m", np.nan))
    )
    has_local_depression_shape = bool(np.isfinite(local_prominence) and local_prominence > 0.0)

    if not profile["has_metric_reference"]:
        label = "insufficient_metric_reference"
        confidence = 0.25
        bottom_visible = False
    elif np.isfinite(vegetation_ratio) and vegetation_ratio >= 0.35:
        label = "vegetation_surface_likely"
        confidence = min(0.95, 0.45 + vegetation_ratio)
        bottom_visible = False
    elif peak_at_boundary:
        label = "boundary_surface_uncertain"
        confidence = 0.65
        bottom_visible = False
    elif monotonic_surface:
        label = "monotonic_slope_surface"
        confidence = 0.60
        bottom_visible = False
    elif not has_diagnostic_surface or not has_local_depression_shape:
        label = "no_clear_visible_depression"
        confidence = 0.55
        bottom_visible = False
    elif fused_views_at_peak < 2:
        label = "single_view_surface_uncertain"
        confidence = 0.50
        bottom_visible = False
    else:
        label = "near_track_bottom_visible_candidate" if near_track_fallback else "bottom_visible_candidate"
        confidence = 0.70 if near_track_fallback else (0.75 if fused_views_at_peak < 4 else 0.85)
        bottom_visible = True

    has_side_profile = bool(
        np.isfinite(profile.get("visible_surface_T_m", np.nan))
        and np.isfinite(profile.get("visible_surface_depth_m", np.nan))
    )
    if bottom_visible:
        ditch_presence_class = "exposed_ditch_bottom_candidate"
    elif has_side_profile and label in {
        "vegetation_surface_likely",
        "boundary_surface_uncertain",
        "no_clear_visible_depression",
        "monotonic_slope_surface",
        "near_track_surface_uncertain",
        "single_view_surface_uncertain",
    }:
        ditch_presence_class = "ditch_corridor_visible_metric_uncertain"
    elif has_side_profile:
        ditch_presence_class = "side_depression_context_visible"
    else:
        ditch_presence_class = "no_image_profile_evidence"

    metric_candidate = label in {
        "bottom_visible_candidate",
        "near_track_bottom_visible_candidate",
    }
    metric_exclusion_reason = "metric_candidate" if metric_candidate else label

    return {
        "visibility_class": label,
        "visibility_confidence": float(confidence),
        "bottom_visible_candidate": bool(bottom_visible),
        "ditch_presence_class": ditch_presence_class,
        "visible_surface_metric_candidate": metric_candidate,
        "metric_exclusion_reason": metric_exclusion_reason,
        "vegetation_ratio": vegetation_ratio,
        "profile_local_prominence_m": local_prominence,
        "profile_trend_corr": trend,
        "peak_at_search_boundary": peak_at_boundary,
        "fused_views_at_peak": int(fused_views_at_peak),
    }


def empty_profile():
    bins = np.arange(CFG.side_t_min_m, CFG.side_t_max_m + CFG.bin_width_m, CFG.bin_width_m)
    centres = 0.5 * (bins[:-1] + bins[1:])
    arr = np.full(len(centres), np.nan)
    return {
        "track_bearing_deg": np.nan,
        "z_rail": np.nan,
        "shoulder_points": 0,
        "n_reference_views": 0,
        "shoulder_z_before_shift": np.nan,
        "shoulder_z_iqr_m": np.nan,
        "datum_shift_m": 0.0,
        "datum_shift_status": "empty",
        "reference_quality_score": 0.0,
        "has_metric_reference": False,
        "zone_points": 0,
        "t_centres": centres,
        "z_profile": arr.copy(),
        "depth_profile": arr.copy(),
        "n_profile": np.zeros(len(centres), dtype=int),
        "n_fused_views": np.zeros(len(centres), dtype=int),
        "visible_surface_depth_m": np.nan,
        "visible_surface_T_m": np.nan,
        "visible_surface_search_zone": "none",
        "visible_surface_search_zone_used_fallback": False,
        "selected_profile_local_prominence_m": np.nan,
        "selected_profile_trend_corr": np.nan,
        "selected_peak_at_search_boundary": False,
        "primary_visible_surface_depth_m": np.nan,
        "primary_visible_surface_T_m": np.nan,
        "primary_profile_local_prominence_m": np.nan,
        "near_track_visible_surface_depth_m": np.nan,
        "near_track_visible_surface_T_m": np.nan,
        "near_track_profile_local_prominence_m": np.nan,
        "diagnostic_max_depth_m": np.nan,
        "diagnostic_max_T_m": np.nan,
        "lidar_guided_visible_surface_depth_m": np.nan,
        "lidar_guided_visible_surface_T_m": np.nan,
        "valid_bins": 0,
    }


def nearest_lidar_row(lidar_df: pd.DataFrame | None, s_cam: float, side: str):
    if lidar_df is None:
        # Independent (LiDAR-free) variant: no ditch reference is consulted.
        return {
            "s": np.nan, "depth_m": np.nan, "bottom_t_m": np.nan,
            "span_x0_m": np.nan, "span_x1_m": np.nan, "priority": "NA",
        }
    idx = (lidar_df["s"] - s_cam).abs().idxmin()
    row = lidar_df.loc[idx]
    exists = int(row.get(f"{side}_ditch_exists", 0)) if pd.notna(row.get(f"{side}_ditch_exists", np.nan)) else 0
    depth = row.get(f"{side}_depth", np.nan) if exists == 1 else np.nan
    bottom_t = row.get(f"{side}_bottom_x", np.nan) if exists == 1 else np.nan
    span_x0 = row.get(f"{side}_span_x0", np.nan) if exists == 1 else np.nan
    span_x1 = row.get(f"{side}_span_x1", np.nan) if exists == 1 else np.nan
    priority = row.get(f"{side}_tdok_priority", "")
    return {
        "s": float(row["s"]),
        "depth_m": float(depth) if pd.notna(depth) else np.nan,
        "bottom_t_m": float(bottom_t) if pd.notna(bottom_t) else np.nan,
        "span_x0_m": float(span_x0) if pd.notna(span_x0) else np.nan,
        "span_x1_m": float(span_x1) if pd.notna(span_x1) else np.nan,
        "priority": priority,
    }


def save_profile_csv(path: Path, profile: dict):
    pd.DataFrame(
        {
            "t_m": profile["t_centres"],
            "z_visible_surface_m": profile["z_profile"],
            "visible_surface_depth_m": profile["depth_profile"],
            "in_automatic_ditch_window": (
                (profile["t_centres"] >= CFG.primary_ditch_search_t_min_m)
                & (profile["t_centres"] <= CFG.primary_ditch_search_t_max_m)
            ),
            "in_near_track_fallback_window": (
                (profile["t_centres"] >= CFG.near_track_search_t_min_m)
                & (profile["t_centres"] <= CFG.near_track_search_t_max_m)
            ),
            "n_points": profile["n_profile"],
            "n_fused_views": profile["n_fused_views"],
        }
    ).to_csv(path, index=False)


def save_diagnostic(path: Path, crop_path: str, depth: np.ndarray, profile: dict, title: str):
    """Publication-standard intermediate-process figure for the panorama method:
    (a) side-looking virtual view, (b) Depth Anything 3 depth, (c) the resulting
    visible-surface cross-section with the lateral search window and the detected
    ditch bottom.  Saved as vector PDF + 400 dpi PNG when figure_style is present.
    """
    if _PUB_STYLE:
        apply_style()
    profile_c = PALETTE["profile"] if _PUB_STYLE else "#1f2937"
    ditch_c = PALETTE["ditch"] if _PUB_STYLE else "#ef4444"
    ruk_c = PALETTE["ruk"] if _PUB_STYLE else "#dc2626"
    sky = PALETTE["search_zone"] if _PUB_STYLE else "#fee2e2"
    ground = "#e7ded0"
    width = CM(17) if _PUB_STYLE else 17.0 / 2.54
    crop = cv2.cvtColor(cv2.imread(crop_path), cv2.COLOR_BGR2RGB)
    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(width, width * 0.46),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]}, layout="constrained")

    # (a)(b) merged into the panel titles (no separate label -> no overlap)
    ax0.imshow(crop); ax0.set_title("(a) Side-looking view", fontsize=9.5); ax0.axis("off")
    im = ax1.imshow(depth, cmap="viridis"); ax1.set_title("(b) DA3 metric depth", fontsize=9.5); ax1.axis("off")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6.5)
    cb.set_label("depth (m)", fontsize=7.5)

    # (c) filled terrain cross-section (same style as the stand-alone figure)
    t = np.asarray(profile["t_centres"], dtype=float)
    d = np.asarray(profile["depth_profile"], dtype=float)
    m = np.isfinite(d) & (t >= 0) & (t <= 9.0)
    tt, hh = t[m], -d[m]                                   # height relative to rail
    lo, hi = float(np.nanmin(hh)), float(np.nanmax(hh))
    pad = 0.18 * max(hi - lo, 0.4)
    ymin = lo - pad
    ax2.fill_between(tt, hh, ymin, color=ground, zorder=1)
    ax2.plot(tt, hh, "-", color=profile_c, lw=1.7, solid_capstyle="round",
             label="panorama visible surface", zorder=3)
    ax2.axhline(0, color=ruk_c, lw=1.0, ls="--", label="rail reference (RUK)", zorder=4)
    smin, smax = CFG.primary_ditch_search_t_min_m, CFG.primary_ditch_search_t_max_m
    ax2.axvspan(smin, smax, color=ruk_c, alpha=0.07, lw=0, zorder=2, label="ditch-search window")
    for xb in (smin, smax):
        ax2.axvline(xb, color=ruk_c, lw=0.8, ls=":", alpha=0.7, zorder=2)
    ti = profile.get("visible_surface_T_m", np.nan)
    if np.isfinite(ti) and m.any():
        di = float(d[m][np.argmin(np.abs(t[m] - ti))])
        ax2.scatter([ti], [-di], marker="*", s=85, color=ditch_c, edgecolor="white",
                    linewidth=0.6, label="detected ditch bottom", zorder=6)
        ax2.annotate(f"depth = {di:.2f} m", xy=(ti, -di), xytext=(6, 16),
                     textcoords="offset points", fontsize=7.5, color=ditch_c,
                     fontweight="bold", zorder=7,
                     arrowprops=dict(arrowstyle="-", color=ditch_c, lw=0.7))
    ax2.set_ylim(ymin, hi + pad)
    ax2.set_xlim(0, 9)
    ax2.set_xlabel("Lateral offset from track centre, $T$ (m)")
    ax2.set_ylabel("Height relative to\nrail reference (m)")
    ax2.set_title("(c) Visible-surface cross-section", fontsize=9.5)
    ax2.grid(True, zorder=0)
    # No legend inside the figure; the elements are explained in the caption.

    if _PUB_STYLE:
        savefig(fig, Path(path).with_suffix(""))    # writes .pdf + .png
    else:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    global LIDAR_ST_AXIS_EN, LOCAL_FRAME_FLIP_COUNT
    LIDAR_ST_AXIS_EN = load_lidar_st_axis()
    LOCAL_FRAME_FLIP_COUNT = 0
    paths = ensure_dirs()
    print("=" * 78)
    print("Multi-view virtual-camera DA3 on Cirrus panoramas")
    print("=" * 78)
    panos = pd.read_csv(CFG.pano_csv)
    jpg_paths = panos["jpg_path"].fillna("").astype(str)
    panos = panos[jpg_paths.map(lambda p: len(p) > 0 and Path(p).exists())].reset_index(drop=True)
    sp = pd.read_csv(CFG.sparmitt_csv)
    # LiDAR is optional.  In the independent (LiDAR-free) variant
    # (PANO_USE_LIDAR=0) no ditch reference is read: the depth, datum and
    # left/right frame all come from the panorama GNSS poses + sparmitt
    # centreline + ballast-shoulder geometry, and the lidar_* columns are left
    # blank for reporting only.
    if USE_LIDAR and CFG.lidar_csv.exists():
        lidar = pd.read_csv(CFG.lidar_csv)
    else:
        lidar = None
        print(f"  LiDAR reference disabled (independent variant); USE_LIDAR={USE_LIDAR}")
    anchors = lateral_anchor_values()
    print(f"  panoramas with JPG: {len(panos)}")
    print(f"  preferred group: i-{CFG.preferred_half_window}..i+{CFG.preferred_half_window}")
    print(f"  lateral anchors per side: {', '.join(f'{v:.1f} m' for v in anchors)}")
    if LIDAR_ST_AXIS_EN is not None:
        axis_bearing = wrap_angle_deg(math.degrees(math.atan2(float(LIDAR_ST_AXIS_EN[0]), float(LIDAR_ST_AXIS_EN[1]))))
        print(f"  side convention: aligned to LiDAR/ST S-axis bearing {axis_bearing:.2f} deg")
    else:
        print("  side convention: sparmitt file order (no LiDAR/ST axis available)")

    model = load_da3_model()
    rows: list[dict] = []
    diag_targets = set(np.linspace(0, len(panos) - 1, CFG.diag_count, dtype=int).tolist())
    t0 = time.time()
    for target_idx, target in panos.iterrows():
        rail_qc = rail_gauge_quality(target, sp, paths)
        for side in ("right", "left"):
            points_by_view: list[np.ndarray] = []
            crop_records: list[dict] = []
            source_indices_all: list[int] = []
            half_windows_used: list[int] = []
            depth_paths: list[str] = []

            for anchor_t in anchors:
                pred, crop_paths, c2w_global, source_indices, half_window_used, depth_path = infer_or_load_cached(
                    model, panos, sp, target_idx, str(target["Filename"]), side, paths, anchor_t_m=anchor_t
                )
                anchor_points = backproject_depths_by_view(pred.depth, pred.intrinsics, c2w_global)
                points_by_view.extend(anchor_points)
                source_indices_all.extend(source_indices)
                half_windows_used.append(int(half_window_used))
                depth_paths.append(str(depth_path))
                for view_pos, (crop_path, source_idx) in enumerate(zip(crop_paths, source_indices)):
                    crop_records.append(
                        {
                            "crop_path": crop_path,
                            "source_index": int(source_idx),
                            "anchor_t_m": float(anchor_t),
                            "depth": pred.depth[view_pos],
                        }
                    )

            lidar_match = nearest_lidar_row(lidar, float(target["S"]), side)
            profile = profile_from_multiview_points(
                points_by_view,
                target,
                sp,
                side,
                target_ditch_t_m=lidar_match["bottom_t_m"],
            )
            profile_path = paths["profiles"] / f"{target['Filename']}_{side}_multiview_profile.csv"
            save_profile_csv(profile_path, profile)

            selected_t = profile["visible_surface_T_m"]
            if not np.isfinite(selected_t):
                selected_t = profile["diagnostic_max_T_m"]
            if not np.isfinite(selected_t):
                selected_t = CFG.target_anchor_t_m
            target_records = [rec for rec in crop_records if rec["source_index"] == int(target_idx)]
            candidate_records = target_records if target_records else crop_records
            selected_record = min(candidate_records, key=lambda rec: abs(float(rec["anchor_t_m"]) - float(selected_t)))
            target_crop_path = selected_record["crop_path"]
            target_depth_map = selected_record["depth"]
            source_indices_unique = list(dict.fromkeys(source_indices_all))
            half_window_used_text = " ".join(str(v) for v in sorted(set(half_windows_used), reverse=True))
            visibility = classify_visibility(profile, target_crop_path)
            combined_metric_quality = 0.75 * profile["reference_quality_score"] + 0.25 * rail_qc["rail_gauge_quality_score"]
            metric_evaluation_eligible = bool(
                profile["has_metric_reference"]
                and combined_metric_quality >= CFG.min_reference_quality_score
                and visibility["visible_surface_metric_candidate"]
            )
            if target_idx in diag_targets:
                fig_path = paths["figures"] / f"multiview_{target['Filename']}_{side}.png"
                save_diagnostic(
                    fig_path,
                    target_crop_path,
                    target_depth_map,
                    profile,
                    f"{target['Filename']} {side}, S={float(target['S']):.1f} m",
                )

            rows.append(
                {
                    "filename": target["Filename"],
                    "side": side,
                    "S_cam": float(target["S"]),
                    "target_index": int(target_idx),
                    "source_indices": " ".join(str(i) for i in source_indices_unique),
                    "n_views": len(source_indices_unique),
                    "n_virtual_views": len(points_by_view),
                    "anchor_t_values_m": " ".join(f"{v:.2f}" for v in anchors),
                    "selected_anchor_t_m": float(selected_record["anchor_t_m"]),
                    "half_window_used": half_window_used_text,
                    "metric_evaluation_eligible": metric_evaluation_eligible,
                    "combined_metric_quality_score": float(combined_metric_quality),
                    "visible_surface_depth_m": profile["visible_surface_depth_m"],
                    "visible_surface_T_m": profile["visible_surface_T_m"],
                    "visible_surface_search_zone": profile["visible_surface_search_zone"],
                    "visible_surface_search_zone_used_fallback": profile["visible_surface_search_zone_used_fallback"],
                    "primary_visible_surface_depth_m": profile["primary_visible_surface_depth_m"],
                    "primary_visible_surface_T_m": profile["primary_visible_surface_T_m"],
                    "primary_profile_local_prominence_m": profile["primary_profile_local_prominence_m"],
                    "near_track_visible_surface_depth_m": profile["near_track_visible_surface_depth_m"],
                    "near_track_visible_surface_T_m": profile["near_track_visible_surface_T_m"],
                    "near_track_profile_local_prominence_m": profile["near_track_profile_local_prominence_m"],
                    "diagnostic_max_depth_m": profile["diagnostic_max_depth_m"],
                    "diagnostic_max_T_m": profile["diagnostic_max_T_m"],
                    "lidar_guided_visible_surface_depth_m": profile["lidar_guided_visible_surface_depth_m"],
                    "lidar_guided_visible_surface_T_m": profile["lidar_guided_visible_surface_T_m"],
                    "track_bearing_deg": profile["track_bearing_deg"],
                    "z_rail": profile["z_rail"],
                    "datum_shift_m": profile["datum_shift_m"],
                    "datum_shift_status": profile["datum_shift_status"],
                    "reference_quality_score": profile["reference_quality_score"],
                    "has_metric_reference": profile["has_metric_reference"],
                    "n_reference_views": profile["n_reference_views"],
                    "shoulder_z_iqr_m": profile["shoulder_z_iqr_m"],
                    "shoulder_points": profile["shoulder_points"],
                    "rail_gauge_quality_score": rail_qc["rail_gauge_quality_score"],
                    "rail_gauge_status": rail_qc["rail_gauge_status"],
                    "rail_gauge_est_m": rail_qc["rail_gauge_est_m"],
                    "rail_gauge_error_m": rail_qc["rail_gauge_error_m"],
                    "rail_edge_support": rail_qc["rail_edge_support"],
                    "rail_projected_samples": rail_qc["rail_projected_samples"],
                    "rail_crop_path": rail_qc["rail_crop_path"],
                    "visibility_class": visibility["visibility_class"],
                    "visibility_confidence": visibility["visibility_confidence"],
                    "bottom_visible_candidate": visibility["bottom_visible_candidate"],
                    "ditch_presence_class": visibility["ditch_presence_class"],
                    "visible_surface_metric_candidate": visibility["visible_surface_metric_candidate"],
                    "metric_exclusion_reason": visibility["metric_exclusion_reason"],
                    "vegetation_ratio": visibility["vegetation_ratio"],
                    "profile_local_prominence_m": visibility["profile_local_prominence_m"],
                    "profile_trend_corr": visibility["profile_trend_corr"],
                    "peak_at_search_boundary": visibility["peak_at_search_boundary"],
                    "fused_views_at_peak": visibility["fused_views_at_peak"],
                    "zone_points": profile["zone_points"],
                    "valid_bins": profile["valid_bins"],
                    "lidar_s": lidar_match["s"],
                    "lidar_depth_m": lidar_match["depth_m"],
                    "lidar_bottom_T_m": lidar_match["bottom_t_m"],
                    "lidar_span_x0_m": lidar_match["span_x0_m"],
                    "lidar_span_x1_m": lidar_match["span_x1_m"],
                    "lidar_priority": lidar_match["priority"],
                    "profile_path": str(profile_path),
                    "depth_path": " ; ".join(depth_paths),
                    "target_crop_path": target_crop_path,
                }
            )

        if (target_idx + 1) % 5 == 0 or target_idx == len(panos) - 1:
            print(f"  [{target_idx + 1:02d}/{len(panos)}] processed in {time.time() - t0:.1f} s")

    result = pd.DataFrame(rows)
    out_csv = CFG.out_dir / "multiview_visible_surface.csv"
    summary_path = CFG.out_dir / "multiview_visible_surface_summary.json"
    result.to_csv(out_csv, index=False)
    both = result.dropna(subset=["visible_surface_depth_m", "lidar_depth_m"])
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "n_panoramas": int(len(panos)),
        "n_side_views": int(len(result)),
        "n_with_lidar_depth": int(len(both)),
        "visible_surface_depth_m": result["visible_surface_depth_m"].describe().to_dict(),
        "diagnostic_max_depth_m": result["diagnostic_max_depth_m"].describe().to_dict(),
        "lidar_guided_visible_surface_depth_m": result["lidar_guided_visible_surface_depth_m"].describe().to_dict(),
        "datum_shift_m": result["datum_shift_m"].describe().to_dict(),
        "datum_shift_status_counts": result["datum_shift_status"].value_counts(dropna=False).to_dict(),
        "reference_quality_score": result["reference_quality_score"].describe().to_dict(),
        "metric_evaluation_eligible_counts": result["metric_evaluation_eligible"].value_counts(dropna=False).to_dict(),
        "rail_gauge_status_counts": result["rail_gauge_status"].value_counts(dropna=False).to_dict(),
        "visibility_class_counts": result["visibility_class"].value_counts(dropna=False).to_dict(),
        "ditch_presence_class_counts": result["ditch_presence_class"].value_counts(dropna=False).to_dict(),
        "side_convention": {
            "frame_tag": CFG.side_frame_tag,
            "st_meta_json": str(CFG.st_meta_json),
            "lidar_st_axis_en": LIDAR_ST_AXIS_EN.tolist() if LIDAR_ST_AXIS_EN is not None else None,
            "local_frame_tangent_flip_calls": int(LOCAL_FRAME_FLIP_COUNT),
        },
    }
    if len(both) > 0:
        err = both["visible_surface_depth_m"] - both["lidar_depth_m"]
        summary["comparison_to_lidar"] = {
            "n": int(len(both)),
            "bias_m": float(err.mean()),
            "rmse_m": float(np.sqrt(np.mean(err.to_numpy() ** 2))),
            "mae_m": float(np.mean(np.abs(err.to_numpy()))),
        }
        guided = both.dropna(subset=["lidar_guided_visible_surface_depth_m"])
        if len(guided) > 0:
            guided_err = guided["lidar_guided_visible_surface_depth_m"] - guided["lidar_depth_m"]
            summary["comparison_to_lidar_at_lidar_bottom_T"] = {
                "n": int(len(guided)),
                "bias_m": float(guided_err.mean()),
                "rmse_m": float(np.sqrt(np.mean(guided_err.to_numpy() ** 2))),
                "mae_m": float(np.mean(np.abs(guided_err.to_numpy()))),
            }
        referenced = both[both["has_metric_reference"]].copy()
        if len(referenced) > 0:
            ref_err = referenced["visible_surface_depth_m"] - referenced["lidar_depth_m"]
            summary["comparison_to_lidar_with_ballast_reference"] = {
                "n": int(len(referenced)),
                "bias_m": float(ref_err.mean()),
                "rmse_m": float(np.sqrt(np.mean(ref_err.to_numpy() ** 2))),
                "mae_m": float(np.mean(np.abs(ref_err.to_numpy()))),
            }
            ref_guided = referenced.dropna(subset=["lidar_guided_visible_surface_depth_m"])
            if len(ref_guided) > 0:
                ref_guided_err = ref_guided["lidar_guided_visible_surface_depth_m"] - ref_guided["lidar_depth_m"]
                summary["comparison_to_lidar_at_lidar_bottom_T_with_ballast_reference"] = {
                    "n": int(len(ref_guided)),
                    "bias_m": float(ref_guided_err.mean()),
                    "rmse_m": float(np.sqrt(np.mean(ref_guided_err.to_numpy() ** 2))),
                    "mae_m": float(np.mean(np.abs(ref_guided_err.to_numpy()))),
                }
        eligible = both[both["metric_evaluation_eligible"]].copy()
        if len(eligible) > 0:
            eligible_err = eligible["visible_surface_depth_m"] - eligible["lidar_depth_m"]
            summary["comparison_to_lidar_metric_eligible"] = {
                "n": int(len(eligible)),
                "bias_m": float(eligible_err.mean()),
                "rmse_m": float(np.sqrt(np.mean(eligible_err.to_numpy() ** 2))),
                "mae_m": float(np.mean(np.abs(eligible_err.to_numpy()))),
            }
            eligible_guided = eligible.dropna(subset=["lidar_guided_visible_surface_depth_m"])
            if len(eligible_guided) > 0:
                eligible_guided_err = eligible_guided["lidar_guided_visible_surface_depth_m"] - eligible_guided["lidar_depth_m"]
                summary["comparison_to_lidar_at_lidar_bottom_T_metric_eligible"] = {
                    "n": int(len(eligible_guided)),
                    "bias_m": float(eligible_guided_err.mean()),
                    "rmse_m": float(np.sqrt(np.mean(eligible_guided_err.to_numpy() ** 2))),
                    "mae_m": float(np.mean(np.abs(eligible_guided_err.to_numpy()))),
                }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nOutputs")
    print(f"  CSV     : {out_csv}")
    print(f"  summary : {summary_path}")
    print(f"  figures : {paths['figures']}")
    if len(both) > 0:
        cmp_ = summary["comparison_to_lidar"]
        print(
            f"  automatic window vs LiDAR: n={cmp_['n']}, bias={cmp_['bias_m']:+.3f} m, "
            f"RMSE={cmp_['rmse_m']:.3f} m, MAE={cmp_['mae_m']:.3f} m"
        )
        if "comparison_to_lidar_at_lidar_bottom_T" in summary:
            cmp2 = summary["comparison_to_lidar_at_lidar_bottom_T"]
            print(
                f"  validation T vs LiDAR: n={cmp2['n']}, bias={cmp2['bias_m']:+.3f} m, "
                f"RMSE={cmp2['rmse_m']:.3f} m, MAE={cmp2['mae_m']:.3f} m"
            )
        if "comparison_to_lidar_with_ballast_reference" in summary:
            cmp3 = summary["comparison_to_lidar_with_ballast_reference"]
            print(
                f"  ballast-referenced automatic subset: n={cmp3['n']}, "
                f"bias={cmp3['bias_m']:+.3f} m, RMSE={cmp3['rmse_m']:.3f} m, MAE={cmp3['mae_m']:.3f} m"
            )
        if "comparison_to_lidar_metric_eligible" in summary:
            cmp4 = summary["comparison_to_lidar_metric_eligible"]
            print(
                f"  metric-eligible subset: n={cmp4['n']}, "
                f"bias={cmp4['bias_m']:+.3f} m, RMSE={cmp4['rmse_m']:.3f} m, MAE={cmp4['mae_m']:.3f} m"
            )


if __name__ == "__main__":
    main()


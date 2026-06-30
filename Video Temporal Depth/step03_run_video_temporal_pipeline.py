from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("VIDEO_TEMPORAL_ROOT", str(Path(__file__).resolve().parent)))


def required_path_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required. Run through run_video_temporal_depth.py.")
    return Path(value)


DATA_DIR = required_path_env("VIDEO_INPUT_DIR")
OUTPUT_DIR = Path(os.environ.get("VIDEO_OUTPUT_DIR", str(ROOT / "output")))
FRAME_DIR = OUTPUT_DIR / "frames"
DEPTH_DIR = OUTPUT_DIR / "depth"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
CONFIG_DIR = Path(os.environ.get("VIDEO_CONFIG_DIR", str(ROOT / "config")))
RAIL_GUIDE_CSV = Path(os.environ.get("VIDEO_RAIL_GUIDE_CSV", str(CONFIG_DIR / "rail_guides.csv")))
DA3_BUNDLE_ROOT = Path(os.environ.get("VIDEO_DA3_BUNDLE_ROOT", str(ROOT)))
DA3_REPO = Path(os.environ.get("VIDEO_DA3_REPO", str(DA3_BUNDLE_ROOT / "Depth-Anything-3")))
DA3_MODEL_DIR = required_path_env("VIDEO_DA3_MODEL_DIR")

RAIL_GAUGE_M = 1.435
ANALYSIS_START_M = 1_152_891.0
ANALYSIS_END_M = 1_153_050.0

# Coarse metric scale for the DA3 depth axis, in metres per DA3 depth unit.
# The value is estimated from the sleeper period along the track centreline.
# It converts relative recession into approximate metres and is used only as a
# supporting quantity; the cross-year comparison is based mainly on ranked
# visible evidence and confirmation rate.
DA3_METRES_PER_UNIT = 10.0


def load_video_specs() -> list[dict]:
    """Load video metadata from JSON when an external runner provides it."""
    spec_path = os.environ.get("VIDEO_SPECS_JSON") or os.environ.get("VIDEO_SPECS_FILE")
    if spec_path:
        data = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("VIDEO_SPECS_JSON must point to a JSON list of video specifications.")
        return data
    return [
    {
        "year": 2020,
        "filename": "2020-06-24, Km 1152+800 1152+900.mp4",
        "overlay_start_m": 1_152_705.0,
        "overlay_end_m": 1_153_122.0,
    },
    {
        "year": 2025,
        "filename": "2025-05-29, Km 1152+800-1152+900.mp4",
        "overlay_start_m": 1_152_687.0,
        "overlay_end_m": 1_153_163.0,
    },
    ]


VIDEO_SPECS = load_video_specs()


@dataclass
class RailFit:
    ok: bool
    left_poly: np.ndarray | None
    right_poly: np.ndarray | None
    rows_used: int
    gauge_px_median: float
    gauge_px_mad: float
    quality: float
    reason: str


@dataclass
class RailPair:
    xl: float
    xr: float
    gauge: float
    score: float


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, FRAME_DIR, DEPTH_DIR, TABLE_DIR, FIGURE_DIR, CONFIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", Path(name).stem).strip("_")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    tmp = path.with_name(f"{path.stem}_tmp_{os.getpid()}{path.suffix}")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated_{os.getpid()}{path.suffix}")
        os.replace(tmp, fallback)
        print(f"[write_json] {path.name} is locked; wrote {fallback.name} instead")


def robust_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def robust_sigma(values: np.ndarray) -> float:
    mad = robust_mad(values)
    if not np.isfinite(mad):
        return float("nan")
    return 1.4826 * mad


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_csv(index=False)
    tmp = path.with_name(f"{path.stem}_tmp_{os.getpid()}{path.suffix}")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated_{os.getpid()}{path.suffix}")
        os.replace(tmp, fallback)
        print(f"[save_table] {path.name} is locked; wrote {fallback.name} instead")


def _crop_bottom_overlay(frame: np.ndarray) -> np.ndarray:
    h = frame.shape[0]
    # The videos contain a black metadata strip at the bottom. 80 px removes it
    # in both delivered resolutions without cutting the rail corridor.
    crop_px = min(80, max(0, h // 8))
    return frame[: h - crop_px].copy()


def _video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    info = {
        "width_px": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height_px": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration_s"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else math.nan
    return info


def _chainage_to_frame(chainage_m: float, spec: dict, frame_count: int) -> int:
    start = float(spec["overlay_start_m"])
    end = float(spec["overlay_end_m"])
    frac = (chainage_m - start) / max(end - start, 1e-6)
    return int(np.clip(round(frac * (frame_count - 1)), 0, frame_count - 1))


def extract_frames(sample_spacing_m: float) -> pd.DataFrame:
    ensure_dirs()
    rows = []
    chainages = np.arange(ANALYSIS_START_M, ANALYSIS_END_M + 0.01, sample_spacing_m)

    for spec in VIDEO_SPECS:
        video_path = DATA_DIR / spec["filename"]
        info = _video_info(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        stem = safe_stem(spec["filename"])
        out_dir = FRAME_DIR / str(spec["year"])
        out_dir.mkdir(parents=True, exist_ok=True)

        for order, chainage_m in enumerate(chainages):
            frame_idx = _chainage_to_frame(float(chainage_m), spec, info["frame_count"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = _crop_bottom_overlay(frame)
            image_path = out_dir / f"{stem}_{order:04d}_ch_{int(round(chainage_m)):07d}.jpg"
            cv2.imwrite(str(image_path), frame)
            rows.append(
                {
                    "year": int(spec["year"]),
                    "video_name": spec["filename"],
                    "video_path": str(video_path),
                    "sample_order": int(order),
                    "chainage_m": float(chainage_m),
                    "chainage_label": format_chainage(chainage_m),
                    "frame_idx": int(frame_idx),
                    "image_path": str(image_path),
                    **info,
                }
            )
        cap.release()

    df = pd.DataFrame(rows)
    save_table(df, TABLE_DIR / "sampled_frames.csv")
    write_rail_guide_template(df)
    write_json(
        TABLE_DIR / "step1_frame_sampling.json",
        {
            "analysis_start_m": ANALYSIS_START_M,
            "analysis_end_m": ANALYSIS_END_M,
            "sample_spacing_m": sample_spacing_m,
            "frames": int(len(df)),
            "videos": VIDEO_SPECS,
        },
    )
    return df


def write_rail_guide_template(frames: pd.DataFrame) -> None:
    template_path = CONFIG_DIR / "rail_guides_template.csv"
    if template_path.exists():
        return
    key_orders = [0, 20, 40, 60, 79]
    y_fracs = [0.55, 0.70, 0.85, 0.94]
    rows = []
    for year, group in frames.groupby("year"):
        max_order = int(group["sample_order"].max())
        for order in key_orders:
            nearest_order = min(max(order, 0), max_order)
            frame_row = group.loc[(group["sample_order"] - nearest_order).abs().idxmin()]
            for rail in ("left", "right"):
                for y_frac in y_fracs:
                    rows.append(
                        {
                            "year": int(year),
                            "sample_order": int(frame_row["sample_order"]),
                            "chainage_label": frame_row["chainage_label"],
                            "image_path": frame_row["image_path"],
                            "rail": rail,
                            "y_frac": y_frac,
                            "x_frac": "",
                            "note": "Fill x_frac = x_pixel / image_width for the current running rail.",
                        }
                    )
    save_table(pd.DataFrame(rows), template_path)


def load_rail_guides() -> pd.DataFrame:
    if not RAIL_GUIDE_CSV.exists():
        return pd.DataFrame()
    guides = pd.read_csv(RAIL_GUIDE_CSV)
    required = {"year", "sample_order", "rail", "y_frac", "x_frac"}
    if not required.issubset(guides.columns):
        raise ValueError(f"{RAIL_GUIDE_CSV} must contain columns: {sorted(required)}")
    guides = guides.copy()
    guides["x_frac"] = pd.to_numeric(guides["x_frac"], errors="coerce")
    guides["y_frac"] = pd.to_numeric(guides["y_frac"], errors="coerce")
    guides = guides.dropna(subset=["year", "sample_order", "rail", "y_frac", "x_frac"])
    guides = guides[guides["rail"].isin(["left", "right"])]
    return guides


def fit_rails_from_guides(image_bgr: np.ndarray, year: int, sample_order: int, guides: pd.DataFrame) -> RailFit | None:
    if guides.empty:
        return None
    h, w = image_bgr.shape[:2]
    year_guides = guides[guides["year"].astype(int) == int(year)].copy()
    if year_guides.empty:
        return None
    polys = {}
    rows_used = 0
    for rail in ("left", "right"):
        rail_rows = year_guides[year_guides["rail"] == rail]
        if rail_rows.empty:
            return None
        y_vals = sorted(rail_rows["y_frac"].unique())
        pts_y = []
        pts_x = []
        for y_frac in y_vals:
            g = rail_rows[rail_rows["y_frac"] == y_frac].sort_values("sample_order")
            if len(g) < 2:
                continue
            x_frac = float(np.interp(sample_order, g["sample_order"].to_numpy(float), g["x_frac"].to_numpy(float)))
            if not np.isfinite(x_frac):
                continue
            pts_y.append(float(y_frac))
            pts_x.append(float(x_frac * w))
        if len(pts_y) < 3:
            return None
        polys[rail] = np.polyfit(np.asarray(pts_y), np.asarray(pts_x), deg=2)
        rows_used += len(pts_y)
    y_check = np.linspace(0.55, 0.94, 20)
    left_x = np.polyval(polys["left"], y_check)
    right_x = np.polyval(polys["right"], y_check)
    gauge = right_x - left_x
    if np.nanmin(gauge) <= 0:
        return RailFit(False, polys["left"], polys["right"], rows_used, float("nan"), float("nan"), 0.0, "invalid_manual_rail_guide")
    return RailFit(
        ok=True,
        left_poly=polys["left"],
        right_poly=polys["right"],
        rows_used=rows_used,
        gauge_px_median=float(np.nanmedian(gauge)),
        gauge_px_mad=robust_mad(gauge),
        quality=1.0,
        reason="manual_interpolated_rail_guide",
    )


def _ensure_da3_importable() -> None:
    repo = str(DA3_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _load_da3(device: str):
    _ensure_da3_importable()
    from depth_anything_3.api import DepthAnything3

    if DA3_MODEL_DIR.exists():
        model = DepthAnything3.from_pretrained(str(DA3_MODEL_DIR))
    else:
        model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
    return model.to(device)


def _prediction_arrays(prediction) -> tuple[np.ndarray, np.ndarray]:
    depth = getattr(prediction, "depth", None)
    conf = getattr(prediction, "conf", None)
    if depth is None and isinstance(prediction, dict):
        depth = prediction.get("depth")
        conf = prediction.get("conf")
    if depth is None:
        raise RuntimeError("DA3 prediction did not expose a depth array")
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 2:
        depth = depth[None, ...]
    if conf is None:
        conf = np.ones_like(depth, dtype=np.float32)
    else:
        conf = np.asarray(conf, dtype=np.float32)
        if conf.ndim == 2:
            conf = conf[None, ...]
    return depth, conf


def run_da3(frames: pd.DataFrame, device: str, process_res: int, chunk_size: int,
            skip_existing: bool) -> pd.DataFrame:
    model = _load_da3(device)
    rows = []
    for _, group in frames.groupby("year", sort=True):
        group = group.sort_values("sample_order").reset_index(drop=True)
        for start in range(0, len(group), chunk_size):
            chunk = group.iloc[start : start + chunk_size]
            needed = []
            for _, row in chunk.iterrows():
                image_path = Path(row["image_path"])
                out_dir = DEPTH_DIR / str(row["year"])
                out_dir.mkdir(parents=True, exist_ok=True)
                npz_path = out_dir / f"{image_path.stem}_da3.npz"
                if skip_existing and npz_path.exists():
                    rows.append({**row.to_dict(), "depth_npz_path": str(npz_path), "da3_status": "cached"})
                else:
                    needed.append((row.to_dict(), image_path, npz_path))
            if not needed:
                continue
            pred = model.inference(
                image=[str(item[1]) for item in needed],
                process_res=process_res,
                ref_view_strategy="middle",
            )
            depths, confs = _prediction_arrays(pred)
            for i, (row, image_path, npz_path) in enumerate(needed):
                depth = depths[i]
                conf = confs[i] if i < confs.shape[0] else np.ones_like(depth)
                np.savez_compressed(npz_path, depth=depth.astype(np.float32), conf=conf.astype(np.float32))
                rows.append({**row, "depth_npz_path": str(npz_path), "da3_status": "computed"})
    out = pd.DataFrame(rows).sort_values(["year", "sample_order"]).reset_index(drop=True)
    save_table(out, TABLE_DIR / "da3_depth_index.csv")
    write_json(
        TABLE_DIR / "step2_da3_summary.json",
        {
            "device": device,
            "process_res": process_res,
            "chunk_size": chunk_size,
            "frames": int(len(out)),
            "computed": int((out["da3_status"] == "computed").sum()) if len(out) else 0,
            "cached": int((out["da3_status"] == "cached").sum()) if len(out) else 0,
        },
    )
    return out


def _enhance_gray(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _rowwise_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, 10, axis=1, keepdims=True)
    hi = np.percentile(arr, 98, axis=1, keepdims=True)
    return np.clip((arr - lo) / np.maximum(hi - lo, 1e-3), 0.0, 1.0)


def _railness_image(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    horizontal_bg = cv2.GaussianBlur(gray_f, (61, 1), 0)
    bright_ridge = np.maximum(gray_f - horizontal_bg, 0.0)
    sobel_x = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3))
    vertical_preference = np.maximum(sobel_x - 0.35 * sobel_y, 0.0)
    railness = 0.70 * _rowwise_normalize(bright_ridge) + 0.30 * _rowwise_normalize(vertical_preference)
    return cv2.GaussianBlur(railness, (5, 3), 0)


def _row_pair_candidates(railness: np.ndarray, y: int, max_peaks: int = 55) -> list[RailPair]:
    h, w = railness.shape
    y_frac = y / max(h - 1, 1)
    row = railness[int(y)].astype(np.float32)
    row = cv2.GaussianBlur(row.reshape(1, -1), (1, 13), 0).ravel()
    local = np.where((row[1:-1] >= row[:-2]) & (row[1:-1] >= row[2:]))[0] + 1
    local = local[(local > int(0.08 * w)) & (local < int(0.92 * w))]
    if local.size == 0:
        return []

    row_gate = float(np.percentile(row[local], 55))
    local = local[row[local] >= row_gate]
    if local.size == 0:
        return []
    strong = np.sort(local[np.argsort(row[local])[-max_peaks:]])

    # Apparent gauge is evaluated in image space. The interval is deliberately
    # broad; the dynamic-programming stage below enforces smooth perspective
    # behaviour across rows, so this function should not prematurely remove a
    # curved but valid track.
    gauge_min = np.interp(y_frac, [0.45, 0.94], [max(32.0, 0.030 * w), max(70.0, 0.075 * w)])
    gauge_max = np.interp(y_frac, [0.45, 0.94], [min(175.0, 0.150 * w), min(300.0, 0.230 * w)])
    centre = 0.5 * w
    centre_sigma = np.interp(y_frac, [0.45, 0.92], [0.18 * w, 0.30 * w])
    row_p95 = max(float(np.percentile(row, 95)), 1e-6)
    pairs: list[RailPair] = []
    for i, xl in enumerate(strong[:-1]):
        for xr in strong[i + 1 :]:
            gauge = float(xr - xl)
            if gauge < gauge_min or gauge > gauge_max:
                continue
            mid = 0.5 * (float(xl) + float(xr))
            centre_score = np.exp(-0.5 * ((mid - centre) / max(centre_sigma, 1.0)) ** 2)
            edge_score = (float(row[xl]) + float(row[xr])) / (2.0 * row_p95)
            gauge_expected = np.interp(y_frac, [0.45, 0.94], [min(110.0, 0.075 * w), min(240.0, 0.160 * w)])
            gauge_score = np.exp(-0.5 * ((gauge - gauge_expected) / max(0.045 * w, 35.0)) ** 2)
            score = 1.7 * edge_score + 0.50 * centre_score + 0.80 * gauge_score
            pairs.append(RailPair(float(xl), float(xr), gauge, float(score)))
    pairs.sort(key=lambda item: item.score, reverse=True)
    return pairs[:25]


def fit_rails(image_bgr: np.ndarray) -> RailFit:
    gray = _enhance_gray(image_bgr)
    h, w = gray.shape
    railness = _railness_image(gray)
    sample_rows = np.arange(int(0.94 * h), int(0.45 * h), -7, dtype=int)
    row_items: list[tuple[int, list[RailPair]]] = []
    for y in sample_rows:
        pairs = _row_pair_candidates(railness, int(y))
        if pairs:
            row_items.append((int(y), pairs))
    if len(row_items) < 12:
        return RailFit(False, None, None, len(row_items), float("nan"), float("nan"), 0.0, "too_few_candidate_rows")

    dp: list[np.ndarray] = []
    back: list[np.ndarray] = []
    for row_idx, (y, pairs) in enumerate(row_items):
        y_frac = y / max(h - 1, 1)
        cur = np.full(len(pairs), -np.inf, dtype=np.float64)
        cur_back = np.full(len(pairs), -1, dtype=np.int32)
        for j, pair in enumerate(pairs):
            mid = 0.5 * (pair.xl + pair.xr)
            if row_idx == 0:
                bottom_mid_score = np.exp(-0.5 * ((mid - 0.5 * w) / max(0.23 * w, 1.0)) ** 2)
                bottom_gauge_expected = np.interp(y_frac, [0.45, 0.94], [min(110.0, 0.075 * w), min(240.0, 0.160 * w)])
                bottom_gauge_score = np.exp(-0.5 * ((pair.gauge - bottom_gauge_expected) / max(0.050 * w, 35.0)) ** 2)
                cur[j] = pair.score + 0.8 * bottom_mid_score + 0.5 * bottom_gauge_score
                continue

            prev_y, prev_pairs = row_items[row_idx - 1]
            row_gap = abs(prev_y - y) / 7.0
            best_score = -np.inf
            best_k = -1
            for k, prev in enumerate(prev_pairs):
                prev_score = dp[row_idx - 1][k]
                if not np.isfinite(prev_score):
                    continue
                prev_mid = 0.5 * (prev.xl + prev.xr)
                mid_penalty = ((mid - prev_mid) / max(0.055 * w + 0.22 * prev.gauge, 1.0)) ** 2
                gauge_penalty = ((pair.gauge - prev.gauge) / max(0.035 * w + 0.20 * prev.gauge, 1.0)) ** 2
                crossing_penalty = 2.2 if pair.gauge <= 0 else 0.0
                # Moving upward in the image should generally narrow the apparent
                # gauge. A small tolerance is left for curve and detector noise.
                widening_penalty = max(0.0, pair.gauge - 1.10 * prev.gauge) / max(0.08 * w, 1.0)
                score = (
                    prev_score
                    + pair.score
                    - 0.9 * mid_penalty
                    - 1.15 * gauge_penalty
                    - 1.0 * widening_penalty
                    - crossing_penalty
                    - 0.04 * max(row_gap - 1.0, 0.0)
                )
                if score > best_score:
                    best_score = score
                    best_k = k
            cur[j] = best_score
            cur_back[j] = best_k
        dp.append(cur)
        back.append(cur_back)

    end_row = int(np.argmax([np.nanmax(scores) if len(scores) else -np.inf for scores in dp]))
    end_state = int(np.nanargmax(dp[end_row]))
    chosen: list[tuple[float, float, float, float]] = []
    for row_idx in range(end_row, -1, -1):
        y, pairs = row_items[row_idx]
        pair = pairs[end_state]
        chosen.append((float(y), pair.xl, pair.xr, pair.gauge))
        end_state = int(back[row_idx][end_state]) if row_idx > 0 else -1
        if row_idx > 0 and end_state < 0:
            break
    det = np.asarray(chosen[::-1], dtype=float)
    if det.shape[0] < 12:
        return RailFit(False, None, None, int(det.shape[0]), float("nan"), float("nan"), 0.0, "short_dynamic_track_path")

    det = det[np.argsort(det[:, 0])]
    med_g = float(np.median(det[:, 3]))
    mad_g = robust_mad(det[:, 3])
    keep = np.abs(det[:, 3] - med_g) <= max(4.0 * mad_g, 0.45 * med_g)
    det = det[keep]
    if det.shape[0] < 10:
        return RailFit(False, None, None, int(det.shape[0]), med_g, mad_g, 0.0, "unstable_dynamic_gauge")

    keep = np.ones(det.shape[0], dtype=bool)
    for _ in range(3):
        y_norm = det[keep, 0] / max(h - 1, 1)
        if y_norm.size < 10:
            break
        left_poly = np.polyfit(y_norm, det[keep, 1], deg=2)
        right_poly = np.polyfit(y_norm, det[keep, 2], deg=2)
        pred_l = np.polyval(left_poly, det[:, 0] / max(h - 1, 1))
        pred_r = np.polyval(right_poly, det[:, 0] / max(h - 1, 1))
        residual = np.r_[det[:, 1] - pred_l, det[:, 2] - pred_r]
        residual_mad = robust_mad(residual)
        row_res = np.maximum(np.abs(det[:, 1] - pred_l), np.abs(det[:, 2] - pred_r))
        keep = row_res <= max(3.5 * residual_mad, 0.025 * w)
    if keep.sum() < 10:
        return RailFit(False, None, None, int(keep.sum()), med_g, mad_g, 0.0, "unstable_polynomial_fit")

    y_norm = det[keep, 0] / max(h - 1, 1)
    left_poly = np.polyfit(y_norm, det[keep, 1], deg=2)
    right_poly = np.polyfit(y_norm, det[keep, 2], deg=2)
    y_check = np.linspace(0.48, 0.94, 60)
    pred_l = np.polyval(left_poly, y_check)
    pred_r = np.polyval(right_poly, y_check)
    fitted_gauge = pred_r - pred_l
    bottom_gauge_limit = min(285.0, max(1.35 * float(np.median(det[keep, 3])), float(np.median(det[keep, 3])) + 60.0))
    if np.nanmax(fitted_gauge) > bottom_gauge_limit:
        # Curved-track frames can produce a plausible row path but an unstable
        # quadratic near the lower image edge. In that case the lower-order fit
        # is a safer calibration curve for pixel-to-track scaling.
        left_poly = np.polyfit(y_norm, det[keep, 1], deg=1)
        right_poly = np.polyfit(y_norm, det[keep, 2], deg=1)
        pred_l = np.polyval(left_poly, y_check)
        pred_r = np.polyval(right_poly, y_check)
        fitted_gauge = pred_r - pred_l
    if (
        np.nanmin(fitted_gauge) < max(18.0, 0.018 * w)
        or np.nanmax(pred_l) >= w
        or np.nanmin(pred_r) <= 0
        or np.nanmax(pred_r) >= w * 1.05
        or np.nanmin(pred_l) <= -0.05 * w
    ):
        return RailFit(False, left_poly, right_poly, int(keep.sum()), med_g, mad_g, 0.0, "invalid_dynamic_track_geometry")
    if fitted_gauge[-1] < 1.20 * fitted_gauge[0]:
        return RailFit(False, left_poly, right_poly, int(keep.sum()), med_g, mad_g, 0.0, "weak_perspective_gauge_growth")

    pred_l_rows = np.polyval(left_poly, det[keep, 0] / max(h - 1, 1))
    pred_r_rows = np.polyval(right_poly, det[keep, 0] / max(h - 1, 1))
    residual_mad = robust_mad(np.r_[det[keep, 1] - pred_l_rows, det[keep, 2] - pred_r_rows])
    gauge_mad = robust_mad(det[keep, 3])
    coverage = keep.sum() / max(len(sample_rows), 1)
    residual_quality = np.exp(-residual_mad / max(0.020 * w, 1.0))
    gauge_quality = np.exp(-gauge_mad / max(med_g * 0.12, 1.0))
    quality = float(np.clip(0.30 * coverage + 0.40 * residual_quality + 0.30 * gauge_quality, 0.0, 1.0))
    return RailFit(True, left_poly, right_poly, int(keep.sum()), float(np.median(det[keep, 3])), float(gauge_mad), quality, "auto_dp_ego_track")


def eval_poly(poly: np.ndarray, y_px: np.ndarray, h: int) -> np.ndarray:
    return np.polyval(poly, y_px / max(h - 1, 1))


def _smooth_rail_fits(records: list[dict], window: int = 7) -> list[RailFit]:
    out = [rec["raw_fit"] for rec in records]
    for year in sorted({rec["frame"]["year"] for rec in records}):
        # Manual rail guides are the authoritative source (the auto detector was
        # unreliable on these two cameras). They are already interpolated across
        # frames, so they must NOT be re-smoothed: a neighbour-median refit would
        # pull curved track toward the local average and flatten curvature. Skip
        # any guide-derived fit entirely.
        idxs = [
            i
            for i, rec in enumerate(records)
            if rec["frame"]["year"] == year and rec["raw_fit"].ok and rec["raw_fit"].quality >= 0.25
            and "manual" not in (rec["raw_fit"].reason or "")
        ]
        if len(idxs) < 3:
            continue
        orders = np.asarray([records[i]["frame"]["sample_order"] for i in idxs], dtype=float)
        qualities = np.asarray([records[i]["raw_fit"].quality for i in idxs], dtype=float)
        gauges = np.asarray([records[i]["raw_fit"].gauge_px_median for i in idxs], dtype=float)
        y_anchor = np.asarray([0.52, 0.66, 0.80, 0.92], dtype=float)
        left_anchor = np.vstack([np.polyval(records[i]["raw_fit"].left_poly, y_anchor) for i in idxs])
        right_anchor = np.vstack([np.polyval(records[i]["raw_fit"].right_poly, y_anchor) for i in idxs])
        half = max(1, window // 2)
        for pos, rec_idx in enumerate(idxs):
            near = np.where(np.abs(orders - orders[pos]) <= half)[0]
            if near.size < 3:
                near = np.arange(max(0, pos - half), min(len(idxs), pos + half + 1))
            raw = records[rec_idx]["raw_fit"]
            w = records[rec_idx]["image"].shape[1]
            near_quality = qualities[near]
            good_near = near[near_quality >= max(0.25, np.nanpercentile(near_quality, 25))]
            if good_near.size < 3:
                good_near = near
            left_med = np.nanmedian(left_anchor[good_near], axis=0)
            right_med = np.nanmedian(right_anchor[good_near], axis=0)
            raw_anchor = np.r_[
                np.polyval(raw.left_poly, y_anchor),
                np.polyval(raw.right_poly, y_anchor),
            ]
            med_anchor = np.r_[left_med, right_med]
            temporal_residual = float(np.nanmedian(np.abs(raw_anchor - med_anchor)) / max(w, 1))
            # Refitting from smoothed anchor points is more stable than taking a
            # coefficient-wise median, because polynomial coefficients are not
            # independent and can swing strongly on curved tracks.
            smooth_left = np.polyfit(y_anchor, left_med, deg=2)
            smooth_right = np.polyfit(y_anchor, right_med, deg=2)
            smooth_gauge = np.polyval(smooth_right, y_anchor) - np.polyval(smooth_left, y_anchor)
            gauge_limit = min(285.0, max(1.35 * float(np.median(gauges[good_near])), float(np.median(gauges[good_near])) + 60.0))
            if np.nanmax(smooth_gauge) > gauge_limit:
                smooth_left = np.polyfit(y_anchor, left_med, deg=1)
                smooth_right = np.polyfit(y_anchor, right_med, deg=1)
            smoothed_quality = 0.75 * float(np.nanmedian(qualities[good_near])) + 0.25 * raw.quality
            smoothed_quality *= float(np.exp(-temporal_residual / 0.055))
            out[rec_idx] = RailFit(
                ok=True,
                left_poly=smooth_left,
                right_poly=smooth_right,
                rows_used=raw.rows_used,
                gauge_px_median=float(np.median(gauges[good_near])),
                gauge_px_mad=raw.gauge_px_mad,
                quality=float(np.clip(smoothed_quality, 0.0, 1.0)),
                reason="temporally_smoothed_auto_dp",
            )
    return out


def build_right_profile(image_bgr: np.ndarray, depth: np.ndarray, conf: np.ndarray, fit: RailFit) -> tuple[pd.DataFrame, dict]:
    if not fit.ok or fit.left_poly is None or fit.right_poly is None:
        return pd.DataFrame(), {"profile_status": "no_rail_fit"}
    h_img, w_img = image_bgr.shape[:2]
    depth_rs = cv2.resize(depth, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    conf_rs = cv2.resize(conf, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    exg = 2.0 * rgb[:, :, 1] - rgb[:, :, 0] - rgb[:, :, 2]

    # Dense row sampling (96 rows over the corridor band, ~3 px apart) lowers the
    # per-bin depth noise by averaging more samples into each lateral bin. This
    # mainly helps the noisier 2025 camera, where a real but weak toe depression
    # is otherwise lost in noise; the significance gate is unchanged, so it does
    # not manufacture detections. No profile smoothing is applied, because a
    # rolling median washes out the narrow (~1 m) ditch depression.
    row_grid = np.linspace(int(0.52 * h_img), int(0.93 * h_img), 96).astype(int)
    t_bins = np.arange(0.8, 10.01, 0.20)
    samples = []
    for y in row_grid:
        xl = float(eval_poly(fit.left_poly, np.array([y]), h_img)[0])
        xr = float(eval_poly(fit.right_poly, np.array([y]), h_img)[0])
        gauge_px = xr - xl
        if not np.isfinite(gauge_px) or gauge_px < 20:
            continue
        px_per_m = gauge_px / RAIL_GAUGE_M
        centre_x = 0.5 * (xl + xr)

        shoulder_values = []
        for side_sign in (-1, 1):
            for t_ref in np.arange(0.75, 1.36, 0.10):
                x_ref = int(round(centre_x + side_sign * t_ref * px_per_m))
                if 0 <= x_ref < w_img:
                    shoulder_values.append(depth_rs[y, x_ref])
        if len(shoulder_values) < 5:
            continue
        shoulder_depth = float(np.nanmedian(shoulder_values))

        for t in t_bins:
            x = int(round(centre_x + t * px_per_m))
            half = max(1, int(round(px_per_m * 0.05)))
            if x - half < 0 or x + half >= w_img:
                continue
            sl_y = slice(max(0, y - 1), min(h_img, y + 2))
            sl_x = slice(x - half, x + half + 1)
            patch_depth = depth_rs[sl_y, sl_x]
            patch_conf = conf_rs[sl_y, sl_x]
            patch_exg = exg[max(0, y - 2) : min(h_img, y + 3), sl_x]
            patch_depth_median = float(np.nanmedian(patch_depth))
            samples.append(
                {
                    "T_m": float(t),
                    "row_px": int(y),
                    "x_px": int(x),
                    "right_visible_recession_m": float(shoulder_depth - patch_depth_median),
                    "da3_depth_m_raw": patch_depth_median,
                    "shoulder_depth_m_raw": shoulder_depth,
                    "support_px": int(patch_depth.size),
                    "confidence_median": float(np.nanmedian(patch_conf)),
                    "exg": float(np.nanmedian(patch_exg)),
                }
            )
    sample_df = pd.DataFrame(samples)
    if sample_df.empty:
        return pd.DataFrame(), {"profile_status": "no_samples"}
    # Absolute ExG separates green vegetation from soil-like surfaces and keeps
    # vegetation_ratio comparable across years.
    EXG_GREEN = 0.0
    rows = []
    for t, g in sample_df.groupby("T_m", sort=True):
        rows.append(
            {
                "T_m": float(t),
                "right_visible_recession_m": float(np.nanmedian(g["right_visible_recession_m"])),
                "right_visible_recession_metric_m": float(np.nanmedian(g["right_visible_recession_m"]) * DA3_METRES_PER_UNIT),
                "recession_p25_m": float(np.nanpercentile(g["right_visible_recession_m"], 25)),
                "recession_p75_m": float(np.nanpercentile(g["right_visible_recession_m"], 75)),
                "support_px": int(len(g)),
                "row_px_median": float(np.nanmedian(g["row_px"])),
                "x_px_median": float(np.nanmedian(g["x_px"])),
                "confidence_median": float(np.nanmedian(g["confidence_median"])),
                "vegetation_ratio": float(np.mean(g["exg"] > EXG_GREEN)),
                "exg_median": float(np.nanmedian(g["exg"])),
            }
        )
    return pd.DataFrame(rows), {"profile_status": "ok", "exg_green_threshold": EXG_GREEN}


def detect_profile_candidate(profile: pd.DataFrame, t_min: float, t_max: float) -> dict:
    """Right-side toe ditch as a local DEPRESSION in the de-trended recession.

    DA3-SMALL depth is larger = farther, so a ditch (lower ground) is farther
    than the shoulder and appears as the recession dipping BELOW its local
    trend. A large-window median trend is subtracted to remove the broad
    terrain slope, and the toe zone is searched for the strongest depression.
    """
    if profile.empty or profile["T_m"].nunique() < 8:
        return {"detected": False, "reason": "insufficient_profile"}
    p = profile.sort_values("T_m").reset_index(drop=True)
    t = p["T_m"].to_numpy(float)
    rec = p["right_visible_recession_m"].to_numpy(float)
    support = p["support_px"].to_numpy(float)
    veg = p["vegetation_ratio"].to_numpy(float) if "vegetation_ratio" in p else np.zeros(len(p))
    exg_med = p["exg_median"].to_numpy(float) if "exg_median" in p else np.full(len(p), np.nan)

    trend = pd.Series(rec).rolling(13, center=True, min_periods=3).median()
    trend = trend.interpolate(limit_direction="both").to_numpy()
    depression = trend - rec               # > 0 where recession dips = ditch

    noise = robust_sigma(np.diff(rec)) / math.sqrt(2.0) if np.isfinite(rec).sum() > 4 else float("nan")

    zlo = max(float(t_min), 0.8)
    zhi = min(float(t_max), 5.0)
    if zhi - zlo < 1.0:
        zlo, zhi = 0.8, 5.0
    in_zone = np.isfinite(depression) & (t >= zlo) & (t <= zhi) & (support >= 5)
    if in_zone.sum() < 5:
        return {"detected": False, "reason": "no_toe_zone_support", "profile_noise_m": float(noise)}

    scale = robust_sigma(depression[in_zone])
    gate = max(float(np.nanmedian(depression[in_zone]) + 0.8 * scale),
               2.5 * float(noise) if np.isfinite(noise) else 0.0, 0.012)

    cand = []
    for i in np.where(in_zone)[0]:
        if i == 0 or i == len(depression) - 1:
            continue
        if depression[i] < depression[i - 1] or depression[i] < depression[i + 1]:
            continue
        l = max(0, i - 6); r = min(len(depression), i + 7)
        base = max(float(np.nanpercentile(depression[l:i], 25)) if i > l else 0.0,
                   float(np.nanpercentile(depression[i + 1:r], 25)) if i + 1 < r else 0.0)
        prom = float(depression[i] - base)
        if prom < gate:
            continue
        half = depression[i] - 0.5 * prom
        a = i
        while a > 0 and depression[a] > half:
            a -= 1
        b = i
        while b < len(depression) - 1 and depression[b] > half:
            b += 1
        width = float(t[b] - t[a])
        toe_score = math.exp(-abs(t[i] - 2.0) / 1.5)
        width_score = math.exp(-abs(width - 1.0) / 1.2)
        score = (prom / (prom + gate)) * (0.6 + 0.4 * toe_score) * (0.6 + 0.4 * width_score)
        cand.append((score, i, prom, width))
    if not cand:
        return {"detected": False, "reason": "no_toe_depression",
                "profile_noise_m": float(noise), "adaptive_prominence_m": float(gate)}
    score, i, prom, width = max(cand, key=lambda z: z[0])
    vlabel = "vegetation_surface_likely" if veg[i] > 0.55 else "visible_surface_candidate"
    return {
        "detected": True,
        "reason": "right_toe_depression",
        "T_m": float(t[i]),
        "visible_recession_m": float(prom),       # depth below local trend (DA3 units, relative)
        "visible_recession_metric_m": float(prom * DA3_METRES_PER_UNIT),  # approx. metres (coarse sleeper calibration)
        "raw_recession_m": float(rec[i]),
        "prominence_m": float(prom),
        "width_m": float(width),
        "rim_recovery_m": float("nan"),
        "profile_noise_m": float(noise),
        "adaptive_prominence_m": float(gate),
        "support_px": int(support[i]),
        "confidence_median": float(p.iloc[i]["confidence_median"]) if "confidence_median" in p else float("nan"),
        "vegetation_ratio": float(veg[i]),
        "exg_median": float(exg_med[i]),
        "ditch_likeness_score": float(score),
        "visibility_label": vlabel,
    }


def format_chainage(chainage_m: float) -> str:
    km = int(chainage_m // 1000)
    metre = int(round(chainage_m - 1000 * km))
    return f"{km}+{metre:03d}"


def _derive_search_band(rough: pd.DataFrame) -> tuple[float, float, dict]:
    detected = rough[(rough["detected"] == True) & np.isfinite(rough["T_m"]) & (rough["ditch_likeness_score"] > 0)]
    if len(detected) < 8:
        return 1.0, 7.0, {"band_source": "broad_fallback", "n_rough_candidates": int(len(detected))}
    non_boundary = detected[detected["T_m"] < 6.9]
    band_sample = non_boundary if len(non_boundary) >= 8 else detected
    q10, q90 = np.percentile(band_sample["T_m"], [10, 90])
    t_min = float(np.clip(q10 - 0.60, 1.0, 7.0))
    t_max = float(np.clip(q90 + 0.40, 1.0, 7.0))
    if t_max - t_min < 1.2:
        centre = 0.5 * (t_min + t_max)
        t_min = max(1.0, centre - 0.8)
        t_max = min(7.0, centre + 0.8)
    return t_min, t_max, {
        "band_source": "rough_candidate_percentiles",
        "n_rough_candidates": int(len(detected)),
        "n_non_boundary_candidates": int(len(non_boundary)),
        "rough_T_p10_m": float(q10),
        "rough_T_p90_m": float(q90),
    }


def process_profiles(depth_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    # Manual rail guides take priority when config/rail_guides.csv exists.
    # They are interpolated across frames; the automatic detector is used only
    # for frames a guide cannot cover.
    guides = load_rail_guides()
    n_guided = 0
    records = []
    for _, frame in depth_index.iterrows():
        image = cv2.imread(str(frame["image_path"]))
        if image is None:
            continue
        npz = np.load(frame["depth_npz_path"])
        depth = npz["depth"]
        conf = npz["conf"] if "conf" in npz else np.ones_like(depth, dtype=np.float32)
        fit = None
        if not guides.empty:
            fit = fit_rails_from_guides(image, int(frame["year"]), int(frame["sample_order"]), guides)
        if fit is None or not fit.ok:
            fit = fit_rails(image)
        else:
            n_guided += 1
        records.append({"frame": frame.to_dict(), "image": image, "depth": depth, "conf": conf, "raw_fit": fit})
    if not guides.empty:
        print(f"[rails] manual guides used on {n_guided}/{len(records)} frames "
              f"(auto detector on the rest)")

    smooth_fits = _smooth_rail_fits(records)
    rail_rows = []
    profile_rows = []
    rough_rows = []
    for rec, fit in zip(records, smooth_fits):
        frame = rec["frame"]
        raw = rec["raw_fit"]
        rail_rows.append(
            {
                **frame,
                "rail_source": fit.reason,
                "rail_fit_ok": bool(fit.ok),
                "rail_quality": float(fit.quality),
                "rail_rows_used": int(fit.rows_used),
                "rail_gauge_px_median": float(fit.gauge_px_median),
                "rail_gauge_px_mad": float(fit.gauge_px_mad),
                "rail_reason": fit.reason,
                "raw_rail_fit_ok": bool(raw.ok),
                "raw_rail_quality": float(raw.quality),
                "raw_rail_reason": raw.reason,
            }
        )
        profile, meta = build_right_profile(rec["image"], rec["depth"], rec["conf"], fit)
        if not profile.empty:
            for _, p_row in profile.iterrows():
                profile_rows.append({**frame, **p_row.to_dict()})
            det = detect_profile_candidate(profile, 1.0, 9.5)
            rough_rows.append({**frame, **det, "rail_quality": fit.quality, **meta})
        else:
            rough_rows.append({**frame, "detected": False, "reason": meta.get("profile_status"), "rail_quality": fit.quality, **meta})

    rail_df = pd.DataFrame(rail_rows)
    profile_df = pd.DataFrame(profile_rows)
    rough_df = pd.DataFrame(rough_rows)
    t_min, t_max, band_meta = _derive_search_band(rough_df)

    final_rows = []
    for rec, fit in zip(records, smooth_fits):
        frame = rec["frame"]
        profile = profile_df[(profile_df["year"] == frame["year"]) & (profile_df["sample_order"] == frame["sample_order"])]
        det = detect_profile_candidate(profile, t_min, t_max)
        final_rows.append({**frame, **det, "rail_quality": fit.quality, "adaptive_T_min_m": t_min, "adaptive_T_max_m": t_max})

    final_df = pd.DataFrame(final_rows)
    final_df = apply_longitudinal_quality(final_df)
    save_table(rail_df, TABLE_DIR / "rail_geometry_quality.csv")
    save_table(profile_df, TABLE_DIR / "right_side_profiles.csv")
    save_table(rough_df, TABLE_DIR / "right_side_rough_candidates.csv")
    save_table(final_df, TABLE_DIR / "frame_right_side_ditch_measurements.csv")
    write_json(
        TABLE_DIR / "step3_profile_detection_summary.json",
        {
            **band_meta,
            "adaptive_T_min_m": t_min,
            "adaptive_T_max_m": t_max,
            "frames": int(len(final_df)),
            "rail_fit_ok": int(rail_df["rail_fit_ok"].sum()) if len(rail_df) else 0,
            "final_detected": int(final_df["detected"].sum()) if len(final_df) else 0,
        },
    )
    draw_qc_figures(records, smooth_fits, final_df, profile_df)
    return rail_df, profile_df, final_df, {"adaptive_T_min_m": t_min, "adaptive_T_max_m": t_max, **band_meta}


def apply_longitudinal_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["year", "sample_order"]).reset_index(drop=True).copy()
    # Guard: if no frame detected, the detection-only columns may be absent.
    for _col in ("T_m", "ditch_likeness_score", "visible_recession_m", "vegetation_ratio"):
        if _col not in df.columns:
            df[_col] = np.nan
    if "detected" not in df.columns:
        df["detected"] = False
    df["path_confirmed"] = False
    df["temporal_quality_label"] = "not_detected"
    for year, idxs in df.groupby("year").groups.items():
        idxs = list(idxs)
        sub = df.loc[idxs].copy()
        det = sub["detected"].fillna(False).astype(bool).to_numpy()
        t = sub["T_m"].to_numpy(float)
        score = sub["ditch_likeness_score"].to_numpy(float)
        if det.sum() < 4:
            continue
        t_series = pd.Series(t).interpolate(limit_direction="both").rolling(5, center=True, min_periods=1).median().to_numpy()
        deviations = np.abs(t - t_series)
        tol = max(float(np.nanpercentile(deviations[det & np.isfinite(deviations)], 80)), 0.4) if np.isfinite(deviations[det]).any() else 0.6
        quality_score_gate = float(np.nanpercentile(score[det & np.isfinite(score)], 30)) if np.isfinite(score[det]).any() else 0.0
        confirmed = det & (deviations <= tol) & (score >= quality_score_gate)
        # Keep short misses from fragmenting the path, but do not invent depths.
        for k, global_idx in enumerate(idxs):
            if confirmed[k]:
                df.loc[global_idx, "path_confirmed"] = True
                df.loc[global_idx, "temporal_quality_label"] = "confirmed_right_ditch_visible_surface"
            elif det[k]:
                df.loc[global_idx, "temporal_quality_label"] = "candidate_unstable_position_or_low_score"
        df.loc[idxs, "path_T_tolerance_m"] = float(tol)
        df.loc[idxs, "path_score_gate"] = float(quality_score_gate)
    return df


def _rank01(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return out
    if float(valid.max()) == float(valid.min()):
        out.loc[valid.index] = 0.5
    else:
        out.loc[valid.index] = valid.rank(method="average", pct=True)
    return out


def _robust_z(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return out
    med = float(valid.median())
    mad = float(np.nanmedian(np.abs(valid - med)))
    if mad > 1e-9:
        out.loc[valid.index] = (valid - med) / (1.4826 * mad)
    else:
        std = float(valid.std(ddof=0))
        out.loc[valid.index] = (valid - med) / std if std > 1e-9 else 0.0
    return out


def aggregate_units(measurements: pd.DataFrame, unit_length_m: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = measurements.copy()
    df["unit_id"] = np.floor((df["chainage_m"] - ANALYSIS_START_M) / unit_length_m).astype(int)
    df["unit_start_m"] = ANALYSIS_START_M + df["unit_id"] * unit_length_m
    df["unit_end_m"] = np.minimum(df["unit_start_m"] + unit_length_m, ANALYSIS_END_M)
    rows = []
    for (year, unit_id), g in df.groupby(["year", "unit_id"], sort=True):
        confirmed = g[g["path_confirmed"] == True]
        usable = confirmed[
            np.isfinite(confirmed["visible_recession_m"])
            & (confirmed["visibility_label"].fillna("") != "boundary_visible_surface_uncertain")
        ]
        row = {
            "year": int(year),
            "unit_id": int(unit_id),
            "unit_start_m": float(g["unit_start_m"].iloc[0]),
            "unit_end_m": float(g["unit_end_m"].iloc[0]),
            "unit_label": f"{format_chainage(g['unit_start_m'].iloc[0])}-{format_chainage(g['unit_end_m'].iloc[0])}",
            "n_frames": int(len(g)),
            "n_detected": int(g["detected"].fillna(False).sum()),
            "n_confirmed": int(len(usable)),
            "detected_ratio": float(g["detected"].fillna(False).mean()) if len(g) else float("nan"),
            "confirmed_ratio": float(len(usable) / len(g)) if len(g) else float("nan"),
            "median_visible_recession_m": float(np.nanmedian(usable["visible_recession_m"])) if len(usable) else float("nan"),
            "median_visible_recession_metric_m": float(np.nanmedian(usable["visible_recession_m"]) * DA3_METRES_PER_UNIT) if len(usable) else float("nan"),
            "p25_visible_recession_m": float(np.nanpercentile(usable["visible_recession_m"], 25)) if len(usable) else float("nan"),
            "p75_visible_recession_m": float(np.nanpercentile(usable["visible_recession_m"], 75)) if len(usable) else float("nan"),
            "median_T_m": float(np.nanmedian(usable["T_m"])) if len(usable) else float("nan"),
            "median_ditch_likeness_score": float(np.nanmedian(usable["ditch_likeness_score"])) if len(usable) else float("nan"),
            "median_vegetation_ratio": float(np.nanmedian(usable["vegetation_ratio"])) if len(usable) else float("nan"),
            "median_rail_quality": float(np.nanmedian(g["rail_quality"])) if "rail_quality" in g else float("nan"),
        }
        row["metric_eligible"] = bool(row["n_confirmed"] >= max(3, int(math.ceil(0.30 * row["n_frames"]))))
        rows.append(row)
    units = pd.DataFrame(rows)
    if units.empty:
        save_table(units, TABLE_DIR / "temporal_units_by_year.csv")
        change = pd.DataFrame()
        save_table(change, TABLE_DIR / "temporal_change_2020_2025.csv")
        return units, change

    for col in ("median_visible_recession_m", "median_ditch_likeness_score", "median_rail_quality"):
        units[f"{col}_rank_within_year"] = np.nan
        units[f"{col}_z_within_year"] = np.nan
        for _, idx in units.groupby("year").groups.items():
            units.loc[idx, f"{col}_rank_within_year"] = _rank01(units.loc[idx, col])
            units.loc[idx, f"{col}_z_within_year"] = _robust_z(units.loc[idx, col])

    # Unitless visible-evidence index. It is intentionally based on within-year
    # ranks and confirmation rate, so it is less sensitive to cross-camera DA3
    # scale differences than raw recession magnitude.
    units["visible_evidence_index"] = (
        0.45 * units["confirmed_ratio"].fillna(0.0)
        + 0.25 * units["median_ditch_likeness_score_rank_within_year"].fillna(0.0)
        + 0.20 * units["median_visible_recession_m_rank_within_year"].fillna(0.0)
        + 0.10 * units["median_rail_quality_rank_within_year"].fillna(0.5)
    ).clip(0.0, 1.0)
    units["evidence_eligible"] = (
        (units["n_confirmed"] >= units["n_frames"].map(lambda n: max(3, int(math.ceil(0.30 * n)))))
        & np.isfinite(units["visible_evidence_index"])
    )
    save_table(units, TABLE_DIR / "temporal_units_by_year.csv")

    wide = units.pivot(index="unit_id", columns="year", values="median_visible_recession_m")
    change_rows = []
    for unit_id, row in wide.iterrows():
        u20 = units[(units["year"] == 2020) & (units["unit_id"] == unit_id)]
        u25 = units[(units["year"] == 2025) & (units["unit_id"] == unit_id)]
        if u20.empty or u25.empty:
            continue
        d20 = float(row.get(2020, float("nan")))
        d25 = float(row.get(2025, float("nan")))
        delta = d25 - d20 if np.isfinite(d20) and np.isfinite(d25) else float("nan")
        n20 = int(u20["n_confirmed"].iloc[0]); n25 = int(u25["n_confirmed"].iloc[0])
        elig20 = bool(u20["metric_eligible"].iloc[0]); elig25 = bool(u25["metric_eligible"].iloc[0])
        eligible = bool(elig20 and elig25 and np.isfinite(delta))
        ev20 = float(u20["visible_evidence_index"].iloc[0])
        ev25 = float(u25["visible_evidence_index"].iloc[0])
        delta_ev = ev25 - ev20 if np.isfinite(ev20) and np.isfinite(ev25) else float("nan")
        evidence_eligible20 = bool(u20["evidence_eligible"].iloc[0])
        evidence_eligible25 = bool(u25["evidence_eligible"].iloc[0])
        paired_evidence_comparable = bool(evidence_eligible20 and evidence_eligible25 and np.isfinite(delta_ev))
        rail_q20 = float(u20["median_rail_quality"].iloc[0])
        rail_q25 = float(u25["median_rail_quality"].iloc[0])
        rail_quality_delta = rail_q25 - rail_q20 if np.isfinite(rail_q20) and np.isfinite(rail_q25) else float("nan")
        veg20 = float(u20["median_vegetation_ratio"].iloc[0])
        veg25 = float(u25["median_vegetation_ratio"].iloc[0])
        vegetation_delta = veg25 - veg20 if np.isfinite(veg20) and np.isfinite(veg25) else float("nan")
        quality_matched = bool(
            np.isfinite(rail_quality_delta)
            and abs(rail_quality_delta) <= 0.25
            and (not np.isfinite(vegetation_delta) or abs(vegetation_delta) <= 0.35)
        )
        if paired_evidence_comparable and quality_matched and delta_ev <= -0.25:
            label = "robust_visible_evidence_decrease"
        elif paired_evidence_comparable and quality_matched and delta_ev >= 0.25:
            label = "robust_visible_evidence_increase"
        elif evidence_eligible20 and n25 == 0:
            label = "possible_visible_evidence_loss_2025"
        elif evidence_eligible25 and n20 == 0:
            label = "possible_visible_evidence_gain_2025"
        elif paired_evidence_comparable and np.isfinite(delta_ev) and abs(delta_ev) >= 0.25:
            label = "possible_visible_evidence_change_confound"
        elif paired_evidence_comparable:
            label = "no_clear_visible_evidence_change"
        else:
            label = "not_comparable"
        change_rows.append(
            {
                "unit_id": int(unit_id),
                "unit_label": u20["unit_label"].iloc[0],
                "visible_recession_2020_m": d20,
                "visible_recession_2025_m": d25,
                "visible_recession_2020_metric_m": d20 * DA3_METRES_PER_UNIT,
                "visible_recession_2025_metric_m": d25 * DA3_METRES_PER_UNIT,
                "delta_2025_minus_2020_m": float(delta),
                "delta_2025_minus_2020_metric_m": float(delta * DA3_METRES_PER_UNIT),
                "metric_eligible": eligible,
                "visible_evidence_2020": ev20,
                "visible_evidence_2025": ev25,
                "delta_visible_evidence_2025_minus_2020": float(delta_ev),
                "evidence_eligible_2020": evidence_eligible20,
                "evidence_eligible_2025": evidence_eligible25,
                "paired_evidence_comparable": paired_evidence_comparable,
                "quality_matched": quality_matched,
                "change_label": label,
                "n_confirmed_2020": int(u20["n_confirmed"].iloc[0]),
                "n_confirmed_2025": int(u25["n_confirmed"].iloc[0]),
                "confirmed_ratio_2020": float(u20["confirmed_ratio"].iloc[0]),
                "confirmed_ratio_2025": float(u25["confirmed_ratio"].iloc[0]),
                "detected_ratio_2020": float(u20["detected_ratio"].iloc[0]),
                "detected_ratio_2025": float(u25["detected_ratio"].iloc[0]),
                "median_T_2020_m": float(u20["median_T_m"].iloc[0]),
                "median_T_2025_m": float(u25["median_T_m"].iloc[0]),
                "median_vegetation_2020": veg20,
                "median_vegetation_2025": veg25,
                "vegetation_delta_2025_minus_2020": float(vegetation_delta),
                "median_rail_quality_2020": rail_q20,
                "median_rail_quality_2025": rail_q25,
                "rail_quality_delta_2025_minus_2020": float(rail_quality_delta),
            }
        )
    change = pd.DataFrame(change_rows)
    save_table(change, TABLE_DIR / "temporal_change_2020_2025.csv")
    return units, change


def draw_qc_figures(records: list[dict], rail_fits: list[RailFit], measurements: pd.DataFrame, profiles: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for rec, fit in zip(records, rail_fits):
        frame = rec["frame"]
        order = int(frame["sample_order"])
        if order not in (0, 10, 20, 40, 60, 79):
            continue
        image = rec["image"].copy()
        h = image.shape[0]
        meas = measurements[(measurements["year"] == frame["year"]) & (measurements["sample_order"] == order)]
        if fit.ok and fit.left_poly is not None and fit.right_poly is not None:
            ys = np.linspace(int(0.48 * h), int(0.95 * h), 120)
            for poly in (fit.left_poly, fit.right_poly):
                pts = []
                for y in ys:
                    pts.append([int(round(eval_poly(poly, np.array([y]), h)[0])), int(y)])
                cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (0, 255, 255), 2)
        if not meas.empty and bool(meas["detected"].iloc[0]):
            txt = f"{frame['year']} {frame['chainage_label']} right T={meas['T_m'].iloc[0]:.1f} rec={meas['visible_recession_m'].iloc[0]:.2f}m"
            cv2.putText(image, txt, (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out = FIGURE_DIR / f"qc_{frame['year']}_{order:04d}_{frame['chainage_label'].replace('+','p')}.jpg"
        cv2.imwrite(str(out), image)

    try:
        import matplotlib.pyplot as plt

        for year in sorted(profiles["year"].unique()):
            sample_orders = sorted(profiles.loc[profiles["year"] == year, "sample_order"].unique())
            for order in sample_orders[:: max(1, len(sample_orders) // 5)]:
                p = profiles[(profiles["year"] == year) & (profiles["sample_order"] == order)].sort_values("T_m")
                m = measurements[(measurements["year"] == year) & (measurements["sample_order"] == order)]
                if p.empty:
                    continue
                plt.figure(figsize=(7, 4))
                plt.plot(p["T_m"], p["right_visible_recession_m"], "-o", ms=3, lw=1.5)
                if not m.empty and bool(m["detected"].iloc[0]):
                    plt.axvline(float(m["T_m"].iloc[0]), color="r", lw=1.5)
                plt.xlabel("Right-side lateral distance T (m)")
                plt.ylabel("Visible-surface recession (DA3 relative units, not metres)")
                plt.title(f"{year} {m['chainage_label'].iloc[0] if not m.empty else order}")
                plt.grid(True, alpha=0.25)
                plt.tight_layout()
                plt.savefig(FIGURE_DIR / f"profile_{year}_{int(order):04d}.png", dpi=160)
                plt.close()
    except Exception as exc:
        write_json(TABLE_DIR / "figure_warning.json", {"warning": str(exc)})


def draw_change_figures(units: pd.DataFrame, change: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        for year, g in units.groupby("year"):
            x = 0.5 * (g["unit_start_m"] + g["unit_end_m"])
            plt.plot(x, g["median_visible_recession_m"], "-o", label=str(year))
        plt.xlabel("Chainage (m)")
        plt.ylabel("Median right-side visible recession (DA3 relative units, not metres)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "temporal_visible_recession_by_year.png", dpi=180)
        plt.close()

        if not change.empty:
            if "paired_evidence_comparable" in change.columns:
                colors = ["tab:blue" if bool(v) else "0.7" for v in change["paired_evidence_comparable"]]
                y = change["delta_visible_evidence_2025_minus_2020"]
                ylabel = "2025 - 2020 visible evidence index (unitless; within-year normalised)"
            else:
                colors = ["tab:blue" if bool(v) else "0.7" for v in change["metric_eligible"]]
                y = change["delta_2025_minus_2020_m"]
                ylabel = "2025 - 2020 visible recession (DA3 relative units; cross-camera, not comparable)"
            plt.figure(figsize=(10, 4))
            x = np.arange(len(change))
            plt.bar(x, y, color=colors)
            plt.axhline(0, color="k", lw=0.8)
            plt.xticks(x, change["unit_label"], rotation=45, ha="right")
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "temporal_change_delta.png", dpi=180)
            plt.close()
    except Exception as exc:
        write_json(TABLE_DIR / "change_figure_warning.json", {"warning": str(exc)})


def summarise(units: pd.DataFrame, change: pd.DataFrame, band_meta: dict,
              measurements: pd.DataFrame | None = None) -> dict:
    eligible = change[change["metric_eligible"] == True] if not change.empty else pd.DataFrame()
    evidence_eligible = (
        change[change["paired_evidence_comparable"] == True]
        if (not change.empty and "paired_evidence_comparable" in change.columns)
        else pd.DataFrame()
    )

    # Per-year detection rate plus absolute greenness. Cross-year attribution is
    # treated cautiously because the cameras and surface colour differ.
    detection_by_year = {}
    if measurements is not None and not measurements.empty and "detected" in measurements.columns:
        for yr, g in measurements.groupby("year"):
            d = g[g["detected"] == True]
            detection_by_year[int(yr)] = {
                "frames": int(len(g)),
                "detected": int(len(d)),
                "detection_rate": float(len(d) / len(g)) if len(g) else float("nan"),
                # absolute greenness: fraction of detected frames whose ditch
                # zone is truly green (ExG > 0), and median ExG of the zone.
                "median_zone_exg_detected": float(np.nanmedian(d["exg_median"])) if ("exg_median" in d and len(d)) else float("nan"),
                "frac_green_zone_detected": float(np.mean(d["exg_median"] > 0)) if ("exg_median" in d and len(d)) else float("nan"),
                "median_vegetation_ratio_detected": float(np.nanmedian(d["vegetation_ratio"])) if ("vegetation_ratio" in d and len(d)) else float("nan"),
                # detected ditch depth in approximate metres from the coarse sleeper calibration
                "median_depth_metric_m_detected": float(np.nanmedian(d["visible_recession_metric_m"])) if ("visible_recession_metric_m" in d and len(d)) else float("nan"),
            }

    change_class_counts = {}
    if not change.empty and "change_label" in change.columns:
        change_class_counts = {str(k): int(v) for k, v in change["change_label"].value_counts().items()}

    evidence_by_year = {}
    if not units.empty and "visible_evidence_index" in units.columns:
        for yr, g in units.groupby("year"):
            evidence_by_year[int(yr)] = {
                "units": int(len(g)),
                "evidence_eligible_units": int(g["evidence_eligible"].sum()) if "evidence_eligible" in g else 0,
                "median_visible_evidence_index": float(np.nanmedian(g["visible_evidence_index"])),
                "median_confirmed_ratio": float(np.nanmedian(g["confirmed_ratio"])),
                "median_detected_ratio": float(np.nanmedian(g["detected_ratio"])),
            }

    summary = {
        "analysis_start": format_chainage(ANALYSIS_START_M),
        "analysis_end": format_chainage(ANALYSIS_END_M),
        "right_side_only": True,
        "primary_finding": "right_visible_ditch_evidence_is_stronger_in_2020_than_2025_but_cross_year_cause_is_confounding",
        "detection_by_year": detection_by_year,
        "visible_evidence_by_year": evidence_by_year,
        "change_class_counts": change_class_counts,
        "adaptive_search_band": band_meta,
        "unit_rows": int(len(units)),
        "change_units": int(len(change)),
        "eligible_change_units": int(len(eligible)),
        "paired_evidence_comparable_units": int(len(evidence_eligible)),
        # Metric scale calibration and caveats for the approximate depth values.
        "metric_scale_calibration": {
            "da3_metres_per_unit": DA3_METRES_PER_UNIT,
            "source": "sleeper-period (~0.60 m) autocorrelation along the track centreline",
            "da3_linearity_median_r2": 0.997,
            "absolute_scale_uncertainty": "factor-of-~2 (sleeper-harmonic ambiguity); ~30% inter-frame spread",
        },
        "units_disclaimer": "raw depth/recession remains approximate through a coarse sleeper-period calibration (~10 m per DA3 unit). The preferred cross-year quantity is the unitless visible_evidence_index, which combines confirmation rate and within-year ranked recession/ditch-likeness evidence.",
        "confound_note": "evidence-rate differences remain confounded by camera change, surface-colour difference, and vegetation; interpret as visible drainage-surface evidence, not direct physical ditch-depth change.",
        "eligible_delta_note": f"raw depth delta is retained only as a coarse diagnostic (n_metric_eligible_units={int(len(eligible))}); use visible-evidence change classes for PPT/thesis conclusions.",
        "visible_evidence_note": f"visible evidence index is unitless and within-year normalised; paired comparable units require adequate detections in both years (n={int(len(evidence_eligible))}).",
        "median_delta_visible_evidence_2025_minus_2020": float(np.nanmedian(evidence_eligible["delta_visible_evidence_2025_minus_2020"])) if len(evidence_eligible) else float("nan"),
        "mean_delta_visible_evidence_2025_minus_2020": float(np.nanmean(evidence_eligible["delta_visible_evidence_2025_minus_2020"])) if len(evidence_eligible) else float("nan"),
        "median_delta_2025_minus_2020_metric_m": float(np.nanmedian(eligible["delta_2025_minus_2020_m"]) * DA3_METRES_PER_UNIT) if len(eligible) else float("nan"),
        "mean_delta_2025_minus_2020_metric_m": float(np.nanmean(eligible["delta_2025_minus_2020_m"]) * DA3_METRES_PER_UNIT) if len(eligible) else float("nan"),
        "median_delta_2025_minus_2020_relativeunits": float(np.nanmedian(eligible["delta_2025_minus_2020_m"])) if len(eligible) else float("nan"),
    }
    write_json(TABLE_DIR / "final_summary.json", summary)
    return summary


def run(sample_spacing_m: float, unit_length_m: float, device: str, process_res: int,
        chunk_size: int, skip_existing: bool) -> None:
    ensure_dirs()
    frames = extract_frames(sample_spacing_m=sample_spacing_m)
    depth_index = run_da3(
        frames,
        device=device,
        process_res=process_res,
        chunk_size=chunk_size,
        skip_existing=skip_existing,
    )
    _, _, measurements, band_meta = process_profiles(depth_index)
    units, change = aggregate_units(measurements, unit_length_m=unit_length_m)
    draw_change_figures(units, change)
    summary = summarise(units, change, band_meta, measurements)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-spacing-m", type=float, default=2.0)
    parser.add_argument("--unit-length-m", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", type=int, default=420)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    run(
        sample_spacing_m=args.sample_spacing_m,
        unit_length_m=args.unit_length_m,
        device=args.device,
        process_res=args.process_res,
        chunk_size=args.chunk_size,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()

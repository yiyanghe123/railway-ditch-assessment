"""Build a continuous chainage profile from the 43 Cirrus DA3 panoramas.

This step does not rerun Depth Anything 3.  It consumes the multi-view
visible-surface reconstruction outputs and turns the per-panorama side profiles into
track-referenced products:

1. an inventory check proving that the usable JPEG panoramas are continuous;
2. a raw target-station table for all 43 panoramas and both sides;
3. a 1 m continuous visible-surface profile in chainage space;
4. a LiDAR-aligned comparison table.

The continuous profile is an interpolation of panorama observations.  It is not
a 1 m image acquisition, and it is not a hidden ditch-bottom reconstruction.
Every grid row therefore records the nearest panorama distance, the number of
source panoramas, the source quality, and the propagated visibility class.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    pano_csv: Path = Path(os.environ.get("PANO_PANO_CSV", ""))
    da3_csv: Path = Path(os.environ.get("PANO_DA3_CSV", ""))
    lidar_csv: Path = Path(os.environ.get("PANO_LIDAR_CSV", ""))
    out_dir: Path = Path(os.environ.get("PANO_CONTINUOUS_OUT_DIR", "output_continuous"))
    fig_dir: Path = Path(os.environ.get("PANO_CONTINUOUS_FIG_DIR", "figures_continuous"))
    grid_step_m: float = 1.0
    max_source_distance_m: float = 15.0
    interpolation_sigma_m: float = 6.0
    max_metric_eval_distance_m: float = 5.5
    # Maximum chainage gap allowed when joining a panorama grid row to the
    # nearest LiDAR section; beyond this the LiDAR row is treated as "no
    # confirmed ditch" rather than silently paired with a distant section.
    max_lidar_join_distance_m: float = 2.0
    # |panorama - LiDAR| band within which the visible surface is reported to
    # agree with the LiDAR depth (used only for the comparison label).
    comparison_tolerance_m: float = 0.30
    visibility_weight_bottom: float = 1.00
    visibility_weight_monotonic: float = 0.75
    visibility_weight_no_clear: float = 0.35
    visibility_weight_boundary: float = 0.20
    visibility_weight_other: float = 0.15


CFG = Config()
# Independent variant switch: PANO_USE_LIDAR=0 skips every LiDAR comparison so
# the continuous panorama products can be built with no ditch reference at all.
USE_LIDAR = os.environ.get("PANO_USE_LIDAR", "1").strip().lower() not in ("0", "false", "no", "")


def ensure_dirs() -> None:
    CFG.out_dir.mkdir(parents=True, exist_ok=True)
    CFG.fig_dir.mkdir(parents=True, exist_ok=True)


def frame_no(name: str) -> int | float:
    m = re.search(r"_(\d{6})(?:\.jpg)?$", str(name))
    return int(m.group(1)) if m else math.nan


def visibility_weight(label: str) -> float:
    label = str(label)
    if label == "bottom_visible_candidate":
        return CFG.visibility_weight_bottom
    if label == "monotonic_slope_surface":
        return CFG.visibility_weight_monotonic
    if label == "no_clear_visible_depression":
        return CFG.visibility_weight_no_clear
    if label == "boundary_surface_uncertain":
        return CFG.visibility_weight_boundary
    return CFG.visibility_weight_other


def safe_obs_string(obs: pd.Series, key: str, default: str = "NA") -> str:
    value = obs.get(key, default)
    return default if pd.isna(value) else str(value)


def weighted_mode(labels: pd.Series, weights: np.ndarray) -> str:
    totals: dict[str, float] = {}
    for label, weight in zip(labels.astype(str), weights):
        totals[label] = totals.get(label, 0.0) + float(weight)
    if not totals:
        return "NA"
    return max(totals.items(), key=lambda kv: kv[1])[0]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    pano = pd.read_csv(CFG.pano_csv)
    da3 = pd.read_csv(CFG.da3_csv)
    lidar = pd.read_csv(CFG.lidar_csv) if (USE_LIDAR and CFG.lidar_csv.exists()) else None
    if not USE_LIDAR:
        # Independent variant: strip any LiDAR columns baked into an existing
        # step-4 product so they cannot leak into the panorama-only outputs.
        da3 = da3.drop(columns=[c for c in da3.columns if "lidar" in c.lower()])

    pano["frame_no"] = pano["Filename"].map(frame_no)
    pano["jpg_exists"] = pano["jpg_path"].map(lambda p: Path(str(p)).exists())
    da3["frame_no"] = da3["filename"].map(frame_no)
    da3["visibility_weight"] = da3["visibility_class"].map(visibility_weight)
    da3["continuous_source_quality"] = (
        pd.to_numeric(da3["combined_metric_quality_score"], errors="coerce").fillna(0.0)
        * da3["visibility_weight"]
    )
    return pano, da3, lidar


def panorama_continuity_report(pano: pd.DataFrame, da3: pd.DataFrame) -> dict:
    jpg = pano[pano["jpg_exists"]].sort_values("frame_no").copy()
    jpg["d_frame"] = jpg["frame_no"].diff()
    jpg["d_time_s"] = jpg["Timestamp"].diff()
    jpg["dxy_m"] = np.hypot(jpg["East"].diff(), jpg["North"].diff())
    jpg["dS_m"] = jpg["S"].diff()
    missing_jpg = sorted(set(range(int(jpg["frame_no"].min()), int(jpg["frame_no"].max()) + 1)) - set(jpg["frame_no"].astype(int)))

    targets = da3[["filename", "frame_no", "S_cam"]].drop_duplicates().sort_values("frame_no").copy()
    targets["d_frame"] = targets["frame_no"].diff()
    targets["dS_m"] = targets["S_cam"].diff()
    missing_targets = sorted(set(range(int(targets["frame_no"].min()), int(targets["frame_no"].max()) + 1)) - set(targets["frame_no"].astype(int)))

    return {
        "matched_metadata_rows": int(len(pano)),
        "usable_jpeg_panoramas": int(len(jpg)),
        "usable_jpeg_frame_min": int(jpg["frame_no"].min()),
        "usable_jpeg_frame_max": int(jpg["frame_no"].max()),
        "usable_jpeg_missing_frame_numbers": missing_jpg,
        "usable_jpeg_frame_diff_counts": {str(k): int(v) for k, v in jpg["d_frame"].value_counts(dropna=True).sort_index().items()},
        "usable_jpeg_spacing_xy_m": {
            "mean": float(jpg["dxy_m"].mean()),
            "min": float(jpg["dxy_m"].min()),
            "max": float(jpg["dxy_m"].max()),
            "std": float(jpg["dxy_m"].std()),
        },
        "usable_jpeg_spacing_S_m": {
            "mean": float(jpg["dS_m"].abs().mean()),
            "min": float(jpg["dS_m"].abs().min()),
            "max": float(jpg["dS_m"].abs().max()),
            "std": float(jpg["dS_m"].abs().std()),
        },
        "da3_target_panoramas": int(len(targets)),
        "da3_side_rows": int(len(da3)),
        "da3_target_missing_frame_numbers": missing_targets,
        "da3_target_frame_diff_counts": {str(k): int(v) for k, v in targets["d_frame"].value_counts(dropna=True).sort_index().items()},
        "n_views_counts": {str(k): int(v) for k, v in da3["n_views"].value_counts().sort_index().items()},
    }


def write_target_station_table(da3: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "filename",
        "frame_no",
        "side",
        "S_cam",
        "source_indices",
        "n_views",
        "metric_evaluation_eligible",
        "combined_metric_quality_score",
        "continuous_source_quality",
        "visible_surface_depth_m",
        "visible_surface_T_m",
        "visible_surface_search_zone",
        "visible_surface_search_zone_used_fallback",
        "primary_visible_surface_depth_m",
        "primary_visible_surface_T_m",
        "primary_profile_local_prominence_m",
        "near_track_visible_surface_depth_m",
        "near_track_visible_surface_T_m",
        "near_track_profile_local_prominence_m",
        "diagnostic_max_depth_m",
        "diagnostic_max_T_m",
        "reference_quality_score",
        "has_metric_reference",
        "n_reference_views",
        "rail_gauge_quality_score",
        "rail_gauge_status",
        "visibility_class",
        "visibility_confidence",
        "bottom_visible_candidate",
        "ditch_presence_class",
        "visible_surface_metric_candidate",
        "metric_exclusion_reason",
        "vegetation_ratio",
        "profile_local_prominence_m",
        "profile_trend_corr",
        "peak_at_search_boundary",
        "fused_views_at_peak",
        "zone_points",
        "valid_bins",
        "lidar_s",
        "lidar_depth_m",
        "lidar_bottom_T_m",
        "lidar_priority",
        "profile_path",
        "target_crop_path",
    ]
    available = [c for c in cols if c in da3.columns]
    station = da3[available].sort_values(["S_cam", "side"]).copy()
    station.to_csv(CFG.out_dir / "panorama_target_station_visible_surface.csv", index=False, float_format="%.6f")
    return station


def stack_target_profiles(da3: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, obs in da3.iterrows():
        path = Path(str(obs.get("profile_path", "")))
        if not path.exists():
            continue
        prof = pd.read_csv(path)
        prof.insert(0, "filename", obs["filename"])
        prof.insert(1, "frame_no", obs["frame_no"])
        prof.insert(2, "side", obs["side"])
        prof.insert(3, "S_cam", obs["S_cam"])
        prof.insert(4, "visibility_class", obs["visibility_class"])
        prof.insert(5, "ditch_presence_class", safe_obs_string(obs, "ditch_presence_class"))
        prof.insert(6, "metric_exclusion_reason", safe_obs_string(obs, "metric_exclusion_reason"))
        prof.insert(7, "metric_evaluation_eligible", bool(obs["metric_evaluation_eligible"]))
        prof.insert(8, "combined_metric_quality_score", obs["combined_metric_quality_score"])
        rows.append(prof)
    stacked = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    stacked.to_csv(CFG.out_dir / "panorama_side_profile_stack.csv", index=False, float_format="%.6f")
    return stacked


def interpolate_depth_for_side(obs: pd.DataFrame, s_grid: np.ndarray, source_set: str) -> pd.DataFrame:
    rows = []
    obs = obs[np.isfinite(obs["visible_surface_depth_m"])].copy()
    if source_set == "metric_eligible":
        obs = obs[obs["metric_evaluation_eligible"] == True].copy()
    for s in s_grid:
        if obs.empty:
            rows.append({"S_grid": s, "source_set": source_set, "continuous_available": False})
            continue
        dist = np.abs(obs["S_cam"].to_numpy(dtype=float) - s)
        keep = dist <= CFG.max_source_distance_m
        nearby = obs.loc[keep].copy()
        nearby_dist = dist[keep]
        if nearby.empty:
            nearest_idx = int(np.argmin(dist))
            rows.append(
                {
                    "S_grid": s,
                    "source_set": source_set,
                    "continuous_available": False,
                    "nearest_source_distance_m": float(dist[nearest_idx]),
                    "nearest_frame_no": int(obs.iloc[nearest_idx]["frame_no"]),
                    "nearest_visibility_class": str(obs.iloc[nearest_idx]["visibility_class"]),
                    "nearest_ditch_presence_class": safe_obs_string(obs.iloc[nearest_idx], "ditch_presence_class"),
                    "nearest_metric_exclusion_reason": safe_obs_string(obs.iloc[nearest_idx], "metric_exclusion_reason"),
                    "source_panorama_count": 0,
                }
            )
            continue
        distance_w = np.exp(-0.5 * (nearby_dist / CFG.interpolation_sigma_m) ** 2)
        quality_w = nearby["continuous_source_quality"].to_numpy(dtype=float)
        if source_set == "metric_eligible":
            quality_w = np.maximum(quality_w, 0.05)
        weights = distance_w * np.maximum(quality_w, 0.03)
        depths = nearby["visible_surface_depth_m"].to_numpy(dtype=float)
        weighted_depth = float(np.sum(weights * depths) / np.sum(weights))
        nearest_local = int(np.argmin(nearby_dist))
        nearest = nearby.iloc[nearest_local]
        nearest_distance = float(nearby_dist[nearest_local])
        row = {
            "S_grid": float(s),
            "source_set": source_set,
            "continuous_available": True,
            "continuous_visible_surface_depth_m": weighted_depth,
            "nearest_source_distance_m": nearest_distance,
            "source_panorama_count": int(len(nearby)),
            "source_frame_numbers": " ".join(str(int(x)) for x in nearby["frame_no"].to_list()),
            "nearest_frame_no": int(nearest["frame_no"]),
            "nearest_S_cam": float(nearest["S_cam"]),
            "nearest_visibility_class": str(nearest["visibility_class"]),
            "nearest_ditch_presence_class": safe_obs_string(nearest, "ditch_presence_class"),
            "nearest_metric_exclusion_reason": safe_obs_string(nearest, "metric_exclusion_reason"),
            "weighted_visibility_class": weighted_mode(nearby["visibility_class"], weights),
            "weighted_ditch_presence_class": weighted_mode(nearby.get("ditch_presence_class", pd.Series("NA", index=nearby.index)), weights),
            "weighted_metric_exclusion_reason": weighted_mode(nearby.get("metric_exclusion_reason", pd.Series("NA", index=nearby.index)), weights),
            "weighted_quality_score": float(np.sum(weights * nearby["combined_metric_quality_score"].to_numpy(dtype=float)) / np.sum(weights)),
            "weighted_source_quality_score": float(np.sum(weights * nearby["continuous_source_quality"].to_numpy(dtype=float)) / np.sum(weights)),
            "metric_evaluation_eligible_continuous": bool(
                source_set == "metric_eligible" and nearest_distance <= CFG.max_metric_eval_distance_m
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_continuous_grid(da3: pd.DataFrame) -> pd.DataFrame:
    s_min = math.ceil(float(da3["S_cam"].min()))
    s_max = math.floor(float(da3["S_cam"].max()))
    s_grid = np.arange(s_min, s_max + CFG.grid_step_m, CFG.grid_step_m)
    frames = []
    for side, sub in da3.groupby("side"):
        for source_set in ("all_reference_accepted", "metric_eligible"):
            interp = interpolate_depth_for_side(sub, s_grid, source_set)
            interp.insert(1, "side", side)
            frames.append(interp)
    grid = pd.concat(frames, ignore_index=True)
    grid.to_csv(CFG.out_dir / "panorama_continuous_visible_surface_1m.csv", index=False, float_format="%.6f")
    return grid


def nearest_lidar_depth(lidar: pd.DataFrame, s: float, side: str) -> dict:
    idx = int(np.argmin(np.abs(lidar["s"].to_numpy(dtype=float) - s)))
    row = lidar.iloc[idx]
    prefix = "left" if side == "left" else "right"
    distance = float(abs(float(row["s"]) - s))
    # Guard against pairing a panorama row with a distant LiDAR section when the
    # LiDAR table does not cover this chainage.
    if distance > CFG.max_lidar_join_distance_m:
        return {
            "lidar_s": float(row["s"]),
            "lidar_distance_m": distance,
            "lidar_ditch_exists": False,
            "lidar_depth_m": math.nan,
            "lidar_bottom_T_m": math.nan,
            "lidar_priority": "NA",
            "lidar_priority_source": "out_of_join_range",
        }
    return {
        "lidar_s": float(row["s"]),
        "lidar_distance_m": distance,
        "lidar_ditch_exists": bool(row.get(f"{prefix}_ditch_exists", False)),
        "lidar_depth_m": pd.to_numeric(pd.Series([row.get(f"{prefix}_depth")]), errors="coerce").iloc[0],
        "lidar_bottom_T_m": pd.to_numeric(pd.Series([row.get(f"{prefix}_bottom_x")]), errors="coerce").iloc[0],
        "lidar_priority": row.get(f"{prefix}_tdok_priority", "NA"),
        "lidar_priority_source": row.get(f"{prefix}_priority_source", "NA"),
    }


def build_lidar_aligned_table(grid: pd.DataFrame, lidar: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    usable = grid[(grid["continuous_available"] == True)].copy()
    for _, g in usable.iterrows():
        row = g.to_dict()
        if lidar is None:
            # Independent variant: emit the panorama continuous product with no
            # LiDAR comparison columns (downstream guards on column presence).
            row["panorama_minus_lidar_depth_m"] = math.nan
            row["comparison_interpretation"] = "lidar_disabled_independent_variant"
            rows.append(row)
            continue
        info = nearest_lidar_depth(lidar, float(g["S_grid"]), str(g["side"]))
        row.update(info)
        lidar_depth = row.get("lidar_depth_m")
        pano_depth = row.get("continuous_visible_surface_depth_m")
        if pd.notna(lidar_depth) and pd.notna(pano_depth):
            row["panorama_minus_lidar_depth_m"] = float(pano_depth) - float(lidar_depth)
        else:
            row["panorama_minus_lidar_depth_m"] = math.nan
        row["comparison_interpretation"] = comparison_interpretation(row)
        rows.append(row)
    aligned = pd.DataFrame(rows)
    aligned.to_csv(CFG.out_dir / "panorama_lidar_aligned_visible_surface_1m.csv", index=False, float_format="%.6f")
    return aligned


def comparison_interpretation(row: dict) -> str:
    if not row.get("lidar_ditch_exists", False):
        return "no_lidar_confirmed_ditch"
    if row.get("source_set") == "metric_eligible" and not row.get("metric_evaluation_eligible_continuous", False):
        return "not_metric_eligible_continuous"
    diff = row.get("panorama_minus_lidar_depth_m")
    if pd.isna(diff):
        return "missing_depth_comparison"
    vis = str(row.get("weighted_visibility_class", row.get("nearest_visibility_class", "")))
    if "boundary" in vis:
        return "image_boundary_visible_surface_not_comparable"
    if "no_clear" in vis:
        return "no_clear_visible_depression"
    if diff < -CFG.comparison_tolerance_m:
        return "panorama_shallower_than_lidar_possible_occlusion"
    if diff > CFG.comparison_tolerance_m:
        return "panorama_deeper_than_lidar_check_projection"
    return "visible_surface_agrees_with_lidar_depth"


def metric_summary(aligned: pd.DataFrame) -> dict:
    out = {}
    if "lidar_ditch_exists" not in aligned.columns:
        return out  # independent (LiDAR-free) variant: nothing to compare
    for source_set, sub in aligned.groupby("source_set"):
        comp = sub[
            (sub["lidar_ditch_exists"] == True)
            & pd.to_numeric(sub["panorama_minus_lidar_depth_m"], errors="coerce").notna()
        ].copy()
        if source_set == "metric_eligible":
            comp = comp[comp["metric_evaluation_eligible_continuous"] == True]
        diff = comp["panorama_minus_lidar_depth_m"].to_numpy(dtype=float)
        out[source_set] = {
            "n": int(len(diff)),
            "bias_m": float(np.mean(diff)) if len(diff) else math.nan,
            "rmse_m": float(np.sqrt(np.mean(diff**2))) if len(diff) else math.nan,
            "mae_m": float(np.mean(np.abs(diff))) if len(diff) else math.nan,
        }
    return out


def plot_depth_profiles(grid: pd.DataFrame, aligned: pd.DataFrame) -> None:
    for source_set in ("all_reference_accepted", "metric_eligible"):
        fig, ax = plt.subplots(figsize=(12, 5))
        for side, color in [("left", "#1476B8"), ("right", "#D95F02")]:
            sub = grid[(grid["source_set"] == source_set) & (grid["side"] == side) & (grid["continuous_available"] == True)].sort_values("S_grid")
            if source_set == "metric_eligible":
                sub = sub[sub["metric_evaluation_eligible_continuous"] == True]
            if len(sub):
                ax.plot(sub["S_grid"], sub["continuous_visible_surface_depth_m"], lw=1.8, color=color, label=f"{side} panorama")
            if "lidar_depth_m" not in aligned.columns:
                continue
            comp = aligned[
                (aligned["source_set"] == source_set)
                & (aligned["side"] == side)
                & (aligned["lidar_ditch_exists"] == True)
                & pd.to_numeric(aligned["lidar_depth_m"], errors="coerce").notna()
            ].sort_values("S_grid")
            if source_set == "metric_eligible":
                comp = comp[comp["metric_evaluation_eligible_continuous"] == True]
            if len(comp):
                ax.scatter(comp["S_grid"], comp["lidar_depth_m"], s=10, color=color, alpha=0.25, marker="x", label=f"{side} LiDAR")
        ax.set_xlabel("Track chainage S (m)")
        ax.set_ylabel("Depth below rail reference (m)")
        ax.set_title(f"Continuous panorama-derived visible-surface profile ({source_set})")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(CFG.fig_dir / f"continuous_visible_surface_depth_{source_set}.png", dpi=180)
        plt.close(fig)


def plot_visibility_strip(grid: pd.DataFrame) -> None:
    source_set = "all_reference_accepted"
    sub = grid[(grid["source_set"] == source_set) & (grid["continuous_available"] == True)].copy()
    labels = [
        "bottom_visible_candidate",
        "monotonic_slope_surface",
        "no_clear_visible_depression",
        "boundary_surface_uncertain",
    ]
    label_to_int = {label: i for i, label in enumerate(labels)}
    sub["label_id"] = sub["weighted_visibility_class"].map(lambda x: label_to_int.get(str(x), len(labels)))
    pivot = sub.pivot_table(index="side", columns="S_grid", values="label_id", aggfunc="first")
    fig, ax = plt.subplots(figsize=(12, 2.7))
    im = ax.imshow(pivot.fillna(len(labels)).to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    cols = pivot.columns.to_numpy(dtype=float)
    tick_idx = np.linspace(0, len(cols) - 1, min(8, len(cols)), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{cols[i]:.0f}" for i in tick_idx])
    ax.set_xlabel("Track chainage S (m)")
    ax.set_title("Propagated panorama visibility class along chainage")
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(labels) + 1))
    cbar.ax.set_yticklabels(labels + ["other/NA"])
    fig.tight_layout()
    fig.savefig(CFG.fig_dir / "continuous_visibility_strip_map.png", dpi=180)
    plt.close(fig)


def plot_profile_heatmap(profile_stack: pd.DataFrame) -> None:
    if profile_stack.empty:
        return
    for side in ("left", "right"):
        sub = profile_stack[profile_stack["side"] == side].copy()
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="t_m", columns="S_cam", values="visible_surface_depth_m", aggfunc="median")
        pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(
            pivot.to_numpy(),
            aspect="auto",
            origin="lower",
            extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
            cmap="magma",
        )
        ax.set_xlabel("Target panorama chainage S (m)")
        ax.set_ylabel("Lateral distance T from track centre (m)")
        ax.set_title(f"Target-station side profile stack ({side})")
        fig.colorbar(im, ax=ax, label="Visible-surface depth (m)")
        fig.tight_layout()
        fig.savefig(CFG.fig_dir / f"target_profile_stack_heatmap_{side}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    print("Continuous panorama-derived visible-surface profile")
    ensure_dirs()
    pano, da3, lidar = load_inputs()
    continuity = panorama_continuity_report(pano, da3)
    station = write_target_station_table(da3)
    profile_stack = stack_target_profiles(da3)
    grid = build_continuous_grid(da3)
    aligned = build_lidar_aligned_table(grid, lidar)

    plot_depth_profiles(grid, aligned)
    plot_visibility_strip(grid)
    plot_profile_heatmap(profile_stack)

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "interpretation": (
            "The 1 m continuous profile is interpolated from 10 m-spaced panorama target stations. "
            "It represents visible-surface evidence, not direct hidden ditch-bottom depth."
        ),
        "panorama_continuity": continuity,
        "target_station_rows": int(len(station)),
        "profile_stack_rows": int(len(profile_stack)),
        "continuous_grid_rows": int(len(grid)),
        "available_grid_rows_by_source_set": {
            k: int(v) for k, v in grid[grid["continuous_available"] == True]["source_set"].value_counts().items()
        },
        "metric_summary_against_lidar": metric_summary(aligned),
        "outputs": {
            "target_station_results": str(CFG.out_dir / "panorama_target_station_visible_surface.csv"),
            "target_side_profile_stack": str(CFG.out_dir / "panorama_side_profile_stack.csv"),
            "continuous_1m_profile": str(CFG.out_dir / "panorama_continuous_visible_surface_1m.csv"),
            "lidar_aligned_1m_profile": str(CFG.out_dir / "panorama_lidar_aligned_visible_surface_1m.csv"),
            "figures": str(CFG.fig_dir),
        },
    }
    (CFG.out_dir / "panorama_continuous_profile_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nContinuity")
    print(json.dumps(continuity, indent=2, ensure_ascii=False))
    print("\nMetric summary against LiDAR")
    print(json.dumps(summary["metric_summary_against_lidar"], indent=2, ensure_ascii=False))
    print("\nOutputs")
    for key, value in summary["outputs"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()


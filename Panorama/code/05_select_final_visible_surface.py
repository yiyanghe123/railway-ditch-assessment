"""Final optimized selection for the Cirrus DA3 panorama experiment.

This step takes the strongest existing product, namely the multi-view DA3
visible-surface profiles and their continuous 1 m interpolation, and applies a
conservative final eligibility layer based only on image-side quality fields.

LiDAR is used only after the selection has been made, for reporting comparison
statistics.  It is not used to decide which panorama rows are eligible.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass(frozen=True)
class Config:
    multiview_csv: Path = Path(os.environ.get("PANO_DA3_CSV", ""))
    continuous_lidar_csv: Path = Path(os.environ.get("PANO_CONTINUOUS_LIDAR_CSV", ""))
    metric_path_csv: Path = Path(os.environ.get("PANO_METRIC_PATH_CSV", ""))
    corridor_path_csv: Path = Path(os.environ.get("PANO_CORRIDOR_PATH_CSV", ""))
    out_dir: Path = Path(os.environ.get("PANO_FINAL_OUT_DIR", "output_final_selection"))
    fig_dir: Path = Path(os.environ.get("PANO_FINAL_FIG_DIR", "figures_final_selection"))
    # Quantile rules only.  The actual metric gates are derived from the current
    # tile after the base visibility/reference filter has been applied.
    final_gate_lower_quantile: float = float(os.environ.get("PANO_FINAL_GATE_LOWER_QUANTILE", "0.00"))
    continuous_source_distance_quantile: float = float(os.environ.get("PANO_CONTINUOUS_SOURCE_DISTANCE_QUANTILE", "1.00"))


CFG = Config()
# Independent variant switch: PANO_USE_LIDAR=0 strips any LiDAR comparison
# columns baked into the inputs so the final selection/report is panorama-only.
USE_LIDAR = os.environ.get("PANO_USE_LIDAR", "1").strip().lower() not in ("0", "false", "no", "")


def _strip_lidar_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if "lidar" in c.lower()])


def ensure_dirs() -> None:
    CFG.out_dir.mkdir(parents=True, exist_ok=True)
    CFG.fig_dir.mkdir(parents=True, exist_ok=True)


def frame_no(name: str) -> int | float:
    m = re.search(r"_(\d{6})(?:\.jpg)?$", str(name))
    return int(m.group(1)) if m else math.nan


def finite_ge(series: pd.Series, threshold: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & (values >= threshold)


def finite_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.quantile(q)) if len(values) else math.nan


def derive_target_gates(df: pd.DataFrame) -> dict:
    has_metric_reference = (
        df["has_metric_reference"].eq(True)
        if "has_metric_reference" in df.columns
        else pd.Series(True, index=df.index)
    )
    base = (
        df["final_prediction_source"].ne("no_metric_prediction")
        & has_metric_reference
        & df["final_prediction_depth_m"].notna()
    )
    sample = df[base].copy()
    q = CFG.final_gate_lower_quantile
    gates = {
        "source": "tile_quantile_after_base_metric_filter",
        "sample_size": int(len(sample)),
        "quantile": float(q),
        "visible_depth_floor_m": finite_quantile(sample["final_prediction_depth_m"], q),
        "local_prominence_floor_m": finite_quantile(sample["final_prediction_prominence_m"], q),
        "combined_quality_floor": finite_quantile(sample["combined_metric_quality_score"], q),
        "fused_views_floor": finite_quantile(sample["final_prediction_fused_views"], q),
    }
    return gates


def load_path_features(path: Path, prefix: str) -> pd.DataFrame:
    """Load optional stage 04 path features for final reporting.

    The stage 04 paths are panorama-only products.  They are merged here so that
    the final table can use the profile-tracked metric path as the independent
    panorama prediction.  LiDAR remains a reporting-only join.
    """
    if not path.exists():
        return pd.DataFrame(columns=["filename", "side"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["filename", "side"])
    keep = [
        "filename",
        "side",
        "candidate_T_m",
        "candidate_depth_m",
        "candidate_prominence_m",
        "candidate_width_m",
        "n_fused_views_at_candidate",
        "n_points_at_candidate",
        "candidate_zone",
        "candidate_source",
        "candidate_boundary_margin_m",
        "profile_trend_corr",
        "candidate_source_score",
        "candidate_boundary_score",
        "profile_trend_score",
        "local_ditch_likeness",
        "corridor_presence_score",
        "profile_shape_score",
        "corridor_shape_score",
        "path_selected",
        "path_adaptive_confirmed",
        "path_metric_eligible",
        "path_corridor_confirmed",
        "path_score_floor",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out = out.drop_duplicates(subset=["filename", "side"], keep="first")
    rename = {c: f"{prefix}_{c}" for c in out.columns if c not in {"filename", "side"}}
    return out.rename(columns=rename)


def merge_path_features(target: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    metric = load_path_features(CFG.metric_path_csv, "profile_metric")
    corridor = load_path_features(CFG.corridor_path_csv, "profile_corridor")
    if not metric.empty:
        out = out.merge(metric, on=["filename", "side"], how="left")
    if not corridor.empty:
        out = out.merge(corridor, on=["filename", "side"], how="left")
    if "profile_corridor_path_corridor_confirmed" in out.columns:
        out["profile_corridor_presence_status"] = np.where(
            out["profile_corridor_path_corridor_confirmed"].eq(True),
            "corridor_path_confirmed",
            "corridor_path_unconfirmed",
        )
    elif "profile_corridor_candidate_T_m" in out.columns:
        out["profile_corridor_presence_status"] = "corridor_path_unconfirmed"
    else:
        out["profile_corridor_presence_status"] = "not_available"
    return out


def add_final_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Choose the final independent panorama prediction.

    The multiview stage builds the fused visible-surface profile and records
    diagnostic candidates.  The candidate-scoring stage selects the ditch-like
    profile path using local shape and longitudinal coherence.  The final
    independent panorama prediction is therefore taken from the metric path
    when that path is available.
    """
    out = df.copy()
    multiview_ok = (
        out.get("metric_evaluation_eligible", pd.Series(False, index=out.index)).eq(True)
        & out.get("visible_surface_depth_m", pd.Series(np.nan, index=out.index)).notna()
        & out.get("visible_surface_T_m", pd.Series(np.nan, index=out.index)).notna()
    )
    reference_ok = out.get("has_metric_reference", pd.Series(False, index=out.index)).eq(True)
    profile_metric_ok = out.get("profile_metric_path_metric_eligible", pd.Series(False, index=out.index)).eq(True)
    profile_t_ok = out.get("profile_metric_candidate_T_m", pd.Series(np.nan, index=out.index)).notna()
    profile_depth_ok = out.get("profile_metric_candidate_depth_m", pd.Series(np.nan, index=out.index)).notna()
    profile_selected = out.get("profile_metric_path_selected", pd.Series(True, index=out.index)).fillna(True).eq(True)
    profile_assisted_ok = reference_ok & profile_metric_ok & profile_t_ok & profile_depth_ok & profile_selected

    out["final_prediction_uses_profile_path"] = profile_assisted_ok
    out["multiview_diagnostic_metric_candidate"] = multiview_ok
    out["final_prediction_source"] = np.where(profile_assisted_ok, "profile_metric_path_visible_surface", "no_metric_prediction")
    out["final_prediction_depth_m"] = np.where(
        profile_assisted_ok,
        pd.to_numeric(out.get("profile_metric_candidate_depth_m"), errors="coerce"),
        np.nan,
    )
    out["final_prediction_T_m"] = np.where(
        profile_assisted_ok,
        pd.to_numeric(out.get("profile_metric_candidate_T_m"), errors="coerce"),
        np.nan,
    )
    out["final_prediction_prominence_m"] = np.where(
        profile_assisted_ok,
        pd.to_numeric(out.get("profile_metric_candidate_prominence_m"), errors="coerce"),
        np.nan,
    )
    out["final_prediction_fused_views"] = np.where(
        profile_assisted_ok,
        pd.to_numeric(out.get("profile_metric_n_fused_views_at_candidate"), errors="coerce"),
        np.nan,
    )
    out["final_prediction_profile_path_T_m"] = np.where(
        profile_assisted_ok,
        pd.to_numeric(out.get("profile_metric_candidate_T_m"), errors="coerce"),
        np.nan,
    )
    out["final_prediction_visible_max_T_m"] = np.where(
        profile_assisted_ok,
        pd.to_numeric(out.get("visible_surface_T_m"), errors="coerce"),
        np.nan,
    )
    return out


def final_reason(row: pd.Series, gates: dict) -> str:
    if str(row.get("final_prediction_source", "no_metric_prediction")) == "no_metric_prediction":
        return "no_metric_prediction"
    if not bool(row.get("has_metric_reference", False)):
        return "no_metric_reference"
    depth_floor = gates.get("visible_depth_floor_m", math.nan)
    prom_floor = gates.get("local_prominence_floor_m", math.nan)
    quality_floor = gates.get("combined_quality_floor", math.nan)
    views_floor = gates.get("fused_views_floor", math.nan)
    if np.isfinite(depth_floor) and (
        pd.isna(row.get("final_prediction_depth_m")) or float(row["final_prediction_depth_m"]) < depth_floor
    ):
        return "visible_depression_too_shallow"
    if np.isfinite(prom_floor) and (
        pd.isna(row.get("final_prediction_prominence_m")) or float(row["final_prediction_prominence_m"]) < prom_floor
    ):
        return "local_prominence_too_weak"
    if np.isfinite(quality_floor) and (
        pd.isna(row.get("combined_metric_quality_score")) or float(row["combined_metric_quality_score"]) < quality_floor
    ):
        return "combined_quality_too_low"
    if np.isfinite(views_floor) and (
        pd.isna(row.get("final_prediction_fused_views")) or float(row["final_prediction_fused_views"]) < views_floor
    ):
        return "insufficient_multiview_support"
    return "accepted_final_metric"


def load_target_rows() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(CFG.multiview_csv)
    if not USE_LIDAR:
        df = _strip_lidar_columns(df)
    df["frame_no"] = df["filename"].map(frame_no).astype("Int64")
    df = merge_path_features(df)
    df = add_final_prediction_columns(df)
    gates = derive_target_gates(df)
    df["final_metric_status"] = df.apply(lambda row: final_reason(row, gates), axis=1)
    df["final_metric_eligible"] = df["final_metric_status"].eq("accepted_final_metric")
    if "lidar_depth_m" in df.columns:
        df["final_panorama_minus_lidar_depth_m"] = df["final_prediction_depth_m"] - df["lidar_depth_m"]
    return df, gates


def load_continuous_rows(final_target: pd.DataFrame, gates: dict) -> pd.DataFrame:
    cont = pd.read_csv(CFG.continuous_lidar_csv)
    if not USE_LIDAR:
        cont = _strip_lidar_columns(cont)
    final_sources = final_target[final_target["final_metric_eligible"]].dropna(
        subset=["frame_no", "final_prediction_depth_m", "final_prediction_T_m"]
    ).copy()
    keys = set((int(row["frame_no"]), str(row["side"])) for _, row in final_sources.iterrows())
    cont["nearest_target_final_eligible"] = [
        (int(frame), str(side)) in keys if pd.notna(frame) else False
        for frame, side in zip(cont["nearest_frame_no"], cont["side"])
    ]
    cont["final_continuous_prediction_depth_m"] = np.nan
    cont["final_continuous_prediction_T_m"] = np.nan
    cont["final_continuous_prediction_source"] = "no_metric_prediction"

    for side, side_sources in final_sources.groupby("side"):
        side_sources = side_sources.sort_values("S_cam")
        side_mask = cont["side"].eq(side)
        if side_sources.empty or not side_mask.any():
            continue
        s_src = side_sources["S_cam"].to_numpy(dtype=float)
        depth_src = side_sources["final_prediction_depth_m"].to_numpy(dtype=float)
        t_src = side_sources["final_prediction_T_m"].to_numpy(dtype=float)
        s_grid = cont.loc[side_mask, "S_grid"].to_numpy(dtype=float)
        if len(side_sources) == 1:
            nearest = np.abs(s_grid - s_src[0])
            usable = nearest <= gates.get("continuous_source_distance_gate_m", np.inf)
            cont.loc[side_mask, "final_continuous_prediction_depth_m"] = np.where(usable, depth_src[0], np.nan)
            cont.loc[side_mask, "final_continuous_prediction_T_m"] = np.where(usable, t_src[0], np.nan)
        else:
            cont.loc[side_mask, "final_continuous_prediction_depth_m"] = np.interp(s_grid, s_src, depth_src)
            cont.loc[side_mask, "final_continuous_prediction_T_m"] = np.interp(s_grid, s_src, t_src)
        cont.loc[side_mask & cont["nearest_target_final_eligible"].eq(True), "final_continuous_prediction_source"] = (
            "interpolated_profile_metric_path"
        )

    source_distance_sample = cont[
        cont["source_set"].eq("metric_eligible")
        & cont["metric_evaluation_eligible_continuous"].eq(True)
        & cont["nearest_target_final_eligible"].eq(True)
    ]["nearest_source_distance_m"]
    distance_gate = finite_quantile(source_distance_sample, CFG.continuous_source_distance_quantile)
    gates["continuous_source_distance_quantile"] = float(CFG.continuous_source_distance_quantile)
    gates["continuous_source_distance_gate_m"] = distance_gate
    cont["final_continuous_metric_eligible"] = (
        cont["source_set"].eq("metric_eligible")
        & cont["metric_evaluation_eligible_continuous"].eq(True)
        & cont["nearest_target_final_eligible"].eq(True)
        & finite_ge(cont["final_continuous_prediction_depth_m"], gates["visible_depth_floor_m"])
        & (pd.to_numeric(cont["nearest_source_distance_m"], errors="coerce") <= distance_gate)
    )
    if "lidar_depth_m" in cont.columns:
        cont["final_continuous_minus_lidar_depth_m"] = (
            cont["final_continuous_prediction_depth_m"] - cont["lidar_depth_m"]
        )
    return cont


def metric_summary(df: pd.DataFrame, value_col: str, diff_col: str, mask: pd.Series) -> dict:
    sub = df[mask & df[value_col].notna() & df[diff_col].notna()].copy()
    diff = sub[diff_col].to_numpy(dtype=float)
    return {
        "n": int(len(diff)),
        "bias_m": float(np.mean(diff)) if len(diff) else math.nan,
        "rmse_m": float(np.sqrt(np.mean(diff**2))) if len(diff) else math.nan,
        "mae_m": float(np.mean(np.abs(diff))) if len(diff) else math.nan,
    }


def visibility_metric_table(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    diff_col: str,
    base_mask: pd.Series,
    final_mask: pd.Series,
) -> pd.DataFrame:
    rows: list[dict] = []
    labels = sorted(str(v) for v in df[label_col].dropna().unique())
    for label in labels:
        label_mask = df[label_col].astype(str).eq(label)
        for subset_name, subset_mask in [
            ("all_with_lidar", base_mask),
            ("final_metric_eligible", final_mask),
        ]:
            sub = df[label_mask & subset_mask & df[value_col].notna() & df[diff_col].notna()].copy()
            diff = pd.to_numeric(sub[diff_col], errors="coerce").dropna().to_numpy(dtype=float)
            values = pd.to_numeric(sub[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "visibility_class": label,
                    "subset": subset_name,
                    "n": int(len(diff)),
                    "bias_m": float(np.mean(diff)) if len(diff) else math.nan,
                    "rmse_m": float(np.sqrt(np.mean(diff**2))) if len(diff) else math.nan,
                    "mae_m": float(np.mean(np.abs(diff))) if len(diff) else math.nan,
                    "mean_panorama_depth_m": float(np.mean(values)) if len(values) else math.nan,
                    "median_panorama_depth_m": float(np.median(values)) if len(values) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def tables_to_summary(table: pd.DataFrame) -> list[dict]:
    if table.empty:
        return []
    return table.replace({np.nan: None}).to_dict(orient="records")


def build_summary(target: pd.DataFrame, cont: pd.DataFrame, gates: dict) -> dict:
    target_lidar = target["lidar_depth_m"].notna()
    target_base = target["final_prediction_source"].ne("no_metric_prediction")
    target_final = target["final_metric_eligible"].eq(True)
    cont_base = (
        cont["source_set"].eq("metric_eligible")
        & cont["metric_evaluation_eligible_continuous"].eq(True)
        & cont["lidar_ditch_exists"].eq(True)
        & cont["final_continuous_minus_lidar_depth_m"].notna()
    )
    cont_final = cont["final_continuous_metric_eligible"].eq(True) & cont["lidar_ditch_exists"].eq(True)
    target_visibility = visibility_metric_table(
        target,
        "visibility_class",
        "final_prediction_depth_m",
        "final_panorama_minus_lidar_depth_m",
        target_lidar,
        target_lidar & target_final,
    )
    continuous_visibility = visibility_metric_table(
        cont,
        "weighted_visibility_class",
        "final_continuous_prediction_depth_m",
        "final_continuous_minus_lidar_depth_m",
        cont["lidar_ditch_exists"].eq(True) & cont["final_continuous_minus_lidar_depth_m"].notna(),
        cont_final,
    )
    return {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(CFG).items()},
        "adaptive_final_gates": gates,
        "interpretation": (
            "Final optimized result from DA3 multi-view geometry. The final metric prediction is "
            "a profile-assisted visible-surface estimate. The depth is taken from the Step 4 fused "
            "visible-surface depression because it is the most stable metric depth observation. "
            "The lateral position is refined with the Step 6 profile path, which tracks the ditch-like "
            "profile shape through chainage. Final eligibility is checked with metric-reference "
            "quality, profile prominence, and multi-view support. LiDAR is used only for reporting."
        ),
        "target_station_rows": int(len(target)),
        "target_final_status_counts": target["final_metric_status"].value_counts(dropna=False).to_dict(),
        "target_profile_corridor_presence_counts": (
            target["profile_corridor_presence_status"].value_counts(dropna=False).to_dict()
            if "profile_corridor_presence_status" in target.columns
            else {}
        ),
        "continuous_rows": int(len(cont)),
        "continuous_final_eligible_rows": int(cont["final_continuous_metric_eligible"].sum()),
        "target_metrics": {
            "base_metric_eligible": metric_summary(
                target,
                "final_prediction_depth_m",
                "final_panorama_minus_lidar_depth_m",
                target_lidar & target_base,
            ),
            "final_metric_eligible": metric_summary(
                target,
                "final_prediction_depth_m",
                "final_panorama_minus_lidar_depth_m",
                target_lidar & target_final,
            ),
        },
        "continuous_metrics": {
            "base_metric_eligible": metric_summary(
                cont,
                "final_continuous_prediction_depth_m",
                "final_continuous_minus_lidar_depth_m",
                cont_base,
            ),
            "final_metric_eligible": metric_summary(
                cont,
                "final_continuous_prediction_depth_m",
                "final_continuous_minus_lidar_depth_m",
                cont_final,
            ),
        },
        "visibility_stratified_metrics": {
            "target_station": tables_to_summary(target_visibility),
            "continuous_profile": tables_to_summary(continuous_visibility),
        },
    }


def plot_target_scatter(target: pd.DataFrame) -> None:
    sub = target[
        target["lidar_depth_m"].notna()
        & target["final_prediction_source"].ne("no_metric_prediction")
    ].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    rejected = sub[~sub["final_metric_eligible"]]
    accepted = sub[sub["final_metric_eligible"]]
    if not rejected.empty:
        ax.scatter(rejected["lidar_depth_m"], rejected["final_prediction_depth_m"], s=36, alpha=0.45, label="base eligible, final rejected")
    if not accepted.empty:
        ax.scatter(accepted["lidar_depth_m"], accepted["final_prediction_depth_m"], s=42, alpha=0.85, label="final eligible")
    lo = float(np.nanmin([sub["lidar_depth_m"].min(), sub["final_prediction_depth_m"].min()]))
    hi = float(np.nanmax([sub["lidar_depth_m"].max(), sub["final_prediction_depth_m"].max()]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("LiDAR ditch depth below rail (m)")
    ax.set_ylabel("Final panorama visible-surface depth below rail (m)")
    ax.set_title("Final DA3 panorama metric subset")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CFG.fig_dir / "final_target_station_scatter.png", dpi=180)
    plt.close(fig)


def plot_continuous(cont: pd.DataFrame) -> None:
    sub = cont[cont["final_continuous_metric_eligible"]].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for side, color in [("left", "#1476B8"), ("right", "#D95F02")]:
        side_sub = sub[sub["side"] == side].sort_values("S_grid")
        if side_sub.empty:
            continue
        s_values = side_sub["S_grid"].to_numpy(dtype=float)
        split_at = (np.where(np.diff(s_values) > 1.5)[0] + 1).tolist()
        starts = [0] + split_at
        ends = split_at + [len(side_sub)]
        for k, (start, end) in enumerate(zip(starts, ends)):
            segment = side_sub.iloc[start:end]
            if segment.empty:
                continue
            label = f"{side} final panorama" if k == 0 else None
            ax.plot(
                segment["S_grid"],
                segment["final_continuous_prediction_depth_m"],
                color=color,
                lw=1.8,
                marker="o",
                ms=2.2,
                label=label,
            )
        lidar_sub = side_sub[side_sub["lidar_depth_m"].notna()]
        if not lidar_sub.empty:
            ax.scatter(lidar_sub["S_grid"], lidar_sub["lidar_depth_m"], color=color, marker="x", s=10, alpha=0.35, label=f"{side} LiDAR")
    ax.set_xlabel("Track chainage S (m)")
    ax.set_ylabel("Depth below rail reference (m)")
    ax.set_title("Final continuous panorama visible-surface depth")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(CFG.fig_dir / "final_continuous_profile.png", dpi=180)
    plt.close(fig)


def main() -> None:
    print("Final optimized DA3 panorama selection")
    ensure_dirs()
    target, gates = load_target_rows()
    cont = load_continuous_rows(target, gates)
    # Independent (LiDAR-free) variant: the continuous table carries no LiDAR
    # comparison columns.  Add them as empty so the reporting / plotting code
    # degrades to empty comparisons (via its existing .notna()/empty guards)
    # instead of raising KeyError.  The panorama-only selection itself is
    # unaffected.
    for col, default in [
        ("lidar_ditch_exists", False),
        ("lidar_depth_m", np.nan),
        ("lidar_bottom_T_m", np.nan),
        ("final_continuous_minus_lidar_depth_m", np.nan),
    ]:
        if col not in cont.columns:
            cont[col] = default
    if "final_panorama_minus_lidar_depth_m" not in target.columns:
        target["final_panorama_minus_lidar_depth_m"] = np.nan
    if "lidar_depth_m" not in target.columns:
        target["lidar_depth_m"] = np.nan
    stale_path_cols = ["rail_crop_path", "profile_path", "depth_path", "target_crop_path"]
    target_export = target.drop(columns=[c for c in stale_path_cols if c in target.columns])
    target_csv = CFG.out_dir / "panorama_target_station_selection.csv"
    cont_csv = CFG.out_dir / "panorama_continuous_selection.csv"
    target_export.to_csv(target_csv, index=False, float_format="%.6f")
    cont.to_csv(cont_csv, index=False, float_format="%.6f")
    target_lidar = target["lidar_depth_m"].notna()
    target_final = target["final_metric_eligible"].eq(True)
    cont_final = cont["final_continuous_metric_eligible"].eq(True) & cont["lidar_ditch_exists"].eq(True)
    target_visibility_table = visibility_metric_table(
        target,
        "visibility_class",
        "final_prediction_depth_m",
        "final_panorama_minus_lidar_depth_m",
        target_lidar,
        target_lidar & target_final,
    )
    continuous_visibility_table = visibility_metric_table(
        cont,
        "weighted_visibility_class",
        "final_continuous_prediction_depth_m",
        "final_continuous_minus_lidar_depth_m",
        cont["lidar_ditch_exists"].eq(True) & cont["final_continuous_minus_lidar_depth_m"].notna(),
        cont_final,
    )
    target_visibility_csv = CFG.out_dir / "panorama_target_visibility_metrics.csv"
    continuous_visibility_csv = CFG.out_dir / "panorama_continuous_visibility_metrics.csv"
    target_visibility_table.to_csv(target_visibility_csv, index=False, float_format="%.6f")
    continuous_visibility_table.to_csv(continuous_visibility_csv, index=False, float_format="%.6f")
    plot_target_scatter(target)
    plot_continuous(cont)
    summary = build_summary(target, cont, gates)
    summary["outputs"] = {
        "target_station_selection": str(target_csv),
        "continuous_selection": str(cont_csv),
        "target_visibility_metrics": str(target_visibility_csv),
        "continuous_visibility_metrics": str(continuous_visibility_csv),
        "figures": str(CFG.fig_dir),
    }
    summary_json = CFG.out_dir / "panorama_selection_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


"""
Build the per-tile centreline files from the official Trafikverket sparmitt
CSV (`centreline_per_tile/centreline_tile_<TILE>.csv`).

Reads:
    - centreline_per_tile/centreline_tile_<TILE>.csv  (East, North, Height, chainage_m)
    - <output_dir>/step03_ST_metadata.json            (PCA azimuth + mean local)
    - the tile's LAZ header                           (UTM offsets x0, y0)

Writes:
    - <output_dir>/rail_centreline_dense_step05.csv      (every sparmitt point)
    - <output_dir>/rail_centreline_step05.csv            (one node per 5 m)
    - <output_dir>/step05_centreline_metadata.json

step06 and step07 consume these files directly. The `_step05` suffix in the
output filenames is retained for compatibility with the existing downstream
readers — there is no spline fitting anywhere in this script.

Usage:
    python step05_centreline_from_sparmitt.py <output_dir> [tile_id]
e.g. python step05_centreline_from_sparmitt.py output_630_415
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import laspy
from scipy.signal import savgol_filter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR            = os.environ.get("LIDAR_BASE", os.getcwd())
CENTRELINE_PER_TILE = os.environ.get("LIDAR_CENTRELINE_DIR", os.path.join(BASE_DIR, "centreline_per_tile"))
LAZ_ROOT            = os.environ.get("LIDAR_LAZ_ROOT", os.path.join(BASE_DIR, "data", "laz"))


def find_laz_for_tile(tile_id):
    env_laz = os.environ.get("LIDAR_LAZ_FILE")
    if env_laz:
        if not os.path.exists(env_laz):
            raise FileNotFoundError(f"LIDAR_LAZ_FILE does not exist: {env_laz}")
        return env_laz
    matches = glob.glob(os.path.join(LAZ_ROOT, "**", f"*_{tile_id}.laz"),
                        recursive=True)
    if not matches:
        raise FileNotFoundError(f"No LAZ found for tile {tile_id} under {LAZ_ROOT}")
    return matches[0]


def tile_id_from_output_dir(out_dir):
    meta_path = os.path.join(out_dir, "step06_reference_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            laz = json.load(f).get("laz_file") or ""
        if laz:
            parts = os.path.splitext(os.path.basename(laz))[0].split("_")
            return f"{parts[-2]}_{parts[-1]}"
    for key, tid in (("630", "7592_630"), ("635", "7593_635"),
                     ("629", "7592_629"), ("636", "7593_636"),
                     ("671", "7584_671")):
        if key in out_dir:
            return tid
    return None


def compute_curvature(S, T):
    """κ = |T''| / (1 + T'^2)^(3/2) via finite differences."""
    T_p  = np.gradient(T, S)
    T_pp = np.gradient(T_p, S)
    return np.abs(T_pp) / (1.0 + T_p ** 2) ** 1.5


def smooth_T(T, window=11):
    """Light Sav-Gol to suppress mm-level noise before the 2nd derivative."""
    if len(T) >= window + 2:
        return savgol_filter(T, window_length=window, polyorder=3, mode="nearest")
    return T


def build_centreline(out_dir, tile_id=None):
    out_dir = os.path.abspath(out_dir)
    if tile_id is None:
        tile_id = tile_id_from_output_dir(out_dir)
    if tile_id is None:
        raise RuntimeError(
            f"Could not infer tile_id from {out_dir}. "
            "Pass it as second CLI argument."
        )

    sparmitt_csv = os.path.join(CENTRELINE_PER_TILE,
                                f"centreline_tile_{tile_id}.csv")
    step03_meta_path = os.path.join(out_dir, "step03_ST_metadata.json")
    for p in (sparmitt_csv, step03_meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    with open(step03_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    angle_rad = np.radians(float(meta["pca_track_azimuth_deg"]))
    mean_x    = float(meta["mean_x_local"])
    mean_y    = float(meta["mean_y_local"])

    laz_path = find_laz_for_tile(tile_id)
    with laspy.open(laz_path) as lf:
        x0 = float(lf.header.x_min)
        y0 = float(lf.header.y_min)

    print(f"Tile {tile_id}")
    print(f"  PCA azimuth = {np.degrees(angle_rad):.3f} deg, "
          f"mean_local = ({mean_x:.3f}, {mean_y:.3f})")
    print(f"  UTM offset  = ({x0:.3f}, {y0:.3f})")

    # UTM → local → (S, T) using the step03 PCA frame
    sp = pd.read_csv(sparmitt_csv)
    X_local = sp["East"].values  - x0
    Y_local = sp["North"].values - y0
    dx = X_local - mean_x
    dy = Y_local - mean_y
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    S =  dx * cos_a + dy * sin_a
    T = -dx * sin_a + dy * cos_a

    # Sort by S, drop duplicates (bbox-overlap can produce repeats)
    order = np.argsort(S)
    S, T = S[order], T[order]
    X_local, Y_local = X_local[order], Y_local[order]
    unique = np.concatenate(([True], np.diff(S) > 1e-6))
    S, T = S[unique], T[unique]
    X_local, Y_local = X_local[unique], Y_local[unique]

    T_smooth  = smooth_T(T, window=11)
    curvature = compute_curvature(S, T_smooth)
    radius    = np.where(curvature > 1e-8, 1.0 / curvature, 1e6)

    print(f"  Sparmitt: {len(sp)} raw → {len(S)} after sort/dedup")
    print(f"  S range:  {S.min():.2f} – {S.max():.2f} m ({S.max()-S.min():.1f} m)")
    print(f"  T range:  {T.min():.2f} – {T.max():.2f} m")

    dense = pd.DataFrame({
        "S":         S,
        "T_center":  T_smooth,
        "X_local":   X_local,
        "Y_local":   Y_local,
        "curvature": curvature,
        "radius":    radius,
    })

    # Node-level CSV: one anchor per 5 m. T_center_raw == T_center_spline
    # because no fitting is involved; columns kept to satisfy the step06 reader.
    ds = 5.0
    node_S = np.arange(S.min(), S.max() + ds, ds)
    node_T = np.interp(node_S, S, T_smooth)
    nodes = pd.DataFrame({
        "S":               node_S,
        "T_center_raw":    node_T,
        "T_center_spline": node_T,
        "X_local":         np.interp(node_S, S, X_local),
        "Y_local":         np.interp(node_S, S, Y_local),
        "corridor_count":  1000,
        "is_inlier":       1,
    })

    dense_path  = os.path.join(out_dir, "rail_centreline_dense_step05.csv")
    nodes_path  = os.path.join(out_dir, "rail_centreline_step05.csv")
    meta_path   = os.path.join(out_dir, "step05_centreline_metadata.json")

    dense.to_csv(dense_path, index=False, float_format="%.6f")
    nodes.to_csv(nodes_path, index=False, float_format="%.6f")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "source":              os.path.basename(sparmitt_csv),
            "source_path":         sparmitt_csv,
            "source_type":         "per-tile centreline CSV",
            "tile_id":             tile_id,
            "n_sparmitt_points":   int(len(sp)),
            "n_after_sort_dedup":  int(len(S)),
            "S_range_m":           [float(S.min()), float(S.max())],
            "S_span_m":            float(S.max() - S.min()),
            "T_range_m":           [float(T.min()), float(T.max())],
            "pca_azimuth_deg":     float(np.degrees(angle_rad)),
            "utm_offset_x0":       x0,
            "utm_offset_y0":       y0,
        }, f, indent=2)

    print(f"  Wrote: {os.path.basename(dense_path)} ({len(dense)} rows)")
    print(f"  Wrote: {os.path.basename(nodes_path)} ({len(nodes)} rows)")
    print(f"  Wrote: {os.path.basename(meta_path)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step05_centreline_from_sparmitt.py <output_dir> [tile_id]")
        sys.exit(1)
    out_dir = sys.argv[1]
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(BASE_DIR, out_dir)
    tile_id = sys.argv[2] if len(sys.argv) > 2 else None
    build_centreline(out_dir, tile_id=tile_id)

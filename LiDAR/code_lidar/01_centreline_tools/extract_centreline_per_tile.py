"""
Cut a full railway centreline text file into per-tile CSV files matching
the LAZ bounding boxes.

Input
-----
  <centreline.txt>    East North Height (space-separated)
  <laz-root>/**/*.laz all LAZ tiles

Output
------
  <output-dir>/
    centreline_tile_<TILE_ID>.csv     East, North, Height, chainage_m
    summary.csv                        one row per tile with E/N range, #points

The `chainage_m` column is cumulative horizontal distance along the
centreline from the first point in the tile — useful as an S coordinate
substitute without needing the step03 PCA rotation.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import laspy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.environ.get("LIDAR_BASE", os.getcwd())
CENTRELINE_FILE = os.environ.get(
    "LIDAR_FULL_CENTRELINE_FILE",
    os.path.join(BASE_DIR, "data", "centreline", "sparmitt.txt"),
)
LAZ_ROOT = os.environ.get("LIDAR_LAZ_ROOT", os.path.join(BASE_DIR, "data", "laz"))
OUT_DIR = os.environ.get("LIDAR_CENTRELINE_DIR", os.path.join(BASE_DIR, "centreline_per_tile"))
BUFFER_M        = 20.0   # include a small buffer beyond the tile extent


def tile_id_from_path(laz_path):
    """las_BD_111_0_1_2024_0_7592_630.laz → '7592_630'"""
    name = os.path.basename(laz_path)
    # Strip 'las_' prefix and '.laz' suffix, take last two underscore-separated fields
    stem = os.path.splitext(name)[0]
    parts = stem.split("_")
    return f"{parts[-2]}_{parts[-1]}"


def get_laz_bbox(laz_path):
    """Read just the LAZ header to get (xmin, xmax, ymin, ymax) without
    loading the point cloud."""
    with laspy.open(laz_path) as f:
        h = f.header
        return float(h.x_min), float(h.x_max), float(h.y_min), float(h.y_max)


def cumulative_chainage(E, N):
    """Cumulative horizontal distance along a polyline."""
    dx = np.diff(E)
    dy = np.diff(N)
    d = np.sqrt(dx * dx + dy * dy)
    return np.concatenate(([0.0], np.cumsum(d)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load full centreline ───────────────────────────
    print(f"Loading centreline: {CENTRELINE_FILE}")
    data = np.loadtxt(CENTRELINE_FILE)
    if data.shape[1] != 3:
        raise RuntimeError(f"Expected 3 columns (E N Z), got {data.shape[1]}")
    E_all, N_all, Z_all = data[:, 0], data[:, 1], data[:, 2]
    print(f"  {len(data):,} points  E=[{E_all.min():.0f}, {E_all.max():.0f}]  "
          f"N=[{N_all.min():.0f}, {N_all.max():.0f}]")
    print()

    # ── Find all LAZ tiles ─────────────────────────────
    laz_paths = sorted(glob.glob(os.path.join(LAZ_ROOT, "**", "*.laz"),
                                 recursive=True))
    print(f"Found {len(laz_paths)} LAZ tile(s):")
    for p in laz_paths:
        print(f"  {p}")
    print()

    # ── Process each tile ──────────────────────────────
    summary_rows = []
    for laz_path in laz_paths:
        tid = tile_id_from_path(laz_path)
        xmin, xmax, ymin, ymax = get_laz_bbox(laz_path)
        # extend by buffer
        xmin -= BUFFER_M; xmax += BUFFER_M
        ymin -= BUFFER_M; ymax += BUFFER_M

        mask = (E_all >= xmin) & (E_all <= xmax) & (N_all >= ymin) & (N_all <= ymax)
        n_pts = int(mask.sum())

        if n_pts == 0:
            print(f"  [skip] tile {tid}: no centreline points in bbox "
                  f"E=[{xmin:.0f}, {xmax:.0f}] N=[{ymin:.0f}, {ymax:.0f}]")
            continue

        E_t = E_all[mask]
        N_t = N_all[mask]
        Z_t = Z_all[mask]

        # Sort along centreline order — centreline file is already ordered
        # by chainage, so we can just keep the original order (mask preserves it)
        chain = cumulative_chainage(E_t, N_t)

        df = pd.DataFrame({
            "East":        E_t,
            "North":       N_t,
            "Height":      Z_t,
            "chainage_m":  chain,
        })
        out_csv = os.path.join(OUT_DIR, f"centreline_tile_{tid}.csv")
        df.to_csv(out_csv, index=False, float_format="%.4f")

        summary_rows.append({
            "tile_id":      tid,
            "n_points":     n_pts,
            "E_min":        float(E_t.min()),
            "E_max":        float(E_t.max()),
            "N_min":        float(N_t.min()),
            "N_max":        float(N_t.max()),
            "Z_min":        float(Z_t.min()),
            "Z_max":        float(Z_t.max()),
            "chainage_m":   float(chain[-1]),
            "laz_path":     laz_path,
            "csv_out":      out_csv,
        })
        print(f"  tile {tid}: {n_pts:4d} pts, length {chain[-1]:6.1f} m "
              f"→ {os.path.basename(out_csv)}")

    # ── Write summary ──────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(OUT_DIR, "summary.csv")
        summary_df.to_csv(summary_csv, index=False, float_format="%.3f")
        print()
        print(f"Summary: {len(summary_rows)} tile(s) written to {summary_csv}")
    else:
        print("\nNo tiles had centreline coverage.")


if __name__ == "__main__":
    main()

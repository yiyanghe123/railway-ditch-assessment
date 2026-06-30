"""Command-line runner for the LiDAR ditch condition pipeline.

This wrapper keeps the algorithm code under code_lidar/ unchanged for normal
use, while exposing the project paths as command-line arguments. It is the
recommended entry point when the input LAZ root, centreline directory, or output
folder changes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CENTRELINE_DIR = ROOT / "centreline_per_tile"
DEFAULT_LAZ_ROOT = ROOT / "data" / "laz"
INNER_RUNNER = ROOT / "code_lidar" / "run" / "run_pipeline.py"


def resolve_path(text: str | None, default: Path | None = None) -> Path | None:
    if text is None:
        return default
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LiDAR ditch condition pipeline.")
    parser.add_argument("--tile", required=True, help="Tile or section id.")
    parser.add_argument("--laz", default=None, help="Explicit LAZ file. If omitted, --laz-root is searched.")
    parser.add_argument("--laz-root", default=str(DEFAULT_LAZ_ROOT), help="Directory searched for *_<tile>.laz.")
    parser.add_argument("--centreline-dir", default=str(DEFAULT_CENTRELINE_DIR),
                        help="Directory containing centreline_tile_<tile>.csv.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to output_<tile>.")
    parser.add_argument("--mode", choices=["full", "downstream", "step07", "step08"], default="full")
    parser.add_argument("--use-adaptive-thresholds", choices=["1", "0"], default="1")
    args = parser.parse_args()

    laz_root = resolve_path(args.laz_root)
    out_dir = resolve_path(args.out_dir, ROOT / f"output_{args.tile}")
    centreline_dir = resolve_path(args.centreline_dir)

    cmd = [
        sys.executable,
        str(INNER_RUNNER),
        "--tile", args.tile,
        "--laz-root", str(laz_root),
        "--out-dir", str(out_dir),
        "--mode", args.mode,
        "--use-adaptive-thresholds", args.use_adaptive_thresholds,
    ]
    if args.laz:
        cmd.extend(["--laz", str(resolve_path(args.laz))])

    env = os.environ.copy()
    env["LIDAR_BASE"] = str(ROOT)
    env["LIDAR_CENTRELINE_DIR"] = str(centreline_dir)
    env["LIDAR_LAZ_ROOT"] = str(laz_root)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"

    print("LiDAR command-line runner")
    print(f"tile           : {args.tile}")
    print(f"laz_root       : {laz_root}")
    print(f"centreline_dir : {centreline_dir}")
    print(f"out_dir        : {out_dir}")
    print(f"mode           : {args.mode}")

    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()

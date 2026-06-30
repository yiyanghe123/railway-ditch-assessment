"""
Generic runner for the LiDAR ditch-detection pipeline.

Examples
--------
Example tile:

    python code_lidar\\run\\run_pipeline.py --tile <tile-id> --out-dir output_<tile-id> --mode full

Re-run only ditch analysis and overview after updating adaptive thresholds:

    python code_lidar\\run\\run_pipeline.py --tile <tile-id> --out-dir output_<tile-id> --mode downstream

Run another tile:

    python code_lidar\\run\\run_pipeline.py --tile <tile-id> --out-dir output_<tile-id> --mode full
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import subprocess
import sys
import time
from pathlib import Path


BASE = Path(os.environ.get("LIDAR_BASE") or Path(__file__).resolve().parents[2])
DEFAULT_LAZ_ROOT = BASE / "data" / "laz"

# Adaptive thresholds are regenerated from this run's own geometry every time
# (no precomputed JSON is reused). The validated thresholds are a single
# bootstrap iteration, so the full run is: cold-start thresholds -> step07
# (pass 1) -> bootstrap thresholds from pass-1 detections -> step07 (final).
_THRESH = "code_lidar/03_adaptive_thresholds/compute_adaptive_thresholds.py"
_THRESH_ARGS = ["--tile", "{tile}", "--out-dir", "{out_dir}", "--laz", "{laz}"]

FULL_STEPS = [
    ("code_lidar/00_calibration/step01_explore.py", "Step 01 - Data exploration", []),
    ("code_lidar/00_calibration/step02_calibrate.py", "Step 02 - Corridor calibration", []),
    ("code_lidar/02_pipeline/step03_track_direction.py", "Step 03 - Track direction", []),
    ("code_lidar/02_pipeline/step04_rotate_and_crosssection.py", "Step 04 - Track-aligned projection", []),
    ("code_lidar/02_pipeline/step05_centreline_from_sparmitt.py", "Step 05 - Official sparmitt centreline", ["{out_dir}", "{tile}"]),
    ("code_lidar/02_pipeline/step06_railpeaks_RUK.py", "Step 06 - Rail-bottom datum", []),
    (_THRESH, "Thresholds - cold start (iter 0)", list(_THRESH_ARGS)),
    ("code_lidar/02_pipeline/step07_full_analysis.py", "Step 07 - Ditch analysis (pass 1)", []),
    (_THRESH, "Thresholds - bootstrap (iter 1)", list(_THRESH_ARGS) + ["--bootstrap"]),
    ("code_lidar/02_pipeline/step07_full_analysis.py", "Step 07 - Ditch analysis (final)", []),
    ("code_lidar/02_pipeline/step08_overview_map.py", "Step 08 - Overview maps", []),
]

# downstream = re-bootstrap + final step07 + step08 (reuses existing pass-1 CSV)
DOWNSTREAM_STEPS = FULL_STEPS[-3:]
STEP07_ONLY = [FULL_STEPS[-2]]
STEP08_ONLY = [FULL_STEPS[-1]]


def configure_stdio() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def resolve_path(path_text: str | None, default: Path | None = None) -> Path:
    if path_text is None:
        if default is None:
            raise ValueError("No path was supplied.")
        return default
    path = Path(path_text)
    if not path.is_absolute():
        path = BASE / path
    return path


def find_laz(tile: str, laz_root: Path) -> Path:
    pattern = str(laz_root / "**" / f"*_{tile}.laz")
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No LAZ file matching tile {tile!r} under {laz_root}")
    if len(matches) > 1:
        print(f"[WARN] Multiple LAZ files matched {tile}; using the first:")
        for m in matches[:5]:
            print(f"       {m}")
    return Path(matches[0])


def format_args(args: list[str], out_dir: Path, tile: str, laz: Path) -> list[str]:
    return [a.format(out_dir=str(out_dir), tile=tile, laz=str(laz)) for a in args]


def run_step(script_rel: str, label: str, extra_args: list[str], env: dict[str, str]) -> None:
    script = BASE / script_rel
    if not script.exists():
        raise FileNotFoundError(f"Missing pipeline script: {script}")

    cmd = [sys.executable, str(script), *extra_args]
    print(f"\n{'=' * 72}")
    print(label)
    print(f"script : {script_rel}")
    if extra_args:
        print(f"args   : {extra_args}")
    print(f"{'=' * 72}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE), env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode} after {elapsed:.1f}s")
    print(f"[OK] {label} completed in {elapsed:.1f}s")


def main() -> None:
    configure_stdio()

    parser = argparse.ArgumentParser(description="Run the LiDAR ditch-detection pipeline for any tile.")
    parser.add_argument("--tile", required=True, help="Tile or section id.")
    parser.add_argument("--laz", default=None, help="Optional explicit LAZ path. If omitted, it is found from --tile.")
    parser.add_argument("--laz-root", default=str(DEFAULT_LAZ_ROOT), help="Root used when searching for the tile LAZ.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Relative paths are resolved from the workspace.")
    parser.add_argument("--mode", choices=["full", "downstream", "step07", "step08"], default="full",
                        help="full runs steps 01-08; downstream runs steps 07-08; step07/step08 run one step.")
    parser.add_argument("--use-adaptive-thresholds", choices=["1", "0"], default="1",
                        help="Forwarded to step07 as USE_ADAPTIVE_THR.")
    args = parser.parse_args()

    laz_root = resolve_path(args.laz_root)
    laz_path = resolve_path(args.laz) if args.laz else find_laz(args.tile, laz_root)
    out_dir = resolve_path(args.out_dir, BASE / f"output_{args.tile}")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env["LIDAR_BASE"] = str(BASE)
    env["LIDAR_TILE_ID"] = args.tile
    env["LIDAR_LAZ_FILE"] = str(laz_path)
    env["LIDAR_LAZ_ROOT"] = str(laz_root)
    env["LIDAR_OUT_DIR"] = str(out_dir)
    env.setdefault("LIDAR_CENTRELINE_DIR", str(BASE / "centreline_per_tile"))
    env["USE_ADAPTIVE_THR"] = args.use_adaptive_thresholds

    if args.mode == "full":
        steps = FULL_STEPS
    elif args.mode == "downstream":
        steps = DOWNSTREAM_STEPS
    elif args.mode == "step07":
        steps = STEP07_ONLY
    else:
        steps = STEP08_ONLY

    print("LiDAR pipeline runner")
    print(f"tile      : {args.tile}")
    print(f"laz       : {laz_path}")
    print(f"out_dir   : {out_dir}")
    print(f"mode      : {args.mode}")
    print(f"adaptive  : {args.use_adaptive_thresholds}")
    print(f"steps     : {len(steps)}")

    total_t0 = time.time()
    for script_rel, label, extra_args in steps:
        run_step(script_rel, label, format_args(extra_args, out_dir, args.tile, laz_path), env)

    print(f"\n{'=' * 72}")
    print(f"PIPELINE COMPLETE in {time.time() - total_t0:.1f}s")
    print(f"Output: {out_dir}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

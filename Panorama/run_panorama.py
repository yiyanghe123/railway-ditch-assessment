"""External runner for the panorama visible-ditch workflow.

The individual scripts read PANO_* environment variables. This runner builds
those variables from command-line arguments and then executes the selected
stages in order.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"


def resolve_path(text: str | None, default: Path | None = None) -> Path | None:
    if text is None:
        return default
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def run_step(script: str, env: dict[str, str]) -> None:
    script_path = CODE_DIR / script
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    print(f"\n[run] {script}")
    subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the panorama visible-ditch workflow.")
    parser.add_argument("--tile", required=True, help="Tile id, for example 7309_809 or 7302_816.")
    parser.add_argument("--tile-dir", default=None, help="Tile directory. Defaults to tiles/<tile>.")
    parser.add_argument("--input-dir", default=None, help="Directory containing panorama inventory and centreline inputs.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <tile-dir>/results.")
    parser.add_argument("--model-dir", required=True,
                        help="External DA3-SMALL model directory.")
    parser.add_argument("--da3-repo", default=str(ROOT / "Depth-Anything-3"))
    parser.add_argument("--use-lidar", choices=["0", "1"], default="0",
                        help="0 keeps the pipeline image-only; 1 joins LiDAR only for reporting.")
    parser.add_argument("--lidar-csv", default=None, help="Optional LiDAR reference CSV used only when --use-lidar 1.")
    parser.add_argument("--run-inventory", action="store_true",
                        help="Run step 01 before steps 02-05. Requires the raw panorama metadata arguments.")
    parser.add_argument("--laz", default=None, help="LAZ file for optional inventory generation.")
    parser.add_argument("--orbit-file", default=None, help="Orbit panorama pose CSV/TXT for optional inventory generation.")
    parser.add_argument("--inf-file", default=None, help="Orbit INF file for optional inventory generation.")
    parser.add_argument("--panorama-root", default=None, help="Root folder containing raw panorama JPG files.")
    args = parser.parse_args()

    tile_dir = resolve_path(args.tile_dir, ROOT / "tiles" / args.tile)
    input_dir = resolve_path(args.input_dir, tile_dir / "inputs")
    output_dir = resolve_path(args.output_dir, tile_dir / "results")
    model_dir = resolve_path(args.model_dir)
    da3_repo = resolve_path(args.da3_repo)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = str(da3_repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PANO_TILE_ID"] = args.tile
    env["PANO_USE_LIDAR"] = args.use_lidar
    env["PANO_PANO_CSV"] = str(input_dir / "panoramas_in_tile.csv")
    env["PANO_SPARMITT_CSV"] = str(input_dir / f"centreline_tile_{args.tile}.csv")
    env["PANO_ST_META_JSON"] = str(input_dir / "step03_ST_metadata.json")
    env["PANO_DA3_MODEL_DIR"] = str(model_dir)
    env["PANO_MULTIVIEW_OUT_DIR"] = str(output_dir / "output_multiview")
    env["PANO_MULTIVIEW_FIG_DIR"] = str(output_dir / "figures_multiview")
    env["PANO_DA3_CSV"] = str(output_dir / "output_multiview" / "multiview_visible_surface.csv")
    env["PANO_CONTINUOUS_OUT_DIR"] = str(output_dir / "output_continuous")
    env["PANO_CONTINUOUS_FIG_DIR"] = str(output_dir / "figures_continuous")
    env["PANO_PROFILE_STACK_CSV"] = str(output_dir / "output_continuous" / "panorama_side_profile_stack.csv")
    env["PANO_TARGET_STATION_CSV"] = str(output_dir / "output_continuous" / "panorama_target_station_visible_surface.csv")
    env["PANO_CONTINUOUS_LIDAR_CSV"] = str(output_dir / "output_continuous" / "panorama_lidar_aligned_visible_surface_1m.csv")
    env["PANO_DITCH_OUT_DIR"] = str(output_dir / "output_profile_ditch_likeness")
    env["PANO_DITCH_FIG_DIR"] = str(output_dir / "figures_profile_ditch_likeness")
    env["PANO_METRIC_PATH_CSV"] = str(output_dir / "output_profile_ditch_likeness" / "visible_ditch_metric_path.csv")
    env["PANO_CORRIDOR_PATH_CSV"] = str(output_dir / "output_profile_ditch_likeness" / "visible_ditch_presence_path.csv")
    env["PANO_FINAL_OUT_DIR"] = str(output_dir / "output_final_selection")
    env["PANO_FINAL_FIG_DIR"] = str(output_dir / "figures_final_selection")

    if args.lidar_csv:
        env["PANO_LIDAR_CSV"] = str(resolve_path(args.lidar_csv))
    if args.run_inventory:
        required = {
            "--laz": args.laz,
            "--orbit-file": args.orbit_file,
            "--inf-file": args.inf_file,
            "--panorama-root": args.panorama_root,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("Inventory generation requires: " + ", ".join(missing))
        env["PANO_LAZ_FILE"] = str(resolve_path(args.laz))
        env["PANO_ORBIT_FILE"] = str(resolve_path(args.orbit_file))
        env["PANO_INF_FILE"] = str(resolve_path(args.inf_file))
        env["PANO_PANORAMA_ROOT"] = str(resolve_path(args.panorama_root))
        env["PANO_INVENTORY_OUT_DIR"] = str(input_dir)
        env["PANO_INVENTORY_FIG_DIR"] = str(output_dir / "figures_inventory")

    print("Panorama external runner")
    print(f"tile       : {args.tile}")
    print(f"input_dir  : {input_dir}")
    print(f"output_dir : {output_dir}")
    print(f"model_dir  : {model_dir}")
    print(f"use_lidar  : {args.use_lidar}")

    if args.run_inventory:
        run_step("01_match_panorama_inventory.py", env)
    run_step("02_reconstruct_multiview_visible_surface.py", env)
    run_step("03_build_continuous_visible_surface_profile.py", env)
    run_step("04_score_visible_ditch_candidates.py", env)
    run_step("05_select_final_visible_surface.py", env)


if __name__ == "__main__":
    main()

"""Command-line runner for the video temporal visible-depth workflow."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def resolve_path(text: str | None, default: Path | None = None) -> Path | None:
    if text is None:
        return default
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def run_script(script: str, args: list[str], env: dict[str, str]) -> None:
    script_path = ROOT / script
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    print(f"\n[run] {script}")
    subprocess.run([sys.executable, str(script_path), *args], cwd=str(ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the video temporal visible-depth workflow.")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing the source inspection videos.")
    parser.add_argument("--output-dir", default=str(ROOT / "output"),
                        help="Directory for frames, depth maps, tables, and figures.")
    parser.add_argument("--config-dir", default=str(ROOT / "config"),
                        help="Directory containing rail_guides.csv.")
    parser.add_argument("--rail-guide-csv", default=None,
                        help="Explicit rail guide CSV. Defaults to <config-dir>/rail_guides.csv.")
    parser.add_argument("--da3-repo", default=str(ROOT / "Depth-Anything-3"),
                        help="Depth Anything 3 source directory.")
    parser.add_argument("--model-dir", required=True,
                        help="Local DA3-SMALL model directory.")
    parser.add_argument("--video-specs-json", default=None,
                        help="Optional JSON list describing videos, years, and chainage overlays.")
    parser.add_argument("--mode", choices=["all", "profiles", "metric"], default="all",
                        help="profiles runs step03; metric runs step04; all runs both.")
    parser.add_argument("--sample-spacing-m", type=float, default=2.0)
    parser.add_argument("--unit-length-m", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", type=int, default=420)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    config_dir = resolve_path(args.config_dir)
    rail_csv = resolve_path(args.rail_guide_csv, config_dir / "rail_guides.csv")
    da3_repo = resolve_path(args.da3_repo)
    model_dir = resolve_path(args.model_dir)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env["VIDEO_TEMPORAL_ROOT"] = str(ROOT)
    env["VIDEO_INPUT_DIR"] = str(input_dir)
    env["VIDEO_OUTPUT_DIR"] = str(output_dir)
    env["VIDEO_CONFIG_DIR"] = str(config_dir)
    env["VIDEO_RAIL_GUIDE_CSV"] = str(rail_csv)
    env["VIDEO_DA3_REPO"] = str(da3_repo)
    env["VIDEO_DA3_MODEL_DIR"] = str(model_dir)
    env["PYTHONPATH"] = str(da3_repo) + os.pathsep + env.get("PYTHONPATH", "")
    if args.video_specs_json:
        env["VIDEO_SPECS_JSON"] = str(resolve_path(args.video_specs_json))

    profile_args = [
        "--sample-spacing-m", str(args.sample_spacing_m),
        "--unit-length-m", str(args.unit_length_m),
        "--device", args.device,
        "--process-res", str(args.process_res),
        "--chunk-size", str(args.chunk_size),
    ]
    if args.no_skip_existing:
        profile_args.append("--no-skip-existing")

    print("Video temporal-depth command-line runner")
    print(f"input_dir  : {input_dir}")
    print(f"output_dir : {output_dir}")
    print(f"config_dir : {config_dir}")
    print(f"da3_repo   : {da3_repo}")
    print(f"model_dir  : {model_dir}")
    print(f"mode       : {args.mode}")

    if args.mode in ("all", "profiles"):
        run_script("step03_run_video_temporal_pipeline.py", profile_args, env)
    if args.mode in ("all", "metric"):
        run_script("step04_analyze_temporal_visible_depth.py", [], env)


if __name__ == "__main__":
    main()

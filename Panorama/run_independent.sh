#!/usr/bin/env bash
# Independent (image-only) panorama ditch-depth pipeline, DA3-SMALL.
# No LiDAR is read for depth/selection (PANO_USE_LIDAR=0); scale comes from the
# panorama GNSS camera poses + known rail geometry only.
# Usage:  bash run_independent.sh 7309_809 /path/to/DA3-SMALL      (or 7302_816)
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
TILE="${1:?usage: bash run_independent.sh <TILE, e.g. 7309_809> <DA3_MODEL_DIR>}"
DA3_MODEL_DIR="${2:?usage: bash run_independent.sh <TILE, e.g. 7309_809> <DA3_MODEL_DIR>}"
T="$HERE/tiles/$TILE"
[ -d "$T" ] || { echo "no tile dir: $T"; exit 1; }

echo "[setup] install Depth-Anything-3 (keep instance CUDA torch) + deps"
echo "[info] 01 panorama inventory is already prepared in tiles/<TILE>/inputs"
pip install --no-input --no-deps ./Depth-Anything-3
pip install --no-input -r requirements_cloud.txt

export PYTHONUNBUFFERED=1
export PANO_USE_LIDAR=0                              # INDEPENDENT: no LiDAR reference
export PANO_TILE_ID="$TILE"
export PANO_PANO_CSV="$T/inputs/panoramas_in_tile.csv"
export PANO_SPARMITT_CSV="$T/inputs/centreline_tile_${TILE}.csv"
export PANO_ST_META_JSON="$T/inputs/step03_ST_metadata.json"
export PANO_DA3_MODEL_DIR="$DA3_MODEL_DIR"           # external SMALL model
# Leave PANO_DA3_MODEL_ID unset so the bundled SMALL model is used.

OUT="$T/results"
export PANO_MULTIVIEW_OUT_DIR="$OUT/output_multiview"
export PANO_MULTIVIEW_FIG_DIR="$OUT/figures_multiview"
export PANO_DA3_CSV="$OUT/output_multiview/multiview_visible_surface.csv"
export PANO_CONTINUOUS_OUT_DIR="$OUT/output_continuous"
export PANO_CONTINUOUS_FIG_DIR="$OUT/figures_continuous"
export PANO_PROFILE_STACK_CSV="$OUT/output_continuous/panorama_side_profile_stack.csv"
export PANO_TARGET_STATION_CSV="$OUT/output_continuous/panorama_target_station_visible_surface.csv"
export PANO_CONTINUOUS_LIDAR_CSV="$OUT/output_continuous/panorama_lidar_aligned_visible_surface_1m.csv"
export PANO_DITCH_OUT_DIR="$OUT/output_profile_ditch_likeness"
export PANO_DITCH_FIG_DIR="$OUT/figures_profile_ditch_likeness"
export PANO_METRIC_PATH_CSV="$OUT/output_profile_ditch_likeness/visible_ditch_metric_path.csv"
export PANO_CORRIDOR_PATH_CSV="$OUT/output_profile_ditch_likeness/visible_ditch_presence_path.csv"
export PANO_FINAL_OUT_DIR="$OUT/output_final_selection"
export PANO_FINAL_FIG_DIR="$OUT/figures_final_selection"

echo "[GPU] 02 multiview visible-surface reconstruction (SMALL)"
python code/02_reconstruct_multiview_visible_surface.py
echo "[CPU] 03 continuous visible-surface profile"
python code/03_build_continuous_visible_surface_profile.py
echo "[CPU] 04 visible-ditch candidate scoring"
python code/04_score_visible_ditch_candidates.py
echo "[CPU] 05 final visible-surface selection"
python code/05_select_final_visible_surface.py
echo "DONE (independent, SMALL). Results: $OUT"




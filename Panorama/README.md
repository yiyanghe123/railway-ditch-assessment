# Panorama Visible-Surface Workflow

This workflow estimates a visible side-ditch surface from posed panorama images. It is an image-based supporting route and does not replace LiDAR ditch-bottom measurement.

Use the main runner from this folder:

```bash
python run_panorama.py \
  --tile <tile-id> \
  --input-dir <path-to-panorama-input-folder> \
  --output-dir <path-to-output-folder> \
  --da3-repo <path-to-Depth-Anything-3> \
  --model-dir <path-to-DA3-model-folder> \
  --use-lidar 0
```

## Inputs

| Input | Description |
|---|---|
| Panorama inventory | Matched panorama names and positions |
| Centreline CSV | Track centreline in the local tile frame |
| Track-frame metadata | Metadata from LiDAR preprocessing |
| Panorama images | Source equirectangular images |
| DA3 source and model | External Depth Anything 3 code and model folder |

Optional LiDAR reference data can be joined for offline comparison, but it should not be used for image-side candidate selection when the goal is an independent panorama experiment.

## Optional Inventory Generation

If the panorama inventory is not prepared, run step 01 through the runner:

```bash
python run_panorama.py \
  --tile <tile-id> \
  --run-inventory \
  --laz <path-to-tile.laz> \
  --orbit-file <path-to-orbit-poses-file> \
  --inf-file <path-to-camera-mount-file> \
  --panorama-root <path-to-panorama-images> \
  --input-dir <path-to-panorama-input-folder> \
  --output-dir <path-to-output-folder> \
  --da3-repo <path-to-Depth-Anything-3> \
  --model-dir <path-to-DA3-model-folder>
```

## Outputs

| Folder or file | Meaning |
|---|---|
| `output_multiview/multiview_visible_surface.csv` | DA3 multi-view visible-surface points |
| `output_continuous/panorama_side_profile_stack.csv` | Side-profile stack before ditch scoring |
| `output_profile_ditch_likeness/visible_ditch_metric_path.csv` | Longitudinal path of metric-depth candidates |
| `output_final_selection/` | Final panorama visible-depth tables and summaries |

## Notes

- Keep panorama images and DA3 weights outside the repository.
- The workflow estimates visible-surface depth. It cannot recover a hidden ditch bed beneath dense vegetation or water.

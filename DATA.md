# Data Requirements

This repository contains code only. To reproduce the workflows, provide the input data through command-line arguments.

## LiDAR Workflow

Required inputs:

| Input | Description |
|---|---|
| LAZ point-cloud tile | Mobile mapping LiDAR point cloud for one tile or section |
| Centreline CSV | Track centreline in the local or project coordinate system |
| Tile or section ID | Identifier used to name outputs |

Typical command:

```bash
python LiDAR/run_lidar.py \
  --tile <tile-id> \
  --laz <path-to-tile.laz> \
  --centreline-dir <path-to-centreline-csv-folder> \
  --out-dir <path-to-output-folder>
```

## Panorama Workflow

Required inputs:

| Input | Description |
|---|---|
| Panorama inventory CSV | Matched panorama names, positions, and headings |
| Centreline CSV | Track centreline for the analysed tile or section |
| Track-frame metadata JSON | Metadata exported from the LiDAR preprocessing stage |
| Panorama images | Source equirectangular panorama images |
| DA3 source and model | External Depth Anything 3 code and local model folder |

Typical command:

```bash
python Panorama/run_panorama.py \
  --tile <tile-id> \
  --input-dir <path-to-panorama-input-folder> \
  --output-dir <path-to-output-folder> \
  --da3-repo <path-to-Depth-Anything-3> \
  --model-dir <path-to-DA3-model-folder> \
  --use-lidar 0
```

## Video Temporal Workflow

Required inputs:

| Input | Description |
|---|---|
| Inspection videos | Source videos for the compared years or surveys |
| Rail guide CSV | In-frame rail guide points used for metric calibration |
| Optional video specs JSON | Video names, years, and chainage overlays |
| DA3 source and model | External Depth Anything 3 code and local model folder |

Typical command:

```bash
python "Video Temporal Depth/run_video_temporal_depth.py" \
  --input-dir <path-to-video-folder> \
  --output-dir <path-to-output-folder> \
  --config-dir <path-to-config-folder> \
  --da3-repo <path-to-Depth-Anything-3> \
  --model-dir <path-to-DA3-model-folder>
```

## Files Excluded From Git

Do not commit:

- `.laz`, `.las`, `.ply`, `.pcd` point-cloud files
- panorama images and inspection videos
- model weights such as `.pt`, `.pth`, `.ckpt`, `.safetensors`
- generated output folders
- local virtual environments and caches

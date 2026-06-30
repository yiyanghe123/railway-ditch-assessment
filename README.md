# Railway Ditch Depth and Condition Assessment

This repository contains the source code for three railway side-ditch assessment workflows:

| Folder | Workflow |
|---|---|
| `LiDAR/` | Track-referenced LiDAR ditch depth and condition assessment |
| `Panorama/` | Panorama-based visible-surface ditch profiling |
| `Video Temporal Depth/` | Inspection-video temporal visible-surface comparison |

The repository contains code only. Raw point clouds, panorama images, inspection videos, generated outputs, and model weights are not included.

## Repository Layout

```text
LiDAR/
  run_lidar.py
  code_lidar/
Panorama/
  run_panorama.py
  code/
Video Temporal Depth/
  run_video_temporal_depth.py
  config/
requirements.txt
requirements-optional.txt
DATA.md
THIRD_PARTY.md
LICENSE.md
```

## Installation

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch first with the CUDA or CPU build that matches your machine. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

`xformers` is optional and platform-dependent:

```bash
pip install -r requirements-optional.txt
```

## Data And Models

The workflows expect external data paths. Keep large or restricted files outside the Git repository and pass their locations through the runner arguments.

Required data are described in [DATA.md](DATA.md).

Depth Anything 3 source code and model weights are not vendored in this repository. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Quick Start

Check that the entry points are available:

```bash
python LiDAR/run_lidar.py --help
python Panorama/run_panorama.py --help
python "Video Temporal Depth/run_video_temporal_depth.py" --help
```

Run the LiDAR workflow for one tile:

```bash
python LiDAR/run_lidar.py \
  --tile <tile-id> \
  --laz <path-to-tile.laz> \
  --centreline-dir <path-to-centreline-csv-folder> \
  --out-dir <path-to-output-folder>
```

Run the panorama workflow:

```bash
python Panorama/run_panorama.py \
  --tile <tile-id> \
  --input-dir <path-to-panorama-input-folder> \
  --output-dir <path-to-output-folder> \
  --da3-repo <path-to-Depth-Anything-3> \
  --model-dir <path-to-DA3-model-folder> \
  --use-lidar 0
```

Run the video temporal workflow:

```bash
python "Video Temporal Depth/run_video_temporal_depth.py" \
  --input-dir <path-to-video-folder> \
  --output-dir <path-to-output-folder> \
  --config-dir <path-to-config-folder> \
  --da3-repo <path-to-Depth-Anything-3> \
  --model-dir <path-to-DA3-model-folder>
```

## Main Outputs

| Workflow | Main output |
|---|---|
| LiDAR | Per-metre ditch geometry, condition, and priority table |
| Panorama | Continuous visible-surface ditch profile and selection tables |
| Video | Frame-level and interval-level temporal visible-depth comparison tables |

## License

This code is released for non-commercial research and educational use. See [LICENSE.md](LICENSE.md).

## Academic Context

This repository accompanies the master's thesis work of Yiyang He at KTH Royal Institute of Technology, 2026. A formal thesis citation will be added after the thesis record is publicly available. Until then, please refer to this repository URL when discussing or reusing the code.

# Video Temporal Visible-Depth Workflow

This workflow samples inspection-video frames, runs Depth Anything 3, calibrates visible depth against rail geometry, and compares visible side-ditch evidence between two surveys.

Use the external runner from this folder:

```bash
python run_video_temporal_depth.py \
  --input-dir <path-to-video-folder> \
  --output-dir <path-to-output-folder> \
  --config-dir <path-to-config-folder> \
  --da3-repo <path-to-Depth-Anything-3> \
  --model-dir <path-to-DA3-model-folder>
```

## Inputs

| Input | Description |
|---|---|
| Inspection videos | Source videos for the compared years or surveys |
| Rail guide CSV | Manual rail guide points used for in-frame metric calibration |
| Optional video specs JSON | Video names, years, and chainage overlays |
| DA3 source and model | External Depth Anything 3 code and model folder |

If no video specs JSON is supplied, the pipeline uses the video definitions inside `step03_run_video_temporal_pipeline.py`.

## Run Modes

| Mode | What it runs |
|---|---|
| `all` | Frame sampling, DA3 depth, profile extraction, temporal aggregation, and metric change analysis |
| `profiles` | Frame sampling, DA3 depth, and visible-profile extraction |
| `metric` | Metric temporal-change analysis only |

Example:

```bash
python run_video_temporal_depth.py \
  --input-dir <path-to-video-folder> \
  --output-dir <path-to-output-folder> \
  --config-dir <path-to-config-folder> \
  --da3-repo <path-to-Depth-Anything-3> \
  --model-dir <path-to-DA3-model-folder> \
  --mode profiles
```

## Optional Video Specs JSON

Use `--video-specs-json` when the video names or chainage overlays change.

```json
[
  {
    "year": 2020,
    "filename": "inspection_2020.mp4",
    "overlay_start_m": 1152705.0,
    "overlay_end_m": 1153122.0
  },
  {
    "year": 2025,
    "filename": "inspection_2025.mp4",
    "overlay_start_m": 1152687.0,
    "overlay_end_m": 1153163.0
  }
]
```

## Outputs

| File or folder | Meaning |
|---|---|
| `frames/` | Sampled frames by year |
| `depth/` | Cached DA3 depth maps |
| `tables/frame_right_side_ditch_measurements.csv` | Frame-level visible ditch measurements |
| `tables/temporal_units_by_year.csv` | Aggregated temporal units by year |
| `tables/temporal_change_2020_2025.csv` | Cross-year visible-evidence change table |
| `tables/metric_temporal_change.csv` | Metric calibrated temporal change table |

## Notes

- Keep inspection videos and DA3 weights outside the repository.
- The video workflow is a visible-surface screening method. It should not be interpreted as direct physical ditch-bed measurement.

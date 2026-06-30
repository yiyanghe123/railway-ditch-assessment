# LiDAR Ditch Condition Workflow

This workflow estimates railway side-ditch geometry, condition labels, and maintenance-priority classes from one mobile mapping LiDAR tile or section.

Use the external runner from this folder:

```bash
python run_lidar.py \
  --tile <tile-id> \
  --laz <path-to-tile.laz> \
  --centreline-dir <path-to-centreline-csv-folder> \
  --out-dir <path-to-output-folder>
```

## Inputs

| Input | Description | Argument |
|---|---|---|
| LAZ tile | Mobile mapping point cloud | `--laz` or `--laz-root` |
| Centreline CSV | `centreline_tile_<tile-id>.csv` | `--centreline-dir` |
| Tile or section ID | Identifier used in filenames | `--tile` |

## Outputs

| File | Meaning |
|---|---|
| `ditch_metrics_final.csv` | Main per-station ditch geometry and condition table |
| `adaptive_thresholds_v2.json` | Data-derived thresholds used by the final pass |
| `rail_reference_framework_step06.csv` | Rail-bottom reference frame |
| `step08_overview/` | Overview maps and diagnostic figures |

## Run Modes

```bash
python run_lidar.py \
  --tile <tile-id> \
  --laz <path-to-tile.laz> \
  --centreline-dir <path-to-centreline-csv-folder> \
  --out-dir <path-to-output-folder> \
  --mode full
```

| Mode | What it runs |
|---|---|
| `full` | Steps 01 to 08, including adaptive threshold bootstrap |
| `downstream` | Recompute thresholds, final ditch analysis, and maps |
| `step07` | Final ditch analysis only |
| `step08` | Overview maps only |

## Notes

- Keep raw LAZ files outside the repository.
- Use command-line arguments to change data locations.
- Run the workflow once per tile or section.

# LiDAR Pipeline Internals

The recommended public entry point is `../run_lidar.py`.

The scripts in this folder are the internal processing stages used by the LiDAR workflow. They read paths from environment variables set by the external runner.

```bash
python ../run_lidar.py \
  --tile <tile-id> \
  --laz <path-to-tile.laz> \
  --centreline-dir <path-to-centreline-csv-folder> \
  --out-dir <path-to-output-folder>
```

Run the stage scripts directly only when debugging a specific processing step.

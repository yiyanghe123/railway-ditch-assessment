"""
Stage 1 — Match Cirrus 360° panoramas to LAZ tile 7592_630.

Goal
----
For every panorama whose UTM position falls inside (or near) the LAZ
tile's bounding box, produce a single row in `panoramas_in_tile.csv`
that contains:

  * the panorama filename (and full disk path to the .jpg),
  * its UTM position (East, North, Height) and orientation (Pan, Tilt, Roll),
  * its position projected into the **same track-aligned (S, T) frame**
    that step03/step09 use for the LiDAR pipeline — which is the only
    way to cross-validate panorama-derived measurements against
    LiDAR-derived metrics on a per-section basis,
  * the camera mounting offsets (DeltaPan/Tilt/Roll, height) loaded from
    `bor_*.inf` so downstream stages can compute pixel→ray geometry.

Pipeline
--------
1.  Load LAZ header → get UTM offsets (x_min, y_min) and tile bbox.
2.  Load step03_ST_metadata.json → PCA azimuth, mean_x_local, mean_y_local.
3.  Load orbit (`orb_*.txt`) — one row per panorama, UTM + Pan/Tilt/Roll.
4.  Load mounting offsets (`bor_*.inf`).
5.  Filter orbit by bbox + buffer to get the panoramas in this tile.
6.  Compute (X_local, Y_local) and (S, T) for each panorama using the
    same formulas as step03's `to_ST()`.
7.  Compute world-frame Pan (= orbit pan + camera DeltaPan, modulo 360°).
8.  Verify every JPG actually exists on disk; record the path.
9.  Write CSV + a diagnostic figure.

Output files (in da3_panorama/output/ and da3_panorama/figures/)
-----------------------------------------------------------------
    panoramas_in_tile.csv          — one row per matched panorama
    matching_meta.json             — config snapshot for reproducibility
    panorama_tile_match.png        — diagnostic plot

Author : Yiyang He (2026-05-08)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ════════════════════════════════════════════════════════════════════════
# Config — paths
# ════════════════════════════════════════════════════════════════════════
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TILE_ID = os.environ.get("PANO_TILE_ID", "tile")

LAZ_FILE = (
    os.environ.get(
        "PANO_LAZ_FILE",
        os.path.join(ROOT_DIR, "data", "laz", "tile.laz"),
    )
)
ST_META_JSON = os.environ.get(
    "PANO_ST_META_JSON",
    os.path.join(ROOT_DIR, "tiles", TILE_ID, "inputs", "step03_ST_metadata.json"),
)

ORBIT_FILE = (
    os.environ.get(
        "PANO_ORBIT_FILE",
        os.path.join(ROOT_DIR, "data", "orbit", "orbit_poses.txt"),
    )
)
INF_FILE = (
    os.environ.get(
        "PANO_INF_FILE",
        os.path.join(ROOT_DIR, "data", "orbit", "camera_mount.inf"),
    )
)
PANORAMA_ROOT = (
    os.environ.get(
        "PANO_PANORAMA_ROOT",
        os.path.join(ROOT_DIR, "data", "panoramas"),
    )
)
SPARMITT_TILE_CSV = (
    os.environ.get(
        "PANO_SPARMITT_CSV",
        os.path.join(ROOT_DIR, "tiles", TILE_ID, "inputs", f"centreline_tile_{TILE_ID}.csv"),
    )
)

OUT_DIR = os.environ.get("PANO_INVENTORY_OUT_DIR", os.path.join(ROOT_DIR, "tiles", TILE_ID, "inputs"))
FIG_DIR = os.environ.get("PANO_INVENTORY_FIG_DIR", os.path.join(ROOT_DIR, "tiles", TILE_ID, "results", "figures_inventory"))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Buffer around the tile bbox (m).  Panoramas are taken from the moving
# vehicle and may sit a few metres outside the LAZ tile boundary while
# still being relevant to the tile's content.
BBOX_BUFFER_M = float(os.environ.get("PANO_BBOX_BUFFER_M", "30.0"))
MAX_SPARMITT_DISTANCE_M = float(os.environ.get("PANO_MAX_SPARMITT_DISTANCE_M", "35.0"))


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════
def parse_inf_file(path: str) -> dict:
    """The .inf file is a Windows-style key=value section.  Read camera
    offset and orientation offsets relative to the trajectory point."""
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("[") or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    # Coerce known numeric keys to float
    for k in (
        "OrbitCameraDeltaX", "OrbitCameraDeltaY", "OrbitCameraDeltaZ",
        "OrbitCameraDeltaPan", "OrbitCameraDeltaTilt", "OrbitCameraDeltaRoll",
        "photo.camera.height",
    ):
        if k in out:
            try:
                out[k] = float(out[k])
            except ValueError:
                pass
    return out


def to_ST(xv, yv, mean_x, mean_y, angle_rad):
    """Same formula as step03's to_ST."""
    cos_a = np.cos(angle_rad); sin_a = np.sin(angle_rad)
    S =  (xv - mean_x) * cos_a + (yv - mean_y) * sin_a
    T = -(xv - mean_x) * sin_a + (yv - mean_y) * cos_a
    return S, T


def find_panorama_jpg(filename_no_ext: str, root: str) -> str | None:
    """Search for a panorama .jpg under any subdirectory of `root`.

    The orbit file lists 'pan_..._011929' (no extension).  The JPGs live
    in tile-keyed subfolders like '75_6_7525/'.
    """
    pattern = os.path.join(root, "**", filename_no_ext + ".jpg")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def add_nearest_sparmitt_columns(df: pd.DataFrame, sparmitt_csv: str) -> tuple[pd.DataFrame, dict]:
    """Attach distance to the delivered sparmitt centreline."""
    if not os.path.exists(sparmitt_csv) or df.empty:
        out = df.copy()
        out["sparmitt_distance_m"] = np.nan
        out["sparmitt_chainage_m"] = np.nan
        return out, {"available": False, "filter_applied": False}

    sp = pd.read_csv(sparmitt_csv)
    sp_xy = sp[["East", "North"]].to_numpy(dtype=float)
    sp_chainage = (
        sp["chainage_m"].to_numpy(dtype=float)
        if "chainage_m" in sp.columns
        else np.arange(len(sp_xy), dtype=float)
    )
    pose_xy = df[["East", "North"]].to_numpy(dtype=float)
    nearest_dist = np.empty(len(df), dtype=float)
    nearest_s = np.empty(len(df), dtype=float)
    for i, xy in enumerate(pose_xy):
        d = np.hypot(sp_xy[:, 0] - xy[0], sp_xy[:, 1] - xy[1])
        j = int(np.argmin(d))
        nearest_dist[i] = float(d[j])
        nearest_s[i] = float(sp_chainage[j])

    out = df.copy()
    out["sparmitt_distance_m"] = nearest_dist
    out["sparmitt_chainage_m"] = nearest_s
    return out, {
        "available": True,
        "filter_applied": False,
        "distance_before_filter_m": {
            "min": float(np.min(nearest_dist)),
            "median": float(np.median(nearest_dist)),
            "max": float(np.max(nearest_dist)),
        },
    }


# ════════════════════════════════════════════════════════════════════════
# 1) Load LAZ header → tile bbox + UTM offsets
# ════════════════════════════════════════════════════════════════════════
print("=" * 72)
print(f"Stage 1 — Match panoramas to LAZ tile {TILE_ID}")
print("=" * 72)

print("\n[1] LAZ header...")
with laspy.open(LAZ_FILE) as f:
    h = f.header
    UTM_OFFSET_X = float(h.x_min)
    UTM_OFFSET_Y = float(h.y_min)
    BBOX_E_LO = float(h.x_min); BBOX_E_HI = float(h.x_max)
    BBOX_N_LO = float(h.y_min); BBOX_N_HI = float(h.y_max)
    BBOX_Z_LO = float(h.z_min); BBOX_Z_HI = float(h.z_max)
    N_LAZ_POINTS = int(h.point_count)
print(f"  UTM bbox: E [{BBOX_E_LO:.1f}, {BBOX_E_HI:.1f}]  ({BBOX_E_HI-BBOX_E_LO:.1f} m)")
print(f"            N [{BBOX_N_LO:.1f}, {BBOX_N_HI:.1f}]  ({BBOX_N_HI-BBOX_N_LO:.1f} m)")
print(f"  UTM offsets used by step03 pipeline: x_min={UTM_OFFSET_X:.3f}, y_min={UTM_OFFSET_Y:.3f}")
print(f"  Total LAZ points: {N_LAZ_POINTS:,}")


# ════════════════════════════════════════════════════════════════════════
# 2) Load step03 ST metadata → PCA azimuth + local rotation centre
# ════════════════════════════════════════════════════════════════════════
print("\n[2] step03 metadata...")
with open(ST_META_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)

PCA_AZIMUTH_DEG = float(meta["pca_track_azimuth_deg"])
ANGLE_RAD       = np.radians(PCA_AZIMUTH_DEG)
MEAN_X_LOCAL    = float(meta["mean_x_local"])
MEAN_Y_LOCAL    = float(meta["mean_y_local"])
print(f"  PCA azimuth (corridor): {PCA_AZIMUTH_DEG:.3f}°")
print(f"  mean_x_local, mean_y_local: ({MEAN_X_LOCAL:.3f}, {MEAN_Y_LOCAL:.3f})")


# ════════════════════════════════════════════════════════════════════════
# 3) Load orbit file
# ════════════════════════════════════════════════════════════════════════
print("\n[3] Orbit file...")
orb = pd.read_csv(ORBIT_FILE, sep="\t")
print(f"  Total panoramas in orbit file: {len(orb):,}")
print(f"  Columns: {list(orb.columns)}")


# ════════════════════════════════════════════════════════════════════════
# 4) Load INF mounting offsets
# ════════════════════════════════════════════════════════════════════════
print("\n[4] Camera mounting offsets...")
mount = parse_inf_file(INF_FILE)
print("  bor_*.inf contents:")
for k, v in mount.items():
    print(f"    {k}: {v}")

CAM_DELTA_X    = float(mount.get("OrbitCameraDeltaX",    0.0))
CAM_DELTA_Y    = float(mount.get("OrbitCameraDeltaY",    0.0))
CAM_DELTA_Z    = float(mount.get("OrbitCameraDeltaZ",    0.0))
CAM_DELTA_PAN  = float(mount.get("OrbitCameraDeltaPan",  0.0))
CAM_DELTA_TILT = float(mount.get("OrbitCameraDeltaTilt", 0.0))
CAM_DELTA_ROLL = float(mount.get("OrbitCameraDeltaRoll", 0.0))
CAM_HEIGHT     = float(mount.get("photo.camera.height",  0.0))
print(f"  Camera position delta (X,Y,Z) = ({CAM_DELTA_X:.3f}, {CAM_DELTA_Y:.3f}, {CAM_DELTA_Z:.3f}) m")
print(f"  Camera orientation delta (P,T,R) = ({CAM_DELTA_PAN:.2f}°, {CAM_DELTA_TILT:.2f}°, {CAM_DELTA_ROLL:.2f}°)")
print(f"  Camera mounting height (above rail/sleeper): {CAM_HEIGHT:.3f} m")


# ════════════════════════════════════════════════════════════════════════
# 5) Filter orbit by tile bbox
# ════════════════════════════════════════════════════════════════════════
print("\n[5] Filtering orbit by LAZ tile bbox...")
in_tile = (
    (orb["East"]  >= BBOX_E_LO - BBOX_BUFFER_M) &
    (orb["East"]  <= BBOX_E_HI + BBOX_BUFFER_M) &
    (orb["North"] >= BBOX_N_LO - BBOX_BUFFER_M) &
    (orb["North"] <= BBOX_N_HI + BBOX_BUFFER_M)
)
sub = orb[in_tile].copy().sort_values("Timestamp").reset_index(drop=True)
print(f"  Buffer: ±{BBOX_BUFFER_M:.0f} m around tile")
print(f"  Matched panoramas: {len(sub)}")

sub, sparmitt_filter_meta = add_nearest_sparmitt_columns(sub, SPARMITT_TILE_CSV)
if sparmitt_filter_meta["available"]:
    before_n = len(sub)
    sub = sub[sub["sparmitt_distance_m"] <= MAX_SPARMITT_DISTANCE_M].copy().reset_index(drop=True)
    sparmitt_filter_meta["filter_applied"] = True
    sparmitt_filter_meta["max_sparmitt_distance_m"] = MAX_SPARMITT_DISTANCE_M
    sparmitt_filter_meta["n_before_filter"] = int(before_n)
    sparmitt_filter_meta["n_after_filter"] = int(len(sub))
    if len(sub):
        sparmitt_filter_meta["distance_after_filter_m"] = {
            "min": float(sub["sparmitt_distance_m"].min()),
            "median": float(sub["sparmitt_distance_m"].median()),
            "max": float(sub["sparmitt_distance_m"].max()),
        }
    print(
        f"  Sparmitt distance filter: kept {len(sub)} of {before_n} "
        f"poses within {MAX_SPARMITT_DISTANCE_M:.1f} m"
    )
    if len(sub):
        print(
            "  Distance to sparmitt after filter: "
            f"median={sub['sparmitt_distance_m'].median():.2f} m, "
            f"max={sub['sparmitt_distance_m'].max():.2f} m"
        )
else:
    print("  Sparmitt distance filter skipped: centreline CSV not found")


# ════════════════════════════════════════════════════════════════════════
# 6) UTM → local → (S, T)  using step03's exact formula
# ════════════════════════════════════════════════════════════════════════
print("\n[6] Computing local (X,Y) and track-aligned (S,T) for each panorama...")
sub["X_local"] = sub["East"]  - UTM_OFFSET_X
sub["Y_local"] = sub["North"] - UTM_OFFSET_Y
S_arr, T_arr = to_ST(
    sub["X_local"].values, sub["Y_local"].values,
    MEAN_X_LOCAL, MEAN_Y_LOCAL, ANGLE_RAD,
)
sub["S"] = S_arr
sub["T"] = T_arr

print(f"  S range (along-track):  [{sub['S'].min():.1f}, {sub['S'].max():.1f}] m")
print(f"  T range (cross-track):  [{sub['T'].min():.2f}, {sub['T'].max():.2f}] m")

# Median spacing along S (sanity check — Cirrus typically captures a panorama every ~5-10 m)
sub_sorted_s = sub.sort_values("S").reset_index(drop=True)
ds = np.diff(sub_sorted_s["S"].values)
ds = ds[ds > 0]   # drop zeros / negatives (vehicle could pass twice)
print(f"  Spacing along S (m): median={np.median(ds):.2f}, "
      f"min={ds.min():.2f}, max={ds.max():.2f}")


# ════════════════════════════════════════════════════════════════════════
# 7) World-frame Pan (= orbit Pan + camera mounting Pan offset)
# ════════════════════════════════════════════════════════════════════════
sub["Pan_world_deg"]  = (sub["Pan"]  + CAM_DELTA_PAN)  % 360.0
sub["Tilt_world_deg"] = sub["Tilt"]  + CAM_DELTA_TILT
sub["Roll_world_deg"] = sub["Roll"]  + CAM_DELTA_ROLL


# ════════════════════════════════════════════════════════════════════════
# 8) Resolve actual JPG paths (search all subfolders of PANORAMA_ROOT)
# ════════════════════════════════════════════════════════════════════════
print("\n[7] Resolving JPG paths on disk...")
jpg_paths: list[str] = []
n_found = 0
n_missing = 0
missing: list[str] = []
for fn in sub["Filename"]:
    p = find_panorama_jpg(fn, PANORAMA_ROOT)
    if p is None:
        jpg_paths.append("")
        n_missing += 1
        missing.append(fn)
    else:
        jpg_paths.append(os.path.normpath(p))
        n_found += 1
sub["jpg_path"] = jpg_paths
print(f"  Found:   {n_found}")
print(f"  Missing: {n_missing}")
if n_missing > 0 and n_missing <= 5:
    for fn in missing:
        print(f"    missing: {fn}")
elif n_missing > 5:
    for fn in missing[:5]:
        print(f"    missing: {fn}")
    print(f"    ... ({n_missing - 5} more)")


# ════════════════════════════════════════════════════════════════════════
# 9) Write CSV + meta JSON
# ════════════════════════════════════════════════════════════════════════
csv_cols = [
    "Filename", "Timestamp",
    "East", "North", "Height",
    "X_local", "Y_local",
    "S", "T",
    "sparmitt_distance_m", "sparmitt_chainage_m",
    "Pan", "Tilt", "Roll",
    "Pan_world_deg", "Tilt_world_deg", "Roll_world_deg",
    "jpg_path",
]
sub_out = sub[csv_cols].sort_values("S").reset_index(drop=True)
csv_path = os.path.join(OUT_DIR, "panoramas_in_tile.csv")
sub_out.to_csv(csv_path, index=False)
print(f"\n[Output] Saved CSV: {csv_path}")

meta_out = {
    "tile_id": TILE_ID,
    "n_matched": int(len(sub_out)),
    "n_jpg_found": int(n_found),
    "n_jpg_missing": int(n_missing),
    "bbox_buffer_m": BBOX_BUFFER_M,
    "sparmitt_filter": sparmitt_filter_meta,
    "laz_bbox": {
        "east":   [BBOX_E_LO, BBOX_E_HI],
        "north":  [BBOX_N_LO, BBOX_N_HI],
        "height": [BBOX_Z_LO, BBOX_Z_HI],
    },
    "utm_offsets": {"x_min": UTM_OFFSET_X, "y_min": UTM_OFFSET_Y},
    "step03": {
        "pca_track_azimuth_deg": PCA_AZIMUTH_DEG,
        "mean_x_local": MEAN_X_LOCAL,
        "mean_y_local": MEAN_Y_LOCAL,
    },
    "camera_mount": {
        "delta_x": CAM_DELTA_X, "delta_y": CAM_DELTA_Y, "delta_z": CAM_DELTA_Z,
        "delta_pan_deg":  CAM_DELTA_PAN,
        "delta_tilt_deg": CAM_DELTA_TILT,
        "delta_roll_deg": CAM_DELTA_ROLL,
        "height_m": CAM_HEIGHT,
    },
    "S_range_m": [float(sub_out["S"].min()), float(sub_out["S"].max())],
    "T_range_m": [float(sub_out["T"].min()), float(sub_out["T"].max())],
    "spacing_m_along_S": {
        "median": float(np.median(ds)),
        "min": float(ds.min()),
        "max": float(ds.max()),
    },
    "input_files": {
        "laz":        LAZ_FILE,
        "orbit":      ORBIT_FILE,
        "inf":        INF_FILE,
        "panorama_root": PANORAMA_ROOT,
        "step03_meta": ST_META_JSON,
    },
}
meta_path = os.path.join(OUT_DIR, "matching_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta_out, f, indent=2, ensure_ascii=False)
print(f"[Output] Saved meta: {meta_path}")


# ════════════════════════════════════════════════════════════════════════
# 10) Diagnostic plot — panoramas overlaid on tile bbox + sparmitt centreline
# ════════════════════════════════════════════════════════════════════════
print("\n[8] Building diagnostic figure...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) UTM view — bbox, panoramas, optional sparmitt centreline
ax = axes[0]
ax.add_patch(plt.Rectangle(
    (BBOX_E_LO, BBOX_N_LO), BBOX_E_HI-BBOX_E_LO, BBOX_N_HI-BBOX_N_LO,
    fill=False, edgecolor="black", lw=1.5, linestyle="--",
    label=f"LAZ tile {TILE_ID} bbox"))
ax.scatter(sub_out["East"], sub_out["North"],
           c=sub_out["S"], cmap="viridis", s=24, edgecolor="k",
           linewidth=0.4, label=f"panoramas (n={len(sub_out)})", zorder=3)

# Sparmitt centreline if available
if os.path.exists(SPARMITT_TILE_CSV):
    sp = pd.read_csv(SPARMITT_TILE_CSV)
    ax.plot(sp["East"], sp["North"], color="red", lw=1.0, alpha=0.7,
            label="Sparmitt centreline", zorder=2)

ax.set_xlabel("UTM East (m)")
ax.set_ylabel("UTM North (m)")
ax.set_title(f"Panorama positions vs LAZ tile {TILE_ID}\n"
             f"(colored by S = along-track)")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.3)
ax.legend(loc="best", fontsize=9)

# (b) (S, T) view — panoramas in the track-aligned frame
ax = axes[1]
ax.axhline(0, color="red", lw=1.0, alpha=0.5, label="PCA centreline reference (T=0)")
ax.scatter(sub_out["S"], sub_out["T"],
           c=sub_out["S"], cmap="viridis", s=24, edgecolor="k",
           linewidth=0.4)
for _, r in sub_out.iterrows():
    ax.text(r["S"], r["T"]+0.4, str(r["Filename"]).split("_")[-1],
            fontsize=6, ha="center", color="gray")
ax.set_xlabel("S — along-track (m)")
ax.set_ylabel("T — cross-track (m)")
ax.set_title("Panoramas in the global PCA (S, T) frame\n"
             "(large T variation is expected on curved tiles)")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

# No in-figure suptitle (the LaTeX caption is the title).
fig.tight_layout()

fig_path = os.path.join(FIG_DIR, "panorama_tile_match.png")
fig.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[Output] Saved figure: {fig_path}")


# ════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  Tile                       : {TILE_ID}")
print(f"  Panoramas matched          : {len(sub_out)}")
print(f"  JPGs found on disk         : {n_found} of {len(sub_out)}")
print(f"  S range covered            : "
      f"{sub_out['S'].min():.1f} -- {sub_out['S'].max():.1f} m  "
      f"(span {sub_out['S'].max() - sub_out['S'].min():.1f} m)")
print(f"  T range in global PCA frame: "
      f"min {sub_out['T'].min():.2f} m, max {sub_out['T'].max():.2f} m, "
      f"std {sub_out['T'].std():.2f} m")
print(f"  Median S spacing           : {np.median(ds):.2f} m  "
      f"(panorama every ~{np.median(ds):.0f} m)")
print()
print(f"  CSV   : {csv_path}")
print(f"  Meta  : {meta_path}")
print(f"  Figure: {fig_path}")
print()
print("  Next stage: run the multi-view virtual-camera DA3 pipeline")
print("  on all available JPG panoramas in the matched tile sequence.")

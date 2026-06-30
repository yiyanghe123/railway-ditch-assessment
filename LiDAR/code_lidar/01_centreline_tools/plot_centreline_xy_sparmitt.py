"""
Re-generate the centreline XY top-down figure using the current
(Trafikverket sparmitt) centreline CSVs in place.

Standalone — does NOT call step05 (which would re-fit B-spline and
overwrite the sparmitt-replaced CSV).  Reads existing files only.

Outputs:
  output_630_415/23_step5_centreline_XY.png        (overwritten)
  output_630_415/22b_step5_centreline_ST.png       (overwritten)
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import laspy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.environ.get("LIDAR_BASE", os.getcwd())
OUT_DIR = os.environ.get("LIDAR_OUT_DIR", os.path.join(BASE_DIR, "output"))
LAZ_FILE = os.environ.get("LIDAR_LAZ_FILE", os.path.join(BASE_DIR, "data", "laz", "tile.laz"))

GROUND_Z_RANGE = 8.0


# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────
print("Loading...")

dense = pd.read_csv(os.path.join(OUT_DIR, "rail_centreline_dense_step05.csv"))
sparse = pd.read_csv(os.path.join(OUT_DIR, "rail_centreline_step05.csv"))

# Read LAZ
las = laspy.read(LAZ_FILE)
x0 = float(las.header.x_min); y0 = float(las.header.y_min)
X = (np.asarray(las.x, dtype=np.float64) - x0).astype(np.float32)
Y = (np.asarray(las.y, dtype=np.float64) - y0).astype(np.float32)
Z = np.asarray(las.z, dtype=np.float32)
del las

# Keep ground points only (same filter as pipeline)
z1 = float(np.percentile(Z, 1))
m = (Z >= z1) & (Z <= z1 + GROUND_Z_RANGE)
X, Y = X[m], Y[m]
print(f"  Ground points: {len(X):,}")
print(f"  Centreline (dense): {len(dense)} points  S=[{dense['S'].min():.0f}, {dense['S'].max():.0f}]")

# Subsample for plotting speed
if len(X) > 200_000:
    idx = np.random.RandomState(42).choice(len(X), 200_000, replace=False)
    X_p, Y_p = X[idx], Y[idx]
else:
    X_p, Y_p = X, Y


# ──────────────────────────────────────────────────────────────
# Figure 23 — XY top-down
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X_p, Y_p, s=0.5, c="lightgray", alpha=0.5, label=f"Ground points (n={len(X):,})")

# Plot centreline
ax.plot(dense["X_local"], dense["Y_local"], "-",
        color="red", lw=1.8, label=f"Centreline — Trafikverket spårmitt ({len(dense)} pts)")

# Mark sparse anchor points
ax.plot(sparse["X_local"], sparse["Y_local"], "o",
        color="darkblue", ms=3, alpha=0.6, label=f"Sparmitt anchors ({len(sparse)} pts)")

ax.set_xlabel("X_local [m]")
ax.set_ylabel("Y_local [m]")
ax.legend(loc="best", fontsize=9)
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.3)
fig.tight_layout()

out_xy = os.path.join(OUT_DIR, "23_step5_centreline_XY.png")
fig.savefig(out_xy, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_xy}")


# ──────────────────────────────────────────────────────────────
# Figure 22b — (S, T) view
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dense["S"], dense["T_center"], "-",
        color="red", lw=1.5, label="Centreline T_center(S)")
ax.axhline(0, color="black", lw=0.5, ls=":", alpha=0.6, label="T = 0 (PCA axis)")
ax.set_xlabel("S [m] (along-track)")
ax.set_ylabel("T_center [m] (lateral offset from PCA axis)")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()

out_st = os.path.join(OUT_DIR, "22b_step5_centreline_ST.png")
fig.savefig(out_st, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_st}")

print("\nDONE — figures regenerated from sparmitt CSV (CSVs themselves untouched).")

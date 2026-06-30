import os
import sys
import json
import warnings
import matplotlib.patches as mpatches
import matplotlib

matplotlib.use("Agg")

import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import minimum_filter1d, maximum_filter1d

FIG_DPI = 300
COLOR_LEFT = "#1f77b4"
COLOR_RIGHT = "#d55e00"
COLOR_REFERENCE = "#222222"
COLOR_GRID = "#d9d9d9"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.55,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# numpy 2.0+ renamed np.trapz to np.trapezoid; alias for backward compatibility.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

# Force UTF-8 on stdout so the Swedish characters Å / Ö in the summary
# block don't crash on Windows GBK consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# =========================================================
# USER SETTINGS
# =========================================================
BASE_DIR = os.environ.get("LIDAR_BASE", os.getcwd())
LAZ_FILE = os.environ.get("LIDAR_LAZ_FILE", os.path.join(BASE_DIR, "data", "laz", "tile.laz"))
OUT_DIR = os.environ.get("LIDAR_OUT_DIR", os.path.join(BASE_DIR, "output"))
STEP03_META = os.path.join(OUT_DIR, "step03_ST_metadata.json")
REFERENCE_CSV = os.path.join(OUT_DIR, "rail_reference_framework_step06.csv")
if not os.path.exists(REFERENCE_CSV):
    legacy_reference_csv = os.path.join(OUT_DIR, "rail_reference_framework_step08.csv")
    if os.path.exists(legacy_reference_csv):
        REFERENCE_CSV = legacy_reference_csv
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# ADAPTIVE THRESHOLDS - load adaptive_thresholds_v2.json if present
# (Pukelsheim 1994; Lehmann 2013; Tukey 1977; Roelens 2018; TDOK 2015:0155)
#
# Set USE_ADAPTIVE_THR=0 in the environment to force the hard-coded
# literature defaults below (used as fall-back if the JSON is missing).
# =========================================================
USE_ADAPTIVE_THR = os.environ.get("USE_ADAPTIVE_THR", "1") == "1"
_ADAPTIVE_JSON   = os.path.join(OUT_DIR, "adaptive_thresholds_v2.json")
_t = {}
SIGMA_NOISE_M   = None
_ADAPTIVE_ITER  = 0
_ADAPTIVE_MODE  = "fixed"
if USE_ADAPTIVE_THR and os.path.exists(_ADAPTIVE_JSON):
    try:
        with open(_ADAPTIVE_JSON, encoding="utf-8") as _f:
            _ad_payload = json.load(_f)
        _t = {n: s["value"] for n, s in _ad_payload["thresholds"].items()
              if s.get("value") is not None}
        SIGMA_NOISE_M   = _ad_payload.get("_meta", {}).get("sigma_global_m")
        _ADAPTIVE_ITER  = _ad_payload.get("_meta", {}).get("bootstrap_iter", 0)
        _ADAPTIVE_MODE  = "adaptive"
        print(f"[ADAPTIVE] Loaded {_ADAPTIVE_JSON}  "
              f"(iter={_ADAPTIVE_ITER}, sigma_global={SIGMA_NOISE_M*1000:.2f} mm)")
    except Exception as _e:
        print(f"[ADAPTIVE] WARNING: failed to load JSON ({_e}); "
              f"falling back to hard-coded literature defaults.")
        _t = {}
        _ADAPTIVE_MODE = "fallback_after_error"
else:
    if USE_ADAPTIVE_THR:
        print(f"[ADAPTIVE] adaptive_thresholds_v2.json not found; "
              f"using hard-coded literature defaults.")
    else:
        print(f"[ADAPTIVE] USE_ADAPTIVE_THR=0; using hard-coded "
              f"literature defaults (forced by env var).")


def _T(name, default):
    """Return adaptive value for `name` if loaded, else `default`."""
    return _t.get(name, default)


# Ground / preprocessing
GROUND_Z_RANGE = 8.0
EXG_VEG_THRESHOLD = _T("EXG_VEG_THRESHOLD", 20.0)   # ExG > 20 indicates vegetation on the 8-bit-equivalent scale.

# Section extraction
LOCAL_SECTION_HALF_U = 0.75
ROUGH_S_PRESELECT_HALF = 2.0
ROUGH_T_PRESELECT_HALF = _T("ROUGH_T_PRESELECT_HALF", 18.0)

# Envelope
ENV_V_RANGE  = _T("ENV_V_RANGE",  16.0)
ENV_BIN_W    = _T("ENV_BIN_W",    0.05)
ENV_Q        = int(_T("ENV_Q",    5))
ENV_MIN_POINTS_PER_BIN = int(_T("ENV_MIN_POINTS_PER_BIN", 3))

# Morphological opening (replaces SavGol)
# Erosion (min filter) + dilation (max filter) preserves ditch depth
# while removing noise spikes. Kernel size in bins:
MORPH_KERNEL_SIZE     = int(_T("MORPH_KERNEL_SIZE",     3))   # 3 bins = 0.15m
MORPH_POST_SMOOTH_WIN = int(_T("MORPH_POST_SMOOTH_WIN", 5))   # very light SavGol on flat areas only

# Ditch search zone (outward distance from centreline)
SIDE_SEARCH_MIN = _T("SIDE_SEARCH_MIN", 0.50)   # start at ballast edge
SIDE_SEARCH_MAX = max(_T("SIDE_SEARCH_MAX", 8.00), _T("OUTER_DITCH_SEARCH_MAX", 15.50))
ENV_V_RANGE = max(ENV_V_RANGE, SIDE_SEARCH_MAX + 0.50)
ROUGH_T_PRESELECT_HALF = max(ROUGH_T_PRESELECT_HALF, ENV_V_RANGE + 2.0)

# Cutting drainage ditch search
# In cuttings the drainage ditch sits at the base of the slope,
# often closer to the track than SIDE_SEARCH_MIN.
CUTTING_SEARCH_MIN          = _T("CUTTING_SEARCH_MIN",          0.50)
CUTTING_SEARCH_MAX          = _T("CUTTING_SEARCH_MAX",          5.00)
CUTTING_DEPTH_FROM_SHOULDER = _T("CUTTING_DEPTH_FROM_SHOULDER", 0.10)

# Ditch acceptance criteria
DITCH_MIN_DEPTH = _T("DITCH_MIN_DEPTH", 0.08)
DITCH_MIN_PROM  = _T("DITCH_MIN_PROM",  0.05)
DITCH_MIN_WIDTH = _T("DITCH_MIN_WIDTH", 0.20)
DITCH_MIN_AREA  = _T("DITCH_MIN_AREA",  0.01)
SHAPE_OUTER_RISE_MIN  = _T("SHAPE_OUTER_RISE_MIN", 0.03)   # absolute floor (shallow ditches)
# Proportional outer-bank rise: a real ditch's outer bank must climb back
# up to a height commensurate with the ditch depth. Without this, an
# L-shaped embankment toe (sharp descent followed by flat floodplain)
# passes the constant 3-cm check and is falsely called a deep ditch.
# FRAC = 0.25 means a 2 m "ditch" must have outer-bank rise >=0.50 m;
# embankments typically have less than 10 cm rise past the toe and are rejected.
# Basis: Rail Baltica Part 2 + TDOK 2015:0155 Section 10.13 - by design the
# outer shoulder is ~ the same elevation as the inner rail bed, so the
# outer bank recovery should be at least the ditch depth. 0.25 is a lenient
# lower bound that still separates ditches from embankment toes.
SHAPE_OUTER_RISE_FRAC = _T("SHAPE_OUTER_RISE_FRAC", 0.25)

# Effective inner-bank slope (whole-inner-side check)
# The inner-slope check currently looks only at the 0.5 m edge-walked
# window around the bottom (a LOCAL slope). An embankment that gently
# descends toward the outer edge and then has a small terminal dip can
# pass this local check - because the local 0.5 m span is steep - while
# the actual inner side is a 5 m gentle slope, not a real ditch bank.
# We add a second, GLOBAL inner-slope criterion: the slope from the
# inner rim (highest envelope point between the search-zone start and
# the ditch bottom) to the bottom must be at least EFFECTIVE_INNER_SLOPE_MIN.
# Rationale: a real railway side ditch concentrates its depth into a
# short bank (>=18 deg design slope, Rail Baltica Part 2). A ditch whose
# "inner bank" is spread over more than about three times the depth in horizontal distance has no
# distinct bank and is almost certainly an embankment toe.
EFFECTIVE_INNER_SLOPE_MIN    = _T("EFFECTIVE_INNER_SLOPE_MIN",    10.0)  # degrees, for GLOBAL xk[0] to bottom
# slope check, applied conditionally (see EFFECTIVE_SLOPE_MIN_BOTTOM_X).
EFFECTIVE_SLOPE_MIN_BOTTOM_X = _T("EFFECTIVE_SLOPE_MIN_BOTTOM_X",  4.0)   # metres from search start
# The global inner-slope check is only applied when the ditch bottom
# lies further than this from the search-zone start (i.e., x >=3.5 m
# from centreline). Rationale: close-to-track ditches with a short
# steep bank + flat platform will have a LOW global slope (because the
# flat platform contributes 0 deg to the average) even though they are
# legitimate - the local slope check inside the bank already validates
# them. Only FAR ditches are suspect of being "embankment toe with
# terminal dip", and there the global slope correctly separates them.

# Non-ditch terrain classification
EMBANKMENT_MIN_DROP = _T("EMBANKMENT_MIN_DROP", 0.12)
CUTTING_MIN_RISE    = _T("CUTTING_MIN_RISE",    0.12)
FLAT_RELIEF_MAX     = _T("FLAT_RELIEF_MAX",     0.08)

# Depth validity filter
DEPTH_VALID_MIN = _T("DEPTH_VALID_MIN", DITCH_MIN_DEPTH)   # = DITCH_MIN_DEPTH (kept in sync by design)
DEPTH_VALID_MAX = _T("DEPTH_VALID_MAX", 2.50)
DEPTH_DEEP_FLAG = _T("DEPTH_DEEP_FLAG", 1.50)

# Longitudinal continuity filter
# A real drainage ditch persists over consecutive sections.
# Literature alignment (Roelens 2018, Cazorzi 2013): isolated single-section
# detections are noise, but multi-section runs should be preserved.
# We use a two-tier system:
# - CONFIRMED: >=DITCH_MIN_RUN_LENGTH of DITCH_CONFIRM_WINDOW sections
# used for TDOK 2015:0155 Section 10.13 diagnosis (shape /
# content / issues / priority fields are populated)
# - CANDIDATE: isolated detections that passed all per-section tests
# retained for inspection only; diagnosis set to "NA"
DITCH_MIN_RUN_LENGTH = int(_T("DITCH_MIN_RUN_LENGTH", 10))   # min consecutive sections for CONFIRMED
DITCH_CONFIRM_WINDOW = 5      # kept for backwards-compatible log/banner
                              # printing only. The actual filter is now
                              # run-length based (scipy.ndimage.label).

# Spatial-coherence filter (position outlier rejection)
# A real drainage ditch runs parallel to the track; its bottom_x
# (lateral distance from centreline) must stay consistent between
# neighbouring cross-sections. If a confirmed detection has a
# bottom_x that deviates > POSITION_OUTLIER_TOL_M from the rolling
# median of its neighbours, it is treated as a position outlier and
# demoted to "candidate" even if the count-based continuity test passed.
#
# Threshold 0.7 m chosen as about two sigma of the expected natural scatter:
# - Roelens / Bailly 2018: LiDAR ditch centreline RMSE about 0.4-0.6 m
# - Habib 2021: +/-7 cm cross-track accuracy on vegetated ground
# - Empirical MAD on tile 630 confirmed ditches about 0.10-0.15 m
# - Edge-walk 1 cm tolerance adds ~30 cm cumulative position noise
# Data-driven check on tile 630 shows P75 of |Delta position| about 0.5-0.65 m and a
# clear gap before P90, about 1.2-1.7 m, confirming 0.7 m as the "knee"
# between legitimate variation and outlier detections.
POSITION_OUTLIER_TOL_M    = _T("POSITION_OUTLIER_TOL_M",    0.7)
POSITION_OUTLIER_WIN_HALF = int(_T("POSITION_OUTLIER_WIN_HALF", 3))
POSITION_OUTLIER_MIN_NBR  = int(_T("POSITION_OUTLIER_MIN_NBR",  2))

# Spatial-prior continuity in detection
# Each cross-section is no longer detected independently. Instead, the
# rolling median of the last DITCH_PRIOR_WINDOW successful detections
# (per side) is passed in as a "prior" lateral position, and candidate
# peaks are sorted by distance to this prior (near-to-far otherwise).
# This eliminates the "oscillation between two real features" that
# a pure prominence-based sort causes when two valid peaks coexist.
DITCH_PRIOR_WINDOW = int(_T("DITCH_PRIOR_WINDOW", 5))    # use last N confirmed bottom_x values as prior

# Main-ditch candidate selection. When the search range includes both the
# ballast shoulder and the outer drainage corridor, a near-track local dip can
# pass the geometric gates before the real side ditch is tested. The detector
# therefore evaluates all valid candidates and selects the strongest
# drainage-shaped depression. These are scoring weights, not acceptance gates.
MAIN_DITCH_DEPTH_WEIGHT = _T("MAIN_DITCH_DEPTH_WEIGHT", 1.00)
MAIN_DITCH_PROM_WEIGHT = _T("MAIN_DITCH_PROM_WEIGHT", 0.45)
MAIN_DITCH_AREA_WEIGHT = _T("MAIN_DITCH_AREA_WEIGHT", 0.35)
MAIN_DITCH_OUTER_POS_WEIGHT = _T("MAIN_DITCH_OUTER_POS_WEIGHT", 0.04)
MAIN_DITCH_PRIOR_WEIGHT = _T("MAIN_DITCH_PRIOR_WEIGHT", 0.18)

# Outer-reach fix (candidate-selection bias)
# The legacy MAIN_DITCH_OUTER_POS_WEIGHT rewards FARTHER candidates (pos_term =
# bottom_x), which on embankment tiles lets candidate selection drift toward the
# outer search boundary even when the real ditch sits inboard. RTK-GNSS on tile
# 7302_816 puts the true ditch at about 12 m, yet detections drift to 14-15 m (the
# 15.5 m search edge). With USE_OUTER_REACH_FIX on we (1) drop the outward
# reward and (2) penalise candidates whose bottom_x sits within
# EDGE_PENALTY_BAND_M of SIDE_SEARCH_MAX, so an inboard thalweg wins ties and the
# rolling spatial prior stops latching onto the boundary. Set OUTER_REACH_FIX=0
# to restore the legacy scoring exactly.
USE_OUTER_REACH_FIX = os.environ.get("OUTER_REACH_FIX", "1") == "1"
EDGE_PENALTY_WEIGHT = _T("EDGE_PENALTY_WEIGHT", 1.5)   # max penalty at the search boundary
EDGE_PENALTY_BAND_M = _T("EDGE_PENALTY_BAND_M", 2.5)   # penalty ramps over this band inboard of the edge
OUTER_PATH_MIN_X = _T("OUTER_PATH_MIN_X", 8.0)
OUTER_PATH_PRIOR_TOL_M = _T("OUTER_PATH_PRIOR_TOL_M", 3.0)
OUTER_SURFACE_MIN_DEPTH = _T("OUTER_SURFACE_MIN_DEPTH", 0.04)
OUTER_SURFACE_MIN_RISE = _T("OUTER_SURFACE_MIN_RISE", 0.10)
LONGITUDINAL_CANDIDATE_PATH = int(_T("LONGITUDINAL_CANDIDATE_PATH", 1))
PATH_MIN_SUPPORT_SECTIONS = int(_T("PATH_MIN_SUPPORT_SECTIONS", 5))
PATH_MAX_ROW_GAP = int(_T("PATH_MAX_ROW_GAP", 5))
PATH_JUMP_PENALTY = _T("PATH_JUMP_PENALTY", 0.35)
PATH_EXTRA_JUMP_PENALTY = _T("PATH_EXTRA_JUMP_PENALTY", 0.15)
PATH_MAX_STABLE_IQR_M = _T("PATH_MAX_STABLE_IQR_M", 2.50)
PATH_CLUSTER_BIN_M = _T("PATH_CLUSTER_BIN_M", 0.50)
PATH_CLUSTER_KERNEL_M = _T("PATH_CLUSTER_KERNEL_M", 1.50)
PATH_CLUSTER_HALF_WIDTH_M = _T("PATH_CLUSTER_HALF_WIDTH_M", 2.25)
CANDIDATE_JSON_MAX_N = int(_T("CANDIDATE_JSON_MAX_N", 10))

# Top-of-bank edge selection. Earlier versions accepted ditch candidates
# from a short local edge-walk around the bottom. The current detector uses
# bank-top recovery instead: starting from the candidate bottom, each side is
# followed until the profile has recovered a data-driven fraction of the bank
# height and the local slope becomes flat enough to be interpreted as terrain
# outside the ditch. The resulting top-of-bank width is both the reported
# physical width and the width used by the geometric acceptance gate.
TOP_BANK_SEARCH_MAX_M = _T("TOP_BANK_SEARCH_MAX_M", 3.0)
TOP_BANK_RECOVERY_FRAC = _T("TOP_BANK_RECOVERY_FRAC", 0.80)
TOP_BANK_FLAT_SLOPE_DEG = _T("TOP_BANK_FLAT_SLOPE_DEG", 8.0)
TOP_BANK_MIN_WIDTH = _T("TOP_BANK_MIN_WIDTH", 0.20)
TOP_BANK_MAX_WIDTH = _T("TOP_BANK_MAX_WIDTH", 8.00)

# Ditch shape validation (DETECTION GATES)
# Geometric plausibility filter in `_try_find_ditch`. These are not
# maintenance-condition limits. They are broad sanity gates derived from
# the adaptive SLOPE_ANOMALY_* bands with a margin, so degraded ditches can
# still be detected and later labelled as SLOPE_ISSUE by diagnose_issues().
DITCH_INNER_SLOPE_MIN_DEG = _T("DITCH_INNER_SLOPE_MIN_DEG", 3.0)
DITCH_INNER_SLOPE_MAX_DEG = _T("DITCH_INNER_SLOPE_MAX_DEG", 80.0)
DITCH_OUTER_SLOPE_MIN_DEG = _T("DITCH_OUTER_SLOPE_MIN_DEG", 3.0)
DITCH_OUTER_SLOPE_MAX_DEG = _T("DITCH_OUTER_SLOPE_MAX_DEG", 80.0)
DITCH_MAX_WIDTH           = _T("DITCH_MAX_WIDTH",           4.0)   # wider than this is likely terrain

# Slope anomaly thresholds (SLOPE_ISSUE DIAGNOSIS)
# Literature basis:
# - Rail Baltica Design Guidelines Part 2 - Hydraulic, Drainage and
# Culverts (2025): nominal railway side-ditch DESIGN slope 1:2
# (>=6.6 deg) to 1:1 (>=5 deg).
# - Trafikverket TDOK 2015:0155 Section 8.4.1 (oppna diken function list).
# - Knighton, A. D. (1981). Asymmetry of river channel cross-sections:
# Quantitative indices. Earth Surf. Process. Landf. - A1 area-based
# asymmetry index, |A1| > 0.5 signals pronounced one-sided form.
# - Frontiers Earth Sci. 2024 (Chen et al., defect detection review):
# protective-structure deformation signatures in point clouds.
# - Roelens et al. 2018 Section 3.2 - LiDAR-derived profiles are smoothed by
# the morphological-opening envelope used here; the resulting
# EFFECTIVE slope is systematically lower than the design slope.
#
# Why the numbers below are lower than the design slopes:
# MLS LiDAR + morphological opening shifts the measured slope
# distribution. Empirical distribution on tile 630 (779 confirmed
# left-side ditches):
# inner slope: median 16.3 deg, 5-95 pct = 4.4 deg - 37.6 deg
# outer slope: median 17.6 deg, 5-95 pct = 3.7 deg - 41.4 deg
# The design value 26.6 deg is the peak of the EXPECTED angle for a
# well-maintained ditch, but the 5-95 pct band represents the natural
# scatter of a population that is mostly within spec. Anomaly
# thresholds are set just OUTSIDE this band so that "slope outside
# 5-95 pct" becomes the operative definition of abnormal.
#
# Interpretation:
# inner < 8 deg indicates a slumped, silted, or eroded-flat bank
# inner > 55 deg indicates an over-steepened wall, collapse scar, or scour undercut
# outer < 8 deg indicates a slumped far bank or eroded fill
# outer > 55 deg indicates an over-steepened far bank
# |A1| > 0.5 indicates one-sided failure (Knighton moderate-to-extreme)
SLOPE_ANOMALY_INNER_MIN = _T("SLOPE_ANOMALY_INNER_MIN", 8.0)
SLOPE_ANOMALY_INNER_MAX = _T("SLOPE_ANOMALY_INNER_MAX", 55.0)
SLOPE_ANOMALY_OUTER_MIN = _T("SLOPE_ANOMALY_OUTER_MIN", 8.0)
SLOPE_ANOMALY_OUTER_MAX = _T("SLOPE_ANOMALY_OUTER_MAX", 55.0)
SLOPE_ASYMMETRY_MAX     = _T("SLOPE_ASYMMETRY_MAX",     0.50)   # |A1| > 0.5 indicates one-sided collapse (Knighton)

# Water / low-density detection
# NIR LiDAR is absorbed by water, which produces dramatically fewer returns.
WATER_DENSITY_RATIO       = _T("WATER_DENSITY_RATIO",       0.25)
# Relative intensity: a populated bin is "low intensity" when it is below this
# fraction of the local-median intensity. Relative (not absolute) because the
# raw intensity scale varies across datasets (8-bit raw vs 16-bit /256-scaled).
WATER_INTENSITY_RATIO     = _T("WATER_INTENSITY_RATIO",     0.60)
WATER_MIN_CONTIGUOUS_BINS = int(_T("WATER_MIN_CONTIGUOUS_BINS", 3))
WATER_SPAN_MARGIN_M       = _T("WATER_SPAN_MARGIN_M",       0.30)  # search only the ditch-bottom span +/- this
# Fallback for deep water that even Pass 4 misses (no ditch span): scan only
# the realistic near-track band and demand a longer contiguous density hole.
WATER_GAP_FALLBACK_MAX_M  = _T("WATER_GAP_FALLBACK_MAX_M",  6.00)
WATER_GAP_FALLBACK_MIN_RUN = int(_T("WATER_GAP_FALLBACK_MIN_RUN", 2 * WATER_MIN_CONTIGUOUS_BINS))

# Water-occlusion depth recovery (bank-slope extrapolation)
# NIR LiDAR does not penetrate standing water, so in a water-filled ditch the
# lower envelope follows the water SURFACE (or the banks where returns are
# sparse), not the physical bed; the measured depth there is systematically
# shallow. Field GNSS on tile 7302_816 (water-filled, ~12 m off-track) shows a
# ~0.5 m underestimate, whereas the dry cess ditch on 7309_809 agrees to ~0.1 m.
# Where water is flagged we recover the dry-bed depth by fitting the two exposed
# banks ABOVE the water line and extrapolating them to their intersection (the
# thalweg) - the standard bathymetric reconstruction used when the bed itself is
# not observed (Roelens et al. 2018 Section 3.2; Bailly et al. 2008). This runs ONLY on
# water-flagged sections, so dry ditches/tiles are left exactly as before, and it
# can only deepen (never raise) a measured bottom. Set WATER_BOTTOM_RECON=0 in
# the environment to reproduce the pre-recovery behaviour.
USE_WATER_BOTTOM_RECON        = os.environ.get("WATER_BOTTOM_RECON", "1") == "1"
WATER_RECON_BANK_MARGIN_M     = _T("WATER_RECON_BANK_MARGIN_M",     0.04)  # exclude bins within this of the water line
WATER_RECON_MIN_BANK_PTS      = int(_T("WATER_RECON_MIN_BANK_PTS",  3))    # min envelope pts per bank for the line fit
WATER_RECON_MIN_SLOPE_DEG     = _T("WATER_RECON_MIN_SLOPE_DEG",     5.0)   # banks gentler than this give no reliable apex
WATER_RECON_MAX_SLOPE_DEG     = _T("WATER_RECON_MAX_SLOPE_DEG",     80.0)
WATER_RECON_MAX_EXTRA_DEPTH_M = _T("WATER_RECON_MAX_EXTRA_DEPTH_M", 1.20)  # cap recovery below the measured water line

# TDOK 2015:0155 Section 10.13 depth thresholds (below RUK)
# Z_ref is now RUK (rail underside) per step06 RAIL_HEIGHT_M correction.
# TDOK 2015:0155 §10.13: depth < 1.0 m below RUK triggers priority Å,
# and cleaned ditches should reach >=1.3 m below RUK.
TDOK_DEPTH_SUFFICIENT_M = _T("TDOK_DEPTH_SUFFICIENT_M", 1.0)   # below this -> SHALLOW (priority Å)
TDOK_DEPTH_TARGET_M     = _T("TDOK_DEPTH_TARGET_M",     1.3)   # post-intervention target depth

# Shape descriptor thresholds (Knighton 1981 + heuristic)
# flatness = width of "near-bottom band" / total top width
# See compute_shape_descriptors() for definitions.
SHAPE_FLATNESS_U_MIN    = _T("SHAPE_FLATNESS_U_MIN",    0.15)
SHAPE_FLATNESS_FLAT_MIN = _T("SHAPE_FLATNESS_FLAT_MIN", 0.35)
SHAPE_DEPTH_DEFICIT_MIN = _T("SHAPE_DEPTH_DEFICIT_MIN", 0.20)
SHAPE_BAND_FRAC         = _T("SHAPE_BAND_FRAC",         0.10)

# Content classifier thresholds (Roelens 2018 alignment)
# DVCI = fraction of points with ExG > EXG_VEG_THRESHOLD within ditch span
CONTENT_DVCI_PARTIAL = _T("CONTENT_DVCI_PARTIAL", 0.20)
CONTENT_DVCI_DENSE   = _T("CONTENT_DVCI_DENSE",   0.50)

# Aggregation / grading
SEGMENT_LEN          = _T("SEGMENT_LEN",          10.0)
MIN_VALID_IN_SEGMENT = int(_T("MIN_VALID_IN_SEGMENT", 3))

RANDOM_SEED = 42


# =========================================================
# HELPERS
# =========================================================
def fmt_n(x):
    return f"{int(x):,}"


def to_ST(xv, yv, mean_x, mean_y, angle_rad):
    cos_a = np.float32(np.cos(angle_rad))
    sin_a = np.float32(np.sin(angle_rad))
    mean_x = np.float32(mean_x)
    mean_y = np.float32(mean_y)
    S = (xv - mean_x) * cos_a + (yv - mean_y) * sin_a
    T = -(xv - mean_x) * sin_a + (yv - mean_y) * cos_a
    return S, T


def make_savgol(y, max_win=21, poly=3):
    y = np.asarray(y)
    if len(y) < 5:
        return y.copy()
    win = min(len(y), max_win)
    if win % 2 == 0:
        win -= 1
    win = max(win, 5)
    poly = min(poly, win - 2)
    return savgol_filter(y, window_length=win, polyorder=poly)


def estimate_top_bank_width(
    x, z, bottom_idx, local_i0=None, local_i1=None,
    allow_local_edge_fallback=True,
):
    """Estimate physical top-of-bank width from a 1-D ditch profile.

    The main detector's `width` is deliberately conservative: it walks only
    a short distance away from the bottom so that broad embankment slopes are
    not accepted as ditches.  For reporting ditch geometry, however, the more
    useful width is the distance between the two bank tops.  This helper
    searches farther from the detected bottom and selects a bank-top point
    where the profile has recovered most of its local relief and, preferably,
    where the side slope has flattened.

    Returns
    -------
    dict with top_width, top_span_x0, top_span_x1, top_width_quality.
    Quality is one of:
      - slope_break: both sides reached a recovered, locally flat bank top
      - height_recovery: both sides recovered in height but no flat break
      - partial: one side used the detector edge as fallback
      - invalid: no plausible top width could be estimated
    """
    out = {
        "top_width": np.nan,
        "top_span_x0": np.nan,
        "top_span_x1": np.nan,
        "top_i0": None,
        "top_i1": None,
        "top_width_quality": "invalid",
    }

    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(x) < 6 or len(z) != len(x):
        return out
    if bottom_idx <= 0 or bottom_idx >= len(x) - 1:
        return out
    if np.any(~np.isfinite(x)) or np.all(~np.isfinite(z)):
        return out

    finite = np.isfinite(z)
    if finite.sum() < 6:
        return out

    z_fill = z.copy()
    if (~finite).any():
        z_fill[~finite] = np.interp(x[~finite], x[finite], z[finite])

    # Smooth only for slope-break detection. The selected height itself
    # still comes from the envelope coordinates.
    z_s = make_savgol(z_fill, max_win=9, poly=2)
    dx_med = float(np.nanmedian(np.diff(x))) if len(x) > 1 else ENV_BIN_W
    dx_med = max(dx_med, 1e-3)
    slope = np.gradient(z_s, x)
    flat_slope = np.tan(np.radians(TOP_BANK_FLAT_SLOPE_DEG))
    min_flat_bins = max(2, int(np.ceil(0.15 / dx_med)))

    bottom_x = float(x[bottom_idx])
    bottom_z = float(z_s[bottom_idx])

    def find_one_side(direction, detector_edge):
        if direction < 0:
            side_idx = np.arange(bottom_idx - 1, -1, -1)
        else:
            side_idx = np.arange(bottom_idx + 1, len(x))

        if len(side_idx) == 0:
            return None, "missing"

        side_idx = side_idx[np.abs(x[side_idx] - bottom_x) <= TOP_BANK_SEARCH_MAX_M]
        if len(side_idx) == 0:
            return None, "missing"

        side_ref = float(np.nanpercentile(z_s[side_idx], 85))
        if not np.isfinite(side_ref) or side_ref <= bottom_z:
            if allow_local_edge_fallback and detector_edge is not None and 0 <= detector_edge < len(x):
                return int(detector_edge), "local_edge"
            return None, "missing"

        recovery_z = bottom_z + TOP_BANK_RECOVERY_FRAC * (side_ref - bottom_z)
        recovered = [int(i) for i in side_idx if z_s[i] >= recovery_z]
        if not recovered:
            if allow_local_edge_fallback and detector_edge is not None and 0 <= detector_edge < len(x):
                return int(detector_edge), "local_edge"
            return None, "missing"

        # Prefer the first recovered point whose terrain farther away from
        # the bottom has become locally flat. This is the slope-break
        # analogue of the top-of-bank rule used in channel morphometry.
        for idx in recovered:
            if direction < 0:
                flat_window = np.arange(max(0, idx - min_flat_bins + 1), idx + 1)
            else:
                flat_window = np.arange(idx, min(len(x), idx + min_flat_bins))
            if len(flat_window) >= min_flat_bins:
                if float(np.nanmedian(np.abs(slope[flat_window]))) <= flat_slope:
                    return idx, "slope_break"

        return recovered[0], "height_recovery"

    left_idx, left_q = find_one_side(-1, local_i0)
    right_idx, right_q = find_one_side(+1, local_i1)

    if left_idx is None or right_idx is None or right_idx <= left_idx:
        return out

    top_width = float(x[right_idx] - x[left_idx])
    if not (TOP_BANK_MIN_WIDTH <= top_width <= TOP_BANK_MAX_WIDTH):
        return out

    if left_q == "slope_break" and right_q == "slope_break":
        quality = "slope_break"
    elif left_q == "local_edge" or right_q == "local_edge":
        quality = "partial"
    else:
        quality = "height_recovery"

    out.update({
        "top_width": top_width,
        "top_span_x0": float(x[left_idx]),
        "top_span_x1": float(x[right_idx]),
        "top_i0": int(left_idx),
        "top_i1": int(right_idx),
        "top_width_quality": quality,
    })
    return out


def gray_exg_from_rgb(R, G, B):
    return 2.0 * G - R - B


def project_to_local_frame(S_vals, T_vals, s0, t0, dtds):
    tangent = np.array([1.0, dtds], dtype=float)
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-dtds, 1.0], dtype=float)
    normal /= np.linalg.norm(normal)
    dS = S_vals - s0
    dT = T_vals - t0
    u = dS * tangent[0] + dT * tangent[1]
    v = dS * normal[0] + dT * normal[1]
    return u, v


def sorted_slice_bounds(sorted_vals, lo, hi):
    i0 = int(np.searchsorted(sorted_vals, lo, side="left"))
    i1 = int(np.searchsorted(sorted_vals, hi, side="right"))
    return i0, i1


# =========================================================
# LOWER ENVELOPE: MORPHOLOGICAL OPENING (Roelens 2016/2018)
# =========================================================
def build_lower_envelope(v_vals, z_vals, i_vals=None,
                         v_range=7.0, bin_w=0.05,
                         q=5, min_points_per_bin=3):
    """
    Build lower envelope on a REGULAR grid using percentile binning +
    morphological opening.  Empty bins (0 points  - e.g. water absorption)
    are filled by linear interpolation so the 1-D filters operate on a
    spatially uniform array (Bailly et al. 2008 gap-aware approach).

    Returns: env_v, env_z (morphological), bin_counts, bin_intensity_median
    """
    bins = np.arange(-v_range, v_range + bin_w, bin_w)
    n_bins = len(bins) - 1
    grid_v = np.array([(bins[j] + bins[j + 1]) / 2.0 for j in range(n_bins)])
    grid_z = np.full(n_bins, np.nan)
    grid_cnt = np.zeros(n_bins, dtype=int)
    grid_i = np.full(n_bins, np.nan)

    for j in range(n_bins):
        m = (v_vals >= bins[j]) & (v_vals < bins[j + 1])
        n = int(m.sum())
        grid_cnt[j] = n
        # Sparse bins (<min_points_per_bin points) are
        # left as NaN and filled later by linear interpolation between
        # well-populated neighbours. Previously a single stray point
        # in an otherwise-empty bin was taken as the envelope value,
        # creating 30+cm dips that look like spurious peaks to find_peaks.
        if n >= min_points_per_bin:
            grid_z[j] = np.percentile(z_vals[m], q)
            if i_vals is not None:
                grid_i[j] = float(np.median(i_vals[m]))
        # else: leave grid_z[j] as NaN; interpolation later

    # Trim leading/trailing NaN
    valid = ~np.isnan(grid_z)
    if valid.sum() < 5:
        idx = np.where(valid)[0]
        return grid_v[idx], grid_z[idx], grid_cnt[idx], grid_i[idx]

    first_valid = np.argmax(valid)
    last_valid = len(valid) - 1 - np.argmax(valid[::-1])
    sl = slice(first_valid, last_valid + 1)
    gv = grid_v[sl].copy()
    gz = grid_z[sl].copy()
    gc = grid_cnt[sl].copy()
    gi = grid_i[sl].copy()

    # Fill gaps with MIN of flanks
    # Linear interpolation across a water-filled ditch averages the
    # two rim elevations and produces a shallow fake bottom, hiding
    # the real depth. Taking the minimum of the nearest valid bin
    # on each side is conservative: it preserves depth if at least
    # one flank touches the true ditch floor (from partial returns
    # off the sidewall), and degenerates to the shallower flank
    # when the gap is adjacent to one well-populated bin only.
    valid_inner = ~np.isnan(gz)
    if valid_inner.sum() >= 2 and (~valid_inner).sum() > 0:
        gz_s = pd.Series(gz)
        left_fill  = gz_s.ffill().values   # propagate last valid forward
        right_fill = gz_s.bfill().values   # propagate next valid backward
        # If only one side has a valid value, use it on both
        left_fill  = np.where(np.isnan(left_fill),  right_fill, left_fill)
        right_fill = np.where(np.isnan(right_fill), left_fill,  right_fill)
        gap_mask = ~valid_inner
        gz[gap_mask] = np.minimum(left_fill[gap_mask], right_fill[gap_mask])

    if len(gz) < 5:
        return gv, gz, gc, gi

    # morphological opening: erosion (min) then dilation (max)
    env_z_eroded = minimum_filter1d(gz, size=MORPH_KERNEL_SIZE, mode="nearest")
    env_z_opened = maximum_filter1d(env_z_eroded, size=MORPH_KERNEL_SIZE, mode="nearest")

    # very light post-smoothing (win=5 = 0.25m) only to remove quantization
    if len(env_z_opened) >= 5:
        env_z_opened = make_savgol(env_z_opened,
                                   max_win=MORPH_POST_SMOOTH_WIN, poly=2)

    return gv, env_z_opened, gc, gi


# =========================================================
# WATER / LOW-DENSITY DETECTION
# =========================================================
def detect_water_in_ditch(bin_v, bin_counts, bin_intensity, side_sign,
                          ditch_exists=False, span_x0=np.nan, span_x1=np.nan,
                          search_min=SIDE_SEARCH_MIN,
                          search_max=SIDE_SEARCH_MAX):
    """
    Detect potential standing water in ditch zone using LiDAR physical properties:
    - NIR absorption gives low point density in water-covered bins
    - Low return intensity from water surface
    Water can only sit inside a confirmed ditch, so detection is restricted to
    the detected ditch-bottom span (+/-WATER_SPAN_MARGIN_M) and requires BOTH a
    density dropout AND an intensity dropout. This avoids false positives on
    dry, sparsely-sampled background bins far from any ditch.
    Returns: (water_flag, water_fraction, n_water_bins)
    """
    if len(bin_v) < 5:
        return False, 0.0, 0

    if side_sign < 0:
        x_from_centre = -bin_v
    else:
        x_from_centre = bin_v

    if ditch_exists and np.isfinite(span_x0) and np.isfinite(span_x1) and span_x1 > span_x0:
        # Primary path: search only the detected ditch-bottom span (+ margin).
        lo = max(search_min, span_x0 - WATER_SPAN_MARGIN_M)
        hi = min(search_max, span_x1 + WATER_SPAN_MARGIN_M)
        min_run = WATER_MIN_CONTIGUOUS_BINS
    else:
        # Fallback: deep water can defeat even Pass 4, leaving no ditch span.
        # Scan only the realistic near-track band and demand a LONGER
        # contiguous density hole so dry background sparsity stays unflagged.
        lo = search_min
        hi = min(search_max, WATER_GAP_FALLBACK_MAX_M)
        min_run = WATER_GAP_FALLBACK_MIN_RUN

    zone = (x_from_centre >= lo) & (x_from_centre <= hi)
    if zone.sum() < 3:
        return False, 0.0, 0

    zone_counts = bin_counts[zone]
    zone_intensity = bin_intensity[zone]

    # local median density as reference
    local_median_count = float(np.median(zone_counts[zone_counts > 0])) \
        if (zone_counts > 0).sum() > 0 else 1.0

    # low density bins (< WATER_DENSITY_RATIO of local median)
    low_density = zone_counts < (WATER_DENSITY_RATIO * local_median_count)

    # A bin is water-suspect only if it is NOT bright. Empty bins (NaN
    # intensity) keep the default True: an empty bin is the strongest water
    # signal (full NIR absorption) and must not be vetoed for lacking returns.
    # Populated bins must additionally be below WATER_INTENSITY_RATIO of the
    # local-median intensity (relative -> scale-robust). This is what rejects
    # dry, sparse-but-bright bins that caused the old false positives.
    not_bright = np.ones_like(low_density)
    valid_i = ~np.isnan(zone_intensity)
    if valid_i.sum() >= 3:
        local_median_i = float(np.median(zone_intensity[valid_i]))
        if local_median_i > 0:
            not_bright[valid_i] = zone_intensity[valid_i] < (WATER_INTENSITY_RATIO * local_median_i)

    water_suspect = low_density & not_bright

    # require contiguous run of at least N bins
    n_water_bins = 0
    max_run = 0
    current_run = 0
    for ws in water_suspect:
        if ws:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    water_flag = max_run >= min_run
    water_frac = float(water_suspect.sum()) / len(water_suspect) if len(water_suspect) > 0 else 0.0
    n_water_bins = int(water_suspect.sum())

    return water_flag, water_frac, n_water_bins


def reconstruct_water_filled_bottom(x, z, bottom_x, span_x0, span_x1,
                                    water_surface_z, z_ref):
    """
    Recover the dry-bed depth of a water-filled ditch by bank-slope extrapolation.

    NIR LiDAR reflects off standing water, so the measured `bottom` of a flooded
    ditch is the water surface, not the bed.  The two banks above the water line
    are still imaged correctly, so fitting a straight line to each exposed bank
    and intersecting them recovers the thalweg (dry-bed) elevation.

    Parameters
    ----------
    x, z : 1-D lower-envelope cross-section for ONE side; x = outward distance
           from the centreline (ascending), z = elevation.  These are the same
           arrays the detector used, so the recovered geometry stays consistent
           with the reported bottom/span.
    bottom_x : detected (water-surface) bottom position [m].
    span_x0, span_x1 : top-of-bank span of the detected ditch [m].
    water_surface_z : envelope elevation at the detected bottom (~ water level).
    z_ref : RUK datum; the returned depth is metres below RUK.

    Returns
    -------
    dict(valid, bottom_z, depth, method)
        valid    : True only when a physically sensible, DEEPER bed was found.
        bottom_z : reconstructed bed elevation (NaN if invalid).
        depth    : z_ref - bottom_z, clipped to the validity range (NaN if invalid).
        method   : short tag for the CSV / logs.
    """
    out = {"valid": False, "bottom_z": np.nan, "depth": np.nan, "method": "none"}

    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(x) < 2 * WATER_RECON_MIN_BANK_PTS or len(z) != len(x):
        return out
    if any(not np.isfinite(v) for v in (bottom_x, span_x0, span_x1, water_surface_z, z_ref)):
        return out
    if span_x1 <= span_x0 or not (span_x0 <= bottom_x <= span_x1):
        return out

    margin = max(WATER_RECON_BANK_MARGIN_M, 0.0)
    # Inner (track-side) bank: span start ->bottom, only points above water.
    inner = (x >= span_x0) & (x <= bottom_x) & (z >= water_surface_z + margin)
    # Outer (field-side) bank: bottom ->span end, only points above water.
    outer = (x >= bottom_x) & (x <= span_x1) & (z >= water_surface_z + margin)
    if int(inner.sum()) < WATER_RECON_MIN_BANK_PTS or int(outer.sum()) < WATER_RECON_MIN_BANK_PTS:
        return out

    # Fit a straight line to each exposed bank: z = a*x + b.
    try:
        a_in, b_in = np.polyfit(x[inner], z[inner], 1)
        a_out, b_out = np.polyfit(x[outer], z[outer], 1)
    except Exception:
        return out

    # Inner bank must descend toward the bottom (a_in < 0); the outer bank must
    # rise (a_out > 0). Reject banks gentler than WATER_RECON_MIN_SLOPE_DEG (no
    # reliable apex) or steeper than WATER_RECON_MAX_SLOPE_DEG (numerically
    # unstable / not a real bank).
    tan_min = np.tan(np.radians(WATER_RECON_MIN_SLOPE_DEG))
    tan_max = np.tan(np.radians(WATER_RECON_MAX_SLOPE_DEG))
    if not (-tan_max <= a_in <= -tan_min):
        return out
    if not (tan_min <= a_out <= tan_max):
        return out

    denom = a_in - a_out
    if abs(denom) < 1e-6:
        return out
    x_star = (b_out - b_in) / denom
    z_star = a_in * x_star + b_in

    # The apex must fall inside the ditch span and below the water surface.
    if not (span_x0 <= x_star <= span_x1):
        return out
    if z_star >= water_surface_z:
        return out

    # Cap how far below the water surface the recovery may go, so a poor bank
    # fit cannot invent an implausibly deep bed.
    z_floor = water_surface_z - WATER_RECON_MAX_EXTRA_DEPTH_M
    if z_star < z_floor:
        z_star = z_floor

    depth = float(np.clip(z_ref - z_star, DEPTH_VALID_MIN, DEPTH_VALID_MAX))

    out.update({
        "valid": True,
        "bottom_z": float(z_star),
        "depth": depth,
        "method": "bank_slope_intersection",
    })
    return out


# =========================================================
# SHAPE DESCRIPTORS (Knighton 1981 + sediment-aware extension)
# =========================================================
def compute_shape_descriptors(xk, zk, i0, i1, pk, inner_slope_deg):
    """
    Compute three geometric descriptors of a ditch cross-section to
    distinguish V-shape (healthy) vs U-shape vs flat-bottom (silted).

    Inputs
    ------
    xk, zk          : 1-D arrays of the lower envelope cross-section
                      (outward-distance and elevation) already filtered
                      to the ditch side.
    i0, i1          : indices into xk/zk for the ditch left/right edges.
    pk              : index of the ditch bottom (argmin of zk[i0..i1]).
    inner_slope_deg : inner-side slope in degrees (track ->bottom).

    Returns (dict)
    --------------
    flatness       : width_bottom_band / top_width.
                     width_bottom_band = longest run of samples within
                     SHAPE_BAND_FRAC x total_depth of the minimum.
                     0 means pure V-shape; 1 means entirely flat.
    asymmetry      : (A_left - A_right) / (A_left + A_right)
                     where A_left, A_right are the cross-section areas
                     on each side of the bottom. Range -1 to +1;
                     0 = symmetric, sign indicates which side is larger.
                     After Knighton (1981) A1 asymmetry index.
    depth_deficit  : 1 - (actual_depth / ideal_V_depth).
                     ideal_V_depth assumes a perfect V with the observed
                     inner slope and top width. 0 = matches ideal V;
                     values near 1 indicate the ditch has filled in.
    """
    nan_out = {
        "flatness": np.nan,
        "asymmetry": np.nan,
        "depth_deficit": np.nan,
    }
    if i1 <= i0 + 1 or pk < i0 or pk > i1:
        return nan_out

    x_sub = xk[i0:i1 + 1]
    z_sub = zk[i0:i1 + 1]
    pk_local = pk - i0
    n = len(x_sub)
    if n < 3:
        return nan_out

    top_width = float(x_sub[-1] - x_sub[0])
    if top_width <= 0:
        return nan_out

    z_min = float(z_sub[pk_local])
    z_edge_max = float(max(z_sub[0], z_sub[-1]))
    total_depth = z_edge_max - z_min
    if total_depth <= 1e-6:
        return nan_out

    # flatness
    # longest contiguous run of samples within SHAPE_BAND_FRAC of z_min.
    band_cut = z_min + SHAPE_BAND_FRAC * total_depth
    in_band = z_sub <= band_cut
    # contiguous run containing pk_local
    left = pk_local
    while left > 0 and in_band[left - 1]:
        left -= 1
    right = pk_local
    while right < n - 1 and in_band[right + 1]:
        right += 1
    bottom_band_width = float(x_sub[right] - x_sub[left]) if right > left else 0.0
    flatness = float(bottom_band_width / top_width)

    # asymmetry (Knighton 1981 A1-style, area-based)
    # areas above z_min under the profile, split at the bottom.
    deficit = np.maximum(0.0, z_edge_max - z_sub)
    if pk_local >= 1:
        a_left = float(np.trapz(deficit[:pk_local + 1], x_sub[:pk_local + 1]))
    else:
        a_left = 0.0
    if pk_local <= n - 2:
        a_right = float(np.trapz(deficit[pk_local:], x_sub[pk_local:]))
    else:
        a_right = 0.0
    denom = a_left + a_right
    asymmetry = float((a_left - a_right) / denom) if denom > 1e-9 else 0.0

    # depth deficit vs ideal V-shape
    # A perfect V with given top width and observed inner slope would have
    # depth = 0.5 * top_width * tan(inner_slope). Compare to actual depth.
    if not np.isnan(inner_slope_deg) and inner_slope_deg > 0:
        ideal_depth = 0.5 * top_width * np.tan(np.radians(inner_slope_deg))
        if ideal_depth > 1e-6:
            depth_deficit = float(1.0 - (total_depth / ideal_depth))
            depth_deficit = float(np.clip(depth_deficit, -1.0, 1.0))
        else:
            depth_deficit = np.nan
    else:
        depth_deficit = np.nan

    # bottom curvature (normalised)
    # Second derivative of the envelope at the ditch bottom, normalised
    # by the natural ditch scale (depth / top_width^2) so that different-
    # sized ditches with the same shape give similar numerical values.
    # Interpretation:
    # V_HEALTHY means the bottom is a sharp point with large positive curvature (> ~2).
    # U_MODERATE means the bottom is rounded with moderate positive curvature (~0.5-2).
    # FLAT_SILTED means the bottom is flat or smoothed with curvature near zero (< ~0.5).
    #
    # Caveat for water-filled ditches: because the envelope in water gaps
    # is reconstructed from min-of-flanks, the curvature at
    # the reconstructed "bottom" is artificially close to 0 regardless of
    # the underlying true shape. classify_shape() therefore falls back to
    # flatness-only when water_flag is set, with reduced confidence.
    if n >= 5:
        dz_dx  = np.gradient(z_sub, x_sub)
        ddz_dx2 = np.gradient(dz_dx, x_sub)
        curvature_raw = float(ddz_dx2[pk_local])
        # Normalise by depth / top_width^2 (natural scaling for depressions)
        scale = total_depth / (top_width ** 2)
        if scale > 1e-6:
            bottom_curvature_norm = float(curvature_raw / scale)
        else:
            bottom_curvature_norm = np.nan
    else:
        bottom_curvature_norm = np.nan

    return {
        "flatness": flatness,
        "asymmetry": asymmetry,
        "depth_deficit": depth_deficit,
        "bottom_curvature_norm": bottom_curvature_norm,
    }


def classify_shape(flatness, depth_deficit, bottom_curvature_norm=np.nan,
                   water_flag=False):
    """
    Map the shape descriptors to discrete shape classes, and
    return a confidence score in [0, 1].

    Classes:
        V_HEALTHY   - narrow V-shaped bottom, healthy geometry
        U_MODERATE  - moderate flat area, acceptable or mildly silted
        FLAT_SILTED  - flat bottom + depth deficit, probable sediment fill
        UNKNOWN_WATER_MASKED
                     - water-suspect ditch; the measured surface is not the
                       physical bottom, so V/U/flat cannot be inferred
        NA           - descriptors unavailable

    Confidence:
        1.0    - both flatness and curvature agree on the class
        0.7    - flatness-only classification (curvature not available)
        0.2    - water-flagged section: ditch exists, but bottom shape is
                not observable because the lower envelope represents the
                water/visible surface rather than the physical bottom.
        0.6    - flatness and curvature disagree (class still assigned, but
                uncertain).

    Returns
    -------
    (class_name : str, confidence : float)
    """
    if np.isnan(flatness):
        return "NA", 0.0

    # If water is detected, the lower envelope around the "bottom" is not a
    # reliable bottom-shape observation. It may be a water surface, sparse-bin
    # reconstruction, or vegetation/water mixed surface. Do not label it V/U/flat.
    if water_flag:
        return "UNKNOWN_WATER_MASKED", 0.2

    # Base classification from flatness (same as legacy behaviour)
    if flatness < SHAPE_FLATNESS_U_MIN:
        base_class = "V_HEALTHY"
    elif flatness < SHAPE_FLATNESS_FLAT_MIN:
        base_class = "U_MODERATE"
    else:
        base_class = "FLAT_SILTED"

    if np.isnan(bottom_curvature_norm):
        # Dry section but curvature unavailable (e.g. very short profile)
        # Fall back to flatness-only, moderate confidence.
        return base_class, 0.7

    # Dry section with curvature: check agreement
    cv = bottom_curvature_norm
    agreement_ok = False
    if base_class == "V_HEALTHY" and cv > 2.0:
        agreement_ok = True
    elif base_class == "U_MODERATE" and 0.5 <= cv <= 3.0:
        agreement_ok = True
    elif base_class == "FLAT_SILTED" and abs(cv) < 1.0:
        agreement_ok = True

    return base_class, (1.0 if agreement_ok else 0.6)


# =========================================================
# DITCH METRICS
# =========================================================
def _compute_confidence(pass_source, prominence, depth, width):
    """
    Composite confidence score in [0, 1] for a single ditch detection.
    The score combines detection source, prominence, depth, and width.

    Components
    ----------
    base (pass source)  : pass1=1.00, pass2=0.75, pass3=0.60, pass4=0.45
    prominence quality  : 0 at DITCH_MIN_PROM, saturates at +0.10 m above
    depth quality       : 0 at DITCH_MIN_DEPTH, saturates at +0.30 m above
    width plausibility  : after the top-of-bank refactor, this is the
                          reported top width.  Railway side ditches in
                          this tile are expected to be a few metres wide;
                          the factor is 1.0 in [1.0, 5.0] m and tapers
                          to zero at TOP_BANK_MIN_WIDTH / TOP_BANK_MAX_WIDTH.

    The three quality terms are combined by geometric mean (so a very
    weak component drags the score down) and then scaled by the pass
    base.  Returns a float clipped to [0, 1].
    """
    base = {
        "pass1":             1.00,
        "pass2_cutting":     0.75,
        "pass2_embankment":  0.75,
        "pass3_toe":         0.60,
        "pass4_gap":         0.45,
    }.get(pass_source, 0.50)

    prom_q = float(np.clip((prominence - DITCH_MIN_PROM) / 0.10, 0.0, 1.0))
    depth_q = float(np.clip((depth - DITCH_MIN_DEPTH) / 0.30, 0.0, 1.0))

    if width < TOP_BANK_MIN_WIDTH or width > TOP_BANK_MAX_WIDTH:
        width_q = 0.0
    elif 1.0 <= width <= 5.0:
        width_q = 1.0
    elif width < 1.0:
        width_q = (width - TOP_BANK_MIN_WIDTH) / max(1e-6, (1.0 - TOP_BANK_MIN_WIDTH))
    else:  # 5.0 < width <= TOP_BANK_MAX_WIDTH
        width_q = max(0.0, 1.0 - (width - 5.0) / max(1e-6, (TOP_BANK_MAX_WIDTH - 5.0)))

    quality = (
        max(prom_q, 0.05)
        * max(depth_q, 0.05)
        * max(width_q, 0.05)
    ) ** (1.0 / 3.0)
    return float(np.clip(base * quality, 0.0, 1.0))


def _non_ditch_fallback(x, z):
    out = {
        "side_type": "FLAT",
        "relief": np.nan,
        "depth": np.nan,
        "width": np.nan,
        "top_width": np.nan,
        "area": np.nan,
        "bottom_x": np.nan,
        "bottom_z": np.nan,
        "span_x0": np.nan,
        "span_x1": np.nan,
        "top_span_x0": np.nan,
        "top_span_x1": np.nan,
        "top_width_quality": "invalid",
        "inner_slope_deg": np.nan,
        "outer_slope_deg": np.nan,
        "flatness": np.nan,
        "asymmetry": np.nan,
        "depth_deficit": np.nan,
        "dvci": np.nan,
        "ditch_exists": 0,
        "pass_source": "none",
        "confidence": 0.0,
        "candidate_json": "[]",
        "candidate_count": 0,
    }
    if len(x) < 4:
        return out

    relief = float(np.nanmax(z) - np.nanmin(z))
    out["relief"] = relief

    diffs = np.diff(z)
    frac_desc = float((diffs < 0).mean()) if len(diffs) else 0.0
    frac_asc = float((diffs > 0).mean()) if len(diffs) else 0.0

    if (z[0] - np.nanmin(z)) >= EMBANKMENT_MIN_DROP and frac_desc >= 0.60:
        out["side_type"] = "EMBANKMENT"
        return out

    if (np.nanmax(z) - z[0]) >= CUTTING_MIN_RISE and frac_asc >= 0.60:
        out["side_type"] = "CUTTING"
        return out

    total_change = float(z[-1] - z[0])
    if relief <= FLAT_RELIEF_MAX:
        out["side_type"] = "FLAT"
    elif total_change <= -EMBANKMENT_MIN_DROP:
        out["side_type"] = "EMBANKMENT"
    elif total_change >= CUTTING_MIN_RISE:
        out["side_type"] = "CUTTING"
    else:
        out["side_type"] = "FLAT"

    return out


def _pack_candidates_for_json(candidates):
    """Keep a compact, serialisable copy of valid local candidates."""
    if not candidates:
        return "[]", 0
    fields = [
        "depth", "width", "top_width", "area", "bottom_x", "bottom_z",
        "span_x0", "span_x1", "top_span_x0", "top_span_x1",
        "inner_slope_deg", "outer_slope_deg", "flatness", "asymmetry",
        "depth_deficit", "prominence", "confidence", "candidate_score",
        "pass_source",
    ]
    packed = []
    for cand in sorted(candidates, key=lambda c: c.get("candidate_score", 0.0), reverse=True)[:CANDIDATE_JSON_MAX_N]:
        row = {}
        for field in fields:
            value = cand.get(field, np.nan)
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            row[field] = value
        packed.append(row)
    return json.dumps(packed, ensure_ascii=False), len(candidates)


def _try_find_ditch(xk, zk, depth_ref_z, min_depth, prior_x=None,
                    report_ref_z=None, pass_source="pass1",
                    return_all=False):
    """
    Core ditch search: find valleys in the envelope, validate shape.
    Uses find_peaks on inverted Z, plus a global-minimum fallback for
    wide flat-bottomed or gap-filled ditches where find_peaks fails.

    Parameters
    ----------
    depth_ref_z : float
        Reference used for SEARCH and the min_depth gate (e.g. shoulder
        for cuttings/embankments, RUK for standard Pass 1).
    min_depth : float
        Minimum depth below depth_ref_z for a peak to qualify.
    prior_x : float or None
        If given, candidate peaks are sorted by |xk[pk] - prior_x|
        (closest to prior first) instead of "near-to-far from centreline".
        This enforces spatial continuity between consecutive cross-sections,
        preventing the algorithm from oscillating between two coexisting
        real peaks when their prominence happens to flip by ~1 mm.
    report_ref_z : float or None
        Reference written into the returned `depth` / `area` so the
        CSV stays RUK-consistent across passes.  Defaults to depth_ref_z.
        Callers that search with a non-RUK reference (cutting shoulder,
        embankment local min, toe-of-slope) pass the RUK z_ref here so
        the reported depth is always "metres below RUK" per TDOK
        2015:0155 Section 10.13.
    pass_source : str
        Tag attached to the detection ("pass1", "pass2_cutting",
        "pass2_embankment", "pass3_toe", "pass4_gap")  - used by the
        confidence calculator and logged to the CSV.

    Returns dict or None.
    """
    if report_ref_z is None:
        report_ref_z = depth_ref_z

    inv = -zk
    peaks, props = find_peaks(inv, prominence=DITCH_MIN_PROM)

    # Keep per-peak prominence so we can feed it into the confidence calc
    peak_prom = {}
    if len(peaks) > 0 and "prominences" in props:
        for pk_i, pr_i in zip(peaks, props["prominences"]):
            peak_prom[int(pk_i)] = float(pr_i)

    # Fallback: if no peaks found, try the global minimum
    # (handles flat-bottomed ditches and interpolated water gaps)
    if len(peaks) == 0:
        global_min = int(np.argmin(zk))
        # Only use as candidate if it's not at the edges
        if 1 < global_min < len(zk) - 2:
            peaks = np.array([global_min])
            # Estimate prominence from the local flank rise
            flank_left = float(np.max(zk[:global_min])) if global_min > 0 else zk[global_min]
            flank_right = (
                float(np.max(zk[global_min + 1:]))
                if global_min < len(zk) - 1 else zk[global_min]
            )
            peak_prom[global_min] = float(
                min(flank_left, flank_right) - zk[global_min]
            )

    if len(peaks) > 0:
        if prior_x is not None and not np.isnan(prior_x):
            # Sort by closeness to prior position (spatial continuity)
            peaks = peaks[np.argsort(np.abs(xk[peaks] - prior_x))]
        else:
            peaks = peaks[np.argsort(xk[peaks])]   # fallback: near ->far

    valid_candidates = []

    def _main_ditch_score(cand, pk_index):
        """Rank valid candidates after geometry validation."""
        depth_term = max(0.0, cand["depth"] / max(DITCH_MIN_DEPTH, 1e-6))
        prom_term = max(0.0, cand["prominence"] / max(DITCH_MIN_PROM, 1e-6))
        area_term = max(0.0, cand["area"] / max(DITCH_MIN_AREA, 1e-6))
        # Small outer-position preference: it only breaks ties between
        # similarly plausible depressions and prevents a shallow ballast dip
        # from winning solely because it is encountered first.
        pos_term = cand["bottom_x"]
        prior_term = 0.0
        if prior_x is not None and not np.isnan(prior_x):
            prior_term = -abs(cand["bottom_x"] - float(prior_x))
        # Outer-reach fix: replace the outward reward with an outer-boundary
        # penalty so the detector prefers an inboard thalweg over a far
        # embankment-toe / surface candidate near the search edge.
        if USE_OUTER_REACH_FIX:
            outer_pos_contrib = 0.0
            margin = SIDE_SEARCH_MAX - cand["bottom_x"]
            edge_pen = (
                -EDGE_PENALTY_WEIGHT * (1.0 - max(0.0, margin) / EDGE_PENALTY_BAND_M)
                if margin < EDGE_PENALTY_BAND_M else 0.0
            )
        else:
            outer_pos_contrib = MAIN_DITCH_OUTER_POS_WEIGHT * pos_term
            edge_pen = 0.0
        return float(
            MAIN_DITCH_DEPTH_WEIGHT * depth_term
            + MAIN_DITCH_PROM_WEIGHT * prom_term
            + MAIN_DITCH_AREA_WEIGHT * np.sqrt(area_term)
            + outer_pos_contrib
            + edge_pen
            + MAIN_DITCH_PRIOR_WEIGHT * prior_term
        )

    for pk in peaks:
        # Search uses depth_ref_z (may be shoulder-relative),
        # but the reported depth / area is always relative to report_ref_z
        # (normally RUK) so the CSV stays consistent across passes.
        depth_search = float(depth_ref_z  - zk[pk])
        depth_report = float(report_ref_z - zk[pk])
        if depth_search < min_depth:
            continue

        outer_z = zk[pk + 1:] if pk + 1 < len(zk) else np.array([])
        if len(outer_z) < 1:
            continue
        # Proportional outer-rise: real ditches have outer banks that
        # recover to near-original elevation. An L-shape embankment toe
        # has < 10 cm rise past the descent, regardless of how "deep"
        # the descent itself is.
        outer_rise = float(outer_z.max() - zk[pk])
        required_outer_rise = max(SHAPE_OUTER_RISE_MIN,
                                  SHAPE_OUTER_RISE_FRAC * depth_search)
        if outer_rise < required_outer_rise:
            continue

        # Top-of-bank edge selection. Earlier versions used a fixed
        # 0.5 m edge-walk cap on each side of the bottom. That was useful
        # as a conservative detector support width, but it made the reported
        # width nearly constant. The current detector follows the ditch and
        # channel-morphometry literature instead: it searches for the two
        # bank tops from height recovery and slope break in the 1-D profile.
        top_width_desc = estimate_top_bank_width(
            xk, zk, pk, allow_local_edge_fallback=False
        )
        if top_width_desc["top_width_quality"] == "invalid":
            continue

        i0 = top_width_desc["top_i0"]
        i1 = top_width_desc["top_i1"]
        width = float(top_width_desc["top_width"])

        # Area is reported relative to report_ref_z.
        deficit_span = np.maximum(0.0, report_ref_z - zk[i0:i1 + 1])
        area = (
            float(np.trapz(deficit_span, xk[i0:i1 + 1]))
            if i1 > i0 else 0.0
        )

        if width < TOP_BANK_MIN_WIDTH or width > TOP_BANK_MAX_WIDTH:
            continue
        if area < DITCH_MIN_AREA:
            continue

        bottom_x = float(xk[pk])
        bottom_z = float(zk[pk])

        # inner slope (track side ->ditch bottom)
        inner_edge_x = float(xk[i0])
        inner_run = max(1e-6, bottom_x - inner_edge_x)
        inner_rise = float(zk[i0] - bottom_z)
        inner_slope_deg = float(np.degrees(np.arctan(inner_rise / inner_run)))

        # outer slope (ditch bottom ->outer edge)
        outer_edge_x = float(xk[i1])
        outer_run = max(1e-6, outer_edge_x - bottom_x)
        outer_rise = float(zk[i1] - bottom_z)
        outer_slope_deg = float(np.degrees(np.arctan(outer_rise / outer_run)))

        # Shape validation: reject non-ditch depressions
        if not (DITCH_INNER_SLOPE_MIN_DEG <= inner_slope_deg <= DITCH_INNER_SLOPE_MAX_DEG):
            continue
        if not (DITCH_OUTER_SLOPE_MIN_DEG <= outer_slope_deg <= DITCH_OUTER_SLOPE_MAX_DEG):
            continue
        # Conditional global eff-slope check
        # Only applied when the ditch bottom is > EFFECTIVE_SLOPE_MIN_BOTTOM_X
        # from the search boundary. Near-track ditches are preserved;
        # far-track ditches with a gentle overall descent (embankment-toe
        # profiles) are caught here even when pass1's find_peaks returns
        # a valid local minimum in the middle of the dip (so asymmetry
        # is near 0 and won't reject).
        bypass_eff_slope = (
            depth_search >= 2.5 * DITCH_MIN_DEPTH
            and area >= 2.0 * DITCH_MIN_AREA
            and outer_rise >= max(SHAPE_OUTER_RISE_MIN, SHAPE_OUTER_RISE_FRAC * depth_search)
        )
        if (not bypass_eff_slope) and (xk[pk] - xk[0]) > EFFECTIVE_SLOPE_MIN_BOTTOM_X:
            eff_run  = float(xk[pk] - xk[0])
            eff_rise = float(zk[0] - zk[pk])
            if eff_run > 0.5 and eff_rise > 0.0:
                eff_inner_slope = float(np.degrees(
                    np.arctan(eff_rise / eff_run)
                ))
                if eff_inner_slope < EFFECTIVE_INNER_SLOPE_MIN:
                    continue

        shape_desc = compute_shape_descriptors(
            xk, zk, i0, i1, pk, inner_slope_deg
        )

        # Reject extreme one-sidedness: |Knighton A1| > SLOPE_ASYMMETRY_MAX.
        # In embankment-toe-plus-terminal-dip profiles the ditch "bottom"
        # ends up at the inner or outer edge of the detected span, giving
        # asymmetry = +/-1. Real ditches have |A1| well under 0.5.
        asym_check = shape_desc.get("asymmetry", np.nan)
        if not np.isnan(asym_check) and abs(asym_check) > SLOPE_ASYMMETRY_MAX:
            continue

        # Attach pass source and confidence.
        prom_use = peak_prom.get(int(pk), DITCH_MIN_PROM)
        confidence = _compute_confidence(
            pass_source, prom_use, depth_report, width
        )

        candidate = {
            "depth": depth_report,
            "width": width,
            "top_width": top_width_desc["top_width"],
            "area": area,
            "bottom_x": bottom_x,
            "bottom_z": bottom_z,
            "span_x0": float(xk[i0]),
            "span_x1": float(xk[i1]),
            "top_span_x0": top_width_desc["top_span_x0"],
            "top_span_x1": top_width_desc["top_span_x1"],
            "top_width_quality": top_width_desc["top_width_quality"],
            "inner_slope_deg": inner_slope_deg,
            "outer_slope_deg": outer_slope_deg,
            "flatness": shape_desc["flatness"],
            "asymmetry": shape_desc["asymmetry"],
            "depth_deficit": shape_desc["depth_deficit"],
            "prominence": prom_use,
            "pass_source": pass_source,
            "confidence": confidence,
        }
        candidate["candidate_score"] = _main_ditch_score(candidate, pk)
        valid_candidates.append(candidate)

    if not valid_candidates:
        return [] if return_all else None
    valid_candidates.sort(key=lambda c: c["candidate_score"], reverse=True)
    if return_all:
        return valid_candidates
    if prior_x is not None and not np.isnan(prior_x) and prior_x >= OUTER_PATH_MIN_X:
        path_candidates = [
            c for c in valid_candidates
            if abs(c["bottom_x"] - float(prior_x)) <= OUTER_PATH_PRIOR_TOL_M
        ]
        if path_candidates:
            valid_candidates = path_candidates
            valid_candidates.sort(key=lambda c: c["candidate_score"], reverse=True)
    return valid_candidates[0]


def _try_prior_guided_outer_surface(xk, zk, report_ref_z, prior_x,
                                    pass_source="pass5_outer_prior_surface"):
    """
    Track an already established outer drainage path when dense bottom returns
    are missing.  This is a visible-surface fallback, not a claim that the
    physical ditch bottom is observed.
    """
    if prior_x is None or np.isnan(prior_x) or prior_x < OUTER_PATH_MIN_X:
        return None
    if len(xk) < 8 or np.isnan(report_ref_z):
        return None

    w = np.abs(xk - float(prior_x)) <= OUTER_PATH_PRIOR_TOL_M
    if w.sum() < 6:
        return None
    idxs = np.where(w)[0]
    z_win = zk[idxs]
    if not np.isfinite(z_win).any():
        return None
    pk = int(idxs[int(np.nanargmin(z_win))])
    if pk <= 1 or pk >= len(xk) - 2:
        return None

    bottom_x = float(xk[pk])
    bottom_z = float(zk[pk])
    depth_report = float(report_ref_z - bottom_z)
    if depth_report < OUTER_SURFACE_MIN_DEPTH:
        return None

    left = zk[max(0, pk - 20):pk]
    right = zk[pk + 1:min(len(zk), pk + 21)]
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return None
    inner_rise_local = float(np.nanpercentile(left, 80) - bottom_z)
    outer_rise_local = float(np.nanpercentile(right, 80) - bottom_z)
    if max(inner_rise_local, outer_rise_local) < OUTER_SURFACE_MIN_RISE:
        return None

    top_width_desc = estimate_top_bank_width(
        xk, zk, pk, allow_local_edge_fallback=True
    )
    if top_width_desc["top_width_quality"] == "invalid":
        return None
    i0 = top_width_desc["top_i0"]
    i1 = top_width_desc["top_i1"]
    width = float(top_width_desc["top_width"])
    if width < TOP_BANK_MIN_WIDTH or width > TOP_BANK_MAX_WIDTH:
        return None

    deficit_span = np.maximum(0.0, report_ref_z - zk[i0:i1 + 1])
    area = (
        float(np.trapz(deficit_span, xk[i0:i1 + 1]))
        if i1 > i0 else 0.0
    )
    if area < 0.5 * DITCH_MIN_AREA:
        return None

    inner_run = max(1e-6, bottom_x - float(xk[i0]))
    outer_run = max(1e-6, float(xk[i1]) - bottom_x)
    inner_slope_deg = float(np.degrees(np.arctan(max(float(zk[i0] - bottom_z), 0.0) / inner_run)))
    outer_slope_deg = float(np.degrees(np.arctan(max(float(zk[i1] - bottom_z), 0.0) / outer_run)))
    if inner_slope_deg < DITCH_INNER_SLOPE_MIN_DEG and outer_slope_deg < DITCH_OUTER_SLOPE_MIN_DEG:
        return None

    shape_desc = compute_shape_descriptors(
        xk, zk, i0, i1, pk, max(inner_slope_deg, DITCH_INNER_SLOPE_MIN_DEG)
    )
    prom_use = float(max(min(max(inner_rise_local, 0.0), max(outer_rise_local, 0.0)), DITCH_MIN_PROM))
    confidence = 0.55 * _compute_confidence(
        "pass3_toe", prom_use, depth_report, width
    )
    return {
        "depth": depth_report,
        "width": width,
        "top_width": top_width_desc["top_width"],
        "area": area,
        "bottom_x": bottom_x,
        "bottom_z": bottom_z,
        "span_x0": float(xk[i0]),
        "span_x1": float(xk[i1]),
        "top_span_x0": top_width_desc["top_span_x0"],
        "top_span_x1": top_width_desc["top_span_x1"],
        "top_width_quality": top_width_desc["top_width_quality"],
        "inner_slope_deg": inner_slope_deg,
        "outer_slope_deg": outer_slope_deg,
        "flatness": shape_desc["flatness"],
        "asymmetry": shape_desc["asymmetry"],
        "depth_deficit": shape_desc["depth_deficit"],
        "prominence": prom_use,
        "pass_source": pass_source,
        "confidence": float(np.clip(confidence, 0.0, 0.65)),
        "candidate_score": 0.0,
    }


def _try_toe_of_slope_ditch(xk, zk, shoulder_z, report_ref_z=None,
                            pass_source="pass3_toe"):
    """
    Detect a cutting drainage ditch at the toe of a rising slope.
    Works for monotonically rising profiles where find_peaks fails.
    Finds the transition point where slope changes from flat (<10 deg) to
    steep (>15 deg), which is the drainage collection point.

    Depth and area are reported relative to `report_ref_z`
    (normally the RUK z_ref), while the search gate still uses the
    shoulder-based CUTTING_DEPTH_FROM_SHOULDER.
    """
    if report_ref_z is None:
        report_ref_z = shoulder_z
    if len(xk) < 6:
        return None

    dz_dx = np.gradient(zk, xk)
    slopes_deg = np.degrees(np.arctan(dz_dx))

    # Find toe: last point where slope < 10 deg before it exceeds 15 deg
    toe_idx = None
    for i in range(1, len(slopes_deg)):
        if slopes_deg[i - 1] < 10.0 and slopes_deg[i] > 15.0:
            toe_idx = i
            break

    if toe_idx is None:
        return None

    # The "ditch bottom" is at or just before the toe
    bottom_idx = toe_idx
    # Look back a few bins for the actual minimum
    search_start = max(0, toe_idx - 5)
    local_min_idx = search_start + int(np.argmin(zk[search_start:toe_idx + 1]))
    bottom_idx = local_min_idx

    bottom_x = float(xk[bottom_idx])
    bottom_z = float(zk[bottom_idx])
    depth_search = float(shoulder_z    - bottom_z)  # gate uses shoulder
    depth_report = float(report_ref_z  - bottom_z)  # CSV uses RUK

    if depth_search < CUTTING_DEPTH_FROM_SHOULDER:
        return None

    top_width_desc = estimate_top_bank_width(
        xk, zk, bottom_idx, allow_local_edge_fallback=False
    )
    if top_width_desc["top_width_quality"] == "invalid":
        return None

    i0 = top_width_desc["top_i0"]
    i1 = top_width_desc["top_i1"]
    width = float(top_width_desc["top_width"])
    if width < TOP_BANK_MIN_WIDTH or width > TOP_BANK_MAX_WIDTH:
        return None

    # Slopes
    inner_run = max(1e-6, bottom_x - xk[0])
    inner_rise = float(zk[0] - bottom_z)  # may be negative (track higher)
    inner_slope_deg = float(np.degrees(np.arctan(abs(inner_rise) / inner_run)))

    outer_run = max(1e-6, float(xk[i1] - bottom_x))
    outer_rise = float(zk[i1] - bottom_z)
    outer_slope_deg = float(np.degrees(np.arctan(outer_rise / outer_run)))

    if outer_slope_deg < DITCH_OUTER_SLOPE_MIN_DEG:
        return None

    # Conditional global eff-slope check (same rationale as pass1).
    if (bottom_x - xk[0]) > EFFECTIVE_SLOPE_MIN_BOTTOM_X:
        if abs(inner_slope_deg) < EFFECTIVE_INNER_SLOPE_MIN:
            return None

    # Proportional outer-rise (rejects toe-of-embankment false positives
    # where the terrain flattens past the descent).
    required_rise = max(SHAPE_OUTER_RISE_MIN,
                        SHAPE_OUTER_RISE_FRAC * depth_search)
    if outer_rise < required_rise:
        return None

    # Area reported relative to report_ref_z.
    deficit = np.maximum(0.0, report_ref_z - zk[i0:i1 + 1])
    area = float(np.trapz(deficit, xk[i0:i1 + 1])) if i1 > i0 else 0.0

    shape_desc = compute_shape_descriptors(
        xk, zk, i0, i1, bottom_idx, inner_slope_deg
    )

    # Reject extreme one-sidedness (Knighton A1): catches toe-of-slope
    # false positives where the "ditch" bottom sits at the edge of the
    # detected span (embankment terminal-dip pattern).
    asym_check = shape_desc.get("asymmetry", np.nan)
    if not np.isnan(asym_check) and abs(asym_check) > SLOPE_ASYMMETRY_MAX:
        return None

    # Prominence proxy from the local outer rise
    prom_use = float(max(outer_rise, DITCH_MIN_PROM))
    confidence = _compute_confidence(
        pass_source, prom_use, depth_report, width
    )

    return {
        "depth": depth_report,
        "width": width,
        "top_width": top_width_desc["top_width"],
        "area": area,
        "bottom_x": bottom_x,
        "bottom_z": bottom_z,
        "span_x0": float(xk[i0]),
        "span_x1": float(xk[i1]),
        "top_span_x0": top_width_desc["top_span_x0"],
        "top_span_x1": top_width_desc["top_span_x1"],
        "top_width_quality": top_width_desc["top_width_quality"],
        "inner_slope_deg": inner_slope_deg,
        "outer_slope_deg": outer_slope_deg,
        "flatness": shape_desc["flatness"],
        "asymmetry": shape_desc["asymmetry"],
        "depth_deficit": shape_desc["depth_deficit"],
        "prominence": prom_use,
        "pass_source": pass_source,
        "confidence": confidence,
    }


def _try_gap_ditch(xk, zk, depth_ref_z, bin_w=0.10, pass_source="pass4_gap"):
    """
    Detect a water-filled ditch by finding a zone of sparse/low points
    flanked by higher terrain (Bailly et al. 2008 approach).
    Uses coarser binning (0.10m) to identify density drops.

    depth_ref_z here is already the RUK z_ref (see caller in
    compute_side_metrics_first_ditch), so no report_ref_z separation
    is needed.
    """
    if len(xk) < 10 or np.isnan(depth_ref_z):
        return None

    # Coarse binning to find density pattern
    x_bins = np.arange(xk.min(), xk.max() + bin_w, bin_w)
    n_bins = len(x_bins) - 1
    if n_bins < 5:
        return None

    bin_z_min = np.full(n_bins, np.nan)
    bin_cnt = np.zeros(n_bins, dtype=int)
    bin_x_mid = np.array([(x_bins[j] + x_bins[j+1]) / 2.0 for j in range(n_bins)])

    for j in range(n_bins):
        m = (xk >= x_bins[j]) & (xk < x_bins[j+1])
        bin_cnt[j] = int(m.sum())
        if m.sum() > 0:
            bin_z_min[j] = float(np.min(zk[m]))

    # Find the lowest elevation zone (candidate ditch bottom)
    valid = ~np.isnan(bin_z_min)
    if valid.sum() < 3:
        return None

    # Median density of populated bins
    populated = bin_cnt[valid]
    if len(populated) < 3:
        return None
    med_cnt = float(np.median(populated[populated > 0])) if (populated > 0).sum() > 0 else 1.0

    # Find the depression: lowest bin_z_min in the middle portion
    # Exclude first and last 2 bins to ensure flanking data exists
    search_sl = slice(2, n_bins - 2)
    search_z = bin_z_min[search_sl].copy()
    search_x = bin_x_mid[search_sl]
    search_cnt = bin_cnt[search_sl]

    # Fill NaN in search zone with interpolation for minimum finding
    sv = ~np.isnan(search_z)
    if sv.sum() < 2:
        return None
    search_z[~sv] = np.interp(search_x[~sv], search_x[sv], search_z[sv])

    bottom_local_idx = int(np.argmin(search_z))
    bottom_idx = bottom_local_idx + 2  # offset for the slice
    bottom_x = float(bin_x_mid[bottom_idx])
    bottom_z = float(search_z[bottom_local_idx])

    depth = float(depth_ref_z - bottom_z)
    if depth < DITCH_MIN_DEPTH:
        return None

    # Check flanking terrain rises on both sides
    left_flank = bin_z_min[:bottom_idx]
    right_flank = bin_z_min[bottom_idx+1:]

    left_valid = left_flank[~np.isnan(left_flank)]
    right_valid = right_flank[~np.isnan(right_flank)]

    if len(left_valid) < 1 or len(right_valid) < 1:
        return None

    left_max = float(np.max(left_valid))
    right_max = float(np.max(right_valid))

    inner_rise = left_max - bottom_z  # track-side rise
    outer_rise = right_max - bottom_z  # outer-side rise

    # Proportional outer-rise (rejects L-shape embankment toes that happen
    # to have a few cm of noise past the descent).
    required_rise = max(SHAPE_OUTER_RISE_MIN,
                        SHAPE_OUTER_RISE_FRAC * depth)
    if inner_rise < required_rise or outer_rise < required_rise:
        return None

    bin_z_for_shape = np.where(np.isnan(bin_z_min), bottom_z, bin_z_min)
    top_width_desc = estimate_top_bank_width(
        bin_x_mid, bin_z_for_shape, bottom_idx,
        allow_local_edge_fallback=False,
    )
    if top_width_desc["top_width_quality"] == "invalid":
        return None

    i0 = top_width_desc["top_i0"]
    i1 = top_width_desc["top_i1"]
    width = float(top_width_desc["top_width"])
    if width < TOP_BANK_MIN_WIDTH or width > TOP_BANK_MAX_WIDTH:
        return None

    # Slopes
    inner_run = max(1e-6, bottom_x - float(bin_x_mid[i0]))
    inner_slope_deg = float(np.degrees(np.arctan(inner_rise / inner_run)))
    outer_run = max(1e-6, float(bin_x_mid[i1]) - bottom_x)
    outer_slope_deg = float(np.degrees(np.arctan(outer_rise / outer_run)))

    # Slope validity (same two-sided bounds as Pass 1).
    if not (DITCH_INNER_SLOPE_MIN_DEG <= inner_slope_deg <= DITCH_INNER_SLOPE_MAX_DEG):
        return None
    if not (DITCH_OUTER_SLOPE_MIN_DEG <= outer_slope_deg <= DITCH_OUTER_SLOPE_MAX_DEG):
        return None

    # Effective (global) inner-slope check - same conditional as Pass 1.
    # Conditional global eff-slope check (same rationale as pass1).
    if (bottom_x - bin_x_mid[0]) > EFFECTIVE_SLOPE_MIN_BOTTOM_X:
        first_valid = next(
            (i for i, v in enumerate(bin_z_min) if not np.isnan(v)), None
        )
        if first_valid is not None and first_valid < bottom_idx:
            run_to_bottom = float(bin_x_mid[bottom_idx] - bin_x_mid[first_valid])
            if run_to_bottom > 0.5:
                eff_rise = float(bin_z_min[first_valid] - bottom_z)
                if eff_rise > 0.0:
                    eff_inner_slope = float(np.degrees(
                        np.arctan(eff_rise / run_to_bottom)
                    ))
                    if eff_inner_slope < EFFECTIVE_INNER_SLOPE_MIN:
                        return None

    # Area
    deficit = np.maximum(0.0, depth_ref_z - search_z)
    area = float(np.trapz(deficit, search_x))
    if area < DITCH_MIN_AREA:
        return None

    # Shape descriptors from the coarse bin profile (NaN-robust)
    shape_desc = compute_shape_descriptors(
        bin_x_mid, bin_z_for_shape, i0, i1, bottom_idx, inner_slope_deg
    )

    # Reject extreme one-sidedness (Knighton A1 asymmetry).
    asym_check = shape_desc.get("asymmetry", np.nan)
    if not np.isnan(asym_check) and abs(asym_check) > SLOPE_ASYMMETRY_MAX:
        return None

    # Prominence proxy = the smaller of the two flank rises
    prom_use = float(max(min(inner_rise, outer_rise), DITCH_MIN_PROM))
    confidence = _compute_confidence(
        pass_source, prom_use, float(depth), float(width)
    )

    return {
        "depth": depth,
        "width": width,
        "top_width": top_width_desc["top_width"],
        "area": area,
        "bottom_x": bottom_x,
        "bottom_z": bottom_z,
        "span_x0": float(bin_x_mid[i0]),
        "span_x1": float(bin_x_mid[i1]),
        "top_span_x0": top_width_desc["top_span_x0"],
        "top_span_x1": top_width_desc["top_span_x1"],
        "top_width_quality": top_width_desc["top_width_quality"],
        "inner_slope_deg": inner_slope_deg,
        "outer_slope_deg": outer_slope_deg,
        "flatness": shape_desc["flatness"],
        "asymmetry": shape_desc["asymmetry"],
        "depth_deficit": shape_desc["depth_deficit"],
        "prominence": prom_use,
        "pass_source": pass_source,
        "confidence": confidence,
    }


def compute_side_metrics_first_ditch(x, z, z_ref, prior_x=None):
    """
    Analyse one side of a cross-section for the nearest valid ditch.
    x = outward distance from centreline (positive, near to far).

    Multi-pass detection strategy:
      Pass 1: Standard peak-based search in full zone (0.5-15.0 m)
      Pass 2: Shoulder-referenced search (for cuttings/embankments)
      Pass 3: Toe-of-slope detection (full zone, any terrain type)
      Pass 4: Data-gap detection (Bailly et al. 2008, water-filled ditches)

    prior_x, if given, is the rolling median of recent confirmed
    bottom_x values for this side  - used as a spatial-continuity
    prior in Pass 1 and Pass 2.
    """
    # If z_ref is NaN, use local shoulder estimate instead of skipping
    use_zref = z_ref
    if np.isnan(z_ref) and len(x) >= 8:
        # Estimate z_ref from the innermost points (near track)
        inner = z[x < 1.0]
        if len(inner) >= 3:
            use_zref = float(np.percentile(inner, 75))

    if len(x) < 8 or np.isnan(use_zref):
        return _non_ditch_fallback(x, z)

    # Full search zone
    keep = (x >= SIDE_SEARCH_MIN) & (x <= SIDE_SEARCH_MAX)
    xk = x[keep]
    zk = z[keep]
    pass1_candidates = []
    if len(xk) >= 8:
        pass1_candidates = _try_find_ditch(
            xk, zk, use_zref, DITCH_MIN_DEPTH,
            prior_x=None,
            report_ref_z=use_zref,
            pass_source="pass1",
            return_all=True,
        )
    candidate_json, candidate_count = _pack_candidates_for_json(pass1_candidates)

    def _outer_prior_override(result):
        if result is None:
            return None
        if prior_x is None or np.isnan(prior_x) or prior_x < OUTER_PATH_MIN_X:
            return result
        if abs(float(result.get("bottom_x", np.nan)) - float(prior_x)) <= OUTER_PATH_PRIOR_TOL_M:
            return result
        outer_result = _try_prior_guided_outer_surface(
            xk, zk, use_zref, prior_x
        )
        return outer_result if outer_result is not None else result

    # Pass 1: Peak-based ditch search
    result = None
    if len(xk) >= 8:
        result = _try_find_ditch(xk, zk, use_zref, DITCH_MIN_DEPTH,
                                 prior_x=prior_x,
                                 report_ref_z=use_zref,
                                 pass_source="pass1")

    if result is not None:
        result = _outer_prior_override(result)
        result["candidate_json"] = candidate_json
        result["candidate_count"] = candidate_count
        return {
            "side_type": "DITCH",
            "relief": float(np.nanmax(zk) - np.nanmin(zk)),
            "dvci": np.nan,
            "ditch_exists": 1,
            **result,
        }

    # Pass 2: Shoulder-referenced search
    # For cuttings or embankments: use outer-half P75 as depth ref
    if len(xk) >= 6:
        outer_half = xk > (xk.min() + xk.max()) / 2
        if outer_half.sum() >= 3:
            shoulder_z = float(np.percentile(zk[outer_half], 75))
            rise = shoulder_z - zk[0]
            drop = zk[0] - shoulder_z

            # Cutting: terrain rises outward
            if rise >= CUTTING_MIN_RISE:
                result2 = _try_find_ditch(
                    xk, zk, shoulder_z, CUTTING_DEPTH_FROM_SHOULDER,
                    prior_x=prior_x,
                    report_ref_z=use_zref,
                    pass_source="pass2_cutting")
                if result2 is not None:
                    result2 = _outer_prior_override(result2)
                    result2["candidate_json"] = candidate_json
                    result2["candidate_count"] = candidate_count
                    return {
                        "side_type": "CUTTING_DITCH",
                        "relief": float(np.nanmax(zk) - np.nanmin(zk)),
                        "dvci": np.nan,
                        "ditch_exists": 1,
                        **result2,
                    }

            # Embankment: terrain drops outward - ditch at the bottom
            if drop >= EMBANKMENT_MIN_DROP:
                # Use the local minimum region as depth reference
                local_min_z = float(np.min(zk))
                local_max_z = float(np.max(zk))
                emb_depth_ref = float(np.percentile(zk[~outer_half], 25)) \
                    if (~outer_half).sum() >= 3 else use_zref
                result2b = _try_find_ditch(
                    xk, zk, emb_depth_ref, CUTTING_DEPTH_FROM_SHOULDER,
                    prior_x=prior_x,
                    report_ref_z=use_zref,
                    pass_source="pass2_embankment")
                if result2b is not None:
                    result2b = _outer_prior_override(result2b)
                    result2b["candidate_json"] = candidate_json
                    result2b["candidate_count"] = candidate_count
                    return {
                        "side_type": "DITCH",
                        "relief": float(local_max_z - local_min_z),
                        "dvci": np.nan,
                        "ditch_exists": 1,
                        **result2b,
                    }

    # Pass 3: Toe-of-slope detection (any terrain)
    # Not just for cuttings - also for embankments where ditch
    # sits at slope transition.
    if len(xk) >= 6:
        outer_half = xk > (xk.min() + xk.max()) / 2
        if outer_half.sum() >= 3:
            shoulder_z = float(np.percentile(zk[outer_half], 75))
            result3 = _try_toe_of_slope_ditch(
                xk, zk, shoulder_z,
                report_ref_z=use_zref,
                pass_source="pass3_toe")
            if result3 is not None:
                result3 = _outer_prior_override(result3)
                result3["candidate_json"] = candidate_json
                result3["candidate_count"] = candidate_count
                return {
                    "side_type": "CUTTING_DITCH",
                    "relief": float(np.nanmax(zk) - np.nanmin(zk)),
                    "dvci": np.nan,
                    "ditch_exists": 1,
                    **result3,
                }

    # Pass 4: Data-gap detection (Bailly et al. 2008)
    if len(xk) >= 8:
        result_gap = _try_gap_ditch(xk, zk, use_zref)
        if result_gap is not None:
            result_gap = _outer_prior_override(result_gap)
            result_gap["candidate_json"] = candidate_json
            result_gap["candidate_count"] = candidate_count
            return {
                "side_type": "DITCH",
                "relief": float(np.nanmax(zk) - np.nanmin(zk)),
                "dvci": np.nan,
                "ditch_exists": 1,
                **result_gap,
            }

    result_outer_prior = _try_prior_guided_outer_surface(
        xk, zk, use_zref, prior_x
    )
    if result_outer_prior is not None:
        result_outer_prior["candidate_json"] = candidate_json
        result_outer_prior["candidate_count"] = candidate_count
        return {
            "side_type": "OUTER_VISIBLE_SURFACE_DITCH",
            "relief": float(np.nanmax(zk) - np.nanmin(zk)),
            "dvci": np.nan,
            "ditch_exists": 1,
            **result_outer_prior,
        }

    # Fallback: no ditch found
    fb_x = xk if len(xk) >= 4 else x
    fb_z = zk if len(xk) >= 4 else z
    return _non_ditch_fallback(fb_x, fb_z)


def compute_dvci_for_span(v_pts, exg_pts, side_sign, x0, x1):
    if np.isnan(x0) or np.isnan(x1) or x1 <= x0:
        return np.nan
    x_pts = -v_pts if side_sign < 0 else v_pts
    m = (x_pts >= x0) & (x_pts <= x1)
    if m.sum() < 5:
        return np.nan
    return float((exg_pts[m] > EXG_VEG_THRESHOLD).mean())


def get_ditch_presence(left_exists, right_exists):
    if left_exists and right_exists:
        return "BOTH"
    if left_exists:
        return "LEFT_ONLY"
    if right_exists:
        return "RIGHT_ONLY"
    return "NONE"


def classify_content(dvci, water_flag):
    """
    Classify ditch content into one of the five Roelens (2018) classes
    based on vegetation density (DVCI proxy) and water presence.

    Inputs
    ------
    dvci       : fraction of points with ExG > EXG_VEG_THRESHOLD in
                 the ditch span. NaN if insufficient points.
    water_flag : bool, from detect_water_in_ditch() (low point
                 density + low intensity, indicating NIR-absorbing water).

    Returns (str)
    -------------
    DRY_CLEAN        : no water, DVCI < CONTENT_DVCI_PARTIAL
    DRY_VEGETATED    : no water, DVCI >= CONTENT_DVCI_PARTIAL
    WET_CLEAR        : water present, DVCI < CONTENT_DVCI_PARTIAL
    WET_PARTIAL_VEG  : water present, CONTENT_DVCI_PARTIAL <= DVCI < CONTENT_DVCI_DENSE
    WET_DENSE_VEG    : water present, DVCI >= CONTENT_DVCI_DENSE
    UNKNOWN          : insufficient data

    Reference: Roelens et al., "Extracting cross sections and water levels
    of vegetated ditches from LiDAR point clouds", 2018. The 5-class
    scheme is adapted; the variance test they use to separate dry-veg
    from wet-dense-veg is replaced here by our independent water_flag
    (low density + low intensity) which performs the same role.
    """
    if np.isnan(dvci):
        # no DVCI data - fall back to wet/dry-only bucket
        return "WET_CLEAR" if water_flag else "DRY_CLEAN"

    if water_flag:
        if dvci < CONTENT_DVCI_PARTIAL:
            return "WET_CLEAR"
        if dvci < CONTENT_DVCI_DENSE:
            return "WET_PARTIAL_VEG"
        return "WET_DENSE_VEG"
    else:
        if dvci < CONTENT_DVCI_PARTIAL:
            return "DRY_CLEAN"
        return "DRY_VEGETATED"


def diagnose_issues(shape_class, content_class, depth_m,
                    inner_slope_deg, outer_slope_deg, asymmetry,
                    ditch_exists, zref_valid):
    """
    Multi-label ditch health diagnosis aligned with:
      - Trafikverket TDOK 2015:0155 Section 10.13 (shallow / overgrown dike)
      - Rail Baltica Design Guidelines Part 2 (2025)  - drainage geometry
      - Roelens et al. 2018  - vegetation + water classification from LiDAR
      - Knighton 1981  - cross-section asymmetry as deformation indicator
      - Chen et al. 2024 (Frontiers Earth Sci.)  - railway protective
        facility defect detection

    Returns
    -------
    list of issue tags, one or more of:
        HEALTHY          - no issues
        SHALLOW          - depth < TDOK_DEPTH_SUFFICIENT_M (1.0 m under RUK)
        SILTED           - shape_class == FLAT_SILTED in a non-water-suspect
                          ditch (sediment build-up / flat-bottom evidence)
        WATER_MASKED_SHAPE
                         - water is detected, so the true V/U/flat bottom
                          shape is hidden and cannot be classified
        OVERGROWN        - content_class contains VEG (TDOK overgrown dike)
        STAGNANT_WATER   - water present AND drainage obstructed:
                          shallow, OR silted bottom, OR dense vegetation
                          blocking flow.  Extended from the shallow-only
                          TDOK Section 10.13 trigger to cover the flow-obstruction
                          cases described in Roelens 2018 Section 3.2.
        SLOPE_ISSUE      - inner slope outside [SLOPE_ANOMALY_INNER_MIN,
                          SLOPE_ANOMALY_INNER_MAX] OR outer slope outside
                          [SLOPE_ANOMALY_OUTER_MIN, SLOPE_ANOMALY_OUTER_MAX]
                          OR |asymmetry| > SLOPE_ASYMMETRY_MAX.  Captures
                          collapse (over-steepened), erosion (under-sloped),
                          and one-sided wall failure (asymmetric).
        NA               - zref invalid, no ditch, or missing depth
    """
    # Guard: if we can't measure, we can't diagnose
    if (not zref_valid) or (ditch_exists == 0) or np.isnan(depth_m):
        return ["NA"]

    issues = []

    # SHALLOW (TDOK Section 10.13 shallow dike)
    if depth_m < TDOK_DEPTH_SUFFICIENT_M:
        issues.append("SHALLOW")

    water_present = content_class in (
        "WET_CLEAR", "WET_PARTIAL_VEG", "WET_DENSE_VEG"
    )

    # SILTED / WATER-MASKED SHAPE
    # A flat lower envelope is direct silting evidence only when the ditch is
    # not water-suspect. With water, the surface can be the water level or a
    # gap-filled reconstruction rather than the physical ditch bottom.
    if shape_class == "UNKNOWN_WATER_MASKED" or water_present:
        issues.append("WATER_MASKED_SHAPE")
    elif shape_class == "FLAT_SILTED":
        issues.append("SILTED")

    # OVERGROWN (TDOK Section 10.13 overgrown dike)
    overgrown = content_class in (
        "DRY_VEGETATED", "WET_PARTIAL_VEG", "WET_DENSE_VEG"
    )
    if overgrown:
        issues.append("OVERGROWN")

    # STAGNANT_WATER (expanded flow-obstruction logic)
    # Water alone is not a problem - railway side ditches are meant to
    # carry water. A problem exists when water is present AND cannot
    # drain. Three flow-obstruction causes (Roelens 2018 Section 3.2, TDOK
    # Section 10.13):
    # (a) insufficient longitudinal/vertical gradient gives SHALLOW
    # (b) sediment fill raising the flowline gives SILTED bottom
    # (c) dense vegetation physically blocking flow gives WET_DENSE_VEG /
    # WET_PARTIAL_VEG
    if water_present:
        cannot_drain = False
        if depth_m < TDOK_DEPTH_SUFFICIENT_M:
            cannot_drain = True                     # cause (a)
        if "SILTED" in issues:
            cannot_drain = True                     # cause (b)
        if content_class in ("WET_PARTIAL_VEG", "WET_DENSE_VEG"):
            cannot_drain = True                     # cause (c)
        if cannot_drain:
            issues.append("STAGNANT_WATER")

    # SLOPE_ISSUE (slope anomaly)
    # Nominal railway side-ditch slope is 1:2 to 1:1 (Rail Baltica 2025).
    # Tolerances below allow natural erosion/sedimentation scatter.
    # Three independent triggers (any ->SLOPE_ISSUE):
    # (i) inner slope outside [SLOPE_ANOMALY_INNER_MIN, MAX]:
    # < 18 deg = sediment-flattened / eroded bank;
    # > 60 deg = collapsed / over-steepened wall.
    # (ii) outer slope outside [SLOPE_ANOMALY_OUTER_MIN, MAX]:
    # same rationale for the far bank.
    # (iii) |asymmetry| > 0.5 (Knighton A1): one side has failed
    # asymmetrically (typical of slope-wash or wall collapse).
    slope_issue = False
    if not np.isnan(inner_slope_deg):
        if inner_slope_deg < SLOPE_ANOMALY_INNER_MIN or \
           inner_slope_deg > SLOPE_ANOMALY_INNER_MAX:
            slope_issue = True
    if not np.isnan(outer_slope_deg):
        if outer_slope_deg < SLOPE_ANOMALY_OUTER_MIN or \
           outer_slope_deg > SLOPE_ANOMALY_OUTER_MAX:
            slope_issue = True
    if not np.isnan(asymmetry) and abs(asymmetry) > SLOPE_ASYMMETRY_MAX:
        slope_issue = True
    if slope_issue:
        issues.append("SLOPE_ISSUE")

    if not issues:
        issues.append("HEALTHY")

    return issues


# TDOK 2015:0155 Appendix 2 records ditch depth in bands including
# <0.5 m, 0.5-1.0 m, 1.0-1.5 m and >1.5 m. The <0.5 m band is used here
# only as a severity descriptor inside a multi-evidence rule. It is not a
# standalone V or M priority threshold.
TDOK_SEVERE_DEPTH_BAND_M = 0.50
PRIORITY_SLOPE_RUN_MIN = 5
PRIORITY_V_RUN_MIN = 5


def priority_evidence_groups(issues, depth_m):
    """Return the physical evidence groups used by the priority mapper."""
    s = set(issues or [])
    has_depth = depth_m is not None and not np.isnan(depth_m)
    severe_depth_band = has_depth and depth_m < TDOK_SEVERE_DEPTH_BAND_M

    groups = set()
    if "SHALLOW" in s:
        groups.add("depth")
    if "SILTED" in s:
        groups.add("shape")
    if "OVERGROWN" in s:
        groups.add("content")
    if "STAGNANT_WATER" in s:
        groups.add("water")
    if "SLOPE_ISSUE" in s:
        groups.add("slope")

    compound_hydraulic_blockage = (
        "STAGNANT_WATER" in s
        and bool(s & {"SILTED", "OVERGROWN", "SLOPE_ISSUE"})
    )
    deformation_plus_fill = (
        "SLOPE_ISSUE" in s
        and bool(s & {"SILTED", "OVERGROWN"})
    )
    strong_compound_case = (
        severe_depth_band
        and compound_hydraulic_blockage
        and bool(s & {"SLOPE_ISSUE", "SILTED"})
        and len(groups) >= 4
    )

    return {
        "groups": groups,
        "severe_depth_band": severe_depth_band,
        "compound_hydraulic_blockage": compound_hydraulic_blockage,
        "deformation_plus_fill": deformation_plus_fill,
        "strong_compound_case": strong_compound_case,
    }


def map_to_tdok_priority(issues, depth_m):
    """
    Map the five evidence groups produced by diagnose_issues() to a TDOK-style
    priority suggestion.

    The mapping avoids an unsupported single-depth rule for V. TDOK 2015:0155
    Section 10.13 supports SHALLOW when ditch depth is less than 1.0 m below
    RUK, and Appendix 2 provides descriptive depth bands, including <0.5 m.
    The automated classifier therefore treats depth as one evidence group
    together with shape, content, water and slope:

      Ö  no detected functional issue
      Å  one ordinary issue, such as SHALLOW, SILTED or OVERGROWN
      M  compound functional deficiency across several evidence groups
      V  sustained strongest compound case after the longitudinal post-filter.
    """
    if not issues or issues == ["NA"]:
        return "NA"

    if issues == ["HEALTHY"]:
        return "Ö"

    s = set(issues)
    evidence = priority_evidence_groups(issues, depth_m)
    evidence_groups = evidence["groups"]
    severe_depth_band = evidence["severe_depth_band"]
    compound_hydraulic_blockage = evidence["compound_hydraulic_blockage"]
    deformation_plus_fill = evidence["deformation_plus_fill"]
    strong_compound_case = evidence["strong_compound_case"]

    # A single station is not enough for V. The strongest local compound case
    # is kept at M here and can only become V if the longitudinal post-filter
    # finds a sustained run of the same condition.
    if strong_compound_case:
        return "M"

    # M: serious compound drainage deficiency. A severe depth band is enough
    # for M only when at least one additional issue is present.
    if severe_depth_band and len(evidence_groups) >= 2:
        return "M"
    if compound_hydraulic_blockage or deformation_plus_fill:
        return "M"
    if len(evidence_groups) >= 3:
        return "M"

    # TDOK §10.13 explicit: SHALLOW, SILTED, OVERGROWN -> Å.
    # SLOPE_ISSUE alone is also kept at Å here; sustained slope problems
    # are promoted later by a run-length rule.
    if s & {"SHALLOW", "SILTED", "OVERGROWN", "STAGNANT_WATER", "SLOPE_ISSUE"}:
        return "Å"

    return "Ö"


def select_example_ditch_sections(df_1m, n=3):
    has_ditch = (
        ((df_1m["left_ditch_exists"] == 1) | (df_1m["right_ditch_exists"] == 1))
        & (df_1m["zref_valid"] == 1)
    )
    d = df_1m[has_ditch].copy()

    if len(d) >= n:
        d["primary_depth"] = np.nanmax(
            np.column_stack([d["left_depth"].values, d["right_depth"].values]),
            axis=1
        )
        d = d[
            (d["primary_depth"] >= DEPTH_VALID_MIN) &
            (d["primary_depth"] <= DEPTH_VALID_MAX)
        ]

    if len(d) >= n:
        targets = [np.percentile(d["primary_depth"], p) for p in [25, 50, 75]]
        picked, used = [], set()
        for t in targets:
            idx = int(np.argmin(np.abs(d["primary_depth"].values - t)))
            row = d.iloc[idx]
            if row.name not in used:
                picked.append(float(row["s"]))
                used.add(row.name)
        if len(picked) == n:
            return picked

    valid = df_1m[
        (df_1m["zref_valid"] == 1) &
        ((df_1m["left_ditch_exists"] == 1) | (df_1m["right_ditch_exists"] == 1))
    ]
    if len(valid) >= n:
        return [float(np.percentile(valid["s"], p)) for p in [25, 50, 75]]

    return list(df_1m["s"].head(n).values)


def resolve_reference_columns(ref_df):
    t_col = "T_center_refined" if "T_center_refined" in ref_df.columns else "T_center"
    d_col = "dT_dS_refined" if "dT_dS_refined" in ref_df.columns else "dT_dS"
    zref_col = "Z_ref_smooth" if "Z_ref_smooth" in ref_df.columns else "Z_ref"
    left_prior_col = "T_left_prior_refined" if "T_left_prior_refined" in ref_df.columns else "T_left_prior"
    right_prior_col = "T_right_prior_refined" if "T_right_prior_refined" in ref_df.columns else "T_right_prior"
    return t_col, d_col, zref_col, left_prior_col, right_prior_col


def build_section_use_flag(ref_df):
    if "section_use_flag" in ref_df.columns:
        return ref_df["section_use_flag"].astype(str)
    if "usable_for_ditch" in ref_df.columns:
        out = np.where(ref_df["usable_for_ditch"].fillna(0).astype(int) == 1, "use", "skip")
        return pd.Series(out, index=ref_df.index)
    out = np.where(ref_df["zref_valid"].fillna(0).astype(int) == 1, "use", "skip")
    return pd.Series(out, index=ref_df.index)


# =========================================================
# MAIN
# =========================================================
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("STEP 07 - Ditch analysis (morphological envelope + water detection)")
print("=" * 60)

# Load Step 3 metadata
if not os.path.exists(STEP03_META):
    raise FileNotFoundError(f"Missing Step 3 metadata: {STEP03_META}")
with open(STEP03_META, "r", encoding="utf-8") as f:
    meta3 = json.load(f)

angle_deg = float(meta3["pca_track_azimuth_deg"])
angle_rad = np.radians(angle_deg)
mean_x = float(meta3["mean_x_local"])
mean_y = float(meta3["mean_y_local"])

print(f"Loaded Step 3 metadata: {STEP03_META}")
print(f"  PCA azimuth: {angle_deg:.2f} deg")
print(f"  Mean XY local: ({mean_x:.2f}, {mean_y:.2f})")

# Read LAZ
# Memory-conservative loading: we have ~60M points per tile, so
# each float64 array is ~460 MB. Cast to float32 on ingest (still
# sub-mm precision after the mean-subtraction) to halve peak RAM.
print("\nReading LAZ...")
las = laspy.read(LAZ_FILE)

# Keep x,y as float64 just long enough to compute the min-subtraction
# offset (needed for numerical stability at UTM scale), then drop.
_x64 = np.asarray(las.x)
_y64 = np.asarray(las.y)
x0 = float(_x64.min()); y0 = float(_y64.min())
x = (_x64 - x0).astype(np.float32); del _x64
y = (_y64 - y0).astype(np.float32); del _y64
z = np.asarray(las.z, dtype=np.float32)
intensity = np.asarray(las.intensity)

if not all(hasattr(las, ch) for ch in ["red", "green", "blue"]):
    raise RuntimeError("RGB is required for ExG vegetation coverage.")

import gc

# Read RGB and compute ExG immediately, freeing intermediates
R_raw = np.asarray(las.red, dtype=np.float32)
G_raw = np.asarray(las.green, dtype=np.float32)
B_raw = np.asarray(las.blue, dtype=np.float32)

if R_raw.max() > 256:
    R_raw /= 256.0
    G_raw /= 256.0
    B_raw /= 256.0
    print("  RGB: 16-bit -> divided by 256")

exg = (2.0 * G_raw - R_raw - B_raw).astype(np.float32)
del R_raw, G_raw, B_raw
gc.collect()

# intensity to 8-bit scale for water detection thresholds
if intensity.max() > 4096:
    I_scaled = (intensity / 256.0).astype(np.float32)
    print("  Intensity: 16-bit -> scaled /256 for water detection")
else:
    I_scaled = intensity.astype(np.float32)
del intensity
gc.collect()

n_total = len(x)
print(f"  Total points: {fmt_n(n_total)}")

# Ground mask
z_floor = float(np.percentile(z, 1))
gnd = (z >= z_floor) & (z <= z_floor + GROUND_Z_RANGE)
print(f"  Ground layer: {fmt_n(gnd.sum())} pts ({gnd.mean()*100:.1f}%)")

# Formal ST from Step 3
# x, y already carry the x0/y0 subtraction and are float32.
# Compute S,T in float32 and immediately subset to the ground mask
# to release the full-cloud arrays as early as possible.
S_all, T_all = to_ST(x, y, mean_x, mean_y, angle_rad)
S_all = S_all.astype(np.float32, copy=False)
T_all = T_all.astype(np.float32, copy=False)
del x, y
gc.collect()

Sg = S_all[gnd]
Tg = T_all[gnd]
del S_all, T_all

Zg = z[gnd]
del z
gc.collect()
Ig = I_scaled[gnd]
del I_scaled
ExGg = exg[gnd]
del exg, gnd
order_g = np.argsort(Sg)
Sg = Sg[order_g]
Tg = Tg[order_g]
Zg = Zg[order_g]
Ig = Ig[order_g]
ExGg = ExGg[order_g]
del order_g

# Read Step 8 reference CSV
print("\nReading rail reference framework...")
if not os.path.exists(REFERENCE_CSV):
    raise FileNotFoundError(f"Missing Step 8 reference CSV: {REFERENCE_CSV}")

ref = pd.read_csv(REFERENCE_CSV)
ref = ref.sort_values("s").reset_index(drop=True)

debug_s_min = os.environ.get("LIDAR_DEBUG_S_MIN")
debug_s_max = os.environ.get("LIDAR_DEBUG_S_MAX")
if debug_s_min is not None or debug_s_max is not None:
    lo = float(debug_s_min) if debug_s_min is not None else -np.inf
    hi = float(debug_s_max) if debug_s_max is not None else np.inf
    before_n = len(ref)
    ref = ref[(ref["s"] >= lo) & (ref["s"] <= hi)].reset_index(drop=True)
    print(f"[DEBUG] Restricted reference stations by S range {lo:.1f}..{hi:.1f}: "
          f"{before_n} -> {len(ref)}")

required_cols = ["s", "gauge_prior", "half_gauge_prior", "Z_ref", "zref_valid"]
for col in required_cols:
    if col not in ref.columns:
        raise RuntimeError(f"Reference CSV missing column: {col}")

t_col, d_col, zref_col, left_prior_col, right_prior_col = resolve_reference_columns(ref)
for col in [t_col, d_col, zref_col]:
    if col not in ref.columns:
        raise RuntimeError(f"Reference CSV missing required resolved column: {col}")

ref["section_use_flag"] = build_section_use_flag(ref)

print(f"  Reference sections: {len(ref)}")
print(f"  zref_valid:         {(ref['zref_valid'] == 1).sum()}")
print(f"  Using centreline:   {t_col}")
print(f"  Using dT/dS:        {d_col}")
print(f"  Using Z_ref:        {zref_col}")

# Per-metre cross-section analysis
print("\nRunning per-metre cross-section analysis...")
print(f"  Envelope: morphological opening (kernel={MORPH_KERNEL_SIZE} bins = {MORPH_KERNEL_SIZE*ENV_BIN_W:.2f}m)")
print(f"  Water detection: density ratio < {WATER_DENSITY_RATIO}, "
      f"min contiguous = {WATER_MIN_CONTIGUOUS_BINS} bins")
print(f"  TDOK 2015:0155 §10.13: SHALLOW if depth < {TDOK_DEPTH_SUFFICIENT_M}m below RUK, "
      f"target >= {TDOK_DEPTH_TARGET_M}m")

rows = []

# Rolling priors for spatial-continuity detection
# Keep the most recent DITCH_PRIOR_WINDOW confirmed bottom_x values
# per side. Their median is passed into the detection routine so
# candidate peaks are sorted by closeness-to-prior rather than by
# distance-from-centreline. This eliminates the prominence-flip
# oscillation between two coexisting real peaks.
recent_left_bx  = []   # FIFO of last confirmed left bottom_x values
recent_right_bx = []   # same for right

def _prior_median(buf):
    """Median of the rolling buffer, or None if empty."""
    return float(np.median(buf)) if len(buf) else None

def _push_prior(buf, bx):
    buf.append(float(bx))
    if len(buf) > DITCH_PRIOR_WINDOW:
        buf.pop(0)


for idx_ref, row in ref.iterrows():
    s0 = float(row["s"])
    t0 = float(row[t_col])
    d0 = float(row[d_col])
    z_ref = float(row[zref_col]) if not pd.isna(row[zref_col]) else np.nan
    zref_valid = int(row["zref_valid"])
    half_gauge = float(row["half_gauge_prior"])
    use_flag = str(row["section_use_flag"])

    if idx_ref % 100 == 0:
        print(f"  Progress: {idx_ref}/{len(ref)}  S={s0:.0f}m", end="\r")

    i0, i1 = sorted_slice_bounds(
        Sg,
        s0 - ROUGH_S_PRESELECT_HALF,
        s0 + ROUGH_S_PRESELECT_HALF
    )
    if i1 <= i0:
        continue

    S_sub, T_sub = Sg[i0:i1], Tg[i0:i1]
    Z_sub, I_sub = Zg[i0:i1], Ig[i0:i1]
    ExG_sub = ExGg[i0:i1]

    tm = np.abs(T_sub - t0) <= ROUGH_T_PRESELECT_HALF
    if tm.sum() < 50:
        continue

    S_sub, T_sub = S_sub[tm], T_sub[tm]
    Z_sub, I_sub = Z_sub[tm], I_sub[tm]
    ExG_sub = ExG_sub[tm]

    u, v = project_to_local_frame(S_sub, T_sub, s0, t0, d0)
    sm = (np.abs(u) <= LOCAL_SECTION_HALF_U) & (np.abs(v) <= ENV_V_RANGE)
    if sm.sum() < 50:
        continue

    v_sec = v[sm]
    z_sec = Z_sub[sm]
    i_sec = I_sub[sm]
    exg_sec = ExG_sub[sm]

    # build lower envelope with morphological opening
    env_v, env_z, bin_counts, bin_i_med = build_lower_envelope(
        v_sec, z_sec, i_vals=i_sec,
        v_range=ENV_V_RANGE, bin_w=ENV_BIN_W,
        q=ENV_Q, min_points_per_bin=ENV_MIN_POINTS_PER_BIN
    )
    if len(env_v) < 20:
        continue

    # Split envelope into left/right sides
    lm = env_v < 0
    x_left = -env_v[lm]
    z_left = env_z[lm]
    ol = np.argsort(x_left)
    x_left, z_left = x_left[ol], z_left[ol]

    rm = env_v > 0
    x_right = env_v[rm]
    z_right = env_z[rm]
    or_ = np.argsort(x_right)
    x_right, z_right = x_right[or_], z_right[or_]

    # Spatial-continuity priors (median of last N confirmed bottom_x)
    left_prior_x  = _prior_median(recent_left_bx)
    right_prior_x = _prior_median(recent_right_bx)

    left_m = compute_side_metrics_first_ditch(
        x_left, z_left, z_ref, prior_x=left_prior_x)
    right_m = compute_side_metrics_first_ditch(
        x_right, z_right, z_ref, prior_x=right_prior_x)

    # Update rolling priors with the new detections (if valid)
    if left_m["ditch_exists"] == 1 and not np.isnan(left_m.get("bottom_x", np.nan)):
        _push_prior(recent_left_bx, left_m["bottom_x"])
    if right_m["ditch_exists"] == 1 and not np.isnan(right_m.get("bottom_x", np.nan)):
        _push_prior(recent_right_bx, right_m["bottom_x"])

    # Water detection per side (only inside a confirmed ditch span)
    left_water, left_water_frac, left_water_bins = detect_water_in_ditch(
        env_v, bin_counts, bin_i_med, side_sign=-1,
        ditch_exists=(left_m["ditch_exists"] == 1),
        span_x0=left_m.get("span_x0", np.nan), span_x1=left_m.get("span_x1", np.nan))
    right_water, right_water_frac, right_water_bins = detect_water_in_ditch(
        env_v, bin_counts, bin_i_med, side_sign=+1,
        ditch_exists=(right_m["ditch_exists"] == 1),
        span_x0=right_m.get("span_x0", np.nan), span_x1=right_m.get("span_x1", np.nan))

    # Water-occlusion depth recovery (bank-slope extrapolation)
    # NIR LiDAR reads the water surface, not the bed, in a flooded ditch, so the
    # measured depth there is systematically shallow. Where water is flagged we
    # recover the dry-bed depth from the two exposed banks (see
    # reconstruct_water_filled_bottom). Dry sections never enter this block, so
    # non-water tiles/sides are unchanged; the recovery can only deepen, never
    # raise, a measured bottom. The original water-surface depth is preserved in
    # `{side}_depth_water_surface` for transparency.
    for side_m, side_water, sx, sz in (
        (left_m, left_water, x_left, z_left),
        (right_m, right_water, x_right, z_right),
    ):
        side_m["depth_water_surface"] = side_m.get("depth", np.nan)
        side_m["water_bottom_z_recon"] = np.nan
        side_m["water_recon_flag"] = 0
        if (USE_WATER_BOTTOM_RECON
                and side_m["ditch_exists"] == 1 and bool(side_water)
                and not np.isnan(side_m.get("depth", np.nan))
                and not np.isnan(z_ref)):
            _rec = reconstruct_water_filled_bottom(
                sx, sz,
                side_m.get("bottom_x", np.nan),
                side_m.get("span_x0", np.nan),
                side_m.get("span_x1", np.nan),
                side_m.get("bottom_z", np.nan),
                z_ref,
            )
            if _rec["valid"] and _rec["depth"] > side_m["depth"]:
                side_m["depth"] = _rec["depth"]
                side_m["water_bottom_z_recon"] = _rec["bottom_z"]
                side_m["water_recon_flag"] = 1

    # DVCI (vegetation cover index)
    if left_m["ditch_exists"] == 1:
        left_m["dvci"] = compute_dvci_for_span(
            v_sec, exg_sec, -1, left_m["span_x0"], left_m["span_x1"]
        )
    if right_m["ditch_exists"] == 1:
        right_m["dvci"] = compute_dvci_for_span(
            v_sec, exg_sec, +1, right_m["span_x0"], right_m["span_x1"]
        )

    # Depth validity filter
    for m_side in (left_m, right_m):
        if m_side["ditch_exists"] == 1 and not np.isnan(m_side["depth"]):
            if not (DEPTH_VALID_MIN <= m_side["depth"] <= DEPTH_VALID_MAX):
                m_side.update({
                    "side_type": "FLAT", "depth": np.nan,
                    "width": np.nan, "top_width": np.nan, "area": np.nan,
                    "bottom_x": np.nan, "bottom_z": np.nan,
                    "span_x0": np.nan, "span_x1": np.nan,
                    "top_span_x0": np.nan, "top_span_x1": np.nan,
                    "top_width_quality": "invalid",
                    "inner_slope_deg": np.nan, "outer_slope_deg": np.nan,
                    "flatness": np.nan, "asymmetry": np.nan,
                    "depth_deficit": np.nan,
                    "dvci": np.nan, "ditch_exists": 0,
                    "pass_source": "none", "confidence": 0.0,
                })

    ditch_presence = get_ditch_presence(
        bool(left_m["ditch_exists"]), bool(right_m["ditch_exists"])
    )

    # Shape class (V / U / FLAT_SILTED) + confidence
    # Now water-aware: if water_flag is set, classification falls back
    # to flatness-only with reduced confidence (0.4), because the
    # reconstructed envelope at the bottom is unreliable. Dry sections
    # use flatness + bottom curvature agreement (confidence 0.6 - .0).
    if left_m["ditch_exists"] == 1:
        left_shape, left_shape_conf = classify_shape(
            left_m.get("flatness", np.nan),
            left_m.get("depth_deficit", np.nan),
            left_m.get("bottom_curvature_norm", np.nan),
            bool(left_water),
        )
    else:
        left_shape, left_shape_conf = "NA", 0.0
    if right_m["ditch_exists"] == 1:
        right_shape, right_shape_conf = classify_shape(
            right_m.get("flatness", np.nan),
            right_m.get("depth_deficit", np.nan),
            right_m.get("bottom_curvature_norm", np.nan),
            bool(right_water),
        )
    else:
        right_shape, right_shape_conf = "NA", 0.0

    # Content class (Roelens 2018 5-class)
    left_content = classify_content(left_m["dvci"], bool(left_water)) \
        if left_m["ditch_exists"] == 1 else "NA"
    right_content = classify_content(right_m["dvci"], bool(right_water)) \
        if right_m["ditch_exists"] == 1 else "NA"

    # Multi-label issue diagnosis (TDOK Section 10.13 + Knighton A1)
    left_issues_list = diagnose_issues(
        left_shape, left_content, left_m["depth"],
        left_m["inner_slope_deg"], left_m.get("outer_slope_deg", np.nan),
        left_m.get("asymmetry", np.nan),          # Knighton A1 for SLOPE_ISSUE
        left_m["ditch_exists"], bool(zref_valid),
    )
    right_issues_list = diagnose_issues(
        right_shape, right_content, right_m["depth"],
        right_m["inner_slope_deg"], right_m.get("outer_slope_deg", np.nan),
        right_m.get("asymmetry", np.nan),         # Knighton A1 for SLOPE_ISSUE
        right_m["ditch_exists"], bool(zref_valid),
    )
    left_issues_str = ",".join(left_issues_list)
    right_issues_str = ",".join(right_issues_list)

    # TDOK priority mapping (V / M / Å / Ö / NA)
    left_priority = map_to_tdok_priority(left_issues_list, left_m["depth"])
    right_priority = map_to_tdok_priority(right_issues_list, right_m["depth"])

    left_deep = (
        left_m["ditch_exists"] == 1 and
        not np.isnan(left_m["depth"]) and
        left_m["depth"] > DEPTH_DEEP_FLAG
    )
    right_deep = (
        right_m["ditch_exists"] == 1 and
        not np.isnan(right_m["depth"]) and
        right_m["depth"] > DEPTH_DEEP_FLAG
    )

    rows.append({
        "s": s0,
        "T_center_used": t0,
        "T_center_init": row.get("T_center", np.nan),
        "T_center_refined": row.get("T_center_refined", np.nan),
        "dT_dS_used": d0,
        "centreline_field_used": t_col,
        "dtds_field_used": d_col,
        "zref_field_used": zref_col,

        "gauge_prior": float(row["gauge_prior"]),
        "half_gauge_prior": half_gauge,
        "T_left_prior_used": float(row[left_prior_col]) if left_prior_col in row else np.nan,
        "T_right_prior_used": float(row[right_prior_col]) if right_prior_col in row else np.nan,

        "Z_ref": z_ref,   # RUK (step06 applies RAIL_HEIGHT_M correction)
        "Z_ref_raw": row.get("Z_ref", np.nan),
        "Z_ref_smooth": row.get("Z_ref_smooth", np.nan),
        "zref_valid": zref_valid,
        "zref_quality": row.get("zref_quality", np.nan),
        "section_use_flag": use_flag,

        "reference_mode": row.get("reference_mode", np.nan),
        "rail_detect_mode": row.get("rail_detect_mode", np.nan),
        "rail_confidence": row.get("rail_confidence", np.nan),
        "center_shift": row.get("center_shift", np.nan),

        "n_section_pts": int(sm.sum()),
        "left_deep_flag": int(left_deep),
        "right_deep_flag": int(right_deep),

        # Left side
        "left_type": left_m["side_type"],
        "left_ditch_exists": left_m["ditch_exists"],
        "left_relief": left_m["relief"],
        "left_depth": left_m["depth"],         # depth below RUK
        "left_width": left_m["width"],
        "left_top_width": left_m.get("top_width", np.nan),
        "left_area": left_m["area"],
        "left_bottom_x": left_m["bottom_x"],
        "left_bottom_z": left_m["bottom_z"],
        "left_span_x0": left_m["span_x0"],
        "left_span_x1": left_m["span_x1"],
        "left_top_span_x0": left_m.get("top_span_x0", np.nan),
        "left_top_span_x1": left_m.get("top_span_x1", np.nan),
        "left_top_width_quality": left_m.get("top_width_quality", "invalid"),
        "left_inner_slope_deg": left_m["inner_slope_deg"],
        "left_outer_slope_deg": left_m.get("outer_slope_deg", np.nan),
        "left_flatness": left_m.get("flatness", np.nan),
        "left_asymmetry": left_m.get("asymmetry", np.nan),
        "left_depth_deficit": left_m.get("depth_deficit", np.nan),
        "left_dvci": left_m["dvci"],
        "left_shape_class": left_shape,
        "left_shape_confidence": left_shape_conf,
        "left_bottom_curvature_norm": left_m.get("bottom_curvature_norm", np.nan),
        "left_content_class": left_content,
        "left_issues": left_issues_str,
        "left_tdok_priority": left_priority,
        "left_priority_source": (
            "measured" if left_m["ditch_exists"] == 1 and left_priority != "NA" else "none"
        ),
        "left_water_flag": int(left_water),
        "left_water_frac": left_water_frac,
        "left_depth_water_surface": left_m.get("depth_water_surface", np.nan),
        "left_water_recon_flag": int(left_m.get("water_recon_flag", 0)),
        "left_water_bottom_z_recon": left_m.get("water_bottom_z_recon", np.nan),
        "left_pass_source": left_m.get("pass_source", "none"),
        "left_confidence":  left_m.get("confidence", 0.0),
        "left_candidate_json": left_m.get("candidate_json", "[]"),
        "left_candidate_count": left_m.get("candidate_count", 0),

        # Right side
        "right_type": right_m["side_type"],
        "right_ditch_exists": right_m["ditch_exists"],
        "right_relief": right_m["relief"],
        "right_depth": right_m["depth"],       # depth below RUK
        "right_width": right_m["width"],
        "right_top_width": right_m.get("top_width", np.nan),
        "right_area": right_m["area"],
        "right_bottom_x": right_m["bottom_x"],
        "right_bottom_z": right_m["bottom_z"],
        "right_span_x0": right_m["span_x0"],
        "right_span_x1": right_m["span_x1"],
        "right_top_span_x0": right_m.get("top_span_x0", np.nan),
        "right_top_span_x1": right_m.get("top_span_x1", np.nan),
        "right_top_width_quality": right_m.get("top_width_quality", "invalid"),
        "right_inner_slope_deg": right_m["inner_slope_deg"],
        "right_outer_slope_deg": right_m.get("outer_slope_deg", np.nan),
        "right_flatness": right_m.get("flatness", np.nan),
        "right_asymmetry": right_m.get("asymmetry", np.nan),
        "right_depth_deficit": right_m.get("depth_deficit", np.nan),
        "right_dvci": right_m["dvci"],
        "right_shape_class": right_shape,
        "right_shape_confidence": right_shape_conf,
        "right_bottom_curvature_norm": right_m.get("bottom_curvature_norm", np.nan),
        "right_content_class": right_content,
        "right_issues": right_issues_str,
        "right_tdok_priority": right_priority,
        "right_priority_source": (
            "measured" if right_m["ditch_exists"] == 1 and right_priority != "NA" else "none"
        ),
        "right_water_flag": int(right_water),
        "right_water_frac": right_water_frac,
        "right_depth_water_surface": right_m.get("depth_water_surface", np.nan),
        "right_water_recon_flag": int(right_m.get("water_recon_flag", 0)),
        "right_water_bottom_z_recon": right_m.get("water_bottom_z_recon", np.nan),
        "right_pass_source": right_m.get("pass_source", "none"),
        "right_confidence":  right_m.get("confidence", 0.0),
        "right_candidate_json": right_m.get("candidate_json", "[]"),
        "right_candidate_count": right_m.get("candidate_count", 0),

        "ditch_presence": ditch_presence,
    })

print(f"\n  Done: {len(rows)} sections")

df_1m = pd.DataFrame(rows)
if len(df_1m) == 0:
    raise RuntimeError("No valid 1 m sections were analysed.")

df_1m = df_1m.sort_values("s").reset_index(drop=True)


def _safe_float(value, default=np.nan):
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _load_candidate_list(text):
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _apply_longitudinal_candidate_path(df, side):
    """Select a smooth outer ditch path from per-section local candidates.

    The single-section detector may prefer a near-track local depression when
    that dip is stronger in one profile.  A real railway side ditch is a
    longitudinal object, so this layer reconsiders all stored local candidates
    and selects a laterally coherent outer path before the normal confirmation
    filter is applied.
    """
    cand_col = f"{side}_candidate_json"
    if cand_col not in df.columns:
        return 0

    states_by_row = []
    for row_i, text in enumerate(df[cand_col].tolist()):
        raw = _load_candidate_list(text)
        cands = []
        for c in raw:
            bx = _safe_float(c.get("bottom_x"))
            depth = _safe_float(c.get("depth"))
            width = _safe_float(c.get("width"))
            if not np.isfinite(bx) or not np.isfinite(depth) or not np.isfinite(width):
                continue
            if bx < OUTER_PATH_MIN_X:
                continue
            if depth < OUTER_SURFACE_MIN_DEPTH:
                continue
            cands.append(c)
        if cands:
            states_by_row.append((row_i, cands))

    if len(states_by_row) < PATH_MIN_SUPPORT_SECTIONS:
        return 0

    all_x = []
    all_w = []
    for _, cands in states_by_row:
        for c in cands:
            bx = _safe_float(c.get("bottom_x"))
            if np.isfinite(bx):
                all_x.append(bx)
                all_w.append(max(_safe_float(c.get("candidate_score"), 0.0), 0.0) + 1.0)
    if len(all_x) < PATH_MIN_SUPPORT_SECTIONS:
        return 0
    all_x = np.asarray(all_x, dtype=float)
    all_w = np.asarray(all_w, dtype=float)

    centres = np.arange(
        OUTER_PATH_MIN_X,
        max(SIDE_SEARCH_MAX, float(np.nanmax(all_x))) + PATH_CLUSTER_BIN_M,
        PATH_CLUSTER_BIN_M,
    )
    if len(centres) == 0:
        return 0
    cluster_scores = []
    for centre in centres:
        kernel = np.exp(-0.5 * ((all_x - centre) / max(PATH_CLUSTER_KERNEL_M, 1e-6)) ** 2)
        cluster_scores.append(float(np.sum(kernel * all_w)))
    target_x = float(centres[int(np.argmax(cluster_scores))])

    clustered = []
    for row_i, cands in states_by_row:
        kept = [
            c for c in cands
            if abs(_safe_float(c.get("bottom_x")) - target_x) <= PATH_CLUSTER_HALF_WIDTH_M
        ]
        if kept:
            clustered.append((row_i, kept))
    states_by_row = clustered
    if len(states_by_row) < PATH_MIN_SUPPORT_SECTIONS:
        return 0

    dp = []
    back = []
    for block_i, (row_i, cands) in enumerate(states_by_row):
        dp_row = []
        back_row = []
        for c in cands:
            bx = _safe_float(c.get("bottom_x"))
            # Outer-reach fix: the legacy +0.10*bx term rewards farther path
            # nodes, reinforcing the drift to the search boundary. When the fix
            # is on, drop it so the path follows the depth/confidence/continuity
            # evidence instead of being pulled outward.
            local_score = (
                _safe_float(c.get("candidate_score"), 0.0)
                + 0.50 * _safe_float(c.get("confidence"), 0.0)
                + (0.0 if USE_OUTER_REACH_FIX else 0.10 * bx)
            )
            best_prev_score = 0.0
            best_prev = None
            if block_i > 0:
                prev_row_i, prev_cands = states_by_row[block_i - 1]
                row_gap = row_i - prev_row_i
                if row_gap <= PATH_MAX_ROW_GAP:
                    for prev_j, prev_c in enumerate(prev_cands):
                        prev_bx = _safe_float(prev_c.get("bottom_x"))
                        jump = abs(bx - prev_bx)
                        penalty = (
                            PATH_JUMP_PENALTY * jump
                            + PATH_EXTRA_JUMP_PENALTY * max(jump - POSITION_OUTLIER_TOL_M, 0.0) ** 2
                            + 0.10 * max(row_gap - 1, 0)
                        )
                        score = dp[block_i - 1][prev_j] - penalty
                        if score > best_prev_score:
                            best_prev_score = score
                            best_prev = (block_i - 1, prev_j)
            dp_row.append(local_score + best_prev_score)
            back_row.append(best_prev)
        dp.append(dp_row)
        back.append(back_row)

    best_block = None
    best_j = None
    best_score = -np.inf
    for block_i, row_scores in enumerate(dp):
        for j, score in enumerate(row_scores):
            if score > best_score:
                best_score = score
                best_block = block_i
                best_j = j

    if best_block is None:
        return 0

    path = []
    cur = (best_block, best_j)
    while cur is not None:
        block_i, j = cur
        row_i, cands = states_by_row[block_i]
        path.append((row_i, cands[j]))
        cur = back[block_i][j]
    path.reverse()

    if len(path) < PATH_MIN_SUPPORT_SECTIONS:
        return 0
    path_x = np.array([_safe_float(c.get("bottom_x")) for _, c in path], dtype=float)
    if np.nanmedian(path_x) < OUTER_PATH_MIN_X:
        return 0
    if np.nanpercentile(path_x, 75) - np.nanpercentile(path_x, 25) > PATH_MAX_STABLE_IQR_M:
        return 0

    updated = 0
    for row_i, cand in path:
        old_x = _safe_float(df.at[df.index[row_i], f"{side}_bottom_x"])
        new_x = _safe_float(cand.get("bottom_x"))
        if np.isfinite(old_x) and old_x >= OUTER_PATH_MIN_X and abs(old_x - new_x) <= 0.25:
            continue

        for field in [
            "depth", "width", "top_width", "area", "bottom_x", "bottom_z",
            "span_x0", "span_x1", "top_span_x0", "top_span_x1",
            "inner_slope_deg", "outer_slope_deg", "flatness", "asymmetry",
            "depth_deficit",
        ]:
            df.at[df.index[row_i], f"{side}_{field}"] = _safe_float(cand.get(field))

        water_flag = bool(df.at[df.index[row_i], f"{side}_water_flag"]) if f"{side}_water_flag" in df.columns else False
        shape, shape_conf = classify_shape(
            _safe_float(cand.get("flatness")),
            _safe_float(cand.get("depth_deficit")),
            np.nan,
            water_flag,
        )
        dvci = _safe_float(df.at[df.index[row_i], f"{side}_dvci"]) if f"{side}_dvci" in df.columns else np.nan
        content = classify_content(dvci, water_flag)
        issues = diagnose_issues(
            shape,
            content,
            _safe_float(cand.get("depth")),
            _safe_float(cand.get("inner_slope_deg")),
            _safe_float(cand.get("outer_slope_deg")),
            _safe_float(cand.get("asymmetry")),
            1,
            bool(df.at[df.index[row_i], "zref_valid"]),
        )

        df.at[df.index[row_i], f"{side}_type"] = "LONGITUDINAL_DITCH_PATH"
        df.at[df.index[row_i], f"{side}_ditch_exists"] = 1
        df.at[df.index[row_i], f"{side}_shape_class"] = shape
        df.at[df.index[row_i], f"{side}_shape_confidence"] = shape_conf
        df.at[df.index[row_i], f"{side}_content_class"] = content
        df.at[df.index[row_i], f"{side}_issues"] = ",".join(issues)
        df.at[df.index[row_i], f"{side}_tdok_priority"] = map_to_tdok_priority(
            issues, _safe_float(cand.get("depth"))
        )
        if f"{side}_priority_source" in df.columns:
            df.at[df.index[row_i], f"{side}_priority_source"] = "longitudinal_candidate_path"
        df.at[df.index[row_i], f"{side}_pass_source"] = (
            str(cand.get("pass_source", "pass1")) + "_path"
        )
        df.at[df.index[row_i], f"{side}_confidence"] = max(
            _safe_float(cand.get("confidence"), 0.0), 0.55
        )
        updated += 1

    return updated


if LONGITUDINAL_CANDIDATE_PATH:
    print("\n[Post] Longitudinal multi-candidate path selection...")
    n_left_path = _apply_longitudinal_candidate_path(df_1m, "left")
    n_right_path = _apply_longitudinal_candidate_path(df_1m, "right")
    df_1m["ditch_presence"] = [
        get_ditch_presence(bool(l), bool(r))
        for l, r in zip(df_1m["left_ditch_exists"], df_1m["right_ditch_exists"])
    ]
    print(f"  Left:  {n_left_path} sections reassigned from candidate path")
    print(f"  Right: {n_right_path} sections reassigned from candidate path")

# =========================================================
# LONGITUDINAL CONTINUITY FILTER (two-tier: confirmed / candidate)
# =========================================================
# Literature alignment:
# - Roelens et al. 2018 treats isolated LiDAR ditch-dropouts as real
# features (geometric reconstruction) rather than rejecting them.
# - Cazorzi et al. 2013 only discards single-pixel clusters.
#
# Two-tier design:
# CONFIRMED - passes 2-of-5 sliding-window test ->used for
# TDOK 2015:0155 Section 10.13 diagnosis (ditch_exists = 1)
# CANDIDATE - detected but fails continuity ->retained for
# inspection planning (ditch_exists = 0, but
# depth/width/slope values preserved in the CSV)
# NONE - no detection at all (ditch_exists = 0)
#
# Downstream code using `(exists == 1)` automatically sees confirmed
# ditches only. The `{side}_ditch_level` column exposes all three.
print("\n[Post] Longitudinal continuity filter (two-tier)...")

for side in ["left", "right"]:
    exists_col = f"{side}_ditch_exists"
    level_col  = f"{side}_ditch_level"
    water_col  = f"{side}_water_flag"
    exists_arr = df_1m[exists_col].values.copy()
    water_arr  = df_1m[water_col].values if water_col in df_1m.columns else np.zeros_like(exists_arr)
    n_before   = int(exists_arr.sum())

    # Initialise level column (default: "none")
    df_1m[level_col] = "none"

    # Run-length-based continuity (replaces the old sliding-window sum).
    # Rule: a section is CONFIRMED iff it belongs to a run of at least
    # DITCH_MIN_RUN_LENGTH consecutive detections. One-section gaps are
    # bridged (via morphological closing with kernel 3) so that minor
    # envelope noise in the middle of a real ditch does not split it
    # into two sub-threshold runs. Multi-section gaps are handled later
    # by the gap-fill stage.
    #
    # This is strictly stronger than the old "sum >=MIN in window" rule:
    # any confirmed section now provably belongs to a >=N-consecutive run,
    # so the minimum possible confirmed-ditch length = N sections = N m.
    from scipy.ndimage import label as _label_runs
    from scipy.ndimage import binary_closing as _bc_runs
    # Bridge 1-section gaps only (kernel 3) - the gap-fill stage later
    # handles wider gaps with its own dedicated re-detection pass.
    closed_arr = _bc_runs(exists_arr.astype(bool),
                          structure=np.ones(3, dtype=bool))
    labels_run, n_runs = _label_runs(closed_arr)
    confirmed = np.zeros_like(exists_arr)
    for run_id in range(1, n_runs + 1):
        run_mask = (labels_run == run_id)
        if run_mask.sum() >= DITCH_MIN_RUN_LENGTH:
            # Confirm only the ORIGINAL detections in this run. Sections
            # that were closed-bridged (originally 0) remain 0 and will
            # be handled by the gap-fill stage downstream.
            confirmed[run_mask & (exists_arr == 1)] = 1
    # water_arr is no longer used in continuity - a water-filled ditch
    # still needs real-geometry evidence over >=N sections. The relaxed
    # "1 of 5 in water" rule was too permissive and let isolated water
    # detections through as confirmed ditches.
    _ = water_arr   # kept in scope, still written to CSV via step07 path

    confirmed_mask = (exists_arr == 1) & (confirmed == 1)
    candidate_mask = (exists_arr == 1) & (confirmed == 0)
    n_confirmed = int(confirmed_mask.sum())
    n_candidate = int(candidate_mask.sum())

    # Mark confirmed ditches
    df_1m.loc[confirmed_mask, level_col] = "confirmed"

    # Demote candidate ditches: set exists=0 but KEEP depth/width/slope/etc.
    # The level column marks them for future inspection planning.
    if n_candidate > 0:
        df_1m.loc[candidate_mask, exists_col] = 0
        df_1m.loc[candidate_mask, level_col]  = "candidate"
        df_1m.loc[candidate_mask, f"{side}_type"] = "ISOLATED_DIP"
        df_1m.loc[candidate_mask, f"{side}_issues"] = "NA"
        df_1m.loc[candidate_mask, f"{side}_tdok_priority"] = "NA"
        if f"{side}_priority_source" in df_1m.columns:
            df_1m.loc[candidate_mask, f"{side}_priority_source"] = "candidate"
        # NOTE: depth/width/bottom_x/slopes/shape/content are PRESERVED
        # on candidates - only the ditch_exists flag is cleared so the
        # two-tier system can tell them apart from confirmed detections.

    print(f"  {side.capitalize()}: {n_before} detected -> "
          f"{n_confirmed} confirmed + {n_candidate} candidate "
          f"(run-length {DITCH_MIN_RUN_LENGTH}/{DITCH_CONFIRM_WINDOW})")

# Spatial-coherence filter (position outlier rejection)
# Real drainage ditches run parallel to the track with a lateral
# position that varies smoothly. A confirmed detection whose
# bottom_x jumps > POSITION_OUTLIER_TOL_M from the rolling median of
# its neighbours is treated as a position outlier and demoted to
# candidate, even if it passed the count-based continuity test.
#
# This catches a failure mode the count filter cannot: sustained but
# geometrically inconsistent detections (e.g. edge-walk hitting an
# embankment slope instead of the real ditch, giving a cluster of
# fake "ditches" at the wrong lateral offset).
print("\n[Post] Spatial-coherence filter (position outlier rejection)...")
for side in ["left", "right"]:
    exists_col = f"{side}_ditch_exists"
    bx_col     = f"{side}_bottom_x"
    level_col  = f"{side}_ditch_level"

    exists_arr = df_1m[exists_col].values.copy()
    bx_arr     = pd.to_numeric(df_1m[bx_col], errors="coerce").values
    n          = len(exists_arr)

    # Keep the original count-filter stats for reporting
    n_before_pos = int(exists_arr.sum())
    outlier_idx  = []

    for i in range(n):
        if exists_arr[i] != 1 or np.isnan(bx_arr[i]):
            continue
        # Collect neighbour bottom_x values in a +/-POSITION_OUTLIER_WIN_HALF window
        lo = max(0, i - POSITION_OUTLIER_WIN_HALF)
        hi = min(n, i + POSITION_OUTLIER_WIN_HALF + 1)
        nbr_bx = [bx_arr[j] for j in range(lo, hi)
                  if j != i and exists_arr[j] == 1 and not np.isnan(bx_arr[j])]
        if len(nbr_bx) < POSITION_OUTLIER_MIN_NBR:
            continue   # gap region - cannot judge position coherence

        med = float(np.median(nbr_bx))
        if abs(bx_arr[i] - med) > POSITION_OUTLIER_TOL_M:
            outlier_idx.append(i)

    n_out = len(outlier_idx)
    pct = (n_out / n_before_pos * 100.0) if n_before_pos > 0 else 0.0

    # Demote: clear exists flag, tag as position_outlier.
    # Geometry fields (depth/width/slopes/shape/content) are PRESERVED
    # so the rows can still be inspected downstream.
    for i in outlier_idx:
        df_1m.at[df_1m.index[i], exists_col] = 0
        df_1m.at[df_1m.index[i], level_col]  = "position_outlier"
        df_1m.at[df_1m.index[i], f"{side}_type"] = "POSITION_OUTLIER"
        df_1m.at[df_1m.index[i], f"{side}_issues"] = "NA"
        df_1m.at[df_1m.index[i], f"{side}_tdok_priority"] = "NA"
        if f"{side}_priority_source" in df_1m.columns:
            df_1m.at[df_1m.index[i], f"{side}_priority_source"] = "position_outlier"
        # Rejected by spatial coherence, so confidence is set to 0.
        if f"{side}_confidence" in df_1m.columns:
            df_1m.at[df_1m.index[i], f"{side}_confidence"] = 0.0
        if f"{side}_pass_source" in df_1m.columns:
            df_1m.at[df_1m.index[i], f"{side}_pass_source"] = "position_outlier"

    print(f"  {side.capitalize()}: {n_before_pos} confirmed -> "
          f"{n_before_pos - n_out} kept + {n_out} position outliers rejected "
          f"({pct:.1f}%, tol = {POSITION_OUTLIER_TOL_M} m)")

# Gap-filling: morphological closing + constrained re-detection
# (Soille 2004; Cazorzi et al. 2013; Roelens et al. 2018)
#
# Step A: binary closing (kernel=3) identifies isolated 1-section gaps.
# Step B: for each gap, rebuild the envelope and re-run ditch detection
# with RELAXED thresholds, using neighbour ditch position as
# a spatial prior. This directly measures depth from the actual
# cross-section instead of interpolating.
# Step C: if re-detection still fails, fall back to neighbour interpolation.
from scipy.ndimage import binary_closing

print("\n[Post] Gap-filling: closing + constrained re-detection (Roelens 2018)...")
CLOSING_KERNEL = np.ones(int(_T("CLOSING_KERNEL", 5)), dtype=bool)
GAP_RELAXED_MIN_DEPTH = _T("GAP_RELAXED_MIN_DEPTH", 0.04)   # half of normal DITCH_MIN_DEPTH
GAP_RELAXED_MIN_PROM  = _T("GAP_RELAXED_MIN_PROM",  0.03)   # half of normal DITCH_MIN_PROM
GAP_POSITION_TOLERANCE = _T("GAP_POSITION_TOLERANCE", 3.0)   # +/-X m of neighbour position
GAP_PRIORITY_RANK = {"Ö": 1, "Å": 2, "M": 3, "V": 4, "NA": 0}
GAP_PRIORITY_INV = {v: k for k, v in GAP_PRIORITY_RANK.items()}

geom_cols = ["depth", "width", "top_width", "area", "bottom_x", "bottom_z",
             "span_x0", "span_x1", "top_span_x0", "top_span_x1",
             "inner_slope_deg", "outer_slope_deg",
             "flatness", "asymmetry", "depth_deficit", "dvci"]


def _nearest_confirmed_gap_priority(df, side, row_i):
    """Inherit the worst priority from nearest non-interpolated confirmed neighbours.

    Interpolated gap fills preserve ditch continuity for maintenance maps, but
    they do not have independent local shape/content evidence.  Their priority
    therefore inherits the more severe priority of the closest confirmed ditch
    on each side of the gap.
    """
    prio_col = f"{side}_tdok_priority"
    exists_col = f"{side}_ditch_exists"
    level_col = f"{side}_ditch_level"
    pass_col = f"{side}_pass_source"

    inherited = []
    for step in (-1, 1):
        j = row_i + step
        while 0 <= j < len(df):
            if df.at[df.index[j], exists_col] == 1:
                level = df.at[df.index[j], level_col] if level_col in df.columns else "confirmed"
                source = df.at[df.index[j], pass_col] if pass_col in df.columns else ""
                prio = df.at[df.index[j], prio_col]
                if (
                    level == "confirmed"
                    and source != "gap_interpolated"
                    and isinstance(prio, str)
                    and prio in GAP_PRIORITY_RANK
                    and GAP_PRIORITY_RANK[prio] > 0
                ):
                    inherited.append(prio)
                    break
            j += step

    if not inherited:
        return "NA"
    return GAP_PRIORITY_INV[max(GAP_PRIORITY_RANK[p] for p in inherited)]

for side in ["left", "right"]:
    exists_col = f"{side}_ditch_exists"
    raw = df_1m[exists_col].values.astype(bool)
    closed = binary_closing(raw, structure=CLOSING_KERNEL)
    newly_filled = closed & (~raw)
    n_gaps = int(newly_filled.sum())
    n_redetected = 0
    n_interpolated = 0

    if n_gaps == 0:
        print(f"  {side.capitalize()}: 0 gaps to fill")
        continue

    filled_idx = np.where(newly_filled)[0]
    valid_idx = np.where(raw)[0]

    for fi in filled_idx:
        s0_gap = float(df_1m.iloc[fi]["s"])

        # Get neighbour ditch bottom_x as spatial prior
        nbr_bottom_x = []
        for di in [-1, 1]:
            ni = fi + di
            if 0 <= ni < len(df_1m) and raw[ni]:
                bx = df_1m.iloc[ni].get(f"{side}_bottom_x", np.nan)
                if not np.isnan(bx):
                    nbr_bottom_x.append(float(bx))
        prior_x = float(np.mean(nbr_bottom_x)) if nbr_bottom_x else np.nan

        # Step B: re-detect from raw point cloud
        redetected = False
        ref_row = ref.iloc[np.argmin(np.abs(ref["s"].values - s0_gap))]
        t0_gap = float(ref_row[t_col])
        d0_gap = float(ref_row[d_col])
        z_ref_gap = float(ref_row[zref_col]) if not pd.isna(ref_row[zref_col]) else np.nan

        i0g, i1g = sorted_slice_bounds(Sg, s0_gap - ROUGH_S_PRESELECT_HALF,
                                       s0_gap + ROUGH_S_PRESELECT_HALF)
        if i1g > i0g:
            S_sub = Sg[i0g:i1g]; T_sub = Tg[i0g:i1g]
            Z_sub = Zg[i0g:i1g]; I_sub = Ig[i0g:i1g]
            ExG_sub = ExGg[i0g:i1g]

            tm = np.abs(T_sub - t0_gap) <= ROUGH_T_PRESELECT_HALF
            if tm.sum() >= 50:
                ug, vg = project_to_local_frame(S_sub[tm], T_sub[tm],
                                                s0_gap, t0_gap, d0_gap)
                smg = (np.abs(ug) <= LOCAL_SECTION_HALF_U) & (np.abs(vg) <= ENV_V_RANGE)

                if smg.sum() >= 50:
                    v_sec_g = vg[smg]
                    z_sec_g = Z_sub[tm][smg]
                    i_sec_g = I_sub[tm][smg]
                    exg_sec_g = ExG_sub[tm][smg]

                    env_vg, env_zg, _, _ = build_lower_envelope(
                        v_sec_g, z_sec_g, i_vals=i_sec_g,
                        v_range=ENV_V_RANGE, bin_w=ENV_BIN_W,
                        q=ENV_Q, min_points_per_bin=ENV_MIN_POINTS_PER_BIN)

                    if len(env_vg) >= 20:
                        # Extract the correct side
                        if side == "left":
                            sm_side = env_vg < 0
                            x_side = -env_vg[sm_side]
                        else:
                            sm_side = env_vg > 0
                            x_side = env_vg[sm_side]
                        z_side = env_zg[sm_side]
                        oo = np.argsort(x_side)
                        x_side, z_side = x_side[oo], z_side[oo]

                        # Relaxed detection: lower thresholds
                        use_zref_gap = z_ref_gap
                        if np.isnan(use_zref_gap) and len(x_side) >= 8:
                            inner = z_side[x_side < 1.0]
                            if len(inner) >= 3:
                                use_zref_gap = float(np.percentile(inner, 75))

                        if len(x_side) >= 8 and not np.isnan(use_zref_gap):
                            keep_g = (x_side >= SIDE_SEARCH_MIN) & (x_side <= SIDE_SEARCH_MAX)
                            xkg, zkg = x_side[keep_g], z_side[keep_g]

                            if len(xkg) >= 6:
                                # Try detection with relaxed thresholds
                                inv_g = -zkg
                                peaks_g, _ = find_peaks(inv_g, prominence=GAP_RELAXED_MIN_PROM)
                                if len(peaks_g) == 0:
                                    gmin = int(np.argmin(zkg))
                                    if 1 < gmin < len(zkg) - 2:
                                        peaks_g = np.array([gmin])

                                # If we have a spatial prior, prefer peaks near it
                                if len(peaks_g) > 0 and not np.isnan(prior_x):
                                    dist_to_prior = np.abs(xkg[peaks_g] - prior_x)
                                    near_mask = dist_to_prior <= GAP_POSITION_TOLERANCE
                                    if near_mask.any():
                                        peaks_g = peaks_g[near_mask]
                                    peaks_g = peaks_g[np.argsort(np.abs(xkg[peaks_g] - prior_x))]

                                for pk in peaks_g:
                                    depth_g = float(use_zref_gap - zkg[pk])
                                    if depth_g < GAP_RELAXED_MIN_DEPTH:
                                        continue
                                    outer_z_g = zkg[pk+1:] if pk+1 < len(zkg) else np.array([])
                                    # Proportional outer-rise (same logic as
                                    # _try_find_ditch - rejects L-shape embankment
                                    # toes that have only a few cm of noise past
                                    # a deep descent).
                                    if len(outer_z_g) < 1:
                                        continue
                                    outer_rise_g = float(outer_z_g.max() - zkg[pk])
                                    required_rise_g = max(SHAPE_OUTER_RISE_MIN,
                                                          SHAPE_OUTER_RISE_FRAC * depth_g)
                                    if outer_rise_g < required_rise_g:
                                        continue

                                    top_width_g = estimate_top_bank_width(
                                        xkg, zkg, pk,
                                        allow_local_edge_fallback=False,
                                    )
                                    if top_width_g["top_width_quality"] == "invalid":
                                        continue

                                    j0 = top_width_g["top_i0"]
                                    j1 = top_width_g["top_i1"]
                                    w_g = float(top_width_g["top_width"])
                                    if w_g < TOP_BANK_MIN_WIDTH or w_g > TOP_BANK_MAX_WIDTH:
                                        continue

                                    deficit_g = np.maximum(0.0, use_zref_gap - zkg[j0:j1+1])
                                    area_g = float(np.trapz(deficit_g, xkg[j0:j1+1])) if j1 > j0 else 0.0
                                    if area_g < DITCH_MIN_AREA:
                                        continue

                                    bx_g = float(xkg[pk])
                                    bz_g = float(zkg[pk])
                                    inner_run = max(1e-6, bx_g - float(xkg[j0]))
                                    inner_slope = float(np.degrees(np.arctan((zkg[j0]-bz_g)/inner_run)))
                                    outer_run = max(1e-6, float(xkg[j1]) - bx_g)
                                    outer_slope = float(np.degrees(np.arctan((zkg[j1]-bz_g)/outer_run)))

                                    # Shape validity (same bounds as Pass 1).
                                    # On an embankment, outer_slope will be
                                    # very small or negative - this rejects it.
                                    if not (DITCH_INNER_SLOPE_MIN_DEG <= inner_slope <= DITCH_INNER_SLOPE_MAX_DEG):
                                        continue
                                    if not (DITCH_OUTER_SLOPE_MIN_DEG <= outer_slope <= DITCH_OUTER_SLOPE_MAX_DEG):
                                        continue

                                    # Conditional global eff-slope check.
                                    if (xkg[pk] - xkg[0]) > EFFECTIVE_SLOPE_MIN_BOTTOM_X:
                                        eff_run_g  = float(xkg[pk] - xkg[0])
                                        eff_rise_g = float(zkg[0] - zkg[pk])
                                        if eff_run_g > 0.5 and eff_rise_g > 0.0:
                                            eff_slope_g = float(np.degrees(
                                                np.arctan(eff_rise_g / eff_run_g)
                                            ))
                                            if eff_slope_g < EFFECTIVE_INNER_SLOPE_MIN:
                                                continue

                                    # Shape descriptors for the gap-detected ditch
                                    shape_gap = compute_shape_descriptors(
                                        xkg, zkg, j0, j1, pk, inner_slope
                                    )

                                    # Reject extreme one-sidedness (Knighton A1).
                                    asym_g = shape_gap.get("asymmetry", np.nan)
                                    if (not np.isnan(asym_g)
                                            and abs(asym_g) > SLOPE_ASYMMETRY_MAX):
                                        continue

                                    # DVCI first (needed for content classifier)
                                    side_sign = -1 if side == "left" else +1
                                    dvci_g = compute_dvci_for_span(
                                        v_sec_g, exg_sec_g, side_sign,
                                        float(xkg[j0]), float(xkg[j1]))

                                    # Classify + diagnose.
                                    # Gap-filled sections conservatively assume
                                    # no water (we didn't re-run water detection)
                                    # ->water_flag=False, but curvature may be
                                    # available from shape_gap.
                                    shape_cls_g, shape_conf_g = classify_shape(
                                        shape_gap["flatness"],
                                        shape_gap["depth_deficit"],
                                        shape_gap.get("bottom_curvature_norm", np.nan),
                                        False,
                                    )
                                    content_cls_g = classify_content(dvci_g, False)
                                    issues_g = diagnose_issues(
                                        shape_cls_g, content_cls_g, depth_g,
                                        inner_slope, outer_slope,
                                        shape_gap.get("asymmetry", np.nan),
                                        1, True,
                                    )
                                    prio_g = map_to_tdok_priority(issues_g, depth_g)

                                    # Write results
                                    df_1m.at[df_1m.index[fi], exists_col] = 1
                                    df_1m.at[df_1m.index[fi], f"{side}_ditch_level"] = "confirmed"
                                    df_1m.at[df_1m.index[fi], f"{side}_type"] = "GAP_REDETECTED"
                                    df_1m.at[df_1m.index[fi], f"{side}_depth"] = depth_g
                                    df_1m.at[df_1m.index[fi], f"{side}_width"] = w_g
                                    if f"{side}_top_width" in df_1m.columns:
                                        df_1m.at[df_1m.index[fi], f"{side}_top_width"] = top_width_g["top_width"]
                                    df_1m.at[df_1m.index[fi], f"{side}_area"] = area_g
                                    df_1m.at[df_1m.index[fi], f"{side}_bottom_x"] = bx_g
                                    df_1m.at[df_1m.index[fi], f"{side}_bottom_z"] = bz_g
                                    df_1m.at[df_1m.index[fi], f"{side}_span_x0"] = float(xkg[j0])
                                    df_1m.at[df_1m.index[fi], f"{side}_span_x1"] = float(xkg[j1])
                                    if f"{side}_top_span_x0" in df_1m.columns:
                                        df_1m.at[df_1m.index[fi], f"{side}_top_span_x0"] = top_width_g["top_span_x0"]
                                    if f"{side}_top_span_x1" in df_1m.columns:
                                        df_1m.at[df_1m.index[fi], f"{side}_top_span_x1"] = top_width_g["top_span_x1"]
                                    if f"{side}_top_width_quality" in df_1m.columns:
                                        df_1m.at[df_1m.index[fi], f"{side}_top_width_quality"] = top_width_g["top_width_quality"]
                                    df_1m.at[df_1m.index[fi], f"{side}_inner_slope_deg"] = inner_slope
                                    df_1m.at[df_1m.index[fi], f"{side}_outer_slope_deg"] = outer_slope
                                    df_1m.at[df_1m.index[fi], f"{side}_flatness"] = shape_gap["flatness"]
                                    df_1m.at[df_1m.index[fi], f"{side}_asymmetry"] = shape_gap["asymmetry"]
                                    df_1m.at[df_1m.index[fi], f"{side}_depth_deficit"] = shape_gap["depth_deficit"]
                                    df_1m.at[df_1m.index[fi], f"{side}_dvci"] = dvci_g
                                    df_1m.at[df_1m.index[fi], f"{side}_shape_class"] = shape_cls_g
                                    if f"{side}_shape_confidence" in df_1m.columns:
                                        df_1m.at[df_1m.index[fi], f"{side}_shape_confidence"] = shape_conf_g
                                    if f"{side}_bottom_curvature_norm" in df_1m.columns:
                                        df_1m.at[df_1m.index[fi], f"{side}_bottom_curvature_norm"] = shape_gap.get("bottom_curvature_norm", np.nan)
                                    df_1m.at[df_1m.index[fi], f"{side}_content_class"] = content_cls_g
                                    df_1m.at[df_1m.index[fi], f"{side}_issues"] = ",".join(issues_g)
                                    df_1m.at[df_1m.index[fi], f"{side}_tdok_priority"] = prio_g
                                    if f"{side}_priority_source" in df_1m.columns:
                                        df_1m.at[df_1m.index[fi], f"{side}_priority_source"] = "measured_gap_redetect"

                                    # Confidence for gap re-detection.
                                    # Use pass4_gap base (0.45) because the
                                    # thresholds were relaxed; the geometry still
                                    # came from the raw point cloud, not interp.
                                    conf_g = _compute_confidence(
                                        "pass4_gap",
                                        max(GAP_RELAXED_MIN_PROM, 0.03),
                                        float(depth_g), float(w_g),
                                    )
                                    if f"{side}_confidence" in df_1m.columns:
                                        df_1m.at[df_1m.index[fi], f"{side}_confidence"] = conf_g
                                    if f"{side}_pass_source" in df_1m.columns:
                                        df_1m.at[df_1m.index[fi], f"{side}_pass_source"] = "pass4_gap_redetect"

                                    redetected = True
                                    n_redetected += 1
                                    break

        # Step C: fallback interpolation if re-detection failed
        if not redetected:
            df_1m.at[df_1m.index[fi], exists_col] = 1
            df_1m.at[df_1m.index[fi], f"{side}_ditch_level"] = "confirmed"
            df_1m.at[df_1m.index[fi], f"{side}_type"] = "GAP_INTERPOLATED"
            # Interpolated fills don't have fresh local shape/content
            # evidence ->label diagnostics NA. Priority is inherited from
            # neighbouring confirmed sections for maintenance-map continuity.
            df_1m.at[df_1m.index[fi], f"{side}_shape_class"] = "NA"
            if f"{side}_shape_confidence" in df_1m.columns:
                df_1m.at[df_1m.index[fi], f"{side}_shape_confidence"] = 0.0
            df_1m.at[df_1m.index[fi], f"{side}_content_class"] = "NA"
            df_1m.at[df_1m.index[fi], f"{side}_issues"] = "NA"
            if f"{side}_top_width_quality" in df_1m.columns:
                df_1m.at[df_1m.index[fi], f"{side}_top_width_quality"] = "interpolated"
            if len(valid_idx) >= 2:
                s_valid = df_1m["s"].values[valid_idx]
                s_gap = df_1m["s"].values[fi]
                for gcol in geom_cols:
                    full_col = f"{side}_{gcol}"
                    if full_col not in df_1m.columns:
                        continue
                    vals = df_1m[full_col].values[valid_idx]
                    good = ~np.isnan(vals)
                    if good.sum() >= 2:
                        val_interp = float(np.interp(s_gap, s_valid[good], vals[good]))
                        df_1m.at[df_1m.index[fi], full_col] = val_interp
            inherited_priority = _nearest_confirmed_gap_priority(df_1m, side, fi)
            df_1m.at[df_1m.index[fi], f"{side}_tdok_priority"] = inherited_priority
            if f"{side}_priority_source" in df_1m.columns:
                df_1m.at[df_1m.index[fi], f"{side}_priority_source"] = (
                    "gap_inherited" if inherited_priority != "NA" else "gap_inherited_na"
                )
            # Interpolated fills have no fresh evidence - assign a low but
            # non-zero confidence so downstream analysis can distinguish
            # them from outright rejections.
            if f"{side}_confidence" in df_1m.columns:
                df_1m.at[df_1m.index[fi], f"{side}_confidence"] = 0.20
            if f"{side}_pass_source" in df_1m.columns:
                df_1m.at[df_1m.index[fi], f"{side}_pass_source"] = "gap_interpolated"
            n_interpolated += 1

    print(f"  {side.capitalize()}: {n_gaps} gap(s) - {n_redetected} re-detected, "
          f"{n_interpolated} interpolated")

# Promote sustained slope deformation only when it is spatially coherent.
# A single 1 m SLOPE_ISSUE is too sensitive for M-level maintenance, but a
# continuous run indicates a real side-slope condition rather than local noise.
print("\n[Post] Priority refinement: sustained slope-issue runs...")
for side in ["left", "right"]:
    exists_col = f"{side}_ditch_exists"
    issues_col = f"{side}_issues"
    prio_col = f"{side}_tdok_priority"
    source_col = f"{side}_priority_source"
    pass_col = f"{side}_pass_source"

    issues = df_1m[issues_col].fillna("").astype(str)
    pass_src = df_1m[pass_col].fillna("").astype(str) if pass_col in df_1m.columns else ""
    slope_mask = (
        df_1m[exists_col].eq(1)
        & issues.str.contains("SLOPE_ISSUE", regex=False)
        & df_1m[prio_col].eq("Å")
        & (pass_src != "gap_interpolated")
    )

    promoted = 0
    arr = slope_mask.to_numpy()
    start = None
    for i, is_slope in enumerate(arr):
        if is_slope and start is None:
            start = i
        is_end = i == len(arr) - 1
        if (not is_slope or is_end) and start is not None:
            end = i if (is_end and is_slope) else i - 1
            if end - start + 1 >= PRIORITY_SLOPE_RUN_MIN:
                idx = df_1m.index[start:end + 1]
                df_1m.loc[idx, prio_col] = "M"
                if source_col in df_1m.columns:
                    df_1m.loc[idx, source_col] = "measured_sustained_slope"
                promoted += len(idx)
            start = None
    print(f"  {side.capitalize()}: {promoted} section(s) promoted to M "
          f"from sustained SLOPE_ISSUE runs (min {PRIORITY_SLOPE_RUN_MIN} m)")

# Promote sustained severe drainage deficiency to V only when the condition is
# spatially coherent and supported by several evidence groups. V is therefore
# not a depth-threshold class; it is an urgent multi-evidence suggestion.
print("\n[Post] Priority refinement: sustained urgent drainage runs...")
for side in ["left", "right"]:
    exists_col = f"{side}_ditch_exists"
    issues_col = f"{side}_issues"
    prio_col = f"{side}_tdok_priority"
    depth_col = f"{side}_depth"
    source_col = f"{side}_priority_source"
    pass_col = f"{side}_pass_source"

    issues = df_1m[issues_col].fillna("").astype(str)
    pass_src = df_1m[pass_col].fillna("").astype(str) if pass_col in df_1m.columns else ""
    compound_obstruction = (
        issues.str.contains("STAGNANT_WATER", regex=False)
        & (
            issues.str.contains("SILTED", regex=False)
            | issues.str.contains("OVERGROWN", regex=False)
            | issues.str.contains("SLOPE_ISSUE", regex=False)
        )
    )
    deformation_plus_fill = (
        issues.str.contains("SLOPE_ISSUE", regex=False)
        & (
            issues.str.contains("SILTED", regex=False)
            | issues.str.contains("OVERGROWN", regex=False)
        )
    )
    strongest_compound = (
        issues.str.contains("STAGNANT_WATER", regex=False)
        & issues.str.contains("SLOPE_ISSUE", regex=False)
        & (
            issues.str.contains("SILTED", regex=False)
            | issues.str.contains("OVERGROWN", regex=False)
        )
    )
    urgent_mask = (
        df_1m[exists_col].eq(1)
        & df_1m[prio_col].eq("M")
        & df_1m[depth_col].lt(TDOK_SEVERE_DEPTH_BAND_M)
        & (strongest_compound | (compound_obstruction & deformation_plus_fill))
        & (pass_src != "gap_interpolated")
    )

    promoted = 0
    arr = urgent_mask.to_numpy()
    start = None
    for i, is_urgent in enumerate(arr):
        if is_urgent and start is None:
            start = i
        is_end = i == len(arr) - 1
        if (not is_urgent or is_end) and start is not None:
            end = i if (is_end and is_urgent) else i - 1
            if end - start + 1 >= PRIORITY_V_RUN_MIN:
                idx = df_1m.index[start:end + 1]
                df_1m.loc[idx, prio_col] = "V"
                if source_col in df_1m.columns:
                    df_1m.loc[idx, source_col] = "measured_sustained_urgent"
                promoted += len(idx)
            start = None
    print(f"  {side.capitalize()}: {promoted} section(s) promoted to V "
          f"from sustained urgent runs (min {PRIORITY_V_RUN_MIN} m)")

# Final consistency guard. Every confirmed measured ditch must carry issue and
# priority labels. This mainly protects boundary sections at the start/end of
# a run, where earlier continuity edits can leave an empty label even though
# the geometry itself is still confirmed.
for side in ["left", "right"]:
    exists_col = f"{side}_ditch_exists"
    issues_col = f"{side}_issues"
    prio_col = f"{side}_tdok_priority"
    source_col = f"{side}_priority_source"

    bad_label = (
        (df_1m[exists_col].fillna(0).astype(int) == 1)
        & (
            df_1m[issues_col].isna()
            | (df_1m[issues_col].astype(str).str.strip() == "")
            | df_1m[prio_col].isna()
            | (df_1m[prio_col].astype(str).str.strip() == "")
        )
    )
    for idx in df_1m.index[bad_label]:
        issues_fix = diagnose_issues(
            df_1m.at[idx, f"{side}_shape_class"],
            df_1m.at[idx, f"{side}_content_class"],
            df_1m.at[idx, f"{side}_depth"],
            df_1m.at[idx, f"{side}_inner_slope_deg"],
            df_1m.at[idx, f"{side}_outer_slope_deg"],
            df_1m.at[idx, f"{side}_asymmetry"],
            df_1m.at[idx, exists_col],
            bool(df_1m.at[idx, "zref_valid"]),
        )
        prio_fix = map_to_tdok_priority(issues_fix, df_1m.at[idx, f"{side}_depth"])
        df_1m.at[idx, issues_col] = ",".join(issues_fix)
        df_1m.at[idx, prio_col] = prio_fix
        if source_col in df_1m.columns:
            df_1m.at[idx, source_col] = "measured" if prio_fix != "NA" else "none"

# recompute ditch_presence after continuity filter + gap-filling
df_1m["ditch_presence"] = [
    get_ditch_presence(bool(row["left_ditch_exists"]), bool(row["right_ditch_exists"]))
    for _, row in df_1m.iterrows()
]

ditch_csv = os.path.join(OUT_DIR, "ditch_metrics_final.csv")
df_1m.to_csv(ditch_csv, index=False, float_format="%.4f")
print(f"  Sections analysed: {len(df_1m)}")
print(f"  Saved: {ditch_csv}")

# 10 m aggregation
# Aggregation rules (aligned with TDOK 2015:0155 Section 8.6 prioritisation):
# depth ->median over confirmed 1m sections
# shape_class ->majority vote (ties broken by worst case)
# content_class ->majority vote (ties broken by worst case)
# issues ->union of all 1m issue tags within the segment
# tdok_priority ->worst (highest urgency) priority across the segment
print("\nAggregating into 10 m segments...")

SHAPE_ORDER   = ["V_HEALTHY", "U_MODERATE", "FLAT_SILTED", "UNKNOWN_WATER_MASKED", "NA"]
CONTENT_ORDER = ["DRY_CLEAN", "DRY_VEGETATED", "WET_CLEAR",
                 "WET_PARTIAL_VEG", "WET_DENSE_VEG", "NA"]
PRIORITY_RANK = {"Ö": 1, "Å": 2, "M": 3, "V": 4, "NA": 0}
INV_PRIORITY  = {v: k for k, v in PRIORITY_RANK.items()}


def _majority_vote(labels, order):
    """Majority vote. On tie, pick the label that appears later in `order`
    (conservative: worst case wins)."""
    clean = [x for x in labels if isinstance(x, str) and x and x != "NA"]
    if not clean:
        return "NA"
    counts = {}
    for lab in clean:
        counts[lab] = counts.get(lab, 0) + 1
    max_n = max(counts.values())
    top = [lab for lab, n in counts.items() if n == max_n]
    if len(top) == 1:
        return top[0]
    # tie ->pick worst by `order`
    return max(top, key=lambda x: order.index(x) if x in order else -1)


def _union_issues(issues_series):
    """Union of comma-separated issue tags across a series."""
    tags = set()
    for s in issues_series:
        if not isinstance(s, str) or not s or s == "NA":
            continue
        for tok in s.split(","):
            tok = tok.strip()
            if tok and tok != "NA":
                tags.add(tok)
    if not tags:
        return "NA"
    # stable ordering for CSV readability
    order = ["HEALTHY", "SHALLOW", "SILTED", "WATER_MASKED_SHAPE",
             "OVERGROWN", "STAGNANT_WATER", "SLOPE_ISSUE"]
    return ",".join(sorted(tags, key=lambda x: order.index(x) if x in order else 99))


def _worst_priority(priority_series):
    """Highest-urgency priority across a segment."""
    ranks = [PRIORITY_RANK.get(p, 0) for p in priority_series
             if isinstance(p, str) and p]
    ranks = [r for r in ranks if r > 0]
    if not ranks:
        return "NA"
    return INV_PRIORITY[max(ranks)]


seg_start = np.floor(df_1m["s"].min() / SEGMENT_LEN) * SEGMENT_LEN
seg_end = np.ceil(df_1m["s"].max() / SEGMENT_LEN) * SEGMENT_LEN
seg_edges = np.arange(seg_start, seg_end + SEGMENT_LEN, SEGMENT_LEN)

seg_rows = []
for a, b in zip(seg_edges[:-1], seg_edges[1:]):
    seg = df_1m[(df_1m["s"] >= a) & (df_1m["s"] < b)]
    if len(seg) == 0:
        continue

    lv = seg[(seg["left_ditch_exists"] == 1) & seg["left_depth"].notna()]
    rv = seg[(seg["right_ditch_exists"] == 1) & seg["right_depth"].notna()]

    l_dep = float(lv["left_depth"].median()) if len(lv) >= 1 else np.nan
    r_dep = float(rv["right_depth"].median()) if len(rv) >= 1 else np.nan
    l_dvc = float(lv["left_dvci"].median()) if len(lv) >= 1 else np.nan
    r_dvc = float(rv["right_dvci"].median()) if len(rv) >= 1 else np.nan

    l_ratio = float((seg["left_ditch_exists"] == 1).mean())
    r_ratio = float((seg["right_ditch_exists"] == 1).mean())
    zref_vld_ratio = float(seg["zref_valid"].mean())

    # water flags at segment level
    l_water_seg = int(seg["left_water_flag"].sum()) if "left_water_flag" in seg else 0
    r_water_seg = int(seg["right_water_flag"].sum()) if "right_water_flag" in seg else 0

    l_present = len(lv) >= MIN_VALID_IN_SEGMENT
    r_present = len(rv) >= MIN_VALID_IN_SEGMENT

    # Aggregated shape / content / issues / priority
    l_shape   = _majority_vote(lv["left_shape_class"].tolist(),   SHAPE_ORDER)
    r_shape   = _majority_vote(rv["right_shape_class"].tolist(),  SHAPE_ORDER)
    l_content = _majority_vote(lv["left_content_class"].tolist(), CONTENT_ORDER)
    r_content = _majority_vote(rv["right_content_class"].tolist(), CONTENT_ORDER)
    l_issues  = _union_issues(lv["left_issues"].tolist())
    r_issues  = _union_issues(rv["right_issues"].tolist())
    l_prio    = _worst_priority(lv["left_tdok_priority"].tolist())
    r_prio    = _worst_priority(rv["right_tdok_priority"].tolist())

    # Segment-level priority = worst across both sides
    seg_prio = _worst_priority([l_prio, r_prio])
    seg_presence = get_ditch_presence(l_present, r_present)

    seg_rows.append({
        "s_start": float(a),
        "s_end": float(b),
        "s_mid": float(0.5 * (a + b)),
        "n_sections": len(seg),
        "left_ditch_ratio": l_ratio,
        "right_ditch_ratio": r_ratio,
        "left_depth_median": l_dep,
        "right_depth_median": r_dep,
        "left_dvci_median": l_dvc,
        "right_dvci_median": r_dvc,
        "left_water_sections": l_water_seg,
        "right_water_sections": r_water_seg,
        "segment_ditch_presence": seg_presence,
        "left_shape_class": l_shape,
        "right_shape_class": r_shape,
        "left_content_class": l_content,
        "right_content_class": r_content,
        "left_issues": l_issues,
        "right_issues": r_issues,
        "left_tdok_priority": l_prio,
        "right_tdok_priority": r_prio,
        "segment_tdok_priority": seg_prio,
        "zref_valid_ratio": zref_vld_ratio,
    })

df_10m = pd.DataFrame(seg_rows)
# Grading output disabled for this workflow.
# df_10m is still built so figures that use it keep working, but we
# don't export condition_segments.csv or the grade bar chart.
print(f"  10 m segments (internal): {len(df_10m)}")

# =========================================================
# FIGURES
# =========================================================
print("\nGenerating figures...")

# Figure 1: Longitudinal depth profiles
fig, ax = plt.subplots(figsize=(15, 5))
mL = (df_1m["left_ditch_exists"] == 1) & df_1m["left_depth"].notna()
mR = (df_1m["right_ditch_exists"] == 1) & df_1m["right_depth"].notna()

ax.plot(df_1m.loc[mL, "s"], df_1m.loc[mL, "left_depth"],
        ".", ms=3, alpha=0.5, label="Left ditch depth (1 m)")
ax.plot(df_1m.loc[mR, "s"], df_1m.loc[mR, "right_depth"],
        ".", ms=3, alpha=0.5, label="Right ditch depth (1 m)")

if mL.sum() >= 5:
    ax.plot(df_1m.loc[mL, "s"],
            make_savgol(df_1m.loc[mL, "left_depth"].values, max_win=31, poly=2),
            lw=2, alpha=0.9, label="Left (smoothed)")
if mR.sum() >= 5:
    ax.plot(df_1m.loc[mR, "s"],
            make_savgol(df_1m.loc[mR, "right_depth"].values, max_win=31, poly=2),
            lw=2, alpha=0.9, label="Right (smoothed)")

deep_L = df_1m[mL & (df_1m["left_deep_flag"] == 1)]
deep_R = df_1m[mR & (df_1m["right_deep_flag"] == 1)]
if len(deep_L):
    ax.scatter(deep_L["s"], deep_L["left_depth"],
               marker="^", s=60, color="steelblue", zorder=5,
               label=f"Left deep (>{DEPTH_DEEP_FLAG} m)")
if len(deep_R):
    ax.scatter(deep_R["s"], deep_R["right_depth"],
               marker="^", s=60, color="tomato", zorder=5,
               label=f"Right deep (>{DEPTH_DEEP_FLAG} m)")

# water flagged sections
water_L = df_1m[mL & (df_1m["left_water_flag"] == 1)]
water_R = df_1m[mR & (df_1m["right_water_flag"] == 1)]
if len(water_L):
    ax.scatter(water_L["s"], water_L["left_depth"],
               marker="v", s=40, color="cyan", zorder=4, alpha=0.7,
               label=f"Left water suspect ({len(water_L)})")
if len(water_R):
    ax.scatter(water_R["s"], water_R["right_depth"],
               marker="v", s=40, color="deepskyblue", zorder=4, alpha=0.7,
               label=f"Right water suspect ({len(water_R)})")

# TDOK 2015:0155 §10.13 depth thresholds (measured below RUK)
for depth, lbl, col in [
    (TDOK_DEPTH_TARGET_M,
     f"Target >= {TDOK_DEPTH_TARGET_M} m (TDOK §10.13 post-cleaning)",
     "green"),
    (TDOK_DEPTH_SUFFICIENT_M,
     f"Sufficient >= {TDOK_DEPTH_SUFFICIENT_M} m (TDOK §10.13 SHALLOW cut-off)",
     "orange"),
]:
    ax.axhline(depth, color=col, lw=1, ls="--", alpha=0.7, label=lbl)

ax.axhline(DEPTH_DEEP_FLAG, color="purple", lw=1.2, ls="-.",
           alpha=0.7, label=f"Deep flag {DEPTH_DEEP_FLAG} m")

ax.set_xlabel("S - along-track distance (m)")
ax.set_ylabel("Ditch depth below RUK (m)")
ax.set_ylim(0, DEPTH_VALID_MAX + 0.1)
ax.legend(fontsize=7, ncol=4)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_depth_profile_new.png"), dpi=FIG_DPI)
plt.close(fig)

# Figure 2: Ditch presence
fig, ax = plt.subplots(figsize=(15, 3.5))
py = {"NONE": 1, "LEFT_ONLY": 2, "RIGHT_ONLY": 3, "BOTH": 4}
pc = {"NONE": "lightgray", "LEFT_ONLY": "steelblue",
      "RIGHT_ONLY": "tomato", "BOTH": "green"}
for _, r in df_10m.iterrows():
    p = r["segment_ditch_presence"]
    ax.plot([r["s_start"], r["s_end"]], [py.get(p, 0), py.get(p, 0)],
            lw=7, color=pc.get(p, "gray"), solid_capstyle="butt")
ax.set_xlabel("S (m)")
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(["NONE", "LEFT_ONLY", "RIGHT_ONLY", "BOTH"])
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_ditch_presence_new.png"), dpi=FIG_DPI)
plt.close(fig)

# Figure 3: Example cross-sections
s_examples = select_example_ditch_sections(df_1m)
fig, axes = plt.subplots(len(s_examples), 1, figsize=(14, 6 * len(s_examples)))
if len(s_examples) == 1:
    axes = [axes]

for ax, s0 in zip(axes, s_examples):
    ref_idx = int(np.argmin(np.abs(ref["s"].values - s0)))
    ref_row = ref.iloc[ref_idx]

    s_ref = float(ref_row["s"])
    t_ref = float(ref_row[t_col])
    d_ref = float(ref_row[d_col])

    i0, i1 = sorted_slice_bounds(
        Sg,
        s_ref - ROUGH_S_PRESELECT_HALF,
        s_ref + ROUGH_S_PRESELECT_HALF
    )
    S_sub, T_sub = Sg[i0:i1], Tg[i0:i1]
    Z_sub, I_sub = Zg[i0:i1], Ig[i0:i1]

    tm = np.abs(T_sub - t_ref) <= ROUGH_T_PRESELECT_HALF
    S_sub, T_sub = S_sub[tm], T_sub[tm]
    Z_sub, I_sub = Z_sub[tm], I_sub[tm]

    u, v = project_to_local_frame(S_sub, T_sub, s_ref, t_ref, d_ref)
    sm = (np.abs(u) <= LOCAL_SECTION_HALF_U) & (np.abs(v) <= ENV_V_RANGE)
    v_sec, z_sec, i_sec = v[sm], Z_sub[sm], I_sub[sm]

    env_v, env_z, _, _ = build_lower_envelope(
        v_sec, z_sec, i_vals=i_sec,
        v_range=ENV_V_RANGE, bin_w=ENV_BIN_W, q=ENV_Q
    )

    row_1m = df_1m.iloc[int(np.argmin(np.abs(df_1m["s"].values - s0)))]
    z_ref = row_1m["Z_ref"]
    half_g = row_1m["half_gauge_prior"]
    l_d = row_1m["left_depth"]
    r_d = row_1m["right_depth"]
    l_w = row_1m["left_width"]
    r_w = row_1m["right_width"]

    sc = ax.scatter(v_sec, z_sec, c=i_sec, cmap="plasma_r",
                    s=0.4, alpha=0.25, vmin=0, vmax=10000)
    plt.colorbar(sc, ax=ax, label="Intensity")

    if len(env_v):
        ax.plot(env_v, env_z, color="darkred", lw=2.5, label="P5 morph. envelope")
    if not np.isnan(z_ref):
        ax.axhline(z_ref, color="cyan", lw=1.5, ls="--",
                   label=f"Z_ref={z_ref:.3f} m")

    ax.axvline(-half_g, color="lime", lw=2.0, ls="--",
               label=f"rail +/-{half_g:.3f} m")
    ax.axvline(+half_g, color="lime", lw=2.0, ls="--")
    ax.axvline(0, color="gray", lw=1, ls=":")
    ax.axvspan(-SIDE_SEARCH_MAX, -SIDE_SEARCH_MIN,
               alpha=0.06, color="cyan", label="left search zone")
    ax.axvspan(+SIDE_SEARCH_MIN, +SIDE_SEARCH_MAX,
               alpha=0.06, color="lime", label="right search zone")

    if not np.isnan(row_1m["left_width"]):
        ax.axvspan(
            -row_1m["left_span_x1"],
            -row_1m["left_span_x0"],
            alpha=0.15, color="steelblue", label="left ditch span"
        )
    if "left_top_width" in row_1m and not np.isnan(row_1m["left_top_width"]):
        ax.axvspan(
            -row_1m["left_top_span_x1"],
            -row_1m["left_top_span_x0"],
            alpha=0.08, color="navy", label="left top-bank span"
        )
    if not np.isnan(row_1m["right_width"]):
        ax.axvspan(
            row_1m["right_span_x0"],
            row_1m["right_span_x1"],
            alpha=0.15, color="tomato", label="right ditch span"
        )
    if "right_top_width" in row_1m and not np.isnan(row_1m["right_top_width"]):
        ax.axvspan(
            row_1m["right_top_span_x0"],
            row_1m["right_top_span_x1"],
            alpha=0.08, color="darkred", label="right top-bank span"
        )

    # water flag indicator
    lw_flag = "WATER" if row_1m.get("left_water_flag", 0) else ""
    rw_flag = "WATER" if row_1m.get("right_water_flag", 0) else ""

    ax.set_xlim(-ENV_V_RANGE, ENV_V_RANGE)
    ax.set_xlabel("Local cross-track v (m)  [0 = centreline]")
    ax.set_ylabel("Z (m)")
    ld_str = "NaN" if np.isnan(l_d) else f"{l_d:.3f}"
    rd_str = "NaN" if np.isnan(r_d) else f"{r_d:.3f}"
    lw_str = "NaN" if np.isnan(l_w) else f"{l_w:.2f}"
    rw_str = "NaN" if np.isnan(r_w) else f"{r_w:.2f}"
    ax.set_title(f"S = {row_1m['s']:.0f} m", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", ncol=3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_example_sections_new.png"), dpi=FIG_DPI)
plt.close(fig)

# Condition grade strip (Fig 4) disabled for this workflow.
# Kept as a comment for reference in case it's needed later.

# Figure 5: Water detection diagnostic
fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
ax = axes[0]
ax.bar(df_1m["s"], df_1m["left_water_frac"], width=0.9, alpha=0.7,
       color="steelblue", label="Left water fraction")
ax.bar(df_1m["s"], df_1m["right_water_frac"], width=0.9, alpha=0.5,
       color="tomato", label="Right water fraction", bottom=0)
ax.set_ylabel("Water suspect fraction")
ax.set_title("Water / low-density detection per section")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(df_1m["s"], df_1m["left_water_flag"], ".", ms=3, color="steelblue",
        alpha=0.7, label="Left water flag")
ax.plot(df_1m["s"], df_1m["right_water_flag"] + 0.05, ".", ms=3, color="tomato",
        alpha=0.7, label="Right water flag")
ax.set_ylabel("Water flag")
ax.set_xlabel("S (m)")
ax.set_yticks([0, 1])
ax.set_yticklabels(["No", "Yes"])
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_water_detection.png"), dpi=FIG_DPI)
plt.close(fig)

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("STEP 07 - SUMMARY")
print("=" * 60)
print(f"1 m output:  {ditch_csv}")
print(f"Sections analysed: {len(df_1m)}")

print(f"\nCentreline field used: {t_col}")
print(f"dT/dS field used:      {d_col}")
print(f"Z_ref field used:      {zref_col}")

print(f"\nEnvelope method: morphological opening (kernel={MORPH_KERNEL_SIZE} bins)")
print(f"Continuity filter: min {DITCH_MIN_RUN_LENGTH} consecutive sections (run-length filter)")
print(f"Shape validation: inner slope {DITCH_INNER_SLOPE_MIN_DEG}-{DITCH_INNER_SLOPE_MAX_DEG} deg, "
      f"top-bank width {TOP_BANK_MIN_WIDTH}-{TOP_BANK_MAX_WIDTH}m")
print("Classification framework (dual-axis, multi-label):")
print(f"  Depth datum:  RUK (TDOK 2015:0155 §10.13)")
print(f"  SHALLOW if depth < {TDOK_DEPTH_SUFFICIENT_M} m below RUK; "
      f"target >= {TDOK_DEPTH_TARGET_M} m")
print(f"  Shape class:   V_HEALTHY / U_MODERATE / FLAT_SILTED for dry sections; "
      f"UNKNOWN_WATER_MASKED when water_flag is set "
      f"(dry flatness < {SHAPE_FLATNESS_U_MIN} / "
      f"< {SHAPE_FLATNESS_FLAT_MIN} / >= {SHAPE_FLATNESS_FLAT_MIN})")
print(f"  Content class (Roelens 2018): DRY_CLEAN / DRY_VEGETATED / "
      f"WET_CLEAR / WET_PARTIAL_VEG / WET_DENSE_VEG")
print(f"  Issues (multi-label): HEALTHY / SHALLOW / SILTED / WATER_MASKED_SHAPE "
      f"/ OVERGROWN / STAGNANT_WATER / SLOPE_ISSUE")
print(f"  TDOK priority: V (2 wk) / M (3 mo) / Å (3 yr) / Ö (opportunistic)")

print("\nSection ditch presence:")
print(df_1m["ditch_presence"].value_counts(dropna=False).to_string())
print("\nLeft terrain type:")
print(df_1m["left_type"].value_counts(dropna=False).to_string())
print("\nRight terrain type:")
print(df_1m["right_type"].value_counts(dropna=False).to_string())

mL = (df_1m["left_ditch_exists"] == 1) & df_1m["left_depth"].notna()
mR = (df_1m["right_ditch_exists"] == 1) & df_1m["right_depth"].notna()

print(f"\nLeft DITCH: {int(mL.sum())} sections")
if mL.sum() > 0:
    print(f"  Median depth : {df_1m.loc[mL, 'left_depth'].median():.3f} m")
    print(f"  Median width : {df_1m.loc[mL, 'left_width'].median():.3f} m")
    if "left_top_width" in df_1m.columns:
        tw_l = df_1m.loc[mL, "left_top_width"].dropna()
        if len(tw_l) > 0:
            print(f"  Median top width : {tw_l.median():.3f} m "
                  f"({len(tw_l)} estimated)")
    print(f"  Median area  : {df_1m.loc[mL, 'left_area'].median():.4f} m2")
    n_deep_l = int(df_1m.loc[mL, 'left_deep_flag'].sum())
    if n_deep_l > 0:
        print(f"  Deep sections (>{DEPTH_DEEP_FLAG} m): {n_deep_l}")
    n_water_l = int(df_1m.loc[mL, 'left_water_flag'].sum())
    if n_water_l > 0:
        print(f"  Water suspect sections: {n_water_l}")

print(f"\nRight DITCH: {int(mR.sum())} sections")
if mR.sum() > 0:
    print(f"  Median depth : {df_1m.loc[mR, 'right_depth'].median():.3f} m")
    print(f"  Median width : {df_1m.loc[mR, 'right_width'].median():.3f} m")
    if "right_top_width" in df_1m.columns:
        tw_r = df_1m.loc[mR, "right_top_width"].dropna()
        if len(tw_r) > 0:
            print(f"  Median top width : {tw_r.median():.3f} m "
                  f"({len(tw_r)} estimated)")
    print(f"  Median area  : {df_1m.loc[mR, 'right_area'].median():.4f} m2")
    n_deep_r = int(df_1m.loc[mR, 'right_deep_flag'].sum())
    if n_deep_r > 0:
        print(f"  Deep sections (>{DEPTH_DEEP_FLAG} m): {n_deep_r}")
    n_water_r = int(df_1m.loc[mR, 'right_water_flag'].sum())
    if n_water_r > 0:
        print(f"  Water suspect sections: {n_water_r}")

# total water
total_water_l = int(df_1m["left_water_flag"].sum())
total_water_r = int(df_1m["right_water_flag"].sum())
print(f"\nWater detection:")
print(f"  Left:  {total_water_l}/{len(df_1m)} sections flagged ({total_water_l/len(df_1m)*100:.1f}%)")
print(f"  Right: {total_water_r}/{len(df_1m)} sections flagged ({total_water_r/len(df_1m)*100:.1f}%)")

print("\n10 m segment ditch presence:")
print(df_10m["segment_ditch_presence"].value_counts(dropna=False).to_string())

print("\nKey improvements:")
print(f"  1. Morphological opening envelope (Roelens 2016/2018) - preserves narrow ditch depth")
print(f"  2. Water/low-density detection (NIR absorption, USGS 2024)")
print(f"  3. TDOK 2015:0155 §10.13 aligned diagnosis - depth below RUK,")
print(f"     thresholds >= {TDOK_DEPTH_SUFFICIENT_M}m sufficient / "
      f">= {TDOK_DEPTH_TARGET_M}m post-intervention target")
print(f"  4. Dual slope: inner + outer ditch slope computed")
print(f"  5. Dual-axis classification (shape + content) + multi-label issue set")
print(f"  6. Longitudinal continuity filter: min {DITCH_MIN_RUN_LENGTH} consecutive sections (run-length filter)")
print(f"  7. Shape validation: slope {DITCH_INNER_SLOPE_MIN_DEG}-{DITCH_INNER_SLOPE_MAX_DEG} deg, "
      f"top-bank width <= {TOP_BANK_MAX_WIDTH}m")

print("\nOutput figures:")
for f in [
    "Fig_depth_profile_new.png",
    "Fig_ditch_presence_new.png",
    "Fig_example_sections_new.png",
    "Fig_water_detection.png",
]:
    print(f"  {f}")

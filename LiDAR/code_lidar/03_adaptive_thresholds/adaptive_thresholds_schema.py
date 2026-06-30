"""
Adaptive Threshold Schema for the railway ditch-detection pipeline (step07).

This module defines:
  • the metadata model (ThresholdSpec) carried by every detection /
    classification threshold used in step07;
  • the 7-category taxonomy that classifies each threshold by its
    justification source (regulatory / instrument / geometric /
    classification / procedural / search / sanity);
  • the THRESHOLD_REGISTRY — a single source of truth that lists all
    ~40 thresholds, their default literature values, and the rule each
    one follows when the adaptive pipeline computes it from data;
  • I/O helpers for serialising/deserialising the registry to/from JSON
    so step07 can consume the result with one `json.load()` at startup.

The actual *computation* of adaptive values lives in
`compute_adaptive_thresholds_v2.py` — this file is **pure metadata**
plus very small data-class plumbing (no LiDAR I/O, no NumPy work).

Design rationale
----------------
• Every threshold becomes auditable: the JSON output records `value`,
  `source`, `rule`, `reference`, and `n_samples` for each one.  An
  examiner can reproduce any number from the data.

• Splitting metadata (this file) from computation
  (compute_adaptive_thresholds_v2.py) lets step07 import the schema
  without having to import LiDAR-loading code, and lets unit tests
  introspect the registry without touching disk.

• Categories are ordered by "how data-driven the value should be":
    A — REGULATORY     never adaptive (fixed by a standard)
    B — INSTRUMENT     always adaptive from σ_noise
    C — GEOMETRIC      adaptive from prior detection bootstrap
    D — CLASSIFICATION adaptive from prior shape/content bootstrap
    E — PROCEDURAL     derived from B/C, or fixed engineering
    F — SEARCH         fixed engineering (corridor geometry)
    G — SANITY         fixed engineering (validity bounds)

References
----------
Pukelsheim 1994 — three-sigma rule
Lehmann 2013 — 3σ-rule for outlier detection
Tukey 1977 — Exploratory Data Analysis (percentile bootstrap)
Cui et al. 2021 — terrain-adaptive LiDAR filtering
Yan et al. 2024 — density-based multilevel terrain noise removal
Roelens et al. 2018 — Drainage ditch extraction from airborne LiDAR
Cazorzi et al. 2013 — Drainage network detection
Bailly et al. 2008 — LiDAR linear features (drainage networks)
Sofia et al. 2011 — channel network statistical thresholds
Soille 2004 — Morphological Image Analysis (closing)
Knighton 1981 — A1 channel asymmetry index
Meyer & Neto 2008 — Excess Green index
Šašak et al. 2023; Velesaca et al. 2020 — saturation-based ground class.
Trafikverket TDOK 2015:0155 — Tillståndsbedömning av avvattnings-
   anläggningar (§8.4.1, §10.13, Bilaga 2)
Rail Baltica RBDG-MAN-016 (2025) — Hydraulic, Drainage and Culverts
Virtanen et al. 2020 — SciPy 1.0 (find_peaks / peak_prominences)
Arastounia 2018 — paired-rail gauge search (Z_ref smoothing weights)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# ════════════════════════════════════════════════════════════════════════
# Enums — category and provenance
# ════════════════════════════════════════════════════════════════════════
class Category(str, Enum):
    REGULATORY     = "A_regulatory"
    INSTRUMENT     = "B_instrument_noise"
    GEOMETRIC      = "C_geometric_bootstrap"
    CLASSIFICATION = "D_classification_bootstrap"
    PROCEDURAL     = "E_procedural"
    SEARCH         = "F_search_zone"
    SANITY         = "G_sanity"


class Source(str, Enum):
    SIGMA_NOISE          = "sigma_noise"          # derived from σ
    FALSE_DEPRESSION_IQR = "false_depression_iqr" # flat-profile Q3 + 1.5 IQR
    BOOTSTRAP            = "bootstrap_percentile" # from prior detections
    LITERATURE           = "literature_fixed"     # immutable from a paper/standard
    LITERATURE_FALLBACK  = "literature_fallback"  # used in cold start
    DERIVED              = "derived"              # function of other thresholds
    ENGINEERING          = "engineering_judgment" # fixed by domain reasoning


# ════════════════════════════════════════════════════════════════════════
# Dataclass — every threshold carries this metadata bundle
# ════════════════════════════════════════════════════════════════════════
@dataclass
class ThresholdSpec:
    """A single ditch-detection threshold and the provenance of its value."""
    name: str
    category: Category
    source: Source
    rule: str                              # human-readable formula
    reference: str                         # literature key
    units: str = ""
    description: str = ""
    used_by: list = field(default_factory=list)  # step07 functions that read it

    # ── Filled in by compute_adaptive_thresholds_v2.py ───────────────
    value: Optional[float] = None
    n_samples: Optional[int] = None
    bootstrap_iter: int = 0                # 0 = literature/cold start

    # ── Convenience ──────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["source"]   = self.source.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ThresholdSpec":
        d = dict(d)
        d["category"] = Category(d["category"])
        d["source"]   = Source(d["source"])
        return cls(**d)


# ════════════════════════════════════════════════════════════════════════
# THRESHOLD REGISTRY
#   The complete, single-source-of-truth list of every threshold step07
#   uses.  Default values reflect the LITERATURE fall-back used at cold
#   start.  Adaptive computation overwrites `value`, `source`,
#   `n_samples`, and `bootstrap_iter`.
# ════════════════════════════════════════════════════════════════════════

def _make_registry() -> dict[str, ThresholdSpec]:
    R: dict[str, ThresholdSpec] = {}

    def add(spec: ThresholdSpec):
        R[spec.name] = spec

    # ─────────────────────────────────────────────────────────────────
    # CATEGORY A — REGULATORY (immutable, tied to standards)
    # ─────────────────────────────────────────────────────────────────
    add(ThresholdSpec(
        name="TDOK_DEPTH_SUFFICIENT_M", category=Category.REGULATORY,
        source=Source.LITERATURE, value=1.0, units="m",
        rule="fixed",
        reference="Trafikverket TDOK 2015:0155 §10.13 (grunt dike threshold)",
        description="Below RUK; SHALLOW issue triggered if measured depth < this.",
        used_by=["diagnose_issues"],
    ))
    add(ThresholdSpec(
        name="TDOK_DEPTH_TARGET_M", category=Category.REGULATORY,
        source=Source.LITERATURE, value=1.3, units="m",
        rule="fixed",
        reference="Trafikverket TDOK 2015:0155 §10.13 (post-maintenance target)",
        description="Recommended ditch depth after maintenance below RUK.",
        used_by=["reporting"],
    ))
    add(ThresholdSpec(
        name="EXG_VEG_THRESHOLD", category=Category.REGULATORY,
        source=Source.LITERATURE, value=20.0, units="ExG (8-bit RGB)",
        rule="fixed",
        reference="Meyer & Neto 2008 — Excess Green vegetation index",
        description="Per-point ExG > this → vegetation; used by DVCI computation.",
        used_by=["compute_dvci_for_span"],
    ))

    # ─────────────────────────────────────────────────────────────────
    # CATEGORY B — INSTRUMENT noise (σ-derived)
    # ─────────────────────────────────────────────────────────────────
    add(ThresholdSpec(
        name="DITCH_MIN_DEPTH", category=Category.INSTRUMENT,
        source=Source.LITERATURE_FALLBACK, value=0.08, units="m",
        rule="max(6 σ_noise, Q3(false local depressions) + 1.5 IQR)",
        reference="Pukelsheim 1994; Lehmann 2013; Tukey 1977; Roelens 2018; Sofia 2011",
        description=(
            "Local lower-envelope depression significance threshold used for "
            "candidate detection. This is not the regulatory ditch depth below RUK."
        ),
        used_by=["_try_find_ditch (Pass 1-4)"],
    ))
    add(ThresholdSpec(
        name="DITCH_MIN_PROM", category=Category.INSTRUMENT,
        source=Source.LITERATURE_FALLBACK, value=0.05, units="m",
        rule="max(1.5 σ_noise, Q3(false local depressions) + 1.5 IQR)",
        reference="Virtanen 2020; Tukey 1977; Roelens 2018; Sofia 2011",
        description="Minimum local peak prominence on the inverted lower envelope.",
        used_by=["_try_find_ditch"],
    ))
    add(ThresholdSpec(
        name="SHAPE_OUTER_RISE_MIN", category=Category.INSTRUMENT,
        source=Source.LITERATURE_FALLBACK, value=0.03, units="m",
        rule="max(3 σ_noise, 0.5 × false-depression threshold)",
        reference="Pukelsheim 1994; Tukey 1977",
        description="Absolute floor for outer-bank rise (small, shallow ditches).",
        used_by=["_try_find_ditch"],
    ))
    add(ThresholdSpec(
        name="DITCH_MIN_AREA", category=Category.INSTRUMENT,
        source=Source.DERIVED, value=0.01, units="m²",
        rule="max(0.5 × DITCH_MIN_WIDTH × DITCH_MIN_DEPTH, 0.005)",
        reference="derived from Cat B + Cat C",
        description="Cross-section area lower bound (kills slivers).",
        used_by=["_try_find_ditch"],
    ))
    add(ThresholdSpec(
        name="WATER_DENSITY_RATIO", category=Category.INSTRUMENT,
        source=Source.LITERATURE_FALLBACK, value=0.25, units="ratio",
        rule="P25 of local bin-count distribution / median bin count",
        reference="Roelens 2018 §3.2 (NIR absorption water signature)",
        description="Bin counts < this fraction of local median → water-suspect.",
        used_by=["detect_water_in_ditch"],
    ))
    add(ThresholdSpec(
        name="WATER_INTENSITY_MAX", category=Category.INSTRUMENT,
        source=Source.LITERATURE_FALLBACK, value=800.0, units="intensity (8-bit eq.)",
        rule="P10 of corridor intensity (when adaptive)",
        reference="Roelens 2018; Velesaca 2020 (low-intensity water proxy)",
        description="Intensity below this in low-density bins → water support.",
        used_by=["detect_water_in_ditch"],
    ))
    add(ThresholdSpec(
        name="ENV_BIN_W", category=Category.INSTRUMENT,
        source=Source.ENGINEERING, value=0.05, units="m",
        rule="fixed (≈ MLS lateral footprint)",
        reference="engineering — Cirrus MLS spec",
        description="Envelope binning width across track.",
        used_by=["build_lower_envelope"],
    ))
    add(ThresholdSpec(
        name="ENV_MIN_POINTS_PER_BIN", category=Category.INSTRUMENT,
        source=Source.ENGINEERING, value=3, units="points",
        rule="fixed (sparse-bin gating)",
        reference="engineering",
        description="Minimum points per envelope bin to compute P5.",
        used_by=["build_lower_envelope"],
    ))

    # ─────────────────────────────────────────────────────────────────
    # CATEGORY C — GEOMETRIC (bootstrap from detected width / slope)
    # ─────────────────────────────────────────────────────────────────
    add(ThresholdSpec(
        name="DITCH_MIN_WIDTH", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=0.20, units="m",
        rule="max(0.20, P1(detected widths) − 0.05)",
        reference="Tukey 1977 + TDOK 2015:0155 Bilaga 2; Roelens 2018 (~0.25 m)",
        description="Minimum accepted detection width.",
        used_by=["_try_find_ditch"],
    ))
    add(ThresholdSpec(
        name="DITCH_MAX_WIDTH", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=4.0, units="m",
        rule="min(6.0, P99(detected widths) + 0.5)",
        reference="Tukey 1977 + TDOK 2015:0155 Bilaga 2; Rail Baltica RBDG-MAN-016",
        description="Maximum accepted detection width (rejects embankment toe).",
        used_by=["_try_find_ditch"],
    ))
    add(ThresholdSpec(
        name="TOP_BANK_MIN_WIDTH", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=0.20, units="m",
        rule="max(0.20, P1(top-of-bank widths) - 0.20)",
        reference="Tukey 1977; Celik 2014 top-of-bank ditch geometry; Passalacqua 2012 channel morphometry",
        description="Lower sanity bound for reported top-of-bank ditch width.",
        used_by=["estimate_top_bank_width"],
    ))
    add(ThresholdSpec(
        name="TOP_BANK_MAX_WIDTH", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=8.00, units="m",
        rule="min(8.00, P99(top-of-bank widths) + 0.50)",
        reference="Tukey 1977; Celik 2014; Passalacqua 2012",
        description="Upper sanity bound for reported top-of-bank ditch width.",
        used_by=["estimate_top_bank_width"],
    ))
    add(ThresholdSpec(
        name="TOP_BANK_SEARCH_MAX_M", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=3.0, units="m",
        rule="clip(0.5 * TOP_BANK_MAX_WIDTH + 0.25, 1.5, 4.0)",
        reference="derived from bootstrap top-of-bank width distribution",
        description="Maximum one-sided search distance from bottom to bank top.",
        used_by=["estimate_top_bank_width"],
    ))
    add(ThresholdSpec(
        name="TOP_BANK_RECOVERY_FRAC", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=0.80, units="fraction",
        rule="clip(1 - false_depression_threshold / median_confirmed_depth, 0.70, 0.90)",
        reference="Tukey 1977 false-depression threshold + confirmed ditch-depth distribution",
        description="Required fraction of local bank-height recovery before a point can be a bank top.",
        used_by=["estimate_top_bank_width"],
    ))
    add(ThresholdSpec(
        name="TOP_BANK_FLAT_SLOPE_DEG", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=8.0, units="deg",
        rule="P90(median absolute slope of flattest lower-envelope side profiles)",
        reference="slope-break/top-of-bank criterion; Celik 2014; Passalacqua 2012; Tukey 1977",
        description="Slope below which recovered terrain is considered locally flat enough for a bank top.",
        used_by=["estimate_top_bank_width"],
    ))
    add(ThresholdSpec(
        name="DITCH_INNER_SLOPE_MIN_DEG", category=Category.GEOMETRIC,
        source=Source.DERIVED, value=3.0, units="deg",
        rule="max(3.0, SLOPE_ANOMALY_INNER_MIN - 3.0)",
        reference="derived from bootstrap slope anomaly band",
        description="Broad inner-bank detection sanity gate derived from the adaptive anomaly band.",
        used_by=["_try_find_ditch"],
    ))
    add(ThresholdSpec(
        name="DITCH_INNER_SLOPE_MAX_DEG", category=Category.GEOMETRIC,
        source=Source.DERIVED, value=80.0, units="deg",
        rule="min(80.0, SLOPE_ANOMALY_INNER_MAX + 30.0)",
        reference="derived from bootstrap slope anomaly band",
        description="Broad inner-bank detection sanity gate derived from the adaptive anomaly band.",
        used_by=["_try_find_ditch"],
    ))
    add(ThresholdSpec(
        name="DITCH_OUTER_SLOPE_MIN_DEG", category=Category.GEOMETRIC,
        source=Source.DERIVED, value=3.0, units="deg",
        rule="max(3.0, SLOPE_ANOMALY_OUTER_MIN - 3.0)",
        reference="derived from bootstrap slope anomaly band",
        description="Broad outer-bank detection sanity gate derived from the adaptive anomaly band.",
        used_by=["_try_find_ditch"],
    ))
    add(ThresholdSpec(
        name="DITCH_OUTER_SLOPE_MAX_DEG", category=Category.GEOMETRIC,
        source=Source.DERIVED, value=80.0, units="deg",
        rule="min(80.0, SLOPE_ANOMALY_OUTER_MAX + 30.0)",
        reference="derived from bootstrap slope anomaly band",
        description="Broad outer-bank detection sanity gate derived from the adaptive anomaly band.",
        used_by=["_try_find_ditch"],
    ))
    add(ThresholdSpec(
        name="SLOPE_ANOMALY_INNER_MIN", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=8.0, units="°",
        rule="P5(confirmed inner slopes)",
        reference="Knighton 1981; Roelens 2018 §3.2; Rail Baltica RBDG-MAN-016",
        description="Inner-slope SLOPE_ISSUE anomaly band lower (sediment-flattened).",
        used_by=["diagnose_issues"],
    ))
    add(ThresholdSpec(
        name="SLOPE_ANOMALY_INNER_MAX", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=55.0, units="°",
        rule="P95(confirmed inner slopes)",
        reference="Knighton 1981; Rail Baltica RBDG-MAN-016",
        description="Inner-slope SLOPE_ISSUE anomaly band upper (collapsed wall).",
        used_by=["diagnose_issues"],
    ))
    add(ThresholdSpec(
        name="SLOPE_ANOMALY_OUTER_MIN", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=8.0, units="°",
        rule="P5(confirmed outer slopes)",
        reference="Knighton 1981; Rail Baltica RBDG-MAN-016",
        description="Outer-slope anomaly band lower.",
        used_by=["diagnose_issues"],
    ))
    add(ThresholdSpec(
        name="SLOPE_ANOMALY_OUTER_MAX", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=55.0, units="°",
        rule="P95(confirmed outer slopes)",
        reference="Knighton 1981; Rail Baltica RBDG-MAN-016",
        description="Outer-slope anomaly band upper.",
        used_by=["diagnose_issues"],
    ))
    add(ThresholdSpec(
        name="EFFECTIVE_INNER_SLOPE_MIN", category=Category.GEOMETRIC,
        source=Source.LITERATURE_FALLBACK, value=10.0, units="°",
        rule="P10(confirmed eff slopes when bottom > 4 m from track)",
        reference="Rail Baltica Part 2 §10.13",
        description="Global inner-slope sanity check (only when bottom > 4 m).",
        used_by=["_try_find_ditch (cond. global slope)"],
    ))
    add(ThresholdSpec(
        name="EFFECTIVE_SLOPE_MIN_BOTTOM_X", category=Category.GEOMETRIC,
        source=Source.ENGINEERING, value=4.0, units="m",
        rule="fixed",
        reference="engineering: separates near-track from far-track ditches",
        description="Bottom_x threshold above which global eff slope check applies.",
        used_by=["_try_find_ditch"],
    ))
    add(ThresholdSpec(
        name="SHAPE_OUTER_RISE_FRAC", category=Category.GEOMETRIC,
        source=Source.LITERATURE, value=0.25, units="fraction",
        rule="fixed (proportional to depth)",
        reference="Rail Baltica Part 2 §10.13 (outer shoulder ≈ rail bed elev.)",
        description="Required outer-bank rise = max(SHAPE_OUTER_RISE_MIN, this × depth).",
        used_by=["_try_find_ditch"],
    ))

    # ─────────────────────────────────────────────────────────────────
    # CATEGORY D — CLASSIFICATION (bootstrap from shape/content)
    # ─────────────────────────────────────────────────────────────────
    add(ThresholdSpec(
        name="SHAPE_FLATNESS_U_MIN", category=Category.CLASSIFICATION,
        source=Source.LITERATURE_FALLBACK, value=0.15, units="ratio",
        rule="P25(confirmed flatness)",
        reference="Knighton 1981 + Roelens 2018 §3.3",
        description="flatness ≥ this → not pure-V (V_HEALTHY upper bound).",
        used_by=["classify_shape"],
    ))
    add(ThresholdSpec(
        name="SHAPE_FLATNESS_FLAT_MIN", category=Category.CLASSIFICATION,
        source=Source.LITERATURE_FALLBACK, value=0.35, units="ratio",
        rule="P75(confirmed flatness)",
        reference="Knighton 1981 + Roelens 2018",
        description="flatness ≥ this → flat-bottom (FLAT_SILTED candidate).",
        used_by=["classify_shape"],
    ))
    add(ThresholdSpec(
        name="SHAPE_DEPTH_DEFICIT_MIN", category=Category.CLASSIFICATION,
        source=Source.LITERATURE_FALLBACK, value=0.20, units="ratio",
        rule="P75(confirmed depth_deficit)",
        reference="Knighton 1981; Roelens 2018",
        description="deficit ≥ this AND flat → SILTED.",
        used_by=["classify_shape"],
    ))
    add(ThresholdSpec(
        name="SHAPE_BAND_FRAC", category=Category.CLASSIFICATION,
        source=Source.ENGINEERING, value=0.10, units="fraction",
        rule="fixed",
        reference="engineering: bottom-band = lowest 10 % of depth",
        description="Used inside compute_shape_descriptors.",
        used_by=["compute_shape_descriptors"],
    ))
    add(ThresholdSpec(
        name="CONTENT_DVCI_PARTIAL", category=Category.CLASSIFICATION,
        source=Source.LITERATURE_FALLBACK, value=0.20, units="fraction",
        rule="P50(confirmed DVCI)",
        reference="Roelens 2018; Meyer & Neto 2008",
        description="DVCI ≥ this → some vegetation (partial cover).",
        used_by=["classify_content"],
    ))
    add(ThresholdSpec(
        name="CONTENT_DVCI_DENSE", category=Category.CLASSIFICATION,
        source=Source.LITERATURE_FALLBACK, value=0.50, units="fraction",
        rule="P85(confirmed DVCI)",
        reference="Roelens 2018; Meyer & Neto 2008",
        description="DVCI ≥ this → densely vegetated.",
        used_by=["classify_content"],
    ))

    # ─────────────────────────────────────────────────────────────────
    # CATEGORY E — PROCEDURAL (derived from B/C, or fixed engineering)
    # ─────────────────────────────────────────────────────────────────
    add(ThresholdSpec(
        name="POSITION_OUTLIER_TOL_M", category=Category.PROCEDURAL,
        source=Source.LITERATURE_FALLBACK, value=0.7, units="m",
        rule="P95(|bottom_x − rolling_median|) of confirmed detections",
        reference="Roelens 2018 (lateral RMSE 0.4-0.6 m); Habib 2021",
        description="Confirmed sections beyond this from rolling median → demoted.",
        used_by=["spatial-coherence post-processing"],
    ))
    add(ThresholdSpec(
        name="POSITION_OUTLIER_WIN_HALF", category=Category.PROCEDURAL,
        source=Source.ENGINEERING, value=3, units="sections",
        rule="fixed",
        reference="engineering: ±3 m local context",
        description="Half-window for rolling median in position-outlier check.",
        used_by=["spatial-coherence"],
    ))
    add(ThresholdSpec(
        name="POSITION_OUTLIER_MIN_NBR", category=Category.PROCEDURAL,
        source=Source.ENGINEERING, value=2, units="sections",
        rule="fixed",
        reference="engineering: need ≥ 2 neighbours to judge outlier",
        description="Minimum neighbours required for position outlier test.",
        used_by=["spatial-coherence"],
    ))
    add(ThresholdSpec(
        name="DITCH_PRIOR_WINDOW", category=Category.PROCEDURAL,
        source=Source.ENGINEERING, value=5, units="sections",
        rule="fixed",
        reference="engineering: 5-section rolling memory of last bottom_x",
        description="Length of rolling-prior memory used in spatial continuity.",
        used_by=["compute_side_metrics_first_ditch"],
    ))
    add(ThresholdSpec(
        name="GAP_RELAXED_MIN_DEPTH", category=Category.PROCEDURAL,
        source=Source.DERIVED, value=0.04, units="m",
        rule="0.5 × DITCH_MIN_DEPTH",
        reference="derived",
        description="Relaxed depth threshold for gap re-detection.",
        used_by=["gap re-detection"],
    ))
    add(ThresholdSpec(
        name="GAP_RELAXED_MIN_PROM", category=Category.PROCEDURAL,
        source=Source.DERIVED, value=0.03, units="m",
        rule="0.5 × DITCH_MIN_PROM",
        reference="derived",
        description="Relaxed prominence for gap re-detection.",
        used_by=["gap re-detection"],
    ))
    add(ThresholdSpec(
        name="GAP_POSITION_TOLERANCE", category=Category.PROCEDURAL,
        source=Source.DERIVED, value=3.0, units="m",
        rule="max(3.0, 4 × POSITION_OUTLIER_TOL_M)",
        reference="derived",
        description="Allowed |bottom_x − neighbour_prior| in gap re-detection.",
        used_by=["gap re-detection"],
    ))
    add(ThresholdSpec(
        name="CLOSING_KERNEL", category=Category.PROCEDURAL,
        source=Source.LITERATURE, value=5, units="sections",
        rule="fixed (binary closing kernel)",
        reference="Soille 2004 (morphological closing); Cazorzi 2013",
        description="Closes gaps up to (kernel-1) sections.",
        used_by=["gap closing"],
    ))
    add(ThresholdSpec(
        name="DITCH_MIN_RUN_LENGTH", category=Category.PROCEDURAL,
        source=Source.ENGINEERING, value=10, units="sections (m)",
        rule="fixed engineering: <10 m has no drainage function",
        reference="engineering; aligned with Roelens 2018; Cazorzi 2013",
        description="Minimum confirmed-run length for CONFIRMED tag.",
        used_by=["run-length filter"],
    ))
    add(ThresholdSpec(
        name="MORPH_KERNEL_SIZE", category=Category.PROCEDURAL,
        source=Source.LITERATURE, value=3, units="bins (×ENV_BIN_W)",
        rule="fixed",
        reference="Roelens 2016/2018 (morphological opening)",
        description="Erosion+dilation kernel size on envelope (= 0.15 m).",
        used_by=["build_lower_envelope"],
    ))
    add(ThresholdSpec(
        name="MORPH_POST_SMOOTH_WIN", category=Category.PROCEDURAL,
        source=Source.ENGINEERING, value=5, units="bins",
        rule="fixed (Sav-Gol post-smoothing)",
        reference="engineering",
        description="Window for light Sav-Gol smoothing after opening.",
        used_by=["build_lower_envelope"],
    ))
    add(ThresholdSpec(
        name="WATER_MIN_CONTIGUOUS_BINS", category=Category.PROCEDURAL,
        source=Source.LITERATURE, value=3, units="bins",
        rule="fixed (≈ 0.15 m run)",
        reference="Roelens 2018",
        description="Minimum contiguous low-density bins for water flag.",
        used_by=["detect_water_in_ditch"],
    ))
    add(ThresholdSpec(
        name="SEGMENT_LEN", category=Category.PROCEDURAL,
        source=Source.ENGINEERING, value=10.0, units="m",
        rule="fixed reporting granularity",
        reference="engineering",
        description="Aggregation segment length for condition reporting.",
        used_by=["aggregate_to_segments"],
    ))
    add(ThresholdSpec(
        name="MIN_VALID_IN_SEGMENT", category=Category.PROCEDURAL,
        source=Source.ENGINEERING, value=3, units="sections",
        rule="fixed",
        reference="engineering",
        description="Minimum valid 1-m sections inside a 10-m segment.",
        used_by=["aggregate_to_segments"],
    ))

    # ─────────────────────────────────────────────────────────────────
    # CATEGORY F — SEARCH ZONE (corridor geometry, fixed engineering)
    # ─────────────────────────────────────────────────────────────────
    add(ThresholdSpec(
        name="SIDE_SEARCH_MIN", category=Category.SEARCH,
        source=Source.ENGINEERING, value=0.50, units="m",
        rule="fixed (track ballast edge)",
        reference="TDOK 2015:0155 Bilaga 2; Rail Baltica RBDG-MAN-016",
        description="Inner edge of ditch search zone (from centreline).",
        used_by=["compute_side_metrics_first_ditch"],
    ))
    add(ThresholdSpec(
        name="SIDE_SEARCH_MAX", category=Category.SEARCH,
        source=Source.ENGINEERING, value=15.50, units="m",
        rule="fixed (extended corridor width upper; includes outer drainage depressions)",
        reference="TDOK 2015:0155 Bilaga 2; Rail Baltica RBDG-MAN-016",
        description="Outer edge of ditch search zone (from centreline).",
        used_by=["compute_side_metrics_first_ditch"],
    ))
    add(ThresholdSpec(
        name="CUTTING_SEARCH_MIN", category=Category.SEARCH,
        source=Source.ENGINEERING, value=0.50, units="m",
        rule="fixed",
        reference="engineering: cutting drainage near track",
        description="Search-zone min for cutting drainage ditches.",
        used_by=["Pass 2 cutting"],
    ))
    add(ThresholdSpec(
        name="CUTTING_SEARCH_MAX", category=Category.SEARCH,
        source=Source.ENGINEERING, value=5.00, units="m",
        rule="fixed",
        reference="engineering: cutting toe within 5 m typical",
        description="Search-zone max for cutting drainage ditches.",
        used_by=["Pass 2 cutting"],
    ))
    add(ThresholdSpec(
        name="CUTTING_DEPTH_FROM_SHOULDER", category=Category.SEARCH,
        source=Source.ENGINEERING, value=0.10, units="m",
        rule="fixed",
        reference="engineering: shoulder-relative depth threshold",
        description="Min depth below shoulder_z for cutting ditch acceptance.",
        used_by=["Pass 2 cutting / embankment"],
    ))
    add(ThresholdSpec(
        name="ENV_V_RANGE", category=Category.SEARCH,
        source=Source.ENGINEERING, value=16.0, units="m",
        rule="fixed (≥ SIDE_SEARCH_MAX + 0.5)",
        reference="engineering: must cover full side-search range",
        description="Half-range of envelope sampling across track.",
        used_by=["build_lower_envelope"],
    ))
    add(ThresholdSpec(
        name="ENV_Q", category=Category.SEARCH,
        source=Source.ENGINEERING, value=5, units="percentile",
        rule="fixed",
        reference="engineering: P5 = robust lower envelope",
        description="Quantile used to pick envelope value per bin.",
        used_by=["build_lower_envelope"],
    ))

    # ─────────────────────────────────────────────────────────────────
    # CATEGORY G — SANITY (validity bounds, literature)
    # ─────────────────────────────────────────────────────────────────
    add(ThresholdSpec(
        name="DEPTH_VALID_MIN", category=Category.SANITY,
        source=Source.DERIVED, value=0.08, units="m",
        rule="= DITCH_MIN_DEPTH",
        reference="derived",
        description="Sanity floor for reported depths.",
        used_by=["validity gating"],
    ))
    add(ThresholdSpec(
        name="DEPTH_VALID_MAX", category=Category.SANITY,
        source=Source.ENGINEERING, value=2.50, units="m",
        rule="fixed engineering",
        reference="engineering: > 2.5 m below RUK is suspect terrain, not ditch",
        description="Sanity ceiling for reported depths.",
        used_by=["validity gating"],
    ))
    add(ThresholdSpec(
        name="DEPTH_DEEP_FLAG", category=Category.SANITY,
        source=Source.LITERATURE, value=1.50, units="m",
        rule="TDOK target + 0.2 m safety margin",
        reference="Trafikverket TDOK 2015:0155 §10.13",
        description="Reported as deep_flag = 1 above this.",
        used_by=["reporting"],
    ))
    add(ThresholdSpec(
        name="SLOPE_ASYMMETRY_MAX", category=Category.SANITY,
        source=Source.LITERATURE, value=0.50, units="A1 dimensionless",
        rule="fixed",
        reference="Knighton 1981 (A1 moderate-to-extreme boundary)",
        description="|asymmetry| > this → reject candidate or trigger SLOPE_ISSUE.",
        used_by=["_try_find_ditch; diagnose_issues"],
    ))
    add(ThresholdSpec(
        name="EMBANKMENT_MIN_DROP", category=Category.SANITY,
        source=Source.ENGINEERING, value=0.12, units="m",
        rule="fixed",
        reference="engineering: classifier trigger (terrain drops outward)",
        description="Inner_z − outer_half_z above which Pass 2b triggers.",
        used_by=["compute_side_metrics_first_ditch (Pass 2b trigger)"],
    ))
    add(ThresholdSpec(
        name="CUTTING_MIN_RISE", category=Category.SANITY,
        source=Source.ENGINEERING, value=0.12, units="m",
        rule="fixed",
        reference="engineering: classifier trigger (terrain rises outward)",
        description="Outer_half_z − inner_z above which Pass 2a triggers.",
        used_by=["compute_side_metrics_first_ditch (Pass 2a trigger)"],
    ))
    add(ThresholdSpec(
        name="FLAT_RELIEF_MAX", category=Category.SANITY,
        source=Source.ENGINEERING, value=0.08, units="m",
        rule="fixed",
        reference="engineering",
        description="Max relief on a side for FLAT terrain classification.",
        used_by=["non-ditch terrain classifier"],
    ))

    return R


THRESHOLD_REGISTRY: dict[str, ThresholdSpec] = _make_registry()


# ════════════════════════════════════════════════════════════════════════
# Convenience helpers
# ════════════════════════════════════════════════════════════════════════
def by_category(cat: Category) -> dict[str, ThresholdSpec]:
    """Return the subset of the registry in a given category."""
    return {n: s for n, s in THRESHOLD_REGISTRY.items() if s.category == cat}


def to_step07_constants(specs: dict[str, ThresholdSpec]) -> dict[str, Any]:
    """Flatten a spec dict to a {name: value} mapping ready for step07 to do
    `globals()[name] = value` (or to be assigned individually).
    """
    return {n: s.value for n, s in specs.items() if s.value is not None}


def save_registry(specs: dict[str, ThresholdSpec],
                  path: str,
                  meta: Optional[dict] = None) -> None:
    """Serialise a spec dict + run metadata to JSON.

    JSON top-level structure:
        {
          "_meta": {...},
          "thresholds": {"DITCH_MIN_DEPTH": {value, source, rule, ...}, ...}
        }
    """
    payload = {
        "_meta": meta or {},
        "thresholds": {n: s.to_dict() for n, s in specs.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_registry(path: str) -> tuple[dict[str, ThresholdSpec], dict]:
    """Load a serialised spec dict.  Returns (specs_dict, meta)."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("thresholds", {})
    specs = {n: ThresholdSpec.from_dict(d) for n, d in raw.items()}
    return specs, payload.get("_meta", {})


def merge_specs(*spec_dicts: dict[str, ThresholdSpec]) -> dict[str, ThresholdSpec]:
    """Right-most wins.  Useful for assembling the full registry from
    per-category dicts."""
    out: dict[str, ThresholdSpec] = {}
    for d in spec_dicts:
        out.update(d)
    return out


def fresh_registry() -> dict[str, ThresholdSpec]:
    """Return a deep-ish copy of the literature-default registry."""
    return {n: ThresholdSpec(**{**asdict(s),
                                 "category": s.category,
                                 "source":   s.source})
            for n, s in THRESHOLD_REGISTRY.items()}


# ════════════════════════════════════════════════════════════════════════
# Self-test
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Registry size: {len(THRESHOLD_REGISTRY)} thresholds")
    for cat in Category:
        sub = by_category(cat)
        print(f"  {cat.value:30s}  {len(sub)} threshold(s)")
        for name in sub:
            spec = sub[name]
            val = f"{spec.value}" if spec.value is not None else "—"
            print(f"     {name:35s}  default={val:>8s}   {spec.source.value}")

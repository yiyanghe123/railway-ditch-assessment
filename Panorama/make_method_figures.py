"""Generate publication-quality INTERMEDIATE-PROCESS figures for the panorama
method, from the already-saved per-station profile stack (no GPU needed).

Figures (vector PDF + 400 dpi PNG) written to tiles/<TILE>/results/figures_method/:
  1. method_cross_section_<TILE>_<side>     one representative station: the
     panorama visible-surface cross-section z(t) below the rail reference, the
     lateral ditch-search window, and the detected visible-surface depression.
  2. method_profile_gallery_<TILE>_<side>   small multiples of the same
     extraction at several chainages (shows per-station behaviour along the run).

Usage:
    python make_method_figures.py --tile 7309_809 --side right
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from figure_style import apply_style, savefig, panel_label, PALETTE, CM  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# lateral ditch-search window used by the independent panorama pipeline
SEARCH_MIN, SEARCH_MAX = 2.0, 7.0
HERE = Path(__file__).resolve().parent


def fnum(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_profiles(tile: str, side: str):
    f = HERE / "tiles" / tile / "results" / "output_continuous" / "panorama_side_profile_stack.csv"
    rows = list(csv.DictReader(open(f)))
    prof = {}
    for r in rows:
        if r.get("side") != side:
            continue
        key = (r["frame_no"], fnum(r["S_cam"]))
        prof.setdefault(key, []).append((fnum(r["t_m"]), fnum(r["visible_surface_depth_m"]),
                                         fnum(r["n_points"])))
    out = {}
    for k, v in prof.items():
        v = sorted([(t, d, n) for t, d, n in v if t is not None])
        t = np.array([a for a, _, _ in v])
        d = np.array([np.nan if b is None else b for _, b, _ in v])
        out[k] = (t, d)
    return out


# lateral window shown in the method figure (focus on the near-track ditch zone;
# beyond ~9 m the monocular reconstruction is unreliable and is not shown)
XVIEW_MIN, XVIEW_MAX = 0.0, 9.0


def detect(t, d):
    """deepest visible-surface bin inside the lateral search window, with a
    local prominence (depth minus the shallower shoulder just outside it)."""
    band = (t >= SEARCH_MIN) & (t <= SEARCH_MAX) & np.isfinite(d)
    if not band.any():
        return None
    bi = np.where(band)[0]
    i = bi[np.nanargmax(d[band])]
    td, dd = float(t[i]), float(d[i])
    rim = ((np.abs(t - td) > 0.6) & (np.abs(t - td) <= 2.5) & np.isfinite(d))
    prom = dd - float(np.nanpercentile(d[rim], 75)) if rim.any() else 0.0
    return td, dd, prom


GROUND = "#e7ded0"        # warm light tan for the filled ground body


def plot_cross_section(ax, t, d, det, legend="none", annotate=True):
    """Filled terrain cross-section: height relative to rail (ditch dips down),
    ground shaded below the visible surface.  legend='outside' | 'none'."""
    m = np.isfinite(d) & (t >= XVIEW_MIN) & (t <= XVIEW_MAX)
    tt, hh = t[m], -d[m]                                  # height relative to rail
    lo, hi = float(np.nanmin(hh)), float(np.nanmax(hh))
    if det is not None:
        lo = min(lo, -det[1])
    pad = 0.18 * max(hi - lo, 0.4)
    ymin = lo - pad
    ax.fill_between(tt, hh, ymin, color=GROUND, zorder=1)
    ax.plot(tt, hh, "-", color=PALETTE["profile"], lw=1.9, solid_capstyle="round",
            label="panorama visible surface", zorder=3)
    ax.axhline(0, color=PALETTE["ruk"], lw=1.1, ls="--", label="rail reference (RUK)", zorder=4)
    ax.axvspan(SEARCH_MIN, SEARCH_MAX, color=PALETTE["ruk"], alpha=0.07, lw=0, zorder=2,
               label="ditch-search window")
    for xb in (SEARCH_MIN, SEARCH_MAX):
        ax.axvline(xb, color=PALETTE["ruk"], lw=0.8, ls=":", alpha=0.7, zorder=2)
    if det is not None:
        ax.scatter([det[0]], [-det[1]], marker="*", s=150, color=PALETTE["ditch"],
                   edgecolor="white", linewidth=0.8, label="detected ditch bottom", zorder=6)
        if annotate:
            ax.annotate(f"{det[1]:.2f} m", xy=(det[0], -det[1]), xytext=(6, 18),
                        textcoords="offset points", fontsize=7.5, color=PALETTE["ditch"],
                        fontweight="bold", zorder=7,
                        arrowprops=dict(arrowstyle="-", color=PALETTE["ditch"], lw=0.7))
    ax.set_ylim(ymin, hi + pad)
    ax.set_xlim(XVIEW_MIN, XVIEW_MAX)
    ax.set_xlabel("Lateral offset from track centre, $T$ (m)")
    ax.set_ylabel("Surface height relative\nto rail reference (m)")
    ax.grid(True, zorder=0)
    if legend == "outside":
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                  handlelength=1.6)


def main():
    global SEARCH_MAX, XVIEW_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="7309_809")
    ap.add_argument("--side", default="right")
    ap.add_argument("--search-max", type=float, default=SEARCH_MAX,
                    help="lateral search-window upper bound (m); set 15 for the wide-window demo")
    ap.add_argument("--suffix", default="", help="filename suffix, e.g. _wide15m")
    a = ap.parse_args()
    SEARCH_MAX = a.search_max
    XVIEW_MAX = max(9.0, SEARCH_MAX + 1.0)            # show the whole search window
    apply_style()
    prof = load_profiles(a.tile, a.side)
    figdir = HERE / "tiles" / a.tile / "results" / "figures_method"

    narrow = SEARCH_MAX <= 8.0
    # narrow window: keep CLEAN near-corridor depressions; wide window: keep all
    # detections so the figure shows WHERE the wide search lands (the demo point).
    scored = []
    for (frame, scam), (t, d) in prof.items():
        det = detect(t, d)
        if det is None:
            continue
        td, dd, prom = det
        if not narrow:
            scored.append((dd, scam, frame, t, d, (td, dd)))
        elif 2.5 <= td <= 6.0 and 0.25 <= dd <= 1.3 and prom >= 0.15:
            scored.append((dd, scam, frame, t, d, (td, dd)))
    if narrow and len(scored) < 3:
        # relax if too few clean examples
        scored = []
        for (frame, scam), (t, d) in prof.items():
            det = detect(t, d)
            if det and np.isfinite(det[1]):
                scored.append((det[1], scam, frame, t, d, (det[0], det[1])))
    if not scored:
        print("no detectable depressions; nothing to plot"); return
    scored.sort(key=lambda x: x[0])
    # representative single station: median-depth detection among the clean set
    rep = scored[len(scored) // 2]

    tile_pretty = a.tile.split("_")[-1]
    # ---- Figure 1: single cross-section, legend outside (LiDAR-figure layout) ----
    fig, ax = plt.subplots(figsize=(CM(13.8), CM(6.2)), constrained_layout=True)
    plot_cross_section(ax, rep[3], rep[4], rep[5], legend="outside", annotate=True)
    # No in-figure title (the LaTeX caption is the title); keep only a short
    # station tag in the corner, matching the LiDAR cross-section figures.
    ax.text(0.015, 0.95, f"$S$ = {rep[1]:.0f} m", transform=ax.transAxes,
            fontsize=8, va="top", ha="left")
    savefig(fig, figdir / f"method_cross_section_{a.tile}_{a.side}{a.suffix}")
    print(f"wrote method_cross_section_{a.tile}_{a.side}.pdf/.png  (S={rep[1]:.1f} m)")

    # ---- Figure 2: gallery of profiles along chainage (full width, 2x3) ----
    by_s = sorted(scored, key=lambda x: x[1])
    pick = [by_s[int(round(i * (len(by_s) - 1) / 5))] for i in range(6)] if len(by_s) >= 6 else by_s
    nrow, ncol = 2, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(CM(17), CM(9)), sharex=True, sharey=True)
    for k, ax in enumerate(axes.ravel()):
        if k >= len(pick):
            ax.axis("off"); continue
        _, scam, frame, t, d, det = pick[k]
        plot_cross_section(ax, t, d, det, legend="none", annotate=True)
        ax.set_title(f"$S$ = {scam:.0f} m", fontsize=8.5)
        panel_label(ax, f"({chr(97 + k)})")
        ax.set_xlabel(""); ax.set_ylabel("")          # use shared labels instead
    # shared axis labels + a shared single-row legend, stacked below the panels
    fig.tight_layout(rect=[0.03, 0.17, 1, 1])
    fig.supylabel("Visible-surface depth below rail (m)", fontsize=9)
    fig.supxlabel("Lateral offset from track centre, $T$ (m)", fontsize=9, y=0.125)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handlelength=1.5, fontsize=7.5)
    savefig(fig, figdir / f"method_profile_gallery_{a.tile}_{a.side}{a.suffix}")
    print(f"wrote method_profile_gallery_{a.tile}_{a.side}.pdf/.png  ({len(pick)} stations)")
    print(f"figures in: {figdir}")


if __name__ == "__main__":
    main()


"""Ablation figure: the SAME panorama cross-section under two lateral search
windows.  Left = 2-7 m (chosen near band): the detected bottom sits on the
near-track ditch.  Right = 2-15.5 m (matching the LiDAR search): the deepest-point
selection drifts onto the far, unreliable surface.

Rendered as a filled terrain cross-section (height relative to the rail
reference, ground shaded below the visible surface) in the unified LiDAR style.

Usage: python make_ablation_figure.py --tile 7309_809 --side right
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from figure_style import apply_style, savefig, PALETTE, CM  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

NEAR_MAX, WIDE_MAX, SMIN, XMAX = 7.0, 15.5, 2.0, 15.4
GROUND = "#e7ded0"        # warm light tan for the filled ground body


def fnum(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_profiles(tile, side):
    f = HERE / "tiles" / tile / "results" / "output_continuous" / "panorama_side_profile_stack.csv"
    prof = {}
    for r in csv.DictReader(open(f)):
        if r.get("side") != side:
            continue
        prof.setdefault((r["frame_no"], fnum(r["S_cam"])), []).append(
            (fnum(r["t_m"]), fnum(r["visible_surface_depth_m"])))
    out = {}
    for k, v in prof.items():
        v = sorted([(t, d) for t, d in v if t is not None])
        out[k] = (np.array([t for t, _ in v]),
                  np.array([np.nan if d is None else d for _, d in v]))
    return out


def detect(t, d, tmax):
    b = (t >= SMIN) & (t <= tmax) & np.isfinite(d)
    if not b.any():
        return None
    i = np.where(b)[0][np.nanargmax(d[b])]
    return float(t[i]), float(d[i])


def panel(ax, t, d, tmax, det, title, ymin, text_x, text_y):
    m = np.isfinite(d) & (t <= XMAX)
    tt, hh = t[m], -d[m]                                  # height relative to rail (down = ditch)
    # ground body
    ax.fill_between(tt, hh, ymin, color=GROUND, zorder=1)
    ax.plot(tt, hh, "-", color=PALETTE["profile"], lw=1.9, solid_capstyle="round",
            label="panorama visible surface", zorder=3)
    # rail reference datum
    ax.axhline(0, color=PALETTE["ruk"], lw=1.1, ls="--", label="rail reference (RUK)", zorder=4)
    # search-window extent: bracket the band with edge lines + a faint top overlay
    ax.axvspan(SMIN, tmax, color=PALETTE["ruk"], alpha=0.07, lw=0, zorder=2,
               label="ditch-search window")
    for xb in (SMIN, tmax):
        ax.axvline(xb, color=PALETTE["ruk"], lw=0.8, ls=":", alpha=0.7, zorder=2)
    if det is not None:
        ax.scatter([det[0]], [-det[1]], marker="*", s=85, color=PALETTE["ditch"],
                   edgecolor="white", linewidth=0.6, label="detected ditch bottom", zorder=6)
        # label in the open strip between the curve and the x-axis (same low
        # height in both panels); a thin leader points to the star
        ax.annotate(f"$T$ = {det[0]:.1f} m,  depth = {det[1]:.2f} m",
                    xy=(det[0], -det[1]), xytext=(text_x, text_y), textcoords="data",
                    fontsize=7.5, color=PALETTE["ditch"], fontweight="bold",
                    ha="center", va="center", zorder=7,
                    arrowprops=dict(arrowstyle="-", color=PALETTE["ditch"], lw=0.7))
    ax.set_xlim(0, WIDE_MAX + 0.3)
    ax.set_xlabel("Lateral offset from track centre, $T$ (m)")
    ax.set_title(title, fontsize=9.5, pad=8)
    ax.grid(True, zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="7309_809")
    ap.add_argument("--side", default="right")
    a = ap.parse_args()
    apply_style()
    prof = load_profiles(a.tile, a.side)

    best = None
    for (frame, scam), (t, d) in prof.items():
        n, w = detect(t, d, NEAR_MAX), detect(t, d, WIDE_MAX)
        if not n or not w:
            continue
        if 2.5 <= n[0] <= 5.5 and 0.3 <= n[1] <= 1.0 and w[0] >= 9.0:
            drift = w[0] - n[0]
            if best is None or drift > best[0]:
                best = (drift, scam, t, d, n, w)
    if best is None:
        print("no clear drift station found"); return
    _, scam, t, d, n, w = best

    # shared height range (height = -depth) over the displayed lateral extent
    m = np.isfinite(d) & (t <= XMAX)
    hh = -d[m]
    hlo, hhi = float(np.nanmin(hh)), float(np.nanmax(hh))
    hlo = min(hlo, -n[1], -w[1])
    span = max(hhi - hlo, 0.5)
    ymin, ymax = hlo - 0.46 * span, hhi + 0.12 * span     # extra room below for the label strip
    text_y = ymin + 0.13 * (ymax - ymin)                  # common low label height (level in both panels)

    fig, axes = plt.subplots(1, 2, figsize=(CM(17), CM(7.2)), sharex=True, sharey=True)
    panel(axes[0], t, d, NEAR_MAX, n, "(a) $2$--$7$ m window", ymin, n[0], text_y)
    panel(axes[1], t, d, WIDE_MAX, w, "(b) $2$--$15.5$ m window", ymin, 8.0, text_y)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    axes[0].set_ylabel("Surface height relative\nto rail reference (m)")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handlelength=1.5, fontsize=7.5)
    figdir = HERE / "tiles" / a.tile / "results" / "figures_method"
    savefig(fig, figdir / f"method_ablation_search_window_{a.tile}_{a.side}")
    print(f"wrote method_ablation_search_window_{a.tile}_{a.side}.pdf/.png  "
          f"(S={scam:.0f} m; 7m->T={n[0]:.1f}, 15.5m->T={w[0]:.1f})")


if __name__ == "__main__":
    main()

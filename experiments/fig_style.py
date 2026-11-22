"""Shared figure style for the ICML figures (docs/notes/fig_plan.md section 4).

Importing this module sets matplotlib rcParams (Agg backend, DejaVu Sans, 8 pt base, fonttype 42).
Signatures are frozen after step 0; additive fixes only.

    save(fig, name, sub=None)        PDF + PNG into paper/figs[/sub]
    frame_tile(ax, img, crop=None, border=None)
    draw_path(ax, ee_px, held, outcome)
    stamp(ax, "FAIL"|"OK")
    box(ax, xy, w, h, label, kind)   kind in frozen|expert|new|plain|input|ghost
    arrow(ax, p0, p1, ...)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "flawed")
OUT = os.path.join(REPO, "paper", "figs")

W1, W2 = 3.25, 6.75  # column and full-width figure widths in inches

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220,
    "axes.grid": False, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

# ------------------------------------------------------------------ palette
INK, MUTED, GREYINK, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
OK, FAIL = "#0ca30c", "#d03b3b"
CUE = "#7b3fa0"
GREY, BLACK = "#8c8c8c", "#000000"
MODEL = {"pi0.5": GREY, "pi0": "#E69F00", "GR00T N1": "#009E73",
         "X-VLA": "#CC79A7", "SmolVLA": "#D55E00"}
ARM = {"R0": "#c6dbef", "R2": "#6baed6", "RAA": "#2171b5", "RAA-o": "#08519c",
       "RAA-c": "#4292c6", "RAA-s": "#08306b", "RAA-sL": "#08306b", "RAA-L": "#2171b5",
       "ReGuide": BLACK}
NAVY = "#08306b"
FROZEN, EXPERT = "#e6e6e6", "#c6dbef"
SMALL, TINY = 7.0, 6.5  # pt


def save(fig, name, sub=None, tight=True):
    """tight=False keeps the exact canvas (fixed 6.75 / 3.25 in widths); default behaviour unchanged."""
    d = OUT if sub is None else os.path.join(OUT, sub)
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        if tight:
            fig.savefig(os.path.join(d, name + "." + ext), bbox_inches="tight", pad_inches=0.03)
        else:
            fig.savefig(os.path.join(d, name + "." + ext))
    plt.close(fig)
    print("wrote", name, flush=True)


def frame_tile(ax, img, crop=None, border=None):
    """Show an image (H,W,3) in ax. crop = (r0, r1, c0, c1) pixel box. border = model colour (2 pt)."""
    if crop is not None:
        r0, r1, c0, c1 = crop
        img = img[r0:r1, c0:c1]
    ax.imshow(img, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(border is not None)
        if border is not None:
            s.set_edgecolor(border)
            s.set_linewidth(2.0)
    return ax


def _halo(lw):
    return [pe.Stroke(linewidth=lw + 3.0, foreground="white"), pe.Normal()]


def draw_path(ax, ee_px, held, outcome):
    """ee_px [T,2] pixel (col,row) in the tile's coordinates; held [T] bool.
    Reach (gripper open) dashed, carry (held) solid, carry colour by outcome ('FAIL'/'fail'/False -> red, else green).
    Grasp = circle at first held row, release = square at last held row."""
    ee = np.asarray(ee_px, float)
    held = np.asarray(held, bool)
    bad = outcome in ("FAIL", "fail", False, 0)
    col = FAIL if bad else OK
    idx = np.nonzero(held)[0]
    g0 = idx[0] if len(idx) else len(ee) - 1
    g1 = idx[-1] if len(idx) else len(ee) - 1
    ax.plot(ee[:g0 + 1, 0], ee[:g0 + 1, 1], ls=(0, (3, 2)), color=INK, lw=1.6, path_effects=_halo(1.6), zorder=5)
    if len(idx):
        ax.plot(ee[g0:g1 + 1, 0], ee[g0:g1 + 1, 1], ls="-", color=col, lw=1.6, path_effects=_halo(1.6), zorder=6)
        ax.plot(ee[g1:, 0], ee[g1:, 1], ls=(0, (3, 2)), color=INK, lw=1.2, path_effects=_halo(1.2), zorder=5)
        ax.plot(*ee[g0], "o", ms=5, mfc="white", mec=INK, mew=1.2, zorder=7)
        ax.plot(*ee[g1], "s", ms=5, mfc="white", mec=INK, mew=1.2, zorder=7)


def stamp(ax, text, loc="upper left"):
    """FAIL / OK tag in the tile corner (axes coordinates)."""
    col = FAIL if text.upper().startswith("F") else OK
    x, y, ha, va = {"upper left": (0.04, 0.95, "left", "top"), "upper right": (0.96, 0.95, "right", "top"),
                    "lower left": (0.04, 0.05, "left", "bottom"), "lower right": (0.96, 0.05, "right", "bottom")}[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="white", lw=0.8), zorder=10)


_KINDS = {  # kind: (facecolor, edgecolor, linewidth, linestyle)
    "frozen": (FROZEN, GREYINK, 0.8, "-"),
    "expert": (EXPERT, NAVY, 0.8, "-"),
    "new": ("white", NAVY, 1.4, "-"),
    "plain": ("white", GREYINK, 0.8, "-"),
    "input": ("white", GREYINK, 0.8, "-"),
    "ghost": ("white", GREYINK, 0.8, (0, (2, 2))),
}


def box(ax, xy, w, h, label="", kind="plain", fontsize=SMALL, color=None, weight="normal", hatch=None, zorder=2):
    """Rounded box with lower-left corner xy. Returns the patch."""
    fc, ec, lw, ls = _KINDS[kind]
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0,rounding_size=%g" % (min(w, h) * 0.12), fc=fc, ec=ec,
                       lw=lw, ls=ls, hatch=hatch, zorder=zorder)
    ax.add_patch(p)
    if label:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center", fontsize=fontsize,
                color=color or (GREYINK if kind == "ghost" else INK), fontweight=weight, zorder=zorder + 1,
                linespacing=1.05)
    return p


def arrow(ax, p0, p1, color=INK, lw=0.9, ls="-", head=6, rad=0.0, zorder=3, shrink=0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw, ls=ls,
                        connectionstyle="arc3,rad=%g" % rad, shrinkA=shrink, shrinkB=shrink, zorder=zorder)
    ax.add_patch(a)
    return a


def cross(ax, xy, size=0.04, color=FAIL, lw=1.4, zorder=8):
    x, y = xy
    ax.plot([x - size, x + size], [y - size, y + size], color=color, lw=lw, zorder=zorder)
    ax.plot([x - size, x + size], [y + size, y - size], color=color, lw=lw, zorder=zorder)


def blank_axes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax

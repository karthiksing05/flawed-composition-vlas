"""Cue schematics (docs/notes/fig_plan.md section 2 and 4).

Interface: draw_cue(key, ax, compact=False); keys F1 F1s2 F1m F2 F7 F6 NL MV.
Axis contract: limits (0,1) x (0,1), equal aspect.  The cue accent (purple) is used for the cue only;
the instructed target is dashed ink, the cue-implied one is purple and solid; a red cross marks the arm
that the cue misdirects.  All schematics read in greyscale (shapes and dashes differ).
compact=True drops the secondary captions (for the 1.2 in gallery tile).
Standalone:  python fig_cues.py   writes paper/figs/fig_cues_all.{pdf,png} and paper/figs/cues/<key>.{pdf,png}.
No data: pure schematic (the numbers live in the captions, DRAFT section 4).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon, Wedge, FancyBboxPatch  # noqa: E402
import fig_style as S  # noqa: E402

KEYS = ["F1", "F1s2", "F1m", "F2", "F7", "F6", "NL", "MV"]
TITLE = {"F1": "F1 place", "F1s2": "F1 scene 2", "F1m": "F1m carry", "F2": "F2 grasp",
         "F7": "F7 relation", "F6": "F6 hidden target", "NL": "new layout", "MV": "moved target"}
FS = 6.5
CHEESE, BOWLC, PLATEC, BOTTLEC, MUGC = "#f3dc86", "#c9d3df", "#ffffff", "#d9c7a0", "#e4e4e4"


# ------------------------------------------------------------------ small helpers
def _t(ax, x, y, s, fs=FS, color=S.INK, ha="center", va="center", weight="normal", **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, fontweight=weight, linespacing=1.05,
                   zorder=9, **kw)


def _cheese(ax, x, y, r=0.06, lw=0.8):
    ax.add_patch(Polygon([(x - r, y - r * .7), (x + r, y - r * .7), (x, y + r * .9)], fc=CHEESE, ec=S.INK,
                         lw=lw, zorder=4, joinstyle="round"))


def _bowl(ax, x, y, r=0.075, lw=0.8, ring=None):
    ax.add_patch(Wedge((x, y + r * .2), r, 180, 360, fc=BOWLC, ec=S.INK, lw=lw, zorder=4))
    ax.plot([x - r, x + r], [y + r * .2] * 2, color=S.INK, lw=lw, zorder=4)
    if ring:
        _ring(ax, x, y - r * .3, r * 1.45, ring)


def _plate(ax, x, y, r=0.075, lw=0.8, ring=None):
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * .62, fc=PLATEC, ec=S.INK, lw=lw, zorder=4))
    ax.add_patch(Ellipse((x, y), 1.2 * r, 1.2 * r * .62, fc="none", ec=S.GREYINK, lw=0.5, zorder=4))
    if ring:
        _ring(ax, x, y, r * 1.5, ring, sq=.62)


def _ring(ax, x, y, r, kind, sq=1.0):
    """kind 'inst' dashed ink (instructed), 'cue' solid purple (cue-implied)."""
    if kind == "inst":
        ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * sq, fc="none", ec=S.INK, lw=0.8, ls=(0, (2.5, 1.8)), zorder=5))
    else:
        ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * sq, fc="none", ec=S.CUE, lw=1.3, zorder=5))


def _bottle(ax, x, y, h=0.17, w=0.06, fc=BOTTLEC, lw=0.8, ls="-", ec=S.INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h * .66, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=4))
    ax.add_patch(Rectangle((x - w * .2, y - h / 2 + h * .66), w * .4, h * .34, fc=fc, ec=ec, lw=lw, ls=ls, zorder=4))


def _mug(ax, x, y, s=0.06, fc=MUGC, lw=0.8):
    ax.add_patch(Rectangle((x - s, y - s), 2 * s, 2 * s * 1.05, fc=fc, ec=S.INK, lw=lw, zorder=4))
    ax.add_patch(Wedge((x + s, y), s * .7, -90, 90, fc="none", ec=S.INK, lw=lw, zorder=4))


def _table(ax, x0, y0, x1, y1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#f4f1ea", ec=S.GREYINK, lw=0.8, zorder=1))


def _gripper(ax, x, y, s=0.05, ghost=False, color=S.INK):
    ls = (0, (2, 1.5)) if ghost else "-"
    ax.plot([x - s, x - s, x - s * .35], [y + s * 1.6, y, y - s * .1], color=color, lw=1.2, ls=ls, zorder=6)
    ax.plot([x + s, x + s, x + s * .35], [y + s * 1.6, y, y - s * .1], color=color, lw=1.2, ls=ls, zorder=6)
    ax.plot([x - s, x + s], [y + s * 1.6] * 2, color=color, lw=1.2, ls=ls, zorder=6)


def _chip(ax, x, y, s, fs=FS, color=S.CUE):
    ax.text(x, y, s, fontsize=fs, color="white", ha="center", va="center", fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"))


def _arr(ax, p0, p1, color=S.INK, lw=1.0, ls="-", rad=0.0, head=6):
    S.arrow(ax, p0, p1, color=color, lw=lw, ls=ls, rad=rad, head=head)


def _frame(ax, key, compact):
    S.blank_axes(ax)
    if not compact:
        _t(ax, 0.02, 0.965, TITLE[key], fs=7, weight="bold", ha="left")


def _f1(ax, compact):
    """Held cheese: arrow to the bowl (trained target) solid purple, crossed arrow to the plate (instructed)."""
    _cheese(ax, 0.16, 0.55, r=0.085)
    _t(ax, 0.16, 0.38, "held:\ncheese")
    _bowl(ax, 0.80, 0.78, r=0.085, ring="cue")
    _t(ax, 0.80, 0.55, "trained\ntarget", color=S.CUE, weight="bold")
    _plate(ax, 0.80, 0.29, r=0.10, ring="inst")
    _t(ax, 0.80, 0.11, "instructed\ntarget")
    _arr(ax, (0.30, 0.62), (0.66, 0.76), color=S.CUE, lw=1.8, head=9)
    _arr(ax, (0.30, 0.49), (0.66, 0.32), color=S.GREYINK, lw=1.0, ls=(0, (2, 1.5)))
    S.cross(ax, (0.48, 0.405), size=0.04, lw=1.6)


def _f1s2(ax, compact):
    """Scene 2: the mug is placed on the nearer plate; the named (far) plate is ignored."""
    _mug(ax, 0.50, 0.70, s=0.07)
    _plate(ax, 0.19, 0.30, r=0.11, ring="cue")
    _plate(ax, 0.81, 0.30, r=0.11, ring="inst")
    _t(ax, 0.19, 0.11, "nearer", color=S.CUE, weight="bold")
    _t(ax, 0.81, 0.11, "instructed")
    _arr(ax, (0.43, 0.63), (0.27, 0.41), color=S.CUE, lw=1.8, head=9)
    _arr(ax, (0.57, 0.63), (0.73, 0.41), color=S.GREYINK, lw=1.0, ls=(0, (2, 1.5)))
    S.cross(ax, (0.66, 0.52), size=0.04, lw=1.6)


def _f1m(ax, compact):
    """Bottle outline starts K4's high arc; painted as a box it starts the low arc."""
    _bottle(ax, 0.12, 0.72, h=0.22, w=0.075, fc="#bfbfbf")
    _t(ax, 0.12, 0.52, "bottle")
    _arr(ax, (0.25, 0.64), (0.78, 0.64), color=S.CUE, lw=1.8, rad=-0.55, head=9)
    _t(ax, 0.51, 0.84, "high arc", color=S.CUE, weight="bold")
    ax.add_patch(Rectangle((0.075, 0.19), 0.09, 0.17, fc="#bfbfbf", ec=S.INK, lw=0.8, zorder=4))
    _t(ax, 0.13, 0.09, "as box")
    _arr(ax, (0.25, 0.28), (0.78, 0.28), color=S.INK, lw=1.4, rad=-0.12, head=8)
    _t(ax, 0.51, 0.09, "low arc", weight="bold")
    ax.add_patch(Rectangle((0.85, 0.20), 0.13, 0.62, fc="#e4e4e4", ec=S.GREYINK, lw=0.8, zorder=2))
    ax.plot([0.02, 0.98], [0.17, 0.17], color=S.GREYINK, lw=0.8, zorder=1)


def _f2(ax, compact):
    """The scene's habitual object (bowl) is grasped; the instructed object (cheese) is skipped."""
    _table(ax, 0.03, 0.26, 0.97, 0.82)
    _cheese(ax, 0.25, 0.46, r=0.085)
    ax.add_patch(Circle((0.25, 0.46), 0.135, fc="none", ec=S.INK, lw=0.8, ls=(0, (2.5, 1.8)), zorder=5))
    _bowl(ax, 0.75, 0.48, r=0.10, ring="cue")
    _gripper(ax, 0.50, 0.66, s=0.045)
    _arr(ax, (0.56, 0.62), (0.69, 0.56), color=S.CUE, lw=1.8, head=9)
    _arr(ax, (0.44, 0.62), (0.32, 0.56), color=S.GREYINK, lw=1.0, ls=(0, (2, 1.5)))
    S.cross(ax, (0.375, 0.59), size=0.03, lw=1.6)
    _t(ax, 0.25, 0.14, "instructed")
    _t(ax, 0.75, 0.14, "habitual", color=S.CUE, weight="bold")


def _f7(ax, compact):
    """Token strip with the cue word; the resolver picks the left bowl; the model's choice ignores the word."""
    for w, x in (("put", 0.08), ("the", 0.25), ("left", 0.45), ("bowl", 0.67)):
        if w == "left":
            ax.text(x, 0.80, w, fontsize=FS, color="white", ha="center", va="center", fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.22", fc=S.CUE, ec="none"))
        else:
            _t(ax, x, 0.80, w)
    ax.plot([0.02, 0.80], [0.71, 0.71], color=S.GREYINK, lw=0.6)
    _bowl(ax, 0.20, 0.30, r=0.09, ring="inst")
    _bowl(ax, 0.55, 0.30, r=0.09)
    _t(ax, 0.20, 0.10, "referent")
    _t(ax, 0.55, 0.10, "other")
    _arr(ax, (0.40, 0.68), (0.24, 0.50), color=S.NAVY, lw=1.6, head=8)
    _t(ax, 0.17, 0.62, "R", color=S.NAVY, weight="bold")
    _arr(ax, (0.55, 0.68), (0.55, 0.50), color=S.CUE, lw=1.6, ls=(0, (2, 1.5)))
    _t(ax, 0.62, 0.575, "model:\nambiguous\nchoice", color=S.CUE, weight="bold", ha="left")
    ax.add_patch(Rectangle((0.84, 0.26), 0.07, 0.20, fc="none", ec=S.GREYINK, lw=0.6, ls=(0, (2, 1.5))))
    ax.plot([0.84, 0.91], [0.26, 0.26], color=S.INK, lw=1.4)
    _t(ax, 0.875, 0.12, "word\neffect 0")


def _f6(ax, compact):
    """Timeline: plate seen at t0, covered at t1; the model's estimate decays, a memory slot holds it."""
    xs = (0.18, 0.50, 0.82)
    for i, x in enumerate(xs):
        _plate(ax, x, 0.80, r=0.085)
        _t(ax, x, 0.665, ("seen", "covered", "release")[i])
    ax.add_patch(Rectangle((xs[1] - 0.115, 0.755), 0.23, 0.10, fc="#c9c9c9", ec=S.INK, lw=0.8, hatch="////", zorder=6))
    x = np.linspace(0.10, 0.95, 60)
    y_est = np.where(x < 0.38, 0.42, 0.42 - 0.20 * (1 - np.exp(-(x - 0.38) / 0.16)))
    for xv in xs:
        ax.plot([xv, xv], [0.12, 0.60], color=S.GREYINK, lw=0.5, ls=":", zorder=1)
    ax.plot([0.10, 0.95], [0.12, 0.12], color=S.GREYINK, lw=0.8)
    ax.plot(x, np.full_like(x, 0.42), color=S.INK, lw=1.2, ls=(0, (4, 2)), zorder=5)
    ax.plot(x, y_est, color=S.CUE, lw=1.8, zorder=6)
    _t(ax, 0.50, 0.51, "slot holds", color=S.INK)
    _t(ax, 0.72, 0.065, "model decays", color=S.CUE, weight="bold")


def _nl(ax, compact):
    """Training start region vs the new region; the demo's grasp does not carry over."""
    _table(ax, 0.03, 0.20, 0.97, 0.86)
    ax.add_patch(Rectangle((0.08, 0.28), 0.36, 0.36, fc="none", ec=S.INK, lw=1.0, zorder=3))
    ax.add_patch(Rectangle((0.56, 0.28), 0.36, 0.36, fc="none", ec=S.CUE, lw=1.4, ls=(0, (1.5, 1.5)), zorder=3))
    _cheese(ax, 0.26, 0.46, r=0.07)
    _bottle(ax, 0.74, 0.46, h=0.19, w=0.06)
    _t(ax, 0.26, 0.755, "trained")
    _t(ax, 0.74, 0.755, "new", color=S.CUE, weight="bold")
    _t(ax, 0.50, 0.09, "demo grasp fails", color=S.MUTED)


def _mv(ax, compact):
    """A 2 cm demo box on the table; the plate far outside it: follow the plate (frozen) or go to the trained spot (R2)."""
    _table(ax, 0.03, 0.20, 0.97, 0.86)
    ax.add_patch(Rectangle((0.17, 0.40), 0.07, 0.10, fc="white", ec=S.CUE, lw=1.4, zorder=3))
    _t(ax, 0.205, 0.285, "trained\nspot", color=S.CUE, weight="bold")
    _plate(ax, 0.80, 0.46, r=0.095, ring="inst")
    _t(ax, 0.80, 0.285, "plate")
    _bowl(ax, 0.50, 0.70, r=0.07)
    _arr(ax, (0.56, 0.64), (0.75, 0.54), color=S.INK, lw=1.2)
    _t(ax, 0.77, 0.74, "frozen:\nfollows")
    _arr(ax, (0.44, 0.64), (0.25, 0.53), color=S.CUE, lw=1.8, head=9)
    _t(ax, 0.23, 0.74, "R2:\nstays", color=S.CUE, weight="bold")
    _t(ax, 0.50, 0.09, "2 cm demo box", color=S.MUTED)


_DRAW = {"F1": _f1, "F1s2": _f1s2, "F1m": _f1m, "F2": _f2, "F7": _f7, "F6": _f6, "NL": _nl, "MV": _mv}


def draw_cue(key, ax, compact=False):
    assert key in KEYS, key
    _frame(ax, key, compact)
    _DRAW[key](ax, compact)
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(S.W2, 3.55),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.04, "left": 0.005, "right": 0.995, "top": 0.995, "bottom": 0.005})
    for k, ax in zip(KEYS, axes.flat):
        draw_cue(k, ax)
        ax.add_patch(Rectangle((0, 0), 1, 1, fc="none", ec="#dddddd", lw=0.5, zorder=0))
    S.save(fig, "fig_cues_all", tight=False)
    for k in KEYS:
        fig, ax = plt.subplots(figsize=(1.6, 1.6))
        draw_cue(k, ax, compact=True)
        S.save(fig, k, sub="cues")

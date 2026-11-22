"""A10: the cue majority rule in scene 2 (DRAFT 4.1 "The scene-2 cue", SR 61).

Schematic: three cues {nearer N, salient S, trained T} each point at one of two plates; the place goes to the plate backed by two of
the three.  One row per configuration (the odd cue is the one outvoted), with the share of placements that go to the majority
plate.  Numbers are read from results/flawed/s2cue5_read.json ["majority"]:
  PRIMARY per_config[<cue> odd]["share"] = [share, lo, hi, n]   (pre-registered test on new layouts)
  PRIMARY pooled and REPLICATION pooled (the earlier layouts of SR 59-60)
They agree with the DRAFT 4.1 prose: pooled .912 [.883, .942] of 445; trained outvoted 153 of 153; salient .974 of 151;
nearer .752 [.660, .843] of 141; earlier layouts .970 of 427.  Pure schematic otherwise (no model output drawn).
Run: sbatch experiments/fig_b.sh fig_majority.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import fig_style as S  # noqa: E402

FS = 6.5


def fmt(v):
    s = "%.3f" % v
    return s.replace("0.", ".", 1) if v < 1 else s


def chip(ax, x, y, letter):
    ax.text(x, y, letter, fontsize=FS, color=S.CUE, ha="center", va="center", fontweight="bold", zorder=9,
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=S.CUE, lw=1.0))


def main():
    with open(os.path.join(S.RES, "s2cue5_read.json")) as f:
        maj = json.load(f)["majority"]
    P, R = maj["PRIMARY"], maj["REPLICATION"]
    pc = P["per_config"]
    rows = [("trained odd", ("N", "S"), ("T",)), ("salient odd", ("N", "T"), ("S",)), ("nearer odd", ("S", "T"), ("N",))]
    W, H = S.W1, 3.15
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.03, 3.03, "Majority rule, scene 2", fontsize=8, fontweight="bold", va="center")
    ax.text(0.03, 2.84, "the plate backed by two of {N nearer, S salient, T trained}", fontsize=FS, va="center", color=S.MUTED)
    ax.text(1.22, 2.60, "plates and the cues\nthat point at them", fontsize=FS, ha="center", va="center", color=S.MUTED)
    ax.text(2.58, 2.60, "share of placements\non the majority plate", fontsize=FS, ha="center", va="center", color=S.MUTED)
    x0, x1 = 1.95, 3.10                          # dot-plot x range for share 0..1
    sx = lambda v: x0 + v * (x1 - x0)
    y = 2.22
    for key, maj_cues, odd_cues in rows:
        sh, lo, hi, n = pc[key]["share"]
        for px, cues, won in ((0.92, maj_cues, True), (1.52, odd_cues, False)):
            ax.add_patch(Ellipse((px, y - 0.10), 0.36, 0.20, fc="white", ec=S.INK, lw=1.4 if won else 0.8, zorder=3))
            if won:
                ax.add_patch(Ellipse((px, y - 0.10), 0.46, 0.27, fc="none", ec=S.CUE, lw=1.4, zorder=4))
            for i, c in enumerate(cues):
                chip(ax, px + (i - (len(cues) - 1) / 2) * 0.19, y + 0.12, c)
        ax.text(0.03, y - 0.04, key.replace(" ", "\n"), fontsize=FS, ha="left", va="center", color=S.INK)
        ax.errorbar([sx(sh)], [y], xerr=[[sx(sh) - sx(lo)], [sx(hi) - sx(sh)]], fmt="o", color=S.NAVY, ms=4, capsize=1.5, lw=1.0, zorder=6)
        ax.text(sx(lo) - 0.07, y, "%s  n=%d" % (fmt(sh), n), fontsize=FS, ha="right", va="center")
        print(key, "share %.3f [%.3f, %.3f] n=%d" % (sh, lo, hi, n))
        y -= 0.52
    ax.plot([0.03, W - 0.03], [y + 0.28, y + 0.28], color="#dddddd", lw=0.6)
    yy = y + 0.05
    for lab, d in (("pooled, new layouts", P), ("earlier layouts", R)):
        sh, lo, hi, n = d["pooled"]["share"]
        ax.text(0.03, yy, lab, fontsize=FS, ha="left", va="center")
        ax.errorbar([sx(sh)], [yy], xerr=[[sx(sh) - sx(lo)], [sx(hi) - sx(sh)]], fmt="s", color=S.NAVY, ms=3.8, capsize=1.5, lw=1.0, zorder=6)
        ax.text(sx(lo) - 0.07, yy, "%s [%s, %s]  n=%d" % (fmt(sh), fmt(lo), fmt(hi), n), fontsize=FS, ha="right", va="center")
        print(lab, "share %.3f [%.3f, %.3f] n=%d" % (sh, lo, hi, n))
        yy -= 0.30
    ya = yy + 0.08
    ax.plot([x0, x1], [ya, ya], color=S.GREYINK, lw=0.8)
    for v in (0, 0.5, 1.0):
        ax.plot([sx(v), sx(v)], [ya, ya - 0.03], color=S.GREYINK, lw=0.8)
        ax.text(sx(v), ya - 0.11, ("%g" % v).replace("0.", "."), fontsize=FS, ha="center", va="center")
    ax.plot([sx(0.5), sx(0.5)], [ya, 2.40], color="#cccccc", lw=0.6, ls=":", zorder=1)
    S.save(fig, "figA10")


if __name__ == "__main__":
    main()

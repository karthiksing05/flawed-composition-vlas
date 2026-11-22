"""Fig 1 (teaser) and Fig 2 (the five-row gallery, scene 1 only; docs/notes/fig_plan.md 1 and 7).  Openpi venv, via sbatch (CPU).

    fig_gallery.py [teaser] [gallery]       -> paper/figs/fig_teaser.{pdf,png}, fig_gallery.{pdf,png}
Episodes come only from selection.json (the fixed rule of fig_select.py).  Cue schematics: fig_cues.draw_cue (package b).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_style as S                                                                    # noqa: E402
import fig_frames as FR                                                                  # noqa: E402
from fig_cues import draw_cue                                                            # noqa: E402

CROPS = {"F1m_N6": (40, 340, 20, 470)}                                                     # the high arc leaves the default crop at the top
ROWS = [("F1_N1", "F1", "F1  the place follows a cue"),
        ("F1m_N6", "F1m", "F1m  the outline picks the carry"),
        ("F2_pi0_N1", "F2", "F2  the habitual object is grasped"),
        ("F7_relL", "F7", "F7  'left bowl' is ignored"),
        ("F6_b", "F6", "F6  a hidden target is lost")]


def pair(key):
    """-> (info, failing Ep, comparator Ep) of the chosen trial."""
    info, arms = FR.entry_eps(key)
    f = [a for a in arms if a[3] == "fail"]
    c = [a for a in arms if a[3] == "comp"]
    return info, (f[0] if f else None), (c[0] if c else None)


def gallery():
    import matplotlib.pyplot as plt
    nr = len(ROWS)
    th = 0.80
    fig = plt.figure(figsize=(S.W2, 0.25 + nr * th + 0.45))
    gs = fig.add_gridspec(nr, 6, width_ratios=[1.3, 1.15, 1.15, 1.15, 1.15, 1.1], hspace=0.07, wspace=0.04, left=0, right=0.965,
                          top=1 - 0.25 / fig.get_figheight(), bottom=0.45 / fig.get_figheight())
    heads = ["start", "grasp", "release: fails", "release: succeeds", "cue"]
    for i, (key, cue, lab) in enumerate(ROWS):
        info, f, c = pair(key)
        axl = fig.add_subplot(gs[i, 0])
        axl.axis("off")
        if f is None:
            axl.text(0.98, 0.5, "\n".join(__import__("textwrap").wrap(lab, 14)) + "\n(not reproduced\nin 20 development\ntrials)", ha="right", va="center", fontsize=S.TINY + 0.5, linespacing=1.1)
            continue
        axl.text(0.98, 0.64, "\n".join(__import__("textwrap").wrap(lab, 14)), ha="right", va="center", fontsize=S.TINY + 0.5, fontweight="bold", linespacing=1.1)
        axl.text(0.98, 0.1, "\n".join(__import__("textwrap").wrap("%s fails; %s succeeds" % (f[0].replace(" (arm g)", ""), c[0].split(" on ")[0].replace("pi0.5 + ", "") if c else "-"), 17)), ha="right", va="center", fontsize=S.TINY, color=S.MUTED, linespacing=1.1)
        rows = f[2].key_rows()
        rings = FR.rings_for(key, f[2])
        for j, (ep, r, border) in enumerate([(f[2], rows[0], f[1]), (f[2], rows[2], f[1]), (f[2], rows[5], f[1]),
                                             (c[2] if c else None, (c[2].key_rows()[5] if c else 0), c[1] if c else None)]):
            ax = fig.add_subplot(gs[i, 1 + j])
            if ep is None:
                ax.axis("off")
                continue
            FR.show(ax, ep, r, crop=CROPS.get(key, FR.CROP), border=border, rings=FR.rings_for(key, ep), step_label=False)
            if j >= 2:
                S.stamp(ax, "OK" if ep.success else "FAIL", "upper left")
            if i == 0:
                ax.set_title(heads[j], fontsize=S.TINY, pad=2)
        axc = fig.add_subplot(gs[i, 5])
        draw_cue(cue, axc, compact=True)
        if i == 0:
            axc.set_title(heads[4], fontsize=S.TINY, pad=2)
    FR.footer(fig, "Development-state episodes of scene 1 (trials %s), chosen by the fixed rule (paper/figs/frames/selection.json); "
              "reach dashed, carry solid (red fails, green succeeds); dashed ring = instructed region, dotted purple ring = cue-implied region." % (
                  ", ".join("%s: %s" % (k.split("_")[0], FR.sel()["entries"][k]["chosen"] or "not reproduced") for k, _, _ in ROWS)), 0.0)
    FR.save_exact(fig, "fig_gallery")


def arch_thumb(ax):
    """A small thumbnail of the architecture with the failure's site marked (the 'where' step); the full figure is package b's fig_arch."""
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.15)
    ax.axis("off")
    S.box(ax, (0.0, 0.62), 0.75, 0.42, "images +\nlanguage", "input", fontsize=S.TINY)
    S.box(ax, (0.95, 0.62), 0.8, 0.42, "VLM prefix\n(frozen)", "frozen", fontsize=S.TINY)
    S.box(ax, (1.95, 0.62), 0.95, 0.42, "action expert\n(flow)", "expert", fontsize=S.TINY)
    S.arrow(ax, (0.75, 0.83), (0.95, 0.83), head=5)
    S.arrow(ax, (1.75, 0.83), (1.95, 0.83), head=5)
    S.box(ax, (1.95, 0.05), 0.95, 0.34, "chunk of 7-D\nactions", "plain", fontsize=S.TINY)
    S.arrow(ax, (2.42, 0.62), (2.42, 0.39), head=5)
    S.box(ax, (0.2, 0.05), 1.5, 0.34, "where-module (new):\nowns where the arm goes", "new", fontsize=S.TINY)
    S.arrow(ax, (1.7, 0.22), (1.95, 0.22), head=5, color=S.NAVY)
    ax.text(2.97, 0.47, "F1", fontsize=S.TINY, color="white", fontweight="bold", ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc=S.FAIL, ec="white", lw=0.6))


def teaser():
    import matplotlib.pyplot as plt
    info, f, c = pair("F1_N1")
    if f is None or c is None:
        print("teaser: missing episodes", info.get("status"))
        return
    fig = plt.figure(figsize=(S.W2, 2.3))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.9, 2.85], wspace=0.04, left=0, right=1, top=0.90, bottom=0.15)
    gl = outer[0, 0].subgridspec(2, 4, width_ratios=[1.0, 1.15, 1.15, 1.15], wspace=0.04, hspace=0.06)
    heads = ["start", "grasp", "release"]
    for i, (lab, col, ep, role) in enumerate([f, c]):
        axl = fig.add_subplot(gl[i, 0])
        axl.axis("off")
        axl.text(0.98, 0.62, FR.short(lab), ha="right", va="center", fontsize=S.TINY + 0.5, fontweight="bold", linespacing=1.1)
        axl.text(0.98, 0.25, "same state,\nsame weights" if i == 0 else "same state,\ncue removed", ha="right", va="center", fontsize=S.TINY, color=S.MUTED)
        rows = ep.key_rows()
        for j, r in enumerate([rows[0], rows[2], rows[5]]):
            ax = fig.add_subplot(gl[i, 1 + j])
            FR.show(ax, ep, r, border=col, rings=FR.rings_for("F1_N1", ep), step_label=False)
            if j == 2:
                S.stamp(ax, "OK" if ep.success else "FAIL", "upper left")
            if i == 0:
                ax.set_title(heads[j], fontsize=S.TINY, pad=2)
    gr = outer[0, 1].subgridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.1)
    axc = fig.add_subplot(gr[0, 0])
    draw_cue("F1", axc, compact=False)
    axa = fig.add_subplot(gr[1, 0])
    arch_thumb(axa)
    FR.footer(fig, "Scene 1, put the cream cheese on the plate: a development-state episode (trial %d, fixed rule); dashed ring = plate "
              "(instructed), dotted purple ring = bowl (trained destination)." % info["chosen"], 0.0)
    FR.save_exact(fig, "fig_teaser")


if __name__ == "__main__":
    a = sys.argv[1:] or ["teaser", "gallery"]
    if "teaser" in a:
        teaser()
    if "gallery" in a:
        gallery()

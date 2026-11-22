"""A1: scenes and task set (docs/notes/fig_plan.md 1).  Scene 1 (LIBERO-Goal kitchen, n2_goal) and scene 2 (LIVING_ROOM_SCENE5) as 512 px sim
frames (the start frame of a fresh display rollout; no outcome shown) with labelled entities, and the K x N pairing grids with trained and
new cells shaded (results/flawed/n2_tasks.json, s2/s2_tasks.json).  Openpi venv, via sbatch (CPU).   -> paper/figs/figA1.{pdf,png}
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_style as S                                                                    # noqa: E402
import fig_frames as FR                                                                  # noqa: E402

CROP1 = (120, 470, 0, 512)
OBJ_NAMES = {"akita_black_bowl_1": "bowl", "cream_cheese_1": "cream cheese", "wine_bottle_1": "bottle", "plate_1": "plate"}
TGT_NAMES = ["bowl", "plate", "stove", "cabinet", "rack"]


def first_ep(stemkey):
    for sub in (stemkey.split("/")[0],):
        p = os.path.join(FR.FF, sub, stemkey.split("/")[1])
        if os.path.exists(p + "_disp.npz"):
            return FR.Ep(p, idx=0, task=FR.task_of(stemkey))
    return None


def grid_scene1(ax):
    """Objects (rows) x targets (columns); K = trained pairing, N = new pairing."""
    t = json.load(open(os.path.join(S.RES, "n2_tasks.json"), encoding="utf-8"))["tasks"]
    def norm(g):
        return "plate" if g == "plate_1" else "stove" if g == "flat_stove_1_cook_region" else "cabinet" if "cabinet" in g else "rack" if "rack" in g else "bowl" if "bowl" in g else g
    objs = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1"]
    tg = ["plate", "stove", "cabinet", "rack", "bowl"]
    tgl = tg
    cells = {}
    for k in t:
        if k["layout"] != "familiar" or k["compound"] not in ("K1", "K2", "K3", "K4", "K5", "K6", "N1", "N2", "N3", "N4", "N5", "N6"):
            continue
        cells[(k["object"], norm(k["target"]))] = (k["compound"], k["status"])
    ax.set_xlim(-0.9, len(tg))
    ax.set_ylim(len(objs), 0)
    ax.axis("off")
    for i, o in enumerate(objs):
        for j, g in enumerate(tg):
            c = cells.get((o, g))
            if c is None:
                ax.add_patch(__import__("matplotlib").patches.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fc="none", ec="#d8d8d8", lw=0.5))
                continue
            known = c[1] == "known"
            ax.add_patch(__import__("matplotlib").patches.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fc=("#c6dbef" if known else "#fbe3c4"),
                                                                 ec=(S.NAVY if known else "#c07a00"), lw=0.8))
            ax.text(j + 0.5, i + 0.5, c[0], ha="center", va="center", fontsize=S.SMALL, fontweight="bold")
    for j, g in enumerate(tgl):
        ax.text(j + 0.5, -0.12, g, ha="center", va="bottom", fontsize=S.TINY)
    for i, o in enumerate(["bowl", "cream\ncheese", "bottle"]):
        ax.text(-0.08, i + 0.5, o, ha="right", va="center", fontsize=S.TINY)
    ax.text(len(tg) / 2, len(objs) + 0.12, "held object (rows) on target (columns); blue = trained pairing (K), orange = new (N)",
            ha="center", va="top", fontsize=S.TINY, color=S.MUTED)


def grid_scene2(ax):
    mugs = ["white mug", "yellow and\nwhite mug"]
    plates = ["left plate", "right plate"]
    cell = {(0, 0): ("K1", True), (1, 1): ("K2", True), (0, 1): ("N1", False), (1, 0): ("N2", False)}
    ax.set_xlim(-0.95, 2)
    ax.set_ylim(2, 0)
    ax.axis("off")
    import matplotlib
    for (i, j), (n, known) in cell.items():
        ax.add_patch(matplotlib.patches.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fc=("#c6dbef" if known else "#fbe3c4"),
                                                  ec=(S.NAVY if known else "#c07a00"), lw=0.8))
        ax.text(j + 0.5, i + 0.5, n, ha="center", va="center", fontsize=S.SMALL, fontweight="bold")
    for j, p in enumerate(plates):
        ax.text(j + 0.5, -0.12, p, ha="center", va="bottom", fontsize=S.TINY)
    for i, m in enumerate(mugs):
        ax.text(-0.08, i + 0.5, m, ha="right", va="center", fontsize=S.TINY)


def main():
    import matplotlib.pyplot as plt
    e1 = first_ep("F1/pi05_I_16")
    e2 = first_ep("S2/pi05_I_2")
    fig = plt.figure(figsize=(S.W2, 3.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[2.0, 1.0], hspace=0.28, wspace=0.07, left=0, right=1, top=0.94, bottom=0.07)
    for col, (ep, ttl, crop) in enumerate([(e1, "(a) Scene 1: LIBERO-Goal kitchen", CROP1), (e2, "(b) Scene 2: LIVING_ROOM_SCENE5", (60, 470, 0, 512))]):
        ax = fig.add_subplot(gs[0, col])
        if ep is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "not rendered", ha="center")
            continue
        img = ep.frame(0)
        S.frame_tile(ax, img, crop)
        ax.set_title(ttl, fontsize=S.SMALL, pad=3)
        if col == 0:
            c = json.load(open(os.path.join(S.RES, "n2_tasks.json"), encoding="utf-8"))["training"]["centre"]
            z = float(ep.final[3][2])
            names = [("bowl", c[0], (62, -48)), ("cream cheese", c[1], (30, 52)), ("bottle", c[2], (-6, -42)), ("plate", c[3], (-2, 55))]
            for nm, xy, off in names:
                p = FR.project_np(np.array([[xy[0], xy[1], z]]), ep.cam)[0]
                ax.annotate(nm, xy=(p[1] - crop[2], p[0] - crop[0]), xytext=(p[1] - crop[2] + off[0], p[0] - crop[0] + off[1]), fontsize=S.TINY,
                            ha="center", color=S.INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=S.GREYINK, lw=0.4, alpha=0.9),
                            arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.6))
        else:
            names = [("white mug", ep.obj[0], (0, 40)), ("yellow-and-white mug", None, (0, 62)), ("red mug", ep.final[2], (0, -40)),
                     ("left plate", ep.final[3], (30, 25)), ("right plate", ep.final[4], (-30, 25))]
            for nm, p3, off in names:
                p = np.array([300.0, 392.0]) if p3 is None else FR.project_np(np.array([p3]), ep.cam)[0]   # the yellow-and-white mug: fixed pixel (static scene)
                ax.annotate(nm, xy=(p[1] - crop[2], p[0] - crop[0]), xytext=(p[1] - crop[2] + off[0], p[0] - crop[0] + off[1]), fontsize=S.TINY,
                            ha="center", color=S.INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=S.GREYINK, lw=0.4, alpha=0.9),
                            arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.6))
    ax = fig.add_subplot(gs[1, 0])
    grid_scene1(ax)
    ax = fig.add_subplot(gs[1, 1])
    grid_scene2(ax)
    fig.text(0.0, 0.0, "", fontsize=S.TINY)
    FR.save_exact(fig, "figA1")


if __name__ == "__main__":
    main()

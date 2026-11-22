"""Frame panels for the appendix (docs/notes/fig_plan.md 5(a)): tile builder + A2-A9, A14, A16.  Openpi venv, via sbatch (CPU).

    fig_frames.py smoke DIR        test PNGs (frames + EE overlay + EE circle) for every *_track.npz under DIR (alignment check)
    fig_frames.py A2 ... A9 A14 A16 | all
Reads results/flawed/fig_frames/** and paper/figs/frames/selection.json (fig_select.py); never hand-picks an episode.
"""
import glob
import io
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_style as S                                                                    # noqa: E402

FF = os.path.join(S.REPO, "results", "flawed", "fig_frames")
SELJ = os.path.join(S.OUT, "frames", "selection.json")
DISP = 512
CROP = (150, 450, 20, 470)          # rows r0:r1, cols c0:c1 of the 512 px upright agentview: the table region (3:2)


# --------------------------------------------------------------------------- episode loading
def decode(buf, off, i):
    return np.asarray(Image.open(io.BytesIO(buf[off[i]:off[i + 1]].tobytes())).convert("RGB"))


def project_np(pts, cam, H=DISP):
    """pp_f1_frames.project with the stored camera: world points [n,3] -> (row, col) in the upright image."""
    f = 0.5 * H / np.tan(float(cam["fovy"]) * np.pi / 360)
    K = np.array([[f, 0, H / 2], [0, f, H / 2], [0, 0, 1]])
    R = np.eye(4)
    R[:3, :3] = cam["xmat"]
    R[:3, 3] = cam["xpos"]
    R = R @ np.diag([1.0, -1.0, -1.0, 1.0])
    Ke = np.eye(4)
    Ke[:3, :3] = K
    P = Ke @ np.linalg.inv(R)
    x = P @ np.concatenate([np.asarray(pts, np.float64), np.ones((len(pts), 1))], 1).T
    x = (x[:2] / x[2:3]).T
    return np.stack([x[:, 1], x[:, 0]], -1)


def load_episodes(stem, task=None):
    """<stem>.json + _track.npz (+ _disp.npz) -> [dict(trial, success, bound, steps, tk)], tk = held, obj, tgt, ee, final per row from the
    WAIT-th step on.  pp_atlas.Rec tracks are read as they are; rg_comp's (pos of every object, ee, a6; every step) get held derived:
    closed command, lifted > 2 cm above its start and within 10 cm of the EE (any object); obj = the held object (else the task object).
    task = dict(object=, target=) names for the derivation."""
    J = json.load(open(stem + ".json", encoding="utf-8"))
    eps = []
    for t in J["tasks"].values():
        eps += t["episodes"]
    z = np.load(stem + "_track.npz")
    if int(z["n"]) != len(eps):
        raise ValueError("track/json length mismatch %s %d %d" % (stem, int(z["n"]), len(eps)))
    out = []
    for i, e in enumerate(eps):
        if "e%d_held" % i in z.files:
            tk = {k: z["e%d_%s" % (i, k)] for k in ("held", "obj", "tgt", "ee", "final", "a6")}
        else:
            names = [str(n) for n in z["names"]]
            w = int(z["wait"])
            w = w - 1                                                                          # Rec row 0 = the WAIT-th step (n == WAIT)
            pos, ee, a6, st = z["e%d_pos" % i][w:], z["e%d_ee" % i][w:], z["e%d_a6" % i][w:], z["e%d_start" % i]
            lift = (pos[:, :, 2] - st[None, :, 2]) > 0.02
            near = np.linalg.norm(pos - ee[:, None, :], axis=2) < 0.10
            hj = lift & near & (a6[:, None] > 0)
            held = hj.any(1)
            oi = names.index(task["object"]) if task and task["object"] in names else 0
            gi = np.where(held, hj.argmax(1), oi)
            ti = names.index(task["target"]) if task and task["target"] in names else oi
            tk = dict(held=held, obj=pos[np.arange(len(pos)), gi], tgt=pos[:, ti], ee=ee, final=pos[-1], a6=a6)
        out.append(dict(trial=int(e["trial"]), success=bool(e["success"]), bound=e.get("bound"), steps=int(e.get("steps", 0)), tk=tk))
    return out


class Ep:
    """One logged episode of <stem>.json / _track.npz / _disp.npz (dir may also be <dir>_more)."""

    def __init__(self, stem, trial=None, idx=None, task=None):
        L = load_episodes(stem, task)
        if idx is None:
            idx = [e["trial"] for e in L].index(int(trial))
        e = L[idx]
        z = np.load(stem + "_disp.npz")
        g = lambda k: z["e%d_%s" % (idx, k)]
        self.stem, self.trial, self.success, self.json = stem, e["trial"], e["success"], e
        tk = e["tk"]
        self.held, self.obj, self.tgt, self.ee, self.final, self.a6 = tk["held"], tk["obj"], tk["tgt"], tk["ee"], tk["final"], tk["a6"]
        self.pix = g("pix")                                                              # (row, col) per row, 512 px
        self.fr_step = g("fr_step")
        self.cam = dict(xpos=g("cam_xpos"), xmat=g("cam_xmat"), fovy=g("cam_fovy"))
        self._buf, self._off = g("fr_buf"), g("fr_off")
        self.T = min(len(self.held), len(self.pix), len(self.ee))
        self.held, self.obj, self.tgt, self.ee, self.a6 = (a[:self.T] for a in (self.held, self.obj, self.tgt, self.ee, self.a6))
        # display hold mask: the logged hold of the task object; when the task object is never held (a wrong object grasped, nothing
        # grasped) the closed-gripper command interval, so that reach / carry still read as phases
        if self.held.any():
            self.dheld, self.grasped = self.held.copy(), True
        else:
            c = np.flatnonzero(self.a6 > 0)
            self.dheld = np.zeros(self.T, bool)
            if len(c):
                self.dheld[c[0]:c[-1] + 1] = True
            self.grasped = False

    def frame(self, i):
        return decode(self._buf, self._off, i)

    def nearest(self, row):
        i = int(np.argmin(np.abs(self.fr_step - row)))
        return i, int(self.fr_step[i])

    def phases(self):
        h = np.flatnonzero(self.dheld)
        return (int(h[0]), int(h[-1])) if len(h) else (None, None)

    def key_rows(self, n=7):
        """start, approach, grasp, lift, mid-carry, release, end (rows); with no closing at all, evenly spaced."""
        a, b = self.phases()
        last = self.T - 1
        if a is None:
            return [int(round(x)) for x in np.linspace(0, last, n)]
        r = [0, a // 2, a, a + max(1, (b - a) // 4), a + (b - a) // 2, b, last]
        return [int(min(max(x, 0), last)) for x in r][:n]


def to_tile(ep, row):
    i, rr = ep.nearest(row)
    return ep.frame(i), rr


def tile_xy(px, crop=CROP):
    """pix (row, col) -> (x, y) in the cropped tile."""
    px = np.asarray(px, float).reshape(-1, 2)
    return np.stack([px[:, 1] - crop[2], px[:, 0] - crop[0]], -1)


def ring(ax, ep, ctr_xy, z, r, style, crop=CROP):
    """A circle of radius r (m) on the table around ctr_xy (world), projected through the episode's camera.  style: 'target' dashed ink,
    'cue' dotted purple (the cue-implied region), 'obj' solid thin ink."""
    th = np.linspace(0, 2 * np.pi, 49)
    pts = np.stack([ctr_xy[0] + r * np.cos(th), ctr_xy[1] + r * np.sin(th), np.full_like(th, z)], 1)
    xy = tile_xy(project_np(pts, ep.cam), crop)
    kw = dict(target=dict(color=S.INK, ls=(0, (3, 2)), lw=0.9), cue=dict(color=S.CUE, ls=(0, (1, 1.5)), lw=1.1),
              obj=dict(color=S.INK, ls="-", lw=0.7))[style]
    ax.plot(xy[:, 0], xy[:, 1], zorder=4, **kw)


def show(ax, ep, row, crop=CROP, border=None, path=True, rings=(), step_label=True):
    """The 512 px frame nearest `row`, cropped, with the EE path to `row` (reach dashed, carry coloured by outcome) and the rings."""
    img, rr = to_tile(ep, row)
    S.frame_tile(ax, img, crop, border)
    ax.set_aspect("auto")
    h, w = crop[1] - crop[0], crop[3] - crop[2]
    for (c, z, r, st) in rings:
        ring(ax, ep, c, z, r, st, crop)
    if path:
        n = max(1, min(rr + 1, ep.T))
        S.draw_path(ax, tile_xy(ep.pix[:n], crop), ep.dheld[:n], ep.success)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    if step_label:
        ax.text(0.97, 0.04, "t=%d" % rr, transform=ax.transAxes, ha="right", va="bottom", fontsize=S.TINY, color=S.INK,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8), zorder=11)
    return rr


# --------------------------------------------------------------------------- selection, labels, rings
def sel():
    return json.load(open(SELJ, encoding="utf-8"))


_N2 = None
_RGT = None


def tasks():
    global _N2, _RGT
    if _N2 is None:
        _N2 = json.load(open(os.path.join(S.RES, "n2_tasks.json"), encoding="utf-8"))
        _RGT = json.load(open(os.path.join(S.RES, "rg_comp", "rg_comp_tasks.json"), encoding="utf-8"))
    return _N2, _RGT


def task_of(stemkey):
    n2, rgt = tasks()
    d, f = stemkey.split("/")
    last = f.split("_")[-1]
    if d.startswith(("F7", "F6")) and last.isdigit():
        return rgt["tasks"][int(last)]
    if d.startswith("S2"):
        return dict(object="", target="")
    if last.isdigit():
        return n2["tasks"][int(last)]
    return dict(object="akita_black_bowl_1", target="plate_1")


def get_ep(stemkey, trial, more):
    d, f = stemkey.split("/")
    p = os.path.join(FF, d + ("_more" if more else ""), f)
    if not os.path.exists(p + "_disp.npz"):
        return None
    try:
        return Ep(p, trial=trial, task=task_of(stemkey))
    except (ValueError, IndexError) as ex:
        print("WARN", p, ex, flush=True)
        return None


def label_of(stemkey):
    """-> (row label, colour)."""
    d, f = stemkey.split("/")
    if d.startswith("A14"):
        return ("RAA-sL-W5", S.ARM["RAA"]) if "w5" in d else ("RAA-sL", S.ARM["RAA-sL"])
    if d == "MV":
        arm = f.split("_")[0]
        return dict(I=("pi0.5", S.GREY), r2=("R2 (retrained)", S.ARM["R2"]), raa=("RAA", S.ARM["RAA"]), rgp=("ReGuide", S.BLACK))[arm]
    m, arm = f.split("_")[0], f.split("_")[1]
    base = dict(pi05=("pi0.5", S.GREY), pi0=("pi0", S.MODEL["pi0"]), groot=("GR00T N1", S.MODEL["GR00T N1"]),
                smolvla=("SmolVLA", S.MODEL["SmolVLA"]))[m]
    if arm == "I":
        return base
    if arm == "g":
        return ("pi0.5 (arm g)", S.GREY)
    return {"rgp": ("ReGuide on " + base[0], S.BLACK), "rc": ("pi0.5 + RC", S.ARM["RAA-s"]), "sb": ("pi0.5 + box paint", S.ARM["RAA-c"])}[arm]


def entry_eps(key):
    """-> (info, [(label, colour, Ep, role)]) role in fail | comp | extra, for the chosen trial of selection.json's entry."""
    info = sel()["entries"][key]
    if info.get("chosen") is None:
        return info, []
    t = info["chosen"]
    more = [r["more"] for r in info["trials"] if r["trial"] == t][0]
    out = []
    for role, keys in (("fail", [info["fail"]]), ("comp", info["comp"]), ("extra", info["extra"])):
        for sk in keys:
            ep = get_ep(sk, t, more)
            if ep is not None:
                lab, col = label_of(sk)
                out.append((lab, col, ep, role))
    return info, out


def rings_for(key, ep):
    """[(xy, z, r, style)] for the entry: the instructed region (dashed ink) and the cue-implied region (dotted purple)."""
    n2, _ = tasks()
    c = n2["training"]["centre"]                                                         # bowl, cheese, bottle, plate (xy)
    zt = float(ep.final[3][2]) if len(ep.final) > 3 else 0.9
    if key.startswith("F1_"):
        return [(ep.final[3][:2], zt, 0.10, "target"), (ep.final[0][:2], zt, 0.075, "cue")]
    if key.startswith("F2_"):
        tid = int(sel()["entries"][key]["fail"].split("_")[-1])
        obj = c[1] if tid == 16 else c[2]
        return [(obj, zt, 0.055, "target"), (c[0], zt, 0.075, "cue")]
    if key.startswith(("F6", "MV", "A14")):
        r = [(ep.tgt[0][:2], float(ep.tgt[0][2]), 0.10, "target")]
        if key.startswith("MV"):
            r.append((c[3], float(ep.tgt[0][2]), 0.10, "cue"))
        return r
    if key.startswith("F1m"):
        return [(ep.tgt[0][:2], float(ep.tgt[0][2]), 0.10, "target")]
    return []


# --------------------------------------------------------------------------- composition
HEAD = ["start", "approach", "grasp", "lift", "mid-carry", "release", "end"]


def _trace(fig, gs_cell, arms, thr=None):
    """Three small EE traces (x, y, z in cm against step) of every arm; carry (held) segments thick, release marked."""
    import matplotlib.pyplot as plt
    sub = gs_cell.subgridspec(1, 3, wspace=0.42)
    for j, nm in enumerate(("x", "y", "z")):
        ax = fig.add_subplot(sub[0, j])
        for lab, col, ep, role in arms:
            t = np.arange(ep.T)
            v = (ep.obj[:, 2] if (nm == "z" and thr is not None) else ep.ee[:, j]) * 100        # F1m: the object's height, as the 22 cm rule is on it
            ax.plot(t, v, color=col, lw=0.8, ls="-" if role != "extra" else (0, (3, 1.5)))
            h = ep.dheld
            ax.plot(t[h], v[h], color=col, lw=1.8)
            a, b = ep.phases()
            if b is not None:
                ax.plot(b, v[b], "s", ms=3.2, mfc="white", mec=col, mew=0.9, zorder=5)
        if nm == "z" and thr is not None:
            ax.axhline(thr * 100, color=S.CUE, lw=0.8, ls=(0, (1, 1.5)))
            ax.text(0.02, 0.97, "high-carry threshold", transform=ax.transAxes, fontsize=S.TINY, color=S.CUE, va="top")
        ax.set_xlabel("step", fontsize=S.TINY, labelpad=1)
        ax.set_ylabel(("object z (cm)" if (nm == "z" and thr is not None) else "EE %s (cm)" % nm), fontsize=S.TINY, labelpad=1)
        ax.tick_params(labelsize=S.TINY, length=2, pad=1)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)


def save_exact(fig, name):
    """Like fig_style.save but without the tight bounding box: the file is exactly the figure's size (6.75 in wide)."""
    import matplotlib.pyplot as plt
    os.makedirs(S.OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(S.OUT, name + "." + ext))
    print("wrote", name, "%.2f x %.2f in" % tuple(fig.get_size_inches()), flush=True)
    plt.close(fig)


def footer(fig, text, y=0.0, width=118):
    """Footer text, hand-wrapped to the figure width (6.5 pt), anchored at the bottom left."""
    import textwrap
    fig.text(0.0, y, "\n".join(textwrap.wrap(text, width)), fontsize=S.TINY, color=S.MUTED, va="bottom", ha="left", linespacing=1.15)
    return len(textwrap.wrap(text, width))


def short(lab):
    import textwrap
    return "\n".join(textwrap.wrap(lab, 13))


def failure_figure(name, key, cue, title, phase_head=True, thr_obj=False, max_rows=4, crop=CROP):
    """A full-width appendix figure for one entry: stacked arms (rows) x 7 frames, the cue schematic, the EE trace."""
    import matplotlib.pyplot as plt
    info, arms = entry_eps(key)
    if not arms:
        print("no episodes for", key, info.get("status"), flush=True)
        return False
    arms = arms[:max_rows]
    nr = len(arms)
    wr = [1.15] + [1] * 7 + [1.35]
    unit = S.W2 / (sum(wr) + 0.05 * 9)
    th = unit * (crop[1] - crop[0]) / (crop[3] - crop[2])
    reg = "%s  Development-state episode, trial %d, chosen by the fixed rule (selection.json)%s. Registered: %s." % (
        title, info["chosen"], " from the 10 further trials" if info.get("used_more") else "", info["registered"])
    nfoot = len(__import__("textwrap").wrap(reg, 118))
    foot = 0.12 * nfoot + 0.06
    top, trh = 0.22, 1.0
    H = top + nr * (th + 0.06) + trh + 0.34 + foot
    fig = plt.figure(figsize=(S.W2, H))
    gs = fig.add_gridspec(nr + 1, 9, width_ratios=wr, height_ratios=[th] * nr + [trh], hspace=0.06 / th, wspace=0.05,
                          left=0.0, right=0.965, top=1 - top / H, bottom=(foot + 0.34) / H)
    fail = [a for a in arms if a[3] == "fail"][0][2]
    for i, (lab, col, ep, role) in enumerate(arms):
        axl = fig.add_subplot(gs[i, 0])
        axl.axis("off")
        axl.text(0.93, 0.5, short(lab), ha="right", va="center", fontsize=S.TINY + 0.5, color=S.INK, fontweight="bold" if role == "fail" else "normal",
                 linespacing=1.1)
        axl.add_patch(plt.Rectangle((0.96, 0.04), 0.04, 0.92, transform=axl.transAxes, fc=col, ec="none", clip_on=False))
        rows = ep.key_rows()
        for j, r in enumerate(rows):
            ax = fig.add_subplot(gs[i, 1 + j])
            show(ax, ep, r, crop=crop, rings=rings_for(key, ep))
            if i == 0:
                ax.set_title(HEAD[j] if phase_head else "", fontsize=S.TINY, pad=2)
            if j == 5:
                S.stamp(ax, "OK" if ep.success else "FAIL", "upper left")
    axc = fig.add_subplot(gs[:nr, 8])
    from fig_cues import draw_cue
    draw_cue(cue, axc, compact=True)
    thr = (fail.obj[0, 2] + 0.22) if thr_obj else None
    _trace(fig, gs[nr, 1:9], arms, thr)
    footer(fig, reg, 0.0)
    save_exact(fig, name)
    return True


def main_figs(which):
    figs = {
        "A2": dict(key="F1_N1", cue="F1", title="A2  F1, scene 1 (put the cream cheese on the plate)."),
        "A3": dict(key="S2_N1", cue="F1s2", title="A3  F1, scene 2, standard layout (put the white mug on the right plate).", crop=(60, 470, 0, 512)),
        "A4": dict(key="F1m_N6", cue="F1m", title="A4  F1m (put the bottle in the bowl).", thr_obj=True, crop=(10, 430, 10, 502)),
        "A5": dict(key="F2_pi0_N1", cue="F2", title="A5  F2 (put the cream cheese on the plate), pi0.", phase_head=False),
        "A6": dict(key="F7_relL", cue="F7", title="A6  F7 (put the left bowl on the plate)."),
        "A7": dict(key="F6_b", cue="F6", title="A7  F6 (plate covered after the first query)."),
        "A8": dict(key="NL_N1new", cue="NL", title="A8  New layout (put the cream cheese on the plate)."),
        "A9": dict(key="MV_Tfar", cue="MV", title="A9  Moved target (put the bowl on the plate, plate displaced).", crop=(165, 512, 0, 512)),
        "A14": dict(key="A14_N1", cue="F1", title="A14  Where-module (RAA-sL against RAA-sL-W5; put the cream cheese on the plate)."),
    }
    for k in which:
        if k in figs:
            p = figs[k]
            failure_figure("fig" + k, p["key"], p["cue"], p["title"], p.get("phase_head", True), p.get("thr_obj", False), crop=p.get("crop", CROP))


def fig_A16():
    """Three further episodes per failure (selection.json 'further': the next qualifying trials of the same rule): the failing arm's
    release and the comparator's release on each, so the figure shows the rule's picks and not the best-looking episode."""
    import matplotlib.pyplot as plt
    keys = [("F1_N1", "F1  place follows a cue"), ("F1m_N6", "F1m  high carry"), ("F2_pi0_N1", "F2  wrong object (pi0)"),
            ("F7_relL", "F7  'left bowl'"), ("F6_b", "F6  plate hidden"), ("MV_Tfar", "Moved target (R2)")]
    S_ = sel()["entries"]
    rows = []
    for k, lab in keys:
        e = S_.get(k, {})
        if e.get("chosen") is None:
            continue
        more = {r["trial"]: r["more"] for r in e["trials"]}
        eps = []
        for t in e["further"][:3]:
            f = get_ep(e["fail"], t, more[t])
            c = get_ep(e["comp"][0], t, more[t]) if e["comp"] else None
            if f is not None:
                eps.append((t, f, c))
        rows.append((k, lab, eps))
    nr = len(rows)
    fig = plt.figure(figsize=(S.W2, 0.3 + nr * 0.72 + 0.42))
    gs = fig.add_gridspec(nr, 7, width_ratios=[1.0] + [1] * 6, hspace=0.12, wspace=0.04, left=0, right=1,
                          top=1 - 0.3 / fig.get_figheight(), bottom=0.42 / fig.get_figheight())
    for i, (k, lab, eps) in enumerate(rows):
        axl = fig.add_subplot(gs[i, 0])
        axl.axis("off")
        axl.text(0.98, 0.5, short(lab), ha="right", va="center", fontsize=S.TINY + 0.5, linespacing=1.1)
        for j, (t, f, c) in enumerate(eps):
            for m, ep in enumerate((f, c)):
                ax = fig.add_subplot(gs[i, 1 + 2 * j + m])
                if ep is None:
                    ax.axis("off")
                    continue
                a, b = ep.phases()
                show(ax, ep, b if b is not None else ep.T - 1, rings=rings_for(k, ep), step_label=False)
                S.stamp(ax, "OK" if ep.success else "FAIL", "upper left")
                ax.text(0.97, 0.04, "trial %d" % t, transform=ax.transAxes, ha="right", va="bottom", fontsize=S.TINY,
                        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8))
                if i == 0:
                    ax.set_title("failing arm" if m == 0 else "comparator", fontsize=S.TINY, pad=2)
    footer(fig, "A16  Further development-state episodes per failure (release tiles): the next qualifying trials of the same fixed rule (selection.json), not the best-looking ones.")
    save_exact(fig, "figA16")
    return True


# --------------------------------------------------------------------------- smoke check
def smoke(d):
    import matplotlib.pyplot as plt
    for tf in sorted(glob.glob(os.path.join(d, "**", "*_track.npz"), recursive=True)):
        stem = tf[:-len("_track.npz")]
        try:
            J = json.load(open(stem + ".json", encoding="utf-8"))
        except Exception as ex:
            print("no json", stem, ex)
            continue
        n = sum(len(t["episodes"]) for t in J["tasks"].values())
        sz = os.path.getsize(tf) / 1e6
        for k in range(n):
            ep = Ep(stem, idx=k)
            print(os.path.basename(stem), "ep", k, "trial", ep.trial, "success", ep.success, "rows", ep.T, "frames", len(ep.fr_step),
                  "npz MB %.1f" % sz, "cam", ep.cam is not None, flush=True)
            rows = ep.key_rows()
            fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
            for ax, r in zip(axes.flat, rows + [rows[-1]]):
                i, rr = ep.nearest(r)
                img = ep.frame(i)
                ax.imshow(img)                                                              # full 512 frame, no crop
                ax.plot(*ep.pix[rr][::-1], "o", ms=12, mfc="none", mec="cyan", mew=2)       # the EE at this frame's row
                a, b = ep.phases()
                ax.plot(ep.pix[:rr + 1, 1], ep.pix[:rr + 1, 0], "-", color="yellow", lw=1)
                ax.set_title("row %d (frame row %d) held %s" % (r, rr, bool(ep.held[min(rr, ep.T - 1)])), fontsize=8)
                ax.axis("off")
            fig.savefig(stem + "_ep%d_check.png" % k, dpi=70, bbox_inches="tight")
            g_i, g_r = ep.nearest(ep.phases()[0] if ep.phases()[0] is not None else 0)
            Image.fromarray(ep.frame(g_i)).save(stem + "_ep%d_grasp512.png" % k)
            plt.close(fig)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "smoke":
        smoke(a[1])
    else:
        todo = ["A2", "A4", "A5", "A6", "A7", "A8", "A9", "A14", "A16", "A3"] if a[0] == "all" else a
        main_figs([k for k in todo if k != "A16"])
        if "A16" in todo:
            fig_A16()

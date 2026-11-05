"""Labels at the object's centre (docs/LOCATOR.md 14; m3_prereg pp_loc4).  openpi venv, GPU.

    train S    -> DATA/ckpt/pp_loc4_CC_s<S>.pt
    read       -> results/flawed/pp_loc4.json

pp_loc3's arm CP, with each label replaced by its demo's cutout centre.  LOC_SMOKE=1: 40 steps, _smoke files, drawn patches in the job directory.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_loc as L                                                          # noqa: E402
import pp_loc2 as L2                                                        # noqa: E402
import pp_loc3 as L3                                                        # noqa: E402

FC, H, SFX = L.FC, L.H, L.SFX
_to_pix = L.to_pix


def ck4(s):
    return os.path.join(FC.DATA, "ckpt", f"pp_loc4_CC_s{s}{SFX}")


def component(mask, dist):
    """The largest connected region of mask within GUARD px of the label, holes filled; None if there is none."""
    from scipy import ndimage as ni
    lab, n = ni.label(mask)
    best, sz = None, 0
    for c in range(1, n + 1):
        p = lab == c
        if dist[p].min() <= L2.GUARD and p.sum() > sz:
            best, sz = p, int(p.sum())
    return None if best is None else ni.binary_fill_holes(best)


def centre_pix(F):
    """Labels [frames, heads, 2]: each demo's cutout centre, from the hand's labels as the search region only."""
    from scipy import ndimage as ni
    base = _to_pix(F, "C")
    Z = np.load(os.path.join(FC.DATA, f"loc_cut{SFX}.npz"))
    first, last, task, demo = Z["first"], Z["last"], Z["task"], Z["demo"]
    rr, cc = np.mgrid[0:H, 0:H]
    out = np.full(base.shape, np.nan)
    per, found, missed = {h: [] for h in range(len(L.HEADS))}, 0, 0
    for hi in range(4):
        for k in L.HEADS[hi][2]:
            for i in np.flatnonzero(task == list(FC.KNOWN).index(k)):
                f = (F["task"] == task[i]) & (F["demo"] == demo[i]) & F["valid"][:, hi]
                if not f.any():
                    continue
                lb = np.nanmean(base[f, hi], 0)
                dist = np.sqrt((rr - lb[0]) ** 2 + (cc - lb[1]) ** 2)
                img = first[i].astype(np.float32)
                if hi < 3:
                    m = ni.binary_opening((np.abs(img - last[i]).max(-1) > L2.CUT_DIFF) & (dist <= L2.CUT_ROI))
                    lo, hi_a = 100, 9000
                else:
                    col = np.median(img[(dist >= L3.RING[0]) & (dist <= L3.RING[1])], 0)
                    m = ni.binary_closing((np.abs(img - col).max(-1) > L3.FG_DIFF) & (dist <= L3.RING[0]), iterations=4)
                    m = ni.binary_opening(ni.binary_fill_holes(m), iterations=6)
                    lo, hi_a = 300, 9000
                p = component(m, dist)
                if p is None or not lo <= p.sum() <= hi_a:
                    missed += 1
                    continue
                c = np.array([rr[p].mean(), cc[p].mean()])
                out[f, hi] = c
                per[hi].append(c)
                found += 1
    for hi in range(4):                                                     # a demo with no cutout takes its head's mean
        v = F["valid"][:, hi] & np.isnan(out[:, hi, 0])
        out[v, hi] = np.mean(per[hi], 0)
    out[F["valid"][:, 4], 4] = np.mean(per[2], 0)                           # bowl as target: the bowl's mean centre
    print(f"[loc4] centres: {found} demos found, {missed} took the mean", flush=True)
    return out


def heights(F):
    """Head heights from the hand: each object's mean hand height at the close; the bowl target as the bowl; the plate at the lowest of the three."""
    tr = F["demo"] < L.N_TRAIN_DEMO
    z = [float(np.nanmean(F["lab_C"][tr & F["valid"][:, h], h, 2])) for h in range(3)]
    return np.array(z + [min(z), z[2]])


def read():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = np.load(L.FRAMES)
    F = {k: F[k] for k in F.files}
    K, R = F["K"], F["R"]
    Dp = np.load(os.path.join(FC.DATA, "demos.npz"))["pos"]
    tr = F["demo"] < L.N_TRAIN_DEMO
    pix, Zs = centre_pix(F), heights(F)
    rows = {n: dict(familiar=dict(e=[], raw=[], med=[], T=[], rawmed=[]), new=dict(e=[], raw=[], med=[], T=[], rawmed=[], n_bad=[])) for n in L.NAMES}
    bs = {}
    for s in (0, 1, 2):
        m = L.net().to(dev)
        m.load_state_dict(torch.load(ck4(s) + ".pt", map_location=dev))
        pr = L.locate(m.eval(), F["d_img"], dev)
        for hi, nm in enumerate(L.NAMES):
            ei = FC.ENT.index(L.HEADS[hi][0])
            v = tr & F["valid"][:, hi]
            z = float(Zs[hi])
            wl = L.lift(K, R, pix[v, hi], np.full(v.sum(), z))
            b = (wl[:, :2] - Dp[F["row"][v], ei][:, :2]).mean(0)
            bs[nm] = b.tolist()
            tb = wl[:, :2].mean(0)
            for li, lay in enumerate(("familiar", "new")):
                mk = F["d_layout"] == li
                o = F["d_pos"][mk, ei]
                w = L.lift(K, R, pr[mk, hi], np.full(mk.sum(), z))
                w2 = w.copy()
                w2[:, :2] -= b
                d = rows[nm][lay]
                d["e"].append(L.stat(w2, o)); d["raw"].append(L.stat(w, o)); d["med"].append(L.stat(w2, o, 50)); d["rawmed"].append(L.stat(w, o, 50))
                d["T"].append(L.stat(np.tile(tb - b, (mk.sum(), 1)), o))
                if lay == "new":
                    d["n_bad"].append(int((np.linalg.norm(w2[:, :2] - o[:, :2], axis=-1) > L2.BAD).sum()))
    out = {nm: dict(b=bs[nm], z=float(Zs[i]), **{y: dict(e=float(np.median(d["e"])), e_seeds=d["e"], e_raw=float(np.median(d["raw"])), raw_seeds=d["raw"], e_median=float(np.median(d["med"])), raw_median=float(np.median(d["rawmed"])), T=float(np.median(d["T"])), **({"n_bad": d["n_bad"]} if y == "new" else {})) for y, d in rows[nm].items()}) for i, nm in enumerate(L.NAMES)}
    within = all(out[n][y]["e"] <= L.BUDGET for n in L.NAMES for y in ("familiar", "new"))
    beats = all(out[n]["new"]["e"] <= out[n]["new"]["T"] / 2 for n in L.NAMES)
    v = "WITHIN BUDGET" if within else ("BEATS A TABLE, OVER BUDGET" if beats else "NO BETTER THAN A TABLE")
    print(f"[loc4] CC: {v}", json.dumps(out), flush=True)
    with open(os.path.join(L3.RES_OUT, "pp_loc4.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(label=v, heads=out), fh, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        L.to_pix = lambda F, arm: centre_pix(F)
        L2.make_bank = L3.bank3
        L2.ck = lambda a, s: ck4(s)
        L2.train("CP", int(sys.argv[2]))
    else:
        read()

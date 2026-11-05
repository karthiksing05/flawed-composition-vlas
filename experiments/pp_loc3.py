"""A plate cutout from figure and ground (docs/LOCATOR.md 12; m3_prereg pp_loc3).  openpi venv, GPU.

    train ARM S    ARM CP or AP (written as CP3 / AP3) -> DATA/ckpt/pp_loc3_<ARM>_s<S>.pt
    read           -> results/flawed/pp_loc3.json

Everything else is pp_loc2's arm CP or AP.  LOC_SMOKE=1: 40 steps, _smoke files, and the drawn patches go to the job directory.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_loc as L                                                          # noqa: E402
import pp_loc2 as L2                                                        # noqa: E402

FC, H, SFX = L.FC, L.H, L.SFX
FG_DIFF, RING = 20.0, (60.0, 70.0)
_bank = L2.make_bank
RES_OUT = L.OUT
if L.SMOKE:
    L.OUT = "/coc/flash10/ksingara3/.claude_tmp/pp"


def ck3(arm, s):
    return os.path.join(FC.DATA, "ckpt", f"pp_loc3_{arm}3_s{s}{SFX}")


def bank3(F, pix, pool, arm, dev):
    """pp_loc2's bank, with the plate's cutout, anchor and offset from figure and ground."""
    import torch
    from scipy import ndimage as ni
    cuts, off, BG, ALPHA = _bank(F, pix, pool, arm, dev)
    Z = np.load(os.path.join(FC.DATA, f"loc_cut{SFX}.npz"))
    first, task, demo = Z["first"], Z["task"], Z["demo"]
    rr, cc = np.mgrid[0:H, 0:H]
    cs, an, ar = [], [], []
    for i in np.flatnonzero(task == list(FC.KNOWN).index("K1")):
        f = (F["task"] == task[i]) & (F["demo"] == demo[i]) & F["valid"][:, 3]
        if not f.any():
            continue
        lb = np.nanmean(pix[f, 3], 0)
        dist = np.sqrt((rr - lb[0]) ** 2 + (cc - lb[1]) ** 2)
        img = first[i].astype(np.float32)
        col = np.median(img[(dist >= RING[0]) & (dist <= RING[1])], 0)
        fg = ni.binary_closing((np.abs(img - col).max(-1) > FG_DIFF) & (dist <= RING[0]), iterations=4)
        fg = ni.binary_opening(ni.binary_fill_holes(fg), iterations=6)
        lab, n = ni.label(fg)
        best, sz = None, 0
        for c in range(1, n + 1):
            p = lab == c
            if dist[p].min() <= L2.GUARD and p.sum() > sz:
                best, sz = p, int(p.sum())
        if best is None:
            continue
        best = ni.binary_fill_holes(best)
        if not 300 <= best.sum() <= 9000:
            continue
        cen = np.round([rr[best].mean(), cc[best].mean()]).astype(int)
        al = ni.gaussian_filter(best.astype(np.float32), 1.0)
        rgba = np.concatenate([img / 255.0, al[..., None]], -1)
        cs.append(np.pad(rgba, ((64, 64), (64, 64), (0, 0)))[cen[0]:cen[0] + 128, cen[1]:cen[1] + 128].transpose(2, 0, 1))
        an.append(cen); ar.append(int(best.sum()))
    cuts[3] = torch.as_tensor(np.stack(cs), device=dev, dtype=torch.float32)
    off[3] = np.nanmean(pix[pool[3], 3], 0) - np.mean(an, 0)
    print("[loc3] plate cutouts", len(cs), "median area", float(np.median(ar)), "offset", np.round(off[3], 1).tolist(), flush=True)
    return cuts, off, BG, ALPHA


def read():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = np.load(L.FRAMES)
    F = {k: F[k] for k in F.files}
    K, R = F["K"], F["R"]
    Dp = np.load(os.path.join(FC.DATA, "demos.npz"))["pos"]
    tr = F["demo"] < L.N_TRAIN_DEMO
    res = {}
    for arm in ("C", "A"):
        lab = F[f"lab_{arm}"]
        pix = L.to_pix(F, arm)
        rows = {n: dict(familiar=dict(e=[], raw=[], med=[], T=[]), new=dict(e=[], raw=[], med=[], T=[], n_bad=[])) for n in L.NAMES}
        for s in (0, 1, 2):
            m = L.net().to(dev)
            m.load_state_dict(torch.load(ck3(arm + "P", s) + ".pt", map_location=dev))
            pr = L.locate(m.eval(), F["d_img"], dev)
            for hi, nm in enumerate(L.NAMES):
                ei = FC.ENT.index(L.HEADS[hi][0])
                v = tr & F["valid"][:, hi]
                z = float(lab[v, hi, 2].mean())
                wl = L.lift(K, R, pix[v, hi], np.full(v.sum(), z))
                b = (wl[:, :2] - Dp[F["row"][v], ei][:, :2]).mean(0)
                tb = wl[:, :2].mean(0)
                for li, lay in enumerate(("familiar", "new")):
                    mk = F["d_layout"] == li
                    o = F["d_pos"][mk, ei]
                    w = L.lift(K, R, pr[mk, hi], np.full(mk.sum(), z))
                    w2 = w.copy()
                    w2[:, :2] -= b
                    d = rows[nm][lay]
                    d["e"].append(L.stat(w2, o)); d["raw"].append(L.stat(w, o)); d["med"].append(L.stat(w2, o, 50))
                    d["T"].append(L.stat(np.tile(tb - b, (mk.sum(), 1)), o))
                    if lay == "new":
                        d["n_bad"].append(int((np.linalg.norm(w2[:, :2] - o[:, :2], axis=-1) > L2.BAD).sum()))
        out = {nm: {y: dict(e=float(np.median(d["e"])), e_seeds=d["e"], e_raw=float(np.median(d["raw"])), e_median=float(np.median(d["med"])), T=float(np.median(d["T"])), **({"n_bad": d["n_bad"]} if y == "new" else {})) for y, d in r.items()} for nm, r in rows.items()}
        within = all(out[n][y]["e"] <= L.BUDGET for n in L.NAMES for y in ("familiar", "new"))
        beats = all(out[n]["new"]["e"] <= out[n]["new"]["T"] / 2 for n in L.NAMES)
        v = "WITHIN BUDGET" if within else ("BEATS A TABLE, OVER BUDGET" if beats else "NO BETTER THAN A TABLE")
        res[arm + "P3"] = dict(label=v, heads=out)
        print(f"[loc3] {arm}P3: {v}", json.dumps(out), flush=True)
    with open(os.path.join(RES_OUT, "pp_loc3.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        arm, seed = sys.argv[2], int(sys.argv[3])
        assert arm in ("CP", "AP")
        L2.make_bank = bank3
        L2.ck = lambda a, s: ck3(a, s)
        L2.train(arm, seed)
    else:
        read()

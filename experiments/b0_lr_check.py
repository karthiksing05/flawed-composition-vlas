"""PLAN_BINDER B0 follow-up: is LIBERO's left / right / front / back in the image (camera view) or the robot frame?  CPU, chunks env.

Reads only S1's rendered frames (results/bind/s1/frames, policy orientation: agentview[::-1, ::-1] as the client sends it) and
S1's labels.  No demo, no sealed file, no model.
  1. world -> image map: per body with segmentation coverage, the token centroid (row, col) vs the world xy (robot base at -x,
     facing +x; +y = the robot's left).  Sign of corr(col, y) and corr(row, x).
  2. TRAINING-label check: for every training instance phrase with a direction word, is the target the extreme instance in the
     image direction the word names (left = smallest col, right = largest col, front = largest row (nearest the image bottom),
     back = smallest row)?  Rate per word, under the plain image reading and the mirrored one.
  3. Dev depth-1 units: target / competitor centroids and where each B0 VLM's peak landed.

    python experiments/b0_lr_check.py -> results/bind/b0_lr_check.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
S1 = os.path.join(ROOT, "results", "bind", "s1")
DIRW = ("left", "right", "front", "back", "middle")


def cent(c):
    c = np.asarray(c, np.float64).reshape(16, 16)
    s = c.sum()
    if s <= 0:
        return None
    r, k = np.mgrid[0:16, 0:16]
    return float((r * c).sum() / s), float((k * c).sum() / s)


def main():
    import s1_read as SR
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    out = {}
    # 1. world -> image
    Y, X, C, R = [], [], [], []
    for f in sorted(glob.glob(os.path.join(S1, "frames", "*", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        pos = json.loads(str(z["pos"]))
        names = [str(b) for b in z["body_names"]]
        for n, p in pos.items():
            if n not in names or "table" in n or "floor" in n:
                continue
            ce = cent(z["body_cov"][names.index(n)])
            if ce is None:
                continue
            X.append(p[0]); Y.append(p[1]); R.append(ce[0]); C.append(ce[1])
    X, Y, R, C = map(np.asarray, (X, Y, R, C))
    out["world_to_image"] = dict(n=int(len(X)), corr_col_y=float(np.corrcoef(C, Y)[0, 1]), corr_col_x=float(np.corrcoef(C, X)[0, 1]),
                                 corr_row_x=float(np.corrcoef(R, X)[0, 1]), corr_row_y=float(np.corrcoef(R, Y)[0, 1]),
                                 slope_col_per_m_y=float(np.polyfit(Y, C, 1)[0]), slope_row_per_m_x=float(np.polyfit(X, R, 1)[0]))
    print("[lr] world->image", out["world_to_image"], flush=True)

    # 2. training labels with direction words, target among same-type instances
    def judge(e):
        z = np.load(os.path.join(ROOT, e["frame"]), allow_pickle=True)
        names = [str(b) for b in z["body_names"]]
        pos = json.loads(str(z["pos"]))
        tg = e["targets"]
        if len(tg) != 1 or tg[0] not in names:
            return None
        t = tg[0]
        sib = [b for b in names if b != t and SR.itype(b) == SR.itype(t)]
        if not sib:
            return None
        cs = {b: cent(z["body_cov"][names.index(b)]) for b in [t] + sib}
        if any(v is None for v in cs.values()):
            return None
        words = [w for w in DIRW if re.search(rf"\b{w}\b", e["phrase"])]
        res = {}
        for w in words:
            if w == "middle":
                continue
            ax = 1 if w in ("left", "right") else 0
            vals = {b: cs[b][ax] for b in cs}
            lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
            img = {"left": lo, "right": hi, "back": lo, "front": hi}[w]
            mir = {"left": hi, "right": lo, "back": hi, "front": lo}[w]
            wv = {b: pos[b] for b in cs if b in pos}
            res[w] = dict(image=float(img == t), mirrored=float(mir == t),
                          world=({b: [round(v, 3) for v in wv[b][:2]] for b in wv}), cent={b: [round(v, 2) for v in cs[b]] for b in cs})
        return res
    tr = {}
    for e in lab["train"]:
        r = judge(e)
        if not r:
            continue
        for w, v in r.items():
            d = tr.setdefault(f"{w}|{e['phrase']}|{e['scene']}", dict(image=[], mirrored=[], ex=v))
            d["image"].append(v["image"]); d["mirrored"].append(v["mirrored"])
    out["train_direction_phrases"] = {k: dict(n=len(v["image"]), image=float(np.mean(v["image"])), mirrored=float(np.mean(v["mirrored"])),
                                              example=v["ex"]) for k, v in sorted(tr.items())}
    for k, v in out["train_direction_phrases"].items():
        print(f"[lr] train {k}: n {v['n']} image-reading {v['image']:.2f} mirrored {v['mirrored']:.2f} ex {v['example']}", flush=True)
    dv = {}
    for e in lab["dev"]:
        r = judge(e)
        if r:
            for w, v in r.items():
                d = dv.setdefault(f"{w}|{e['phrase']}|{e['tid']}", dict(image=[], mirrored=[], ex=v))
                d["image"].append(v["image"]); d["mirrored"].append(v["mirrored"])
    out["dev_direction_phrases"] = {k: dict(n=len(v["image"]), image=float(np.mean(v["image"])), mirrored=float(np.mean(v["mirrored"])),
                                            example=v["ex"]) for k, v in sorted(dv.items())}
    for k, v in out["dev_direction_phrases"].items():
        print(f"[lr] dev {k}: n {v['n']} image-reading {v['image']:.2f} mirrored {v['mirrored']:.2f} ex {v['example']}", flush=True)

    # 3. dev depth-1 units: where each VLM's peak landed
    preds = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "bind", "b0", "*__*.npz"))):
        nm = os.path.basename(f)[:-4]
        if not any(a in nm for a in ("qwen", "molmo")):
            continue
        pz = np.load(f, allow_pickle=True)
        preds[nm] = {str(k): int(p) for k, p in zip(pz["keys"], pz["peak"])}
    d1 = []
    for e in lab["dev"]:
        if e["depth"] != 1:
            continue
        z = np.load(os.path.join(ROOT, e["frame"]), allow_pickle=True)
        names = [str(b) for b in z["body_names"]]
        t = e["targets"][0]
        sib = [b for b in names if b != t and SR.itype(b) == SR.itype(t)]
        u = dict(tid=e["tid"], role=e["role"], phrase=e["phrase"], seen=e["seen"], target=t,
                 cent={b: [round(v, 2) for v in (cent(z["body_cov"][names.index(b)]) or (-1, -1))] for b in [t] + sib}, peaks={})
        key = f"{e['role']}|{e['frame']}"
        for nm, pr in preds.items():
            pk = pr.get(key, -1)
            if pk < 0:
                u["peaks"][nm] = None
                continue
            on = [b for b in names if float(z["body_cov"][names.index(b)][pk]) > 0]
            u["peaks"][nm] = dict(rc=[pk // 16, pk % 16], on=on)
        d1.append(u)
        print(f"[lr] d1 {u['tid']} {u['role']} {u['phrase']!r} seen {u['seen']} target {t} cent {u['cent']} | "
              + " ".join(f"{nm}:{(v or {}).get('rc')}{(v or {}).get('on')}" for nm, v in u["peaks"].items()), flush=True)
    out["dev_depth1"] = d1
    json.dump(out, open(os.path.join(ROOT, "results", "bind", "b0_lr_check.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

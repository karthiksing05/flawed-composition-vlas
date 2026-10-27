"""PLAN_BINDER B0 SUPPLEMENTARY rescore (not pre-registered; does NOT replace results/bind/b0.json): depth-1 units under the
MIRRORED left/right reading.  CPU, chunks env.

Finding behind it (experiments/b0_lr_check.py, results/bind/b0_lr_check.json): in the policy's image (agentview[::-1, ::-1]),
world +y (the ROBOT's left) lies on the IMAGE LEFT (corr(col, y) = -0.99), and LIBERO's "left" instance is world -y in every
training scene, i.e. on the IMAGE RIGHT (16 / 16 "left bowl", 32 / 32 "left plate", ...).  "front / back" agree with the image
(front = nearer the image bottom).  So LIBERO's left / right is the true camera view's, and the policy image is that view
mirrored left-right.  A VLM that reads "left" in the image it is shown picks the other instance.

Mirrored reading: for a unit whose phrase says left / right and whose target is one of several same-type instances, the target
becomes the instance at that image extreme (left = smallest centroid column, right = largest).  Every other unit, and every
other word (front / back / middle), is unchanged.  Same hit metric (s1_read.hit, the same competitors), same task-cluster
bootstrap (d4_read.rate), the committed B0 predictions (results/bind/b0/*.npz), no re-prediction.
Descriptive extra: the caddy's left / right compartments under a left <-> right name swap (the caddy is in no training scene).

    python experiments/b0_mirror.py -> results/bind/b0_mirror.json
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


def ccol(c):
    c = np.asarray(c, np.float64).reshape(16, 16)
    return float((np.arange(16)[None, :] * c).sum() / max(c.sum(), 1e-9))


def main():
    import d4_read as DR
    import s1_read as SR
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    b0 = json.load(open(os.path.join(ROOT, "results", "bind", "b0.json"), encoding="utf-8"))
    preds = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "bind", "b0", "*__*.npz"))):
        pz = np.load(f, allow_pickle=True)
        preds[os.path.basename(f)[:-4]] = {str(k): int(p) for k, p in zip(pz["keys"], pz["peak"])}
    ck = sorted(preds)
    units, caddy = [], []
    for ix, e in enumerate(lab["dev"]):
        z = np.load(os.path.join(ROOT, e["frame"]), allow_pickle=True)
        names, cov = SR.masks(z)
        bodies = [str(b) for b in z["body_names"]]
        tg = e["targets"][0]
        lr = re.search(r"\b(left|right)\b", e["phrase"])
        key = f"{e['role']}|{e['frame']}"
        bu = b0["units"][ix]
        assert (bu["tid"], bu["role"], bu["phrase"]) == (e["tid"], e["role"], e["phrase"])
        if e["depth"] == 1:
            mt = tg
            if lr and tg in bodies:
                sib = [tg] + [b for b in bodies if b != tg and SR.itype(b) == SR.itype(tg)]
                cols = {b: ccol(cov[names.index(b)]) for b in sib}
                mt = (min if lr.group(1) == "left" else max)(cols, key=cols.get)
            comp, _ = SR.competitors(mt, names, bodies)
            t, cc = cov[names.index(mt)], [cov[names.index(c)] for c in comp]
            u = dict(tid=e["tid"], role=e["role"], phrase=e["phrase"], seen=e["seen"], target=tg, mirrored_target=mt,
                     changed=mt != tg)
            for c in ck:
                u[c] = SR.hit(preds[c].get(key, -1), t, cc)
                u[c + "__prereg"] = bu[c]
            for m in ("b1b", "s1b", "s1c", "b1a", "robopoint"):
                u[m + "__prereg"] = bu[m]
            units.append(u)
        elif lr and "compartment" in e["phrase"] and "caddy" in tg:
            mt = tg.replace("_left_", "_TMP_").replace("_right_", "_left_").replace("_TMP_", "_right_")
            comp, _ = SR.competitors(mt, names, bodies)
            t, cc = cov[names.index(mt)], [cov[names.index(c)] for c in comp]
            u = dict(tid=e["tid"], phrase=e["phrase"], target=tg, mirrored_target=mt,
                     col_target=ccol(cov[names.index(tg)]), col_mirrored=ccol(cov[names.index(mt)]))
            for c in ck:
                u[c] = SR.hit(preds[c].get(key, -1), t, cc)
                u[c + "__prereg"] = bu[c]
            caddy.append(u)

    def per(us, k):
        d = {}
        for u in us:
            d.setdefault(u["tid"], []).append(float(u[k]))
        return DR.rate(d)
    cells = {}
    for nm, us in (("d1_all", units), ("d1_seen", [u for u in units if u["seen"]]), ("d1_heldout", [u for u in units if not u["seen"]]),
                   ("d1_leftright_units", [u for u in units if u["changed"]]), ("caddy_leftright", caddy)):
        if not us:
            continue
        cells[nm] = dict(n=len(us), tasks=sorted({u["tid"] for u in us}),
                         mirrored={c: per(us, c) for c in ck}, prereg={c: per(us, c + "__prereg") for c in ck})
        print(f"[b0 mirror] {nm}: n {len(us)} tasks {cells[nm]['tasks']}", flush=True)
        for c in ck:
            m, p = cells[nm]["mirrored"][c], cells[nm]["prereg"][c]
            print(f"[b0 mirror]   {c:28s} mirrored {m['mean']:.3f} [{m['lo']:.2f}, {m['hi']:.2f}]   pre-registered {p['mean']:.3f}", flush=True)
    for u in caddy:
        print(f"[b0 mirror] caddy {u['tid']} {u['phrase']!r}: target col {u['col_target']:.2f}, swapped col {u['col_mirrored']:.2f}", flush=True)
    bar_i = {c: dict(mirrored=cells["d1_heldout"]["mirrored"][c]["mean"], prereg=cells["d1_heldout"]["prereg"][c]["mean"],
                     mirrored_pass=cells["d1_heldout"]["mirrored"][c]["mean"] >= 0.70) for c in ck}
    json.dump(dict(note="SUPPLEMENTARY, not pre-registered; the pre-registered B0 numbers are results/bind/b0.json",
                   rule=__doc__, bar_i_mirrored=bar_i, cells=cells, units=units, caddy=caddy),
              open(os.path.join(ROOT, "results", "bind", "b0_mirror.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

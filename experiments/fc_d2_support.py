"""FLAWED_COMPOSITION D2 controls, from the task definitions alone (no model, no demo, no outcome): task novelty of every D2 task
relative to the archive VLAs' training set (the 40 LIBERO-Spatial / Object / Goal / 10 tasks every pi0.5 / GR00T / X-VLA LIBERO
checkpoint was fine-tuned on).  CPU.

  u      the split's compound-divergence proxy: the fraction of the task's compounds absent from training (c_split.unseen_frac)
  score  the D0 predicted-debt score of fc_d0_support.py with the 40 tasks as the training set: the components 1 - seen(v,o),
         1 - seen(o,t), 1 - seen(o,s), 1 - seen(t,s), -log1p(P_o), -log1p(P_t), -log1p(L_o), -log1p(L_t), averaged over steps, ranked
         within the block (L90 or LI), averaged over the available components
-> results/flawed/d2_support.json

    python experiments/fc_d2_support.py
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
OUT = os.path.join(ROOT, "results", "flawed")
ALL = ("u_vo", "u_ot", "u_os", "u_ts", "nP_o", "nP_t", "nL_o", "nL_t")


def main():
    import c_split as CS
    from fc_d0_support import rankdata
    from lifelong_ood.policy import factors as F
    train = F.load(F.TRAIN_SUITES)
    seen, P_o, P_t, L_o, L_t = set(), *(collections.defaultdict(set) for _ in range(4))
    for t in train:
        for v, o, g in CS.steps(t):
            P_o[o].add(g)
            L_o[o].add(t.scene)
            if g:
                P_t[g].add(o)
                L_t[g].add(t.scene)
        seen |= set(CS.compounds(t))
    fr = json.load(open(os.path.join(OUT, "d2_frames.json"), encoding="utf-8"))
    ood = {f"{t.suite.split('_')[1]}:{t.index}": t for t in F.load(F.OOD_SUITES)}
    l90 = {t.tid: t for t in F.load(("libero_90",))}
    tasks = {"L90": {t: l90[t] for t in fr["l90"]}, "LI": {k: ood[k] for k in fr["li"]}}
    out = {}
    for blk, ts in tasks.items():
        rows = {}
        for key, t in ts.items():
            s = "s:" + t.scene
            comp = collections.defaultdict(list)
            for v, o, g in CS.steps(t):
                o_, v_ = "o:" + o, "v:" + v
                comp["u_vo"].append(1.0 - ((v_, o_) in seen))
                comp["u_os"].append(1.0 - ((o_, s) in seen))
                comp["nP_o"].append(-math.log1p(len(P_o[o])))
                comp["nL_o"].append(-math.log1p(len(L_o[o])))
                if g:
                    t_ = "t:" + g
                    comp["u_ot"].append(1.0 - ((o_, t_) in seen))
                    comp["u_ts"].append(1.0 - ((t_, s) in seen))
                    comp["nP_t"].append(-math.log1p(len(P_t[g])))
                    comp["nL_t"].append(-math.log1p(len(L_t[g])))
            rows[key] = dict(language=t.language, scene=t.scene, u=float(CS.unseen_frac(t, seen)),
                             components={k: float(np.mean(v)) for k, v in comp.items()})
        keys = sorted(rows)
        ranks = {}
        for k in ALL:
            have = [x for x in keys if k in rows[x]["components"]]
            ranks[k] = dict(zip(have, rankdata([rows[x]["components"][k] for x in have]).tolist()))
        for x in keys:
            v = [ranks[k][x] for k in ALL if x in ranks[k]]
            rows[x]["score"] = float(np.mean(v)) if v else None
        out[blk] = rows
    with open(os.path.join(OUT, "d2_support.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for blk, rows in out.items():
        print(blk, len(rows), "u mean", np.mean([r["u"] for r in rows.values()]), "score range",
              min(r["score"] for r in rows.values()), max(r["score"] for r in rows.values()))


if __name__ == "__main__":
    main()

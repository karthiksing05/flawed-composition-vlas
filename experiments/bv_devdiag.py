"""PLAN_BINDVLA 11, offline DEV slot-hit diagnostic (pre-registered with the phase-2 dev read, m3_prereg tag pbv2): each BindVLA
checkpoint's final-level slot mask on frame 0 of the dev tasks' own demos (experiments/bv_devrefs.py: contact-event referents; the
demos are scored only, never trained on or executed), hit = argmax within Chebyshev 1 of the referent token (chance 9 / 256).
Steps pair with transports as in b1_bind.training_occurrences.  Grouped by CD bin, by role and by path depth.  vanilla_frozen
has no slots (no role-indexed attention), so only BindVLA checkpoints are scored.

    python experiments/bv_devdiag.py --ckpt results/bindvla/main2/ckpt/step_060000.pt [--ckpt ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
CACHE = "/coc/flash10/ksingara3/robot/data/bindvla_cache"


def boot(h, n=2000, seed=0, groups=None):
    """task-cluster bootstrap of the mean hit (groups = task ids)."""
    h = np.asarray(h, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        bt = [rng.choice(h, len(h)).mean() for _ in range(n)]
    else:
        g = np.asarray(groups)
        u = np.unique(g)
        idx = {x: np.flatnonzero(g == x) for x in u}
        bt = [np.concatenate([h[idx[x]] for x in rng.choice(u, len(u))]).mean() for _ in range(n)]
    return [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    a = ap.parse_args()
    from lifelong_ood import l90c
    from lifelong_ood.policy.backbone_bindvla import BindVLABackbone
    import b1_bind as B
    import s1_labels as SL
    from lifelong_ood.policy import primitives as P
    z = np.load(os.path.join(CACHE, "devrefs.npz"), allow_pickle=True)
    evs = json.loads(str(z["events"]))
    meta = json.load(open(os.path.join(ROOT, "results", "bindvla", "devrefs.json"), encoding="utf-8"))["per_task"]
    for t in set(z["tid"].tolist()):
        l90c.check_task(t)
    res = {}
    for ck in a.ckpt:
        bk = BindVLABackbone(ck, device="cuda")
        cfg = bk.bvcfg
        rec = []
        for i in range(len(z["img"])):
            tid, b = str(z["tid"][i]), str(z["bin"][i])
            lang = meta[tid]["language"]
            steps = [SL.split_step(s) for s in P.steps_of(lang)]
            pairs = B.transports(evs[i])
            if len(pairs) < len(steps):
                continue
            o = bk.observation({"observation/image": z["img"][i], "observation/wrist_image": z["img"][i],
                                "observation/state": np.zeros(8, np.float32), "prompt": lang})
            with torch.no_grad():
                _, _, dg = bk.bv.prefix_cache(o.batch)
            M = dg["mask"][0].float().cpu().numpy()
            VV = dg["level_valid"][0].cpu().numpy()
            for s, (og, (gr, re_)) in enumerate(zip(steps[: cfg.n_steps], pairs)):
                for role, e in ((0, gr), (1, re_ if og[1] else None)):
                    if e is None or not VV[s, role].any():
                        continue
                    tr, tc = divmod(int(e["token"]), 16)
                    pr, pc = divmod(int(M[s, role].argmax()), 16)
                    rec.append(dict(tid=tid, bin=b, role=["object", "goal"][role], depth=int(VV[s, role].sum()),
                                    hit=float(max(abs(pr - tr), abs(pc - tc)) <= 1)))
        out = {"n": len(rec)}
        keys = [("all", lambda r: True)] + [(f"bin_{x}", (lambda x: lambda r: r["bin"] == x)(x)) for x in ("0.1", "0.3", "0.5", "part")] + \
               [(x, (lambda x: lambda r: r["role"] == x)(x)) for x in ("object", "goal")] + \
               [(f"depth{d}", (lambda d: lambda r: r["depth"] == d)(d)) for d in (1, 2, 3)]
        for k, f in keys:
            rs = [r for r in rec if f(r)]
            if rs:
                out[k] = dict(n=len(rs), tasks=len({r["tid"] for r in rs}), hit=float(np.mean([r["hit"] for r in rs])),
                              ci95_task=boot([r["hit"] for r in rs], groups=[r["tid"] for r in rs]), chance=9 / 256)
        out["per_task"] = {t: float(np.mean([r["hit"] for r in rec if r["tid"] == t])) for t in sorted({r["tid"] for r in rec})}
        res[ck] = out
        del bk
        torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(ROOT, "results", "bindvla", "devdiag.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_task"} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()

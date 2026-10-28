"""PLAN_BINDVLA hierarchy prediction (a), diagnostic only: per-level slot-mask hits of a BindVLA checkpoint against SIMULATOR masks
(scoring only) on the 61 rendered matched training demos (results/bind/b1/demo/<tid>.npz: frame 0 at 224 in the policy
orientation, body_cov [n_bodies, 256] sim masks, grasp / release referent tokens).  These demos come from TRAINING cells, so the
read is in-distribution (the held-out-episode diagnostic is in bv_train.py).

Per (demo, step, role) with a referent token: the referent body = the sim body covering that token with the smallest area
(tables, i.e. bodies covering > 40% of the image, are skipped).
  level-1 hit   argmax of the level-1 mask inside the referent body's mask (body_cov > 0.5)
  final hit     argmax of the final (deepest valid level) mask within Chebyshev 1 of the referent token
  mass          level-1 mask mass on the body; final mask mass in the 3 x 3 window; chance = area fractions
Grouped by the role's path depth (number of non-empty levels).

    python experiments/bv_levels.py --ckpt results/bindvla/main/ckpt/step_060000.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    from lifelong_ood.policy.backbone_bindvla import BindVLABackbone
    bk = BindVLABackbone(a.ckpt, device="cuda")
    cfg = bk.bvcfg
    import b1_bind as B
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "bind", "b1", "demo", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        prompt = str(z["prompt"])
        ev = json.loads(str(z["events"]))
        pairs = B.transports(ev)
        obs = {"observation/image": z["img"], "observation/wrist_image": z["wimg"] if "wimg" in z else z["img"],
               "observation/state": z["state8"] if "state8" in z else np.zeros(8, np.float32), "prompt": prompt}
        o = bk.observation(obs)
        _, _, dg = bk.bv.prefix_cache(o.batch)
        LV = dg["levels"][0].float().cpu().numpy()            # [S, 2, L, 256]
        FM = dg["mask"][0].float().cpu().numpy()
        LVV = dg["level_valid"][0].cpu().numpy()
        cov = np.asarray(z["body_cov"], np.float32) > 0.5
        area = cov.mean(1)
        for s, (g, r) in enumerate(pairs[: cfg.n_steps]):
            for role, e in ((0, g), (1, r)):
                if e is None or not LVV[s, role].any():
                    continue
                tok = int(e["token"])
                cand = [i for i in range(len(cov)) if cov[i, tok] and area[i] < 0.4]
                if not cand:
                    continue
                b = min(cand, key=lambda i: area[i])
                tr, tc = divmod(tok, 16)
                win = np.zeros(256, bool)
                for rr in range(max(0, tr - 1), min(16, tr + 2)):
                    for cc in range(max(0, tc - 1), min(16, tc + 2)):
                        win[rr * 16 + cc] = True
                m1, mf = LV[s, role, 0], FM[s, role]
                pf = int(mf.argmax())
                rows.append(dict(tid=str(z["tid"]), step=s, role=["object", "goal"][role], depth=int(LVV[s, role].sum()),
                                 hit1=bool(cov[b, int(m1.argmax())]), hitf=bool(win[pf]), hitf_body=bool(cov[b, pf]),
                                 mass1=float(m1[cov[b]].sum()), chance1=float(area[b]), massf=float(mf[win].sum()), chancef=float(win.mean())))
    out = {"n": len(rows), "ckpt": a.ckpt}
    for key, sel in [("all", lambda r: True)] + [(f"depth{d}", (lambda d: lambda r: r["depth"] == d)(d)) for d in (1, 2, 3)] + \
                    [(k, (lambda k: lambda r: r["role"] == k)(k)) for k in ("object", "goal")]:
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        m = lambda k: float(np.mean([r[k] for r in rs]))                                  # noqa: E731
        out[key] = dict(n=len(rs), level1_hit=m("hit1"), final_hit=m("hitf"), final_in_body=m("hitf_body"), level1_mass=m("mass1"),
                        level1_chance=m("chance1"), final_mass=m("massf"), final_chance=m("chancef"),
                        product_l1_x_final_given_l1=float(m("hit1") * (np.mean([r["hitf"] for r in rs if r["hit1"]]) if any(r["hit1"] for r in rs) else 0.0)))
    os.makedirs(os.path.join(ROOT, "results", "bindvla"), exist_ok=True)
    json.dump(dict(summary=out, rows=rows), open(os.path.join(ROOT, "results", "bindvla", "levels.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

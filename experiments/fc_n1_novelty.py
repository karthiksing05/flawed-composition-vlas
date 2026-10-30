"""FLAWED_COMPOSITION N1 (section 4.9): per-task INPUT novelty (nearest-training distances) and COMPOUND novelty, from the
rendered initial scenes (fc_n1_render.py) and the task definitions.  No outcome is read.  SmolVLA venv, one GPU (frozen SigLIP,
BindVLA binders).

Per task t (means over its 10 rendered initial states; positions relative to the robot base):
  (a) positions  referent = step 0's moved / manipulated instance (factors goal atom, first argument), goal = its target instance
                 (second argument; a region maps to the instance whose name is its longest prefix).  d_a = mean over the roles
                 present of min over training tasks of the Euclidean distance between the role's positions (a training task
                 without that role is skipped for it)
  (b) state      d_b = min over training tasks of [Chamfer distance between the two scenes' object + fixture position sets
                 + Euclidean distance between the end-effector positions + between the gripper openings]
  (c) visual     e = the frozen SigLIP grid of the agentview first frame (SmolVLM2's vision tower, 2 x 2 pooled to 16 x 16,
                 mean over tokens), mean over states; d_c = min over training tasks of the cosine distance
  (d) slot pose  BindVLA blocks only: the checkpoint's own step-0 object and goal final-mask moments (mu, sigma) at the first
                 frame with the task's instruction; d_d = min over training tasks of the Euclidean distance
  composite      the mean within-block rank of d_a, d_b, d_c (primary input novelty)
Compound novelty: the D0 score and u (BindVLA blocks: fc_d0_support / c_split against the 76 LIBERO-90-C training tasks; D2
blocks: fc_d2_support against the 40 LIBERO tasks).
Blocks and training sets: bv-main2, bv-vanilla (dev 29 vs the 76 training tasks), bv-c1 (dev 29 vs its 72); pi05-L90 (the 54
unsealed LIBERO-90 tasks), pi05-LI, groot-LI, xvla-LI (Li's 20) vs the 40 LIBERO tasks.
-> results/flawed/n1_novelty.json

    python experiments/fc_n1_novelty.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
OUT = os.path.join(ROOT, "results", "flawed")
RD = os.path.join(OUT, "n1_render")
C1_HELD = ("libero_90:37", "libero_object:9", "libero_spatial:8", "libero_spatial:9")
CK = {"main2": "results/bindvla/main2/ckpt/step_060000.pt", "c1": "results/bindvla/c1main/ckpt/step_020000.pt",
      "vanilla": "results/bindvla/vanilla/ckpt/step_060000.pt"}


def inst_of(arg, names):
    if arg in names:
        return arg
    c = [n for n in names if arg.startswith(n)]
    return max(c, key=len) if c else None


def chamfer(P, Q):
    d = np.linalg.norm(P[:, None] - Q[None], axis=-1)
    return float(0.5 * (d.min(1).mean() + d.min(0).mean()))


def main():
    from lifelong_ood.policy import factors as F
    from lifelong_ood.policy.backbone_bindvla import BindVLABackbone
    from lifelong_ood.policy import bindvla as BV
    import c_split as CS
    tl = json.load(open(os.path.join(OUT, "n1_tasks.json"), encoding="utf-8"))
    tasks = {t.tid: t for t in F.load(CS.SUITES)}
    for t in F.load(F.OOD_SUITES):
        tasks[f"{t.suite.split('_')[1]}:{t.index}"] = t
    fn = {**{t: os.path.join(RD, f"{t.split(':')[0]}_{t.split(':')[1]}.npz") for t in tl["tasks"]},
          **{f"{p}:{i}": os.path.join(RD, f"libero_{p}_ood_{i}.npz") for p in ("goal", "spatial") for i in range(10)}}
    Z = {k: dict(np.load(v, allow_pickle=True)) for k, v in fn.items()}
    feat = {}
    vis = BindVLABackbone(CK["vanilla"], device="cuda")
    for k, z in Z.items():
        names = [str(x) for x in z["names"]]
        rel = z["pos"] - z["base"][:, None]                                   # [S, n, 3]
        g = tasks[k].goal[0] if tasks[k].goal else ()
        ro = inst_of(g[1], names) if len(g) > 1 else None
        rg = inst_of(g[2], names) if len(g) > 2 else None
        with torch.no_grad():
            img = torch.from_numpy(z["img"]).permute(0, 3, 1, 2).float().cuda() / 255.0
            e = vis.bv.vision(img)[0].mean(1).mean(0).float().cpu().numpy()
        feat[k] = dict(obj=rel[:, names.index(ro)].mean(0) if ro else None, goal=rel[:, names.index(rg)].mean(0) if rg else None,
                       scene=rel.mean(0), eef=(z["state8"][:, :3] - z["base"]).mean(0), grip=z["state8"][:, 6:8].mean(0), emb=e,
                       ref_names=[ro, rg])
    del vis
    torch.cuda.empty_cache()
    pose = {}
    for m in ("main2", "c1"):
        bk = BindVLABackbone(CK[m], device="cuda")
        pose[m] = {}
        for k in list(tl["bv_train"]) + list(tl["dev"]):
            z = Z[k]
            obs = [{"observation/image": z["img"][j], "observation/wrist_image": z["wimg"][j], "observation/state": z["state8"][j],
                    "prompt": tasks[k].language} for j in range(len(z["img"]))]
            with torch.no_grad():
                _, _, dg = bk.bv.prefix_cache(bk.observations(obs).batch)
                mu, sig = BV.moments(dg["mask"][:, 0].float(), BV.grid_pos(dg["mask"].device))      # [S, 2 roles, 2]
            pose[m][k] = torch.cat([mu, sig], -1).reshape(len(obs), -1).mean(0).cpu().numpy()
        del bk
        torch.cuda.empty_cache()

    def dists(test, train, m=None):
        out = {}
        for k in test:
            f = feat[k]
            da = []
            for role in ("obj", "goal"):
                if f[role] is None:
                    continue
                c = [np.linalg.norm(f[role] - feat[t][role]) for t in train if feat[t][role] is not None and t != k]
                if c:
                    da.append(min(c))
            db = min(chamfer(f["scene"], feat[t]["scene"]) + np.linalg.norm(f["eef"] - feat[t]["eef"]) + np.linalg.norm(f["grip"] - feat[t]["grip"])
                     for t in train if t != k)
            em = f["emb"] / np.linalg.norm(f["emb"])
            dc = min(1 - float(em @ (feat[t]["emb"] / np.linalg.norm(feat[t]["emb"]))) for t in train if t != k)
            r = dict(d_a=float(np.mean(da)) if da else None, d_b=float(db), d_c=float(dc), referent=f["ref_names"])
            if m is not None:
                r["d_d"] = float(min(np.linalg.norm(pose[m][k] - pose[m][t]) for t in train if t != k))
            out[k] = r
        return out
    from fc_d0_support import rankdata
    d0 = json.load(open(os.path.join(OUT, "d0_support.json"), encoding="utf-8"))["per_task"]
    d2s = json.load(open(os.path.join(OUT, "d2_support.json"), encoding="utf-8"))
    seen76 = set()
    for t in tl["bv_train"]:
        seen76 |= set(CS.compounds(tasks[t]))
    bv72 = [t for t in tl["bv_train"] if t not in C1_HELD]
    blocks = {"bv-main2": (tl["dev"], tl["bv_train"], "main2"), "bv-c1": (tl["dev"], bv72, "c1"), "bv-vanilla": (tl["dev"], tl["bv_train"], None),
              "pi05-L90": (tl["l90"], tl["lib40"], None)}
    for b in ("pi05-LI", "groot-LI", "xvla-LI"):
        blocks[b] = (sorted(d2s["LI"]), tl["lib40"], None)
    res = {}
    for b, (test, train, m) in blocks.items():
        D = dists(test, train, m)
        for k in test:
            if b.startswith("bv-"):
                D[k].update(score=d0[k]["score_all"], u=float(CS.unseen_frac(tasks[k], seen76)))
            else:
                src = d2s["L90" if b == "pi05-L90" else "LI"][k]
                D[k].update(score=src["score"], u=src["u"])
        keys = sorted(D)
        for c in ("d_a", "d_b", "d_c"):
            have = [k for k in keys if D[k][c] is not None]
            for k, r in zip(have, rankdata([D[k][c] for k in have]) / len(have)):
                D[k][f"r_{c}"] = float(r)
        for k in keys:
            v = [D[k].get(f"r_{c}") for c in ("d_a", "d_b", "d_c") if D[k].get(f"r_{c}") is not None]
            D[k]["input_composite"] = float(np.mean(v))
        res[b] = D
    with open(os.path.join(OUT, "n1_novelty.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for b, D in res.items():
        print(b, len(D), {c: round(float(np.mean([r[c] for r in D.values() if r.get(c) is not None])), 4) for c in ("d_a", "d_b", "d_c", "d_d") if any(r.get(c) is not None for r in D.values())},
              "no referent:", sum(r["d_a"] is None for r in D.values()))


if __name__ == "__main__":
    main()

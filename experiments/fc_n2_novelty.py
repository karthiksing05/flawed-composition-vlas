"""FLAWED_COMPOSITION N2 (section 4.11): (1) the compound factor checked by rule against the 40 LIBERO training tasks of pi0.5_libero
and ar0s/groot_libero, and (2) N1's input-novelty measures for every N2 task, from the renders of fc_n2_tasks.py.  No outcome is
read.  SmolVLA venv, one GPU (the frozen SigLIP tower N1 used).

(1) Each N2 BDDL's goal atom is parsed with lifelong_ood/policy/factors.py's own rules into a slot (pred, object type, target type,
    region).  KNOWN iff some training task has a slot with the same (object type, target type, region); NEW iff none has, while the
    object type is moved in some training task and (target type, region) is some training task's target.  Asserted to match the
    design in n2_tasks.json.
(2) d_a (referent and goal positions), d_b (scene + end-effector + gripper), d_c (SigLIP cosine): fc_n1_novelty's definitions with
    the 40 LIBERO tasks (results/flawed/n1_render, n1_tasks.json lib40) as training set; composite = mean within-N2 rank of the three.
-> results/flawed/n2_novelty.json

    python experiments/fc_n2_novelty.py
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


def slot_of(text):
    from lifelong_ood.policy import factors as F
    import re
    inst = dict(F._typed(F._section(text, "objects")), **F._typed(F._section(text, "fixtures")))
    goal = [tuple(a.split()) for a in re.findall(r"\(([A-Z][A-Za-z]* [^()]*)\)", F._section(text, "goal"))]
    assert len(goal) == 1 and goal[0][0] in ("In", "On"), goal
    pred, a, b = goal[0]
    tgt, region = F._target(b, inst)
    return (pred, inst.get(a, a), tgt, region)


def main():
    from lifelong_ood.policy import factors as F
    from lifelong_ood.policy.backbone_bindvla import BindVLABackbone
    from fc_n1_novelty import CK, chamfer, inst_of
    from fc_d0_support import rankdata
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    tl = json.load(open(os.path.join(OUT, "n1_tasks.json"), encoding="utf-8"))
    train = {t.tid: t for t in F.load(F.TRAIN_SUITES)}
    assert len(train) == 40 and sorted(train) == sorted(tl["lib40"])
    tr_slots = [(k, s) for k, t in train.items() for s in t.slots if s[0] in ("In", "On")]
    comp = {}
    for c in sorted({t["compound"] for t in n2["tasks"]}):
        t = next(x for x in n2["tasks"] if x["compound"] == c)
        s = slot_of(open(os.path.join(ROOT, t["bddl"]), encoding="utf-8").read())
        same = sorted(k for k, x in tr_slots if x[1:] == s[1:])
        obj_seen = sorted({k for k, x in tr_slots if x[1] == s[1]})
        tgt_seen = sorted({k for k, x in tr_slots if x[2:] == s[2:]})
        status = "known" if same else ("new" if obj_seen and tgt_seen else "atom-novel")
        assert status == t["status"], (c, s, status, t["status"])
        comp[c] = dict(slot=list(s), status=status, training_tasks_with_slot=same, n_training_tasks_moving_object=len(obj_seen),
                       n_training_tasks_with_target=len(tgt_seen), language=t["language"])
    # input novelty
    def feat_of(z, goal):
        names = [str(x) for x in z["names"]]
        rel = z["pos"] - z["base"][:, None]
        ro = inst_of(goal[1], names) if len(goal) > 1 else None
        rg = inst_of(goal[2], names) if len(goal) > 2 else None
        with torch.no_grad():
            img = torch.from_numpy(z["img"]).permute(0, 3, 1, 2).float().cuda() / 255.0
            e = vis.bv.vision(img)[0].mean(1).mean(0).float().cpu().numpy()
        return dict(obj=rel[:, names.index(ro)].mean(0) if ro else None, goal=rel[:, names.index(rg)].mean(0) if rg else None,
                    scene=rel.mean(0), eef=(z["state8"][:, :3] - z["base"]).mean(0), grip=z["state8"][:, 6:8].mean(0), emb=e,
                    ref_names=[ro, rg])
    vis = BindVLABackbone(CK["vanilla"], device="cuda")
    feat = {}
    for k in tl["lib40"]:
        z = dict(np.load(os.path.join(OUT, "n1_render", f"{k.split(':')[0]}_{k.split(':')[1]}.npz"), allow_pickle=True))
        g = train[k].goal[0] if train[k].goal else ()
        feat[k] = feat_of(z, g)
    for t in n2["tasks"]:
        z = dict(np.load(os.path.join(OUT, "n2_render", f"{n2['suite']}_{t['tid']}.npz"), allow_pickle=True))
        feat[t["tid"]] = feat_of(z, tuple(t["goal"]))
    res = {}
    for t in n2["tasks"]:
        f = feat[t["tid"]]
        da = []
        for role in ("obj", "goal"):
            if f[role] is None:
                continue
            c = [np.linalg.norm(f[role] - feat[k][role]) for k in tl["lib40"] if feat[k][role] is not None]
            if c:
                da.append(min(c))
        db = min(chamfer(f["scene"], feat[k]["scene"]) + np.linalg.norm(f["eef"] - feat[k]["eef"]) + np.linalg.norm(f["grip"] - feat[k]["grip"])
                 for k in tl["lib40"])
        em = f["emb"] / np.linalg.norm(f["emb"])
        dc = min(1 - float(em @ (feat[k]["emb"] / np.linalg.norm(feat[k]["emb"]))) for k in tl["lib40"])
        res[str(t["tid"])] = dict(key=t["key"], compound=t["compound"], status=t["status"], layout=t["layout"], d_a=float(np.mean(da)),
                                  d_b=float(db), d_c=float(dc), referent=f["ref_names"])
    keys = sorted(res, key=int)
    for c in ("d_a", "d_b", "d_c"):
        for k, r in zip(keys, rankdata([res[k][c] for k in keys]) / len(keys)):
            res[k][f"r_{c}"] = float(r)
    for k in keys:
        res[k]["input_composite"] = float(np.mean([res[k][f"r_{c}"] for c in ("d_a", "d_b", "d_c")]))
    cells = {}
    for st in ("known", "new"):
        for lay in ("familiar", "new"):
            rr = [r for r in res.values() if r["status"] == st and r["layout"] == lay]
            cells[f"{st}/{lay}"] = {c: float(np.mean([r[c] for r in rr])) for c in ("d_a", "d_b", "d_c", "input_composite")}
    out = dict(compounds=comp, tasks=res, cells=cells, training_set="the 40 LIBERO tasks (n1_tasks.json lib40)",
               parser_hash=F.parser_hash())
    with open(os.path.join(OUT, "n2_novelty.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for c, v in comp.items():
        print(c, v["status"], v["slot"], "with slot:", v["training_tasks_with_slot"])
    for c, v in cells.items():
        print(c, {k: round(x, 4) for k, x in v.items()})


if __name__ == "__main__":
    main()

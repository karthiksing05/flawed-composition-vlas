"""PLAN_SOLVE S3b read (docs/PLAN_S3.md; pre-registration experiments/prereg/ps3b.py, ledger m3_prereg tag ps3b): RoboPoint-13B
(experiments/s3b_robopoint.py, results/bind/s3b/pred_dev.npz) as a point binder on S1's 240 dev units, scored exactly as S1
(s1_read.hit: peak in the target and covered >= every competitor; s1_read.iou).  A copy of s1b_read.py with the prediction file
swapped; the field "s1b_hit" below holds ROBOPOINT's hit (kept for the shared cell code); S1b's and S1c's own hits on the same units
are carried as ref_s1b / ref_s1c (from results/bind/s1b.json, s1c.json).  CPU (chunks env).
NO pass bar (comparison only; pre-declared report: RoboPoint vs S1b / S1c / B1a by depth x seen / held-out / held-out body-seen).

    python experiments/s3b_read.py [--record]  -> results/bind/s3b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
S1 = os.path.join(ROOT, "results", "bind", "s1")
OD = os.path.join(ROOT, "results", "bind", "s3b")


def main():
    import d4_read as DR
    import s1_read as SR
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    pz = np.load(os.path.join(OD, "pred_dev.npz"), allow_pickle=True)
    pred = {str(k): (int(p), m, json.loads(str(mt))) for k, p, m, mt in zip(pz["keys"], pz["peak"], pz["mask"], pz["meta"])}
    s1 = {f"{u['role']}|{u['frame']}": u for u in json.load(open(os.path.join(ROOT, "results", "bind", "s1.json")))["units"]}
    ref = {}
    for nm in ("s1b", "s1c"):        # S1b / S1c hits on the same units, keyed like s1 (role|frame)
        js = json.load(open(os.path.join(ROOT, "results", "bind", f"{nm}.json")))
        ref[nm] = [(u["tid"], u["role"], u["phrase"], u["s1b_hit"]) for u in js["units"]]   # same loop order over labels["dev"] as here (asserted below)
        assert len(ref[nm]) == len(lab["dev"])
    cache = {}

    def frame(f):
        if f not in cache:
            z = np.load(os.path.join(ROOT, f), allow_pickle=True)
            cache[f] = (z, *SR.masks(z), [str(b) for b in z["body_names"]])
        return cache[f]
    seen_types = set()
    for e in lab["train"]:
        _, names, _, bodies = frame(e["frame"])
        for t in e["targets"]:
            seen_types.add(SR.itype(t) if t in bodies else SR.competitors(t, names, bodies)[1])
    units = []
    for ix, e in enumerate(lab["dev"]):
        z, names, cov, bodies = frame(e["frame"])
        tg = e["targets"][0]
        t = cov[names.index(tg)]
        comp, btype = SR.competitors(tg, names, bodies)
        if tg in bodies:
            btype = SR.itype(tg)
        cc = [cov[names.index(c)] for c in comp]
        key = f"{e['role']}|{e['frame']}"
        pk, msk, meta = pred[key]
        o = s1[key]
        for nm in ref:
            assert ref[nm][ix][:3] == (e["tid"], e["role"], e["phrase"]), (nm, ix)
        rb, rc = ref["s1b"][ix][3], ref["s1c"][ix][3]
        units.append(dict(tid=e["tid"], role=e["role"], phrase=e["phrase"], target=tg, depth=e["depth"], seen=e["seen"],
                          body_type=btype, body_seen=btype in seen_types, s1b_hit=SR.hit(pk, t, cc), s1b_iou=SR.iou(msk, t),
                          s1b_body=meta.get("body"), s1b_relation=meta.get("relation"), detected=pk >= 0,
                          b1b_hit=o["b1b_hit"], b1a_hit=o["b1a_hit"], anchor_hit=o["anchor_hit"], ref_s1b=rb, ref_s1c=rc))

    def per(us, k):
        d = {}
        for u in us:
            if u.get(k) is not None:
                d.setdefault(u["tid"], []).append(float(u[k]))
        return DR.rate(d)
    cells = {}
    for depth in (0, 1, 2):
        for split in ("all", "seen", "heldout", "heldout_body_seen", "heldout_body_unseen"):
            us = [u for u in units if u["depth"] == depth and (
                split == "all" or (split == "seen" and u["seen"]) or (split == "heldout" and not u["seen"])
                or (split == "heldout_body_seen" and not u["seen"] and u["body_seen"])
                or (split == "heldout_body_unseen" and not u["seen"] and not u["body_seen"]))]
            cells[f"d{depth}_{split}"] = dict(n=len(us), tasks=sorted({u["tid"] for u in us}), s1b_hit=per(us, "s1b_hit"),
                                              s1b_iou=per(us, "s1b_iou"), b1b_hit=per(us, "b1b_hit"), b1a_hit=per(us, "b1a_hit"),
                                              anchor_hit=per(us, "anchor_hit"), detected=per(us, "detected"),
                                              ref_s1b=per(us, "ref_s1b"), ref_s1c=per(us, "ref_s1c"))
    by_role = {r: dict(s1b_hit=per([u for u in units if u["role"] == r], "s1b_hit"),
                       b1a_hit=per([u for u in units if u["role"] == r], "b1a_hit"),
                       b1b_hit=per([u for u in units if u["role"] == r], "b1b_hit")) for r in ("object", "goal")}
    pt = {}
    for u in units:
        p = pt.setdefault(f"{u['tid']}|{u['role']}", dict(phrase=u["phrase"], body=u["s1b_body"], relation=u["s1b_relation"],
                                                         target=u["target"], depth=u["depth"], seen=u["seen"],
                                                         body_seen=u["body_seen"], s1b=[], b1a=[], b1b=[]))
        for k in ("s1b", "b1a", "b1b"):
            p[k].append(u[f"{k}_hit"])
    pt = {k: dict(v, **{m: float(np.mean(v[m])) for m in ("s1b", "b1a", "b1b")}) for k, v in pt.items()}
    bar = {k: dict(n=cells[k]["n"], robopoint=cells[k]["s1b_hit"]["mean"], s1b=cells[k]["ref_s1b"]["mean"], s1c=cells[k]["ref_s1c"]["mean"],
                   b1a=cells[k]["b1a_hit"]["mean"]) for k in ("d0_all", "d0_heldout", "d1_all", "d1_heldout", "d2_all", "d2_seen",
                                                            "d2_heldout", "d2_heldout_body_seen", "d2_heldout_body_unseen")}
    bar["verdict"] = "REPORT (no pass bar)"
    json.dump(dict(bar=bar, cells=cells, by_role=by_role, per_task=pt, units=units, seen_body_types=sorted(seen_types)),
              open(os.path.join(ROOT, "results", "bind", "s3b.json"), "w"), indent=1)
    for k, v in cells.items():
        if v["n"]:
            print(f"[s3b read] {k}: n {v['n']} ({len(v['tasks'])} tasks) | RoboPoint {v['s1b_hit']} | S1b {v['ref_s1b']['mean']} | S1c {v['ref_s1c']['mean']} | B1a {v['b1a_hit']['mean']} | "
                  f"B1b {v['b1b_hit']['mean']} | anchor {v['anchor_hit']['mean']} | RoboPoint iou {v['s1b_iou']['mean']} | "
                  f"detected {v['detected']['mean']}", flush=True)
    for k, v in sorted(pt.items()):
        print(f"[s3b read] {k}: {v['phrase']!r} = body {v['body']!r} + rel {v['relation']!r} -> {v['target']} d{v['depth']} "
              f"seen {v['seen']} body_seen {v['body_seen']}: RoboPoint {v['s1b']:.2f} B1a {v['b1a']:.2f} B1b {v['b1b']:.2f}", flush=True)
    print(f"[s3b read] REPORT {json.dumps(bar)}", flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("s3b_grounder", bench="libero_90_c", tag="read", seed=0, command="python experiments/s3b_read.py --record",
                            metrics=dict(bar=bar, cells=cells, by_role=by_role),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(note="S3b RoboPoint-13B point binder accuracy (S1 hit metric)"), script="experiments/s3b_read.py"))


if __name__ == "__main__":
    main()

"""FLAWED_COMPOSITION N1 read (sections 4.9, 4.10; m3_prereg tag pn1): does input novelty, rather than compound novelty, explain
per-task held-out failure?  CPU.

Rows = (block, task) with failure y: bv-main2 / bv-c1 / bv-vanilla = 1 - closed-loop dev success (results/bindvla/c1_read.json,
29 tasks x 10); pi05-L90 / pi05-LI / groot-LI / xvla-LI = 1 - zs / 20 (the D2 outcome files).  Novelty from n1_novelty.json:
input = the composite (mean within-block rank of d_a, d_b, d_c), compound = mean within-block rank of the D0 score and u.
Every variable is rank-transformed to (0, 1] within its block; CIs are 2,000-draw bootstraps over task identities (seed 0).
  N1a  pooled Spearman of input novelty with failure, block as stratum (residualised on block dummies); holds if rho > 0, CI excludes 0
  N1b  partial(input | score, u, block) and the reverse partial(compound | input, block).  "Where, not what" holds if the first
       CI excludes 0 (rho > 0) AND the reverse CI includes 0.  Falsified as stated: the first CI includes 0 (input novelty adds
       nothing beyond compound novelty).  Both CIs excluding 0: both matter
  N1c  main2 and C1: Spearman of the per-task D1 dev excess (d1_read.json per_task_delta_F) with the block's input composite, over
       the 29 dev tasks; holds if rho > 0 and the CI excludes 0
Secondary: each component (d_a, d_b, d_c; d_d in the BindVLA blocks) in N1a / N1b; the pool without the BindVLA blocks; per block;
the BindVLA blocks of main2 and C1 with the continuous outcomes (a) the share of episodes with neither success nor first contact on
the named object (12.4 logs) and (b) the gripper's closest approach to the named object over the episode (the logged rollouts:
min over policy queries of |eef - named object position|; tasks whose named object is a fixture have none).
-> results/flawed/n1_read.json

    python experiments/fc_n1_read.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
OUT = os.path.join(ROOT, "results", "flawed")
RR = os.path.join(ROOT, "results", "robot")
BV = os.path.join(ROOT, "results", "bindvla")


def closest_approach(tag, cl):
    """per task mean over logged episodes of min_q |eef(q) - named object position at q|."""
    named = {(r["tid"], int(r["trial"])): r["named"] for r in cl["rows"]}
    per = {}
    for js in glob.glob(os.path.join(BV, "eval", f"devlog_bindvla_{tag}_*.json")):
        suite = os.path.basename(js).split(f"devlog_bindvla_{tag}_")[1].rsplit("_", 1)[0]
        for f in sorted(glob.glob(js[:-5] + "_rec/*.npz")):
            z = np.load(f, allow_pickle=True)
            if "obj_pos" not in z or "qstate" not in z:
                continue
            tid = f"{suite}:{int(z['task'])}"
            nm = named.get((tid, int(z["trial"])))
            names = [str(x) for x in z["obj_names"]]
            if nm is None or nm not in names:
                continue
            P, qt, qs = z["obj_pos"], np.asarray(z["qt"]).astype(int), z["qstate"]
            j = names.index(nm)
            d = [np.linalg.norm(qs[q][:3] - P[min(qt[q], len(P) - 1), j]) for q in range(len(qt))]
            per.setdefault(tid, []).append(float(min(d)))
    return {t: float(np.mean(v)) for t, v in per.items()}


def main():
    from fc_d2_read import boot_partial
    nov = json.load(open(os.path.join(OUT, "n1_novelty.json"), encoding="utf-8"))
    c1r = json.load(open(os.path.join(BV, "c1_read.json"), encoding="utf-8"))["per_task_c1"]
    zs = {b: {k: v["zs"] for k, v in json.load(open(os.path.join(RR, f), encoding="utf-8"))["per_task"].items()}
          for b, f in (("pi05-L90", "g8_read.json"), ("pi05-LI", "g15_read_pi05.json"), ("groot-LI", "g18_read_groot_libero.json"),
                       ("xvla-LI", "g18_read_xvla.json"))}
    rows = []
    for b, D in nov.items():
        for k, r in D.items():
            if b.startswith("bv-"):
                y = 1.0 - c1r[k][b.split("-")[1]]
            else:
                kk = k.split(":")[1] if b == "pi05-L90" else k
                if kk not in zs[b]:
                    continue
                y = 1.0 - zs[b][kk] / 20.0
            rows.append(dict(block=b, task=k, y=y, input=r["input_composite"], d_a=r["d_a"], d_b=r["d_b"], d_c=r["d_c"], d_d=r.get("d_d"),
                             score=r["score"], u=r["u"]))
    # compound novelty = mean within-block rank of score and u
    from fc_d2_read import rank01
    for b in {r["block"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["block"] == b]
        rs, ru = rank01([rows[i]["score"] for i in idx]), rank01([rows[i]["u"] for i in idx])
        for i, a1, a2 in zip(idx, rs, ru):
            rows[i]["compound"] = float(0.5 * (a1 + a2))
    dist = {}
    for b in sorted({r["block"] for r in rows}):
        y = np.array([r["y"] for r in rows if r["block"] == b])
        dist[b] = dict(n=len(y), mean_success=float(1 - y.mean()), always_fail=int((y == 1).sum()), always_succeed=int((y == 0).sum()),
                       between=int(((y > 0) & (y < 1)).sum()))
    strat = ("score_none",)
    for r in rows:
        r["score_none"] = 0.0                                   # a constant control: N1a's stratified correlation = block dummies only
    res = dict(success_distribution=dist, n_rows=len(rows))
    res["N1a"] = boot_partial(rows, "input", controls=strat)
    res["N1b_input_beyond_compound"] = boot_partial(rows, "input", controls=("score", "u"))
    res["N1b_compound_beyond_input"] = boot_partial(rows, "compound", controls=("input",))
    a, rv = res["N1b_input_beyond_compound"], res["N1b_compound_beyond_input"]
    first = a["rho"] > 0 and a["ci95"][0] > 0
    rev_null = rv["ci95"][0] <= 0 <= rv["ci95"][1]
    res["N1a_verdict"] = "holds" if (res["N1a"]["rho"] > 0 and res["N1a"]["ci95"][0] > 0) else "falsified"
    res["N1b_verdict"] = ("where, not what: holds" if first and rev_null else
                          "both matter" if first else "falsified: input novelty adds nothing beyond compound novelty")
    sec = {}
    for c in ("d_a", "d_b", "d_c"):
        sec[f"N1a_{c}"] = boot_partial([r for r in rows if r[c] is not None], c, controls=strat)
        sec[f"N1b_{c}"] = boot_partial([r for r in rows if r[c] is not None], c, controls=("score", "u"))
    bvr = [r for r in rows if r["block"] in ("bv-main2", "bv-c1")]
    sec["N1a_d_d_bv"] = boot_partial(bvr, "d_d", controls=strat)
    nobv = [r for r in rows if not r["block"].startswith("bv-")]
    sec["N1a_no_bv"] = boot_partial(nobv, "input", controls=strat)
    sec["N1b_no_bv"] = boot_partial(nobv, "input", controls=("score", "u"))
    sec["N1b_rev_no_bv"] = boot_partial(nobv, "compound", controls=("input",))
    per = {}
    for b in sorted({r["block"] for r in rows}):
        br = [r for r in rows if r["block"] == b]
        per[b] = dict(N1a=boot_partial(br, "input", controls=strat), N1b=boot_partial(br, "input", controls=("score", "u")),
                      N1b_rev=boot_partial(br, "compound", controls=("input",)))
    # continuous outcomes, BindVLA main2 / C1
    cont = {}
    for m, tag, f in (("main2", "main2_60000", "cldiag_bindvla_main2_60000.json"), ("c1", "c1main_20000", "cldiag_bindvla_c1main_20000.json")):
        cl = json.load(open(os.path.join(BV, f), encoding="utf-8"))
        nc = {}
        for r in cl["rows"]:
            if r["named"] is not None:
                nc.setdefault(r["tid"], []).append(float(not (r["success"] or r["first_contact"] == "named")))
        ca = closest_approach(tag, cl)
        b = f"bv-{m}"
        for name, val in (("named_contact_failure", {t: float(np.mean(v)) for t, v in nc.items()}), ("closest_approach", ca)):
            rr = [dict(r, y=val[r["task"]]) for r in rows if r["block"] == b and r["task"] in val]
            cont[f"{m}_{name}"] = dict(n=len(rr), N1a=boot_partial(rr, "input", controls=strat),
                                       N1b=boot_partial(rr, "input", controls=("score", "u")),
                                       N1b_rev=boot_partial(rr, "compound", controls=("input",)))
    # N1c
    d1 = json.load(open(os.path.join(OUT, "d1_read.json"), encoding="utf-8"))["per_task_delta_F"]
    n1c = {}
    for m in ("main2", "c1"):
        rr = [dict(block=f"bv-{m}", task=k, y=d1[m][k], input=r["input_composite"], score_none=0.0)
              for k, r in nov[f"bv-{m}"].items() if k in d1[m]]
        n1c[m] = boot_partial(rr, "input", controls=strat)
        n1c[m]["verdict"] = "holds" if (n1c[m]["rho"] > 0 and n1c[m]["ci95"][0] > 0) else "falsified"
    res.update(secondary=sec, per_block=per, continuous=cont, N1c=n1c, rows=rows)
    with open(os.path.join(OUT, "n1_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()

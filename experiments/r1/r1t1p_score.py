"""Score the pinned Gate T1 rerun (compositional-architecture registry: temporal_T1_pinned_registered + temporal_T1_pinned_amended).
Numpy only, no simulation.  Reads results/r1/t1p/{none,grasp,random,oracle,perm}/s{S}/{group}.json, writes results/r1/t1p/t1p_score.json.

Order (as registered): (1) validity check: every ungated cell has identical success AND steps across the five arms; if not, stop;
(2) T1-P1..P4 + falsifier with the temporal_T1_registered thresholds on grasp vs random; (3) T1-P5 grasp vs perm and T1-P6 grasp vs none,
exact one-sided McNemar on gated pair cells, net >= +.03 and p < .05; (4) context: the same contrasts per seed, and the 1x prefix (e < 10).
"""
import json
import math
import os
import sys

import numpy as np

OUT = "/coc/flash10/ksingara3/flawed-composition-vlas/results/r1/t1p"
ARMS = ("none", "grasp", "random", "oracle", "perm")
S3 = ["single_HB", "single_OW", "single_TC"]
P3 = ["pair_HBOW", "pair_HBTC", "pair_OWTC"]
UNGATED = [(s, "single_HB") for s in (0, 1, 2)] + [(s, "single_TC") for s in (0, 1)] + [(s, "pair_HBTC") for s in (0, 1)]
GATED_PAIRS = [(s, "pair_HBOW") for s in (0, 1, 2)] + [(s, "pair_OWTC") for s in (0, 1, 2)] + [(2, "pair_HBTC")]


def load(arm, s, g):
    f = f"{OUT}/{arm}/s{s}/{g}.json"
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def vec(d, key="success", prefix=None):
    out = []
    for c in sorted(d["cells"]):
        v = d["cells"][c][key]
        out += list(v[:prefix] if prefix else v)
    return out


def mcnemar(a, b):
    """exact one-sided p for 'a beats b' on paired 0/1 vectors; returns (a_only, b_only, net_frac, p)"""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    ao, bo = int((a & ~b).sum()), int((b & ~a).sum())
    n = ao + bo
    p = sum(math.comb(n, k) for k in range(ao, n + 1)) / 2 ** n if n else 1.0
    return dict(a_only=ao, b_only=bo, net=round((ao - bo) / len(a), 4), p_one_sided=round(p, 4), n_episodes=len(a))


def main():
    res = dict(missing=[], validity={}, verdict={})
    data = {}
    for arm in ARMS:
        for s in (0, 1, 2):
            for g in S3 + P3:
                d = load(arm, s, g)
                if d is None:
                    res["missing"].append(f"{arm}/s{s}/{g}")
                data[arm, s, g] = d
    if res["missing"]:
        print("MISSING", res["missing"]); json.dump(res, open(f"{OUT}/t1p_score.json", "w", encoding="utf-8"), indent=1); return 1

    # (1) validity
    ok_all = True
    for s, g in UNGATED:
        ref_s, ref_t = vec(data["none", s, g]), vec(data["none", s, g], "steps")
        row = {}
        for arm in ARMS[1:]:
            same = vec(data[arm, s, g]) == ref_s and vec(data[arm, s, g], "steps") == ref_t
            row[arm] = same; ok_all &= same
        res["validity"][f"s{s}/{g}"] = row
    res["validity_pass"] = bool(ok_all)
    print("validity", "PASS" if ok_all else "FAIL", json.dumps(res["validity"]))

    m = lambda v: round(float(np.mean(v)), 4)
    tab = {}
    for arm in ARMS:
        b = {}
        for s in (0, 1, 2):
            b[f"seed{s}"] = dict(by_group={g: m(vec(data[arm, s, g])) for g in S3 + P3},
                                 pairs=m(sum((vec(data[arm, s, g]) for g in P3), [])), singles=m(sum((vec(data[arm, s, g]) for g in S3), [])))
        b["mean"] = {k: m([b[f"seed{s}"][k] for s in (0, 1, 2)]) for k in ("pairs", "singles")}
        b["mean"].update({g: m([b[f"seed{s}"]["by_group"][g] for s in (0, 1, 2)]) for g in S3 + P3})
        tab[arm] = b
    res["arms"] = tab
    if not ok_all:
        res["verdict"] = "NOT SCORED: validity check failed (registered stop)"
        json.dump(res, open(f"{OUT}/t1p_score.json", "w", encoding="utf-8"), indent=1); print(res["verdict"]); return 0

    A, B = tab["grasp"], tab["random"]
    diffs = [A[f"seed{s}"]["pairs"] - B[f"seed{s}"]["pairs"] for s in (0, 1, 2)]
    v = {}
    v["T1-P1"] = dict(holds=bool(A["mean"]["pairs"] >= .60 and A["mean"]["pair_OWTC"] >= .47), pairs=A["mean"]["pairs"], OWTC=A["mean"]["pair_OWTC"])
    v["T1-P2"] = dict(holds=bool(abs(A["mean"]["pair_HBTC"] - .711) <= .07), HBTC=A["mean"]["pair_HBTC"], HBTC_pinned_none=tab["none"]["mean"]["pair_HBTC"])
    v["T1-P3"] = dict(holds=bool(A["mean"]["singles"] >= .55), singles=A["mean"]["singles"])
    v["T1-P4"] = dict(holds=bool(A["mean"]["pairs"] - B["mean"]["pairs"] >= .05 and sum(x >= .05 - 1e-9 for x in diffs) >= 2),
                      mean_diff=round(A["mean"]["pairs"] - B["mean"]["pairs"], 4), per_seed_diff=[round(x, 4) for x in diffs])
    v["falsifier"] = dict(triggered=bool(A["mean"]["pairs"] < .55 or A["mean"]["pair_OWTC"] < .40 or not v["T1-P4"]["holds"]))
    gp = lambda arm, prefix=None: sum((vec(data[arm, s, g], prefix=prefix) for s, g in GATED_PAIRS), [])
    for name, other in (("T1-P5", "perm"), ("T1-P6", "none")):
        t = mcnemar(gp("grasp"), gp(other))
        v[name] = dict(holds=bool(t["net"] >= .03 and t["p_one_sided"] < .05), vs=other, **t)
    res["verdict"] = v
    ctx = {}
    for other in ("random", "perm", "none", "oracle"):
        ctx[f"gated_pairs_grasp_vs_{other}"] = mcnemar(gp("grasp"), gp(other))
        ctx[f"gated_pairs_grasp_vs_{other}_per_seed"] = {f"s{s}": mcnemar(sum((vec(data["grasp", s2, g]) for s2, g in GATED_PAIRS if s2 == s), []),
                                                                          sum((vec(data[other, s2, g]) for s2, g in GATED_PAIRS if s2 == s), [])) for s in (0, 1, 2)}
        ctx[f"gated_pairs_grasp_vs_{other}_prefix_e_lt_10"] = mcnemar(gp("grasp", 10), gp(other, 10))
    res["context"] = ctx
    json.dump(res, open(f"{OUT}/t1p_score.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(v, indent=1))
    for arm in ARMS:
        print(arm, json.dumps(tab[arm]["mean"]))
    print(json.dumps(ctx, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

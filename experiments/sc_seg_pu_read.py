"""Read of PU-L (SCAFFOLD_RESULTS 40): U from poses estimated with LEARNED masks (sc_seg.py pu: the segmenter's pred() in place of the
simulator's segmentation; depth, calibration and Estimator unchanged) vs the same pipeline on simulator masks (sc_pu.py percep arm,
SR 38).  numpy + json only.  Writes results/flawed/sc_seg_pu_read.json (refuses to overwrite).

    python experiments/sc_seg_pu_read.py

Cells: known (K1-K6) / new pairings (N1-N6) x familiar / new layout; seeds 10-12; N2 main states.  Primary: learned - percep per cell,
bootstrapped over tasks and states jointly (2,000 draws, rng 0).  Also reported (no verdict): learned - oracle.
Verdict (registered in m3_prereg sc_seg_pu before any non-smoke learned rollout):
  COSTS   if any cell's CI upper bound < -.10;
  HOLDS   else if every cell's CI lower bound >= -.10;
  UNRESOLVED otherwise.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
SEEDS = (10, 11, 12)
COMPS = tuple("K%d" % i for i in range(1, 7)) + tuple("N%d" % i for i in range(1, 7))
GROUPS = dict(known=COMPS[:6], new_pairings=COMPS[6:])
LAYOUTS = ("familiar", "new")
B, MARGIN = 2000, 0.10
DST = os.path.join(OUT, "sc_seg_pu_read.json")


def main():
    assert not os.path.exists(DST), "exists: " + DST
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    ks = [int(k) for k in n2["states"]["main"]]

    def tid(c, l):
        return next(t for t in n2["tasks"] if t["compound"] == c and t["layout"] == l)["tid"]

    def succ(path):
        R = json.load(open(path, encoding="utf-8"))
        assert not R.get("smoke"), path
        d = {int(e["trial"]): e for e in R["eps"]}
        assert sorted(d) == sorted(ks), (path, sorted(d))
        for e in R["eps"]:
            assert e.get("error") is None, (path, e.get("error"))
        return np.array([float(bool(d[k]["success"])) for k in ks])

    def path(arm, s, c, l):
        if arm == "learned":
            return os.path.join(OUT, "sc_seg", "pu_learned", "learned_s%d_%d.json" % (s, tid(c, l)))
        return os.path.join(OUT, "sc_pu", "%s_s%d_%d.json" % (arm, s, tid(c, l)))

    def arr(arm, comps, l):
        return np.array([[succ(path(arm, s, c, l)) for s in SEEDS] for c in comps])        # [tasks, seeds, states]

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()),
                    ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    res = dict(primary={}, vs_oracle={}, per_task={})
    for g, comps in GROUPS.items():
        for l in LAYOUTS:
            L, P, O = arr("learned", comps, l), arr("percep", comps, l), arr("oracle", comps, l)
            k = "%s|%s" % (g, l)
            res["primary"][k] = diff(L, P)
            res["vs_oracle"][k] = diff(L, O)
            for i, c in enumerate(comps):
                res["per_task"].setdefault(c, {})[l] = dict(learned=float(L[i].mean()), percep=float(P[i].mean()), oracle=float(O[i].mean()))
    cells = list(res["primary"].values())
    if any(x["ci"][1] < -MARGIN for x in cells):
        lab = "COSTS"
    elif all(x["ci"][0] >= -MARGIN for x in cells):
        lab = "HOLDS"
    else:
        lab = "UNRESOLVED"
    res["verdict"] = lab
    print("[seg_pu_read] verdict", lab, flush=True)
    for k in res["primary"]:
        x, y = res["primary"][k], res["vs_oracle"][k]
        print("[seg_pu_read] %-22s learned %.3f percep %.3f oracle %.3f | L-P %+.3f [%+.3f, %+.3f] | L-O %+.3f [%+.3f, %+.3f]" % (
            k, x["a"], x["b"], y["b"], x["delta"], x["ci"][0], x["ci"][1], y["delta"], y["ci"][0], y["ci"][1]), flush=True)
    for c in COMPS:
        print("[seg_pu_read] %s %s" % (c, json.dumps({l: {a: round(v, 3) for a, v in d.items()} for l, d in res["per_task"][c].items()})), flush=True)
    with open(DST + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(DST + ".tmp", DST)


if __name__ == "__main__":
    main()

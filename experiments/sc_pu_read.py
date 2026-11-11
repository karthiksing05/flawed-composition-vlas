"""Read of PU (SCAFFOLD_RESULTS 38): U from perceived poses (sc_pu.py percep arm) vs U from simulator poses (sc_pu.py oracle arm).
numpy + json only (CPU, any venv).  Writes results/flawed/sc_pu_read.json (refuses to overwrite).

    python experiments/sc_pu_read.py

Cells: known (K1-K6) / new pairings (N1-N6) x familiar / new layout; seeds 10-12; N2 main states.  Each cell's difference percep -
oracle is bootstrapped over tasks and states jointly (2,000 draws, rng 0), as sc_t4's read.
Sanity: the oracle arm must reproduce sc_t4's U rollouts (same checkpoints, states and seeds): episode agreement >= .90 overall,
else the verdict is INVALID.
Verdict (registered in m3_prereg sc_pu before any non-smoke rollout):
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
B, MARGIN, AGREE = 2000, 0.10, 0.90
DST = os.path.join(OUT, "sc_pu_read.json")


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
        if arm == "U":
            return os.path.join(OUT, "sc_t4", "U_s%d_%d.json" % (s, tid(c, l)))
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
        return dict(percep=float(a.mean()), oracle=float(b.mean()), delta=float((a - b).mean()),
                    ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    res = dict(cells={}, sanity={}, per_task={})
    agree_all = []
    for g, comps in GROUPS.items():
        for l in LAYOUTS:
            P, O, U = arr("percep", comps, l), arr("oracle", comps, l), arr("U", comps, l)
            res["cells"]["%s|%s" % (g, l)] = diff(P, O)
            ag = (O == U)
            agree_all.append(ag.ravel())
            res["sanity"]["%s|%s" % (g, l)] = dict(agree=float(ag.mean()), oracle=float(O.mean()), t4_U=float(U.mean()))
            for i, c in enumerate(comps):
                res["per_task"].setdefault(c, {})[l] = dict(percep=float(P[i].mean()), oracle=float(O[i].mean()), t4_U=float(U[i].mean()))
    agree = float(np.concatenate(agree_all).mean())
    res["sanity"]["agree_overall"] = agree
    cells = list(res["cells"].values())
    if agree < AGREE:
        lab = "INVALID"
    elif any(x["ci"][1] < -MARGIN for x in cells):
        lab = "COSTS"
    elif all(x["ci"][0] >= -MARGIN for x in cells):
        lab = "HOLDS"
    else:
        lab = "UNRESOLVED"
    res["verdict"] = lab
    print("[pu_read] verdict", lab, "| oracle-vs-T4 agreement %.3f" % agree, flush=True)
    for k, x in res["cells"].items():
        print("[pu_read] %-22s percep %.3f oracle %.3f delta %+.3f [%+.3f, %+.3f] | T4 agree %.3f" % (
            k, x["percep"], x["oracle"], x["delta"], x["ci"][0], x["ci"][1], res["sanity"][k]["agree"]), flush=True)
    for c in COMPS:
        print("[pu_read] %s %s" % (c, json.dumps({l: {a: round(v, 3) for a, v in d.items()} for l, d in res["per_task"][c].items()})), flush=True)
    with open(DST + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(DST + ".tmp", DST)


if __name__ == "__main__":
    main()

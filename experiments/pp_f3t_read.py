"""PLAN_PARTS targeted F3 read (docs/PARTS_RESULTS.md section 7; m3_prereg pp_f3t).  CPU.

Data: pi0.5, N1-N6, both layouts, N2 states 4-13, N2's seeds.
  intact  results/flawed/pp_f3_roll/pi05_intact_<tid>.json (pp_f3's intact re-run; it matched N2's recorded episodes exactly)
  T, R    results/flawed/pp_f3t_roll/pi05_f3t_<arm>_<tid>.json
Per layout: pooled s, g (success or the object bound) and p = sum s / sum g over the 6 tasks x 10 states, per arm.
CI: 2,000-draw percentile bootstrap (seed 0) resampling tasks and states jointly, every arm on the same draw (paired); pp_f3_read's.
Verdict (familiar layout, primary):
  place_up      dp_T = p(T) - p(intact) > 0 with its CI excluding 0
  grasp_kept    dg_T = g(T) - g(intact) >= -0.10 (point estimate)
  specific      dp_T - dp_R = p(T) - p(R) > 0 with its CI excluding 0
  F3t TIES iff place_up and grasp_kept and specific.
  Otherwise: "raises place, not specific" (place_up and grasp_kept), "raises place, costs grasp" (place_up only), "does not raise place".
-> results/flawed/pp_f3t_read.json

    python experiments/pp_f3t_read.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
STATES = list(range(4, 14))
NEW = ["N1", "N2", "N3", "N4", "N5", "N6"]
B, SEED, TOL_G = 2000, 0, 0.10
ARMS = ("intact", "T", "R")


def episodes(f, obj):
    eps = [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
    return {int(e["trial"]): (float(bool(e["success"])), float(bool(e["success"]) or e.get("bound") == obj), e.get("error")) for e in eps}


def main():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
    res = {}
    for lay in ("familiar", "new"):
        S = {a: np.full((len(NEW), len(STATES)), np.nan) for a in ARMS}
        G = {a: np.full((len(NEW), len(STATES)), np.nan) for a in ARMS}
        errs, missing = 0, []
        for i, c in enumerate(NEW):
            t = tid[(c, lay)]
            for a in ARMS:
                f = (os.path.join(OUT, "pp_f3_roll", f"pi05_intact_{t['tid']}.json") if a == "intact"
                     else os.path.join(OUT, "pp_f3t_roll", f"pi05_f3t_{a}_{t['tid']}.json"))
                if not os.path.exists(f):
                    missing.append(f)
                    continue
                E = episodes(f, t["object"])
                for j, k in enumerate(STATES):
                    if k in E:
                        S[a][i, j], G[a][i, j] = E[k][0], E[k][1]
                        errs += E[k][2] is not None
        if missing:
            res[lay] = dict(missing=missing)
            continue

        def stats(ti, si):
            r = {}
            for a in ARMS:
                s, g = S[a][np.ix_(ti, si)], G[a][np.ix_(ti, si)]
                r[a] = dict(s=float(np.nanmean(s)), g=float(np.nanmean(g)), p=float(np.nansum(s) / max(np.nansum(g), 1e-9)))
            return r
        pt = stats(np.arange(len(NEW)), np.arange(len(STATES)))
        rng = np.random.default_rng(SEED)
        keys = [(x, a, b) for x in ("p", "s", "g") for a, b in (("T", "intact"), ("R", "intact"), ("T", "R"))]
        bs = {k: [] for k in keys}
        for _ in range(B):
            ti, si = rng.integers(0, len(NEW), len(NEW)), rng.integers(0, len(STATES), len(STATES))
            r = stats(ti, si)
            for x, a, b in keys:
                bs[(x, a, b)].append(r[a][x] - r[b][x])
        d = {f"d{x}_{a}-{b}": [pt[a][x] - pt[b][x], float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for (x, a, b), v in bs.items()}
        per = {c: {a: dict(s=float(np.nanmean(S[a][i])), g=float(np.nanmean(G[a][i]))) for a in ARMS} for i, c in enumerate(NEW)}
        res[lay] = dict(arms=pt, delta=d, per_task=per, episode_errors=int(errs))
    v = None
    fam = res.get("familiar", {})
    if "delta" in fam:
        D = fam["delta"]
        place_up = D["dp_T-intact"][0] > 0 and D["dp_T-intact"][1] > 0
        grasp_kept = D["dg_T-intact"][0] >= -TOL_G
        specific = D["dp_T-R"][0] > 0 and D["dp_T-R"][1] > 0
        v = dict(place_up=place_up, grasp_kept=grasp_kept, specific=specific,
                 verdict=("ties" if place_up and grasp_kept and specific else "raises place, not specific" if place_up and grasp_kept
                          else "raises place, costs grasp" if place_up else "does not raise place"))
    out = dict(layouts=res, F3t=v, tolerance_grasp=TOL_G)
    json.dump(out, open(os.path.join(OUT, "pp_f3t_read.json"), "w", encoding="utf-8"), indent=1)
    print("F3t", json.dumps(v), flush=True)


if __name__ == "__main__":
    main()

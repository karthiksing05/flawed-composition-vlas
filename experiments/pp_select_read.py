"""Selection check read (docs/PLACE_DIAG.md 6; m3_prereg pp_select).  CPU, no simulator.  Saved rollouts of P_min, new layout, dev states and seeds.
Start states are compared on the first 64 entries of the saved state vectors.  -> results/flawed/pp_select_read.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from fa1g_read import radius                                                # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
SEEDS, B = (10, 11, 12), 2000
n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
ks = n2["states"]["main"]
TASKS, CONTRASTS = ("K1", "N1", "N4", "K6", "N6"), (("N1", "K1"), ("N4", "K1"), ("N6", "K6"))


def task(c):
    return next(t for t in n2["tasks"] if t["compound"] == c and t["layout"] == "new")


D = {}
for c in TASKS:
    t = task(c)
    rad = radius(t["object"], t["target"], cal)
    E = {}
    for sd in SEEDS:
        for v in json.load(open(os.path.join(OUT, "pp_cons_roll", f"P2naf_s{sd}_main_{t['tid']}.json"), encoding="utf-8"))["tasks"].values():
            for e in v["episodes"]:
                E[(sd, int(e["trial"]))] = e
    a = np.array([[E[(sd, k)]["min_dxy_held"] is not None and E[(sd, k)]["min_dxy_held"] <= rad for k in ks] for sd in SEEDS])
    s = np.array([[bool(E[(sd, k)]["success"]) for k in ks] for sd in SEEDS])
    d0 = np.array([[np.linalg.norm(E[(sd, k)]["onset_canon"]["foot"][:2]) if E[(sd, k)]["onset_canon"] else np.nan for k in ks] for sd in SEEDS])
    D[c] = dict(a=a, s=s, d0=d0, S=np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy")))


def rate(a, s, kk):
    x, y = a[:, kk], s[:, kk]
    return float((x & y).sum() / x.sum()) if x.sum() else float("nan")


def boot(f):
    rng = np.random.default_rng(0)
    v = np.array([f(rng.choice(len(ks), len(ks))) for _ in range(B)])
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


allk = np.arange(len(ks))
res = dict(contrasts={}, d0={})
for x, y in CONTRASTS:
    X, Y = D[x], D[y]
    Sx, Sy = X["S"][ks], Y["S"][ks]
    same = float(np.mean(np.all(np.abs(Sx - Sy)[:, :64] <= 0.01, axis=1))) if Sx.shape == Sy.shape else 0.0
    both = X["a"] & Y["a"]
    Ga = rate(X["a"], X["s"], allk) - rate(Y["a"], Y["s"], allk)
    Gs = rate(both, X["s"], allk) - rate(both, Y["s"], allk)
    ci = boot(lambda kk: rate(both, X["s"], kk) - rate(both, Y["s"], kk))
    if same < 0.9:
        v = "NOT TESTABLE"
    elif both.sum() < 15:
        v = "TOO FEW"
    elif ci[0] <= 0.0 <= ci[1] and abs(Gs) <= 0.5 * abs(Ga):
        v = "START-STATE SELECTION"
    else:
        v = "NOT SELECTION"
    r = dict(state_shape=[list(Sx.shape), list(Sy.shape)], states_same=same, shared=int(both.sum()), G_all=Ga, G_s=Gs, G_s_ci=ci, verdict=v,
             place_shared=[rate(both, X["s"], allk), rate(both, Y["s"], allk)])
    res["contrasts"][f"{x}-{y}"] = r
    print("VERDICT", x, y, json.dumps(r), flush=True)
for tg in (("K1", "N1", "N4"), ("K6", "N6")):
    pool = np.concatenate([D[c]["d0"][D[c]["a"]] for c in tg])
    e = np.nanpercentile(pool, (100 / 3, 200 / 3))
    for c in tg:
        A, S, d = D[c]["a"], D[c]["s"], D[c]["d0"]
        cell = []
        for lo, hi in ((-np.inf, e[0]), (e[0], e[1]), (e[1], np.inf)):
            m = A & (d >= lo) & (d < hi)
            cell.append([int(m.sum()), int((m & S).sum())])
        res["d0"][c] = dict(edges=[float(z) for z in e], tertiles=cell, median=float(np.nanmedian(d[A])))
        print("D0", c, json.dumps(res["d0"][c]), flush=True)
json.dump(res, open(os.path.join(OUT, "pp_select_read.json"), "w", encoding="utf-8"), indent=1)

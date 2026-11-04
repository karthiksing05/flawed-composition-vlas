"""In-hand test read (docs/PLACE_DIAG.md 4.1; m3_prereg pp_inhand).  CPU.  Dev states 4-23, seeds 10-12.  -> results/flawed/pp_inhand_read.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

OUT = os.path.join(ROOT, "results", "flawed")
SEEDS, B = (10, 11, 12), 2000
TASKS, LAYS = ("K1", "N1", "N4", "K6", "N6"), ("new", "familiar")
CONTRASTS = (("N1", "K1"), ("N4", "K1"), ("N6", "K6"))
ks = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))["states"]["main"]
EDGES = (15.0, 30.0)


def load(arm, comp, lay, d="pp_inhand_trace", pre=None):
    E = {}
    for sd in SEEDS:
        nm = f"{pre or arm}_s{sd}_{comp}_{lay}.json"
        for e in json.load(open(os.path.join(OUT, d, nm), encoding="utf-8"))["episodes"]:
            E[(sd, e["trial"])] = e
    return E


def arr(E):
    a = np.array([[E[(sd, k)]["arrival"] is not None for k in ks] for sd in SEEDS])
    s = np.array([[bool(E[(sd, k)]["success"]) for k in ks] for sd in SEEDS])
    r = [[E[(sd, k)]["arrival"] for k in ks] for sd in SEEDS]
    return a, s, r


def rate(a, s, kk):
    x, y = a[:, kk], s[:, kk]
    return float((x & y).sum() / x.sum()) if x.sum() else float("nan")


def boot(f):
    rng = np.random.default_rng(0)
    v = np.array([f(rng.choice(len(ks), len(ks))) for _ in range(B)])
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def stratum(t):
    return 0 if t < EDGES[0] else (1 if t < EDGES[1] else 2)


def cells(a, s, r, kk):
    c = {}
    for i in range(len(SEEDS)):
        for j in kk:
            if a[i, j]:
                z = c.setdefault(stratum(r[i][j]["tilt"]), [0, 0])
                z[0] += 1
                z[1] += int(s[i, j])
    return c


def adjusted(x, y, kk):
    cA, cB = cells(*x, kk), cells(*y, kk)
    tot = sum(v[0] for v in cA.values()) + sum(v[0] for v in cB.values())
    if not tot:
        return float("nan"), 0.0, cA, cB
    num, mass = 0.0, 0.0
    for z in set(cA) & set(cB):
        w = (cA[z][0] + cB[z][0]) / tot
        num += w * (cA[z][1] / cA[z][0] - cB[z][1] / cB[z][0])
        mass += w
    return (num / mass if mass else float("nan")), mass, cA, cB


def med(a, r, f):
    v = [r[i][j][f] for i in range(len(SEEDS)) for j in range(len(ks)) if a[i, j]]
    return [float(np.percentile(v, q)) for q in (10, 50, 90)] if v else None


res = dict(tasks={}, contrasts={}, agreement={})
allk = np.arange(len(ks))
for lay in LAYS:
    T = {c: {arm: load(arm, c, lay) for arm in ("base", "norot")} for c in TASKS}
    A = {c: {arm: arr(T[c][arm]) for arm in ("base", "norot")} for c in TASKS}
    for c in TASKS:
        st = load(None, c, lay, "pp_place_trace", "P")
        res["agreement"][f"{c}|{lay}"] = float(np.mean([bool(st[k]["success"]) == bool(T[c]["base"][k]["success"]) for k in st]))
        rt = {}
        for arm in ("base", "norot"):
            a, s, r = A[c][arm]
            E = T[c][arm]
            still = [E[(sd, k)]["end"] for sd in SEEDS for k in ks if E[(sd, k)]["arrival"] is not None and not E[(sd, k)]["success"] and E[(sd, k)]["held_end"]]
            rt[arm] = dict(arrived=int(a.sum()), place=rate(a, s, allk), success=float(s.mean()), hold=float(np.mean([E[x]["held_step"] is not None for x in E])),
                           tilt=med(a, r, "tilt"), tilt_onset=med(a, r, "tilt_onset"), hand_turn=med(a, r, "hand_turn"), slip=med(a, r, "slip"),
                           offset_shift=med(a, r, "offset_shift"), tilt_max_after=med(a, r, "tilt_max_after"), by_tilt=cells(a, s, r, allk),
                           still_held=len(still), still_held_end=[float(np.median([e[k] for e in still])) for k in ("d", "h", "tilt")] if still else None,
                           zeroed_median=float(np.median([E[x]["zeroed"] for x in E])))
        res["tasks"][f"{c}|{lay}"] = rt
        print("[task]", c, lay, json.dumps(rt), "agree", res["agreement"][f"{c}|{lay}"], flush=True)
    for x, y in CONTRASTS:
        bx, by = A[x]["base"], A[y]["base"]
        G = rate(bx[0], bx[1], allk) - rate(by[0], by[1], allk)
        ciG = boot(lambda kk: rate(bx[0], bx[1], kk) - rate(by[0], by[1], kk))
        a_, mass, cA, cB = adjusted(bx, by, allk)
        ciA = boot(lambda kk: adjusted(bx, by, kk)[0])
        if ciG[0] <= 0.0 <= ciG[1]:
            ob = "NO GAP"
        elif mass < 0.5:
            ob = "UNDETERMINED"
        elif abs(a_) <= 0.5 * abs(G):
            ob = "TILT-MEDIATED"
        else:
            ob = "NOT EXPLAINED BY TILT"
        low = x if G < 0 else y
        b, n = A[low]["base"], A[low]["norot"]
        D = rate(n[0], n[1], allk) - rate(b[0], b[1], allk)
        ciD = boot(lambda kk: rate(n[0], n[1], kk) - rate(b[0], b[1], kk))
        iv = "ROTATION-CAUSAL" if ciD[0] > 0.0 and D > 0.10 else "NO EFFECT"
        nx, ny = A[x]["norot"], A[y]["norot"]
        Gn = rate(nx[0], nx[1], allk) - rate(ny[0], ny[1], allk)
        res["contrasts"][f"{x}-{y}|{lay}"] = dict(G=G, G_ci=ciG, A=a_, A_ci=ciA, overlap=mass, first=cA, second=cB, observational=ob, lower_task=low,
                                                  delta=D, delta_ci=ciD, intervention=iv, G_norot=Gn)
        print("VERDICT", x, y, lay, ob, "|", iv, json.dumps({k: res["contrasts"][f"{x}-{y}|{lay}"][k] for k in ("G", "G_ci", "A", "A_ci", "overlap", "lower_task", "delta", "delta_ci", "G_norot")}), flush=True)
        print("CELLS", x, y, lay, json.dumps(dict(first=cA, second=cB)), flush=True)
json.dump(res, open(os.path.join(OUT, "pp_inhand_read.json"), "w", encoding="utf-8"), indent=1, default=float)

"""The second hand-over's read (docs/HAND_OVER.md 7; m3_prereg pp_hand2).  CPU.  -> results/flawed/pp_hand2_read.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_atlas as PA                                                       # noqa: E402

n2 = PA.tasks()
ks = n2["states"]["main"]
tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
H1D, H2D = os.path.join(PA.OUT, "pp_hand"), os.path.join(PA.OUT, "pp_hand2")


def load(stem, t):
    R = json.load(open(stem + ".json", encoding="utf-8"))
    E = R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]
    Tr = PA.load_track(stem + "_track.npz")
    assert len(E) == len(ks) == len(Tr) and all(e.get("error") is None for e in E), (stem, len(E), len(Tr))
    A = {s: np.zeros(len(ks)) for s in PA.STAGES}
    for j, (e, r) in enumerate(zip(E, Tr)):
        assert int(e["trial"]) == ks[j]
        d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
        A[PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])[0]][j] = 1.0
    return A


def get(model, arm, c, lay):
    t = tid[(c, lay)]
    if arm == "intact":
        return load(os.path.join(PA.OD, f"{model}_{t['tid']}"), t)
    if arm == "Hs10":
        return load(os.path.join(H1D, f"{model}_H_{t['tid']}"), t)
    if arm == "Hrep":
        a, b = (load(os.path.join(H2D, f"{model}_{x}_{t['tid']}"), t) for x in ("Hs11", "Hs12"))
        return {s: 0.5 * (a[s] + b[s]) for s in PA.STAGES}
    return load(os.path.join(H2D, f"{model}_{arm}_{t['tid']}"), t)


def boot(f):
    rng = np.random.default_rng(0)
    v = [f(rng.choice(len(PA.NEW), len(PA.NEW)), rng.choice(len(ks), len(ks))) for _ in range(PA.B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


ti0, si0 = np.arange(len(PA.NEW)), np.arange(len(ks))


def S(D, a, ti, si):
    return float(np.mean([D[a][PA.NEW[i]]["success"][si].mean() for i in ti]))


def delta(D, a, b):
    e = S(D, a, ti0, si0) - S(D, b, ti0, si0)
    lo, hi = boot(lambda t, s: S(D, a, t, s) - S(D, b, t, s))
    return dict(delta=e, ci=[lo, hi], passes=bool(lo > 0.0 and e > 0.10))


def est(D, a, names):
    def excess(ti, si):
        out = []
        for i in ti:
            c = PA.NEW[i]
            ko, kt = PA.PAIR[c]
            out.append(sum(D[a][c][n][si] - 0.5 * (D[a][ko][n][si] + D[a][kt][n][si]) for n in names).mean())
        return float(np.mean(out))
    e = excess(ti0, si0)
    ci = boot(excess)
    return dict(E=e, ci=ci, breaks=bool(ci[0] > 0.0 and e > 0.10))


def hand_verdict(D, a):
    d = delta(D, a, "intact")
    w, x = est(D, a, PA.GROUP["within"]), est(D, a, PA.GROUP["cross"])
    v = ("NO FIX" if not d["passes"] else {(False, False): "FIXED", (True, False): "RAISED, WITHIN STILL BREAKS", (False, True): "RAISED, CROSS STILL BREAKS", (True, True): "RAISED, BOTH STILL BREAK"}[(w["breaks"], x["breaks"])])
    return dict(verdict=v, s=S(D, a, ti0, si0), delta=d, within=w, cross=x, known=float(np.mean([D[a][c]["success"].mean() for c in PA.KNOWN])), per_task={c: float(D[a][c]["success"].mean()) for c in PA.NEW})


res = {}
for model in ("pi05", "groot"):
    res[model] = {}
    for lay in ("familiar", "new"):
        D = {a: {c: get(model, a, c, lay) for c in PA.NEW + PA.KNOWN} for a in ("intact", "Hs10", "Hrep", "H1")}
        r = {a: hand_verdict(D, a) for a in ("Hrep", "H1")}
        r["intact_s"], r["intact_known"] = S(D, "intact", ti0, si0), float(np.mean([D["intact"][c]["success"].mean() for c in PA.KNOWN]))
        r["Hs10_s"] = S(D, "Hs10", ti0, si0)
        r["H1_minus_Hs10"] = delta(D, "H1", "Hs10")
        r["replicates"] = bool(r["Hrep"]["verdict"] == "FIXED")
        r["contact_verdict"] = "CONTACT NOT NEEDED" if (r["H1"]["verdict"] == "FIXED" and r["Hs10_s"] - r["H1"]["s"] <= 0.10) else "CONTACT MATTERS"
        res[model][lay] = r
        print(model, lay, "replicates", r["replicates"], r["contact_verdict"], json.dumps(r), flush=True)
D = {a: {c: get("pi05", a, c, "familiar") for c in PA.NEW} for a in ("Hs10", "T01", "T02", "T04", "O01", "O02", "O04")}
base = S(D, "Hs10", ti0, si0)
bud = dict(base=base)
for f in ("T", "O"):
    rows = {a: dict(s=S(D, a, ti0, si0), minus_base=delta(D, a, "Hs10"), per_task={c: float(D[a][c]["success"].mean()) for c in PA.NEW}) for a in (f + "01", f + "02", f + "04")}
    within = [int(a[1:]) / 100.0 for a in rows if rows[a]["s"] >= base - 0.10]
    keep = 0.0
    for r_ in (0.01, 0.02, 0.04):
        if r_ in within:
            keep = r_
        else:
            break
    bud[f] = dict(rows=rows, budget_m=keep, verdict=("BELOW 0.01 M" if keep == 0.0 else f"{keep:.2f} M"))
    print("budget", f, json.dumps(bud[f]), flush=True)
res["budget"] = bud
json.dump(res, open(os.path.join(PA.OUT, "pp_hand2_read.json"), "w", encoding="utf-8"), indent=1)

"""The hold gate's read (docs/HOLD_GATE.md 3; m3_prereg pp_gate).  CPU.  -> results/flawed/pp_gate_read.json"""
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
GD = os.path.join(PA.OUT, "pp_gate")


def load(model, arm, c, lay):
    t = tid[(c, lay)]
    stem = os.path.join(PA.OD, f"{model}_{t['tid']}") if arm == "intact" else os.path.join(GD, f"{model}_{arm}_{t['tid']}")
    R = json.load(open(stem + ".json", encoding="utf-8"))
    E = [e for v in R["tasks"].values() for e in v["episodes"]]
    z = np.load(stem + "_track.npz")
    Tr = PA.load_track(stem + "_track.npz")
    assert len(E) == len(ks) == len(Tr) and all(e.get("error") is None for e in E), (stem, len(E), len(Tr))
    A = {s: np.zeros(len(ks)) for s in PA.STAGES}
    held_end, fired = np.zeros(len(ks)), np.zeros(len(ks))
    for j, (e, r) in enumerate(zip(E, Tr)):
        assert int(e["trial"]) == ks[j]
        d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
        s = PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])[0]
        A[s][j] = 1.0
        held_end[j] = float(not e["success"] and bool(r["held"][-1]) and d[-1] > 0.05)
        if f"e{j}_fired" in z:
            fired[j] = float(z[f"e{j}_fired"])
    return A, held_end, fired


def boot(f):
    rng = np.random.default_rng(0)
    v = [f(rng.choice(len(PA.NEW), len(PA.NEW)), rng.choice(len(ks), len(ks))) for _ in range(PA.B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


res = {}
for model in ("pi05", "groot"):
    res[model] = {}
    for lay in ("familiar", "new"):
        D = {arm: {c: load(model, arm, c, lay) for c in PA.NEW} for arm in ("intact", "G", "GA")}
        Kn = {arm: {c: load(model, arm, c, lay) for c in PA.KNOWN} for arm in ("intact", "G")}

        def S(arm, ti, si):
            return float(np.mean([D[arm][PA.NEW[i]][0]["success"][si].mean() for i in ti]))

        ti0, si0 = np.arange(len(PA.NEW)), np.arange(len(ks))

        def delta(a, b):
            e = S(a, ti0, si0) - S(b, ti0, si0)
            lo, hi = boot(lambda t, s: S(a, t, s) - S(b, t, s))
            return dict(delta=e, ci=[lo, hi], passes=bool(lo > 0.0 and e > 0.10))

        r = dict(s={a: S(a, ti0, si0) for a in D}, G_minus_intact=delta("G", "intact"), GA_minus_G=delta("GA", "G"), GA_minus_intact=delta("GA", "intact"))
        r["verdict"] = {(True, True): "TIMING AND BINDING", (True, False): "TIMING", (False, True): "BINDING NEEDS THE GATE", (False, False): "NOT TIMING"}[(r["G_minus_intact"]["passes"], r["GA_minus_G"]["passes"])]
        r["known"] = {a: float(np.mean([Kn[a][c][0]["success"].mean() for c in PA.KNOWN])) for a in Kn}
        r["known_damage"] = bool(r["known"]["intact"] - r["known"]["G"] > 0.05)
        r["per_task"] = {a: {c: {s: float(D[a][c][0][s].mean()) for s in PA.STAGES} for c in PA.NEW} for a in D}
        r["held_end_share"] = {a: float(np.mean([D[a][c][1].mean() for c in PA.NEW])) for a in D}
        r["fired_mean"] = {a: float(np.mean([D[a][c][2].mean() for c in PA.NEW])) for a in ("G", "GA")}

        def excess(arm, names, ti, si):
            karm = "G" if arm == "GA" else arm
            out = []
            for i in ti:
                c = PA.NEW[i]
                ko, kt = PA.PAIR[c]
                out.append(sum(D[arm][c][0][n][si] - 0.5 * (Kn[karm][ko][0][n][si] + Kn[karm][kt][0][n][si]) for n in names).mean())
            return float(np.mean(out))

        r["E"] = {a: {g: dict(E=excess(a, n, ti0, si0), ci=boot(lambda t, s: excess(a, n, t, s))) for g, n in list(PA.GROUP.items()) + [(s, (s,)) for s in PA.STAGES[1:]]} for a in D}
        res[model][lay] = r
        print(model, lay, r["verdict"], json.dumps({k: r[k] for k in ("s", "G_minus_intact", "GA_minus_G", "GA_minus_intact", "known", "known_damage", "held_end_share", "fired_mean")}), flush=True)
        print(" per_task", json.dumps(r["per_task"]), flush=True)
        print(" E", json.dumps(r["E"]), flush=True)
json.dump(res, open(os.path.join(PA.OUT, "pp_gate_read.json"), "w", encoding="utf-8"), indent=1)

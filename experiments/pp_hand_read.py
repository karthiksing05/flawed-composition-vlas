"""The hand-over's read (docs/HAND_OVER.md 3; m3_prereg pp_hand).  CPU.  -> results/flawed/pp_hand_read.json"""
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
HD = os.path.join(PA.OUT, "pp_hand")


def load(stem, t):
    R = json.load(open(stem + ".json", encoding="utf-8"))
    E = R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]
    z = np.load(stem + "_track.npz")
    Tr = PA.load_track(stem + "_track.npz")
    assert len(E) == len(ks) == len(Tr) and all(e.get("error") is None for e in E), (stem, len(E), len(Tr))
    A = {s: np.zeros(len(ks)) for s in PA.STAGES}
    handed = np.zeros(len(ks))
    for j, (e, r) in enumerate(zip(E, Tr)):
        assert int(e["trial"]) == ks[j]
        d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
        A[PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])[0]][j] = 1.0
        if f"e{j}_handed" in z:
            handed[j] = float(int(z[f"e{j}_handed"]) >= 0)
    return A, handed


def stem(model, arm, t):
    if arm == "H":
        return os.path.join(HD, f"{model}_H_{t['tid']}")
    if arm == "pmin":
        return os.path.join(PA.OD, f"pmin_s10_{t['tid']}")
    return os.path.join(PA.OD, f"{model}_{t['tid']}")


def boot(f):
    rng = np.random.default_rng(0)
    v = [f(rng.choice(len(PA.NEW), len(PA.NEW)), rng.choice(len(ks), len(ks))) for _ in range(PA.B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


res = {}
for model in ("pi05", "groot"):
    res[model] = {}
    for lay in ("familiar", "new"):
        D = {a: {c: load(stem(model, a, tid[(c, lay)]), tid[(c, lay)]) for c in PA.NEW + PA.KNOWN} for a in ("intact", "H", "pmin")}
        ti0, si0 = np.arange(len(PA.NEW)), np.arange(len(ks))

        def S(a, ti, si):
            return float(np.mean([D[a][PA.NEW[i]][0]["success"][si].mean() for i in ti]))

        def delta(a, b):
            e = S(a, ti0, si0) - S(b, ti0, si0)
            lo, hi = boot(lambda t, s: S(a, t, s) - S(b, t, s))
            return dict(delta=e, ci=[lo, hi], passes=bool(lo > 0.0 and e > 0.10))

        def excess(a, names, ti, si):
            out = []
            for i in ti:
                c = PA.NEW[i]
                ko, kt = PA.PAIR[c]
                out.append(sum(D[a][c][0][n][si] - 0.5 * (D[a][ko][0][n][si] + D[a][kt][0][n][si]) for n in names).mean())
            return float(np.mean(out))

        def est(a, names):
            e = excess(a, names, ti0, si0)
            ci = boot(lambda t, s: excess(a, names, t, s))
            return dict(E=e, ci=ci, breaks=bool(ci[0] > 0.0 and e > 0.10))

        r = dict(s={a: S(a, ti0, si0) for a in D}, H_minus_intact=delta("H", "intact"), H_minus_pmin=delta("H", "pmin"))
        r["E"] = {a: {g: est(a, n) for g, n in list(PA.GROUP.items()) + [(s, (s,)) for s in PA.STAGES[1:]]} for a in ("intact", "H")}
        w, x = r["E"]["H"]["within"]["breaks"], r["E"]["H"]["cross"]["breaks"]
        r["verdict"] = ("NO FIX" if not r["H_minus_intact"]["passes"] else
                        {(False, False): "FIXED", (True, False): "RAISED, WITHIN STILL BREAKS", (False, True): "RAISED, CROSS STILL BREAKS", (True, True): "RAISED, BOTH STILL BREAK"}[(w, x)])
        r["known"] = {a: float(np.mean([D[a][c][0]["success"].mean() for c in PA.KNOWN])) for a in D}
        r["known_cost"] = bool(r["known"]["intact"] - r["known"]["H"] > 0.05)
        r["per_task"] = {a: {c: {s: float(D[a][c][0][s].mean()) for s in PA.STAGES} for c in PA.NEW + PA.KNOWN} for a in D}
        h = np.stack([D["H"][c][1] for c in PA.NEW])
        hs = np.stack([D["H"][c][0]["success"] for c in PA.NEW])
        r["handed_share"] = float(h.mean())
        r["success_given_handed"] = float(hs[h > 0].mean()) if h.sum() else None
        res[model][lay] = r
        print(model, lay, r["verdict"], json.dumps({k: r[k] for k in ("s", "H_minus_intact", "H_minus_pmin", "known", "known_cost", "handed_share", "success_given_handed")}), flush=True)
        print(" E", json.dumps(r["E"]), flush=True)
        print(" per_task_H", json.dumps(r["per_task"]["H"]), flush=True)
json.dump(res, open(os.path.join(PA.OUT, "pp_hand_read.json"), "w", encoding="utf-8"), indent=1)

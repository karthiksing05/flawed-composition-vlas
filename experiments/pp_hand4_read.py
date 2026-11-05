"""Arm LR's read (docs/LOCATOR.md 11; m3_prereg pp_hand4).  CPU.  -> results/flawed/pp_hand4_read.json"""
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
H2D, H3D, H4D = (os.path.join(PA.OUT, x) for x in ("pp_hand2", "pp_hand3", "pp_hand4"))
NT, KT, SEEDS = ("N2", "N3", "N5"), ("K4", "K6"), (0, 1, 2)


def load(stem, t):
    R = json.load(open(stem + ".json", encoding="utf-8"))
    E = [e for v in R["tasks"].values() for e in v["episodes"]] if "tasks" in R else R["eps"]
    Tr = PA.load_track(stem + "_track.npz")
    assert len(E) == len(ks) == len(Tr) and all(e.get("error") is None for e in E), (stem, len(E), len(Tr))
    A = {s: np.zeros(len(ks)) for s in PA.STAGES}
    for j, (e, r) in enumerate(zip(E, Tr)):
        assert int(e["trial"]) == ks[j]
        d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
        A[PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])[0]][j] = 1.0
    return A


def pooled(d, pre, t):
    a = [load(os.path.join(d, f"{pre}{s}_{t['tid']}"), t) for s in SEEDS]
    return {s_: np.mean([x[s_] for x in a], 0) for s_ in PA.STAGES}


def get(model, arm, c, lay):
    t = tid[(c, lay)]
    if arm == "intact":
        return load(os.path.join(PA.OD, f"{model}_{t['tid']}"), t)
    if arm == "H1":
        return load(os.path.join(H2D, f"{model}_H1_{t['tid']}"), t)
    if arm == "LS":
        return pooled(H3D, f"{model}_LS", t)
    return pooled(H4D, f"{model}_LR", t)


def S(D, a, ti, si):
    return float(np.mean([D[a][NT[i]]["success"][si].mean() for i in ti]))


def boot(f):
    rng = np.random.default_rng(0)
    v = [f(rng.choice(len(NT), len(NT)), rng.choice(len(ks), len(ks))) for _ in range(PA.B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


ti0, si0 = np.arange(len(NT)), np.arange(len(ks))
res = {}
for model in ("pi05", "groot"):
    res[model] = {}
    for lay in ("familiar", "new"):
        D = {a: {c: get(model, a, c, lay) for c in NT + KT} for a in ("intact", "H1", "LS", "LR")}
        rd = S(D, "LR", ti0, si0) - S(D, "intact", ti0, si0)
        ci = boot(lambda t, s: S(D, "LR", t, s) - S(D, "intact", t, s))
        loss = S(D, "H1", ti0, si0) - S(D, "LR", ti0, si0)
        lci = boot(lambda t, s: S(D, "H1", t, s) - S(D, "LR", t, s))
        gain = boot(lambda t, s: S(D, "LR", t, s) - S(D, "LS", t, s))
        raises = bool(ci[0] > 0.0 and rd > 0.10)
        v = "NO FIX" if not raises else ("PERCEPTION HOLDS" if loss <= 0.10 else "PERCEPTION COSTS")
        k1, kl = float(np.mean([D["H1"][c]["success"].mean() for c in KT])), float(np.mean([D["LR"][c]["success"].mean() for c in KT]))
        err, handed, ok_handed = {}, [], []
        for c in NT:
            t = tid[(c, lay)]
            for s in SEEDS:
                st = os.path.join(H4D, f"{model}_LR{s}_{t['tid']}")
                z = np.load(st + "_track.npz")
                R = json.load(open(st + ".json", encoding="utf-8"))
                E = [e for x in R["tasks"].values() for e in x["episodes"]]
                for i in range(int(z["n"])):
                    k = int(z[f"e{i}_handed"])
                    handed.append(k >= 0)
                    if k < 0:
                        continue
                    ok_handed.append(bool(E[i]["success"]))
                    dv = np.linalg.norm(z[f"e{i}_est"] - z[f"e{i}_obj"][:, :2], axis=1)
                    hl = z[f"e{i}_held"][k:]
                    n = int(np.argmin(hl)) if not hl.all() else len(hl)
                    q = err.setdefault(t["object"], dict(at=[], carry=[]))
                    q["at"].append(float(dv[k])); q["carry"].append(float(dv[k:k + max(n, 1)].max()))
        err = {o: {f: dict(n=len(x), p50=float(np.median(x)), p95=float(np.percentile(x, 95))) for f, x in q.items()} for o, q in err.items()}
        res[model][lay] = dict(verdict=v, s_intact=S(D, "intact", ti0, si0), s_H1=S(D, "H1", ti0, si0), s_LS=S(D, "LS", ti0, si0), s_LR=S(D, "LR", ti0, si0),
                               raise_=dict(delta=rd, ci=ci, passes=raises), loss=dict(delta=loss, ci=lci), LR_minus_LS=dict(delta=S(D, "LR", ti0, si0) - S(D, "LS", ti0, si0), ci=gain),
                               known=dict(H1=k1, LS=float(np.mean([D["LS"][c]["success"].mean() for c in KT])), LR=kl, known_cost=bool(k1 - kl > 0.05)),
                               per_task={c: {a: float(D[a][c]["success"].mean()) for a in D} for c in NT + KT},
                               stages_LR={s_: float(np.mean([D["LR"][c][s_].mean() for c in NT])) for s_ in PA.STAGES},
                               handed=float(np.mean(handed)), success_when_handed=float(np.mean(ok_handed)) if ok_handed else None, estimate_error=err)
        print(model, lay, v, json.dumps(res[model][lay]), flush=True)
json.dump(res, open(os.path.join(PA.OUT, "pp_hand4_read.json"), "w", encoding="utf-8"), indent=1)

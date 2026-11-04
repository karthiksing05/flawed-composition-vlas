"""V1 read (docs/RIVAL_TESTS.md 1.3; m3_prereg pp_rival1).  CPU.  -> results/flawed/pp_v1_read.json"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
NEW = ["N1", "N2", "N3", "N4", "N5", "N6"]
n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
ks = n2["states"]["main"]
B, NEAR = 2000, 0.10


def load(arm, lay):
    """-> s, g, near_wrong, near_true, held [tasks, states], with errors counted."""
    A = {x: np.full((len(NEW), len(ks)), np.nan) for x in ("s", "g", "near_wrong", "near_true", "hold")}
    errs = 0
    meta = {}
    for i, c in enumerate(NEW):
        t = tid[(c, lay)]
        f = os.path.join(OUT, "pp_v1_roll", f"pi05_{arm}_{t['tid']}")
        R = json.load(open(f + ".json", encoding="utf-8"))
        Tr = json.load(open(f + "_track.json", encoding="utf-8"))
        meta[c] = dict(pre=Tr["pre"], post=Tr["post"], wrong=Tr["wrong"])
        E = [e for v in R["tasks"].values() for e in v["episodes"]]
        trk = {int(x["episode"]): x for x in Tr["episodes"] if x["episode"] is not None}
        for e in E:
            j = ks.index(int(e["trial"]))
            x = trk[int(e["trial"])]
            s = bool(e["success"])
            A["s"][i, j], A["g"][i, j] = float(s), float(s or e.get("bound") == t["object"])
            p = np.array(x["final"][t["object"]])[:2]
            A["near_wrong"][i, j] = float(np.linalg.norm(p - np.array(x["final"][Tr["wrong"]])[:2]) <= NEAR)
            A["near_true"][i, j] = float(np.linalg.norm(p - np.array(x["final"][t["target"]])[:2]) <= NEAR)
            A["hold"][i, j] = float(x["hold_step"] is not None)
            errs += e.get("error") is not None
    return A, errs, meta


def pr(A, ti, si):
    s, g = A["s"][np.ix_(ti, si)], A["g"][np.ix_(ti, si)]
    return float(np.nansum(s) / max(np.nansum(g), 1e-9))


def nr(A, ti, si):
    g = A["g"][np.ix_(ti, si)]
    return float(np.nansum(A["near_wrong"][np.ix_(ti, si)] * g) / max(np.nansum(g), 1e-9))


def boot(f):
    rng = np.random.default_rng(0)
    n, m = len(NEW), len(ks)
    v = np.array([f(rng.choice(n, n), rng.choice(m, m)) for _ in range(B)])
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


res = dict(layouts={})
ti0, si0 = np.arange(len(NEW)), np.arange(len(ks))
for lay in ("familiar", "new"):
    arms = ["intact", "A", "S"] + (["W"] if lay == "familiar" else [])
    D, Er, Me = {}, {}, {}
    for a in arms:
        D[a], Er[a], Me[a] = load(a, lay)
    r = dict(errors=Er, arms={}, per_task={}, prompts=Me)
    for a in arms:
        r["arms"][a] = dict(s=float(np.nanmean(D[a]["s"])), g=float(np.nanmean(D[a]["g"])), p=pr(D[a], ti0, si0), hold=float(np.nanmean(D[a]["hold"])))
        r["per_task"][a] = {c: dict(s=float(np.nanmean(D[a]["s"][i])), g=float(np.nanmean(D[a]["g"][i]))) for i, c in enumerate(NEW)}
    dp = pr(D["A"], ti0, si0) - pr(D["intact"], ti0, si0)
    cp = boot(lambda t, s: pr(D["A"], t, s) - pr(D["intact"], t, s))
    r["delta_p_A"] = [dp, *cp]
    ds = float(np.nanmean(D["S"]["s"]) - np.nanmean(D["intact"]["s"]))
    r["delta_s_S"] = [ds, *boot(lambda t, s: float(np.nanmean(D["S"]["s"][np.ix_(t, s)]) - np.nanmean(D["intact"]["s"][np.ix_(t, s)])))]
    da = float(np.nanmean(D["A"]["s"]) - np.nanmean(D["intact"]["s"]))
    r["delta_s_A"] = [da, *boot(lambda t, s: float(np.nanmean(D["A"]["s"][np.ix_(t, s)]) - np.nanmean(D["intact"]["s"][np.ix_(t, s)])))]
    rec = cp[0] > 0.0 and dp > 0.10
    r["RECOVERS"] = bool(rec)
    if lay == "familiar":
        st = nr(D["W"], ti0, si0) - nr(D["intact"], ti0, si0)
        cs = boot(lambda t, s: nr(D["W"], t, s) - nr(D["intact"], t, s))
        r["delta_steer"] = [st, *cs]
        ste = cs[0] > 0.0 and st > 0.10
        r["STEERS"] = bool(ste)
        r["verdict"] = {(True, True): "INSTRUCTION BINDING", (False, True): "HELD-OBJECT BINDING", (False, False): "NOT THE INSTRUCTION", (True, False): "RECOVERS WITHOUT STEERING"}[(bool(rec), bool(ste))]
    agree = []
    for i, c in enumerate(NEW):
        t = tid[(c, lay)]
        E = {int(e["trial"]): e for v in json.load(open(os.path.join(OUT, "n2_roll", f"pi05_main_{t['tid']}.json"), encoding="utf-8"))["tasks"].values() for e in v["episodes"]}
        agree += [float(bool(E[k]["success"]) == bool(D["intact"]["s"][i, j])) for j, k in enumerate(ks) if k in E]
    r["intact_agrees_with_recorded"] = float(np.mean(agree))
    res["layouts"][lay] = r
    print("V1", lay, json.dumps({k: v for k, v in r.items() if k != "prompts"}), flush=True)
json.dump(res, open(os.path.join(OUT, "pp_v1_read.json"), "w", encoding="utf-8"), indent=1)

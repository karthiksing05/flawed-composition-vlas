"""V1 offset read (docs/RIVAL_TESTS.md 5; m3_prereg pp_rival3).  CPU, no new episodes.  -> results/flawed/pp_v1_offset.json"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
ks = n2["states"]["main"]
kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"]
B, RAD, MIN_V, MIN_N = 2000, 0.15, 0.02, 5


def xy(v):
    return np.asarray(v, np.float64)[:2]


def load(arm, comp):
    """-> d [states, 2] from the reference target (W: its wrong target; else the true one), NaN where excluded; held_end [states]."""
    t = tid[(comp, "familiar")]
    f = os.path.join(OUT, "pp_v1_roll", f"pi05_{arm}_{t['tid']}")
    E = [e for v in json.load(open(f + ".json", encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
    Tr = json.load(open(f + "_track.json", encoding="utf-8"))
    trk = {int(x["episode"]): x for x in Tr["episodes"]}
    ref = Tr["wrong"] if arm == "W" else t["target"]
    D, H = np.full((len(ks), 2), np.nan), np.zeros(len(ks))
    for e in E:
        j, x = ks.index(int(e["trial"])), trk[int(e["trial"])]
        if not (bool(e["success"]) or e.get("bound") == t["object"]):
            continue
        d = xy(x["final"][t["object"]]) - xy(x["final"][ref])
        if np.linalg.norm(d) <= RAD:
            D[j], H[j] = d, float(x["held_end"])
    return D, H, ref


CHEESE = {"N1": "K1", "N2": "K2", "N3": "K3"}
P = {c: xy(kp[k]["release_off"]) - xy(kp["K6"]["grasp_off"]) for c, k in CHEESE.items()}
Q = xy(kp["K6"]["grasp_off"]) - xy(kp["K4"]["grasp_off"])
GROUPS = [(("N2", "N3"), ("N5", "N6")), (("N1",), ("N4",))]
rot = lambda v: np.array([-v[1], v[0]])                                     # noqa: E731


def beta1(D, si, turn=False):
    num, den = 0.0, 0.0
    for c, p in P.items():
        v = rot(p) if turn else p
        d = D[c][si]
        ok = np.isfinite(d[:, 0])
        num += float((d[ok] @ v).sum())
        den += float(ok.sum() * (v @ v))
    return num / den if den > 0 else np.nan


def beta2(W, si, turn=False):
    cs = []
    for cheese, bottle in GROUPS:
        a = np.concatenate([W[c][si] for c in cheese])
        b = np.concatenate([W[c][si] for c in bottle])
        a, b = a[np.isfinite(a[:, 0])], b[np.isfinite(b[:, 0])]
        if not len(a) or not len(b):
            return np.nan
        cs.append(b.mean(0) - a.mean(0))
    v = rot(Q) if turn else Q
    return float(np.mean(cs, 0) @ v / (v @ v))


def boot(f):
    rng = np.random.default_rng(0)
    v = np.array([f(rng.choice(len(ks), len(ks))) for _ in range(B)])
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def call(ci, testable):
    if not testable:
        return "unclear"
    return "yes" if ci[0] > 0.5 else ("no" if ci[1] < 0.5 else "unclear")


S0 = np.arange(len(ks))
A = {c: load("A", c)[0] for c in CHEESE}
I = {c: load("intact", c)[0] for c in CHEESE}
W, WH = {}, {}
for c in ("N1", "N2", "N3", "N4", "N5", "N6"):
    W[c], WH[c], _ = load("W", c)
n = lambda D: int(np.isfinite(D[:, 0]).sum())                               # noqa: E731
res = dict(p={c: v.tolist() for c, v in P.items()}, q=Q.tolist(), p_norm={c: float(np.linalg.norm(v)) for c, v in P.items()}, q_norm=float(np.linalg.norm(Q)))
res["n"] = dict(A={c: n(A[c]) for c in A}, intact={c: n(I[c]) for c in I}, W={c: n(W[c]) for c in W})
res["mean_d"] = dict(A={c: np.nanmean(A[c], 0).tolist() for c in A}, intact={c: np.nanmean(I[c], 0).tolist() for c in I}, W={c: np.nanmean(W[c], 0).tolist() for c in W})
res["held_end_W"] = {c: float(WH[c][np.isfinite(W[c][:, 0])].mean()) for c in W}
b1, c1 = beta1(A, S0), boot(lambda s: beta1(A, s))
b2, c2 = beta2(W, S0), boot(lambda s: beta2(W, s))
t1 = min(res["p_norm"].values()) >= MIN_V and min(res["n"]["A"].values()) >= MIN_N
t2 = res["q_norm"] >= MIN_V and min(res["n"]["W"].values()) >= MIN_N
res["beta1"], res["beta2"] = [b1, *c1], [b2, *c2]
res["call1"], res["call2"] = call(c1, t1), call(c2, t2)
res["verdict"] = {("yes", "yes"): "OFFSET CARRIES", ("no", "no"): "NOT THE OFFSET"}.get((res["call1"], res["call2"]), "UNRESOLVED")
res["placebo_beta1_rot90"] = [beta1(A, S0, True), *boot(lambda s: beta1(A, s, True))]
res["placebo_beta2_rot90"] = [beta2(W, S0, True), *boot(lambda s: beta2(W, s, True))]
res["beta1_intact"] = [beta1(I, S0), *boot(lambda s: beta1(I, s))]
print(json.dumps(res), flush=True)
json.dump(res, open(os.path.join(OUT, "pp_v1_offset.json"), "w", encoding="utf-8"), indent=1)

"""E0 (docs/THEORY_COMPOSITION.md 2, partner counts): does an object with one training target fail new pairings more than one with two?  Read on existing tracks; no rollouts.

    python experiments/sc_e0.py read        -> results/flawed/sc_e0_read.json

Groups are defined by the task's object in n2_tasks.json (cream cheese vs wine bottle), among compounds N1-N6; N3 is the cheese task whose target is the cabinet.
Registered arms: GR00T frozen (n2_roll/groot_main), GR00T at 2 and 10 actions per query (sc_t1/groot_r2, groot_r10), pi0.5 at 2 and 10 (sc_t1/pi05_r2, pi05_r10).
pi0.5 intact (n2_roll/pi05_main) was seen before registration and is descriptive only.
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
OD = os.path.join(OUT, "sc_t1")
NEW = [f"N{i}" for i in range(1, 7)]
LAYOUTS = ("familiar", "new")
B = 2000
ARMS = {"groot_frozen": ("groot", 5), "groot_r2": ("groot", 2), "groot_r10": ("groot", 10), "pi05_r2": ("pi05", 2), "pi05_r10": ("pi05", 10)}
DESCRIPTIVE = {"pi05_intact": ("pi05", 5)}


def read():
    meta = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    ks = meta["states"]["main"]
    T = {(t["compound"], t["layout"]): t for t in meta["tasks"]}

    cheese = [c for c in NEW if "cheese" in T[(c, "familiar")]["object"]]
    bottle = [c for c in NEW if "bottle" in T[(c, "familiar")]["object"]]
    assert len(cheese) == 3 and len(bottle) == 3 and not set(cheese) & set(bottle), (cheese, bottle)
    n3 = [c for c in cheese if "cabinet" in T[(c, "familiar")]["target"]]
    assert len(n3) == 1, n3
    n3 = n3[0]
    others = [c for c in cheese if c != n3]
    order = cheese + bottle                                                  # rows: cheese x3, then bottle x3

    def eps(m, r, c, l):
        t = T[(c, l)]["tid"]
        p = os.path.join(OUT, "n2_roll", f"{m}_main_{t}.json") if r == 5 else os.path.join(OD, f"{m}_r{r}_{t}.json")
        if os.path.isdir(p):                                                # t1_roll.sh's mv landed inside the client's video folder
            p = os.path.join(p, os.path.basename(p) + ".tmp")
        by = {int(e.get("trial", e.get("init_state", -1))): e for e in json.load(open(p, encoding="utf-8"))["tasks"][str(t)]["episodes"]}
        return [by[k] for k in ks], T[(c, l)]["object"]

    succ = lambda e, o: float(bool(e["success"]))                            # noqa: E731
    bound = lambda e, o: float(bool(e["success"]) or e.get("bound") == o)     # noqa: E731

    def arr(m, r, l, f):                                                    # [6 tasks (cheese, bottle), states]
        out = []
        for c in order:
            E, o = eps(m, r, c, l)
            out.append([f(e, o) for e in E])
        return np.array(out)

    def gap(x):
        return float(x[3:].mean() - x[:3].mean())

    def gap_ci(x):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            s = rng.choice(x.shape[1], x.shape[1])                          # states jointly across tasks; tasks fixed
            v.append(gap(x[:, s]))
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def verdict(x):
        lo, hi = gap_ci(x)
        return "HOLDS" if lo > 0 else ("REVERSED" if hi < 0 else "UNCLEAR")

    def describe(m, r):
        d = {}
        for l in LAYOUTS:
            S, G = arr(m, r, l, succ), arr(m, r, l, bound)
            per = {}
            for i, c in enumerate(order):
                g_ = float(G[i].mean())
                per[c] = dict(success=float(S[i].mean()), grasped_right=g_, place_given_grasp=(float(S[i].mean() / g_) if g_ > 0 else None))
            d[l] = dict(per_task=per, cheese=float(S[:3].mean()), bottle=float(S[3:].mean()), gap=gap(S), gap_ci=gap_ci(S),
                        n3_minus_mean_n1_n2=float(S[order.index(n3)].mean() - np.mean([S[order.index(c)].mean() for c in others])))
        return d

    res = dict(groups=dict(cheese=cheese, bottle=bottle, n3=n3), arms={}, descriptive={}, verdicts={})
    for k, (m, r) in ARMS.items():
        res["arms"][k] = describe(m, r)
        res["verdicts"][k] = verdict(arr(m, r, "familiar", succ))
    for k, (m, r) in DESCRIPTIVE.items():
        res["descriptive"][k] = describe(m, r)
        res["descriptive"][k]["verdict_not_registered"] = verdict(arr(m, r, "familiar", succ))

    pooled = np.mean([arr("groot", r, "familiar", succ) for r in (5, 2, 10)], 0)   # the three GR00T files averaged
    res["groot_pooled"] = dict(gap=gap(pooled), gap_ci=gap_ci(pooled), verdict=verdict(pooled),
                               cheese=float(pooled[:3].mean()), bottle=float(pooled[3:].mean()))
    vs = res["verdicts"]
    if res["groot_pooled"]["verdict"] == "REVERSED":
        res["overall"] = "REFUTED"
    elif res["groot_pooled"]["verdict"] == "HOLDS" and "REVERSED" not in vs.values():
        res["overall"] = "SUPPORTED"
    else:
        res["overall"] = "UNRESOLVED"
    res["abs_gap_groot_pooled_lt_pi05"] = {k: bool(abs(res["groot_pooled"]["gap"]) < abs(res["arms"][k]["familiar"]["gap"])) for k in ("pi05_r2", "pi05_r10")}
    print("[e0]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_e0_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read",))
    ap.parse_args()
    if os.path.exists(os.path.join(OUT, "sc_e0_read.json")):
        raise SystemExit("sc_e0_read.json exists")
    read()

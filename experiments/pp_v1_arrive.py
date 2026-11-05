"""V1 arrival read (docs/RIVAL_TESTS.md 4; m3_prereg pp_rival2).  CPU, no new episodes.  -> results/flawed/pp_v1_arrive.json"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
NEW = ["N1", "N2", "N3", "N4", "N5", "N6"]
n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
ks = n2["states"]["main"]
B, R, RS = 2000, 0.10, 0.03


def load(arm, lay):
    A = {x: np.zeros((len(NEW), len(ks))) for x in ("fail", "near", "seat", "held")}
    for i, c in enumerate(NEW):
        t = tid[(c, lay)]
        f = os.path.join(OUT, "pp_v1_roll", f"pi05_{arm}_{t['tid']}")
        E = [e for v in json.load(open(f + ".json", encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
        trk = {int(x["episode"]): x for x in json.load(open(f + "_track.json", encoding="utf-8"))["episodes"]}
        for e in E:
            j = ks.index(int(e["trial"]))
            x = trk[int(e["trial"])]
            if bool(e["success"]) or e.get("bound") != t["object"]:
                continue
            d = float(np.linalg.norm(np.array(x["final"][t["object"]])[:2] - np.array(x["final"][t["target"]])[:2]))
            A["fail"][i, j], A["near"][i, j], A["seat"][i, j], A["held"][i, j] = 1.0, float(d <= R), float(d <= RS), float(x["held_end"])
    return A


def share(A, k, ti, si):
    return float(A[k][np.ix_(ti, si)].sum() / max(A["fail"][np.ix_(ti, si)].sum(), 1e-9))


def ci(A, k):
    rng = np.random.default_rng(0)
    v = []
    for _ in range(B):
        ti, si = rng.choice(len(NEW), len(NEW)), rng.choice(len(ks), len(ks))
        if A["fail"][np.ix_(ti, si)].sum() > 0:
            v.append(share(A, k, ti, si))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


T0, S0 = np.arange(len(NEW)), np.arange(len(ks))
res = {}
for lay in ("familiar", "new"):
    for arm in ("intact", "A"):
        A = load(arm, lay)
        r = dict(n_fail=int(A["fail"].sum()))
        for k in ("near", "seat", "held"):
            r[k] = [share(A, k, T0, S0), *ci(A, k)]
        r["per_task"] = {c: dict(n_fail=int(A["fail"][i].sum()), near=int(A["near"][i].sum()), seat=int(A["seat"][i].sum()), held=int(A["held"][i].sum())) for i, c in enumerate(NEW)}
        res[f"{arm}|{lay}"] = r
        print("ARRIVE", arm, lay, json.dumps(r), flush=True)
lo, hi = res["intact|familiar"]["near"][1:]
res["verdict"] = "ARRIVES, FAILS TO SEAT" if lo > 0.5 else ("DOES NOT ARRIVE" if hi < 0.5 else "MIXED")
print("VERDICT", res["verdict"], flush=True)
json.dump(res, open(os.path.join(OUT, "pp_v1_arrive.json"), "w", encoding="utf-8"), indent=1)

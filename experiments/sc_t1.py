"""T1 (docs/SCAFFOLD_RESULTS.md 32; m3_prereg sc_t1): actions per query on frozen VLAs.  Rollouts through experiments/fc_n2_client.py (t1_roll.sh); this file holds the array table and the read.

    python experiments/sc_t1.py job --index 0       prints MODEL REPLAN TIDS for one array task
    python experiments/sc_t1.py read                -> results/flawed/sc_t1_read.json
"""
import argparse
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
OD = os.path.join(OUT, "sc_t1")
MODELS, REPLANS = ("pi05", "groot"), (2, 10)
KNOWN, NEW = [f"K{i}" for i in range(1, 7)], [f"N{i}" for i in range(1, 7)]
B, MARGIN = 2000, 0.10


def n2():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def tid(meta, c, l):
    return next(t["tid"] for t in meta["tasks"] if t["compound"] == c and t["layout"] == l)


def jobs():
    meta = n2()
    groups = [[tid(meta, c, l) for c in cs] for cs, l in itertools.product((KNOWN, NEW), ("familiar", "new"))]
    return [(m, r, g) for m, r, g in itertools.product(MODELS, REPLANS, groups)]


def read():
    meta = n2()
    ks = meta["states"]["main"]

    def eps(m, r, c, l):
        t = tid(meta, c, l)
        p = os.path.join(OUT, "n2_roll", f"{m}_main_{t}.json") if r == 5 else os.path.join(OD, f"{m}_r{r}_{t}.json")
        if os.path.isdir(p):                                                # t1_roll.sh's mv landed inside the client's video folder, named from --out
            p = os.path.join(p, os.path.basename(p) + ".tmp")
        by = {int(e["trial"]): e for e in json.load(open(p, encoding="utf-8"))["tasks"][str(t)]["episodes"]}
        return [by[k] for k in ks], meta["tasks"][t]["object"]

    def arr(m, r, cs, l, f):
        return np.array([[f(e, eps(m, r, c, l)[1]) for e in eps(m, r, c, l)[0]] for c in cs])

    succ = lambda e, o: float(bool(e["success"]))                            # noqa: E731
    bound = lambda e, o: float(bool(e["success"]) or e.get("bound") == o)     # noqa: E731

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[1], a.shape[1])
            v.append(float((a[t][:, s] - b[t][:, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    res = dict(success={}, grasped_right={}, labels={})
    for m, r in itertools.product(MODELS, REPLANS):
        for g, cs in (("known", KNOWN), ("new_pairings", NEW)):
            for l in ("familiar", "new"):
                k = f"{m}|r{r}|{g}|{l}"
                res["success"][k] = diff(arr(m, r, cs, l, succ), arr(m, 5, cs, l, succ))
                res["grasped_right"][k] = diff(arr(m, r, cs, l, bound), arr(m, 5, cs, l, bound))
        x = res["success"][f"{m}|r{r}|new_pairings|familiar"]
        res["labels"][f"{m}|r{r}"] = "GAIN" if x["ci"][0] > MARGIN else ("LOSS" if x["ci"][1] < -MARGIN else "NO CLEAR CHANGE")
    res["overall"] = "CHUNK-SENSITIVE" if "GAIN" in res["labels"].values() else "NO GAIN FROM CHUNKING"
    print("[t1]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_t1_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("job", "read"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "job":
        m_, r_, g_ = jobs()[a.index]
        print(m_, r_, "+".join(str(x) for x in g_))
    else:
        read()

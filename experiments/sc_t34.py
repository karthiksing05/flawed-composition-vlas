"""T34 (docs/SCAFFOLD_RESULTS.md 34; m3_prereg sc_t34): the single-GPU ladder.  Rollouts (R0, R1, later R2) sit in results/flawed/sc_t34/{rung}_main_{tid}.json, written by experiments/fc_n2_client.py.  This file holds the read.

    python experiments/sc_t34.py read [--rungs r0 r1] [--tag SUFFIX]      -> results/flawed/sc_t34_read{SUFFIX}.json
"""
import argparse
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
OD = os.path.join(OUT, "sc_t34")
KNOWN, NEW = [f"K{i}" for i in range(1, 7)], [f"N{i}" for i in range(1, 7)]
CHEESE, BOTTLE = ["N1", "N2", "N3"], ["N4", "N5", "N6"]
EXTRA = ["K7", "K8", "N7", "N8"]
LAYOUTS = ("familiar", "new")
B, MARGIN, FIT = 2000, 0.10, 0.80


def n2():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def read():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--rungs", nargs="+", default=["r0", "r1"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--dir", default="sc_t34", help="results/flawed/<dir> holds the rollouts; sc_t37 for the matched ladder (SCAFFOLD_RESULTS 37)")
    ap.add_argument("--primary", default="", help="'ra:rb': the pair whose composition label is the overall label (default: second rung minus first)")
    a = ap.parse_args()
    rungs = a.rungs
    assert len(rungs) >= 2 and len(set(rungs)) == len(rungs), rungs
    OD = os.path.join(OUT, a.dir)                                            # noqa: N806  (shadows the module default on purpose)
    outp = os.path.join(OUT, f"{a.dir}_read{('_' + a.tag.lstrip('_')) if a.tag else ''}.json")
    if os.path.exists(outp):
        raise SystemExit(f"{outp} exists; pass --tag with a new suffix")

    meta = n2()
    ks = meta["states"]["main"]

    def tid(c, l):
        return next(t["tid"] for t in meta["tasks"] if t["compound"] == c and t["layout"] == l)

    def eps(path, c, l):
        t = tid(c, l)
        if os.path.isdir(path):                                              # the client's video-folder quirk (sc_t1.read): the file sits inside, named from --out
            path = os.path.join(path, os.path.basename(path) + ".tmp")
        E = json.load(open(path, encoding="utf-8"))["tasks"][str(t)]["episodes"]
        by = {int(e.get("trial", e.get("init_state", -1))): e for e in E}
        return [by[k] for k in ks], meta["tasks"][t]["object"]

    succ = lambda e, o: float(bool(e["success"]))                            # noqa: E731
    bound = lambda e, o: float(bool(e["success"]) or e.get("bound") == o)     # noqa: E731

    def P(kind, name, c, l):
        t = tid(c, l)
        d = {"rung": os.path.join(OD, f"{name}_main_{t}.json"),
             "frozen": os.path.join(OUT, "n2_roll", f"pi05_main_{t}.json"),
             "t8b": os.path.join(OUT, "sc_t8v", f"pi05g8_main_{t}.json"),
             "t16s": os.path.join(OUT, "sc_t16s", f"s_main_{t}.json")}
        return d[kind]

    def arr(kind, name, cs, l, f):
        out = []
        for c in cs:
            E, o = eps(P(kind, name, c, l), c, l)
            out.append([f(e, o) for e in E])
        return np.array(out)

    def diff(x, y):                                                         # [tasks, states], paired; sc_t1's scheme
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(x.shape[0], x.shape[0])
            s = rng.choice(x.shape[1], x.shape[1])
            v.append(float((x[t][:, s] - y[t][:, s]).mean()))
        return dict(a=float(x.mean()), b=float(y.mean()), delta=float((x - y).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    def verdicts(ra, rb):
        comp = diff(arr("rung", ra, NEW, "familiar", succ), arr("rung", rb, NEW, "familiar", succ))
        cost = diff(arr("rung", ra, KNOWN, "familiar", succ), arr("rung", rb, KNOWN, "familiar", succ))
        return dict(composition_diff=comp, cost_diff=cost,
                    composition="STRUCTURE WINS" if comp["ci"][0] > MARGIN else ("STRUCTURE LOSES" if comp["ci"][1] < -MARGIN else "NO CLEAR DIFFERENCE"),
                    cost="NO COST" if cost["ci"][0] >= -MARGIN else "COST")

    res = dict(rungs=rungs, note="a minus b; composition on N1-N6 familiar, cost on K1-K6 familiar", success={}, grasped_right={}, place_given_grasp={}, per_task={}, vs_frozen={}, vs_t8b={}, vs_t16s={}, verdicts={})

    k0 = float(arr("rung", rungs[0], KNOWN, "familiar", succ).mean())
    res["fit_gate"] = dict(rung=rungs[0], known_familiar=k0, threshold=FIT, underfit=bool(k0 < FIT))

    for r in rungs:
        for l in LAYOUTS:
            for g, cs in (("known", KNOWN), ("new_pairings", NEW), ("cheese", CHEESE), ("bottle", BOTTLE)):
                res["success"][f"{r}|{g}|{l}"] = float(arr("rung", r, cs, l, succ).mean())
                res["grasped_right"][f"{r}|{g}|{l}"] = float(arr("rung", r, cs, l, bound).mean())
            for c in KNOWN + NEW + EXTRA:
                s_, b_ = arr("rung", r, [c], l, succ).mean(), arr("rung", r, [c], l, bound).mean()
                res["per_task"][f"{r}|{c}|{l}"] = dict(success=float(s_), grasped_right=float(b_), place_given_grasp=(float(s_ / b_) if b_ > 0 else None))
            gp = arr("rung", r, NEW, l, succ).mean() / max(arr("rung", r, NEW, l, bound).mean(), 1e-9)
            res["place_given_grasp"][f"{r}|new_pairings|{l}"] = float(gp)

    # each rung against frozen pi05_libero's recorded episodes, and against T8b and T16s when their files exist
    def have(kind, l, cs):
        return all(os.path.exists(P(kind, "", c, l)) for c in cs)

    refs = (("frozen", "vs_frozen"), ("t8b", "vs_t8b"), ("t16s", "vs_t16s"))
    for r in rungs:
        for l in LAYOUTS:
            for g, cs in (("known", KNOWN), ("new_pairings", NEW)):
                for kind, key in refs:
                    if have(kind, l, KNOWN + NEW):
                        res[key][f"{r}|{g}|{l}"] = diff(arr("rung", r, cs, l, succ), arr(kind, "", cs, l, succ))
    for kind, key in refs[1:]:
        if not res[key]:
            res[key] = "files absent; not read"

    # verdicts: R1 - R0 first; with more than two rungs, adjacent pairs and each rung against the first
    pairs = [(rungs[1], rungs[0])]
    pairs += [(rungs[i + 1], rungs[i]) for i in range(1, len(rungs) - 1)]
    pairs += [(r, rungs[0]) for r in rungs[2:]]
    pairs += [(rungs[i + 1], rungs[i]) for i in range(0, 1) if (rungs[i + 1], rungs[i]) not in pairs]
    for ra, rb in dict.fromkeys(pairs):
        res["verdicts"][f"{ra} minus {rb}"] = verdicts(ra, rb)
    pa, pb = a.primary.split(":") if a.primary else (rungs[1], rungs[0])
    if f"{pa} minus {pb}" not in res["verdicts"]:
        res["verdicts"][f"{pa} minus {pb}"] = verdicts(pa, pb)
    first = res["verdicts"][f"{pa} minus {pb}"]
    res["primary"] = f"{pa} minus {pb}"
    res["composition"], res["cost"] = first["composition"], first["cost"]
    res["overall"] = "UNDERFIT" if res["fit_gate"]["underfit"] else first["composition"]
    print("[t34]", json.dumps(res), flush=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap0 = argparse.ArgumentParser()
    ap0.add_argument("what", choices=("read",))
    ap0.parse_known_args()
    read()

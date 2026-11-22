"""RAA-F dev read (docs/notes/raaf_plan.md 6.2, 6.3; selection only, seen dev states).  Numpy only; run on a compute node (sbatch).

A NEW reader (raa_read.py, raasl_read.py, wa_read.py are imported, not edited).  Needed because raa_read.grid reads every arm's "rx" files from
results/flawed/raa, while the RAA-F rollouts live in their own directories.

Phase A (SR 73's 60 states: dev trials 4-23 from <stem>_main_<tid>.json, new trials 24-63 from <stem>_rx_<tid>.json; K1-K6, N1-N6):
    paired bootstrap B = 2000, rng 0 (raa_read.diff: tasks and states resampled), route - RAA-sL (registered comparator, results/flawed/raa raasl_*)
    and route - A0 (servo-matched comparator; B - A0 and A - A0).
Phase B (rg_comp rgc main states, wa_s.py / wa_sb.py full arm, <dir>/full_<tid>.json): F7 pool (tids 0-3), F6_b (tid 8), F6_a (9), K1t (10), K2t (7),
    unique-name controls (4-6); route - RAA-sL (results/flawed/wasl) and route - A0 by wa_read.bootm (states resampled, F7 pool: the four
    cells' states resampled jointly).  wa_read.read is also run on each route's directory (dissociation labels, descriptive).

    python experiments/raaf_dev_read.py read --routes A,B,A0 --out results/flawed/raaf/raaf_dev_read.json
Writes once (never overwrites).  Route directories: results/flawed/raaf/dev_<route> (Phase A: raaf<route>_{main,rx}_<tid>.json; Phase B: dev_<route>/wa/).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raa_read as RR                                                        # noqa: E402
import wa_read as WR                                                         # noqa: E402

OUT = RR.OUT
RAAF = os.path.join(OUT, "raaf")
TRIALS = RR.DEV + RR.NEWS


def src(arm):
    """-> (main dir, rx dir, stem) of an arm."""
    if arm == "raasl":
        return RR.RD, RR.RD, "raasl"
    d = os.path.join(RAAF, "dev_%s" % arm)
    return d, d, "raaf%s" % arm


def grid(meta, arm, comps, what):
    dm, dr, stem = src(arm)
    out = []
    for c in comps:
        tid = next(t["tid"] for t in meta["tasks"] if t["compound"] == c and t["layout"] == "familiar")
        obj = next(t["object"] for t in meta["tasks"] if t["tid"] == tid)
        by = dict(RR.episodes(os.path.join(dm, "%s_main_%d.json" % (stem, tid)), tid))
        by.update(RR.episodes(os.path.join(dr, "%s_rx_%d.json" % (stem, tid)), tid))
        out.append([float(bool(by[k]["success"])) if what == "succ" else float(bool(by[k]["success"]) or by[k].get("bound") == obj) for k in TRIALS])
    return np.array(out)


def phaseA(arms, pairs):
    meta = RR.n2()
    G = {}
    for arm in arms:
        for grp, comps in (("K", RR.KNOWN), ("N", RR.NEW), ("cheese", RR.CHEESE), ("bottle", RR.BOTTLE)):
            G[(arm, grp)] = (grid(meta, arm, comps, "succ"), grid(meta, arm, comps, "bound"))
    res = dict(success={}, per_task={}, contrasts={})
    for arm in arms:
        for grp in ("K", "N", "cheese", "bottle"):
            res["success"]["%s|%s" % (arm, grp)] = float(G[(arm, grp)][0].mean())
        for c in RR.KNOWN + RR.NEW:
            res["per_task"]["%s|%s" % (arm, c)] = float(grid(meta, arm, [c], "succ").mean())
    for a, b in pairs:
        for grp in ("N", "K", "cheese", "bottle"):
            d = RR.diff(G[(a, grp)], G[(b, grp)], "mean")
            row = dict(d)
            if grp == "K":
                row["noninf_-.05"] = RR.label(d, 0.05, "noninf")
            if grp == "N":
                row["support"] = "SUPPORTED" if d["ci"][0] > 0 else ("REVERSED" if d["ci"][1] < 0 else "NOT SUPPORTED")
            res["contrasts"]["%s - %s | %s" % (a, b, grp)] = row
    return res


def wdir(arm):
    return os.path.join(OUT, "wasl") if arm == "raasl" else os.path.join(RAAF, "dev_%s" % arm, "wa")


def phaseB(arms, pairs):
    meta = json.load(open(WR.META, encoding="utf-8"))
    rows7, rows6 = meta["states"]["f7"]["main"], meta["states"]["f6"]["main"]
    rows_of = lambda t: rows7 if t <= 7 else rows6                           # noqa: E731
    X = {}
    for arm in arms:
        for t in WR.KEYS:
            d, _ = WR.load(os.path.join(wdir(arm), "full_%d.json" % t), rows_of(t))
            X[(arm, t)] = None if d is None or len(d) < len(rows_of(t)) else np.array([d[k] for k in rows_of(t)], float)
    cells = {"F7pool": (0, 1, 2, 3), "F6_b": (8,), "F6_a": (9,), "K1t": (10,), "K2t": (7,), "uniq(4-6)": (4, 5, 6), "K(K1t+K2t)": (7, 10)}

    def fn(arm, tids):
        xs = [X[(arm, t)] for t in tids]
        if any(x is None for x in xs):
            return None
        return lambda I, xs=xs: float(np.mean([x[I[0]].mean() for x in xs]))
    res = dict(success={}, ci={}, contrasts={})
    for arm in arms:
        for name, tids in cells.items():
            f = fn(arm, tids)
            if f is not None:
                n = len(rows_of(tids[0]))
                res["success"]["%s|%s" % (arm, name)] = f([np.arange(n)])
                res["ci"]["%s|%s" % (arm, name)] = WR.bootm(f, [n])
    for a, b in pairs:
        for name in ("F7pool", "F6_b", "F6_a"):
            fa, fb = fn(a, cells[name]), fn(b, cells[name])
            if fa is None or fb is None:
                res["contrasts"]["%s - %s | %s" % (a, b, name)] = "MISSING"
                continue
            n = len(rows_of(cells[name][0]))
            ci = WR.bootm(lambda I, fa=fa, fb=fb: fa(I) - fb(I), [n])
            res["contrasts"]["%s - %s | %s" % (a, b, name)] = dict(delta=ci[0], ci=ci[1:], label=WR.lab_support(ci))
    return res


def read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: %s" % out
    routes = a.routes.replace("+", ",").split(",")
    arms = ["raasl"] + routes
    pairs = [(r, "raasl") for r in routes] + ([("B", "A0"), ("A", "A0")] if "A0" in routes else [])
    pairs += [("Aplus", "A"), ("Aplus", "A0"), ("Actrl", "raasl"), ("Actrl", "Aplus"),
              ("Aplus0", "raasl"), ("Aplus0", "A"), ("Aplus0", "A0"), ("Aplus0", "Aplus")]
    pairs = [p for p in pairs if p[0] in arms and p[1] in arms]
    res = dict(B=RR.B, routes=routes, phaseA=phaseA(arms, pairs), phaseB=phaseB(arms, pairs))
    print("[raaf-dev-read]", json.dumps(res), flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for r in routes:                                                         # dissociation labels per route (wa_read, descriptive)
        o = out.replace(".json", "_wa_%s.json" % r)
        try:
            WR.read(argparse.Namespace(phase="main", wa=wdir(r), out=o))
        except Exception as e:                                               # noqa: BLE001  (wa_read's print loop on unrolled noM arms, as SR 80)
            print("[raaf-dev-read] wa_read %s: %r (json written: %s)" % (r, e, os.path.exists(o)), flush=True)
    print("[raaf-dev-read] -> %s" % out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read",))
    ap.add_argument("--routes", default="A,B,A0")
    ap.add_argument("--out", required=True)
    read(ap.parse_args())

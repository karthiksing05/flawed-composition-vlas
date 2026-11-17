"""RAA-min read (docs/SCAFFOLD_RESULTS.md 73; m3_prereg raa).  Numpy only (openpi venv).

States: the dev states R0 / R2 / R2l were read on (n2_states, harness trials 4-23, results/flawed/sc_t37) plus 40 new familiar-layout
states (fc_n2_rx.py, harness trials 24-63, k = 540-579).  Every arm is read on the same 60 states; R0, R2, R2l on the 40 new states are
rolled in this test (results/flawed/raa/<name>_rx_<tid>.json), on the dev states they are read from sc_t37 unchanged.  Tasks: K1-K6
(known) and N1-N6 (new pairings), familiar layout.

    python experiments/raa_read.py read [--tag T]       -> results/flawed/raa_read[_T].json (refuses to overwrite)
    python experiments/raa_read.py selftest             reproduces sc_t37_read.json's R2 - R0 on the dev states (+.400 [+.167, +.625])

Estimator (sc_t34's): paired bootstrap, B = 2000, rng 0; each draw resamples tasks and states independently and averages the paired
differences over the grid.  Placed-given-right-grasp is the ratio of the arm's success to its right-grasp rate over the resampled grid.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
RD = os.path.join(OUT, "raa")
KNOWN, NEW = ["K%d" % i for i in range(1, 7)], ["N%d" % i for i in range(1, 7)]
CHEESE, BOTTLE = ["N1", "N2", "N3"], ["N4", "N5", "N6"]
DEV, NEWS = list(range(4, 24)), list(range(24, 64))
B, MARGIN_K, MARGIN_RG, FIT = 2000, 0.05, 0.10, 0.80
# arm -> (directory of the dev-state files, file stem); every arm's new-state files are RD/<stem>_rx_<tid>.json
ARMS = {"r0": (os.path.join(OUT, "sc_t37"), "r0"), "r2": (os.path.join(OUT, "sc_t37"), "r2"), "r2l": (os.path.join(OUT, "sc_t37"), "r2l"),
        "rrel": (RD, "rrel"), "raa": (RD, "raa"), "rgp": (RD, "rgp")}


def n2():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def episodes(path, tid):
    if os.path.isdir(path):                                                  # the client's video-folder quirk (sc_t34)
        path = os.path.join(path, os.path.basename(path) + ".tmp")
    E = json.load(open(path, encoding="utf-8"))["tasks"][str(tid)]["episodes"]
    return {int(e.get("trial", e.get("init_state", -1))): e for e in E}


def grid(meta, arm, comps, trials, what):
    """-> [tasks, states] of success (what 'succ') or right-grasp (what 'bound'): the arm's episodes on `trials`."""
    d, stem = ARMS[arm]
    out = []
    for c in comps:
        tid = next(t["tid"] for t in meta["tasks"] if t["compound"] == c and t["layout"] == "familiar")
        obj = next(t["object"] for t in meta["tasks"] if t["tid"] == tid)
        by = {}
        dev = [k for k in trials if k in DEV]
        new = [k for k in trials if k in NEWS]
        if dev:
            by.update(episodes(os.path.join(d, "%s_main_%d.json" % (stem, tid)), tid))
        if new:
            by.update(episodes(os.path.join(RD, "%s_rx_%d.json" % (stem, tid)), tid))
        row = []
        for k in trials:
            e = by[k]
            row.append(float(bool(e["success"])) if what == "succ" else float(bool(e["success"]) or e.get("bound") == obj))
        out.append(row)
    return np.array(out)



def diff(xs, ys, kind="mean"):
    """xs, ys = (succ, bound) grids [tasks, states] of arm a and arm b.  kind 'mean' on succ, 'pgg' the placed-given-right-grasp ratio."""
    rng = np.random.default_rng(0)
    nt, ns = xs[0].shape

    def stat(g, t, s):
        sub = g[0][t][:, s]
        if kind == "mean":
            return float(sub.mean())
        return float(sub.mean() / max(float(g[1][t][:, s].mean()), 1e-9))
    v = []
    for _ in range(B):
        t = rng.choice(nt, nt)
        s = rng.choice(ns, ns)
        v.append(stat(xs, t, s) - stat(ys, t, s))
    t0, s0 = np.arange(nt), np.arange(ns)
    a, b = stat(xs, t0, s0), stat(ys, t0, s0)
    return dict(a=a, b=b, delta=a - b, ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])


def label(d, margin=0.0, kind="help"):
    lo, hi = d["ci"]
    if kind == "help":                                                       # a minus b: a beats b iff lo > margin
        return "A BEATS B" if lo > margin else ("B BEATS A" if hi < -margin else "NO CLEAR DIFFERENCE")
    if kind == "noninf":                                                     # a not worse than b by more than margin
        return "NON-INFERIOR" if lo > -margin else ("COST" if hi < 0 else "NOT SHOWN")
    raise ValueError(kind)


def read(a):
    meta = n2()
    trials = DEV + NEWS
    outp = os.path.join(OUT, "raa_read%s.json" % (("_" + a.tag.lstrip("_")) if a.tag else ""))
    if os.path.exists(outp):
        raise SystemExit("%s exists; pass --tag with a new suffix" % outp)
    G = {}
    for arm in ARMS:
        for grp, comps in (("K", KNOWN), ("N", NEW), ("cheese", CHEESE), ("bottle", BOTTLE)):
            G[(arm, grp)] = (grid(meta, arm, comps, trials, "succ"), grid(meta, arm, comps, trials, "bound"))
    res = dict(states=dict(dev=DEV, new=NEWS), B=B, success={}, grasped_right={}, place_given_grasp={}, per_task={}, verdicts={})
    for arm in ARMS:
        for grp in ("K", "N", "cheese", "bottle"):
            s, b = G[(arm, grp)]
            res["success"]["%s|%s" % (arm, grp)] = float(s.mean())
            res["grasped_right"]["%s|%s" % (arm, grp)] = float(b.mean())
            res["place_given_grasp"]["%s|%s" % (arm, grp)] = float(s.mean() / max(float(b.mean()), 1e-9))
        for c in KNOWN + NEW:
            s, b = grid(meta, arm, [c], trials, "succ"), grid(meta, arm, [c], trials, "bound")
            res["per_task"]["%s|%s" % (arm, c)] = dict(success=float(s.mean()), grasped_right=float(b.mean()),
                                                       place_given_grasp=float(s.mean() / max(float(b.mean()), 1e-9)))
    fit = dict(r0_known=res["success"]["r0|K"], raa_known=res["success"]["raa|K"], rrel_known=res["success"]["rrel|K"], threshold=FIT)
    fit["r0_underfit"], fit["raa_underfit"], fit["rrel_underfit"] = (fit["r0_known"] < FIT, fit["raa_known"] < FIT, fit["rrel_known"] < FIT)
    res["fit_gate"] = fit
    V = res["verdicts"]

    def pair(name, ra, rb, grp, kind, margin=0.0, what="mean"):
        d = diff(G[(ra, grp)], G[(rb, grp)], what)
        V[name] = dict(a=ra, b=rb, group=grp, metric="success" if what == "mean" else "placed given right grasp", **d,
                       label=label(d, margin, kind) if kind != "none" else None)
        return V[name]
    v1a = pair("V1a RAA - R2, success, N1-N6", "raa", "r2", "N", "help")
    v1b = pair("V1b RAA - R2, placed given right grasp, N1-N6", "raa", "r2", "N", "help", what="pgg")
    nb = (v1a["label"] == "A BEATS B") + (v1b["label"] == "A BEATS B")
    V["V1 primary"] = dict(label={2: "SUPPORTED", 1: "PARTIAL"}.get(nb, "NOT SUPPORTED"),
                           rule="SUPPORTED iff V1a and V1b lower bounds > 0; PARTIAL iff exactly one; else NOT SUPPORTED")
    pair("V1l RAA - R2l (learned masks, no channel), success, N1-N6", "raa", "r2l", "N", "help")
    pair("V1l2 RAA - R2l, placed given right grasp, N1-N6", "raa", "r2l", "N", "help", what="pgg")
    pair("V2a Rrel - R0, success, N1-N6", "rrel", "r0", "N", "help")
    pair("V2b Rrel - R0, placed given right grasp, N1-N6", "rrel", "r0", "N", "help", what="pgg")
    pair("V2c RAA - Rrel, success, N1-N6 (what the views add to the channel)", "raa", "rrel", "N", "help")
    pair("V3a RAA - rgp, success, N1-N6", "raa", "rgp", "N", "noninf", MARGIN_RG)
    pair("V3b RAA - rgp, success, N1-N6 (beats)", "raa", "rgp", "N", "help")
    pair("V3c Rrel - rgp, success, N1-N6", "rrel", "rgp", "N", "noninf", MARGIN_RG)
    pair("K1 RAA - R2, success, K1-K6", "raa", "r2", "K", "noninf", MARGIN_K)
    pair("K2 Rrel - R0, success, K1-K6", "rrel", "r0", "K", "noninf", MARGIN_K)
    pair("K3 RAA - R0, success, K1-K6", "raa", "r0", "K", "noninf", MARGIN_K)
    pair("K4 RAA - rgp, success, K1-K6", "raa", "rgp", "K", "noninf", MARGIN_K)
    pair("D1 RAA - R2, cheese pairings N1-N3", "raa", "r2", "cheese", "help")
    pair("D2 RAA - R2, bottle pairings N4-N6", "raa", "r2", "bottle", "help")
    res["overall"] = "UNDERFIT (R0 under .80 on K1-K6)" if fit["r0_underfit"] else V["V1 primary"]["label"]      # sc_t34's gate: R0 only
    print("[raa-read]", json.dumps(res), flush=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raa-read] -> %s" % outp, flush=True)


def selftest(a):
    """The dev-state R2 - R0 of sc_t37_read.json (N1-N6 familiar, trials 4-23): same grid, same rng, same draws."""
    meta = n2()
    g = {r: (grid(meta, r, NEW, DEV, "succ"), grid(meta, r, NEW, DEV, "bound")) for r in ("r0", "r2")}
    d = diff(g["r2"], g["r0"])
    ref = json.load(open(os.path.join(OUT, "sc_t37_read.json"), encoding="utf-8"))
    want = ref["verdicts"]["r2 minus r0"]["composition_diff"]
    print("[raa-read] selftest: R2 - R0 delta %.4f ci %s; sc_t37_read.json: delta %.4f ci %s" % (d["delta"], d["ci"], want["delta"], want["ci"]), flush=True)
    assert abs(d["delta"] - want["delta"]) < 1e-9 and np.allclose(d["ci"], want["ci"]), "estimator does not reproduce sc_t37_read.json"
    print("[raa-read] selftest ok", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read", "selftest"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"read": read, "selftest": selftest}[a.what](a)

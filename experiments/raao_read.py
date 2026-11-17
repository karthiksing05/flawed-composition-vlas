"""RAA-o read (docs/SCAFFOLD_RESULTS.md 76; m3_prereg raao).  Numpy only (openpi venv).  EXTENDS raa_read.py without changing it:
it imports raa_read's grid / diff / label (same states, same bootstrap B = 2000, rng 0) and adds the arm raao (rollouts
results/flawed/raa/raao_{main,rx}_<tid>.json).

Registered contrasts (experiments/prereg/raao_pred.txt):
  W1a  RAA-o - RAA, success, N1-N6                         help: lower bound > 0
  W1b  RAA-o - RAA, placed given right grasp, N1-N6        help: lower bound > 0
  W1   primary: SUPPORTED iff W1a and W1b lower bounds > 0; PARTIAL iff exactly one; else NOT SUPPORTED (as V1)
  W2   RAA-o - rgp, success, N1-N6                         non-inferiority margin -.10 (reported either way), W2b = beats
  WK   RAA-o - RAA, success, K1-K6                         non-inferiority margin -.05
Reported with no label: RAA-o - R2 (V1's contrast), per-task success / right-grasp / placed-given-right-grasp, cheese vs bottle.

    python experiments/raao_read.py selftest     raa_read's dev-state selftest + the section 73 V1a / V1b recomputed from the rollout files
                                                 against results/flawed/raa_read.json
    python experiments/raao_read.py read [--tag T]   -> results/flawed/raao_read[_T].json (refuses to overwrite)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raa_read as RR                                                        # noqa: E402

OUT, RD = RR.OUT, RR.RD
RR.ARMS["raao"] = (RD, "raao")                                               # extends the arm table; raa_read.py itself is unchanged
MARGIN_K, MARGIN_RG = 0.05, 0.10


def selftest(a):
    RR.selftest(a)                                                           # sc_t37_read.json's R2 - R0 on the dev states
    meta = RR.n2()
    trials = RR.DEV + RR.NEWS
    g = {r: (RR.grid(meta, r, RR.NEW, trials, "succ"), RR.grid(meta, r, RR.NEW, trials, "bound")) for r in ("raa", "r2")}
    ref = json.load(open(os.path.join(OUT, "raa_read.json"), encoding="utf-8"))["verdicts"]
    for key, kind in (("V1a RAA - R2, success, N1-N6", "mean"), ("V1b RAA - R2, placed given right grasp, N1-N6", "pgg")):
        d = RR.diff(g["raa"], g["r2"], kind)
        want = ref[key]
        print("[raao-read] selftest %s: delta %.4f ci %s; raa_read.json: delta %.4f ci %s" % (key[:3], d["delta"], d["ci"], want["delta"], want["ci"]), flush=True)
        assert abs(d["delta"] - want["delta"]) < 1e-9 and np.allclose(d["ci"], want["ci"]), "does not reproduce section 73's " + key
    print("[raao-read] selftest ok (section 73's V1a and V1b reproduced)", flush=True)


def read(a):
    meta = RR.n2()
    trials = RR.DEV + RR.NEWS
    outp = os.path.join(OUT, "raao_read%s.json" % (("_" + a.tag.lstrip("_")) if a.tag else ""))
    if os.path.exists(outp):
        raise SystemExit("%s exists; pass --tag with a new suffix" % outp)
    arms = ("r0", "r2", "raa", "raao", "rgp")
    G = {}
    for arm in arms:
        for grp, comps in (("K", RR.KNOWN), ("N", RR.NEW), ("cheese", RR.CHEESE), ("bottle", RR.BOTTLE)):
            G[(arm, grp)] = (RR.grid(meta, arm, comps, trials, "succ"), RR.grid(meta, arm, comps, trials, "bound"))
    res = dict(states=dict(dev=RR.DEV, new=RR.NEWS), B=RR.B, success={}, grasped_right={}, place_given_grasp={}, per_task={}, verdicts={})
    for arm in arms:
        for grp in ("K", "N", "cheese", "bottle"):
            s, b = G[(arm, grp)]
            res["success"]["%s|%s" % (arm, grp)] = float(s.mean())
            res["grasped_right"]["%s|%s" % (arm, grp)] = float(b.mean())
            res["place_given_grasp"]["%s|%s" % (arm, grp)] = float(s.mean() / max(float(b.mean()), 1e-9))
        for c in RR.KNOWN + RR.NEW:
            s, b = RR.grid(meta, arm, [c], trials, "succ"), RR.grid(meta, arm, [c], trials, "bound")
            res["per_task"]["%s|%s" % (arm, c)] = dict(success=float(s.mean()), grasped_right=float(b.mean()),
                                                       place_given_grasp=float(s.mean() / max(float(b.mean()), 1e-9)))
    res["fit_gate"] = dict(r0_known=res["success"]["r0|K"], threshold=RR.FIT, r0_underfit=res["success"]["r0|K"] < RR.FIT)
    V = res["verdicts"]

    def pair(name, ra, rb, grp, kind, margin=0.0, what="mean"):
        d = RR.diff(G[(ra, grp)], G[(rb, grp)], what)
        V[name] = dict(arm_a=ra, arm_b=rb, group=grp, metric="success" if what == "mean" else "placed given right grasp", **d,
                       label=RR.label(d, margin, kind) if kind != "none" else None)
        return V[name]
    w1a = pair("W1a RAA-o - RAA, success, N1-N6", "raao", "raa", "N", "help")
    w1b = pair("W1b RAA-o - RAA, placed given right grasp, N1-N6", "raao", "raa", "N", "help", what="pgg")
    nb = (w1a["label"] == "A BEATS B") + (w1b["label"] == "A BEATS B")
    V["W1 primary"] = dict(label={2: "SUPPORTED", 1: "PARTIAL"}.get(nb, "NOT SUPPORTED"),
                           rule="SUPPORTED iff W1a and W1b lower bounds > 0; PARTIAL iff exactly one; else NOT SUPPORTED")
    pair("W2a RAA-o - rgp, success, N1-N6 (non-inferiority -.10)", "raao", "rgp", "N", "noninf", MARGIN_RG)
    pair("W2b RAA-o - rgp, success, N1-N6 (beats)", "raao", "rgp", "N", "help")
    pair("WK RAA-o - RAA, success, K1-K6 (non-inferiority -.05)", "raao", "raa", "K", "noninf", MARGIN_K)
    pair("X1 RAA-o - R2, success, N1-N6 (V1's contrast, no label weight)", "raao", "r2", "N", "none")
    pair("X2 RAA-o - RAA, cheese pairings N1-N3", "raao", "raa", "cheese", "help")
    pair("X3 RAA-o - RAA, bottle pairings N4-N6", "raao", "raa", "bottle", "help")
    pair("X4 RAA-o - R0, success, K1-K6", "raao", "r0", "K", "noninf", MARGIN_K)
    res["overall"] = "UNDERFIT (R0 under .80 on K1-K6)" if res["fit_gate"]["r0_underfit"] else V["W1 primary"]["label"]
    print("[raao-read]", json.dumps(res), flush=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raao-read] -> %s" % outp, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read", "selftest"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"read": read, "selftest": selftest}[a.what](a)

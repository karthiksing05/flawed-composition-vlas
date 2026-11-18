"""RAA-s read (docs/SCAFFOLD_RESULTS.md 78; m3_prereg raas).  Numpy only (openpi venv).  EXTENDS raa_read.py and raao_read.py without
changing them: same states (dev trials 4-23 + new k 540-579), same bootstrap (B = 2000, rng 0), same grid / diff / label; adds the arm raas
(rollouts results/flawed/raa/raas_{main,rx}_<tid>.json).  RAA-o's rows are SR 76's rollout files, unchanged.

Registered contrasts (docs/notes/raas_plan.md, experiments/prereg/raas_pred.txt):
  S1a  RAA-s - RAA-o, success, N1-N6                       help: lower bound > 0
  S1b  RAA-s - RAA-o, placed given right grasp, N1-N6      help: lower bound > 0
  S1   primary: SUPPORTED iff S1a and S1b lower bounds > 0; PARTIAL iff exactly one; else NOT SUPPORTED (as W1)
  S2   RAA-s - RAA, success, N1-N6                         help: lower bound > 0
  S3   RAA-s - rgp, success, N1-N6                         non-inferiority margin -.10 (reported either way), S3b = beats
  SK   RAA-s - RAA-o, success, K1-K6                       non-inferiority margin -.05
Reported with no label: RAA-s - R2, RAA-s - R0 (K), cheese vs bottle, per-task success / right-grasp / placed-given-right-grasp.

    python experiments/raas_read.py selftest    raao_read's selftest (SR 73's V1a / V1b) + SR 76's W1a / W1b / WK recomputed from the rollout files
                                                against results/flawed/raao_read.json (the same machinery must reproduce them exactly)
    python experiments/raas_read.py read [--tag T]   -> results/flawed/raas_read[_T].json (refuses to overwrite)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raa_read as RR                                                        # noqa: E402
import raao_read as RO                                                       # noqa: E402,F401  (registers the raao arm in RR.ARMS)
import raac_read as RC                                                       # noqa: E402,F401  (registers the raac arm: RAA-s is read against it as an unlabelled reference)

OUT, RD = RR.OUT, RR.RD
RR.ARMS["raas"] = (RD, "raas")                                               # extends the arm table; raa_read.py / raao_read.py are unchanged
MARGIN_K, MARGIN_RG = 0.05, 0.10


def selftest(a):
    RO.selftest(a)                                                           # SR 73's V1a / V1b (and the dev-state selftest)
    meta = RR.n2()
    trials = RR.DEV + RR.NEWS
    g = {r: {grp: (RR.grid(meta, r, comps, trials, "succ"), RR.grid(meta, r, comps, trials, "bound")) for grp, comps in (("N", RR.NEW), ("K", RR.KNOWN))}
         for r in ("raao", "raa")}
    ref = json.load(open(os.path.join(OUT, "raao_read.json"), encoding="utf-8"))["verdicts"]
    for key, grp, kind in (("W1a RAA-o - RAA, success, N1-N6", "N", "mean"), ("W1b RAA-o - RAA, placed given right grasp, N1-N6", "N", "pgg"),
                           ("WK RAA-o - RAA, success, K1-K6 (non-inferiority -.05)", "K", "mean")):
        d = RR.diff(g["raao"][grp], g["raa"][grp], kind)
        want = ref[key]
        print("[raas-read] selftest %s: delta %.4f ci %s; raao_read.json: delta %.4f ci %s" % (key[:3], d["delta"], d["ci"], want["delta"], want["ci"]), flush=True)
        assert abs(d["delta"] - want["delta"]) < 1e-9 and np.allclose(d["ci"], want["ci"]), "does not reproduce section 76's " + key
    print("[raas-read] selftest ok (section 73's V1a / V1b and section 76's W1a / W1b / WK reproduced exactly)", flush=True)


def read(a):
    meta = RR.n2()
    trials = RR.DEV + RR.NEWS
    outp = os.path.join(OUT, "raas_read%s.json" % (("_" + a.tag.lstrip("_")) if a.tag else ""))
    if os.path.exists(outp):
        raise SystemExit("%s exists; pass --tag with a new suffix" % outp)
    arms = ("r0", "r2", "raa", "raao", "raac", "raas", "rgp")
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
    u1a = pair("S1a RAA-s - RAA-o, success, N1-N6", "raas", "raao", "N", "help")
    u1b = pair("S1b RAA-s - RAA-o, placed given right grasp, N1-N6", "raas", "raao", "N", "help", what="pgg")
    nb = (u1a["label"] == "A BEATS B") + (u1b["label"] == "A BEATS B")
    V["S1 primary"] = dict(label={2: "SUPPORTED", 1: "PARTIAL"}.get(nb, "NOT SUPPORTED"),
                           rule="SUPPORTED iff S1a and S1b lower bounds > 0; PARTIAL iff exactly one; else NOT SUPPORTED")
    pair("S2 RAA-s - RAA, success, N1-N6", "raas", "raa", "N", "help")
    pair("S3a RAA-s - rgp, success, N1-N6 (non-inferiority -.10)", "raas", "rgp", "N", "noninf", MARGIN_RG)
    pair("S3b RAA-s - rgp, success, N1-N6 (beats)", "raas", "rgp", "N", "help")
    pair("SK RAA-s - RAA-o, success, K1-K6 (non-inferiority -.05)", "raas", "raao", "K", "noninf", MARGIN_K)
    pair("X1 RAA-s - R2, success, N1-N6 (no label weight)", "raas", "r2", "N", "none")
    pair("X2 RAA-s - RAA-o, cheese pairings N1-N3", "raas", "raao", "cheese", "help")
    pair("X3 RAA-s - RAA-o, bottle pairings N4-N6", "raas", "raao", "bottle", "help")
    pair("X4 RAA-s - R0, success, K1-K6", "raas", "r0", "K", "noninf", MARGIN_K)
    pair("X5 RAA-s - RAA-c, success, N1-N6 (no label weight)", "raas", "raac", "N", "none")
    pair("X6 RAA-s - RAA-c, success, K1-K6 (no label weight)", "raas", "raac", "K", "none")
    res["overall"] = "UNDERFIT (R0 under .80 on K1-K6)" if res["fit_gate"]["r0_underfit"] else V["S1 primary"]["label"]
    print("[raas-read]", json.dumps(res), flush=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raas-read] -> %s" % outp, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read", "selftest"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"read": read, "selftest": selftest}[a.what](a)

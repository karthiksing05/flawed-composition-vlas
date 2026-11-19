"""RAA-L read (docs/SCAFFOLD_RESULTS.md 80; docs/notes/raal_plan.md; m3_prereg raal).  Numpy only (openpi venv; run on a compute node).
EXTENDS raa_read.py, raao_read.py and wa_read.py without changing them (imports their grid / diff / label / bootm / read).

PHASE A (SR 73's 60 registered states, dev trials 4-23 + new k 540-579; tasks K1-K6, N1-N6; rollouts results/flawed/raa/raal_{main,rx}_<tid>.json;
paired bootstrap B = 2000, rng 0, tasks and states resampled, raa_read machinery):
  L1   RAA-L - RAA, success, N1-N6                    SUPPORTED iff lower bound > 0 (else NOT SUPPORTED; REVERSED iff upper bound < 0)
  L2a  RAA-L - rgp, success, N1-N6, non-inferiority margin -.10 (NON-INFERIOR iff lower bound > -.10); L2b = beats rgp (lower bound > 0)
  LK   RAA-L - RAA, success, K1-K6, non-inferior at -.05
  fit gate (as SR 73): R0 on K1-K6 >= .80 (SR 73's R0 rollouts, unchanged)
  reported with no label: RAA-L - R2, RAA-L - R0 (K1-K6), RAA-L - RAA-o / RAA-c / RAA-s when their files exist, cheese vs bottle, per task.

PHASE B (rg_comp rgc main states, wa.py arms full / noR / noM on the RAA-L checkpoint, rollouts results/flawed/wal/<arm>_<tid>.json): wa_read.read with
--wa results/flawed/wal, then the gated reading below.  D1, D2, D3a, D3b, DK are wa_read's registered labels, unchanged.
  TESTABLE gates (full only): F6_a >= .50 (for D2 and D3a: the F6_b contrasts), mean of the unique-name controls tids 4-6 >= .50 (for D1 and D3b: the F7 contrasts),
  K1t + K2t >= .70 (floor).  A contrast on a cell whose gate fails is reported with its label and marked NOT TESTABLE.
  PRIMARY (gated): DISSOCIATION SHOWN iff D1 and D2 SUPPORTED, D3a and D3b EQUIVALENT, and all three gates TESTABLE.  Otherwise wa_read's rule (SR 79):
  PARTIAL iff D1 and D2 SUPPORTED with a D3 not EQUIVALENT, else NOT SHOWN; "UNTESTABLE" is reported if D1, D2 and both D3 pass but a gate fails.
  Descriptive (wa_read's d1-d11): full vs rc / I (frozen pi0.5, SR 70), F6_b vs F6_a, per-task rates.

    python experiments/raal_read.py selftest                reproduces SR 73's V1a / V1b and V3a (RAA - rgp) from the rollout files, and wa_read's selftest
    python experiments/raal_read.py read [--tag T]          -> results/flawed/raal_read[_T].json (refuses to overwrite)
    python experiments/raal_read.py readb [--tag T]         -> results/flawed/raal_wa_read[_T].json (wa_read) + raal_wa_gated[_T].json (refuses to overwrite)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raa_read as RR                                                        # noqa: E402
import raao_read as RO                                                       # noqa: E402,F401  (registers raao)
import raac_read as RC                                                       # noqa: E402,F401  (registers raac)
import raas_read as RS                                                       # noqa: E402,F401  (registers raas)
import wa_read as WR                                                         # noqa: E402

OUT, RD = RR.OUT, RR.RD
RR.ARMS["raal"] = (RD, "raal")                                               # extends the arm table; raa_read.py is unchanged
MARGIN_K, MARGIN_RG = 0.05, 0.10
WAL = os.path.join(OUT, "wal")
TIDS12 = (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26)


def have(arm):
    d, stem = RR.ARMS[arm]
    return all(os.path.exists(os.path.join(RD, "%s_rx_%d.json" % (stem, t))) for t in TIDS12)


def selftest(a):
    RO.selftest(a)                                                           # raa_read's dev-state selftest + SR 73's V1a / V1b
    meta = RR.n2()
    trials = RR.DEV + RR.NEWS
    g = {r: (RR.grid(meta, r, RR.NEW, trials, "succ"), RR.grid(meta, r, RR.NEW, trials, "bound")) for r in ("raa", "rgp")}
    d = RR.diff(g["raa"], g["rgp"], "mean")
    want = json.load(open(os.path.join(OUT, "raa_read.json"), encoding="utf-8"))["verdicts"]["V3a RAA - rgp, success, N1-N6"]
    print("[raal-read] selftest V3a: delta %.4f ci %s; raa_read.json: delta %.4f ci %s" % (d["delta"], d["ci"], want["delta"], want["ci"]), flush=True)
    assert abs(d["delta"] - want["delta"]) < 1e-9 and np.allclose(d["ci"], want["ci"]), "does not reproduce section 73's V3a"
    WR.selftest(a)
    print("[raal-read] selftest ok", flush=True)


def read(a):
    meta = RR.n2()
    trials = RR.DEV + RR.NEWS
    outp = os.path.join(OUT, "raal_read%s.json" % (("_" + a.tag.lstrip("_")) if a.tag else ""))
    if os.path.exists(outp):
        raise SystemExit("%s exists; pass --tag with a new suffix" % outp)
    arms = ["r0", "r2", "raa", "raal", "rgp"] + [x for x in ("raao", "raac", "raas") if have(x)]
    G = {}
    for arm in arms:
        for grp, comps in (("K", RR.KNOWN), ("N", RR.NEW), ("cheese", RR.CHEESE), ("bottle", RR.BOTTLE)):
            G[(arm, grp)] = (RR.grid(meta, arm, comps, trials, "succ"), RR.grid(meta, arm, comps, trials, "bound"))
    res = dict(states=dict(dev=RR.DEV, new=RR.NEWS), B=RR.B, arms=arms, success={}, grasped_right={}, place_given_grasp={}, per_task={}, verdicts={})
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
        lab = None
        if kind == "support":
            lo, hi = d["ci"]
            lab = "SUPPORTED" if lo > 0 else ("REVERSED" if hi < 0 else "NOT SUPPORTED")
        elif kind != "none":
            lab = RR.label(d, margin, kind)
        V[name] = dict(arm_a=ra, arm_b=rb, group=grp, metric="success" if what == "mean" else "placed given right grasp", **d, label=lab)
        return V[name]
    pair("L1 RAA-L - RAA, success, N1-N6", "raal", "raa", "N", "support")
    pair("L2a RAA-L - rgp, success, N1-N6 (non-inferiority -.10)", "raal", "rgp", "N", "noninf", MARGIN_RG)
    pair("L2b RAA-L - rgp, success, N1-N6 (beats)", "raal", "rgp", "N", "help")
    pair("LK RAA-L - RAA, success, K1-K6 (non-inferiority -.05)", "raal", "raa", "K", "noninf", MARGIN_K)
    pair("X1 RAA-L - R2, success, N1-N6 (no label weight)", "raal", "r2", "N", "none")
    pair("X2 RAA-L - R0, success, K1-K6 (no label weight)", "raal", "r0", "K", "none")
    pair("X3 RAA-L - RAA, placed given right grasp, N1-N6 (no label weight)", "raal", "raa", "N", "none", what="pgg")
    pair("X4 RAA-L - RAA, cheese pairings N1-N3 (no label weight)", "raal", "raa", "cheese", "none")
    pair("X5 RAA-L - RAA, bottle pairings N4-N6 (no label weight)", "raal", "raa", "bottle", "none")
    for x in ("raao", "raac", "raas"):
        if x in arms:
            pair("X6 RAA-L - %s, success, N1-N6 (no label weight)" % x, "raal", x, "N", "none")
    res["primary"] = V["L1 RAA-L - RAA, success, N1-N6"]["label"]
    res["overall"] = "UNDERFIT (R0 under .80 on K1-K6)" if res["fit_gate"]["r0_underfit"] else res["primary"]
    print("[raal-read]", json.dumps(res), flush=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raal-read] -> %s" % outp, flush=True)


def readb(a):
    sfx = ("_" + a.tag.lstrip("_")) if a.tag else ""
    o1, o2 = os.path.join(OUT, "raal_wa_read%s.json" % sfx), os.path.join(OUT, "raal_wa_gated%s.json" % sfx)
    for o in (o1, o2):
        if os.path.exists(o):
            raise SystemExit("%s exists; pass --tag with a new suffix" % o)
    WR.read(argparse.Namespace(phase="main", wa=WAL, out=o1))
    r = json.load(open(o1, encoding="utf-8"))
    gt = {k: (v["label"] == "TESTABLE") for k, v in r["gates"].items()}
    inp = r["primary_inputs"]
    allg = all(gt.values()) and len(gt) == 3
    shown_rule = inp["D1"] == "SUPPORTED" and inp["D2"] == "SUPPORTED" and inp["D3a"] == "EQUIVALENT" and inp["D3b"] == "EQUIVALENT"
    g = dict(wa_read_primary=r["primary"], inputs=inp, gates=r["gates"],
             testable={"D1 (F7: unique-name gate)": gt.get("F7_full_unique_controls"), "D3b (F7: unique-name gate)": gt.get("F7_full_unique_controls"),
                       "D2 (F6_b: F6_a gate)": gt.get("F6_full_F6_a"), "D3a (F6_b: F6_a gate)": gt.get("F6_full_F6_a"), "DK (K floor gate)": gt.get("floor_full_K")},
             primary_gated=("DISSOCIATION SHOWN" if (shown_rule and allg) else ("UNTESTABLE (D1, D2, D3a, D3b pass, a gate fails)" if shown_rule else r["primary"])),
             deltas={k: dict(delta=v["delta"], ci=v["ci"], label=v.get("label")) for k, v in r["contrasts"].items()})
    with open(o2, "w", encoding="utf-8") as fh:
        json.dump(g, fh, indent=1)
    print("[raal-read] PRIMARY (gated): %s; gates %s" % (g["primary_gated"], {k: v["label"] for k, v in r["gates"].items()}), flush=True)
    print("[raal-read] -> %s, %s" % (o1, o2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read", "readb", "selftest"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"read": read, "readb": readb, "selftest": selftest}[a.what](a)

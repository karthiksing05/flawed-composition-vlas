"""RAA-sL read (docs/SCAFFOLD_RESULTS.md 82; docs/notes/raasl_plan.md; m3_prereg raasl).  Numpy only (openpi venv; run on a compute node).
Adapted from raal_read.py.  EXTENDS raa_read.py, raao_read.py, raac_read.py, raas_read.py, raas_b_read.py, raal_read.py and wa_read.py without changing them
(imports their grid / diff / label / bootm / read).  RAA-sL = pi05_g8v_raas_l_ae: the RAA-s recipe (servo residual, RAA-c state token) initialised from openpi pi05_libero.

PHASE A (SR 73's 60 registered states, dev trials 4-23 + new k 540-579; tasks K1-K6, N1-N6; rollouts results/flawed/raa/raasl_{main,rx}_<tid>.json;
paired bootstrap B = 2000, rng 0, tasks and states resampled, raa_read machinery).  "RAA-s(78b)" = arm raasb = results/flawed/raa/raas_b_*.json (SR 78b, served WITH servo):
  P1   PRIMARY  RAA-sL - RAA-s(78b), success, N1-N6    SUPPORTED iff lower bound > 0 (else NOT SUPPORTED; REVERSED iff upper bound < 0)
  P2   RAA-sL - RAA-L (SR 80), success, N1-N6          SUPPORTED iff lower bound > 0 (the servo's effect on the broad init)
  P3a  RAA-sL - rgp, success, N1-N6, non-inferiority margin -.10 (NON-INFERIOR iff lower bound > -.10); P3b = beats rgp (lower bound > 0)
  PK   RAA-sL - RAA-s(78b), success, K1-K6, non-inferior at -.05
  fit gate (as SR 73): R0 on K1-K6 >= .80 (SR 73's R0 rollouts, unchanged)
  reported with no label: RAA-sL - RAA, - R2, - R0 (K1-K6), - RAA-o, - RAA-c, - RAA-s(78); placed given right grasp vs RAA-s(78b) and RAA-L; cheese vs bottle vs RAA-s(78b);
  per task.

PHASE B (rg_comp rgc main states, wa_s.py arms full / noR / noM on the RAA-sL checkpoint, rollouts results/flawed/wasl/<arm>_<tid>.json): wa_read.read with --wa
results/flawed/wasl, then the gated reading below (as raal_read.readb: D1, D2, D3a, D3b, DK = wa_read's registered labels, unchanged; gates F6_a >= .50, unique controls
>= .50, K1t + K2t >= .70).  Descriptive: full F6_b vs frozen I (.517 in SR 79 / 80, its rollouts in the wa_read tables) and rgp, rc; full vs SR 79's RAA / SR 80's RAA-L.
wa_read.read crashes in its print loop after writing the json when a contrast has no delta (noMe arms not rolled; SR 80 deviation 1): readb catches that, requires the
json to exist, and gates from it with .get.

    python experiments/raasl_read.py selftest            reproduces SR 73's V1a / V1b, SR 76's W1a / W1b / WK, SR 73's V3a and wa_read's selftest from the rollout files
    python experiments/raasl_read.py read [--tag T]      -> results/flawed/raasl_read[_T].json (refuses to overwrite)
    python experiments/raasl_read.py readb [--tag T]     -> results/flawed/raasl_wa_read[_T].json (wa_read) + raasl_wa_gated[_T].json (refuses to overwrite)
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
import raas_b_read as RB                                                     # noqa: E402,F401  (registers raasb = SR 78b)
import raal_read as RL                                                       # noqa: E402,F401  (registers raal)
import wa_read as WR                                                         # noqa: E402

OUT, RD = RR.OUT, RR.RD
RR.ARMS["raasl"] = (RD, "raasl43")                                           # SEED 2 COPY of raasl_read.py: arm "raasl" = the seed-43 rollouts raasl43_*; "raasl1" = SR 82's seed-1 rollouts
RR.ARMS["raasl1"] = (RD, "raasl")                                            # extends the arm table; raa_read.py is unchanged
MARGIN_K, MARGIN_RG = 0.05, 0.10
WASL = os.path.join(OUT, "wasl43")
TIDS12 = RL.TIDS12


def have(arm):
    d, stem = RR.ARMS[arm]
    return all(os.path.exists(os.path.join(RD, "%s_rx_%d.json" % (stem, t))) for t in TIDS12)


def selftest(a):
    RB.selftest(a)                                                           # SR 73's V1a / V1b (raao_read) + SR 76's W1a / W1b / WK
    meta = RR.n2()
    trials = RR.DEV + RR.NEWS
    g = {r: (RR.grid(meta, r, RR.NEW, trials, "succ"), RR.grid(meta, r, RR.NEW, trials, "bound")) for r in ("raa", "rgp")}
    d = RR.diff(g["raa"], g["rgp"], "mean")
    want = json.load(open(os.path.join(OUT, "raa_read.json"), encoding="utf-8"))["verdicts"]["V3a RAA - rgp, success, N1-N6"]
    print("[raasl-read] selftest V3a: delta %.4f ci %s; raa_read.json: delta %.4f ci %s" % (d["delta"], d["ci"], want["delta"], want["ci"]), flush=True)
    assert abs(d["delta"] - want["delta"]) < 1e-9 and np.allclose(d["ci"], want["ci"]), "does not reproduce section 73's V3a"
    WR.selftest(a)
    print("[raasl-read] selftest ok", flush=True)


def read(a):
    meta = RR.n2()
    trials = RR.DEV + RR.NEWS
    outp = os.path.join(OUT, "raasl43_read%s.json" % (("_" + a.tag.lstrip("_")) if a.tag else ""))
    if os.path.exists(outp):
        raise SystemExit("%s exists; pass --tag with a new suffix" % outp)
    arms = ["r0", "r2", "raa", "raasl", "raasl1", "raasb", "raal", "rgp"] + [x for x in ("raao", "raac", "raas") if have(x)]
    assert have("raasl") and have("raasl1") and have("raasb") and have("raal"), "missing rollout files"
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
    pair("P1 RAA-sL - RAA-s(78b), success, N1-N6", "raasl", "raasb", "N", "support")
    pair("P2 RAA-sL - RAA-L, success, N1-N6", "raasl", "raal", "N", "support")
    pair("P3a RAA-sL - rgp, success, N1-N6 (non-inferiority -.10)", "raasl", "rgp", "N", "noninf", MARGIN_RG)
    pair("P3b RAA-sL - rgp, success, N1-N6 (beats)", "raasl", "rgp", "N", "help")
    pair("PK RAA-sL - RAA-s(78b), success, K1-K6 (non-inferiority -.05)", "raasl", "raasb", "K", "noninf", MARGIN_K)
    pair("X1 RAA-sL - RAA, success, N1-N6 (no label weight)", "raasl", "raa", "N", "none")
    pair("X2 RAA-sL - R2, success, N1-N6 (no label weight)", "raasl", "r2", "N", "none")
    pair("X3 RAA-sL - R0, success, K1-K6 (no label weight)", "raasl", "r0", "K", "none")
    pair("X4 RAA-sL - RAA-s(78b), placed given right grasp, N1-N6 (no label weight)", "raasl", "raasb", "N", "none", what="pgg")
    pair("X5 RAA-sL - RAA-L, placed given right grasp, N1-N6 (no label weight)", "raasl", "raal", "N", "none", what="pgg")
    pair("X6 RAA-sL - RAA-s(78b), cheese pairings N1-N3 (no label weight)", "raasl", "raasb", "cheese", "none")
    pair("X7 RAA-sL - RAA-s(78b), bottle pairings N4-N6 (no label weight)", "raasl", "raasb", "bottle", "none")
    pair("X8 RAA-sL - RAA-L, K1-K6 (no label weight)", "raasl", "raal", "K", "none")
    pair("S1 seed 2 - seed 1, success, N1-N6 (replication, no label weight)", "raasl", "raasl1", "N", "none")
    pair("S2 seed 2 - seed 1, success, K1-K6 (replication, no label weight)", "raasl", "raasl1", "K", "none")
    pair("S3 seed 2 - seed 1, success, bottle N4-N6 (no label weight)", "raasl", "raasl1", "bottle", "none")
    pair("S4 seed 2 - seed 1, success, cheese N1-N3 (no label weight)", "raasl", "raasl1", "cheese", "none")
    for x in ("raao", "raac", "raas"):
        if x in arms:
            pair("X9 RAA-sL - %s, success, N1-N6 (no label weight)" % x, "raasl", x, "N", "none")
    res["primary"] = V["P1 RAA-sL - RAA-s(78b), success, N1-N6"]["label"]
    res["overall"] = "UNDERFIT (R0 under .80 on K1-K6)" if res["fit_gate"]["r0_underfit"] else res["primary"]
    print("[raasl-read]", json.dumps(res), flush=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raasl-read] -> %s" % outp, flush=True)


def readb(a):
    sfx = ("_" + a.tag.lstrip("_")) if a.tag else ""
    o1, o2 = os.path.join(OUT, "raasl43_wa_read%s.json" % sfx), os.path.join(OUT, "raasl43_wa_gated%s.json" % sfx)
    for o in (o1, o2):
        if os.path.exists(o):
            raise SystemExit("%s exists; pass --tag with a new suffix" % o)
    try:
        WR.read(argparse.Namespace(phase="main", wa=WASL, out=o1))
    except KeyError as e:                                                    # wa_read's print loop on a contrast without delta (noMe not rolled), after the json is written
        if not os.path.exists(o1):
            raise
        print("[raasl-read] wa_read print loop KeyError %r after writing %s (noMe contrasts, as SR 80); continuing from the json" % (e, o1), flush=True)
    r = json.load(open(o1, encoding="utf-8"))
    gt = {k: (v["label"] == "TESTABLE") for k, v in r["gates"].items()}
    inp = r["primary_inputs"]
    allg = all(gt.values()) and len(gt) == 3
    shown_rule = inp["D1"] == "SUPPORTED" and inp["D2"] == "SUPPORTED" and inp["D3a"] == "EQUIVALENT" and inp["D3b"] == "EQUIVALENT"
    g = dict(wa_read_primary=r["primary"], inputs=inp, gates=r["gates"],
             testable={"D1 (F7: unique-name gate)": gt.get("F7_full_unique_controls"), "D3b (F7: unique-name gate)": gt.get("F7_full_unique_controls"),
                       "D2 (F6_b: F6_a gate)": gt.get("F6_full_F6_a"), "D3a (F6_b: F6_a gate)": gt.get("F6_full_F6_a"), "DK (K floor gate)": gt.get("floor_full_K")},
             primary_gated=("DISSOCIATION SHOWN" if (shown_rule and allg) else ("UNTESTABLE (D1, D2, D3a, D3b pass, a gate fails)" if shown_rule else r["primary"])),
             deltas={k: dict(delta=v.get("delta"), ci=v.get("ci"), label=v.get("label")) for k, v in r["contrasts"].items()})
    with open(o2, "w", encoding="utf-8") as fh:
        json.dump(g, fh, indent=1)
    print("[raasl-read] PRIMARY (gated): %s; gates %s" % (g["primary_gated"], {k: v["label"] for k, v in r["gates"].items()}), flush=True)
    print("[raasl-read] -> %s, %s" % (o1, o2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read", "readb", "selftest"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"read": read, "readb": readb, "selftest": selftest}[a.what](a)

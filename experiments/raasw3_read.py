"""RAA-sL-W read (SR 84; docs/notes/raasw_plan.md; m3_prereg raasw).  Numpy only (openpi venv; run on a compute node).  Adapted from raasl_read.py, which it imports (registering
raasl and every earlier arm) without changing it.  W1 = RAA-sL with the xy of transport owned by the module (arm raasw1, results/flawed/raa/raasw1_*; Phase B results/flawed/wasw1),
W2 = W1 + release gate (raasw2 / wasw2).  Reference arm "raasl" = SR 82's rollouts, same states.

PHASE A (paired bootstrap B = 2000, rng 0, tasks and states resampled, raa_read machinery), for W in (W1, W2):
  A1 PRIMARY-A  W - RAA-sL, success, N1-N6    SUPPORTED iff lower > 0 (REVERSED iff upper < 0; else NOT SUPPORTED)
  A2a  W - rgp, N1-N6, NON-INFERIOR iff lower > -.10;  A2b  W beats rgp iff lower > 0
  AK   W - RAA-sL, K1-K6, NON-INFERIOR iff lower > -.05;  fit gate R0 K1-K6 >= .80
PHASE B: wa_read.read on results/flawed/wasw<W> (D1', D2', D3a, D3b, DK, gates, descriptive d-contrasts vs I / rgp / rc), then the SR 82 gated reading, plus
  B1 full_W - full RAA-sL (results/flawed/wasl), F6_b: SUPPORTED iff lower > 0;  B2 full_W - frozen I, F6_b: SUPPORTED iff lower > 0 (REVERSED iff upper < 0);
  B3 descriptive: full_W - RAA-sL full on F7 pool, F6_a, K1t + K2t; noM replication (noM_W vs SR 82's noM, per state).

    python experiments/raasw_read.py selftest            reproduces SR 82's P3a / P1 from the files and SR 82's D2 (wa_read on results/flawed/wasl, scratch output) + raasl_read's selftest
    python experiments/raasw_read.py read [--tag T]      -> results/flawed/raasw_read[_T].json (refuses to overwrite)
    python experiments/raasw_read.py readb [--tag T]     -> results/flawed/raasw_wa_read_w{1,2}[_T].json + raasw_wa_gated[_T].json (refuses to overwrite)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raa_read as RR                                                        # noqa: E402
import raasl_read as RSL                                                     # noqa: E402  (registers raasl and every earlier arm)
import wa_read as WR                                                         # noqa: E402

OUT, RD = RR.OUT, RR.RD
for w in (1, 2, 3, 4):
    RR.ARMS["raasw%d" % w] = (RD, "raasw%d" % w)                             # extends the arm table; raa_read.py is unchanged
MARGIN_K, MARGIN_RG = 0.05, 0.10
WASL = os.path.join(OUT, "wasl")
TIDS12 = RSL.TIDS12
SCR = "/coc/flash10/ksingara3/.claude_tmp/pp"


def have(arm):
    return RSL.have(arm)


def selftest(a):
    RSL.selftest(a)                                                          # SR 73 / 76 / 70 selfchecks via raasl_read, then wa_read's
    meta = RR.n2()
    trials = RR.DEV + RR.NEWS
    G = {r: (RR.grid(meta, r, RR.NEW, trials, "succ"), RR.grid(meta, r, RR.NEW, trials, "bound")) for r in ("raasl", "rgp", "raasb")}
    want = json.load(open(os.path.join(OUT, "raasl_read.json"), encoding="utf-8"))["verdicts"]
    d = RR.diff(G["raasl"], G["rgp"], "mean")
    w = want["P3a RAA-sL - rgp, success, N1-N6 (non-inferiority -.10)"]
    print("[raasw3-read] selftest P3a: delta %.4f ci %s; raasl_read.json: delta %.4f ci %s" % (d["delta"], d["ci"], w["delta"], w["ci"]), flush=True)
    assert abs(d["delta"] - w["delta"]) < 1e-9 and np.allclose(d["ci"], w["ci"]), "does not reproduce SR 82's P3a"
    d = RR.diff(G["raasl"], G["raasb"], "mean")
    w = want["P1 RAA-sL - RAA-s(78b), success, N1-N6"]
    assert abs(d["delta"] - w["delta"]) < 1e-9 and np.allclose(d["ci"], w["ci"]), "does not reproduce SR 82's P1"
    G1 = (RR.grid(meta, "raasw1", RR.NEW, trials, "succ"), RR.grid(meta, "raasw1", RR.NEW, trials, "bound"))
    d = RR.diff(G1, G["raasl"], "mean")
    w = json.load(open(os.path.join(OUT, "raasw_read.json"), encoding="utf-8"))["verdicts"]["A1 PRIMARY-A W1 - RAA-sL, success, N1-N6"]
    assert abs(d["delta"] - w["delta"]) < 1e-9 and np.allclose(d["ci"], w["ci"]), "does not reproduce SR 84's A1 (W1)"
    print("[raasw3-read] selftest: SR 84's A1 (W1) reproduced exactly", flush=True)
    o = os.path.join(SCR, "raasw3_selftest_wa_read_%d.json" % os.getpid())
    assert not os.path.exists(o)
    try:
        WR.read(argparse.Namespace(phase="main", wa=WASL, out=o))
    except KeyError:
        pass
    r = json.load(open(o, encoding="utf-8"))
    w = json.load(open(os.path.join(OUT, "raasl_wa_read.json"), encoding="utf-8"))
    for k in ("D1 full - noR, F7", "D2 full - noM, F6_b"):
        assert abs(r["contrasts"][k]["delta"] - w["contrasts"][k]["delta"]) < 1e-9 and np.allclose(r["contrasts"][k]["ci"], w["contrasts"][k]["ci"]), k
    print("[raasw3-read] selftest: SR 82's P1, P3a, D1, D2 reproduced exactly from the SR 82 files (wa_read scratch output %s)" % o, flush=True)
    print("[raasw3-read] selftest ok", flush=True)


def read(a):
    meta = RR.n2()
    trials = RR.DEV + RR.NEWS
    outp = os.path.join(OUT, "raasw3_read%s.json" % (("_" + a.tag.lstrip("_")) if a.tag else ""))
    if os.path.exists(outp):
        raise SystemExit("%s exists; pass --tag with a new suffix" % outp)
    assert have("raasl") and have("raasb") and have("raasw1") and have("raasw2") and have("raasw3") and have("raasw4") and have("raal"), "missing rollout files"
    arms = ["r0", "r2", "raa", "raasl", "raasb", "raal", "rgp", "raasw1", "raasw2", "raasw3", "raasw4"] + [x for x in ("raao", "raac", "raas") if have(x)]
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
            res["per_task"]["%s|%s" % (arm, c)] = dict(success=float(s.mean()), grasped_right=float(b.mean()), place_given_grasp=float(s.mean() / max(float(b.mean()), 1e-9)))
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
    for w in (3, 4):
        W, ar = "W%d" % w, "raasw%d" % w
        pair("A1 PRIMARY-A %s - RAA-sL, success, N1-N6" % W, ar, "raasl", "N", "support")
        pair("A2a %s - rgp, success, N1-N6 (non-inferiority -.10)" % W, ar, "rgp", "N", "noninf", MARGIN_RG)
        pair("A2b %s - rgp, success, N1-N6 (beats)" % W, ar, "rgp", "N", "help")
        pair("AK %s - RAA-sL, success, K1-K6 (non-inferiority -.05)" % W, ar, "raasl", "K", "noninf", MARGIN_K)
        pair("X1 %s - RAA-s(78b), success, N1-N6 (no label weight)" % W, ar, "raasb", "N", "none")
        pair("X2 %s - RAA-L, success, N1-N6 (no label weight)" % W, ar, "raal", "N", "none")
        pair("X3 %s - RAA-sL, bottle pairings N4-N6 (no label weight)" % W, ar, "raasl", "bottle", "none")
        pair("X4 %s - RAA-sL, cheese pairings N1-N3 (no label weight)" % W, ar, "raasl", "cheese", "none")
        pair("X5 %s - RAA-sL, placed given right grasp, N1-N6 (no label weight)" % W, ar, "raasl", "N", "none", what="pgg")
        pair("X6 %s - rgp, success, K1-K6 (no label weight)" % W, ar, "rgp", "K", "none")
    pair("X7 W4 - W3, success, N1-N6 (no label weight)", "raasw4", "raasw3", "N", "none")
    pair("X8 W4 - W3, success, K1-K6 (no label weight)", "raasw4", "raasw3", "K", "none")
    for w in (3, 4):
        pair("X9 W%d - W1 (SR 84), success, N1-N6 (no label weight)" % w, "raasw%d" % w, "raasw1", "N", "none")
        pair("X10 W%d - W1 (SR 84), success, K1-K6 (no label weight)" % w, "raasw%d" % w, "raasw1", "K", "none")
    res["primary"] = {"W3": V["A1 PRIMARY-A W3 - RAA-sL, success, N1-N6"]["label"], "W4": V["A1 PRIMARY-A W4 - RAA-sL, success, N1-N6"]["label"]}
    res["overall"] = "UNDERFIT (R0 under .80 on K1-K6)" if res["fit_gate"]["r0_underfit"] else res["primary"]
    print("[raasw3-read]", json.dumps(res), flush=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raasw3-read] -> %s" % outp, flush=True)


def _vec(d, rows):
    return None if d is None else np.array([d.get(k, np.nan) for k in rows], float)


def readb(a):
    sfx = ("_" + a.tag.lstrip("_")) if a.tag else ""
    o2 = os.path.join(OUT, "raasw3_wa_gated%s.json" % sfx)
    outs = {w: os.path.join(OUT, "raasw3_wa_read_w%d%s.json" % (w, sfx)) for w in (3, 4)}
    for o in list(outs.values()) + [o2]:
        if os.path.exists(o):
            raise SystemExit("%s exists; pass --tag with a new suffix" % o)
    meta = json.load(open(WR.META, encoding="utf-8"))
    rows7, rows6 = meta["states"]["f7"]["main"], meta["states"]["f6"]["main"]
    rows_of = lambda t: rows7 if t <= 7 else rows6                           # noqa: E731
    G = {}
    for w in (3, 4):
        wd = os.path.join(OUT, "wasw%d" % w)
        try:
            WR.read(argparse.Namespace(phase="main", wa=wd, out=outs[w]))
        except KeyError as e:                                                # wa_read's print loop on a contrast without delta (noMe not rolled), after the json is written
            if not os.path.exists(outs[w]):
                raise
            print("[raasw3-read] wa_read print loop KeyError %r after writing %s (noMe contrasts, as SR 80 / 82); continuing from the json" % (e, outs[w]), flush=True)
        r = json.load(open(outs[w], encoding="utf-8"))
        gt = {k: (v["label"] == "TESTABLE") for k, v in r["gates"].items()}
        inp = r["primary_inputs"]
        allg = all(gt.values()) and len(gt) == 3
        shown_rule = inp["D1"] == "SUPPORTED" and inp["D2"] == "SUPPORTED" and inp["D3a"] == "EQUIVALENT" and inp["D3b"] == "EQUIVALENT"
        g = dict(wa_read_primary=r["primary"], inputs=inp, gates=r["gates"],
                 testable={"D1 (F7: unique-name gate)": gt.get("F7_full_unique_controls"), "D3b (F7: unique-name gate)": gt.get("F7_full_unique_controls"),
                           "D2 (F6_b: F6_a gate)": gt.get("F6_full_F6_a"), "D3a (F6_b: F6_a gate)": gt.get("F6_full_F6_a"), "DK (K floor gate)": gt.get("floor_full_K")},
                 primary_gated=("DISSOCIATION SHOWN" if (shown_rule and allg) else ("UNTESTABLE (D1, D2, D3a, D3b pass, a gate fails)" if shown_rule else r["primary"])),
                 deltas={k: dict(delta=v.get("delta"), ci=v.get("ci"), label=v.get("label")) for k, v in r["contrasts"].items()})
        # extra registered / descriptive contrasts against SR 82's RAA-sL (same states) and the frozen I (wa_read's reference files)
        wad = os.path.abspath(os.path.join(OUT, "wasw%d" % w))
        X = {}
        for arm, d in (("full_W", wad), ("full_L", WASL), ("noM_W", wad), ("noM_L", WASL), ("full_W1", os.path.abspath(os.path.join(OUT, "wasw1")))):
            for t in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
                dd, _ = WR.load(os.path.join(d, "%s_%d.json" % (arm.split("_")[0], t)), rows_of(t))
                X[(arm, t)] = _vec(dd, rows_of(t))
        for t in WR.KEYS:
            dd, _ = WR.load(os.path.join(WR.REF, "pi05_I_%d.json" % t), rows_of(t))
            X[("I", t)] = _vec(dd, rows_of(t))
            dd, _ = WR.load(os.path.join(WR.REF, "pi05_rgp_%d.json" % t), rows_of(t))
            X[("rgp", t)] = _vec(dd, rows_of(t))
        ex = {}

        def cell_fn(arm, cell):
            if cell == "F7":
                xs = [X[(arm, t)] for t in (0, 1, 2, 3)]
                return None if any(x is None for x in xs) else (lambda I, xs=xs: float(np.mean([np.nanmean(x[I[0]]) for x in xs])))
            if cell == "K":
                a_, b_ = X[(arm, 10)], X[(arm, 7)]
                return None if a_ is None or b_ is None else (lambda I: 0.5 * (float(np.nanmean(a_[I[0]])) + float(np.nanmean(b_[I[1]]))))
            t = {"F6_b": 8, "F6_a": 9}[cell]
            x = X[(arm, t)]
            return None if x is None else (lambda I, x=x: float(np.nanmean(x[I[0]])))
        nsz = {"F7": [len(rows7)], "F6_b": [len(rows6)], "F6_a": [len(rows6)], "K": [len(rows6), len(rows7)]}

        def contrast(name, a1, a2, cell, kind):
            fa, fb = cell_fn(a1, cell), cell_fn(a2, cell)
            if fa is None or fb is None:
                ex[name] = dict(label="MISSING")
                return
            ci = WR.bootm(lambda I: fa(I) - fb(I), nsz[cell])
            lab = None
            if kind == "support" and ci[1] is not None:
                lab = "SUPPORTED" if ci[1] > 0 else ("REVERSED" if ci[2] < 0 else "NOT SUPPORTED")
            ex[name] = dict(delta=ci[0], ci=[ci[1], ci[2]], label=lab, a=a1, b=a2, cell=cell)
        contrast("B1 full_W - full RAA-sL (SR 82), F6_b", "full_W", "full_L", "F6_b", "support")
        contrast("B2 full_W - frozen I, F6_b", "full_W", "I", "F6_b", "support")
        contrast("B3a full_W - full RAA-sL, F7 pool", "full_W", "full_L", "F7", None)
        contrast("B3b full_W - full RAA-sL, F6_a", "full_W", "full_L", "F6_a", None)
        contrast("B3c full_W - full RAA-sL, K1t + K2t", "full_W", "full_L", "K", None)
        contrast("B3d full_W - rgp, F6_b", "full_W", "rgp", "F6_b", None)
        contrast("B3e full_W - rgp, F7 pool", "full_W", "rgp", "F7", None)
        contrast("B3f full_W - frozen I, F7 pool", "full_W", "I", "F7", None)
        contrast("B3g noM_W - noM RAA-sL (SR 82), F6_b", "noM_W", "noM_L", "F6_b", None)
        contrast("B3h full_W - full W1 (SR 84), F6_b", "full_W", "full_W1", "F6_b", None)
        contrast("B3i full_W - full W1 (SR 84), F7 pool", "full_W", "full_W1", "F7", None)
        contrast("B3j full_W - full W1 (SR 84), K1t + K2t", "full_W", "full_W1", "K", None)
        contrast("B3k full_W - full W1 (SR 84), F6_a", "full_W", "full_W1", "F6_a", None)
        same = {}
        for t, nm in ((8, "F6_b"), (0, "F7_relL"), (1, "F7_relR"), (2, "F7_pos"), (3, "F7_neg")):
            a_, b_ = X[("noM_W", t)], X[("noM_L", t)]
            if a_ is not None and b_ is not None:
                same[nm] = dict(identical=int(np.sum(a_ == b_)), n=int(len(a_)))
        g["extra"] = ex
        g["noM_replication_states_identical"] = same
        G["W%d" % w] = g
        print("[raasw3-read] W%d PRIMARY (gated): %s; gates %s; B1 %s; B2 %s; noM replication %s" % (w, g["primary_gated"], {k: v["label"] for k, v in r["gates"].items()},
                                                                                              ex.get("B1 full_W - full RAA-sL (SR 82), F6_b"), ex.get("B2 full_W - frozen I, F6_b"), same), flush=True)
    with open(o2, "w", encoding="utf-8") as fh:
        json.dump(G, fh, indent=1)
    print("[raasw3-read] -> %s, %s" % (list(outs.values()), o2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read", "readb", "selftest"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"read": read, "readb": readb, "selftest": selftest}[a.what](a)

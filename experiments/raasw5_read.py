"""SR 87 read (docs/notes/raasw5_plan.md; m3_prereg raasw5).  Numpy only (openpi venv; compute node).  The FRESH-state confirmatory run of W5 (W4 + C1 release-point aim + C2 handoff radius) with the attribution arms W5c2 (W4 + C2)
and W5c1 (W4 + C1) and the baselines RAA-sL, W4, rgp, I re-rolled on the same fresh states.  Machinery: raa_read.diff / label (paired bootstrap B = 2000, rng 0, tasks and states resampled) and wa_read.read / bootm (pointed at the
fresh rgc states and result directories); nothing here reads an SR 70-86 rollout.
Results layout (results/flawed/raasw5/):  a_<arm>/<arm>_main_<tid>.json (Phase A, arms w5 w5c2 w5c1 w4 raasl; n2 tids, trials 0-59 = k 580-639), a_rgp/pi05_rgp_<tid>.json;
                                          b_<arm>/{full,noR,noM}_<tid>.json (Phase B; arms w5 w5c2 w5c1 w4 raasl), b_ref/pi05_{I,rgp}_<tid>.json   (rgc tids 0-10, rows of results/flawed/rg_comp_x main).

PHASE A (N1-N6 = tids 16-26, K1-K6 = tids 0-10; 60 states):  A1 W5 - RAA-sL N1-N6 (SUPPORTED iff lower > 0, REVERSED iff upper < 0);  A2a W5 - rgp N1-N6 (NON-INFERIOR iff lower > -.10), A2b (beats, lower > 0);
  AK W5 - RAA-sL K1-K6 (NON-INFERIOR iff lower > -.05);  A3 W5 - W4 K1-K6 (C1 + C2 effect; SUPPORTED iff lower > 0);  descriptive: per cell, W5 - W4 N, W5c2 - W4, W5c1 - W4, W5 - W5c2, W5 - W5c1, K5 alone.
PHASE B:  wa_read.read on b_w5 (D1', D2', D3a, D3b, DK, gates, gated PRIMARY);  B1 full_W5 - full RAA-sL F6_b, B2 full_W5 - frozen I F6_b (SUPPORTED iff lower > 0);  C2 attribution W5c2 - W4 on the unique-name controls (tids 4-6 pooled)
  and F6_a (descriptive, no label weight); B3 descriptive contrasts.
    python experiments/raasw5_read.py read [--tag T]     -> results/flawed/raasw5_read[_T].json, raasw5_wa_read_w5[_T].json, raasw5_wa_gated[_T].json (refuse to overwrite)
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
BASE = os.path.join(OUT, "raasw5")
RGCX = os.path.join(OUT, "rg_comp_x", "rg_comp_x_tasks.json")
KNOWN, NEW = RR.KNOWN, RR.NEW
ARMS_A = ("w5", "w5c2", "w5c1", "w4", "raasl")
TRIALS = list(range(60))
MARGIN_K, MARGIN_RG = 0.05, 0.10
LAB = {"w5": "W5", "w5c2": "W5c2", "w5c1": "W5c1", "w4": "W4", "raasl": "RAA-sL", "rgp": "rgp", "I": "I"}


def tid_of(meta, comp):
    return next(t["tid"] for t in meta["tasks"] if t["compound"] == comp and t["layout"] == "familiar")


def grid(meta, arm, comps, what):
    out = []
    for c in comps:
        tid = tid_of(meta, c)
        obj = next(t["object"] for t in meta["tasks"] if t["tid"] == tid)
        p = os.path.join(BASE, "a_rgp", "pi05_rgp_%d.json" % tid) if arm == "rgp" else os.path.join(BASE, "a_" + arm, "%s_main_%d.json" % (arm, tid))
        by = RR.episodes(p, tid)
        assert sorted(by) == TRIALS, (arm, c, sorted(by))
        out.append([float(bool(by[k]["success"])) if what == "succ" else float(bool(by[k]["success"]) or by[k].get("bound") == obj) for k in TRIALS])
    return np.array(out)


def have_a(arm):
    meta = RR.n2()
    ps = [(os.path.join(BASE, "a_rgp", "pi05_rgp_%d.json" % tid_of(meta, c)) if arm == "rgp" else os.path.join(BASE, "a_" + arm, "%s_main_%d.json" % (arm, tid_of(meta, c)))) for c in KNOWN + NEW]
    return all(os.path.exists(p) for p in ps)


def read(a):
    sfx = ("_" + a.tag.lstrip("_")) if a.tag else ""
    outp = os.path.join(OUT, "raasw5_read%s.json" % sfx)
    o2 = os.path.join(OUT, "raasw5_wa_gated%s.json" % sfx)
    ow = {w: os.path.join(OUT, "raasw5_wa_read_%s%s.json" % (w, sfx)) for w in ("w5",)}
    for o in [outp, o2] + list(ow.values()):
        if os.path.exists(o):
            raise SystemExit("%s exists; pass --tag with a new suffix" % o)
    meta = RR.n2()
    arms = [x for x in ARMS_A + ("rgp",) if have_a(x)]
    assert all(x in arms for x in ("w5", "w4", "raasl", "rgp")), "missing Phase A rollout files for the registered arms: %s" % arms
    G = {}
    for arm in arms:
        for grp, comps in (("K", KNOWN), ("N", NEW), ("cheese", RR.CHEESE), ("bottle", RR.BOTTLE)):
            G[(arm, grp)] = (grid(meta, arm, comps, "succ"), grid(meta, arm, comps, "bound"))
    res = dict(states="fresh n2 familiar states k = 580-639 (trial t = k - 580)", B=RR.B, arms=arms, success={}, grasped_right={}, per_task={}, verdicts={})
    for arm in arms:
        for grp in ("K", "N", "cheese", "bottle"):
            s, b = G[(arm, grp)]
            res["success"]["%s|%s" % (arm, grp)] = float(s.mean())
            res["grasped_right"]["%s|%s" % (arm, grp)] = float(b.mean())
        for c in KNOWN + NEW:
            s = grid(meta, arm, [c], "succ")
            res["per_task"]["%s|%s" % (arm, c)] = dict(success=float(s.mean()), grasped_right=float(grid(meta, arm, [c], "bound").mean()))
    V = res["verdicts"]

    def pair(name, ra, rb, grp, kind, margin=0.0, what="mean", comps=None):
        if ra not in arms or rb not in arms:
            V[name] = dict(label="MISSING")
            return
        if comps is None:
            xs, ys = G[(ra, grp)], G[(rb, grp)]
        else:
            xs = (grid(meta, ra, comps, "succ"), grid(meta, ra, comps, "bound"))
            ys = (grid(meta, rb, comps, "succ"), grid(meta, rb, comps, "bound"))
        d = RR.diff(xs, ys, what)
        lab = None
        if kind == "support":
            lo, hi = d["ci"]
            lab = "SUPPORTED" if lo > 0 else ("REVERSED" if hi < 0 else "NOT SUPPORTED")
        elif kind != "none":
            lab = RR.label(d, margin, kind)
        V[name] = dict(arm_a=ra, arm_b=rb, group=grp if comps is None else comps, metric="success" if what == "mean" else "placed given right grasp", **d, label=lab)
    pair("A1 PRIMARY-A W5 - RAA-sL, success, N1-N6", "w5", "raasl", "N", "support")
    pair("A2a W5 - rgp, success, N1-N6 (non-inferiority -.10)", "w5", "rgp", "N", "noninf", MARGIN_RG)
    pair("A2b W5 - rgp, success, N1-N6 (beats)", "w5", "rgp", "N", "help")
    pair("AK W5 - RAA-sL, success, K1-K6 (non-inferiority -.05)", "w5", "raasl", "K", "noninf", MARGIN_K)
    pair("A3 W5 - W4, success, K1-K6 (C1 + C2 effect)", "w5", "w4", "K", "support")
    pair("X1 W5 - W4, success, N1-N6", "w5", "w4", "N", "none")
    pair("X2 W5 - W4, success, K5 alone", "w5", "w4", "K", "none", comps=["K5"])
    pair("X3 W5 - RAA-sL, success, K5 alone", "w5", "raasl", "K", "none", comps=["K5"])
    pair("X4 W5 - rgp, success, K1-K6", "w5", "rgp", "K", "none")
    pair("X5 W5 - W4 + RAA-sL: W4 - RAA-sL, success, K1-K6 (the SR 85 cost, on fresh states)", "w4", "raasl", "K", "none")
    pair("X6 W4 - RAA-sL, success, N1-N6", "w4", "raasl", "N", "none")
    for x, y in (("w5c2", "w4"), ("w5c1", "w4"), ("w5", "w5c2"), ("w5", "w5c1")):
        for grp in ("N", "K"):
            pair("C %s - %s, success, %s1-%s6" % (LAB[x], LAB[y], grp, grp), x, y, grp, "none")
    pair("X7 W5 - RAA-sL, placed given right grasp, N1-N6", "w5", "raasl", "N", "none", what="pgg")
    pair("X8 W5 - RAA-sL, bottle pairings N4-N6", "w5", "raasl", "bottle", "none")
    pair("X9 W5 - RAA-sL, cheese pairings N1-N3", "w5", "raasl", "cheese", "none")
    res["primary"] = V["A1 PRIMARY-A W5 - RAA-sL, success, N1-N6"]["label"]
    print("[raasw5-read]", json.dumps(res), flush=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raasw5-read] -> %s" % outp, flush=True)

    # ---------------- Phase B
    WR.META = RGCX
    WR.REF = os.path.join(BASE, "b_ref")
    wbm = json.load(open(RGCX, encoding="utf-8"))
    rows7, rows6 = wbm["states"]["f7"]["main"], wbm["states"]["f6"]["main"]
    rows_of = lambda t: rows7 if t <= 7 else rows6                           # noqa: E731
    G2 = {}
    wd = os.path.join(BASE, "b_w5")
    try:
        WR.read(argparse.Namespace(phase="main", wa=wd, out=ow["w5"]))
    except KeyError as e:                                                    # wa_read's print loop on a contrast without delta (noMe not rolled), after the json is written
        if not os.path.exists(ow["w5"]):
            raise
        print("[raasw5-read] wa_read print loop KeyError %r after writing %s (noMe contrasts, as SR 80 / 82 / 85); continuing from the json" % (e, ow["w5"]), flush=True)
    r = json.load(open(ow["w5"], encoding="utf-8"))
    gt = {k: (v["label"] == "TESTABLE") for k, v in r["gates"].items()}
    inp = r["primary_inputs"]
    allg = all(gt.values()) and len(gt) == 3
    shown_rule = inp["D1"] == "SUPPORTED" and inp["D2"] == "SUPPORTED" and inp["D3a"] == "EQUIVALENT" and inp["D3b"] == "EQUIVALENT"
    g = dict(wa_read_primary=r["primary"], inputs=inp, gates=r["gates"],
             primary_gated=("DISSOCIATION SHOWN" if (shown_rule and allg) else ("UNTESTABLE (D1, D2, D3a, D3b pass, a gate fails)" if shown_rule else r["primary"])),
             deltas={k: dict(delta=v.get("delta"), ci=v.get("ci"), label=v.get("label")) for k, v in r["contrasts"].items()})
    X = {}
    for arm in ARMS_A:
        for t in range(11):
            dd, _ = WR.load(os.path.join(BASE, "b_" + arm, "full_%d.json" % t), rows_of(t))
            X[(arm, t)] = None if dd is None else np.array([dd.get(k, np.nan) for k in rows_of(t)], float)
    for arm in ("w5",):
        for kind in ("noM", "noR"):
            for t in range(11):
                dd, _ = WR.load(os.path.join(BASE, "b_" + arm, "%s_%d.json" % (kind, t)), rows_of(t))
                X[(arm + kind, t)] = None if dd is None else np.array([dd.get(k, np.nan) for k in rows_of(t)], float)
    for ref in ("I", "rgp"):
        for t in range(11):
            dd, _ = WR.load(os.path.join(WR.REF, "pi05_%s_%d.json" % (ref, t)), rows_of(t))
            X[(ref, t)] = None if dd is None else np.array([dd.get(k, np.nan) for k in rows_of(t)], float)
    ex = {}

    def cell_fn(arm, cell):
        if cell == "F7":
            xs = [X[(arm, t)] for t in (0, 1, 2, 3)]
            return None if any(x is None for x in xs) else (lambda I, xs=xs: float(np.mean([np.nanmean(x[I[0]]) for x in xs])))
        if cell == "K":
            a_, b_ = X[(arm, 10)], X[(arm, 7)]
            return None if a_ is None or b_ is None else (lambda I: 0.5 * (float(np.nanmean(a_[I[0]])) + float(np.nanmean(b_[I[1]]))))
        if cell == "controls":
            xs = [X[(arm, t)] for t in (4, 5, 6)]
            return None if any(x is None for x in xs) else (lambda I, xs=xs: float(np.mean([np.nanmean(x[I[0]]) for x in xs])))
        t = {"F6_b": 8, "F6_a": 9, "K5t": None}.get(cell)
        if t is None:
            t = int(cell.split("_")[-1])
        x = X[(arm, t)]
        return None if x is None else (lambda I, x=x: float(np.nanmean(x[I[0]])))
    nsz = {"F7": [len(rows7)], "controls": [len(rows7)], "F6_b": [len(rows6)], "F6_a": [len(rows6)], "K": [len(rows6), len(rows7)]}

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
    contrast("B1 full_W5 - full RAA-sL, F6_b", "w5", "raasl", "F6_b", "support")
    contrast("B2 full_W5 - frozen I, F6_b", "w5", "I", "F6_b", "support")
    contrast("B3a full_W5 - full RAA-sL, F7 pool", "w5", "raasl", "F7", None)
    contrast("B3b full_W5 - full RAA-sL, F6_a", "w5", "raasl", "F6_a", None)
    contrast("B3c full_W5 - full RAA-sL, K1t + K2t", "w5", "raasl", "K", None)
    contrast("B3d full_W5 - full RAA-sL, unique-name controls (tids 4-6)", "w5", "raasl", "controls", None)
    contrast("B3e full_W5 - rgp, F6_b", "w5", "rgp", "F6_b", None)
    contrast("B3f full_W5 - rgp, F7 pool", "w5", "rgp", "F7", None)
    contrast("B3g full_W5 - frozen I, F7 pool", "w5", "I", "F7", None)
    contrast("B3h full_W5 - full W4, F6_b", "w5", "w4", "F6_b", None)
    contrast("B3i full_W5 - full W4, F7 pool", "w5", "w4", "F7", None)
    contrast("B3j full_W5 - full W4, unique-name controls", "w5", "w4", "controls", None)
    contrast("B3k full_W5 - full W4, F6_a", "w5", "w4", "F6_a", None)
    contrast("B3l full_W5 - full W4, K1t + K2t", "w5", "w4", "K", None)
    for cell in ("controls", "F6_a", "F7", "F6_b", "K"):
        contrast("C2 attribution: full_W5c2 - full_W4, %s" % cell, "w5c2", "w4", cell, None)
        contrast("C1 attribution: full_W5 - full_W5c2, %s" % cell, "w5", "w5c2", cell, None)
        contrast("C1 only: full_W5c1 - full_W4, %s" % cell, "w5c1", "w4", cell, None)
        contrast("W5 - W5c1, %s" % cell, "w5", "w5c1", cell, None)
    contrast("F6_b W4 - RAA-sL", "w4", "raasl", "F6_b", None)
    contrast("unique-name controls W4 - RAA-sL", "w4", "raasl", "controls", None)
    g["extra"] = ex
    g["per_cell_success"] = {("%s|%d" % (arm, t)): (None if X.get((arm, t)) is None else float(np.nanmean(X[(arm, t)]))) for arm in ARMS_A + ("w5noM", "w5noR", "I", "rgp") for t in range(11)}
    g["n_states"] = dict(f7=len(rows7), f6=len(rows6))
    G2["W5"] = g
    print("[raasw5-read] W5 PRIMARY (gated): %s; gates %s; B1 %s; B2 %s" % (g["primary_gated"], {k: v["label"] for k, v in r["gates"].items()}, ex.get("B1 full_W5 - full RAA-sL, F6_b"), ex.get("B2 full_W5 - frozen I, F6_b")), flush=True)
    with open(o2, "w", encoding="utf-8") as fh:
        json.dump(G2, fh, indent=1)
    print("[raasw5-read] -> %s, %s" % (ow["w5"], o2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read",))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    read(a)

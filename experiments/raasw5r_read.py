"""SR 91 read (docs/notes/sr91_plan.md; m3_prereg raasw5r).  Numpy only (openpi venv; compute node).  The registered REPLICATION of W5 (where_own 5, SR 87 design, no code change) on the FOURTH fresh-state set,
with RAA-sL, rgp, frozen I re-rolled on the same states, and the SECONDARY pooled analysis over the three fresh sets W5 was evaluated on without design changes after it was frozen (SR 87, SR 90, SR 91).
Machinery: raa_read.diff / label (paired bootstrap B = 2000, rng 0, tasks and states resampled) and wa_read.read / bootm (pointed at the fourth fresh rgc states and result directories).
Results layout (results/flawed/raasw5r/):  a_<arm>/<arm>_main_<tid>.json (Phase A, arms w5 raasl; n2 tids, trials 0-59 = k 700-759), a_rgp/pi05_rgp_<tid>.json;
                                           b_w5/{full,noR,noM}_<tid>.json, b_raasl/full_<tid>.json, b_ref/pi05_{I,rgp}_<tid>.json   (rgc tids; rows of results/flawed/rg_comp_z main).

REGISTERED (per set, SR 91 only):  Phase A  A2a PRIMARY-R W5 - rgp N1-N6 (NON-INFERIOR iff lower > -.10), A2b (beats, lower > 0), A1 W5 - RAA-sL N1-N6 (SUPPORTED iff lower > 0, REVERSED iff upper < 0),
  AK W5 - RAA-sL K1-K6 (NON-INFERIOR iff lower > -.05).  Phase B (wa_read.read on b_w5): D1', D2', D3a, D3b, DK, gates (K floor .70, unique-name controls .50, F6_a .50), gated PRIMARY; B1 W5 - RAA-sL F6_b, B2 W5 - frozen I F6_b.
SECONDARY (pooled over SR 87 / SR 90 / SR 91; every set's data seen except SR 91's): A1, A2a, A2b, AK on N1-N6 / K1-K6 and B1, B2 on F6_b; per draw each set resamples tasks, then states, independently; statistic = mean over the sets of
  the per-set paired difference.   Descriptive pooled: K5 alone, per-set deltas.
    python experiments/raasw5r_read.py read [--tag T]     -> results/flawed/raasw5r_read[_T].json, raasw5r_wa_read_w5[_T].json, raasw5r_wa_gated[_T].json, raasw5r_pooled[_T].json (refuse to overwrite)
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
BASE = os.path.join(OUT, "raasw5r")
RGCZ = os.path.join(OUT, "rg_comp_z", "rg_comp_z_tasks.json")
KNOWN, NEW = RR.KNOWN, RR.NEW
ARMS_A = ("w5", "raasl")
TRIALS = list(range(60))
MARGIN_K, MARGIN_RG = 0.05, 0.10
LAB = {"w5": "W5", "raasl": "RAA-sL", "rgp": "rgp", "I": "I"}
SETS = {"SR87": (os.path.join(OUT, "raasw5"), os.path.join(OUT, "rg_comp_x", "rg_comp_x_tasks.json")),
        "SR90": (os.path.join(OUT, "raasw6"), os.path.join(OUT, "rg_comp_y", "rg_comp_y_tasks.json")),
        "SR91": (BASE, RGCZ)}


def tid_of(meta, comp):
    return next(t["tid"] for t in meta["tasks"] if t["compound"] == comp and t["layout"] == "familiar")


def apath(base, meta, arm, c):
    tid = tid_of(meta, c)
    return os.path.join(base, "a_rgp", "pi05_rgp_%d.json" % tid) if arm == "rgp" else os.path.join(base, "a_" + arm, "%s_main_%d.json" % (arm, tid))


def grid(meta, arm, comps, what, base=BASE):
    out = []
    for c in comps:
        tid = tid_of(meta, c)
        obj = next(t["object"] for t in meta["tasks"] if t["tid"] == tid)
        by = RR.episodes(apath(base, meta, arm, c), tid)
        assert sorted(by) == TRIALS, (base, arm, c, sorted(by))
        out.append([float(bool(by[k]["success"])) if what == "succ" else float(bool(by[k]["success"]) or by[k].get("bound") == obj) for k in TRIALS])
    return np.array(out)


def have_a(arm, base=BASE):
    meta = RR.n2()
    return all(os.path.exists(apath(base, meta, arm, c)) for c in KNOWN + NEW)


def pooled(outp, only=None):
    """SECONDARY pooled analysis over SR 87 / SR 90 / SR 91 (the first two are SEEN: pre-registration states this)."""
    meta = RR.n2()
    PG = {}
    for sname, (base, rgc) in [(k, v) for k, v in SETS.items() if only is None or k in only]:
        for arm in ("w5", "raasl", "rgp"):
            for grp, comps in (("K", KNOWN), ("N", NEW), ("K5", ["K5"])):
                PG[(sname, arm, grp)] = grid(meta, arm, comps, "succ", base)
    B6 = {}
    for sname, (base, rgc) in [(k, v) for k, v in SETS.items() if only is None or k in only]:
        rows6 = json.load(open(rgc, encoding="utf-8"))["states"]["f6"]["main"]
        for arm, fn in (("w5", "b_w5/full_8.json"), ("raasl", "b_raasl/full_8.json"), ("I", "b_ref/pi05_I_8.json")):
            d, _ = WR.load(os.path.join(base, fn), rows6)
            assert d is not None and all(k in d for k in rows6), (sname, arm)
            B6[(sname, arm)] = np.array([d[k] for k in rows6], float)[None, :]            # [1 task, states]

    def pool_diff(get_a, get_b, names=None):
        names = tuple(only or SETS) if names is None else names
        rng = np.random.default_rng(0)
        per = {}

        def stat(j, t, s):
            return float(get_a(j)[t][:, s].mean()) - float(get_b(j)[t][:, s].mean())
        v = []
        for _ in range(RR.B):
            tot = 0.0
            for j in names:
                nt, ns = get_a(j).shape
                t = rng.choice(nt, nt)
                s = rng.choice(ns, ns)
                tot += stat(j, t, s)
            v.append(tot / len(names))
        for j in names:
            nt, ns = get_a(j).shape
            per[j] = stat(j, np.arange(nt), np.arange(ns))
        d0 = float(np.mean(list(per.values())))
        return dict(delta=d0, ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], per_set=per,
                    a=float(np.mean([get_a(j).mean() for j in names])), b=float(np.mean([get_b(j).mean() for j in names])))
    res = dict(note="SECONDARY pooled analysis over SR 87 / SR 90 / SR 91; SR 87 and SR 90 data were SEEN before registration", B=RR.B, sets=list(only or SETS), verdicts={})
    V = res["verdicts"]

    def add(name, a, b, grp, kind, margin=0.0):
        d = pool_diff(lambda j: PG[(j, a, grp)], lambda j: PG[(j, b, grp)])
        d["label"] = None if kind == "none" else (("SUPPORTED" if d["ci"][0] > 0 else ("REVERSED" if d["ci"][1] < 0 else "NOT SUPPORTED")) if kind == "support" else RR.label(d, margin, kind))
        V[name] = d
    add("P-A1 W5 - RAA-sL, N1-N6 (pooled)", "w5", "raasl", "N", "support")
    add("P-A2a W5 - rgp, N1-N6 (pooled; non-inferiority -.10)", "w5", "rgp", "N", "noninf", MARGIN_RG)
    add("P-A2b W5 - rgp, N1-N6 (pooled; beats)", "w5", "rgp", "N", "help")
    add("P-AK W5 - RAA-sL, K1-K6 (pooled; non-inferiority -.05)", "w5", "raasl", "K", "noninf", MARGIN_K)
    add("P-X1 W5 - rgp, K1-K6 (pooled)", "w5", "rgp", "K", "none")
    add("P-X2 W5 - RAA-sL, K5 alone (pooled)", "w5", "raasl", "K5", "none")
    for name, b in (("P-B1 W5 - RAA-sL, F6_b (pooled)", "raasl"), ("P-B2 W5 - frozen I, F6_b (pooled)", "I")):
        d = pool_diff(lambda j: B6[(j, "w5")], lambda j: B6[(j, b)])
        d["label"] = "SUPPORTED" if d["ci"][0] > 0 else ("REVERSED" if d["ci"][1] < 0 else "NOT SUPPORTED")
        V[name] = d
    print("[raasw5r-pooled]", json.dumps(res), flush=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raasw5r-pooled] -> %s" % outp, flush=True)


def read(a):
    sfx = ("_" + a.tag.lstrip("_")) if a.tag else ""
    outp = os.path.join(OUT, "raasw5r_read%s.json" % sfx)
    o2 = os.path.join(OUT, "raasw5r_wa_gated%s.json" % sfx)
    o3 = os.path.join(OUT, "raasw5r_pooled%s.json" % sfx)
    ow = {w: os.path.join(OUT, "raasw5r_wa_read_%s%s.json" % (w, sfx)) for w in ("w5",)}
    for o in [outp, o2, o3] + list(ow.values()):
        if os.path.exists(o):
            raise SystemExit("%s exists; pass --tag with a new suffix" % o)
    meta = RR.n2()
    arms = [x for x in ARMS_A + ("rgp",) if have_a(x)]
    assert all(x in arms for x in ("w5", "raasl", "rgp")), "missing Phase A rollout files for the registered arms: %s" % arms
    G = {}
    for arm in arms:
        for grp, comps in (("K", KNOWN), ("N", NEW), ("cheese", RR.CHEESE), ("bottle", RR.BOTTLE)):
            G[(arm, grp)] = (grid(meta, arm, comps, "succ"), grid(meta, arm, comps, "bound"))
    res = dict(states="fresh n2 familiar states k = 700-759 (trial t = k - 700)", B=RR.B, arms=arms, success={}, grasped_right={}, per_task={}, verdicts={})
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
    pair("A2a PRIMARY-R W5 - rgp, success, N1-N6 (non-inferiority -.10)", "w5", "rgp", "N", "noninf", MARGIN_RG)
    pair("A2b W5 - rgp, success, N1-N6 (beats)", "w5", "rgp", "N", "help")
    pair("A1 W5 - RAA-sL, success, N1-N6", "w5", "raasl", "N", "support")
    pair("AK W5 - RAA-sL, success, K1-K6 (non-inferiority -.05)", "w5", "raasl", "K", "noninf", MARGIN_K)
    pair("X2 W5 - RAA-sL, success, K5 alone", "w5", "raasl", "K", "none", comps=["K5"])
    pair("X4 W5 - rgp, success, K1-K6", "w5", "rgp", "K", "none")
    pair("X7 W5 - RAA-sL, placed given right grasp, N1-N6", "w5", "raasl", "N", "none", what="pgg")
    pair("X8 W5 - RAA-sL, bottle pairings N4-N6", "w5", "raasl", "bottle", "none")
    pair("X9 W5 - RAA-sL, cheese pairings N1-N3", "w5", "raasl", "cheese", "none")
    res["primary"] = V["A2a PRIMARY-R W5 - rgp, success, N1-N6 (non-inferiority -.10)"]["label"]
    print("[raasw5r-read]", json.dumps(res), flush=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raasw5r-read] -> %s" % outp, flush=True)

    # ---------------- Phase B
    WR.META = RGCZ
    WR.REF = os.path.join(BASE, "b_ref")
    wbm = json.load(open(RGCZ, encoding="utf-8"))
    rows7, rows6 = wbm["states"]["f7"]["main"], wbm["states"]["f6"]["main"]
    rows_of = lambda t: rows7 if t <= 7 else rows6                           # noqa: E731
    G2 = {}
    wd = os.path.join(BASE, "b_w5")
    try:
        WR.read(argparse.Namespace(phase="main", wa=wd, out=ow["w5"]))
    except KeyError as e:                                                    # wa_read's print loop on a contrast without delta (noMe not rolled), after the json is written
        if not os.path.exists(ow["w5"]):
            raise
        print("[raasw5r-read] wa_read print loop KeyError %r after writing %s (noMe contrasts, as SR 80 / 82 / 85 / 87 / 90); continuing from the json" % (e, ow["w5"]), flush=True)
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
    for kind in ("noM", "noR"):
        for t in range(11):
            dd, _ = WR.load(os.path.join(BASE, "b_w5", "%s_%d.json" % (kind, t)), rows_of(t))
            X[("w5" + kind, t)] = None if dd is None else np.array([dd.get(k, np.nan) for k in rows_of(t)], float)
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
    g["extra"] = ex
    g["per_cell_success"] = {("%s|%d" % (arm, t)): (None if X.get((arm, t)) is None else float(np.nanmean(X[(arm, t)]))) for arm in ARMS_A + ("w5noM", "w5noR", "I", "rgp") for t in range(11)}
    g["n_states"] = dict(f7=len(rows7), f6=len(rows6))
    G2["W5"] = g
    print("[raasw5r-read] W5 PRIMARY (gated): %s; gates %s; B1 %s; B2 %s" % (g["primary_gated"], {k: v["label"] for k, v in r["gates"].items()}, ex.get("B1 full_W5 - full RAA-sL, F6_b"), ex.get("B2 full_W5 - frozen I, F6_b")), flush=True)
    with open(o2, "w", encoding="utf-8") as fh:
        json.dump(G2, fh, indent=1)
    print("[raasw5r-read] -> %s, %s" % (ow["w5"], o2), flush=True)
    pooled(o3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read", "pooled"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="", help="pooled mode only: comma-free list of sets joined by +, e.g. SR87+SR90 (a dry run before SR 91 data exist)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.what == "read":
        read(a)
    else:                                                                    # pooled only (e.g. a rerun without the registered read)
        sfx = ("_" + a.tag.lstrip("_")) if a.tag else ""
        o3 = os.path.join(OUT, "raasw5r_pooled%s.json" % sfx)
        o3 = a.out or o3
        assert not os.path.exists(o3), o3
        pooled(o3, only=a.only.split("+") if a.only else None)

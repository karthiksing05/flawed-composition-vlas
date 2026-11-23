"""RVW5 reader (docs/notes/plan_review_atlas.md 3.3; prereg experiments/prereg/rvw5_pred.txt).  Numpy only (openpi venv, compute node via sbatch).

Phase A (N1-N6 / K1-K6 on the 5th fresh set, n2 familiar states k = 760-819, trial t = k - 760; results/flawed/rvw5/a_<arm>/<arm>_main_<tid>.json): raa_read.diff's scheme (paired,
B = 2000, rng 0, tasks then states resampled), re-implemented here only so the bootstrap VECTOR is available for the Holm p-values (the draws are identical to raa_read.diff's).
Phase B (rgc cells on the 5th set, results/flawed/rvw5/b_<arm>/full_<tid>.json; rows of results/flawed/rg_comp_5/rg_comp_5_tasks.json main): wa_read.load / bootm.
Registered contrasts (the prediction text):
  d1' PRIMARY pzA0 - pzsL N: SUPPORTED iff upper < 0, REVERSED iff lower > 0, else NOT SHOWN.  d4' same on K: NON-INFERIOR iff lower > -.05.  REPLICATES iff d1' SUPPORTED and d4' NI.
  d2' pzA0 - A0 (N), d3' pzsL - raasl (N): reported.  c0 A0 - raasl N, cA A - raasl N: SUPPORTED-for-module iff upper < 0.
  r1 PRIMARY w5 - now N: SUPPORTED iff lower > 0.  r2 w5 - now F6_b.  r3 now - A0 N: DEPENDENT iff upper < 0, EQUIVALENT iff CI inside +-.10, else NOT SHOWN.  rK now - w5 K.  Ladder descriptive.
  h1 PRIMARY H - A0 N pooled over the dev set (results/flawed/raaf/dev_H vs dev_A0, 60 states) and the 5th set (mean of the per-set differences, each set resampled tasks then states):
     SUPPORTED iff lower > 0, REVERSED iff upper < 0.  h2 H - raasl N (5th): NI iff lower > -.10, COST iff upper < -.10.  h3 H - A N (5th): SUPPORTED iff lower > 0.  hK H - raasl K (5th): NI iff lower > -.05.
     hF H - raasl and H - A0 on F6_b (5th): reported.
  Holm over {d1', r1, h1}: one-sided p in each registered direction from the bootstrap, (1 + #{draws on the wrong side}) / (B + 1); step-down adjusted p; a primary SURVIVES iff its adjusted p < .025
  (the one-sided equivalent of the 95% two-sided interval rule).  Both raw and adjusted are written.
    python experiments/rvw5_read.py read --part d|r|h|all --out results/flawed/rvw5/rvw5_read_<part>.json     (reads only arms whose files all exist; outputs opened with "x")
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
BASE = os.path.join(OUT, "rvw5")
RGC5 = os.path.join(OUT, "rg_comp_5", "rg_comp_5_tasks.json")
DEVH = os.path.join(OUT, "raaf")
KNOWN, NEW = RR.KNOWN, RR.NEW
TRIALS5, TRIALS_DEV = list(range(60)), RR.DEV + RR.NEWS
B = RR.B


def tid_of(meta, comp):
    return next(t["tid"] for t in meta["tasks"] if t["compound"] == comp and t["layout"] == "familiar")


def apath5(meta, arm, c):
    return os.path.join(BASE, "a_" + arm, "%s_main_%d.json" % (arm, tid_of(meta, c)))


def have_a(meta, arm):
    return all(os.path.exists(apath5(meta, arm, c)) for c in KNOWN + NEW)


def grid5(meta, arm, comps):
    out = []
    for c in comps:
        tid = tid_of(meta, c)
        by = RR.episodes(apath5(meta, arm, c), tid)
        assert sorted(by) == TRIALS5, (arm, c, sorted(by))
        out.append([float(bool(by[k]["success"])) for k in TRIALS5])
    return np.array(out)


def griddev(meta, arm, comps):
    """dev-state 60-trial grid of an arm kept in results/flawed/raaf/dev_<arm> (A0, H) or results/flawed/raa (raasl)."""
    import raaf_dev_read as RD
    return RD.grid(meta, arm, comps, "succ")


def have_dev(meta, arm):
    import raaf_dev_read as RD
    dm, dr, stem = RD.src(arm)
    return all(os.path.exists(os.path.join(dm, "%s_main_%d.json" % (stem, tid_of(meta, c)))) and os.path.exists(os.path.join(dr, "%s_rx_%d.json" % (stem, tid_of(meta, c))))
               for c in KNOWN + NEW)


def boot_diff(ga, gb, pooled=None):
    """ga, gb = [tasks, states] success grids (or lists of such for a pooled analysis: mean over sets of the per-set difference, sets resampled independently).
    -> dict(a, b, delta, ci, draws).  Same rng and draws as raa_read.diff for a single set."""
    GA = ga if isinstance(ga, list) else [ga]
    GB = gb if isinstance(gb, list) else [gb]
    rng = np.random.default_rng(0)
    v = []
    for _ in range(B):
        tot = 0.0
        for xa, xb in zip(GA, GB):
            nt, ns = xa.shape
            t = rng.choice(nt, nt)
            s = rng.choice(ns, ns)
            tot += float(xa[t][:, s].mean()) - float(xb[t][:, s].mean())
        v.append(tot / len(GA))
    v = np.array(v)
    per = [float(xa.mean() - xb.mean()) for xa, xb in zip(GA, GB)]
    d0 = float(np.mean(per))
    return dict(a=float(np.mean([x.mean() for x in GA])), b=float(np.mean([x.mean() for x in GB])), delta=d0,
                ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], per_set=per if len(GA) > 1 else None,
                p_le0=float((1 + (v <= 0).sum()) / (B + 1)), p_ge0=float((1 + (v >= 0).sum()) / (B + 1)))


def lab_sup(d):                                                              # a - b: SUPPORTED iff lower > 0
    return "SUPPORTED" if d["ci"][0] > 0 else ("REVERSED" if d["ci"][1] < 0 else "NOT SHOWN")


def lab_neg(d):                                                              # a - b: SUPPORTED iff upper < 0 (the servo-kept arm is better)
    return "SUPPORTED" if d["ci"][1] < 0 else ("REVERSED" if d["ci"][0] > 0 else "NOT SHOWN")


def lab_ni(d, m):
    return "NON-INFERIOR" if d["ci"][0] > -m else ("COST" if d["ci"][1] < 0 else "NOT SHOWN")


def holm(ps, alpha=0.025):
    """ps: dict name -> raw one-sided p.  -> dict name -> dict(raw, adj, survives)."""
    order = sorted(ps, key=lambda k: ps[k])
    m = len(order)
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        adj[k] = run
    return {k: dict(raw=ps[k], adj=adj[k], survives=bool(adj[k] < alpha)) for k in ps}


def phaseB_cells(arms):
    meta = json.load(open(RGC5, encoding="utf-8"))
    rows7, rows6 = meta["states"]["f7"]["main"], meta["states"]["f6"]["main"]
    X = {}
    for arm in arms:
        for t in (7, 8, 9, 10):
            d, _ = WR.load(os.path.join(BASE, "b_" + arm, "full_%d.json" % t), rows6)
            X[(arm, t)] = None if d is None or len(d) < len(rows6) else np.array([d[k] for k in rows6], float)
    return X, rows6


def read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    meta = RR.n2()
    part = a.part
    arms_all = ("pzsL", "pzA0", "raasl", "A0", "A", "w5", "now", "H", "raasl43")
    arms = [x for x in arms_all if have_a(meta, x)]
    res = dict(states="n2 familiar fresh set k = 760-819 (trial t = k - 760); rgc 5th set rg_comp_5", B=B, part=part, arms=arms, success={}, contrasts={}, notes=[])
    G = {}
    for arm in arms:
        for grp, comps in (("K", KNOWN), ("N", NEW), ("cheese", RR.CHEESE), ("bottle", RR.BOTTLE)):
            G[(arm, grp)] = grid5(meta, arm, comps)
            res["success"]["%s|%s" % (arm, grp)] = float(G[(arm, grp)].mean())
        for c in KNOWN + NEW:
            res["success"]["%s|task|%s" % (arm, c)] = float(grid5(meta, arm, [c]).mean())
    C = res["contrasts"]

    def need(*xs):
        return all(x in arms for x in xs)

    def add(name, a_, b_, grp, label):
        if not need(a_, b_):
            C[name] = "MISSING (arm files absent: %s)" % [x for x in (a_, b_) if x not in arms]
            return None
        d = boot_diff(G[(a_, grp)], G[(b_, grp)])
        d["label"] = label(d) if label else None
        C[name] = d
        return d
    if part in ("d", "all"):
        d1 = add("d1' PRIMARY pzA0 - pzsL, N", "pzA0", "pzsL", "N", lab_neg)
        d4 = add("d4' pzA0 - pzsL, K (NI -.05)", "pzA0", "pzsL", "K", lambda d: lab_ni(d, 0.05))
        if d1 and d4:
            C["REPLICATES (d1' SUPPORTED and d4' NI)"] = bool(d1["label"] == "SUPPORTED" and d4["label"] == "NON-INFERIOR")
        add("d2' pzA0 - A0 (pi0 vs pi0.5, no servo), N", "pzA0", "A0", "N", None)
        add("d3' pzsL - raasl (pi0 vs pi0.5, servo), N", "pzsL", "raasl", "N", None)
        add("c0 A0 - raasl, N", "A0", "raasl", "N", lab_neg)
        add("cA A - raasl, N", "A", "raasl", "N", lab_neg)
        for grp in ("cheese", "bottle"):
            add("d1' pzA0 - pzsL, %s" % grp, "pzA0", "pzsL", grp, None)
    if part in ("r", "all"):
        r1 = add("r1 PRIMARY w5 - now, N", "w5", "now", "N", lab_sup)
        add("rK now - w5, K (reported)", "now", "w5", "K", None)

        def r3lab(d):
            return "DEPENDENT" if d["ci"][1] < 0 else ("EQUIVALENT" if (d["ci"][0] > -0.10 and d["ci"][1] < 0.10) else "NOT SHOWN")
        add("r3 now - A0, N", "now", "A0", "N", r3lab)
        add("ladder w5 - raasl (owner removal), N", "w5", "raasl", "N", None)
        add("ladder raasl - now (servo removal), N", "raasl", "now", "N", None)
        add("ladder w5 - now (both), N", "w5", "now", "N", None)
        add("ladder now - raasl, K", "now", "raasl", "K", None)
        BX, rows6 = phaseB_cells([x for x in ("w5", "now", "raasl", "H", "pzsL", "pzA0") if os.path.isdir(os.path.join(BASE, "b_" + x))])
        res["phaseB_success"] = {"%s|tid%d" % (k[0], k[1]): (None if v is None else float(v.mean())) for k, v in BX.items()}

        def bdiff(name, a_, b_, tid, label):
            xa, xb = BX.get((a_, tid)), BX.get((b_, tid))
            if xa is None or xb is None:
                C[name] = "MISSING"
                return
            ci = WR.bootm(lambda I, xa=xa, xb=xb: float(xa[I[0]].mean() - xb[I[0]].mean()), [len(rows6)])
            d = dict(delta=ci[0], ci=ci[1:], a=float(xa.mean()), b=float(xb.mean()))
            d["label"] = label(d) if label else None
            C[name] = d
        bdiff("r2 w5 - now, F6_b", "w5", "now", 8, lab_sup)
        bdiff("ladder w5 - raasl, F6_b", "w5", "raasl", 8, None)
        bdiff("ladder raasl - now, F6_b", "raasl", "now", 8, None)
        for t, nm in ((7, "K2t"), (10, "K1t"), (9, "F6_a")):
            bdiff("w5 - now, %s" % nm, "w5", "now", t, None)
        bdiff("pi0 F6_b pzA0 - pzsL", "pzA0", "pzsL", 8, None)
        bdiff("pi0 F6_a pzA0 - pzsL", "pzA0", "pzsL", 9, None)
    if part in ("h", "all"):
        hav = need("H")
        devH = have_dev(meta, "H") and have_dev(meta, "A0")
        if hav and need("A0"):
            if devH:
                gd = {a_: griddev(meta, a_, NEW) for a_ in ("H", "A0")}
                d = boot_diff([gd["H"], G[("H", "N")]], [gd["A0"], G[("A0", "N")]])
                d["label"] = lab_sup(d)
                C["h1 PRIMARY H - A0, N, pooled dev + 5th set"] = d
                d5 = boot_diff(G[("H", "N")], G[("A0", "N")])
                d5["label"] = lab_sup(d5)
                C["h1 per set: 5th set only"] = d5
                dd = boot_diff(gd["H"], gd["A0"])
                dd["label"] = lab_sup(dd)
                C["h1 per set: dev set only"] = dd
                gk = griddev(meta, "H", KNOWN)
                res["H_dev_K"] = float(gk.mean())
                res["H_gate_dev_K_ge_.70"] = bool(gk.mean() >= 0.70)
            else:
                C["h1 PRIMARY"] = "MISSING (dev rollouts of H or A0 absent)"
        add("h2 H - raasl, N (5th set; NI -.10)", "H", "raasl", "N", lambda d: lab_ni(d, 0.10))
        add("h3 H - A, N (5th set)", "H", "A", "N", lab_sup)
        add("hK H - raasl, K (5th set; NI -.05)", "H", "raasl", "K", lambda d: lab_ni(d, 0.05))
        BX, rows6 = phaseB_cells(["H", "raasl", "A0"] if os.path.isdir(os.path.join(BASE, "b_H")) else ["raasl"])
        res["phaseB_success"] = {"%s|tid%d" % (k[0], k[1]): (None if v is None else float(v.mean())) for k, v in BX.items()}
    if part == "all" or part == "h":
        ps = {}
        for key, nm, side in (("d1'", "d1' PRIMARY pzA0 - pzsL, N", "p_ge0"), ("r1", "r1 PRIMARY w5 - now, N", "p_le0"), ("h1", "h1 PRIMARY H - A0, N, pooled dev + 5th set", "p_le0")):
            if isinstance(C.get(nm), dict):
                ps[key] = C[nm][side]
        res["holm"] = holm(ps)
        res["holm_note"] = ("family {d1', r1, h1}; raw p one-sided in the registered direction (d1': P(delta >= 0); r1, h1: P(delta <= 0)); adjusted by step-down; survives iff adj < .025; "
                            "primaries present: %s" % sorted(ps))
    print("[rvw5-read]", json.dumps(res)[:6000], flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[rvw5-read] -> %s" % out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read",))
    ap.add_argument("--part", default="all", choices=("d", "r", "h", "all"))
    ap.add_argument("--out", required=True)
    read(ap.parse_args())

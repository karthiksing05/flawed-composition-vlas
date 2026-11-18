"""WA read (docs/notes/wa_plan.md; m3_prereg tag wa).  Numpy only (any venv; run on a compute node).

Cells (rg_comp's rgc suite, MAIN states, unchanged): F7 = tids 0-3 (relL, relR, pos, neg; 40 states each), F6_b = tid 8, F6_a = tid 9 (60), K2t = tid 7
(40), K1t = tid 10 (60).  Arms (all on the RAA checkpoint): full, noR, noM, noMe (descriptive); references on frozen pi0.5 from results/flawed/rg_comp_roll:
I, rgp, rc (descriptive only; same states and branch files, identity checked by trial id and by the start-scene snapshot of each episode).

Estimator (rg_comp.bootm: B = 2000, numpy default_rng(0), percentile 2.5 / 97.5, paired by state): the F7 pool = mean over the four F7 tasks of the
per-task success rate, ONE draw of the 40 state indices per bootstrap sample applied to all four tasks (as rg_comp's families); K1t + K2t = the mean of the
two tasks' rates, each task's states resampled separately in the same draw (rg_comp's K).

REGISTERED CONTRASTS
  D1  full - noR on F7 (pool of tids 0-3)          SUPPORTED iff lower bound > 0   (else NOT SUPPORTED; REVERSED iff upper bound < 0)
  D2  full - noM on F6_b (tid 8)                   SUPPORTED iff lower bound > 0   (same)
  D3a noR - full on F6_b                           EQUIVALENT iff the interval lies inside (-.10, +.10), else NOT EQUIVALENT
  D3b noM - full on F7 pool                        the same
  DK  full - noR on K1t + K2t                      NON-INFERIOR iff lower bound > -.05, else NOT SHOWN (COST iff upper bound < 0)
  PRIMARY: DISSOCIATION SHOWN iff D1 and D2 SUPPORTED and D3a and D3b EQUIVALENT; PARTIAL iff D1 and D2 SUPPORTED, D3 not both EQUIVALENT; else NOT SHOWN.
DESCRIPTIVE: full vs rc (frozen pi0.5 + F7R + slot + ReGuide, SR 70) on the F7 pool and F6_b; F6_b vs F6_a (full); per task rates; noMe; I, rgp, rc.

    python experiments/wa_read.py read --phase smoke|main [--wa DIR] --out X.json
    python experiments/wa_read.py selftest        reproduces SR 70's rc - I on F6_b (+.283 [+.150, +.417]) and rc - rgp on pi0.5 SP/NG from the stored rollouts
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
META = os.path.join(OUT, "rg_comp", "rg_comp_tasks.json")
REF = os.path.join(OUT, "rg_comp_roll")
WAD = os.path.join(OUT, "wa")
B = 2000
EQ, NI = 0.10, 0.05
F7T = (0, 1, 2, 3)
KEYS = {0: "F7_SP_relL", 1: "F7_SP_relR", 2: "F7_NG_pos", 3: "F7_NG_neg", 4: "F7_SP_uniL", 5: "F7_SP_uniR", 6: "F7_NG_uniP", 7: "K2t", 8: "F6_b", 9: "F6_a", 10: "K1t"}


def bootm(fn, ns, seed=0):
    """rg_comp.bootm, restated: -> [point, lo, hi]; fn(list of index arrays, one per state set), each resampled in the same draw."""
    rng = np.random.default_rng(seed)
    pt = fn([np.arange(n) for n in ns])
    v = []
    for _ in range(B):
        v.append(fn([rng.integers(0, n, n) for n in ns]))
    v = np.array(v, float)
    v = v[np.isfinite(v)]
    if not len(v) or not np.isfinite(pt):
        return [float(pt), None, None]
    return [float(pt), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load(path, rows):
    """-> {trial: success} of one result file, or None.  Also returns the start snapshots {trial: array} from its track file."""
    if not os.path.exists(path):
        return None, None
    eps = json.load(open(path, encoding="utf-8"))["tasks"]
    eps = list(eps.values())[0]["episodes"]
    s = {int(e["trial"]): float(bool(e["success"])) for e in eps}
    fz = path.replace(".json", "_track.npz")
    st = {}
    if os.path.exists(fz):
        z = np.load(fz)
        for i, e in enumerate(eps):
            if i < int(z["n"]):
                st[int(e["trial"])] = z[f"e{i}_start"]
    return {k: s[k] for k in rows if k in s}, st


def lab_support(ci):
    if ci[1] is None:
        return "MISSING"
    return "SUPPORTED" if ci[1] > 0 else ("REVERSED" if ci[2] < 0 else "NOT SUPPORTED")


def lab_equiv(ci):
    if ci[1] is None:
        return "MISSING"
    return "EQUIVALENT" if (ci[1] > -EQ and ci[2] < EQ) else "NOT EQUIVALENT"


def lab_ni(ci):
    if ci[1] is None:
        return "MISSING"
    return "NON-INFERIOR" if ci[1] > -NI else ("COST" if ci[2] < 0 else "NOT SHOWN")


def read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    meta = json.load(open(META, encoding="utf-8"))
    rows7, rows6 = meta["states"]["f7"][a.phase], meta["states"]["f6"][a.phase]
    rows_of = lambda t: rows7 if t <= 7 else rows6                         # noqa: E731
    wad = os.path.abspath(a.wa)
    R, S = {}, {}
    for arm in ("full", "noR", "noM", "noMe"):
        for t in KEYS:
            R[(arm, t)], S[(arm, t)] = load(os.path.join(wad, "%s_%d.json" % (arm, t)), rows_of(t))
    for arm in ("I", "rgp", "rc"):
        for t in KEYS:
            R[(arm, t)], S[(arm, t)] = load(os.path.join(REF, "pi05_%s_%d.json" % (arm, t)), rows_of(t))
    res = dict(phase=a.phase, B=B, wa=wad, missing=[], success={}, n={}, contrasts={}, identity={}, mechanism={})

    def X(arm, t):
        d = R[(arm, t)]
        n = len(rows_of(t))
        return None if d is None else np.array([d.get(k, np.nan) for k in rows_of(t)], float)
    for (arm, t), d in R.items():
        if d is None or len(d) < len(rows_of(t)):
            res["missing"].append("%s/%s" % (arm, KEYS[t]))
    # state identity: the same trial ids, and the same start snapshot, between every WA arm / reference on a task
    for t in KEYS:
        base = next(((arm, S[(arm, t)]) for arm in ("full", "noR", "I") if S[(arm, t)]), None)
        for arm in ("full", "noR", "noM", "noMe", "I", "rgp", "rc"):
            if base is None or not S[(arm, t)] or arm == base[0]:
                continue
            ks = [k for k in rows_of(t) if k in base[1] and k in S[(arm, t)]]
            same = all(base[1][k].shape == S[(arm, t)][k].shape and np.allclose(base[1][k], S[(arm, t)][k], atol=1e-5) for k in ks)
            res["identity"]["%s vs %s / %s" % (arm, base[0], KEYS[t])] = dict(n=len(ks), start_snapshot_equal=bool(same),
                                                                           trials_equal=bool(set(R[(arm, t)] or []) == set(R[base[0], t] or [])))

    def rate(arm, t):
        x = X(arm, t)
        return None if x is None else (lambda I, x=x: float(np.nanmean(x[I[0]])))

    def pool(arm):
        xs = [X(arm, t) for t in F7T]
        if any(x is None for x in xs):
            return None
        return lambda I, xs=xs: float(np.mean([np.nanmean(x[I[0]]) for x in xs]))

    def kpool(arm):
        a_, b_ = X(arm, 10), X(arm, 7)
        if a_ is None or b_ is None:
            return None
        return lambda I: 0.5 * (float(np.nanmean(a_[I[0]])) + float(np.nanmean(b_[I[1]])))
    cells = {"F7": (pool, [len(rows7)]), "F6_b": (lambda arm: rate(arm, 8), [len(rows6)]), "F6_a": (lambda arm: rate(arm, 9), [len(rows6)]),
             "K": (kpool, [len(rows6), len(rows7)]), "K1t": (lambda arm: rate(arm, 10), [len(rows6)]), "K2t": (lambda arm: rate(arm, 7), [len(rows7)])}
    for t in F7T:
        cells[KEYS[t]] = ((lambda arm, t=t: rate(arm, t)), [len(rows7)])
    for cell, (fn, ns) in cells.items():
        for arm in ("full", "noR", "noM", "noMe", "I", "rgp", "rc"):
            f = fn(arm)
            if f is not None:
                res["success"]["%s|%s" % (arm, cell)] = bootm(f, ns)

    # GATES (descriptive labels on the executor, not a change to the primary rule; rg_comp's rule restated for RAA):
    #   floor  full's K (K1t, K2t) >= .70;  F7 full's unique-name controls (tids 4-6: one visible bowl, the F7 layouts) mean >= .50;  F6 full's F6_a >= .50
    g = {}
    kf = res["success"].get("full|K")
    ux = [X("full", t) for t in (4, 5, 6)]
    uni = None if any(x is None for x in ux) else float(np.mean([np.nanmean(x) for x in ux]))
    a6 = None if X("full", 9) is None else float(np.nanmean(X("full", 9)))
    g["floor_full_K"] = dict(value=None if kf is None else kf[0], label=None if kf is None else ("TESTABLE" if kf[0] >= 0.70 else "NOT TESTABLE"))
    g["F7_full_unique_controls"] = dict(value=uni, label=None if uni is None else ("TESTABLE" if uni >= 0.50 else "NOT TESTABLE"))
    g["F6_full_F6_a"] = dict(value=a6, label=None if a6 is None else ("TESTABLE" if a6 >= 0.50 else "NOT TESTABLE"))
    res["gates"] = g

    def diff(name, ca, cb, cell, labeller, note=""):
        fn, ns = cells[cell]
        fa, fb = fn(ca), fn(cb)
        if fa is None or fb is None:
            res["contrasts"][name] = dict(label="MISSING", a=ca, b=cb, cell=cell)
            return None
        ci = bootm(lambda I: fa(I) - fb(I), ns)
        res["contrasts"][name] = dict(a=ca, b=cb, cell=cell, delta=ci[0], ci=[ci[1], ci[2]], label=labeller(ci) if labeller else None, note=note)
        return res["contrasts"][name]["label"]
    L1 = diff("D1 full - noR, F7", "full", "noR", "F7", lab_support)
    L2 = diff("D2 full - noM, F6_b", "full", "noM", "F6_b", lab_support)
    L3a = diff("D3a noR - full, F6_b", "noR", "full", "F6_b", lab_equiv, "R is not engaged on F6_b: noR is a replicate of full")
    L3b = diff("D3b noM - full, F7", "noM", "full", "F7", lab_equiv)
    diff("DK full - noR, K1t + K2t", "full", "noR", "K", lab_ni, "R is not engaged on K1t / K2t: noR is a replicate of full")
    if L1 == "SUPPORTED" and L2 == "SUPPORTED":
        res["primary"] = "DISSOCIATION SHOWN" if (L3a == "EQUIVALENT" and L3b == "EQUIVALENT") else "PARTIAL"
    else:
        res["primary"] = "NOT SHOWN"
    res["primary_inputs"] = dict(D1=L1, D2=L2, D3a=L3a, D3b=L3b)
    # descriptive
    diff("d1 full - rc (SR 70, frozen pi0.5 + R + slot + ReGuide), F7", "full", "rc", "F7", None)
    diff("d2 full - rc, F6_b", "full", "rc", "F6_b", None)
    diff("d3 full - I (frozen pi0.5), F7", "full", "I", "F7", None)
    diff("d4 full - I, F6_b", "full", "I", "F6_b", None)
    xb, xa = X("full", 8), X("full", 9)
    if xb is not None and xa is not None:
        ci = bootm(lambda I: float(np.nanmean(xb[I[0]]) - np.nanmean(xa[I[0]])), [len(rows6)])
        res["contrasts"]["d5 full F6_b - full F6_a (the unhidden ceiling)"] = dict(a="full F6_b", b="full F6_a", cell="F6", delta=ci[0], ci=[ci[1], ci[2]])
    diff("d6 noMe - full, F6_b (empty-mask hold only withheld)", "noMe", "full", "F6_b", None)
    diff("d7 noMe - full, F7", "noMe", "full", "F7", None)
    diff("d8 full - noMe, F6_b", "full", "noMe", "F6_b", None)
    diff("d9 noM - noMe, F6_b", "noM", "noMe", "F6_b", None)
    for t in F7T:
        diff("d10 full - noR, %s" % KEYS[t], "full", "noR", KEYS[t], None)
        diff("d11 noM - full, %s" % KEYS[t], "noM", "full", KEYS[t], None)
    # mechanism: the client side records and the server's per-query log
    for arm in ("full", "noR", "noM", "noMe"):
        for t in KEYS:
            f = os.path.join(wad, "%s_%d_wa.json" % (arm, t))
            if not os.path.exists(f):
                continue
            eps = json.load(open(f, encoding="utf-8"))["eps"]
            m = dict(episodes=len(eps))
            fr = [e["front"]["res"] for e in eps if e.get("front") and e["front"].get("res")]
            if fr:
                m["front_ok"] = float(np.mean([bool(r.get("ok")) for r in fr]))
                m["referent_bound"] = float(np.mean([bool(r.get("used_correct", r.get("resolver_correct"))) for r in fr if r.get("ok")])) if any(r.get("ok") for r in fr) else None
                m["n_comp"] = [int(r.get("n_comp", -1)) for r in fr][:5]
            q = [x for e in eps for x in e["queries"]]
            if q:
                m["queries"] = len(q)
                m["hidden_obj_share"] = float(np.mean([not x["vis_obj"] for x in q]))
                m["hidden_tgt_share"] = float(np.mean([not x["vis_tgt"] for x in q]))
                m["cover_on_share"] = float(np.mean([x["hidden_truth"] for x in q]))
                m["err_obj_cm_median"] = float(np.median([x["err_obj_cm"] for x in q]))
                m["err_tgt_cm_median"] = float(np.median([x["err_tgt_cm"] for x in q]))
            res["mechanism"]["%s/%s" % (arm, KEYS[t])] = m
    rec = {}
    for f in sorted(glob.glob(os.path.join(wad, "chan_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = "%s/%s/%s" % (r["arm"], KEYS.get(r["tid"], r["tid"]), r["role"])
            d = rec.setdefault(k, dict(n=0, omitted=0, stale=0, err=[], src={}))
            d["n"] += 1
            d["omitted"] += int(bool(r["omitted"]))
            d["stale"] += int(r["src"] in ("occl", "last", "prior"))
            d["err"].append(r["err_cm"])
            d["src"][r["src"]] = d["src"].get(r["src"], 0) + 1
    res["channel"] = {k: dict(queries=d["n"], omitted=d["omitted"], omitted_share=d["omitted"] / d["n"], stale_share=d["stale"] / d["n"],
                              err_median_cm=float(np.median(d["err"])), err_p90_cm=float(np.percentile(d["err"], 90)), src=d["src"])
                      for k, d in sorted(rec.items())}
    print("[wa-read] " + json.dumps(res), flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[wa-read] -> %s" % out, flush=True)
    print("[wa-read] PRIMARY: %s  %s" % (res["primary"], res["primary_inputs"]), flush=True)
    for k, v in res["contrasts"].items():
        print("[wa-read] %-62s %+.3f [%s, %s] %s" % (k, v["delta"], None if v["ci"][0] is None else "%+.3f" % v["ci"][0],
                                                  None if v["ci"][1] is None else "%+.3f" % v["ci"][1], v.get("label")), flush=True)
    for k, v in res["success"].items():
        print("[wa-read] success %-18s %.3f [%s, %s]" % (k, v[0], None if v[1] is None else "%.3f" % v[1], None if v[2] is None else "%.3f" % v[2]), flush=True)


def selftest(a):
    """SR 70 (rg_comp_read, pi0.5, main): rc - I on F6_b +.283 [+.150, +.417]; rc - rgp SP +.550 [+.488, +.625], NG +.663 [+.587, +.738]."""
    meta = json.load(open(META, encoding="utf-8"))
    rows7, rows6 = meta["states"]["f7"]["main"], meta["states"]["f6"]["main"]
    X = {}
    for arm in ("I", "rgp", "rc"):
        for t in (0, 1, 2, 3, 8):
            d, _ = load(os.path.join(REF, "pi05_%s_%d.json" % (arm, t)), rows6 if t == 8 else rows7)
            X[(arm, t)] = np.array([d[k] for k in (rows6 if t == 8 else rows7)], float)
    ci = bootm(lambda I: X[("rc", 8)][I[0]].mean() - X[("I", 8)][I[0]].mean(), [len(rows6)])
    print("[wa-read] selftest rc - I F6_b %+.3f [%+.3f, %+.3f] (SR 70: +.283 [+.150, +.417])" % tuple(ci), flush=True)
    assert abs(ci[0] - 0.283) < 6e-4 and abs(ci[1] - 0.150) < 6e-4 and abs(ci[2] - 0.417) < 6e-4, ci
    for fam, (t1, t2), want in (("SP", (0, 1), (0.550, 0.488, 0.625)), ("NG", (2, 3), (0.663, 0.587, 0.738))):
        f = lambda I, arm=None: 0.0                                          # noqa: E731
        ci = bootm(lambda I: 0.5 * sum(X[("rc", t)][I[0]].mean() - X[("rgp", t)][I[0]].mean() for t in (t1, t2)), [len(rows7)])
        print("[wa-read] selftest rc - rgp %s %+.3f [%+.3f, %+.3f] (SR 70: %+.3f [%+.3f, %+.3f])" % ((fam,) + tuple(ci) + want), flush=True)
        assert all(abs(x - y) < 6e-4 for x, y in zip(ci, want)), ci
    print("[wa-read] selftest ok", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read", "selftest"))
    ap.add_argument("--phase", default="main", choices=("smoke", "main"))
    ap.add_argument("--wa", default=WAD)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"read": read, "selftest": selftest}[a.what](a)

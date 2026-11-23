"""NPH reader (docs/notes/plan_necessity_power.md 3.3; prereg experiments/prereg/nph_pred.txt).  Numpy only (openpi venv, compute node via sbatch).

Built on rvw5_read.py (imported unchanged: boot_diff, lab_*, holm) for the registered tasks+states bootstrap; the new part is the primary rule n1 (tasks fixed, states-only bootstrap, widened by
the measured seed variance, with a no-catastrophic-task flag).  Phase A files: results/flawed/nph6/a_<arm>/<arm>_main_<tid>.json (n2 familiar states k = 820-879, trial t = k - 820); arms
pzH = pi0-H*, pzA0 = pi0-A0, pzsL = pi0-sL.

Notation: d_ts paired difference (task t, state s) on N1-N6, 60 states; d_t task mean; Delta = mean_t d_t; v_t = var_s(d_ts) / 60 (ddof 1); sigma_s = .033 (K: .049), sigma_i = .070 (frozen).
  n1 PRIMARY pzH - pzA0, N: interval Delta +- 1.96 sqrt(var_B + 2 sigma_s^2), var_B = variance of the states-only bootstrap (tasks fixed, one state index shared by all tasks, B = 2000, rng 0).
     flag = some task with d_t + 1.96 sqrt(v_t + 2 sigma_s^2 + 2 sigma_i^2) < -.10.  SUPPORTED iff lower > 0 and no flag; FLAGGED iff lower > 0 and a flag; REVERSED iff upper < 0;
     EQUIVALENT iff the interval lies inside +-.10; else NOT SHOWN.
  n1r same contrast, registered tasks+states bootstrap (raa_read.diff scheme): SUPPORTED iff lower > 0.
  n2 pzH - pzsL, N, interval as n1 (flag reported): NON-INFERIOR iff lower > -.10; COST iff upper < -.10.
  d1'' pzA0 - pzsL, N: SUPPORTED iff upper < 0, under both the seed-widened rule and the registered rule.
  nK pzH - pzA0, K, interval as n1 with sigma_s = .049: NON-INFERIOR iff lower > -.05.
  Sensitivities (reported, not used for the label): n1 with sigma_s = .05; n1 with the finite-population factor (6 of 8 pairings) on the task component of the registered bootstrap variance.
  Holm over the necessity primaries {d1', r1, h1, n1}: one-sided raw p's; d1', r1, h1 from results/flawed/rvw5/rvw5_read_{d,r,h}.json; n1 from the seed-widened normal.
    python experiments/nph_read.py read --part gate|main --out results/flawed/nph6/nph_read_<part>.json      (outputs opened with "x")
    python experiments/nph_read.py read --part main --base results/flawed/rvw5 --arm-map pzH=H --out <scratch>.json      (SELF-TEST on the existing 5th-set grids; not a read of the 6th set)
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raa_read as RR                                                        # noqa: E402
import rvw5_read as V                                                        # noqa: E402

OUT = RR.OUT
KNOWN, NEW = RR.KNOWN, RR.NEW
TRIALS = list(range(60))
B = V.B
SIG_S, SIG_SK, SIG_I = 0.033, 0.049, 0.070
Z = 1.96


def grid(meta, base, amap, arm, comps):
    d, stem = amap.get(arm, (arm, arm))
    out = []
    for c in comps:
        tid = V.tid_of(meta, c)
        by = RR.episodes(os.path.join(base, "a_" + d, "%s_main_%d.json" % (stem, tid)), tid)
        assert sorted(by) == TRIALS, (arm, c, sorted(by))
        out.append([float(bool(by[k]["success"])) for k in TRIALS])
    return np.array(out)


def have(meta, base, amap, arm, comps):
    d, stem = amap.get(arm, (arm, arm))
    return all(os.path.exists(os.path.join(base, "a_" + d, "%s_main_%d.json" % (stem, V.tid_of(meta, c)))) for c in comps)


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def widened(ga, gb, sig_s, comps, fpc=None):
    """Primary-rule statistics for ga - gb ([tasks, states] success grids)."""
    d = ga - gb
    nt, ns = d.shape
    dt = d.mean(1)
    delta = float(dt.mean())
    rng = np.random.default_rng(0)
    draws = np.array([d[:, rng.choice(ns, ns)].mean() for _ in range(B)])
    var_b = float(np.var(draws, ddof=1))
    se = math.sqrt(var_b + 2.0 * sig_s ** 2)
    half = Z * se
    v_t = d.var(1, ddof=1) / ns
    flag_ub = dt + Z * np.sqrt(v_t + 2.0 * sig_s ** 2 + 2.0 * SIG_I ** 2)
    res = dict(delta=delta, ci=[delta - half, delta + half], half=half, var_B=var_b, sigma_s=sig_s, se=se, p_le0=float(1.0 - phi(delta / se)), p_ge0=float(phi(delta / se)),
               per_task={c: dict(d_t=float(dt[i]), v_t=float(v_t[i]), flag_bound=float(flag_ub[i]), flagged=bool(flag_ub[i] < -0.10)) for i, c in enumerate(comps)},
               flagged_tasks=[c for i, c in enumerate(comps) if flag_ub[i] < -0.10], a=float(ga.mean()), b=float(gb.mean()))
    if fpc is not None:                                                     # task component of the registered bootstrap variance scaled by (1 - 6/8)
        rng = np.random.default_rng(0)
        full = []
        for _ in range(B):
            t = rng.choice(nt, nt)
            s = rng.choice(ns, ns)
            full.append(float(ga[t][:, s].mean()) - float(gb[t][:, s].mean()))
        var_full = float(np.var(full, ddof=1))
        var_task = max(var_full - var_b, 0.0)
        se2 = math.sqrt(var_b + (1.0 - fpc) * var_task + 2.0 * sig_s ** 2)
        res["fpc"] = dict(frac=fpc, var_full=var_full, var_task=var_task, se=se2, ci=[delta - Z * se2, delta + Z * se2])
    return res


def lab_n1(r):
    lo, hi = r["ci"]
    if lo > 0:
        return "FLAGGED" if r["flagged_tasks"] else "SUPPORTED"
    if hi < 0:
        return "REVERSED"
    if lo > -0.10 and hi < 0.10:
        return "EQUIVALENT"
    return "NOT SHOWN"


def raw_p_prev():
    ps = {}
    for key, fn, nm, side in (("d1'", "rvw5_read_d.json", "d1' PRIMARY pzA0 - pzsL, N", "p_ge0"), ("r1", "rvw5_read_r.json", "r1 PRIMARY w5 - now, N", "p_le0"),
                              ("h1", "rvw5_read_h.json", "h1 PRIMARY H - A0, N, pooled dev + 5th set", "p_le0")):
        p = os.path.join(OUT, "rvw5", fn)
        if os.path.exists(p):
            ps[key] = json.load(open(p, encoding="utf-8"))["contrasts"][nm][side]
    return ps


def read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    meta = RR.n2()
    base = os.path.abspath(a.base) if a.base else os.path.join(OUT, "nph6")
    amap = {}
    for kv in (a.arm_map or []):
        k, v = kv.split("=")
        amap[k] = (v, v)
    res = dict(states="n2 familiar fresh set k = 820-879 (trial t = k - 820)" if not a.base else "SELF-TEST on %s with arm map %s" % (base, amap), B=B, part=a.part, sigma=dict(s=SIG_S, sK=SIG_SK, i=SIG_I),
               success={}, contrasts={}, notes=[])
    if a.part == "gate":
        assert have(meta, base, amap, "pzH", KNOWN), "pzH K files incomplete"
        g = grid(meta, base, amap, "pzH", KNOWN)
        res["gate"] = dict(K_mean=float(g.mean()), per_task={c: float(g[i].mean()) for i, c in enumerate(KNOWN)}, threshold=0.70, passes=bool(g.mean() >= 0.70))
        print("[nph-read]", json.dumps(res), flush=True)
        with open(out, "x", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        return
    arms = [x for x in ("pzH", "pzA0", "pzsL") if have(meta, base, amap, x, KNOWN + NEW)]
    res["arms"] = arms
    G = {}
    for arm in arms:
        for grp, comps in (("K", KNOWN), ("N", NEW), ("cheese", RR.CHEESE), ("bottle", RR.BOTTLE)):
            G[(arm, grp)] = grid(meta, base, amap, arm, comps)
            res["success"]["%s|%s" % (arm, grp)] = float(G[(arm, grp)].mean())
        for c in KNOWN + NEW:
            res["success"]["%s|task|%s" % (arm, c)] = float(grid(meta, base, amap, arm, [c]).mean())
    C = res["contrasts"]
    need = lambda *xs: all(x in arms for x in xs)                           # noqa: E731
    if need("pzH", "pzA0"):
        n1 = widened(G[("pzH", "N")], G[("pzA0", "N")], SIG_S, NEW, fpc=None)
        n1["label"] = lab_n1(n1)
        C["n1 PRIMARY pzH - pzA0, N (seed-widened, tasks fixed)"] = n1
        n1r = V.boot_diff(G[("pzH", "N")], G[("pzA0", "N")])
        n1r["label"] = "SUPPORTED" if n1r["ci"][0] > 0 else ("REVERSED" if n1r["ci"][1] < 0 else "NOT SHOWN")
        C["n1r pzH - pzA0, N (registered tasks+states bootstrap)"] = n1r
        s05 = widened(G[("pzH", "N")], G[("pzA0", "N")], 0.05, NEW, fpc=0.75)
        s05["label"] = lab_n1(s05)
        C["sens n1 sigma_s = .05 (+ FPC 6/8 at sigma_s = .05, key fpc)"] = s05
        sf = widened(G[("pzH", "N")], G[("pzA0", "N")], SIG_S, NEW, fpc=0.75)
        C["sens n1 FPC 6/8 at sigma_s = .033 (key fpc: task variance x (1 - 6/8))"] = dict(delta=sf["delta"], fpc=sf["fpc"])
        nk = widened(G[("pzH", "K")], G[("pzA0", "K")], SIG_SK, KNOWN)
        nk["label"] = "NON-INFERIOR" if nk["ci"][0] > -0.05 else ("COST" if nk["ci"][1] < -0.05 else "NOT SHOWN")
        C["nK pzH - pzA0, K (NI -.05, sigma_s .049)"] = nk
        for grp in ("cheese", "bottle"):
            C["n1 by pairing group: %s" % grp] = V.boot_diff(G[("pzH", grp)], G[("pzA0", grp)])
    if need("pzH", "pzsL"):
        n2 = widened(G[("pzH", "N")], G[("pzsL", "N")], SIG_S, NEW)
        n2["label"] = "NON-INFERIOR" if n2["ci"][0] > -0.10 else ("COST" if n2["ci"][1] < -0.10 else "NOT SHOWN")
        C["n2 pzH - pzsL, N (NI -.10; seed-widened)"] = n2
        r = V.boot_diff(G[("pzH", "N")], G[("pzsL", "N")])
        r["label"] = "NON-INFERIOR" if r["ci"][0] > -0.10 else ("COST" if r["ci"][1] < -0.10 else "NOT SHOWN")
        C["n2 registered-bootstrap version"] = r
    if need("pzA0", "pzsL"):
        d1 = widened(G[("pzA0", "N")], G[("pzsL", "N")], SIG_S, NEW)
        d1["label"] = "SUPPORTED" if d1["ci"][1] < 0 else ("REVERSED" if d1["ci"][0] > 0 else "NOT SHOWN")
        C["d1'' pzA0 - pzsL, N (seed-widened)"] = d1
        d1r = V.boot_diff(G[("pzA0", "N")], G[("pzsL", "N")])
        d1r["label"] = V.lab_neg(d1r)
        C["d1'' pzA0 - pzsL, N (registered)"] = d1r
        res["d1pp_SUPPORTED_both"] = bool(d1["label"] == "SUPPORTED" and d1r["label"] == "SUPPORTED")
        C["d1'' K pzA0 - pzsL (reported)"] = V.boot_diff(G[("pzA0", "K")], G[("pzsL", "K")])
    key = "n1 PRIMARY pzH - pzA0, N (seed-widened, tasks fixed)"
    if key in C:
        ps = raw_p_prev()
        ps["n1"] = C[key]["p_le0"]
        res["holm"] = V.holm(ps)
        res["holm_note"] = "family {d1', r1, h1, n1}; raw one-sided p; d1', r1, h1 from rvw5_read_{d,r,h}.json; n1 from the seed-widened normal; survives iff adjusted < .025; present: %s" % sorted(ps)
    print("[nph-read]", json.dumps(res)[:7000], flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[nph-read] -> %s" % out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("read",))
    ap.add_argument("--part", default="main", choices=("gate", "main"))
    ap.add_argument("--base", default=None)
    ap.add_argument("--arm-map", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    read(ap.parse_args())

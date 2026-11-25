"""NPHJ reader (experiments/prereg/nphj_pred.txt).  Numpy only (openpi venv, compute node via sbatch).  Reuses nphs_read.py's hierarchical bootstrap and nph_read.py's grid loaders unchanged.
New arms (results/flawed/nphj7/a_<arm>_s<seed>/<arm>_s<seed>_main_<tid>.json): pzA24 (pi0-A0 recipe, 24,001 steps; seeds 42, 43), pzJ (jointly trained where-branch; seeds 42-45).
Comparators (existing, results/flawed/nphs7): pzA0 = pi0-A0 (6k) and pzsL = pi0-sL, seeds 42-45.  Same 7th set (k 880-939, trial t = k - 880).
  gate --arm A --seed S --out <json> : K1-K6 mean >= .70 ("passes")   [outputs opened with "x"]
  main --out <json>                  : u1, u2, j1 (PRIMARY), j2, jK, seed SD, paired t (descriptive), J magnitude at 5 cm, Holm over {u1, j1}.
Hierarchical bootstrap (every interval): resample seeds with replacement within each arm (independently per arm, among that arm's valid seeds; comparators use all four seeds), then states within each task
(one resample per task, shared by the two arms; tasks fixed); B = 2000, np.random.default_rng(0), 95% percentile interval; point = mean over the arm's valid seeds of the task-mean success.
Labels: u1 A24 - sL on N: "UNDER-TRAINING RULED OUT" iff upper < 0 else "NOT RULED OUT".  u2 A24 - A0 on N: reported.  j1 PRIMARY J - A0 on N: SUPPORTED lower > 0 | REVERSED upper < 0 | EQUIVALENT inside +-.10 | NOT SHOWN.
j2 J - sL on N: NON-INFERIOR lower > -.10.  jK J - A0 on K: NON-INFERIOR lower > -.05.  Gates: K mean < .70 for a (arm, seed) -> that seed is dropped (UNTESTABLE); fewer than 3 valid J seeds -> j1, j2, jK UNTESTABLE;
no valid A24 seed -> u1, u2 UNTESTABLE.  Holm over {u1, j1}: one-sided bootstrap p (u1: P(delta >= 0), j1: P(delta <= 0)) = (1 + #draws on the wrong side) / (B + 1); survives iff adjusted < .025."""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nph_read as NR                                                        # noqa: E402
import nphs_read as NS                                                       # noqa: E402

RR, V = NR.RR, NR.V
OLD = os.path.join(RR.OUT, "nphs7")
NEW = os.path.join(RR.OUT, "nphj7")
SEEDS = (42, 43, 44, 45)
ASEEDS = {"pzA24": (42, 43), "pzJ": (42, 43, 44, 45)}


def name(arm, s):
    return "%s_s%d" % (arm, s)


def grid(arm, s, comps):
    base = OLD if arm in ("pzA0", "pzsL") else NEW
    nm = name(arm, s)
    return NR.grid(RR.n2(), base, {nm: (nm, nm)}, nm, comps)


def gate_read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    nm = name(a.arm, a.seed)
    assert NR.have(RR.n2(), NEW, {nm: (nm, nm)}, nm, NR.KNOWN), "K files incomplete"
    g = grid(a.arm, a.seed, NR.KNOWN)
    res = dict(arm=a.arm, seed=a.seed, gate=dict(K_mean=float(g.mean()), per_task={c: float(g[i].mean()) for i, c in enumerate(NR.KNOWN)}, threshold=0.70, passes=bool(g.mean() >= 0.70)))
    print("[nphj-read]", json.dumps(res), flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def tcheck(da):
    d = np.asarray(da, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return dict(n=n, mean=float(d.mean()) if n else None, t=None, df=n - 1, p_two_sided=None)
    t = float(d.mean() / (d.std(ddof=1) / math.sqrt(n)))
    p = None
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        pass
    return dict(n=n, mean=float(d.mean()), t=t, df=n - 1, p_two_sided=p)


def main_read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    res = dict(states="n2 fresh set k = 880-939 (trial t = k - 880)", B=NS.B, gates={}, success={}, seed_sd={}, contrasts={}, descriptive={}, notes=[])
    valid = {}
    for arm in ("pzA24", "pzJ"):
        valid[arm] = []
        for s in ASEEDS[arm]:
            g = json.load(open(os.path.join(NEW, "nphj_gate_%s_s%d.json" % (arm, s)), encoding="utf-8"))["gate"]
            res["gates"]["%s|s%d" % (arm, s)] = g
            if g["passes"]:
                valid[arm].append(s)
    res["valid_seeds"] = valid
    G = {}
    for arm in ("pzA0", "pzsL", "pzA24", "pzJ"):
        ss = SEEDS if arm in ("pzA0", "pzsL") else valid[arm]
        for s in ss:
            for grp, comps in (("K", NR.KNOWN), ("N", NR.NEW)):
                G[(arm, s, grp)] = grid(arm, s, comps)
                res["success"]["%s|s%d|%s" % (arm, s, grp)] = float(G[(arm, s, grp)].mean())
    for arm in ("pzA0", "pzsL", "pzA24", "pzJ"):
        for grp in ("N", "K"):
            v = [G[(arm, s, grp)].mean() for s in SEEDS if (arm, s, grp) in G]
            res["seed_sd"]["%s|%s" % (arm, grp)] = dict(n_seeds=len(v), mean=float(np.mean(v)) if v else None, sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
    get = lambda arm, grp, ss: {s: G[(arm, s, grp)] for s in ss}                 # noqa: E731
    C, ps = res["contrasts"], {}
    nA, nJ = len(valid["pzA24"]), len(valid["pzJ"])
    if nA >= 1:
        r = NS.hboot(get("pzA24", "N", valid["pzA24"]), get("pzsL", "N", SEEDS), valid["pzA24"], SEEDS)
        r["label"] = "UNDER-TRAINING RULED OUT" if r["ci"][1] < 0 else "NOT RULED OUT"
        if nA < 2:
            r["note"] = "one valid A0-24k seed: the seed level of the bootstrap is degenerate"
        C["u1 pi0-A0-24k - pi0-sL, N"] = r
        ps["u1"] = r["p_ge0"]
        r = NS.hboot(get("pzA24", "N", valid["pzA24"]), get("pzA0", "N", SEEDS), valid["pzA24"], SEEDS)
        r["label"] = "REPORTED"
        C["u2 pi0-A0-24k - pi0-A0 (6k), N"] = r
        res["descriptive"]["u1_paired_t_matched_seeds"] = tcheck([G[("pzA24", s, "N")].mean() - G[("pzsL", s, "N")].mean() for s in valid["pzA24"]])
        res["descriptive"]["u2_paired_t_matched_seeds"] = tcheck([G[("pzA24", s, "N")].mean() - G[("pzA0", s, "N")].mean() for s in valid["pzA24"]])
    else:
        for k in ("u1 pi0-A0-24k - pi0-sL, N", "u2 pi0-A0-24k - pi0-A0 (6k), N"):
            C[k] = dict(label="UNTESTABLE", reason="no valid pi0-A0-24k seed (K gate >= .70): %s" % valid["pzA24"])
    if nJ >= 3:
        r = NS.hboot(get("pzJ", "N", valid["pzJ"]), get("pzA0", "N", SEEDS), valid["pzJ"], SEEDS)
        lo, hi = r["ci"]
        r["label"] = "SUPPORTED" if lo > 0 else ("REVERSED" if hi < 0 else ("EQUIVALENT" if lo > -0.10 and hi < 0.10 else "NOT SHOWN"))
        C["j1 PRIMARY J - pi0-A0, N"] = r
        ps["j1"] = r["p_le0"]
        r = NS.hboot(get("pzJ", "N", valid["pzJ"]), get("pzsL", "N", SEEDS), valid["pzJ"], SEEDS)
        r["label"] = "NON-INFERIOR" if r["ci"][0] > -0.10 else ("COST" if r["ci"][1] < -0.10 else "NOT SHOWN")
        C["j2 J - pi0-sL, N (NI -.10)"] = r
        r = NS.hboot(get("pzJ", "K", valid["pzJ"]), get("pzA0", "K", SEEDS), valid["pzJ"], SEEDS)
        r["label"] = "NON-INFERIOR" if r["ci"][0] > -0.05 else ("COST" if r["ci"][1] < -0.05 else "NOT SHOWN")
        C["jK J - pi0-A0, K (NI -.05)"] = r
        res["descriptive"]["j1_paired_t_matched_seeds"] = tcheck([G[("pzJ", s, "N")].mean() - G[("pzA0", s, "N")].mean() for s in valid["pzJ"]])
        res["descriptive"]["j2_paired_t_matched_seeds"] = tcheck([G[("pzJ", s, "N")].mean() - G[("pzsL", s, "N")].mean() for s in valid["pzJ"]])
    else:
        for k in ("j1 PRIMARY J - pi0-A0, N", "j2 J - pi0-sL, N (NI -.10)", "jK J - pi0-A0, K (NI -.05)"):
            C[k] = dict(label="UNTESTABLE", reason="fewer than 3 valid J seeds (K gate >= .70): %s" % valid["pzJ"])
    res["holm"] = V.holm(ps)
    res["holm_note"] = "family {u1, j1}; survives iff adjusted < .025; present: %s" % sorted(ps)
    mags = []
    for s in SEEDS:
        f = os.path.join(NEW, "..", "nphj7", "jmag_s%d_6000.json" % s)
        if os.path.exists(f):
            m = json.load(open(f, encoding="utf-8"))
            mags.append(dict(seed=s, ratio_row0=m["ratio_row0_mean"], ratio_rows_mean=m["ratio_rows_mean"]))
    res["j_magnitude_5cm_over_servo"] = dict(per_seed=mags, mean_row0=float(np.mean([m["ratio_row0"] for m in mags])) if mags else None,
                                             mean_rows=float(np.mean([m["ratio_rows_mean"] for m in mags])) if mags else None, prediction=">= 0.5 (row 0)")
    print("[nphj-read]", json.dumps(res)[:6000], flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[nphj-read] -> %s" % out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("gate", "main"))
    ap.add_argument("--arm", choices=("pzA24", "pzJ"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gate_read(a) if a.what == "gate" else main_read(a)

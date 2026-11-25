"""NPHP reader (experiments/prereg/nphp_pred.txt).  Numpy only (openpi venv, compute node via sbatch).  Reuses nph_read.py's grid loaders unchanged.
Arms pqH / pqA0 / pqsL, seeds 42-45: files results/flawed/nphp8/a_<arm>_s<seed>/<arm>_s<seed>_main_<tid>.json (8th set k = 940-999, trial t = k - 940).
  gate --seed S --out <json>   : pi0.5-H* seed S K1-K6 mean >= .70 ("passes")   [outputs opened with "x"]
  main --out <json>            : p1 (PRIMARY), p2, pd, pK, seed SD per arm, Holm over {p1, pd}.
Hierarchical bootstrap (every interval): resample seeds with replacement within each arm (independently), then states within each task (one resample per task, shared by the arms; tasks fixed);
B = 2000, np.random.default_rng(0), 95% percentile interval; point = mean over valid seeds of the task-mean difference.  Valid H* seeds = gate passes; fewer than 3 -> p1 / p2 / pK UNTESTABLE.
Labels: p1 SUPPORTED lower > 0 | REVERSED upper < 0 | EQUIVALENT inside +-.10 | NOT SHOWN.  p2 NON-INFERIOR lower > -.10 (COST upper < -.10).  pd SUPPORTED upper < 0.  pK NON-INFERIOR lower > -.05.
Holm: one-sided bootstrap p = (1 + #draws on the wrong side of 0) / (B + 1); survives iff adjusted < .025."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nph_read as NR                                                        # noqa: E402

RR, V = NR.RR, NR.V
BASE = os.path.join(RR.OUT, "nphp8")
SEEDS = (42, 43, 44, 45)
B = 2000


def name(arm, s):
    return "%s_s%d" % (arm, s)


def amap():
    return {name(a, s): (name(a, s), name(a, s)) for a in ("pqH", "pqA0", "pqsL") for s in SEEDS}


def paired_t(d):
    """Descriptive check (not registered): paired t over the per-seed differences."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return dict(n=n, mean=float(d.mean()) if n else None, t=None, df=n - 1, p_one_sided=None)
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    try:
        from scipy import stats
        p = float(stats.t.sf(t, n - 1))
    except Exception:
        p = None
    return dict(n=n, mean=float(d.mean()), sd=float(d.std(ddof=1)), t=t, df=n - 1, p_one_sided_pos=p)


def hboot(A, B_, sa, sb):
    """A, B_: dict seed -> [6, 60] grid.  -> dict(delta, ci, p_le0, p_ge0)."""
    XA, XB = np.stack([A[s] for s in sa]), np.stack([B_[s] for s in sb])          # [m, 6, 60]
    delta = float(XA.mean() - XB.mean())
    rng = np.random.default_rng(0)
    nt, ns = XA.shape[1], XA.shape[2]
    draws = np.empty(B)
    for b in range(B):
        ia = rng.integers(0, len(sa), len(sa))
        ib = rng.integers(0, len(sb), len(sb))
        idx = rng.integers(0, ns, (nt, ns))
        ga = np.take_along_axis(XA[ia], np.broadcast_to(idx, (len(sa), nt, ns)), axis=2).mean()
        gb = np.take_along_axis(XB[ib], np.broadcast_to(idx, (len(sb), nt, ns)), axis=2).mean()
        draws[b] = ga - gb
    lo, hi = np.percentile(draws, [2.5, 97.5])
    psd = {str(s): float(A[s].mean() - B_[s].mean()) for s in sa if s in B_}
    return dict(paired_t=paired_t(list(psd.values())), delta=delta, ci=[float(lo), float(hi)], p_le0=float((1 + (draws <= 0).sum()) / (B + 1)), p_ge0=float((1 + (draws >= 0).sum()) / (B + 1)),
                seeds_a=list(sa), seeds_b=list(sb), per_seed_delta={str(s): float(A[s].mean() - B_[s].mean()) for s in sa if s in B_})


def main_read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    meta, am = RR.n2(), amap()
    res = dict(states="n2 familiar fresh set k = 940-999 (trial t = k - 940)", B=B, gates={}, success={}, seed_sd={}, contrasts={}, notes=[])
    valid = []
    for s in SEEDS:
        g = json.load(open(os.path.join(BASE, "nphp_gate_s%d.json" % s), encoding="utf-8"))["gate"]
        res["gates"][str(s)] = g
        if g["passes"]:
            valid.append(s)
    res["valid_hstar_seeds"] = valid
    G = {}
    for arm in ("pqH", "pqA0", "pqsL"):
        for s in SEEDS:
            if arm == "pqH" and s not in valid:
                continue
            for grp, comps in (("K", NR.KNOWN), ("N", NR.NEW)):
                G[(arm, s, grp)] = NR.grid(meta, BASE, am, name(arm, s), comps)
                res["success"]["%s|s%d|%s" % (arm, s, grp)] = float(G[(arm, s, grp)].mean())
    for arm in ("pqH", "pqA0", "pqsL"):
        for grp in ("N", "K"):
            v = [G[(arm, s, grp)].mean() for s in SEEDS if (arm, s, grp) in G]
            res["seed_sd"]["%s|%s" % (arm, grp)] = dict(n_seeds=len(v), mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
    get = lambda arm, grp, ss: {s: G[(arm, s, grp)] for s in ss}                 # noqa: E731
    C = res["contrasts"]
    ps = {}
    if len(valid) >= 3:
        r = hboot(get("pqH", "N", valid), get("pqA0", "N", valid), valid, valid)
        lo, hi = r["ci"]
        r["label"] = "SUPPORTED" if lo > 0 else ("REVERSED" if hi < 0 else ("EQUIVALENT" if lo > -0.10 and hi < 0.10 else "NOT SHOWN"))
        C["p1 PRIMARY pi0.5-H* - pi0.5-A0, N, valid pairs"] = r
        ps["p1"] = r["p_le0"]
        r = hboot(get("pqH", "N", valid), get("pqsL", "N", valid), valid, valid)
        r["label"] = "NON-INFERIOR" if r["ci"][0] > -0.10 else ("COST" if r["ci"][1] < -0.10 else "NOT SHOWN")
        C["p2 pi0.5-H* - pi0.5-sL, N (NI -.10)"] = r
        r = hboot(get("pqH", "K", valid), get("pqA0", "K", valid), valid, valid)
        r["label"] = "NON-INFERIOR" if r["ci"][0] > -0.05 else ("COST" if r["ci"][1] < -0.05 else "NOT SHOWN")
        C["pK pi0.5-H* - pi0.5-A0, K (NI -.05)"] = r
    else:
        for k in ("p1 PRIMARY pi0.5-H* - pi0.5-A0, N, valid pairs", "p2 pi0.5-H* - pi0.5-sL, N (NI -.10)", "pK pi0.5-H* - pi0.5-A0, K (NI -.05)"):
            C[k] = dict(label="UNTESTABLE", reason="fewer than 3 valid pi0.5-H* seeds (K gate >= .70): %s" % valid)
    r = hboot(get("pqA0", "N", SEEDS), get("pqsL", "N", SEEDS), SEEDS, SEEDS)
    r["label"] = "SUPPORTED" if r["ci"][1] < 0 else ("REVERSED" if r["ci"][0] > 0 else "NOT SHOWN")
    C["pd pi0.5-A0 - pi0.5-sL, N, all seeds"] = r
    ps["pd"] = r["p_ge0"]
    res["holm"] = V.holm(ps)
    res["holm_note"] = "family {p1, pd}; survives iff adjusted < .025; present: %s" % sorted(ps)
    print("[nphp-read]", json.dumps(res)[:6000], flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[nphp-read] -> %s" % out, flush=True)


def gate_read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    meta, am = RR.n2(), amap()
    nm = name("pqH", a.seed)
    assert NR.have(meta, BASE, am, nm, NR.KNOWN), "pqH K files incomplete"
    g = NR.grid(meta, BASE, am, nm, NR.KNOWN)
    res = dict(seed=a.seed, gate=dict(K_mean=float(g.mean()), per_task={c: float(g[i].mean()) for i, c in enumerate(NR.KNOWN)}, threshold=0.70, passes=bool(g.mean() >= 0.70)))
    print("[nphp-read]", json.dumps(res), flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("gate", "main"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gate_read(a) if a.what == "gate" else main_read(a)

"""NPHS reader (experiments/prereg/nphs_pred.txt).  Numpy only (openpi venv, compute node via sbatch).  Reuses nph_read.py's grid loaders unchanged.
Arms pzH / pzA0 / pzsL, seeds 42-45: files results/flawed/nphs7/a_<arm>_s<seed>/<arm>_s<seed>_main_<tid>.json (7th set k = 880-939, trial t = k - 880).
  gate --seed S --out <json>   : pi0-H* seed S K1-K6 mean >= .70 ("passes")   [outputs opened with "x"]
  main --out <json>            : n1s (PRIMARY), n2s, d1s, nKs, seed SD per arm, Holm over {n1s, d1s}.
Hierarchical bootstrap (every interval): resample seeds with replacement within each arm (independently), then states within each task (one resample per task, shared by the arms; tasks fixed);
B = 2000, np.random.default_rng(0), 95% percentile interval; point = mean over valid seeds of the task-mean difference.  Valid H* seeds = gate passes; fewer than 3 -> n1s / n2s / nKs UNTESTABLE.
Labels: n1s SUPPORTED lower > 0 | REVERSED upper < 0 | EQUIVALENT inside +-.10 | NOT SHOWN.  n2s NON-INFERIOR lower > -.10 (COST upper < -.10).  d1s SUPPORTED upper < 0.  nKs NON-INFERIOR lower > -.05.
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
BASE = os.path.join(RR.OUT, "nphs7")
SEEDS = (42, 43, 44, 45)
B = 2000


def name(arm, s):
    return "%s_s%d" % (arm, s)


def amap():
    return {name(a, s): (name(a, s), name(a, s)) for a in ("pzH", "pzA0", "pzsL") for s in SEEDS}


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
    return dict(delta=delta, ci=[float(lo), float(hi)], p_le0=float((1 + (draws <= 0).sum()) / (B + 1)), p_ge0=float((1 + (draws >= 0).sum()) / (B + 1)),
                seeds_a=list(sa), seeds_b=list(sb), per_seed_delta={str(s): float(A[s].mean() - B_[s].mean()) for s in sa if s in B_})


def main_read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    meta, am = RR.n2(), amap()
    res = dict(states="n2 familiar fresh set k = 880-939 (trial t = k - 880)", B=B, gates={}, success={}, seed_sd={}, contrasts={}, notes=[])
    valid = []
    for s in SEEDS:
        g = json.load(open(os.path.join(BASE, "nphs_gate_s%d.json" % s), encoding="utf-8"))["gate"]
        res["gates"][str(s)] = g
        if g["passes"]:
            valid.append(s)
    res["valid_hstar_seeds"] = valid
    G = {}
    for arm in ("pzH", "pzA0", "pzsL"):
        for s in SEEDS:
            if arm == "pzH" and s not in valid:
                continue
            for grp, comps in (("K", NR.KNOWN), ("N", NR.NEW)):
                G[(arm, s, grp)] = NR.grid(meta, BASE, am, name(arm, s), comps)
                res["success"]["%s|s%d|%s" % (arm, s, grp)] = float(G[(arm, s, grp)].mean())
    for arm in ("pzH", "pzA0", "pzsL"):
        for grp in ("N", "K"):
            v = [G[(arm, s, grp)].mean() for s in SEEDS if (arm, s, grp) in G]
            res["seed_sd"]["%s|%s" % (arm, grp)] = dict(n_seeds=len(v), mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
    get = lambda arm, grp, ss: {s: G[(arm, s, grp)] for s in ss}                 # noqa: E731
    C = res["contrasts"]
    ps = {}
    if len(valid) >= 3:
        r = hboot(get("pzH", "N", valid), get("pzA0", "N", valid), valid, valid)
        lo, hi = r["ci"]
        r["label"] = "SUPPORTED" if lo > 0 else ("REVERSED" if hi < 0 else ("EQUIVALENT" if lo > -0.10 and hi < 0.10 else "NOT SHOWN"))
        C["n1s PRIMARY pi0-H* - pi0-A0, N, valid pairs"] = r
        ps["n1s"] = r["p_le0"]
        r = hboot(get("pzH", "N", valid), get("pzsL", "N", valid), valid, valid)
        r["label"] = "NON-INFERIOR" if r["ci"][0] > -0.10 else ("COST" if r["ci"][1] < -0.10 else "NOT SHOWN")
        C["n2s pi0-H* - pi0-sL, N (NI -.10)"] = r
        r = hboot(get("pzH", "K", valid), get("pzA0", "K", valid), valid, valid)
        r["label"] = "NON-INFERIOR" if r["ci"][0] > -0.05 else ("COST" if r["ci"][1] < -0.05 else "NOT SHOWN")
        C["nKs pi0-H* - pi0-A0, K (NI -.05)"] = r
    else:
        for k in ("n1s PRIMARY pi0-H* - pi0-A0, N, valid pairs", "n2s pi0-H* - pi0-sL, N (NI -.10)", "nKs pi0-H* - pi0-A0, K (NI -.05)"):
            C[k] = dict(label="UNTESTABLE", reason="fewer than 3 valid pi0-H* seeds (K gate >= .70): %s" % valid)
    r = hboot(get("pzA0", "N", SEEDS), get("pzsL", "N", SEEDS), SEEDS, SEEDS)
    r["label"] = "SUPPORTED" if r["ci"][1] < 0 else ("REVERSED" if r["ci"][0] > 0 else "NOT SHOWN")
    C["d1s pi0-A0 - pi0-sL, N, all seeds"] = r
    ps["d1s"] = r["p_ge0"]
    res["holm"] = V.holm(ps)
    res["holm_note"] = "family {n1s, d1s}; survives iff adjusted < .025; present: %s" % sorted(ps)
    print("[nphs-read]", json.dumps(res)[:6000], flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[nphs-read] -> %s" % out, flush=True)


def gate_read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    meta, am = RR.n2(), amap()
    nm = name("pzH", a.seed)
    assert NR.have(meta, BASE, am, nm, NR.KNOWN), "pzH K files incomplete"
    g = NR.grid(meta, BASE, am, nm, NR.KNOWN)
    res = dict(seed=a.seed, gate=dict(K_mean=float(g.mean()), per_task={c: float(g[i].mean()) for i, c in enumerate(NR.KNOWN)}, threshold=0.70, passes=bool(g.mean() >= 0.70)))
    print("[nphs-read]", json.dumps(res), flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("gate", "main"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gate_read(a) if a.what == "gate" else main_read(a)

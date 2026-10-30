"""FLAWED_COMPOSITION D1x / D2 read (sections 4.7, 4.8; m3_prereg tag pd2).  CPU.

D1x (main2, C1; results/flawed/d1x_<m>.json).  Per compound C, task means of the action error with C's own prefix (E_true) and
with the interchanged prefix (E_rec).  rho_in = (mean E_rec - mean E_true) / mean E_true over the in-support compounds, rho_dev the
same over the dev compounds, delta = rho_dev - rho_in (the expert-read combination debt); agreement = the share of frames whose
recombined output is nearer to the true compound's than to A's and to B's.  CIs: 2,000-draw task bootstraps (seed 0).
  D1x-a  factored on support: holds if the upper CI bound of rho_in < 0.10; falsified if the lower bound > 0.10; else inconclusive
  D1x-b  expert-read combination debt: present if delta > 0 with CI excluding 0; absent if the CI includes 0; negative otherwise

D2 (results/flawed/d2_<model>.json, d2_support.json; outcomes = the policy-alone arm `zs`, 20 episodes per task, of g8_read.json
(pi0.5, LIBERO-90), g15_read_pi05.json, g18_read_groot_libero.json, g18_read_xvla.json (Li OOD)).  Rows = (block, task):
  failure y = 1 - zs / 20;  purity P = mean over roles of other / own (roles with own > 0);  bypass B = d_byp / (d_byp + d_role);
  debt x = mean of the available within-block ranks of P and B (rows with neither are dropped);  controls = the D0-style split score
  and u (fc_d2_support.py).
  KEY  pooled partial Spearman: within each block every variable is rank-transformed to (0, 1]; x and y are each residualised by
       OLS on [score, u, block dummies]; rho = Pearson(residuals).  CI: 2,000-draw bootstrap over task identities (a Li task carries
       its rows of all three models).  Holds if rho > 0 and the CI excludes 0; falsified otherwise ("debt adds nothing beyond
       novelty").  Per block: the same within the block.  Secondary: P alone, B alone, pi0.5's combination error on LIBERO-90,
       the raw (uncontrolled) Spearman, and the pool of the two most graded blocks (pi0.5-LI, GR00T-LI).
-> results/flawed/d2_read.json

    python experiments/fc_d2_read.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
OUT = os.path.join(ROOT, "results", "flawed")
RR = os.path.join(ROOT, "results", "robot")
NB = 2000
DIR, SUF = OUT, ""
BLOCKS = (("pi05", "L90", "g8_read.json"), ("pi05", "LI", "g15_read_pi05.json"), ("groot", "LI", "g18_read_groot_libero.json"),
          ("xvla", "LI", "g18_read_xvla.json"))


def rank01(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r / len(x)


def pmean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def boot_mean(v, seed=0):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    rng = np.random.default_rng(seed)
    bt = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return dict(mean=float(v.mean()), ci95=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))], n=len(v))


def d1x():
    res = {}
    for m in ("main2", "c1"):
        R = json.load(open(os.path.join(DIR, f"d1x_{m}{SUF}.json"), encoding="utf-8"))["records"]
        per = {}
        for r in R:
            per.setdefault((r["set"], r["tid"]), []).append(r)
        T = {s: {t: dict(true=np.mean([x["err"]["true"] for x in v]), rec=np.mean([x["err"]["rec"] for x in v]),
                         A=np.mean([x["err"]["A"] for x in v]), B=np.mean([x["err"]["B"] for x in v]),
                         agree=np.mean([x["agree"] for x in v]), n=len(v))
                 for (ss, t), v in per.items() if ss == s} for s in ("IN", "DEV")}
        rng = np.random.default_rng(0)

        def rel(d, idx):
            tr, rc = np.array([d[k]["true"] for k in idx]), np.array([d[k]["rec"] for k in idx])
            return (rc.mean() - tr.mean()) / tr.mean()
        kin, kdv = sorted(T["IN"]), sorted(T["DEV"])
        bi, bd = [], []
        for _ in range(NB):
            si = [kin[i] for i in rng.integers(0, len(kin), len(kin))]
            sd = [kdv[i] for i in rng.integers(0, len(kdv), len(kdv))]
            bi.append(rel(T["IN"], si)), bd.append(rel(T["DEV"], sd))
        bi, bd = np.array(bi), np.array(bd)
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]   # noqa: E731
        row = dict(n_in_tasks=len(kin), n_dev_tasks=len(kdv), n_in_frames=int(sum(T["IN"][k]["n"] for k in kin)),
                   n_dev_frames=int(sum(T["DEV"][k]["n"] for k in kdv)))
        for s, ks in (("IN", kin), ("DEV", kdv)):
            for q in ("true", "rec", "A", "B"):
                row[f"{s}_E_{q}"] = float(np.mean([T[s][k][q] for k in ks]))
            row[f"{s}_agree"] = boot_mean([T[s][k]["agree"] for k in ks])
        row["rho_in"], row["rho_in_ci95"] = float(rel(T["IN"], kin)), ci(bi)
        row["rho_dev"], row["rho_dev_ci95"] = float(rel(T["DEV"], kdv)), ci(bd)
        row["delta"], row["delta_ci95"] = row["rho_dev"] - row["rho_in"], ci(bd - bi)
        row["D1x_a"] = "holds" if row["rho_in_ci95"][1] < 0.10 else ("falsified" if row["rho_in_ci95"][0] > 0.10 else "inconclusive")
        row["D1x_b"] = ("present" if row["delta_ci95"][0] > 0 else "absent" if row["delta_ci95"][0] <= 0 <= row["delta_ci95"][1] else "negative")
        row["per_task"] = {s: {k: {kk: float(vv) for kk, vv in v.items()} for k, v in T[s].items()} for s in T}
        res[m] = row
    return res


def partial(rows, xkey, ykey="y", controls=("score", "u")):
    blocks = sorted({r["block"] for r in rows})
    X = np.array([r[xkey] for r in rows], float)
    Y = np.array([r[ykey] for r in rows], float)
    Z = [np.array([r[c] for r in rows], float) for c in controls]
    D = [np.array([r["block"] == b for r in rows], float) for b in blocks]
    M = np.column_stack(Z + D) if len(blocks) > 1 else np.column_stack(Z + [np.ones(len(rows))])
    rx = X - M @ np.linalg.lstsq(M, X, rcond=None)[0]
    ry = Y - M @ np.linalg.lstsq(M, Y, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def ranked(rows, keys):
    out = [dict(r) for r in rows]
    for b in {r["block"] for r in rows}:
        idx = [i for i, r in enumerate(out) if r["block"] == b]
        for k in keys:
            v = rank01([out[i][k] for i in idx])
            for i, x in zip(idx, v):
                out[i][k] = float(x)
    return out


def boot_partial(rows, xkey, controls=("score", "u"), raw=False, seed=0):
    rows = [r for r in rows if r.get(xkey) is not None]
    keys = [xkey, "y"] + list(controls)

    def stat(rs):
        rs = ranked(rs, keys)
        if raw:
            x, y = np.array([r[xkey] for r in rs]), np.array([r["y"] for r in rs])
            return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan
        return partial(rs, xkey, controls=controls)
    point = stat(rows)
    tasks = sorted({r["task"] for r in rows})
    by = {t: [r for r in rows if r["task"] == t] for t in tasks}
    rng = np.random.default_rng(seed)
    bt = []
    for _ in range(NB):
        rs = [r for t in rng.choice(tasks, len(tasks)) for r in by[t]]
        bt.append(stat(rs))
    bt = np.array(bt)
    ok = bt[np.isfinite(bt)]
    return dict(rho=point, ci95=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) else None,
                n_rows=len(rows), n_tasks=len(tasks), nan_draws=int((~np.isfinite(bt)).sum()))


def d2():
    sup = json.load(open(os.path.join(OUT, "d2_support.json"), encoding="utf-8"))
    rows, table, dist = [], {}, {}
    for model, blk, f in BLOCKS:
        D = json.load(open(os.path.join(DIR, f"d2_{model}{SUF}.json"), encoding="utf-8"))["blocks"][blk]
        zs = {k: v["zs"] for k, v in json.load(open(os.path.join(RR, f), encoding="utf-8"))["per_task"].items()}
        name = f"{model}-{blk}"
        br = []
        for key, r in D.items():
            okey = key.split(":")[1] if blk == "L90" else key
            if okey not in zs:
                continue
            p = r.get("purity")
            P = pmean([p["obj_other"] / p["obj_own"] if p and p["obj_own"] > 0 else None,
                       p["goal_other"] / p["goal_own"] if p and p["goal_own"] > 0 else None]) if p else None
            b = r.get("bypass")
            B = b["d_byp"] / (b["d_byp"] + b["d_role"]) if b and (b["d_byp"] + b["d_role"]) > 0 else None
            br.append(dict(block=name, task=key, y=1.0 - zs[okey] / 20.0, zs=zs[okey], P=P, B=B, comb=r.get("comb_err"),
                           score=sup[blk][key]["score"], u=sup[blk][key]["u"]))
        for k in ("P", "B"):
            have = [i for i, r in enumerate(br) if r[k] is not None]
            rk = rank01([br[i][k] for i in have])
            for i, v in zip(have, rk):
                br[i][f"r{k}"] = float(v)
        for r in br:
            r["x"] = pmean([r.get("rP"), r.get("rB")])
        br = [r for r in br if r["x"] is not None]
        rows += br
        z = np.array([r["zs"] for r in br])
        dist[name] = dict(n=len(z), mean_success=float(z.mean() / 20), at_0=int((z == 0).sum()), at_20=int((z == 20).sum()),
                          between=int(((z > 0) & (z < 20)).sum()), hist_0_1_5_6_14_15_19_20=[int((z == 0).sum()), int(((z >= 1) & (z <= 5)).sum()),
                                                                                              int(((z >= 6) & (z <= 14)).sum()), int(((z >= 15) & (z <= 19)).sum()), int((z == 20).sum())])
        table[name] = dict(purity=boot_mean([r["P"] for r in br if r["P"] is not None]),
                           bypass=boot_mean([r["B"] for r in br if r["B"] is not None]),
                           combination=boot_mean([r["comb"] for r in br if r["comb"] is not None]) if any(r["comb"] is not None for r in br) else None,
                           n_purity=sum(r["P"] is not None for r in br), n_bypass=sum(r["B"] is not None for r in br),
                           per_block_key=boot_partial(br, "x"), per_block_raw=boot_partial(br, "x", raw=True),
                           per_block_P=boot_partial(br, "P"), per_block_B=boot_partial(br, "B"))
        if blk == "L90":
            table[name]["per_block_comb"] = boot_partial(br, "comb")
    key = boot_partial(rows, "x")
    res = dict(rows=rows, success_distribution=dist, table=table, KEY=key,
               KEY_verdict="holds" if (key["rho"] > 0 and key["ci95"][0] > 0) else "falsified: debt adds nothing beyond novelty",
               pooled_raw=boot_partial(rows, "x", raw=True), pooled_P=boot_partial(rows, "P"), pooled_B=boot_partial(rows, "B"),
               pooled_graded2=boot_partial([r for r in rows if r["block"] in ("pi05-LI", "groot-LI")], "x"))
    return res


def main():
    global DIR, SUF
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=OUT)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    DIR, SUF = a.dir, a.suffix
    res = dict(D1x=d1x(), D2=d2())
    with open(os.path.join(DIR, f"d2_read{SUF}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    if SUF:
        print("[d2_read] smoke ok")
        return
    show = dict(D1x={m: {k: v for k, v in r.items() if k != "per_task"} for m, r in res["D1x"].items()},
                D2={k: v for k, v in res["D2"].items() if k != "rows"})
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()

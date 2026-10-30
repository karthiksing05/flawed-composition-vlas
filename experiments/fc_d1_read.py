"""FLAWED_COMPOSITION D1 read: aggregate experiments/fc_d1_debt.py's per-frame records into the debt table and test the pre-registered
predictions D1a, D1b, D1c and D0 (docs/FLAWED_COMPOSITION.md section 4.5; m3_prereg tag pd01).  CPU.

Task sets.  In-support = the 72 training tasks every model trained on (the 76 minus C1's 4 held-out-layout tasks); its LIBERO-90
subset (43 cells) is the in-support comparator of the combination debt (same demo source and scene families as the dev tasks).
Dev = the 29 dev tasks.  Every CI is a 2,000-draw percentile bootstrap over tasks (seed 0).

  combination debt   E_dev(d) = mean intact action error over task d's dev frames; E_in = mean over the 43 in-support LIBERO-90
                     cells of their mean intact error on held-out episodes; delta_F_rel = (mean_d E_dev(d) - E_in) / E_in (dev and
                     in-support tasks resampled independently); per task delta_F(d) = E_dev(d) - E_in
  purity debt        per task, sum of the message's movement under the other-role swap / sum under the own-role swap; mean over
                     tasks; delta_P = mean of the object and goal values.  Levels: tok (binder output; BindVLA only), kv
  bypass debt        per task, sum delta_byp / (sum delta_byp + sum delta_role) over frames with a same-scene partner; mean over tasks
  insufficiency      I = mean over in-support tasks of the mean action error with the bypass cut (cut); also cutS and I - E_in
  D1a  main2: holds if the lower CI bound of delta_F_rel > the upper CI bound of delta_P (tok); falsified if delta_F_rel <= delta_P
       (point estimates); otherwise inconclusive
  D1b  main2: Spearman rho(delta_F(d), dev failure(d)) over the 29 tasks; holds if rho > 0 and its CI excludes 0; falsified otherwise
  D1c  paired over in-support tasks: dB = B_C1 - B_main2, dI = I_C1 - I_main2.  Holds if CI(dB) < 0 and CI(dI) > 0; falsified if
       both fall (CI(dB) < 0 and CI(dI) < 0) or neither changes (both CIs include 0); otherwise partial
  D0   Spearman rho(D0 predicted-debt score, delta_F_main2(d)) over the 29 tasks; holds if rho > 0 and its CI excludes 0
-> results/flawed/d1_read.json

    python experiments/fc_d1_read.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
OUT = os.path.join(ROOT, "results", "flawed")
C1_HELD = ("libero_90:37", "libero_object:9", "libero_spatial:8", "libero_spatial:9")
NB = 2000
MODELS = ("main2", "c1", "vanilla")


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def boot_spear(a, b, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bt = []
    for _ in range(NB):
        i = rng.integers(0, len(a), len(a))
        bt.append(spearman(a[i], b[i]))
    bt = np.asarray(bt)
    ok = bt[~np.isnan(bt)]
    return dict(rho=spearman(a, b), ci95=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) else None,
                n=len(a), nan_draws=int(np.isnan(bt).sum()))


def boot_mean(v, seed=0):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    rng = np.random.default_rng(seed)
    bt = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return dict(mean=float(v.mean()), ci95=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))], n=len(v))


def per_task(rows, f):
    out = {}
    for r in rows:
        x = f(r)
        if x is not None:
            out.setdefault(r["tid"], []).append(x)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=OUT, help="where the d1_<model>.json files are (smoke: the scratch dir)")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    d0 = json.load(open(os.path.join(OUT, "d0_support.json"), encoding="utf-8"))["per_task"]
    c1r = json.load(open(os.path.join(ROOT, "results", "bindvla", "c1_read.json"), encoding="utf-8"))["per_task_c1"]
    from lifelong_ood import l90c
    sp = l90c.load_split()
    support = [t for t in sp["train"] if t not in C1_HELD]
    l90_support = [t for t in support if t.startswith("libero_90:")]
    dev = [t for b in ("0.1", "0.3", "0.5", "part") for t in sp["dev"][b]]
    D = {m: json.load(open(os.path.join(a.dir, f"d1_{m}{a.suffix}.json"), encoding="utf-8")) for m in MODELS}
    res = {"task_sets": dict(support=len(support), l90_support=len(l90_support), dev=len(dev))}
    table, pert = {}, {}
    for m in MODELS:
        R = D[m]["records"]
        IN = [r for r in R if r["set"] == "IN" and r["tid"] in support]
        DV = [r for r in R if r["set"] == "DEV"]
        row = {}
        # combination debt (primary: action error; secondary: grasp-only, release-only, flow, site)
        for name, sel, val in (("err", None, lambda r: r["err"]["intact"]), ("err_grasp", "grasp", lambda r: r["err"]["intact"]),
                               ("err_release", "release", lambda r: r["err"]["intact"]), ("flow", None, lambda r: r["flow"]),
                               ("site", None, lambda r: r["site"])):
            fi = per_task([r for r in IN if r["tid"] in l90_support and (sel is None or r["phase"] == sel)], val)
            fd = per_task([r for r in DV if sel is None or r["phase"] == sel], val)
            ein = np.array([np.mean(fi[t]) for t in l90_support if t in fi])
            edv = {t: float(np.mean(fd[t])) for t in dev if t in fd}
            ed = np.array(list(edv.values()))
            rng = np.random.default_rng(0)
            bt = [(ed[rng.integers(0, len(ed), len(ed))].mean() - (e := ein[rng.integers(0, len(ein), len(ein))].mean())) / e
                  for _ in range(NB)]
            row[f"combination_{name}"] = dict(E_in=float(ein.mean()), E_dev=float(ed.mean()), rel=float((ed.mean() - ein.mean()) / ein.mean()),
                                              ci95=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                                              n_in_tasks=len(ein), n_dev_tasks=len(ed))
            if name == "err":
                pert[m] = {t: v - float(ein.mean()) for t, v in edv.items()}
        # purity
        P = [p for p in D[m]["purity"] if p["tid"] in support]
        for lvl in ("tok", "kv"):
            for role in ("obj", "goal"):
                k_o, k_w = f"{lvl}_{role}_other", f"{lvl}_{role}_own"
                rows = [p for p in P if k_o in p]
                if not rows:
                    continue
                num, den = per_task(rows, lambda p: p[k_o]), per_task(rows, lambda p: p[k_w])
                v = [float(np.sum(num[t]) / np.sum(den[t])) for t in num if np.sum(den[t]) > 0]
                row[f"purity_{lvl}_{role}"] = boot_mean(v)
            if f"purity_{lvl}_obj" in row and f"purity_{lvl}_goal" in row:
                tk = sorted(set(per_task(P, lambda p: p.get(f"{lvl}_obj_other"))) & set(per_task(P, lambda p: p.get(f"{lvl}_goal_other"))))
                num_o, den_o = per_task(P, lambda p: p.get(f"{lvl}_obj_other")), per_task(P, lambda p: p.get(f"{lvl}_obj_own"))
                num_g, den_g = per_task(P, lambda p: p.get(f"{lvl}_goal_other")), per_task(P, lambda p: p.get(f"{lvl}_goal_own"))
                v = [0.5 * (np.sum(num_o[t]) / np.sum(den_o[t]) + np.sum(num_g[t]) / np.sum(den_g[t])) for t in tk
                     if np.sum(den_o[t]) > 0 and np.sum(den_g[t]) > 0]
                row[f"purity_{lvl}"] = boot_mean(v)
        # bypass
        bp = [r for r in IN if "role" in r["delta"]]
        for k in ("byp", "st", "glob"):
            if not any(k in r["delta"] for r in bp):
                continue
            num, den = per_task(bp, lambda r: r["delta"][k]), per_task(bp, lambda r: r["delta"]["role"])
            v = {t: float(np.sum(num[t]) / (np.sum(num[t]) + np.sum(den[t]))) for t in num if np.sum(num[t]) + np.sum(den[t]) > 0}
            row[f"bypass_{k}"] = boot_mean(list(v.values()))
            row[f"bypass_{k}"]["per_task"] = v
            row[f"delta_{k}_raw"] = boot_mean([np.mean(num[t]) for t in num])
        row["delta_role_raw"] = boot_mean([np.mean(x) for x in per_task(bp, lambda r: r["delta"]["role"]).values()])
        # insufficiency
        for k in ("intact", "cut", "cutS"):
            v = per_task(IN, lambda r: r["err"][k])
            row[f"in_support_err_{k}"] = boot_mean([np.mean(x) for x in v.values()])
            row[f"in_support_err_{k}"]["per_task"] = {t: float(np.mean(x)) for t, x in v.items()}
        row["probe"] = D[m]["probe"]
        table[m] = row
    res["table"] = table
    # D1a
    t2 = table["main2"]
    F_, P_ = t2["combination_err"], t2.get("purity_tok")
    d1a = dict(delta_F_rel=F_["rel"], delta_F_ci95=F_["ci95"], delta_P=P_["mean"], delta_P_ci95=P_["ci95"])
    d1a["verdict"] = "holds" if F_["ci95"][0] > P_["ci95"][1] else ("falsified" if F_["rel"] <= P_["mean"] else "inconclusive")
    res["D1a"] = d1a
    # D1b
    key = {"main2": "main2", "c1": "c1", "vanilla": "vanilla"}
    d1b = {}
    for m in MODELS:
        ts = [t for t in dev if t in pert[m]]
        fail = [1.0 - c1r[t][key[m]] for t in ts]
        d1b[m] = boot_spear([pert[m][t] for t in ts], fail)
    cl = {}
    for m, f in (("main2", "cldiag_bindvla_main2_60000.json"), ("c1", "cldiag_bindvla_c1main_20000.json")):
        rows = json.load(open(os.path.join(ROOT, "results", "bindvla", f), encoding="utf-8"))["rows"]
        per = {}
        for r in rows:
            if r["named"] is None:
                continue
            per.setdefault(r["tid"], []).append(float(r["success"] or r["first_contact"] == "named"))
        ts = [t for t in dev if t in per and t in pert[m]]
        cl[m] = boot_spear([pert[m][t] for t in ts], [1.0 - np.mean(per[t]) for t in ts])
    v = d1b["main2"]
    d1b["verdict_main2"] = "holds" if (v["rho"] > 0 and v["ci95"][0] > 0) else "falsified"
    d1b["secondary_named_contact_failure"] = cl
    res["D1b"] = d1b
    # D1c
    sup_b = set(table["main2"]["bypass_byp"]["per_task"]) & set(table["c1"]["bypass_byp"]["per_task"])
    tb = sorted(sup_b)
    dB = np.array([table["c1"]["bypass_byp"]["per_task"][t] - table["main2"]["bypass_byp"]["per_task"][t] for t in tb])
    ti = sorted(set(table["main2"]["in_support_err_cut"]["per_task"]) & set(table["c1"]["in_support_err_cut"]["per_task"]))
    dI = np.array([table["c1"]["in_support_err_cut"]["per_task"][t] - table["main2"]["in_support_err_cut"]["per_task"][t] for t in ti])
    bB, bI = boot_mean(dB), boot_mean(dI)
    lowB, highI = bB["ci95"][1] < 0, bI["ci95"][0] > 0
    fallI = bI["ci95"][1] < 0
    noneB, noneI = bB["ci95"][0] <= 0 <= bB["ci95"][1], bI["ci95"][0] <= 0 <= bI["ci95"][1]
    verdict = "holds" if (lowB and highI) else ("falsified" if ((lowB and fallI) or (noneB and noneI)) else "partial")
    dS = [table["c1"]["bypass_st"]["per_task"][t] - table["main2"]["bypass_st"]["per_task"][t]
          for t in sorted(set(table["c1"]["bypass_st"]["per_task"]) & set(table["main2"]["bypass_st"]["per_task"]))]
    res["D1c"] = dict(dB=bB, dI=bI, verdict=verdict, n_tasks_B=len(tb), n_tasks_I=len(ti), secondary_dB_state_only=boot_mean(dS))
    # D0
    ts = [t for t in dev if t in pert["main2"]]
    d0r = {"primary": boot_spear([d0[t]["score_all"] for t in ts], [pert["main2"][t] for t in ts])}
    d0r["verdict"] = "holds" if (d0r["primary"]["rho"] > 0 and d0r["primary"]["ci95"][0] > 0) else "falsified"
    for m in ("c1", "vanilla"):
        d0r[f"score_all_vs_{m}"] = boot_spear([d0[t]["score_all"] for t in ts], [pert[m][t] for t in ts])
    for s in ("score_binder", "score_motor"):
        d0r[f"{s}_vs_main2"] = boot_spear([d0[t][s] for t in ts], [pert["main2"][t] for t in ts])
    binord = {"0.1": 0, "0.3": 1, "0.5": 2, "part": 3}
    d0r["bin_only_vs_main2"] = boot_spear([binord[d0[t]["bin"]] for t in ts], [pert["main2"][t] for t in ts])
    d0r["score_all_vs_failure_main2"] = boot_spear([d0[t]["score_all"] for t in ts], [1.0 - c1r[t]["main2"] for t in ts])
    res["D0"] = d0r
    res["per_task_delta_F"] = pert
    with open(os.path.join(a.dir, f"d1_read{a.suffix}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    if a.suffix:
        print("[d1_read] smoke ok")
        return
    show = {k: v for k, v in res.items() if k not in ("table", "per_task_delta_F")}
    print(json.dumps(show, indent=1))
    for m in MODELS:
        print(m, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "per_task"})
                             for k, v in table[m].items()}, indent=None))


if __name__ == "__main__":
    main()

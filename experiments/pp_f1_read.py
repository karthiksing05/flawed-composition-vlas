"""PLAN_PARTS F1 + F2 read (docs/PARTS_RESULTS.md section 1; m3_prereg tag pp_f123).  CPU, no model.

Inputs: results/flawed/pp_f1_pi05.json, pp_f1_groot.json (pp_f1_mix.py), pp_f1_oracle.json (pp_f1_oracle.py), N2's main rollouts
(results/flawed/n2_roll/<model>_main_<tid>.json, states 4-23).
Per model, direction (tg = target into the grasp chunk, grasp frames, both layouts; op = object into the place chunk, place frames,
familiar layout) and cell u (compound x layout), component k:
  m_k(u) = sum_f d_k(f) / sum_f d_full_own(f) over u's frames   (the cross-role patch's change relative to the own-role swap's)
  oracle arms: d averaged over the arm's 3 checkpoints per frame first.
Pooled m_k = mean over cells; 95% CI = 2,000-draw percentile bootstrap (seed 0) over compounds (a compound's cells move together).
F1 (per VLA):  A floor_dir = max over A's components of the CI upper bound of A's pooled m.  F1 HOLDS iff some component k of the
               VLA (W, V, Rest, KV_l / Q_l, E_j) in some direction has CI lower bound of m_k > floor_dir.
Location:      k*_dir = argmax of the pooled point m_k over the VLA's role-message components {W, V, KV_l (pi0.5) | Q_l (GR00T)}
               (expert layers carry no role message to purify; Rest is not a role message).
F2 (per VLA):  rho_g = Spearman over the tg cells of m_{k*_tg}(u) with 1 - g(u); rho_p = Spearman over the op cells of m_{k*_op}(u)
               with 1 - p(u) (p = successes / grasps in the cell; a cell without a grasp drops out).  Bootstrap over compounds as above;
               degenerate draws are skipped and counted.  F2 TIES iff (rho_g > 0 and its CI excludes 0) or (rho_p > 0 and its CI
               excludes 0).
-> results/flawed/pp_f1_read.json

    python experiments/pp_f1_read.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
B, SEED = 2000, 0
DIRS = {"tg": "grasp", "op": "place"}


def cells_of(tasks, ph):
    return {k: v for k, v in tasks.items() if not v.get("skipped") and v["phase"] == ph}


def m_table(cells, comps):
    """-> {comp: {cell: m}}; cells: {key: dict(d=...)}."""
    out = {}
    for k in comps:
        out[k] = {}
        for u, r in cells.items():
            x = np.asarray(r["d"].get(k, []), float)
            own = np.asarray(r["d"]["full_own"], float)
            if len(x) == 0 or np.isnan(x).any():
                continue
            out[k][u] = float(x.sum() / own.sum())
    return out


def compound_of(u):
    return u.split("|")[0]


def boot_mean(vals):
    """vals {cell: m} -> (point, lo, hi) with the compound bootstrap."""
    if not vals:
        return None
    cells = sorted(vals)
    comps = sorted({compound_of(u) for u in cells})
    by = {c: [vals[u] for u in cells if compound_of(u) == c] for c in comps}
    pt = float(np.mean([vals[u] for u in cells]))
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(comps), len(comps))
        bs.append(np.mean([x for i in pick for x in by[comps[i]]]))
    return [pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def ranks(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):                       # average ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_rho(mx, fail):
    cells = sorted(set(mx) & set(fail))
    comps = sorted({compound_of(u) for u in cells})
    by = {c: [u for u in cells if compound_of(u) == c] for c in comps}
    pt = spearman([mx[u] for u in cells], [fail[u] for u in cells])
    rng = np.random.default_rng(SEED)
    bs, deg = [], 0
    for _ in range(B):
        pick = rng.integers(0, len(comps), len(comps))
        us = [u for i in pick for u in by[comps[i]]]
        r = spearman([mx[u] for u in us], [fail[u] for u in us])
        if r is None:
            deg += 1
        else:
            bs.append(r)
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else [None, None]
    return dict(rho=pt, ci=ci, n_cells=len(cells), degenerate_draws=deg,
                ties=bool(pt is not None and pt > 0 and ci[0] is not None and ci[0] > 0))


def n2_outcomes(model):
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    g, p, s = {}, {}, {}
    for t in n2["tasks"]:
        f = os.path.join(OUT, "n2_roll", f"{model}_main_{t['tid']}.json")
        eps = [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
        ss = np.array([bool(e["success"]) for e in eps], float)
        gg = np.array([bool(e["success"]) or e.get("bound") == t["object"] for e in eps], float)
        u = f"{t['compound']}|{t['layout']}"
        s[u], g[u] = float(ss.mean()), float(gg.mean())
        p[u] = float(ss.sum() / gg.sum()) if gg.sum() > 0 else None
    return s, g, p


def oracle_cells(runs, arm):
    """Average d over the arm's checkpoints per frame -> pseudo-task dict like a VLA's."""
    keys = [k for k in runs if k.startswith(arm + "|")]
    out = {}
    for u, r in runs[keys[0]].items():
        if r.get("skipped"):
            out[u] = r
            continue
        d = {c: np.mean([runs[k][u]["d"][c] for k in keys], 0).tolist() for c in r["d"]}
        out[u] = dict(r, d=d)
    return out


def profile(tasks):
    res = {}
    for dn, ph in DIRS.items():
        cells = cells_of(tasks, ph)
        comps = sorted({c for r in cells.values() for c in r["d"]} - {"full_own"})
        tab = m_table(cells, comps)
        res[dn] = dict(m={k: boot_mean(v) for k, v in tab.items()}, cells={k: v for k, v in tab.items()}, n_cells=len(cells))
    return res


def main():
    orc = json.load(open(os.path.join(OUT, "pp_f1_oracle.json"), encoding="utf-8"))
    refs = {arm: profile(oracle_cells(orc["runs"], arm)) for arm in ("A", "H", "R")}
    floor = {dn: max(v[2] for k, v in refs["A"][dn]["m"].items() if k not in ("full_cross",) and v is not None) for dn in DIRS}
    out = dict(floor_A=floor, references={a: {dn: r[dn]["m"] for dn in DIRS} for a, r in refs.items()}, models={})
    for model in ("pi05", "groot"):
        R = json.load(open(os.path.join(OUT, f"pp_f1_{model}.json"), encoding="utf-8"))
        P = profile(R["tasks"])
        comps_all = {dn: {k: v for k, v in P[dn]["m"].items() if k != "full_cross" and v is not None} for dn in DIRS}
        above = {dn: sorted(k for k, v in comps_all[dn].items() if v[1] > floor[dn]) for dn in DIRS}
        f1 = any(above[dn] for dn in DIRS)
        role = lambda k: k in ("W", "V") or k.startswith("KV_") or k.startswith("Q_")      # noqa: E731
        loc = {dn: max((k for k in comps_all[dn] if role(k)), key=lambda k: comps_all[dn][k][0]) for dn in DIRS}
        s, g, p = n2_outcomes(model)
        mx_g = {u.split("|")[0] + "|" + u.split("|")[2]: v for u, v in P["tg"]["cells"][loc["tg"]].items()}
        mx_p = {u.split("|")[0] + "|" + u.split("|")[2]: v for u, v in P["op"]["cells"][loc["op"]].items()}
        fail_g = {u: 1 - g[u] for u in mx_g}
        fail_p = {u: 1 - p[u] for u in mx_p if p.get(u) is not None}
        rg, rp = boot_rho(mx_g, fail_g), boot_rho(mx_p, fail_p)
        out["models"][model] = dict(profile={dn: P[dn]["m"] for dn in DIRS}, n_cells={dn: P[dn]["n_cells"] for dn in DIRS},
                                    above_floor=above, F1="holds" if f1 else "fails", located=loc,
                                    F2=dict(rho_g=rg, rho_p=rp, verdict="ties" if (rg["ties"] or rp["ties"]) else "does not tie"),
                                    cells_tg={u: v for u, v in mx_g.items()}, cells_op={u: v for u, v in mx_p.items()},
                                    fail_g=fail_g, fail_p=fail_p)
        print("F1", model, out["models"][model]["F1"], "located", loc, "F2", out["models"][model]["F2"]["verdict"], flush=True)
    json.dump(out, open(os.path.join(OUT, "pp_f1_read.json"), "w", encoding="utf-8"), indent=1)
    print("floor_A", json.dumps(floor))


if __name__ == "__main__":
    main()

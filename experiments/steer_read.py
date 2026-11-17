"""SR 74: the steerability failure predictor, read (CPU, openpi venv, numpy only).  Registered mapping (fixed in the prereg row):

Per probed query (experiments/steer_probe.py record): d = summed xyz delta of a chunk's first 5 steps (X-VLA: the 5th target minus the hand;
the first 8 steps as a descriptive variant), v_old = p_old - eef, v_new = p_new - eef (simulator positions of the trained and the swapped-in
entity).  steer = 1 iff angle(d, v_new) < angle(d, v_old); queries with |d| < 1e-9 are dropped (counted).  Scored for the cf chunk (the
counterfactual instruction), the true chunk (sanity), the absent-word chunk (validity control) and the nonsense-word chunk.
  g_s(o) = mean steer(cf) over grasp-phase queries whose swapped-in object is o;  p_s(t) = mean over place-phase queries whose swapped-in
  target is t, per policy, pooled over trained tasks, states and queries.   s_hat(o, t) = g_s(o) * p_s(t).  No fitting.
Cells: (policy, new task), new tasks N1-N6 = (cheese, plate) (cheese, stove) (cheese, cabinet) (wine, plate) (wine, stove) (wine, bowl) in scene 1,
S2N1 (white mug, right plate) and S2N2 (yellow-white mug, left plate) in scene 2.  Target = the canonical measured row of results/flawed/steer/
targets.json (steer_targets.py).  Policies without a target (X-VLA) are probed and excluded from T0-T2.  Wrapper-free subset (headline) =
frozen pi0.5, GR00T, pi0, SmolVLA intact.
T0   label = cell success >= .5; AUC (ties .5) and Spearman (average ranks) of s_hat against label / success; bootstrap B = 2000, numpy
     default_rng(0), tasks and policies drawn jointly with replacement, cells = the drawn (policy, task) pairs that exist; draws with one
     class (AUC) or a constant vector (Spearman) are dropped and counted; CI = 2.5 / 97.5 percentiles.  PASS iff AUC >= .75 and CI lower > .5.
     Headline: wrapper-free cells.  Also reported: all scene-1 policies; all cells of both scenes.
T1   Spearman between policy-mean s_hat and policy-mean measured success over the 6 new tasks, across scene-1 policies; bootstrap over
     policies and tasks jointly.  PASS iff rho >= .6 and CI lower > 0.
T2   leave-one-family-out (families pi05 [front ends and ladder included], groot, pi0, smolvla): kappa = sum(y s) / sum(s^2) on the other
     families' cells; prediction clip(kappa s, 0, 1); mean absolute error on the held-out family against the constant (the other families'
     mean success).  PASS iff the calibrated s_hat beats the constant in at least 3 of the 4 families.  Within-family cell-level Spearman
     (bootstrap over tasks and the family's policies) is reported alongside.
T3   scene 2: p_s of the plate-word swap on K1, K2 (pooled and per plate), s_hat(S2N1, S2N2), against measured N1, N2.  SUPPORTS the diagnosis
     iff every probed input-only policy has pooled p_s <= .25.
Baselines (same cells, same bootstrap; policy-level scores): trained-task success alone (mean success of the probe's own K1-K4, K6 episodes);
nonsense-word sensitivity = share of probed queries whose nonsense-word chunk differs in direction from the true chunk by more than 30 degrees.
Validity: absent = share of absent-word chunks nearer p_old than p_new; true = the same for the true chunk; valid iff |absent - true| <= .15.

    python experiments/steer_read.py --phase main|smoke [--out results/flawed/steer_read.json]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE]
FL = os.path.join(ROOT, "results", "flawed")
B, SEED = 2000, 0
FAMILY = dict(pi05="pi05", groot="groot", pi0="pi0", smolvla="smolvla", xvla="xvla", r0="pi05", r1="pi05", r2="pi05", r2b="pi05",
              r2w="pi05", r2l="pi05")
WF = ("pi05_I", "groot_I", "pi0_I", "smolvla_I")
INPUT_ONLY_S2 = ("pi05_I", "pi0_I", "pi05_tvp")
NEWT = {  # task -> (scene, object entity, target entity)
    "N1": (1, "cream_cheese_1", "plate_1"), "N2": (1, "cream_cheese_1", "flat_stove_1_cook_region"),
    "N3": (1, "cream_cheese_1", "wooden_cabinet_1_top_side"), "N4": (1, "wine_bottle_1", "plate_1"),
    "N5": (1, "wine_bottle_1", "flat_stove_1_cook_region"), "N6": (1, "wine_bottle_1", "akita_black_bowl_1"),
    "S2N1": (2, "porcelain_mug_1", "plate_2"), "S2N2": (2, "white_yellow_mug_1", "plate_1")}
TIDS = {1: (0, 2, 4, 6, 10), 2: (0, 1)}


def ang(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    return float(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1)))


def score(rec, kind, dk="d5"):
    d = rec[dk].get(kind)
    if d is None:
        return float("nan")
    d, e = np.asarray(d), np.asarray(rec["eef"])
    a_new, a_old = ang(d, np.asarray(rec["p_new"]) - e), ang(d, np.asarray(rec["p_old"]) - e)
    if np.isnan(a_new) or np.isnan(a_old):
        return float("nan")
    return float(a_new < a_old)


def pid_of(fname):
    b = os.path.basename(fname)
    pol, arm = b.split("_")[0], b.split("_")[1]
    if pol in FAMILY and pol.startswith("r") and pol not in ("pi05", "pi0"):
        return pol
    return pol + "_" + arm


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _auc(s, y):
    p, n = s[y], s[~y]
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def pct(v):
    v = [x for x in v if not np.isnan(x)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2


def boot_cells(cells, fn):
    """cells: dict (policy, task) -> (score, success).  Joint bootstrap over the policies and tasks present; fn(scores, successes) -> float."""
    P = sorted({p for p, _ in cells})
    T = sorted({t for _, t in cells})
    rng = np.random.default_rng(SEED)
    v, dropped = [], 0
    for _ in range(B):
        ps, ts = rng.choice(len(P), len(P)), rng.choice(len(T), len(T))
        c = [cells[(P[i], T[j])] for i in ps for j in ts if (P[i], T[j]) in cells]
        x = fn(np.array([a for a, _ in c]), np.array([b for _, b in c])) if c else float("nan")
        if np.isnan(x):
            dropped += 1
        else:
            v.append(x)
    return pct(v), dropped


def stat_block(cells):
    s = np.array([cells[k][0] for k in sorted(cells)])
    y = np.array([cells[k][1] for k in sorted(cells)])
    out = dict(n_cells=len(cells), n_policies=len({p for p, _ in cells}), n_pos=int((y >= .5).sum()))
    f_auc = lambda a, b: auc_(a, b)                                          # noqa: E731
    f_rho = lambda a, b: spearman(a, b)                                      # noqa: E731
    out["auc"] = auc_(s, y)
    out["auc_ci"], out["auc_dropped"] = boot_cells(cells, f_auc)
    out["rho"] = spearman(s, y)
    out["rho_ci"], out["rho_dropped"] = boot_cells(cells, f_rho)
    out["pass"] = bool(out["auc"] >= .75 and out["auc_ci"][0] > .5) if not np.isnan(out["auc"]) else None
    return out


def auc_(s, b):
    y = np.asarray(b) >= .5
    if y.all() or (~y).all():
        return float("nan")
    return _auc(np.asarray(s, float), y)


def t1(sc, tg, pols, tasks):
    def stat(pi, ti):
        xs = [np.mean([sc[(pols[i], tasks[j])] for j in ti]) for i in pi]
        ys = [np.mean([tg[(pols[i], tasks[j])] for j in ti]) for i in pi]
        return spearman(xs, ys)
    rho = stat(range(len(pols)), range(len(tasks)))
    rng = np.random.default_rng(SEED)
    v = [stat(rng.choice(len(pols), len(pols)), rng.choice(len(tasks), len(tasks))) for _ in range(B)]
    nd = sum(np.isnan(x) for x in v)
    return dict(n_policies=len(pols), rho=rho, ci=pct(v), dropped=int(nd), passed=bool(rho >= .6 and pct(v)[0] > 0) if not np.isnan(rho) else None)


def main():
    phase = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else "main"
    base = os.path.join(FL, "steer" if phase == "main" else "steer_smoke")
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(FL, "steer_read.json" if phase == "main" else "steer_smoke_read.json")
    assert not os.path.exists(out), "never overwrite " + out
    T = json.load(open(os.path.join(FL, "steer", "targets.json"), encoding="utf-8"))
    tgt = {(c["policy"], c["task"] if c["scene"] == 1 else "S2" + c["task"]): c["success"] for c in T["cells"]}
    R = {}                                                                  # (scene, pid) -> list of records
    ksucc = {}
    for sc in (1, 2):
        for f in sorted(glob.glob(os.path.join(base, "s%d_*" % sc, "*_steer.jsonl"))):
            pid = pid_of(f)
            recs = [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
            R.setdefault((sc, pid), []).extend(recs)
            jf = f.replace("_steer.jsonl", ".json")
            if os.path.exists(jf):
                eps = [e for v in json.load(open(jf, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
                ksucc.setdefault((sc, pid), []).extend(bool(e["success"]) for e in eps)
    res = dict(phase=phase, mapping=__doc__.split("    python")[0].strip(), policies={}, scene2={})
    GS, PS, SENS, KS, VALID = {}, {}, {}, {}, {}
    for (sc, pid), recs in sorted(R.items()):
        P = {}
        for dk in ("d5", "d8"):
            g, p, nz = {}, {}, 0
            for r in recs:
                s = score(r, "cf", dk)
                if np.isnan(s):
                    nz += 1
                    continue
                (g if r["phase"] == "grasp" else p).setdefault(r["cand"], []).append(s)
            P[dk] = dict(g={k: [float(np.mean(v)), len(v)] for k, v in g.items()}, p={k: [float(np.mean(v)), len(v)] for k, v in p.items()},
                         dropped=nz)
        val = {}
        for ph in ("grasp", "place", "all"):
            rs = [r for r in recs if ph == "all" or r["phase"] == ph]
            ab = [score(r, "absent") for r in rs]
            tr = [score(r, "true") for r in rs]
            cf = [score(r, "cf") for r in rs]
            ns = [ang(np.asarray(r["d5"]["nonsense"]), np.asarray(r["d5"]["true"])) for r in rs]
            m = lambda x: float(np.nanmean(x)) if len(x) and not np.all(np.isnan(x)) else float("nan")      # noqa: E731
            val[ph] = dict(n=len(rs), cf_steer=m(cf), absent_nearer_old=1 - m(ab), true_nearer_old=1 - m(tr), true_steer=m(tr),
                           absent_steer=m(ab), nonsense_change30=m([float(a > np.pi / 6) for a in ns if not np.isnan(a)]),
                           nonsense_change15=m([float(a > np.pi / 12) for a in ns if not np.isnan(a)]),
                           nonsense_change60=m([float(a > np.pi / 3) for a in ns if not np.isnan(a)]))
            val[ph]["valid"] = bool(abs(val[ph]["absent_nearer_old"] - val[ph]["true_nearer_old"]) <= .15)
        P["validity"] = val
        P["k_success"] = None if (sc, pid) not in ksucc else dict(mean=float(np.mean(ksucc[(sc, pid)])), n=len(ksucc[(sc, pid)]))
        P["n_queries"] = len(recs)
        res["policies"]["s%d_%s" % (sc, pid)] = P
        GS[(sc, pid)], PS[(sc, pid)] = {k: v[0] for k, v in P["d5"]["g"].items()}, {k: v[0] for k, v in P["d5"]["p"].items()}
        SENS[(sc, pid)] = val["all"]["nonsense_change30"]
        KS[(sc, pid)] = None if P["k_success"] is None else P["k_success"]["mean"]
    # ---- scene 1 cells
    def cells_for(scene, pols, variant="prod", dk="d5"):
        sc_, tg_ = {}, {}
        for pid in pols:
            for t, (s_, o, tt) in NEWT.items():
                if s_ != scene or (pid, t) not in tgt or (scene, pid) not in GS:
                    continue
                g, p = GS[(scene, pid)].get(o), PS[(scene, pid)].get(tt)
                if g is None or p is None:
                    continue
                v = {"prod": g * p, "g": g, "p": p, "mean": (g + p) / 2.0, "ktrain": KS[(scene, pid)], "nonsense": SENS[(scene, pid)]}[variant]
                if v is None or np.isnan(v):
                    continue
                sc_[(pid, t)], tg_[(pid, t)] = v, tgt[(pid, t)]
        return sc_, tg_
    pols1 = sorted(pid for (sc, pid) in GS if sc == 1 and pid in {p for p, _ in tgt})
    res["scene1_policies"] = pols1
    if phase == "main":
        for name, pols in (("wrapper_free", [p for p in pols1 if p in WF]), ("scene1_all", pols1)):
            for variant in ("prod", "g", "p", "mean", "ktrain", "nonsense"):
                sc_, tg_ = cells_for(1, pols, variant)
                cells = {k: (sc_[k], tg_[k]) for k in sc_}
                if len(cells) >= 4:
                    res.setdefault("T0", {}).setdefault(name, {})[variant] = stat_block(cells)
        # both scenes
        sc1, tg1 = cells_for(1, pols1)
        sc2, tg2 = cells_for(2, sorted(pid for (sc, pid) in GS if sc == 2))
        both = {k: (sc1[k], tg1[k]) for k in sc1}
        both.update({k: (sc2[k], tg2[k]) for k in sc2})
        if len(both) >= 4:
            res.setdefault("T0", {})["both_scenes_prod"] = stat_block(both)
        # T1
        tasks = [t for t in NEWT if NEWT[t][0] == 1]
        for name, pols in (("scene1_all", pols1), ("wrapper_free", [p for p in pols1 if p in WF])):
            for variant in ("prod", "ktrain", "nonsense"):
                sc_, tg_ = cells_for(1, pols, variant)
                full = [p for p in pols if all((p, t) in sc_ for t in tasks)]
                if len(full) >= 3:
                    res.setdefault("T1", {}).setdefault(name, {})[variant] = t1(sc_, tg_, full, tasks)
        # T2
        sc_, tg_ = cells_for(1, pols1)
        fam = {p: FAMILY[p.split("_")[0]] for p in pols1}
        kt, _ = cells_for(1, pols1, "ktrain")
        T2 = {}
        for F in ("pi05", "groot", "pi0", "smolvla"):
            tr = [k for k in sc_ if fam[k[0]] != F]
            te = [k for k in sc_ if fam[k[0]] == F]
            if not te or not tr:
                continue
            x, y = np.array([sc_[k] for k in tr]), np.array([tg_[k] for k in tr])
            kap = float((x * y).sum() / max((x * x).sum(), 1e-12))
            mae = float(np.mean([abs(tg_[k] - min(1.0, max(0.0, kap * sc_[k]))) for k in te]))
            mae_c = float(np.mean([abs(tg_[k] - y.mean()) for k in te]))
            xk = np.array([kt[k] for k in tr if k in kt])
            yk = np.array([tg_[k] for k in tr if k in kt])
            kk = float((xk * yk).sum() / max((xk * xk).sum(), 1e-12)) if len(xk) else float("nan")
            mae_k = float(np.mean([abs(tg_[k] - min(1.0, max(0.0, kk * kt[k]))) for k in te if k in kt])) if len(xk) else float("nan")
            wf = [k for k in te]
            rho = spearman([sc_[k] for k in wf], [tg_[k] for k in wf])
            sub = {k: (sc_[k], tg_[k]) for k in te}
            ci, dr = boot_cells(sub, lambda a, b: spearman(a, b)) if len({p for p, _ in sub}) >= 1 and len(sub) >= 6 else ([float("nan")] * 2, 0)
            T2[F] = dict(kappa=kap, mae_calibrated=mae, mae_constant=mae_c, mae_ktrain_calibrated=mae_k, beats_constant=bool(mae < mae_c),
                         n_cells=len(te), within_family_rho=rho, within_family_rho_ci=ci, within_dropped=dr)
        if T2:
            res["T2"] = dict(families=T2, passed=bool(sum(v["beats_constant"] for v in T2.values()) >= 3), n_families=len(T2))
        # T3
        for (sc, pid), _ in sorted(R.items()):
            if sc != 2:
                continue
            pv = PS[(2, pid)]
            ps = [x for k in R[(2, pid)] if k["phase"] == "place" for x in [score(k, "cf")] if not np.isnan(x)]
            res["scene2"][pid] = dict(p_s_pooled=float(np.mean(ps)) if ps else None, n_place=len(ps), p_s_by_plate=PS[(2, pid)], g_s=GS[(2, pid)],
                                      s_hat={t: (GS[(2, pid)].get(NEWT[t][1], np.nan) * PS[(2, pid)].get(NEWT[t][2], np.nan)) for t in ("S2N1", "S2N2")},
                                      measured={t: tgt.get((pid, t)) for t in ("S2N1", "S2N2")})
        io = [pid for pid in INPUT_ONLY_S2 if pid in res["scene2"] and res["scene2"][pid]["p_s_pooled"] is not None]
        if io:
            res["T3"] = dict(input_only_probed=io, supports_diagnosis=bool(all(res["scene2"][p]["p_s_pooled"] <= .25 for p in io)))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    show(res, phase)
    print("wrote", out, flush=True)


def show(res, phase):
    f = lambda x: "  NA " if x is None or (isinstance(x, float) and np.isnan(x)) else "%.3f" % x          # noqa: E731
    print("== per policy (d5): queries, K success, mean g_s over words, mean p_s over words, validity (all phases)")
    for k, P in res["policies"].items():
        g = [v[0] for v in P["d5"]["g"].values()]
        p = [v[0] for v in P["d5"]["p"].values()]
        V = P["validity"]["all"]
        print("%-14s q %4d K %s | g %s p %s | cf %s true_steer %s absent_nearer_old %s true_nearer_old %s valid %s | ns30 %s" % (
            k, P["n_queries"], f(None if P["k_success"] is None else P["k_success"]["mean"]), f(np.mean(g) if g else None),
            f(np.mean(p) if p else None), f(V["cf_steer"]), f(V["true_steer"]), f(V["absent_nearer_old"]), f(V["true_nearer_old"]), V["valid"],
            f(V["nonsense_change30"])))
        print("               g_s %s | p_s %s" % ({a[:8]: (round(b[0], 2), b[1]) for a, b in P["d5"]["g"].items()},
                                                    {a[:8]: (round(b[0], 2), b[1]) for a, b in P["d5"]["p"].items()}))
    for key in ("T0", "T1", "T2", "T3", "scene2"):
        if key in res:
            print("==", key)
            print(json.dumps(res[key], indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()

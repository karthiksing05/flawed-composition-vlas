"""FLAWED_COMPOSITION prototype A, stage 1b read (section 5.6; m3_prereg tag pa1b).  CPU.

Data: results/flawed/a1b_roll/<arm>_s<seed>_main_<tid>.json (fa1b_roll.py) on N2's 16 FAMILIAR-layout tasks, main start states 4-23.
Main arms Hb, Rb, Ab with seeds 0, 1, 2; ablations Ab_cmd, Ab_nooff, Ab_nosum with seed 0.
Per episode s, g, r as stage 1 (fa1_read.py: s = LIBERO success; g = s or the bind track's first object moved with the gripper closed
is the task's object; r = s or the relaxed placement test).

PRIMARY class: the pick-and-place pairings, known K1-K6 and new N1-N6.  DESCRIPTIVE class (declared before the read; not object +
target form): K7 / N7 (the drawer is opened before the place), K8 / N8 (a push, no grasp).
Per arm, pooled over its seeds: s(status) = mean over the status's compounds of the task success rate (seeds pooled per task);
g likewise; p = s given g pooled over episodes.
CIs: 2,000-draw percentile bootstrap (seed 0), resampling the 6 known and the 6 new compounds within status and the 20 start states,
jointly for every arm and seed (a state draw is applied to every seed; seeds are not resampled).

Pre-registered criteria (pa1b), all on the primary class with oracle state, A = Ab, H = Hb, R = Rb:
  floor  s_H(known) < 0.5 -> "uninformative"
  C1     parity: s_A(known) - s_H(known) >= -0.10 (point estimate, mean over seeds; CI reported)
  C2     compositional ratio: s_A(new) / s_A(known) >= 0.8 (point) and its CI's lower bound >= 0.6
  C3     multiplicativity: for every new pairing j = (o, t), D_j = s_A(j) - g_A(o) * p_A(t) has a 95% CI containing 0; g_A(o) = the
         episode-pooled g over the known pick-and-place tasks with object o; p_A(t) = s given g pooled over the known pick-and-place
         tasks with target t (for C3 only the start states are resampled: the compounds are the definition)
  C4     baselines: s_A(new) - s_H(new) > 0 and s_A(new) - s_R(new) > 0, both CIs excluding 0
  verdict, the first that applies: floor -> "uninformative"; C1 and C2 and C3 and C4 -> "compositional"; otherwise
         "not compositional (fails: <the failing criteria in the order C1, C2, C3, C4>)"
Descriptive: per-seed values; strict / relaxed p; ablation contrasts (Ab - Y on s(known), s(new), p_rel(new)); the descriptive class
per compound; the habit table; multiplicativity for H and R; the closed-loop grasp-identity test (for each arm, seed, object and
start state, whether the first held step and the end-effector position there are identical across every familiar task with that
object: the object phase cannot see the target); stage 1's A on the same cells (results/flawed/a1_roll, seed 0).
-> results/flawed/a1b_read.json

    python experiments/fa1b_read.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_read as R1                                                        # noqa: E402  (at, RAD, TRAINED, CLASSES)

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "flawed")
MAIN, SEEDS = ("Hb", "Rb", "Ab"), (0, 1, 2)
ABL = ("Ab_cmd", "Ab_nooff", "Ab_nosum")
RUNS = [(m, s) for m in MAIN for s in SEEDS] + [(m, 0) for m in ABL]
PRIMARY = [f"K{i}" for i in range(1, 7)] + [f"N{i}" for i in range(1, 7)]
DESCR = ["K7", "K8", "N7", "N8"]
B, SEED = 2000, 0


def load():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    fam = [t for t in n2["tasks"] if t["layout"] == "familiar"]
    comps = sorted({t["compound"] for t in fam}, key=lambda c: (c[0] != "K", int(c[1:])))
    ks = n2["states"]["main"]
    info = {t["compound"]: t for t in fam}
    X, EP, missing, err = {}, {}, [], 0
    runs = RUNS + [("A_stage1", 0)]
    for m, sd in runs:
        X[(m, sd)] = {x: np.full((len(comps), len(ks)), np.nan) for x in ("S", "G", "R", "H")}
        for t in fam:
            ci = comps.index(t["compound"])
            f = (os.path.join(OUT, "a1_roll", f"A_main_{t['tid']}.json") if m == "A_stage1" else
                 os.path.join(OUT, "a1b_roll", f"{m}_s{sd}_main_{t['tid']}.json"))
            if not os.path.exists(f):
                missing.append(os.path.basename(f))
                continue
            o, tg = t["object"], t["target"]
            others = [x for x in R1.RAD if x not in (o, tg) and x not in R1.TRAINED.get(o, [])]
            for e in [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
                j = ks.index(int(e["trial"]))
                s = bool(e["success"])
                g = s or e.get("bound") == o
                r = s or (tg in R1.RAD and R1.at(e["final"], o, tg, cal))
                X[(m, sd)]["S"][ci, j], X[(m, sd)]["G"][ci, j], X[(m, sd)]["R"][ci, j] = s, g, r
                if t["status"] == "new" and not s:
                    cls = ("instructed" if r else
                           "habit" if any(R1.at(e["final"], o, x, cal) for x in R1.TRAINED.get(o, [])) else
                           "other" if any(R1.at(e["final"], o, x, cal) for x in others) else "elsewhere")
                    X[(m, sd)]["H"][ci, j] = R1.CLASSES.index(cls) + (0 if g else 10)
                EP[(m, sd, t["compound"], j)] = e
                err += e.get("error") is not None
    return comps, ks, info, X, EP, missing, err


def pool(X, arm, seeds):
    return {k: np.stack([X[(arm, s)][k] for s in seeds]) for k in ("S", "G", "R")}          # [n_seeds, 16, n_states]


def ratio(num, den):
    d = den.sum()
    return float(num.sum() / d) if d > 0 else float("nan")


def arm_stats(P, comps, kn, nw, kk, info):
    S, G, R = (P[k][:, :, kk] for k in ("S", "G", "R"))
    r = {}
    for st, idx in (("known", kn), ("new", nw)):
        r[f"s_{st}"] = float(S[:, idx].mean())
        r[f"g_{st}"] = float(G[:, idx].mean())
        r[f"p_{st}"] = ratio(S[:, idx] * G[:, idx], G[:, idx])
        r[f"prel_{st}"] = ratio(R[:, idx] * G[:, idx], G[:, idx])
    r["ratio"] = r["s_new"] / r["s_known"] if r["s_known"] > 0 else float("nan")
    return r


def mult(P, comps, kk, info):
    """Per new primary pairing: observed s, predicted g(o) * p(t) from the known primary tasks."""
    S, G = P["S"][:, :, kk], P["G"][:, :, kk]
    known = [c for c in PRIMARY if c[0] == "K"]
    out = {}
    for c in PRIMARY:
        if c[0] != "N":
            continue
        o, t = info[c]["object"], info[c]["target"]
        ko = [comps.index(k) for k in known if info[k]["object"] == o]
        kt = [comps.index(k) for k in known if info[k]["target"] == t]
        g_o = float(G[:, ko].mean())
        p_t = ratio(S[:, kt] * G[:, kt], G[:, kt])
        obs = float(S[:, comps.index(c)].mean())
        out[c] = dict(obs=obs, g_o=g_o, p_t=p_t, pred=g_o * p_t, D=obs - g_o * p_t, g_obs=float(G[:, comps.index(c)].mean()),
                      p_obs=ratio(S[:, comps.index(c)] * G[:, comps.index(c)], G[:, comps.index(c)]))
    return out


def evaluate(X, comps, info, kn, nw, kk):
    P = {m: pool(X, m, SEEDS) for m in MAIN}
    P.update({m: pool(X, m, (0,)) for m in ABL + ("A_stage1",)})
    for m in MAIN:
        for s in SEEDS:
            P[f"{m}_s{s}"] = pool(X, m, (s,))
    st = {m: arm_stats(P[m], comps, kn, nw, kk, info) for m in P}
    c = {"C1": st["Ab"]["s_known"] - st["Hb"]["s_known"], "C2": st["Ab"]["ratio"],
         "C4_vs_H": st["Ab"]["s_new"] - st["Hb"]["s_new"], "C4_vs_R": st["Ab"]["s_new"] - st["Rb"]["s_new"]}
    for y in ABL + ("A_stage1",):
        for k in ("s_known", "s_new", "prel_new", "g_new"):
            c[f"Ab_minus_{y}_{k}"] = st["Ab"][k] - st[y][k]
    for m in ("Ab", "Hb", "Rb"):
        for j, d in mult(P[m], comps, kk, info).items():
            c[f"D_{m}_{j}"] = d["D"]
    return st, c


def boot(X, comps, info):
    idx = np.array([comps.index(c) for c in PRIMARY])
    kn, nw = idx[:6], idx[6:]
    n = X[RUNS[0]]["S"].shape[-1]
    pst, pc = evaluate(X, comps, info, kn, nw, np.arange(n))
    rng = np.random.default_rng(SEED)
    ds, dc = [], []
    for _ in range(B):
        a, b, kk = rng.choice(kn, len(kn)), rng.choice(nw, len(nw)), rng.choice(n, n)
        st, c = evaluate(X, comps, info, a, b, kk)
        c3 = {k: v for k, v in evaluate(X, comps, info, kn, nw, kk)[1].items() if k.startswith("D_")}   # C3: states only
        c.update(c3)
        ds.append(st)
        dc.append(c)

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2

    cells = {m: {k: dict(est=pst[m][k], ci=ci([d[m][k] for d in ds])) for k in pst[m]} for m in pst}
    con = {k: dict(est=pc[k], ci=ci([d[k] for d in dc])) for k in pc}
    P = {m: pool(X, m, SEEDS) for m in MAIN}
    P["A_stage1"] = pool(X, "A_stage1", (0,))
    mt = {m: mult(P[m], comps, np.arange(n), info) for m in P}
    return cells, con, mt


def verdict(cells, con):
    v = dict(floor=cells["Hb"]["s_known"]["est"] < 0.5, C1=con["C1"]["est"] >= -0.10,
             C2=con["C2"]["est"] >= 0.8 and con["C2"]["ci"][0] >= 0.6,
             C3=all(con[k]["ci"][0] <= 0 <= con[k]["ci"][1] for k in con if k.startswith("D_Ab_")),
             C4=all(con[k]["est"] > 0 and con[k]["ci"][0] > 0 for k in ("C4_vs_H", "C4_vs_R")))
    if v["floor"]:
        v["verdict"] = "uninformative"
    elif all(v[k] for k in ("C1", "C2", "C3", "C4")):
        v["verdict"] = "compositional"
    else:
        v["verdict"] = "not compositional (fails: " + ", ".join(k for k in ("C1", "C2", "C3", "C4") if not v[k]) + ")"
    return v


def grasp_identity(EP, info, ks):
    """Per run: share of (object, state) groups whose first held step and end-effector position there agree across all familiar
    tasks with that object (groups where no task's episode reaches a held step are counted separately)."""
    out = {}
    objs = sorted({info[c]["object"] for c in info})
    for m, sd in RUNS:
        same, diff, none = 0, 0, 0
        for o in objs:
            cs = [c for c in info if info[c]["object"] == o]
            if len(cs) < 2:
                continue
            for j in range(len(ks)):
                hs = [(EP[(m, sd, c, j)]["held_step"], tuple(np.round(EP[(m, sd, c, j)]["held_ee"], 6)) if EP[(m, sd, c, j)]["held_ee"] else None)
                      for c in cs if (m, sd, c, j) in EP]
                if all(h[0] is None for h in hs):
                    none += 1
                elif len(set(hs)) == 1:
                    same += 1
                else:
                    diff += 1
        out[f"{m}_s{sd}"] = dict(identical=same, differ=diff, never_held=none,
                                 share_identical=same / (same + diff) if same + diff else float("nan"))
    return out


def main():
    comps, ks, info, X, EP, missing, err = load()
    assert not missing, f"missing roll files: {missing[:5]} ... ({len(missing)})"
    for k, v in X.items():
        for x in ("S", "G", "R"):
            assert np.isfinite(v[x]).all(), (k, x)
    cells, con, mt = boot(X, comps, info)
    v = verdict(cells, con)
    per = {c: {f"{m}_s{sd}": dict(s=float(X[(m, sd)]["S"][i].mean()), g=float(X[(m, sd)]["G"][i].mean()),
                                  r=float(X[(m, sd)]["R"][i].mean())) for m, sd in RUNS + [("A_stage1", 0)]}
           for i, c in enumerate(comps)}
    habit = {}
    for m, sd in RUNS:
        h = X[(m, sd)]["H"][[comps.index(c) for c in PRIMARY if c[0] == "N"]]
        n_gf = int(np.isin(h, [0, 1, 2, 3]).sum())
        habit[f"{m}_s{sd}"] = dict(n_g_fail=n_gf, **{cl: (float((h == i).sum() / n_gf) if n_gf else float("nan"))
                                                      for i, cl in enumerate(R1.CLASSES)})
    res = dict(phase="main", layout="familiar", n_states=len(ks), compounds=comps, primary=PRIMARY, descriptive=DESCR, verdict=v,
               cells=cells, contrasts=con, multiplicativity=mt, per_compound=per, habit=habit,
               grasp_identity=grasp_identity(EP, info, ks), errors=err)
    json.dump(res, open(os.path.join(OUT, "a1b_read.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(verdict=v, contrasts=con), indent=1))


if __name__ == "__main__":
    main()

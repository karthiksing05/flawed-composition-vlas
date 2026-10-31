"""FLAWED_COMPOSITION prototype A, the read engine for stage 1c onward (sections 5.7+): stage 1b's estimators, criteria C1-C4 and
verdict order (experiments/fa1b_read.py, section 5.6) with the factored arm A, its ablations and reference arms as parameters.  CPU.

Rollout files: results/flawed/<dir>/<arm>_s<seed>_main_<tid>.json, the directory looked up per arm (ROLLDIR; stage 1b's Hb / Rb runs
are read from a1b_roll).  PRIMARY class K1-K6 / N1-N6, descriptive class K7 / K8 / N7 / N8, familiar layout, main states 4-23.
  floor  s_H(known) < 0.5 -> "uninformative"
  C1     s_A(known) - s_H(known) >= -0.10 (point)
  C2     s_A(new) / s_A(known) >= 0.8 (point) and CI lower bound >= 0.6
  C3     every new pairing's CI of s_A(o, t) - g_A(o) p_A(t) contains 0 (start states resampled only)
  C4     s_A(new) - s_H(new) > 0 and s_A(new) - s_R(new) > 0, both CIs excluding 0
  verdict: floor -> "uninformative"; C1-C4 -> "compositional"; else "not compositional (fails: ...)"
CIs: 2,000-draw percentile bootstrap (seed 0) over the 6 + 6 compounds within status and the 20 states, joint over arms and seeds.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_read as R1                                                        # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "flawed")
PRIMARY = [f"K{i}" for i in range(1, 7)] + [f"N{i}" for i in range(1, 7)]
DESCR = ["K7", "K8", "N7", "N8"]
B, SEED, SEEDS = 2000, 0, (0, 1, 2)
ROLLDIR = {"Hb": "a1b_roll", "Rb": "a1b_roll", "Ab": "a1b_roll", "Ab_cmd": "a1b_roll", "Ab_nooff": "a1b_roll", "Ab_nosum": "a1b_roll",
           "Ac": "a1c_roll", "Ac_nobear": "a1c_roll", "Ac_noft": "a1c_roll"}


class Read:
    def __init__(self, A, abl, refs, H="Hb", R="Rb", layout="familiar", a_seeds=SEEDS):
        """A: the factored arm (3 seeds); abl: its seed-0 ablations; refs: earlier arms shown beside it, as (name, seeds).
        H, R: the holistic and role-routed baselines (stage 1g on: retrained on the same data as A); layout: N2's familiar or new
        cells; a_seeds: A's seeds (stage 1j's confirmatory read uses fresh seeds 3-5).  The defaults are stages 1b-1f's read."""
        self.A, self.abl, self.H, self.R, self.layout = A, tuple(abl), H, R, layout
        self.main = (H, R, A)
        self.seeds = {H: SEEDS, R: SEEDS, A: tuple(a_seeds)}
        self.runs = [(m, s) for m in self.main for s in self.seeds[m]] + [(m, 0) for m in self.abl]
        self.refs = list(refs)
        self.all_runs = self.runs + [(m, s) for m, ss in self.refs for s in ss if (m, s) not in self.runs]

    def path(self, m, sd, tid):
        if m == "A_stage1":
            return os.path.join(OUT, "a1_roll", f"A_main_{tid}.json")
        return os.path.join(OUT, ROLLDIR[m], f"{m}_s{sd}_main_{tid}.json")

    def load(self):
        n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
        cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
        fam = [t for t in n2["tasks"] if t["layout"] == self.layout]
        comps = sorted({t["compound"] for t in fam}, key=lambda c: (c[0] != "K", int(c[1:])))
        ks = n2["states"]["main"]
        info = {t["compound"]: t for t in fam}
        X, EP, missing, err = {}, {}, [], 0
        for m, sd in self.all_runs:
            X[(m, sd)] = {x: np.full((len(comps), len(ks)), np.nan) for x in ("S", "G", "R", "H")}
            for t in fam:
                ci = comps.index(t["compound"])
                f = self.path(m, sd, t["tid"])
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
    return {k: np.stack([X[(arm, s)][k] for s in seeds]) for k in ("S", "G", "R")}


def ratio(num, den):
    d = den.sum()
    return float(num.sum() / d) if d > 0 else float("nan")


def arm_stats(P, kn, nw, kk):
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
        i = comps.index(c)
        obs = float(S[:, i].mean())
        out[c] = dict(obs=obs, g_o=g_o, p_t=p_t, pred=g_o * p_t, D=obs - g_o * p_t, g_obs=float(G[:, i].mean()),
                      p_obs=ratio(S[:, i] * G[:, i], G[:, i]))
    return out


def run(rd, out_name):
    comps, ks, info, X, EP, missing, err = rd.load()
    assert not missing, f"missing roll files: {missing[:5]} ... ({len(missing)})"
    for k, v in X.items():
        for x in ("S", "G", "R"):
            assert np.isfinite(v[x]).all(), (k, x)
    A, HB, RB = rd.A, rd.H, rd.R
    groups = {m: rd.seeds[m] for m in rd.main}
    groups.update({m: (0,) for m in rd.abl})
    groups.update({m: tuple(ss) for m, ss in rd.refs})
    for m in rd.main:
        for s in rd.seeds[m]:
            groups[f"{m}_s{s}"] = (s,)

    Ps = {}
    for name in groups:
        if name in rd.main or name in rd.abl or name in dict(rd.refs):
            Ps[name] = pool(X, name, groups[name])
        else:
            Ps[name] = pool(X, name[:-3], groups[name])

    def evaluate(kn, nw, kk):
        st = {m: arm_stats(Ps[m], kn, nw, kk) for m in Ps}
        c = {"C1": st[A]["s_known"] - st[HB]["s_known"], "C2": st[A]["ratio"],
             "C4_vs_H": st[A]["s_new"] - st[HB]["s_new"], "C4_vs_R": st[A]["s_new"] - st[RB]["s_new"]}
        for y in list(rd.abl) + [m for m, _ in rd.refs]:
            for k in ("s_known", "s_new", "prel_new", "g_new", "p_new"):
                c[f"{A}_minus_{y}_{k}"] = st[A][k] - st[y][k]
        return st, c

    def dvals(kk):
        return {f"D_{m}_{j}": d["D"] for m in (A, HB, RB) for j, d in mult(Ps[m], comps, kk, info).items()}

    idx = np.array([comps.index(c) for c in PRIMARY])
    kn, nw = idx[:6], idx[6:]
    n = len(ks)
    pst, pc = evaluate(kn, nw, np.arange(n))
    pc.update(dvals(np.arange(n)))
    rng = np.random.default_rng(SEED)
    ds, dc = [], []
    for _ in range(B):
        a, b, kk = rng.choice(kn, len(kn)), rng.choice(nw, len(nw)), rng.choice(n, n)
        st, c = evaluate(a, b, kk)
        c.update(dvals(kk))
        ds.append(st)
        dc.append(c)

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2

    cells = {m: {k: dict(est=pst[m][k], ci=ci([d[m][k] for d in ds])) for k in pst[m]} for m in pst}
    con = {k: dict(est=pc[k], ci=ci([d[k] for d in dc])) for k in pc}
    v = dict(floor=cells[HB]["s_known"]["est"] < 0.5, C1=con["C1"]["est"] >= -0.10,
             C2=con["C2"]["est"] >= 0.8 and con["C2"]["ci"][0] >= 0.6,
             C3=all(con[k]["ci"][0] <= 0 <= con[k]["ci"][1] for k in con if k.startswith(f"D_{A}_")),
             C4=all(con[k]["est"] > 0 and con[k]["ci"][0] > 0 for k in ("C4_vs_H", "C4_vs_R")))
    if v["floor"]:
        v["verdict"] = "uninformative"
    elif all(v[k] for k in ("C1", "C2", "C3", "C4")):
        v["verdict"] = "compositional"
    else:
        v["verdict"] = "not compositional (fails: " + ", ".join(k for k in ("C1", "C2", "C3", "C4") if not v[k]) + ")"
    mt = {m: mult(Ps[m], comps, np.arange(n), info) for m in [A, HB, RB] + [x for x, _ in rd.refs]}
    per = {c: {f"{m}_s{sd}": dict(s=float(X[(m, sd)]["S"][i].mean()), g=float(X[(m, sd)]["G"][i].mean()),
                                  r=float(X[(m, sd)]["R"][i].mean())) for m, sd in rd.all_runs} for i, c in enumerate(comps)}
    habit = {}
    for m, sd in rd.runs:
        h = X[(m, sd)]["H"][[comps.index(c) for c in PRIMARY if c[0] == "N"]]
        n_gf = int(np.isin(h, [0, 1, 2, 3]).sum())
        habit[f"{m}_s{sd}"] = dict(n_g_fail=n_gf, **{cl: (float((h == i).sum() / n_gf) if n_gf else float("nan"))
                                                      for i, cl in enumerate(R1.CLASSES)})
    gid = {}
    objs = sorted({info[c]["object"] for c in info})
    for m, sd in rd.runs:
        same, diff, none = 0, 0, 0
        for o in objs:
            cs = [c for c in info if info[c]["object"] == o]
            if len(cs) < 2:
                continue
            for j in range(n):
                hs = [(EP[(m, sd, c, j)]["held_step"], tuple(np.round(EP[(m, sd, c, j)]["held_ee"], 6)) if EP[(m, sd, c, j)]["held_ee"] else None)
                      for c in cs if (m, sd, c, j) in EP]
                if all(h[0] is None for h in hs):
                    none += 1
                elif len(set(hs)) == 1:
                    same += 1
                else:
                    diff += 1
        gid[f"{m}_s{sd}"] = dict(identical=same, differ=diff, never_held=none, share_identical=same / (same + diff) if same + diff else float("nan"))
    res = dict(A=A, phase="main", layout=rd.layout, n_states=n, compounds=comps, primary=PRIMARY, descriptive=DESCR, verdict=v,
               cells=cells, contrasts=con, multiplicativity=mt, per_compound=per, habit=habit, grasp_identity=gid, errors=err)
    json.dump(res, open(os.path.join(OUT, out_name), "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(verdict=v), indent=1))
    return res

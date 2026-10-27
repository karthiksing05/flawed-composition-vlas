"""PLAN_BINDER B1.3 read (docs/PLAN_BINDER.md section 13; pre-registration experiments/prereg/pb1_3.py, ledger m3_prereg tag pb1_3).
CPU, chunks env.  Simulator labels are used here for scoring only.

PRIMARY (generated test units, results/bind/b1_3/units.json "test"; clusters = test COMPOUNDS, 3 layouts each):
  binding error (1 - S1's hit) vs the compound's CD bin (x = 0.1 / 0.3 / 0.5, the realised bin) for the FACTORISED binder
  (B1.3, the primary arm) and the HOLISTIC binder (B1b retrained on the generated pool).  OLS slope of unit error on x, with a
  2000-draw compound-cluster bootstrap (compounds resampled within each depth x bin cell, the same draw for both methods).
  INTERACTION = slope(holistic) - slope(factorised).  T1/T2 prediction CONFIRMED iff the interaction's 95% CI lies above 0;
  reported with it: each slope's CI (T2 predicts the factorised CI includes 0; T1 predicts the holistic CI lies above 0), the
  same per depth, and with x = the unit's own CD (D_C(train || unit), alpha 0.1).
NEW-SET BARS (per arm, point estimates, two-level compound bootstrap CIs): (i) depth-1 hit >= 0.70; (ii) depth-2 hit >= 0.70;
  (iii) depth-0 hit >= GroundingDINO's depth-0 hit on the same units.  B1.3 PASSES the new-set bars iff the primary arm passes all three.
ALSO: CD x depth hit tables for every arm (B1.3 siglip / dino, B1b-gen, S1c as trained in S1, GroundingDINO, Qwen2.5-VL, Molmo;
  the VLMs under LIBERO's mirrored convention (the labels) and the literal image-side convention); per role; drawer and caddy units;
  per-word channel assignment (prequential DL); left / right geometry prototypes (the mirrored check); T2 budget per depth and bin;
  atom invariance (permutation test); parameter count.
SECONDARY (S1's 240 dev units, the 11 recombination tasks for the CD slope): the same three bars as B1 / B1.1 / B1.2.

    python experiments/b1_3_read.py [--record]  -> results/bind/b1_3.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
B13 = os.path.join(ROOT, "results", "bind", "b1_3")
S1 = os.path.join(ROOT, "results", "bind", "s1")
ARMS = ("siglip", "dino")
N_BOOT = 2000
BINX = {"0.1": 0.1, "0.3": 0.3, "0.5": 0.5}


def words(p):
    return re.findall(r"[a-z]+", p.lower())


def literal_target(u, meta, names):
    """the image-side (literal) referent of a lateral unit: LIBERO's "left" (smallest world y) is the IMAGE RIGHT, so the literal
    "left" is the largest-y instance (and vice versa); compartments swap left <-> right.  Others unchanged."""
    q, tg = u["tuple"][3], u["targets"][0]
    if q not in ("left", "right"):
        return tg
    if u["tuple"][4] == "compartment":
        o = tg.replace("_left_contain_region", "_TMP").replace("_right_contain_region", "_left_contain_region").replace("_TMP", "_right_contain_region")
        return o if o in names else tg
    if u["tuple"][4]:
        return tg
    t = u["tuple"][2]
    inst = [(o["name"], meta["pos"][o["name"]][1]) for o in meta["objects"] if o["type"] == t and o["name"] in meta["pos"]]
    if len(inst) < 2:
        return tg
    ys = sorted(inst, key=lambda x: x[1])
    return ys[-1][0] if q == "left" else ys[0][0]


def main():
    import d4_read as DR
    import s1_read as SR
    import b1_3_gen as G
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    U = json.load(open(os.path.join(B13, "units.json"), encoding="utf-8"))
    prereg = json.load(open(os.path.join(B13, "prereg_frozen.json"), encoding="utf-8"))
    PRIMARY = prereg["primary"]
    test = U["test"]
    P, SEL = {}, {}
    for arm in ARMS:
        p = os.path.join(B13, arm, "pred_test.pkl")
        if os.path.exists(p):
            P[arm] = pickle.load(open(p, "rb"))["pred"]
            SEL[arm] = json.load(open(os.path.join(B13, arm, "select.json"), encoding="utf-8"))
    assert PRIMARY in P
    ext = {}
    z = np.load(os.path.join(B13, "b1b", "pred_test.npz"), allow_pickle=True)
    ext["b1b_gen"] = {str(k): int(np.argmax(l)) for k, l in zip(z["keys"], z["logits"])}
    for m, f in (("s1c", "s1c/pred_test.npz"), ("gdino", "b1a/pred_test.npz"), ("qwen", "vlm/qwen.npz"), ("molmo", "vlm/molmo.npz")):
        fp = os.path.join(B13, f)
        if os.path.exists(fp):
            z = np.load(fp, allow_pickle=True)
            ext[m] = {str(k): int(p) for k, p in zip(z["keys"], z["peak"])}
    methods = list(P) + list(ext)
    cache = {}

    def frame(f):
        if f not in cache:
            z = np.load(os.path.join(ROOT, f), allow_pickle=True)
            names, cov = SR.masks(z)
            cache[f] = (names, cov, [str(b) for b in z["body_names"]], json.loads(str(z["meta"])))
        return cache[f]
    units = []
    for ix, e in enumerate(test):
        names, cov, bodies, meta = frame(e["frame"])
        tg = e["targets"][0]
        lit = literal_target(e, meta, names)
        key = f"{e['role']}|{e['frame']}"
        u = dict(ix=ix, compound=e["compound"], tid=e["compound"], role=e["role"], phrase=e["phrase"], depth=e["depth"], bin=e["bin"],
                 x=BINX[e["bin"]], cd_unit=e["cd_unit"], u=e["u"], tuple=e["tuple"], target=tg, frame=e["frame"], lateral=lit != tg)
        for conv, t_ in (("", tg), ("_lit", lit)):
            t = cov[names.index(t_)]
            comp, _ = SR.competitors(t_, names, bodies)
            cc = [cov[names.index(c)] for c in comp if c != t_]
            for m in methods:
                pk = P[m][ix]["peak"] if m in P else ext[m].get(key, -1)
                u[m + conv] = SR.hit(pk, t, cc)
        # atoms of the factorised arms (scoring only)
        t = cov[names.index(tg)]
        comp, _ = SR.competitors(tg, names, bodies)
        cc = [cov[names.index(c)] for c in comp if c != tg]
        if tg in bodies:
            ref_body = [tg]
            type_set = [b for b in bodies if SR.itype(b) == SR.itype(tg)]
        else:
            ref_body = [b for b in bodies if tg.startswith(b + "_")]
            type_set = ref_body
        refm = np.max([cov[names.index(b)] for b in ref_body], 0) if ref_body else np.zeros(256)
        typem = np.max([cov[names.index(b)] for b in type_set], 0) if type_set else np.zeros(256)
        for arm, pr in P.items():
            p = pr[ix]
            hits_c = np.array([SR.hit(int(k), t, cc) for k in p["cand_peak"]])
            on_type = np.array([typem[int(k)] > 0 for k in p["cand_peak"]])
            at = {"@proposal": float(not hits_c.any())}
            if p["parent"] >= 0:
                at["@parent"] = float(refm[int(p["cand_peak"][p["parent"]])] <= 0)
            for w, s in p["per"].items():
                if w == "@role":
                    continue
                ch = SEL[arm]["channels"].get(w)
                if ch == "app":
                    at[w] = float(not on_type[int(np.argmax(s))])
                elif ch == "geo":
                    pool = np.where(on_type)[0]
                    at[w] = float(not (len(pool) and hits_c[pool[int(np.argmax(s[pool]))]]))
                elif ch == "both":
                    at[w] = float(not hits_c[int(np.argmax(s))])
            u[arm + "__atoms"] = at
        units.append(u)

    def per(us, k):
        d = {}
        for x in us:
            if x.get(k) is not None:
                d.setdefault(x["tid"], []).append(float(x[k]))
        return DR.rate(d)
    cols = methods + [m + "_lit" for m in ("qwen", "molmo", "gdino") if m in ext]
    cells = {}
    for d in (0, 1, 2, None):
        for b in list(G.BINS) + [None]:
            for role in (None, "object", "goal"):
                us = [x for x in units if (d is None or x["depth"] == d) and (b is None or x["bin"] == b) and (role is None or x["role"] == role)]
                if us:
                    cells[f"d{'all' if d is None else d}_b{'all' if b is None else b}" + (f"_{role}" if role else "")] = dict(
                        n=len(us), compounds=len({x["tid"] for x in us}), hit={m: per(us, m) for m in cols})
    for nm, fn in (("drawer_units", lambda x: x["tuple"][4] == "drawer"), ("caddy_units", lambda x: x["tuple"][4] == "compartment"),
                   ("lateral_units", lambda x: x["lateral"])):
        us = [x for x in units if fn(x)]
        cells[nm] = dict(n=len(us), compounds=len({x["tid"] for x in us}), hit={m: per(us, m) for m in cols})
    # ── new-set bars ──
    gd0 = cells["d0_ball"]["hit"]["gdino"]["mean"]
    bars = {}
    for m in methods:
        i_, ii, iii = cells["d1_ball"]["hit"][m], cells["d2_ball"]["hit"][m], cells["d0_ball"]["hit"][m]
        b = dict(i_depth1=i_, i_pass=bool(i_["mean"] >= 0.70), ii_depth2=ii, ii_pass=bool(ii["mean"] >= 0.70), iii_depth0=iii,
                 iii_groundingdino=gd0, iii_pass=bool(iii["mean"] >= gd0))
        b["pass"] = b["i_pass"] and b["ii_pass"] and b["iii_pass"]
        bars[m] = b
    # ── PRIMARY: error vs CD bin, factorised vs holistic, compound-cluster bootstrap ──
    rng = np.random.default_rng(0)
    comp_of = collections.defaultdict(list)
    for i, x in enumerate(units):
        comp_of[x["tid"]].append(i)
    strata = collections.defaultdict(list)
    for c, ii in comp_of.items():
        x = units[ii[0]]
        strata[(x["depth"], x["bin"])].append(c)

    def slope_of(idx, m, xk="x", depth=None):
        sel = [i for i in idx if depth is None or units[i]["depth"] == depth]
        xs = np.array([units[i][xk] for i in sel])
        ys = np.array([1.0 - units[i][m] for i in sel])
        return float(np.polyfit(xs, ys, 1)[0]) if len(sel) > 2 and np.ptp(xs) > 0 else float("nan")
    draws = []
    for _ in range(N_BOOT):
        idx = []
        for s, cs in strata.items():
            for c in rng.choice(cs, len(cs), replace=True):
                idx.extend(comp_of[c])
        draws.append(idx)
    allidx = list(range(len(units)))
    FACT, HOL = PRIMARY, "b1b_gen"

    def test_slopes(xk="x", depth=None):
        out = {}
        for m in methods:
            bs = [slope_of(ix, m, xk, depth) for ix in draws]
            out[m] = dict(slope=slope_of(allidx, m, xk, depth), lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)))
        inter = [slope_of(ix, HOL, xk, depth) - slope_of(ix, FACT, xk, depth) for ix in draws]
        out["interaction_holistic_minus_factorised"] = dict(value=slope_of(allidx, HOL, xk, depth) - slope_of(allidx, FACT, xk, depth),
                                                           lo=float(np.nanpercentile(inter, 2.5)), hi=float(np.nanpercentile(inter, 97.5)))
        if "dino" in P and PRIMARY != "dino":
            i2 = [slope_of(ix, HOL, xk, depth) - slope_of(ix, "dino", xk, depth) for ix in draws]
            out["interaction_holistic_minus_dino"] = dict(lo=float(np.nanpercentile(i2, 2.5)), hi=float(np.nanpercentile(i2, 97.5)))
        return out
    primary = dict(bin=test_slopes("x"), unit_cd=test_slopes("cd_unit"), per_depth={f"d{d}": test_slopes("x", d) for d in (0, 1, 2)})
    it = primary["bin"]["interaction_holistic_minus_factorised"]
    verdict_primary = "CONFIRMED" if it["lo"] > 0 else "NOT CONFIRMED"
    err_table = {m: {f"d{d}": {b: (float(np.mean([1 - x[m] for x in units if x["depth"] == d and x["bin"] == b])) if any(
        x["depth"] == d and x["bin"] == b for x in units) else None) for b in G.BINS} for d in (0, 1, 2)} for m in methods}
    # ── T2 budget, invariance ──
    budget, inv = {}, {}
    for arm in P:
        bd = {}
        for key, sel in [(f"d{d}", lambda x, d=d: x["depth"] == d) for d in (0, 1, 2)] + [(f"bin{b}", lambda x, b=b: x["bin"] == b) for b in G.BINS]:
            us = [x for x in units if sel(x)]
            eps = collections.defaultdict(list)
            for x in us:
                for k, v in x[arm + "__atoms"].items():
                    eps[k].append(v)
            sums = [sum(x[arm + "__atoms"].values()) for x in us]
            bd[key] = dict(observed_error=float(np.mean([1 - x[arm] for x in us])), union_bound_mean=float(np.mean([min(1.0, s) for s in sums])),
                           violations=int(sum(1 for x, s in zip(us, sums) if (1 - x[arm]) > s)), n=len(us),
                           atom_eps={k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in sorted(eps.items())})
        budget[arm] = bd
        tab = collections.defaultdict(list)
        for x in units:
            for k, v in x[arm + "__atoms"].items():
                tab[k].append((x["role"] + "|" + " ".join(w for w in words(x["phrase"]) if w != k), v))
        res = {}
        for k, rows in tab.items():
            ctxs = sorted({c for c, _ in rows})
            if len(ctxs) < 2:
                continue
            y = np.array([v for _, v in rows])
            lab_ = np.array([ctxs.index(c) for c, _ in rows])

            def spread(lv):
                return float(np.var([y[lv == i].mean() for i in range(len(ctxs)) if (lv == i).any()]))
            obs = spread(lab_)
            null = [spread(rng.permutation(lab_)) for _ in range(1000)]
            res[k] = dict(n=len(rows), n_contexts=len(ctxs), spread=obs, p_perm=float((1 + sum(n_ >= obs for n_ in null)) / 1001),
                          eps_range=[float(min(y[lab_ == i].mean() for i in range(len(ctxs)))), float(max(y[lab_ == i].mean() for i in range(len(ctxs))))])
        inv[arm] = res
    mdl = {arm: dict(chosen=SEL[arm]["chosen"], channels=SEL[arm]["channels"], dl=SEL[arm]["dl"], lexicon_floats=SEL[arm]["lexicon_floats"],
                     pca_floats=SEL[arm]["pca_floats"], n_content=SEL[arm]["n_content"], n_words=SEL[arm]["n_words"],
                     posfree_probe_r2_generated=SEL[arm].get("posfree_probe_r2_generated"),
                     channel_counts=dict(collections.Counter(str(c) for c in SEL[arm]["channels"].values())),
                     proto_geo={w: SEL[arm]["proto_geo"][w] for w in ("left", "right", "front", "back", "middle", "top", "bottom")
                                if w in SEL[arm]["proto_geo"]}) for arm in P}
    pc = collections.defaultdict(dict)
    for x in units:
        pc[x["tid"]] = dict(phrase=x["phrase"], depth=x["depth"], bin=x["bin"], role=x["role"],
                            hits={m: float(np.mean([y[m] for y in units if y["tid"] == x["tid"]])) for m in methods})
    dev = dev_read(DR, SR, PRIMARY)
    out = dict(verdict_primary=verdict_primary, primary_arm=PRIMARY, primary=primary, bars=bars, new_set_pass=bars[PRIMARY]["pass"], cells=cells,
               error_table=err_table, budget=budget, invariance=inv, mdl=mdl, per_compound=pc, dev=dev, stats=U["stats"],
               units=[{k: v for k, v in x.items()} for x in units])
    json.dump(out, open(os.path.join(ROOT, "results", "bind", "b1_3.json"), "w", encoding="utf-8"), indent=1, default=float)
    for k in sorted(cells):
        v = cells[k]
        print(f"[b1.3 read] {k}: n {v['n']} ({v['compounds']} cpds) " + " | ".join(f"{m} {v['hit'][m]['mean']:.3f}" for m in cols), flush=True)
    for m, b in bars.items():
        print(f"[b1.3 read] NEW-SET BAR {m}: d1 {b['i_depth1']['mean']:.3f} {b['i_pass']} | d2 {b['ii_depth2']['mean']:.3f} {b['ii_pass']} | "
              f"d0 {b['iii_depth0']['mean']:.3f} vs gdino {gd0:.3f} {b['iii_pass']} | PASS {b['pass']}", flush=True)
    for k, v in primary["bin"].items():
        print(f"[b1.3 read] PRIMARY slope vs CD bin {k}: {json.dumps(v)}", flush=True)
    for k, v in primary["unit_cd"].items():
        print(f"[b1.3 read] slope vs unit CD {k}: {json.dumps(v)}", flush=True)
    for d, t in primary["per_depth"].items():
        print(f"[b1.3 read] per-depth {d}: fact {json.dumps(t[FACT])} hol {json.dumps(t[HOL])} inter {json.dumps(t['interaction_holistic_minus_factorised'])}", flush=True)
    for m in methods:
        print(f"[b1.3 read] error by depth x bin {m}: {json.dumps(err_table[m])}", flush=True)
    for arm in P:
        print(f"[b1.3 read] channels {arm}: {mdl[arm]['channel_counts']} {json.dumps(mdl[arm]['channels'])}", flush=True)
        print(f"[b1.3 read] proto_geo {arm}: " + json.dumps({w: [round(v, 2) for v in g] for w, g in mdl[arm]['proto_geo'].items()}), flush=True)
        print(f"[b1.3 read] budget {arm}: " + json.dumps({k: (round(b['observed_error'], 3), round(b['union_bound_mean'], 3), b['violations'], b['n']) for k, b in budget[arm].items()}), flush=True)
        print(f"[b1.3 read] invariance {arm}: " + json.dumps({k: (round(v['p_perm'], 3), v['n'], v['n_contexts']) for k, v in inv[arm].items()}), flush=True)
    print(f"[b1.3 read] DEV bars: {json.dumps(dev['bars'])}", flush=True)
    print(f"[b1.3 read] DEV CD slopes: {json.dumps(dev['cd_curve'])}", flush=True)
    print(f"[b1.3 read] PRIMARY {verdict_primary}; new-set bars PASS {bars[PRIMARY]['pass']}", flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("b1_3_bind", bench="libero_90_c", tag="read", seed=0, command="python experiments/b1_3_read.py --record",
                            metrics=dict(verdict_primary=verdict_primary, primary_arm=PRIMARY, primary=primary, bars=bars, cells=cells,
                                         error_table=err_table, budget=budget, mdl={k: {kk: vv for kk, vv in v.items() if kk != "dl"} for k, v in mdl.items()},
                                         dev=dict(bars=dev["bars"], cd_curve=dev["cd_curve"], cells=dev["cells"]), stats=U["stats"]),
                            bytes=dict(experts=int(4 * mdl[PRIMARY]["lexicon_floats"]), tree=0, codes=0, samples=0, teacher_copies=0,
                                       total=int(4 * (mdl[PRIMARY]["lexicon_floats"] + mdl[PRIMARY]["pca_floats"]))),
                            config=dict(note="B1.3 prequential channel MDL + relative-only geometry; generated MCD unit set (S1 hit metric)"),
                            script="experiments/b1_3_read.py"))


def dev_read(DR, SR, PRIMARY):
    """SECONDARY: S1's 240 dev units, the same bars and CD slope as B1 / B1.1 / B1.2 (b1_2_read.py), for the B1.3 arms and B1b-gen."""
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    b0 = json.load(open(os.path.join(ROOT, "results", "bind", "b0.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(ROOT, "results", "l90c", "split.json"), encoding="utf-8"))
    cd = {k: v["cd"] for k, v in split["dev_per_task"].items()}
    refs = {"b1b": "b1b", "s1c": "s1c", "gdino": "b1a", "qwen": "qwen__bbox_phrase", "molmo": "molmo__point_phrase"}
    P = {}
    for arm in ARMS:
        p = os.path.join(B13, arm, "pred_dev.pkl")
        if os.path.exists(p):
            P[arm] = [x["peak"] for x in pickle.load(open(p, "rb"))["pred"]]
    z = np.load(os.path.join(B13, "b1b", "pred_dev.npz"), allow_pickle=True)
    hol = {str(k): int(np.argmax(l)) for k, l in zip(z["keys"], z["logits"])}
    P["b1b_gen"] = [hol[f"{e['role']}|{e['frame']}"] for e in lab["dev"]]
    seen_types = set(b0["seen_body_types"])
    units = []
    for ix, e in enumerate(lab["dev"]):
        zz = np.load(os.path.join(ROOT, e["frame"]), allow_pickle=True)
        names, cov = SR.masks(zz)
        bodies = [str(b) for b in zz["body_names"]]
        tg = e["targets"][0]
        t = cov[names.index(tg)]
        comp, btype = SR.competitors(tg, names, bodies)
        if tg in bodies:
            btype = SR.itype(tg)
        cc = [cov[names.index(c)] for c in comp]
        bu = b0["units"][ix]
        u = dict(tid=e["tid"], role=e["role"], phrase=e["phrase"], depth=e["depth"], seen=e["seen"], body_seen=btype in seen_types,
                 body_type=btype, cd=cd.get(e["tid"]), bin=split["dev_per_task"].get(e["tid"], {}).get("bin"))
        for k, src in refs.items():
            u[k] = bu[src]
        for m, pk in P.items():
            u[m] = SR.hit(pk[ix], t, cc)
        units.append(u)
    methods = list(P) + list(refs)

    def per(us, k):
        d = {}
        for x in us:
            d.setdefault(x["tid"], []).append(float(x[k]))
        return DR.rate(d)
    cells = {}
    for d in (0, 1, 2):
        for s in ("all", "seen", "heldout"):
            us = [x for x in units if x["depth"] == d and (s == "all" or (s == "seen") == x["seen"])]
            if us:
                cells[f"d{d}_{s}"] = dict(n=len(us), tasks=len({x["tid"] for x in us}), hit={m: per(us, m) for m in methods})
    d2s = [x for x in units if x["depth"] == 2 and x["body_seen"]]
    cells["d2_body_seen_all"] = dict(n=len(d2s), hit={m: per(d2s, m) for m in methods})
    for nm, fn in (("caddy_units", lambda x: "caddy" in (x["body_type"] or "")), ("drawer_units", lambda x: "drawer" in x["phrase"])):
        us = [x for x in units if fn(x)]
        cells[nm] = dict(n=len(us), hit={m: per(us, m) for m in methods})
    gd0 = cells["d0_all"]["hit"]["gdino"]["mean"]
    bars = {}
    for m in methods:
        i_, ii, iii = cells["d1_heldout"]["hit"][m]["mean"], cells["d2_body_seen_all"]["hit"][m]["mean"], cells["d0_all"]["hit"][m]["mean"]
        bars[m] = dict(i=i_, ii=ii, iii=iii, pass_=bool(i_ >= 0.7 and ii >= 0.7 and iii >= gd0))
    rng = np.random.default_rng(0)
    cdu = [x for x in units if x["cd"] is not None]
    tids = sorted({x["tid"] for x in cdu})

    def slope(us, m):
        xs = np.array([x["cd"] for x in us])
        ys = np.array([1.0 - x[m] for x in us])
        return float(np.polyfit(xs, ys, 1)[0]) if np.ptp(xs) > 0 else float("nan")
    curve = {}
    for m in methods:
        bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(tids, len(tids), replace=True)
            bs.append(slope([x for t in pick for x in cdu if x["tid"] == t], m))
        curve[m] = dict(slope=slope(cdu, m), lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)), n_tasks=len(tids))
    pt = {}
    for x in units:
        pt.setdefault(f"{x['tid']}|{x['role']}", dict(phrase=x["phrase"], depth=x["depth"], seen=x["seen"], hits={m: [] for m in methods}))
        for m in methods:
            pt[f"{x['tid']}|{x['role']}"]["hits"][m].append(x[m])
    pt = {k: dict(v, hits={m: float(np.mean(h)) for m, h in v["hits"].items()}) for k, v in pt.items()}
    return dict(bars=bars, groundingdino_d0=gd0, cells=cells, cd_curve=curve, per_task=pt)


if __name__ == "__main__":
    main()

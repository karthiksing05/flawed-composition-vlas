"""PLAN_BINDER B1 read (docs/PLAN_BINDER.md section 7; pre-registration experiments/prereg/pb1.py, ledger m3_prereg tag pb1).  CPU.

Scores the unsupervised prototype binder (experiments/b1_bind.py; results/bind/b1/<arm>/pred_dev.pkl) on S1's 240 dev units EXACTLY
as S1 / S1c / B0 (s1_read.hit, the competitor rule; d4_read.rate, the two-level task-cluster bootstrap).  Simulator labels are
used here, for scoring and diagnostics only.

BARS (S1c's / B0's, point estimates, per arm): (i) held-out depth-1 hit >= 0.70; (ii) depth-2 seen-body hit >= 0.70;
(iii) depth-0 hit >= GroundingDINO alone (0.786).  VERDICT: B1 PASS iff the PRIMARY arm (siglip: the policy's own features)
passes all three.  dino (DINOv2-small) is the comparison arm.
Also: caddy units; part-proposal recall vs sim masks; the MDL channel of every word; confounds (words with identical training
contexts); error vs compound divergence for B1 vs B1b (S1's holistic binder) with a task-bootstrap slope CI; the T2 error budget
over atoms (region proposal, role, parent resolution, each content word) and the atom-invariance test.

    python experiments/b1_read.py [--record]  -> results/bind/b1.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
S1 = os.path.join(ROOT, "results", "bind", "s1")
B1 = os.path.join(ROOT, "results", "bind", "b1")
ARMS = ("siglip", "dino")
PRIMARY = "siglip"
REFS = {"b1b": "b1b", "s1c": "s1c", "s1b": "s1b", "gdino": "b1a", "qwen": "qwen__bbox_phrase", "molmo": "molmo__point_phrase",
        "robopoint": "robopoint"}
SPLITS = ("all", "seen", "heldout", "heldout_body_seen", "heldout_body_unseen")
N_BOOT = 2000


def words(p):
    return re.findall(r"[a-z]+", p.lower())


def main():
    import d4_read as DR
    import s1_read as SR
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    b0 = json.load(open(os.path.join(ROOT, "results", "bind", "b0.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(ROOT, "results", "l90c", "split.json"), encoding="utf-8"))
    cd = {k: v["cd"] for k, v in split["dev_per_task"].items()}
    P, SEL = {}, {}
    for arm in ARMS:
        p = os.path.join(B1, arm, "pred_dev.pkl")
        if os.path.exists(p):
            P[arm] = pickle.load(open(p, "rb"))["pred"]
            SEL[arm] = json.load(open(os.path.join(B1, arm, "select.json"), encoding="utf-8"))
            assert len(P[arm]) == len(lab["dev"])
    assert PRIMARY in P
    cache = {}

    def frame(f):
        if f not in cache:
            z = np.load(os.path.join(ROOT, f), allow_pickle=True)
            names, cov = SR.masks(z)
            cache[f] = (names, cov, [str(b) for b in z["body_names"]])
        return cache[f]
    seen_types = set(b0["seen_body_types"])
    units, by_frame = [], {}
    for ix, e in enumerate(lab["dev"]):
        names, cov, bodies = frame(e["frame"])
        tg = e["targets"][0]
        t = cov[names.index(tg)]
        comp, btype = SR.competitors(tg, names, bodies)
        if tg in bodies:
            btype = SR.itype(tg)
        cc = [cov[names.index(c)] for c in comp]
        bu = b0["units"][ix]
        assert (bu["tid"], bu["role"], bu["phrase"]) == (e["tid"], e["role"], e["phrase"])
        # the target's reference body (scoring only): the body itself; a region's body; a table region's reference body
        if tg in bodies:
            ref_body = [tg]
            type_set = [b for b in bodies if SR.itype(b) == SR.itype(tg)]
        else:
            ref_body = [b for b in bodies if tg.startswith(b + "_")] or [b for b in bodies if btype and SR.itype(b) == btype]
            type_set = ref_body
        refm = np.max([cov[names.index(b)] for b in ref_body], 0) if ref_body else np.zeros(256)
        typem = np.max([cov[names.index(b)] for b in type_set], 0) if type_set else np.zeros(256)
        u = dict(ix=ix, tid=e["tid"], role=e["role"], phrase=e["phrase"], target=tg, depth=e["depth"], seen=e["seen"], body_type=btype,
                 body_seen=btype in seen_types, cd=cd.get(e["tid"]), frame=e["frame"])
        for k, src in REFS.items():
            u[k] = bu[src]
        for arm, pr in P.items():
            p = pr[ix]
            u[arm] = SR.hit(p["peak"], t, cc)
            cm = np.unpackbits(p["cand_mask"], axis=1)[:, :256].astype(bool)
            u[arm + "__iou"] = SR.iou(cm[p["chosen"]] if p["chosen"] >= 0 else np.zeros(256, bool), t)
            # ── atoms (scoring only) ──
            hits_c = np.array([SR.hit(int(k), t, cc) for k in p["cand_peak"]])
            on_type = np.array([typem[int(k)] > 0 for k in p["cand_peak"]])
            ious = [SR.iou(m, t) for m in cm]
            at = {"@proposal": float(not hits_c.any())}
            if p["parent"] >= 0:
                at["@parent"] = float(refm[int(p["cand_peak"][p["parent"]])] <= 0)
            at["@role"] = None
            for w, s in p["per"].items():
                if w == "@role":
                    continue
                ch = SEL[arm]["channels"].get(w)
                if ch == "app":
                    at[w] = float(not on_type[int(np.argmax(s))])
                elif ch == "geo":
                    pool = np.where(on_type)[0]
                    at[w] = float(not (len(pool) and hits_c[pool[int(np.argmax(s[pool]))]]))
            u[arm + "__atoms"] = at
            u[arm + "__best_iou"] = float(max(ious)) if ious else 0.0
            u[arm + "__peak"] = int(p["peak"])
        units.append(u)
        by_frame.setdefault(e["frame"], []).append(len(units) - 1)
    for f, ids in by_frame.items():                              # role atom: the peak lands on the OTHER role's target
        for i in ids:
            for arm in P:
                others = [j for j in ids if units[j]["role"] != units[i]["role"]]
                names, cov, _ = frame(f)
                pk = units[i][arm + "__peak"]
                e_role = float(any(pk >= 0 and cov[names.index(units[j]["target"])][pk] > 0 and cov[names.index(units[i]["target"])][pk] <= 0
                                   for j in others))
                units[i][arm + "__atoms"]["@role"] = e_role

    def per(us, k):
        d = {}
        for u in us:
            if u.get(k) is not None:
                d.setdefault(u["tid"], []).append(float(u[k]))
        return DR.rate(d)

    def sel(depth, split_, role=None):
        return [u for u in units if u["depth"] == depth and (role is None or u["role"] == role) and (
            split_ == "all" or (split_ == "seen" and u["seen"]) or (split_ == "heldout" and not u["seen"])
            or (split_ == "heldout_body_seen" and not u["seen"] and u["body_seen"])
            or (split_ == "heldout_body_unseen" and not u["seen"] and not u["body_seen"]))]
    methods = list(P) + list(REFS)
    cells = {}
    for depth in (0, 1, 2):
        for s in SPLITS:
            for role in (None, "object", "goal"):
                us = sel(depth, s, role)
                if us:
                    cells[f"d{depth}_{s}" + (f"_{role}" if role else "")] = dict(n=len(us), tasks=sorted({u["tid"] for u in us}),
                                                                                hit={m: per(us, m) for m in methods})
    d2s = [u for u in units if u["depth"] == 2 and u["body_seen"]]
    cells["d2_body_seen_all"] = dict(n=len(d2s), tasks=sorted({u["tid"] for u in d2s}), hit={m: per(d2s, m) for m in methods})
    cad = [u for u in units if "caddy" in (u["body_type"] or "")]
    cells["caddy_units"] = dict(n=len(cad), tasks=sorted({u["tid"] for u in cad}), hit={m: per(cad, m) for m in methods})
    gd0 = cells["d0_all"]["hit"]["gdino"]["mean"]
    bars = {}
    for arm in P:
        i_, ii, iii = cells["d1_heldout"]["hit"][arm], cells["d2_body_seen_all"]["hit"][arm], cells["d0_all"]["hit"][arm]
        b = dict(i_depth1_heldout=i_, i_pass=bool(i_["mean"] >= 0.70), ii_depth2_body_seen=ii, ii_pass=bool(ii["mean"] >= 0.70),
                 iii_depth0=iii, iii_groundingdino=gd0, iii_pass=bool(iii["mean"] >= gd0), caddy=cells["caddy_units"]["hit"][arm],
                 iou_mean=float(np.mean([u[arm + "__iou"] for u in units])))
        b["pass"] = b["i_pass"] and b["ii_pass"] and b["iii_pass"]
        bars[arm] = b
    verdict = "PASS" if bars[PRIMARY]["pass"] else "FAIL"

    # ── part-proposal recall vs sim masks (scoring only) ──
    recall = {}
    for arm in P:
        r = {}
        for u in units:
            kind = "body" if u["target"] in frame(u["frame"])[2] else ("table_region" if any(u["target"].startswith(t + "_") for t in SR.TABLES) else "part_region")
            r.setdefault(kind, []).append(u[arm + "__best_iou"])
            r.setdefault("peak_" + kind, []).append(1.0 - u[arm + "__atoms"]["@proposal"])
        recall[arm] = {k: dict(iou50=float(np.mean(np.array(v) >= 0.5)), mean_best_iou=float(np.mean(v)), n=len(v)) if not k.startswith("peak_")
                       else dict(any_candidate_hits=float(np.mean(v)), n=len(v)) for k, v in r.items()}

    # ── confounds: content words whose training contexts (distinct role phrases) coincide ──
    conf = {}
    for arm in P:
        wo = {w: sorted(set(v)) for w, v in SEL[arm]["word_occ"].items()}
        ctx = {w: frozenset(x.split("|", 1)[1] for x in v) for w, v in wo.items()}
        groups = {}
        for w, c in ctx.items():
            if SEL[arm]["channels"].get(w):
                groups.setdefault(c, []).append(w)
        conf[arm] = dict(identical=[sorted(g) for g in groups.values() if len(g) > 1],
                         near=sorted({tuple(sorted((w1, w2))) for w1 in ctx for w2 in ctx if w1 < w2 and SEL[arm]["channels"].get(w1)
                                      and SEL[arm]["channels"].get(w2) and len(ctx[w1] & ctx[w2]) / len(ctx[w1] | ctx[w2]) >= 0.8
                                      and ctx[w1] != ctx[w2]}))

    # ── error vs compound divergence (recombination dev tasks with a CD value) ──
    rng = np.random.default_rng(0)
    cdu = [u for u in units if u["cd"] is not None]
    tids = sorted({u["tid"] for u in cdu})

    def slope(us, m):
        x = np.array([u["cd"] for u in us])
        y = np.array([1.0 - u[m] for u in us])
        return float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float("nan")
    cdcurve = {}
    for m in list(P) + ["b1b", "s1c", "gdino", "qwen", "molmo"]:
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(tids, len(tids), replace=True)
            us = [u for t in pick for u in cdu if u["tid"] == t]
            boots.append(slope(us, m))
        bins = {}
        for u in cdu:
            bins.setdefault(split["dev_per_task"][u["tid"]]["bin"], []).append(1.0 - u[m])
        cdcurve[m] = dict(slope=slope(cdu, m), lo=float(np.nanpercentile(boots, 2.5)), hi=float(np.nanpercentile(boots, 97.5)),
                          error_by_bin={k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in sorted(bins.items())},
                          n_units=len(cdu), n_tasks=len(tids))

    # ── T2 error budget and the atom-invariance test ──
    budget, inv = {}, {}
    for arm in P:
        bd = {}
        for depth in (0, 1, 2):
            us = [u for u in units if u["depth"] == depth]
            eps = {}
            for u in us:
                for k, v in u[arm + "__atoms"].items():
                    if v is not None:
                        eps.setdefault(k, []).append(v)
            per_unit_sum = [sum(v for v in u[arm + "__atoms"].values() if v is not None) for u in us]
            bd[f"d{depth}"] = dict(observed_error=float(np.mean([1 - u[arm] for u in us])),
                                   union_bound_mean_sum=float(np.mean([min(1.0, s) for s in per_unit_sum])),
                                   atom_eps={k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in sorted(eps.items())},
                                   violations=int(sum(1 for u, s in zip(us, per_unit_sum) if (1 - u[arm]) > s)), n=len(us))
        budget[arm] = bd
        # invariance: per atom, its error in each co-occurring context (the rest of the compound = the role phrase without
        # the atom); permutation test of the between-context spread against shuffling context labels across units
        tab = {}
        for u in units:
            for k, v in u[arm + "__atoms"].items():
                if v is None:
                    continue
                ctxk = u["role"] + "|" + " ".join(w for w in words(u["phrase"]) if w != k)
                tab.setdefault(k, []).append((ctxk, v, u["tid"]))
        res = {}
        for k, rows in tab.items():
            ctxs = sorted({c for c, _, _ in rows})
            if len(ctxs) < 2:
                continue
            y = np.array([v for _, v, _ in rows])
            lab_ = np.array([ctxs.index(c) for c, _, _ in rows])

            def spread(lv):
                ms = [y[lv == i].mean() for i in range(len(ctxs)) if (lv == i).any()]
                return float(np.var(ms))
            obs = spread(lab_)
            null = [spread(rng.permutation(lab_)) for _ in range(1000)]
            res[k] = dict(contexts={c: dict(eps=float(y[lab_ == i].mean()), n=int((lab_ == i).sum())) for i, c in enumerate(ctxs)},
                          spread=obs, p_perm=float((1 + sum(n_ >= obs for n_ in null)) / 1001), n=len(rows))
        inv[arm] = res

    # ── referent purity of the contact events (the 61 simulator-rendered matched demos; sim masks, diagnostics only) ──
    import glob
    import b1_bind as BB
    tasks = json.load(open(os.path.join(S1, "tasks.json"), encoding="utf-8"))
    pur = {"grasp": [], "release": []}
    for fz in sorted(glob.glob(os.path.join(B1, "demo", "*.npz"))):
        z = np.load(fz, allow_pickle=True)
        tid = str(z["tid"])
        if tid in BB.SKIP_CELLS:
            continue
        nm, cv = SR.masks(z)
        atoms = [x for x in tasks[tid]["goal"] if len(x) == 3]
        for (g, r), at in zip(BB.transports(json.loads(str(z["events"]))), atoms):
            if at[1] in nm:
                pur["grasp"].append(float(cv[nm.index(at[1])][g["token"]] > 0))
            if r is not None and at[2] in nm:
                pur["release"].append(float(cv[nm.index(at[2])][r["token"]] > 0))
    purity = {k: dict(on_target=float(np.mean(v)) if v else None, n=len(v)) for k, v in pur.items()}
    # ── MDL table ──
    mdl = {arm: dict(chosen=SEL[arm]["chosen"], channels=SEL[arm]["channels"], gain=SEL[arm]["gain"], lexicon_floats=SEL[arm]["lexicon_floats"],
                     pca_floats=SEL[arm]["pca_floats"], n_occ=SEL[arm]["n_occ"], n_frames=SEL[arm].get("n_frames"),
                     n_content=SEL[arm]["n_content"], n_words=SEL[arm]["n_words"]) for arm in P}
    pt = {}
    for u in units:
        p = pt.setdefault(f"{u['tid']}|{u['role']}", dict(phrase=u["phrase"], target=u["target"], depth=u["depth"], seen=u["seen"],
                                                         body_seen=u["body_seen"], hits={m: [] for m in methods}))
        for m in methods:
            p["hits"][m].append(u[m])
    pt = {k: dict(v, hits={m: float(np.mean(x)) for m, x in v["hits"].items()}) for k, v in pt.items()}
    out = dict(verdict=verdict, primary=PRIMARY, bars=bars, cells=cells, per_task=pt, recall=recall, purity=purity, confounds=conf, cd_curve=cdcurve,
               budget=budget, invariance=inv, mdl=mdl, units=[{k: v for k, v in u.items() if not k.endswith("__peak")} for u in units])
    json.dump(out, open(os.path.join(ROOT, "results", "bind", "b1.json"), "w", encoding="utf-8"), indent=1, default=float)
    for k in ("d0_all", "d0_seen", "d0_heldout", "d1_all", "d1_seen", "d1_heldout", "d2_all", "d2_seen", "d2_heldout",
              "d2_body_seen_all", "caddy_units"):
        v = cells.get(k)
        if v:
            print(f"[b1 read] {k}: n {v['n']} ({len(v['tasks'])} tasks) " + " | ".join(
                f"{m} {v['hit'][m]['mean']:.3f}" for m in methods), flush=True)
    for k, v in sorted(pt.items()):
        print(f"[b1 read] {k}: {v['phrase']!r} d{v['depth']} seen {v['seen']}: " + " ".join(f"{m} {x:.2f}" for m, x in v["hits"].items()), flush=True)
    for arm, b in bars.items():
        print(f"[b1 read] BAR {arm}: i {b['i_depth1_heldout']['mean']:.3f} {b['i_pass']} | ii {b['ii_depth2_body_seen']['mean']:.3f} "
              f"{b['ii_pass']} | iii {b['iii_depth0']['mean']:.3f} vs {gd0:.3f} {b['iii_pass']} | PASS {b['pass']} | caddy {b['caddy']['mean']:.3f}", flush=True)
    for m, c in cdcurve.items():
        print(f"[b1 read] CD slope {m}: {c['slope']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] bins {c['error_by_bin']}", flush=True)
    for arm in P:
        print(f"[b1 read] recall {arm}: {recall[arm]}", flush=True)
        print(f"[b1 read] budget {arm}: " + json.dumps({d: (b['observed_error'], b['union_bound_mean_sum'], b['violations']) for d, b in budget[arm].items()}), flush=True)
        print(f"[b1 read] invariance {arm}: " + json.dumps({k: (round(v['p_perm'], 3), v['n']) for k, v in inv[arm].items()}), flush=True)
        print(f"[b1 read] confounds {arm}: {conf[arm]['identical']}", flush=True)
    print(f"[b1 read] referent purity (matched demos, sim masks): {purity}", flush=True)
    print(f"[b1 read] VERDICT {verdict}", flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("b1_bind", bench="libero_90_c", tag="read", seed=0, command="python experiments/b1_read.py --record",
                            metrics=dict(verdict=verdict, bars=bars, cells=cells, cd_curve=cdcurve, recall=recall, purity=purity, budget=budget,
                                         mdl={arm: {k: v for k, v in m.items() if k != "gain"} for arm, m in mdl.items()}, confounds=conf),
                            bytes=dict(experts=int(4 * mdl[PRIMARY]["lexicon_floats"]), tree=0, codes=0, samples=0, teacher_copies=0,
                                       total=int(4 * (mdl[PRIMARY]["lexicon_floats"] + mdl[PRIMARY]["pca_floats"]))),
                            config=dict(note="B1 unsupervised prototype binder (S1 hit metric)"), script="experiments/b1_read.py"))


if __name__ == "__main__":
    main()

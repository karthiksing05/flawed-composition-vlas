"""FLAWED_COMPOSITION prototype A, stage 1 read (section 5.5; m3_prereg tag pa1).  CPU.

Data: results/flawed/a1_roll/<arm>_<phase>_<tid>.json (fa1_roll.py), arms H, R, A, A_noframe, A_nogate, A_nosum; N2's 16 compounds
(8 known K1-K8, 8 new N1-N8) x {familiar, new} layout; start states (smoke 0-3, main 4-23) and flow noise shared by every arm.
Per episode:
  s   success (LIBERO's strict goal check)
  g   s, or the first object moved while the commanded gripper is closed (the bind track) is the task's object
  r   relaxed placement: s, or at episode end the object is within the instructed target's footprint: horizontal distance to the
      target reference (plate / bowl body, stove cook site, cabinet top-side site, rack top site, stove-front site)
      <= max(RAD[target], dxy_rest + 0.03) and height above it within [dz_rest - 0.04, dz_rest + 0.06], dz_rest and dxy_rest from
      fa1_data.py calib (results/flawed/a1_calib.json), whether or not it is stable or released.  The drawer target (K7, N7) has no
      relaxed test: r = s there.
  p   s given g (strict) and r given g (relaxed), pooled ratios sum(x * g) / sum(g)
Cells: s and g are means over compounds of the task-cell rates; p is pooled over the cell's episodes.
Pre-registered (pa1), probability scale, 95% CIs from a 2,000-draw percentile bootstrap (seed 0) resampling compounds within each
status and start states, jointly for every arm:
  floor   s_H(known, familiar) < 0.5 on the main run: uninformative (the oracle-state recipe failed), stop
  A1      parity: s_A(known, fam) - s_H(known, fam) >= -0.10 (point estimate)
  A2      placement: p_rel,A(new, fam) - p_rel,H(new, fam) > 0 and p_rel,A(new, fam) - p_rel,R(new, fam) > 0, both CIs excluding 0
  A3      layout: L_g(X) = mean_c [g(c, fam) - g(c, new)] over all 16 compounds; L_g(H) - L_g(A) > 0 and L_g(R) - L_g(A) > 0, both
          CIs excluding 0
  A4      relocation: for H and for R, C_p(X) = p_rel,X(known, fam) - p_rel,X(new, fam) > 0 with the CI excluding 0 (both arms)
  A5      (descriptive) for each ablation Y in A_noframe, A_nogate, A_nosum: p_rel,A(new, fam) - p_rel,Y(new, fam) and
          L_g(Y) - L_g(A), with CIs; Y "carries" A2 / A3 when that contrast's CI excludes 0 above 0
  Verdict, the first that applies: floor -> "uninformative"; not A1 -> "A1 fails: A is a worse policy"; A2 and A3 -> "confirmed";
  A2 and not A3 -> "placement confirmed, layout not"; otherwise (A1 holds, A2 fails) -> "killed: the factored motor does not fix
  placement".  A4 is reported beside the verdict (holds / fails) and does not change it.
Secondary: strict p; the pick-and-place pairs only (K1-K6, N1-N6); success effects; per-compound table; the habit table: new-compound
episodes with g = 1 and s = 0, the object's end position classified in order as instructed (relaxed test at the instructed target),
habit (relaxed test at any target this object is moved to in a training task), other (at any other scene target) or elsewhere.
Listed factorisation violations (pre-registered): K7 / N7 (articulated drawer opened before the place), K8 / N8 (no gripper closure:
A's target term never runs).
-> results/flawed/a1_read.json   (--smoke: results/flawed/a1_smoke.json, the gate: s_H(known, familiar) >= 0.5 on the smoke)

    python experiments/fa1_read.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
ARMS = ("H", "R", "A", "A_noframe", "A_nogate", "A_nosum")
B, SEED = 2000, 0
RAD = {"plate_1": 0.08, "akita_black_bowl_1": 0.06, "flat_stove_1_cook_region": 0.09, "wooden_cabinet_1_top_side": 0.10,
       "wine_rack_1_top_region": 0.10, "main_table_stove_front_region": 0.06}
DZ_LO, DZ_HI, DXY_PAD = -0.04, 0.06, 0.03
TRAINED = {"akita_black_bowl_1": ["plate_1", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side"],
           "wine_bottle_1": ["wooden_cabinet_1_top_side", "wine_rack_1_top_region"],
           "cream_cheese_1": ["akita_black_bowl_1"], "plate_1": ["main_table_stove_front_region"]}
CLASSES = ("instructed", "habit", "other", "elsewhere")


def at(final, o, tg, cal):
    c = cal.get(f"{o}|{tg}")
    if c is None:
        return False
    p, ref = np.array(final[o]), np.array(final[tg])
    dz, dxy = p[2] - ref[2], np.linalg.norm(p[:2] - ref[:2])
    return bool(dxy <= max(RAD[tg], float(np.mean(c["dxy"])) + DXY_PAD) and c["dz_mean"] + DZ_LO <= dz <= c["dz_mean"] + DZ_HI)


def load(phase):
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    comps = sorted({t["compound"] for t in n2["tasks"]}, key=lambda c: (c[0] != "K", int(c[1:])))
    ks = n2["states"][phase]
    shape = (len(comps), 2, len(ks))
    A = {m: {x: np.full(shape, np.nan) for x in ("S", "G", "R", "H")} for m in ARMS}
    err, missing = {m: 0 for m in ARMS}, []
    for t in n2["tasks"]:
        ci, li = comps.index(t["compound"]), (0 if t["layout"] == "familiar" else 1)
        o, tg = t["object"], t["target"]
        others = [x for x in RAD if x not in (o, tg) and x not in TRAINED.get(o, [])]
        for m in ARMS:
            f = os.path.join(OUT, "a1_roll", f"{m}_{phase}_{t['tid']}.json")
            if not os.path.exists(f):
                missing.append(os.path.basename(f))
                continue
            eps = [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
            for e in eps:
                j = ks.index(int(e["trial"]))
                s = bool(e["success"])
                g = s or e.get("bound") == o
                r = s or (tg in RAD and at(e["final"], o, tg, cal))
                A[m]["S"][ci, li, j], A[m]["G"][ci, li, j], A[m]["R"][ci, li, j] = s, g, r
                if t["status"] == "new" and not s:
                    cls = ("instructed" if r else
                           "habit" if any(at(e["final"], o, x, cal) for x in TRAINED.get(o, [])) else
                           "other" if any(at(e["final"], o, x, cal) for x in others) else "elsewhere")
                    A[m]["H"][ci, li, j] = CLASSES.index(cls) + (0 if g else 10)
                err[m] += e.get("error") is not None
    status = np.array([c[0] == "K" for c in comps])
    return n2, comps, status, A, err, missing


def ratio(num, den):
    d = den.sum()
    return float(num.sum() / d) if d > 0 else float("nan")


def stats(X, known, new, kk):
    """X: one arm's {S, G, R, H} [16, 2, n]; known / new compound indices (resampled), kk state indices (resampled)."""
    S, G, R = (X[k][:, :, kk] for k in ("S", "G", "R"))
    r, allc = {}, np.concatenate([known, new])
    for st, idx in (("known", known), ("new", new)):
        for li, lay in enumerate(("fam", "new")):
            r[f"s_{st}_{lay}"] = float(S[idx, li].mean())
            r[f"g_{st}_{lay}"] = float(G[idx, li].mean())
            r[f"p_{st}_{lay}"] = ratio(S[idx, li] * G[idx, li], G[idx, li])
            r[f"prel_{st}_{lay}"] = ratio(R[idx, li] * G[idx, li], G[idx, li])
    g = G.mean(-1)
    s = S.mean(-1)
    r["L_g"] = float((g[allc, 0] - g[allc, 1]).mean())
    r["L_s"] = float((s[allc, 0] - s[allc, 1]).mean())
    r["C_s"] = float(s[known].mean() - s[new].mean())
    r["C_p_fam"] = r["prel_known_fam"] - r["prel_new_fam"]
    r["C_pstrict_fam"] = r["p_known_fam"] - r["p_new_fam"]
    r["L_p"] = ratio(R[allc, 0] * G[allc, 0], G[allc, 0]) - ratio(R[allc, 1] * G[allc, 1], G[allc, 1])
    return r


def contrasts(st):
    c = {"A1": st["A"]["s_known_fam"] - st["H"]["s_known_fam"],
         "A2_vs_H": st["A"]["prel_new_fam"] - st["H"]["prel_new_fam"], "A2_vs_R": st["A"]["prel_new_fam"] - st["R"]["prel_new_fam"],
         "A3_vs_H": st["H"]["L_g"] - st["A"]["L_g"], "A3_vs_R": st["R"]["L_g"] - st["A"]["L_g"],
         "A4_H": st["H"]["C_p_fam"], "A4_R": st["R"]["C_p_fam"],
         "A2strict_vs_H": st["A"]["p_new_fam"] - st["H"]["p_new_fam"], "A2strict_vs_R": st["A"]["p_new_fam"] - st["R"]["p_new_fam"]}
    for y in ("A_noframe", "A_nogate", "A_nosum"):
        c[f"A5_{y}_p"] = st["A"]["prel_new_fam"] - st[y]["prel_new_fam"]
        c[f"A5_{y}_Lg"] = st[y]["L_g"] - st["A"]["L_g"]
        c[f"A5_{y}_s_known_fam"] = st[y]["s_known_fam"] - st["A"]["s_known_fam"]
    return c


def boot(A, status, sub=None, arms=ARMS):
    idx = np.arange(len(status)) if sub is None else np.asarray(sub)
    known, new = idx[status[idx]], idx[~status[idx]]
    n = A[arms[0]]["S"].shape[-1]
    one = lambda kn, nw, kk: {m: stats(A[m], kn, nw, kk) for m in arms}         # noqa: E731
    point = one(known, new, np.arange(n))
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(B):
        kn, nw, kk = rng.choice(known, len(known)), rng.choice(new, len(new)), rng.choice(n, n)
        draws.append(one(kn, nw, kk))

    def ci(get):
        v = np.array([get(d) for d in draws], float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2

    cells = {m: {k: dict(est=point[m][k], ci=ci(lambda d, m=m, k=k: d[m][k])) for k in point[m]} for m in arms}
    out = dict(cells=cells)
    if set(ARMS) <= set(arms):
        pc = contrasts(point)
        dc = [contrasts(d) for d in draws]
        out["contrasts"] = {k: dict(est=pc[k], ci=[float(np.nanpercentile([d[k] for d in dc], 2.5)),
                                                    float(np.nanpercentile([d[k] for d in dc], 97.5))]) for k in pc}
    return out


def verdict(cells, con):
    lo = lambda k: con[k]["ci"][0] > 0 and con[k]["est"] > 0                    # noqa: E731
    v = dict(floor=cells["H"]["s_known_fam"]["est"] < 0.5, A1=con["A1"]["est"] >= -0.10, A2=lo("A2_vs_H") and lo("A2_vs_R"),
             A3=lo("A3_vs_H") and lo("A3_vs_R"), A4=lo("A4_H") and lo("A4_R"))
    for y in ("A_noframe", "A_nogate", "A_nosum"):
        v[f"A5_{y}_carries_A2"] = lo(f"A5_{y}_p")
        v[f"A5_{y}_carries_A3"] = lo(f"A5_{y}_Lg")
    if v["floor"]:
        v["verdict"] = "uninformative: H below 0.5 on known / familiar with oracle state"
    elif not v["A1"]:
        v["verdict"] = "A1 fails: A is a worse policy"
    elif v["A2"] and v["A3"]:
        v["verdict"] = "confirmed"
    elif v["A2"]:
        v["verdict"] = "placement confirmed, layout not"
    else:
        v["verdict"] = "killed: the factored motor does not fix placement"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    phase = "smoke" if a.smoke else "main"
    n2, comps, status, A, err, missing = load(phase)
    if a.smoke:
        sm = {m: dict(s_known_fam=float(np.nanmean(A[m]["S"][status, 0])), g_known_fam=float(np.nanmean(A[m]["G"][status, 0])),
                      n=int(np.isfinite(A[m]["S"][status, 0]).sum())) for m in ARMS}
        res = dict(phase="smoke", cells=sm, gate=dict(H_s_known_fam=sm["H"]["s_known_fam"], passed=sm["H"]["s_known_fam"] >= 0.5),
                   errors=err, missing=missing)
        json.dump(res, open(os.path.join(OUT, "a1_smoke.json"), "w", encoding="utf-8"), indent=1)
        print(json.dumps(res, indent=1))
        return
    assert not missing, f"missing roll files: {missing[:5]} ... ({len(missing)})"
    for m in ARMS:
        for x in ("S", "G", "R"):
            assert np.isfinite(A[m][x]).all(), (m, x)
    full = boot(A, status)
    v = verdict(full["cells"], full["contrasts"])
    pp = [i for i, c in enumerate(comps) if int(c[1:]) <= 6]
    sec = dict(pick_place=boot(A, status, sub=pp))
    sec["pick_place_verdict"] = verdict(sec["pick_place"]["cells"], sec["pick_place"]["contrasts"])
    per = {}
    for i, c in enumerate(comps):
        per[c] = {m: dict(s_fam=float(A[m]["S"][i, 0].mean()), s_new=float(A[m]["S"][i, 1].mean()), g_fam=float(A[m]["G"][i, 0].mean()),
                          g_new=float(A[m]["G"][i, 1].mean()), r_fam=float(A[m]["R"][i, 0].mean()), r_new=float(A[m]["R"][i, 1].mean()))
                  for m in ARMS}
    habit = {}
    for m in ARMS:
        h = A[m]["H"][~status]
        for li, lay in enumerate(("fam", "new")):
            hl = h[:, li]
            n_gf = int(np.isin(hl, [0, 1, 2, 3]).sum())
            habit[f"{m}_{lay}"] = dict(n_g_fail=n_gf, **{cl: (float((hl == i).sum() / n_gf) if n_gf else float("nan"))
                                                          for i, cl in enumerate(CLASSES)})
    res = dict(phase="main", n_states=len(n2["states"]["main"]), compounds=comps, verdict=v, **full, secondary=sec,
               per_compound=per, habit=habit, errors=err)
    json.dump(res, open(os.path.join(OUT, "a1_read.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(verdict=v, contrasts=full["contrasts"]), indent=1))


if __name__ == "__main__":
    main()

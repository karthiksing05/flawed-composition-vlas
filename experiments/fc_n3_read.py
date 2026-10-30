"""FLAWED_COMPOSITION N3 read (section 4.12; m3_prereg tag pn3): the phase dissociation (layout -> grasp, compound -> placement) in
LIBERO-Spatial's scene.  CPU.

Data: results/flawed/n3_roll/<model>_<phase>_<tid>.json (+ _pos.json, the client-side end-of-episode positions), models pi05, groot
and (if its main ran) xvla; 8 pairs (K_i known, N_i new: same BDDL, same states, the target replaced) x {familiar, new} layout;
start states k (smoke 0-3, main 4-23) shared by every cell and model.
Per episode:
  s   success (LIBERO's strict goal check)
  g   s, or the first object moved while the commanded gripper is closed (the client's bind track) is bowl_1, the named bowl
  r   relaxed placement: s, or at episode end bowl_1 is within the instructed target's footprint: horizontal distance to the target
      reference (plate body; stove cook-region site; cabinet top-side site) <= RAD[target] and height above it within
      [dz_rest - 0.04, dz_rest + 0.06] (dz_rest from fc_n3_tasks.py's teleport-and-settle check), whether or not it is stable
  p   s given g (strict) and r given g (relaxed), pooled ratios sum(x * g) / sum(g)
Per model (probability scale), c over the 16 compounds, i over the 8 pairs, cell means over states:
  L_g = mean_c [g(c, fam) - g(c, new)]                C_g = mean_i [gbar(K_i) - gbar(N_i)]     (gbar: mean over layouts)
  L_p = p(all c, fam) - p(all c, new)                 C_p = p(known, both layouts) - p(new, both layouts)   (relaxed: primary)
  H1 = L_g - C_g > 0        H2 = C_p - L_p > 0 (relaxed)        H3 = (L_g - C_g) - (L_p - C_p) > 0        each with its 95% CI
  excluding 0.  H4 (secondary): among new-compound episodes with g = 1 and s = 0, the end position is classified, in this order, as
  instructed (relaxed test at the instructed target), habit (relaxed test at the plate, the target of the known task with the same
  bowl and referent), other (relaxed test at the scene's remaining fixture target) or elsewhere; H4 = rate(habit) - rate(other) > 0,
  CI excluding 0 (the chance expectation for the trained habit is the rate at the control target).
CIs: 2,000-draw percentile bootstrap (seed 0), resampling pairs (K_i and N_i together) and start states, jointly for all models.
Verdict (pre-registered, pn3), over the models whose main ran:
  per model: H3 holds = estimate > 0 and CI lower bound > 0.
  confirmed                      H3 holds in every model, and H1 and H2 hold in every model
  confirmed on the interaction   H3 holds in every model, but H1 or H2 fails in at least one
  mixed                          H3 holds in at least one model and fails in at least one
  killed                         H3 fails in every model
  (no model ran its main: "floored at the gate: stopped")
Secondary: strict p; success effects L and C; per-pair table; H1-H3 without pair 2 (table centre, whose referent the translation
only approximately keeps); the habit table on all new-compound failures regardless of g.
-> results/flawed/n3_read.json   (--smoke: results/flawed/n3_smoke.json, the gate only)

    python experiments/fc_n3_read.py [--smoke] [--models pi05,groot[,xvla]]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
B, SEED = 2000, 0
RAD = {"plate": 0.08, "stove": 0.09, "cabinet": 0.10}
DZ_LO, DZ_HI = -0.04, 0.06
SITE = {"stove": "flat_stove_1_cook_region", "cabinet": "wooden_cabinet_1_top_side"}
CLASSES = ("instructed", "habit", "other", "elsewhere")


def at(final, sites, tgt, dz_rest):
    b = np.array(final["akita_black_bowl_1"])
    ref = np.array(final["plate_1"]) if tgt == "plate" else np.array(sites[SITE[tgt]])
    dz = b[2] - ref[2]
    return bool(np.linalg.norm(b[:2] - ref[:2]) <= RAD[tgt] and dz_rest[tgt] + DZ_LO <= dz <= dz_rest[tgt] + DZ_HI)


def load(phase, models):
    n3 = json.load(open(os.path.join(OUT, "n3_tasks.json"), encoding="utf-8"))
    dz_rest = {k: v["mean"] for k, v in n3["dz_rest"].items()}
    npair = len(n3["pairs"])
    ks = n3["states"][phase]
    shape = (2 * npair, 2, len(ks))
    A = {m: {x: np.full(shape, np.nan) for x in ("S", "G", "R", "H")} for m in models}   # H: habit class index (new, g=1, s=0)
    err, missing = {m: 0 for m in models}, []
    other_of = {p["ci"]: ("cabinet" if p["new_target"] == "stove" else "stove") for p in n3["pairs"]}
    for t in n3["tasks"]:
        ci = t["pair"] + (0 if t["status"] == "known" else npair)
        li = 0 if t["layout"] == "familiar" else 1
        for m in models:
            f = os.path.join(OUT, "n3_roll", f"{m}_{phase}_{t['tid']}.json")
            fp = f[:-5] + "_pos.json"
            if not (os.path.exists(f) and os.path.exists(fp)):
                missing.append(os.path.basename(f))
                continue
            eps = [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
            pos = json.load(open(fp, encoding="utf-8"))
            assert len(pos) == len(eps), (f, len(pos), len(eps))
            for e, q in zip(eps, pos):
                j = ks.index(int(e["trial"]))
                s = bool(e["success"])
                g = s or e.get("bound") == t["object"]
                r = s or at(q["final"], q["sites"], t["target"], dz_rest)
                A[m]["S"][ci, li, j], A[m]["G"][ci, li, j], A[m]["R"][ci, li, j] = s, g, r
                if t["status"] == "new" and not s:
                    cls = ("instructed" if at(q["final"], q["sites"], t["target"], dz_rest) else
                           "habit" if at(q["final"], q["sites"], "plate", dz_rest) else
                           "other" if (other_of[t["pair"]] in SITE and SITE[other_of[t["pair"]]] in q["sites"] and
                                       at(q["final"], q["sites"], other_of[t["pair"]], dz_rest)) else "elsewhere")
                    A[m]["H"][ci, li, j] = CLASSES.index(cls) + (0 if g else 10)
                err[m] += e.get("error") is not None
    return n3, npair, A, err, missing


def ratio(num, den):
    d = den.sum()
    return float(num.sum() / d) if d > 0 else float("nan")


def effects(a, ii, kk, npair):
    S, G, R, H = (a[x][:, :, kk] for x in ("S", "G", "R", "H"))
    K, N = np.asarray(ii), np.asarray(ii) + npair
    allc = np.concatenate([K, N])
    gm, sm = G.mean(-1), S.mean(-1)
    r = {}
    r["L_g"] = float((gm[allc, 0] - gm[allc, 1]).mean())
    r["C_g"] = float(gm[K].mean(1).mean() - gm[N].mean(1).mean())
    for nm, X in (("R", R), ("S", S)):
        sfx = "" if nm == "R" else "_strict"
        r["L_p" + sfx] = ratio((X * G)[allc][:, 0], G[allc][:, 0]) - ratio((X * G)[allc][:, 1], G[allc][:, 1])
        r["C_p" + sfx] = ratio((X * G)[K], G[K]) - ratio((X * G)[N], G[N])
        r["H2" + sfx] = r["C_p" + sfx] - r["L_p" + sfx]
        r["H3" + sfx] = (r["L_g"] - r["C_g"]) - (r["L_p" + sfx] - r["C_p" + sfx])
    r["H1"] = r["L_g"] - r["C_g"]
    r["L_s"] = float((sm[allc, 0] - sm[allc, 1]).mean())
    r["C_s"] = float(sm[K].mean(1).mean() - sm[N].mean(1).mean())
    for st, cs in (("known", K), ("new", N)):
        for li, lay in enumerate(("familiar", "new")):
            c = f"{st}_{lay}"
            r[f"s_{c}"] = float(S[cs][:, li].mean())
            r[f"g_{c}"] = float(G[cs][:, li].mean())
            r[f"r_{c}"] = float(R[cs][:, li].mean())
            r[f"p_strict_{c}"] = ratio((S * G)[cs][:, li], G[cs][:, li])
            r[f"p_relaxed_{c}"] = ratio((R * G)[cs][:, li], G[cs][:, li])
    h = H[N]
    gf = h < 10                                                         # new-compound, g = 1, s = 0
    n_gf = int(gf.sum())
    for i, cl in enumerate(CLASSES):
        r[f"habit_{cl}"] = float((h == i).sum() / n_gf) if n_gf else float("nan")
        r[f"habit_all_{cl}"] = float(((h == i) | (h == 10 + i)).sum() / np.isfinite(h).sum()) if np.isfinite(h).sum() else float("nan")
    r["habit_n"] = n_gf
    r["H4"] = r["habit_habit"] - r["habit_other"]
    return r


def boot(A, models, npair, pairs=None):
    pairs = np.arange(npair) if pairs is None else np.asarray(pairs)
    n = A[models[0]]["S"].shape[-1]
    point = {m: effects(A[m], pairs, np.arange(n), npair) for m in models}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(B):
        ii, kk = rng.choice(pairs, len(pairs)), rng.choice(n, n)
        draws.append({m: effects(A[m], ii, kk, npair) for m in models})
    out = {}
    for m in models:
        out[m] = {}
        for k, v in point[m].items():
            x = np.array([d[m][k] for d in draws], float)
            x = x[np.isfinite(x)]
            out[m][k] = dict(est=v, ci95=[float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [float("nan")] * 2)
    return out


def holds(x):
    return x["est"] > 0 and x["ci95"][0] > 0


def verdict(r, models):
    if not models:
        return "floored at the gate: stopped"
    h3 = {m: holds(r[m]["H3"]) for m in models}
    h12 = {m: holds(r[m]["H1"]) and holds(r[m]["H2"]) for m in models}
    if all(h3.values()):
        return "confirmed" if all(h12.values()) else "confirmed on the interaction"
    if any(h3.values()):
        return "mixed: H3 holds in " + ", ".join(m for m in models if h3[m]) + "; fails in " + ", ".join(m for m in models if not h3[m])
    return "killed: H3 fails in every model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default="pi05,groot")
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if m]
    phase = "smoke" if a.smoke else "main"
    n3, npair, A, err, missing = load(phase, models)
    assert not missing, f"missing result files: {missing}"
    if a.smoke:
        res = dict(models=models, errors=err, n_states=A[models[0]]["S"].shape[-1], gate={}, cells={})
        for m in models:
            x = float(A[m]["S"][:npair, 0].mean())
            res["gate"][m] = dict(baseline_success=x, passes=bool(x >= 0.20), episode_errors=err[m])
            res["cells"][m] = {f"{st}/{lay}": dict(s=float(A[m]["S"][sl, li].mean()), g=float(A[m]["G"][sl, li].mean()),
                                                   r=float(A[m]["R"][sl, li].mean()))
                               for st, sl in (("known", slice(0, npair)), ("new", slice(npair, 2 * npair)))
                               for li, lay in enumerate(("familiar", "new"))}
        with open(os.path.join(OUT, "n3_smoke.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print(json.dumps(res, indent=1))
        return
    sm = json.load(open(os.path.join(OUT, "n3_smoke.json"), encoding="utf-8"))
    assert all(sm["gate"][m]["passes"] for m in models), "a model below the gate has no main read"
    r = boot(A, models, npair)
    res = dict(primary=r, models=models, gate=sm["gate"], verdict=verdict(r, models), errors=err,
               n_states=A[models[0]]["S"].shape[-1], episodes={m: int(np.isfinite(A[m]["S"]).sum()) for m in models},
               per_model_hypotheses={m: {h: holds(r[m][h]) for h in ("H1", "H2", "H3", "H4", "H2_strict", "H3_strict")} for m in models})
    res["without_pair2"] = {m: {k: v for k, v in d.items() if k in ("H1", "H2", "H3", "H4", "L_g", "C_g", "L_p", "C_p")}
                            for m, d in boot(A, models, npair, pairs=[i for i in range(npair) if n3["pairs"][i]["spatial_index"] != 2]).items()}
    res["per_pair"] = {m: {f"{p['known']['key']}/{p['new']['key']}": {
        f"{st}_{lay}": dict(s=float(A[m]["S"][p["ci"] + off, li].mean()), g=float(A[m]["G"][p["ci"] + off, li].mean()),
                            r=float(A[m]["R"][p["ci"] + off, li].mean()))
        for st, off in (("known", 0), ("new", npair)) for li, lay in enumerate(("familiar", "new"))}
        for p in n3["pairs"]} for m in models}
    res["per_pair_language"] = {f"{p['known']['key']}/{p['new']['key']}": [p["known"]["language"], p["new"]["language"]] for p in n3["pairs"]}
    with open(os.path.join(OUT, "n3_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_pair",)}, indent=1))


if __name__ == "__main__":
    main()

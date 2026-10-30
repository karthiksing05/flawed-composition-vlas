"""FLAWED_COMPOSITION N2 read (section 4.11; m3_prereg tag pn2): the compound x layout factorial in one scene, pi0.5 and GR00T N1.5.
CPU.

Data: results/flawed/n2_roll/<model>_<phase>_<tid>.json (fc_n2_roll.sh), models pi05 and groot, 8 known + 8 new compounds x
{familiar, new} layout, start states k (smoke 0-3, main 4-23) shared by every cell and model (same base placement and noise seed).
Per episode: s = success; g = success or the first object moved while the commanded gripper is closed (the client's bind track) is
the task's object; p = s given g.
Per model, on the probability scale (primary), with c over compounds, s(c, l) the task-cell success rate:
  L   layout effect      mean_c [s(c, fam) - s(c, new)]                         (> 0: the new layout hurts)
  C   compound effect    mean_{known c} sbar(c) - mean_{new c} sbar(c), sbar = mean over the two layouts   (> 0: the new compound hurts)
  I   interaction        layout drop on known compounds - layout drop on new compounds
  DD  model x factor     (L_groot - C_groot) - (L_pi05 - C_pi05)                  (> 0: GR00T leans to the layout axis, pi0.5 to the compound axis)
CIs: 2,000-draw percentile bootstrap (seed 0) resampling compounds within each status AND start states, jointly for both models
(the pairing of layouts, compounds and models through shared states and noise is kept).
Verdict (pre-registered, pn2):
  gate (smoke)    a model whose known-compound / familiar-layout smoke success is < 0.20 is floored: stop, report
  uninformative   a model with success <= 0.05 in all three other cells; both models -> the experiment is uninformative
  P1  L_groot > 0 (CI excludes 0)     P2  C_pi05 > 0 (CI excludes 0)     P3  DD > 0 (CI excludes 0)
  double dissociation: P1, P2, P3, L_groot > C_groot and C_pi05 > L_pi05 (points)
  partial: P3 without P1 or P2, or P1 and P2 without P3
  reversed: DD < 0 with the CI excluding 0
  two axes dropped (kill): DD's CI includes 0 and, in both models, L and C both > 0 with CIs excluding 0 and L - C's CI including 0
  otherwise inconclusive
Secondary: g and p (pooled ratios); the log-odds scale (task cells smoothed (s n + 0.5) / (n + 1)); pick-and-place pairs only
(K1-K6, N1-N6); per-cell N1 input novelty (n2_novelty.json) and per model the Spearman of the task composite with failure over the
16 tasks; the manipulation check mean_c [d(new) - d(fam)] for d_a, d_b, d_c; episode errors.
-> results/flawed/n2_read.json   (--smoke: results/flawed/n2_smoke.json, the gate only)

    python experiments/fc_n2_read.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
MODELS = ("pi05", "groot")
B, SEED = 2000, 0


def load(phase):
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    comps = sorted({t["compound"] for t in n2["tasks"]}, key=lambda c: (c[0] != "K", int(c[1:])))
    ks = n2["states"][phase]
    S = {m: np.full((len(comps), 2, len(ks)), np.nan) for m in MODELS}
    G = {m: np.full((len(comps), 2, len(ks)), np.nan) for m in MODELS}
    err, missing = {m: 0 for m in MODELS}, []
    for t in n2["tasks"]:
        ci, li = comps.index(t["compound"]), (0 if t["layout"] == "familiar" else 1)
        for m in MODELS:
            f = os.path.join(OUT, "n2_roll", f"{m}_{phase}_{t['tid']}.json")
            if not os.path.exists(f):
                missing.append(os.path.basename(f))
                continue
            eps = [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
            for e in eps:
                j = ks.index(int(e["trial"]))
                S[m][ci, li, j] = float(bool(e["success"]))
                G[m][ci, li, j] = float(bool(e["success"]) or e.get("bound") == t["object"])
                err[m] += e.get("error") is not None
    status = np.array([c[0] == "K" for c in comps])
    return comps, status, S, G, err, missing


def effects(S, G, known, new, kk):
    """S, G [16, 2, n] for one model; known / new compound indices (resampled); kk state indices (resampled)."""
    s = S[:, :, kk].mean(-1)                                       # [16, 2]
    allc = np.concatenate([known, new])
    drop = s[:, 0] - s[:, 1]
    sbar = s.mean(1)
    r = dict(L=float(drop[allc].mean()), C=float(sbar[known].mean() - sbar[new].mean()),
             I=float(drop[known].mean() - drop[new].mean()))
    r["L_minus_C"] = r["L"] - r["C"]
    for st, idx in (("known", known), ("new", new)):
        for li, lay in enumerate(("familiar", "new")):
            r[f"s_{st}_{lay}"] = float(s[idx, li].mean())
            r[f"g_{st}_{lay}"] = float(G[idx][:, li][:, kk].mean())
            gg, ss = G[idx][:, li][:, kk], S[idx][:, li][:, kk]
            r[f"p_{st}_{lay}"] = float(ss.sum() / gg.sum()) if gg.sum() > 0 else float("nan")
    gm = G[:, :, kk].mean(-1)
    gdrop = gm[:, 0] - gm[:, 1]
    r.update(L_g=float(gdrop[allc].mean()), C_g=float(gm.mean(1)[known].mean() - gm.mean(1)[new].mean()),
             I_g=float(gdrop[known].mean() - gdrop[new].mean()))

    def pr(cs, li):
        gg, ss = G[cs][:, li][:, kk], S[cs][:, li][:, kk]
        return ss.sum() / gg.sum() if gg.sum() > 0 else np.nan
    r.update(L_p=float(pr(allc, 0) - pr(allc, 1)),
             C_p=float((S[known][:, :, kk].sum() / max(G[known][:, :, kk].sum(), 1e-9)) - (S[new][:, :, kk].sum() / max(G[new][:, :, kk].sum(), 1e-9))),
             I_p=float((pr(known, 0) - pr(known, 1)) - (pr(new, 0) - pr(new, 1))))
    n = len(kk)
    lg = np.log((s * n + 0.5) / (n - s * n + 0.5))
    ldrop = lg[:, 0] - lg[:, 1]
    r.update(L_logit=float(ldrop[allc].mean()), C_logit=float(lg.mean(1)[known].mean() - lg.mean(1)[new].mean()),
             I_logit=float(ldrop[known].mean() - ldrop[new].mean()))
    return r


def both(S, G, known, new, kk):
    e = {m: effects(S[m], G[m], known, new, kk) for m in MODELS}
    for sfx in ("", "_g", "_p", "_logit"):
        e[f"DD{sfx}"] = (e["groot"][f"L{sfx}"] - e["groot"][f"C{sfx}"]) - (e["pi05"][f"L{sfx}"] - e["pi05"][f"C{sfx}"])
    return e


def boot(S, G, status, sub=None):
    idx = np.arange(len(status)) if sub is None else np.asarray(sub)
    known, new = idx[status[idx]], idx[~status[idx]]
    n = S[MODELS[0]].shape[-1]
    point = both(S, G, known, new, np.arange(n))
    rng = np.random.default_rng(SEED)
    draws = [both(S, G, rng.choice(known, len(known)), rng.choice(new, len(new)), rng.choice(n, n)) for _ in range(B)]

    def ci(get):
        v = np.array([get(d) for d in draws], float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2
    out = {}
    for m in MODELS:
        out[m] = {k: dict(est=v, ci95=ci(lambda d, m=m, k=k: d[m][k])) for k, v in point[m].items()}
    for k in [x for x in point if x.startswith("DD")]:
        out[k] = dict(est=point[k], ci95=ci(lambda d, k=k: d[k]))
    return out


def excl(x):
    return x["ci95"][0] > 0 or x["ci95"][1] < 0


def verdict(r, gate):
    if any(not g["passes"] for g in gate.values()):
        return "floored at the gate: stopped"
    uninf = {m: all(r[m][f"s_{c}"]["est"] <= 0.05 for c in ("known_new", "new_familiar", "new_new")) for m in MODELS}
    if all(uninf.values()):
        return "uninformative: success floors in every new cell for both models"
    P1 = r["groot"]["L"]["est"] > 0 and r["groot"]["L"]["ci95"][0] > 0
    P2 = r["pi05"]["C"]["est"] > 0 and r["pi05"]["C"]["ci95"][0] > 0
    P3 = r["DD"]["est"] > 0 and r["DD"]["ci95"][0] > 0
    cross = r["groot"]["L"]["est"] > r["groot"]["C"]["est"] and r["pi05"]["C"]["est"] > r["pi05"]["L"]["est"]
    if P1 and P2 and P3 and cross:
        v = "double dissociation holds"
    elif P3 or (P1 and P2):
        v = "partial"
    elif r["DD"]["est"] < 0 and r["DD"]["ci95"][1] < 0:
        v = "reversed"
    elif (not excl(r["DD"]) and all(r[m]["L"]["ci95"][0] > 0 and r[m]["C"]["ci95"][0] > 0 and not excl(r[m]["L_minus_C"]) for m in MODELS)):
        v = "two axes dropped: both factors hurt equally in both models"
    else:
        v = "inconclusive"
    return v + "".join(f" ({m} uninformative)" for m in MODELS if uninf[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    phase = "smoke" if a.smoke else "main"
    comps, status, S, G, err, missing = load(phase)
    assert not missing, f"missing result files: {missing}"
    gate = {}
    if a.smoke or os.path.exists(os.path.join(OUT, "n2_smoke.json")):
        sm = load("smoke")[2] if not a.smoke else S
        for m in MODELS:
            x = float(sm[m][status, 0].mean())
            gate[m] = dict(baseline_success=x, passes=bool(x >= 0.20))
    if a.smoke:
        cells = {m: {f"{st}/{lay}": dict(s=float(S[m][status == (st == 'known'), li].mean()), g=float(G[m][status == (st == 'known'), li].mean()))
                     for st in ("known", "new") for li, lay in enumerate(("familiar", "new"))} for m in MODELS}
        per_task = {m: {c: [float(S[m][i, 0].mean()), float(S[m][i, 1].mean())] for i, c in enumerate(comps)} for m in MODELS}
        res = dict(gate=gate, cells=cells, per_compound_success_fam_new=per_task, errors=err, n_states=S[MODELS[0]].shape[-1])
        with open(os.path.join(OUT, "n2_smoke.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print(json.dumps(res, indent=1))
        return
    r = boot(S, G, status)
    res = dict(primary=r, gate=gate, verdict=verdict(r, gate), errors=err, n_states=S[MODELS[0]].shape[-1], compounds=comps,
               episodes={m: int(np.isfinite(S[m]).sum()) for m in MODELS})
    res["pick_place_only"] = boot(S, G, status, sub=[i for i, c in enumerate(comps) if int(c[1:]) <= 6])
    res["per_compound"] = {m: {c: dict(s_fam=float(S[m][i, 0].mean()), s_new=float(S[m][i, 1].mean()), g_fam=float(G[m][i, 0].mean()),
                                       g_new=float(G[m][i, 1].mean())) for i, c in enumerate(comps)} for m in MODELS}
    # N1 input novelty per cell, its manipulation check, and its relation to failure
    nov = json.load(open(os.path.join(OUT, "n2_novelty.json"), encoding="utf-8"))
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    res["novelty_cells"] = nov["cells"]
    rng = np.random.default_rng(SEED)
    man = {}
    for d in ("d_a", "d_b", "d_c", "input_composite"):
        diff = np.array([nov["tasks"][str(2 * i + 1)][d] - nov["tasks"][str(2 * i)][d] for i in range(len(comps))])
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
        man[d] = dict(est=float(diff.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    res["manipulation_check_new_minus_familiar"] = man
    def _rank(v):                                                  # average ranks (ties share the mean rank)
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[o] = np.arange(len(v), dtype=float)
        for u in np.unique(v):
            m_ = v == u
            r[m_] = r[m_].mean()
        return r

    def spearmanr(x_, y_):                                         # scipy is absent from this venv: Pearson of average ranks
        a_, b_ = _rank(np.asarray(x_, float)), _rank(np.asarray(y_, float))
        if a_.std() == 0 or b_.std() == 0:
            return (float("nan"),)
        return (float(np.corrcoef(a_, b_)[0, 1]),)
    rel = {}
    for m in MODELS:
        x = np.array([nov["tasks"][str(t["tid"])]["input_composite"] for t in n2["tasks"]])
        y = np.array([1 - S[m][comps.index(t["compound"]), 0 if t["layout"] == "familiar" else 1].mean() for t in n2["tasks"]])
        cid = np.array([comps.index(t["compound"]) for t in n2["tasks"]])
        rho = float(spearmanr(x, y)[0])
        bs = []
        for _ in range(B):
            pick = rng.integers(0, len(comps), len(comps))
            sel = np.concatenate([np.where(cid == c)[0] for c in pick])
            v = spearmanr(x[sel], y[sel])[0]
            if np.isfinite(v):
                bs.append(v)
        rel[m] = dict(rho=rho, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_tasks=len(x))
    res["input_novelty_vs_failure"] = rel
    with open(os.path.join(OUT, "n2_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_compound"}, indent=1))


if __name__ == "__main__":
    main()

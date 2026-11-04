"""Place diagnostic read (docs/PLACE_DIAG.md 2; m3_prereg pp_place).  CPU.  Dev states 4-23, seeds 10-12.
  --stage t0     T0 (failure taxonomy from saved rollouts) and P_min's m-conditioned 3F beside the frozen one (post-hoc, PLAN_PARTS 10.14)
                 -> results/flawed/pp_place_t0.json
  --stage final  T1, T2, T4 and the verdict per contrast -> results/flawed/pp_place_read.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from fa1g_read import radius                                                # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
SEEDS = (10, 11, 12)
ARMS = {"P_min": ("pp_cons_roll", "P2naf"), "P_cov": ("pp_cov_roll", "P2cov"), "P_cov_h": ("pp_cov_roll", "P2covh")}
CONTRASTS = (("N1", "K1"), ("N4", "K1"), ("N6", "K6"))
TASKS = ("K1", "N1", "N4", "K6", "N6")
LAYS = ("new", "familiar")
TOL, TILT, B = 0.03, 15.0, 2000
n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
ks = n2["states"]["main"]


def task(comp, lay):
    return next(t for t in n2["tasks"] if t["compound"] == comp and t["layout"] == lay)


def stored(arm, t):
    d, nm = ARMS[arm]
    out = {}
    for sd in SEEDS:
        for v in json.load(open(os.path.join(OUT, d, f"{nm}_s{sd}_main_{t['tid']}.json"), encoding="utf-8"))["tasks"].values():
            for e in v["episodes"]:
                out[(sd, int(e["trial"]))] = e
    return out


def t0():
    res = dict(taxonomy={}, radii={}, m3F={})
    comps = [f"K{i}" for i in range(1, 7)] + [f"N{i}" for i in range(1, 7)]
    for arm in ARMS:
        for lay in LAYS:
            for c in comps:
                t = task(c, lay)
                rad = radius(t["object"], t["target"], cal)
                res["radii"][f"{c}|{t['object']}|{t['target']}"] = rad
                E = list(stored(arm, t).values())
                A = [e for e in E if e["min_dxy_held"] is not None and e["min_dxy_held"] <= rad]
                cl = dict(success=0, held_at_end=0, end_off_centre=0, end_within_tol=0)
                ends = []
                pt_name = t["target"]
                for e in A:
                    pos = np.array(e["final"][t["object"]])
                    pt = np.array(e["final"][pt_name])
                    dxy = float(np.linalg.norm((pos - pt)[:2]))
                    if e["success"]:
                        cl["success"] += 1
                    else:
                        ends.append(dxy)
                        if e["held_end"]:
                            cl["held_at_end"] += 1
                        elif dxy >= TOL:
                            cl["end_off_centre"] += 1
                        else:
                            cl["end_within_tol"] += 1
                res["taxonomy"][f"{arm}|{lay}|{c}"] = dict(n=len(E), arrived=len(A), **cl, regrasp=int(sum(e["held_segments"] > 1 for e in A)),
                                                           fail_end_d_med=float(np.median(ends)) if ends else None)
                print("[t0]", arm, lay, c, f"rad {rad:.3f}", json.dumps(res["taxonomy"][f"{arm}|{lay}|{c}"]), flush=True)
    res["m3F"] = m_three_factor()
    fn = os.path.join(OUT, "pp_place_t0.json")
    json.dump(res, open(fn, "w", encoding="utf-8"), indent=1)


def m_three_factor():
    """P_min's 3F with an object's factors pooled over only its known pairings under the new pairing's m, beside the frozen pool.  Post-hoc."""
    import torch  # noqa: F401
    import fa1_common as FC
    from pp_p_roll import load_any
    modes = []
    for sd in SEEDS:
        m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_full_naf_s{sd}", "P.npz"))
        modes.append({e: int(m.tmode[m.tpart[FC.ENT.index(e)]]) for e in FC.ENT if int(m.tpart[FC.ENT.index(e)]) >= 0})
    same = all(x == modes[0] for x in modes)
    res = dict(modes=modes[0], same_across_seeds=same, label="post-hoc; no verdict (PLAN_PARTS 10.14)")
    if not same:
        return res
    for lay in LAYS:
        fam = {t["compound"]: t for t in n2["tasks"] if t["layout"] == lay and int(t["compound"][1:]) <= 6}
        H, AR, S = {}, {}, {}
        for c, t in fam.items():
            rad = radius(t["object"], t["target"], cal)
            E = stored("P_min", t)
            H[c] = np.array([[E[(sd, k)]["held_step"] is not None for k in ks] for sd in SEEDS], float)
            AR[c] = np.array([[E[(sd, k)]["min_dxy_held"] is not None and E[(sd, k)]["min_dxy_held"] <= rad for k in ks] for sd in SEEDS], float)
            S[c] = np.array([[bool(E[(sd, k)]["success"]) for k in ks] for sd in SEEDS], float)
        known = [c for c in fam if c[0] == "K"]

        def est(kk, cond):
            o2 = {}
            for c in sorted(x for x in fam if x[0] == "N"):
                o, tg = fam[c]["object"], fam[c]["target"]
                ko = [k for k in known if fam[k]["object"] == o]
                if cond:
                    ko = [k for k in ko if modes[0][fam[k]["target"]] == modes[0][tg]]
                kt = [k for k in known if fam[k]["target"] == tg]
                if not ko or not kt:
                    o2[c] = None
                    continue
                hs = np.concatenate([H[k][:, kk].ravel() for k in ko])
                ars = np.concatenate([AR[k][:, kk].ravel() for k in ko])
                arq = np.concatenate([AR[k][:, kk].ravel() for k in kt])
                sq = np.concatenate([S[k][:, kk].ravel() for k in kt])
                pred = hs.mean() * (ars * hs).sum() / max(hs.sum(), 1) * (sq * arq).sum() / max(arq.sum(), 1)
                o2[c] = float(S[c][:, kk].mean() - pred)
            return o2

        n = len(ks)
        for cond, nm in ((False, "frozen"), (True, "m_conditioned")):
            pt = est(np.arange(n), cond)
            rng = np.random.default_rng(0)
            dr = [est(rng.choice(n, n), cond) for _ in range(B)]
            res[f"{lay}|{nm}"] = {c: None if pt[c] is None else dict(D3=pt[c], ci=[float(np.percentile([d[c] for d in dr], q)) for q in (2.5, 97.5)]) for c in pt}
            print("[3F]", lay, nm, json.dumps(res[f"{lay}|{nm}"]), flush=True)
    return res


def load_trace(comp, lay):
    E = {}
    for sd in SEEDS:
        for e in json.load(open(os.path.join(OUT, "pp_place_trace", f"P_s{sd}_{comp}_{lay}.json"), encoding="utf-8"))["episodes"]:
            E[(sd, e["trial"])] = e
    return E


def arrays(E, r):
    """-> arrived, success, and the release dict per (seed, state)."""
    key = f"{round(r, 4):.4f}"
    rel = [[E[(sd, k)]["release"].get(key) for k in ks] for sd in SEEDS]
    a = np.array([[x is not None for x in row] for row in rel])
    s = np.array([[bool(E[(sd, k)]["success"]) for k in ks] for sd in SEEDS])
    return a, s, rel


def rate(a, s, kk):
    x, y = a[:, kk], s[:, kk]
    return float((x & y).sum() / x.sum()) if x.sum() else float("nan")


def boot(f):
    rng = np.random.default_rng(0)
    v = np.array([f(rng.choice(len(ks), len(ks))) for _ in range(B)])
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def strata(rel, hmed):
    if rel["none"]:
        return "none"
    if rel["tilt"] > TILT:
        return "tilted"
    d = "d0" if rel["d"] < 0.015 else ("d1" if rel["d"] < TOL else "d2")
    return f"{d}|{'hlo' if rel['h'] < hmed else 'hhi'}"


def adjusted(aA, sA, rA, aB, sB, rB, hmed, kk):
    """-> (standardised difference, overlapping mass, cells)."""
    cA, cB = {}, {}
    for a, s, r, c in ((aA, sA, rA, cA), (aB, sB, rB, cB)):
        for i in range(len(SEEDS)):
            for j in kk:
                if a[i, j]:
                    z = c.setdefault(strata(r[i][j], hmed), [0, 0])
                    z[0] += 1
                    z[1] += int(s[i, j])
    tot = sum(v[0] for v in cA.values()) + sum(v[0] for v in cB.values())
    if not tot:
        return float("nan"), 0.0, {}
    num, mass = 0.0, 0.0
    for z in set(cA) & set(cB):
        w = (cA[z][0] + cB[z][0]) / tot
        num += w * (cA[z][1] / cA[z][0] - cB[z][1] / cB[z][0])
        mass += w
    return (num / mass if mass else float("nan")), mass, dict(first=cA, second=cB)


def drop(comp, lay):
    D = json.load(open(os.path.join(OUT, "pp_place_drop", f"{comp}_{lay}.json"), encoding="utf-8"))
    use = [r for r in D["drops"] if r["d"] <= 0.02]
    kept = [r for r in use if not r["excluded"]]
    cells = {}
    for r in D["drops"]:
        if not r["excluded"]:
            cells.setdefault(f"d{r['d']:.2f}|h{r['h']:.2f}", []).append(r["success"])
    return dict(goal=D["goal"], rate=float(np.mean([r["success"] for r in kept])) if kept else float("nan"),
                n=len(kept), excluded_share=1.0 - len(kept) / len(use), cells={k: float(np.mean(v)) for k, v in sorted(cells.items())},
                end_tilt_med=float(np.median([r["end_tilt"] for r in kept])) if kept else None)


def final():
    res = dict(contrasts={}, tasks={}, drops={}, agreement={})
    ta = json.load(open(os.path.join(OUT, "pp_place_t3.json"), encoding="utf-8"))
    res["t3"] = dict(exact=ta["exact"], control_alive=ta["control_alive"])
    if not (ta["exact"] and ta["control_alive"]):
        res["verdict"] = "STOPPED: T3 not exact (PLACE_DIAG 2)"
    for lay in LAYS:
        T = {c: load_trace(c, lay) for c in TASKS}
        for c in TASKS:
            t = task(c, lay)
            st = stored("P_min", t)
            res["agreement"][f"{c}|{lay}"] = float(np.mean([bool(st[k]["success"]) == bool(T[c][k]["success"]) for k in T[c]]))
            r = radius(t["object"], t["target"], cal)
            a, s, rel = arrays(T[c], r)
            R = [rel[i][j] for i in range(len(SEEDS)) for j in range(len(ks)) if a[i, j]]
            X = [x for x in R if not x["none"]]
            q = lambda f: [float(np.percentile([x[f] for x in X], p)) for p in (10, 50, 90)] if X else None  # noqa: E731
            cl = dict(held_end=0, off=0, within=0)
            for i, sd in enumerate(SEEDS):
                for j, k in enumerate(ks):
                    if a[i, j] and not s[i, j]:
                        e = T[c][(sd, k)]
                        cl["held_end" if e["held_end"] else ("off" if e["end"]["d"] >= TOL else "within")] += 1
            res["tasks"][f"{c}|{lay}"] = dict(radius=r, arrived=int(a.sum()), place=float((a & s).sum() / max(a.sum(), 1)), no_open=int(sum(x["none"] for x in R)),
                                              d=q("d"), h=q("h"), tilt=q("tilt"), tgt_moved=q("tgt_moved"), fail=cl,
                                              fail_end_contact=int(sum(T[c][(sd, k)]["end"]["contact"] for i, sd in enumerate(SEEDS) for j, k in enumerate(ks) if a[i, j] and not s[i, j])))
            res["drops"][f"{c}|{lay}"] = drop(c, lay)
            print("[task]", c, lay, json.dumps(res["tasks"][f"{c}|{lay}"]), "agree", res["agreement"][f"{c}|{lay}"], flush=True)
            print("[drop]", c, lay, json.dumps(res["drops"][f"{c}|{lay}"]), flush=True)
        for x, y in CONTRASTS:
            tx, ty = task(x, lay), task(y, lay)
            rx, ry = radius(tx["object"], tx["target"], cal), radius(ty["object"], ty["target"], cal)
            ax, sx, _ = arrays(T[x], rx)
            ay, sy, _ = arrays(T[y], ry)
            rc = min(rx, ry)
            cx, ux, relx = arrays(T[x], rc)
            cy, uy, rely = arrays(T[y], rc)
            allk = np.arange(len(ks))
            G = rate(ax, sx, allk) - rate(ay, sy, allk)
            Gc = rate(cx, ux, allk) - rate(cy, uy, allk)
            ciG = boot(lambda kk: rate(ax, sx, kk) - rate(ay, sy, kk))
            ciGc = boot(lambda kk: rate(cx, ux, kk) - rate(cy, uy, kk))
            hs = [r["h"] for rel, a in ((relx, cx), (rely, cy)) for i in range(len(SEEDS)) for j in range(len(ks))
                  if a[i, j] for r in [rel[i][j]] if not r["none"] and r["tilt"] <= TILT]
            hmed = float(np.median(hs)) if hs else 0.0
            A, mass, cells = adjusted(cx, ux, relx, cy, uy, rely, hmed, allk)
            ciA = boot(lambda kk: adjusted(cx, ux, relx, cy, uy, rely, hmed, kk)[0])
            D = res["drops"][f"{x}|{lay}"]["rate"] - res["drops"][f"{y}|{lay}"]["rate"]
            if ciG[0] <= 0.0 <= ciG[1]:
                v = "NOISE"
            elif ciGc[0] <= 0.0 <= ciGc[1]:
                v = "ARRIVAL-DEFINITION"
            elif mass < 0.5:
                v = "UNDETERMINED"
            elif abs(A) <= 0.5 * abs(Gc):
                v = "RELEASE-STATE"
            elif np.sign(D) == np.sign(Gc) and abs(D) >= 0.10:
                v = "SAME-STATE: PHYSICS"
            else:
                v = "SAME-STATE: UNMEASURED"
            res["contrasts"][f"{x}-{y}|{lay}"] = dict(G=G, G_ci=ciG, G_c=Gc, G_c_ci=ciGc, r_common=rc, A=A, A_ci=ciA, overlap=mass, hmed=hmed, D=D, verdict=v, cells=cells)
            print("VERDICT", x, y, lay, v, json.dumps({k: res["contrasts"][f"{x}-{y}|{lay}"][k] for k in ("G", "G_ci", "G_c", "G_c_ci", "A", "A_ci", "overlap", "D")}), flush=True)
            print("CELLS", x, y, lay, json.dumps(cells), flush=True)
    json.dump(res, open(os.path.join(OUT, "pp_place_read.json"), "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("t0", "final"))
    a = ap.parse_args()
    (t0 if a.stage == "t0" else final)()

"""F6O2 read (CPU, any venv with numpy): F6 memory with a physical occluder (f6o, SR 45.5) on NEW displaced-layout states, plus
arm M, the memory overlay (experiments/f6o2_tasks.py; docs/SCAFFOLD_RESULTS.md 45.6).  Per model: success in a (visible), b
(occluded-late), c (never-visible), d (moved-beside), M (b + the memory overlay); paired contrasts over start states,
bootstrapped over states (fx_read.boot: B = 2,000, seed 0).  The reference is fx's (fx_ref.py, run on these states by
f6o2_ref.py: servo with the plate pose frozen at t = 0, planner seeds 10-12, and P_min s10-s12; the better system's per-state
mean, as fx_read.load_ref).

Registered verdict: on the NEW main states alone (rows of f6o2_tasks.json states.main), per model, first matching row for F6
(f6o_read.verdict, unchanged):
    NO POWER        s(a) < .50  or  reference < .60
    MEMORY ABSENT   s(d) - s(b) > .10 with CI lower bound > 0,  and  s(b) - s(c) CI upper bound < .10
    MEMORY USED     s(b) - s(c) > .10 with CI lower bound > 0
    PARTIAL         otherwise
and for the fix (arm M):
    NO POWER        when the F6 row is NO POWER
    MEMORY RESTORED s(M) - s(b) > .10 with CI lower bound > 0,  and  s(M)'s CI lower bound >= s(b)'s CI upper bound
    NO VERDICT      otherwise (s(M) - s(b) reported)
F6 HOLDS iff MEMORY ABSENT for both pi0.5 and GR00T; F6 FIX iff MEMORY RESTORED for both.  Reported with no verdict: s(M) - s(d)
(the gap to an unhidden plate), s(M) - s(c), s(a) - s(d), the same F6 rows pooled over f6o's main states 4-23 (results/flawed/f6o/
roll, a-d only) and the new main states, the occluder leak (share of later queries with plate pixels), where failed bowls end,
and the overlay's own diagnostics (segmenter IoU against the true plate at the first query, centre error, painted pixels).

    python experiments/f6o2_read.py --phase smoke|main --roll DIR [--ref DIR] [--models pi05+groot] [--out X.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fx_read as FR                                                       # noqa: E402
import f6o_read as OR                                                      # noqa: E402

META = os.path.join(ROOT, "results", "flawed", "f6o2", "f6o2_tasks.json")
META1 = os.path.join(ROOT, "results", "flawed", "f6o", "f6o_tasks.json")
ROLL1 = os.path.join(ROOT, "results", "flawed", "f6o", "roll")
REF1 = os.path.join(ROOT, "results", "flawed", "fx", "ref")
EPS = FR.EPS
NAMES = ("a", "b", "c", "d", "M")


def ref_systems(ref, tid, phase, states):
    """fx_read.load_ref's per-system per-state means (servo seeds 10-12, P_min s10-s12) -> {kind: array}."""
    kinds = {"servo": [f"servo_{tid}_{phase}{x}.json" for x in ("", "_p11", "_p12")],
             "pmin": [f"pmin_{tid}_{phase}_s{s}.json" for s in (10, 11, 12)]}
    per = {}
    for kind, names in kinds.items():
        rows = []
        for nm in names:
            fn = os.path.join(ref, nm)
            if os.path.exists(fn):
                eps = {int(e["trial"]): bool(e["success"]) for e in json.load(open(fn, encoding="utf-8"))["eps"]}
                rows.append([eps.get(k, np.nan) for k in states])
        if rows:
            per[kind] = np.nanmean(np.array(rows, float), 0)
    return per


def best(per):
    if not per:
        return None
    return float(max(np.nanmean(v) for v in per.values()))


def load_shards(roll, model, t, states, nshard):
    """f6o_read.load over the unsharded file <model>_<tid>.json and the shards <model>-s<j>_<tid>.json."""
    d = {}
    for tag in [model] + [f"{model}-s{j}" for j in range(nshard)]:
        x = OR.load(roll, tag, t, states)
        if x:
            d.update(x)
    return d


def mem_diag(roll, model, t, nshard):
    eps = []
    for tag in [model] + [f"{model}-s{j}" for j in range(nshard)]:
        fn = os.path.join(roll, f"{tag}_{t['tid']}_mem.json")
        if os.path.exists(fn):
            eps += json.load(open(fn, encoding="utf-8"))["eps"]
    if not eps:
        return None
    rec = [e["record"] or {} for e in eps]
    ok = [r for r in rec if r.get("ok")]
    tr = np.array([x for e in eps for x in e["trace"]], float) if any(e["trace"] for e in eps) else np.zeros((0, 3))
    return dict(n=len(eps), record_ok=len(ok), seg_iou_median=float(np.median([r["iou"] for r in ok])) if ok else None,
                seg_iou_min=float(np.min([r["iou"] for r in ok])) if ok else None,
                centre_xy_err_median=float(np.median([r["centre_xy_err"] for r in ok])) if ok else None,
                centre_xy_err_max=float(np.max([r["centre_xy_err"] for r in ok])) if ok else None,
                painted_px_median_agent_wrist=[float(np.median(tr[:, 1])), float(np.median(tr[:, 2]))] if len(tr) else None,
                queries_wrist_painted_share=float(np.mean(tr[:, 2] > 0)) if len(tr) else None)


def cell(D, states, ref, tr_xy, meta, names):
    n = len(states)
    s = {c: FR.col(D[c], states) for c in names}
    mean = lambda x, i: float(np.nanmean(x[i]))                            # noqa: E731
    diff = lambda x, y: FR.boot(lambda i: mean(s[x], i) - mean(s[y], i), n)   # noqa: E731
    row = dict(n=n, success={c: FR.boot(lambda i, c=c: mean(s[c], i), n) for c in names}, ref=ref,
               d_minus_b=diff("d", "b"), b_minus_c=diff("b", "c"), a_minus_d_disc_cost=diff("a", "d"), a_minus_b=diff("a", "b"))
    if "M" in names:
        row.update(M_minus_b=diff("M", "b"), M_minus_d=diff("M", "d"), M_minus_c=diff("M", "c"))
    row["verdict"] = OR.verdict(row["success"]["a"][0], row["d_minus_b"], row["b_minus_c"], ref)
    if "M" in names:
        mb, sM, sb = row["M_minus_b"], row["success"]["M"], row["success"]["b"]
        if row["verdict"] == "NO POWER":
            fv = "NO POWER"
        elif mb[1] is not None and mb[0] > EPS and mb[1] > 0 and sM[1] is not None and sb[2] is not None and sM[1] >= sb[2]:
            fv = "MEMORY RESTORED"
        else:
            fv = "NO VERDICT"
        row["fix_verdict"] = fv
    vis, ends = {}, {}
    for c in names:
        va = [e.get("vis_agent") for e in D[c].values() if e.get("vis_agent") is not None]
        vw = [e.get("vis_wrist") for e in D[c].values() if e.get("vis_wrist") is not None]
        vis[c] = dict(later_queries_agent=float(np.mean(np.concatenate([v[1:] for v in va]))) if va else None,
                      later_queries_wrist=float(np.mean(np.concatenate([v[1:] for v in vw]))) if vw else None,
                      later_queries_any=float(np.mean(np.concatenate([np.maximum(x[1:], y[1:]) for x, y in zip(va, vw)]))) if va else None)
        cnt = dict(n_fail=0, habit=0, near_plate=0, elsewhere=0)
        for st in states:
            e = D[c][st]
            if e["success"] or not e["final"]:
                continue
            cnt["n_fail"] += 1
            bxy, pxy = np.array(e["final"][FR.B1][:2]), np.array(e["final"]["plate_1"][:2])
            if np.linalg.norm(bxy - pxy) <= FR.NEAR:
                cnt["near_plate"] += 1
            elif np.linalg.norm(bxy - tr_xy) <= FR.NEAR:
                cnt["habit"] += 1
            else:
                cnt["elsewhere"] += 1
        ends[c] = cnt
    row.update(plate_visible=vis, failed_bowl_ends=ends, per_state={c: [int(D[c][st]["success"]) for st in states] for c in names})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("smoke", "main"))
    ap.add_argument("--roll", required=True)
    ap.add_argument("--ref", default=os.path.join(ROOT, "results", "flawed", "f6o2", "ref"))
    ap.add_argument("--models", default="pi05+groot")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.out:
        assert not os.path.exists(a.out), f"{a.out} exists"
    meta = json.load(open(META, encoding="utf-8"))
    meta1 = json.load(open(META1, encoding="utf-8"))
    states = meta["states"][a.phase]
    old = meta["states"]["f6o_main"]
    nsh = int(meta["shards"])
    models = a.models.split("+")
    T = {t["key"].split("_")[-1]: t for t in meta["tasks"]}
    T1 = {t["cond"]: t for t in meta1["tasks"] if t["layout"] == "disp"}
    tr_xy = np.array(meta1["plate_training_xy"])
    ref_new = ref_systems(a.ref, 2, a.phase, states)
    ref_old = ref_systems(REF1, 2, "main", old)
    ref_pool = {k: np.concatenate([ref_old[k], ref_new[k]]) for k in ref_new if k in ref_old}
    res = dict(phase=a.phase, states=states, f6o_states=old, models=models, B=FR.B, roll=a.roll,
               ref=dict(new={k: float(np.nanmean(v)) for k, v in ref_new.items()}, pooled={k: float(np.nanmean(v)) for k, v in ref_pool.items()}),
               per_model={})
    for m in models:
        D = {c: load_shards(a.roll, m, T[c], states, nsh) for c in NAMES}
        out = dict(n_loaded={c: len(D[c]) for c in NAMES})
        if any(len(D[c]) < len(states) for c in NAMES):
            out["new"] = dict(verdict="MISSING")
        else:
            out["new"] = cell(D, states, best(ref_new), tr_xy, meta, NAMES)
            out["new"]["memory_overlay"] = mem_diag(a.roll, m, T["M"], nsh)
            if a.phase == "main":
                D1 = {c: OR.load(ROLL1, m, T1[c], old) for c in "abcd"}
                if all(v is not None and len(v) == len(old) for v in D1.values()):
                    P = {c: {**D1[c], **D[c]} for c in "abcd"}
                    out["pooled"] = cell(P, old + states, best(ref_pool), tr_xy, meta, ("a", "b", "c", "d"))
                else:
                    out["pooled"] = dict(verdict="MISSING f6o")
        res["per_model"][m] = out
    core = [m for m in ("pi05", "groot") if m in res["per_model"]]
    v = lambda m, k: res["per_model"][m].get("new", {}).get(k)              # noqa: E731
    res["F6_HOLDS"] = bool(len(core) == 2 and all(v(m, "verdict") == "MEMORY ABSENT" for m in core))
    res["F6_FIX"] = bool(len(core) == 2 and all(v(m, "fix_verdict") == "MEMORY RESTORED" for m in core))
    js = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(js)
    f = lambda x: "-" if x is None else ("[" + ", ".join("-" if y is None else "%+.3f" % y for y in x) + "]" if isinstance(x, list) else "%.3f" % x)   # noqa: E731
    print(f"ref {res['ref']}")
    for m, out in res["per_model"].items():
        for part in ("new", "pooled"):
            r = out.get(part)
            if r is None:
                continue
            if "success" not in r:
                print(f"{m} {part}: {r['verdict']} {out['n_loaded']}")
                continue
            print(f"{m} {part} (n={r['n']}): {r['verdict']} fix {r.get('fix_verdict')} | " +
                  " ".join(f"{c} {f(r['success'][c])}" for c in r["success"]) + f" | ref {f(r['ref'])}")
            print(f"    d-b {f(r['d_minus_b'])} b-c {f(r['b_minus_c'])} a-d {f(r['a_minus_d_disc_cost'])}" +
                  (f" M-b {f(r['M_minus_b'])} M-d {f(r['M_minus_d'])} M-c {f(r['M_minus_c'])}" if "M_minus_b" in r else ""))
            print(f"    plate visible later {json.dumps(r['plate_visible'])}")
            print(f"    failed ends {json.dumps(r['failed_bowl_ends'])}")
            if r.get("memory_overlay"):
                print(f"    overlay {json.dumps(r['memory_overlay'])}")
    print(f"F6 HOLDS (MEMORY ABSENT, new states, pi05 and groot) = {res['F6_HOLDS']}; F6 FIX (MEMORY RESTORED, both) = {res['F6_FIX']}")


if __name__ == "__main__":
    main()

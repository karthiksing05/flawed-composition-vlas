"""F6O read (CPU, any venv with numpy): F6 memory with a physical occluder in every condition (experiments/f6o_tasks.py;
docs/SCAFFOLD_RESULTS.md 45.5).  Per model and layout (fam = the plate on its training spot, disp = displaced): success in
a (visible: both covers parked), b (occluded-late: cover 0 onto the plate after the first query), c (never-visible: cover 0 on
the plate from the start), d (moved-beside: cover 0 moves at b's step to a spot beside the plate, which stays visible); paired
contrasts over start states, bootstrapped over states (fx_read.boot: B = 2,000, seed 0).  The reference is fx's scripted servo /
P_min that reads the plate pose once at t = 0 (fx_ref.py, freeze; fx tid 0 for fam, 2 for disp; the better system's per-state
mean, as fx_read.load_ref): it renders nothing and the covers do not collide, so it is the reference for b on these same states.

Verdict per model and layout, first matching row (mapping fixed before the main runs, m3_prereg f6o):
    NO POWER        s(a) < .50  or  reference (for b) < .60
    MEMORY ABSENT   s(d) - s(b) > .10 with CI lower bound > 0,  and  s(b) - s(c) CI upper bound < .10
    MEMORY USED     s(b) - s(c) > .10 with CI lower bound > 0
    PARTIAL         otherwise
d, not a, is b's comparison: d has b's visual event (a large disc appearing mid-table beside the plate at the same step) without
hiding the plate, so d - b isolates the loss of the plate's view.  disp is the headline layout: in fam the cover on the plate also
hides most of the bowl (render check: bowl agentview px c/a .07-.49), so fam's b and c lose the bowl too; fam is reported under
the same mapping, descriptive only.  MEMORY ABSENT on disp for a model = F6 HOLDS for it; F6 HOLDS (overall) iff MEMORY ABSENT on
disp for both pi0.5 and GR00T.
Reported with no verdict: s(a) - s(d) (the cost of the disc appearing), s(a) - s(b), the share of policy queries at which the
plate had any pixel in agentview / wrist (from the client's track: the occluder's in-rollout check), where the bowl ends in failed
episodes (training spot = habit, the true plate, a cover, elsewhere), and the grasped object.

    python experiments/f6o_read.py --phase smoke|main --roll DIR [--ref DIR] [--models pi05+groot] [--out X.json]
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

META = os.path.join(ROOT, "results", "flawed", "f6o", "f6o_tasks.json")
EPS, NEAR, B1 = FR.EPS, FR.NEAR, FR.B1


def load(roll, model, t, states):
    d = FR.load_task(roll, model, t, states)
    if d is None:
        return None
    fz = os.path.join(roll, f"{model}_{t['tid']}_track.npz")
    if os.path.exists(fz):
        z = np.load(fz)
        for i in range(int(z["n"])):
            k = int(z[f"e{i}_k"])
            if k not in d:
                continue
            px = z[f"e{i}_plate_px"]
            q = px[px[:, 0] >= 0]
            d[k]["q_px"] = q.tolist()
            d[k]["vis_agent"] = (q[:, 0] > 0).astype(float).tolist()
            d[k]["vis_wrist"] = (q[:, 1] > 0).astype(float).tolist()
    return d


CONDS = ("a", "b", "c", "d")


def verdict(sa, db, bc, ref):
    if sa < 0.50 or ref is None or ref < 0.60:
        return "NO POWER"
    if db[1] is not None and db[0] > EPS and db[1] > 0 and bc[2] is not None and bc[2] < EPS:
        return "MEMORY ABSENT"
    if bc[1] is not None and bc[0] > EPS and bc[1] > 0:
        return "MEMORY USED"
    return "PARTIAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("smoke", "main"))
    ap.add_argument("--roll", required=True)
    ap.add_argument("--ref", default=os.path.join(ROOT, "results", "flawed", "fx", "ref"))
    ap.add_argument("--models", default="pi05+groot")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    meta = json.load(open(META, encoding="utf-8"))
    if a.out:
        assert not os.path.exists(a.out), f"{a.out} exists"
    states = meta["states"][a.phase]
    n = len(states)
    models = a.models.split("+")
    tr = np.array(meta["plate_training_xy"])
    T = {(t["layout"], t["cond"]): t for t in meta["tasks"]}
    res = dict(phase=a.phase, states=states, models=models, B=FR.B, per_model={}, ref={})
    for lay in ("fam", "disp"):
        t0 = T[(lay, "a")]
        r = FR.load_ref(a.ref, dict(tid=t0["fx_tid"], key=f"F6O_{lay}"), a.phase, states)
        res["ref"][lay] = None if r is None else dict(mean=float(np.nanmean(r)), by_system=FR.REF_KIND.get(f"F6O_{lay}"))
    for m in models:
        out = {}
        for lay in ("fam", "disp"):
            D = {c: load(a.roll, m, T[(lay, c)], states) for c in CONDS}
            if any(v is None or len(v) < n for v in D.values()):
                out[lay] = dict(verdict="MISSING", n={c: (None if v is None else len(v)) for c, v in D.items()})
                continue
            s = {c: FR.col(D[c], states) for c in D}
            mean = lambda x, i: float(np.nanmean(x[i]))                    # noqa: E731
            ref = res["ref"][lay]["mean"] if res["ref"][lay] else None
            diff = lambda x, y: FR.boot(lambda i: mean(s[x], i) - mean(s[y], i), n)   # noqa: E731
            db, bc = diff("d", "b"), diff("b", "c")
            row = dict(success={c: FR.boot(lambda i, c=c: mean(s[c], i), n) for c in s}, d_minus_b=db, b_minus_c=bc,
                       a_minus_d_disc_cost=diff("a", "d"), a_minus_b=diff("a", "b"), a_minus_c=diff("a", "c"), ref=ref)
            vis, ends, grasp = {}, {}, {}
            for c in CONDS:
                va = [e.get("vis_agent") for e in D[c].values() if e.get("vis_agent") is not None]
                vw = [e.get("vis_wrist") for e in D[c].values() if e.get("vis_wrist") is not None]
                vis[c] = dict(first_query_agent=float(np.mean([v[0] for v in va])) if va else None,
                              later_queries_agent=float(np.mean(np.concatenate([v[1:] for v in va]))) if va else None,
                              later_queries_wrist=float(np.mean(np.concatenate([v[1:] for v in vw]))) if vw else None,
                              later_queries_any=float(np.mean(np.concatenate([np.maximum(x[1:], y[1:]) for x, y in zip(va, vw)]))) if va else None)
                cnt = dict(n_fail=0, habit=0, near_plate=0, on_cover=0, elsewhere=0)
                for st in states:
                    e = D[c][st]
                    if e["success"] or not e["final"]:
                        continue
                    cnt["n_fail"] += 1
                    bxy, pxy = np.array(e["final"][B1][:2]), np.array(e["final"]["plate_1"][:2])
                    parks = np.array(meta["parks"][lay][st])
                    cov = [parks[1]] + ([parks[0]] if c == "a" else [np.array(meta["beside"][lay][st])] if c == "d" else [])  # covers off the plate at the end
                    if np.linalg.norm(bxy - pxy) <= NEAR:
                        cnt["near_plate"] += 1
                    elif np.linalg.norm(bxy - tr) <= NEAR:
                        cnt["habit"] += 1
                    elif min(np.linalg.norm(bxy - q) for q in cov) <= meta["cover"]["radius"]:
                        cnt["on_cover"] += 1
                    else:
                        cnt["elsewhere"] += 1
                ends[c] = cnt
                g = [D[c][st]["grasped"] for st in states]
                grasp[c] = {x: g.count(x) for x in sorted(set(map(str, g)))}
            row.update(plate_visible=vis, failed_bowl_ends=ends, grasped=grasp,
                       per_state={c: [int(D[c][st]["success"]) for st in states] for c in D})
            row["verdict"] = verdict(row["success"]["a"][0], db, bc, ref)
            out[lay] = row
        out["memory_absent_disp"] = out.get("disp", {}).get("verdict") == "MEMORY ABSENT"
        res["per_model"][m] = out
    core = [m for m in ("pi05", "groot") if m in res["per_model"]]
    res["F6_HOLDS"] = bool(len(core) == 2 and all(res["per_model"][m]["memory_absent_disp"] for m in core))
    js = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(js)
    print(f"ref {res['ref']}")
    for m, out in res["per_model"].items():
        for lay in ("fam", "disp"):
            r = out[lay]
            if r["verdict"] == "MISSING":
                print(f"{m} {lay}: MISSING {r['n']}")
                continue
            print(f"{m} {lay}: {r['verdict']} | a {r['success']['a']} b {r['success']['b']} c {r['success']['c']} d {r['success']['d']} | "
                  f"d-b {r['d_minus_b']} b-c {r['b_minus_c']} a-d (disc cost) {r['a_minus_d_disc_cost']} | ref {r['ref']}")
            print(f"    plate visible {r['plate_visible']}")
            print(f"    failed ends {r['failed_bowl_ends']} grasped {r['grasped']}")
        print(f"{m}: MEMORY ABSENT on disp = {out['memory_absent_disp']}")
    print(f"F6 HOLDS (MEMORY ABSENT on disp for pi05 and groot) = {res['F6_HOLDS']}")


if __name__ == "__main__":
    main()

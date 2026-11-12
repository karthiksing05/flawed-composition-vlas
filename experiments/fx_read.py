"""FX diagnostics, the read (CPU, any venv with numpy).  Per task and model: success, the object grasped (the first object lifted
>= LIFT above its start height, from the client's track), and for F6 where the bowl ends.  Contrasts are paired over start states
and bootstrapped over states (B = 2,000 draws, seed 0).  The verdict mappings are fixed in the docstrings of f6() and f7().

    python experiments/fx_read.py --phase smoke|main --roll DIR [--ref DIR] [--models pi05+groot] [--out X.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx_tasks as FT                                                      # noqa: E402

B, LIFT, NEAR, EPS = 2000, 0.03, 0.10, 0.10
B1, B2 = FT.B1, FT.B2


def load_task(roll, model, t, states):
    """-> dict state -> episode record, or None if the run is missing."""
    fj = os.path.join(roll, f"{model}_{t['tid']}.json")
    fz = fj.replace(".json", "_track.npz")
    if not os.path.exists(fj):
        return None
    eps = json.load(open(fj, encoding="utf-8"))["tasks"][str(t["tid"])]["episodes"]
    z = np.load(fz) if os.path.exists(fz) else None
    names = [str(x) for x in z["names"]] if z is not None else []
    out = {}
    for i, e in enumerate(eps):
        r = dict(success=bool(e["success"]), bound=e.get("bound"), error=e.get("error"), grasped=None, final={})
        if z is not None and i < int(z["n"]):
            start, P = z[f"e{i}_start"], z[f"e{i}_pos"]
            r["closed_at_end"] = bool(z[f"e{i}_a6"][-1] > 0)
            r["hidden_steps"] = int(z[f"e{i}_hidden"].sum())
            lift = P[:, :, 2] - start[None, :, 2]
            first = [int(np.argmax(lift[:, j] >= LIFT)) if (lift[:, j] >= LIFT).any() else None for j in range(len(names))]
            cand = [(f, names[j]) for j, f in enumerate(first) if f is not None]
            r["grasped"] = min(cand)[1] if cand else "none"
            r["final"] = {n: P[-1, j].tolist() for j, n in enumerate(names)}
            r["start"] = {n: start[j].tolist() for j, n in enumerate(names)}
        out[int(e["trial"])] = r
    return {k: out[k] for k in states if k in out}


REF_KIND = {}                                                               # task key -> {kind: mean success}


def load_ref(ref, t, phase, states):
    """-> per-state reference success.  Each reference system is averaged over its seeds on disk (servo: planner seeds 10, 11,
    12 = servo_<tid>_<phase>[_p11|_p12].json; P_min: checkpoints s10-s12 = pmin_<tid>_<phase>_s<seed>.json).  The task's
    reference is the system with the higher mean success (registered: a reference counts if either simple system succeeds);
    both means are reported in REF_KIND."""
    if not ref:
        return None
    kinds = {"servo": [f"servo_{t['tid']}_{phase}{x}.json" for x in ("", "_p11", "_p12")],
             "pmin": [f"pmin_{t['tid']}_{phase}_s{s}.json" for s in (10, 11, 12)]}
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
    if not per:
        return None
    REF_KIND[t["key"]] = {k: float(np.nanmean(v)) for k, v in per.items()}
    return max(per.values(), key=lambda v: float(np.nanmean(v)))


def boot(fn, n, seed=0):
    """-> (point, lo, hi): fn(index array) over states, bootstrap percentile 95% CI."""
    rng = np.random.default_rng(seed)
    pt = fn(np.arange(n))
    draws = np.array([fn(rng.integers(0, n, n)) for _ in range(B)], float)
    draws = draws[np.isfinite(draws)]
    if not len(draws) or not np.isfinite(pt):
        return [float(pt), None, None]
    return [float(pt), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def col(recs, states, key="success"):
    return np.array([float(recs[k][key]) if (recs and k in recs) else np.nan for k in states], float)


def f6(data, refs, meta, states, models):
    """F6, per model.  s(.) = success; drop = s(disp_vis) - s(disp_hid); artifact = s(fam_vis) - s(fam_hid); ref = the servo with
    the remembered plate on the displaced states.  First matching row:
      NO POWER           s(disp_vis) < 0.30 (the VLA does not use the visible plate) or ref < 0.80 (criterion 1 unmet)
      HIDING ARTIFACT    artifact > 0.10 and its CI lower bound > 0 (removing the plate breaks the VLA where no memory is needed)
      MEMORY FAILURE     drop > 0.10 and its CI lower bound > 0
      NO MEMORY FAILURE  drop's CI upper bound < 0.10
      INCONCLUSIVE       otherwise
    Reported with no verdict: s(disp_abs), s(disp_hid) - s(disp_abs), and where the bowl ends in failed episodes (within 0.10 m
    of the plate's training spot = habit; of the true plate = near miss; still lifted; gripper commanded closed at the end)."""
    tr = np.array(meta["f6"]["plate_training_xy"])
    T = {t["key"]: t for t in meta["tasks"] if t["diag"] == "F6"}
    out = {}
    ref = refs.get("F6_disp_vis")
    ref_fam = refs.get("F6_fam_vis")
    for m in models:
        D = {k: data.get((m, k)) for k in T}
        if any(v is None for v in D.values()):
            out[m] = dict(verdict="MISSING", missing=[k for k, v in D.items() if v is None])
            continue
        s = {k: col(D[k], states) for k in T}
        n = len(states)
        mean = lambda a, i: float(np.nanmean(a[i]))                         # noqa: E731
        r = dict(success={k: mean(s[k], np.arange(n)) for k in T},
                 drop=boot(lambda i: mean(s["F6_disp_vis"], i) - mean(s["F6_disp_hid"], i), n),
                 artifact=boot(lambda i: mean(s["F6_fam_vis"], i) - mean(s["F6_fam_hid"], i), n),
                 hid_minus_abs=boot(lambda i: mean(s["F6_disp_hid"], i) - mean(s["F6_disp_abs"], i), n),
                 ref_disp=None if ref is None else float(np.nanmean(ref)), ref_fam=None if ref_fam is None else float(np.nanmean(ref_fam)))
        ends = {}
        for k in T:
            c = dict(n_fail=0, habit=0, near_plate=0, elsewhere=0, bowl_lifted_at_end=0, closed_at_end=0)
            for st in states:
                e = D[k].get(st)
                if e is None or e["success"] or not e["final"]:
                    continue
                c["n_fail"] += 1
                c["bowl_lifted_at_end"] += int(e["final"][B1][2] - e["start"][B1][2] >= LIFT)
                c["closed_at_end"] += int(e.get("closed_at_end", False))
                bxy, pxy = np.array(e["final"][B1][:2]), np.array(e["final"]["plate_1"][:2])
                if np.linalg.norm(bxy - tr) <= NEAR:
                    c["habit"] += 1
                elif np.linalg.norm(bxy - pxy) <= NEAR:
                    c["near_plate"] += 1
                else:
                    c["elsewhere"] += 1
            ends[k] = c
        r["failed_bowl_ends"] = ends
        sv, d, a = r["success"]["F6_disp_vis"], r["drop"], r["artifact"]
        if sv < 0.30 or r["ref_disp"] is None or r["ref_disp"] < 0.80:
            v = "NO POWER"
        elif a[1] is not None and a[0] > EPS and a[1] > 0:
            v = "HIDING ARTIFACT"
        elif d[1] is not None and d[0] > EPS and d[1] > 0:
            v = "MEMORY FAILURE"
        elif d[2] is not None and d[2] < EPS:
            v = "NO MEMORY FAILURE"
        else:
            v = "INCONCLUSIVE"
        r["verdict"] = v
        out[m] = r
    return out


def f7(data, refs, meta, states, models):
    """F7, per model and family (SP left / right; NG on / not on the plate).  The two relational tasks (relA, relB) share every
    image; uniA / uniB name the referent uniquely (the other bowl removed).  U = mean success over uniA, uniB; R = mean success over
    relA, relB; delta = U - R.  C = the share of relational episodes that grasped the referent among those that grasped a bowl
    (an instruction-blind policy scores 0.5 by construction); both = the share of states where both relational episodes grasped
    their referent (instruction-blind: 0).  ref = the servo's success on relA, relB given the referent.  First matching row:
      NO POWER            U < 0.50 or ref < 0.60 (floor lowered from .80 after the smoke's references of .67-1.0; SR 45)
      RELATION IGNORED    delta > 0.10 with CI lower bound > 0, and C's CI upper bound < 0.75 (referent at chance or by habit)
      RELATION USED       C's CI lower bound > 0.75
      PARTIAL             otherwise
    Reported with no verdict: amb's choice (which bowl the policy takes unprompted) and the share of relational episodes whose
    grasped bowl sits where amb's did on the same state (habit)."""
    fam = dict(SP=("relL", "relR", "uniL", "uniR"), NG=("pos", "neg", "uniP", "uniT"))
    lay = meta["f7"]["layouts"]
    out = {}
    for m in models:
        out[m] = {}
        for F, (ra, rb, ua, ub) in fam.items():
            ks = [f"F7_{F}_{c}" for c in (ra, rb, ua, ub, "amb")]
            D = {k: data.get((m, k)) for k in ks}
            if any(D[k] is None for k in ks[:4]):
                out[m][F] = dict(verdict="MISSING", missing=[k for k in ks if D[k] is None])
                continue
            n = len(states)
            s = {k: col(D[k], states) for k in ks if D[k] is not None}

            def gr(k, st):
                e = D[k].get(st)
                return None if e is None else e["grasped"]
            hit = np.array([[gr(ks[j], st) == B1 for j in (0, 1)] for st in states], float)
            bowl = np.array([[gr(ks[j], st) in (B1, B2) for j in (0, 1)] for st in states], float)
            mean = lambda a, i: float(np.nanmean(a[i]))                     # noqa: E731
            U = lambda i: (mean(s[ks[2]], i) + mean(s[ks[3]], i)) / 2         # noqa: E731
            R = lambda i: (mean(s[ks[0]], i) + mean(s[ks[1]], i)) / 2         # noqa: E731
            C = lambda i: float(hit[i].sum() / bowl[i].sum()) if bowl[i].sum() else np.nan   # noqa: E731
            ref = [refs.get(k) for k in ks[:2]]
            r = dict(success={k: mean(s[k], np.arange(n)) for k in s}, U=boot(U, n), R=boot(R, n), delta=boot(lambda i: U(i) - R(i), n),
                     C=boot(C, n), both=boot(lambda i: float(np.mean(hit[i].sum(1) == 2)), n),
                     bowl_grasp_rate=float(bowl.mean()),
                     ref=None if any(x is None for x in ref) else float(np.nanmean(np.concatenate(ref))),
                     grasped={k: {str(st): gr(k, st) for st in states} for k in ks if D[k] is not None})
            if D[ks[4]] is not None:
                # where amb's bowl sat at the start, in the referent-bowl1 frame of state ra: bowl_1 = side A (left / on the plate)
                same, tot, sideA = 0, 0, 0
                for st in states:
                    g = gr(ks[4], st)
                    if g not in (B1, B2):
                        continue
                    a_side = g == B1                                         # amb runs on ra's states for SP, rb's for NG
                    if F == "NG":
                        a_side = g == B2                                     # NG amb = state T (bowl_1 on the table)
                    sideA += a_side
                    for j, k in ((0, ks[0]), (1, ks[1])):
                        g2 = gr(k, st)
                        if g2 not in (B1, B2):
                            continue
                        tot += 1
                        side2 = (g2 == B1) if j == 0 else (g2 == B2)        # rb's states swap the bowls
                        same += side2 == a_side
                n_amb = sum(gr(ks[4], st) in (B1, B2) for st in states)
                r["amb_sideA_share"] = float(sideA / n_amb) if n_amb else None
                r["rel_same_side_as_amb"] = float(same / tot) if tot else None
                r["sideA"] = "left (+y)" if F == "SP" else "on the plate"
            if r["ref"] is None or r["ref"] < 0.60 or r["success"][ks[2]] + r["success"][ks[3]] < 1.0:   # ref floor .60 (set after smoke, SR 45)
                v = "NO POWER"
            elif r["delta"][1] is not None and r["delta"][0] > EPS and r["delta"][1] > 0 and r["C"][2] is not None and r["C"][2] < 0.75:
                v = "RELATION IGNORED"
            elif r["C"][1] is not None and r["C"][1] > 0.75:
                v = "RELATION USED"
            else:
                v = "PARTIAL"
            r["verdict"] = v
            out[m][F] = r
        # LIBERO-PRO's negation (no verdict)
        ks = ["F7P_trained", "F7P_notmin", "F7P_notpro"]
        P = {}
        for k in ks:
            d = data.get((m, k))
            if d is None:
                continue
            g = [d[st]["grasped"] for st in states if st in d]
            P[k] = dict(success=float(np.mean([d[st]["success"] for st in states if st in d])),
                        grasped_bowl1=float(np.mean([x == B1 for x in g])) if g else None,
                        grasped_bowl2=float(np.mean([x == B2 for x in g])) if g else None, ref=None if refs.get(k) is None else float(np.nanmean(refs[k])))
        out[m]["PRO"] = P
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("smoke", "main"))
    ap.add_argument("--roll", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--models", default="pi05+groot")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    meta = json.load(open(FT.META, encoding="utf-8"))
    states = meta["states"][a.phase]
    models = a.models.split("+")
    data, refs = {}, {}
    for t in meta["tasks"]:
        for m in models:
            d = load_task(a.roll, m, t, states)
            if d is not None:
                data[(m, t["key"])] = d
        r = load_ref(a.ref, t, a.phase, states)
        if r is not None:
            refs[t["key"]] = r
    res = dict(phase=a.phase, states=states, models=models, B=B,
               per_task={f"{m}:{k}": dict(success=float(np.mean([e["success"] for e in d.values()])) if d else None,
                                          grasped=[d[st]["grasped"] for st in states if st in d], n=len(d))
                         for (m, k), d in sorted(data.items())},
               ref={k: float(np.nanmean(v)) for k, v in refs.items()}, ref_by_system=REF_KIND,
               F6=f6(data, refs, meta, states, models), F7=f7(data, refs, meta, states, models))
    js = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(js)
    for k, v in res["per_task"].items():
        print(f"{k:22s} success {v['success']:.2f} n={v['n']} grasped {v['grasped']}")
    for k, v in res["ref"].items():
        print(f"ref {k:16s} {v:.2f}  by system {REF_KIND.get(k)}")
    for m in models:
        f = res["F6"].get(m, {})
        print(f"F6 {m}: {f.get('verdict')} drop {f.get('drop')} artifact {f.get('artifact')} ref {f.get('ref_disp')}")
        for F in ("SP", "NG"):
            g = res["F7"].get(m, {}).get(F, {})
            print(f"F7 {m} {F}: {g.get('verdict')} U {g.get('U')} R {g.get('R')} C {g.get('C')} both {g.get('both')} ref {g.get('ref')} "
                  f"amb sideA {g.get('amb_sideA_share')} rel=amb {g.get('rel_same_side_as_amb')}")
        print(f"F7P {m}: {res['F7'].get(m, {}).get('PRO')}")


if __name__ == "__main__":
    main()

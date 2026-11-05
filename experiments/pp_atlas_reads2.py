"""Reads C and D on the break atlas (docs/BREAK_ATLAS.md 8; m3_prereg pp_atlas3).  CPU, recorded tracks only.  -> results/flawed/pp_atlas_reads2.json"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_atlas as PA                                                       # noqa: E402

n2 = PA.tasks()
ks = n2["states"]["main"]
tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
LIFT, SLACK, NEAR = 0.01, 0.02, 0.10
STEMS = dict(pi05=["pi05_{tid}"], groot=["groot_{tid}"], pmin=["pmin_s10_{tid}", "pmin_s11_{tid}", "pmin_s12_{tid}"], servo=["servo_s10_{tid}"])
OBJ = dict(cheese=("N1", "N2", "N3", "K6"), bottle=("N4", "N5", "N6", "K4"))


def episodes(stem):
    p = os.path.join(PA.OD, stem)
    R = json.load(open(p + ".json", encoding="utf-8"))
    E = R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]
    return E, PA.load_track(p + "_track.npz")


def dist(r):
    return np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)


def read_c():
    out = {}
    for m in ("pi05", "groot"):
        rows, per = [], {}
        for c in PA.NEW:
            t = tid[(c, "familiar")]
            E, Tr = episodes(f"{m}_{t['tid']}")
            for e, r in zip(E, Tr):
                d = dist(r)
                s, _, last = PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])
                if s != "HANDOFF" or d[-1] > NEAR:
                    continue
                idx = np.flatnonzero(r["held"])
                gap = np.linalg.norm(r["ee"] - r["obj"], axis=1)
                g = float(np.median(gap[idx]))
                z = r["obj"][:, 2] - r["obj"][0, 2]
                step = np.linalg.norm(np.diff(r["obj"][:, :2], axis=0), axis=1)[last:]
                k = np.arange(last + 1, len(d))
                carried = (z[k] >= LIFT) & (r["a6"][k] > 0) & (gap[k] <= g + SLACK)
                tot = float(step.sum())
                share = float(step[carried].sum() / tot) if tot > 0 else 0.0
                x = dict(task=c, carried_share=share, travel=tot, table_share=float(step[z[k] < LIFT].sum() / tot) if tot > 0 else 0.0,
                         lift_at_last=float(z[last]), lift_plus5=float(z[min(last + 5, len(z) - 1)]), lift_plus20=float(z[min(last + 20, len(z) - 1)]),
                         closed_after=float((r["a6"][k] > 0).mean()), near_gap_after=float((gap[k] <= g + SLACK).mean()), last=int(last), final_lift=float(z[-1]), runs=int(1 + np.sum(np.diff(idx) > 1)))
                rows.append(x)
                per.setdefault(c, []).append(x)
        lab = [float(x["carried_share"] > 0.5) for x in rows]
        sh = float(np.mean(lab)) if lab else None
        call = None if sh is None else ("MISSED CARRY" if sh > 2 / 3 else ("REAL DROP" if sh < 1 / 3 else "MIXED"))
        out[m] = dict(n=len(rows), share_missed_carry=sh, call=call,
                      per_task={c: dict(n=len(v), **{k: float(np.median([x[k] for x in v])) for k in v[0] if k != "task"}) for c, v in per.items()})
        print("C", m, json.dumps(out[m]), flush=True)
    return out


def yaw(q):
    w, x, y, z = [float(v) for v in q]
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def by_state(sysname, comps, lay):
    """-> grasp-failure rate, success rate [states] over the tasks and the system's seeds."""
    G, S = np.zeros(len(ks)), np.zeros(len(ks))
    for c in comps:
        t = tid[(c, lay)]
        for st in STEMS[sysname]:
            E, Tr = episodes(st.format(tid=t["tid"]))
            for j, (e, r) in enumerate(zip(E, Tr)):
                s = PA.classify(r["held"], dist(r), bool(e["success"]), e.get("bound"), t["object"])[0]
                G[j] += float(s == "GRASP")
                S[j] += float(s == "success")
    n = len(comps) * len(STEMS[sysname])
    return G / n, S / n


def read_d():
    out = {"objects": {}}
    # the start layout: every task of a layout shares its states, so the K tasks' first track rows give each movable's start
    Tr = {c: episodes(f"servo_s10_{tid[(c, 'new')]['tid']}")[1] for c in ("K1", "K4", "K6")}
    start = {j: dict(bowl=Tr["K1"][j]["obj"][0, :2], plate=Tr["K1"][j]["tgt"][0, :2], bottle=Tr["K4"][j]["obj"][0, :2], cheese=Tr["K6"][j]["obj"][0, :2]) for j in range(len(ks))}
    for name, comps in OBJ.items():
        K = "K6" if name == "cheese" else "K4"
        gs, _ = by_state("servo", comps, "new")
        gp, _ = by_state("pmin", comps, "new")
        L = [j for j in range(len(ks)) if gs[j] >= 0.9 and gp[j] >= 0.9]
        C = [j for j in range(len(ks)) if gs[j] <= 0.1 and gp[j] <= 0.1]
        r = dict(states_L=[ks[j] for j in L], states_C=[ks[j] for j in C], servo_by_state=gs.round(3).tolist(), pmin_by_state=gp.round(3).tolist())
        r["systems"] = {}
        ceil = np.ones(len(ks), bool)
        for m in STEMS:
            g, s = by_state(m, comps, "new")
            ceil &= g >= 1.0
            f = lambda X, I: float(X[I].mean()) if len(I) else None         # noqa: E731
            r["systems"][m] = dict(grasp_fail_L=f(g, L), grasp_fail_C=f(g, C), success_L=f(s, L), success_C=f(s, C))
        r["states_no_system_holds"] = [ks[j] for j in np.flatnonzero(ceil)]
        tr = Tr[K]
        geo = {}
        for nm, I in (("L", L), ("C", C)):
            if I:
                others = [min((float(np.linalg.norm(start[j][name] - start[j][o])), o) for o in start[j] if o != name) for j in I]
                geo[nm] = dict(xy=[np.round(start[j][name], 3).tolist() for j in I], yaw=[round(yaw(tr[j]["quat"][0]), 1) for j in I], nearest=[(round(a, 3), b) for a, b in others],
                               ee_obj_min=float(np.mean([np.linalg.norm(tr[j]["ee"] - tr[j]["obj"], axis=1).min() for j in I])),
                               ee_z_min_above_obj=float(np.mean([(tr[j]["ee"][:, 2] - tr[j]["obj"][:, 2]).min() for j in I])))
        r["geometry"] = geo
        out["objects"][name] = r
    sizes = [(len(v["states_L"]), len(v["states_C"])) for v in out["objects"].values()]
    out["call"] = "OBJECT-STATE LOCKED" if all(a >= 6 and a + b >= 16 for a, b in sizes) else "NOT LOCKED"
    print("D", json.dumps(out), flush=True)
    return out


res = dict(C=read_c(), D=read_d())
json.dump(res, open(os.path.join(PA.OUT, "pp_atlas_reads2.json"), "w", encoding="utf-8"), indent=1)

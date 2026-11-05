"""Checks on the break atlas (docs/BREAK_ATLAS.md 6; m3_prereg pp_atlas2).  CPU, recorded tracks only.  -> results/flawed/pp_atlas_checks.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_atlas as PA                                                       # noqa: E402

n2 = PA.tasks()
ks = n2["states"]["main"]
tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
NEAR = 0.10
STEMS = dict(pi05=["pi05_{tid}"], groot=["groot_{tid}"], pmin=["pmin_s10_{tid}", "pmin_s11_{tid}", "pmin_s12_{tid}"], servo=["servo_s10_{tid}"])
CB = ("N1", "N2", "N3", "N4", "N5", "N6", "K4", "K6")
BOWL = ("K1", "K2", "K3")


def episodes(stem):
    p = os.path.join(PA.OD, stem)
    R = json.load(open(p + ".json", encoding="utf-8"))
    E = R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]
    return E, PA.load_track(p + "_track.npz")


def dist(r):
    return np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)


def check_a():
    out = {}
    for m in ("pi05", "groot"):
        rows = {}
        for c in PA.NEW:
            t = tid[(c, "familiar")]
            E, Tr = episodes(f"{m}_{t['tid']}")
            for e, r in zip(E, Tr):
                d = dist(r)
                s, _, last = PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])
                if s != "HANDOFF":
                    continue
                idx = np.flatnonzero(r["held"])
                rows.setdefault(c, []).append(dict(
                    near=float(d[-1] <= NEAR), d0=float(d[idx[0]]), dmin_held=float(d[idx].min()), dmin_all=float(d.min()), dfinal=float(d[-1]),
                    lift=float(r["obj"][last, 2] - r["obj"][0, 2]), ee_obj=float(np.linalg.norm(r["ee"][last] - r["obj"][last])),
                    opened=float(bool((r["a6"][idx[0]:last + 1] < 0).any())), hold_steps=int(len(idx)), last=int(last), T=int(len(d))))
        allr = [x for v in rows.values() for x in v]
        share = float(np.mean([x["near"] for x in allr])) if allr else None
        call = None if share is None else ("LABEL FAILS" if share > 0.5 else ("LABEL HOLDS" if share < 0.25 else "AMBIGUOUS"))
        out[m] = dict(n=len(allr), share_near=share, call=call,
                      per_task={c: dict(n=len(v), **{k: float(np.median([x[k] for x in v])) for k in v[0]}) for c, v in rows.items()})
        print("A", m, json.dumps(out[m]), flush=True)
    return out


def grasp_by_state(sysname, comp, lay):
    """-> [states] GRASP rate, success rate over the system's seeds."""
    t = tid[(comp, lay)]
    G, S = np.zeros(len(ks)), np.zeros(len(ks))
    for st in STEMS[sysname]:
        E, Tr = episodes(st.format(tid=t["tid"]))
        for j, (e, r) in enumerate(zip(E, Tr)):
            s, _, _ = PA.classify(r["held"], dist(r), bool(e["success"]), e.get("bound"), t["object"])
            G[j] += float(s == "GRASP") / len(STEMS[sysname])
            S[j] += float(s == "success") / len(STEMS[sysname])
    return G, S


def sets(lay):
    gs = np.mean([grasp_by_state("servo", c, lay)[0] for c in CB], 0)
    gp = np.mean([grasp_by_state("pmin", c, lay)[0] for c in CB], 0)
    L = [j for j in range(len(ks)) if gs[j] >= 0.9 and gp[j] >= 0.9]
    C = [j for j in range(len(ks)) if gs[j] <= 0.1 and gp[j] <= 0.1]
    return gs, gp, L, C


def check_b():
    out = {}
    for lay in ("new", "familiar"):
        gs, gp, L, C = sets(lay)
        r = dict(states_L=[ks[j] for j in L], states_C=[ks[j] for j in C], n_L=len(L), n_C=len(C), servo_by_state=gs.tolist(), pmin_by_state=gp.tolist())
        r["call"] = "STATE-LOCKED" if len(L) >= 8 and len(L) + len(C) >= 18 else "NOT STATE-LOCKED"
        r["systems"] = {}
        for m in STEMS:
            r["systems"][m] = {}
            for name, tasks_ in (("cheese_bottle", CB), ("new_pairings", PA.NEW), ("bowl_known", BOWL)):
                A = [grasp_by_state(m, c, lay) for c in tasks_]
                G, S = np.array([a[0] for a in A]), np.array([a[1] for a in A])
                f = lambda X, idx: float(X[:, idx].mean()) if len(idx) else None  # noqa: E731
                r["systems"][m][name] = dict(grasp_L=f(G, L), grasp_C=f(G, C), success_L=f(S, L), success_C=f(S, C))
        geo = {}
        for c in ("K4", "K6", "K1"):
            t = tid[(c, lay)]
            _, Tr = episodes(f"servo_s10_{t['tid']}")
            for nm, idx in (("L", L), ("C", C)):
                if idx:
                    geo[f"{c}_{nm}"] = dict(obj_xy=np.mean([Tr[j]["obj"][0, :2] for j in idx], 0).tolist(), ee_xy=np.mean([Tr[j]["ee"][0, :2] for j in idx], 0).tolist(),
                                            obj_xy_each=[np.round(Tr[j]["obj"][0, :2], 3).tolist() for j in idx],
                                            max_lift=float(np.mean([(Tr[j]["obj"][:, 2] - Tr[j]["obj"][0, 2]).max() for j in idx])),
                                            ee_obj_min=float(np.mean([np.linalg.norm(Tr[j]["ee"] - Tr[j]["obj"], axis=1).min() for j in idx])),
                                            ee_xy_min=float(np.mean([np.linalg.norm((Tr[j]["ee"] - Tr[j]["obj"])[:, :2], axis=1).min() for j in idx])))
        r["servo_geometry"] = geo
        out[lay] = r
        print("B", lay, json.dumps(r), flush=True)
    return out


res = dict(A=check_a(), B=check_b())
json.dump(res, open(os.path.join(PA.OUT, "pp_atlas_checks.json"), "w", encoding="utf-8"), indent=1)

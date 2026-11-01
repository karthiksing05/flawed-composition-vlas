"""FLAWED_COMPOSITION prototype A, stage 1l DIAGNOSTIC (section 5.20; descriptive only): why the cheese's relocated grasp did not improve
with three times the generated support (5.19).  Two sources:
  planner   every generated attempt of 1g and 1k (results/flawed/a1g_gen_K*.json, a1k_gen_K*.json): the four movable objects'
            positions, success, the phase reached.  Per object: yield, failure phase histogram, yield against the object's
            position (x, y terciles of the N2 box) and against its clearance (distance to the nearest other movable object or
            fixture centre, terciles)
  policy    the traced episodes of results/flawed/a1k_trace (fa1k_trace.py; Ak traced in 1l with every entity's start position):
            per (model, object, layout), the miss share (closed without a hold) against the LOCAL PLANNER YIELD at the start
            position (share of successful planner attempts among the 15 nearest attempts of that object's tasks), against the
            object's clearance and its position terciles; the mean grasp error vector (object frame) of missed grasps
-> results/flawed/a1l_diag.json

    python experiments/fa1l_diag.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "flawed")
KNN = 15


def terc(v, edges):
    return int(np.searchsorted(edges, v))


def main():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    mov = n2["movable"]
    fixp = np.array(n2["constraints"]["fixture_xy"])
    att = {}
    for f in sorted(glob.glob(os.path.join(OUT, "a1g_gen_K*.json")) + glob.glob(os.path.join(OUT, "a1k_gen_K*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        oe = FC.roles(lang[d["task"]])[0]
        mi = mov.index(oe)
        for e in d["episodes"]:
            if not e.get("layout"):
                continue
            P = np.array(e["positions"])
            others = np.r_[np.delete(P, mi, 0), fixp]
            clr = float(np.linalg.norm(others - P[mi], axis=1).min())
            a = att.setdefault(oe, dict(xy=[], ok=[], phase=[], clr=[], task=[], src=[]))
            a["xy"].append(P[mi]); a["ok"].append(bool(e["success"])); a["phase"].append(int(e["phase"])); a["clr"].append(clr)
            a["task"].append(d["task"]); a["src"].append(os.path.basename(f)[:3])
    rep = dict(planner={}, policy={})
    edges = {}
    for oe, a in att.items():
        xy, ok, ph, clr = np.array(a["xy"]), np.array(a["ok"]), np.array(a["phase"]), np.array(a["clr"])
        ex, ey, ec = np.percentile(xy[:, 0], [33.3, 66.7]), np.percentile(xy[:, 1], [33.3, 66.7]), np.percentile(clr, [33.3, 66.7])
        edges[oe] = (ex, ey, ec)
        r = dict(attempts=int(len(ok)), yield_=float(ok.mean()),
                 fail_phase={int(p): int(((~ok) & (ph == p)).sum()) for p in np.unique(ph[~ok])} if (~ok).any() else {},
                 yield_by_x=[float(ok[[terc(v, ex) == i for v in xy[:, 0]]].mean()) for i in range(3)],
                 yield_by_y=[float(ok[[terc(v, ey) == i for v in xy[:, 1]]].mean()) for i in range(3)],
                 yield_by_clearance=[float(ok[[terc(v, ec) == i for v in clr]].mean()) for i in range(3)],
                 x_edges=ex.tolist(), y_edges=ey.tolist(), clearance_edges=ec.tolist())
        rep["planner"][oe] = r
        print(f"[fa1l] planner {oe:20s} n={r['attempts']} yield={r['yield_']:.2f} fail_phase={r['fail_phase']} by_x={np.round(r['yield_by_x'], 2).tolist()} "
              f"by_y={np.round(r['yield_by_y'], 2).tolist()} by_clr={np.round(r['yield_by_clearance'], 2).tolist()} clr_edges={np.round(ec, 3).tolist()}", flush=True)
    groups = {}
    for f in sorted(glob.glob(os.path.join(OUT, "a1k_trace", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        oe = d["object"]
        if oe not in att:
            continue
        A = att[oe]
        axy, aok = np.array(A["xy"]), np.array(A["ok"])
        ex, ey, ec = edges[oe]
        for e in d["episodes"]:
            st = e.get("start")
            if st is None:
                continue
            xy = np.array(st["obj"][:2])
            nn = np.argsort(np.linalg.norm(axy - xy, axis=1))[:KNN]
            row = dict(miss=e["hold_class"] == "closed_no_hold", success=e["success"], ly=float(aok[nn].mean()), tx=terc(xy[0], ex), ty=terc(xy[1], ey),
                       gvec=e.get("grasp_ee_obj"))
            if "all" in st:
                P = np.array(st["all"])[:, :2]
                others = np.r_[np.array([P[FC.ENT.index(m)] for m in mov if m != oe]), fixp]
                row["clr"] = float(np.linalg.norm(others - xy, axis=1).min())
                row["tc"] = terc(row["clr"], ec)
            groups.setdefault((d["name"], oe, d["layout"]), []).append(row)
    for (nm, oe, lay), R in sorted(groups.items()):
        miss, ly = np.array([r["miss"] for r in R]), np.array([r["ly"] for r in R])
        lyt = np.percentile(ly, [33.3, 66.7])
        g = dict(n=len(R), miss=float(miss.mean()), success=float(np.mean([r["success"] for r in R])),
                 local_planner_yield_med=float(np.median(ly)),
                 miss_by_local_yield=[float(miss[[terc(v, lyt) == i for v in ly]].mean()) if any(terc(v, lyt) == i for v in ly) else None for i in range(3)],
                 local_yield_edges=lyt.tolist(),
                 miss_by_x=[float(np.mean([r["miss"] for r in R if r["tx"] == i])) if any(r["tx"] == i for r in R) else None for i in range(3)],
                 miss_by_y=[float(np.mean([r["miss"] for r in R if r["ty"] == i])) if any(r["ty"] == i for r in R) else None for i in range(3)])
        if all("clr" in r for r in R):
            g["miss_by_clearance"] = [float(np.mean([r["miss"] for r in R if r["tc"] == i])) if any(r["tc"] == i for r in R) else None for i in range(3)]
            g["clearance_med"] = float(np.median([r["clr"] for r in R]))
        mv = [r["gvec"] for r in R if r["miss"] and r["gvec"] is not None]
        g["miss_grasp_vec_mean"] = np.mean(mv, 0).tolist() if mv else None
        okv = [r["gvec"] for r in R if r["success"] and r["gvec"] is not None]
        g["success_grasp_vec_mean"] = np.mean(okv, 0).tolist() if okv else None
        rep["policy"][f"{nm}|{oe}|{lay}"] = g
        print(f"[fa1l] policy {nm:4s} {oe:20s} {lay:8s} n={g['n']} miss={g['miss']:.2f} S={g['success']:.2f} ly_med={g['local_planner_yield_med']:.2f} "
              f"miss_by_ly={g['miss_by_local_yield']} (edges {np.round(lyt, 2).tolist()}) by_x={g['miss_by_x']} by_y={g['miss_by_y']} "
              f"by_clr={g.get('miss_by_clearance')} clr_med={g.get('clearance_med')} gvec_miss={None if mv is None or not mv else np.round(g['miss_grasp_vec_mean'], 3).tolist()} "
              f"gvec_ok={None if not okv else np.round(g['success_grasp_vec_mean'], 3).tolist()}", flush=True)
    with open(os.path.join(OUT, "a1l_diag.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(rep, knn=KNN, descriptive=True), fh, indent=1)


if __name__ == "__main__":
    main()

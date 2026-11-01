"""FLAWED_COMPOSITION prototype A, stage 1p DIAGNOSTIC (section 5.31; descriptive only): what the policy does with the cheese beside
the wine rack.  From the traced new-layout cheese episodes (results/flawed/a1k_trace, tids 11 and 17) of the chosen models, split by
the evaluation state's cheese-rack clearance (fa1o_cover.py: below 0.10 m = near): the hold classes, the end effector minus the cheese
at the first close (cheese frame, cm) and the step of the first close; the same for the training data's near-rack cheese carries
(gen_lp.npz rows at the first close).  -> results/flawed/a1p_near.json

    python experiments/fa1p_near.py --names Ap1+Ao1
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402
import fa1g_plan as FG                                                      # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "flawed")


def rack_clr(p, n2):
    fix = n2["constraints"]["fixture_xy"][n2["fixtures"].index("wine_rack_1")]
    return float(np.linalg.norm(np.asarray(p) - np.asarray(fix)) - FG.FIXR["wine_rack_1"] - FG.MOVR["cream_cheese_1"])


def med(v):
    return float(np.median(v)) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="Ap1+Ao1")
    a = ap.parse_args()
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    ci = n2["movable"].index("cream_cheese_1")
    out = {}
    for nm in a.names.split("+"):
        g = {}
        for f in sorted(glob.glob(os.path.join(OUT, "a1k_trace", f"{nm}_s*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            if d["name"] != nm or d["tid"] not in (11, 17):
                continue
            for e in d["episodes"]:
                near = rack_clr(n2["positions"]["new"][e["trial"]][ci], n2) < 0.10
                g.setdefault("near" if near else "far", []).append(e)
        for k, E in sorted(g.items()):
            cls = {}
            for e in E:
                c = "success" if e["success"] else e["hold_class"]
                cls[c] = cls.get(c, 0) + 1
            z = [100 * e["grasp_ee_obj"][2] for e in E if "grasp_ee_obj" in e]
            xy = [100 * math.hypot(*e["grasp_ee_obj"][:2]) for e in E if "grasp_ee_obj" in e]
            fc = [e["first_close"] for e in E if "first_close" in e]
            r = dict(n=len(E), classes=cls, z_close_med_cm=med(z), z_close_iqr_cm=[float(np.percentile(z, q)) for q in (25, 75)] if z else None,
                     xy_close_med_cm=med(xy), first_close_med=med(fc))
            out[f"{nm}|{k}"] = r
            print(f"[fa1p] near {nm:4s} {k:4s} {json.dumps(r)}", flush=True)
    G = dict(np.load(os.path.join(FC.DATA, "gen_lp.npz")))
    key = G["task"].astype(np.int64) * 1000 + G["demo"]
    tix = list(FC.KNOWN).index("K6")
    fixw = None
    rows = {"near": [], "far": []}
    rk = FC.ENT.index("wine_rack_1_top_region")
    for kk in np.unique(key[G["task"] == tix]):
        m = np.nonzero(key == kk)[0]
        p0 = G["pos"][m[0]]
        d = float(np.linalg.norm(p0[FC.ENT.index("cream_cheese_1"), :2] - p0[rk, :2]))
        cl = np.nonzero(G["action"][m, 6] > 0)[0]
        if not len(cl):
            continue
        i = m[cl[0]]
        v = G["ee_pos"][i] - G["pos"][i, FC.ENT.index("cream_cheese_1")]
        yaw = float(G["yaw"][i, FC.ENT.index("cream_cheese_1")])
        c, s = math.cos(yaw), math.sin(yaw)
        rows["near" if d < 0.20 else "far"].append([100 * (c * v[0] + s * v[1]), 100 * (-s * v[0] + c * v[1]), 100 * v[2], d])
    for k, R in rows.items():
        R = np.array(R)
        r = dict(n=int(len(R)), z_close_med_cm=med(R[:, 2]) if len(R) else None, xy_close_med_cm=med(np.hypot(R[:, 0], R[:, 1])) if len(R) else None)
        out[f"data|{k}"] = r
        print(f"[fa1p] near data K6 carries, cheese-to-rack-slot xy {'< 0.20' if k == 'near' else '>= 0.20'} m: {json.dumps(r)}", flush=True)
    with open(os.path.join(OUT, "a1p_near.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

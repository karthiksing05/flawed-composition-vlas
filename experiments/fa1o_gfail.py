"""FLAWED_COMPOSITION prototype A, stage 1o DIAGNOSTIC (section 5.29; descriptive only): why the scripted planner cannot grasp the
cheese beside the wine rack.  For the K6 generator attempts of stage 1k (a1k_gen_K6.json: same seeds, same layout sampler) whose
cheese-rack clearance is below --max_clr, and as a control for an equal number above 0.20 m, every grasp OPTION of fa1l_cgen is run
(grasp test only: approach, descent, close, lift; fresh env, no recording into any data file).  Per run, at the first close command:
  ee_obj        end effector minus cheese in the cheese frame (cm): did the descent reach the grasp point?
  pushed        cheese displacement from the start (cm)
  reach_err     distance from the end effector to the descent waypoint (cm)
  held_lift     the planner's own grasp test result
-> results/flawed/a1o_gfail.json

    python experiments/fa1o_gfail.py --max_clr 0.06 --n 12
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402
import fa1g_plan as FG                                                      # noqa: E402
import fa1l_cgen as CG                                                      # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "flawed")


def clr_rack(p, n2):
    fix = n2["constraints"]["fixture_xy"][n2["fixtures"].index("wine_rack_1")]
    return float(np.linalg.norm(np.asarray(p) - np.asarray(fix)) - FG.FIXR["wine_rack_1"] - FG.MOVR["cream_cheese_1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_clr", type=float, default=0.06)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    n2, rl = FG.lang_roles()
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"]["K6"]
    t = next(x for x in n2["tasks"] if x["compound"] == "K6" and x["layout"] == "familiar")
    bddl = os.path.join(FC.ROOT if hasattr(FC, "ROOT") else os.path.dirname(HERE), t["bddl"])
    oe, te = rl["K6"]
    ci = n2["movable"].index("cream_cheese_1")
    gk = json.load(open(os.path.join(OUT, "a1k_gen_K6.json"), encoding="utf-8"))
    near = [e for e in gk["episodes"] if e.get("layout") and clr_rack(e["positions"][ci], n2) < a.max_clr][:a.n]
    far = [e for e in gk["episodes"] if e.get("layout") and clr_rack(e["positions"][ci], n2) > 0.20][:a.n]
    lay_env = FC.make_env(bddl)
    res = []
    for grp, E in (("near", near), ("far", far)):
        for e in E:
            seed = int(e["seed"])
            s0, p = FG.layout(lay_env, n2, np.random.default_rng(seed), seed)
            if s0 is None:
                continue
            for opt in CG.OPTIONS:
                r = CG.episode(bddl, s0, oe, te, kp, opt, seed, record=True, full=False)
                ep = r["ep"]
                act = np.array(ep["action"])
                cl = np.nonzero(act[:, 6] > 0)[0]
                row = dict(group=grp, seed=seed, clr=clr_rack(p[ci], n2), option=list(opt), held_lift=r["held_lift"])
                if len(cl):
                    i = int(cl[0])
                    po, ee, yaw = np.array(ep["objp"][i]), np.array(ep["ee_pos"][i]), float(np.array(ep["yaw"][i])[FC.ENT.index(oe)])
                    v = ee - po
                    c, s = math.cos(yaw), math.sin(yaw)
                    row.update(ee_obj_cm=[100 * (c * v[0] + s * v[1]), 100 * (-s * v[0] + c * v[1]), 100 * v[2]],
                               pushed_cm=100 * float(np.linalg.norm(po - np.array(ep["objp"][0]))), first_close=i)
                res.append(row)
                print(f"[fa1o] gfail {json.dumps({k: (round(v, 2) if isinstance(v, float) else ([round(x, 1) for x in v] if isinstance(v, list) and v and isinstance(v[0], float) else v)) for k, v in row.items()})}", flush=True)
    lay_env.close()
    summ = {}
    for grp in ("near", "far"):
        R = [x for x in res if x["group"] == grp]
        if not R:
            continue
        summ[grp] = dict(runs=len(R), layouts=len({x["seed"] for x in R}), held_lift=float(np.mean([x["held_lift"] for x in R])),
                         any_option=float(np.mean([any(y["held_lift"] for y in R if y["seed"] == s) for s in {x["seed"] for x in R}])),
                         ee_z_med_cm=float(np.median([x["ee_obj_cm"][2] for x in R if "ee_obj_cm" in x])),
                         ee_xy_med_cm=float(np.median([math.hypot(*x["ee_obj_cm"][:2]) for x in R if "ee_obj_cm" in x])),
                         pushed_med_cm=float(np.median([x["pushed_cm"] for x in R if "pushed_cm" in x])))
        print(f"[fa1o] gfail summary {grp}: {json.dumps(summ[grp])}", flush=True)
    with open(os.path.join(OUT, "a1o_gfail.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(max_clr=a.max_clr, runs=res, summary=summ, note="grasp tests only; nothing enters any data file"), fh, indent=1)


if __name__ == "__main__":
    main()

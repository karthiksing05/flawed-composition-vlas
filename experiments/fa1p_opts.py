"""Section 5.31 check (descriptive): which grasp option the generator chose for the near-fixture cheese carries, and whether that choice
is a function of what the policy can see.  The collision-aware generator (fa1l_cgen, fa1p_sgen) tries the OPTIONS in a fixed order
and keeps the first whose simulated grasp test holds: a choice made by trial in the simulator, not by the geometry.  For every K6
attempt of 1l and 1p with a chosen option: the option, the nearest fixture and its bearing from the cheese (layout frame, in 45 deg
sectors), and for each (fixture, sector) cell how concentrated the choice is (share of the most common option).  Also the share of
attempts per cell where the first option, (0, 0), already held.
-> results/flawed/a1p_opts.json

    python experiments/fa1p_opts.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1g_plan as FG                                                      # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "flawed")


def main():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    fix, fixn = n2["constraints"]["fixture_xy"], n2["fixtures"]
    ci = n2["movable"].index("cream_cheese_1")
    files = [os.path.join(OUT, "a1l_cgen_K6.json")] + sorted(glob.glob(os.path.join(OUT, "a1p_sgen", "K6_b*_c*.json")))
    cells, opts_all = {}, {}
    for f in files:
        for e in json.load(open(f, encoding="utf-8"))["episodes"]:
            if not e.get("layout") or "tried" not in e:
                continue
            p = np.asarray(e["positions"][ci])
            c = {fn: float(np.linalg.norm(p - np.asarray(fix[i])) - FG.FIXR[fn] - FG.MOVR["cream_cheese_1"]) for i, fn in enumerate(fixn)}
            near = min(c, key=c.get)
            if c[near] >= 0.08:
                continue
            v = np.asarray(fix[fixn.index(near)]) - p
            sec = int(((math.degrees(math.atan2(v[1], v[0])) + 360 + 22.5) % 360) // 45)
            opt = tuple(e["option"]) if e.get("option") else None
            k = f"{near}|sector{sec}"
            x = cells.setdefault(k, dict(attempts=0, none=0, first_held=0, options={}))
            x["attempts"] += 1
            x["none"] += int(opt is None)
            x["first_held"] += int(bool(e["tried"]) and e["tried"][0]["held_lift"])
            if opt is not None:
                x["options"][str(list(opt))] = x["options"].get(str(list(opt)), 0) + 1
                opts_all[str(list(opt))] = opts_all.get(str(list(opt)), 0) + 1
    for k, x in sorted(cells.items()):
        n = sum(x["options"].values())
        x["top_share"] = (max(x["options"].values()) / n) if n else None
        print(f"[fa1p] opts {k:28s} {json.dumps(x)}", flush=True)
    print(f"[fa1p] opts all near-fixture chosen options: {json.dumps(opts_all)}", flush=True)
    with open(os.path.join(OUT, "a1p_opts.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(cells=cells, options=opts_all), fh, indent=1)


if __name__ == "__main__":
    main()

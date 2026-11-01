"""Section 5.30 check (descriptive): which fixture the stratified K6 (cheese) attempts sat beside, and how many kept carries lie within
0.10 m of the wine rack (the clearance of fa1o_cover.py).  -> stdout and results/flawed/a1p_which.json

    python experiments/fa1p_which.py
"""
import glob
import json
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
    out = {}
    for task, obj in (("K6", "cream_cheese_1"), ("K4", "wine_bottle_1"), ("K5", "wine_bottle_1"), ("K1", "akita_black_bowl_1")):
        ci = n2["movable"].index(obj)
        r = {}
        for f in sorted(glob.glob(os.path.join(OUT, "a1p_sgen", f"{task}_b*_c*.json"))):
            for e in json.load(open(f, encoding="utf-8"))["episodes"]:
                p = np.asarray(e["positions"][ci])
                c = {fn: float(np.linalg.norm(p - np.asarray(fix[i])) - FG.FIXR[fn] - FG.MOVR[obj]) for i, fn in enumerate(fixn)}
                near = min(c, key=c.get)
                x = r.setdefault(near, dict(attempts=0, kept=0))
                x["attempts"] += 1
                x["kept"] += int(bool(e.get("kept")))
                y = r.setdefault("rack_within_0.10", dict(attempts=0, kept=0))
                if c["wine_rack_1"] < 0.10:
                    y["attempts"] += 1
                    y["kept"] += int(bool(e.get("kept")))
        out[task] = r
        print(f"[fa1p] which {task}: {json.dumps(r)}", flush=True)
    with open(os.path.join(OUT, "a1p_which.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

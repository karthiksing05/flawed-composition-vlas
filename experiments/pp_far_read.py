"""PLAN_PARTS round diagnostics (descriptive): summary of pp_far_diag.py's roll files against its demo profile.  Per file: episodes,
holds, successes; the start offset; where the hand first comes within 4 cm of the demos' close height ("low"); the first close's
offset; close calls per episode; the hand's yaw at the hold's onset; split by start offset (inside / beyond the object part's clip
radius r; for the cheese also nearer than its demos' 10th-percentile start).  CPU.  -> results/flawed/pp_far_read.json

    python experiments/pp_far_read.py imd10_K4_new+imd11_K4_new+...
"""
import json
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "flawed")


def med(v):
    v = [x for x in v if x is not None]
    return None if not v else float(np.median(v))


def summ(E, demo):
    return dict(n=len(E), held=int(sum(e["held_step"] is not None for e in E)), success=int(sum(e["success"] for e in E)),
                none_moved=int(sum(e["bound"] is None for e in E)),
                start_dxy=med([e["start"]["dxy"] for e in E if e["start"]]),
                low_dxy=med([e["low"]["dxy"] for e in E if e["low"]]), no_low=int(sum(e["low"] is None for e in E)),
                close_dxy=med([e["first_close"]["dxy"] for e in E if e["first_close"]]),
                close_dz=med([e["first_close"]["dz"] for e in E if e["first_close"]]), no_close=int(sum(e["first_close"] is None for e in E)),
                close_beyond_demo_p90=int(sum(e["first_close"] is not None and e["first_close"]["dxy"] > demo["close_dxy"][2] for e in E)),
                n_close_med=med([e["n_close"] for e in E]),
                yaw_onset=[None if e["yaw_onset"] is None else round(e["yaw_onset"], 1) for e in E if e["held_step"] is not None])


def main():
    demo = json.load(open(os.path.join(OUT, "pp_far_demo.json"), encoding="utf-8"))
    res = dict(demo={k: {q: v[q] for q in ("start_dxy", "start_dz", "close_dxy", "close_dz")} for k, v in demo.items()}, runs={})
    for nm in sys.argv[1].split("+"):
        z = json.load(open(os.path.join(OUT, f"pp_far_{nm}.json"), encoding="utf-8"))
        E, d = z["episodes"], demo[z["task"]]
        r = E[0]["r"]
        out = dict(all=summ(E, d), inside_r=summ([e for e in E if e["start"]["dxy"] <= r], d),
                   beyond_r=summ([e for e in E if e["start"]["dxy"] > r], d), r=r)
        if z["task"] == "K6":
            out["nearer_than_demo_p10"] = summ([e for e in E if e["start"]["dxy"] < d["start_dxy"][0]], d)
            out["at_or_beyond_demo_p10"] = summ([e for e in E if e["start"]["dxy"] >= d["start_dxy"][0]], d)
        res["runs"][nm] = out
        print("[far_read]", nm, json.dumps(out), flush=True)
    json.dump(res, open(os.path.join(OUT, "pp_far_read.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

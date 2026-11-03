"""PLAN_PARTS round diagnostics (descriptive, dev data only): why held episodes do not arrive.  For each pick-and-place compound
and layout of one arm's S2 rollouts, over the episodes that hold the object but never come within the pair's calibrated radius of
the target (fa1g_read.radius), report:
  dropped     the hold ended before the episode did (held_end false)
  segments    median number of hold segments
  min_dxy     median closest held object-to-target distance, against the radius
  onset_d     median horizontal object-to-target distance at the hold's onset (canonical frame, onset_canon.foot)
  onset_d (arrived)  the same over episodes that arrived
CPU.  -> stdout and results/flawed/pp_arrive_<name>.json

    python experiments/pp_arrive_diag.py --dir pp_r6_roll --name P2rt --seeds 10+11+12
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1x_read as RX                                                       # noqa: E402
from fa1g_read import radius                                                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--seeds", required=True)
a = ap.parse_args()
n2 = json.load(open(os.path.join(RX.OUT, "n2_tasks.json"), encoding="utf-8"))
cal = json.load(open(os.path.join(RX.OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
out = {}
med = lambda v: float(np.median(v)) if len(v) else None                      # noqa: E731
for t in n2["tasks"]:
    if int(t["compound"][1:]) > 6:
        continue
    rad = radius(t["object"], t["target"], cal)
    E = []
    for sd in a.seeds.split("+"):
        f = os.path.join(RX.OUT, a.dir, f"{a.name}_s{sd}_main_{t['tid']}.json")
        if os.path.exists(f):
            E += [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
    H = [e for e in E if e["held_step"] is not None]
    if not H:
        continue
    arr = [e for e in H if e["min_dxy_held"] is not None and e["min_dxy_held"] <= rad]
    na = [e for e in H if not (e["min_dxy_held"] is not None and e["min_dxy_held"] <= rad)]
    od = lambda e: float(np.linalg.norm(e["onset_canon"]["foot"][:2])) if e.get("onset_canon") else np.nan   # noqa: E731
    r = dict(radius=rad, n_held=len(H), n_not_arrived=len(na), dropped=float(np.mean([not e["held_end"] for e in na])) if na else None,
             segments=med([e["held_segments"] for e in na]), min_dxy=med([e["min_dxy_held"] for e in na if e["min_dxy_held"] is not None]),
             onset_d=med([od(e) for e in na]), onset_d_arrived=med([od(e) for e in arr]),
             dropped_arrived=float(np.mean([not e["held_end"] for e in arr])) if arr else None,
             onset_z=med([e["onset_canon"]["foot"][2] for e in na if e.get("onset_canon")]),
             onset_z_arrived=med([e["onset_canon"]["foot"][2] for e in arr if e.get("onset_canon")]))
    out[f"{t['compound']}|{t['layout']}"] = r
    print(t["compound"], t["layout"], t["object"], t["target"], json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}),
          flush=True)
json.dump(out, open(os.path.join(RX.OUT, f"pp_arrive_{a.name}.json"), "w", encoding="utf-8"), indent=1)

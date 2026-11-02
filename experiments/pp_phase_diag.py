"""PLAN_PARTS round diagnostics (descriptive, dev data only): per compound and layout, the phase breakdown of one arm's S2 rollouts.
  bound_right  the first object the episode moved is the instruction's object
  hold         the object held at some step
  arrival      held within the pair's calibrated radius of the target (given hold)
  place        success given arrival
  steps_to_hold  median steps until the hold starts
CPU.  -> stdout and results/flawed/pp_phase_diag_<name>.json

    python experiments/pp_phase_diag.py --dir pp_r2_roll --name P2r32 --seeds 10+11+12
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
ap.add_argument("--phase", default="main")
a = ap.parse_args()
n2 = json.load(open(os.path.join(RX.OUT, "n2_tasks.json"), encoding="utf-8"))
cal = json.load(open(os.path.join(RX.OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
out = {}
for t in n2["tasks"]:
    if int(t["compound"][1:]) > 6:
        continue
    rad = radius(t["object"], t["target"], cal)
    E = []
    for sd in a.seeds.split("+"):
        f = os.path.join(RX.OUT, a.dir, f"{a.name}_s{sd}_{a.phase}_{t['tid']}.json")
        if os.path.exists(f):
            E += [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
    if not E:
        continue
    h = np.array([e["held_step"] is not None for e in E])
    ar = np.array([e["min_dxy_held"] is not None and e["min_dxy_held"] <= rad for e in E])
    s = np.array([bool(e["success"]) for e in E])
    br = np.array([e.get("bound") == t["object"] for e in E])
    nb = np.array([e.get("bound") is None for e in E])
    r = dict(n=len(E), s=float(s.mean()), bound_right=float(br.mean()), bound_none=float(nb.mean()), hold=float(h.mean()),
             arrival=float(ar[h].mean()) if h.any() else None, place=float(s[ar].mean()) if ar.any() else None,
             steps_to_hold=float(np.median([e["held_step"] for e in E if e["held_step"] is not None])) if h.any() else None)
    out[f"{t['compound']}|{t['layout']}"] = r
    print(t["compound"], t["layout"], json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(out, open(os.path.join(RX.OUT, f"pp_phase_diag_{a.name}.json"), "w", encoding="utf-8"), indent=1)

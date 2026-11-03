"""PLAN_PARTS round 11 (docs/PARTS_RESULTS.md section 38), reported quantities with no verdict.  From pp_grasp_yaw_roll.py files
results/flawed/pp_gyaw_<arm><seed>_<task>_familiar.json:
- mode-hopping rate: the share of episodes whose largest pre-hold change of the hand's yaw (mod 180 deg) between consecutive
  calls exceeds 20 deg;
- mode delivery: among held episodes, the share whose grasp turn (onset yaw - start yaw, mod 180 deg) lies nearest (circularly,
  on the doubled angle) to the inventory component of the task's own m.  The inventory and m are the P_imd full build's of the
  same seed (pp_train_full_imd_s<seed>.json), for both arms.
CPU.  -> results/flawed/pp_r11_hop.json

    python experiments/pp_hop_read.py --arms imd+tdc --seeds 10+11+12
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
TGT = {"K4": "wooden_cabinet_1_top_side", "K5": "wine_rack_1_top_region"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="imd+tdc")
    ap.add_argument("--seeds", default="10+11+12")
    ap.add_argument("--thresh", type=float, default=20.0)
    a = ap.parse_args()
    res = {}
    for arm in a.arms.split("+"):
        res[arm] = {}
        for task in ("K4", "K5"):
            hops, deliv, n, nh = 0, 0, 0, 0
            for sd in [int(x) for x in a.seeds.split("+")]:
                inv = json.load(open(os.path.join(OUT, f"pp_train_full_imd_s{sd}.json"), encoding="utf-8"))["interface"]
                mu2 = np.radians(2 * np.array(inv["mu_deg"]))
                want = inv["tmode"][TGT[task]]
                E = json.load(open(os.path.join(OUT, f"pp_gyaw_{arm}{sd}_{task}_familiar.json"), encoding="utf-8"))["episodes"]
                for e in E:
                    n += 1
                    hops += int(e["max_pre_hold_yaw_step"] > a.thresh)
                    if e["yaw_at_onset"] is not None:
                        nh += 1
                        turn2 = np.radians(2 * (e["yaw_at_onset"] - e["yaw_start"]))
                        d = np.abs(np.angle(np.exp(1j * (turn2 - mu2))))
                        deliv += int(int(np.argmin(d)) == int(want))
            res[arm][task] = dict(n=n, hop_rate=hops / max(n, 1), held=nh, delivery=deliv / max(nh, 1))
            print("[hop]", arm, task, json.dumps(res[arm][task]), flush=True)
    json.dump(res, open(os.path.join(OUT, "pp_r11_hop.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

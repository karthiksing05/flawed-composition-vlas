"""PLAN_PARTS round diagnostics (descriptive, dev data only): which grasp mode does a closed-loop episode take, and does it predict
the outcome?  fa1h_roll.episode unchanged, model.sample wrapped (pp_close_diag.Wrap).  Per episode: the hand's yaw modulo 180 deg
(pp_grasp_modes.yaw2) at the first call after the hold's onset (the grasp the object part delivered), the episode's
success, and the largest change of that yaw between consecutive calls before the onset.  LIBERO venv, CPU.
-> results/flawed/pp_gyaw_<name>_<task>_<layout>.json

    python experiments/pp_grasp_yaw_roll.py --ckpt pp_full_tdc_s10/P.npz --name tdc10 --task N4 --layout familiar
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import fa1h_roll as RH                                                      # noqa: E402
from pp_close_diag import Wrap                                              # noqa: E402
from pp_grasp_modes import yaw2                                             # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--states", default="+".join(str(k) for k in range(4, 24)))
    a = ap.parse_args()
    torch.set_num_threads(1)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", a.ckpt))
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = {int(x["trial"]): x for x in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    t = next(x for x in n2["tasks"] if x["compound"] == a.task and x["layout"] == a.layout)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    eps = []
    for k in [int(x) for x in a.states.split("+")]:
        w = Wrap(m)
        e = RH.episode(w, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
        y = [float(np.degrees(yaw2(bb["ee_rot"].numpy())[0]) / 2) for bb, _ in w.calls]
        h = [float(bb["held"][0]) > 0.5 for bb, _ in w.calls]
        on = next((i for i, x in enumerate(h) if x), None)
        pre = y[:on] if on is not None else y
        jump = max((abs((np.degrees(np.angle(np.exp(1j * np.radians(2 * (q - p))))) / 2)) for p, q in zip(pre, pre[1:])), default=0.0)
        r = dict(state=k, success=bool(e["success"]), held=e["held_step"] is not None, yaw_at_onset=None if on is None else y[on],
                 max_pre_hold_yaw_step=float(jump), yaw_start=y[0] if y else None)
        eps.append(r)
        print("[gyaw]", a.name, a.task, a.layout, json.dumps(r), flush=True)
    json.dump(dict(task=a.task, layout=a.layout, ckpt=a.ckpt, episodes=eps),
              open(os.path.join(OUT, f"pp_gyaw_{a.name}_{a.task}_{a.layout}.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

"""PLAN_PARTS round diagnostics (descriptive, demo rows only, fitted data never executed): does an object's GRASP depend on the task's
target?  Per known task K1-K6, per demo: the hand's yaw (world; the end effector's x axis projected on the table plane) at the
episode start and at the first close row (not held, chunk's first gripper command > 0), and the turn between them; yaw is taken
modulo 180 deg (a parallel gripper is symmetric under a half turn).  Per task: circular mean and SD of each, and the
distribution of the turn (median, 10th / 90th percentiles, absolute degrees).  The object term is target-blind, so the tasks
of one object share one part: if their grasp yaws differ, the part is trained on a target-dependent mixture.
CPU.  -> results/flawed/pp_grasp_modes.json

    python experiments/pp_grasp_modes.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import pp_p_train as PT                                                     # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")


def yaw2(R):
    """Hand yaw x 2 (radians) from the rotation's x axis; doubling makes the half-turn symmetry a full turn."""
    return 2 * np.arctan2(R[:, 1, 0], R[:, 0, 0])


def cstats(a2):
    z = np.mean(np.exp(1j * a2))
    sd = np.sqrt(-2 * np.log(max(abs(z), 1e-12)))
    return dict(mean_deg=float(np.degrees(np.angle(z)) / 2), circ_sd_deg=float(np.degrees(sd) / 2))


def main():
    import fa1_common as FC
    res = {}
    for task in ("K1", "K2", "K3", "K4", "K5", "K6"):
        b, X, _, _, _, demos = PT.rows(torch.device("cpu"), [task])
        D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
        names = list(FC.KNOWN)
        keep = D["task"] == names.index(task)
        demo = D["demo"][keep]
        R = b["ee_rot"].numpy()
        close = ((b["held"] < 0.5) & (X[:, 0, 6] > 0)).numpy()
        s2, c2 = [], []
        for d in sorted(set(demo.tolist())):
            ii = np.nonzero(demo == d)[0]
            jc = [i for i in ii if close[i]]
            if not jc:
                continue
            s2.append(yaw2(R[ii[:1]])[0])
            c2.append(yaw2(R[jc[:1]])[0])
        s2, c2 = np.array(s2), np.array(c2)
        turn = np.degrees(np.abs(np.angle(np.exp(1j * (c2 - s2))))) / 2
        res[task] = dict(n=len(c2), start=cstats(s2), close=cstats(c2),
                         turn_abs_deg=dict(median=float(np.median(turn)), p10=float(np.percentile(turn, 10)), p90=float(np.percentile(turn, 90))))
        print("[grasp]", task, json.dumps(res[task]), flush=True)
    json.dump(res, open(os.path.join(OUT, "pp_grasp_modes.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

"""Section 5.27 check (descriptive): print the rows around the first close command for a few cheese demos and generated carries
(step, prev_grip, held, commanded dz, commanded gripper, ee z in cm), to read the close-to-lift schedule each source teaches.

    python experiments/fa1o_peek.py --gen gen_lc.npz
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="gen_lc.npz")
    a = ap.parse_args()
    D0 = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db0 = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    G = dict(np.load(os.path.join(FC.DATA, a.gen)))
    n2 = json.load(open(os.path.join(os.path.dirname(HERE), "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    for src, D, held in (("demos", D0, Db0["held"]), ("generated", G, G["held"])):
        task, demo = np.asarray(D["task"]), np.asarray(D["demo"])
        obj = np.array([rl[int(t)][0] for t in task])
        ds = np.unique(demo[obj == "cream_cheese_1"])
        pick = list(ds[:3]) + (list(ds[-3:]) if src == "generated" else [])
        for d in pick:
            ix = np.nonzero(demo == d)[0]
            st = np.asarray(D["step"])[ix]
            print(f"[peek] {src} demo {d} rows {len(ix)} steps sorted {bool(np.all(np.diff(st) > 0))} task {int(task[ix[0]])}", flush=True)
            cl = np.nonzero(np.asarray(D["prev_grip"])[ix] > 0)[0]
            if not len(cl):
                continue
            for k in range(max(0, cl[0] - 3), min(len(ix), cl[0] + 18)):
                i = ix[k]
                print(f"   k {k:3d} step {int(D['step'][i]):3d} prev {float(D['prev_grip'][i]):+.0f} held {float(held[i]):.0f} "
                      f"dz {float(D['action'][i][2]):+.2f} g {float(D['action'][i][6]):+.0f} eez {100 * float(D['ee_pos'][i][2]):6.2f}", flush=True)


if __name__ == "__main__":
    main()

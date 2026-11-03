"""PLAN_PARTS round diagnostics (descriptive, demo data only): how far the human demos turn the hand while carrying, per known task.
Per demo, over its held rows: the largest rotation angle of the end effector relative to its rotation at the hold's onset (deg), and
that angle at the last held row.  Per task: medians and the 90th percentile over demos.  CPU.
-> results/flawed/pp_carry_rot.json

    python experiments/pp_carry_rot.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
from fa1g_train import rot0_rows                                            # noqa: E402


def ang(R):
    return float(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))))


def main():
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    r0 = rot0_rows(D, Db["held"])
    held = Db["held"] > 0.5
    out, KN = {}, list(FC.KNOWN)
    for t in sorted({int(x) for x in D["task"]}):
        mx, last = [], []
        for d in sorted({int(x) for x in D["demo"][D["task"] == t]}):
            m = (D["task"] == t) & (D["demo"] == d) & held
            if not m.any():
                continue
            a = [ang(r0[i].T @ D["ee_rot"][i]) for i in np.where(m)[0]]
            mx.append(max(a))
            last.append(a[-1])
        out[KN[t]] = dict(n=len(mx), max_med=float(np.median(mx)), max_p90=float(np.percentile(mx, 90)),
                                last_med=float(np.median(last)))
        print("[carry]", KN[t], json.dumps({k: round(v, 2) for k, v in out[KN[t]].items()}), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "flawed", "pp_carry_rot.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

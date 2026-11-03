"""PLAN_PARTS round diagnostics (descriptive, demo data only): the target term's horizontal object-to-target offset over carries.
Per known pick-and-place task: per demo, the horizontal distance from the held object to its target at the hold's onset, and over all
held rows the 99th percentile.  Plus the pairing N4 would need in the familiar scene: at each K4 demo's hold onset, the distance from
the bottle to the plate (the plate is in every scene).  CPU.  -> results/flawed/pp_carry_dist.json

    python experiments/pp_carry_dist.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402


def main():
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    KN = list(FC.KNOWN)
    held = Db["held"] > 0.5
    out = {}

    def onsets(t, o, g):
        r = []
        for d in sorted({int(x) for x in D["demo"][D["task"] == t]}):
            m = np.where((D["task"] == t) & (D["demo"] == d) & held)[0]
            if len(m):
                i = m[0]
                r.append(float(np.linalg.norm(D["pos"][i, o, :2] - D["pos"][i, g, :2])))
        return r

    for t, c in enumerate(KN):
        if c in ("K7", "K8"):
            continue
        ob, tg = FC.roles(lang[c])
        o, g = FC.ENT.index(ob), FC.ENT.index(tg)
        m = (D["task"] == t) & held
        dd = np.linalg.norm(D["pos"][m, o, :2] - D["pos"][m, g, :2], axis=1)
        on = onsets(t, o, g)
        out[c] = dict(object=ob, target=tg, onset_med=float(np.median(on)), onset_max=float(np.max(on)), held_p99=float(np.percentile(dd, 99)))
        print("[dist]", c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[c].items()}), flush=True)
    on = onsets(KN.index("K4"), FC.ENT.index("wine_bottle_1"), FC.ENT.index("plate_1"))
    out["N4_from_K4_onsets"] = dict(onset_med=float(np.median(on)), onset_min=float(np.min(on)), onset_max=float(np.max(on)))
    print("[dist] N4 (bottle to plate at K4 hold onsets)", json.dumps({k: round(v, 3) for k, v in out["N4_from_K4_onsets"].items()}), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "flawed", "pp_carry_dist.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

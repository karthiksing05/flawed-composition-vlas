"""PLAN_PARTS round diagnostics (descriptive, demo data only): do the demos approach along straight lines toward the object?
On not-held demo rows of K1-K6 whose horizontal hand-object distance exceeds 0.06 m, the angle (deg) between the commanded
translation (the row's first action, a[0:3]) and the 3D direction from the hand to the object, and the same with only the
horizontal direction plus zero vertical motion (the horizontal-transport hypothesis of section 21).  Also the elevation of
the hand-object offset (deg above horizontal) by distance bin.  CPU.  -> results/flawed/pp_ray_check.json

    python experiments/pp_ray_check.py
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


def angle(u, v):
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def main():
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    KN = list(FC.KNOWN)
    out = {}
    for c in ("K1", "K2", "K3", "K4", "K5", "K6"):
        t = KN.index(c)
        o = FC.ENT.index(FC.roles(lang[c])[0])
        m = (D["task"] == t) & (Db["held"] < 0.5)
        d = D["pos"][m, o] - D["ee_pos"][m]                                     # hand -> object
        a = D["actions"][m, :3] if "actions" in D else D["action"][m, :3]
        h = np.linalg.norm(d[:, :2], axis=1)
        sel = (h > 0.06) & (np.linalg.norm(a, axis=1) > 1e-3)
        d3, a3 = d[sel], a[sel]
        dh = d3.copy()
        dh[:, 2] = 0
        el = np.degrees(np.arctan2(-d3[:, 2], h[sel]))
        bins = {}
        for lo, hi in ((0.06, 0.10), (0.10, 0.15), (0.15, 0.25)):
            b = (h[sel] >= lo) & (h[sel] < hi)
            if b.any():
                bins[f"{lo}-{hi}"] = dict(n=int(b.sum()), elev_med=float(np.median(el[b])),
                                          ang3_med=float(np.median(angle(a3[b], d3[b]))), angh_med=float(np.median(angle(a3[b], dh[b]))))
        out[c] = dict(n=int(sel.sum()), ang3_med=float(np.median(angle(a3, d3))), angh_med=float(np.median(angle(a3, dh))), bins=bins)
        print("[ray]", c, json.dumps(out[c]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "flawed", "pp_ray_check.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

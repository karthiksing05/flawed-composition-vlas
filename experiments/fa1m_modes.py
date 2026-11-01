"""FLAWED_COMPOSITION prototype A, stage 1m DIAGNOSTIC (section 5.25; descriptive): are the training grasps of each object one mode or
several?  Per object and per source (human demos / 1g-1k generated (gen_k) / 1l collision-aware generated (gen_lc rows beyond
gen_k)), at the first close command of every episode: the end effector minus the object in the OBJECT frame (cm), the gripper's
yaw relative to the object's (deg, mod 180) and its tilt from vertical (deg).  Mean and standard deviation per source.
-> results/flawed/a1m_modes.json

    python experiments/fa1m_modes.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "flawed")


def main():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    G = dict(np.load(os.path.join(FC.DATA, "gen_lc.npz")))
    rows = {}
    for src0, X in (("demo", D), ("gen", G)):
        for ti, k in enumerate(FC.KNOWN):
            if k not in lang or ti > 5:
                continue
            oe = FC.roles(lang[k])[0]
            oi = FC.ENT.index(oe)
            m = X["task"] == ti
            for dm in np.unique(X["demo"][m]):
                idx = np.nonzero(m & (X["demo"] == dm))[0]
                idx = idx[np.argsort(X["step"][idx])]
                a = X["action"][idx, 6]
                if not (a > 0).any():
                    continue
                i = idx[int(np.argmax(a > 0))]
                y = float(X["yaw"][i, oi])
                c, s = math.cos(y), math.sin(y)
                v = X["ee_pos"][i] - X["pos"][i, oi]
                off = [c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]]
                R = X["ee_rot"][i]
                gy = math.degrees(math.atan2(R[1, 0], R[0, 0]) - y) % 180.0
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, -R[2, 2]))))
                src = src0 if src0 == "demo" else ("gen_lc" if int(dm) >= 400 else "gen_k")
                rows.setdefault((oe, src), []).append(off + [gy, tilt])
    out = {}
    for (oe, src), r in sorted(rows.items()):
        r = np.array(r)
        out[f"{oe}|{src}"] = dict(n=int(len(r)), off_cm_mean=(r[:, :3].mean(0) * 100).round(2).tolist(),
                                  off_cm_std=(r[:, :3].std(0) * 100).round(2).tolist(), yaw_deg_mean=float(r[:, 3].mean()),
                                  yaw_deg_std=float(r[:, 3].std()), tilt_deg_mean=float(r[:, 4].mean()), tilt_deg_std=float(r[:, 4].std()))
        print(f"[fa1m] modes {oe:20s} {src:7s} {json.dumps(out[f'{oe}|{src}'])}", flush=True)
    with open(os.path.join(OUT, "a1m_modes.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

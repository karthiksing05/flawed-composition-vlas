"""FLAWED_COMPOSITION 5.8 (stage 1d), the bypass measurement made on TRAINING states only, before stage 1d: how much each input of
Ac's target term reveals the HELD OBJECT's identity (condition (B): no input of the place stage may carry the held object's identity
beyond the geometry its role message declares).  On held rows of the pick-and-place demos (K1-K6), per input group (the foot in the
canonical frame; the end effector's tilt; the finger positions; the last command), a nearest-centroid classifier of the held object
(bowl / bottle / cheese), fitted on demos 0-39 and scored on demos 40-49 of every task; chance = the majority share.  Also the per-object
median of each group.  CPU.  -> results/flawed/a1d_proxy.json

    python experiments/fa1d_proxy.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402
import fa1c_common as FK                                                    # noqa: E402

OUT = os.path.join(FC.ROOT, "results", "flawed")


def main():
    D = {k: v for k, v in np.load(os.path.join(FC.DATA, "demos.npz")).items()}
    Db = np.load(os.path.join(FC.DATA, "demos_b.npz"))
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    oi = np.array([FC.ENT.index(rl[t][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in D["task"]])
    zr, cy = FK.canon_rows(D, Db["held"], oi, ti)
    m = (Db["held"] > 0.5) & (D["task"] <= 5)
    idx = np.nonzero(m)[0]
    n = np.arange(len(oi))
    foot = D["pos"][n, oi].copy()
    foot[:, 2] = foot[:, 2] - zr + D["pos"][:, FK.TABLE, 2]
    rel = foot - D["pos"][n, ti]
    c, s = np.cos(cy), np.sin(cy)
    fc = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], -1)
    groups = {"foot": fc, "tilt": D["ee_rot"][:, 2, :], "fingers": D["grip"], "last_cmd": D["prev_grip"][:, None]}
    y = oi[idx]
    tr, te = D["demo"][idx] < 40, D["demo"][idx] >= 40
    classes = np.unique(y)
    chance = float(max(np.mean(y[te] == k) for k in classes))
    res = dict(n_rows=int(len(idx)), chance=chance, groups={})
    for g, X in groups.items():
        Z = X[idx].astype(np.float64)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        Z = (Z - mu) / sd
        cent = np.stack([Z[tr][y[tr] == k].mean(0) for k in classes])
        pred = classes[np.argmin(((Z[te][:, None, :] - cent[None]) ** 2).sum(-1), 1)]
        acc = float(np.mean(pred == y[te]))
        med = {FC.ENT[k]: np.median(X[idx][y == k], 0).round(4).tolist() for k in classes}
        res["groups"][g] = dict(identity_accuracy=acc, median_per_object=med)
        print(f"[fa1d] {g}: held-object identity accuracy {acc:.3f} (chance {chance:.3f}); medians {json.dumps(med)}", flush=True)
    json.dump(res, open(os.path.join(OUT, "a1d_proxy.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

"""FLAWED_COMPOSITION 5.7, the support prediction made BEFORE stage 1c's rollouts: how far each pairing's place phase starts outside the
target term's training support, in stage 1b's input coordinates (the end effector in the target frame; fa1b_diag2.py) and in stage 1c's
canonical coordinates (the foot in the bearing frame, which reduces the onset to the carry distance d and the foot height).
  training  per target: the first hold onset of each demo of the known pick-and-place tasks with that target (demos, exact foot)
  pairings  per pick-and-place task: stage 1b's Ab grasp onsets (a1b_roll, 3 seeds; the object phase is the same kind of controller),
            foot approximated as held_ee + the object's median training grasp offset (horizontal), height = table + the lift rule
  gaps      median over episodes of the distance to the nearest training onset: gap_b (end effector, target frame, 3-d) and gap_c
            (|d - nearest training d|, horizontal carry distance in the canonical frame)
CPU.  -> results/flawed/a1c_support.json

    python experiments/fa1c_support.py
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
    D = np.load(os.path.join(FC.DATA, "demos.npz"))
    Db = np.load(os.path.join(FC.DATA, "demos_b.npz"))
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = {k: FC.roles(lang[k]) for k in FC.KNOWN}
    oi = np.array([FC.ENT.index(rl[list(FC.KNOWN)[t]][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[list(FC.KNOWN)[t]][1]) for t in D["task"]])
    zr, cy = FK.canon_rows(D, Db["held"], oi, ti)
    held = Db["held"] > 0.5
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    first = {}
    for i in np.nonzero(held & np.r_[True, (~held[:-1]) | (key[1:] != key[:-1])])[0]:
        first.setdefault(int(key[i]), i)
    tr_ee, tr_d, offs = {}, {}, {}
    for i in sorted(first.values()):
        k = list(FC.KNOWN)[D["task"][i]]
        offs.setdefault(rl[k][0], []).append(Db["off"][i])
        if int(k[1:]) > 6:
            continue
        t = rl[k][1]
        pt = D["pos"][i, ti[i]]
        tr_ee.setdefault(t, []).append(D["ee_pos"][i] - pt)
        tr_d.setdefault(t, []).append(float(np.linalg.norm(D["pos"][i, oi[i], :2] - pt[:2])))
    off_med = {o: np.median(v, 0) for o, v in offs.items()}
    res = dict(train_d={t: [float(np.min(v)), float(np.median(v)), float(np.max(v))] for t, v in tr_d.items()}, pairs={})
    fam = [t for t in n2["tasks"] if t["layout"] == "familiar" and int(t["compound"][1:]) <= 6]
    for t in sorted(fam, key=lambda t: (t["compound"][0] != "K", int(t["compound"][1:]))):
        o, tg = t["object"], t["target"]
        E, dd = np.array(tr_ee[tg]), np.array(tr_d[tg])
        gb, gc, ds = [], [], []
        for s in (0, 1, 2):
            f = os.path.join(OUT, "a1b_roll", f"Ab_s{s}_main_{t['tid']}.json")
            for e in [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
                if e["held_ee"] is None:
                    continue
                pt = np.array(e["final"][tg])
                rel = np.array(e["held_ee"]) - pt
                gb.append(float(np.linalg.norm(E - rel, axis=1).min()))
                d = float(np.linalg.norm(np.array(e["held_ee"])[:2] + off_med[o][:2] - pt[:2]))
                ds.append(d)
                gc.append(float(np.abs(dd - d).min()))
        res["pairs"][t["compound"]] = dict(object=o, target=tg, n=len(gb), gap_b=float(np.median(gb)), gap_c=float(np.median(gc)),
                                           d_median=float(np.median(ds)))
        print(t["compound"], json.dumps(res["pairs"][t["compound"]]), flush=True)
    print("train_d", json.dumps(res["train_d"]), flush=True)
    json.dump(res, open(os.path.join(OUT, "a1c_support.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

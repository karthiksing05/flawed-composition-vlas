"""FLAWED_COMPOSITION 5.7, descriptive (after the pa1c read; not a test): stage 1c's place failures in the canonical frame.  For Ac (3
seeds) on each pick-and-place task, over episodes that reach a hold:
  onset   the foot's canonical position at the hold's onset (-d along the bearing, lateral, height) and its gap to the training onsets
          for that target (|d - nearest trained d|, from demos as in fa1c_support.py)
  end     placed (s); failures split into still held at the end / released within 3 cm of the target / released farther; for the
          released-far failures, the median end offset of the object from the target in the canonical frame (along the bearing,
          lateral) and the share that end BEYOND the target (overshoot, along > 0) vs short of it (along < 0)
  segs    the share of hold-reaching episodes with more than one hold (a re-grasp, counted against the place by Proposition 3 (iii))
CPU.  -> results/flawed/a1c_diag.json

    python experiments/fa1c_diag.py
"""
import json
import math
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
    held = Db["held"] > 0.5
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    first = {}
    for i in np.nonzero(held & np.r_[True, (~held[:-1]) | (key[1:] != key[:-1])])[0]:
        first.setdefault(int(key[i]), i)
    tr_d = {}
    for i in sorted(first.values()):
        k = list(FC.KNOWN)[D["task"][i]]
        if int(k[1:]) <= 6:
            tr_d.setdefault(rl[k][1], []).append(float(np.linalg.norm(D["pos"][i, oi[i], :2] - D["pos"][i, ti[i], :2])))
    res = {}
    fam = [t for t in n2["tasks"] if t["layout"] == "familiar" and int(t["compound"][1:]) <= 6]
    for t in sorted(fam, key=lambda t: (t["compound"][0] != "K", int(t["compound"][1:]))):
        o, tg = t["object"], t["target"]
        dd = np.array(tr_d[tg])
        rows = []
        for s in (0, 1, 2):
            f = os.path.join(OUT, "a1c_roll", f"Ac_s{s}_main_{t['tid']}.json")
            for e in [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
                oc = e.get("onset_canon")
                if oc is None:
                    continue
                cy = oc["cyaw"]
                c, sn = math.cos(cy), math.sin(cy)
                rel = np.array(e["final"][o]) - np.array(e["final"][tg])
                along, lat = c * rel[0] + sn * rel[1], -sn * rel[0] + c * rel[1]
                d = -oc["foot"][0]
                rows.append(dict(s=bool(e["success"]), held_end=bool(e["held_end"]), segs=int(e["held_segments"]),
                                 dxy=float(np.linalg.norm(rel[:2])), along=float(along), lat=float(lat), d=float(d),
                                 gap=float(np.abs(dd - d).min()), foot=oc["foot"]))
        fail = [x for x in rows if not x["s"]]
        far = [x for x in fail if not x["held_end"] and x["dxy"] > 0.03]
        res[t["compound"]] = dict(
            object=o, target=tg, n_hold=len(rows), placed=float(np.mean([x["s"] for x in rows])) if rows else None,
            onset_d_median=float(np.median([x["d"] for x in rows])), onset_gap_median=float(np.median([x["gap"] for x in rows])),
            onset_lateral_median=float(np.median([x["foot"][1] for x in rows])), onset_height_median=float(np.median([x["foot"][2] for x in rows])),
            multi_hold=float(np.mean([x["segs"] > 1 for x in rows])),
            fail_held_end=float(np.mean([x["held_end"] for x in fail])) if fail else None,
            fail_near=float(np.mean([(not x["held_end"]) and x["dxy"] <= 0.03 for x in fail])) if fail else None,
            fail_far=float(len(far) / len(fail)) if fail else None,
            far_along_median=float(np.median([x["along"] for x in far])) if far else None,
            far_lat_median=float(np.median([x["lat"] for x in far])) if far else None,
            far_overshoot=float(np.mean([x["along"] > 0 for x in far])) if far else None)
        print(t["compound"], json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[t["compound"]].items()}), flush=True)
    json.dump(res, open(os.path.join(OUT, "a1c_diag.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

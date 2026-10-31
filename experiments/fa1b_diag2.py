"""FLAWED_COMPOSITION 5.6, descriptive (after the pa1b read; not a test): how far each new pairing's place phase starts outside the
target term's training support (Proposition 3's (Supp)), split into its two parts.
  start gap   the end-effector position at the grasp onset, in the target's frame: per new pairing (Ab rollouts, 3 seeds, held_ee
              minus the target's end position, which for fixtures is their fixed position) against the training grasp onsets for
              that target (demos: the first held step of each demo of the known tasks with that target); reported as the median
              distance to the nearest training onset
  offset gap  the grasp offset (held object minus end effector at the onset; demos only, per object) of the new held object against
              the offsets trained with that target
CPU.  -> results/flawed/a1b_diag2.json

    python experiments/fa1b_diag2.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(FC.ROOT, "results", "flawed")


def main():
    D = np.load(os.path.join(FC.DATA, "demos.npz"))
    Db = np.load(os.path.join(FC.DATA, "demos_b.npz"))
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = {k: FC.roles(lang[k]) for k in FC.KNOWN}
    held = Db["held"] > 0.5
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    onset = np.nonzero(held & np.r_[True, (~held[:-1]) | (key[1:] != key[:-1])])[0]
    first = {}
    for i in onset:                                                         # the first onset of each demo
        first.setdefault(int(key[i]), i)
    onset = np.array(sorted(first.values()))
    tr_start, tr_off = {}, {}
    for i in onset:
        k = list(FC.KNOWN)[D["task"][i]]
        if int(k[1:]) > 6:
            continue
        o, t = rl[k]
        ti = FC.ENT.index(t)
        tr_start.setdefault(t, []).append(D["ee_pos"][i] - D["pos"][i, ti])
        tr_off.setdefault(t, []).append(Db["off"][i])
    off_obj = {}
    for i in onset:
        k = list(FC.KNOWN)[D["task"][i]]
        off_obj.setdefault(rl[k][0], []).append(Db["off"][i])
    res = dict(train_offset_median={o: np.median(v, 0).round(4).tolist() for o, v in off_obj.items()}, pairs={})
    fam = [t for t in n2["tasks"] if t["layout"] == "familiar" and int(t["compound"][1:]) <= 6]
    for t in sorted(fam, key=lambda t: (t["compound"][0] != "K", int(t["compound"][1:]))):
        o, tg = t["object"], t["target"]
        S = np.array(tr_start[tg])
        gaps = []
        for s in (0, 1, 2):
            f = os.path.join(OUT, "a1b_roll", f"Ab_s{s}_main_{t['tid']}.json")
            for e in [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]:
                if e["held_ee"] is None:
                    continue
                rel = np.array(e["held_ee"]) - np.array(e["final"][tg])
                gaps.append(float(np.linalg.norm(S - rel, axis=1).min()))
        oo = np.median(off_obj[o], 0) if o in off_obj else None
        og = float(np.linalg.norm(np.array(tr_off[tg]) - oo, axis=1).min()) if oo is not None else None
        res["pairs"][t["compound"]] = dict(object=o, target=tg, n=len(gaps), start_gap_median=float(np.median(gaps)) if gaps else None,
                                           start_gap_p90=float(np.percentile(gaps, 90)) if gaps else None, offset_gap=og)
        print(t["compound"], json.dumps(res["pairs"][t["compound"]]), flush=True)
    print("offsets", json.dumps(res["train_offset_median"]), flush=True)
    json.dump(res, open(os.path.join(OUT, "a1b_diag2.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

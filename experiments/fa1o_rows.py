"""FLAWED_COMPOSITION prototype A, stage 1n DIAGNOSTIC (section 5.27; descriptive only): what the OBJECT term is taught between the
first close command and the contact hold.  The object term is trained on every row with held = 0 (the gate), so the rows with the
gripper commanded closed but no contact hold yet are its only teaching about what to do while the fingers close.  Per object and
source (demos, generated), over the training rows (demos.npz + demos_b.npz + the generated file):
  per demo segment from the first close command to the hold onset:
    gap           steps from the first close command to the hold onset
    lift_pre      end-effector rise (cm) during that gap
  over the closed-not-held rows (prev > 0, held = 0):
    share         their share of the object term's rows
    chunk_dz      the summed commanded z action over the row's horizon-8 chunk (scaled action units)
    after_hold    share of those rows that lie AFTER a hold has begun in the same demo (hold lost or flickered)
-> results/flawed/a1o_rows.json

    python experiments/fa1o_rows.py --gen gen_lc.npz
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
OBJ = ("cream_cheese_1", "akita_black_bowl_1", "wine_bottle_1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="gen_lc.npz")
    a = ap.parse_args()
    D0 = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db0 = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    G = dict(np.load(os.path.join(FC.DATA, a.gen)))
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    out = {}
    for src, D, held in (("demos", D0, Db0["held"]), ("generated", G, G["held"])):
        task, prev = np.asarray(D["task"]), np.asarray(D["prev_grip"])
        demo = task.astype(np.int64) * 10000 + np.asarray(D["demo"])          # demo ids repeat across tasks in demos.npz
        ee, act = np.asarray(D["ee_pos"]), np.asarray(D["action"])
        held = np.asarray(held) > 0.5
        obj = np.array([rl[int(t)][0] for t in task])
        for o in OBJ:
            gaps, lifts, dz, after, n_obj, n_cnh = [], [], [], [], 0, 0
            for d in np.unique(demo[obj == o]):
                ix = np.nonzero((demo == d) & (obj == o))[0]
                if not len(ix):
                    continue
                p, h, z = prev[ix], held[ix], ee[ix, 2]
                n_obj += int((~h).sum())
                cl = np.nonzero(p > 0)[0]
                hs = np.nonzero(h)[0]
                if len(cl) and len(hs) and hs[0] >= cl[0]:
                    gaps.append(int(hs[0] - cl[0]))
                    lifts.append(100 * float(z[hs[0]] - z[cl[0]]))
                first_hold = hs[0] if len(hs) else len(ix)
                for k in np.nonzero((p > 0) & ~h)[0]:
                    n_cnh += 1
                    dz.append(float(act[ix[k:k + 8], 2].sum()))
                    after.append(bool(k > first_hold))
            r = dict(n_demos=int(len(np.unique(demo[obj == o]))), n_obj_rows=n_obj, n_closed_not_held=n_cnh,
                     share=(n_cnh / n_obj if n_obj else None),
                     gap_med=(float(np.median(gaps)) if gaps else None), gap_iqr=([float(np.percentile(gaps, q)) for q in (25, 75)] if gaps else None),
                     lift_pre_med=(float(np.median(lifts)) if lifts else None),
                     lift_pre_iqr=([float(np.percentile(lifts, q)) for q in (25, 75)] if lifts else None),
                     chunk_dz_med=(float(np.median(dz)) if dz else None), chunk_dz_pos=(float(np.mean(np.array(dz) > 0.5)) if dz else None),
                     after_hold=(float(np.mean(after)) if after else None))
            out[f"{src}|{o}"] = r
            print(f"[fa1o] rows {src:9s} {o:20s} {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})}", flush=True)
    with open(os.path.join(OUT, f"a1o_rows_{a.gen.split('.')[0]}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

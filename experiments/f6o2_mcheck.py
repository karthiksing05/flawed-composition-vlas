"""F6O2 overlay check for arm M's final occlusion rule (experiments/f6o2_tasks.py, Memory.paint: the gripper-column exclusion),
policy-free, on the states f6o2_tasks.py built (results/flawed/f6o2/f6o2_tasks.json; the states are not redrawn).  Runs
f6o2_tasks.m_check on every smoke + main row and f6o2_tasks.descent_check on the first two smoke rows.  LIBERO venv, EGL, GPU.
-> results/flawed/f6o2/mcheck.json, check_M2.png, check_M2_descent.png (refuses to overwrite)

    python experiments/f6o2_mcheck.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import fx_tasks as FT                                                      # noqa: E402
import f6o_tasks as OT                                                     # noqa: E402
import f6o2_tasks as O2                                                    # noqa: E402

OUTJ = os.path.join(O2.OUT, "mcheck.json")


def main():
    import imageio
    assert not os.path.exists(OUTJ), OUTJ
    meta = json.load(open(O2.META, encoding="utf-8"))
    CV = meta["cover"]
    OT.install(CV["radius"], CV["half_h"], CV["n"])
    S = np.load(os.path.join(O2.ST, f"{O2.SUITE}_0.npy"))
    env = FT.make_env(os.path.join(O2.ROOT, meta["tasks"][0]["bddl"]))
    env.seed(0)
    rows = meta["states"]["smoke"] + meta["states"]["main"]
    per, frame_rows = [], []
    for i, r in enumerate(rows):
        fr = O2.m_check(env, S[r], meta["parks"][r], CV["zc"], frames_at=O2.WAIT + O2.HIDE_AT + 2 * O2.REPLAN if i < 4 else None)
        tot = {c: [sum(x[1] for x in fr["iou"][c]), sum(x[2] for x in fr["iou"][c]), sum(x[3] for x in fr["iou"][c])] for c in O2.CAMS}
        per.append(dict(row=r, record=fr["record"], iou_per_query=fr["iou"],
                        iou={c: (None if not v[1] else v[0] / v[1]) for c, v in tot.items()}, true_px={c: v[2] for c, v in tot.items()}))
        if "img" in fr:
            im = fr["img"]
            frame_rows.append(np.concatenate([im["a"][0], im["b"][0], im["M"][0], im["a"][1], im["b"][1], im["M"][1]], axis=1))
        if i < 6 or i % 20 == 0:
            print(f"[mcheck] row {r}: IoU {per[-1]['iou']} true px {per[-1]['true_px']}", flush=True)
    imageio.imwrite(os.path.join(O2.OUT, "check_M2.png"), np.concatenate(frame_rows, axis=0))
    # wrist IoU only where the true plate is a real target in the wrist (>= 200 px summed over the 4 queries)
    out = dict(rule=dict(mem_h=O2.MEM_H, mem_ee=O2.MEM_EE, mem_ee_dz=O2.MEM_EE_DZ), per_state=per)
    for c in O2.CAMS:
        v = [p["iou"][c] for p in per if p["iou"][c] is not None]
        w = [p["iou"][c] for p in per if p["iou"][c] is not None and p["true_px"][c] >= 200]
        out[c] = dict(median=float(np.median(v)), min=float(np.min(v)), n=len(v),
                      median_true_ge200=float(np.median(w)) if w else None, n_true_ge200=len(w))
    desc, drows = {}, []
    top = CV["zc"] + CV["half_h"]
    for r in meta["states"]["smoke"][:2]:
        d_, im = O2.descent_check(env, S[r], meta["parks"][r], CV["zc"], top)
        desc[str(r)] = d_
        drows += [im[n] for n in ("a", "b", "M")]
        print(f"[mcheck] descent row {r}: M {d_['M']}", flush=True)
    imageio.imwrite(os.path.join(O2.OUT, "check_M2_descent.png"), np.concatenate(drows, axis=0))
    out["descent"] = desc
    env.close()
    with open(OUTJ, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"[mcheck] agentview {out[O2.CAMS[0]]} wrist {out[O2.CAMS[1]]}", flush=True)
    print("[mcheck] done", flush=True)


if __name__ == "__main__":
    main()

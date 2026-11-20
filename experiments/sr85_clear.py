"""SR 85: where in the DEMOS does xy transport of a held object begin, in the channel's own coordinates (state = [dx, dy, dz (m, target minus object), role id])?
Used to fix CLEAR for W3 from the training demos (a population disjoint from every evaluated rollout).  Not a result.
    python experiments/sr85_clear.py --out results/flawed/sr85_clear.json    (refuses to overwrite)
"""
import argparse, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq

LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"


def q(v, ps=(1, 5, 10, 25, 50, 75, 90)):
    return {"p%d" % p: round(float(np.percentile(v, p)), 4) for p in ps} if len(v) else {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite")
    fs = sorted(glob.glob(os.path.join(LR, "data", "chunk-000", "episode_*.parquet")))
    E = []
    for f in fs:
        t = pq.read_table(f).to_pydict()
        st = np.asarray(t["state"], float); ac = np.asarray(t["actions"], float)
        h = np.where(np.isclose(st[:, 3], 2.0))[0]
        if len(h) < 5:
            continue
        rel = st[h, :3]; xy = np.linalg.norm(rel[:, :2], axis=1); c = -rel[:, 2]          # c = object height above the target's reference point (m)
        d0 = xy[0]
        on = np.where(xy <= d0 - 0.03)[0]
        thr = 0.03
        far = xy > 0.02
        E.append(dict(d0=d0, elev=float(rel[0, 2]), c0=float(c[0]), c_onset=float(c[on[0]]) if len(on) else None, c_end=float(c[-1]), cmin_far=float(c[far].min()) if far.any() else None,
                      frac_far_open=float((c[far] >= thr).mean()) if far.any() else None, frac_all_open=float((c >= thr).mean()),
                      n_far=int(far.sum()), first_open_far=int(np.argmax((c >= thr) & far)) if ((c >= thr) & far).any() else None,
                      xy_on_before_open=float(xy[np.argmax(c >= thr)]) if (c >= thr).any() else None))
    res = dict(n_episodes=len(E), n_files=len(fs))
    for name, sel in (("all", lambda e: True), ("target_above_object_>5cm", lambda e: e["elev"] > 0.05), ("target_not_above", lambda e: e["elev"] <= 0.05)):
        S = [e for e in E if sel(e)]
        res[name] = dict(n=len(S), elev=q([e["elev"] for e in S]), c0_at_grasp=q([e["c0"] for e in S]), c_at_xy_onset=q([e["c_onset"] for e in S if e["c_onset"] is not None]),
                         c_at_release=q([e["c_end"] for e in S]), min_c_while_xy_gt2cm=q([e["cmin_far"] for e in S if e["cmin_far"] is not None]),
                         frac_held_frames_far_with_c_ge_3cm=q([e["frac_far_open"] for e in S if e["frac_far_open"] is not None]),
                         frac_eps_with_any_far_frame_c_lt_3cm=float(np.mean([e["cmin_far"] < 0.03 for e in S if e["cmin_far"] is not None])),
                         frac_eps_c_ge_3cm_at_last_held_frame=float(np.mean([e["c_end"] >= 0.03 for e in S])),
                         xy_remaining_when_first_c_ge_3cm=q([e["xy_on_before_open"] for e in S if e["xy_on_before_open"] is not None]))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

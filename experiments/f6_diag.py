"""F6 diagnosis (docs/notes/f6_diag.md): what are the F6_b failures?  Reads REGISTERED tracks only (no writes there), numpy only, via sbatch.
  results/flawed/f6o2/roll/<model>-s<shard>_<tid>_track.npz   tid 0..4 = cond a b c d M, 4 shards x 30 states (120 main states, disp layout)
  results/flawed/rg_comp_roll/pi05_{I_8,I_9,rc_8,mem_8,rgp_8}_track.npz   rg_comp's F6_b / F6_a (60 states)
  results/flawed/fig_frames/F6{,_more}/pi05_{I_8,I_9,rc_8}_track.npz     this session's 20 display rollouts (same harness as rg_comp_roll)
Per episode: held = closed command & the bowl lifted > 2 cm above its start & within 10 cm of the EE; release = last held row.
Output: stdout + results/flawed/fig_frames/f6_diag.json
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "flawed")
BOWL, PLATE = "akita_black_bowl_1", "plate_1"
COVER_R = 0.1135                                   # f6o2_tasks.json "radius"
TRAIN = np.array(json.load(open(os.path.join(RES, "f6o", "f6o_tasks.json"), encoding="utf-8"))["plate_training_xy"])
CONDS = ("a", "b", "c", "d", "M")


def episodes(stem):
    J = json.load(open(stem + ".json", encoding="utf-8"))
    E = [e for t in J["tasks"].values() for e in t["episodes"]]
    z = np.load(stem + "_track.npz")
    names = [str(n) for n in z["names"]]
    assert int(z["n"]) == len(E), (stem, int(z["n"]), len(E))
    out = []
    for i, e in enumerate(E):
        g = lambda k: z["e%d_%s" % (i, k)]
        out.append(dict(trial=int(e["trial"]), success=bool(e["success"]), names=names, pos=g("pos"), start=g("start"), ee=g("ee"), a6=g("a6"),
                        hidden=g("hidden") if "e%d_hidden" % i in z.files else None,
                        px=g("plate_px") if "e%d_plate_px" % i in z.files else None))
    return out


def feat(e):
    n = e["names"]
    ib, ip = n.index(BOWL), n.index(PLATE)
    pos, st, ee, a6 = e["pos"], e["start"], e["ee"], e["a6"]
    T = len(pos)
    bowl, pl0 = pos[:, ib], st[ip]
    held = ((bowl[:, 2] - st[ib, 2]) > 0.02) & (np.linalg.norm(bowl - ee, axis=1) < 0.10) & (a6 > 0)
    h = np.flatnonzero(held)
    d = dict(trial=e["trial"], success=e["success"], T=T, plate_train_d=float(np.linalg.norm(pl0[:2] - TRAIN)),
             held_any=bool(len(h)), n_held=int(len(h)), held_end=bool(held[-1]))
    dend = float(np.linalg.norm(bowl[-1, :2] - pl0[:2]))
    d.update(end_dxy=dend, end_dz=float(bowl[-1, 2] - pl0[2]))
    if len(h):
        on, r = int(h[0]), int(h[-1])
        dall = np.linalg.norm(bowl[:, :2] - pl0[:2], axis=1)
        for k in (5, 10, 20, 40):
            d["dxy_r+%d" % k] = float(dall[min(r + k, T - 1)])
        d["shift10"] = float(np.linalg.norm(bowl[min(r + 10, T - 1), :2] - bowl[r, :2]))
        d["min_dxy_after"] = float(dall[r:].min())
        d["ever_within3_after"] = bool(dall[r:].min() < 0.03)
        d["bowl_dz_r+10"] = float(bowl[min(r + 10, T - 1), 2] - pl0[2])
        d["bowl_dz_maxafter"] = float(bowl[r:, 2].max() - pl0[2])
        d["ee_bowl_r+10"] = float(np.linalg.norm(ee[min(r + 10, T - 1), :2] - bowl[min(r + 10, T - 1), :2]))
        d["open_frac_r+1..5"] = float((a6[r + 1:r + 6] < 0).mean()) if r + 1 < T else None
        d["n_segments"] = int(np.sum(np.diff(np.r_[0, held.astype(int)]) == 1))
        d["dist_end_over_3cm"] = bool(dend > 0.03)
        r3 = max(r - 3, 0)
        d["ee_speed_rel"] = float(np.linalg.norm(ee[r, :2] - ee[r3, :2]) / max(r - r3, 1))
        d["ee_vx_rel"] = float((ee[r, 0] - ee[r3, 0]) / max(r - r3, 1))
        d["ee_vy_rel"] = float((ee[r, 1] - ee[r3, 1]) / max(r - r3, 1))
        d["ee_vz_rel"] = float((ee[r, 2] - ee[r3, 2]) / max(r - r3, 1))
        d["bowl_vx_after"] = float((bowl[min(r + 5, T - 1), 0] - bowl[r, 0]) / 5)
        u = (TRAIN - pl0[:2]) / max(np.linalg.norm(TRAIN - pl0[:2]), 1e-9)          # unit vector plate -> trained plate spot
        pv = np.array([-u[1], u[0]])
        o_rel, o_end = bowl[r, :2] - pl0[:2], bowl[-1, :2] - pl0[:2]
        d.update({"rel_dx": float(o_rel[0]), "rel_dy": float(o_rel[1]), "end_dx": float(o_end[0]), "end_dy": float(o_end[1]), "tr_dx": float((TRAIN - pl0[:2])[0]), "tr_dy": float((TRAIN - pl0[:2])[1]), "rel_proj": float(o_rel @ u), "rel_perp": float(o_rel @ pv), "end_proj": float(o_end @ u), "end_perp": float(o_end @ pv),
                  "drift_proj": float((o_end - o_rel) @ u), "drift_perp": float((o_end - o_rel) @ pv)})
    if len(h):
        on, r = int(h[0]), int(h[-1])
        dh = np.linalg.norm(bowl[h, :2] - pl0[:2], axis=1)
        d.update(onset=on, rel=r, rel_dxy=float(np.linalg.norm(bowl[r, :2] - pl0[:2])), rel_ee_z=float(ee[r, 2] - pl0[2]),
                 rel_bowl_dz=float(bowl[r, 2] - pl0[2]), min_dxy_held=float(dh.min()))
    if e["hidden"] is not None and e["hidden"].any():
        d["cover_step"] = int(np.flatnonzero(e["hidden"])[0])
    if e["px"] is not None:
        q = e["px"][:, 0]
        q = q[q >= 0]
        d["plate_px_later"] = float(q[1:].mean()) if len(q) > 1 else None
        d["plate_px_first"] = float(q[0]) if len(q) else None
    # class
    if d["success"]:
        c = "success"
    elif not d["held_any"]:
        c = "no_grasp"
    elif d["held_end"]:
        c = "held_to_timeout"
    elif d["rel_dxy"] > 0.10:
        c = "released_far(>10cm)"
    elif d["rel_dxy"] > 0.03:
        c = "released_3-10cm"
    else:
        c = "released_within_3cm"
    d["cls"] = c
    return d


def med(v):
    v = [x for x in v if x is not None]
    return None if not v else [round(float(np.percentile(v, p)), 4) for p in (25, 50, 75)]


def summ(F):
    out = dict(n=len(F), success=round(float(np.mean([f["success"] for f in F])), 3))
    cls = {}
    for f in F:
        cls[f["cls"]] = cls.get(f["cls"], 0) + 1
    out["classes"] = cls
    for nm, sel in (("succ", [f for f in F if f["success"]]), ("fail", [f for f in F if not f["success"]])):
        out[nm] = dict(n=len(sel), plate_train_d=med([f["plate_train_d"] for f in sel]), rel_dxy=med([f.get("rel_dxy") for f in sel]),
                       rel_ee_z=med([f.get("rel_ee_z") for f in sel]), rel_bowl_dz=med([f.get("rel_bowl_dz") for f in sel]),
                       end_dxy=med([f["end_dxy"] for f in sel]), end_dz=med([f["end_dz"] for f in sel]), onset=med([f.get("onset") for f in sel]),
                       rel=med([f.get("rel") for f in sel]), T=med([f["T"] for f in sel]), cover_step=med([f.get("cover_step") for f in sel]),
                       plate_px_later=med([f.get("plate_px_later") for f in sel]), min_dxy_held=med([f.get("min_dxy_held") for f in sel]),
                       held_end=int(sum(f["held_end"] for f in sel)),
                       **{k: med([f.get(k) for f in sel]) for k in ("ee_speed_rel", "ee_vx_rel", "ee_vy_rel", "ee_vz_rel", "bowl_vx_after", "rel_dx", "rel_dy", "end_dx", "end_dy", "tr_dx", "tr_dy", "rel_proj", "rel_perp", "end_proj", "end_perp", "drift_proj", "drift_perp", "dxy_r+5", "dxy_r+10", "dxy_r+20", "dxy_r+40", "shift10", "min_dxy_after", "bowl_dz_r+10", "bowl_dz_maxafter", "ee_bowl_r+10", "open_frac_r+1..5", "n_segments")},
                       ever_within3_after=int(sum(bool(f.get("ever_within3_after")) for f in sel)), end_over_3cm=int(sum(bool(f.get("dist_end_over_3cm")) for f in sel)))
    return out


def main():
    R = {}
    for model in ("pi05", "groot"):
        for ci, c in enumerate(CONDS):
            F = []
            for s in range(4):
                stem = os.path.join(RES, "f6o2", "roll", "%s-s%d_%d" % (model, s, ci))
                if os.path.exists(stem + "_track.npz"):
                    F += [feat(e) for e in episodes(stem)]
            if F:
                R["f6o2/%s/%s" % (model, c)] = summ(F)
    for arm, lab in (("I_8", "b"), ("I_9", "a"), ("rc_8", "RC"), ("mem_8", "mem"), ("rgp_8", "rgp")):
        stem = os.path.join(RES, "rg_comp_roll", "pi05_" + arm)
        if os.path.exists(stem + "_track.npz"):
            R["rg_comp_roll/pi05/%s(%s)" % (arm, lab)] = summ([feat(e) for e in episodes(stem)])
    for arm in ("I_8", "I_9", "rc_8"):
        F = []
        for sub in ("F6", "F6_more"):
            stem = os.path.join(RES, "fig_frames", sub, "pi05_" + arm)
            if os.path.exists(stem + "_track.npz"):
                F += [feat(e) for e in episodes(stem)]
        if F:
            R["fig_frames/pi05/%s" % arm] = summ(F)
    for k, v in R.items():
        print("==", k, json.dumps(v), flush=True)
    with open(os.path.join(RES, "fig_frames", "f6_diag.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)


if __name__ == "__main__":
    main()

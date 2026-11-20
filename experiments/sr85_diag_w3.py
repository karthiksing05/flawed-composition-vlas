"""SR 85 DIAGNOSTIC 2 (after the read; not registered): W3 on K5: does the W1 xy servo drag the held object into an elevated target before the expert lifts it clear?
Reads per-step instr npz (experiments/raa_instr.py: obj, tgt, eef, a, contact, succ) of W1 and RAA-sL on K3 (tid 4) and K5 (tid 8), 20 states.
    python experiments/sr85_diag.py --out results/flawed/sr85_diag.json     (refuses to overwrite)
"""
import argparse, glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "flawed", "raa_instr")


def ep(f):
    d = np.load(f, allow_pickle=True)
    obj, tgt, eef, a = (np.asarray(d[k], float) for k in ("obj", "tgt", "eef", "a"))
    con = np.asarray(d["contact"]).reshape(len(obj), -1)[:, 0] > 0.5
    succ = bool(np.asarray(d["succ"]).reshape(-1).any())
    n = len(obj)
    z0 = obj[0, 2]
    dz = obj[:, 2] - tgt[:, 2]                               # object z minus target body-origin z
    lifted = np.where(obj[:, 2] > z0 + 0.02)[0]
    r = dict(n=n, succ=succ, z0_minus_tgtz_cm=100 * (z0 - tgt[0, 2]), tgt_z=tgt[0, 2], obj_z0=z0,
             final_xy_err_cm=100 * float(np.linalg.norm(obj[-1, :2] - tgt[-1, :2])), max_obj_rise_cm=100 * float(obj[:, 2].max() - z0),
             min_dz_cm=100 * float(dz.min()), lifted_step=int(lifted[0]) if len(lifted) else None)
    if len(lifted):
        s0 = lifted[0]
        xy = np.linalg.norm(obj[:, :2] - obj[s0, :2], axis=1)
        mv = np.where((np.arange(n) >= s0) & (xy > 0.03))[0]
        r["xy_moves_step"] = int(mv[0]) if len(mv) else None
        if len(mv):
            k = mv[0]
            r["at_xy_move"] = dict(obj_z_minus_tgt_cm=100 * float(dz[k]), obj_rise_cm=100 * float(obj[k, 2] - z0), eef_z_minus_tgt_cm=100 * float(eef[k, 2] - tgt[k, 2]),
                                   xy_dist_to_tgt_cm=100 * float(np.linalg.norm(obj[k, :2] - tgt[k, :2])), steps_since_lift=int(k - s0))
        dist = np.linalg.norm(obj[:, :2] - tgt[:, :2], axis=1)
        far = dist > 0.05
        r["contact_while_xy_gt5cm"] = int((con & far & (np.arange(n) >= s0)).sum())
        r["steps_obj_below_tgt_z_while_xy_lt15cm"] = int(((dz < 0) & (dist < 0.15) & (np.arange(n) >= s0)).sum())
        # stall: commanded |a_xy| large but object xy speed tiny, after lift
        sp = np.r_[0, np.linalg.norm(np.diff(obj[:, :2], axis=0), axis=1)]
        cmd = np.r_[0, np.linalg.norm(a[1:, :2], axis=1)] if a.ndim == 2 else np.zeros(n)
        st = (np.arange(n) >= s0) & (cmd > 0.5) & (sp < 0.0015) & far
        r["stall_steps"] = int(st.sum())
        r["stall_with_contact"] = int((st & con).sum())
    return r


def med(v):
    v = [x for x in v if x is not None]
    return None if not v else float(np.median(v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite")
    res = {}
    for arm, dd in (("W3", "raasw3K"),):
        for name, tid in (("K5", 8),):
            fs = sorted(glob.glob(os.path.join(D, dd, "instr_raac_%d_t*.npz" % tid)))
            E = [ep(f) for f in fs]
            at = [e["at_xy_move"] for e in E if "at_xy_move" in e]
            res["%s|%s" % (name, arm)] = dict(
                n=len(E), success=sum(e["succ"] for e in E), tgt_above_obj0_cm=med([-e["z0_minus_tgtz_cm"] for e in E]),
                n_lifted=sum(e["lifted_step"] is not None for e in E), n_xy_moved=len(at),
                at_xy_move_obj_z_minus_tgt_cm_med=med([x["obj_z_minus_tgt_cm"] for x in at]), at_xy_move_obj_rise_cm_med=med([x["obj_rise_cm"] for x in at]),
                at_xy_move_obj_below_tgt_z=sum(x["obj_z_minus_tgt_cm"] < 0 for x in at), at_xy_move_obj_below_tgt_plus3=sum(x["obj_z_minus_tgt_cm"] < 3 for x in at),
                at_xy_move_steps_since_lift_med=med([x["steps_since_lift"] for x in at]),
                eps_contact_while_xy_gt5=sum(e.get("contact_while_xy_gt5cm", 0) > 0 for e in E), eps_stall=sum(e.get("stall_steps", 0) >= 5 for e in E),
                eps_stall_with_contact=sum(e.get("stall_with_contact", 0) >= 3 for e in E),
                med_steps_obj_below_tgtz_xy_lt15=med([e.get("steps_obj_below_tgt_z_while_xy_lt15cm") for e in E]), med_final_xy_err_cm=med([e["final_xy_err_cm"] for e in E]),
                med_min_dz_cm=med([e["min_dz_cm"] for e in E]), per_episode=E)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    for k, v in res.items():
        print(k, {x: y for x, y in v.items() if x != "per_episode"})


if __name__ == "__main__":
    main()

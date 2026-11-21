"""RAA-F Route B check S-B1 (docs/notes/raaf_plan.md 4, 9): is the projected crop centre within 8 px of the rendered segmentation?
LIBERO venv, GPU (EGL).  Per task K1-K6: K / E of the fixed agentview camera (saved for the crop builder), then NF frames per task over the
first ND rendered demos (sc_rv render's npz: client-orientation seg, names, slots).  The ground-truth slot position (chan_XX.npz pos, the
demo's own recorded scene) of the OBJECT role (fa1_common.roles) and of the TARGET role is projected with raafb_crop.project_client and
compared with the centroid of that entity's cleaned mask.  Movable roles (bowl, cheese, bottle, plate: body origin vs silhouette centroid)
are the kill set; fixture targets (a site vs the whole fixture's silhouette) are reported separately and are NOT a like-for-like reference.
Also: sc_pu.cam_model / project (the plan's named code) converted to the client orientation must agree with the K / E route.
    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/raafb_proj.py --out DIR
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="K1+K2+K3+K4+K5+K6")
    ap.add_argument("--nd", type=int, default=5)
    ap.add_argument("--nf", type=int, default=20)
    ap.add_argument("--min_px", type=int, default=20)
    a = ap.parse_args()
    import fa1_common as FC
    import sc_pu as PU
    import sc_rv as RV
    from f0b_firstframes_seg import CU                                       # robosuite.utils.camera_utils inlined (it imports h5py, absent from the LIBERO venv)
    import raafb_crop as CR
    os.makedirs(a.out, exist_ok=True)
    by_c, _ = RV.roles_of()
    allrows, KE = [], {}
    for c in a.tasks.split("+"):
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(bddl_file_name=RV._bddl(c), camera_heights=RV.SIZE, camera_widths=RV.SIZE)
        env.seed(0)
        env.reset()
        sim = env.env.sim
        K = CU.get_camera_intrinsic_matrix(sim, "agentview", 256, 256)
        E = CU.get_camera_extrinsic_matrix(sim, "agentview")
        f, Rc, cp = PU.cam_model(sim, "agentview")
        KE["K_" + c], KE["E_" + c] = K, E
        Et = RV.entity_tables(env)
        raw = np.load(RV.raw_path(RV.RV_ROOT, c))
        files = RV.demo_files(RV.RV_ROOT, c)[:a.nd]
        obj, tgt = by_c[c]
        for fpath in files:
            z = np.load(fpath)
            j = int(z["demo"])
            ch = np.load(RV.chan_path(RV.RV_ROOT, c, j))
            n = len(z["actions"])
            st_raw = raw["states_%d" % j]
            kept = np.asarray(z["kept"])
            pos = np.asarray(ch["pos"], np.float64)
            for role, slot in (("object", obj), ("target", tgt)):
                si = FC.ENT.index(slot)
                ei = RV.slot_index(z["names"], z["slot_names"], z["slot_ents"], slot)
                if ei is None:
                    continue
                for t in np.unique(np.linspace(0, n - 1, a.nf).astype(int)):
                    m = RV.clean_mask(np.asarray(z["seg"][t]) == ei)
                    ys, xs = np.nonzero(m)
                    if len(ys) < a.min_px:
                        continue
                    ref = np.array([ys.mean(), xs.mean()])
                    env.set_init_state(st_raw[kept[t]])                         # geometry reference: the mean of the entity's visible geoms (3D)
                    gs = [g for g in np.flatnonzero(Et["geom_ent"] == ei) if g not in set(Et["hide"].tolist())]
                    gm = np.mean(np.asarray(env.env.sim.data.geom_xpos)[gs], 0) if gs else pos[t, si]
                    pg = CR.project_client(gm, K, E)
                    pr = CR.project_client(pos[t, si], K, E)
                    r2, c2 = PU.project(pos[t, si], f, Rc, cp)                # upright convention -> client: col mirrored
                    pr2 = np.array([r2, (RV.SIZE - 1) - c2])
                    rr, cc = int(round(pr[0])), int(round(pr[1]))
                    inside = 0 <= rr < 256 and 0 <= cc < 256
                    allrows.append(dict(task=c, demo=j, t=int(t), role=role, slot=slot, movable=slot in FC.MOVABLE, npx=int(len(ys)),
                                        err=float(np.linalg.norm(pr - ref)), err_dr=float(pr[0] - ref[0]), err_dc=float(pr[1] - ref[1]),
                                        hit=bool(inside and m[rr, cc]), pu_vs_ke=float(np.linalg.norm(pr - pr2)),
                                        proj=pr.tolist(), ref=ref.tolist(), err_geom=float(np.linalg.norm(pg - ref)),
                                        off3=(gm - pos[t, si]).tolist(), pos3=pos[t, si].tolist()))
        env.close()
        print("[sb1] %s done, rows so far %d" % (c, len(allrows)), flush=True)
    np.savez(os.path.join(a.out, "camKE.npz"), **KE)

    # per-slot constant world lift = median (entity geom centre - slot position): the body origin sits at the object's base, the silhouette centroid
    # does not.  err_lift = the projection of (slot position + that constant) against the same centroid.  Diagnostic only (wiring it in would be
    # a constant per object in raafb_rv.py and raafb_crop.client_uv).
    LIFT = {}
    for sl in set(r["slot"] for r in allrows):
        LIFT[sl] = np.median(np.array([r["off3"] for r in allrows if r["slot"] == sl]), 0)
    for r in allrows:
        pl = CR.project_client(np.asarray(r["pos3"]) + LIFT[r["slot"]], K=KE["K_" + r["task"]], E=KE["E_" + r["task"]])
        r["err_lift"] = float(np.linalg.norm(pl - np.asarray(r["ref"])))

    def st(rows, name):
        if not rows:
            return dict(name=name, n=0)
        e = np.array([r["err"] for r in rows])
        g = np.array([r["err_geom"] for r in rows])
        lf = np.array([r["err_lift"] for r in rows])
        return dict(name=name, n=len(rows), median=float(np.median(e)), p90=float(np.percentile(e, 90)), mean=float(e.mean()), max=float(e.max()),
                    geom_median=float(np.median(g)), geom_p90=float(np.percentile(g, 90)), lift_median=float(np.median(lf)),
                    lift_p90=float(np.percentile(lf, 90)),
                    hit=float(np.mean([r["hit"] for r in rows])), mean_dr=float(np.mean([r["err_dr"] for r in rows])),
                    mean_dc=float(np.mean([r["err_dc"] for r in rows])), pu_vs_ke_max=float(max(r["pu_vs_ke"] for r in rows)))
    S = [st(allrows, "all"), st([r for r in allrows if r["movable"]], "movable"), st([r for r in allrows if not r["movable"]], "fixture")]
    for role in ("object", "target"):
        S.append(st([r for r in allrows if r["movable"] and r["role"] == role], "movable_" + role))
    for sl in sorted(set(r["slot"] for r in allrows)):
        S.append(st([r for r in allrows if r["slot"] == sl], "slot:" + sl))
    json.dump(dict(summary=S, rows=allrows), open(os.path.join(a.out, "proj_check.json"), "w"), indent=1)
    for s in S:
        print("[sb1] " + json.dumps(s), flush=True)
    mv = S[1]
    print("[sb1] KILL (median > 8 px on movable roles): %s" % ("YES" if mv.get("n", 0) and mv["median"] > 8.0 else "no"), flush=True)


if __name__ == "__main__":
    main()

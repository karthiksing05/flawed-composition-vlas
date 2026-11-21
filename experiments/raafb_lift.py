"""RAA-F Route B: fixed per-class 3D offsets (body origin -> mean of the entity's visible geom centres, world axes) read from the simulator's own
geoms at the task's reset state (no demo data, nothing fitted to the projection check).  LIBERO venv.  -> OUT/lift.json.  Fixtures keep the site (offset 0)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))


def main():
    import fa1_common as FC
    import sc_rv as RV
    out = sys.argv[1]
    res, per = {}, {}
    for c in ("K1", "K2", "K3", "K4", "K5", "K6"):
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(bddl_file_name=RV._bddl(c), camera_heights=256, camera_widths=256)
        env.seed(0); env.reset()
        E = RV.entity_tables(env)
        sim = env.env.sim
        hide = set(E["hide"].tolist())
        for slot in FC.MOVABLE:
            if slot not in E["names"]:
                continue
            ei = E["names"].index(slot)
            gs = [g for g in np.flatnonzero(E["geom_ent"] == ei) if g not in hide]
            gm = np.mean(np.asarray(sim.data.geom_xpos)[gs], 0)
            body = np.asarray(sim.data.body_xpos[env.env.obj_body_id[slot]], np.float64)
            per.setdefault(slot, {})[c] = (gm - body).tolist()
        env.close()
    for slot, d in per.items():
        v = np.array(list(d.values()))
        res[slot] = np.median(v, 0).tolist()
        print("[lift] %s per task %s spread max %.4f m" % (slot, {k: np.round(x, 4).tolist() for k, x in d.items()}, float(np.abs(v - np.median(v, 0)).max())), flush=True)
    json.dump(dict(lift=res, per_task=per, note="gm - body origin at reset, world axes, metres; fixtures 0"), open(out, "w"), indent=1)
    print("[lift] " + json.dumps(res), flush=True)


main()

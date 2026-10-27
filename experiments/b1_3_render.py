"""PLAN_BINDER B1.3 (docs/PLAN_BINDER.md section 13): render the generated layouts (results/bind/b1_3/plan.json, b1_3_plan.py).
LIBERO env (python 3.8).  No policy, no demo, no rollout: each layout's BDDL is written, the env is built, seeded, reset, and given
the client's 10 settle steps (s1_render.frame), then the policy-orientation agentview image (224) + simulator segmentation (S1's
format) + camera K / E + every body's and region site's world position are stored.  Simulator data are used for SCORING and for
the scripted grasp / release referents (b1_3_units.py) only.

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/b1_3_render.py --shard i/n [--only id,id]
-> results/bind/b1_3/frames/<set>/<id>.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
B13 = os.path.join(ROOT, "results", "bind", "b1_3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="")
    ap.add_argument("--smoke", type=int, default=0)      # N random layouts (one per family in turn) -> frames/smoke; backend check only
    a = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import f0b_firstframes_seg as F0
    import libero_client as LC
    import s1_render as SR
    import b1_3_gen as G
    W = SR._where()
    if a.smoke:
        import random
        rng = random.Random(5)
        lays = []
        while len(lays) < a.smoke:
            L = G.make_layout(rng, list(G.FAMILIES)[len(lays) % 4])
            if L:
                L.update(set="smoke", id=f"sm{len(lays):02d}", seed=len(lays))
                lays.append(L)
    else:
        plan = json.load(open(os.path.join(B13, "plan.json"), encoding="utf-8"))
        lays = plan["layouts"]
    if a.only:
        keep = set(a.only.split(":"))
        lays = [L for L in lays if L["id"] in keep]
    i, n = map(int, a.shard.split("/"))
    lays = lays[i::n]
    os.makedirs(os.path.join(B13, "bddl"), exist_ok=True)
    fails = {}
    for L in lays:
        od = os.path.join(B13, "frames", L["set"])
        os.makedirs(od, exist_ok=True)
        p = os.path.join(od, L["id"] + ".npz")
        if os.path.exists(p):
            continue
        t0 = time.time()
        bp = os.path.join(B13, "bddl", L["id"] + ".bddl")
        with open(bp, "w", encoding="utf-8") as fh:
            fh.write(G.bddl_text(L))
        try:
            env = OffScreenRenderEnv(bddl_file_name=bp, camera_heights=256, camera_widths=256)
            env.seed(int(L.get("seed", 0)))
            env.reset()
            sim = env.env.sim
            fr = SR.frame(env, sim, W, LC)
            K = F0.CU.get_camera_intrinsic_matrix(sim, "agentview", 256, 256)
            E = F0.CU.get_camera_extrinsic_matrix(sim, "agentview")
            inner = env.env
            pos = {nm: np.asarray(sim.data.body_xpos[inner.obj_body_id[nm]], float).tolist() for nm in inner.obj_body_id}
            sites = {str(nm): xyz.tolist() for nm, xyz in zip(fr["site_names"], fr["site_pos"])}
            meta = dict(L, pos=pos, sites=sites)
            env.close()
        except Exception as e:                                                       # noqa: BLE001
            fails[L["id"]] = repr(e)[:300]
            print(f"[b1.3 render] {L['id']} FAILED: {e!r}\n{traceback.format_exc()[-800:]}", flush=True)
            try:
                env.close()
            except Exception:                                                        # noqa: BLE001
                pass
            continue
        np.savez_compressed(p + ".tmp.npz", tid=L["id"], k=0, language="", K=K, E=E, meta=json.dumps(meta), **fr)
        os.replace(p + ".tmp.npz", p)
        print(f"[b1.3 render] {L['set']} {L['id']} {L['family']}: {len(L['objects'])} objects, {len(L['fixtures'])} fixtures, "
              f"{len(fr['site_names'])} sites, {time.time() - t0:.0f}s", flush=True)
    if fails:
        json.dump(fails, open(os.path.join(B13, f"render_fails_{i}_{n}.json"), "w", encoding="utf-8"), indent=1)
    print(f"[b1.3 render] shard {a.shard} done; {len(fails)} failures", flush=True)


if __name__ == "__main__":
    main()

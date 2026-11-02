"""PLAN_PARTS round diagnostics (descriptive, dev data only): trace where the end effector goes in rollouts of one checkpoint.
fa1h_roll.episode, unchanged, with the simulator's step wrapped to record, per step, the end effector, the instruction object's
position and the gripper command.  Summary per episode: the closest end-effector-to-object distance (xy, z) and when it happens,
the distance at step 0, whether the gripper closed within 3 cm of the object, and the object's lift.  LIBERO venv, CPU.
-> results/flawed/pp_trace_<name>.json

    python experiments/pp_trace.py --ckpt pp_full_af_s10/P.npz --name af10 --tids 0+1+10+11 --states 4+5+6+7+8+9
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import fa1h_roll as RH                                                      # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
REC = {}


class Rec:
    def __init__(self, env, oe):
        self._e, self._oe = env, oe

    def __getattr__(self, k):
        return getattr(self._e, k)

    def step(self, a):
        r = self._e.step(a)
        sim = self._e.env.sim
        REC.setdefault("ee", []).append(np.asarray(r[0]["robot0_eef_pos"]).tolist())
        REC.setdefault("obj", []).append(np.asarray(sim.data.body_xpos[self._e.env.obj_body_id[self._oe]]).tolist())
        REC.setdefault("g", []).append(float(a[6]))
        return r

    def set_init_state(self, s):
        return self._e.set_init_state(s)

    def reset(self):
        return self._e.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tids", required=True)
    ap.add_argument("--states", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, meta = load_any(os.path.join(FC.DATA, "ckpt", a.ckpt))
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = {int(b["trial"]): b for b in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    orig = FC.fresh_env
    res = []
    for tid in [int(x) for x in a.tids.split("+")]:
        t = n2["tasks"][tid]
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        oe = t["object"]
        for k in [int(x) for x in a.states.split("+")]:
            REC.clear()
            FC.fresh_env = lambda bddl, seed=7: Rec(orig(bddl, seed), oe)        # noqa: E731
            try:
                e = RH.episode(model, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
            finally:
                FC.fresh_env = orig
            ee, ob, g = np.array(REC["ee"]), np.array(REC["obj"]), np.array(REC["g"])
            ee, ob, g = ee[10:], ob[10:], g[10:]                                   # after the 10 settle steps
            dxy = np.linalg.norm(ee[:, :2] - ob[:, :2], axis=1)
            dz = ee[:, 2] - ob[:, 2]
            i = int(np.argmin(dxy + np.abs(dz)))
            closed_near = bool(((g > 0) & (dxy < 0.03)).any())
            res.append(dict(tid=tid, compound=t["compound"], layout=t["layout"], state=k, success=e["success"], held=e["held_step"] is not None,
                            d0_xy=float(dxy[0]), min_dxy=float(dxy.min()), dz_at_min=float(dz[i]), step_min=i, closed_near=closed_near,
                            first_close=int(np.argmax(g > 0)) if (g > 0).any() else None, dxy_at_first_close=float(dxy[int(np.argmax(g > 0))]) if (g > 0).any() else None,
                            lift=float(ob[:, 2].max() - ob[0, 2]), ee_start=ee[0].tolist(), obj_start=ob[0].tolist(),
                            ee_min_xy=ee[i].tolist()))
            r = res[-1]
            print(f"[trace] {a.name} {t['key']} s{k}: succ={r['success']} held={r['held']} d0={r['d0_xy']:.3f} min_dxy={r['min_dxy']:.3f} "
                  f"dz={r['dz_at_min']:.3f} @{r['step_min']} close@{r['first_close']} dxy@close={r['dxy_at_first_close']} lift={r['lift']:.3f}", flush=True)
    json.dump(res, open(os.path.join(OUT, f"pp_trace_{a.name}.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

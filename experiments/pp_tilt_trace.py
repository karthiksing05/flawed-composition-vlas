"""PLAN_PARTS round diagnostics (descriptive, dev data only): where a held object's tilt comes from.
fa1h_roll.episode, unchanged, with the simulator's step wrapped to record, per step, the end effector's position and rotation, the
instruction object's and the target's positions and rotations, and the commanded action.  Per episode, from the lift (the object
first 2 cm above its start):
  max_obj_tilt     the object's largest tilt (body z vs world z, deg) after the lift
  onset            first step after the lift with tilt > 30 deg (None: never), relative to the lift and to the first gripper open
  ee_turn          the end effector's rotation angle since the lift (deg), at the onset (or at its maximum when no onset)
  slip             the change of the object's rotation relative to the end effector since the lift (deg), at the same step
  cmd_rot          the accumulated commanded rotation |sum a[3:6]| from the lift to that step
  h_above_target   the object's height above the target's origin at the onset; d_target_xy its horizontal distance
ee_turn >> slip: the hand turns the object; slip >> ee_turn: the object pivots in (or falls from) the grasp, e.g. on contact.
LIBERO venv, CPU.  -> results/flawed/pp_tilt_<name>.json

    python experiments/pp_tilt_trace.py --ckpt pp_full_rc_s10/P.npz --name rc10_N4 --tids 22 --states 4+5+6
"""
import argparse
import json
import math
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


def mat_wxyz(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ang(R):
    return float(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))))


class Rec:
    def __init__(self, env, oe, te):
        self._e, self._oe, self._te = env, oe, te

    def __getattr__(self, k):
        return getattr(self._e, k)

    def step(self, a):
        r = self._e.step(a)
        sim, E = self._e.env.sim, self._e.env
        bo = E.obj_body_id[self._oe]
        REC.setdefault("ee", []).append(np.asarray(r[0]["robot0_eef_pos"]).tolist())
        q = np.asarray(r[0]["robot0_eef_quat"])                               # robosuite: (x, y, z, w)
        REC.setdefault("eeq", []).append([q[3], q[0], q[1], q[2]])
        REC.setdefault("obj", []).append(np.asarray(sim.data.body_xpos[bo]).tolist())
        REC.setdefault("objq", []).append(np.asarray(sim.data.body_xquat[bo]).tolist())
        bt = E.obj_body_id.get(self._te) if hasattr(E.obj_body_id, "get") else None
        REC.setdefault("tgt", []).append(np.asarray(sim.data.body_xpos[bt]).tolist() if bt is not None else [np.nan] * 3)
        REC.setdefault("a", []).append(np.asarray(a, dtype=float).tolist())
        return r

    def set_init_state(self, s):
        return self._e.set_init_state(s)

    def reset(self):
        return self._e.reset()


def summarise(e):
    ee, eeq, ob, obq, tg, A = (np.array(REC[k][10:]) for k in ("ee", "eeq", "obj", "objq", "tgt", "a"))
    Ree = [mat_wxyz(q) for q in eeq]
    Rob = [mat_wxyz(q) for q in obq]
    tilt = np.array([math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))) for R in Rob])
    up = np.where(ob[:, 2] > ob[0, 2] + 0.02)[0]
    out = dict(success=e["success"], held=e["held_step"] is not None, tilt0=float(tilt[0]), lifted=bool(len(up)))
    if not len(up):
        return out
    L = int(up[0])
    post = tilt[L:]
    on = np.where(post > 30.0)[0]
    j = L + int(on[0]) if len(on) else L + int(np.argmax(post))
    opn = np.where(A[L:, 6] < 0)[0]
    rel0 = Ree[L].T @ Rob[L]
    out.update(lift_step=L, max_obj_tilt=float(post.max()), tilt_at_lift=float(tilt[L]),
               onset=(int(on[0]) if len(on) else None), first_open=(int(opn[0]) if len(opn) else None),
               at=j - L, ee_turn=ang(Ree[L].T @ Ree[j]), slip=ang(rel0.T @ (Ree[j].T @ Rob[j])),
               cmd_rot=float(np.linalg.norm(A[L:j + 1, 3:6].sum(0))),
               h_above_target=float(ob[j, 2] - tg[j, 2]), d_target_xy=float(np.linalg.norm(ob[j, :2] - tg[j, :2])),
               ee_turn_max=float(max(ang(Ree[L].T @ R) for R in Ree[L:])),
               slip_max=float(max(ang(rel0.T @ (Ree[i].T @ Rob[i])) for i in range(L, len(Rob)))))
    return out


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
        for k in [int(x) for x in a.states.split("+")]:
            REC.clear()
            FC.fresh_env = lambda bddl, seed=7: Rec(orig(bddl, seed), t["object"], t["target"])   # noqa: E731
            try:
                e = RH.episode(model, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
            finally:
                FC.fresh_env = orig
            r = dict(tid=tid, key=t["key"], state=k, **summarise(e))
            res.append(r)
            print("[tilt]", a.name, json.dumps({x: (round(v, 3) if isinstance(v, float) else v) for x, v in r.items()}), flush=True)
    json.dump(res, open(os.path.join(OUT, f"pp_tilt_{a.name}.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

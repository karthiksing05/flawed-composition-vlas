"""Place diagnostic T4 (docs/PLACE_DIAG.md 1.3; m3_prereg pp_place): the drop map, the same state with no hand.  No policy, no checkpoint.
From a settled N2 state the object is put upright (its starting orientation), at rest, with its foot at (d, bearing, h) from the target's
origin, then 60 steps of zero motion with an open gripper, recording LIBERO's own success check.  A drop where the robot touches the object is
marked excluded.  LIBERO venv, CPU.  -> results/flawed/pp_place_drop/<compound>_<layout>.json

    python experiments/pp_place_drop.py --task K1:new
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import fa1b_common as FB                                                    # noqa: E402
import fa1c_common as FK                                                    # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
STATES, DS, NB, STEPS, WAIT = (4, 5, 6, 7), (0.0, 0.01, 0.02, 0.03, 0.04), 8, 60, 10
HS = {"plate_1": (0.01, 0.04), "akita_black_bowl_1": (0.05, 0.08)}


def robot_touch(sim, og):
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        for x, y in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
            if y in og:
                nm = sim.model.geom_id2name(x) or ""
                if nm.startswith("robot0") or nm.startswith("gripper0"):
                    return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    comp, lay = a.task.split(":")
    od = os.path.join(OUT, "pp_place_drop")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"{comp}_{lay}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    t = next(x for x in n2["tasks"] if x["compound"] == comp and x["layout"] == lay)
    oe, te = t["object"], t["target"]
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    env = FC.fresh_env(os.path.join(ROOT, t["bddl"]))
    env.reset()
    E = env.env
    sim = E.sim
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    og = con.obj[oe]
    joint = E.get_object(oe).joints[-1]
    goal = str(E.parsed_problem.get("goal_state"))
    rows = []
    for k in STATES:
        env.set_init_state(S[k])
        E.timestep, E.done = 0, False
        for _ in range(WAIT):
            obs, _, _, _ = env.step(FC.DUMMY)
        flat = sim.get_state().flatten().copy()
        pos = FC.scene(env, obs, dj)["pos"]
        tgt, ztab = pos[FC.ENT.index(te)].copy(), float(pos[FK.TABLE, 2])
        zrest = float(pos[FC.ENT.index(oe), 2])
        quat = np.array(sim.data.body_xquat[E.obj_body_id[oe]], np.float64)
        for h in HS[te]:
            for d in DS:
                for b in (range(NB) if d > 0 else (0,)):
                    th = 2.0 * math.pi * b / NB
                    env.set_init_state(flat)
                    E.timestep, E.done = 0, False
                    qp = [tgt[0] + d * math.cos(th), tgt[1] + d * math.sin(th), h + tgt[2] + zrest - ztab] + quat.tolist()
                    sim.data.set_joint_qpos(joint, np.array(qp))
                    sim.data.set_joint_qvel(joint, np.zeros(6))
                    sim.forward()
                    ok, touch = False, False
                    for _ in range(STEPS):
                        obs, _, done, _ = env.step(FC.DUMMY)
                        touch = touch or robot_touch(sim, og)
                        if done:
                            ok = True
                            break
                    p = np.array(sim.data.body_xpos[E.obj_body_id[oe]], np.float64)
                    q = np.array(sim.data.body_xquat[E.obj_body_id[oe]], np.float64)
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)))))
                    pt = FC.scene(env, obs, dj)["pos"][FC.ENT.index(te)]
                    rows.append(dict(state=k, d=d, bearing=b, h=h, success=bool(ok), excluded=bool(touch),
                                     end_d=float(np.linalg.norm((p - pt)[:2])), end_tilt=float(tilt)))
        print("[drop]", comp, lay, k, "done", np.mean([r["success"] for r in rows]), flush=True)
    env.close()
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(compound=comp, layout=lay, object=oe, target=te, goal=goal, host=os.uname().nodename, drops=rows), fh, indent=1)


if __name__ == "__main__":
    main()

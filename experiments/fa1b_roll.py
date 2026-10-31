"""FLAWED_COMPOSITION prototype A, stage 1b (section 5.6): closed-loop rollouts of one stage-1b arm (factored_flow.ARMS_B), in process on
CPU (LIBERO venv).  Stage 1's protocol (fa1_roll.py) unchanged: N2's BDDLs, start states and noise seed bases; a fresh env per
episode re-seeded with 7; 10 settle steps; 300 policy steps; an 8-step chunk per query, the first 5 executed; actions clipped to
[-1, 1]; the client's bind track; success = LIBERO's done.  Added: the physical-contact phase signal (fa1b_common.Held for the task's
role object, stepped after every simulator step from the first policy step on) and its grasp offset are fed to the arm at every
query.  Logged per episode, beyond stage 1's fields: the first held step, the end-effector position there, the number of held
segments, held at the end.

-> results/flawed/a1b_roll/<arm>_s<seed>_<phase>_<tid>.json

    python experiments/fa1b_roll.py --arm Ab --ckpt b_s0 --phase main --tids 0+1+2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import fa1_common as FC                                                     # noqa: E402
import fa1b_common as FB                                                    # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
MAX_STEPS, WAIT, REPLAN = 300, 10, 5


def episode(model, bddl, state, lang, seed_base):
    oe, te = FC.roles(lang)
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(state)
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    hm = FB.Held()
    t, calls, done, err, plan = 0, 0, False, None, []
    prev, bound, bound_step, start_pos, t0 = -1.0, None, None, None, time.time()
    held_step, held_ee, segs, h_on, off = None, None, 0, False, np.zeros(3)
    try:
        while t < MAX_STEPS + WAIT:
            if t < WAIT:
                obs, _, done, _ = env.step(FC.DUMMY)
                t += 1
                if t == WAIT:                                               # the machine starts on the first policy step's state
                    h_on, off = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
                continue
            if not plan:
                b = FF.batch_from_scene(FC.scene(env, obs, dj), oi, ti, prev, held=float(h_on), off=off)
                g = torch.Generator().manual_seed(int(seed_base) + calls)
                ch = model.sample(b, generator=g)[0].numpy()
                plan = [np.clip(x, -1.0, 1.0) for x in ch[:REPLAN]]
                calls += 1
            act = plan.pop(0)
            obs, _, done, _ = env.step(act.tolist())
            prev = float(act[6])
            was = h_on
            h_on, off = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
            if h_on and not was:
                segs += 1
                if held_step is None:
                    held_step, held_ee = t - WAIT, np.asarray(obs["robot0_eef_pos"]).tolist()
            if bound is None:
                mn, mp = FC.movable_positions(env)
                if start_pos is None:
                    start_pos = mp
                elif act[6] > 0:
                    disp = np.linalg.norm(mp - start_pos, axis=-1)
                    if disp.max() > 0.03:
                        bound, bound_step = mn[int(np.argmax(disp))], t - WAIT
            if done:
                break
            t += 1
    except Exception as ex:                                                 # noqa: BLE001
        err = repr(ex)
    sc = FC.scene(env, obs, dj)
    env.close()
    return dict(success=bool(done), steps=t - WAIT, calls=calls, secs=time.time() - t0, error=err, bound=bound, bound_step=bound_step,
                final={e: sc["pos"][i].tolist() for i, e in enumerate(FC.ENT)}, final_grip=sc["grip"].tolist(), last_cmd=prev,
                drawer=sc["drawer"], held_step=held_step, held_ee=held_ee, held_segments=segs, held_end=bool(h_on))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=FF.ARMS_B)
    ap.add_argument("--ckpt", required=True, help="dir under DATA/ckpt, e.g. b_s0")
    ap.add_argument("--phase", required=True, choices=("smoke", "main"))
    ap.add_argument("--tids", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, meta = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", a.ckpt, f"{a.arm}.npz"))
    od = os.path.join(OUT, "a1b_roll")
    os.makedirs(od, exist_ok=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = json.load(open(os.path.join(OUT, f"n2_branches_{a.phase}.json"), encoding="utf-8"))
    for tid in [int(x) for x in a.tids.split("+")]:
        out = os.path.join(od, f"{a.arm}_s{meta['seed']}_{a.phase}_{tid}.json")
        if os.path.exists(out):
            print(f"[fa1b] {out} exists", flush=True)
            continue
        t = n2["tasks"][tid]
        assert t["tid"] == tid
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        eps = []
        for b in br:
            e = episode(model, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            e.update(trial=int(b["trial"]))
            eps.append(e)
            print(f"[fa1b] {a.arm} s{meta['seed']} {a.phase} task {tid} {t['key']} trial {b['trial']}: success={e['success']} "
                  f"bound={e['bound']} held_step={e['held_step']} ({e['secs']:.0f}s)", flush=True)
        res = dict(arm=a.arm, ckpt=a.ckpt, ckpt_meta=meta, phase=a.phase,
                   tasks={str(tid): dict(language=t["language"], success_rate=float(np.mean([e["success"] for e in eps])), episodes=eps)})
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

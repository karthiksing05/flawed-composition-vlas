"""FLAWED_COMPOSITION prototype A, stage 1 (section 5.5): closed-loop rollouts of one trained arm on N2's tasks, in process (the
policy is a small torch model on CPU; LIBERO venv, py3.8, no GPU, no renderer: fa1_data.py equiv showed the camera-free env steps
bit-identically to N2's).

N2's protocol (libero_client.py's branch path, fc_n2_roll.sh's settings): per episode a fresh env, numpy and the env re-seeded
(seed 7), reset, the N2 start state set, 10 settle steps of the dummy action, then 300 policy steps; the policy is queried every 5
steps (an 8-step chunk, the first 5 executed; actions clipped to [-1, 1]); the client's bind track (the first movable object displaced
> 3 cm from its position after the first policy step while the commanded gripper is closed); success = LIBERO's done.  Flow noise:
torch generator seeded seed_base + query index (seed_base from n2_branches_<phase>.json: 13000000 + 1000 k), identical across arms.
Oracle inputs: fa1_common.scene at every query, roles from the template parse of the task's prompt.
Logged per episode, beyond N2's fields: final positions of every slot entity, final finger qpos, the last gripper command.
--phase dev rolls the known tasks from the start states of the held-out demos 45-49 (raw_<K>.npz; recipe development only, never an
eval cell).

-> results/flawed/a1_roll/<arm>_<phase>_<tid>.json   (dev: results/flawed/a1_dev/<arm>_dev_<tag>_<K>.json)

    python experiments/fa1_roll.py --arm A --ckpt final --phase main --tids 0+1+2
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
    t, calls, done, err, plan = 0, 0, False, None, []
    prev, bound, bound_step, start_pos, t0 = -1.0, None, None, None, time.time()
    try:
        while t < MAX_STEPS + WAIT:
            if t < WAIT:
                obs, _, done, _ = env.step(FC.DUMMY)
                t += 1
                continue
            if not plan:
                b = FF.batch_from_scene(FC.scene(env, obs, dj), oi, ti, prev)
                g = torch.Generator().manual_seed(int(seed_base) + calls)
                ch = model.sample(b, generator=g)[0].numpy()
                plan = [np.clip(x, -1.0, 1.0) for x in ch[:REPLAN]]
                calls += 1
            act = plan.pop(0)
            obs, _, done, _ = env.step(act.tolist())
            prev = float(act[6])
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
                drawer=sc["drawer"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=FF.ARMS)
    ap.add_argument("--ckpt", required=True, help="tag under DATA/ckpt")
    ap.add_argument("--phase", required=True, choices=("smoke", "main", "dev"))
    ap.add_argument("--tids", default="")
    ap.add_argument("--max-eps", type=int, default=0, help="at most this many episodes per task (0: all of the phase)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, meta = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", a.ckpt, f"{a.arm}.npz"))
    od = os.path.join(OUT, "a1_dev" if a.phase == "dev" else "a1_roll")
    os.makedirs(od, exist_ok=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    if a.phase == "dev":
        for k, name in FC.KNOWN.items():
            if a.tids and k not in a.tids.split("+"):
                continue
            t = next(x for x in n2["tasks"] if x["compound"] == k and x["layout"] == "familiar")
            r = np.load(os.path.join(FC.DATA, f"raw_{k}.npz"))
            eps = []
            for dm in range(45, 50):
                st = r["states"][np.nonzero(r["demo"] == dm)[0][0]]
                e = episode(model, os.path.join(ROOT, t["bddl"]), st, t["language"], 14000000 + 1000 * dm)
                e.update(trial=dm)
                eps.append(e)
                print(f"[fa1] dev {a.arm} {k} demo-start {dm}: success={e['success']} bound={e['bound']} ({e['secs']:.0f}s)", flush=True)
            with open(os.path.join(od, f"{a.arm}_dev_{a.ckpt}_{k}.json"), "w", encoding="utf-8") as fh:
                json.dump(dict(arm=a.arm, ckpt=a.ckpt, meta=meta, tasks={k: dict(language=t["language"], episodes=eps)}), fh, indent=1)
        return
    br = json.load(open(os.path.join(OUT, f"n2_branches_{a.phase}.json"), encoding="utf-8"))
    if a.max_eps:
        br = br[: a.max_eps]
    for tid in [int(x) for x in a.tids.split("+")]:
        out = os.path.join(od, f"{a.arm}_{a.phase}_{tid}.json")
        if os.path.exists(out):
            print(f"[fa1] {out} exists", flush=True)
            continue
        t = n2["tasks"][tid]
        assert t["tid"] == tid
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        eps = []
        for b in br:
            e = episode(model, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            e.update(trial=int(b["trial"]))
            eps.append(e)
            print(f"[fa1] {a.arm} {a.phase} task {tid} {t['key']} trial {b['trial']}: success={e['success']} bound={e['bound']} "
                  f"({e['secs']:.0f}s)", flush=True)
        res = dict(arm=a.arm, ckpt=a.ckpt, ckpt_meta=meta, phase=a.phase,
                   tasks={str(tid): dict(language=t["language"], success_rate=float(np.mean([e["success"] for e in eps])), episodes=eps)})
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

"""FLAWED_COMPOSITION prototype A, stage 1i DIAGNOSTIC (section 5.15; descriptive only, no verdict): where the known pick-and-place
tasks K5 (bottle -> rack) and K6 (cheese -> bowl) fail.  The registered rollouts log only end-of-episode fields, so each registered
episode is RE-RUN with the same model, state and noise seeds (fresh env and re-seed per episode, as registered) and a per-step trace;
the re-run's success is checked against the registered file (reproduction rate reported).  Episode logic = fa1h_roll.episode
(stage 1d's plus rot0, which every arm before 1h ignores).  Per step: the held object's position and tilt (angle of its body z axis
from vertical), the target reference, the end effector, the commanded gripper, the held flag, the finger contacts.
Per episode summary (fa1i_diag.py classifies):
  hold_steps, held_segments, first hold, the LAST release (held on -> off) step, at release: dxy to the target, dz (object minus
  target reference) against the calibrated rest dz, tilt; final: dxy, dz, tilt; open commands while held; success.
CPU (LIBERO venv).  -> results/flawed/a1i_trace/<name>_s<seed>_<tid>.json

    python experiments/fa1i_trace.py --arm Ah_rot --name Ah_rot --ckpt h_s0 --src a1h_roll --tids 8+10
"""
import argparse
import json
import math
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
import fa1c_common as FK                                                    # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
MAX_STEPS, WAIT, REPLAN = 300, 10, 5


def tilt_deg(env, name):
    R = np.array(env.env.sim.data.body_xmat[env.env.obj_body_id[name]]).reshape(3, 3)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))))


def episode(model, bddl, state, lang, seed_base):
    oe, te = FC.roles(lang)
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(state)
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    hm, cn = FB.Held(), FK.Canon()
    t, calls, done, plan, prev = 0, 0, False, [], -1.0
    h_on, off, zr, cy, rot0 = False, np.zeros(3), 0.0, 0.0, None
    tr = {k: [] for k in ("obj", "tgt", "ee", "tilt", "cmd", "held", "lt", "rt")}

    def track():
        h, o = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
        z, c = cn.step(h, con.pos(oe), lambda: (FC.scene(env, obs, dj)["pos"][ti], float(FC.scene(env, obs, dj)["pos"][FK.TABLE, 2])))
        return h, o, z, c

    err = None
    try:
        while t < MAX_STEPS + WAIT:
            if t < WAIT:
                obs, _, done, _ = env.step(FC.DUMMY)
                t += 1
                if t == WAIT:
                    h_on, off, zr, cy = track()
                    rot0 = FC.scene(env, obs, dj)["ee_rot"].copy() if h_on else None
                continue
            if not plan:
                b = FF.batch_from_scene(FC.scene(env, obs, dj), oi, ti, prev, held=float(h_on), off=off, zrest=zr, cyaw=cy, rot0=rot0)
                g = torch.Generator().manual_seed(int(seed_base) + calls)
                ch = model.sample(b, generator=g)[0].numpy()
                plan = [np.clip(x, -1.0, 1.0) for x in ch[:REPLAN]]
                calls += 1
            act = plan.pop(0)
            obs, _, done, _ = env.step(act.tolist())
            prev = float(act[6])
            was = h_on
            h_on, off, zr, cy = track()
            if h_on and not was:
                rot0 = FC.scene(env, obs, dj)["ee_rot"].copy()
            elif not h_on:
                rot0 = None
            sc = FC.scene(env, obs, dj)
            lt, rt = con.touching(oe)
            tr["obj"].append(con.pos(oe).tolist()); tr["tgt"].append(sc["pos"][ti].tolist()); tr["ee"].append(sc["ee_pos"].tolist())
            tr["tilt"].append(tilt_deg(env, oe)); tr["cmd"].append(prev); tr["held"].append(bool(h_on)); tr["lt"].append(bool(lt))
            tr["rt"].append(bool(rt))
            if done:
                break
            t += 1
    except Exception as ex:                                                 # noqa: BLE001
        err = repr(ex)
    env.close()
    return dict(success=bool(done), error=err, trace=tr)


def summarise(e, rest_dz):
    tr = e["trace"]
    H = np.array(tr["held"], bool)
    O, Tg, cmd = np.array(tr["obj"]), np.array(tr["tgt"]), np.array(tr["cmd"])
    n = len(H)
    s = dict(success=e["success"], steps=n, hold_steps=int(H.sum()), rest_dz=rest_dz)
    if n == 0:
        return s
    on = np.nonzero(H[1:] & ~H[:-1])[0] + 1
    offs = np.nonzero(~H[1:] & H[:-1])[0] + 1
    s["held_segments"] = int(len(on) + int(H[0]))
    s["first_hold"] = int(np.argmax(H)) if H.any() else None
    s["open_cmds_while_held"] = int(np.sum((cmd[1:] < 0) & (cmd[:-1] > 0) & H[:-1]))
    s["held_at_end"] = bool(H[-1])
    dxy = np.linalg.norm(O[:, :2] - Tg[:, :2], axis=1)
    dz = O[:, 2] - Tg[:, 2]
    s["min_dxy_held"] = float(dxy[H].min()) if H.any() else None
    if len(offs):
        r = int(offs[-1])
        s.update(release_step=r, release_dxy=float(dxy[r - 1]), release_dz=float(dz[r - 1]), release_tilt=float(tr["tilt"][r - 1]),
                 release_cmd_open=bool(cmd[r - 1] < 0), max_drop_after=float(dz[r - 1] - dz[r:].min()) if r < n else 0.0)
    s.update(final_dxy=float(dxy[-1]), final_dz=float(dz[-1]), final_tilt=float(tr["tilt"][-1]), max_tilt=float(np.max(tr["tilt"])))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True, help="registered rollout dir under results/flawed")
    ap.add_argument("--tids", default="8+10")
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, meta = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", a.ckpt, f"{a.arm}.npz"))
    od = os.path.join(OUT, "a1i_trace")
    os.makedirs(od, exist_ok=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    for tid in [int(x) for x in a.tids.split("+")]:
        t = n2["tasks"][tid]
        reg = json.load(open(os.path.join(OUT, a.src, f"{a.name}_s{meta['seed']}_main_{tid}.json"), encoding="utf-8"))
        reg_s = {int(x["trial"]): bool(x["success"]) for v in reg["tasks"].values() for x in v["episodes"]}
        rest = float(cal[f"{t['object']}|{t['target']}"]["dz_mean"])
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        eps, t0 = [], time.time()
        for b in br:
            e = episode(model, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            sm = summarise(e, rest)
            sm.update(trial=int(b["trial"]), registered_success=reg_s[int(b["trial"])], error=e["error"])
            eps.append(dict(summary=sm, trace={k: (np.round(v, 4).tolist() if k in ("obj", "tgt", "ee", "tilt") else v)
                                                  for k, v in e["trace"].items()}))
            print(f"[fa1i] {a.name} s{meta['seed']} {t['key']} trial {b['trial']}: success={sm['success']} "
                  f"(registered {sm['registered_success']}) ({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(od, f"{a.name}_s{meta['seed']}_{tid}.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(arm=a.arm, name=a.name, ckpt=a.ckpt, src=a.src, tid=tid, key=t["key"], host=os.uname().nodename,
                           reproduced=float(np.mean([x["summary"]["success"] == x["summary"]["registered_success"] for x in eps])),
                           episodes=eps), fh, indent=0)


if __name__ == "__main__":
    main()

"""FLAWED_COMPOSITION prototype A, stage 1k DIAGNOSTIC (section 5.18; descriptive only): the object-side GRASP and CARRY of the
cheese, the bottle and (as the easy object) the bowl on relocated against familiar layouts.  Each registered episode of the chosen
known tasks is re-run with a per-step trace (fa1i_trace's protocol: same model, state, noise seeds; fresh re-seeded env), adding
the object's yaw and the commanded gripper so that the grasp can be read in the OBJECT frame.
Per episode:
  first_close   first step the gripper is commanded closed; ee minus object at that step, rotated into the object's frame (yaw)
  start         object position / yaw and the end effector in the object frame at the first policy step (for the support check)
  hold class    success | never_closed | closed_no_hold (grasp missed) | slipped (first hold lost with dxy > r while the gripper
                was commanded closed) | released_far (first hold lost with dxy > r after an open command) | carried_to_target
                (first release within r, or still held at the end; failed later)
  hold_ee_obj   ee minus object in the object frame at the first hold step; max_lift of the held object
CPU (LIBERO venv).  -> results/flawed/a1k_trace/<name>_s<seed>_<tid>.json

    python experiments/fa1k_trace.py --arm Ai --name Aj --ckpt j_s3 --src a1j_roll --tids 0+1+6+7+10+11
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
import fa1_read as R1                                                       # noqa: E402
import fa1b_common as FB                                                    # noqa: E402
import fa1c_common as FK                                                    # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
MAX_STEPS, WAIT, REPLAN = 300, 10, 5


def objframe(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]]


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
    tr = {k: [] for k in ("obj", "yaw", "tgt", "ee", "cmd", "held", "lt", "rt")}
    start = None

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
                    sc = FC.scene(env, obs, dj)
                    start = dict(obj=sc["pos"][oi].tolist(), yaw=float(sc["yaw"][oi]), all=sc["pos"].tolist(), ee=sc["ee_pos"].tolist(),
                                 ee_obj=objframe(sc["ee_pos"] - sc["pos"][oi], float(sc["yaw"][oi])))
                continue
            if not plan:
                b = FF.batch_from_scene(FC.scene(env, obs, dj), oi, ti, prev, held=float(h_on), off=off, zrest=zr, cyaw=cy, rot0=rot0)
                g = torch.Generator().manual_seed(int(seed_base) + calls)
                ch = model.sample(b, generator=g)[0].numpy()
                plan = [np.clip(x, -1.0, 1.0) for x in ch[:REPLAN]]
                calls += 1
            act = plan.pop(0)
            sc0 = FC.scene(env, obs, dj)
            obs, _, done, _ = env.step(act.tolist())
            prev = float(act[6])
            was = h_on
            h_on, off, zr, cy = track()
            if h_on and not was:
                rot0 = FC.scene(env, obs, dj)["ee_rot"].copy()
            elif not h_on:
                rot0 = None
            lt, rt = con.touching(oe)
            tr["obj"].append(sc0["pos"][oi].tolist()); tr["yaw"].append(float(sc0["yaw"][oi])); tr["tgt"].append(sc0["pos"][ti].tolist())
            tr["ee"].append(sc0["ee_pos"].tolist()); tr["cmd"].append(prev); tr["held"].append(bool(h_on))
            tr["lt"].append(bool(lt)); tr["rt"].append(bool(rt))
            if done:
                break
            t += 1
    except Exception as ex:                                                 # noqa: BLE001
        err = repr(ex)
    env.close()
    return dict(success=bool(done), error=err, trace=tr, start=start)


def summarise(e, r):
    tr = e["trace"]
    H, cmd = np.array(tr["held"], bool), np.array(tr["cmd"])
    O, Tg, EE, Y = np.array(tr["obj"]), np.array(tr["tgt"]), np.array(tr["ee"]), np.array(tr["yaw"])
    s = dict(success=e["success"], start=e["start"], steps=len(H))
    if len(H) == 0:
        s["hold_class"] = "never_closed"
        return s
    closed = np.nonzero(cmd > 0)[0]
    if len(closed):
        i = int(closed[0])
        s["first_close"] = i
        s["grasp_ee_obj"] = objframe(EE[i] - O[i], float(Y[i]))
        s["grasp_touch"] = [bool(tr["lt"][i]), bool(tr["rt"][i])]
    s["first_hold"] = int(np.argmax(H)) if H.any() else None
    if H.any():
        j = s["first_hold"]
        s["hold_ee_obj"] = objframe(EE[j] - O[j], float(Y[j]))
        s["max_lift"] = float(O[H, 2].max() - O[0, 2])
    segs = int(np.sum(H[1:] & ~H[:-1]) + int(H[0]))
    s["held_segments"] = segs
    dxy = np.linalg.norm(O[:, :2] - Tg[:, :2], axis=1)
    offs = np.nonzero(~H[1:] & H[:-1])[0] + 1
    if len(offs):
        k = int(offs[0])
        s["first_release_dxy"] = float(dxy[k - 1])
        s["first_release_cmd_open"] = bool(cmd[k - 1] < 0)
    if e["success"]:
        s["hold_class"] = "success"
    elif not len(closed):
        s["hold_class"] = "never_closed"
    elif not H.any():
        s["hold_class"] = "closed_no_hold"
    elif len(offs) and s["first_release_dxy"] > r:
        s["hold_class"] = "released_far" if s["first_release_cmd_open"] else "slipped"
    else:
        s["hold_class"] = "carried_to_target"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", default="", help="registered rollout dir (reproduction check); empty = none")
    ap.add_argument("--tids", default="0+1+6+7+10+11")
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, meta = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", a.ckpt, f"{a.arm}.npz"))
    od = os.path.join(OUT, "a1k_trace")
    os.makedirs(od, exist_ok=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    for tid in [int(x) for x in a.tids.split("+")]:
        t = n2["tasks"][tid]
        r = max(R1.RAD[t["target"]], float(np.mean(cal[f"{t['object']}|{t['target']}"]["dxy"])) + R1.DXY_PAD)
        reg_s = {}
        if a.src:
            p = os.path.join(OUT, a.src, f"{a.name}_s{meta['seed']}_main_{tid}.json")
            if os.path.exists(p):
                reg = json.load(open(p, encoding="utf-8"))
                reg_s = {int(x["trial"]): bool(x["success"]) for v in reg["tasks"].values() for x in v["episodes"]}
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        eps, t0 = [], time.time()
        for b in br:
            e = episode(model, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            sm = summarise(e, r)
            sm.update(trial=int(b["trial"]), registered_success=reg_s.get(int(b["trial"])), error=e["error"])
            eps.append(sm)
            print(f"[fa1k] {a.name} s{meta['seed']} {t['key']} trial {b['trial']}: {sm['hold_class']} success={sm['success']} "
                  f"(registered {sm['registered_success']}) ({time.time() - t0:.0f}s)", flush=True)
        rep = [x["success"] == x["registered_success"] for x in eps if x["registered_success"] is not None]
        with open(os.path.join(od, f"{a.name}_s{meta['seed']}_{tid}.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(arm=a.arm, name=a.name, ckpt=a.ckpt, src=a.src, tid=tid, key=t["key"], compound=t["compound"],
                           layout=t["layout"], object=t["object"], host=os.uname().nodename,
                           reproduced=(float(np.mean(rep)) if rep else None), episodes=eps), fh, indent=0)


if __name__ == "__main__":
    main()

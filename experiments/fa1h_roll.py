"""FLAWED_COMPOSITION prototype A, stage 1h (section 5.14): closed-loop rollouts.  Stage 1d's episode (fa1d_roll.episode) with ONE
addition: the end effector's rotation at the hold's onset (rot0), captured on the step the held machine turns on and kept while it
stays on (the current rotation while off) -- the same rule as the training rows (fa1g_train.rot0_rows).  Every other field, the
seeds, the 300 steps and replanning every 5 are unchanged.  One job per model and layout (fa1g_roll.py's convention, --name).

-> results/flawed/<outdir>/<name>_s<seed>_main_<tid>.json

    python experiments/fa1h_roll.py --arm Ah --name Ah --ckpt h_s0 --outdir a1h_roll --tids 0+2
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


def episode(model, bddl, state, lang, seed_base):
    lex = getattr(model, "role_lex", None)                                  # LADDER_RESULTS 5 (L4): roles from the library's lexicon
    if lex:
        from lifelong_ood.policy.parts_flow import role_parse
        parsed = role_parse(lex, lang)
        if parsed is None:                                                  # no induced roles: a failure with no steps
            return dict(success=False, steps=0, calls=0, secs=0.0, error="no induced roles", bound=None, bound_step=None, final={},
                        final_grip=None, last_cmd=-1.0, drawer=None, held_step=None, held_ee=None, held_segments=0, held_end=False,
                        onset_canon=None, min_dxy_held=None, roles=None,
                        pgate=dict(event_step=None, event_segments=0, event_end=False, steps=0, agree=0, event_only=0, oracle_only=0))
        oe, te = parsed
    else:
        oe, te = FC.roles(lang)
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(state)
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    hm, cn = FB.Held(), FK.Canon()
    # LADDER_RESULTS 1 (L1): with model.pgate, the POLICY's gate, Canon, rot0 and d0t follow the proprioceptive event; the oracle
    # signal above is still stepped and logged (the read's instrument).  pgate None: exactly the L0 episode.
    pg = getattr(model, "pgate", None)
    if pg:
        from lifelong_ood.policy.parts_flow import ProprioHeld
        det, cnp = ProprioHeld(pg["tw"], pg["lift"]), FK.Canon()
        ptr = dict(on=False, step=None, segs=0, agree=0, n=0, ev_only=0, or_only=0)
        mv = None                                                           # SCAFFOLD_RESULTS 24: set at the first gate, from the object's position then
    t, calls, done, err, plan = 0, 0, False, None, []
    prev, bound, bound_step, start_pos, t0 = -1.0, None, None, None, time.time()
    held_step, held_ee, onset_c, segs, h_on, off, zr, cy = None, None, None, 0, False, np.zeros(3), 0.0, 0.0
    min_dxy, rot0, d0t = None, None, None

    def tgt():
        p = FC.scene(env, obs, dj)["pos"]
        return p[ti], float(p[FK.TABLE, 2])

    def track():
        h, o = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
        z, c = cn.step(h, con.pos(oe), tgt)
        return h, o, z, c

    def policy_gate(h, z, c):
        """L1: the policy's (held, zrest, cyaw) from the proprioceptive event (demo rows' features: scene grip and ee_pos);
        pgate None: the oracle's, unchanged."""
        if not pg:
            return h, z, c
        sc = FC.scene(env, obs, dj)
        hp = det.step(sc["grip"], prev, sc["ee_pos"][2])
        if pg.get("motion"):                                                # SCAFFOLD_RESULTS 24: held, or the hand moves the object with nothing held
            nonlocal mv
            if mv is None:
                from lifelong_ood.policy.parts_flow import HandMoved
                mv = HandMoved(sc["pos"][oi])
            hp = bool(mv.step(sc["pos"][oi], prev)) or hp
        zp, cp = cnp.step(hp, con.pos(oe), tgt)
        ptr["n"] += 1
        ptr["agree"] += int(hp == h)
        ptr["ev_only"] += int(hp and not h)
        ptr["or_only"] += int(h and not hp)
        if hp and not ptr["on"]:
            ptr["segs"] += 1
            if ptr["step"] is None:
                ptr["step"] = t - WAIT
        ptr["on"] = hp
        return hp, zp, cp

    def onset_d(h, was, d0):
        """PARTS_RESULTS 33 (P_ts): the horizontal object-to-target distance at the hold's onset (None while not held)."""
        if h and not was:
            return float(np.linalg.norm(np.asarray(con.pos(oe))[:2] - np.asarray(tgt()[0])[:2]))
        return d0 if h else None

    def onset_rot(h, was, r0):
        if h and not was:
            return FC.scene(env, obs, dj)["ee_rot"].copy()
        return r0 if h else None

    try:
        while t < MAX_STEPS + WAIT:
            if t < WAIT:
                obs, _, done, _ = env.step(FC.DUMMY)
                t += 1
                if t == WAIT:
                    h_on, off, zr, cy = track()
                    p_on, p_zr, p_cy = policy_gate(h_on, zr, cy)
                    rot0 = onset_rot(p_on, False, None)
                    d0t = onset_d(p_on, False, None)
                    ee0 = FC.scene(env, obs, dj)["ee_pos"].copy()        # the episode start (PARTS_RESULTS 27, P_hs)
                continue
            if not plan:
                b = FF.batch_from_scene(FC.scene(env, obs, dj), oi, ti, prev, held=float(p_on), off=off, zrest=p_zr, cyaw=p_cy,
                                        rot0=rot0)
                if float(getattr(model, "hscale", 0.0)) > 0:
                    b["ee0"] = torch.tensor(np.asarray(ee0), dtype=torch.float32)[None]
                if bool(getattr(model, "tscale", False)) and d0t is not None:   # not held: the model uses the current distance
                    b["d0t"] = torch.tensor([d0t], dtype=torch.float32)
                g = torch.Generator().manual_seed(int(seed_base) + calls)
                ch = model.sample(b, generator=g)[0].numpy()
                plan = [np.clip(x, -1.0, 1.0) for x in ch[:REPLAN]]
                calls += 1
            act = plan.pop(0)
            obs, _, done, _ = env.step(act.tolist())
            prev = float(act[6])
            was, p_was = h_on, p_on
            h_on, off, zr, cy = track()
            p_on, p_zr, p_cy = policy_gate(h_on, zr, cy)
            rot0 = onset_rot(p_on, p_was, rot0)
            d0t = onset_d(p_on, p_was, d0t)
            if h_on and not was:
                segs += 1
                if held_step is None:
                    held_step, held_ee = t - WAIT, np.asarray(obs["robot0_eef_pos"]).tolist()
                    pt, ztab = tgt()
                    p = con.pos(oe)
                    foot = np.array([p[0], p[1], p[2] - zr + ztab])
                    c, s = math.cos(cy), math.sin(cy)
                    QT = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
                    onset_c = dict(foot=(QT @ (foot - pt)).tolist(), ee=(QT @ (np.asarray(held_ee) - pt)).tolist(), cyaw=cy, zrest=zr)
            if h_on:
                dxy = float(np.linalg.norm(con.pos(oe)[:2] - np.asarray(tgt()[0])[:2]))
                min_dxy = dxy if min_dxy is None else min(min_dxy, dxy)
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
    out = dict(success=bool(done), steps=t - WAIT, calls=calls, secs=time.time() - t0, error=err, bound=bound, bound_step=bound_step,
               final={e: sc["pos"][i].tolist() for i, e in enumerate(FC.ENT)}, final_grip=sc["grip"].tolist(), last_cmd=prev,
               drawer=sc["drawer"], held_step=held_step, held_ee=held_ee, held_segments=segs, held_end=bool(h_on), onset_canon=onset_c,
               min_dxy_held=min_dxy)
    if pg:                                                                  # L1: the event's trace next to the oracle's
        out["pgate"] = dict(event_step=ptr["step"], event_segments=ptr["segs"], event_end=bool(ptr["on"]), steps=ptr["n"],
                            agree=ptr["agree"], event_only=ptr["ev_only"], oracle_only=ptr["or_only"])
    if lex:
        out["roles"] = [oe, te]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=FF.ARMS_H + FF.ARMS_I + FF.ARMS_O)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tids", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, meta = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", a.ckpt, f"{a.arm}.npz"))
    od = os.path.join(OUT, a.outdir)
    os.makedirs(od, exist_ok=True)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    for tid in [int(x) for x in a.tids.split("+")]:
        out = os.path.join(od, f"{a.name}_s{meta['seed']}_main_{tid}.json")
        if os.path.exists(out):
            print(f"[fa1h] {out} exists", flush=True)
            continue
        t = n2["tasks"][tid]
        assert t["tid"] == tid
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        eps = []
        for b in br:
            e = episode(model, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            e.update(trial=int(b["trial"]))
            eps.append(e)
            print(f"[fa1h] {a.name} s{meta['seed']} task {tid} {t['key']} trial {b['trial']}: success={e['success']} "
                  f"bound={e['bound']} held_step={e['held_step']} ({e['secs']:.0f}s)", flush=True)
        res = dict(arm=a.arm, name=a.name, ckpt=a.ckpt, ckpt_meta=meta, phase="main", host=os.uname().nodename,
                   tasks={str(tid): dict(language=t["language"], success_rate=float(np.mean([e["success"] for e in eps])), episodes=eps)})
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

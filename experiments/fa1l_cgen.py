"""FLAWED_COMPOSITION prototype A, stage 1l (section 5.21), DATA: a COLLISION-AWARE variant of stage 1g's scripted oracle-state planner,
with ROLE-WISE acceptance.  5.20: on relocated layouts the demo-fitted top-down grasp is blocked by the tall fixtures (wine rack,
cabinet) beside the cheese / bottle.  Per attempt (known pairings K1-K6 only; layout = 1g's sampler; seed 910000 + 1000 * task + k):
  1. grasp test: for each grasp OPTION in order -- yaw offsets 0, +90, +45, -45 deg about the vertical through the object, then
     0 and +90 deg with a 25 deg and a 40 deg tilt AWAY from the nearest fixture -- run the planner's approach (timeout 80 steps,
     1g: 40, so that the hand settles above the grasp point before descending), descent, close and lift from the
     start state (fresh env each time); the first option whose hold is on after the lift (fa1b_common.Held) is chosen;
  2. the full episode is run once more from the start state with the chosen option and recorded (the planner's phases 0-7 as
     1g's, the release unchanged: demo-fitted release offset and rotation);
  3. acceptance by ROLE: the whole episode if LIBERO's predicate fires; else its GRASP SEGMENT (rows up to the end of the lift) if
     the hold was on after the lift; else nothing.
No demo action is executed or replayed; every action comes from a proportional controller on the live simulator state.
    gen    (LIBERO venv, CPU): --task K6 --n 150 (attempts) -> DATA/gen_lc/<task>.npz, results/flawed/a1l_cgen_<task>.json
    merge  (openpi venv, CPU): DATA/gen_k.npz + the task files -> DATA/gen_lc.npz (new demo ids 400 + j; seg = 1 for grasp-segment
           rows; gen = 1) -> results/flawed/a1l_cgen.json

    python experiments/fa1l_cgen.py gen|merge [--task K6 --n 150]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402
import fa1g_plan as FG                                                      # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
BASE_SEED = 910000
OPTIONS = [(0, 0), (90, 0), (45, 0), (-45, 0), (0, 25), (90, 25), (0, 40), (90, 40)]   # (yaw offset, tilt away from nearest fixture), deg
APPROACH_STEPS = 80                                                         # phase-0 timeout (1g: 40), so a turned grasp can settle
FIXTURES = FC.ENT[4:8]


def rz(d):
    c, s = math.cos(d), math.sin(d)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_axis(a, th):
    a = np.asarray(a, float) / max(1e-9, np.linalg.norm(a))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def grasp_pose(kp, sc, oi, opt, jit):
    """The option's grasp rotation and offset (object-centred): yaw about the vertical, then a tilt whose axis z x d tips the hand's
    up-vector toward d = the unit xy direction from the nearest fixture to the object (away from it)."""
    dy, tl = math.radians(opt[0]), math.radians(opt[1])
    RG, goff = rz(dy) @ np.array(kp["grasp_rot"]), rz(dy) @ (np.array(kp["grasp_off"]) + jit)
    if tl:
        po = sc["pos"][oi]
        fx = [FC.ENT.index(f) for f in FIXTURES]
        clr = [np.linalg.norm(sc["pos"][f][:2] - po[:2]) - FC.SIZE[FC.ENT[f]][0] for f in fx]
        d = po[:2] - sc["pos"][fx[int(np.argmin(clr))]][:2]
        d = np.r_[d / max(1e-9, np.linalg.norm(d)), 0.0]
        Rt = rot_axis(np.cross([0, 0, 1.0], d), tl)
        RG, goff = Rt @ RG, Rt @ goff
    return RG, goff


def episode(bddl, s0, oe, te, kp, opt, seed, record, full):
    import fa1b_common as FB
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    rng = np.random.default_rng(seed + 7919)
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(s0)
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    for _ in range(FG.WAIT):
        obs, _, done, _ = env.step(FC.DUMMY)
    jit = rng.normal(0, 0.01, (2, 3))
    cz = kp["carry_z_above_ref"] + rng.normal(0, 0.02)
    sc = FC.scene(env, obs, dj)
    RG, goff = grasp_pose(kp, sc, oi, opt, jit[0])
    roff, RR = np.array(kp["release_off"]) + jit[1], np.array(kp["release_rot"])
    zref = float(sc["pos"][8, 2])
    hm = FB.Held()
    ep = {x: [] for x in ("step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "touch", "objp")}
    phase, hold, prev, done, step, in_phase, rel_up, lift_end, held_lift = 0, 0, -1.0, False, 0, 0, None, None, False
    last = sc["ee_pos"]
    while step < FG.MAX_STEPS and not done and (full or phase <= 3):
        sc = FC.scene(env, obs, dj)
        po, pt = sc["pos"][oi], sc["pos"][ti]
        if phase == 0:
            wp, Rt, g = po + goff + [0, 0, 0.10], RG, -1.0
        elif phase == 1:
            wp, Rt, g = po + goff, RG, -1.0
        elif phase == 2:
            wp, Rt, g = last if hold else sc["ee_pos"], RG, 1.0
        elif phase == 3:
            wp, Rt, g = np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), RG, 1.0
        elif phase == 4:
            wp, Rt, g = np.array([pt[0] + roff[0], pt[1] + roff[1], zref + cz]), RR, 1.0
        elif phase == 5:
            wp, Rt, g = pt + roff, RR, 1.0
        elif phase == 6:
            wp, Rt, g = last, RR, -1.0
        else:
            wp, Rt, g = rel_up, RR, -1.0
        a, err = FG.ctrl(sc, wp, Rt, g, rng)
        if phase in (2, 6):
            a[:6] = 0.0
        lt, rt = con.touching(oe)
        h_on, _ = hm.step(lt, rt, con.pos(oe), sc["ee_pos"])
        if record:
            for x in ("ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                ep[x].append(sc[x])
            ep["step"].append(step); ep["action"].append(a); ep["prev_grip"].append(prev)
            ep["touch"].append((lt, rt)); ep["objp"].append(con.pos(oe))
        last = sc["ee_pos"]
        obs, _, done, _ = env.step(a.tolist())
        prev, step = float(a[6]), step + 1
        in_phase += 1
        if phase in (0, 3, 4) and err < 0.015 or phase in (1, 5) and err < 0.008 or phase in (1, 3, 4, 5) and in_phase >= 40 \
                or phase == 0 and in_phase >= APPROACH_STEPS:
            if phase == 3:
                lt, rt = con.touching(oe)
                held_lift = bool(hm.step(lt, rt, con.pos(oe), FC.scene(env, obs, dj)["ee_pos"])[0])
                lift_end = step
            phase, in_phase = phase + 1, 0
        elif phase in (2, 6):
            hold += 1
            if hold >= 12:
                phase, hold, in_phase = phase + 1, 0, 0
                if phase == 7:
                    rel_up = sc["ee_pos"] + [0, 0, 0.10]
    env.close()
    return dict(success=bool(done), held_lift=held_lift, lift_end=lift_end, steps=step, ep=ep)


def gen(task, n):
    n2, rl = FG.lang_roles()
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"][task]
    t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
    bddl = os.path.join(ROOT, t["bddl"])
    oe, te = rl[task]
    tix = list(FC.KNOWN).index(task)
    rows = {x: [] for x in ("ep", "seg", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "touch", "objp")}
    rep, kept, t0 = [], 0, time.time()
    lay_env = FC.make_env(bddl)
    for k in range(n):
        seed = BASE_SEED + 1000 * tix + k
        s0, p = FG.layout(lay_env, n2, np.random.default_rng(seed), seed)
        if s0 is None:
            rep.append(dict(seed=seed, layout=False))
            continue
        chosen, tried = None, []
        for oi_, opt in enumerate(OPTIONS):
            r = episode(bddl, s0, oe, te, kp, opt, seed, record=False, full=False)
            tried.append(dict(option=list(opt), held_lift=r["held_lift"]))
            if r["held_lift"]:
                chosen = opt
                break
        res = dict(seed=seed, layout=True, positions=np.round(p, 4).tolist(), tried=tried, option=list(chosen) if chosen else None)
        if chosen is not None:
            r = episode(bddl, s0, oe, te, kp, chosen, seed, record=True, full=True)
            res.update(success=r["success"], held_lift=r["held_lift"], steps=r["steps"])
            keep = None
            if r["success"]:
                keep, seg = len(r["ep"]["step"]), 0
            elif r["held_lift"] and r["lift_end"]:
                keep, seg = int(r["lift_end"]), 1
            res["kept"] = None if keep is None else ("episode" if seg == 0 else "grasp_segment")
            if keep:
                for x in r["ep"]:
                    rows[x] += r["ep"][x][:keep]
                rows["ep"] += [kept] * keep
                rows["seg"] += [seg] * keep
                kept += 1
        rep.append(res)
        print(f"[fa1l] cgen {task} {k} seed {seed}: option={res['option']} kept={res.get('kept')} success={res.get('success')} "
              f"tried={[(x['option'], x['held_lift']) for x in tried]} "
              f"({time.time() - t0:.0f}s)", flush=True)
    lay_env.close()
    os.makedirs(os.path.join(FC.DATA, "gen_lc"), exist_ok=True)
    np.savez_compressed(os.path.join(FC.DATA, "gen_lc", f"{task}.npz"), task_index=tix, **{x: np.asarray(v) for x, v in rows.items()})
    with open(os.path.join(OUT, f"a1l_cgen_{task}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(task=task, attempts=n, kept=kept, base=BASE_SEED, options=OPTIONS, episodes=rep, generated=True,
                       note="collision-aware variant of stage 1g's scripted oracle-state planner; role-wise acceptance; no demo action "
                            "executed or replayed"), fh, indent=1)


def merge(tasks):
    import fa1b_common as FB
    from fa1_data import noop_mask
    G0 = dict(np.load(os.path.join(FC.DATA, "gen_k.npz")))
    keys = ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "held", "off", "touch", "lift")
    cols = {x: list(G0[x]) for x in keys}
    seg = [0] * len(G0["task"])
    _, rl = FG.lang_roles()
    rep, per_obj = {}, {}
    for k in tasks:
        G = dict(np.load(os.path.join(FC.DATA, "gen_lc", f"{k}.npz")))
        tix = int(G["task_index"])
        r = dict(episodes=0, segments=0, rows=0)
        for e in (np.unique(G["ep"]) if len(G["ep"]) else []):
            m = np.nonzero(G["ep"] == e)[0]
            keep = m[noop_mask(G["action"][m])]
            hm, prev = FB.Held(), -1.0
            sg = int(G["seg"][m[0]])
            for s, i in enumerate(keep):
                lt, rt = bool(G["touch"][i][0]), bool(G["touch"][i][1])
                h, off = hm.step(lt, rt, G["objp"][i], G["ee_pos"][i])
                for x in ("action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                    cols[x].append(G[x][i])
                cols["task"].append(tix); cols["demo"].append(400 + int(e)); cols["step"].append(s); cols["prev_grip"].append(prev)
                cols["held"].append(float(h)); cols["off"].append(off); cols["touch"].append((lt, rt)); cols["lift"].append(G["objp"][i][2] - hm.z0)
                seg.append(sg)
                prev = float(G["action"][i][6])
            r["segments" if sg else "episodes"] += 1
            r["rows"] += len(keep)
        rep[k] = r
        o = per_obj.setdefault(rl[k][0], dict(episodes=0, segments=0, rows=0))
        for x in o:
            o[x] += r[x]
    out = {x: np.asarray(v, np.float32 if x not in ("task", "demo", "step") else np.int32) for x, v in cols.items()}
    assert out["demo"].max() < 1000
    np.savez_compressed(os.path.join(FC.DATA, "gen_lc.npz"), gen=np.ones(len(out["task"]), np.int8), seg=np.asarray(seg, np.int8), **out)
    for k in rep:
        print(f"[fa1l] merge {k}: {json.dumps(rep[k])}", flush=True)
    print(f"[fa1l] per object (new data): {json.dumps(per_obj)}; rows {len(out['task'])}", flush=True)
    with open(os.path.join(OUT, "a1l_cgen.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(tasks=rep, per_object_new=per_obj, rows=int(len(out["task"])), segment_rows=int(np.sum(seg)), generated=True), fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "merge"))
    ap.add_argument("--task", default="K6")
    ap.add_argument("--tasks", default="K1+K2+K3+K4+K5+K6")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    if a.mode == "gen":
        gen(a.task, a.n)
    else:
        merge(a.tasks.split("+"))

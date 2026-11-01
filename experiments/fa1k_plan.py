"""FLAWED_COMPOSITION prototype A, stage 1k (section 5.19), DATA: MORE generated varied-layout carries of the KNOWN pick-and-place
pairings, so that every carried object has the same object-side support (the same number of generated relocated grasps).
Planner, keypoints, layout sampler, acceptance rule and episode protocol are stage 1g's, unchanged (fa1g_plan.py: layout, ctrl,
a1g_keypoints.json); only the seed block and the output files differ.  No demo action is executed or replayed.

    gen    (LIBERO venv, CPU): --task K6 --n 150.  Attempt k uses seed GEN_BASE_K + 1000 * task + k, GEN_BASE_K = 900500, so the
           seeds never meet 1g's (900000 + 1000 * task + k, k < 500) nor N2's (600000-600023, 700000-700023); attempts are capped at
           k < 500 -> DATA/gen_k/<task>.npz, results/flawed/a1k_gen_<task>.json
    merge  (openpi venv, CPU): DATA/gen_g.npz plus the new task files -> DATA/gen_k.npz (demo ids: 1g's 100-149 kept; new episodes
           150 + j), fields as 1g's merge (held / off / touch / lift stepped with fa1b_common.Held, no-op steps dropped) and gen = 1
           -> results/flawed/a1k_gen.json (per object: generated episodes, carry distances, start-position spread)

    python experiments/fa1k_plan.py gen|merge [--task K6 --n 150]
"""
import argparse
import json
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
GEN_BASE_K, K_MAX = 900500, 500
MAX_STEPS, WAIT = FG.MAX_STEPS, FG.WAIT


def gen(task, n):
    import fa1b_common as FB
    n2, rl = FG.lang_roles()
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"][task]
    t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
    bddl = os.path.join(ROOT, t["bddl"])
    oe, te = rl[task]
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    tix = list(FC.KNOWN).index(task)
    rows = {x: [] for x in ("ep", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "touch", "objp")}
    rep, kept, k, t0 = [], 0, 0, time.time()
    lay_env = FC.make_env(bddl)
    while kept < n and k < K_MAX:
        seed = GEN_BASE_K + 1000 * tix + k
        rng = np.random.default_rng(seed)
        s0, p = FG.layout(lay_env, n2, rng, seed)
        k += 1
        if s0 is None:
            rep.append(dict(seed=seed, layout=False))
            continue
        env = FC.fresh_env(bddl)
        env.reset()
        obs = env.set_init_state(s0)
        dj = FC.drawer_joint(env)
        con = FB.Contact(env, [oe])
        for _ in range(WAIT):
            obs, _, done, _ = env.step(FC.DUMMY)
        jit = rng.normal(0, 0.01, (2, 3))
        cz = kp["carry_z_above_ref"] + rng.normal(0, 0.02)
        goff, roff = np.array(kp["grasp_off"]) + jit[0], np.array(kp["release_off"]) + jit[1]
        RG, RR = np.array(kp["grasp_rot"]), np.array(kp["release_rot"])
        ep = {x: [] for x in rows if x != "ep"}
        phase, hold, prev, done, step, in_phase, rel_up = 0, 0, -1.0, False, 0, 0, None
        sc = FC.scene(env, obs, dj)
        zref = float(sc["pos"][8, 2])
        while step < MAX_STEPS and not done:
            sc = FC.scene(env, obs, dj)
            po, pt = sc["pos"][oi], sc["pos"][ti]
            if phase == 0:
                wp, Rt, g = po + goff + [0, 0, 0.10], RG, -1.0
            elif phase == 1:
                wp, Rt, g = po + goff, RG, -1.0
            elif phase == 2:
                wp, Rt, g = ep["ee_pos"][-1] if hold else sc["ee_pos"], RG, 1.0
            elif phase == 3:
                wp, Rt, g = np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), RG, 1.0
            elif phase == 4:
                wp, Rt, g = np.array([pt[0] + roff[0], pt[1] + roff[1], zref + cz]), RR, 1.0
            elif phase == 5:
                wp, Rt, g = pt + roff, RR, 1.0
            elif phase == 6:
                wp, Rt, g = ep["ee_pos"][-1], RR, -1.0
            else:
                wp, Rt, g = rel_up, RR, -1.0
            a, err = FG.ctrl(sc, wp, Rt, g, rng)
            if phase in (2, 6):
                a[:6] = 0.0
            for x in ("ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                ep[x].append(sc[x])
            ep["step"].append(step); ep["action"].append(a); ep["prev_grip"].append(prev)
            ep["touch"].append(con.touching(oe)); ep["objp"].append(con.pos(oe))
            obs, _, done, _ = env.step(a.tolist())
            prev, step = float(a[6]), step + 1
            in_phase += 1
            if phase in (0, 3, 4) and err < 0.015 or phase in (1, 5) and err < 0.008 or phase in (0, 1, 3, 4, 5) and in_phase >= 40:
                phase, in_phase = phase + 1, 0
            elif phase in (2, 6):
                hold += 1
                if hold >= 12:
                    phase, hold, in_phase = phase + 1, 0, 0
                    if phase == 7:
                        rel_up = sc["ee_pos"] + [0, 0, 0.10]
        env.close()
        d0 = float(np.linalg.norm(ep["objp"][0][:2] - ep["pos"][0][ti][:2]))
        rep.append(dict(seed=seed, layout=True, success=bool(done), steps=step, phase=phase, carry_dist=d0, positions=np.round(p, 4).tolist()))
        print(f"[fa1k] {task} attempt {k - 1} seed {seed}: success={done} steps={step} phase={phase} carry={d0:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if done:
            for x in ep:
                rows[x] += ep[x]
            rows["ep"] += [kept] * len(ep["step"])
            kept += 1
    lay_env.close()
    os.makedirs(os.path.join(FC.DATA, "gen_k"), exist_ok=True)
    np.savez_compressed(os.path.join(FC.DATA, "gen_k", f"{task}.npz"), task_index=tix, **{x: np.asarray(v) for x, v in rows.items()})
    with open(os.path.join(OUT, f"a1k_gen_{task}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(task=task, kept=kept, attempts=k, base=GEN_BASE_K, keypoints=kp, episodes=rep, generated=True,
                       note="stage 1g's scripted oracle-state planner, new seed block; no demo action executed or replayed"), fh, indent=1)


def merge(tasks):
    import fa1b_common as FB
    from fa1_data import noop_mask
    _, rl = FG.lang_roles()
    G0 = dict(np.load(os.path.join(FC.DATA, "gen_g.npz")))
    keys = ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "held", "off", "touch", "lift")
    cols = {x: list(G0[x]) for x in keys}
    rep = {}
    for k in tasks:
        G = dict(np.load(os.path.join(FC.DATA, "gen_k", f"{k}.npz")))
        tix = int(G["task_index"])
        n_ep, n_rows = 0, 0
        for e in np.unique(G["ep"]):
            m = np.nonzero(G["ep"] == e)[0]
            keep = m[noop_mask(G["action"][m])]
            hm, prev = FB.Held(), -1.0
            for s, i in enumerate(keep):
                lt, rt = bool(G["touch"][i][0]), bool(G["touch"][i][1])
                h, off = hm.step(lt, rt, G["objp"][i], G["ee_pos"][i])
                for x in ("action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                    cols[x].append(G[x][i])
                cols["task"].append(tix); cols["demo"].append(150 + int(e)); cols["step"].append(s); cols["prev_grip"].append(prev)
                cols["held"].append(float(h)); cols["off"].append(off); cols["touch"].append((lt, rt)); cols["lift"].append(G["objp"][i][2] - hm.z0)
                prev = float(G["action"][i][6])
            n_ep += 1
            n_rows += len(keep)
        rep[k] = dict(new_episodes=n_ep, new_rows=n_rows)
    out = {x: np.asarray(v, np.float32 if x not in ("task", "demo", "step") else np.int32) for x, v in cols.items()}
    assert out["demo"].max() < 1000
    np.savez_compressed(os.path.join(FC.DATA, "gen_k.npz"), gen=np.ones(len(out["task"]), np.int8), **out)
    per_obj = {}
    for tix, k in enumerate(FC.KNOWN):
        if k not in rl:
            continue
        m = (out["task"] == tix) & (out["step"] == 0)
        oi = FC.ENT.index(rl[k][0])
        o = per_obj.setdefault(rl[k][0], dict(episodes=0, xy=[]))
        o["episodes"] += int(m.sum())
        o["xy"] += out["pos"][m, oi, :2].tolist()
    per_obj = {ob: dict(episodes=v["episodes"], xy_std=np.std(np.array(v["xy"]), 0).tolist() if v["xy"] else None) for ob, v in per_obj.items()}
    for k in rep:
        print(f"[fa1k] merge {k}: {json.dumps(rep[k])}", flush=True)
    print(f"[fa1k] per object: {json.dumps(per_obj)}; rows {len(out['task'])}", flush=True)
    with open(os.path.join(OUT, "a1k_gen.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(tasks=rep, per_object=per_obj, rows=int(len(out["task"])), generated=True), fh, indent=1)


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
        merge(a.task and [t for t in a.tasks.split("+")])

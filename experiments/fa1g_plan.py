"""FLAWED_COMPOSITION prototype A, stage 1g (section 5.12), DATA: GENERATED varied-layout carries for the KNOWN pick-and-place pairings
(K1-K6) from a scripted ORACLE-STATE planner.  No demo action is executed or replayed: the planner's keypoints are FITTED from the
demos' recorded states (mode stats), and every generated action comes from a proportional controller on the live simulator state.

    stats  (openpi venv, CPU): per known task, from DATA/demos.npz + demos_b.npz (recorded states only): the grasp offset (end effector
           minus object at the first close command), the grasp rotation, the carry height (highest end effector while held), the
           release offset (end effector minus target at the first open command after the grasp) and the release rotation; medians
           over the 50 demos, rotations = the demo rotation nearest the chordal mean -> results/flawed/a1g_keypoints.json
    gen    (LIBERO venv, CPU): --task K1 --n 50.  Per attempt k: N2's familiar sampler with seed GEN_BASE + 1000 * task + k (never an N2
           seed: those are 600000-600023 and 700000-700023), then the four movable objects relocated uniformly in N2's layout box
           (n2_tasks.json constraints) with N2's clearance rules but WITHOUT the 0.10 m exclusion from the training spot, so that both
           familiar-like and relocated layouts occur; accepted if 10 settle steps move no object > 2 cm.  The episode is the rollouts'
           (fresh env, 10 dummy steps, 300 policy steps).  The planner: approach above the grasp point, descend, close, lift to the
           carry height, carry above the release point, descend, open, retreat; waypoints jittered per episode (1 cm, carry height
           2 cm).  Kept only if LIBERO's success predicate fires.  Rows recorded as demos.npz's (scene before the action, action,
           prev command), plus the raw finger contacts for the held label -> DATA/gen/<task>.npz, results/flawed/a1g_gen_<task>.json
    merge  (openpi venv, CPU): the 6 task files -> DATA/gen_g.npz, demos.npz's fields plus held / off / touch / lift (fa1b_common.Held
           stepped over each episode's kept steps, no-op steps dropped by fa1_data.noop_mask as for the demos) and gen = 1; demo ids
           100 + j -> results/flawed/a1g_gen.json (counts, carry distances against the demos')

    python experiments/fa1g_plan.py stats|gen|merge [--task K1 --n 50]
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

OUT = os.path.join(ROOT, "results", "flawed")
GEN_BASE, PP = 900000, ["K1", "K2", "K3", "K4", "K5", "K6"]
MAX_STEPS, WAIT = 300, 10
MOVR = {"akita_black_bowl_1": 0.06, "cream_cheese_1": 0.045, "wine_bottle_1": 0.045, "plate_1": 0.10}
FIXR = {"wooden_cabinet_1": 0.13, "flat_stove_1": 0.11, "wine_rack_1": 0.10}


def lang_roles():
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    return n2, {k: FC.roles(lang[k]) for k in FC.KNOWN}


def rot_mean(Rs):
    M = np.mean(Rs, 0)
    return Rs[int(np.argmin([np.linalg.norm(R - M) for R in Rs]))]


def stats():
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    _, rl = lang_roles()
    out = {}
    for ti, k in enumerate(FC.KNOWN):
        if k not in PP:
            continue
        oi, tg = FC.ENT.index(rl[k][0]), FC.ENT.index(rl[k][1])
        G, RG, CZ, RO, RR, CD, NS = [], [], [], [], [], [], []
        for dm in np.unique(D["demo"][D["task"] == ti]):
            idx = np.nonzero((D["task"] == ti) & (D["demo"] == dm))[0]
            a, h = D["action"][idx, 6], Db["held"][idx] > 0.5
            if not h.any():
                continue
            ic = int(np.argmax(a > 0))
            io = ic + int(np.argmax(a[ic:] < 0)) if (a[ic:] < 0).any() else len(idx) - 1
            hs = np.nonzero(h)[0]
            i, j, r = idx[ic], idx[io], idx[hs]
            G.append(D["ee_pos"][i] - D["pos"][i, oi])
            RG.append(D["ee_rot"][i])
            CZ.append(float(D["ee_pos"][r, 2].max() - D["pos"][r[0], 8, 2]))
            RO.append(D["ee_pos"][j] - D["pos"][j, tg])
            RR.append(D["ee_rot"][j])
            CD.append(float(np.linalg.norm(D["pos"][idx[0], oi, :2] - D["pos"][idx[0], tg, :2])))
            NS.append(len(idx))
        G, RO = np.array(G), np.array(RO)
        out[k] = dict(object=rl[k][0], target=rl[k][1], n=len(G), grasp_off=np.median(G, 0).tolist(), grasp_off_mad=np.median(np.abs(G - np.median(G, 0)), 0).tolist(),
                      grasp_rot=rot_mean(np.array(RG)).tolist(), carry_z_above_ref=float(np.median(CZ)),
                      release_off=np.median(RO, 0).tolist(), release_off_mad=np.median(np.abs(RO - np.median(RO, 0)), 0).tolist(),
                      release_rot=rot_mean(np.array(RR)).tolist(), carry_dist=[float(min(CD)), float(np.median(CD)), float(max(CD))],
                      steps_median=float(np.median(NS)))
        print(f"[fa1g] {k}: {json.dumps({x: (np.round(v, 3).tolist() if isinstance(v, list) else v) for x, v in out[k].items() if 'rot' not in x})}", flush=True)
    with open(os.path.join(OUT, "a1g_keypoints.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(tasks=out, ref_slot=FC.ENT[8], note="fitted from recorded demo states; carry_z_above_ref = highest held end "
                                                         "effector minus the slot-8 reference height at the hold's first row"), fh, indent=1)


def rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def ctrl(sc, wp, Rt, grip, rng):
    dp = np.asarray(wp) - sc["ee_pos"]
    a = np.zeros(7)
    a[:3] = np.clip(dp / 0.02, -1, 1)
    a[3:6] = np.clip(rotvec(np.asarray(Rt) @ sc["ee_rot"].T) / 0.2, -1, 1)
    a[:3] += rng.normal(0, 0.03, 3)
    a[6] = grip
    return np.clip(a, -1, 1), float(np.linalg.norm(dp))


def layout(env, n2, rng, seed):
    """N2's familiar sampler state for `seed`, the movable objects relocated in N2's box (no exclusion from the training spot)."""
    c = n2["constraints"]
    lo, hi, gap = np.array(c["box"][0]), np.array(c["box"][1]), c["gap"]
    fixp, sf = np.array(c["fixture_xy"]), c["stove_front_world"]
    mov, fixn = n2["movable"], n2["fixtures"]
    np.random.seed(seed)
    env.seed(seed)
    env.reset()
    s_fam = env.env.sim.get_state().flatten()

    def ok(p):
        for i, nm in enumerate(mov):
            r = MOVR[nm]
            if any(np.linalg.norm(p[i] - p[j]) < r + MOVR[mov[j]] + gap for j in range(i)):
                return False
            if any(np.linalg.norm(p[i] - fixp[f]) < r + FIXR[fn] + gap for f, fn in enumerate(fixn)):
                return False
            cx, cy = np.clip(p[i][0], sf[0], sf[2]), np.clip(p[i][1], sf[1], sf[3])
            if np.hypot(p[i][0] - cx, p[i][1] - cy) < r + gap:
                return False
        return True

    for _ in range(2000):
        p = rng.uniform(lo, hi, size=(len(mov), 2))
        if not ok(p):
            continue
        env.reset()
        env.set_init_state(s_fam)
        sim = env.env.sim
        for i, nm in enumerate(mov):
            m = sim.model
            a = int(m.jnt_qposadr[m.body_jntadr[env.env.obj_body_id[nm]]])
            sim.data.qpos[a:a + 2] = p[i]
        sim.forward()
        s_new = sim.get_state().flatten()
        env.reset()
        env.set_init_state(s_new)
        sim = env.env.sim
        q0 = np.array([sim.data.body_xpos[env.env.obj_body_id[n]] for n in mov])
        for _ in range(10):
            env.step(FC.DUMMY)
        q1 = np.array([sim.data.body_xpos[env.env.obj_body_id[n]] for n in mov])
        if np.linalg.norm(q1[:, :2] - q0[:, :2], axis=-1).max() <= 0.02 and np.allclose(q0[:, :2], p, atol=1e-4):
            return s_new, p
    return None, None


def gen(task, n):
    import fa1b_common as FB
    n2, rl = lang_roles()
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"][task]
    t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
    bddl = os.path.join(ROOT, t["bddl"])
    oe, te = rl[task]
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    tix = list(FC.KNOWN).index(task)
    rows = {x: [] for x in ("ep", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "touch", "objp")}
    rep, kept, k, t0 = [], 0, 0, time.time()
    lay_env = FC.make_env(bddl)
    while kept < n and k < 6 * n:
        seed = GEN_BASE + 1000 * tix + k
        rng = np.random.default_rng(seed)
        s0, p = layout(lay_env, n2, rng, seed)
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
        phase, hold, prev, done, step, in_phase = 0, 0, -1.0, False, 0, 0
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
            a, err = ctrl(sc, wp, Rt, g, rng)
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
        print(f"[fa1g] {task} attempt {k - 1} seed {seed}: success={done} steps={step} phase={phase} carry={d0:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if done:
            for x in ep:
                rows[x] += ep[x]
            rows["ep"] += [kept] * len(ep["step"])
            kept += 1
    lay_env.close()
    os.makedirs(os.path.join(FC.DATA, "gen"), exist_ok=True)
    np.savez_compressed(os.path.join(FC.DATA, "gen", f"{task}.npz"), task_index=tix, **{x: np.asarray(v) for x, v in rows.items()})
    with open(os.path.join(OUT, f"a1g_gen_{task}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(task=task, kept=kept, attempts=k, base=GEN_BASE, keypoints=kp, episodes=rep, generated=True,
                       note="scripted oracle-state planner; no demo action executed or replayed"), fh, indent=1)


def merge():
    import fa1b_common as FB
    from fa1_data import noop_mask
    _, rl = lang_roles()
    cols = {x: [] for x in ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip",
                            "held", "off", "touch", "lift")}
    rep = {}
    for k in PP:
        G = dict(np.load(os.path.join(FC.DATA, "gen", f"{k}.npz")))
        oi = FC.ENT.index(rl[k][0])
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
                cols["task"].append(tix); cols["demo"].append(100 + int(e)); cols["step"].append(s); cols["prev_grip"].append(prev)
                cols["held"].append(float(h)); cols["off"].append(off); cols["touch"].append((lt, rt)); cols["lift"].append(G["objp"][i][2] - hm.z0)
                prev = float(G["action"][i][6])
            n_ep += 1
            n_rows += len(keep)
        rep[k] = dict(episodes=n_ep, rows=n_rows)
    out = {x: np.asarray(v, np.float32 if x not in ("task", "demo", "step") else np.int32) for x, v in cols.items()}
    np.savez_compressed(os.path.join(FC.DATA, "gen_g.npz"), gen=np.ones(len(out["task"]), np.int8), **out)
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    for k in PP:
        tix = list(FC.KNOWN).index(k)
        oi, ti = FC.ENT.index(rl[k][0]), FC.ENT.index(rl[k][1])
        for name, X in (("demo", D), ("gen", out)):
            m = (X["task"] == tix) & (X["step"] == 0)
            d = np.linalg.norm(X["pos"][m, oi, :2] - X["pos"][m, ti, :2], axis=-1)
            rep[k][f"carry_{name}"] = [float(d.min()), float(np.median(d)), float(d.max())]
        rep[k]["held_share"] = float(out["held"][out["task"] == tix].mean())
        print(f"[fa1g] merge {k}: {json.dumps(rep[k])}", flush=True)
    with open(os.path.join(OUT, "a1g_gen.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(tasks=rep, rows=int(len(out["task"])), generated=True), fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("stats", "gen", "merge"))
    ap.add_argument("--task", default="K1")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    if a.mode == "stats":
        stats()
    elif a.mode == "gen":
        gen(a.task, a.n)
    else:
        merge()

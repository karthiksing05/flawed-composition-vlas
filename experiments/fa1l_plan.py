"""FLAWED_COMPOSITION prototype A, stage 1l (section 5.21), DATA: ROLE-WISE ACCEPTANCE of the generated attempts.  1g and 1k kept a
generated attempt only if the whole pick and place succeeded, so an object's grasp was supervised only at positions where the
scripted PLACE also worked (5.20: every planner failure is a placement failure, phase 7, and the policy misses the grasp exactly
where the planner's local yield is low).  The object term is target-blind; its role is approach, close and lift.  Here every
1g / 1k attempt that FAILED the task is re-run (same task, same seed, stage 1g's planner and layout sampler unchanged) and its
GRASP SEGMENT is kept -- the rows from the first step to the end of the lift (planner phase 3) -- if the hold is on at that point
(fa1b_common.Held: both fingers touching, lifted >= LIFT).  Nothing after the lift is kept, so no failed placement is ever
trained on.  No demo action is executed or replayed; no new pairing is generated.

    gen    (LIBERO venv, CPU): --task K6 -> DATA/gen_l/<task>.npz, results/flawed/a1l_gen_<task>.json
    merge  (openpi venv, CPU): DATA/gen_k.npz plus the grasp segments -> DATA/gen_l.npz (segment demo ids 400 + j; field seg = 1
           for grasp-segment rows, 0 otherwise; gen = 1) -> results/flawed/a1l_gen.json

    python experiments/fa1l_plan.py gen|merge [--task K6]
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
MAX_STEPS, WAIT = FG.MAX_STEPS, FG.WAIT


def failed_seeds(task):
    out = []
    for f in (f"a1g_gen_{task}.json", f"a1k_gen_{task}.json"):
        d = json.load(open(os.path.join(OUT, f), encoding="utf-8"))
        out += [int(e["seed"]) for e in d["episodes"] if e.get("layout") and not e["success"]]
    return out


def gen(task):
    import fa1b_common as FB
    n2, rl = FG.lang_roles()
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"][task]
    t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
    bddl = os.path.join(ROOT, t["bddl"])
    oe, te = rl[task]
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    tix = list(FC.KNOWN).index(task)
    rows = {x: [] for x in ("ep", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "touch", "objp")}
    rep, kept, t0 = [], 0, time.time()
    lay_env = FC.make_env(bddl)
    for seed in failed_seeds(task):
        rng = np.random.default_rng(seed)
        s0, p = FG.layout(lay_env, n2, rng, seed)
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
        goff = np.array(kp["grasp_off"]) + jit[0]
        RG = np.array(kp["grasp_rot"])
        ep = {x: [] for x in rows if x != "ep"}
        phase, hold, prev, done, step, in_phase = 0, 0, -1.0, False, 0, 0
        hm = FB.Held()
        sc = FC.scene(env, obs, dj)
        zref = float(sc["pos"][8, 2])
        while step < MAX_STEPS and not done and phase <= 3:
            sc = FC.scene(env, obs, dj)
            po = sc["pos"][oi]
            if phase == 0:
                wp, Rt, g = po + goff + [0, 0, 0.10], RG, -1.0
            elif phase == 1:
                wp, Rt, g = po + goff, RG, -1.0
            elif phase == 2:
                wp, Rt, g = ep["ee_pos"][-1] if hold else sc["ee_pos"], RG, 1.0
            else:
                wp, Rt, g = np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), RG, 1.0
            a, err = FG.ctrl(sc, wp, Rt, g, rng)
            if phase == 2:
                a[:6] = 0.0
            for x in ("ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                ep[x].append(sc[x])
            ep["step"].append(step); ep["action"].append(a); ep["prev_grip"].append(prev)
            lt, rt = con.touching(oe)
            ep["touch"].append((lt, rt)); ep["objp"].append(con.pos(oe))
            h_on, _ = hm.step(lt, rt, con.pos(oe), sc["ee_pos"])
            obs, _, done, _ = env.step(a.tolist())
            prev, step = float(a[6]), step + 1
            in_phase += 1
            if phase in (0, 3) and err < 0.015 or phase == 1 and err < 0.008 or phase in (0, 1, 3) and in_phase >= 40:
                phase, in_phase = phase + 1, 0
            elif phase == 2:
                hold += 1
                if hold >= 12:
                    phase, hold, in_phase = phase + 1, 0, 0
        lt, rt = con.touching(oe)
        h_end, _ = hm.step(lt, rt, con.pos(oe), FC.scene(env, obs, dj)["ee_pos"])
        env.close()
        ok = bool(h_end and phase == 4)
        rep.append(dict(seed=seed, layout=True, grasp_ok=ok, steps=step, phase=phase, positions=np.round(p, 4).tolist()))
        print(f"[fa1l] {task} seed {seed}: grasp_ok={ok} steps={step} phase={phase} ({time.time() - t0:.0f}s)", flush=True)
        if ok:
            for x in ep:
                rows[x] += ep[x]
            rows["ep"] += [kept] * len(ep["step"])
            kept += 1
    lay_env.close()
    os.makedirs(os.path.join(FC.DATA, "gen_l"), exist_ok=True)
    np.savez_compressed(os.path.join(FC.DATA, "gen_l", f"{task}.npz"), task_index=tix, **{x: np.asarray(v) for x, v in rows.items()})
    with open(os.path.join(OUT, f"a1l_gen_{task}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(task=task, kept=kept, attempts=len(rep), episodes=rep, generated=True, segment="grasp (phases 0-3)",
                       note="stage 1g's planner on 1g/1k's failed seeds; grasp segment kept when the hold is on after the lift"), fh, indent=1)


def merge(tasks):
    import fa1b_common as FB
    from fa1_data import noop_mask
    G0 = dict(np.load(os.path.join(FC.DATA, "gen_k.npz")))
    keys = ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "held", "off", "touch", "lift")
    cols = {x: list(G0[x]) for x in keys}
    seg = [0] * len(G0["task"])
    rep, base = {}, 400
    for k in tasks:
        G = dict(np.load(os.path.join(FC.DATA, "gen_l", f"{k}.npz")))
        tix = int(G["task_index"])
        n_ep, n_rows = 0, 0
        for e in np.unique(G["ep"]) if len(G["ep"]) else []:
            m = np.nonzero(G["ep"] == e)[0]
            keep = m[noop_mask(G["action"][m])]
            hm, prev = FB.Held(), -1.0
            for s, i in enumerate(keep):
                lt, rt = bool(G["touch"][i][0]), bool(G["touch"][i][1])
                h, off = hm.step(lt, rt, G["objp"][i], G["ee_pos"][i])
                for x in ("action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                    cols[x].append(G[x][i])
                cols["task"].append(tix); cols["demo"].append(base + int(e)); cols["step"].append(s); cols["prev_grip"].append(prev)
                cols["held"].append(float(h)); cols["off"].append(off); cols["touch"].append((lt, rt)); cols["lift"].append(G["objp"][i][2] - hm.z0)
                seg.append(1)
                prev = float(G["action"][i][6])
            n_ep += 1
            n_rows += len(keep)
        rep[k] = dict(segments=n_ep, rows=n_rows)
    out = {x: np.asarray(v, np.float32 if x not in ("task", "demo", "step") else np.int32) for x, v in cols.items()}
    assert out["demo"].max() < 1000
    np.savez_compressed(os.path.join(FC.DATA, "gen_l.npz"), gen=np.ones(len(out["task"]), np.int8), seg=np.asarray(seg, np.int8), **out)
    _, rl = FG.lang_roles()
    per_obj = {}
    for k, r in rep.items():
        o = per_obj.setdefault(rl[k][0], dict(segments=0, rows=0))
        o["segments"] += r["segments"]; o["rows"] += r["rows"]
    for k in rep:
        print(f"[fa1l] merge {k}: {json.dumps(rep[k])}", flush=True)
    print(f"[fa1l] per object: {json.dumps(per_obj)}; rows {len(out['task'])} (segment rows {int(np.sum(seg))})", flush=True)
    with open(os.path.join(OUT, "a1l_gen.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(tasks=rep, per_object=per_obj, rows=int(len(out["task"])), segment_rows=int(np.sum(seg)), generated=True), fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "merge"))
    ap.add_argument("--task", default="K6")
    ap.add_argument("--tasks", default="K1+K2+K3+K4+K5+K6")
    a = ap.parse_args()
    if a.mode == "gen":
        gen(a.task)
    else:
        merge(a.tasks.split("+"))

"""T6b (docs/SCAFFOLD_RESULTS.md 22; m3_prereg sc_t6b): T6's chains with the boundary rebuilt on hand state.  LIBERO venv, CPU, no training, U's checkpoints.
Boundary, both arms: at the release event, hold still with the command open until the fingers stop moving; rise RISE straight up.
    R: then a straight line to the start pose, until within NEAR (at most RET_MAX steps), then the second skill.
    D: the second skill starts at once, from wherever the hand is.
The last skill runs as U alone runs: until the goal or SKILL_MAX, with no stop at its release.  Clauses parse through S1's refusing Lexicon.

    python experiments/sc_t6b.py roll --index 0
    python experiments/sc_t6b.py read               -> results/flawed/sc_t6b_read.json
SC_SMOKE=1: 2 smoke states, results under sc_t6b_smoke.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import sc_t6 as T                                                           # noqa: E402

SMOKE = T.SMOKE
OD = os.path.join(T.OUT, "sc_t6b_smoke" if SMOKE else "sc_t6b")
ARMS = ("R", "D")
RISE, RET_MAX, OPEN_MAX, STILL, ELBOW_ON, ELBOW_OFF = 0.08, 150, 20, 0.001, 0.02, 0.25


def combos():
    if SMOKE:
        return [(a, n, 10, "familiar") for a, n in itertools.product(ARMS, T.CHAINS)]
    return list(itertools.product(ARMS, T.CHAINS, T.SEEDS, ("familiar", "new")))


def episode(model, bddl, state, clauses, seed_base, arm):
    import torch
    import fa1_common as FC
    import fa1b_common as FB
    import fa1c_common as FK
    from lifelong_ood.policy import factored_flow as FF
    from lifelong_ood.policy.parts_flow import ProprioHeld
    from lifelong_ood.scaffold.stores import Lexicon
    lex = Lexicon(model.role_lex["situations"])
    parsed = [lex.parse(c) for c in clauses]
    if any(p.reason is not None for p in parsed):
        return dict(success=False, error="refused", clauses=[])
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(state)
    dj = FC.drawer_joint(env)
    for _ in range(T.WAIT):
        obs, _, done, _ = env.step(FC.DUMMY)
    sc0 = FC.scene(env, obs, dj)
    ee0, R0 = sc0["ee_pos"].copy(), sc0["ee_rot"].copy()
    done, err, rep, calls = False, None, [], 0
    goals = env.env.parsed_problem["goal_state"]
    t0 = time.time()
    try:
        prev = -1.0
        for ci, p in enumerate(parsed):
            oe, te = p.object, p.target
            last = ci + 1 == len(parsed)
            oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
            con = FB.Contact(env, [oe])
            hm = FB.Held()
            det, cnp = ProprioHeld(model.pgate["tw"], model.pgate["lift"]), FK.Canon()

            def tgt():
                q = FC.scene(env, obs, dj)["pos"]
                return q[ti], float(q[FK.TABLE, 2])

            def gate(pv):
                sc = FC.scene(env, obs, dj)
                _, o = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
                hp = det.step(sc["grip"], pv, sc["ee_pos"][2])
                zp, cp = cnp.step(hp, con.pos(oe), tgt)
                return hp, o, zp, cp

            p_on, off, zr, cy = gate(prev)
            rot0, d0t, released, held_step, plan, t = None, None, False, None, [], 0
            while t < T.SKILL_MAX and not done and (last or not released):
                if not plan:
                    b = FF.batch_from_scene(FC.scene(env, obs, dj), oi, ti, prev, held=float(p_on), off=off, zrest=zr, cyaw=cy, rot0=rot0)
                    if bool(getattr(model, "tscale", False)) and d0t is not None:
                        b["d0t"] = torch.tensor([d0t], dtype=torch.float32)
                    g = torch.Generator().manual_seed(int(seed_base) + calls)
                    plan = [np.clip(x, -1.0, 1.0) for x in model.sample(b, generator=g)[0].numpy()[:T.REPLAN]]
                    calls += 1
                act = plan.pop(0)
                obs, _, done, _ = env.step(act.tolist())
                prev = float(act[6])
                was = p_on
                p_on, off, zr, cy = gate(prev)
                t += 1
                if p_on and not was:
                    if held_step is None:
                        held_step = t
                    rot0 = FC.scene(env, obs, dj)["ee_rot"].copy()
                    d0t = float(np.linalg.norm(np.asarray(con.pos(oe))[:2] - np.asarray(tgt()[0])[:2]))
                if was and not p_on and prev <= 0:
                    released = True
                if not p_on:
                    rot0, d0t = None, None
            rep.append(dict(oe=oe, te=te, steps=t, held_step=held_step, released=released))
            if last or done:
                continue
            rep[-1]["goal_at_release"] = bool(env.env._eval_predicate(goals[0]))
            w0 = None                                                       # the boundary: open, hold still, until the fingers stop
            for r in range(OPEN_MAX):
                sc = FC.scene(env, obs, dj)
                w = float(sc["grip"][0] - sc["grip"][1])
                if w0 is not None and abs(w - w0) < STILL:
                    break
                w0 = w
                obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
                if done:
                    break
            rep[-1]["open_steps"] = r
            z1 = FC.scene(env, obs, dj)["ee_pos"][2] + RISE
            for r in range(30):                                             # straight up
                ee = FC.scene(env, obs, dj)["ee_pos"]
                if done or ee[2] >= z1 - T.NEAR:
                    break
                obs, _, done, _ = env.step([0, 0, float(np.clip((z1 - ee[2]) / 0.02, -1, 1)), 0, 0, 0, -1.0])
            rep[-1]["rise_steps"] = r
            if arm == "R":
                r, arrived, unlock, unlocks = 0, False, False, 0
                while r < RET_MAX and not done:                             # a straight line in space, at the fastest speed the action allows
                    sc = FC.scene(env, obs, dj)
                    d = ee0 - sc["ee_pos"]
                    if np.linalg.norm(d) < T.NEAR:
                        arrived = True
                        break
                    if os.environ.get("SC_PROBE") and r % 15 == 0:          # smoke only: why a return stalls
                        sim = env.env.sim
                        q = sim.data.qpos[env.env.robots[0]._ref_joint_pos_indexes]
                        con_ = sorted({tuple(sorted((sim.model.geom_id2name(sim.data.contact[i].geom1) or "?", sim.model.geom_id2name(sim.data.contact[i].geom2) or "?"))) for i in range(sim.data.ncon)})
                        con_ = [c for c in con_ if any(k in c[0] + c[1] for k in ("robot0", "gripper0"))]
                        print("[probe]", r, "ee", np.round(sc["ee_pos"], 3).tolist(), "q", np.round(q, 2).tolist(), "contacts", con_, flush=True)
                    v = d / 0.02
                    a = np.zeros(7)
                    a[:3] = v / max(1.0, float(np.abs(v).max()))
                    rb = env.env.robots[0]                                  # the elbow's toggle lock: at its limit an inward pull presses it harder against the stop, so push outward until it has bent
                    q4 = float(env.env.sim.data.qpos[rb._ref_joint_pos_indexes[3]])
                    lim4 = float(env.env.sim.model.jnt_range[rb._ref_joint_indexes[3]][1])
                    if q4 > lim4 - ELBOW_ON:
                        unlock = True
                    elif q4 < lim4 - ELBOW_OFF:
                        unlock = False
                    if unlock:
                        u = (sc["ee_pos"] - np.asarray(rb.base_pos))[:2]
                        a[:3] = np.r_[u / np.linalg.norm(u), 0.0]
                        unlocks += 1
                    a[3:6] = np.clip(T.rotvec(R0 @ sc["ee_rot"].T) / 0.2, -1, 1)
                    a[6] = -1.0
                    obs, _, done, _ = env.step(a.tolist())
                    r += 1
                rep[-1]["return_steps"], rep[-1]["arrived"], rep[-1]["unlock_steps"] = r, arrived, unlocks
            prev = -1.0
            rep[-1]["first_goal_at_transition"] = bool(env.env._eval_predicate(goals[0]))
        out_goals = [bool(env.env._eval_predicate(g)) for g in goals]
    except Exception as ex:                                                 # noqa: BLE001
        err, out_goals = repr(ex), []
    env.close()
    return dict(success=bool(done), error=err, clauses=rep, goals=out_goals, secs=time.time() - t0)


def roll(arm, name, seed, lay):
    import torch
    import fa1_common as FC
    from pp_p_roll import load_any
    n2 = T.tasks()
    t = T.task(n2, T.CHAINS[name][0], lay)
    fn = os.path.join(OD, f"{arm}_{name}_s{seed}_{lay}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc_U_s{seed}", "P.npz"))
    S = np.load(os.path.join(T.OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    br = json.load(open(os.path.join(T.OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:3]
    eps = []
    for b in br:
        clauses = [T.task(n2, c, lay)["language"] for c in T.CHAINS[name]]
        e = episode(m, os.path.join(T.BD, name + ".bddl"), S[int(b["trial"])], clauses, b["seed_base"], arm)
        eps.append(dict(trial=int(b["trial"]), **e))
        print("[t6b]", arm, name, seed, lay, b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(arm=arm, name=name, seed=seed, layout=lay, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = T.tasks()
    ks = n2["states"]["main"]

    def load(arm, n, s, l):
        E = json.load(open(os.path.join(OD, f"{arm}_{n}_s{s}_{l}.json"), encoding="utf-8"))["eps"]
        for e in E:
            assert e.get("error") is None, (n, e.get("error"))
        d = {int(e["trial"]): e for e in E}
        return [d[k] for k in ks]

    def u(c, s, l):
        E = json.load(open(os.path.join(T.OUT, "sc_t4", f"U_s{s}_{T.task(n2, c, l)['tid']}.json"), encoding="utf-8"))["eps"]
        d = {int(e["trial"]): e for e in E}
        return np.array([float(d[k]["success"]) for k in ks])

    names = list(T.CHAINS)

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(T.B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    res = dict(V={}, per_chain={})
    arr = {}
    for arm, l in itertools.product(ARMS, ("familiar", "new")):
        ch = np.array([[[float(e["success"]) for e in load(arm, n, s, l)] for s in T.SEEDS] for n in names])
        pr = np.array([[u(T.CHAINS[n][0], s, l) * u(T.CHAINS[n][1], s, l) for s in T.SEEDS] for n in names])
        arr[arm, l] = ch
        res["V"][f"{arm}|{l}"] = diff(ch, pr)
        for i, n in enumerate(names):
            E = [e for s in T.SEEDS for e in load(arm, n, s, l)]
            c0 = [e["clauses"][0] for e in E]
            res["per_chain"][f"{arm}|{n}|{l}"] = dict(
                chain=float(ch[i].mean()), product=float(pr[i].mean()),
                first_released=float(np.mean([c["released"] for c in c0])),
                first_goal_at_release=float(np.mean([bool(c.get("goal_at_release")) for c in c0])),
                first_goal_at_transition=float(np.mean([bool(c.get("first_goal_at_transition")) for c in c0])),
                arrived=float(np.mean([bool(c.get("arrived")) for c in c0])), return_steps=float(np.median([c.get("return_steps", 0) for c in c0])),
                second_held=float(np.mean([len(e["clauses"]) > 1 and e["clauses"][1]["held_step"] is not None for e in E])),
                first_goal_end=float(np.mean([len(e["goals"]) > 0 and e["goals"][0] for e in E])), second_goal_end=float(np.mean([len(e["goals"]) > 1 and e["goals"][1] for e in E])))
    res["D_minus_R"] = {l: diff(arr["D", l], arr["R", l]) for l in ("familiar", "new")}
    lab = lambda x: "BREAKS" if x["ci"][1] < -T.MARGIN else ("HOLDS" if x["ci"][0] >= -T.MARGIN else "UNRESOLVED")     # noqa: E731
    res["label"] = {arm: lab(res["V"][f"{arm}|familiar"]) for arm in ARMS}
    print("[t6b]", json.dumps(res), flush=True)
    with open(os.path.join(T.OUT, "sc_t6b_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        arm_, n_, s_, l_ = combos()[a.index]
        roll(arm_, n_, s_, l_)
    else:
        read()

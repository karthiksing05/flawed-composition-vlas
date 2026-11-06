"""T14 (docs/SCAFFOLD_RESULTS.md 27; m3_prereg sc_t14): chains that mix verbs.  T11's Um checkpoints, T13's mode per clause, T6b's boundary R.  LIBERO venv, CPU, no training.
A close clause ends at its release.  An open clause ends once its object has moved MV_ON from where the clause began and then moved under MV_STILL over MV_GAP steps.  The last clause runs to the goal or SKILL_MAX.

    python experiments/sc_t14.py make               writes results/flawed/sc_t14_bddl/*.bddl
    python experiments/sc_t14.py roll --index 0
    python experiments/sc_t14.py read               -> results/flawed/sc_t14_read.json
SC_SMOKE=1: 2 smoke states, M1, M3, D1, seed 10, familiar, results under sc_t14_smoke.
"""
import argparse
import itertools
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import sc_t6 as T                                                           # noqa: E402
import sc_t6b as B                                                          # noqa: E402

SMOKE = T.SMOKE
OD = os.path.join(T.OUT, "sc_t14_smoke" if SMOKE else "sc_t14")
BD = os.path.join(T.OUT, "sc_t14_bddl")
MODES = os.path.join(T.OUT, "sc_t12_modes.json")
HORIZON = "ValueError('executing action in terminated episode')"
VERDICT = dict(M1=("K8", "K2"), M2=("K8", "N3"), M3=("N8", "N1"), M4=("K4", "K8"), M5=("K4", "N8"), M6=("N3", "N8"))
DEPENDENT = dict(D1=("K8", "K1"), D2=("K8", "N1"))
CHAINS = {**VERDICT, **DEPENDENT}


def make():
    n2 = T.tasks()
    os.makedirs(BD, exist_ok=True)
    for name, cs in CHAINS.items():
        ps = [os.path.join(ROOT, T.task(n2, c, "familiar")["bddl"]) for c in cs]
        gs = [T.goal_of(p) for p in ps]
        new, n = re.subn(r"\(:goal\s*\(And\s*.*?\)\s*\)\s*\)", "(:goal\n    (And " + " ".join(gs) + ")\n  )", open(ps[0], encoding="utf-8").read(), flags=re.S)
        assert n == 1, name
        with open(os.path.join(BD, name + ".bddl"), "w", encoding="utf-8") as fh:
            fh.write(new)
        print(name, gs, flush=True)


def combos():
    if SMOKE:
        return [(n, 10, "familiar") for n in ("M1", "M3", "D1")]
    return list(itertools.product(CHAINS, T.SEEDS, ("familiar", "new")))


def episode(model, bddl, state, clauses, seed_base, end="still", rho=None):
    """end (SCAFFOLD_RESULTS 28), an open clause, besides going still: "near" also ends it when the object is within rho of its target; "goal" also ends it on its own goal predicate."""
    import torch
    import fa1_common as FC
    import fa1b_common as FB
    import fa1c_common as FK
    from lifelong_ood.policy import factored_flow as FF
    from lifelong_ood.policy import parts_flow as PF
    words = PF.verb_words(json.load(open(MODES, encoding="utf-8")), {w for w, p in model.role_lex["pg"].items() if p > 0})
    parsed = [PF.role_parse(model.role_lex, c) for c in clauses]
    verbs = [PF.verb_parse(words, c) for c in clauses]
    if any(p is None for p in parsed) or any(v is None for v in verbs):
        return dict(success=False, error="refused", clauses=[], goals=[])
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
        for ci, (oe, te) in enumerate(parsed):
            verb = verbs[ci]
            last = ci + 1 == len(parsed)
            oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
            con = FB.Contact(env, [oe])
            hm = FB.Held()
            det, cnp = PF.ProprioHeld(model.pgate["tw"], model.pgate["lift"]), FK.Canon()
            sc = FC.scene(env, obs, dj)
            mv, p0, hist, moved = PF.HandMoved(sc["pos"][oi]), np.asarray(sc["pos"][oi], float).copy(), [], False
            arrive = None

            def tgt():
                q = FC.scene(env, obs, dj)["pos"]
                return q[ti], float(q[FK.TABLE, 2])

            def gate(pv):
                sc = FC.scene(env, obs, dj)
                _, o = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
                hp = det.step(sc["grip"], pv, sc["ee_pos"][2])
                hp = bool(mv.step(sc["pos"][oi], pv)) or hp or arrive is not None
                zp, cp = cnp.step(hp, con.pos(oe), tgt)
                return hp, o, zp, cp

            p_on, off, zr, cy = gate(prev)
            rot0, d0t, ended, held_step, plan, t = None, None, False, None, [], 0
            while t < T.SKILL_MAX and not done and (last or not ended):
                if not plan:
                    b = FF.batch_from_scene(FC.scene(env, obs, dj), oi, ti, prev, held=float(p_on), off=off, zrest=zr, cyaw=cy, rot0=rot0)
                    if bool(getattr(model, "tscale", False)) and d0t is not None:
                        b["d0t"] = torch.tensor([d0t], dtype=torch.float32)
                    g = torch.Generator().manual_seed(int(seed_base) + calls)
                    plan = [np.clip(x, -1.0, 1.0) for x in model.sample(b, generator=g)[0].numpy()[:T.REPLAN]]
                    calls += 1
                act = plan.pop(0)
                if verb == "open":
                    if act[6] > 0 and arrive is None:
                        arrive = t
                    act = act.copy()
                    act[6] = -1.0
                det_was = det.on
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
                if not p_on:
                    rot0, d0t = None, None
                if verb == "close":
                    ended = det_was and not det.on and prev <= 0
                else:
                    p = np.asarray(FC.scene(env, obs, dj)["pos"][oi], float)
                    moved = moved or float(np.linalg.norm(p - p0)) > PF.MV_ON
                    hist = (hist + [p])[-(PF.MV_GAP + 1):]
                    ended = moved and len(hist) > PF.MV_GAP and float(np.linalg.norm(p - hist[0])) < PF.MV_STILL
                    if end == "nearonly":                                   # SCAFFOLD_RESULTS 29: no end by stillness
                        ended = not last and moved and float(np.linalg.norm(p[:2] - np.asarray(tgt()[0])[:2])) <= rho
                    elif not last and end == "near":
                        ended = ended or (moved and float(np.linalg.norm(p[:2] - np.asarray(tgt()[0])[:2])) <= rho)
                    elif not last and end == "goal":
                        ended = ended or bool(env.env._eval_predicate(goals[ci]))
            rep.append(dict(oe=oe, te=te, verb=verb, steps=t, held_step=held_step, arrive=arrive, ended=bool(ended)))
            if last or done:
                continue
            rep[-1]["goal_at_end"] = bool(env.env._eval_predicate(goals[ci]))
            w0 = None
            for _ in range(B.OPEN_MAX):
                sc = FC.scene(env, obs, dj)
                w = float(sc["grip"][0] - sc["grip"][1])
                if w0 is not None and abs(w - w0) < B.STILL:
                    break
                w0 = w
                obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
                if done:
                    break
            z1 = FC.scene(env, obs, dj)["ee_pos"][2] + B.RISE
            for _ in range(30):
                ee = FC.scene(env, obs, dj)["ee_pos"]
                if done or ee[2] >= z1 - T.NEAR:
                    break
                obs, _, done, _ = env.step([0, 0, float(np.clip((z1 - ee[2]) / 0.02, -1, 1)), 0, 0, 0, -1.0])
            r, arrived, unlock = 0, False, False
            while r < B.RET_MAX and not done:
                sc = FC.scene(env, obs, dj)
                d = ee0 - sc["ee_pos"]
                if np.linalg.norm(d) < T.NEAR:
                    arrived = True
                    break
                v = d / 0.02
                a = np.zeros(7)
                a[:3] = v / max(1.0, float(np.abs(v).max()))
                rb = env.env.robots[0]
                q4 = float(env.env.sim.data.qpos[rb._ref_joint_pos_indexes[3]])
                lim4 = float(env.env.sim.model.jnt_range[rb._ref_joint_indexes[3]][1])
                if q4 > lim4 - B.ELBOW_ON:
                    unlock = True
                elif q4 < lim4 - B.ELBOW_OFF:
                    unlock = False
                if unlock:
                    u = (sc["ee_pos"] - np.asarray(rb.base_pos))[:2]
                    a[:3] = np.r_[u / np.linalg.norm(u), 0.0]
                a[3:6] = np.clip(T.rotvec(R0 @ sc["ee_rot"].T) / 0.2, -1, 1)
                a[6] = -1.0
                obs, _, done, _ = env.step(a.tolist())
                r += 1
            rep[-1]["return_steps"], rep[-1]["arrived"] = r, arrived
            rep[-1]["goal_at_transition"] = bool(env.env._eval_predicate(goals[ci]))
            prev = -1.0
        out_goals = [bool(env.env._eval_predicate(g)) for g in goals]
    except Exception as ex:                                                 # noqa: BLE001
        err, out_goals = repr(ex), []
    env.close()
    return dict(success=bool(done), error=err, clauses=rep, goals=out_goals, secs=time.time() - t0)


def roll(name, seed, lay):
    import torch
    import fa1_common as FC
    from pp_p_roll import load_any
    n2 = T.tasks()
    t = T.task(n2, CHAINS[name][0], lay)
    fn = os.path.join(OD, f"{name}_s{seed}_{lay}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc11_Um_s{seed}", "P.npz"))
    S = np.load(os.path.join(T.OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    br = json.load(open(os.path.join(T.OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        clauses = [T.task(n2, c, lay)["language"] for c in CHAINS[name]]
        e = episode(m, os.path.join(BD, name + ".bddl"), S[int(b["trial"])], clauses, b["seed_base"])
        eps.append(dict(trial=int(b["trial"]), **e))
        print("[t14]", name, seed, lay, b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(name=name, seed=seed, layout=lay, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = T.tasks()
    ks = n2["states"]["main"]

    def load(n, s, l):
        E = json.load(open(os.path.join(OD, f"{n}_s{s}_{l}.json"), encoding="utf-8"))["eps"]
        for e in E:
            assert e.get("error") in (None, HORIZON), (n, e.get("error"))
            assert not (e.get("error") == HORIZON and e["success"]), n
        d = {int(e["trial"]): e for e in E}
        return [d[k] for k in ks]

    def alone(c, s, l):                                                     # a push alone: T12's V; a grasp alone: T11's Um
        tid = T.task(n2, c, l)["tid"]
        p = ("sc_t12", "V") if c in ("K8", "N8") else ("sc_t11", "Um")
        E = json.load(open(os.path.join(T.OUT, p[0], f"{p[1]}_s{s}_{tid}.json"), encoding="utf-8"))["eps"]
        d = {int(e["trial"]): e for e in E}
        return np.array([float(d[k]["success"]) for k in ks])

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(T.B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    def group(names, l):
        ch = np.array([[[float(e["success"]) for e in load(n, s, l)] for s in T.SEEDS] for n in names])
        pr = np.array([[alone(CHAINS[n][0], s, l) * alone(CHAINS[n][1], s, l) for s in T.SEEDS] for n in names])
        return ch, pr

    res = dict(V={}, dependent={}, per_chain={})
    for l in ("familiar", "new"):
        res["V"][l] = diff(*group(list(VERDICT), l))
        res["dependent"][l] = diff(*group(list(DEPENDENT), l))
        for n in CHAINS:
            ch, pr = group([n], l)
            E = [e for s in T.SEEDS for e in load(n, s, l)]
            res["per_chain"][f"{n}|{l}"] = dict(chain=float(ch.mean()), product=float(pr.mean()), past_horizon=float(np.mean([e.get("error") == HORIZON for e in E])),
                                                alone=[float(np.mean([alone(c, s, l).mean() for s in T.SEEDS])) for c in CHAINS[n]],
                                                first_ended=float(np.mean([len(e["clauses"]) > 0 and e["clauses"][0]["ended"] for e in E])),
                                                first_goal_at_transition=float(np.mean([len(e["clauses"]) > 0 and bool(e["clauses"][0].get("goal_at_transition")) for e in E])),
                                                goal_end=[float(np.mean([len(e["goals"]) > j and e["goals"][j] for e in E])) for j in range(2)])
    x = res["V"]["familiar"]
    res["label"] = "BREAKS" if x["ci"][1] < -T.MARGIN else ("HOLDS" if x["ci"][0] >= -T.MARGIN else "UNRESOLVED")
    print("[t14]", json.dumps(res), flush=True)
    with open(os.path.join(T.OUT, "sc_t14_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("make", "roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "make":
        make()
    elif a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        roll(*combos()[a.index])
    else:
        read()

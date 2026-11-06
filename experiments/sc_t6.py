"""T6 (docs/SCAFFOLD_RESULTS.md 17; m3_prereg sc_t6): two-skill chains no demo shows.  LIBERO venv, CPU, no training, U's checkpoints.
A clause-level schema: split the instruction at " then ", parse each clause by the library's lexicon, run the first skill until the release event (hold on, then off with the command open),
open and rise and return to the start pose, run the second skill.  At most 300 steps per skill.  Success: both goals true at once, by a BDDL goal joined from the two tasks' goals.
Also U on the drawer and push tasks, descriptive.

    python experiments/sc_t6.py make               writes results/flawed/sc_t6_bddl/*.bddl
    python experiments/sc_t6.py roll --index 0
    python experiments/sc_t6.py read               -> results/flawed/sc_t6_read.json
SC_SMOKE=1: 2 smoke states, 3 combos, results under sc_t6_smoke.
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
OUT = os.path.join(ROOT, "results", "flawed")
SMOKE = bool(os.environ.get("SC_SMOKE"))
OD = os.path.join(OUT, "sc_t6_smoke" if SMOKE else "sc_t6")
BD = os.path.join(OUT, "sc_t6_bddl")
SEEDS = (10, 11, 12)
CHAINS = dict(C1=("K1", "N2"), C2=("K4", "K1"), C3=("K2", "K4"), C4=("N1", "K3"), C5=("N4", "N2"), C6=("N3", "K2"))
SEQ = ("K7", "N7", "K8", "N8")
SKILL_MAX, RET_MAX, NEAR, REPLAN, WAIT, B, MARGIN = 300, 60, 0.02, 5, 10, 2000, 0.10


def tasks():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def task(n2, c, l):
    return next(t for t in n2["tasks"] if t["compound"] == c and t["layout"] == l)


def goal_of(path):
    return re.search(r"\(:goal\s*\(And\s*(.*?\))\s*\)\s*\)", open(path, encoding="utf-8").read(), re.S).group(1).strip()


def make():
    n2 = tasks()
    os.makedirs(BD, exist_ok=True)
    for name, (a, b) in CHAINS.items():
        ta, tb = task(n2, a, "familiar"), task(n2, b, "familiar")
        pa, pb = os.path.join(ROOT, ta["bddl"]), os.path.join(ROOT, tb["bddl"])
        txt = open(pa, encoding="utf-8").read()
        new, n = re.subn(r"\(:goal\s*\(And\s*.*?\)\s*\)\s*\)", "(:goal\n    (And " + goal_of(pa) + " " + goal_of(pb) + ")\n  )", txt, flags=re.S)
        assert n == 1, name
        with open(os.path.join(BD, name + ".bddl"), "w", encoding="utf-8") as fh:
            fh.write(new)
        print(name, goal_of(pa), goal_of(pb), flush=True)


def combos():
    if SMOKE:
        return [("C1", 10, "familiar"), ("C5", 10, "new"), ("K7", 10, "familiar")]
    return [(c, s, l) for c, s, l in itertools.product(list(CHAINS) + list(SEQ), SEEDS, ("familiar", "new"))]


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = float(np.arccos(c))
    if th < 1e-6:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def chain_episode(model, bddl, state, clauses, seed_base):
    import torch
    import fa1_common as FC
    import fa1b_common as FB
    import fa1c_common as FK
    from lifelong_ood.policy import factored_flow as FF
    from lifelong_ood.policy.parts_flow import ProprioHeld, role_parse
    parsed = [role_parse(model.role_lex, c) for c in clauses]
    if any(p is None for p in parsed):
        return dict(success=False, error="no induced roles", clauses=[])
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(state)
    dj = FC.drawer_joint(env)
    for _ in range(WAIT):
        obs, _, done, _ = env.step(FC.DUMMY)
    sc0 = FC.scene(env, obs, dj)
    ee0, R0 = sc0["ee_pos"].copy(), sc0["ee_rot"].copy()
    done, err, rep, calls = False, None, [], 0
    goals = env.env.parsed_problem["goal_state"]
    t0 = time.time()
    try:
        prev = -1.0
        for ci, (oe, te) in enumerate(parsed):
            oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
            con = FB.Contact(env, [oe])
            hm = FB.Held()
            det, cnp = ProprioHeld(model.pgate["tw"], model.pgate["lift"]), FK.Canon()

            def tgt():
                p = FC.scene(env, obs, dj)["pos"]
                return p[ti], float(p[FK.TABLE, 2])

            def gate(pv):
                sc = FC.scene(env, obs, dj)
                _, o = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
                hp = det.step(sc["grip"], pv, sc["ee_pos"][2])
                zp, cp = cnp.step(hp, con.pos(oe), tgt)
                return hp, o, zp, cp

            p_on, off, zr, cy = gate(prev)
            rot0, d0t, ever, released, held_step, plan, t = None, None, False, False, None, [], 0
            while t < SKILL_MAX and not done and not released:
                if not plan:
                    b = FF.batch_from_scene(FC.scene(env, obs, dj), oi, ti, prev, held=float(p_on), off=off, zrest=zr, cyaw=cy, rot0=rot0)
                    if bool(getattr(model, "tscale", False)) and d0t is not None:
                        b["d0t"] = torch.tensor([d0t], dtype=torch.float32)
                    g = torch.Generator().manual_seed(int(seed_base) + calls)
                    plan = [np.clip(x, -1.0, 1.0) for x in model.sample(b, generator=g)[0].numpy()[:REPLAN]]
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
                    ever = True
                    rot0 = FC.scene(env, obs, dj)["ee_rot"].copy()
                    d0t = float(np.linalg.norm(np.asarray(con.pos(oe))[:2] - np.asarray(tgt()[0])[:2]))
                if was and not p_on and prev <= 0:
                    released = True
                if not p_on:
                    rot0, d0t = None, None
            rep.append(dict(oe=oe, te=te, steps=t, held_step=held_step, released=released, goal_true_at_end=None))
            if ci + 1 < len(parsed) and not done:
                r = 0
                while r < RET_MAX and not done:                             # the schema's transition: open, up, back to the start pose
                    sc = FC.scene(env, obs, dj)
                    ee = sc["ee_pos"]
                    if np.linalg.norm(ee - ee0) < NEAR and sc["grip"][0] - sc["grip"][1] > 0:
                        break
                    wp = np.array([ee[0], ee[1], ee0[2]]) if ee[2] < ee0[2] - NEAR else ee0
                    a = np.zeros(7)
                    a[:3] = np.clip((wp - ee) / 0.02, -1, 1)
                    a[3:6] = np.clip(rotvec(R0 @ sc["ee_rot"].T) / 0.2, -1, 1)
                    a[6] = -1.0
                    obs, _, done, _ = env.step(a.tolist())
                    prev = -1.0
                    r += 1
                rep[-1]["return_steps"] = r
                rep[-1]["first_goal_at_transition"] = bool(env.env._eval_predicate(goals[0]))
        out_goals = [bool(env.env._eval_predicate(g)) for g in goals]
    except Exception as ex:                                                 # noqa: BLE001
        err, out_goals = repr(ex), []
    env.close()
    return dict(success=bool(done), error=err, clauses=rep, goals=out_goals, secs=time.time() - t0)


def roll(name, seed, lay):
    import torch
    import fa1_common as FC
    from fa1h_roll import episode
    from pp_p_roll import load_any
    n2 = tasks()
    first = CHAINS[name][0] if name in CHAINS else name
    t = task(n2, first, lay)
    fn = os.path.join(OD, f"{name}_s{seed}_{lay}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc_U_s{seed}", "P.npz"))
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    br = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        if name in CHAINS:
            clauses = [task(n2, c, lay)["language"] for c in CHAINS[name]]
            e = chain_episode(m, os.path.join(BD, name + ".bddl"), S[int(b["trial"])], " then ".join(clauses).split(" then "), b["seed_base"])
            eps.append(dict(trial=int(b["trial"]), **e))
        else:
            e = episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
            eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], held_step=e["held_step"], steps=e["steps"], roles=e.get("roles")))
        print("[t6]", name, seed, lay, b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(name=name, seed=seed, layout=lay, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = tasks()
    ks = n2["states"]["main"]

    def load(name, s, l):
        E = json.load(open(os.path.join(OD, f"{name}_s{s}_{l}.json"), encoding="utf-8"))["eps"]
        for e in E:
            assert e.get("error") is None, (name, e.get("error"))
        d = {int(e["trial"]): e for e in E}
        return [d[k] for k in ks]

    def u(c, s, l):
        E = json.load(open(os.path.join(OUT, "sc_t4", f"U_s{s}_{task(n2, c, l)['tid']}.json"), encoding="utf-8"))["eps"]
        d = {int(e["trial"]): e for e in E}
        return np.array([float(d[k]["success"]) for k in ks])

    names = list(CHAINS)

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    res = dict(V={}, per_chain={}, seq={})
    for l in ("familiar", "new"):
        ch = np.array([[[float(e["success"]) for e in load(n, s, l)] for s in SEEDS] for n in names])
        pr = np.array([[u(CHAINS[n][0], s, l) * u(CHAINS[n][1], s, l) for s in SEEDS] for n in names])
        res["V"][l] = diff(ch, pr)
        for i, n in enumerate(names):
            E = [e for s in SEEDS for e in load(n, s, l)]
            res["per_chain"][f"{n}|{l}"] = dict(chain=float(ch[i].mean()), product=float(pr[i].mean()),
                                                first_alone=float(np.mean([u(CHAINS[n][0], s, l).mean() for s in SEEDS])), second_alone=float(np.mean([u(CHAINS[n][1], s, l).mean() for s in SEEDS])),
                                                first_goal_at_transition=float(np.mean([bool(e["clauses"][0].get("first_goal_at_transition")) for e in E])),
                                                first_released=float(np.mean([e["clauses"][0]["released"] for e in E])),
                                                second_held=float(np.mean([len(e["clauses"]) > 1 and e["clauses"][1]["held_step"] is not None for e in E])),
                                                second_goal_end=float(np.mean([len(e["goals"]) > 1 and e["goals"][1] for e in E])), first_goal_end=float(np.mean([len(e["goals"]) > 0 and e["goals"][0] for e in E])))
        for c in SEQ:
            E = [e for s in SEEDS for e in json.load(open(os.path.join(OD, f"{c}_s{s}_{l}.json"), encoding="utf-8"))["eps"]]     # descriptive; a lexicon's refusal is a result here (amended after the first read crashed on K8, no verdict depends on it)
            assert all(e.get("error") in (None, "no induced roles") for e in E), c
            res["seq"][f"{c}|{l}"] = dict(success=float(np.mean([e["success"] for e in E])), refused=float(np.mean([e.get("error") == "no induced roles" for e in E])), n=len(E))
    x = res["V"]["familiar"]
    res["label"] = "BREAKS" if x["ci"][1] < -MARGIN else ("HOLDS" if x["ci"][0] >= -MARGIN else "UNRESOLVED")
    print("[t6]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_t6_read.json"), "w", encoding="utf-8") as fh:
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
        n_, s_, l_ = combos()[a.index]
        roll(n_, s_, l_)
    else:
        read()

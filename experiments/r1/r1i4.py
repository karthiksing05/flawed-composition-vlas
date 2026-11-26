"""I4 (parts over time on a robot policy).  Registration: compositional-architecture results/tables/compose_stream_prediction.json,
addendum interp_I45_registered; spec docs/PLAN_INTERP_I45.md v3 (commit 62c80db), section I4 and "Round 2 findings".
  run --seed S --group G [--eps_mult M] [--smoke] [--out DIR]   one job per (seed, group)
        -> {OUT}/s{S}/{G}.json  (per cell, per arm: success, steps, on_t, off_t)   and  {OUT}/s{S}/{G}.npz  (chunk / step / episode logs)
  Offline activity pass: r1i4_activity.py.  Scorer: r1i4_score.py.
OUT = env I4_OUT or --out, default results/r1/i4; --smoke writes to results/r1/i4_smoke (1 cell, 2 episodes) unless overridden.

Literal-reading choices (the final report lists them too):
 - Arms per cell (derived from the plan's episode counts none 1485 / grasp 1165 / scramble 855 / C3 90):
     none      every cell (all routed parts every step; it steps the segmenter and logs onset/segments, gating nothing)
     grasp     cells where some routed part is gated (ASSIGN != ALL) + cell 0 of every group (C1 needs grasp on ungated cell 0)
     scramble  cells where the gated parts are exactly {OW}  (single_OW and pair_HBOW all seeds, pair_OWTC seeds 0-1)
     none2     second none run, pair_OWTC cell 0 only (C3)
 - Noise: per episode, torch Generator.manual_seed(e) before the arm runs (episode 0 = T1's stream); env seed np.random.seed(e) as T1
   (seed 0 * 100003 + e).  One x0 per replan (t = 0, 8, ...), drawn from that generator exactly as SegRouted.sample does.
 - Sampler = SegRouted.sample re-implemented step for step (same operations, same order) so the path can be logged; the
   active set is [k for k in newvals(cell) if segment in ASSIGN[k]] (none: all routed parts).
 - Gated chunk = a chunk start at which the grasp rule's active set differs from all routed parts.
 - Scramble: in each gated chunk both the gated set and the all-on set are integrated from the SAME x0; executed action (action space,
   pre-clip, the first 8 of the chunk) = a_grasp on unflipped DOFs, a_none + (a_none - a_grasp) on flipped DOFs.  Signs: one per arm DOF
   (actions 0-6) per episode from numpy default_rng([20261128, seed, group index, cell index, e]); the gripper (action 7) is never flipped;
   an all-(+1) draw is redrawn.  Flip drawn for every scramble episode, gated or not.
 - Logged chunk path = x_0..x_9, the 10 states at which the velocity is evaluated (x_10 is the chunk output).  For scramble gated
   chunks the path is the gated (grasp-rule) integration; both pre-clip action chunks are logged (alt_none, alt_grasp).
 - eef = stripped state [66:69] (proprio 45:77, eef pos 21:24), object = [0:3], both at the observation the action was chosen from.
 - contact flag = env.env._check_grasp(...) (the oracle raw held flag), per step; segment = the proprio SegState segment after update.
 - --eps_mult M multiplies EPS for single_OW and the three pair groups only (full run M = 3 gives single_OW 15, pairs 30); single_HB and
   single_TC stay at 5.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/coc/flash10/ksingara3/flawed-composition-vlas/experiments/r1")
import r1, r1s, r1seg, r1t1
from r1s import *  # noqa (np, torch, b1, b1_data, b1_eval, TASKS, GROUPS, EPS, RES, newvals, dropped, H)
from r1seg import ProprioHeld
from r1t1 import SegRouted, SegState, ASSIGN, ALL, tw_value

ARMS = ["none", "grasp", "scramble", "none2"]
SCALED = ("single_OW", "pair_HBOW", "pair_HBTC", "pair_OWTC")
EEF, OBJ = (66, 69), (0, 3)
CLIPLO = None


def out_dir(a):
    if a.out: return a.out
    if os.environ.get("I4_OUT"): return os.environ["I4_OUT"]
    return RES + ("/i4_smoke" if a.smoke else "/i4")


@torch.no_grad()
def flow(model, state, x0, act, task, steps=10):
    """SegRouted.sample step for step, from a given x0; returns (final x, path (steps, H, 8) = x_0..x_{steps-1})"""
    x = x0.clone(); path = []
    for i in range(steps):
        path.append(x)
        t = torch.full((state.shape[0],), i / steps)
        v = model.base.velocity(x, t, state, task)
        if act:
            d = torch.stack([model.parts[k].velocity(x, t, state, task) - v for k in act])
            v = v + d.max(0).values.clamp(min=0) + d.min(0).values.clamp(max=0)
        x = x + v / steps
    return x, torch.cat(path, 0)


def flip_signs(seed, gi, ci, e):
    rng = np.random.default_rng([20261128, seed, gi, ci, e])
    while True:
        s = rng.choice([1, -1], 7)
        if (s == -1).any():
            break
    return np.concatenate([s, [1]]).astype(np.int8)


def run_episode(model, env, cell, arm, e, tw, signs, rec):
    assign, norm = model.assign, model.norm
    on_all = newvals(cell)
    task = torch.zeros(1, dtype=torch.long)
    flipmask = signs == -1 if signs is not None else None
    np.random.seed(e)
    obs = b1_eval._obs(env.reset())
    gen = torch.Generator().manual_seed(e)
    det, st = ProprioHeld(tw), SegState()
    ep = len(rec["ep"])
    ok, t, plan, done, cmd, max_steps = False, 0, [], False, 0.0, 500
    nan88 = np.full((8, 8), np.nan, np.float32)
    while t < max_steps and not done:
        s = b1_data.strip_task_id(np.asarray(obs, np.float32)[None], cell, check=(t == 0))
        seg = st.update(det.step(s[0, r1seg.GRIP[0]:r1seg.GRIP[1]], cmd, s[0, r1seg.ZIDX]), t)
        contact = bool(env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=[g for g in env.env.object.contact_geoms]))
        if not plan:
            sn = norm.state(s).astype(np.float32)
            stt = torch.tensor(sn, dtype=torch.float32)
            x0 = torch.randn(1, H, 8, generator=gen)
            gated_set = [k for k in on_all if seg in assign[k]]
            act = list(on_all) if arm in ("none", "none2") else gated_set
            gated = act != list(on_all)
            alt_n, alt_g = nan88, nan88
            if arm == "scramble" and gated:
                xg, path = flow(model, stt, x0, gated_set, task)
                xn, _ = flow(model, stt, x0, list(on_all), task)
                ag = norm.unact(xg[0].numpy())[:8]; an = norm.unact(xn[0].numpy())[:8]
                aex = np.where(flipmask[None, :], an + (an - ag), ag)
                plan = list(aex); alt_n, alt_g = an.astype(np.float32), ag.astype(np.float32)
            else:
                xx, path = flow(model, stt, x0, act, task)
                plan = list(norm.unact(xx[0].numpy())[:8])
            rec["ch"].append(dict(ep=ep, t=t, state=sn[0], x0=x0[0].numpy(), path=path.numpy(), active=[k in act for k in TASKS], gated=gated,
                                  seg=seg, contact=contact, plan_pre=np.stack(plan[:8]).astype(np.float32), alt_none=alt_n, alt_grasp=alt_g))
        a_pre = plan.pop(0)
        a = np.clip(a_pre, env.action_space.low, env.action_space.high)
        cmd = float(a[r1seg.CMD])
        step = dict(ep=ep, t=t, pre=np.asarray(a_pre, np.float32), post=np.asarray(a, np.float64), eef=s[0, EEF[0]:EEF[1]].copy(),
                    obj=s[0, OBJ[0]:OBJ[1]].copy(), contact=contact, seg=seg)
        out = env.step(a)
        obs, done = b1_eval._obs(out[0]), bool(out[2])
        t += 1
        if env.env._check_success():
            ok = True
        step["succ"] = ok
        rec["sp"].append(step)
        if ok:
            break
    rec["ep"].append(dict(arm=ARMS.index(arm), cell=rec["cur_cell"], e=e, success=bool(ok), steps=t,
                          on_t=-1 if st.on_t is None else st.on_t, off_t=-1 if st.off_t is None else st.off_t,
                          flip=signs if signs is not None else np.zeros(8, np.int8)))
    return ok, t, st.on_t, st.off_t


def pack(rec):
    ch, sp, ep = rec["ch"], rec["sp"], rec["ep"]
    z = {}
    for k in ("arm", "cell", "e", "success", "steps", "on_t", "off_t", "flip"):
        z["ep_" + k] = np.array([x[k] for x in ep])
    for k in ("ep", "t", "state", "x0", "path", "active", "gated", "seg", "contact", "plan_pre", "alt_none", "alt_grasp"):
        z["ch_" + k] = np.array([x[k] for x in ch])
    for k in ("ep", "t", "pre", "post", "eef", "obj", "contact", "seg", "succ"):
        z["sp_" + k] = np.array([x[k] for x in sp])
    return z


def cmd_run(a):
    torch.set_num_threads(1)
    OUT = out_dir(a)
    r1s.S = f"{RES}/tokctl/s{a.seed}"
    model = SegRouted(ASSIGN[a.seed])
    assign = ASSIGN[a.seed]
    gi = list(GROUPS).index(a.group)
    cells = [c for c in GROUPS[a.group] if c not in dropped()]
    if a.smoke:
        cells = cells[:1]
    n_ep = 2 if a.smoke else EPS[a.group] * (a.eps_mult if a.group in SCALED else 1)
    tw = tw_value()
    rec = dict(ch=[], sp=[], ep=[], cur_cell=0)
    res = dict(seed=a.seed, group=a.group, n_ep=n_ep, smoke=bool(a.smoke), arms=ARMS, assign={k: sorted(v) for k, v in assign.items()},
               cell_names=[b1.cell_name(c) for c in cells], cells={})
    t00 = time.time()
    for ci, c in enumerate(cells):
        model.set_cell(c)
        on = newvals(c)
        gated_parts = {k for k in on if assign[k] != ALL}
        arms = ["none"]
        if gated_parts or ci == 0: arms.append("grasp")
        if gated_parts == {"OW"}: arms.append("scramble")
        if a.group == "pair_OWTC" and ci == 0: arms.append("none2")
        env = b1_eval.make_env(c)
        rec["cur_cell"] = ci
        info = {arm: dict(success=[], steps=[], on_t=[], off_t=[], seconds=0.0) for arm in arms}
        t0 = time.time()
        for e in range(n_ep):
            for arm in arms:
                ta = time.time()
                signs = flip_signs(a.seed, gi, ci, e) if arm == "scramble" else None
                ok, n, on_t, off_t = run_episode(model, env, c, arm, e, tw, signs, rec)
                d = info[arm]; d["success"].append(bool(ok)); d["steps"].append(n); d["on_t"].append(on_t); d["off_t"].append(off_t)
                d["seconds"] += time.time() - ta
        res["cells"][b1.cell_name(c)] = dict(index=ci, arms=arms, gated_parts=sorted(gated_parts), info=info)
        print(b1.cell_name(c), arms, {arm: (sum(v["success"]), len(v["success"]), round(v["seconds"] / max(len(v["success"]), 1), 1)) for arm, v in info.items()},
              f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{OUT}/s{a.seed}", exist_ok=True)
    np.savez_compressed(f"{OUT}/s{a.seed}/{a.group}.npz", **pack(rec))
    res["seconds"] = time.time() - t00
    json.dump(res, open(f"{OUT}/s{a.seed}/{a.group}.json", "w", encoding="utf-8"))
    print("saved", f"{OUT}/s{a.seed}/{a.group}", f"{res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--group", default="pair_OWTC")
    p.add_argument("--eps_mult", type=int, default=1); p.add_argument("--smoke", action="store_true"); p.add_argument("--out", default="")
    a = p.parse_args()
    {"run": cmd_run}[a.cmd](a)

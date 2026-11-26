"""Gate T1 (temporal composition): segment-gated smax over the existing rank-8 parts (results/r1/tokctl/s{S}/parts), rollouts, no training.
Registration: compositional-architecture results/tables/compose_stream_prediction.json, last addendum temporal_T1_registered.
  lengths                                  -> results/r1/t1/lengths.json  (empirical L1 S0/S1 lengths of base-train demos, for arm b)
  run --arm none|grasp|random|oracle --seed S --group G [--cell0] -> results/r1/t1/{arm}/s{S}/{G}.json
  report                                   -> results/r1/t1/t1.json

Choices (literal readings):
 - Gate: SegRouted = r1s.Routed (CTL_R8 rule, smax over the label-routed parts, base + parts in output space at every flow step) except the
   parts active in the current segment are the only ones in the smax; a part outside its segments contributes zero (base alone if none).
   The segment is fixed per sampled chunk (at the chunk start, every 8 steps) from the online state at that step.
 - Online boundary (arm grasp): L1 ProprioHeld (copied via r1seg) stepped each env step on obs (grip = obs[73:75], z = obs[68] of the 77-dim
   stripped state) and the last applied (clipped) gripper command act[7] > 0 (0 at t = 0); tw = results/r1/seg/t0_stats.json l1_fit.tw.
   Debounce: the raw held flag must differ from the committed one for 3 consecutive steps to commit (no retroactive dating online).
   S0 = before the first committed onset, S1 = while committed held, S2 = after the first committed release (stays S2).
 - Arm random: per episode S0 length and S1 length drawn independently from the empirical base-train marginals (lengths.json, built with the
   same online rule on stored demos with cmd = act[t-1, 7]); S0 = t < L0, S1 = L0 <= t < L0 + L1, S2 after.  One RandomState(20261126 + seed)
   advanced sequentially over episodes in run order.
 - Arm oracle: raw held = env.env._check_grasp(robots[0].gripper, object.contact_geoms), same debounce.
 - Arm none = E empty (all parts always active): must reproduce CTL_R8.
 - Part-to-segment assignment per seed is the registered one (ASSIGN); labels route parts as in Table 9 (r1s.newvals).
 - Panel is not rerun (no part is on there by construction); triple is not part of the registered scores.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/coc/flash10/ksingara3/flawed-composition-vlas/experiments/r1")
import r1, r1s, r1seg
from r1s import *  # noqa (np, torch, b1, b1_data, load_cell, TASKS, GROUPS, EPS, RES, newvals, dropped, H)
from r1seg import ProprioHeld

OUT = os.environ.get("T1_OUT", RES + "/t1")   # T1_OUT: pinned-node rerun (temporal_T1_pinned_registered) writes to results/r1/t1p
ALL = {0, 1, 2}
ASSIGN = {0: {"HB": ALL, "OW": {0}, "TC": ALL}, 1: {"HB": ALL, "OW": {0}, "TC": ALL}, 2: {"HB": ALL, "OW": {0}, "TC": {1, 2}}}
DEB = 3


def tw_value():
    return json.load(open(RES + "/seg/t0_stats.json", encoding="utf-8"))["l1_fit"]["tw"]


class SegState:
    """debounced held flag -> segment 0/1/2"""
    def __init__(self):
        self.c, self.cnt, self.seg, self.on_t, self.off_t = False, 0, 0, None, None

    def update(self, raw, t):
        if bool(raw) != self.c:
            self.cnt += 1
            if self.cnt >= DEB:
                self.c, self.cnt = bool(raw), 0
                if self.c and self.seg == 0:
                    self.seg, self.on_t = 1, t
                elif (not self.c) and self.seg == 1:
                    self.seg, self.off_t = 2, t
        else:
            self.cnt = 0
        return self.seg


class SegRouted(r1s.Routed):
    def __init__(self, assign):
        super().__init__("smax"); self.assign = assign; self.seg = 0; self.all_on = False

    def set_cell(self, cell):
        self.on = newvals(cell)

    @torch.no_grad()
    def sample(self, state, task=None, steps=10, gen=None, z=None):
        act = [k for k in self.on if self.all_on or self.seg in self.assign[k]]
        x = torch.randn(state.shape[0], H, 8, generator=gen)
        for i in range(steps):
            t = torch.full((state.shape[0],), i / steps)
            v = self.base.velocity(x, t, state, task)
            if act:
                d = torch.stack([self.parts[k].velocity(x, t, state, task) - v for k in act])
                v = v + d.max(0).values.clamp(min=0) + d.min(0).values.clamp(max=0)
            x = x + v / steps
        return x


def cmd_lengths(a):
    sp = r1.make_split(0); dr = r1s.dropped(); tw = tw_value()
    L0, L1, n, fired, rel = [], [], 0, 0, 0
    for c in [c for c in sp["train"] if c not in dr]:
        o, ac = load_cell(c)
        for i in range(len(o)):
            det, st = ProprioHeld(tw), SegState(); n += 1
            for t in range(o.shape[1]):
                cmd = ac[i, t - 1, r1seg.CMD] if t else 0.0
                st.update(det.step(o[i, t, r1seg.GRIP[0]:r1seg.GRIP[1]], cmd, o[i, t, r1seg.ZIDX]), t)
            T = o.shape[1]
            L0.append(st.on_t if st.on_t is not None else T)
            if st.on_t is not None:
                fired += 1
                L1.append((st.off_t if st.off_t is not None else T) - st.on_t); rel += st.off_t is not None
    res = dict(L0=L0, L1=L1, n=n, onset_frac=fired / n, release_frac=rel / n, L0_mean=float(np.mean(L0)), L1_mean=float(np.mean(L1)), tw=tw)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(OUT + "/lengths.json", "w", encoding="utf-8"))
    print({k: v for k, v in res.items() if k not in ("L0", "L1")}, flush=True)


def rollout(model, norm, cell, n_episodes, arm, tw, lens, rng, seed=0, max_steps=500):
    env = b1_eval.make_env(cell)
    cond = dict(task=torch.zeros(1, dtype=torch.long))
    model.eval(); t0 = time.time()
    succ, steps, info = [], [], []
    gen = torch.Generator().manual_seed(seed)
    model.all_on = arm == "none"
    for e in range(n_episodes):
        np.random.seed(seed * 100003 + e)
        obs = b1_eval._obs(env.reset())
        ok, t, plan, done, cmd = False, 0, [], False, 0.0
        det, st = ProprioHeld(tw), SegState()
        if arm == "random":
            l0 = int(rng.choice(lens["L0"])); l1 = int(rng.choice(lens["L1"]))
        while t < max_steps and not done:
            s = b1_data.strip_task_id(np.asarray(obs, np.float32)[None], cell, check=(t == 0))
            if arm == "grasp":
                model.seg = st.update(det.step(s[0, r1seg.GRIP[0]:r1seg.GRIP[1]], cmd, s[0, r1seg.ZIDX]), t)
            elif arm == "oracle":
                raw = env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=[g for g in env.env.object.contact_geoms])
                model.seg = st.update(raw, t)
            elif arm == "random":
                model.seg = 0 if t < l0 else (1 if t < l0 + l1 else 2)
            if not plan:
                st_ = torch.tensor(norm.state(s), dtype=torch.float32)
                ch = model.sample(st_, steps=10, gen=gen, **cond)[0].numpy()
                plan = list(norm.unact(ch)[:8])
            a = np.clip(plan.pop(0), env.action_space.low, env.action_space.high)
            cmd = float(a[r1seg.CMD])
            out = env.step(a)
            obs, done = b1_eval._obs(out[0]), bool(out[2])
            t += 1
            if env.env._check_success():
                ok = True; break
        succ.append(bool(ok)); steps.append(t)
        info.append(dict(on_t=st.on_t, off_t=st.off_t, final_seg=int(model.seg)))
    return dict(success=succ, steps=steps, info=info, seconds=time.time() - t0)


def cmd_run(a):
    torch.set_num_threads(1)
    d = f"{RES}/tokctl/s{a.seed}"
    r1s.S = d
    model = SegRouted(ASSIGN[a.seed])
    cells = r1s.GROUPS[a.group]; dr = r1s.dropped(); cells = [c for c in cells if c not in dr]
    if a.cell0:
        cells = cells[:1]
    lens = json.load(open(OUT + "/lengths.json", encoding="utf-8")) if a.arm == "random" else None
    rng = np.random.RandomState(20261126 + a.seed)
    tw = tw_value(); res = dict(arm=a.arm, seed=a.seed, group=a.group, assign={k: sorted(v) for k, v in ASSIGN[a.seed].items()}, cells={})
    for c in cells:
        model.set_cell(c)
        r = rollout(model, model.norm, c, r1s.EPS[a.group], a.arm, tw, lens, rng, seed=0)
        res["cells"][b1.cell_name(c)] = r
        print(b1.cell_name(c), r["success"], [(i["on_t"], i["off_t"]) for i in r["info"]][:4], f"{r['seconds']:.0f}s", flush=True)
    os.makedirs(f"{OUT}/{a.arm}/s{a.seed}", exist_ok=True)
    json.dump(res, open(f"{OUT}/{a.arm}/s{a.seed}/{a.group}{'_cell0' if a.cell0 else ''}.json", "w", encoding="utf-8"))


def succ_of(f):
    if not os.path.exists(f): return None
    return [x for c in json.load(open(f, encoding="utf-8"))["cells"].values() for x in c["success"]]


def cmd_report(a):
    S3 = ["single_HB", "single_OW", "single_TC"]; P3 = ["pair_HBOW", "pair_HBTC", "pair_OWTC"]
    m = lambda v: None if v is None or not len(v) else round(float(np.mean(v)), 3)
    ref = lambda s, g: succ_of(f"{RES}/tokctl/s{s}/eval/routed_smax__{g}.json")
    arm_ = lambda arm, s, g: succ_of(f"{OUT}/{arm}/s{s}/{g}.json")
    out = dict(arms={}, ref={})
    def block(get):
        b = {}
        for s in (0, 1, 2):
            gs = {g: get(s, g) for g in S3 + P3}
            ok = all(v is not None for v in gs.values())
            b[f"seed{s}"] = dict(by_group={g: m(v) for g, v in gs.items()},
                                 pairs=m(sum((gs[g] for g in P3), [])) if ok else None, singles=m(sum((gs[g] for g in S3), [])) if ok else None)
        for k in ("pairs", "singles"):
            v = [b[f"seed{s}"][k] for s in (0, 1, 2)]
            b.setdefault("mean", {})[k] = m(v) if None not in v else None
        for g in P3 + S3:
            v = [b[f"seed{s}"]["by_group"][g] for s in (0, 1, 2)]
            b["mean"][g] = m(v) if None not in v else None
        return b
    out["ref_CTL_R8"] = block(ref)
    for arm in ("grasp", "random", "oracle", "none"):
        out["arms"][arm] = block(lambda s, g, arm=arm: arm_(arm, s, g))
    A, B = out["arms"]["grasp"], out["arms"]["random"]
    ver = {}
    if A["mean"]["pairs"] is not None:
        ver["T1-P1"] = dict(holds=bool(A["mean"]["pairs"] >= .60 and A["mean"]["pair_OWTC"] >= .47), pairs=A["mean"]["pairs"], OWTC=A["mean"]["pair_OWTC"])
        ver["T1-P2"] = dict(holds=bool(abs(A["mean"]["pair_HBTC"] - .711) <= .07), HBTC=A["mean"]["pair_HBTC"])
        ver["T1-P3"] = dict(holds=bool(A["mean"]["singles"] >= .55), singles=A["mean"]["singles"], panel="unchanged by construction (parts off), not rerun")
        if B["mean"]["pairs"] is not None:
            diffs = [A[f"seed{s}"]["pairs"] - B[f"seed{s}"]["pairs"] for s in (0, 1, 2)]
            ver["T1-P4"] = dict(holds=bool(A["mean"]["pairs"] - B["mean"]["pairs"] >= .05 and sum(x > 0.05 - 1e-9 for x in diffs) >= 2),
                                mean_diff=round(A["mean"]["pairs"] - B["mean"]["pairs"], 3), per_seed_diff=[round(x, 3) for x in diffs],
                                note="per-seed rule read as: (a)-(b) >= .05 in at least 2 of 3 seeds")
        ver["falsifier"] = dict(triggered=bool(A["mean"]["pairs"] < .55 or A["mean"]["pair_OWTC"] < .40 or ("T1-P4" in ver and not ver["T1-P4"]["holds"])))
    out["verdict"] = ver
    # online boundary firing per objective (arm grasp / oracle)
    fire = {}
    for arm in ("grasp", "oracle"):
        agg = {}
        for s in (0, 1, 2):
            for g in S3 + P3:
                f = f"{OUT}/{arm}/s{s}/{g}.json"
                if not os.path.exists(f): continue
                for cn, r in json.load(open(f, encoding="utf-8"))["cells"].items():
                    obj = cn.split("_")[-1]
                    for i in r["info"]:
                        z = agg.setdefault(obj, dict(n=0, onset=0, release=0, on_t=[]))
                        z["n"] += 1; z["onset"] += i["on_t"] is not None; z["release"] += i["off_t"] is not None
                        if i["on_t"] is not None: z["on_t"].append(i["on_t"])
        fire[arm] = {k: dict(n=v["n"], onset_frac=v["onset"] / v["n"], release_frac=v["release"] / v["n"], onset_step_median=float(np.median(v["on_t"])) if v["on_t"] else None) for k, v in agg.items()}
    out["boundary_firing"] = fire
    json.dump(out, open(OUT + "/t1.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(out["verdict"], indent=1)); print(json.dumps(fire))
    for k in ("ref_CTL_R8",):
        print(k, json.dumps(out[k]["mean"]))
    for arm in out["arms"]:
        print(arm, json.dumps(out["arms"][arm]["mean"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd"); p.add_argument("--arm", default="grasp"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group", default="pair_OWTC"); p.add_argument("--cell0", action="store_true")
    a = p.parse_args()
    {"lengths": cmd_lengths, "run": cmd_run, "report": cmd_report}[a.cmd](a)

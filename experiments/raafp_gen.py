"""RAA-F Route A+ generator, planner-yield diagnosis and per-step state recorder (docs/notes/raafp_diag.md).  LIBERO venv, CPU.  sbatch only.

A copy of the working path of experiments/fa1g_plan.py `gen` (imports layout(), ctrl(), FIXR / MOVR from it; fa1g_plan.py is never edited), plus
  * rules: the spot rule's numbers are parameters (RULES), `orig` = the plan's / raaf_plan_dry.py's rule, `prop` = the PROPOSED rule;
  * `ktarget` rule = the fa1g control (destination = the K task's own target, the success predicate), to check the copy against fa1g's yield;
  * a failure classifier (no_layout / no_spot / timeout_phase<p> / grasp_miss / drop_in_transit / collision / tilt / unsettled / release_off_spot);
  * the free-area measurement (`area`);
  * per-step simulator states for every KEPT demo, in sc_rv's raw layout (states_0 [T, nstate] = sim.get_state().flatten() BEFORE action t,
    actions_0 [T, 7]; plus init_state, spot, seed), one file per demo: <out>/<obj>/demo_<k>.npz.

    python experiments/raafp_gen.py area --out <json>
    python experiments/raafp_gen.py dry  --rule orig|prop|ktarget --n 30 --out <json> [--states-dir <dir>] [--objs bowl,cream_cheese,wine_bottle]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import fa1_common as FC                                                      # noqa: E402
import fa1g_plan as FP                                                       # noqa: E402

BASE = 93_000_000
OBJ = [("bowl", "akita_black_bowl_1", "K1"), ("cream_cheese", "cream_cheese_1", "K6"), ("wine_bottle", "wine_bottle_1", "K4")]
KT = {"akita_black_bowl_1": ["plate_1", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side"],
      "cream_cheese_1": ["akita_black_bowl_1"], "wine_bottle_1": ["wooden_cabinet_1_top_side", "wine_rack_1_top_region"]}
# d_start / d_tgt: min distance (m) from the object's start / from each K target of the object; clear: extra clearance beyond the footprint radius
# of each fixture / other movable (center distance < R + clear is excluded); stove_m: margin around N8's stove-front region; ext: the sampling
# box is the N2 box grown by ext on every side (the layout() sampler is unchanged, so objects stay in the N2 box).
RULES = {
    "orig": dict(d_start=0.20, d_tgt=0.20, clear=0.12, stove_m=0.05, ext=0.0),
    # PROPOSED (needs orchestrator approval): see docs/notes/raafp_diag.md
    "prop": dict(d_start=0.20, d_tgt=0.20, gap=0.03, stove_m=0.05, ext=0.05),
    "prop_e10": dict(d_start=0.20, d_tgt=0.20, gap=0.03, stove_m=0.05, ext=0.10),
    "prop_g02": dict(d_start=0.20, d_tgt=0.20, gap=0.02, stove_m=0.05, ext=0.05),
    # release = "hold": the object (not the hand) is brought to the spot, from the live hand-object offset (see attempt()).  ext = [xlo, ylo, xhi, yhi].
    "h_a": dict(d_start=0.20, d_tgt=0.20, gap=0.03, stove_m=0.05, ext=[0.07, 0.05, 0.0, 0.0], rel="hold"),
    "h_b": dict(d_start=0.20, d_tgt=0.20, gap=0.02, stove_m=0.05, ext=[0.07, 0.05, 0.0, 0.0], rel="hold"),
    "h_c": dict(d_start=0.20, d_tgt=0.20, gap=0.03, stove_m=0.05, ext=0.05, rel="hold"),
    "h_d": dict(d_start=0.20, d_tgt=0.20, gap=0.02, stove_m=0.05, ext=0.0, rel="hold"),
    # h_e / h_f: + phase-4 (transit) limit 100 steps instead of 40 and episode cap 360 (carries up to 45 cm need > 40 steps); h_e also trims the east edge
    "h_e": dict(d_start=0.20, d_tgt=0.20, gap=0.02, stove_m=0.05, ext=[0.07, 0.05, -0.05, 0.0], rel="hold", t4=100, max_steps=360),
    "h_f": dict(d_start=0.20, d_tgt=0.20, gap=0.02, stove_m=0.05, ext=[0.07, 0.05, 0.0, 0.0], rel="hold", t4=100, max_steps=360),
    "h_g": dict(d_start=0.20, d_tgt=0.20, gap=0.03, stove_m=0.05, ext=[0.07, 0.05, -0.05, 0.0], rel="hold", t4=100, max_steps=360),
    # h_h / h_i: + redraw the layout when no spot exists, or (h_h) when the object starts east of x 0.04 (reach) / within 5 cm edge gap of another thing
    "h_h": dict(d_start=0.20, d_tgt=0.20, gap=0.02, stove_m=0.05, ext=[0.07, 0.05, 0.0, 0.0], rel="hold", t4=100, max_steps=360, redraw=True, start_x_max=0.04, start_gap=0.05),
    "h_i": dict(d_start=0.20, d_tgt=0.20, gap=0.02, stove_m=0.05, ext=[0.07, 0.05, 0.0, 0.0], rel="hold", t4=100, max_steps=360, redraw=True),
    "h_j": dict(d_start=0.20, d_tgt=0.20, gap=0.02, stove_m=0.05, ext=[0.07, 0.05, 0.0, 0.0], rel="hold", t4=100, max_steps=360, redraw=True, start_x_max=0.04),
}


def free_mask(n2, sc, oe, rule, step=0.01):
    """Grid of the sampling box; bool free[ix, iy] and per-constraint exclusion masks."""
    c = n2["constraints"]
    e = np.broadcast_to(np.asarray(rule["ext"], float), (4,)) if np.ndim(rule["ext"]) else np.full(4, float(rule["ext"]))   # xlo, ylo, xhi, yhi
    lo, hi = np.array(c["box"][0]) - e[:2], np.array(c["box"][1]) + e[2:]
    fixp, fixn = np.array(c["fixture_xy"]), n2["fixtures"]
    sf = np.asarray(c["stove_front_world"], float)
    xs, ys = np.arange(lo[0], hi[0] + 1e-9, step), np.arange(lo[1], hi[1] + 1e-9, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack((X, Y), -1)
    d = lambda q: np.linalg.norm(P - np.asarray(q)[:2], axis=-1)                     # noqa: E731
    p0 = sc["pos"][FC.ENT.index(oe)][:2]
    # "clear" = extra center-distance clearance beyond the other thing's radius (the plan's code; ignores the object's own radius);
    # "gap" = edge-to-edge gap: center distance >= R_other + R_own + gap
    cl = rule["clear"] if "clear" in rule else FP.MOVR[oe] + rule["gap"]
    ex = {"start": d(p0) < rule["d_start"],
          "ktargets": np.any([d(sc["pos"][FC.ENT.index(t)][:2]) < rule["d_tgt"] for t in KT[oe]], 0),
          "fixtures": np.any([d(fixp[f]) < FP.FIXR[fn] + cl for f, fn in enumerate(fixn)], 0),
          "movables": np.any([d(sc["pos"][FC.ENT.index(m)]) < FP.MOVR[m] + cl for m in FP.MOVR if m != oe], 0),
          "stovefront": (X >= sf[0] - rule["stove_m"]) & (X <= sf[2] + rule["stove_m"]) & (Y >= sf[1] - rule["stove_m"]) & (Y <= sf[3] + rule["stove_m"])}
    free = ~np.any(list(ex.values()), 0)
    return free, ex, (lo, hi, step, xs, ys)


def spot(n2, sc, oe, rule, rng):
    free, _, (lo, hi, step, xs, ys) = free_mask(n2, sc, oe, rule, step=0.005)
    ix = np.argwhere(free)
    if len(ix) == 0:
        return None
    i = ix[rng.integers(len(ix))]
    return np.array([xs[i[0]], ys[i[1]]]) + rng.uniform(-step / 2, step / 2, 2) * 0.99


def make_layout_scene(lay_env, n2, bddl, seed):
    rng = np.random.default_rng(seed)
    s0, _ = FP.layout(lay_env, n2, rng, seed)
    if s0 is None:
        return rng, None, None, None, None
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(s0)
    dj = FC.drawer_joint(env)
    return rng, s0, env, obs, dj


def area(a):
    n2, _ = FP.lang_roles()
    res = {}
    variants = dict(RULES)
    variants.update({"no_dtgt": dict(RULES["orig"], d_tgt=0.0), "no_stovefront": dict(RULES["orig"], stove_m=-9.0)})
    for gp in ((0.02, 0.03, 0.05) if a.grid else ()):
        for dd in (0.20, 0.15):
            for ex in (0.0, 0.05, 0.10):
                variants["gap%.2f_d%.2f_ext%.2f" % (gp, dd, ex)] = dict(d_start=dd, d_tgt=dd, gap=gp, stove_m=0.05, ext=ex)
    for oix, (name, oe, task) in enumerate(OBJ):
        t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
        bddl = os.path.join(FP.ROOT, t["bddl"])
        lay_env = FC.make_env(bddl)
        rows = []
        for k in range(a.n):
            seed = BASE + 1000 * oix + k
            rng, s0, env, obs, dj = make_layout_scene(lay_env, n2, bddl, seed)
            if s0 is None:
                rows.append(dict(seed=seed, layout=False))
                continue
            for _ in range(FP.WAIT):
                obs, _, _, _ = env.step(FC.DUMMY)
            sc = FC.scene(env, obs, dj)
            r = dict(seed=seed, layout=True)
            for vn, rule in variants.items():
                free, ex, (lo, hi, step, xs, ys) = free_mask(n2, sc, oe, rule)
                r[vn] = dict(free_cm2=float(free.sum() * (step * 100) ** 2), box_cm2=float(free.size * (step * 100) ** 2),
                             excl_alone_cm2={e: float(m.sum() * (step * 100) ** 2) for e, m in ex.items()})
                # largest free-space clearance of any single spot to the nearest constraint
                if vn == "orig":
                    r[vn]["frac_excl"] = {e: float(m.mean()) for e, m in ex.items()}
            rows.append(r)
            env.close()
            print("[raafp] area %s k %d: free orig %.0f prop %.0f cm2 of %.0f" % (name, k, r["orig"]["free_cm2"], r["prop"]["free_cm2"], r["orig"]["box_cm2"]), flush=True)
        lay_env.close()
        ok = [r for r in rows if r.get("layout")]
        summ = {}
        for vn in variants:
            f = np.array([r[vn]["free_cm2"] for r in ok])
            summ[vn] = dict(mean_cm2=float(f.mean()), median_cm2=float(np.median(f)), min_cm2=float(f.min()), max_cm2=float(f.max()),
                            frac_layouts_nonempty=float((f > 0).mean()), frac_layouts_ge_25cm2=float((f >= 25).mean()),
                            frac_layouts_ge_100cm2=float((f >= 100).mean()), box_cm2=ok[0][vn]["box_cm2"])
        summ["mean_excl_alone_frac_orig"] = {e: float(np.mean([r["orig"]["frac_excl"][e] for r in ok])) for e in ok[0]["orig"]["frac_excl"]}
        res[name] = dict(task=task, n=len(rows), summary=summ, rows=rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(dict(variants=variants, res=res), fh, indent=1)


def attempt(n2, kp, bddl, lay_env, oe, task_target, k, oix, rule_name, states_dir=None, name="", j=0):
    seed = BASE + 1000 * oix + k + 50_000_000 * j
    ktarget = rule_name == "ktarget"
    t4 = 40 if ktarget else RULES[rule_name].get("t4", 40)
    max_steps = FP.MAX_STEPS if ktarget else RULES[rule_name].get("max_steps", FP.MAX_STEPS)
    hold_mode = (not ktarget) and RULES[rule_name].get("rel") == "hold"
    rng, s0, env, obs, dj = make_layout_scene(lay_env, n2, bddl, seed)
    if s0 is None:
        return dict(seed=seed, cls="no_layout", success=False)
    oi = FC.ENT.index(oe)
    ti = FC.ENT.index(task_target)
    sc = FC.scene(env, obs, dj)
    rl_ = RULES.get(rule_name, {})
    if rl_.get("start_x_max") is not None and sc["pos"][oi][0] > rl_["start_x_max"]:
        env.close()
        return dict(seed=seed, cls="layout_rejected", why="start_x", success=False)
    if rl_.get("start_gap") is not None:
        c_ = n2["constraints"]
        gaps = [np.linalg.norm(sc["pos"][oi][:2] - sc["pos"][FC.ENT.index(m)][:2]) - FP.MOVR[m] - FP.MOVR[oe] for m in FP.MOVR if m != oe]
        gaps += [np.linalg.norm(sc["pos"][oi][:2] - np.array(c_["fixture_xy"])[f]) - FP.FIXR[fn] - FP.MOVR[oe] for f, fn in enumerate(n2["fixtures"])]
        if min(gaps) < rl_["start_gap"]:
            env.close()
            return dict(seed=seed, cls="layout_rejected", why="start_gap", success=False)
    sp = None
    if not ktarget:
        sp = spot(n2, sc, oe, RULES[rule_name], rng)
        if sp is None:
            env.close()
            return dict(seed=seed, cls="no_spot", success=False)
    for _ in range(FP.WAIT):
        obs, _, done, _ = env.step(FC.DUMMY)
    jit = rng.normal(0, 0.01, (2, 3))
    cz = kp["carry_z_above_ref"] + rng.normal(0, 0.02)
    goff, roff = np.array(kp["grasp_off"]) + jit[0], np.array(kp["release_off"]) + jit[1]
    RG, RR = np.array(kp["grasp_rot"]), np.array(kp["release_rot"])
    sc = FC.scene(env, obs, dj)
    zref = float(sc["pos"][8, 2])
    p_start = sc["pos"][oi].copy()
    pos0 = np.array([sc["pos"][FC.ENT.index(m)] for m in FP.MOVR])
    sim = env.env.sim
    init_state = sim.get_state().flatten().copy()
    states, acts = [], []
    phase, hold, prev, step, in_phase, rel_up, ee_hold, done = 0, 0, -1.0, 0, 0, None, None, False
    zmax, rel_info, phase_log = 0.0, None, {}
    while step < max_steps and (phase < 8 if not ktarget else not done):
        sc = FC.scene(env, obs, dj)
        po = sc["pos"][oi]
        zmax = max(zmax, float(po[2] - p_start[2]))
        pt = sc["pos"][ti] if ktarget else np.array([sp[0], sp[1], zref])
        if phase == 0:
            wp, Rt, g = po + goff + [0, 0, 0.10], RG, -1.0
        elif phase == 1:
            wp, Rt, g = po + goff, RG, -1.0
        elif phase == 2:
            wp, Rt, g = ee_hold if hold else sc["ee_pos"], RG, 1.0
        elif phase == 3:
            wp, Rt, g = np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), RG, 1.0
        elif phase == 4:
            wp, Rt, g = (np.array([sp[0] + (sc["ee_pos"] - po)[0], sp[1] + (sc["ee_pos"] - po)[1], zref + cz]) if hold_mode else np.array([pt[0] + roff[0], pt[1] + roff[1], zref + cz])), RR, 1.0
        elif phase == 5:
            wp, Rt, g = (np.array([sp[0] + (sc["ee_pos"] - po)[0], sp[1] + (sc["ee_pos"] - po)[1], p_start[2] + 0.01 + (sc["ee_pos"] - po)[2]]) if hold_mode else pt + roff), RR, 1.0
        elif phase == 6:
            wp, Rt, g = ee_hold, RR, -1.0
        else:
            wp, Rt, g = rel_up, RR, -1.0
        a, err = FP.ctrl(sc, wp, Rt, g, rng)
        if phase in (2, 6):
            a[:6] = 0.0
        if phase in (2, 6) and hold == 0:
            ee_hold = sc["ee_pos"].copy()
        if phase == 6 and hold == 0:
            rel_info = dict(obj_to_target_xy=float(np.linalg.norm(po[:2] - pt[:2])), obj_z_above_start=float(po[2] - p_start[2]),
                            obj_to_ee_xy=float(np.linalg.norm(po[:2] - sc["ee_pos"][:2])), ee_z_above_table=float(sc["ee_pos"][2] - zref))
        states.append(sim.get_state().flatten().copy())
        acts.append(a.copy())
        obs, _, done, _ = env.step(a.tolist())
        prev, step, in_phase = float(a[6]), step + 1, in_phase + 1
        lim = t4 if phase == 4 else 40
        if phase in (0, 3, 4) and err < 0.015 or phase in (1, 5) and err < 0.008 or phase in (0, 1, 3, 4, 5, 7) and in_phase >= lim or phase == 7 and err < 0.02:
            if in_phase >= lim and not (phase in (0, 3, 4) and err < 0.015 or phase in (1, 5) and err < 0.008 or phase == 7 and err < 0.02):
                phase_log[phase] = "timeout%d(err=%.3f)" % (lim, err)
            phase, in_phase = phase + 1, 0
        elif phase in (2, 6):
            hold += 1
            if hold >= 12:
                phase, hold, in_phase = phase + 1, 0, 0
                if phase == 7:
                    rel_up = sc["ee_pos"] + [0, 0, 0.10]
        if ktarget and done:
            break
    rec = dict(seed=seed, steps=step, phase=int(phase), zmax_lift=zmax, release=rel_info, phase_timeouts={str(k_): v for k_, v in phase_log.items()},
               spot_xy=None if sp is None else np.round(sp, 4).tolist(), start_xy=np.round(p_start[:2], 4).tolist())
    if ktarget:
        rec.update(success=bool(done), cls="ok" if done else ("timeout_phase%d" % phase))
        if not done and phase >= 4 and zmax < 0.02:
            rec["cls"] = "grasp_miss"
        env.close()
        return rec
    if phase < 8:
        rec.update(success=False, cls="grasp_miss" if (phase >= 3 and zmax < 0.02) else "timeout_phase%d" % phase)
        env.close()
        return rec
    sc1 = FC.scene(env, obs, dj)
    p1 = sc1["pos"][oi].copy()
    for _ in range(20):
        obs, _, _, _ = env.step(FC.DUMMY)
    sc2 = FC.scene(env, obs, dj)
    R = np.asarray(sim.data.body_xmat[env.env.obj_body_id[oe]]).reshape(3, 3)
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    pos1 = np.array([sc2["pos"][FC.ENT.index(m)] for m in FP.MOVR])
    others = max(float(np.linalg.norm(pos1[i, :2] - pos0[i, :2])) for i, m in enumerate(FP.MOVR) if m != oe)
    err_xy = float(np.linalg.norm(sc2["pos"][oi][:2] - sp))
    settle = float(np.linalg.norm(sc2["pos"][oi] - p1))
    moved = float(np.linalg.norm(sc2["pos"][oi][:2] - p_start[:2]))
    rec.update(final_xy_err=err_xy, tilt_deg=tilt, settle_move=settle, others_max_move=others, moved_from_start=moved)
    if zmax < 0.02 or moved < 0.05:
        cls = "grasp_miss"
    elif rel_info and rel_info["obj_to_target_xy"] > 0.06 and err_xy >= 0.03:
        cls = "drop_in_transit"
    elif others >= 0.02:
        cls = "collision"
    elif tilt >= 20:
        cls = "tilt"
    elif settle >= 0.01:
        cls = "unsettled"
    elif err_xy >= 0.03:
        cls = "release_off_spot"
    else:
        cls = "ok"
    rec.update(cls=cls, success=cls == "ok")
    if rec["success"] and states_dir:
        d = os.path.join(states_dir, name)
        os.makedirs(d, exist_ok=True)
        states.append(sim.get_state().flatten().copy())
        np.savez_compressed(os.path.join(d, "demo_%d.npz" % k), states_0=np.asarray(states), actions_0=np.asarray(acts), init_state=init_state,
                            spot=sp, seed=seed, wait_steps=FP.WAIT, bddl=bddl, note="states_0[t] = sim state BEFORE actions_0[t]; last row = final state")
    env.close()
    return rec


def dry(a):
    n2, _ = FP.lang_roles()
    kps = json.load(open(os.path.join(FP.OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"]
    res, t0 = {}, time.time()
    want = a.objs.split(",")
    for oix, (name, oe, task) in enumerate(OBJ):
        if name not in want:
            continue
        t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
        bddl = os.path.join(FP.ROOT, t["bddl"])
        lay_env = FC.make_env(bddl)
        tgt = n2["tasks"][0]["target"]
        tgt = t["target"]
        rows = []
        for k in range(a.n):
            if a.keep and sum(1 for r_ in rows if r_["success"]) >= a.keep:
                break
            j, rej = 0, []
            while True:
                r = attempt(n2, kps[task], bddl, lay_env, oe, tgt, k, oix, a.rule, a.states_dir, name, j)
                if RULES.get(a.rule, {}).get("redraw") and (r["cls"] in ("layout_rejected", "no_spot", "no_layout")) and j < 40:
                    rej.append(r["cls"] + (":" + r["why"] if "why" in r else ""))
                    j += 1
                    continue
                break
            r["redraws"] = rej
            rows.append(r)
            print("[raafp] %s %s k %d: %s steps %s (%.0fs)" % (a.rule, name, k, r["cls"], r.get("steps"), time.time() - t0), flush=True)
        lay_env.close()
        ok = [r for r in rows if r["success"]]
        cls = {}
        for r in rows:
            cls[r["cls"]] = cls.get(r["cls"], 0) + 1
        res[name] = dict(task=task, rule=a.rule, attempts=len(rows), layout_draws=sum(1 + len(r.get("redraws", [])) for r in rows), kept=len(ok), yield_=len(ok) / len(rows),
                         steps_mean_kept=float(np.mean([r["steps"] for r in ok])) if ok else None, classes=cls, seconds=time.time() - t0, rows=rows)
        print("[raafp] %s %s: yield %d/%d classes %s" % (a.rule, name, len(ok), len(rows), json.dumps(cls)), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(dict(rule=a.rule, rule_params=RULES.get(a.rule), res=res), fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("area", "dry"))
    ap.add_argument("--rule", default="orig", choices=tuple(list(RULES) + ["ktarget"]))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=0)
    ap.add_argument("--base", type=int, default=93_000_000)
    ap.add_argument("--states-dir", default=None)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--objs", default="bowl,cream_cheese,wine_bottle")
    a = ap.parse_args()
    BASE = a.base
    assert not os.path.exists(a.out), a.out
    area(a) if a.mode == "area" else dry(a)

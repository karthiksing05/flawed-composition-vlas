"""RAA-F Step 0, check S0-5 (docs/notes/raaf_plan.md 3.2, 5): dry run of Route A+'s generator.  LIBERO venv, CPU.  sbatch only.

A copy of experiments/fa1g_plan.py `gen` (imported: layout(), ctrl(), keypoints) whose destination is a free TABLE SPOT instead of the task's
target: seed 93_000_000 + 1000 obj + k; the spot is uniform in N2's layout box with >= 20 cm from the object's start and from every K target of
that object, >= 12 cm clear of every fixture / movable footprint (fa1g MOVR / FIXR), outside the stove-front region expanded by 5 cm (N8's
region, BDDL stove_front, +5 cm).  Keypoints: the object's own K task (bowl K1, cream cheese K6, wine bottle K4), the release offset applied to
the spot at table height (the plan's fallback "K-task release keypoints unchanged with the spot as the reference" is the ONLY variant run here).
Success filter (3.2): after the retreat, 20 settle steps; the object's xy within 3 cm of the spot, tilt < 20 deg from upright, moved < 1 cm over the
settle steps, no other movable moved > 2 cm.  The dry run saves no states and no images: yield and step counts only (10 attempts per object).

    python experiments/raaf_plan_dry.py --n 10 --out results/flawed/raaf/step0/s0_5_planner.json     (the output must not exist)
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


def spot(n2, sc, oe, rng):
    c = n2["constraints"]
    lo, hi = np.array(c["box"][0]), np.array(c["box"][1])
    fixp, fixn = np.array(c["fixture_xy"]), n2["fixtures"]
    sf = np.asarray(c["stove_front_world"], float)
    p0 = sc["pos"][FC.ENT.index(oe)][:2]
    for _ in range(5000):
        p = rng.uniform(lo, hi)
        if np.linalg.norm(p - p0) < 0.20:
            continue
        if any(np.linalg.norm(p - sc["pos"][FC.ENT.index(t)][:2]) < 0.20 for t in KT[oe]):
            continue
        if any(np.linalg.norm(p - fixp[f]) < FP.FIXR[fn] + 0.12 for f, fn in enumerate(fixn)):
            continue
        if any(m != oe and np.linalg.norm(p - sc["pos"][FC.ENT.index(m)][:2]) < FP.MOVR[m] + 0.12 for m in FP.MOVR):
            continue
        if sf[0] - 0.05 <= p[0] <= sf[2] + 0.05 and sf[1] - 0.05 <= p[1] <= sf[3] + 0.05:
            continue
        return p
    return None


def attempt(n2, rl, kp, bddl, lay_env, oi_name, k, oix):
    import fa1b_common as FB
    seed = BASE + 1000 * oix + k
    rng = np.random.default_rng(seed)
    s0, _ = FP.layout(lay_env, n2, rng, seed)
    if s0 is None:
        return dict(seed=seed, layout=False)
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(s0)
    dj = FC.drawer_joint(env)
    oi = FC.ENT.index(oi_name)
    sc = FC.scene(env, obs, dj)
    sp = spot(n2, sc, oi_name, rng)
    if sp is None:
        env.close()
        return dict(seed=seed, layout=True, spot=False)
    for _ in range(FP.WAIT):
        obs, _, done, _ = env.step(FC.DUMMY)
    jit = rng.normal(0, 0.01, (2, 3))
    cz = kp["carry_z_above_ref"] + rng.normal(0, 0.02)
    goff, roff = np.array(kp["grasp_off"]) + jit[0], np.array(kp["release_off"]) + jit[1]
    RG, RR = np.array(kp["grasp_rot"]), np.array(kp["release_rot"])
    sc = FC.scene(env, obs, dj)
    zref = float(sc["pos"][8, 2])
    pos0 = np.array([sc["pos"][FC.ENT.index(m)] for m in FP.MOVR])
    phase, hold, prev, step, in_phase, rel_up, ee_hold = 0, 0, -1.0, 0, 0, None, None
    while step < FP.MAX_STEPS and phase < 8:
        sc = FC.scene(env, obs, dj)
        po = sc["pos"][oi]
        pt = np.array([sp[0], sp[1], zref])
        if phase == 0:
            wp, Rt, g = po + goff + [0, 0, 0.10], RG, -1.0
        elif phase == 1:
            wp, Rt, g = po + goff, RG, -1.0
        elif phase == 2:
            wp, Rt, g = ee_hold if hold else sc["ee_pos"], RG, 1.0
        elif phase == 3:
            wp, Rt, g = np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), RG, 1.0
        elif phase == 4:
            wp, Rt, g = np.array([pt[0] + roff[0], pt[1] + roff[1], zref + cz]), RR, 1.0
        elif phase == 5:
            wp, Rt, g = pt + roff, RR, 1.0
        elif phase == 6:
            wp, Rt, g = ee_hold, RR, -1.0
        else:
            wp, Rt, g = rel_up, RR, -1.0
        a, err = FP.ctrl(sc, wp, Rt, g, rng)
        if phase in (2, 6):
            a[:6] = 0.0
        if phase in (2, 6) and hold == 0:
            ee_hold = sc["ee_pos"].copy()
        obs, _, _, _ = env.step(a.tolist())
        prev, step, in_phase = float(a[6]), step + 1, in_phase + 1
        if phase in (0, 3, 4) and err < 0.015 or phase in (1, 5) and err < 0.008 or phase in (0, 1, 3, 4, 5, 7) and in_phase >= 40 or phase == 7 and err < 0.02:
            phase, in_phase = phase + 1, 0
        elif phase in (2, 6):
            hold += 1
            if hold >= 12:
                phase, hold, in_phase = phase + 1, 0, 0
                if phase == 7:
                    rel_up = sc["ee_pos"] + [0, 0, 0.10]
    rec = dict(seed=seed, layout=True, spot=True, steps=step, phase=phase, spot_xy=np.round(sp, 4).tolist(),
               carry_dist=float(np.linalg.norm(sp - sc["pos"][oi][:2])))
    if phase >= 8:
        sc1 = FC.scene(env, obs, dj)
        p1 = sc1["pos"][oi].copy()
        for _ in range(20):
            obs, _, _, _ = env.step(FC.DUMMY)
        sc2 = FC.scene(env, obs, dj)
        inner, sim = env.env, env.env.sim
        R = np.asarray(sim.data.body_xmat[inner.obj_body_id[oi_name]]).reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        pos1 = np.array([sc2["pos"][FC.ENT.index(m)] for m in FP.MOVR])
        others = [float(np.linalg.norm(pos1[i, :2] - pos0[i, :2])) for i, m in enumerate(FP.MOVR) if m != oi_name]
        rec.update(final_xy_err=float(np.linalg.norm(sc2["pos"][oi][:2] - sp)), tilt_deg=tilt, settle_move=float(np.linalg.norm(sc2["pos"][oi] - p1)),
                   others_max_move=max(others))
        rec["success"] = bool(rec["final_xy_err"] < 0.03 and tilt < 20 and rec["settle_move"] < 0.01 and rec["others_max_move"] < 0.02)
    else:
        rec["success"] = False
    env.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), a.out
    n2, _ = FP.lang_roles()
    kps = json.load(open(os.path.join(FP.OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"]
    res = {}
    t0 = time.time()
    for oix, (name, oe, task) in enumerate(OBJ):
        t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
        bddl = os.path.join(FP.ROOT, t["bddl"])
        lay_env = FC.make_env(bddl)
        rows = []
        for k in range(a.n):
            r = attempt(n2, None, kps[task], bddl, lay_env, oe, k, oix)
            rows.append(r)
            print("[raaf_plan_dry] %s k %d: %s (%.0fs)" % (name, k, json.dumps({x: r[x] for x in r if x in ("success", "steps", "phase", "final_xy_err", "tilt_deg", "settle_move", "others_max_move", "spot", "layout")}), time.time() - t0), flush=True)
        lay_env.close()
        ok = [r for r in rows if r.get("success")]
        res[name] = dict(task=task, attempts=len(rows), kept=len(ok), yield_=len(ok) / max(len(rows), 1),
                         steps_mean_kept=float(np.mean([r["steps"] for r in ok])) if ok else None,
                         seconds=time.time() - t0, rows=rows)
        print("[raaf_plan_dry] %s: yield %d/%d" % (name, len(ok), len(rows)), flush=True)
    res["kill_k1_any_object_below_30pct"] = [k for k, v in res.items() if isinstance(v, dict) and v["yield_"] < 0.30]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

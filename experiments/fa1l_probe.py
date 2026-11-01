"""FLAWED_COMPOSITION prototype A, stage 1l DIAGNOSTIC probe (section 5.20; descriptive only): HOW the scripted planner's grasp fails.
The 1l re-runs (fa1l_plan.py) showed that most failed generated attempts never hold the object after the lift, so planner
failures are mostly grasp failures (not placement failures, as 5.20 first read from the phase counter).  This probe re-runs
planner phases 0-3 (approach, descend, close, lift; fa1g_plan's planner unchanged) on 1g/1k seeds -- every failed seed and up to
40 successful ones per task -- and records, per attempt:
  descend_exit   'reached' (err < 0.008) or 'timeout' (40 steps) for phase 1; the remaining error and its vector (ee minus waypoint)
  rot_err_deg    angle between the end effector's rotation and the grasp rotation at the close
  obj_shift      how far the object moved in xy during approach and descent (pushed before the close)
  touch          finger contacts at the end of the close; held after the lift (fa1b_common.Held)
  obj_xy, ee0    start object position; start end effector
-> results/flawed/a1l_probe_<task>.json

    python experiments/fa1l_probe.py --task K6
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
import fa1g_plan as FG                                                      # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")


def seeds(task):
    bad, good = [], []
    for f in (f"a1g_gen_{task}.json", f"a1k_gen_{task}.json"):
        d = json.load(open(os.path.join(OUT, f), encoding="utf-8"))
        for e in d["episodes"]:
            if e.get("layout"):
                (good if e["success"] else bad).append(int(e["seed"]))
    return bad, good[:40]


def rot_deg(Ra, Rb):
    c = max(-1.0, min(1.0, (np.trace(np.asarray(Ra) @ np.asarray(Rb).T) - 1) / 2))
    return math.degrees(math.acos(c))


def run(task, seed, lay_env, n2, kp, bddl, oe, FB):
    oi = FC.ENT.index(oe)
    rng = np.random.default_rng(seed)
    s0, p = FG.layout(lay_env, n2, rng, seed)
    if s0 is None:
        return None
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(s0)
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    for _ in range(FG.WAIT):
        obs, _, done, _ = env.step(FC.DUMMY)
    jit = rng.normal(0, 0.01, (2, 3))
    cz = kp["carry_z_above_ref"] + rng.normal(0, 0.02)
    goff, RG = np.array(kp["grasp_off"]) + jit[0], np.array(kp["grasp_rot"])
    sc = FC.scene(env, obs, dj)
    zref, o0, ee0 = float(sc["pos"][8, 2]), sc["pos"][oi].copy(), sc["ee_pos"].copy()
    hm = FB.Held()
    phase, hold, step, in_phase, rec, last = 0, 0, 0, 0, {}, None
    while step < FG.MAX_STEPS and phase <= 3:
        sc = FC.scene(env, obs, dj)
        po = sc["pos"][oi]
        if phase == 0:
            wp, Rt, g = po + goff + [0, 0, 0.10], RG, -1.0
        elif phase == 1:
            wp, Rt, g = po + goff, RG, -1.0
        elif phase == 2:
            wp, Rt, g = last if hold else sc["ee_pos"], RG, 1.0
        else:
            wp, Rt, g = np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), RG, 1.0
        a, err = FG.ctrl(sc, wp, Rt, g, rng)
        if phase == 2:
            a[:6] = 0.0
        last = sc["ee_pos"]
        lt, rt = con.touching(oe)
        hm.step(lt, rt, con.pos(oe), sc["ee_pos"])
        obs, _, done, _ = env.step(a.tolist())
        step += 1
        in_phase += 1
        if phase == 1:
            sim = env.env.sim if hasattr(env, "env") else env.sim
            for ci in range(sim.data.ncon):
                c = sim.data.contact[ci]
                b1 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom1]) or "?"
                b2 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom2]) or "?"
                rob = [b for b in (b1, b2) if b.startswith(("robot0", "gripper0"))]
                if len(rob) == 1:
                    other = b2 if rob[0] == b1 else b1
                    rec.setdefault("descend_contacts", {})
                    rec["descend_contacts"][f"{rob[0]}|{other}"] = rec["descend_contacts"].get(f"{rob[0]}|{other}", 0) + 1
        if phase == 1 and (err < 0.008 or in_phase >= 40):
            rec.update(descend_exit="reached" if err < 0.008 else "timeout", descend_err=float(err),
                       descend_vec=(sc["ee_pos"] - wp).tolist(), rot_err_deg=rot_deg(sc["ee_rot"], RG),
                       obj_shift=float(np.linalg.norm(sc["pos"][oi][:2] - o0[:2])), descend_steps=in_phase,
                       ee_z_minus_obj=float(sc["ee_pos"][2] - sc["pos"][oi][2]))
        if phase in (0, 3) and err < 0.015 or phase == 1 and err < 0.008 or phase in (0, 1, 3) and in_phase >= 40:
            if phase == 0:
                rec.update(approach_exit="reached" if err < 0.015 else "timeout", approach_err=float(err))
            phase, in_phase = phase + 1, 0
        elif phase == 2:
            hold += 1
            if hold >= 12:
                lt, rt = con.touching(oe)
                rec.update(touch=[bool(lt), bool(rt)])
                phase, hold, in_phase = phase + 1, 0, 0
    sc = FC.scene(env, obs, dj)
    lt, rt = con.touching(oe)
    h, _ = hm.step(lt, rt, con.pos(oe), sc["ee_pos"])
    lift = float(con.pos(oe)[2] - o0[2])
    env.close()
    rec.update(held=bool(h), obj_xy=o0[:2].tolist(), ee0=ee0.tolist(), lift=lift)
    return rec


def main():
    import fa1b_common as FB
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="K6")
    ap.add_argument("--max_bad", type=int, default=1000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    n2, rl = FG.lang_roles()
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"][a.task]
    t = next(x for x in n2["tasks"] if x["compound"] == a.task and x["layout"] == "familiar")
    bddl = os.path.join(ROOT, t["bddl"])
    oe = rl[a.task][0]
    lay_env = FC.make_env(bddl)
    bad, good = seeds(a.task)
    bad = bad[:a.max_bad]
    out, t0 = [], time.time()
    for lab, ss in (("task_failed", bad), ("task_ok", good)):
        for s in ss:
            r = run(a.task, s, lay_env, n2, kp, bddl, oe, FB)
            if r is None:
                continue
            r.update(seed=s, group=lab)
            out.append(r)
            print(f"[fa1l] probe {a.task} {lab} {s}: held={r['held']} descend={r.get('descend_exit')} err={r.get('descend_err', -1):.3f} "
                  f"rot={r.get('rot_err_deg', -1):.1f} shift={r.get('obj_shift', -1):.3f} touch={r.get('touch')} ({time.time() - t0:.0f}s)", flush=True)
    lay_env.close()
    summ = {}
    for lab in ("task_failed", "task_ok"):
        for hv in (True, False):
            R = [r for r in out if r["group"] == lab and r["held"] == hv]
            if not R:
                continue
            summ[f"{lab}|held={hv}"] = dict(
                n=len(R), descend_timeout=float(np.mean([r.get("descend_exit") == "timeout" for r in R])),
                descend_err_med=float(np.median([r.get("descend_err", np.nan) for r in R])),
                descend_vec_mean=np.nanmean([r.get("descend_vec", [np.nan] * 3) for r in R], 0).tolist(),
                rot_err_med=float(np.median([r.get("rot_err_deg", np.nan) for r in R])),
                obj_shift_med=float(np.median([r.get("obj_shift", np.nan) for r in R])),
                ee_z_minus_obj_med=float(np.median([r.get("ee_z_minus_obj", np.nan) for r in R])),
                touch_both=float(np.mean([bool(r.get("touch")) and all(r["touch"]) for r in R])),
                touch_one=float(np.mean([bool(r.get("touch")) and any(r["touch"]) and not all(r["touch"]) for r in R])),
                obj_xy_mean=np.mean([r["obj_xy"] for r in R], 0).tolist(),
                descend_contact_share={k: float(np.mean([k in r.get("descend_contacts", {}) for r in R]))
                                       for k in sorted({k for r in R for k in r.get("descend_contacts", {})})})
            print(f"[fa1l] probe {a.task} {lab} held={hv}: {json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in summ[f'{lab}|held={hv}'].items()})}", flush=True)
    with open(os.path.join(OUT, f"a1l_probe_{a.task}{a.tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(task=a.task, attempts=out, summary=summ, descriptive=True), fh, indent=1)


if __name__ == "__main__":
    main()

"""Read G (docs/HAND_OVER.md 7.4; m3_prereg pp_hand2): can the servo hold the object on the new-layout states where it failed, given more time to reach?
CPU, LIBERO venv.  pp_n4d.plan_episode's phases 0-3 with the reach timeout as the only change; stops at the first hold.  -> results/flawed/pp_read_g.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import pp_atlas as PA                                                       # noqa: E402
from pp_l1_floor import kp_for                                              # noqa: E402

n2 = PA.tasks()
ks = n2["states"]["main"]
br = {int(b["trial"]): b for b in PA.branches()}
tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
SHORT, LONG = 40, 120
R_REACH = 0.05


def contacts(env):
    """Names of the geoms touching any robot or gripper geom, excluding the robot's own."""
    sim = env.env.sim
    out = set()
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        a, b = sim.model.geom_id2name(int(c.geom1)) or "", sim.model.geom_id2name(int(c.geom2)) or ""
        ra, rb = ("robot" in a or "gripper" in a), ("robot" in b or "gripper" in b)
        if ra != rb:
            out.add(b if ra else a)
    return sorted(out)


def attempt(t, kp, state, seed_base, timeout):
    import fa1_common as FC
    import fa1b_common as FB
    from fa1g_plan import ctrl
    rng = np.random.default_rng(int(seed_base) + 10)
    oe = t["object"]
    oi = FC.ENT.index(oe)
    env = FC.fresh_env(os.path.join(ROOT, t["bddl"]))
    env.reset()
    obs = env.set_init_state(state)
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    hm = FB.Held()
    for _ in range(10):
        obs, _, _, _ = env.step(FC.DUMMY)
    jit = rng.normal(0, 0.01, (2, 3))
    cz = kp["carry_z_above_ref"] + rng.normal(0, 0.02)
    goff = np.array(kp["grasp_off"]) + jit[0]
    RG = np.array(kp["grasp_rot"])
    phase, hold, step, in_phase, zref = 0, 0, 0, 0, float(FC.scene(env, obs, dj)["pos"][8, 2])
    best, at, held = 1e9, [], False
    lo, hi = None, None
    jn = [j for j in env.env.robots[0].robot_joints]
    ids = [env.env.sim.model.joint_name2id(j) for j in jn]
    lo, hi = env.env.sim.model.jnt_range[ids, 0], env.env.sim.model.jnt_range[ids, 1]
    qadr = [env.env.sim.model.jnt_qposadr[i] for i in ids]
    lim = False
    while step < 300 and not held and phase <= 3:
        sc = FC.scene(env, obs, dj)
        po = sc["pos"][oi]
        if phase == 0:
            wp, g = po + goff + [0, 0, 0.10], -1.0
        elif phase == 1:
            wp, g = po + goff, -1.0
        elif phase == 2:
            wp, g = po + goff, 1.0
        else:
            wp, g = np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), 1.0
        a, err = ctrl(sc, wp, RG, g, rng)
        obs, _, _, _ = env.step(a.tolist())
        h, _ = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
        held = bool(h)
        dist = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - (con.pos(oe) + goff)))
        if dist < best:
            q = env.env.sim.data.qpos[qadr]
            best, at, lim = dist, contacts(env), bool(((q - lo) < 0.05).any() or ((hi - q) < 0.05).any())
        step, in_phase = step + 1, in_phase + 1
        if phase in (0, 3) and err < 0.015 or phase == 1 and err < 0.008 or phase in (0, 1, 3) and in_phase >= timeout:
            phase, in_phase = phase + 1, 0
        elif phase == 2:
            hold += 1
            if hold >= 12:
                phase, hold, in_phase = 3, 0, 0
    env.close()
    return dict(held=held, min_a=best, contacts=at, at_limit=lim, steps=step)


if os.environ.get("READG_SMOKE"):                                           # smoke states only; writes nothing
    t = tid[("N1", "new")]
    S = np.load(os.path.join(PA.OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    for b in json.load(open(os.path.join(PA.OUT, "n2_branches_smoke.json"), encoding="utf-8"))[:2]:
        print("smoke", b["trial"], json.dumps(attempt(t, kp_for("N1"), S[int(b["trial"])], b["seed_base"], SHORT)), flush=True)
    sys.exit(0)

res = {}
for comp, name in (("N1", "cream_cheese_1"), ("N4", "wine_bottle_1")):
    t = tid[(comp, "new")]
    assert t["object"] == name
    S = np.load(os.path.join(PA.OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    kp = kp_for(comp)
    stem = os.path.join(PA.OD, f"servo_s10_{t['tid']}")
    R = json.load(open(stem + ".json", encoding="utf-8"))
    Tr = PA.load_track(stem + "_track.npz")
    fail = []
    for k, e, r in zip(ks, R["eps"], Tr):
        d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
        if PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])[0] == "GRASP":
            fail.append(k)
    rows = []
    for k in fail:
        ctl = attempt(t, kp, S[k], br[k]["seed_base"], SHORT)
        lng = attempt(t, kp, S[k], br[k]["seed_base"], LONG)
        rows.append(dict(state=k, short=ctl, long=lng))
        print(name, k, json.dumps(rows[-1]), flush=True)
    rate = float(np.mean([x["long"]["held"] for x in rows])) if rows else None
    v = "NOT A BLOCK" if (rate or 0) > 0.5 else ("BLOCK" if (rate or 0) < 0.2 else "MIXED")
    res[name] = dict(states=fail, control_hold_rate=float(np.mean([x["short"]["held"] for x in rows])) if rows else None, long_hold_rate=rate, verdict=v, rows=rows)
    print(name, v, "control", res[name]["control_hold_rate"], "long", rate, flush=True)
json.dump(res, open(os.path.join(PA.OUT, "pp_read_g.json"), "w", encoding="utf-8"), indent=1)

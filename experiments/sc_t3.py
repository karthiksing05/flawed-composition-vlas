"""T3 (docs/SCAFFOLD_RESULTS.md 2; m3_prereg sc_t03): is contact-local behaviour layout-invariant?  LIBERO venv, CPU.  Dev states only.
fa1h_roll.episode's oracle-state episode for P_min, with an optional hand-over from the servo (pp_n4d.plan_episode's phases and controller):
  none   the episode from the start (the equivalence check against the atlas's tracks)
  A      the servo moves the hand to the grasp pre-pose (object + grasp_off + 0.10 m up, grasp rotation); then P_min, 300 steps
  B      the servo grasps, lifts and carries to above the release point; then P_min, 300 steps
A state where the servo does not reach the hand-over counts as not reached, and P_min does not run.

    python experiments/sc_t3.py smoke
    python experiments/sc_t3.py run --index 0          (0-143: seeds x modes x tids)
    python experiments/sc_t3.py read                   -> results/flawed/sc_t3_read.json
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
OD = os.path.join(OUT, "sc_t3")
SEEDS = (10, 11, 12)
MODES = ("A", "B")
COMPS = ("K1", "K2", "K3", "K4", "K5", "K6", "N1", "N2", "N3", "N4", "N5", "N6")
MAX_STEPS, WAIT, REPLAN, REACH_T, ERR, RERR, ARRIVE, B, MARGIN, FLOOR = 300, 10, 5, 120, 0.015, 0.10, 0.10, 2000, 0.10, 0.70
SOLID = (4, 5, 6)                                                           # stove cook region, cabinet top side, wine rack (FC.ENT order)


def tasks():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def task(n2, comp, lay):
    return next(t for t in n2["tasks"] if t["compound"] == comp and t["layout"] == lay)


def episode(model, bddl, state, lang, seed_base, kp, mode, seed):
    import torch
    import fa1_common as FC
    import fa1b_common as FB
    import fa1c_common as FK
    from fa1g_plan import ctrl, rotvec
    from lifelong_ood.policy import factored_flow as FF
    assert not getattr(model, "pgate", None) and not getattr(model, "role_lex", None), "P_min's oracle gate and given roles only"
    oe, te = FC.roles(lang)
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(state)
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    hm, cn = FB.Held(), FK.Canon()
    prev, h_on, off, zr, cy, rot0, d0t, done = -1.0, False, np.zeros(3), 0.0, 0.0, None, None, False
    ever_held, min_dxy, err = False, None, None

    def tgt():
        p = FC.scene(env, obs, dj)["pos"]
        return p[ti], float(p[FK.TABLE, 2])

    def track():
        h, o = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
        z, c = cn.step(h, con.pos(oe), tgt)
        return h, o, z, c

    def onset(was):
        """rot0 and d0t, fa1h_roll's rules, after h_on has been updated."""
        nonlocal rot0, d0t
        rot0 = (FC.scene(env, obs, dj)["ee_rot"].copy() if h_on and not was else rot0) if h_on else None
        d0t = (float(np.linalg.norm(np.asarray(con.pos(oe))[:2] - np.asarray(tgt()[0])[:2])) if h_on and not was else d0t) if h_on else None

    for _ in range(WAIT):
        obs, _, done, _ = env.step(FC.DUMMY)
    h_on, off, zr, cy = track()
    onset(False)
    sc0 = FC.scene(env, obs, dj)
    clear = min(float(np.linalg.norm(sc0["pos"][oi, :2] - sc0["pos"][f, :2])) for f in SOLID)
    reached, pre_steps, pre_err, pre_rerr = True, 0, None, None
    if mode != "none":
        rng = np.random.default_rng(int(seed_base) + int(seed))
        jit = rng.normal(0, 0.01, (2, 3))
        cz = kp["carry_z_above_ref"] + rng.normal(0, 0.02)
        goff, roff = np.array(kp["grasp_off"]) + jit[0], np.array(kp["release_off"]) + jit[1]
        RG, RR = np.array(kp["grasp_rot"]), np.array(kp["release_rot"])
        zref = float(sc0["pos"][8, 2])
        phase, in_phase, hold = 0, 0, 0
        last = 0 if mode == "A" else 4
        reached = False
        while phase <= last and pre_steps < 5 * REACH_T:
            sc = FC.scene(env, obs, dj)
            po, pt = sc["pos"][oi], sc["pos"][ti]
            if phase == 0:
                wp, Rt, g = po + goff + [0, 0, 0.10], RG, -1.0
            elif phase in (1, 2):
                wp, Rt, g = po + goff, RG, -1.0 if phase == 1 else 1.0
            elif phase == 3:
                wp, Rt, g = np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), RG, 1.0
            else:
                wp, Rt, g = np.array([pt[0] + roff[0], pt[1] + roff[1], zref + cz]), RR, 1.0
            a, e = ctrl(sc, wp, Rt, g, rng)
            obs, _, done, _ = env.step(a.tolist())
            prev = float(a[6])
            was = h_on
            h_on, off, zr, cy = track()
            onset(was)
            ever_held = ever_held or h_on
            pre_steps, in_phase = pre_steps + 1, in_phase + 1
            sc = FC.scene(env, obs, dj)
            e = float(np.linalg.norm(np.asarray(wp) - sc["ee_pos"]))
            r = float(np.linalg.norm(rotvec(np.asarray(Rt) @ sc["ee_rot"].T)))
            pre_err, pre_rerr = e, r
            if phase == 2:
                hold += 1
                if hold >= 12:
                    phase, in_phase = 3, 0
            elif phase == last:
                if e < ERR and r < RERR:
                    reached = True
                    break
                if in_phase >= REACH_T:
                    break
            elif phase == 1 and e < 0.008 or phase in (0, 3) and e < ERR or in_phase >= REACH_T:
                phase, in_phase = phase + 1, 0
        if mode == "B":
            dxy = float(np.linalg.norm(np.asarray(con.pos(oe))[:2] - np.asarray(tgt()[0])[:2]))
            reached = bool(h_on and dxy <= ARRIVE)                          # arrival is the atlas's: the planner tolerates a stalled phase 4
    base = dict(reached=bool(reached), pre_steps=pre_steps, pre_err=pre_err, pre_rerr=pre_rerr, clear=clear, start_obj=sc0["pos"][oi].tolist(), held_at_handover=bool(h_on))
    if not reached:
        env.close()
        return dict(base, success=False, held=bool(ever_held), error=None, steps=0, min_dxy_held=None)
    t, calls, plan = 0, 0, []
    try:
        while t < MAX_STEPS:
            if not plan:
                b = FF.batch_from_scene(FC.scene(env, obs, dj), oi, ti, prev, held=float(h_on), off=off, zrest=zr, cyaw=cy, rot0=rot0)
                if bool(getattr(model, "tscale", False)) and d0t is not None:
                    b["d0t"] = torch.tensor([d0t], dtype=torch.float32)
                g = torch.Generator().manual_seed(int(seed_base) + calls)
                ch = model.sample(b, generator=g)[0].numpy()
                plan = [np.clip(x, -1.0, 1.0) for x in ch[:REPLAN]]
                calls += 1
            act = plan.pop(0)
            obs, _, done, _ = env.step(act.tolist())
            prev = float(act[6])
            was = h_on
            h_on, off, zr, cy = track()
            onset(was)
            ever_held = ever_held or h_on
            if h_on:
                dxy = float(np.linalg.norm(con.pos(oe)[:2] - np.asarray(tgt()[0])[:2]))
                min_dxy = dxy if min_dxy is None else min(min_dxy, dxy)
            if done:
                break
            t += 1
    except Exception as ex:                                                 # noqa: BLE001
        err = repr(ex)
    env.close()
    return dict(base, success=bool(done), held=bool(ever_held), error=err, steps=t, min_dxy_held=min_dxy)


def run_one(seed, mode, tid, branches, outdir):
    import torch
    import fa1_common as FC
    from pp_l1_floor import kp_for
    from pp_p_roll import load_any
    fn = os.path.join(outdir, f"{mode}_s{seed}_{tid}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return None
    torch.set_num_threads(1)
    n2 = tasks()
    t = n2["tasks"][tid]
    assert t["tid"] == tid
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_full_naf_s{seed}", "P.npz"))
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
    kp = kp_for(t["compound"])
    eps = []
    for b in branches:
        t0 = time.time()
        e = episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"], kp, mode, seed)
        e.update(trial=int(b["trial"]), secs=time.time() - t0)
        eps.append(e)
        print("[t3]", mode, seed, t["compound"], t["layout"], b["trial"], json.dumps({k: e[k] for k in ("reached", "held", "success", "pre_steps", "pre_err", "pre_rerr", "clear", "secs", "error")}), flush=True)
    os.makedirs(outdir, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(compound=t["compound"], layout=t["layout"], mode=mode, seed=seed, tid=tid, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)
    return eps


def combos():
    n2 = tasks()
    tids = [task(n2, c, l)["tid"] for c in COMPS for l in ("familiar", "new")]
    return list(itertools.product(SEEDS, MODES, tids))


def br():
    return json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))


def smoke():
    """Smoke states 0-3, no dev read: each mode on one familiar and one new task.  Prints every field; writes under results/flawed/sc_t3_smoke."""
    n2 = tasks()
    sm = json.load(open(os.path.join(OUT, "n2_branches_smoke.json"), encoding="utf-8"))[:2]
    od = os.path.join(OUT, "sc_t3_smoke")
    for comp in ("K6", "N4"):
        for lay in ("familiar", "new"):
            for mode in ("none", "A", "B"):
                run_one(10, mode, task(n2, comp, lay)["tid"], sm, od)


def equiv():
    """The equivalence check: mode none against the atlas's P_min tracks, seed 10, dev states 4-7 of two tasks, success only."""
    import pp_atlas as PA
    n2 = tasks()
    agree, n = 0, 0
    for comp, lay in (("K6", "familiar"), ("N1", "new")):
        t = task(n2, comp, lay)
        rows = json.load(open(os.path.join(PA.OD, f"pmin_s10_{t['tid']}.json"), encoding="utf-8"))["eps"]
        ref = {int(e["trial"]): bool(e["success"]) for e in rows}
        eps = run_one(10, "none", t["tid"], [b for b in br() if int(b["trial"]) in (4, 5, 6, 7)], os.path.join(OUT, "sc_t3_equiv"))
        for e in eps or []:
            agree += int(bool(e["success"]) == ref[e["trial"]])
            n += 1
    print(f"[equiv] outcomes agree {agree} of {n}", flush=True)


def read():
    n2 = tasks()
    ks = n2["states"]["main"]
    A = {}
    for seed, mode, tid in combos():
        R = json.load(open(os.path.join(OD, f"{mode}_s{seed}_{tid}.json"), encoding="utf-8"))
        assert [int(e["trial"]) for e in R["eps"]] == list(ks), (mode, seed, tid)
        assert all(e["error"] is None for e in R["eps"]), (mode, seed, tid)
        for j, e in enumerate(R["eps"]):
            A[(mode, R["layout"], R["compound"], seed, j)] = e

    def arr(mode, lay, f):
        return np.array([[[f(A[(mode, lay, c, s, j)]) for j in range(len(ks))] for s in SEEDS] for c in COMPS], float)

    G = {}
    for mode in MODES:
        for lay in ("familiar", "new"):
            G[(mode, lay)] = dict(r=arr(mode, lay, lambda e: e["reached"]), h=arr(mode, lay, lambda e: e["reached"] and e["held"]),
                                  s=arr(mode, lay, lambda e: e["reached"] and e["success"]), c=arr(mode, lay, lambda e: e["clear"]))

    def cond(mode, lay, key, ti=None, sj=None, mask=None):
        g = G[(mode, lay)]
        ti = np.arange(len(COMPS)) if ti is None else ti
        sj = np.arange(len(ks)) if sj is None else sj
        m = np.ones_like(g["r"], bool) if mask is None else mask(g)
        r = (g["r"] * m)[ti][:, :, sj].sum()
        return float((g[key] * m)[ti][:, :, sj].sum() / r) if r > 0 else float("nan")

    def contrast(mode, key, mask=None, ti=None):
        rng = np.random.default_rng(0)
        base = np.arange(len(COMPS)) if ti is None else ti
        d = cond(mode, "new", key, base, None, mask) - cond(mode, "familiar", key, base, None, mask)
        v = []
        for _ in range(B):
            tt = rng.choice(base, len(base))
            v.append(cond(mode, "new", key, tt, rng.choice(len(ks), len(ks)), mask) - cond(mode, "familiar", key, tt, rng.choice(len(ks), len(ks)), mask))
        return dict(delta=d, ci=[float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))], familiar=cond(mode, "familiar", key, base, None, mask), new=cond(mode, "new", key, base, None, mask))

    DA, DB = contrast("A", "h"), contrast("B", "s")
    if DA["familiar"] < FLOOR or DB["familiar"] < FLOOR:
        lab = "INSTRUMENT FAILS"
    elif DA["ci"][1] < -MARGIN or DB["ci"][1] < -MARGIN:
        lab = "CONTACT FAILS"
    elif DA["ci"][0] >= -MARGIN and DB["ci"][0] >= -MARGIN:
        lab = "CONTACT INVARIANT"
    else:
        lab = "UNRESOLVED"
    near = lambda g: g["c"] < 0.10                                          # noqa: E731
    far = lambda g: g["c"] >= 0.10                                          # noqa: E731
    Ti = {"known": np.array([i for i, c in enumerate(COMPS) if c[0] == "K"]), "new_pairings": np.array([i for i, c in enumerate(COMPS) if c[0] == "N"])}
    rep = dict(label=lab, A_hold=DA, B_success=DB,
               A_success=contrast("A", "s"),
               reached={f"{m}|{l}": float(G[(m, l)]["r"].mean()) for m in MODES for l in ("familiar", "new")},
               unconditional_success={f"{m}|{l}": float(G[(m, l)]["s"].mean()) for m in MODES for l in ("familiar", "new")},
               far_from_fixture=dict(A_hold=contrast("A", "h", far), B_success=contrast("B", "s", far)),
               near_fixture=dict(A_hold=contrast("A", "h", near), B_success=contrast("B", "s", near)),
               by_group={g: dict(A_hold=contrast("A", "h", None, ti), B_success=contrast("B", "s", None, ti)) for g, ti in Ti.items()},
               per_task={c: {f"{m}|{l}": dict(reached=float(G[(m, l)]["r"][i].mean()), hold=cond(m, l, "h", np.array([i])), success=cond(m, l, "s", np.array([i]))) for m in MODES for l in ("familiar", "new")} for i, c in enumerate(COMPS)})
    intact = {}
    for lay in ("familiar", "new"):
        v = [np.mean([bool(e["success"]) for e in next(iter(json.load(open(os.path.join(OUT, "pp_cons_roll", f"P2naf_s{s}_main_{task(n2, c, lay)['tid']}.json"), encoding="utf-8"))["tasks"].values()))["episodes"]]) for c in COMPS for s in SEEDS]
        intact[lay] = float(np.mean(v))
    rep["intact_success"] = intact
    print("[t3]", lab, json.dumps(rep), flush=True)
    with open(os.path.join(OUT, "sc_t3_read.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("smoke", "equiv", "run", "read"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "run":
        s, m, tid = combos()[a.index]
        run_one(s, m, tid, br(), OD)
    else:
        {"smoke": smoke, "equiv": equiv, "read": read}[a.what]()

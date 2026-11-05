"""T3b (docs/SCAFFOLD_RESULTS.md 5; m3_prereg sc_t3b): who owns the new-layout loss after the pre-pose?  LIBERO venv, CPU.  Dev states only.
Every episode starts from T3's pre-pose.  Cell = grasp owner + carry owner, P (P_min) or S (servo).  P_min finishes every episode.

    python experiments/sc_t3b.py smoke
    python experiments/sc_t3b.py run --index 0          (0-287: seeds x cells x tids)
    python experiments/sc_t3b.py read                   -> results/flawed/sc_t3b_read.json
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
from sc_t3 import ARRIVE, COMPS, ERR, MAX_STEPS, OUT, REACH_T, REPLAN, RERR, SEEDS, WAIT, br, task, tasks     # noqa: E402

OD = os.path.join(OUT, "sc_t3b")
CELLS = ("PP", "PS", "SP", "SS")
GROUPS = dict(bowl=("K1", "K2", "K3"), cheese=("K6", "N1", "N2", "N3"), bottle=("K4", "N4", "N5", "N6"), bottle_side=("K5",))
B, DIS, EFFECT = 2000, 0.05, 0.10
RACK = 6                                                                    # FC.ENT


def kp_b(comp):
    """pp_l1_floor.kp_for, with a new pairing's carry height from its target's task, shifted by the difference in grasp height."""
    from pp_l1_floor import OBJ_K, TGT_K, kp_for
    out = dict(kp_for(comp))
    if comp[0] == "N":
        kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"]
        n2 = tasks()
        t = task(n2, comp, "familiar")
        ko, kt = kp[OBJ_K[t["object"]]], kp[TGT_K[t["target"]]]
        out["carry_z_above_ref"] = float(kt["carry_z_above_ref"]) + float(ko["grasp_off"][2]) - float(kt["grasp_off"][2])
    return out


def episode(model, bddl, state, lang, seed_base, kp, cell, seed):
    import torch
    import fa1_common as FC
    import fa1b_common as FB
    import fa1c_common as FK
    from fa1g_plan import ctrl, rotvec
    from lifelong_ood.policy import factored_flow as FF
    from pp_atlas import tilt_deg
    assert not getattr(model, "pgate", None) and not getattr(model, "role_lex", None)
    oe, te = FC.roles(lang)
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(state)
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    hm, cn = FB.Held(), FK.Canon()
    st = dict(prev=-1.0, h=False, off=np.zeros(3), zr=0.0, cy=0.0, rot0=None, d0t=None, done=False, ever=False, t=0, calls=0, servo=0, arrive=None, opened=None)
    plan = []
    obsb = [obs]

    def tgt():
        p = FC.scene(env, obsb[0], dj)["pos"]
        return p[ti], float(p[FK.TABLE, 2])

    def track():
        h, o = hm.step(*con.touching(oe), con.pos(oe), obsb[0]["robot0_eef_pos"])
        z, c = cn.step(h, con.pos(oe), tgt)
        return h, o, z, c

    def tilt():
        E = env.env
        return tilt_deg(E.sim.data.body_xquat[E.obj_body_id[oe]])

    def after(a):
        """fa1h_roll's rules for the gate's state, after one environment step."""
        st["prev"] = float(a[6])
        was = st["h"]
        st["h"], st["off"], st["zr"], st["cy"] = track()
        if st["h"]:
            d = float(np.linalg.norm(np.asarray(con.pos(oe))[:2] - np.asarray(tgt()[0])[:2]))
            if not was:
                st["rot0"], st["d0t"] = FC.scene(env, obsb[0], dj)["ee_rot"].copy(), d
            if st["arrive"] is None and d <= ARRIVE:
                st["arrive"] = dict(tilt=tilt(), d=d)
            if st["opened"] is None and a[6] < 0:
                st["opened"] = dict(tilt=tilt(), d=d)
        else:
            st["rot0"], st["d0t"] = None, None
        st["ever"] = st["ever"] or st["h"]

    def servo_step(wp, Rt, g, rng):
        sc = FC.scene(env, obsb[0], dj)
        a, _ = ctrl(sc, wp, Rt, g, rng)
        obsb[0], _, st["done"], _ = env.step(a.tolist())
        after(a)
        st["servo"] += 1
        sc = FC.scene(env, obsb[0], dj)
        return float(np.linalg.norm(np.asarray(wp) - sc["ee_pos"])), float(np.linalg.norm(rotvec(np.asarray(Rt) @ sc["ee_rot"].T)))

    def policy_step():
        if not plan:
            b = FF.batch_from_scene(FC.scene(env, obsb[0], dj), oi, ti, st["prev"], held=float(st["h"]), off=st["off"], zrest=st["zr"], cyaw=st["cy"], rot0=st["rot0"])
            if bool(getattr(model, "tscale", False)) and st["d0t"] is not None:
                b["d0t"] = torch.tensor([st["d0t"]], dtype=torch.float32)
            g = torch.Generator().manual_seed(int(seed_base) + st["calls"])
            ch = model.sample(b, generator=g)[0].numpy()
            plan.extend(np.clip(x, -1.0, 1.0) for x in ch[:REPLAN])
            st["calls"] += 1
        act = plan.pop(0)
        obsb[0], _, st["done"], _ = env.step(act.tolist())
        after(act)
        st["t"] += 1

    err = None
    for _ in range(WAIT):
        obsb[0], _, st["done"], _ = env.step(FC.DUMMY)
    st["h"], st["off"], st["zr"], st["cy"] = track()
    sc0 = FC.scene(env, obsb[0], dj)
    rack = float(np.linalg.norm(sc0["pos"][oi, :2] - sc0["pos"][RACK, :2]))
    rng = np.random.default_rng(int(seed_base) + int(seed))                 # T3's draws, in T3's order
    jit = rng.normal(0, 0.01, (2, 3))
    cz = kp["carry_z_above_ref"] + rng.normal(0, 0.02)
    goff, roff = np.array(kp["grasp_off"]) + jit[0], np.array(kp["release_off"]) + jit[1]
    RG, RR = np.array(kp["grasp_rot"]), np.array(kp["release_rot"])
    zref = float(sc0["pos"][8, 2])
    reached = False
    for k in range(REACH_T):
        po = FC.scene(env, obsb[0], dj)["pos"][oi]
        e, r = servo_step(po + goff + [0, 0, 0.10], RG, -1.0, rng)
        if e < ERR and r < RERR:
            reached = True
            break
    base = dict(reached=reached, rack=rack, start_obj=sc0["pos"][oi].tolist())
    if not reached:
        env.close()
        return dict(base, success=False, held=bool(st["ever"]), error=None, steps=0, servo_steps=st["servo"], arrive=None, opened=None, held_by_servo=False)
    try:
        if cell[0] == "P":
            while st["t"] < MAX_STEPS and not st["h"] and not st["done"]:
                policy_step()
        else:                                                               # the servo's grasp: T3's phases 1-3, until a hold
            for _ in range(REACH_T):
                if st["h"]:
                    break
                po = FC.scene(env, obsb[0], dj)["pos"][oi]
                e, _ = servo_step(po + goff, RG, -1.0, rng)
                if e < 0.008:
                    break
            for _ in range(12):
                if st["h"]:
                    break
                servo_step(FC.scene(env, obsb[0], dj)["pos"][oi] + goff, RG, 1.0, rng)
            for _ in range(REACH_T):
                if st["h"]:
                    break
                sc = FC.scene(env, obsb[0], dj)
                e, _ = servo_step(np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), RG, 1.0, rng)
                if e < ERR:
                    break
        held_by_servo = bool(cell[0] == "S" and st["h"])
        if cell[1] == "S" and st["h"] and not st["done"]:
            plan.clear()
            for _ in range(REACH_T):
                if st["done"]:
                    break
                sc = FC.scene(env, obsb[0], dj)
                e, _ = servo_step(np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), RG, 1.0, rng)
                if e < ERR:
                    break
            for _ in range(REACH_T):
                if st["done"]:
                    break
                sc = FC.scene(env, obsb[0], dj)
                pt = sc["pos"][ti]
                e, r = servo_step(np.array([pt[0] + roff[0], pt[1] + roff[1], zref + cz]), RR, 1.0, rng)
                if e < ERR and r < RERR:
                    break
        if cell[0] == "S" or cell[1] == "S":
            plan.clear()
        while st["t"] < MAX_STEPS and not st["done"]:
            policy_step()
    except Exception as ex:                                                 # noqa: BLE001
        err = repr(ex)
    env.close()
    return dict(base, success=bool(st["done"]), held=bool(st["ever"]), error=err, steps=st["t"], servo_steps=st["servo"], arrive=st["arrive"], opened=st["opened"], held_by_servo=held_by_servo)


def run_one(seed, cell, tid, branches, outdir):
    import torch
    import fa1_common as FC
    from pp_p_roll import load_any
    fn = os.path.join(outdir, f"{cell}_s{seed}_{tid}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    n2 = tasks()
    t = n2["tasks"][tid]
    assert t["tid"] == tid
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_full_naf_s{seed}", "P.npz"))
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
    kp = kp_b(t["compound"])
    eps = []
    for b in branches:
        t0 = time.time()
        e = episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"], kp, cell, seed)
        e.update(trial=int(b["trial"]), secs=time.time() - t0)
        eps.append(e)
        print("[t3b]", cell, seed, t["compound"], t["layout"], b["trial"], json.dumps({k: e[k] for k in ("reached", "held", "held_by_servo", "success", "steps", "servo_steps", "arrive", "opened", "rack", "error")}), flush=True)
    os.makedirs(outdir, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(compound=t["compound"], layout=t["layout"], cell=cell, seed=seed, tid=tid, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def combos():
    n2 = tasks()
    tids = [task(n2, c, l)["tid"] for c in COMPS for l in ("familiar", "new")]
    return list(itertools.product(SEEDS, CELLS, tids))


def smoke():
    """Smoke states 0-1: every cell on K1, N3 and N4, both layouts."""
    n2 = tasks()
    sm = json.load(open(os.path.join(OUT, "n2_branches_smoke.json"), encoding="utf-8"))[:2]
    for comp in ("K1", "N3", "N4"):
        for lay in ("familiar", "new"):
            for cell in CELLS:
                run_one(10, cell, task(n2, comp, lay)["tid"], sm, os.path.join(OUT, "sc_t3b_smoke"))


def read():
    n2 = tasks()
    ks = n2["states"]["main"]
    A = {}
    agree, n_ag = 0, 0
    for seed, cell, tid in combos():
        R = json.load(open(os.path.join(OD, f"{cell}_s{seed}_{tid}.json"), encoding="utf-8"))
        assert [int(e["trial"]) for e in R["eps"]] == list(ks), (cell, seed, tid)
        assert all(e["error"] is None for e in R["eps"]), (cell, seed, tid)
        for j, e in enumerate(R["eps"]):
            A[(cell, R["layout"], R["compound"], seed, j)] = e
        if cell == "PP":
            R3 = json.load(open(os.path.join(OUT, "sc_t3", f"A_s{seed}_{tid}.json"), encoding="utf-8"))
            for e, f in zip(R["eps"], R3["eps"]):
                agree += int(e["success"] == f["success"] and e["reached"] == f["reached"] and e["held"] == f["held"])
                n_ag += 1
    dis = 1.0 - agree / n_ag
    C_I = {c: i for i, c in enumerate(COMPS)}
    GI = {g: [C_I[c] for c in cs] for g, cs in GROUPS.items()}
    G_N = list(GROUPS)

    def arr(cell, lay, f):
        return np.array([[[f(A[(cell, lay, c, s, j)]) for j in range(len(ks))] for s in SEEDS] for c in COMPS], float)

    def owned(c, e):
        """A hold counts for its cell's grasp owner: in a servo-grasp cell, P_min's later pick-up after a failed servo grasp does not."""
        return e["reached"] and (e["held_by_servo"] if c[0] == "S" else e["held"])

    X = {(c, l): dict(r=arr(c, l, lambda e: e["reached"]), h=arr(c, l, lambda e, c=c: owned(c, e)), s=arr(c, l, lambda e, c=c: owned(c, e) and e["success"]), u=arr(c, l, lambda e: e["reached"] and e["success"])) for c in CELLS for l in ("familiar", "new")}

    def sub(a, gs, sj):
        return np.concatenate([a[GI[g]][:, :, sj].ravel() for g in gs])

    def sgh(c, l, gs, sj):
        h = sub(X[(c, l)]["h"], gs, sj).sum()
        return float(sub(X[(c, l)]["s"], gs, sj).sum() / h) if h > 0 else float("nan")

    def contrast(a, b, l, gs=None):
        gs = G_N if gs is None else gs
        sall = np.arange(len(ks))
        d = sgh(a, l, gs, sall) - sgh(b, l, gs, sall)
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            g2 = [gs[i] for i in rng.choice(len(gs), len(gs))]
            sj = rng.choice(len(ks), len(ks))
            v.append(sgh(a, l, g2, sj) - sgh(b, l, g2, sj))
        return dict(delta=d, ci=[float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))])

    G, C = contrast("SP", "PP", "new"), contrast("PS", "PP", "new")
    eff = lambda x: x["ci"][0] > 0.0 and x["delta"] > EFFECT                # noqa: E731
    lab = "INSTRUMENT FAILS" if dis > DIS else ("BOTH" if eff(G) and eff(C) else ("CARRY OWNS IT" if eff(C) else ("GRASP OWNS IT" if eff(G) else "NEITHER")))

    def tilts(c, l, key, gs):
        v = [A[(c, l, x, s, j)][key]["tilt"] for g in gs for x in GROUPS[g] for s in SEEDS for j in range(len(ks)) if A[(c, l, x, s, j)][key]]
        return dict(n=len(v), median=float(np.median(v)) if v else None, p90=float(np.percentile(v, 90)) if v else None)

    sall = np.arange(len(ks))
    rep = dict(label=lab, disagreement=dis, G=G, C=C, G_familiar=contrast("SP", "PP", "familiar"), C_familiar=contrast("PS", "PP", "familiar"), SS_vs_PP_new=contrast("SS", "PP", "new"),
               cells={f"{c}|{l}": dict(reached=float(X[(c, l)]["r"].mean()), hold_given_reached=float(X[(c, l)]["h"].sum() / max(X[(c, l)]["r"].sum(), 1)), s_given_hold=sgh(c, l, G_N, sall), unconditional=float(X[(c, l)]["u"].mean())) for c in CELLS for l in ("familiar", "new")},
               per_object={g: {f"{c}|{l}": dict(hold=float(sub(X[(c, l)]["h"], [g], sall).sum() / max(sub(X[(c, l)]["r"], [g], sall).sum(), 1)), s_given_hold=sgh(c, l, [g], sall), unconditional=float(sub(X[(c, l)]["u"], [g], sall).mean()), tilt_arrive=tilts(c, l, "arrive", [g]), tilt_open=tilts(c, l, "opened", [g])) for c in CELLS for l in ("familiar", "new")} for g in G_N},
               per_task={x: {f"{c}|{l}": dict(hold=float(X[(c, l)]["h"][C_I[x]].sum() / max(X[(c, l)]["r"][C_I[x]].sum(), 1)), success=float(X[(c, l)]["u"][C_I[x]].mean())) for c in CELLS for l in ("familiar", "new")} for x in COMPS})
    rk = {}
    for c in CELLS:
        for bin_, f in (("<.10", lambda v: v < 0.10), (">=.10", lambda v: v >= 0.10)):
            E = [A[(c, "new", x, s, j)] for x in GROUPS["cheese"] for s in SEEDS for j in range(len(ks)) if f(A[(c, "new", x, s, j)]["rack"])]
            rk[f"{c}|{bin_}"] = dict(n=len(E), reached=float(np.mean([e["reached"] for e in E])) if E else None, success=float(np.mean([e["success"] for e in E])) if E else None)
    rep["cheese_rack_strata_new"] = rk
    print("[t3b]", lab, json.dumps(rep), flush=True)
    with open(os.path.join(OUT, "sc_t3b_read.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("smoke", "run", "read"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "run":
        s, c, tid = combos()[a.index]
        run_one(s, c, tid, br(), OD)
    else:
        {"smoke": smoke, "read": read}[a.what]()

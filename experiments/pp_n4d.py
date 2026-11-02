"""PLAN_PARTS round 1, the N4 diagnostic (docs/PARTS_RESULTS.md section 10; m3_prereg pp_n4d).

gen     (LIBERO venv, CPU) 50 generated N4 (bottle on plate) carries from stage 1g's scripted oracle-state planner (fa1g_plan.py's
        controller, phases, layout sampler and acceptance rule, unchanged) with keypoints COMPOSED from the demos by the relocation
        rule: grasp offset / rotation, carry height, release rotation from K4; release offset = K4 grasp offset + (K1 release
        offset - K1 grasp offset).  Seeds 950000 + k (no earlier generator used them).  Rows in gen_g.npz's format (held / off by
        fa1b_common.Held over the kept steps, no-ops dropped) -> DATA/gen/pp_n4d_N4.npz, results/flawed/pp_n4d_gen.json
oracle  (LIBERO venv, CPU) the planner on N2's familiar dev states 4-23 (N2's start states; planner noise seeded 13000000 + 1000 k):
        N4 with the composed keypoints, K4 and K1 with their own (a1g_keypoints.json) -> results/flawed/pp_n4d_roll/oracle_<task>.json
        (the rollout episode fields success, held_step, min_dxy_held, final)
train   (openpi venv, GPU) single-task PartsFlow (rank 32) on ONE task's generated carries (N4: pp_n4d_N4.npz; K4 / K1: their rows of
        gen_g.npz), P's build recipe (30,000 steps, AdamW 3e-4, batch 512, EMA 0.999) -> DATA/ckpt/pp_n4d_<task>_s<seed>/P.npz
read    (CPU) -> results/flawed/pp_n4d_read.json (the mapping of section 10)

    python experiments/pp_n4d.py gen | oracle --task N4 | train --task N4 --seed 0 | read
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
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
GEN_BASE, N_GEN = 950000, 50
TASKS = {"N4": ("wine_bottle_1", "plate_1"), "K4": ("wine_bottle_1", "wooden_cabinet_1_top_side"), "K1": ("akita_black_bowl_1", "plate_1")}
TID = {"N4": 22, "K4": 6, "K1": 0}                                         # familiar task ids (2 x compound index)
DEV = list(range(4, 24))
B = 2000


def keypoints(task):
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"]
    if task != "N4":
        return kp[task]
    k4, k1 = kp["K4"], kp["K1"]
    out = dict(k4)
    out["release_off"] = (np.array(k4["grasp_off"]) + np.array(k1["release_off"]) - np.array(k1["grasp_off"])).tolist()
    out["object"], out["target"] = TASKS["N4"]
    out["composed"] = "K4 grasp / carry / release rotation; release_off = K4 grasp_off + (K1 release_off - K1 grasp_off)"
    return out


def plan_episode(bddl, s0, kp, oe, te, rng, servo=False):
    """fa1g_plan.gen's planner on one start state -> (rows dict, info).  Controller and phase rules unchanged.
    servo=True (PARTS_RESULTS 19, instrument v2): in the close / open phases keep servoing to the grasp / release point instead of
    freezing the motion."""
    import fa1b_common as FB
    from fa1g_plan import ctrl
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(s0)
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    hm = FB.Held()
    for _ in range(10):
        obs, _, done, _ = env.step(FC.DUMMY)
    jit = rng.normal(0, 0.01, (2, 3))
    cz = kp["carry_z_above_ref"] + rng.normal(0, 0.02)
    goff, roff = np.array(kp["grasp_off"]) + jit[0], np.array(kp["release_off"]) + jit[1]
    RG, RR = np.array(kp["grasp_rot"]), np.array(kp["release_rot"])
    ep = {x: [] for x in ("step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "touch", "objp")}
    phase, hold, prev, done, step, in_phase = 0, 0, -1.0, False, 0, 0
    sc = FC.scene(env, obs, dj)
    zref = float(sc["pos"][8, 2])
    held_step, min_dxy, rel_up = None, None, None
    while step < 300 and not done:
        sc = FC.scene(env, obs, dj)
        po, pt = sc["pos"][oi], sc["pos"][ti]
        if phase == 0:
            wp, Rt, g = po + goff + [0, 0, 0.10], RG, -1.0
        elif phase == 1:
            wp, Rt, g = po + goff, RG, -1.0
        elif phase == 2 and servo:
            wp, Rt, g = po + goff, RG, 1.0
        elif phase == 2:
            wp, Rt, g = ep["ee_pos"][-1] if hold else sc["ee_pos"], RG, 1.0
        elif phase == 3:
            wp, Rt, g = np.array([sc["ee_pos"][0], sc["ee_pos"][1], zref + cz]), RG, 1.0
        elif phase == 4:
            wp, Rt, g = np.array([pt[0] + roff[0], pt[1] + roff[1], zref + cz]), RR, 1.0
        elif phase == 5:
            wp, Rt, g = pt + roff, RR, 1.0
        elif phase == 6 and servo:
            wp, Rt, g = pt + roff, RR, -1.0
        elif phase == 6:
            wp, Rt, g = ep["ee_pos"][-1], RR, -1.0
        else:
            wp, Rt, g = rel_up, RR, -1.0
        a, err = ctrl(sc, wp, Rt, g, rng)
        if phase in (2, 6) and not servo:
            a[:6] = 0.0
        for x in ("ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
            ep[x].append(sc[x])
        ep["step"].append(step); ep["action"].append(a); ep["prev_grip"].append(prev)
        ep["touch"].append(con.touching(oe)); ep["objp"].append(con.pos(oe))
        obs, _, done, _ = env.step(a.tolist())
        h, _ = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
        if h and held_step is None:
            held_step = step
        if h:
            d = float(np.linalg.norm(np.asarray(con.pos(oe))[:2] - np.asarray(FC.scene(env, obs, dj)["pos"][ti])[:2]))
            min_dxy = d if min_dxy is None else min(min_dxy, d)
        prev, step = float(a[6]), step + 1
        in_phase += 1
        if phase in (0, 3, 4) and err < 0.015 or phase in (1, 5) and err < 0.008 or phase in (0, 1, 3, 4, 5) and in_phase >= 40:
            phase, in_phase = phase + 1, 0
        elif phase in (2, 6):
            hold += 1
            if hold >= 12:
                phase, hold, in_phase = phase + 1, 0, 0
                if phase == 7:
                    rel_up = sc["ee_pos"] + [0, 0, 0.10]
    fin = FC.scene(env, obs, dj)
    env.close()
    info = dict(success=bool(done), steps=step, phase=phase, held_step=held_step, min_dxy_held=min_dxy,
                final={e: fin["pos"][i].tolist() for i, e in enumerate(FC.ENT)}, error=None)
    return ep, info


def gen():
    import fa1b_common as FB
    from fa1_data import noop_mask
    from fa1g_plan import layout
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    t = next(x for x in n2["tasks"] if x["compound"] == "N4" and x["layout"] == "familiar")
    bddl = os.path.join(ROOT, t["bddl"])
    oe, te = TASKS["N4"]
    kp = keypoints("N4")
    lay_env = FC.make_env(bddl)
    cols = {x: [] for x in ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "held", "off")}
    rep, kept, k, t0 = [], 0, 0, time.time()
    while kept < N_GEN and k < 6 * N_GEN:
        seed = GEN_BASE + k
        rng = np.random.default_rng(seed)
        s0, p = layout(lay_env, n2, rng, seed)
        k += 1
        if s0 is None:
            rep.append(dict(seed=seed, layout=False))
            continue
        ep, info = plan_episode(bddl, s0, kp, oe, te, rng)
        rep.append(dict(seed=seed, layout=True, success=info["success"], steps=info["steps"], phase=info["phase"]))
        print(f"[pp_n4d gen] attempt {k - 1}: success={info['success']} phase={info['phase']} ({time.time() - t0:.0f}s)", flush=True)
        if not info["success"]:
            continue
        A = np.asarray(ep["action"])
        keep = np.nonzero(noop_mask(A))[0]
        hm, prev = FB.Held(), -1.0
        for s, i in enumerate(keep):
            lt, rt = bool(ep["touch"][i][0]), bool(ep["touch"][i][1])
            h, off = hm.step(lt, rt, ep["objp"][i], ep["ee_pos"][i])
            for x in ("action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                cols[x].append(ep[x][i])
            cols["task"].append(100); cols["demo"].append(kept); cols["step"].append(s); cols["prev_grip"].append(prev)
            cols["held"].append(float(h)); cols["off"].append(off)
            prev = float(A[i][6])
        kept += 1
    lay_env.close()
    out = {x: np.asarray(v, np.float32 if x not in ("task", "demo", "step") else np.int32) for x, v in cols.items()}
    os.makedirs(os.path.join(FC.DATA, "gen"), exist_ok=True)
    np.savez_compressed(os.path.join(FC.DATA, "gen", "pp_n4d_N4.npz"), **out)
    with open(os.path.join(OUT, "pp_n4d_gen.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(kept=kept, attempts=k, base=GEN_BASE, rows=int(len(out["task"])), keypoints=kp, episodes=rep,
                       note="scripted oracle-state planner (fa1g_plan), keypoints composed from demos; no demo action executed"), fh, indent=1)
    print(f"[pp_n4d gen] kept {kept} of {k}", flush=True)


def oracle(task):
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    t = n2["tasks"][TID[task]]
    assert t["compound"] == task and t["layout"] == "familiar"
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{TID[task]}.npy"))
    oe, te = TASKS[task]
    kp = keypoints(task)
    eps = []
    for k in DEV:
        rng = np.random.default_rng(13000000 + 1000 * k)
        _, info = plan_episode(os.path.join(ROOT, t["bddl"]), S[k], kp, oe, te, rng)
        info["trial"] = k
        eps.append(info)
        print(f"[pp_n4d oracle] {task} state {k}: success={info['success']} phase={info['phase']}", flush=True)
    od = os.path.join(OUT, "pp_n4d_roll")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"oracle_{task}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(task=task, keypoints=kp, tasks={str(TID[task]): dict(episodes=eps)}), fh, indent=1)


def train(task, seed):
    import torch
    import fa1c_common as FK
    from fa1_train import chunks
    from fa1g_train import rot0_rows
    from pp_p_train import fit
    from lifelong_ood.policy import parts_flow as PF
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if task == "N4":
        G = dict(np.load(os.path.join(FC.DATA, "gen", "pp_n4d_N4.npz")))
    else:
        G0 = dict(np.load(os.path.join(FC.DATA, "gen_g.npz")))
        m = G0["task"] == list(FC.KNOWN).index(task)
        G = {k: v[m] for k, v in G0.items() if hasattr(v, "__len__") and len(v) == len(m)}
    n = len(G["task"])
    oe, te = TASKS[task]
    oi, ti = np.full(n, FC.ENT.index(oe)), np.full(n, FC.ENT.index(te))
    zr, cy = FK.canon_rows(G, G["held"], oi, ti)
    r0 = rot0_rows(G, G["held"])
    X = chunks(G, 8)
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    center = D["ee_pos"].mean(0).tolist()                                    # P's centre (the demos of all 8 known tasks)
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v), dtype=dt, device=dev)   # noqa: E731
    b = dict(ee_pos=T(G["ee_pos"]), ee_rot=T(G["ee_rot"]), grip=T(G["grip"]), prev=T(G["prev_grip"]), pos=T(G["pos"]), yaw=T(G["yaw"]),
             drawer=T(G["drawer"]), oi=T(oi, torch.long), ti=T(ti, torch.long), held=T(G["held"]), off=T(G["off"]), zrest=T(zr), cyaw=T(cy),
             rot0=T(r0))
    size = np.array([FC.SIZE[e] for e in FC.ENT], np.float32)
    m = PF.PartsFlow(8, center=center, size=size, rank=32).to(dev)
    ema, curve, secs = fit(m, list(m.parameters()), b, T(np.clip(X, -1, 1)), 30000, 512, 3e-4, seed, 1e-4, log=f"n4d {task} s{seed}")
    d = os.path.join(FC.DATA, "ckpt", f"pp_n4d_{task}_s{seed}")
    os.makedirs(d, exist_ok=True)
    ema.cpu().save(os.path.join(d, "P.npz"), dict(tag=f"n4d_{task}", seed=seed, tasks=[task], center=center, steps=30000, data="generated"))
    with open(os.path.join(OUT, f"pp_n4d_train_{task}_s{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(task=task, seed=seed, n_rows=n, n_episodes=int(len(np.unique(G["demo"]))), secs=secs, curve=curve), fh, indent=1)


# ── read ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def ep_list(path):
    return [e for v in json.load(open(path, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]


def arrays(paths, task, rad, ref_xy):
    """-> per (seed, state) arrays: success, held, arrived, release-site error (xy), final object height above the plate."""
    oe, te = TASKS[task]
    S, H, A, E, Z = (np.full((len(paths), len(DEV)), np.nan) for _ in range(5))
    for i, p in enumerate(paths):
        for e in ep_list(p):
            j = DEV.index(int(e["trial"]))
            S[i, j] = float(bool(e["success"]))
            H[i, j] = float(e["held_step"] is not None)
            A[i, j] = float(e["min_dxy_held"] is not None and e["min_dxy_held"] <= rad)
            fo, ft = np.asarray(e["final"][oe]), np.asarray(e["final"][te])
            E[i, j] = float(np.linalg.norm(fo[:2] - ft[:2] - ref_xy)) if te == "plate_1" else np.nan
            Z[i, j] = float(fo[2] - ft[2])
    return S, H, A, E, Z


def phases(S, H, A, kk):
    s, h, a = S[:, kk], H[:, kk], A[:, kk]
    return dict(s=float(np.nanmean(s)), hold=float(np.nanmean(h)), arrival=float(np.nansum(a * h) / max(np.nansum(h), 1)),
                place_on_arrival=float(np.nansum(s * a) / max(np.nansum(a), 1)))


def boot(fn):
    rng = np.random.default_rng(0)
    v = [fn(rng.integers(0, len(DEV), len(DEV))) for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def read():
    from fa1g_read import radius
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    ti_k1 = list(FC.KNOWN).index("K1")
    rel = []
    for dm in np.unique(D["demo"][D["task"] == ti_k1]):                       # K1 demos: object - plate xy at the first open after grasp
        idx = np.nonzero((D["task"] == ti_k1) & (D["demo"] == dm))[0]
        a, h = D["action"][idx, 6], Db["held"][idx] > 0.5
        if not h.any():
            continue
        ic = int(np.argmax(a > 0))
        if not (a[ic:] < 0).any():
            continue
        j = idx[ic + int(np.argmax(a[ic:] < 0))]
        rel.append(D["pos"][j, FC.ENT.index("akita_black_bowl_1"), :2] - D["pos"][j, FC.ENT.index("plate_1"), :2])
    ref_xy = np.median(np.array(rel), 0)
    R = os.path.join(OUT, "pp_n4d_roll")
    arms = {}
    for task in ("N4", "K4", "K1"):
        rad = radius(*TASKS[task], cal)
        arms[f"a|{task}"] = arrays([os.path.join(R, f"N4d_{task}_s{s}_main_{TID[task]}.json") for s in (0, 1, 2)], task, rad, ref_xy)
        arms[f"P|{task}"] = arrays([os.path.join(R, f"P2r32_s{s}_main_{TID[task]}.json") for s in (10, 11, 12)], task, rad, ref_xy)
        arms[f"o|{task}"] = arrays([os.path.join(R, f"oracle_{task}.json")], task, rad, ref_xy)
    allk = np.arange(len(DEV))
    res = {}
    for k, (S, H, A, E, Z) in arms.items():
        r = phases(S, H, A, allk)
        r["s_ci"] = boot(lambda kk: float(np.nanmean(S[:, kk])))
        r["release_xy_err_median"] = float(np.nanmedian(E)) if np.isfinite(E).any() else None
        r["final_dz_median"] = float(np.nanmedian(Z))
        res[k] = r

    def dlt(t1, t2, key):
        (S1, H1, A1, _, _), (S2, H2, A2, _, _) = arms[t1], arms[t2]
        f = lambda kk: phases(S1, H1, A1, kk)[key] - phases(S2, H2, A2, kk)[key]   # noqa: E731
        return [f(allk), *boot(f)]
    d = {f"{key}|{t}": dlt(f"a|{t}", f"P|{t}", key) for t in ("N4", "K4", "K1") for key in ("s", "hold", "arrival", "place_on_arrival")}
    D4 = d["s|N4"]
    if D4[0] <= 0.10:
        verdict, phase = "competence", None
    elif D4[1] > 0:
        verdict = "combination"
        phase = max(("hold", "arrival", "place_on_arrival"), key=lambda k: d[f"{k}|N4"][0])
    else:
        verdict, phase = "unresolved", None
    flag = [t for t in ("K4", "K1") if d[f"s|{t}"][0] > 0.10 and d[f"s|{t}"][1] > 0]
    out = dict(arms=res, deltas_a_minus_P=d, verdict=verdict, failing_phase=phase, control_flag=flag, demo_release_ref_xy=ref_xy.tolist(),
               gen=json.load(open(os.path.join(OUT, "pp_n4d_gen.json"), encoding="utf-8"))["kept"])
    json.dump(out, open(os.path.join(OUT, "pp_n4d_read.json"), "w", encoding="utf-8"), indent=1)
    print("[pp_n4d] verdict", verdict, phase, "flag", flag, json.dumps(D4), flush=True)
    for k, r in res.items():
        print("[pp_n4d]", k, json.dumps({x: (round(v, 3) if isinstance(v, float) else v) for x, v in r.items()}), flush=True)
    for k, v in d.items():
        print("[pp_n4d] a-P", k, [round(x, 3) for x in v], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "oracle", "train", "read"))
    ap.add_argument("--task", default="N4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"gen": gen, "read": read}.get(a.mode, lambda: None)() if a.mode in ("gen", "read") else (
        oracle(a.task) if a.mode == "oracle" else train(a.task, a.seed))

"""S2CUE2 (docs/SCAFFOLD_RESULTS.md 58; m3_prereg tag s2_cue2): SR 57 found that in scene 2 (LIBERO-10 LIVING_ROOM_SCENE5) frozen pi0.5
carries every held mug to the plate on the side it STARTED from when the two mugs' start xy are swapped (P: 287/287), identity
ruled out.  In that layout the start side is also the side of the NEARER plate and fixes the CARRY DIRECTION.  This probe changes
only the INITIAL STATE again (same tasks, prompts, servers, client, launcher, branches) to ask (1) which cue routes the place when
neither plate is nearer and the mugs stand on no side, and (2) whether the position routing holds on a second model (pi0).

Manipulations (built here from SR 50's states; P is SR 57's, reused read-only from results/flawed/s2cue/P):
    C   central start: both the white and the yellow-and-white mug stand on the perpendicular bisector of the two plates' centres
        (equidistant from both plates, so neither plate is nearer and neither mug is on a side), each at its usual depth (x) order:
        the white mug at the smaller x (as in training).  The two mugs keep their own sampled x when they do not collide; otherwise
        they are pushed apart symmetrically along x (and the pair is shifted to larger x if the front mug touches the red mug) in
        5 mm steps until no mug-mug or mug-plate contact, then 1 cm extra clearance.  Each mug keeps its own height and orientation.
    Cp  (C') as C with the depth order swapped: the yellow-and-white mug at the smaller x.
    M   mirrored plates: plate_1 (the BDDL's "left plate" object) and plate_2 exchange xy; the mugs stand where they were sampled.
        The two plates are the same object model, so M is checked for pixel identity with the unmanipulated start (build).
States k = 0-63 are built (smoke 0-3, main 4-43 used); branches: SR 57's branches_40.json (SR 50's seed rule).

Modes:
    LIBERO venv (EGL):  python experiments/s2cue2.py build        -> results/flawed/s2cue2/{C,Cp,M}/s2_living_<tid>.npy, s2cue2_states.json
                        python experiments/s2cue2.py ref --manip C|Cp|M --tid T --branches B --out-dir D   (CPU)
                        python experiments/s2cue2.py summary DIR                                           (smoke)
                        python experiments/s2cue2.py read [DIR --selftest --out X]                         (CPU)
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import s2                                                                   # noqa: E402
import s2cue                                                                # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2cue2")
ROLL = os.path.join(OUT, "s2cue2_roll")
READ = os.path.join(OUT, "s2cue2_read.json")
STATES_JSON = os.path.join(CD, "s2cue2_states.json")
FRAMES = "/coc/flash10/ksingara3/.claude_tmp/pp/s2cue2_frames.npz"
SHEET = "/coc/flash10/ksingara3/.claude_tmp/pp/s2cue2_sheet.png"
W, Y, R = s2.MUGS
PL, PR = s2.PLATES                                                          # plate_1 = the instructions' "left plate" object
BUILT = ("C", "Cp", "M")
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
MAIN = list(range(4, 44))
SMOKE = list(range(0, 4))
# (model, manipulation, front-end arm, role); every run covers tids 0-3
RUNS = (("pi05", "C", "I", "PRIMARY"), ("pi05", "C", "tv", "SECONDARY"), ("pi05", "Cp", "I", "PRIMARY"),
        ("pi05", "Cp", "tv", "SECONDARY"), ("pi05", "M", "I", "PRIMARY"),
        ("pi0", "P", "I", "PRIMARY"), ("pi0", "P", "tv", "SECONDARY"), ("pi0", "M", "I", "PRIMARY"))
CUES = ("start_side", "nearer_plate", "plate_object", "instruction", "lr_order", "depth_order", "step")
TOL_SIDE, TOL_NEAR, TOL_LR, TOL_DEPTH = 0.05, 0.02, 0.03, 0.02
N_MIN, N_SEP = 20, 10
MAPPING = (
    "Models frozen: pi0.5 (pp_serve_purify --model pi05 --mode intact --comp= --seed 13) and pi0 (serve_guided --mode ours --seed 13), "
    "s2.py client unchanged run by s2_roll.sh unchanged (client seed 7, --bind track), FRESH = the manipulated states, SR 57's "
    "branches_40.json (SR 50's main states k = 4-43 and branch seeds).  Runs: pi0.5 C I (PRIMARY), C tv, C' I (PRIMARY), C' tv, M I "
    "(PRIMARY); pi0 P I (PRIMARY, SR 57's P states), P tv, M I (PRIMARY); tasks K1, K2, N1, N2 each (40 episodes).  Per episode: the "
    "HELD mug = libero_client's 'bound' (first object moved while the gripper is commanded closed) if it is the white or the "
    "yellow-and-white mug; a PLACEMENT = the held mug ends On exactly one plate (final LIBERO predicates).  Plates are scored by "
    "SIDE (the plate with the smaller y = the left plate; under M that is plate_2), since the two plates are the same object model.  "
    "Start positions = the mugs' and plates' xy after the 10 settle steps (s2cue2_states.json).  Cue predictions for the held mug's "
    "plate: START SIDE = the side of the table's centre line y = 0 the held mug starts on (undefined iff |y| < .05); NEARER PLATE = "
    "the plate nearer the held mug's start (undefined iff |d_left - d_right| < .02 m); PLATE OBJECT / IDENTITY = the held mug's "
    "trained plate OBJECT (white -> plate_1, yellow-and-white -> plate_2), at whichever side it stands; INSTRUCTION = the side the "
    "language names ('left plate' / 'right plate'), defined only when the held mug is the named mug; MUG LEFT/RIGHT ORDER = the left "
    "plate iff the held mug is the left one of the two (undefined iff |dy| < .03); DEPTH ORDER = the left plate iff the held mug is the "
    "one at the smaller x (the white mug's trained depth slot; undefined iff |dx| < .02); STEP = the left plate (the trained first "
    "carry).  Share f(c) = placements on c's plate / placements where c is defined, pooled over the run's four tasks; CI = 2.5 / 97.5 "
    "percentiles of B = 2000 bootstrap draws of the 40 states (numpy default_rng(0), s2cue.boot_share).  Per run and cue: ROUTES iff "
    "f >= .70 and CI lo > .50; RULED OUT iff f <= .30 and CI hi < .50; else UNRESOLVED; UNTESTABLE iff < 20 placements define c; NOT "
    "DEFINED iff none.  Separation: a cue is LABELLED only if, against every other cue d that is defined together with it on >= 10 "
    "placements, it disagrees with d on >= 10 placements; otherwise its share verdict is reported as INSEPARABLE from those d (if they "
    "route together the label is 'c OR d ROUTES').  Primary verdicts: pi0.5 C I, C' I, M I; pi0 P I, M I.  Success: per task, the "
    "named mug ends On the plate the language names (by side; under M this differs from the BDDL's object goal, which is also "
    "reported); paired to SR 50 on the same 40 states and branches (same model, same arm; pi0 tv against SR 50 pi0 I, which has no "
    "tv), state bootstrap, tasks fixed (s2.boot2_states).  C1 per manipulation (C, C', M): the reference planner (s2.plan_episode, "
    "unchanged; under M the target is the plate at the named side) on the same states; C1 MET iff its N1-N2 success by side >= .80.  "
    "Descriptive: first-held shares, held x destination tables, M versus SR 50 per-episode agreement (held mug and destination side).")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def state_dir(manip):
    return os.path.join(OUT, "s2cue", "P") if manip == "P" else os.path.join(CD, manip)


# ---------------------------------------------------------------- build (LIBERO venv, EGL)
def _tilt(E, n):
    w, x, y, z = E.sim.data.body_xquat[E.obj_body_id[n]]
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


def build(a):
    import s2_tasks as T2
    for m in BUILT:
        assert not os.path.exists(os.path.join(CD, m)), "exists: " + os.path.join(CD, m)
    assert not os.path.exists(STATES_JSON)
    M_ = s2.meta()
    S = np.load(os.path.join(s2.SD, "states", "%s_0.npy" % s2.SUITE))
    SP = np.load(os.path.join(state_dir("P"), "%s_0.npy" % s2.SUITE))
    for t in M_["tasks"][:4]:
        assert np.array_equal(np.load(os.path.join(s2.SD, "states", "%s_%d.npy" % (s2.SUITE, t["tid"]))), S)
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))
    env.seed(7)
    E = env.env
    jn = {n: E.objects_dict[n].joints[-1] for n in s2.MOVABLE}

    def pos(n):
        return np.array(E.sim.data.body_xpos[E.obj_body_id[n]], np.float64)

    def geoms(n):
        mdl = E.sim.model
        b0 = E.obj_body_id[n]
        bodies = set()
        for b in range(mdl.nbody):
            c = b
            while c > 0 and c != b0:
                c = mdl.body_parentid[c]
            if c == b0:
                bodies.add(b)
        return set(int(g) for g in np.where(np.isin(mdl.geom_bodyid, list(bodies)))[0])

    def touching(x, y):
        gx, gy = geoms(x), geoms(y)
        d = E.sim.data
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in gx and c.geom2 in gy) or (c.geom1 in gy and c.geom2 in gx):
                return True
        return False

    def qpos(n):
        return np.array(E.sim.data.get_joint_qpos(jn[n]))

    def setxy(n, q, xy):
        q = q.copy()
        q[:2] = xy
        E.sim.data.set_joint_qpos(jn[n], q)
        E.sim.data.set_joint_qvel(jn[n], np.zeros(6))

    def settle(obs, k_steps):
        for _ in range(k_steps):
            obs, _, _, _ = env.step(T2.DUMMY)
        return obs

    def img(o):
        return (np.ascontiguousarray(np.asarray(o["agentview_image"])[::-1, ::-1]),
                np.ascontiguousarray(np.asarray(o["robot0_eye_in_hand_image"])[::-1, ::-1]))

    def snapshot(obs):
        p = {n: pos(n) for n in s2.MOVABLE}
        return dict(xy={n: p[n][:2].tolist() for n in s2.MOVABLE}, eef=np.asarray(obs["robot0_eef_pos"], float).tolist(),
                    tilt=max(_tilt(E, n) for n in s2.MUGS),
                    on=[int(bool(E._eval_predicate(["on", m_, p_]))) for m_ in s2.MUGS for p_ in s2.PLATES],
                    mug_contacts=[int(touching(x, y)) for x, y in ((W, Y), (W, R), (Y, R))])

    def finish(sp, label, k, keep_frames):
        """Load the built state, run the client's 10 settle steps, return (state, checks, frames)."""
        env.reset()
        obs = env.set_init_state(sp)
        a0 = {n: pos(n) for n in s2.MOVABLE}
        obs = settle(obs, s2.WAIT)
        snap = snapshot(obs)
        snap["drift_xy"] = float(max(np.linalg.norm((pos(n) - a0[n])[:2]) for n in s2.MOVABLE))
        return snap, (img(obs) if keep_frames else None)

    rows, frames = [], dict(agent=[], wrist=[], names=[])
    for k in range(len(S)):
        row = dict(k=k)
        kf = k < 4
        # ---- unmanipulated (SR 50) and P (SR 57, read-only) start positions after the settle steps
        snap0, fr0 = finish(S[k], "orig", k, kf)
        row["orig"] = snap0
        row["P"], _ = finish(SP[k], "P", k, False)
        # ---- C and Cp
        for m, white_front in (("C", True), ("Cp", False)):
            env.reset()
            env.set_init_state(S[k])
            b0 = {n: pos(n) for n in s2.MOVABLE}
            q0 = {n: qpos(n) for n in (W, Y)}
            pl, pr = b0[PL][:2], b0[PR][:2]
            mid, v = (pl + pr) / 2, pr - pl

            def ybis(x):
                return mid[1] - (x - mid[0]) * v[0] / v[1]
            xf, xb = sorted([b0[W][0], b0[Y][0]])
            front, back = (W, Y) if white_front else (Y, W)
            gap, shift, it, ok_geom = 0.0, 0.0, 0, False
            while it < 80:
                x_f, x_b = xf - gap + shift, xb + gap + shift
                setxy(front, q0[front], [x_f, ybis(x_f)])
                setxy(back, q0[back], [x_b, ybis(x_b)])
                E.sim.forward()
                ff = touching(front, back)
                fr = touching(front, R) or touching(back, R)
                fp = any(touching(m_, p_) for m_ in (W, Y) for p_ in s2.PLATES)
                if not (ff or fr or fp):
                    ok_geom = True
                    break
                gap += 0.005 if ff or fp else 0.0
                shift += 0.005 if fr else 0.0
                it += 1
            # clearance: 1 cm on the mug-mug gap (and on the shift away from the red mug if one was needed); if the settled state
            # fails the checks (a mug leaning on a neighbour or tipping in the 10 settle steps), retry with +1 cm gap and +1 cm shift
            g0, s0 = gap, shift
            for retry in range(6):
                gap = g0 + 0.01 * (retry + 1)
                shift = (s0 + 0.01 * (retry + 1)) if (s0 > 0 or retry > 0) else 0.0
                env.reset()
                env.set_init_state(S[k])
                x_f, x_b = xf - gap + shift, xb + gap + shift
                setxy(front, q0[front], [x_f, ybis(x_f)])
                setxy(back, q0[back], [x_b, ybis(x_b)])
                E.sim.forward()
                sp = np.array(E.sim.get_state().flatten())
                snap, fr = finish(sp, m, k, kf)
                xyw, xyy = np.array(snap["xy"][W]), np.array(snap["xy"][Y])
                spl, spr = np.array(snap["xy"][PL]), np.array(snap["xy"][PR])
                eq = [float(abs(np.linalg.norm(xy - spl) - np.linalg.norm(xy - spr))) for xy in (xyw, xyy)]
                snap.update(gap_added=gap, shift=shift, iters=it, retry=retry, geom_ok=ok_geom, eq_err=eq, dx=float(xyy[0] - xyw[0]),
                            dy=float(xyy[1] - xyw[1]), white_front=white_front)
                snap["ok"] = bool(ok_geom and snap["drift_xy"] < 0.01 and snap["tilt"] < 10 and not any(snap["on"])
                                  and not any(snap["mug_contacts"]) and max(eq) < TOL_NEAR / 2
                                  and abs(snap["dx"]) > TOL_DEPTH and (snap["dx"] > 0) == white_front and abs(snap["dy"]) < TOL_LR)
                if snap["ok"]:
                    break
            row[m] = snap
            row[m + "_state"] = sp
            if kf:
                frames["agent"].append(fr[0])
                frames["wrist"].append(fr[1])
                frames["names"].append("%s_k%d" % (m, k))
        # ---- M: the plates exchange xy
        env.reset()
        env.set_init_state(S[k])
        qa, qb = qpos(PL), qpos(PR)
        setxy(PL, qa, qb[:2])
        setxy(PR, qb, qa[:2])
        E.sim.forward()
        sp = np.array(E.sim.get_state().flatten())
        snap, fr = finish(sp, "M", k, kf)
        snap["swap_err"] = float(max(np.linalg.norm(np.array(snap["xy"][PL]) - np.array(snap0["xy"][PR])),
                                     np.linalg.norm(np.array(snap["xy"][PR]) - np.array(snap0["xy"][PL]))))
        if kf:
            snap["pix_maxabs_vs_orig"] = dict(agent=int(np.abs(fr[0].astype(int) - fr0[0].astype(int)).max()),
                                              wrist=int(np.abs(fr[1].astype(int) - fr0[1].astype(int)).max()))
            frames["agent"] += [fr0[0], fr[0]]
            frames["wrist"] += [fr0[1], fr[1]]
            frames["names"] += ["orig_k%d" % k, "M_k%d" % k]
        snap["ok"] = bool(snap["swap_err"] < 0.01 and snap["drift_xy"] < 0.01 and snap["tilt"] < 10 and not any(snap["on"]))
        row["M"] = snap
        row["M_state"] = sp
        rows.append(row)
        print("[s2cue2 build] k %d | C %s | Cp %s | M %s" % (k, json.dumps({x: row["C"][x] for x in (
            "xy", "gap_added", "shift", "iters", "retry", "eq_err", "dx", "dy", "drift_xy", "tilt", "mug_contacts", "ok")}),
            json.dumps({x: row["Cp"][x] for x in ("gap_added", "shift", "retry", "eq_err", "dx", "dy", "drift_xy", "tilt", "mug_contacts", "ok")}),
            json.dumps({x: row["M"].get(x) for x in ("swap_err", "drift_xy", "tilt", "pix_maxabs_vs_orig", "ok")})), flush=True)
    env.close()
    allok = {m: all(r[m]["ok"] for r in rows) for m in BUILT}
    allok.update({m + "_main": all(r[m]["ok"] for r in rows if r["k"] in MAIN) for m in BUILT})
    for m in BUILT:
        A = np.stack([r.pop(m + "_state") for r in rows])
        os.makedirs(os.path.join(CD, m))
        for t in M_["tasks"][:4]:
            np.save(os.path.join(CD, m, "%s_%d.npy" % (s2.SUITE, t["tid"])), A)
    with open(STATES_JSON, "w", encoding="utf-8") as fh:
        json.dump(dict(rule=__doc__.split("Manipulations")[1].split("States k")[0].strip(), source=os.path.join(s2.SD, "states"),
                       P_source=state_dir("P"), ok=allok, rows=rows), fh, indent=1)
    os.makedirs(os.path.dirname(FRAMES), exist_ok=True)
    np.savez_compressed(FRAMES, agent=np.stack(frames["agent"]), wrist=np.stack(frames["wrist"]), names=np.array(frames["names"]))
    try:
        from PIL import Image
        ims = frames["agent"]
        ncol = 4
        nrow = (len(ims) + ncol - 1) // ncol
        h, w_ = ims[0].shape[:2]
        sheet = np.full((nrow * h, ncol * w_, 3), 255, np.uint8)
        for i, im in enumerate(ims):
            sheet[(i // ncol) * h:(i // ncol + 1) * h, (i % ncol) * w_:(i % ncol + 1) * w_] = im
        Image.fromarray(sheet).save(SHEET)
        print("[s2cue2 build] sheet order %s -> %s" % (frames["names"], SHEET), flush=True)
    except Exception as ex:                                                 # noqa: BLE001
        print("[s2cue2 build] no sheet: %r" % ex, flush=True)
    print("[s2cue2 build] ok %s; wrote %s" % (allok, CD), flush=True)


# ---------------------------------------------------------------- reference planner (LIBERO venv, CPU)
def side_plate(manip, side):
    """The plate OBJECT standing at side 'L' / 'R' under the manipulation."""
    if manip == "M":
        return PR if side == "L" else PL
    return PL if side == "L" else PR


def plate_side(manip, plate):
    return None if plate not in (PL, PR) else ("L" if side_plate(manip, "L") == plate else "R")


def lang_side(t):
    return "L" if t["target"] == PL else "R"                                 # the language's "left plate" is the BDDL's plate_1


def ref(a):
    import time
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    assert t["tid"] == a.tid and a.manip in BUILT
    os.makedirs(a.out_dir, exist_ok=True)
    fn = os.path.join(a.out_dir, "ref_%s_%d.json" % (a.manip, a.tid))
    assert not os.path.exists(fn), "exists: " + fn
    kp = s2.kp_for(t["key"])
    te = side_plate(a.manip, lang_side(t))
    S = np.load(os.path.join(state_dir(a.manip), "%s_%d.npy" % (s2.SUITE, a.tid)))
    eps = []
    for b in json.load(open(a.branches, encoding="utf-8")):
        rng = np.random.default_rng(int(b["seed_base"]) + 10)                # s2.ref's rule
        t0 = time.time()
        try:
            e = s2.plan_episode(os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], kp, t["object"], te, rng, int(t["max_steps"]))
            e["success_side"] = bool(s2.on_of(e["on"], t["object"], te))
        except Exception as ex:                                             # noqa: BLE001
            e = dict(success=False, success_side=False, error=repr(ex))
        e.update(trial=int(b["trial"]), secs=time.time() - t0, target_object=te)
        eps.append(e)
        print("[s2cue2 ref] %s %s trial %d success(BDDL) %s success(side) %s held %s phase %s on %s final dxy %s (%.0fs)" % (
            a.manip, t["key"], b["trial"], e["success"], e["success_side"], e.get("held_step"), e.get("phase"), e.get("on"),
            e.get("final_dxy"), e["secs"]), flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(manip=a.manip, tid=a.tid, key=t["key"], kp=kp, target_object=te, eps=eps), fh, indent=1)


# ---------------------------------------------------------------- scoring (CPU)
def cue_preds(manip, st, t, h):
    """-> {cue: 'L' | 'R' | None} for the held mug h, from the settled start positions st (s2cue2_states.json row[manip])."""
    xy = {n: np.array(v) for n, v in st["xy"].items()}
    o = Y if h == W else W
    pls, prs = xy[side_plate(manip, "L")], xy[side_plate(manip, "R")]
    m = xy[h]
    dl, dr = np.linalg.norm(m - pls), np.linalg.norm(m - prs)
    dy, dx = m[1] - xy[o][1], m[0] - xy[o][0]
    return dict(start_side=None if abs(m[1]) < TOL_SIDE else ("L" if m[1] < 0 else "R"),
                nearer_plate=None if abs(dl - dr) < TOL_NEAR else ("L" if dl < dr else "R"),
                plate_object=plate_side(manip, s2.HABIT[h]),
                instruction=lang_side(t) if h == t["object"] else None,
                lr_order=None if abs(dy) < TOL_LR else ("L" if dy < 0 else "R"),
                depth_order=None if abs(dx) < TOL_DEPTH else ("L" if dx < 0 else "R"),
                step="L")


def score(stem, t, ks, manip, srows):
    out = []
    for k, r in zip(ks, s2.stage_rows(stem, t, ks)):
        h = r["bound"]
        dest = s2cue.plate_of(r["on"], h) if h in (W, Y) else None
        d = dict(k=k, success_bddl=bool(r["success"]), held=h, dest=dest, dest_side=plate_side(manip, dest),
                 success_side=bool(s2.on_of(r["on"], t["object"], side_plate(manip, lang_side(t)))),
                 white=plate_side(manip, s2cue.plate_of(r["on"], W)), yw=plate_side(manip, s2cue.plate_of(r["on"], Y)))
        d["pred"] = cue_preds(manip, srows[k][manip if manip else "orig"], t, h) if h in (W, Y) else None
        out.append(d)
    return out


def placements(E):
    return [(i, e) for eps in E.values() for i, e in enumerate(eps) if e["dest_side"] in ("L", "R")]


def label_run(E, ks):
    P_ = placements(E)
    res = {}
    for c in CUES:
        F, D = np.zeros(len(ks)), np.zeros(len(ks))
        for i, e in P_:
            p = e["pred"][c]
            if p is not None:
                D[i] += 1
                F[i] += float(p == e["dest_side"])
        sh = s2cue.boot_share(F, D)
        if sh[3] == 0:
            v = "NOT DEFINED"
        elif sh[3] < N_MIN:
            v = "UNTESTABLE"
        elif sh[0] >= 0.70 and sh[1] > 0.50:
            v = "ROUTES"
        elif sh[0] <= 0.30 and sh[2] < 0.50:
            v = "RULED OUT"
        else:
            v = "UNRESOLVED"
        res[c] = dict(share=sh, verdict=v)
    for c in CUES:
        pair = {}
        for d in CUES:
            if d == c:
                continue
            both = [(e["pred"][c], e["pred"][d]) for _, e in P_ if e["pred"][c] is not None and e["pred"][d] is not None]
            pair[d] = dict(n_both=len(both), n_disagree=int(sum(x != y for x, y in both)))
        insep = [d for d, v in pair.items() if v["n_both"] >= N_SEP and v["n_disagree"] < N_SEP]
        res[c].update(pairs=pair, inseparable_from=insep, separated=not insep)
        v = res[c]["verdict"]
        if v in ("NOT DEFINED", "UNTESTABLE") or not insep:
            res[c]["label"] = v
        else:
            res[c]["label"] = "%s, INSEPARABLE from %s" % (v, "/".join(insep))
    groups, seen = [], set()
    for c in CUES:
        if c in seen or res[c]["verdict"] != "ROUTES":
            continue
        g = [c] + [d for d in res[c]["inseparable_from"] if res[d]["verdict"] == "ROUTES"]
        seen.update(g)
        groups.append(" OR ".join(g) + " ROUTES")
    return dict(cues=res, routing=groups, n_placements=len(P_))


def describe(E):
    d = {}
    for c, eps in E.items():
        d[c] = dict(success_side=float(np.mean([e["success_side"] for e in eps])),
                    success_bddl=float(np.mean([e["success_bddl"] for e in eps])),
                    held={str(m): float(np.mean([e["held"] == m for e in eps])) for m in (W, Y, R, PL, PR, None)},
                    held_dest={m: {str(p): int(sum(e["held"] == m and e["dest_side"] == p for e in eps)) for p in ("L", "R", None)}
                               for m in (W, Y)},
                    white_final={str(p): float(np.mean([e["white"] == p for e in eps])) for p in ("L", "R", None)},
                    yw_final={str(p): float(np.mean([e["yw"] == p for e in eps])) for p in ("L", "R", None)}, n=len(eps))
    return d


def summary(d):
    M_ = s2.meta()
    srows = {r["k"]: r for r in json.load(open(STATES_JSON, encoding="utf-8"))["rows"]}
    for model, manip, fe, _ in RUNS:
        for tid in TIDS:
            stem = os.path.join(d, manip, "%s_%s_%d" % (model, fe, tid))
            if not (os.path.exists(stem + ".json") and os.path.exists(stem + "_track.npz")):
                continue
            t = M_["tasks"][tid]
            R_ = json.load(open(stem + ".json", encoding="utf-8"))
            ks = [int(e["trial"]) for v in R_["tasks"].values() for e in v["episodes"]]
            for e in score(stem, t, ks, manip, srows):
                print("[s2cue2 smoke] %s %s %s %s %s" % (model, manip, fe, t["key"], json.dumps(e)), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_")) if os.path.isdir(d) else []:
        R_ = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2cue2 smoke] %s: %s" % (f, [(e["trial"], e["success_side"], e.get("on")) for e in R_["eps"]]), flush=True)


def read(a):
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    M_ = s2.meta()
    T = {t["tid"]: t for t in M_["tasks"]}
    srows = {r["k"]: r for r in json.load(open(STATES_JSON, encoding="utf-8"))["rows"]}
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD, base_ks, mism = SMOKE, a.dir, None, "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_cue2"]
        assert len(rows) == 1, "expected exactly one s2_cue2 prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        mism += [p for p, h in rows[0]["metrics"]["sr50_reused_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD, base_ks = MAIN, ROLL, M_["states"]["main"]
    res = dict(states=ks, mapping=MAPPING, code_mismatch=mism, runs={}, reference={})
    for model, manip, fe, role in RUNS:
        name = "%s %s %s" % (model, manip, fe)
        E, miss = {}, []
        for tid in TIDS:
            stem = os.path.join(RD, manip, "%s_%s_%d" % (model, fe, tid))
            if not (os.path.exists(stem + ".json") and os.path.exists(stem + "_track.npz")):
                miss.append(stem)
                continue
            E[T[tid]["key"]] = score(stem, T[tid], ks, manip, srows)
        if miss:
            res["runs"][name] = "MISSING %s" % miss
            print("== %s: MISSING %d" % (name, len(miss)), flush=True)
            continue
        row = dict(role=role, describe=describe(E), labels=label_run(E, ks))
        if base_ks is not None:
            row["vs_sr50"], agree = {}, {}
            bfe = "I" if model == "pi0" else fe
            for tid in TIDS:
                c = T[tid]["key"]
                b = s2.stage_rows(os.path.join(s2.ROLL, "%s_%s_%d" % (model, bfe, tid)), T[tid], base_ks)[:len(ks)]
                x = np.array([[float(e["success_side"]) for e in E[c]]])
                y = np.array([[float(r_["success"]) for r_ in b]])
                row["vs_sr50"][c] = dict(manip=float(x.mean()), sr50=float(y.mean()), sr50_arm=bfe, d=s2.boot2_states(x, y))
                if manip == "M":
                    bb = score(os.path.join(s2.ROLL, "%s_%s_%d" % (model, bfe, tid)), T[tid], base_ks, None, srows)[:len(ks)]
                    agree[c] = dict(held=float(np.mean([e["held"] == f["held"] for e, f in zip(E[c], bb)])),
                                    held_and_dest_side=float(np.mean([e["held"] == f["held"] and e["dest_side"] == f["dest_side"]
                                                                      for e, f in zip(E[c], bb)])))
            if agree:
                row["M_vs_sr50_agreement"] = agree
        res["runs"][name] = row
        L = row["labels"]
        print("== %s (%s): placements %d | routing %s" % (name, role, L["n_placements"], L["routing"]), flush=True)
        for c, v in L["cues"].items():
            print("   %-13s share %s -> %s" % (c, np.round([x if x is not None else np.nan for x in v["share"][:3]], 3).tolist()
                                              + [v["share"][3]], v["label"]), flush=True)
        for c, v in row["describe"].items():
            print("   %s %s" % (c, json.dumps(v)), flush=True)
        if "vs_sr50" in row:
            print("   vs SR 50 %s" % json.dumps(row["vs_sr50"]), flush=True)
        if "M_vs_sr50_agreement" in row:
            print("   M vs SR 50 agreement %s" % json.dumps(row["M_vs_sr50_agreement"]), flush=True)
    RF = a.ref_dir or RD
    for manip in BUILT:
        per = {}
        for tid in TIDS:
            f = os.path.join(RF, "ref_%s_%d.json" % (manip, tid))
            if os.path.exists(f):
                ee = json.load(open(f, encoding="utf-8"))["eps"]
                assert [e["trial"] for e in ee] == ks, (f, [e["trial"] for e in ee])
                per[T[tid]["key"]] = dict(success_side=float(np.mean([e["success_side"] for e in ee])),
                                          errors=int(sum(e.get("error") is not None for e in ee)))
        new = [per[c]["success_side"] for c in ("N1", "N2") if c in per]
        res["reference"][manip] = dict(per_task=per, success_new=float(np.mean(new)) if len(new) == 2 else None,
                                       C1=(None if len(new) < 2 else ("C1 MET" if np.mean(new) >= 0.80 else "C1 UNMET")))
        print("== reference %s: %s" % (manip, json.dumps(res["reference"][manip])), flush=True)
    print("== code hash mismatches: %s" % mism, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, out)
    print("[s2cue2 read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--manip", choices=BUILT)
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--ref-dir")
    ap.add_argument("--out", default=READ)
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    dict(build=build, ref=ref, summary=lambda a: summary(a.dir), read=read)[A.mode](A)

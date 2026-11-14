"""S2CUE (docs/SCAFFOLD_RESULTS.md 57; m3_prereg tag s2_cue): in scene 2 (LIBERO-10 LIVING_ROOM_SCENE5, SR 50) F1 is total (0/720 on
the new pairings) and the identity-hiding role views (tv paints every mug, the held one included, and the non-target plate in the
target phase; SR 54) give exactly 0.  Which input cue then routes the PLACE?  The candidates left once identity is hidden:
    (p) the held mug's START position (training always takes the white mug from its region, left of centre, to the left plate, and
        the yellow-and-white mug from its region, right of centre, to the right plate: start region = the nearer plate),
    (q) the step in the trained two-step sequence (the first carry goes to the left plate, the second to the right plate),
    (r) the arm's own state (pi0.5's observation/state; SR 53 found pi05_libero never reads it -- `statecheck` re-measures it on this
        checkpoint with this scene's prompts and renders),
    (s) the rest of the scene layout.
Each manipulation changes only the INITIAL STATE (no task information is added; prompts, servers, clients, branches unchanged):
    P  swap starts: the white mug's and the yellow-and-white mug's xy are exchanged (each keeps its own height and orientation; every
       mug's yaw range in the BDDL is 0, so both stand upright as sampled).  The white mug then starts in the yellow-and-white mug's
       region and vice versa.
    Q  step 1 done: the white mug starts ON the left plate (dropped from just above the plate's centre and settled), the trained
       first step's end state; everything else as sampled.  K1 is satisfied at t = 0 under Q, so Q runs K2, N1, N2 only.
States: SR 50's main states k = 4-43 (the first 40 of 4-63; results/flawed/s2/states), branches seed_base 23000000 + 1000 k (SR 50's
rule; branches_40.json is branches_main.json's first 40 rows).  Front-end arms: tv (PRIMARY: identity hidden in the target phase) and
I (SECONDARY: identity visible), both s2.py's client, unchanged, run by s2_roll.sh, unchanged, with FRESH pointed at the manipulated
states.  pi0.5 only (GR00T is NOT TESTABLE per SR 50: K1-K2 .508 < .70).  Reference (C1): s2.plan_episode, unchanged, from the
manipulated states.

Modes:
    LIBERO venv (EGL):  python experiments/s2cue.py build          -> results/flawed/s2cue/{P,Q}/s2_living_<tid>.npy (k 0-63),
                                                                       branches_40.json, s2cue_states.json; FRAMES (statecheck input)
                        python experiments/s2cue.py ref --manip P|Q --tid T --branches B --out-dir D       (CPU)
                        python experiments/s2cue.py summary DIR                                            (smoke)
                        python experiments/s2cue.py read [--out results/flawed/s2cue_read.json] [--selftest]   (CPU)
    openpi venv (GPU):  python experiments/s2cue.py statecheck --out X.json
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

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2cue")
ROLL = os.path.join(OUT, "s2cue_roll")
READ = os.path.join(OUT, "s2cue_read.json")
FRAMES = "/coc/flash10/ksingara3/.claude_tmp/pp/s2cue_frames.npz"
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
MANIP_TIDS = dict(P=(0, 1, 2, 3), Q=(1, 2, 3))                             # tid 0 K1, 1 K2, 2 N1, 3 N2
FE = ("tv", "I")                                                            # tv PRIMARY, I SECONDARY
MODEL = "pi05"
MAIN = list(range(4, 44))
SMOKE = list(range(0, 4))
# what each candidate cue predicts for the plate the HELD (first-moved) mug ends on
POS = dict(P={W: PR, Y: PL}, Q={W: None, Y: PR})                            # start region's trained plate (None: never a start)
STEP = dict(P={W: PL, Y: PL}, Q={W: None, Y: PR})                           # P: the first carry -> left; Q: the second carry -> right
N_MIN_ALL, N_MIN_STRATUM = 20, 10
MAPPING = (
    "pi0.5, frozen, pp_serve_purify --mode intact --seed 13; s2.py client unchanged (arms tv PRIMARY, I SECONDARY), client seed 7, "
    "--bind track; SR 50's main states k = 4-43 (40) with SR 50's branch seeds, manipulated at the initial state only: P (white and "
    "yellow-and-white mugs' start xy swapped; tasks K1, K2, N1, N2) and Q (white mug starts on the left plate; tasks K2, N1, N2).  "
    "Per episode: the HELD mug = the first mug moved (libero_client's 'bound'); a PLACEMENT = the held mug ends On exactly one plate "
    "(final LIBERO predicates).  Cue predictions for the held mug's plate: INSTRUCTION = the task's target if the held mug is the "
    "named mug (else not followed); IDENTITY (trained mapping) = white -> left, yellow-and-white -> right; POSITION = the trained "
    "plate of the region the held mug STARTED in (P: white -> right, yellow-and-white -> left; Q: yellow-and-white -> right, white "
    "undefined); STEP = the trained plate of this carry's place in the two-step program (P: first carry -> left for either mug; Q: "
    "yellow-and-white -> right, white undefined).  Share f(c) = placements on c's plate / placements where c is defined, pooled over "
    "the manipulation's tasks; CI = 2.5 / 97.5 percentiles of B = 2000 bootstrap draws of the 40 states (numpy default_rng(0), tasks "
    "pooled, the same state draw for every task); strata = placements with the white mug held and with the yellow-and-white mug held.  "
    "P, per front-end arm: cue c ROUTES iff f(c) >= .70 and CI lo > .50 and every stratum with >= 10 placements where c is defined "
    "has f(c) >= .50; RULED OUT iff CI hi < .50; else UNRESOLVED; UNTESTABLE iff fewer than 20 placements.  POSITION ROUTES and STEP "
    "ROUTES can both hold only when the white-held stratum (the one where they disagree) has < 10 placements: then the label is "
    "POSITION OR STEP (inseparable); likewise IDENTITY OR STEP when the yellow-and-white-held stratum has < 10.  Under P position and "
    "identity always disagree.  Q, per front-end arm, on Q-N2 (the brief's test): placements = the yellow-and-white mug ends On "
    "exactly one plate (whichever mug moved first); PROGRAM CONTINUES (the trained second step routes) iff the share on the right "
    "plate >= .70 and CI lo > .50; INSTRUCTION ROUTES iff the share on the left plate >= .70 and lo > .50; else UNRESOLVED; "
    "UNTESTABLE iff < 10 placements.  Under Q position, identity and step all say right for that mug, so Q does not separate them; "
    "only P does.  STATE (r): RULED OUT BY CONSTRUCTION iff statecheck gives max |action change| = 0.0 under every state swap while "
    "image and prompt swaps move the actions; else a state-swap arm would be needed (not registered).  C1 per manipulation: the "
    "reference planner's success on N1-N2 >= .80 -> C1 MET, else C1 UNMET (the cue labels stand either way; C1 says whether the "
    "instructed placement is reachable from these starts).  Descriptive: per-task success and the paired difference to SR 50's I / tv "
    "on the same 40 states and branches (state bootstrap, tasks fixed); the same cue shares on SR 50's unmanipulated rollouts (where "
    "position = identity); first-moved mug shares; the other mug's final plate; Q-N1 and Q-K2 placements.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------- build (LIBERO venv)
def _tilt(env, n):
    E = env.env
    w, x, y, z = E.sim.data.body_xquat[E.obj_body_id[n]]
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


def build(a):
    import s2_tasks as T2
    for m in ("P", "Q"):
        assert not os.path.exists(os.path.join(CD, m)), "exists: " + os.path.join(CD, m)
    M = s2.meta()
    S = np.load(os.path.join(s2.SD, "states", "%s_0.npy" % s2.SUITE))
    for t in M["tasks"][:4]:
        assert np.array_equal(np.load(os.path.join(s2.SD, "states", "%s_%d.npy" % (s2.SUITE, t["tid"]))), S)
    env = T2.make_env(os.path.join(ROOT, M["tasks"][0]["bddl"]))
    env.seed(7)
    E = env.env
    jn = {n: E.objects_dict[n].joints[-1] for n in s2.MOVABLE}

    class _Sim:                                                             # env.reset() can rebuild the MjSim: always the current one
        def __getattr__(self, k):
            return getattr(E.sim, k)
    sim = _Sim()

    def pos(n):
        return np.array(E.sim.data.body_xpos[E.obj_body_id[n]], np.float64)

    def settle(k_steps):
        for _ in range(k_steps):
            env.step(T2.DUMMY)

    PS, QS, rows, frames = [], [], [], dict(agent=[], wrist=[])
    for k in range(len(S)):
        row = dict(k=k)
        # ---- P: swap the two mugs' xy
        env.reset()
        obs = env.set_init_state(S[k])
        if k < 3:                                                           # statecheck input: start renders after the settle steps
            o = obs
            for _ in range(s2.WAIT):
                o, _, _, _ = env.step(T2.DUMMY)
            frames["agent"].append(np.ascontiguousarray(np.asarray(o["agentview_image"])[::-1, ::-1]))
            frames["wrist"].append(np.ascontiguousarray(np.asarray(o["robot0_eye_in_hand_image"])[::-1, ::-1]))
            env.reset()
            env.set_init_state(S[k])
        p0 = {n: pos(n) for n in s2.MOVABLE}
        qw, qy = np.array(sim.data.get_joint_qpos(jn[W])), np.array(sim.data.get_joint_qpos(jn[Y]))
        assert qw.shape == (7,) and qy.shape == (7,)
        nw, ny = qw.copy(), qy.copy()
        nw[:2], ny[:2] = qy[:2], qw[:2]
        sim.data.set_joint_qpos(jn[W], nw)
        sim.data.set_joint_qpos(jn[Y], ny)
        sim.forward()
        sp = np.array(sim.get_state().flatten())
        env.reset()
        env.set_init_state(sp)
        a0 = {n: pos(n) for n in s2.MOVABLE}
        settle(s2.WAIT)
        a1 = {n: pos(n) for n in s2.MOVABLE}
        row["P"] = dict(white_xy=a1[W][:2].tolist(), yw_xy=a1[Y][:2].tolist(),
                        swap_err=float(max(np.linalg.norm(a0[W][:2] - p0[Y][:2]), np.linalg.norm(a0[Y][:2] - p0[W][:2]))),
                        drift_xy=float(max(np.linalg.norm((a1[n] - a0[n])[:2]) for n in s2.MOVABLE)),
                        tilt=max(_tilt(env, n) for n in s2.MUGS),
                        on=[int(bool(E._eval_predicate(["on", m_, p_]))) for m_ in s2.MUGS for p_ in s2.PLATES])
        row["P"]["ok"] = bool(row["P"]["swap_err"] < 1e-6 and row["P"]["drift_xy"] < 0.01 and row["P"]["tilt"] < 10 and not any(row["P"]["on"]))
        PS.append(sp)
        # ---- Q: the white mug starts on the left plate (dropped from DZ above its own height at the plate's centre, settled)
        q_ok, tried = None, []
        for dz in (0.015, 0.025, 0.035, 0.05):
            env.reset()
            env.set_init_state(S[k])
            b0 = {n: pos(n) for n in s2.MOVABLE}
            q = np.array(sim.data.get_joint_qpos(jn[W]))
            q[:2] = b0[PL][:2]
            q[2] = q[2] + dz
            sim.data.set_joint_qpos(jn[W], q)
            sim.data.set_joint_qvel(jn[W], np.zeros(6))
            sim.forward()
            settle(60)
            sq = np.array(sim.get_state().flatten())
            env.reset()
            env.set_init_state(sq)
            c0 = {n: pos(n) for n in s2.MOVABLE}
            settle(s2.WAIT)
            c1 = {n: pos(n) for n in s2.MOVABLE}
            on = [int(bool(E._eval_predicate(["on", m_, p_]))) for m_ in s2.MUGS for p_ in s2.PLATES]
            chk = dict(dz=dz, on_white_left=s2.on_of(on, W, PL), on=on, tilt_white=_tilt(env, W),
                       white_to_plate_xy=float(np.linalg.norm((c1[W] - c1[PL])[:2])),
                       plate_moved_xy=float(np.linalg.norm((c1[PL] - b0[PL])[:2])),
                       others_moved_xy=float(max(np.linalg.norm((c1[n] - b0[n])[:2]) for n in (Y, R, PR))),
                       drift_after_xy=float(max(np.linalg.norm((c1[n] - c0[n])[:2]) for n in s2.MOVABLE)))
            chk["ok"] = bool(chk["on_white_left"] == 1 and sum(on) == 1 and chk["tilt_white"] < 10 and chk["plate_moved_xy"] < 0.01
                             and chk["others_moved_xy"] < 0.005 and chk["drift_after_xy"] < 0.005)
            tried.append(chk)
            if chk["ok"]:
                q_ok = sq
                break
        row["Q"] = dict(tried=tried, ok=q_ok is not None)
        QS.append(q_ok if q_ok is not None else sq)
        rows.append(row)
        print("[s2cue build] k %d P %s | Q %s" % (k, json.dumps({x: (np.round(v, 4).tolist() if isinstance(v, (list, float)) else v)
                                                                 for x, v in row["P"].items()}), json.dumps(tried[-1])), flush=True)
    env.close()
    PS, QS = np.stack(PS), np.stack(QS)
    for m, A in (("P", PS), ("Q", QS)):
        os.makedirs(os.path.join(CD, m))
        for t in M["tasks"][:4]:
            np.save(os.path.join(CD, m, "%s_%d.npy" % (s2.SUITE, t["tid"])), A)
    br = json.load(open(os.path.join(s2.SD, "branches_main.json"), encoding="utf-8"))
    br40 = [b for b in br if int(b["trial"]) in MAIN]
    assert [int(b["trial"]) for b in br40] == MAIN and [int(b["seed_base"]) for b in br40] == [23000000 + 1000 * k for k in MAIN]
    with open(os.path.join(CD, "branches_40.json"), "w", encoding="utf-8") as fh:
        json.dump(br40, fh, indent=1)
    allok = dict(P=all(r["P"]["ok"] for r in rows), Q=all(r["Q"]["ok"] for r in rows),
                 P_main=all(r["P"]["ok"] for r in rows if r["k"] in MAIN), Q_main=all(r["Q"]["ok"] for r in rows if r["k"] in MAIN))
    with open(os.path.join(CD, "s2cue_states.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(rule="P: white / yellow-and-white mug xy swapped (own z, quat kept); Q: white mug dropped at the left plate's "
                            "centre from the first dz in (.015, .025, .035, .05) m above its own height that passes, 60 settle steps",
                       source=os.path.join(s2.SD, "states"), ok=allok, rows=rows), fh, indent=1)
    os.makedirs(os.path.dirname(FRAMES), exist_ok=True)
    np.savez_compressed(FRAMES, agent=np.stack(frames["agent"]), wrist=np.stack(frames["wrist"]))
    print("[s2cue build] ok %s; wrote %s and %s" % (allok, CD, FRAMES), flush=True)


# ---------------------------------------------------------------- statecheck (openpi venv, GPU)
def statecheck(a):
    """f1m_cue.statecheck's bit-for-bit test with this scene's prompts and start renders (FRAMES) plus random images: does the
    served pi0.5 (pp_f1_mix.Pi05 = PolicyBackbone('pi05_libero'), the object pp_serve_purify --model pi05 serves) read
    observation/state?"""
    sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
    from lifelong_ood.policy.serve import GuidedPolicy
    from pp_f1_mix import Pi05
    Mo = Pi05()
    pol = GuidedPolicy(Mo.bk, hooks=None, num_steps=Mo.bk.num_steps, seed=13)
    rng = np.random.default_rng(0)
    idx = (np.arange(224) * 256 / 224).astype(int)
    imgs = []
    if os.path.exists(FRAMES):
        z = np.load(FRAMES)
        imgs += [("render%d" % i, z["agent"][i][idx][:, idx], z["wrist"][i][idx][:, idx]) for i in range(len(z["agent"]))]
    imgs += [("random%d" % i, rng.integers(0, 256, (224, 224, 3), dtype=np.uint8), rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
             for i in range(2)]
    base = np.array([-0.06, 0.02, 1.05, 3.10, 0.05, -0.08, 0.012, -0.012], np.float32)
    var = {"grip_open": np.r_[base[:6], [0.039, -0.039]].astype(np.float32),
           "grip_closed_on_mug": np.r_[base[:6], [0.025, -0.025]].astype(np.float32),
           "eef_y_mirror": base * np.array([1, -1, 1, 1, 1, 1, 1, 1], np.float32),
           "eef_y_minus_0.30": base + np.array([0, -0.3, 0, 0, 0, 0, 0, 0], np.float32),
           "eef_z_plus_0.20": base + np.array([0, 0, 0.2, 0, 0, 0, 0, 0], np.float32),
           "all_random": (base + rng.normal(0, 1.0, 8)).astype(np.float32), "zeros": np.zeros(8, np.float32)}
    prompts = [t["language"] for t in s2.meta()["tasks"][:4]] if os.path.exists(os.path.join(s2.SD, "s2_tasks.json")) else [
        "put the white mug on the left plate", "put the yellow and white mug on the right plate", "put the white mug on the right plate",
        "put the yellow and white mug on the left plate"]
    res = dict(config=Mo.bk.train_config.name, discrete_state_input=bool(getattr(Mo.bk.train_config.model, "discrete_state_input", None)),
               pi05=bool(getattr(Mo.bk.train_config.model, "pi05", None)), prompts=prompts, cases=[])

    def run(img, wimg, st, prompt, sb):
        o = {"observation/image": img, "observation/wrist_image": wimg, "observation/state": st, "prompt": prompt,
             "meta/episode_start": True, "meta/seed_base": sb, "meta/branch_at": 0, "meta/variant": "s2"}
        return np.asarray(pol.infer(o)["actions"], np.float64)
    for s, (nm, img, wimg) in enumerate(imgs):
        img2 = imgs[(s + 1) % len(imgs)][1]
        for pi_, pr in enumerate(prompts):
            sb = 23000000 + 1000 * s
            a0 = run(img, wimg, base, pr, sb)
            row = dict(image=nm, prompt=pr, repeat_maxabs=float(np.abs(a0 - run(img, wimg, base, pr, sb)).max()),
                       state_maxabs={k: float(np.abs(run(img, wimg, v, pr, sb) - a0).max()) for k, v in var.items()},
                       image_maxabs=float(np.abs(run(img2, wimg, base, pr, sb) - a0).max()),
                       prompt_maxabs=float(np.abs(run(img, wimg, base, prompts[(pi_ + 2) % len(prompts)], sb) - a0).max()),
                       action_absmean=float(np.abs(a0).mean()))
            res["cases"].append(row)
            print("[s2cue statecheck] %s" % json.dumps(row), flush=True)
    res["state_invariant"] = all(v == 0.0 for c in res["cases"] for v in c["state_maxabs"].values())
    res["repeat_exact"] = all(c["repeat_maxabs"] == 0.0 for c in res["cases"])
    res["image_sensitive"] = all(c["image_maxabs"] > 1e-3 for c in res["cases"])
    res["prompt_sensitive"] = all(c["prompt_maxabs"] > 1e-3 for c in res["cases"])
    print("[s2cue statecheck] config %s pi05 %s discrete_state_input %s | state invariant %s repeat exact %s image sensitive %s prompt "
          "sensitive %s" % (res["config"], res["pi05"], res["discrete_state_input"], res["state_invariant"], res["repeat_exact"],
                            res["image_sensitive"], res["prompt_sensitive"]), flush=True)
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), out
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out, flush=True)


# ---------------------------------------------------------------- reference planner (LIBERO venv, CPU)
def ref(a):
    import time
    M = s2.meta()
    t = M["tasks"][a.tid]
    assert t["tid"] == a.tid and a.tid in MANIP_TIDS[a.manip]
    os.makedirs(a.out_dir, exist_ok=True)
    fn = os.path.join(a.out_dir, "ref_%s_%d.json" % (a.manip, a.tid))
    assert not os.path.exists(fn), "exists: " + fn
    kp = s2.kp_for(t["key"])
    S = np.load(os.path.join(CD, a.manip, "%s_%d.npy" % (s2.SUITE, a.tid)))
    eps = []
    for b in json.load(open(a.branches, encoding="utf-8")):
        rng = np.random.default_rng(int(b["seed_base"]) + 10)                # s2.ref's rule
        t0 = time.time()
        try:
            e = s2.plan_episode(os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], kp, t["object"], t["target"], rng, int(t["max_steps"]))
        except Exception as ex:                                             # noqa: BLE001
            e = dict(success=False, error=repr(ex))
        e.update(trial=int(b["trial"]), secs=time.time() - t0)
        eps.append(e)
        print("[s2cue ref] %s %s trial %d success %s held %s phase %s on %s final dxy %s (%.0fs)" % (
            a.manip, t["key"], b["trial"], e["success"], e.get("held_step"), e.get("phase"), e.get("on"), e.get("final_dxy"), e["secs"]),
            flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(manip=a.manip, tid=a.tid, key=t["key"], kp=kp, eps=eps), fh, indent=1)


# ---------------------------------------------------------------- per-episode scoring (CPU)
def plate_of(on, mug):
    ps = [p for p in s2.PLATES if s2.on_of(on, mug, p)]
    return ps[0] if len(ps) == 1 else (None if not ps else "both")


def score(stem, t, ks, manip):
    """-> per episode: success, held mug, the held mug's plate, both mugs' plates, and what each cue predicts."""
    out = []
    for k, r in zip(ks, s2.stage_rows(stem, t, ks)):
        h = r["bound"]
        d = dict(k=k, success=r["success"], held=h, dest=plate_of(r["on"], h) if h in (W, Y) else None,
                 white=plate_of(r["on"], W), yw=plate_of(r["on"], Y))
        if h in (W, Y):
            d["pred"] = dict(instruction=(t["target"] if h == t["object"] else "NOT_NAMED"), identity=s2.HABIT[h],
                             position=POS[manip][h] if manip else s2.HABIT[h], step=STEP[manip][h] if manip else PL)
        out.append(d)
    return out


def boot_share(F, D, B=2000):
    """F, D: [n_states] counts (followed, defined) pooled over tasks -> [share, lo, hi, n_defined]."""
    F, D = np.asarray(F, float), np.asarray(D, float)
    n = int(D.sum())
    if n == 0:
        return [None, None, None, 0]
    rng = np.random.default_rng(0)
    v = []
    for _ in range(B):
        i = rng.choice(len(F), len(F))
        dd = D[i].sum()
        v.append(F[i].sum() / dd if dd else np.nan)
    return [float(F.sum() / D.sum()), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)), n]


def shares(E, ks, cue, held=None):
    """E: {task: [episode]} -> boot_share over placements (held mug On exactly one plate) where cue c is defined."""
    F, D = np.zeros(len(ks)), np.zeros(len(ks))
    for eps in E.values():
        for i, e in enumerate(eps):
            if e["dest"] not in (PL, PR) or (held and e["held"] != held):
                continue
            p = e["pred"][cue]
            if p is None:
                continue
            D[i] += 1
            F[i] += float(p == e["dest"])
    return boot_share(F, D)


def cue_verdict(sh, strata):
    if sh[3] < N_MIN_ALL:
        return "UNTESTABLE"
    if sh[0] >= 0.70 and sh[1] > 0.50 and all(s[0] >= 0.50 for s in strata if s[3] >= N_MIN_STRATUM):
        return "ROUTES"
    if sh[2] < 0.50:
        return "RULED OUT"
    return "UNRESOLVED"


def p_labels(E, ks):
    res = {}
    for c in ("position", "identity", "step", "instruction"):
        sh = shares(E, ks, c)
        st = {m: shares(E, ks, c, held=m) for m in (W, Y)}
        res[c] = dict(share=sh, strata=st, verdict=cue_verdict(sh, list(st.values())))
    nW = shares(E, ks, "identity", held=W)[3]
    nY = shares(E, ks, "identity", held=Y)[3]
    routes = [c for c in res if res[c]["verdict"] == "ROUTES"]
    lab = list(routes)
    if "position" in routes and "step" in routes:
        lab = [x for x in lab if x not in ("position", "step")] + ["POSITION OR STEP (inseparable: white-held placements %d < 10)" % nW]
    if "identity" in routes and "step" in routes:
        lab = [x for x in lab if x not in ("identity", "step")] + ["IDENTITY OR STEP (inseparable: yellow-and-white-held placements %d < 10)" % nY]
    return dict(cues=res, routing=lab, n_white_held=nW, n_yw_held=nY)


def q_label(eps, ks):
    F, D, L = np.zeros(len(ks)), np.zeros(len(ks)), np.zeros(len(ks))
    for i, e in enumerate(eps):
        if e["yw"] in (PL, PR):
            D[i] += 1
            F[i] += float(e["yw"] == PR)
            L[i] += float(e["yw"] == PL)
    r, l_ = boot_share(F, D), boot_share(L, D)
    if r[3] < 10:
        v = "UNTESTABLE"
    elif r[0] >= 0.70 and r[1] > 0.50:
        v = "PROGRAM CONTINUES"
    elif l_[0] >= 0.70 and l_[1] > 0.50:
        v = "INSTRUCTION ROUTES"
    else:
        v = "UNRESOLVED"
    return dict(yw_on_right=r, yw_on_left=l_, verdict=v)


def describe(E):
    d = {}
    for c, eps in E.items():
        n = len(eps)
        d[c] = dict(success=float(np.mean([e["success"] for e in eps])),
                    held={m: float(np.mean([e["held"] == m for e in eps])) for m in (W, Y, R, PL, PR, None)},
                    held_dest={m: {p: int(sum(e["held"] == m and e["dest"] == p for e in eps)) for p in (PL, PR, None, "both")} for m in (W, Y)},
                    white_final={p: float(np.mean([e["white"] == p for e in eps])) for p in (PL, PR, None, "both")},
                    yw_final={p: float(np.mean([e["yw"] == p for e in eps])) for p in (PL, PR, None, "both")}, n=n)
    return d


# ---------------------------------------------------------------- smoke summary / read (CPU, LIBERO venv)
def summary(d):
    M = s2.meta()
    for manip in ("P", "Q"):
        for fe in FE:
            for tid in MANIP_TIDS[manip]:
                stem = os.path.join(d, manip, "%s_%s_%d" % (MODEL, fe, tid))
                if not (os.path.exists(stem + ".json") and os.path.exists(stem + "_track.npz")):
                    continue
                t = M["tasks"][tid]
                R_ = json.load(open(stem + ".json", encoding="utf-8"))
                ks = [int(e["trial"]) for v in R_["tasks"].values() for e in v["episodes"]]
                for e in score(stem, t, ks, manip):
                    print("[s2cue smoke] %s %s %s %s" % (manip, fe, t["key"], json.dumps(e)), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_")) if os.path.isdir(d) else []:
        R_ = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2cue smoke] %s: %s" % (f, [(e["trial"], e["success"], e.get("on")) for e in R_["eps"]]), flush=True)


def read(a):
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    M = s2.meta()
    T = {t["tid"]: t for t in M["tasks"]}
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD = SMOKE, a.dir
        mism, base_ks = "selftest", None
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_cue"]
        assert len(rows) == 1, "expected exactly one s2_cue prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD, base_ks = MAIN, ROLL, M["states"]["main"]
    res = dict(states=ks, mapping=MAPPING, code_mismatch=mism, manip={}, reference={})
    for manip in ("P", "Q"):
        res["manip"][manip] = {}
        for fe in FE:
            E, miss = {}, []
            for tid in MANIP_TIDS[manip]:
                stem = os.path.join(RD, manip, "%s_%s_%d" % (MODEL, fe, tid))
                if not (os.path.exists(stem + ".json") and os.path.exists(stem + "_track.npz")):
                    miss.append(stem)
                    continue
                E[T[tid]["key"]] = score(stem, T[tid], ks, manip)
            if miss:
                res["manip"][manip][fe] = "MISSING %s" % miss
                continue
            row = dict(describe=describe(E))
            if manip == "P":
                row["labels"] = p_labels(E, ks)
            else:
                row["labels"] = dict(Q_N2=q_label(E["N2"], ks), Q_N1=q_label(E["N1"], ks), Q_K2=q_label(E["K2"], ks),
                                     held_cues=p_labels(E, ks))
            # paired success difference to SR 50 on the same states and branches (descriptive)
            if base_ks is not None:
                row["vs_sr50"] = {}
                for tid in MANIP_TIDS[manip]:
                    c = T[tid]["key"]
                    b = s2.stage_rows(os.path.join(s2.ROLL, "%s_%s_%d" % (MODEL, fe, tid)), T[tid], base_ks)[:len(ks)]
                    x = np.array([[float(e["success"]) for e in E[c]]])
                    y = np.array([[float(r_["success"]) for r_ in b]])
                    row["vs_sr50"][c] = dict(manip=float(x.mean()), sr50=float(y.mean()), d=s2.boot2_states(x, y))
            res["manip"][manip][fe] = row
            pl = row["labels"] if manip == "P" else row["labels"]["held_cues"]
            brief = dict(routing=pl["routing"], n_white_held=pl["n_white_held"], n_yw_held=pl["n_yw_held"], cues={})
            for c, v in pl["cues"].items():
                brief["cues"][c] = dict(share=v["share"], verdict=v["verdict"],
                                        strata={m: [v["strata"][m][0], v["strata"][m][3]] for m in v["strata"]})
            if manip == "Q":
                brief["Q"] = {x: row["labels"][x] for x in ("Q_N2", "Q_N1", "Q_K2")}
            print("== %s %s: %s" % (manip, fe, json.dumps(brief, default=str)), flush=True)
            for c, v in row["describe"].items():
                print("   %s %s" % (c, json.dumps(v, default=str)), flush=True)
            if "vs_sr50" in row:
                print("   vs SR 50 %s" % json.dumps(row["vs_sr50"]), flush=True)
    # SR 50's unmanipulated rollouts on the same states (position = identity there; descriptive)
    if base_ks is not None:
        res["sr50_same_states"] = {}
        for fe in FE:
            E = {T[tid]["key"]: score(os.path.join(s2.ROLL, "%s_%s_%d" % (MODEL, fe, tid)), T[tid], base_ks, None)[:len(ks)] for tid in (0, 1, 2, 3)}
            res["sr50_same_states"][fe] = dict(describe=describe(E), labels=p_labels(E, ks))
            print("== SR 50 %s (unmanipulated, k 4-43): routing %s" % (fe, res["sr50_same_states"][fe]["labels"]["routing"]), flush=True)
    # the reference (C1)
    RF = a.ref_dir or RD
    for manip in ("P", "Q"):
        per = {}
        for tid in MANIP_TIDS[manip]:
            f = os.path.join(RF, "ref_%s_%d.json" % (manip, tid))
            if os.path.exists(f):
                ee = json.load(open(f, encoding="utf-8"))["eps"]
                assert [e["trial"] for e in ee] == ks, (f, [e["trial"] for e in ee])
                per[T[tid]["key"]] = dict(success=float(np.mean([e["success"] for e in ee])),
                                          errors=int(sum(e.get("error") is not None for e in ee)))
        new = [per[c]["success"] for c in ("N1", "N2") if c in per]
        res["reference"][manip] = dict(per_task=per, success_new=float(np.mean(new)) if len(new) == 2 else None,
                                       C1=(None if len(new) < 2 else ("C1 MET" if np.mean(new) >= 0.80 else "C1 UNMET")))
        print("== reference %s: %s" % (manip, json.dumps(res["reference"][manip])), flush=True)
    print("== code hash mismatches: %s" % mism, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, out)
    print("[s2cue read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "statecheck", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--manip", choices=("P", "Q"))
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--ref-dir")
    ap.add_argument("--out", default=READ)
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    dict(build=build, statecheck=statecheck, ref=ref, summary=lambda a: summary(a.dir), read=read)[A.mode](A)

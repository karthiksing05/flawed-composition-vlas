"""SCENE 2 (docs/SCAFFOLD_RESULTS.md 50; m3_prereg tag scene2): do the break atlas's F1 (the held object picks the destination; the
cross-chunk break) and the test-time role-view front end replicate in a SECOND trained scene, LIBERO-10's LIVING_ROOM_SCENE5?
Tasks, states and branches: experiments/s2_tasks.py (results/flawed/s2/).  Rollouts: experiments/s2_roll.sh -> results/flawed/s2_roll/.

Harness.  experiments/libero_client.py, unchanged, with a suite s2_living registered in LIBERO's benchmark dictionary (as
fc_n2_client.py registers n2_goal); start states only from --fresh, every episode through the branch path (fresh env, re-seeded);
client seed 7, --bind track, --replan 5, --wait 10.  The env is wrapped by Rec2: pp_atlas.Rec's oracle track (Held from finger contacts
and lift, object and target positions from step WAIT on) with this scene's bodies, plus the final LIBERO 'On' predicate of every
mug on every plate.  Nothing the policy sees changes in arm I.
Arms (client --arm):
    I    intact: the task instruction, raw images.
    tv   role views: tv_frozen.Front, unchanged (ProprioHeld role sequencer with sc_rv_lex.json's threshold, stepped on every env step;
         R2's role_paint rule with this scene's movables = the three mugs and two plates), but the masks are the SIMULATOR's entity
         maps (sc_rv.seg_px of the current state), because the learned segmenter has no mug class and one plate class (s2_tasks.py
         check).  Object phase: every movable except the task's mug is painted magenta (both plates and the other two mugs);
         target phase: every movable except the target plate (the held mug included).  Prompt unchanged.
    tvp  tv plus role phrases switched by the same detector: "object: the white mug" / "target: the right plate" (R1's format).
The policy is served frozen: pi0.5 and GR00T by pp_serve_purify.py --mode intact --comp= --seed 13; pi0 by serve_guided.py --mode
ours --seed 13 (pp_atlas_more_roll.sh's server).
Reference (C1): a scripted planner on simulator state (pp_n4d.plan_episode's phases and controller, servo on), keypoints fitted from
the LIBERO-10 compound's raw demos, one cycle per mug (s2_tasks.py keypoints), new pairings composed by kp_for's relocation rule.

Modes:
    python experiments/s2.py client --arm I|tv|tvp <libero_client args: --port --suite s2_living --tasks T --trials 1 --fresh D
                                   --branches B --videos 0 --bind track --out X.json>   (LIBERO venv; one task per process)
    python experiments/s2.py ref --tid T --branches B --out-dir D                        (LIBERO venv, CPU)
    python experiments/s2.py summary DIR                                                 (smoke)
    python experiments/s2.py read [--out results/flawed/s2_read.json]                    (LIBERO venv, CPU)

Registered mapping (read): MAPPING below.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
SD = os.path.join(OUT, "s2")
ROLL = os.path.join(OUT, "s2_roll")
READ = os.path.join(OUT, "s2_read.json")
SUITE = "s2_living"
WAIT = 10
MUGS = ("porcelain_mug_1", "white_yellow_mug_1", "red_coffee_mug_1")
PLATES = ("plate_1", "plate_2")
MOVABLE = MUGS + PLATES
NEW, KNOWN = ("N1", "N2"), ("K1", "K2")
PAIR = dict(N1=("K1", "K2"), N2=("K2", "K1"))
HABIT = {"porcelain_mug_1": "plate_1", "white_yellow_mug_1": "plate_2"}      # each mug's trained plate
MODELS = dict(pi05=("I", "tv", "tvp"), groot=("I", "tv"), pi0=("I",))
EPS = 0.10
CARRY_STEPS = int(os.environ.get("S2_CARRY_STEPS", "120"))
MAPPING = (
    "Main states k = 4-63 (60), familiar layout of LIVING_ROOM_SCENE5.  Success matrices [task, state]; paired bootstrap B = 2000, "
    "numpy default_rng(0), tasks and states drawn jointly (tv_frozen.boot2 / boot1 / atlas_est, unchanged); CI = 2.5 / 97.5.  "
    "(a) F1 atlas per model, arm I: tv_frozen.atlas_est (pp_atlas's estimator, stage rule and bar) with NEW = N1, N2 and partners "
    "N1 = (K1 object, K2 target), N2 = (K2 object, K1 target); a group BREAKS iff E > .10 and CI lo > 0.  A model is TESTABLE iff its "
    "intact success on K1-K2 is >= .70.  SCENE-2 F1 verdict over pi0.5 and GR00T: REPLICATES iff the cross-chunk group breaks on both "
    "(both testable); PARTIAL iff it breaks on exactly one; NOT REPLICATED iff on neither testable model; NOT TESTABLE iff neither is "
    "testable.  pi0 gets the per-model verdict only.  C1: the reference planner's success on N1-N2 >= .80 -> C1 MET, else C1 UNMET.  "
    "(b) Front end: pi0.5 PRIMARY d = tvp - I on N1-N2: CONFIRMED iff d > .10 and lo > 0; REFUTED iff hi < .10; else UNRESOLVED "
    "(tv_fresh's rule).  GR00T PRIMARY d = tv - I on N1-N2: TV WINS iff lo > 0; else NO CLEAR DIFFERENCE (tv_g's rule).  pi0.5 "
    "SECONDARY tv - I: the pi0.5 primary's rule.  GUARD per arm on K1-K2: NO COST iff (arm - I) lo > -.10, else COST.  SCENE-2 FIX "
    "verdict: REPLICATES iff both primaries win (CONFIRMED, TV WINS); PARTIAL iff one; NOT REPLICATED iff neither.  "
    "Descriptive: per-task success; N - K (N1 vs K1, N2 vs K2, same states); grasp correctness on N (first moved object = the named "
    "mug); habit placements (the named mug ends On its trained plate instead of the named one); atlas excess per arm; stage shares; "
    "KC / NC success and per-atom completion under I; the reference per task; state-only bootstrap (tasks fixed) for every delta; "
    "detector diagnostics (tv_frozen.side_summary).")


def meta():
    return json.load(open(os.path.join(SD, "s2_tasks.json"), encoding="utf-8"))


# ---------------------------------------------------------------- the oracle track (LIBERO venv)
class Rec2:
    """pp_atlas.Rec for this scene: from the WAIT-th step on, Held (fa1b_common, finger contacts + lift) of the task's object and the
    object / target / end-effector positions; at close, every mug-on-plate predicate.  The policy's inputs are untouched."""

    def __init__(self, env, oe, te):
        self._e, self._oe, self._te, self.n = env, oe, te, 0
        self._con, self._hm, self._last = None, None, None
        self.rows = dict(held=[], obj=[], tgt=[], ee=[], quat=[], a6=[])
        self.on = None

    def __getattr__(self, k):
        return getattr(self._e, k)

    def _pos(self, n):
        E = self._e.env
        return np.array(E.sim.data.body_xpos[E.obj_body_id[n]], np.float64)

    def step(self, a):
        import fa1b_common as FB
        r = self._e.step(a)
        self.n += 1
        if self.n < WAIT:
            return r
        if self._con is None:
            self._con, self._hm = FB.Contact(self._e, [self._oe]), FB.Held()
        h, _ = self._hm.step(*self._con.touching(self._oe), self._con.pos(self._oe), r[0]["robot0_eef_pos"])
        E = self._e.env
        R = self.rows
        R["held"].append(bool(h))
        R["obj"].append(self._pos(self._oe))
        R["tgt"].append(self._pos(self._te))
        R["ee"].append(np.array(r[0]["robot0_eef_pos"], np.float64))
        R["quat"].append(np.array(E.sim.data.body_xquat[E.obj_body_id[self._oe]], np.float64))
        R["a6"].append(float(a[6]))
        return r

    def finish(self):
        """Before the env closes: final positions of every movable and On(mug, plate) for every pair."""
        E = self._e.env
        self._last = np.stack([self._pos(n) for n in MOVABLE])
        self.on = np.array([int(bool(E._eval_predicate(["on", m, p]))) for m in MUGS for p in PLATES], np.int8)

    def data(self):
        R = self.rows
        d = dict(held=np.array(R["held"], bool), a6=np.array(R["a6"], np.float32))
        for k in ("obj", "tgt", "ee", "quat"):
            d[k] = np.array(R[k], np.float32).reshape(-1, 4 if k == "quat" else 3)
        d["final"] = np.array(self._last if self._last is not None else np.zeros((len(MOVABLE), 3)), np.float32)
        d["on"] = self.on if self.on is not None else np.full(len(MUGS) * len(PLATES), -1, np.int8)
        return d


def on_of(on, mug, plate):
    return int(on[MUGS.index(mug) * len(PLATES) + PLATES.index(plate)])


def load_track(path):
    z = np.load(path)
    return [{k: z["e%d_%s" % (i, k)] for k in ("held", "obj", "tgt", "ee", "quat", "a6", "final", "on")} for i in range(int(z["n"]))]


# ---------------------------------------------------------------- client (LIBERO venv)
class SimSeg:
    """sc_seg.Segmenter's ent() interface over the simulator's entity map of the current state (privileged masks)."""

    def __init__(self, env):
        import sc_rv as RV
        self.RV, self.env, self.E, self.path = RV, env, None, "SIMULATOR (sc_rv.seg_px)"

    def ent(self, img, cam, names):
        if self.E is None:
            self.E = self.RV.entity_tables(self.env)
            assert list(self.E["names"]) == list(names), (self.E["names"], names)
        return self.RV.seg_px(self.env, self.E, cam)


def register_suite(LC):
    import collections
    from libero.libero import benchmark
    M = meta()
    Task = collections.namedtuple("Task", "name language problem_folder bddl_file")

    class S2Suite:
        def __init__(self):
            self.tasks = [Task(t["key"], t["language"], os.path.dirname(os.path.join(ROOT, t["bddl"])), os.path.basename(t["bddl"]))
                          for t in M["tasks"]]
            self.n_tasks = len(self.tasks)

        def get_task(self, i):
            return self.tasks[i]

        def get_task_init_states(self, i):
            raise RuntimeError("scene 2 runs only from --fresh start states")

    orig = benchmark.get_benchmark_dict

    def bdict(*a, **k):
        d = dict(orig(*a, **k))
        d[SUITE] = S2Suite
        return d
    benchmark.get_benchmark_dict = bdict
    LC.MAX_STEPS[SUITE] = int(M["max_steps"])
    return M


def client_main():
    def take(name, default=None):
        if name in sys.argv:
            i = sys.argv.index(name)
            v = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return v
        return default
    sys.argv.pop(1)                                                         # "client"
    arm = take("--arm")
    assert arm in ("I", "tv", "tvp"), arm
    import pp_atlas as PA
    import libero_client as LC
    M = register_suite(LC)
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = M["tasks"][TID]
    assert TASK["tid"] == TID
    ms = int(TASK["max_steps"])
    if "--max-steps" in sys.argv:
        assert int(sys.argv[sys.argv.index("--max-steps") + 1]) == ms
    else:
        sys.argv += ["--max-steps", str(ms)]
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "scene-2 episodes need --fresh and --branches"
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    X, P, Front = None, None, None
    if arm != "I":
        assert len(TASK["atoms"]) == 1, "the role sequencer handles one pick-and-place: no front end on compounds"
        import sc_rv as RV
        import tv_frozen as TF
        RV.MOVABLE_OBJ = MOVABLE                                            # this scene's movables (role_paint and Front read it)
        X = json.load(open(RV.lex_path(""), encoding="utf-8"))
        Front = TF.Front
        if arm == "tvp":
            P = list(TASK["phrases"])
    ORIG, RECS, SIDE, CUR = LC.OffScreenRenderEnv, [], [], {}
    SIDEF = OUTF.replace(".json", "_tv.json")

    class S2Env(Rec2):
        def __init__(self, env):
            super().__init__(env, TASK["object"], TASK["target"])
            self.front = None
            CUR["env"] = self

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            if Front is not None:
                self.front = Front(self._e, "tv", TASK["object"], TASK["target"], X, SimSeg(self._e))
                self.front.prompts = dict(object=0, target=0)
            return obs

        def step(self, a):
            r = super().step(a)
            if self.front is None:
                return r
            obs = self.front.step(r[0], self.n, float(a[6]))
            return (obs,) + tuple(r[1:])

        def close(self):
            if self.n > 0:
                self.finish()
                RECS.append(self.data())
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
                if self.front is not None:
                    i = len(SIDE)
                    rec = self.front.record(self.rows["held"])
                    rec.update(arm=arm, trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, tid=TID, key=TASK["key"],
                               prompts=self.front.prompts)
                    SIDE.append(rec)
                    tmp = SIDEF + ".tmp%d" % os.getpid()
                    with open(tmp, "w", encoding="utf-8") as fh:
                        json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], phrases=P, lex=X,
                                       masks="simulator", movable=list(MOVABLE), eps=SIDE), fh, indent=1)
                    os.replace(tmp, SIDEF)
                    print("[s2tv] %s %s trial %s: onsets %s (oracle first held %s) held queries %d/%d painted %s prompts %s" % (
                        arm, TASK["key"], rec["trial"], rec["onsets"][:4], rec["oracle_first_held"], rec["held_queries"], rec["queries"],
                        rec["painted_queries"], rec["prompts"]), flush=True)
                self.front = None
            self._e.close()

    if arm == "tvp":
        Base = LC._wcp.WebsocketClientPolicy

        class PClient(Base):
            def infer(self, el):
                env = CUR.get("env")
                f = None if env is None else env.front
                assert str(el["prompt"]) == TASK["language"], el["prompt"]
                el = dict(el)
                on = bool(f is not None and f.on)
                el["prompt"] = P[int(on)]
                if f is not None:
                    f.prompts["target" if on else "object"] += 1
                return super().infer(el)
        LC._wcp.WebsocketClientPolicy = PClient

    LC.OffScreenRenderEnv = lambda *a, **k: S2Env(ORIG(*a, **k))
    print("[s2] arm %s task %s %r object %s target %s max_steps %d phrases %s lex %s" % (
        arm, TASK["key"], TASK["language"], TASK["object"], TASK["target"], ms, P, None if X is None else (X["tw"], X["lift"])), flush=True)
    LC.main()


# ---------------------------------------------------------------- reference planner (LIBERO venv, CPU)
def kp_for(key):
    kp = json.load(open(os.path.join(SD, "keypoints.json"), encoding="utf-8"))["tasks"]
    if key in kp:
        return kp[key]
    ko, kt = kp[PAIR[key][0]], kp[PAIR[key][1]]
    out = dict(ko)
    out["release_off"] = (np.array(ko["grasp_off"]) + np.array(kt["release_off"]) - np.array(kt["grasp_off"])).tolist()
    return out


def plan_episode(bddl, s0, kp, oe, te, rng, max_steps):
    """pp_n4d.plan_episode (servo=True) with this scene's bodies: approach above the grasp point, descend, close, lift to the carry
    height (above the target's start height), carry above the release point, descend, open, retreat.  Controller fa1g_plan.ctrl."""
    import fa1_common as FC
    import fa1b_common as FB
    import s2_tasks as T2
    from fa1g_plan import ctrl
    env = FC.fresh_env(bddl)
    env.reset()
    obs = env.set_init_state(s0)
    con, hm = FB.Contact(env, [oe]), FB.Held()
    for _ in range(WAIT):
        obs, _, done, _ = env.step(FC.DUMMY)
    E = env.env

    def pos(n):
        return np.array(E.sim.data.body_xpos[E.obj_body_id[n]], np.float64)

    def sc():
        return dict(ee_pos=np.asarray(obs["robot0_eef_pos"], np.float64), ee_rot=T2.quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64)))
    jit = rng.normal(0, 0.01, (2, 3))
    cz = kp["carry_z_above_tgt"] + rng.normal(0, 0.02)
    goff, roff = np.array(kp["grasp_off"]) + jit[0], np.array(kp["release_off"]) + jit[1]
    RG, RR = np.array(kp["grasp_rot"]), np.array(kp["release_rot"])
    zref = float(pos(te)[2])
    phase, hold, done, step, in_phase, held_step, rel_up, at_open, hxy_last = 0, 0, False, 0, 0, None, None, None, None
    while step < max_steps and not done:
        s = sc()
        po, pt = pos(oe), pos(te)
        if phase == 0:
            wp, Rt, g = po + goff + [0, 0, 0.10], RG, -1.0
        elif phase == 1:
            wp, Rt, g = po + goff, RG, -1.0
        elif phase == 2:
            wp, Rt, g = po + goff, RG, 1.0
        elif phase == 3:
            wp, Rt, g = np.array([s["ee_pos"][0], s["ee_pos"][1], zref + cz]), RG, 1.0
        elif phase in (4, 5, 6):
            # object-centred release (smoke-tuned, before the prereg): the hand goes where the HELD MUG's centre lies over the
            # plate's centre plus the release jitter, using the live hand-minus-mug offset; height from the fitted release offset
            hxy = (s["ee_pos"] - po)[:2] if phase in (4, 5) else hxy_last
            hxy_last = hxy
            xy = pt[:2] + hxy + jit[1][:2]
            z = zref + cz if phase == 4 else pt[2] + roff[2]
            wp, Rt, g = np.array([xy[0], xy[1], z]), RR, (1.0 if phase < 6 else -1.0)
        else:
            wp, Rt, g = rel_up, RR, -1.0
        a, err = ctrl(s, wp, Rt, g, rng)
        obs, _, done, _ = env.step(a.tolist())
        h, _ = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
        if h and held_step is None:
            held_step = step
        step += 1
        in_phase += 1
        # phase time limits: 40 steps, except the carry above the release point (phase 4), which gets CARRY_STEPS because this
        # scene's new pairings carry ~0.45 m (the trained halves ~0.15-0.2 m); a limit that fires early releases short of the plate
        lim = CARRY_STEPS if phase == 4 else 40
        if phase in (0, 3, 4) and err < 0.015 or phase in (1, 5) and err < 0.008 or phase in (0, 1, 3, 4, 5) and in_phase >= lim:
            if phase == 5:
                at_open = dict(dxy=float(np.linalg.norm((pos(oe) - pos(te))[:2])), held=bool(h), step=step)
            phase, in_phase = phase + 1, 0
        elif phase in (2, 6):
            hold += 1
            if hold >= 12:
                phase, hold, in_phase = phase + 1, 0, 0
                if phase == 7:
                    rel_up = s["ee_pos"] + [0, 0, 0.10]
    on = [int(bool(E._eval_predicate(["on", m, p]))) for m in MUGS for p in PLATES]
    fin = float(np.linalg.norm((pos(oe) - pos(te))[:2]))
    env.close()
    return dict(success=bool(done), steps=step, phase=phase, held_step=held_step, on=on, at_open=at_open, final_dxy=fin, error=None)


def ref(a):
    M = meta()
    t = M["tasks"][a.tid]
    assert t["tid"] == a.tid and len(t["atoms"]) == 1
    os.makedirs(a.out_dir, exist_ok=True)
    fn = os.path.join(a.out_dir, "ref_%d.json" % a.tid)
    assert not os.path.exists(fn), "exists: " + fn
    kp = kp_for(t["key"])
    S = np.load(os.path.join(SD, "states", "%s_%d.npy" % (SUITE, a.tid)))
    eps = []
    for b in json.load(open(a.branches, encoding="utf-8")):
        rng = np.random.default_rng(int(b["seed_base"]) + 10)
        t0 = time.time()
        try:
            e = plan_episode(os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], kp, t["object"], t["target"], rng, int(t["max_steps"]))
        except Exception as ex:                                             # noqa: BLE001
            e = dict(success=False, error=repr(ex))
        e.update(trial=int(b["trial"]), secs=time.time() - t0)
        eps.append(e)
        print("[s2ref] %s trial %d success %s held %s phase %s on %s at_open %s final dxy %s (%.0fs)" % (
            t["key"], b["trial"], e["success"], e.get("held_step"), e.get("phase"), e.get("on"), e.get("at_open"), e.get("final_dxy"),
            e["secs"]), flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(tid=a.tid, key=t["key"], kp=kp, eps=eps), fh, indent=1)


# ---------------------------------------------------------------- read helpers (CPU)
def stage_rows(stem, t, ks):
    """-> per state: stage, subtype, success, bound, on (final predicates)."""
    import pp_atlas as PA
    R = json.load(open(stem + ".json", encoding="utf-8"))
    E = [e for v in R["tasks"].values() for e in v["episodes"]]
    Tr = load_track(stem + "_track.npz")
    assert len(E) == len(ks) and len(Tr) == len(ks), (stem, len(E), len(Tr), len(ks))
    out = []
    for k, e, r in zip(ks, E, Tr):
        assert int(e["trial"]) == k and e.get("error") is None, (stem, k, e.get("error"))
        d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
        s, x, _ = PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])
        out.append(dict(stage=s, sub=x, success=bool(e["success"]), bound=e.get("bound"), on=[int(v) for v in r["on"]]))
    return out


def summary(d):
    M = meta()
    for f in sorted(x for x in os.listdir(d) if x.endswith(".json") and not x.endswith("_tv.json") and not x.startswith("ref_")):
        stem = os.path.join(d, f[:-5])
        if not os.path.exists(stem + "_track.npz"):
            continue
        R = json.load(open(stem + ".json", encoding="utf-8"))
        tid = int(list(R["tasks"].keys())[0])
        t = M["tasks"][tid]
        ks = [int(e["trial"]) for v in R["tasks"].values() for e in v["episodes"]]
        side = json.load(open(stem + "_tv.json", encoding="utf-8"))["eps"] if os.path.exists(stem + "_tv.json") else [None] * len(ks)
        print("== %s %s %r" % (f, t["key"], t["language"]), flush=True)
        for k, row, s in zip(ks, stage_rows(stem, t, ks), side):
            on = {"%s>%s" % (m.split("_")[0][:5], p): on_of(row["on"], m, p) for m in MUGS for p in PLATES if on_of(row["on"], m, p)}
            print("  " + json.dumps(dict(trial=k, success=row["success"], stage=row["stage"], sub=row["sub"], bound=row["bound"], on=on,
                                         onsets=None if s is None else s["onsets"][:4],
                                         oracle_first_held=None if s is None else s["oracle_first_held"],
                                         prompts=None if s is None else s.get("prompts"))), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_")):
        R = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("== %s %s: %s" % (f, R["key"], [(e["trial"], e["success"], e.get("phase")) for e in R["eps"]]), flush=True)


def boot2_states(X, Y, B=2000):
    """Paired difference with the tasks fixed, states resampled (descriptive)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    rng = np.random.default_rng(0)
    ns = X.shape[1]
    v = [float((X[:, si] - Y[:, si]).mean()) for si in (rng.choice(ns, ns) for _ in range(B))]
    return [float((X - Y).mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def read(a):
    import pp_atlas as PA
    import tv_frozen as TF
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "scene2"]
    assert len(rows) == 1, "expected exactly one scene2 prereg row, found %d" % len(rows)
    M = meta()
    ks = M["states"]["main"]
    T = {t["key"]: t for t in M["tasks"]}
    br = json.load(open(os.path.join(SD, "branches_main.json"), encoding="utf-8"))
    br_ok = [int(b["trial"]) for b in br] == ks and [int(b["seed_base"]) for b in br] == [23000000 + 1000 * k for k in ks]
    comps = NEW + KNOWN
    res = dict(states=ks, B=PA.B, mapping=MAPPING, branches_ok=br_ok, models={})
    # the reference (C1)
    RF = {}
    for c in comps:
        f = os.path.join(ROLL, "ref_%d.json" % T[c]["tid"])
        if os.path.exists(f):
            E = json.load(open(f, encoding="utf-8"))["eps"]
            assert [e["trial"] for e in E] == ks
            RF[c] = np.array([float(e["success"]) for e in E])
    ref_row = None
    if all(c in RF for c in comps):
        rn = np.stack([RF[c] for c in NEW])
        ref_row = dict(per_task={c: float(RF[c].mean()) for c in comps}, success_new=TF.boot1(rn),
                       success_known=TF.boot1(np.stack([RF[c] for c in KNOWN])))
        ref_row["C1"] = "C1 MET" if ref_row["success_new"][0] >= 0.80 else "C1 UNMET"
    res["reference"] = ref_row
    for m, arms in MODELS.items():
        A, H = {}, {}
        for arm in arms:
            files = {c: os.path.join(ROLL, "%s_%s_%d" % (m, arm, T[c]["tid"])) for c in comps}
            miss = [f for f in files.values() if not (os.path.exists(f + ".json") and os.path.exists(f + "_track.npz"))]
            if miss:
                A[arm] = "MISSING %d of %d tasks" % (len(miss), len(files))
                continue
            RATE, succ, per, grasp_ok, wrong_obj, habit, other = {}, {}, {}, {}, {}, {}, {}
            for c in comps:
                rr = stage_rows(files[c], T[c], ks)
                RATE[c] = {s: np.array([float(r["stage"] == s) for r in rr]) for s in PA.STAGES}
                succ[c] = np.array([float(r["success"]) for r in rr])
                per[c] = {s: float(RATE[c][s].mean()) for s in PA.STAGES}
                o, tg = T[c]["object"], T[c]["target"]
                grasp_ok[c] = float(np.mean([r["bound"] == o for r in rr]))
                wrong_obj[c] = float(np.mean([r["bound"] not in (None, o) for r in rr]))
                hp = HABIT[o]
                habit[c] = float(np.mean([on_of(r["on"], o, hp) for r in rr])) if hp != tg else None
                op = [p for p in PLATES if p != tg][0]
                other[c] = float(np.mean([on_of(r["on"], o, op) for r in rr]))
                if arm != "I":
                    R0 = json.load(open(os.path.join(ROLL, "%s_I_%d.json" % (m, T[c]["tid"])), encoding="utf-8"))
                    R1 = json.load(open(files[c] + ".json", encoding="utf-8"))
                    da = {k: (R0["args"].get(k), R1["args"].get(k)) for k in set(R0["args"]) | set(R1["args"])
                          if k not in ("out", "port", "videos") and R0["args"].get(k) != R1["args"].get(k)}
                    e0 = [(e["trial"], e.get("branch"), e.get("variant"), e.get("at")) for v in R0["tasks"].values() for e in v["episodes"]]
                    e1 = [(e["trial"], e.get("branch"), e.get("variant"), e.get("at")) for v in R1["tasks"].values() for e in v["episodes"]]
                    H.setdefault(arm, {})[c] = dict(args_diff=da, episodes_match=e0 == e1)
            atl = TF.atlas_est(RATE, NEW, PAIR, ks, PA)
            new_m, kn_m = np.stack([succ[c] for c in NEW]), np.stack([succ[c] for c in KNOWN])
            A[arm] = dict(new=new_m, known=kn_m, row=dict(
                success_new=TF.boot1(new_m), success_known=TF.boot1(kn_m), per_task_success={c: float(succ[c].mean()) for c in comps},
                n_minus_k=TF.boot2(new_m, kn_m), n_minus_k_states=boot2_states(new_m, kn_m),
                stage_shares_new={s: float(np.mean([per[c][s] for c in NEW])) for s in PA.STAGES}, per_task_stages=per,
                grasp_named_mug=grasp_ok, wrong_object_first=wrong_obj, on_trained_plate_instead=habit, on_other_plate=other, atlas=atl,
                side=None if arm == "I" else TF.side_summary([files[c] + "_tv.json" for c in comps])))
        # compounds under I
        comp = {}
        for c in ("KC", "NC"):
            f = os.path.join(ROLL, "%s_I_%d" % (m, T[c]["tid"]))
            if os.path.exists(f + ".json") and os.path.exists(f + "_track.npz"):
                rr = stage_rows(f, T[c], ks)
                s = np.array([[float(r["success"]) for r in rr]])
                comp[c] = dict(success=TF.boot1(s), atoms={"%s>%s" % (o, p): float(np.mean([on_of(r["on"], o, p) for r in rr])) for o, p in T[c]["atoms"]},
                               swapped_atoms={"%s>%s" % (o, [q for q in PLATES if q != p][0]): float(np.mean([on_of(r["on"], o, [q for q in PLATES if q != p][0]) for r in rr]))
                                              for o, p in T[c]["atoms"]})
        row = {arm: (v if isinstance(v, str) else v["row"]) for arm, v in A.items()}

        def dl(x, y, key="new", fn=TF.boot2):
            if isinstance(A.get(x), dict) and isinstance(A.get(y), dict):
                return fn(A[x][key], A[y][key])
            return None
        D = {"%s_minus_I" % x: dl(x, "I") for x in arms if x != "I"}
        D.update({"%s_minus_I_states" % x: dl(x, "I", fn=boot2_states) for x in arms if x != "I"})
        D["known"] = {"%s_minus_I" % x: dl(x, "I", "known") for x in arms if x != "I"}
        if "tvp" in arms:
            D["tvp_minus_tv"] = dl("tvp", "tv")
        V = {}
        if isinstance(A.get("I"), dict):
            kI = row["I"]["success_known"][0]
            V["testable"] = bool(kI >= 0.70)
            V["atlas_I"] = row["I"]["atlas"]["verdict"]
            V["cross_breaks_I"] = bool(row["I"]["atlas"]["groups"]["cross"]["breaks"])
        rule_c = lambda d: "CONFIRMED" if (d[0] > EPS and d[1] > 0) else ("REFUTED" if d[2] < EPS else "UNRESOLVED")   # noqa: E731
        if m == "pi05":
            if D.get("tvp_minus_I"):
                V["primary_tvp_minus_I"] = rule_c(D["tvp_minus_I"])
            if D.get("tv_minus_I"):
                V["secondary_tv_minus_I"] = rule_c(D["tv_minus_I"])
        if m == "groot" and D.get("tv_minus_I"):
            V["primary_tv_minus_I"] = "TV WINS" if D["tv_minus_I"][1] > 0 else "NO CLEAR DIFFERENCE"
        V["guard"] = {x: ("NO COST" if d[1] > -0.10 else "COST") for x, d in D["known"].items() if d}
        hok = all(not h["args_diff"] and h["episodes_match"] for hh in H.values() for h in hh.values()) if H else None
        res["models"][m] = dict(arms=row, deltas=D, verdicts=V, compounds=comp, harness=H, harness_matches_I=hok)
        print("== %s: verdicts %s | harness %s" % (m, json.dumps(V), hok), flush=True)
        for arm, r in row.items():
            if isinstance(r, str):
                print("   %s: %s" % (arm, r), flush=True)
                continue
            print("   %s: new %s known %s per task %s | atlas %s within %s cross %s | grasp named %s | trained plate instead %s" % (
                arm, np.round(r["success_new"], 3).tolist(), np.round(r["success_known"], 3).tolist(),
                {c: round(v, 3) for c, v in r["per_task_success"].items()}, r["atlas"]["verdict"],
                np.round([r["atlas"]["groups"]["within"]["E"]] + r["atlas"]["groups"]["within"]["ci"], 3).tolist(),
                np.round([r["atlas"]["groups"]["cross"]["E"]] + r["atlas"]["groups"]["cross"]["ci"], 3).tolist(),
                {c: round(v, 3) for c, v in r["grasp_named_mug"].items()},
                {c: (None if v is None else round(v, 3)) for c, v in r["on_trained_plate_instead"].items()}), flush=True)
        for k_, d in D.items():
            if isinstance(d, list):
                print("   %s %s" % (k_, np.round(d, 3).tolist()), flush=True)
        print("   known deltas %s | compounds %s" % (json.dumps(D["known"]), json.dumps(comp)), flush=True)
    MV = res["models"]
    tst = {m: MV.get(m, {}).get("verdicts", {}).get("testable") for m in ("pi05", "groot")}
    brk = {m: MV.get(m, {}).get("verdicts", {}).get("cross_breaks_I") for m in ("pi05", "groot")}
    if not any(tst.values()):
        f1 = "NOT TESTABLE"
    else:
        nb = sum(bool(brk[m]) and bool(tst[m]) for m in brk)
        f1 = "REPLICATES" if nb == 2 else ("PARTIAL" if nb == 1 else "NOT REPLICATED")
    wins = [MV.get("pi05", {}).get("verdicts", {}).get("primary_tvp_minus_I") == "CONFIRMED",
            MV.get("groot", {}).get("verdicts", {}).get("primary_tv_minus_I") == "TV WINS"]
    fx = "REPLICATES" if all(wins) else ("PARTIAL" if any(wins) else "NOT REPLICATED")
    res["scene2"] = dict(F1=f1, testable=tst, cross_breaks=brk, FIX=fx, C1=None if ref_row is None else ref_row["C1"])
    print("== SCENE 2: F1 %s (testable %s, cross breaks %s) | FIX %s | reference %s" % (
        f1, tst, brk, fx, None if ref_row is None else (np.round(ref_row["success_new"], 3).tolist(), ref_row["per_task"], ref_row["C1"])), flush=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, out)
    print("[s2read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--out", default=READ)
    A = ap.parse_args()
    if A.mode == "ref":
        ref(A)
    elif A.mode == "summary":
        summary(A.dir)
    else:
        read(A)

"""S2FIX6 (docs/SCAFFOLD_RESULTS.md 67; m3_prereg tag s2_fix6): SR 63's named-mug hand-over with perception (arm Hp) plus one rule,
KEEP THE TARGET CLEAR (arm Hc), on NEW states, layout O, tasks K1, K2, N1, N2 of scene 2 (LIBERO-10 LIVING_ROOM_SCENE5), on frozen
pi0.5 (PRIMARY) and pi0 (SECONDARY).

Why: under Hp the N2 limit is the VLA's first step.  It carries the white mug, which N2 does not name, onto the left plate, which N2
instructs, so that plate is occupied before the named mug is held (SR 63, SR 66).  Putting the white mug back where it was grasped
makes the VLA grasp it again (SR 64, 240 of 240).  On K2 under Hp the white mug held first goes to its trained left plate (SR 66: 59
of 60) and the trained program then grasps the named mug, which P places.

Facts (SR 50's task table, s2_tasks.TASKS and results/flawed/s2/bddl): K1 white -> left (plate_1), K2 yellow-and-white -> right
(plate_2), N1 white -> right, N2 yellow-and-white -> left; the trained compound KC is white -> left, yellow-and-white -> right.  So on
K2 the plate the instruction does NOT name (left) is the white mug's trained plate: Hc there replaces the VLA's own trained first
carry by P's carry to the same plate.  On N2 the unnamed plate is the right plate, so the white mug is moved off N2's target.

ARM Hp: SR 63's arm, run through SR 63's client unchanged (`s2fix3.py client --arm Hp`).

ARM Hc (this file, `client --arm Hc`): Hp plus CLEAR-AWAY.
  - The instruction text (the prompt the VLA receives) is parsed for the mug names (s2fix4.parse_named, SR 64's parser, unchanged)
    and for the plate names it contains (PLATE_PHRASES: "left plate", "right plate") -> NAMED mugs, NAMED_PLATES.  On the four
    single-step tasks NAMED = {the task's object} and NAMED_PLATES = {the task's target} (asserted).
  - Hold events, detector, perception, target registration, the held-mug rule and P are Hp's, unchanged (s2fix3.Percept,
    ProprioHeld, s2fix.Place).  Hp's registration (one OWLv2 + depth scene at n = WAIT) already detects both plates and assigns
    "left" / "right" by SIDE_RULE; Hc keeps the OTHER side's point as well (no extra detector call, no extra simulator read).
  - At a detector onset with no named hand-over committed and neither P nor a clear-away active, the perceived held mug m decides:
      m in NAMED                                -> P takes over to the target plate (Hp's rule);
      m not None, m not in NAMED, no clear-away
      yet in this episode (cap CL_CAP = 1)       -> CLEAR-AWAY, then control returns to the VLA with the same instruction;
      otherwise (m None, or the cap reached)    -> the VLA keeps control (Hp's rule).
  - CLEAR-AWAY.  Destination = the plate the instruction does NOT name (exactly one plate unnamed), at its perceived registration
    point: P (s2fix.Place, unchanged, with the constants of the trained half ending on that plate, s2fix.HALF_OF_PLATE, as Hp does for
    its target; rng default_rng(seed_base + 30)) carries and releases the mug there.  After P's release and 10 cm retreat (phase 4,
    err < .015 or CL_RETREAT steps) the clear-away is done and the VLA acts again.  FALLBACK (both plates named, neither named, or the
    other plate not perceived at registration): SR 64's scripted put-back to the grasp spot (s2fix4.PutBack, unchanged; the grasp
    spot is the hand's lowest point since the executed gripper command last closed).  If the hold detector is off for 3 steps during
    P's carry or descent (a drop), the clear-away aborts and the VLA acts again (s2fix.Place's / PutBack's own abort rule).  Every
    started clear-away counts toward the cap, so an episode cannot loop.
  - What Hc reads beyond Hp: the instruction's plate words (the VLA receives the text; Hp already reads the target plate, as the
    task's target) and the second plate of Hp's own registration scene.  No lookup of trained pairings (the destination is
    "the plate not named", never "the mug's trained plate"); no new simulator read (depth and camera calibration stay privileged, as
    in Hp).  The simulator oracle (H's held-mug rule, Hc's decision on the oracle's held mug, the other plate's registration error,
    and after every clear-away the mug's displacement, tilt and on-plate predicates) is LOGGED in the side file, never fed.
  - For an instruction that names both mugs (KC, NC) the rule never fires on them; KC / NC are not run here.
END Hc.

New states: LIBERO's sampler on K1's BDDL, seeds 970000 + k (none equal to an SR 50-66 state), k 0-3 smoke, 4-63 main; branches
seed_base 44000000 + 1000 k.  Servers / clients: SR 63's (pi0.5: pp_serve_purify --mode intact --comp= --seed 13; pi0: serve_guided
--mode ours --seed 13 with BACKBONE_CONFIG=pi0_libero), client seed 7, --bind track.

Modes (LIBERO venv, py3.8):
    hist [--out X]                        (CPU)   the key risk on EXISTING outputs: SR 63/64/66's Hp side files (no new rollout)
    states                                (EGL)   -> results/flawed/s2fix6/O/s2_living_<tid>.npy, branches_{smoke,main}.json
    client --arm Hc <s2.py client args>   (needs the OWLv2 sidecar on $S2FIX3_DET_PORT)
    ref --tid T --branches B --out-dir D  (CPU, C1)
    summary DIR                           (smoke)
    read [DIR --selftest --out X]         (CPU)
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import s2                                                                   # noqa: E402
import s2cue                                                                # noqa: E402
import s2fix3 as F3                                                         # noqa: E402
import s2fix4 as F4                                                         # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2fix6")
ROLL = os.path.join(OUT, "s2fix6_roll")
READ = os.path.join(OUT, "s2fix6_read.json")
HIST = os.path.join(CD, "hist.json")
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
BASE, BR_BASE, N_STATES = 970000, 44000000, 64
SMOKE = list(range(0, 4))
MAIN = list(range(4, 64))
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
HELD_R = F3.HELD_R
NI_MARGIN = F3.NI_MARGIN
FLOOR = F3.FLOOR
BAR = 0.50                                                                  # SR 54's bar on N1+N2
RUNS = dict(pi05=("I", "Hp", "Hc"), pi0=("I", "Hp", "Hc"))
PRIMARY_MODEL = "pi05"
PLATE_PHRASES = ((PL, "left plate"), (PR, "right plate"))
CL_CAP = 1
CL_RETREAT = 30
HIST_DIRS = (("SR63", "s2fix3_roll"), ("SR64", "s2fix4_roll"), ("SR66", "s2fix5_roll"))
MAPPING = (
    "NEW states (LIBERO sampler on K1's BDDL, seeds 970000 + k, none equal to an SR 50-66 state), main k = 4-63 (60), branches "
    "seed_base 44000000 + 1000 k; layout O; tasks K1, K2, N1, N2; client seed 7, --bind track.  Models: frozen pi0.5 (PRIMARY; "
    "pp_serve_purify --model pi05 --mode intact --comp= --seed 13) and frozen pi0 (SECONDARY; serve_guided --mode ours --seed 13, "
    "BACKBONE_CONFIG=pi0_libero).  Arms per model: I (s2.py client), Hp (SR 63's arm, s2fix3.py client --arm Hp, unchanged), Hc (Hp + "
    "CLEAR-AWAY: at a hold of a mug the instruction does not name, P carries it to the plate the instruction does not name, fallback "
    "the grasp spot, cap 1; s2fix6.py client --arm Hc).  Success = the BDDL goal.  Matrices [task, state].  Paired bootstrap over "
    "states, tasks fixed (s2.boot2_states, B = 2000, numpy default_rng(0); CI = 2.5 / 97.5).  Per model, registered labels: (V1) Hc - "
    "Hp on N2: CLEAR-AWAY HELPS iff lo > 0; CLEAR-AWAY HURTS iff hi < 0; else NO CLEAR DIFFERENCE.  (V2, SR 62's rule) dN = Hc - I on "
    "N1+N2, dK = Hc - I on K1+K2: FIX WORKS iff dN lo > 0 AND dK lo > -.05 (NON-INFERIOR at margin -.05); PARTIAL iff exactly one "
    "holds; FAILS iff neither.  (V3) BAR: MEETS BAR iff Hc's N1+N2 success >= .50 AND dN lo > 0 (SR 54's FIXED); else BELOW BAR; "
    "descriptive beside it: the state-bootstrap CI of Hc's N1+N2 success (s2.boot2_states against zeros) and whether its lower bound "
    "is >= .50.  PRIMARY: V1, V2, V3 on pi0.5.  SR 50's floor on every model: TESTABLE iff its I K1+K2 success >= .70 on these "
    "states; a NOT TESTABLE model's labels read NOT TESTABLE with the rule's outcome descriptive.  MECHANISM (registered read, "
    "descriptive): per task, clear-aways fired (share of episodes, s2cue.boot_share), kind (plate / spot), result (done / abort), "
    "held mug (simulator rule and perceived), the cleared mug's final plate per the logged oracle; on N2 the share of the 60 episodes "
    "in which the named mug is held at an onset after a completed clear-away, in which P takes over the named mug after it (the "
    "registered mechanism share; a success can end the episode during P's descent, before its release is logged), and in which P's "
    "release of the named mug is logged after it, with N2 success given each; the next onset's held mug after a clear-away (named / same / other / none); "
    "for Hp the same N2 shares after the first hold of an unnamed mug.  SECONDARY: Hp's FIX label per model (SR 63 replication); Hc "
    "- Hp on N1+N2, K1+K2, K2 and all four tasks (GAIN iff lo > 0, COST iff hi < 0, else NO CLEAR DIFFERENCE).  Descriptive: "
    "perception accuracy (SR 63's accuracy() on the Hp / Hc side files) and the cleared-to plate's registration error, final plate "
    "of each mug (s2fix2.describe), episode length and time-outs.  C1: the reference planner (s2.plan_episode via s2cue3.ref_eps, "
    "unchanged) on the main states; MET iff its N1-N2 success >= .80.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def parse_plates(language):
    s, out = " " + language.lower() + " ", set()
    for p, ph in PLATE_PHRASES:
        if re.search(r"\b" + re.escape(ph) + r"\b", s):
            out.add(p)
    return out


def clear_dest(named_plates):
    """-> the one plate the instruction does not name, or None (both or neither named: the fallback)."""
    un = [p for p in s2.PLATES if p not in named_plates]
    return un[0] if len(un) == 1 else None


# ---------------------------------------------------------------- the key risk on existing outputs (CPU; before states / smoke)
def after_unnamed(H, rr, obj):
    """Hp side files -> per episode: first onset unnamed (sim), named held at a later onset, P released the named mug after it."""
    out = []
    for h, r in zip(H, rr):
        ev = h["events"]
        i0 = next((i for i, e in enumerate(ev) if e["sim_held"] is not None and e["sim_held"] != obj), None)
        if i0 is None:
            out.append(dict(unnamed=False, named_after=False, handover_after=False, released_after=False, success=bool(r["success"])))
            continue
        row0 = ev[i0]["row"]
        out.append(dict(unnamed=True, first_unnamed=i0 == 0, named_after=any(e["sim_held"] == obj for e in ev[i0 + 1:]),
                        handover_after=any(hd["row"] > row0 for hd in h["handovers"]),
                        released_after=h["committed"] is not None and h["committed"] > row0, success=bool(r["success"])))
    return out


def hist(a):
    out = os.path.abspath(a.out or HIST)
    assert not os.path.exists(out), "never overwrite: " + out
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    res = {}
    for sr, d in HIST_DIRS:
        RD = os.path.join(OUT, d)
        for model in ("pi05", "pi0"):
            run = F3.load_run(RD, model, "Hp", MAIN)
            if isinstance(run, str):
                res["%s %s" % (sr, model)] = run
                continue
            R_ = {}
            for c in ("N2", "K2"):
                o = T[c]["object"]
                E = after_unnamed(run["side"][c], run["rows"][c], o)
                n = len(E)
                R_[c] = dict(
                    n=n, success=float(np.mean([e["success"] for e in E])),
                    first_hold_unnamed=int(sum(e.get("first_unnamed", False) for e in E)),
                    any_unnamed_hold=int(sum(e["unnamed"] for e in E)),
                    named_held_after_unnamed=int(sum(e["named_after"] for e in E)),
                    handover_after_unnamed=int(sum(e["handover_after"] for e in E)),
                    released_after_unnamed=int(sum(e["released_after"] for e in E)),
                    success_given_released_after=(float(np.mean([e["success"] for e in E if e["released_after"]]))
                                                  if any(e["released_after"] for e in E) else None),
                    white_final=s2fix2_describe_plate(run["rows"][c], W))
            res["%s %s" % (sr, model)] = R_
            print("[s2fix6 hist] %s %s Hp: %s" % (sr, model, json.dumps(R_)), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[s2fix6 hist] wrote %s" % out, flush=True)


def s2fix2_describe_plate(rr, m):
    return {str(p): int(sum(s2cue.plate_of(r["on"], m) == p for r in rr)) for p in (PL, PR, None, "both")}


# ---------------------------------------------------------------- states (LIBERO venv, EGL)
def states(a):
    import s2_tasks as T2
    od = os.path.join(CD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    for t in M_["tasks"]:
        print("[s2fix6 states] parse %s %r -> NAMED %s plates %s dest %s (object %s target %s)" % (
            t["key"], t["language"], sorted(F4.parse_named(t["language"])), sorted(parse_plates(t["language"])),
            clear_dest(parse_plates(t["language"])), t.get("object"), t.get("target")), flush=True)
    for t in M_["tasks"][:4]:
        assert F4.parse_named(t["language"]) == {t["object"]}, (t["key"], t["language"])
        assert parse_plates(t["language"]) == {t["target"]}, (t["key"], t["language"])
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))

    def draw(k):                                                            # s2_tasks.build's rule with a new base
        np.random.seed(BASE + k)
        env.seed(BASE + k)
        env.reset()
        return env.sim.get_state().flatten()
    S = np.stack([draw(k) for k in range(N_STATES)])
    env.close()
    old = [np.load(os.path.join(s2.SD, "states", "%s_0.npy" % s2.SUITE))] + [
        np.load(os.path.join(OUT, d, L, "%s_0.npy" % s2.SUITE)) for d, L in
        (("s2cue3", "O"), ("s2cue4", "O"), ("s2cue5", "O"), ("s2fix2", "O"), ("s2fix2", "PD2"), ("s2fix3", "O"), ("s2fix4", "O"),
         ("s2fix5", "O"))]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("same as an SR 50-66 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    for ph, ks in (("smoke", SMOKE), ("main", MAIN)):
        with open(os.path.join(CD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE + 1000 * k, at=0, variant="s2") for k in ks], fh, indent=1)
    print("[s2fix6 states] %d new states (seed %d + k), none equal to an SR 50-66 state; wrote %s" % (len(S), BASE, CD), flush=True)


# ---------------------------------------------------------------- client: arm Hc (LIBERO venv)
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
    assert arm == "Hc", arm
    import pp_atlas as PA
    import libero_client as LC
    import sc_rv as RV
    import s2_tasks as T2
    import s2fix as SF
    from lifelong_ood.policy.parts_flow import ProprioHeld
    M = s2.register_suite(LC)
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = M["tasks"][TID]
    assert TASK["tid"] == TID and len(TASK["atoms"]) == 1
    NAMED = F4.parse_named(TASK["language"])
    assert NAMED == {TASK["object"]}, (NAMED, TASK["object"])
    NPL = parse_plates(TASK["language"])
    assert NPL == {TASK["target"]}, (NPL, TASK["target"])
    DEST = clear_dest(NPL)                                                  # the plate the instruction does not name (or None)
    ms = int(TASK["max_steps"])
    assert "--max-steps" not in sys.argv
    sys.argv += ["--max-steps", str(ms)]
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    KP = json.load(open(os.path.join(s2.SD, "keypoints.json"), encoding="utf-8"))["tasks"]
    KT = KP[SF.HALF_OF_PLATE[TASK["target"]]]
    assert KT["target"] == TASK["target"], (KT["target"], TASK["target"])
    KD = None if DEST is None else KP[SF.HALF_OF_PLATE[DEST]]
    assert KD is None or KD["target"] == DEST, (KD, DEST)
    PER = F3.Percept()
    ORIG, RECS, SIDE = LC.OffScreenRenderEnv, [], []
    SIDEF = OUTF.replace(".json", "_h.json")

    def rule(m, n_cl):
        if m is not None and m in NAMED:
            return "place"
        if m is not None and n_cl < CL_CAP:
            return "clear"
        return "vla: cap reached" if m is not None else "vla: other mug"

    class HEnv(s2.Rec2):
        def __init__(self, env):
            super().__init__(env, TASK["object"], TASK["target"])
            self.det, self.pl, self.cl = None, None, None

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            i = len(RECS)
            self.k = int(br[i]["trial"]) if i < len(br) else None
            self.sb = int(br[i]["seed_base"]) if i < len(br) else 0
            self.rng = np.random.default_rng(self.sb + 20)                 # SR 54's P rng rule
            self.rng_cl = np.random.default_rng(self.sb + 30)
            self.det = ProprioHeld(X["tw"], X["lift"])
            self.pl, self.cl, self.cl_kind, self.obs = None, None, None, obs
            self.h = dict(onsets=[], events=[], handovers=[], aborts=[], committed=None, p_steps=0, phase_max=-1,
                          clears=[], cl_steps=0, n_cl=0)
            self.reg, self.mem, self.tgt_p, self.dst_p, self.gr, self.start = None, {}, None, None, None, None
            return obs

        def _tgt(self):
            return np.asarray(self.tgt_p, np.float64)

        def _sim_held(self, ee):
            d = {m: float(np.linalg.norm(self._pos(m)[:2] - np.asarray(ee, np.float64)[:2])) for m in s2.MUGS}
            m = min(d, key=d.get)
            return (m if d[m] < HELD_R else None), d

        def _per_held(self, ee):
            d = {m: float(np.linalg.norm(np.asarray(p)[:2] - np.asarray(ee, np.float64)[:2])) for m, p in self.mem.items()}
            if not d:
                return None, d
            m = min(d, key=d.get)
            return (m if d[m] < HELD_R else None), d

        def _sim_all(self):
            return {n.split("_")[0]: np.round(self._pos(n), 4).tolist() for n in s2.MOVABLE}

        def _quat(self, n):
            E = self._e.env
            return np.array(E.sim.data.body_xquat[E.obj_body_id[n]], np.float64)

        def _register(self, obs):                                           # Hp's registration, unchanged, + the other plate kept
            sc = PER.scene(self._e, obs)
            self.mem = {m: v["pos"] for m, v in sc["mugs"].items()}
            side = F3.side_word(TASK["target"])
            self.tgt_p = sc[side]
            dside = None if DEST is None else F3.side_word(DEST)
            self.dst_p = None if dside is None else sc[dside]
            tp, op = self._pos(TASK["target"]), self._pos([p for p in s2.PLATES if p != TASK["target"]][0])
            orc = dict(sim=self._sim_all())
            if self.tgt_p is not None:
                e = np.asarray(self.tgt_p) - tp
                orc.update(tgt_err_xy_cm=round(100 * float(np.linalg.norm(e[:2])), 2), tgt_err_z_cm=round(100 * float(e[2]), 2),
                           side_ok=bool(np.linalg.norm(np.asarray(self.tgt_p)[:2] - tp[:2]) < np.linalg.norm(np.asarray(self.tgt_p)[:2] - op[:2])))
            if self.dst_p is not None:
                dp = self._pos(DEST)
                e = np.asarray(self.dst_p) - dp
                orc.update(dst_err_xy_cm=round(100 * float(np.linalg.norm(e[:2])), 2), dst_err_z_cm=round(100 * float(e[2]), 2),
                           dst_side_ok=bool(np.linalg.norm(np.asarray(self.dst_p)[:2] - dp[:2]) < np.linalg.norm(np.asarray(self.dst_p)[:2] - tp[:2])))
            orc["mug_ok"] = {m.split("_")[0]: bool(min(s2.MUGS, key=lambda x: np.linalg.norm(self._pos(x)[:2] - np.asarray(p)[:2])) == m)
                             for m, p in self.mem.items()}
            self.reg = dict(row=len(self.rows["held"]) - 1, scene=sc, side=side, target=self.tgt_p, dest=DEST, dest_side=dside,
                            dest_point=self.dst_p, oracle=orc)
            self.start = {m: self._pos(m) for m in s2.MUGS}                 # oracle log only

        def _end_clear(self, res, row, ee):
            c = self.h["clears"][-1]
            sm = c["sim_mug"]
            c.update(row_end=row, result=res, ee_end=np.round(ee, 4).tolist(),
                     log=(self.cl.log["phases"] if self.cl_kind == "plate" else self.cl.log))
            if sm is not None:
                d = self._pos(sm) - self.start[sm]
                c.update(sim_disp_xy_cm=round(100 * float(np.linalg.norm(d[:2])), 2), sim_dz_cm=round(100 * float(d[2]), 2),
                         sim_tilt_deg=round(F3.tilt_deg(self._quat(sm)), 1),
                         sim_on_plate={p: int(bool(self._e.env._eval_predicate(["on", sm, p]))) for p in s2.PLATES})
            self.cl, self.cl_kind = None, None

        def step(self, a):
            ee0 = np.asarray(self.obs["robot0_eef_pos"], np.float64)
            rot0 = T2.quat_mat(np.asarray(self.obs["robot0_eef_quat"], np.float64))
            if self.pl is not None:
                a = self.pl.action(ee0, rot0, self._tgt()).tolist()
                self.h["p_steps"] += 1
            elif self.cl is not None:
                a = (self.cl.action(ee0, rot0, np.asarray(self.dst_p, np.float64)) if self.cl_kind == "plate"
                     else self.cl.action(ee0, rot0)).tolist()
                self.h["cl_steps"] += 1
            r = super().step(a)
            self.obs = r[0]
            if self.n < s2.WAIT:
                return r
            if self.reg is None:
                self._register(r[0])
            row = len(self.rows["held"]) - 1
            ee = np.asarray(r[0]["robot0_eef_pos"], np.float64)
            if float(a[6]) > 0:                                             # the grasp spot: the hand at its lowest since the close
                if self.gr is None or ee[2] < self.gr[2]:
                    self.gr = ee.copy()
            else:
                self.gr = None
            was = self.det.on
            on = bool(self.det.step(np.asarray(r[0]["robot0_gripper_qpos"], np.float64), float(a[6]), float(ee[2])))
            if on and not was:
                self.h["onsets"].append(row)
            if self.pl is not None:
                res = self.pl.advance(on, ee)
                self.h["phase_max"] = max(self.h["phase_max"], self.pl.phase)
                if self.pl.phase >= 3 and self.h["committed"] is None:
                    self.h["committed"] = row
                if res == "abort":
                    self.h["aborts"].append(dict(row=row, phase=self.pl.phase, log=self.pl.log["phases"]))
                    self.pl = None
            elif self.cl is not None:
                if self.cl_kind == "plate":
                    res = self.cl.advance(on, ee)
                    if res is None and self.cl.phase == 4 and (self.cl.err < 0.015 or self.cl.in_phase >= CL_RETREAT):
                        res = "done"
                    if res is None and self.cl.phase >= 3:
                        self.h["clears"][-1]["released"] = True
                else:
                    res = self.cl.advance(on, ee)
                if res is not None:
                    self._end_clear(res, row, ee)
            elif on and not was and self.h["committed"] is None:
                sh, sd = self._sim_held(ee)
                ev = dict(row=row, oracle_held=bool(self.rows["held"][-1]), n_cl=self.h["n_cl"])
                sc = PER.scene(self._e, r[0])
                self.mem.update({m: v["pos"] for m, v in sc["mugs"].items()})
                held, dd = self._per_held(ee)
                ev.update(detected={m.split("_")[0]: dict(pos=np.round(v["pos"], 4).tolist(), score=v["score"])
                                    for m, v in sc["mugs"].items()})
                ev.update(held=held, dist={m.split("_")[0]: round(v, 4) for m, v in dd.items()},
                          sim_held=sh, sim_dist={m.split("_")[0]: round(v, 4) for m, v in sd.items()},
                          oracle_decision=rule(sh, self.h["n_cl"]))
                dec = rule(held, self.h["n_cl"])
                if dec == "place" and self.tgt_p is None:
                    dec = "vla: no target"
                ev["decision"] = dec
                rot = T2.quat_mat(np.asarray(r[0]["robot0_eef_quat"], np.float64))
                if dec == "place":
                    pz = self._tgt()
                    self.pl = SF.Place(KT, self.rng, ee, rot, float(pz[2]))
                    self.h["handovers"].append(dict(row=row, oracle_held=ev["oracle_held"]))
                elif dec == "clear":
                    if DEST is not None and self.dst_p is not None:
                        self.cl_kind = "plate"
                        self.cl = SF.Place(KD, self.rng_cl, ee, rot, float(self.dst_p[2]))
                    else:                                                   # fallback: SR 64's put-back to the grasp spot
                        self.cl_kind = "spot"
                        gp = self.gr if self.gr is not None else ee
                        self.cl = F4.PutBack(gp, self.rng_cl, ee, rot)
                    self.h["n_cl"] += 1
                    self.h["clears"].append(dict(row=row, mug=held, sim_mug=sh, kind=self.cl_kind,
                                                 dest=DEST if self.cl_kind == "plate" else "grasp spot",
                                                 dest_point=None if self.cl_kind != "plate" else np.round(self.dst_p, 4).tolist(),
                                                 onset_ee=np.round(ee, 4).tolist(), released=False))
                    ev["clear_kind"] = self.cl_kind
                self.h["events"].append(ev)
            return r

        def close(self):
            if self.n > 0:
                self.finish()
                RECS.append(self.data())
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
                i = len(SIDE)
                rec = dict(arm=arm, layout="O", trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, tid=TID, key=TASK["key"])
                rec.update(self.h)
                rec["reg"] = self.reg
                rec["p_final_phase"] = None if self.pl is None else self.pl.phase
                rec["p_log"] = None if self.pl is None else self.pl.log["phases"]
                rec["cl_active_at_end"] = self.cl is not None
                SIDE.append(rec)
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, layout="O", tid=TID, key=TASK["key"], language=TASK["language"], named=sorted(NAMED),
                                   named_plates=sorted(NPL), dest=DEST, cl_cap=CL_CAP, lex=X, kt=KT, kd=KD, held_r=HELD_R, cal=F3.CAL,
                                   side_rule=F3.SIDE_RULE, queries=F3.QUERIES, detector=PER.model,
                                   place="SR 54 object-blind P (PERCEIVED target position)",
                                   clear="SR 54 object-blind P to the PERCEIVED unnamed plate; fallback SR 64 put-back to the grasp spot",
                                   eps=SIDE),
                              fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
                os.replace(tmp, SIDEF)
                print("[s2fix6] %s %s trial %s: reg %s dst %s | events %s | clears %s | handovers %s aborts %d committed %s phase_max %d "
                      "p_steps %d cl_steps %d" % (
                          arm, TASK["key"], rec["trial"], None if self.reg is None else self.reg["oracle"].get("tgt_err_xy_cm"),
                          None if self.reg is None else self.reg["oracle"].get("dst_err_xy_cm"),
                          [(e["row"], e["held"] and e["held"].split("_")[0], e["sim_held"] and e["sim_held"].split("_")[0], e["decision"])
                           for e in rec["events"][:6]],
                          [(c["row"], c.get("row_end"), c["kind"], c.get("result"), c.get("sim_on_plate"), c.get("sim_tilt_deg"))
                           for c in rec["clears"]],
                          [h["row"] for h in rec["handovers"]], len(rec["aborts"]), rec["committed"], rec["phase_max"], rec["p_steps"],
                          rec["cl_steps"]), flush=True)
                self.pl, self.cl = None, None
            self._e.close()

    LC.OffScreenRenderEnv = lambda *a, **k: HEnv(ORIG(*a, **k))
    print("[s2fix6] arm %s task %s %r object %s NAMED %s target %s plates named %s clear-to %s max_steps %d lex %s held_r %.2f "
          "detector %s cal %s side %s cl_cap %d" % (
              arm, TASK["key"], TASK["language"], TASK["object"], sorted(NAMED), TASK["target"], sorted(NPL), DEST, ms,
              (X["tw"], X["lift"]), HELD_R, PER.model, F3.CAL, F3.SIDE_RULE, CL_CAP), flush=True)
    LC.main()


# ---------------------------------------------------------------- reference planner (C1; LIBERO venv, CPU)
def ref(a):
    import s2cue3
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    assert t["tid"] == a.tid
    os.makedirs(a.out_dir, exist_ok=True)
    fn = os.path.join(a.out_dir, "ref_O_%d.json" % a.tid)
    assert not os.path.exists(fn), "exists: " + fn
    S = np.load(os.path.join(CD, "O", "%s_%d.npy" % (s2.SUITE, a.tid)))
    br = {int(b["trial"]): int(b["seed_base"]) for b in json.load(open(a.branches, encoding="utf-8"))}
    kp, eps = s2cue3.ref_eps(S, t, t["target"], sorted(br), lambda k: br[k])
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(layout="O", tid=a.tid, key=t["key"], kp=kp, eps=eps), fh, indent=1)


# ---------------------------------------------------------------- read helpers (CPU)
def ep_steps(RD, model, arm, tid):
    Rj = json.load(open(F3.stem_of(RD, model, arm, tid) + ".json", encoding="utf-8"))
    return [int(e["steps"]) for v in Rj["tasks"].values() for e in v["episodes"]]


def clear_stats(run, RD, model, ks):
    """Hc side files -> per task: clear-aways fired, kind, result, held mug, where the cleared mug ended, what came next."""
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    out = {}
    for c, H in run["side"].items():
        if H is None:
            continue
        rr, o = run["rows"][c], T[c]["object"]
        succ = run["succ"][c]
        fired = np.array([float(bool(h.get("clears"))) for h in H])
        cls = [h["clears"][0] for h in H if h.get("clears")]
        nxt, ind_held, ind_rel, ind_ho = {}, np.zeros(len(H)), np.zeros(len(H)), np.zeros(len(H))
        for i, h in enumerate(H):
            if not h.get("clears"):
                continue
            cl = h["clears"][0]
            if cl.get("result") != "done":
                continue
            re_ = cl["row_end"]
            later = [e for e in h["events"] if e["row"] > re_]
            key = "none" if not later else ("named" if later[0]["sim_held"] == o else
                                            ("same" if later[0]["sim_held"] == cl["sim_mug"] else str(later[0]["sim_held"])))
            nxt[key] = nxt.get(key, 0) + 1
            ind_held[i] = float(any(e["sim_held"] == o for e in later))
            ind_rel[i] = float(h["committed"] is not None and h["committed"] > re_)
            ind_ho[i] = float(any(hd["row"] > re_ for hd in h["handovers"]))   # P took the named mug after the clear-away
        one = np.ones(len(H))
        st = ep_steps(RD, model, "Hc", T[c]["tid"])
        clr_on = {}
        for cl in cls:
            sp = cl.get("sim_on_plate") or {}
            k_ = "+".join(sorted(p for p, v in sp.items() if v)) or "none"
            clr_on[k_] = clr_on.get(k_, 0) + 1
        out[c] = dict(
            fired=s2cue.boot_share(fired, one), n_fired=int(fired.sum()),
            kind={x: int(sum(cl["kind"] == x for cl in cls)) for x in ("plate", "spot")},
            results={str(x): int(sum(cl.get("result") == x for cl in cls)) for x in ("done", "abort", None)},
            released=int(sum(bool(cl.get("released")) for cl in cls)),
            sim_mug={str(m and m.split("_")[0]): int(sum(cl["sim_mug"] == m for cl in cls)) for m in (W, Y, R, None)},
            perceived_mug={str(m and m.split("_")[0]): int(sum(cl["mug"] == m for cl in cls)) for m in (W, Y, R)},
            cleared_mug_on_plate_after=clr_on,
            tipped=int(sum((cl.get("sim_tilt_deg") or 0) > 45 for cl in cls)),
            steps_median=float(np.median([cl["row_end"] - cl["row"] for cl in cls if cl.get("row_end") is not None])) if cls else None,
            onset_row_median=float(np.median([cl["row"] for cl in cls])) if cls else None,
            next_onset_held=nxt,
            named_held_after_clear=s2cue.boot_share(ind_held, one), named_released_after_clear=s2cue.boot_share(ind_rel, one),
            success_given_released_after=float(succ[ind_rel > 0].mean()) if (ind_rel > 0).any() else None,
            named_handover_after_clear=s2cue.boot_share(ind_ho, one),
            success_given_handover_after=float(succ[ind_ho > 0].mean()) if (ind_ho > 0).any() else None,
            success_given_fired=float(succ[fired > 0].mean()) if (fired > 0).any() else None,
            success_given_not_fired=float(succ[fired == 0].mean()) if (fired == 0).any() else None,
            decisions={x: int(sum(e["decision"] == x for h in H for e in h["events"])) for x in
                       ("place", "clear", "vla: other mug", "vla: cap reached", "vla: no target")},
            episodes_with_place=int(sum(len(h["handovers"]) > 0 for h in H)),
            steps_median_episode=float(np.median(st)), timed_out=int(sum(s >= 400 and not r["success"] for s, r in zip(st, rr))),
            dst_err_xy_cm_median=(float(np.median([h["reg"]["oracle"]["dst_err_xy_cm"] for h in H
                                                   if h.get("reg") and "dst_err_xy_cm" in h["reg"]["oracle"]]))
                                  if any(h.get("reg") and "dst_err_xy_cm" in h["reg"]["oracle"] for h in H) else None),
            dst_err_xy_cm_max=(float(np.max([h["reg"]["oracle"]["dst_err_xy_cm"] for h in H
                                             if h.get("reg") and "dst_err_xy_cm" in h["reg"]["oracle"]]))
                               if any(h.get("reg") and "dst_err_xy_cm" in h["reg"]["oracle"] for h in H) else None),
            dst_side_ok=int(sum(bool(h.get("reg") and h["reg"]["oracle"].get("dst_side_ok")) for h in H)))
    return out


def hp_after(run, RD, model, arm):
    """Hp (or Hc) -> on N2 and K2: the after-unnamed shares, as `hist` computes them on SR 63/64/66."""
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    out = {}
    for c in ("N2", "K2"):
        H = run["side"][c]
        if H is None:
            continue
        E = after_unnamed(H, run["rows"][c], T[c]["object"])
        one = np.ones(len(E))
        out[c] = dict(any_unnamed_hold=int(sum(e["unnamed"] for e in E)),
                      first_hold_unnamed=int(sum(e.get("first_unnamed", False) for e in E)),
                      named_held_after_unnamed=s2cue.boot_share(np.array([float(e["named_after"]) for e in E]), one),
                      handover_after_unnamed=s2cue.boot_share(np.array([float(e["handover_after"]) for e in E]), one),
                      released_after_unnamed=s2cue.boot_share(np.array([float(e["released_after"]) for e in E]), one))
    return out


def summary(d):
    M_ = s2.meta()
    for model, arms in RUNS.items():
        for arm in arms:
            for tid in TIDS:
                st = F3.stem_of(d, model, arm, tid)
                if not (os.path.exists(st + ".json") and os.path.exists(st + "_track.npz")):
                    continue
                t = M_["tasks"][tid]
                Rj = json.load(open(st + ".json", encoding="utf-8"))
                E = [e for v in Rj["tasks"].values() for e in v["episodes"]]
                ks = [int(e["trial"]) for e in E]
                H = json.load(open(st + "_h.json", encoding="utf-8"))["eps"] if os.path.exists(st + "_h.json") else [None] * len(ks)
                for k, e, r, h in zip(ks, E, s2.stage_rows(st, t, ks), H):
                    on = {"%s>%s" % (m.split("_")[0][:5], p): 1 for m in s2.MUGS for p in s2.PLATES if s2.on_of(r["on"], m, p)}
                    ev = None if h is None else [(x["row"], x.get("held") and x["held"].split("_")[0],
                                                  x.get("sim_held") and x["sim_held"].split("_")[0], x.get("decision")) for x in h.get("events", [])]
                    cl = None if h is None else [(c["row"], c.get("row_end"), c["kind"], c.get("result"), c.get("sim_on_plate"),
                                                  c.get("sim_tilt_deg")) for c in h.get("clears", [])]
                    print("[s2fix6 summary] %s %s %s k%d success %s steps %s bound %s on %s | events %s | clears %s | phase_max %s" % (
                        model, arm, t["key"], k, r["success"], e["steps"], r["bound"], on, ev, cl, None if h is None else h["phase_max"]),
                        flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_") and x.endswith(".json")):
        Rj = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2fix6 summary] %s: %s" % (f, [(e["trial"], e["success"]) for e in Rj["eps"]]), flush=True)


def gate(lab, testable):
    return lab if testable else "NOT TESTABLE (rule: %s)" % lab


def read(a):
    from lifelong_ood import ledger
    import s2fix2
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD, mism = SMOKE, a.dir, "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_fix6"]
        assert len(rows) == 1, "expected exactly one s2_fix6 prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD = MAIN, ROLL
    res = dict(states=ks, mapping=MAPPING, code_mismatch=mism, models={})
    nulr = {k: {m: dict(predicted=None) for m in s2.MUGS} for k in ks}
    ALLT = s2.KNOWN + s2.NEW
    for model, arms in RUNS.items():
        runs = {arm: F3.load_run(RD, model, arm, ks) for arm in arms}
        R_ = dict(arms={}, deltas={}, verdicts={})
        ok = {arm: not isinstance(v, str) for arm, v in runs.items()}
        for arm, run in runs.items():
            if isinstance(run, str):
                R_["arms"][arm] = run
                print("== %s %s: %s" % (model, arm, run), flush=True)
                continue
            R_["arms"][arm] = dict(success={c: float(v.mean()) for c, v in run["succ"].items()},
                                   new=float(F3.mat(run, s2.NEW).mean()), known=float(F3.mat(run, s2.KNOWN).mean()),
                                   describe=s2fix2.describe(run, nulr, ks))
            print("== %s %s: per task %s | N1+N2 %.3f | K1+K2 %.3f" % (
                model, arm, {c: round(v, 3) for c, v in R_["arms"][arm]["success"].items()}, R_["arms"][arm]["new"],
                R_["arms"][arm]["known"]), flush=True)
            for c, x in R_["arms"][arm]["describe"].items():
                print("      %s %s" % (c, json.dumps(x)), flush=True)

        def dd(x, y, tasks):
            return s2.boot2_states(F3.mat(runs[x], tasks), F3.mat(runs[y], tasks))
        if ok.get("I"):
            kI = float(F3.mat(runs["I"], s2.KNOWN).mean())
            tst = bool(kI >= FLOOR)
            R_["testable"] = dict(intact_K=kI, floor=FLOOR, TESTABLE=tst)
            for arm in ("Hp", "Hc"):
                if not ok.get(arm):
                    continue
                dN, dK = dd(arm, "I", s2.NEW), dd(arm, "I", s2.KNOWN)
                R_["deltas"]["%s-I N" % arm], R_["deltas"]["%s-I K" % arm] = dN, dK
                lab = F3.fix_label(dN, dK)
                R_["verdicts"]["%s FIX label (rule)" % arm] = lab
                R_["verdicts"]["%s FIX label (registered)" % arm] = gate(lab, tst)
                R_["verdicts"]["%s-I K non-inferiority" % arm] = F3.ni_label(dK)
                R_["deltas"]["%s-I per task" % arm] = {c: s2.boot2_states(runs[arm]["succ"][c][None], runs["I"]["succ"][c][None]) for c in ALLT}
                xN = float(F3.mat(runs[arm], s2.NEW).mean())
                bar = "MEETS BAR" if (dN[1] > 0 and xN >= BAR) else "BELOW BAR"
                R_["verdicts"]["%s BAR (rule)" % arm] = bar
                R_["verdicts"]["%s BAR (registered)" % arm] = gate(bar, tst)
                ci = s2.boot2_states(F3.mat(runs[arm], s2.NEW), np.zeros((2, len(ks))))
                R_.setdefault("bar", {})[arm] = dict(new=xN, bar=BAR, sr54_rule=F3.sr54_rule(dN, xN), new_ci=ci,
                                                     lower_bound_ge_bar=bool(ci[1] >= BAR))
            if ok.get("Hc") and ok.get("Hp"):
                d2 = s2.boot2_states(runs["Hc"]["succ"]["N2"][None], runs["Hp"]["succ"]["N2"][None])
                R_["deltas"]["Hc-Hp N2"] = d2
                lab = F3.diff_label(d2, "CLEAR-AWAY HELPS", "CLEAR-AWAY HURTS")
                R_["verdicts"]["Hc-Hp N2 CLEAR-AWAY (rule)"] = lab
                R_["verdicts"]["Hc-Hp N2 CLEAR-AWAY (registered)"] = gate(lab, tst)
                for nm, tasks in (("N", s2.NEW), ("K", s2.KNOWN), ("K2", ("K2",)), ("all", ALLT)):
                    d_ = dd("Hc", "Hp", list(tasks))
                    R_["deltas"]["Hc-Hp %s" % nm] = d_
                    R_["verdicts"]["Hc-Hp %s (secondary)" % nm] = F3.diff_label(d_, "GAIN", "COST")
                R_["deltas"]["Hc-Hp per task"] = {c: s2.boot2_states(runs["Hc"]["succ"][c][None], runs["Hp"]["succ"][c][None]) for c in ALLT}
        if ok.get("Hc"):
            R_["clear"] = clear_stats(runs["Hc"], RD, model, ks)
            print("== %s Hc MECHANISM (clear-away): %s" % (model, json.dumps(R_["clear"])), flush=True)
        for arm in ("Hp", "Hc"):
            if ok.get(arm):
                R_["after_unnamed_%s" % arm] = hp_after(runs[arm], RD, model, arm)
                print("== %s %s after the first unnamed hold: %s" % (model, arm, json.dumps(R_["after_unnamed_%s" % arm])), flush=True)
                R_["perception_%s" % arm] = F3.accuracy(runs[arm], ks)
                print("== %s %s perception accuracy (vs simulator oracle): %s" % (model, arm, json.dumps(R_["perception_%s" % arm])), flush=True)
        for arm in arms:
            if ok.get(arm):
                st_ = {c: ep_steps(RD, model, arm, T_["tid"]) for c, T_ in ((t["key"], t) for t in s2.meta()["tasks"][:4])}
                R_.setdefault("steps", {})[arm] = {c: float(np.median(v)) for c, v in st_.items()}
                R_.setdefault("timed_out", {})[arm] = {c: int(sum(s >= 400 and not r["success"] for s, r in zip(v, runs[arm]["rows"][c])))
                                                      for c, v in st_.items()}
        for k_, v in R_["deltas"].items():
            print("   %s %s %s" % (model, k_, json.dumps(v if not isinstance(v, list) else np.round(v, 3).tolist())), flush=True)
        print("   VERDICTS %s: %s | testable %s | bar %s | median steps %s | timed out %s" % (
            model, json.dumps(R_["verdicts"]), json.dumps(R_.get("testable")), json.dumps(R_.get("bar")), json.dumps(R_.get("steps")),
            json.dumps(R_.get("timed_out"))), flush=True)
        res["models"][model] = R_
    ref_ = {}
    for tid in TIDS:
        f = os.path.join(RD, "ref_O_%d.json" % tid)
        if os.path.exists(f):
            ee = json.load(open(f, encoding="utf-8"))["eps"]
            assert [e["trial"] for e in ee] == ks, f
            ref_[s2.meta()["tasks"][tid]["key"]] = dict(success=float(np.mean([e["success"] for e in ee])),
                                                        errors=int(sum(e.get("error") is not None for e in ee)))
    nn = [ref_[c]["success"] for c in s2.NEW if c in ref_]
    res["reference"] = dict(per_task=ref_, success_new=float(np.mean(nn)) if len(nn) == 2 else None,
                            C1=None if len(nn) < 2 else ("C1 MET" if np.mean(nn) >= 0.80 else "C1 UNMET"))
    print("== reference: %s" % json.dumps(res["reference"]), flush=True)
    print("== code hash mismatches: %s" % mism, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, out)
    print("[s2fix6 read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("hist", "states", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    if A.mode == "read" and A.out is None:
        A.out = READ
    dict(hist=hist, states=states, ref=ref, summary=lambda a: summary(a.dir), read=read)[A.mode](A)

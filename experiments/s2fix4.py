"""S2FIX4 (docs/SCAFFOLD_RESULTS.md 64; m3_prereg tag s2_fix4): SR 63's named-mug hand-over with perception (arm Hp) plus one
structural rule, PUT-BACK (arm Hb), on NEW states, layout O, tasks K1, K2, N1, N2 of scene 2 (LIBERO-10 LIVING_ROOM_SCENE5), on frozen
pi0.5 (PRIMARY), pi0 and GR00T (SECONDARY).

Why: under Hp, N2 is the weak pairing (pi0.5 .30, pi0 .03, GR00T .02).  The VLA's trained first step grasps the white mug, which N2's
instruction does not name, and fills N2's target (left) plate with it before the named mug is ever held (SR 62, SR 63).

ARM Hp: SR 63's arm, run through SR 63's client unchanged (`s2fix3.py client --arm Hp`).

ARM Hb (this file, `client --arm Hb`): Hp plus PUT-BACK.
  - The instruction text (the prompt the VLA receives) is parsed for the mug names it contains (NAME_PHRASES, longest phrase first,
    each match removed before the next is searched) -> NAMED.  On the four single-step tasks NAMED = {the task's object} (asserted).
  - Hold events, detector, perception, target registration, the held-mug rule and P are Hp's, unchanged (s2fix3.Percept, ProprioHeld,
    s2fix.Place with the registered target).
  - At a detector onset with no hand-over committed and neither P nor a put-back active, the perceived held mug m decides:
      m in NAMED                         -> P takes over (Hp's rule);
      m not None, m not in NAMED, and
      fewer than PB_CAP put-backs so far -> PUT-BACK (below), then control returns to the VLA with the same instruction;
      otherwise (m None or cap reached)  -> the VLA keeps control (Hp's rule).
  - PUT-BACK (scripted, proprioception only): the set-down point is the hand position at its lowest point since the executed gripper
    command last turned to close (the same z_min ProprioHeld tracks), i.e. where the mug was grasped.  Phases (fa1g_plan.ctrl, the hand
    rotation held at the onset's, rng default_rng(seed_base + 30)):
      0 carry    to (grasp x, grasp y, max(onset z, grasp z + PB_CLEAR)), gripper closed (<= PB_CARRY steps, or err < .015)
      1 descend  to the grasp point + PB_DZ in z (<= PB_DESC steps, or err < .008)
      2 open     OPEN_STEPS (12) steps
      3 retreat  PB_UP (10 cm) straight up (<= PB_RETREAT steps, or err < .015) -> done
    During 0-1, if the hold detector is off for 3 consecutive steps (the mug was dropped), the put-back aborts and the VLA acts again.
    Every started put-back counts toward PB_CAP (2 per episode), so an episode cannot loop.
  - For an instruction that names both mugs (the scene's trained compound KC, or the swapped compound NC), NAMED holds every mug the
    rule can see except the red distractor, so PUT-BACK never fires on either mug by construction.  K1 and K2 are single-step: each
    names one mug, so on K2 the rule does fire at the trained program's first hold (the white mug).  (KC / NC are not run here.)
  - What Hb reads beyond Hp: the instruction's text (which the VLA already receives; Hp already reads the named mug, as the task's
    object) and the hand's own position at the grasp (proprioception; Hp's ProprioHeld already reads the hand height).  No object pose
    is read by the put-back; no simulator read is added (depth and camera calibration stay rgp's remaining privilege, as in Hp).
  - The simulator oracle (H's held-mug rule, Hb's decision on the oracle's held mug, and after every put-back the mug's displacement
    from its start position and its tilt) is LOGGED in the side file, never fed.
END Hb.

New states: LIBERO's sampler on K1's BDDL, seeds 950000 + k (none equal to an SR 50-63 state), k 0-3 smoke, 4-63 main; branches
seed_base 40000000 + 1000 k.  Development states (put-back mechanics only): SR 63's states (results/flawed/s2fix3/O), k 4-19.
Servers / clients: SR 63's (pi0.5, GR00T: pp_serve_purify --mode intact --comp= --seed 13; pi0: serve_guided --mode ours --seed 13 with
BACKBONE_CONFIG=pi0_libero), client seed 7, --bind track.

Modes (LIBERO venv, py3.8):
    states                                (EGL)   -> results/flawed/s2fix4/O/s2_living_<tid>.npy, branches_{smoke,main,dev}.json
    client --arm Hb <s2.py client args>   (needs the OWLv2 sidecar on $S2FIX3_DET_PORT)
    ref --tid T --branches B --out-dir D  (CPU, C1)
    summary DIR                           (smoke / dev)
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

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2fix4")
ROLL = os.path.join(OUT, "s2fix4_roll")
READ = os.path.join(OUT, "s2fix4_read.json")
DEV_DIR = os.path.join(OUT, "s2fix3", "O")                                  # SR 63's states: development only
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
BASE, BR_BASE, N_STATES = 950000, 40000000, 64
SMOKE = list(range(0, 4))
MAIN = list(range(4, 64))
DEV = list(range(4, 20))
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
HELD_R = F3.HELD_R
NI_MARGIN = F3.NI_MARGIN
FLOOR = F3.FLOOR
BAR = 0.50                                                                  # SR 54's bar on N1+N2
RUNS = dict(pi05=("I", "Hp", "Hb"), pi0=("I", "Hp", "Hb"), groot=("I", "Hp", "Hb"))
PRIMARY_MODEL = "pi05"
NAME_PHRASES = ((Y, "yellow and white mug"), (W, "white mug"), (R, "red mug"))   # longest first
PB_CAP = 2
PB_CLEAR, PB_DZ, PB_UP = 0.03, 0.01, 0.10
PB_CARRY, PB_DESC, PB_RETREAT = 60, 40, 30
OPEN_STEPS, OFF_ABORT = 12, 3
MAPPING = (
    "NEW states (LIBERO sampler on K1's BDDL, seeds 950000 + k, none equal to an SR 50-63 state), main k = 4-63 (60), branches "
    "seed_base 40000000 + 1000 k; layout O; tasks K1, K2, N1, N2; client seed 7, --bind track.  Models: frozen pi0.5 (PRIMARY; "
    "pp_serve_purify --model pi05 --mode intact --comp= --seed 13), frozen pi0 (serve_guided --mode ours --seed 13, "
    "BACKBONE_CONFIG=pi0_libero) and frozen GR00T (pp_serve_purify --model groot --mode intact --comp= --seed 13), SECONDARY.  Arms "
    "per model: I (s2.py client), Hp (SR 63's arm, s2fix3.py client --arm Hp, unchanged), Hb (Hp + PUT-BACK, s2fix4.py client --arm "
    "Hb, cap 2 per episode).  Success = the BDDL goal.  Matrices [task, state].  Paired bootstrap over states, tasks fixed "
    "(s2.boot2_states, B = 2000, numpy default_rng(0); CI = 2.5 / 97.5).  Per model, three registered labels: (1) PUT-BACK: Hb - Hp "
    "on N2: PUT-BACK HELPS iff lo > 0; PUT-BACK HURTS iff hi < 0; else NO CLEAR DIFFERENCE.  (2) FIX (SR 62's rule): dN = Hb - I on "
    "N1+N2, dK = Hb - I on K1+K2; FIX WORKS iff dN lo > 0 AND dK lo > -.05 (NON-INFERIOR at margin -.05); PARTIAL iff exactly one "
    "holds; FAILS iff neither.  (3) BAR: MEETS BAR iff Hb's N1+N2 success >= .50 AND dN lo > 0 (SR 54's FIXED); else BELOW BAR.  "
    "PRIMARY: the three labels on pi0.5.  SR 50's floor on every model: TESTABLE iff its I K1+K2 success >= .70 on these states; a "
    "NOT TESTABLE model's labels read NOT TESTABLE with the rule's outcome descriptive.  SECONDARY: Hp's FIX label per model (SR 63's "
    "replication); Hb - Hp on N1+N2, on K1+K2 and on K2 (PUT-BACK COSTS iff hi < 0; else NO CLEAR DIFFERENCE or GAINS iff lo > 0).  "
    "Descriptive: put-backs per task (fired, done, aborted, steps, the mug's displacement from its start and tilt per the logged "
    "oracle), the held mug at the next onset after a put-back, episodes reaching the cap, episode length, perception accuracy (SR 63's "
    "accuracy() on Hp and Hb side files), the final plate of each mug.  C1: the reference planner (s2.plan_episode, unchanged) on the "
    "main states; MET iff its N1-N2 success >= .80.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def parse_named(language):
    """The mug names an instruction contains (longest phrase first; each match is removed before the next phrase is searched)."""
    s, out = " " + language.lower() + " ", set()
    for m, ph in NAME_PHRASES:
        pat = r"\b" + re.escape(ph) + r"\b"
        if re.search(pat, s):
            out.add(m)
            s = re.sub(pat, " ", s)
    return out


# ---------------------------------------------------------------- the put-back (client side; proprioception only)
class PutBack:
    """Scripted put-back of the held mug to the point where it was grasped.  Reads the hand pose only."""

    def __init__(self, gp, rng, ee, rot):
        self.gp = np.asarray(gp, np.float64).copy()
        self.rng, self.rot = rng, np.asarray(rot)
        self.zc = max(float(ee[2]), float(self.gp[2]) + PB_CLEAR)
        self.phase, self.in_phase, self.off, self.up, self.err = 0, 0, 0, None, 1.0
        self.hold_wp = None
        self.log = []

    def action(self, ee, ee_rot):
        from fa1g_plan import ctrl
        s = dict(ee_pos=np.asarray(ee, np.float64), ee_rot=ee_rot)
        ph = self.phase
        if ph == 0:
            wp, g = np.array([self.gp[0], self.gp[1], self.zc]), 1.0
        elif ph == 1:
            wp, g = self.gp + np.array([0.0, 0.0, PB_DZ]), 1.0
            self.hold_wp = wp
        elif ph == 2:
            wp, g = self.hold_wp, -1.0
        else:
            wp, g = self.up, -1.0
        a, err = ctrl(s, wp, self.rot, g, self.rng)
        self.err = err
        return a

    def advance(self, det_on, ee):
        """After the env step.  -> 'done' (released and retreated), 'abort' (dropped), or None."""
        ph = self.phase
        self.in_phase += 1
        if ph <= 1:
            self.off = 0 if det_on else self.off + 1
            if self.off >= OFF_ABORT:
                self.log.append(dict(phase=ph, steps=self.in_phase, err=round(float(self.err), 4), abort=True))
                return "abort"
        nxt = False
        if ph == 0:
            nxt = self.err < 0.015 or self.in_phase >= PB_CARRY
        elif ph == 1:
            nxt = self.err < 0.008 or self.in_phase >= PB_DESC
        elif ph == 2:
            nxt = self.in_phase >= OPEN_STEPS
            if nxt:
                self.up = np.asarray(ee, np.float64) + [0, 0, PB_UP]
        elif ph == 3:
            nxt = self.err < 0.015 or self.in_phase >= PB_RETREAT
        if nxt:
            self.log.append(dict(phase=ph, steps=self.in_phase, err=round(float(self.err), 4)))
            if ph == 3:
                return "done"
            self.phase, self.in_phase = ph + 1, 0
        return None


# ---------------------------------------------------------------- states (LIBERO venv, EGL)
def states(a):
    import s2_tasks as T2
    od = os.path.join(CD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    for t in M_["tasks"]:
        print("[s2fix4 states] parse %s %r -> NAMED %s (object %s)" % (t["key"], t["language"], sorted(parse_named(t["language"])),
                                                                       t.get("object")), flush=True)
    for t in M_["tasks"][:4]:
        assert parse_named(t["language"]) == {t["object"]}, (t["key"], t["language"])
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
        (("s2cue3", "O"), ("s2cue4", "O"), ("s2cue5", "O"), ("s2fix2", "O"), ("s2fix2", "PD2"), ("s2fix3", "O"))]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("same as an SR 50-63 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    for ph, ks in (("smoke", SMOKE), ("main", MAIN)):
        with open(os.path.join(CD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE + 1000 * k, at=0, variant="s2") for k in ks], fh, indent=1)
    br3 = json.load(open(os.path.join(OUT, "s2fix3", "branches_main.json"), encoding="utf-8"))
    dev = [b for b in br3 if int(b["trial"]) in DEV]
    assert [int(b["trial"]) for b in dev] == DEV
    with open(os.path.join(CD, "branches_dev.json"), "w", encoding="utf-8") as fh:
        json.dump(dev, fh, indent=1)
    print("[s2fix4 states] %d new states (seed %d + k), none equal to an SR 50-63 state; wrote %s (dev branches: SR 63's k %d-%d)" % (
        len(S), BASE, CD, DEV[0], DEV[-1]), flush=True)


# ---------------------------------------------------------------- client: arm Hb (LIBERO venv)
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
    assert arm == "Hb", arm
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
    NAMED = parse_named(TASK["language"])
    assert NAMED == {TASK["object"]}, (NAMED, TASK["object"])
    ms = int(TASK["max_steps"])
    assert "--max-steps" not in sys.argv
    sys.argv += ["--max-steps", str(ms)]
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    KP = json.load(open(os.path.join(s2.SD, "keypoints.json"), encoding="utf-8"))["tasks"]
    KT = KP[SF.HALF_OF_PLATE[TASK["target"]]]
    assert KT["target"] == TASK["target"], (KT["target"], TASK["target"])
    PER = F3.Percept()
    ORIG, RECS, SIDE = LC.OffScreenRenderEnv, [], []
    SIDEF = OUTF.replace(".json", "_h.json")

    def rule(m, n_pb):
        if m is not None and m in NAMED:
            return "place"
        if m is not None and n_pb < PB_CAP:
            return "put back"
        return "vla: cap reached" if m is not None else "vla: other mug"

    class HEnv(s2.Rec2):
        def __init__(self, env):
            super().__init__(env, TASK["object"], TASK["target"])
            self.det, self.pl, self.pb = None, None, None

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            i = len(RECS)
            self.k = int(br[i]["trial"]) if i < len(br) else None
            self.sb = int(br[i]["seed_base"]) if i < len(br) else 0
            self.rng = np.random.default_rng(self.sb + 20)                 # SR 54's P rng rule
            self.rng_pb = np.random.default_rng(self.sb + 30)
            self.det = ProprioHeld(X["tw"], X["lift"])
            self.pl, self.pb, self.obs = None, None, obs
            self.h = dict(onsets=[], events=[], handovers=[], aborts=[], committed=None, p_steps=0, phase_max=-1,
                          putbacks=[], pb_steps=0, n_pb=0)
            self.reg, self.mem, self.tgt_p, self.gr, self.start = None, {}, None, None, None
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

        def _register(self, obs):                                           # Hp's registration, unchanged
            sc = PER.scene(self._e, obs)
            self.mem = {m: v["pos"] for m, v in sc["mugs"].items()}
            side = F3.side_word(TASK["target"])
            self.tgt_p = sc[side]
            tp, op = self._pos(TASK["target"]), self._pos([p for p in s2.PLATES if p != TASK["target"]][0])
            orc = dict(sim=self._sim_all())
            if self.tgt_p is not None:
                e = np.asarray(self.tgt_p) - tp
                orc.update(tgt_err_xy_cm=round(100 * float(np.linalg.norm(e[:2])), 2), tgt_err_z_cm=round(100 * float(e[2]), 2),
                           side_ok=bool(np.linalg.norm(np.asarray(self.tgt_p)[:2] - tp[:2]) < np.linalg.norm(np.asarray(self.tgt_p)[:2] - op[:2])))
            orc["mug_ok"] = {m.split("_")[0]: bool(min(s2.MUGS, key=lambda x: np.linalg.norm(self._pos(x)[:2] - np.asarray(p)[:2])) == m)
                             for m, p in self.mem.items()}
            self.reg = dict(row=len(self.rows["held"]) - 1, scene=sc, side=side, target=self.tgt_p, oracle=orc)
            self.start = {m: self._pos(m) for m in s2.MUGS}                 # oracle log only

        def step(self, a):
            ee0 = np.asarray(self.obs["robot0_eef_pos"], np.float64)
            rot0 = T2.quat_mat(np.asarray(self.obs["robot0_eef_quat"], np.float64))
            if self.pl is not None:
                a = self.pl.action(ee0, rot0, self._tgt()).tolist()
                self.h["p_steps"] += 1
            elif self.pb is not None:
                a = self.pb.action(ee0, rot0).tolist()
                self.h["pb_steps"] += 1
            r = super().step(a)
            self.obs = r[0]
            if self.n < s2.WAIT:
                return r
            if self.reg is None:
                self._register(r[0])
            row = len(self.rows["held"]) - 1
            ee = np.asarray(r[0]["robot0_eef_pos"], np.float64)
            if float(a[6]) > 0:                                             # the grasp point: the hand at its lowest since the close
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
            elif self.pb is not None:
                res = self.pb.advance(on, ee)
                if res is not None:
                    pb = self.h["putbacks"][-1]
                    sm = pb["sim_mug"]
                    pb.update(row_end=row, result=res, log=self.pb.log, ee_end=np.round(ee, 4).tolist())
                    if sm is not None:
                        d = self._pos(sm) - self.start[sm]
                        pb.update(sim_disp_xy_cm=round(100 * float(np.linalg.norm(d[:2])), 2), sim_dz_cm=round(100 * float(d[2]), 2),
                                  sim_tilt_deg=round(F3.tilt_deg(self._quat(sm)), 1),
                                  sim_on_plate={p: int(bool(self._e.env._eval_predicate(["on", sm, p]))) for p in s2.PLATES})
                    self.pb = None
            elif on and not was and self.h["committed"] is None:
                sh, sd = self._sim_held(ee)
                ev = dict(row=row, oracle_held=bool(self.rows["held"][-1]), n_pb=self.h["n_pb"])
                sc = PER.scene(self._e, r[0])
                self.mem.update({m: v["pos"] for m, v in sc["mugs"].items()})
                held, dd = self._per_held(ee)
                ev.update(detected={m.split("_")[0]: dict(pos=np.round(v["pos"], 4).tolist(), score=v["score"])
                                    for m, v in sc["mugs"].items()})
                ev.update(held=held, dist={m.split("_")[0]: round(v, 4) for m, v in dd.items()},
                          sim_held=sh, sim_dist={m.split("_")[0]: round(v, 4) for m, v in sd.items()},
                          oracle_decision=rule(sh, self.h["n_pb"]))
                dec = rule(held, self.h["n_pb"])
                if dec == "place" and self.tgt_p is None:
                    dec = "vla: no target"
                ev["decision"] = dec
                if dec == "place":
                    pz = self._tgt()
                    self.pl = SF.Place(KT, self.rng, ee, T2.quat_mat(np.asarray(r[0]["robot0_eef_quat"], np.float64)), float(pz[2]))
                    self.h["handovers"].append(dict(row=row, oracle_held=ev["oracle_held"]))
                elif dec == "put back":
                    gp = self.gr if self.gr is not None else ee
                    self.pb = PutBack(gp, self.rng_pb, ee, T2.quat_mat(np.asarray(r[0]["robot0_eef_quat"], np.float64)))
                    self.h["n_pb"] += 1
                    self.h["putbacks"].append(dict(row=row, mug=held, sim_mug=sh, grasp_pt=np.round(gp, 4).tolist(),
                                                   onset_ee=np.round(ee, 4).tolist()))
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
                rec["pb_active_at_end"] = self.pb is not None
                SIDE.append(rec)
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, layout="O", tid=TID, key=TASK["key"], language=TASK["language"], named=sorted(NAMED),
                                   pb_cap=PB_CAP, lex=X, kt=KT, held_r=HELD_R, cal=F3.CAL, side_rule=F3.SIDE_RULE, queries=F3.QUERIES,
                                   detector=PER.model, place="SR 54 object-blind P (PERCEIVED target position)",
                                   putback="scripted, to the grasp point (proprioception)", eps=SIDE),
                              fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
                os.replace(tmp, SIDEF)
                print("[s2fix4] %s %s trial %s: reg %s | events %s | putbacks %s | handovers %s aborts %d committed %s phase_max %d "
                      "p_steps %d pb_steps %d" % (
                          arm, TASK["key"], rec["trial"], None if self.reg is None else self.reg["oracle"].get("tgt_err_xy_cm"),
                          [(e["row"], e["held"] and e["held"].split("_")[0], e["sim_held"] and e["sim_held"].split("_")[0], e["decision"])
                           for e in rec["events"][:6]],
                          [(p["row"], p.get("result"), p.get("sim_disp_xy_cm"), p.get("sim_tilt_deg")) for p in rec["putbacks"]],
                          [h["row"] for h in rec["handovers"]], len(rec["aborts"]), rec["committed"], rec["phase_max"], rec["p_steps"],
                          rec["pb_steps"]), flush=True)
                self.pl, self.pb = None, None
            self._e.close()

    LC.OffScreenRenderEnv = lambda *a, **k: HEnv(ORIG(*a, **k))
    print("[s2fix4] arm %s task %s %r object %s NAMED %s target %s max_steps %d lex %s held_r %.2f detector %s cal %s side %s pb_cap %d" % (
        arm, TASK["key"], TASK["language"], TASK["object"], sorted(NAMED), TASK["target"], ms, (X["tw"], X["lift"]), HELD_R, PER.model,
        F3.CAL, F3.SIDE_RULE, PB_CAP), flush=True)
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


def putback_stats(run, RD, model, ks):
    """Hb side files -> per task: put-backs fired / done / aborted, steps, the logged oracle's displacement and tilt, the held mug
    (oracle) at the next onset after a put-back, episodes at the cap, success given >= 1 put-back, episode length."""
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    out = {}
    for c, H in run["side"].items():
        if H is None:
            continue
        rr, o = run["rows"][c], T[c]["object"]
        pbs = [p for h in H for p in h.get("putbacks", [])]
        nxt = {}
        for h in H:
            for p in h.get("putbacks", []):
                later = [e for e in h["events"] if e["row"] > p.get("row_end", 10 ** 9)]
                key = "none" if not later else ("named" if later[0]["sim_held"] == o else
                                                ("same" if later[0]["sim_held"] == p["sim_mug"] else str(later[0]["sim_held"])))
                nxt[key] = nxt.get(key, 0) + 1
        st = ep_steps(RD, model, "Hb", T[c]["tid"])
        with_pb = [i for i, h in enumerate(H) if h.get("putbacks")]
        disp = [p["sim_disp_xy_cm"] for p in pbs if p.get("sim_disp_xy_cm") is not None]
        out[c] = dict(
            episodes_with_putback=len(with_pb), putbacks=len(pbs),
            results={x: int(sum(p.get("result") == x for p in pbs)) for x in ("done", "abort", None)},
            sim_mug={str(m and m.split("_")[0]): int(sum(p["sim_mug"] == m for p in pbs)) for m in (W, Y, R, None)},
            perceived_mug={str(m and m.split("_")[0]): int(sum(p["mug"] == m for p in pbs)) for m in (W, Y, R)},
            steps_median=float(np.median([p["row_end"] - p["row"] for p in pbs if p.get("row_end") is not None])) if pbs else None,
            disp_xy_cm_median=float(np.median(disp)) if disp else None, disp_xy_cm_max=float(np.max(disp)) if disp else None,
            tipped=int(sum((p.get("sim_tilt_deg") or 0) > 45 for p in pbs)),
            on_a_plate_after=int(sum(any((p.get("sim_on_plate") or {}).values()) for p in pbs)),
            next_onset_held=nxt, cap_reached=int(sum(h.get("n_pb", 0) >= PB_CAP for h in H)),
            success_given_putback=float(np.mean([rr[i]["success"] for i in with_pb])) if with_pb else None,
            success_given_no_putback=(float(np.mean([rr[i]["success"] for i in range(len(H)) if i not in with_pb]))
                                      if len(with_pb) < len(H) else None),
            episodes_with_place=int(sum(len(h["handovers"]) > 0 for h in H)),
            decisions={x: int(sum(e["decision"] == x for h in H for e in h["events"])) for x in
                       ("place", "put back", "vla: other mug", "vla: cap reached", "vla: no target")},
            steps_median_episode=float(np.median(st)), timed_out=int(sum(s >= 400 and not r["success"] for s, r in zip(st, rr))))
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
                    pb = None if h is None else [(p["row"], p.get("row_end"), p.get("result"), p.get("sim_disp_xy_cm"), p.get("sim_tilt_deg"),
                                                  p.get("sim_on_plate")) for p in h.get("putbacks", [])]
                    print("[s2fix4 summary] %s %s %s k%d success %s steps %s bound %s on %s | events %s | putbacks %s | phase_max %s" % (
                        model, arm, t["key"], k, r["success"], e["steps"], r["bound"], on, ev, pb, None if h is None else h["phase_max"]),
                        flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_") and x.endswith(".json")):
        Rj = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2fix4 summary] %s: %s" % (f, [(e["trial"], e["success"]) for e in Rj["eps"]]), flush=True)


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
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_fix4"]
        assert len(rows) == 1, "expected exactly one s2_fix4 prereg row, found %d" % len(rows)

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
            for arm in ("Hp", "Hb"):
                if not ok.get(arm):
                    continue
                dN, dK = dd(arm, "I", s2.NEW), dd(arm, "I", s2.KNOWN)
                R_["deltas"]["%s-I N" % arm], R_["deltas"]["%s-I K" % arm] = dN, dK
                lab = F3.fix_label(dN, dK)
                R_["verdicts"]["%s FIX label (rule)" % arm] = lab
                R_["verdicts"]["%s FIX label (registered)" % arm] = gate(lab, tst)
                R_["verdicts"]["%s-I K non-inferiority" % arm] = F3.ni_label(dK)
                R_["deltas"]["%s-I per task" % arm] = {c: s2.boot2_states(runs[arm]["succ"][c][None], runs["I"]["succ"][c][None]) for c in ALLT}
                if arm == "Hb":
                    xN = float(F3.mat(runs["Hb"], s2.NEW).mean())
                    bar = "MEETS BAR" if (dN[1] > 0 and xN >= BAR) else "BELOW BAR"
                    R_["verdicts"]["Hb BAR (rule)"] = bar
                    R_["verdicts"]["Hb BAR (registered)"] = gate(bar, tst)
                    R_["bar"] = dict(Hb_new=xN, bar=BAR, sr54_rule=F3.sr54_rule(dN, xN))
            if ok.get("Hb") and ok.get("Hp"):
                d2 = s2.boot2_states(runs["Hb"]["succ"]["N2"][None], runs["Hp"]["succ"]["N2"][None])
                R_["deltas"]["Hb-Hp N2"] = d2
                lab = F3.diff_label(d2, "PUT-BACK HELPS", "PUT-BACK HURTS")
                R_["verdicts"]["Hb-Hp N2 PUT-BACK (rule)"] = lab
                R_["verdicts"]["Hb-Hp N2 PUT-BACK (registered)"] = gate(lab, tst)
                for nm, tasks in (("N", s2.NEW), ("K", s2.KNOWN), ("K2", ("K2",)), ("all", ALLT)):
                    d_ = dd("Hb", "Hp", list(tasks))
                    R_["deltas"]["Hb-Hp %s" % nm] = d_
                    R_["verdicts"]["Hb-Hp %s (secondary)" % nm] = F3.diff_label(d_, "PUT-BACK GAINS", "PUT-BACK COSTS")
                R_["deltas"]["Hb-Hp per task"] = {c: s2.boot2_states(runs["Hb"]["succ"][c][None], runs["Hp"]["succ"][c][None]) for c in ALLT}
        if ok.get("Hb"):
            R_["putback"] = putback_stats(runs["Hb"], RD, model, ks)
            print("== %s Hb put-back: %s" % (model, json.dumps(R_["putback"])), flush=True)
        for arm in ("Hp", "Hb"):
            if ok.get(arm):
                R_["perception_%s" % arm] = F3.accuracy(runs[arm], ks)
                print("== %s %s perception accuracy (vs simulator oracle): %s" % (model, arm, json.dumps(R_["perception_%s" % arm])), flush=True)
        for arm in arms:
            if ok.get(arm):
                R_.setdefault("steps", {})[arm] = {c: float(np.median(ep_steps(RD, model, arm, T_["tid"])))
                                                   for c, T_ in ((t["key"], t) for t in s2.meta()["tasks"][:4])}
        for k_, v in R_["deltas"].items():
            print("   %s %s %s" % (model, k_, json.dumps(v if not isinstance(v, list) else np.round(v, 3).tolist())), flush=True)
        print("   VERDICTS %s: %s | testable %s | bar %s | median steps %s" % (model, json.dumps(R_["verdicts"]), json.dumps(R_.get("testable")),
                                                                         json.dumps(R_.get("bar")), json.dumps(R_.get("steps"))), flush=True)
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
    print("[s2fix4 read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("states", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    if A.mode == "read" and A.out is None:
        A.out = READ
    dict(states=states, ref=ref, summary=lambda a: summary(a.dir), read=read)[A.mode](A)

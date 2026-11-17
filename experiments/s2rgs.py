"""S2RGS (docs/SCAFFOLD_RESULTS.md 72; m3_prereg tag s2_rgs): which pose source carries the scene-2 N1 gap of ReGuide?  SR 68's ReGuide port
(experiments/s2rg.py; rg.py's controller class, UNCHANGED) runs on perceived poses (rgp: OWLv2 + rendered depth + camera calibration) or on
simulator poses (rgo); N1 is .417 against .833 (SR 69, SR 68).  SR 69 found the rigid held-mug estimate 5.7 cm off at the N1 release, and a
box tracker did not help, so "the N1 residual is the perceived held-mug carry" is only consistent with the data.  This probe swaps ONE
pose source at a time, on frozen pi0.5, NEW states, layout O, tasks K1 K2 N1 N2.

Facts (SR 50's task table): K1 white -> left (plate_1), K2 yellow-and-white -> right (plate_2), N1 white -> right, N2 yellow-and-white
-> left.

THE ARMS (everything not named here is s2rg.py's client, line for line: entities, round declaration, statistics
results/flawed/s2rg/rg_stats_s2.json, registration at the first gated step, re-detection at ProprioHeld onsets, Hp's held-mug identity rule
(nearest perceived mug within HELD_R = .08 m of the hand), the interaction state "the instructed mug is held", the hand-back handling; prompt
and images never changed):
  - rgp   perception everywhere (SR 68's arm): OWLv2 sidecar + rendered depth + camera calibration (s2fix3.Percept, constants unchanged).
          While attached, the instructed mug follows the hand rigidly from the offset fixed at the attachment.
  - rgo   simulator everywhere (SR 68's privileged arm): simulator body positions of every mug and plate at every step; rg.py's oracle
          interaction rule on the instructed mug.
  - rgoH  rgp, but the HELD-mug pose comes from the simulator: from the attachment step (perceived: ProprioHeld onset + the perceived
          identity rule, as rgp) until the release, the instructed mug's position is the simulator body position; at the release the
          mug's remembered position is the simulator position (rgp: where the hand left it by the rigid estimate).  The target plate,
          the other plate, the resting mugs (including the instructed mug before the attachment) and the hold events are perceived.
  - rgoT  rgp, but the TARGET-plate pose comes from the simulator (the instructed destination's body position, set at the registration);
          the held mug (rigid estimate), the other plate, all mugs and the hold events are perceived.  The target plate is then always
          "perceived", so ReGuide is active in episodes where rgp could not find the plate (the pose source includes existence).
  - The rigid perceived held-mug estimate is also LOGGED at each release of rgoH (never fed), and every arm logs the simulator's mug
    position, as SR 69's mechanism rows.
END ARMS.

New states: LIBERO's sampler on K1's BDDL, seeds 1000000 + k, k 0-59 (none equal to an SR 50-69 state); branches seed_base 49000000 +
1000 k.  Smoke on SR 68's smoke states (results/flawed/s2rg/O, k 0-3; development).  Servers: SR 63's.

Modes (LIBERO venv, py3.8):
    states                                (EGL)   -> results/flawed/s2rgs/O/s2_living_<tid>.npy, branches_main.json
    client --arm rgp|rgo|rgoH|rgoT <s2.py client args>  (rgp, rgoH, rgoT need the OWLv2 sidecar on $S2FIX3_DET_PORT)
    ref --tid T --branches B --out-dir D  (CPU, C1)
    summary DIR                           (smoke)
    read [DIR --selftest --ks 0,1,2,3 --out X]  (CPU)
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import s2                                                                   # noqa: E402
import s2cue                                                                # noqa: E402
import s2fix3 as F3                                                         # noqa: E402
import s2fix4 as F4                                                         # noqa: E402
import s2fix6 as F6                                                         # noqa: E402
import s2rg as SRG                                                          # noqa: E402
import s2rgt as RT                                                          # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2rgs")
ROLL = os.path.join(OUT, "s2rgs_roll")
READ = os.path.join(OUT, "s2rgs_read.json")
STATS = SRG.STATS                                                           # SR 68's statistics, unchanged
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
MUGS, PLATES = s2.MUGS, s2.PLATES
BASE, BR_BASE, N_STATES = 1000000, 49000000, 60
MAIN = list(range(0, 60))
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
HELD_R = F3.HELD_R
NI_MARGIN = F3.NI_MARGIN
FLOOR = F3.FLOOR
BAR = 0.50
EQ_BAND = 0.10
ARMS = ("I", "rgp", "rgo", "rgoH", "rgoT")
RUNS = dict(pi05=ARMS)
PRIMARY_MODEL = "pi05"
SIDEARMS = ("rgp", "rgo", "rgoH", "rgoT")
MAPPING = (
    "NEW states (LIBERO sampler on K1's BDDL, seeds 1000000 + k, k = 0-59, none equal to an SR 50-69 state), branches seed_base "
    "49000000 + 1000 k; layout O; tasks K1, K2, N1, N2; client seed 7, --bind track.  Model: frozen pi0.5 only (pp_serve_purify --model "
    "pi05 --mode intact --comp= --seed 13).  Arms: I (s2.py client), rgp (SR 68's arm: ReGuide on OWLv2 + rendered depth + calibration), "
    "rgo (simulator poses everywhere, PRIVILEGED), rgoH (rgp with the HELD-mug pose from the simulator from the attachment to the release), "
    "rgoT (rgp with the TARGET-plate pose from the simulator).  Success = the BDDL goal.  Matrices [task, state].  Paired bootstrap over "
    "states, tasks fixed (s2.boot2_states, B = 2000, numpy default_rng(0); CI = 2.5 / 97.5).  REGISTERED CONTRASTS, each on N1 (primary) "
    "and on N1+N2: (V1) rgoH - rgp: HELD POSE CARRIES iff lo > 0; HELD POSE LOWERS iff hi < 0; else NO CLEAR DIFFERENCE.  (V2) rgoT - "
    "rgp: TARGET POSE CARRIES iff lo > 0; TARGET POSE LOWERS iff hi < 0; else NO CLEAR DIFFERENCE.  (V3) rgo - rgoH: REMAINDER IN OTHER "
    "POSES iff lo > 0; rgo BELOW rgoH iff hi < 0; else NO CLEAR DIFFERENCE, with EQUIVALENT (inside +-.10) reported beside it.  K1+K2 "
    "non-inferiority at -.05 against I for each of rgp, rgo, rgoH, rgoT (NON-INFERIOR iff lo > -.05; INFERIOR iff hi < -.05; else NOT "
    "SHOWN).  Prediction direction (pre-registered): if the held-mug carry accounts for the N1 gap, V1 > 0 (CARRIES) and V3 ~ 0.  SR 50's "
    "floor: the model is TESTABLE iff I's K1+K2 >= .70 on these states; otherwise labels read NOT TESTABLE.  SECONDARY: each perceiving "
    "or oracle arm against I (SR 62-69's FIX WORKS / PARTIAL / FAILS rule, the .50 BAR); rgo - rgp on N1 and N1+N2 (replicates the SR 68 "
    "gap); replication of rgp (SR 69: N1 .417) and of rgo (SR 68: N1 .833).  MECHANISM (descriptive): per arm, the share of N1 episodes "
    "whose white mug ends on no plate; at each release of the attached instructed mug, the xy error of the arm's estimate and, for rgoH, of "
    "the rigid perceived estimate (logged, never fed), and the share of estimates and of real mugs within the controller's release region; "
    "registration error of the perceiving arms; ReGuide's transports, episode length, time-outs.  C1: the reference planner on the main "
    "states; MET iff its N1-N2 success >= .80.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


short = SRG.short


def nm(n):
    return n if n in PLATES else short(n)


# ---------------------------------------------------------------- states (EGL)
def states(a):
    import s2_tasks as T2
    od = os.path.join(CD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    for t in M_["tasks"][:4]:
        assert F4.parse_named(t["language"]) == {t["object"]}, (t["key"], t["language"])
        assert F6.parse_plates(t["language"]) == {t["target"]}, (t["key"], t["language"])
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
         ("s2fix5", "O"), ("s2fix6", "O"), ("s2rg", "O"), ("s2rgt", "O"))]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("same as an SR 50-69 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    with open(os.path.join(CD, "branches_main.json"), "w", encoding="utf-8") as fh:
        json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE + 1000 * k, at=0, variant="s2") for k in MAIN], fh, indent=1)
    print("[s2rgs states] %d new states (seed %d + k), none equal to an SR 50-69 state; wrote %s" % (len(S), BASE, CD), flush=True)


# ---------------------------------------------------------------- client: arms rgp / rgo / rgoH / rgoT (LIBERO venv)
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
    assert arm in SIDEARMS, arm
    ORACLE, H_SIM, T_SIM = arm == "rgo", arm == "rgoH", arm == "rgoT"
    import rg as RG
    RG.MOVABLE, RG.DESTS, RG.MASKED = tuple(MUGS), tuple(PLATES), ()       # SR 68's port: this scene's referents and destinations
    import pp_atlas as PA
    import libero_client as LC
    import sc_rv as RV
    import s2_tasks as T2
    from lifelong_ood.policy.parts_flow import ProprioHeld
    M = s2.register_suite(LC)
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = M["tasks"][TID]
    assert TASK["tid"] == TID and len(TASK["atoms"]) == 1
    NAMED, NPL = F4.parse_named(TASK["language"]), F6.parse_plates(TASK["language"])
    assert len(NAMED) == 1 and len(NPL) == 1, (NAMED, NPL)
    OE, DE = list(NAMED)[0], list(NPL)[0]
    assert (OE, DE) == (TASK["object"], TASK["target"]), (OE, DE, TASK)
    ms = int(TASK["max_steps"])
    assert "--max-steps" not in sys.argv
    sys.argv += ["--max-steps", str(ms)]
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    S = json.load(open(STATS, encoding="utf-8"))
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    PER = None if ORACLE else F3.Percept()
    ORIG, RECS, SIDE, CUR = LC.OffScreenRenderEnv, [], [], dict(env=None, plan=None, chunk=None, k=0)
    SIDEF = OUTF.replace(".json", "_rg.json")

    class PlanDeque(collections.deque):                                     # libero_client's action queue, with a handle (rg.py)
        def __init__(self, *a):
            super().__init__(*a)
            CUR["plan"] = self
    LC.collections = types.SimpleNamespace(deque=PlanDeque)

    class RGEnv(s2.Rec2):
        def __init__(self, env):
            super().__init__(env, TASK["object"], TASK["target"])
            self.rg = None
            CUR["env"] = self

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            self.rg, self.obs, self.cmd = RG.ReGuide(S, OE, DE), obs, -1.0
            self.reg, self.mem, self.att, self.rest, self.hid, self.inactive = None, {}, None, None, None, False
            self.plates = {}
            self.det = ProprioHeld(X["tw"], X["lift"])
            self.diag = dict(chunk_len=[], n_q=0, reg=None, onsets=[], attached_rows=[], inactive=False, rel=[])
            return obs

        def _yaw0(self):
            return {n: 0.0 for n in MUGS + PLATES}

        def _sim_held(self, ee):
            d = {m: float(np.linalg.norm(self._pos(m)[:2] - np.asarray(ee, np.float64)[:2])) for m in MUGS}
            m = min(d, key=d.get)
            return m if d[m] < HELD_R else None

        def _per_held(self, ee):
            d = {m: float(np.linalg.norm(np.asarray(p)[:2] - np.asarray(ee, np.float64)[:2])) for m, p in self.mem.items()}
            if not d:
                return None
            m = min(d, key=d.get)
            return m if d[m] < HELD_R else None

        def register(self, obs):
            if ORACLE:                                                      # simulator poses; the plates' positions for the log
                self.plates = {p: np.array(self._pos(p), np.float64) for p in PLATES}
                self.mem = {m: np.array(self._pos(m), np.float64) for m in MUGS}
                self.inactive = False
                self.diag["inactive"] = False
                self.reg = True
                return
            sc = PER.scene(self._e, obs)
            self.mem = {m: np.asarray(v["pos"], np.float64) for m, v in sc["mugs"].items()}
            self.plates = {}
            if sc["left"] is not None:
                self.plates[PL] = np.asarray(sc["left"], np.float64)
            if sc["right"] is not None:
                self.plates[PR] = np.asarray(sc["right"], np.float64)
            err = {nm(n): round(100 * float(np.linalg.norm((p - self._pos(n))[:2])), 3)          # perceived poses against the simulator
                   for n, p in list(self.mem.items()) + list(self.plates.items())}
            zerr = {nm(n): round(100 * float(p[2] - self._pos(n)[2]), 3) for n, p in list(self.mem.items()) + list(self.plates.items())}
            tp, op = self._pos(DE), self._pos([p for p in PLATES if p != DE][0])
            side_ok = None
            if DE in self.plates:
                side_ok = bool(np.linalg.norm(self.plates[DE][:2] - tp[:2]) < np.linalg.norm(self.plates[DE][:2] - op[:2]))
            found = sorted(short(n) for n in list(self.mem) + list(self.plates))
            if T_SIM:                                                       # rgoT: the target plate's pose is the simulator's
                self.plates[DE] = np.array(self._pos(DE), np.float64)
            self.inactive = OE not in self.mem or DE not in self.plates
            self.diag["inactive"] = self.inactive
            self.diag["reg"] = dict(row=len(self.rows["held"]) - 1, err_xy_cm=err, err_z_cm=zerr, found=found,
                                    tgt_err_xy_cm=err.get(nm(DE)), tgt_side_ok=side_ok, obj_err_xy_cm=err.get(nm(OE)),
                                    scores={short(m): v["score"] for m, v in sc["mugs"].items()}, tgt_swapped=bool(T_SIM))
            self.reg = True

        def log_release(self, row, q, rigid):
            sq = self._pos(OE)
            pxy, _ = self.rg.dest_xy(dict(pos=dict(self.plates), yaw=self._yaw0()))
            om = float(self.rg.ds["omega"])
            self.diag["rel"].append(dict(row=row, est=np.round(q, 4).tolist(), sim=np.round(sq, 4).tolist(),
                                         err_xy_cm=round(100 * float(np.linalg.norm((q - sq)[:2])), 3),
                                         err_z_cm=round(100 * float(q[2] - sq[2]), 3),
                                         rigid_err_xy_cm=round(100 * float(np.linalg.norm((rigid - sq)[:2])), 3),
                                         rigid_in=bool(np.linalg.norm(rigid[:2] - pxy) <= om),
                                         est_in=bool(np.linalg.norm(q[:2] - pxy) <= om), sim_in=bool(np.linalg.norm(sq[:2] - pxy) <= om),
                                         sim_to_target_cm=round(100 * float(np.linalg.norm((sq - self._pos(DE))[:2])), 3),
                                         mode=self.rg.mode))

        def state(self):
            obs = self.obs
            ee = np.asarray(obs["robot0_eef_pos"], np.float64)
            Rm = T2.quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64))
            grip = np.asarray(obs["robot0_gripper_qpos"], np.float64)
            row = len(self.rows["held"]) - 1
            was = self.det.on
            on = bool(self.det.step(grip, self.cmd, ee[2]))                  # perceiving arms: the hold event; rgo: logged only
            onset = on and not was
            if ORACLE:
                pos = {n: np.array(self._pos(n), np.float64) for n in MUGS + PLATES}
                o = pos[OE]
                rel = Rm.T @ (o - ee)
                if self.att is None:
                    self.rest = float(o[2]) if self.rest is None else min(self.rest, float(o[2]))
                    if o[2] > self.rest + RG.EFFECT and np.linalg.norm(o - ee) < RG.HOLD_NEAR:
                        self.att = rel
                        self.diag["attached_rows"].append(row)
                elif np.linalg.norm(rel - self.att) > RG.CO_MOTION:
                    self.log_release(row, o.copy(), ee + Rm @ self.att)
                    self.att, self.rest = None, float(o[2])
                held = self.att is not None
                q = o.copy()
                if onset:
                    self.diag["onsets"].append(dict(row=row, sim_held=short(self._sim_held(ee)), held=None, attached=held))
            else:
                if onset:
                    sc = PER.scene(self._e, obs)
                    for m, v in sc["mugs"].items():
                        self.mem[m] = np.asarray(v["pos"], np.float64)
                    self.hid = self._per_held(ee)
                    self.diag["onsets"].append(dict(row=row, sim_held=short(self._sim_held(ee)), held=short(self.hid),
                                                    attached=self.hid == OE))
                if not on:
                    self.hid = None
                held = bool(on and self.hid == OE)
                if held and self.att is None:
                    self.att = Rm.T @ (self.mem[OE] - ee)                    # rigid offset in the hand frame, fixed at the hold onset
                    self.diag["attached_rows"].append(row)
                if not held and self.att is not None:
                    rigid = ee + Rm @ self.att
                    q_ = np.array(self._pos(OE), np.float64) if H_SIM else rigid          # rgoH: the simulator's mug stays where it is
                    self.log_release(row, q_, rigid)
                    self.mem[OE] = q_                                        # the released mug stays where it was left
                    self.att = None
                if self.att is None:
                    q = self.mem[OE]
                elif H_SIM:
                    q = np.array(self._pos(OE), np.float64)                  # rgoH: the held mug's pose is the simulator's
                else:
                    q = ee + Rm @ self.att
                pos = {m: np.array(p, copy=True) for m, p in self.mem.items()}
                pos.update({p: np.array(v, copy=True) for p, v in self.plates.items()})
                pos[OE] = np.array(q, copy=True)
            return dict(ee=ee, R=Rm, width=float(grip[0] - grip[1]), pos=pos, yaw=self._yaw0(), held=held, q=q, cmd=self.cmd)

        def step(self, a):
            if self.n < s2.WAIT or self.rg is None:
                r = super().step(a)
                self.obs = r[0]
                return r
            if self.reg is None:
                self.register(self.obs)
            if self.inactive:                                               # instructed mug or plate not perceived: the policy alone
                r = super().step(a)
                self.obs = r[0]
                return r
            ch = CUR.get("chunk")
            rem = None
            if ch is not None:
                rem = ch[CUR["k"]:]
                CUR["k"] += 1
            s = self.state()
            s["rem"] = rem
            act, hb = self.rg.decide(s, list(a), rem)
            r = super().step(act)
            self.obs, self.cmd = r[0], float(act[6])
            if hb:
                pl = CUR.get("plan")
                if pl is not None:
                    pl.clear()
                CUR["chunk"] = None
            return r

        def close(self):
            if self.n > 0 and self.rg is not None:
                self.finish()
                RECS.append(self.data())
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
                i = len(SIDE)
                held = np.asarray(self.rows["held"], bool)
                rec = self.rg.record()
                rec.update(trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, tid=TID, key=TASK["key"], arm=arm,
                           oracle_first_held=int(np.argmax(held)) if held.any() else None, **self.diag)
                SIDE.append(rec)
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], roles=[OE, DE], stats=STATS,
                                   movable=list(RG.MOVABLE), dests=list(RG.DESTS), masked=list(RG.MASKED), lex=X, held_r=HELD_R,
                                   detector=None if PER is None else PER.model, cal=F3.CAL, side_rule=F3.SIDE_RULE,
                                   held_sim=H_SIM, target_sim=T_SIM, oracle=ORACLE, eps=SIDE),
                              fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
                os.replace(tmp, SIDEF)
                print("[s2rgs] %s %s trial %s: inactive %s reg %s | onsets %s | rel %s | transports %d releases %d steps_rg %d | events %s" % (
                    arm, TASK["key"], rec["trial"], rec["inactive"], None if rec["reg"] is None else rec["reg"].get("tgt_err_xy_cm"),
                    [(o_["row"], o_["sim_held"], o_["held"], o_["attached"]) for o_ in rec["onsets"][:4]],
                    [(r_["row"], r_["err_xy_cm"], r_["rigid_err_xy_cm"], r_["sim_in"]) for r_ in rec["rel"][:4]],
                    rec["transports"], rec["releases"], rec["steps_rg"],
                    [(e["t"], e["kind"], e.get("what"), e.get("why")) for e in rec["events"]][:8]), flush=True)
                self.rg = None
            self._e.close()

    Base = LC._wcp.WebsocketClientPolicy

    class RGClient(Base):
        def infer(self, el):
            env = CUR.get("env")
            if env is not None and env.rg is not None and env.rg.mode is not None:
                CUR["chunk"] = None                                         # ReGuide steers: no query; the env replaces these actions
                return {"actions": np.zeros((RG.REPLAN, 7))}
            out = super().infer(el)
            CUR["chunk"], CUR["k"] = np.asarray(out["actions"], np.float64), 0
            if env is not None and env.rg is not None:
                env.diag["chunk_len"].append(int(len(out["actions"])))
                env.diag["n_q"] += 1
            return out
    LC._wcp.WebsocketClientPolicy = RGClient
    LC.OffScreenRenderEnv = lambda *a, **k: RGEnv(ORIG(*a, **k))
    print("[s2rgs] arm %s held_sim %s target_sim %s oracle %s task %s %r roles %s -> %s max_steps %d stats %s detector %s" % (
        arm, H_SIM, T_SIM, ORACLE, TASK["key"], TASK["language"], OE, DE, ms, STATS, None if PER is None else PER.model), flush=True)
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
    fr = a.fresh or os.path.join(CD, "O")
    S = np.load(os.path.join(fr, "%s_%d.npy" % (s2.SUITE, a.tid)))
    br = {int(b["trial"]): int(b["seed_base"]) for b in json.load(open(a.branches, encoding="utf-8"))}
    kp, eps = s2cue3.ref_eps(S, t, t["target"], sorted(br), lambda k: br[k])
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(layout="O", tid=a.tid, key=t["key"], kp=kp, eps=eps), fh, indent=1)


# ---------------------------------------------------------------- read helpers (CPU)
def lab(d, pos, neg):
    return F3.diff_label(d, pos, neg)


def rel_stats(side, tasks):
    """Release events of the attached instructed mug: the arm's estimate and the rigid perceived estimate against the simulator."""
    E = [r for c in tasks for e in side[c] for r in e.get("rel", [])]
    if not E:
        return dict(n=0)
    x = np.array([r["err_xy_cm"] for r in E])
    g = np.array([r["rigid_err_xy_cm"] for r in E])
    return dict(n=len(E), est_med_cm=float(np.median(x)), est_p90_cm=float(np.percentile(x, 90)),
                rigid_med_cm=float(np.median(g)), rigid_p90_cm=float(np.percentile(g, 90)),
                est_in=float(np.mean([r["est_in"] for r in E])), rigid_in=float(np.mean([r["rigid_in"] for r in E])),
                sim_in=float(np.mean([r["sim_in"] for r in E])),
                rigid_in_sim_out=int(sum(r["rigid_in"] and not r["sim_in"] for r in E)))


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
                side = json.load(open(st + "_rg.json", encoding="utf-8"))["eps"] if os.path.exists(st + "_rg.json") else [None] * len(ks)
                for k, e, r, h in zip(ks, E, s2.stage_rows(st, t, ks), side):
                    on = {"%s>%s" % (m.split("_")[0][:5], p): 1 for m in MUGS for p in PLATES if s2.on_of(r["on"], m, p)}
                    x = None
                    if h is not None:
                        x = dict(onsets=[(o_["row"], o_["sim_held"], o_["held"], o_["attached"]) for o_ in h["onsets"][:4]],
                                 starts=[(ev["t"], ev["what"], ev.get("why")) for ev in h["events"] if ev["kind"] == "start"][:6],
                                 inactive=h.get("inactive"),
                                 reg=None if not h.get("reg") else (h["reg"].get("tgt_err_xy_cm"), h["reg"].get("obj_err_xy_cm")),
                                 rel=[(r_["row"], r_["err_xy_cm"], r_["rigid_err_xy_cm"], r_["est_in"], r_["rigid_in"], r_["sim_in"])
                                      for r_ in h.get("rel", [])][:4])
                    print("[s2rgs summary] %s %s %s k%d success %s steps %s on %s | %s" % (
                        model, arm, t["key"], k, r["success"], e["steps"], on, json.dumps(x)), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_") and x.endswith(".json")):
        Rj = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2rgs summary] %s: %s" % (f, [(e["trial"], e["success"]) for e in Rj["eps"]]), flush=True)


def read(a):
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD, mism = [int(x) for x in a.ks.split(",")], a.dir, "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_rgs"]
        assert len(rows) == 1, "expected exactly one s2_rgs prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD = MAIN, ROLL
    res = dict(states=ks, mapping=MAPPING, code_mismatch=mism, models={})
    ALLT = s2.KNOWN + s2.NEW
    one = np.ones(len(ks))
    for model, arms in RUNS.items():
        runs = {arm: F3.load_run(RD, model, arm, ks) for arm in arms}
        sides = {arm: SRG.load_rg(RD, model, arm, ks) for arm in arms if arm in SIDEARMS}
        ok = {arm: not isinstance(v, str) for arm, v in runs.items()}
        for arm in SIDEARMS:
            if isinstance(sides.get(arm), str):
                ok[arm] = False
        R_ = dict(arms={}, deltas={}, verdicts={}, mechanism={})
        for arm, run in runs.items():
            if not ok[arm]:
                R_["arms"][arm] = run if isinstance(run, str) else sides[arm]
                print("== %s %s: %s" % (model, arm, R_["arms"][arm]), flush=True)
                continue
            R_["arms"][arm] = dict(success={c: float(v.mean()) for c, v in run["succ"].items()},
                                   new=float(F3.mat(run, s2.NEW).mean()), known=float(F3.mat(run, s2.KNOWN).mean()),
                                   plates=SRG.plates_of(run))
            print("== %s %s: per task %s | N1+N2 %.3f | K1+K2 %.3f" % (
                model, arm, {c: round(v, 3) for c, v in R_["arms"][arm]["success"].items()}, R_["arms"][arm]["new"],
                R_["arms"][arm]["known"]), flush=True)
            print("      plates %s" % json.dumps(R_["arms"][arm]["plates"]), flush=True)

        def dd(x, y, tasks):
            return s2.boot2_states(F3.mat(runs[x], list(tasks)), F3.mat(runs[y], list(tasks)))
        tst = None
        if ok.get("I"):
            kI = float(F3.mat(runs["I"], s2.KNOWN).mean())
            tst = bool(kI >= FLOOR)
            R_["testable"] = dict(intact_K=kI, floor=FLOOR, TESTABLE=tst)
            for arm in SIDEARMS:
                if not ok.get(arm):
                    continue
                dN, dK = dd(arm, "I", s2.NEW), dd(arm, "I", s2.KNOWN)
                R_["deltas"]["%s-I N" % arm], R_["deltas"]["%s-I K" % arm] = dN, dK
                flab = F3.fix_label(dN, dK)
                R_["verdicts"]["%s FIX label (registered)" % arm] = SRG.gate(flab, tst)
                R_["verdicts"]["%s-I K non-inferiority (registered)" % arm] = SRG.gate(F3.ni_label(dK), tst)
                xN = float(F3.mat(runs[arm], s2.NEW).mean())
                R_["verdicts"]["%s BAR (registered)" % arm] = SRG.gate("MEETS BAR" if (dN[1] > 0 and xN >= BAR) else "BELOW BAR", tst)
        for name, (x, y, pos, neg) in (("V1 rgoH-rgp", ("rgoH", "rgp", "HELD POSE CARRIES", "HELD POSE LOWERS")),
                                       ("V2 rgoT-rgp", ("rgoT", "rgp", "TARGET POSE CARRIES", "TARGET POSE LOWERS")),
                                       ("V3 rgo-rgoH", ("rgo", "rgoH", "REMAINDER IN OTHER POSES", "rgo BELOW rgoH"))):
            if not (ok.get(x) and ok.get(y)):
                continue
            for tn, tasks in (("N1", ("N1",)), ("N1+N2", s2.NEW)):
                d_ = dd(x, y, tasks)
                R_["deltas"]["%s %s" % (name, tn)] = d_
                eqv = bool(d_[1] >= -EQ_BAND and d_[2] <= EQ_BAND)
                R_["verdicts"]["%s %s (registered)" % (name, tn)] = SRG.gate(lab(d_, pos, neg), bool(tst))
                R_["verdicts"]["%s %s equivalent within +-.10 (descriptive)" % (name, tn)] = eqv
        if ok.get("rgo") and ok.get("rgp"):
            for tn, tasks in (("N1", ("N1",)), ("N1+N2", s2.NEW), ("K1+K2", s2.KNOWN)):
                d_ = dd("rgo", "rgp", tasks)
                R_["deltas"]["rgo-rgp %s" % tn] = d_
                R_["verdicts"]["rgo-rgp %s (secondary)" % tn] = lab(d_, "GAIN", "COST")
        # mechanism (m1): N1 white mug on no plate
        NOP = {}
        for arm in arms:
            if ok.get(arm):
                ind = np.array([float(s2cue.plate_of(r["on"], W) is None) for r in runs[arm]["rows"]["N1"]])
                NOP[arm] = dict(share=s2cue.boot_share(ind, one), count=int(ind.sum()), ind=ind.tolist())
        R_["mechanism"]["N1_white_no_plate"] = NOP
        print("== %s MECHANISM N1 white on no plate: %s" % (model, json.dumps({k_: {kk: vv for kk, vv in v.items() if kk != "ind"}
                                                                                for k_, v in NOP.items()})), flush=True)
        for arm in SIDEARMS:
            if ok.get(arm):
                R_["mechanism"]["release_%s" % arm] = dict(N1=rel_stats(sides[arm], ("N1",)), all=rel_stats(sides[arm], ALLT))
                R_["mechanism"]["reguide_%s" % arm] = SRG.rg_diag(sides[arm], runs[arm])
                print("== %s %s release: %s" % (model, arm, json.dumps(R_["mechanism"]["release_%s" % arm])), flush=True)
                print("== %s %s ReGuide: %s" % (model, arm, json.dumps(R_["mechanism"]["reguide_%s" % arm])), flush=True)
                if arm != "rgo":
                    R_["perception_%s" % arm] = SRG.perception(sides[arm])
                    R_["perception_%s" % arm]["errors"] = RT.reg_err(sides[arm])
                    print("== %s %s perception: %s" % (model, arm, json.dumps(R_["perception_%s" % arm])), flush=True)
        for arm in arms:
            if ok.get(arm):
                st_ = {t["key"]: F6.ep_steps(RD, model, arm, t["tid"]) for t in s2.meta()["tasks"][:4]}
                R_.setdefault("steps", {})[arm] = {c: float(np.median(v)) for c, v in st_.items()}
                R_.setdefault("timed_out", {})[arm] = {c: int(sum(s >= 400 and not r["success"] for s, r in zip(v, runs[arm]["rows"][c])))
                                                      for c, v in st_.items()}
        for k_, v in R_["deltas"].items():
            print("   %s %s %s" % (model, k_, json.dumps(v if not isinstance(v, list) else np.round(v, 3).tolist())), flush=True)
        print("   VERDICTS %s: %s | testable %s | median steps %s | timed out %s" % (
            model, json.dumps(R_["verdicts"]), json.dumps(R_.get("testable")), json.dumps(R_.get("steps")), json.dumps(R_.get("timed_out"))),
            flush=True)
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
    print("[s2rgs read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("states", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--fresh")
    ap.add_argument("--out-dir")
    ap.add_argument("--out")
    ap.add_argument("--ks", default="0,1,2,3")
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    if A.mode == "read" and A.out is None:
        A.out = READ
    dict(states=states, ref=ref, summary=lambda a: summary(a.dir), read=read)[A.mode](A)

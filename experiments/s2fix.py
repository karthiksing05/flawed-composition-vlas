"""S2FIX (docs/SCAFFOLD_RESULTS.md 54, written to docs/sr54_scene2_fix.md; m3_prereg tag scene2_fix): does ANY structural change that
adds no task information fix F1 in scene 2 (LIBERO-10 LIVING_ROOM_SCENE5, SR 50), where every VLA scores 0 of 720 on the new pairings
N1 (white mug -> right plate) and N2 (yellow-and-white mug -> left plate) and the role views (tv / tvp) give exactly 0?

Everything is s2.py's harness (tasks, states k = 4-63, branches, servers, client seed 7, --bind track, Rec2's oracle track), imported
unchanged; s2.py, s2_roll.sh and tv_frozen.py are hashed in the scene2 row and are not edited.

Arm (a), the target-phase view.  Mode `paintcheck` prints exactly what tv paints in this scene per phase (entities, pixels per camera)
from the code path tv runs (sc_rv.role_paint with RV.MOVABLE_OBJ = s2.MOVABLE, sc_rv.seg_px, sc_rv.paint_many).  If the non-target
plate is painted in the target phase, tv already IS the proposed tvx and arm (a) is read off SR 50's tv / tvp rollouts (reused).

Arm (b), the hand-over (HP-style; the proposal's hand-over variant, ARCHITECTURE_PROPOSAL 2; overlaps ReGuide / LiLo-VLA, NOVELTY.md):
    hI    frozen VLA, intact prompt and images, until the role sequencer fires; then the place policy P acts.
    htvp  frozen VLA under tvp (tv_frozen.Front with simulator masks + role phrases, exactly SR 50's tvp client) until the role
          sequencer fires; then P acts.
The role sequencer is ProprioHeld with sc_rv_lex.json's threshold (the tv detector), stepped on every env step from WAIT on the finger
qpos, the EXECUTED gripper command and the hand height.  It reads no object pose, contact or identity.
P (object-blind place policy) reads ONLY: proprioception (hand pose) and the TARGET plate's simulator position (privileged label:
the target role's entity, as the role phrase names it).  It never reads the held object's identity or pose.  Its constants are the
target plate's own fitted keypoints (results/flawed/s2/keypoints.json, the trained half whose target is this plate: carry height above
the plate, hand-minus-plate offset at release), so they are a property of the plate, not of the object.  Controller fa1g_plan.ctrl,
as the SR 50 reference planner.  Phases:
    0 lift     hand straight up to plate_z0 + carry_z, gripper closed, rotation held at the hand-over rotation (<= 40 steps)
               at the end of the lift the sequencer must still be on, else P aborts and the VLA acts again (failed grasp)
    1 carry    above plate + release_off (xy), same height (<= CARRY_STEPS steps)
    2 descend  to plate_z + release_off z (<= 40 steps)
    3 open     12 steps (committed: from here P keeps control to the end)
    4 retreat  10 cm up, then hold.
    During 0-2, if the sequencer is off for 3 consecutive steps (the mug was dropped), P aborts and the VLA acts again; a later onset
    hands over again.  The VLA's server is still queried every 5 steps throughout (its chunks are ignored while P acts).
P's rng: numpy default_rng(seed_base + 20) per episode (the controller's action noise, as the reference).

Modes (LIBERO venv, py3.8, EGL):
    python experiments/s2fix.py paintcheck --out-dir D                  (SR 50's smoke state 0-1, tids 0-3)
    python experiments/s2fix.py client --arm hI|htvp <s2.py client args>
    python experiments/s2fix.py summary DIR
    python experiments/s2fix.py read [--out results/flawed/s2fix_read.json]   (CPU)
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
ROLL = os.path.join(OUT, "s2fix_roll")
READ = os.path.join(OUT, "s2fix_read.json")
WAIT = s2.WAIT
ARMS = ("hI", "htvp")
HALF_OF_PLATE = {"plate_1": "K1", "plate_2": "K2"}                         # the trained half whose target is this plate
CARRY_STEPS = 120
LIFT_LIM, DESC_LIM, OPEN_STEPS, OFF_ABORT = 40, 40, 12, 3
EPS = 0.10
# registered (read) -- reused arms come from SR 50's results/flawed/s2_roll, new arms from results/flawed/s2fix_roll
MODELS = dict(pi05=dict(reused=("I", "tv", "tvp"), new=("hI", "htvp")), groot=dict(reused=("I", "tv"), new=("hI",)))
PRIMARY = ("pi05", "htvp")
MAPPING = (
    "Main states k = 4-63 (60), SR 50's tasks, branches, servers and client flags.  Success matrices [task, state].  Paired "
    "bootstrap B = 2000, numpy default_rng(0): JOINT (tasks and states drawn jointly, tv_frozen.boot2) and STATE-ONLY (tasks fixed, "
    "s2.boot2_states); CI = 2.5 / 97.5.  For every arm X (new: pi0.5 htvp PRIMARY, pi0.5 hI SECONDARY, GR00T hI SECONDARY and "
    "GR00T NOT TESTABLE per SR 50; reused from SR 50: pi0.5 tv, tvp, GR00T tv = arm (a) when paintcheck shows the non-target plate "
    "is painted in the target phase), pooled over N1+N2 with d = X - I (JOINT CI): FIXED iff CI lo > 0 and success(X) on N1+N2 >= "
    ".50; NOT FIXED iff CI hi < .10; else PARTIAL.  Per task (N1, N2 separately; C1 is met only on N2): the same rule with the "
    "state-only CI of X - I on that task and success(X) on that task.  Trained-half cost on K1+K2: X - I (JOINT CI) NO COST iff lo > "
    "-.10, else COST; per task K1, K2 descriptive.  Reused I / tv / tvp are valid iff every file hash recorded in the scene2_fix row "
    "for them still matches (checked in the read; a mismatch voids the reuse and the read refuses).  Descriptive: hand-over fired "
    "share, oracle Held at hand-over, the mug moved first ('bound'), success given the named mug bound, P aborts, P phase reached, "
    "habit placements, final mug-to-target distance, atlas excess per arm, detector agreement with tv's detector (htvp).")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def reused_files():
    """The SR 50 rollouts this read reuses (absolute paths)."""
    M = s2.meta()
    T = {t["key"]: t for t in M["tasks"]}
    out = []
    for m, d in MODELS.items():
        for arm in d["reused"]:
            for c in s2.NEW + s2.KNOWN:
                stem = os.path.join(s2.ROLL, "%s_%s_%d" % (m, arm, T[c]["tid"]))
                out += [stem + ".json", stem + "_track.npz"]
    return out


# ---------------------------------------------------------------- arm (a): what tv paints (LIBERO venv)
def paintcheck(a):
    import imageio
    import sc_rv as RV
    import s2_tasks as T2
    RV.MOVABLE_OBJ = s2.MOVABLE                                             # exactly what s2.client_main sets for tv / tvp
    os.makedirs(a.out_dir, exist_ok=True)
    M = s2.meta()
    rep, sheet = [], []
    for t in M["tasks"][:4]:
        S = np.load(os.path.join(s2.SD, "states", "%s_%d.npy" % (s2.SUITE, t["tid"])))
        env = T2.make_env(os.path.join(ROOT, t["bddl"]))
        env.seed(7)
        for k in (0, 1):
            env.reset()
            obs = env.set_init_state(S[k])
            for _ in range(WAIT):
                obs, _, _, _ = env.step(T2.DUMMY)
            E = RV.entity_tables(env)
            names = list(E["names"])
            inner = env.env
            row = dict(task=t["key"], object=t["object"], target=t["target"], state=k, names=names,
                       objects=list(inner.objects_dict.keys()), fixtures=list(inner.fixtures_dict.keys()), phases={})
            imgs = []
            for held in (False, True):
                ents, act = RV.role_paint(t["object"], t["target"], held)
                ids = [names.index(e) for e in ents if e in names]
                ph = dict(painted=ents, painted_in_names=[e for e in ents if e in names], active=act, cams={})
                for cam, key in zip(RV.CAMS, ("agentview_image", "robot0_eye_in_hand_image")):
                    raw = np.ascontiguousarray(np.asarray(obs[key])[::-1, ::-1])
                    g = np.asarray(RV.seg_px(env, E, cam))
                    img, n = RV.paint_many(raw, g, ids)
                    px = {nm: int((g == names.index(nm)).sum()) for nm in s2.MOVABLE if nm in names}
                    mag = np.all(img == RV.MAGENTA, -1)
                    ph["cams"][cam] = dict(total_painted=n, entity_px=px,
                                           painted_px={nm: int((mag & (g == names.index(nm))).sum()) for nm in s2.MOVABLE if nm in names})
                    if k == 0:
                        imgs += [raw, img] if not held else [img]
                ph["phase"] = "target" if held else "object"
                row["phases"]["target" if held else "object"] = ph
            rep.append(row)
            if k == 0:
                sheet.append(np.concatenate(imgs, 1))
            print("[paintcheck] %s state %d object %s target %s | objects %s fixtures %s" % (
                t["key"], k, t["object"], t["target"], row["objects"], row["fixtures"]), flush=True)
            for p, ph in row["phases"].items():
                print("   %s phase: painted %s (active, unpainted: %s)" % (p, ph["painted"], ph["active"]), flush=True)
                for cam, c in ph["cams"].items():
                    print("      %s painted px per entity %s of entity px %s" % (cam, c["painted_px"], c["entity_px"]), flush=True)
        env.close()
    imageio.imwrite(os.path.join(a.out_dir, "paint_sheet.png"), np.concatenate(sheet, 0))
    with open(os.path.join(a.out_dir, "paintcheck.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    def other_plate_px(r, cam):
        op = [p for p in s2.PLATES if p != r["target"]][0]
        return r["phases"]["target"]["cams"][cam]["painted_px"].get(op, 0)
    tp = all(other_plate_px(r, "agentview") > 0 for r in rep)
    tt = all(r["phases"]["target"]["cams"][c]["painted_px"].get(r["target"], 0) == 0 for r in rep for c in r["phases"]["target"]["cams"])
    print("[paintcheck] non-target plate painted (agentview) in the target phase on every task/state: %s; target plate never "
          "painted in the target phase: %s" % (tp, tt), flush=True)


# ---------------------------------------------------------------- arm (b): the place policy (LIBERO venv)
class Place:
    """Object-blind place policy: hand pose + the target plate's simulator position only."""

    def __init__(self, kt, rng, ee, rot, pz0):
        self.kt, self.rng = kt, rng
        self.cz = float(kt["carry_z_above_tgt"])
        self.roff = np.asarray(kt["release_off"], np.float64)
        self.rot, self.x0, self.z_carry = np.asarray(rot), np.asarray(ee, np.float64).copy(), pz0 + self.cz
        self.phase, self.in_phase, self.off, self.up = 0, 0, 0, None
        self.log = dict(phases=[], at_open=None)

    def action(self, ee, ee_rot, pt):
        from fa1g_plan import ctrl
        s = dict(ee_pos=np.asarray(ee, np.float64), ee_rot=ee_rot)
        ph = self.phase
        if ph == 0:
            wp, g = np.array([self.x0[0], self.x0[1], self.z_carry]), 1.0
        elif ph == 1:
            wp, g = np.array([pt[0] + self.roff[0], pt[1] + self.roff[1], self.z_carry]), 1.0
        elif ph == 2:
            wp, g = np.array([pt[0] + self.roff[0], pt[1] + self.roff[1], pt[2] + self.roff[2]]), 1.0
        elif ph == 3:
            wp, g = self.hold_wp, -1.0
        else:
            wp, g = self.up, -1.0
        if ph == 2:
            self.hold_wp = wp
        a, err = ctrl(s, wp, self.rot, g, self.rng)
        self.err = err
        return a

    def advance(self, det_on, ee):
        """After the env step.  -> 'abort' (control back to the VLA) or None."""
        ph = self.phase
        self.in_phase += 1
        if ph <= 2:
            self.off = 0 if det_on else self.off + 1
            if self.off >= OFF_ABORT:
                return "abort"
        nxt = False
        if ph == 0:
            nxt = self.err < 0.015 or self.in_phase >= LIFT_LIM
            if nxt and not det_on:
                return "abort"
        elif ph == 1:
            nxt = self.err < 0.015 or self.in_phase >= CARRY_STEPS
        elif ph == 2:
            nxt = self.err < 0.008 or self.in_phase >= DESC_LIM
        elif ph == 3:
            nxt = self.in_phase >= OPEN_STEPS
            if nxt:
                self.up = np.asarray(ee, np.float64) + [0, 0, 0.10]
        elif ph == 4:
            nxt = False
        if nxt:
            self.log["phases"].append(dict(phase=ph, steps=self.in_phase, err=round(float(self.err), 4)))
            self.phase, self.in_phase = ph + 1, 0
        return None


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
    assert arm in ARMS, arm
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
    ms = int(TASK["max_steps"])
    assert "--max-steps" not in sys.argv
    sys.argv += ["--max-steps", str(ms)]
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    KP = json.load(open(os.path.join(s2.SD, "keypoints.json"), encoding="utf-8"))["tasks"]
    KT = KP[HALF_OF_PLATE[TASK["target"]]]
    assert KT["target"] == TASK["target"], (KT["target"], TASK["target"])
    Front, P = None, None
    if arm == "htvp":
        import tv_frozen as TF
        RV.MOVABLE_OBJ = s2.MOVABLE
        Front = TF.Front
        P = list(TASK["phrases"])
    ORIG, RECS, SIDE, CUR = LC.OffScreenRenderEnv, [], [], {}
    SIDEF = OUTF.replace(".json", "_h.json")

    class HEnv(s2.Rec2):
        def __init__(self, env):
            super().__init__(env, TASK["object"], TASK["target"])
            self.front, self.det, self.pl = None, None, None
            CUR["env"] = self

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            i = len(RECS)
            self.sb = int(br[i]["seed_base"]) if i < len(br) else 0
            self.rng = np.random.default_rng(self.sb + 20)
            self.det = ProprioHeld(X["tw"], X["lift"])
            self.pl, self.obs = None, obs
            self.h = dict(onsets=[], handovers=[], aborts=[], committed=None, disagree=0, p_steps=0, phase_max=-1)
            if Front is not None:
                self.front = Front(self._e, "tv", TASK["object"], TASK["target"], X, s2.SimSeg(self._e))
                self.front.prompts = dict(object=0, target=0)
            return obs

        def _tgt(self):
            E = self._e.env
            return np.array(E.sim.data.body_xpos[E.obj_body_id[TASK["target"]]], np.float64)

        def step(self, a):
            if self.pl is not None:
                ee = np.asarray(self.obs["robot0_eef_pos"], np.float64)
                a = self.pl.action(ee, T2.quat_mat(np.asarray(self.obs["robot0_eef_quat"], np.float64)), self._tgt()).tolist()
                self.h["p_steps"] += 1
            r = super().step(a)
            self.obs = r[0]
            if self.n < WAIT:
                return r
            row = len(self.rows["held"]) - 1
            was = self.det.on
            on = bool(self.det.step(np.asarray(r[0]["robot0_gripper_qpos"], np.float64), float(a[6]), float(r[0]["robot0_eef_pos"][2])))
            if on and not was:
                self.h["onsets"].append(row)
            out = r
            if self.front is not None:
                obs = self.front.step(r[0], self.n, float(a[6]))
                self.h["disagree"] += int(bool(self.front.on) != on)
                out = (obs,) + tuple(r[1:])
            if self.pl is not None:
                res = self.pl.advance(on, r[0]["robot0_eef_pos"])
                self.h["phase_max"] = max(self.h["phase_max"], self.pl.phase)
                if self.pl.phase >= 3 and self.h["committed"] is None:
                    self.h["committed"] = row
                if res == "abort":
                    self.h["aborts"].append(dict(row=row, phase=self.pl.phase, log=self.pl.log["phases"]))
                    self.pl = None
            elif on and not was and self.h["committed"] is None:
                pz = self._tgt()
                self.pl = Place(KT, self.rng, r[0]["robot0_eef_pos"], T2.quat_mat(np.asarray(r[0]["robot0_eef_quat"], np.float64)), float(pz[2]))
                self.h["handovers"].append(dict(row=row, oracle_held=bool(self.rows["held"][-1])))
            return out

        def close(self):
            if self.n > 0:
                self.finish()
                RECS.append(self.data())
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
                i = len(SIDE)
                rec = dict(arm=arm, trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, tid=TID, key=TASK["key"])
                rec.update(self.h)
                rec["p_final_phase"] = None if self.pl is None else self.pl.phase
                rec["p_log"] = None if self.pl is None else self.pl.log["phases"]
                if self.front is not None:
                    rec["front"] = self.front.record(self.rows["held"])
                    rec["prompts"] = self.front.prompts
                SIDE.append(rec)
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], phrases=P, lex=X, kt=KT,
                                   masks="simulator" if Front is not None else None, place="object-blind P (target plate sim position)",
                                   eps=SIDE), fh, indent=1)
                os.replace(tmp, SIDEF)
                print("[s2fix] %s %s trial %s: onsets %s handovers %s aborts %d committed %s phase_max %d p_steps %d disagree %d" % (
                    arm, TASK["key"], rec["trial"], rec["onsets"][:4], [h["row"] for h in rec["handovers"]], len(rec["aborts"]),
                    rec["committed"], rec["phase_max"], rec["p_steps"], rec["disagree"]), flush=True)
                self.front, self.pl = None, None
            self._e.close()

    if arm == "htvp":
        Base = LC._wcp.WebsocketClientPolicy

        class PClient(Base):
            def infer(self, el):                                            # s2.client_main's tvp prompt switch, unchanged
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

    LC.OffScreenRenderEnv = lambda *a, **k: HEnv(ORIG(*a, **k))
    print("[s2fix] arm %s task %s %r object %s target %s max_steps %d phrases %s lex %s kt %s" % (
        arm, TASK["key"], TASK["language"], TASK["object"], TASK["target"], ms, P, (X["tw"], X["lift"]),
        dict(carry_z=KT["carry_z_above_tgt"], release_off=KT["release_off"])), flush=True)
    LC.main()


# ---------------------------------------------------------------- smoke summary / read (CPU)
def summary(d):
    M = s2.meta()
    for f in sorted(x for x in os.listdir(d) if x.endswith(".json") and not x.endswith("_h.json")):
        stem = os.path.join(d, f[:-5])
        if not os.path.exists(stem + "_track.npz"):
            continue
        R = json.load(open(stem + ".json", encoding="utf-8"))
        tid = int(list(R["tasks"].keys())[0])
        t = M["tasks"][tid]
        ks = [int(e["trial"]) for v in R["tasks"].values() for e in v["episodes"]]
        side = json.load(open(stem + "_h.json", encoding="utf-8"))["eps"]
        print("== %s %s %r" % (f, t["key"], t["language"]), flush=True)
        for k, row, s in zip(ks, s2.stage_rows(stem, t, ks), side):
            on = {"%s>%s" % (m.split("_")[0][:5], p): s2.on_of(row["on"], m, p) for m in s2.MUGS for p in s2.PLATES if s2.on_of(row["on"], m, p)}
            print("  " + json.dumps(dict(trial=k, success=row["success"], stage=row["stage"], bound=row["bound"], on=on,
                                         handovers=s["handovers"], aborts=[(x["row"], x["phase"]) for x in s["aborts"]],
                                         committed=s["committed"], phase_max=s["phase_max"], disagree=s["disagree"])), flush=True)


def rule(d, x):
    """d = [point, lo, hi] of X - I; x = success(X)."""
    if d[1] > 0 and x >= 0.50:
        return "FIXED"
    if d[2] < EPS:
        return "NOT FIXED"
    return "PARTIAL"


def read(a):
    import pp_atlas as PA
    import tv_frozen as TF
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    M = s2.meta()
    if a.selftest:                                                          # smoke states 0-3, smoke dirs, no ledger: plumbing only
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks = M["states"]["smoke"]
        RDIR = "/coc/flash10/ksingara3/.claude_tmp/pp/s2_smoke/%s"
        NDIR = "/coc/flash10/ksingara3/.claude_tmp/pp/s2fix_smoke/%s"
        s2_code_mismatch, mine_mismatch = "selftest", "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "scene2_fix"]
        assert len(rows) == 1, "expected exactly one scene2_fix prereg row, found %d" % len(rows)
        row = rows[0]
        s2row = [r for r in ledger.load("m3_prereg") if r.get("tag") == "scene2"]
        assert len(s2row) == 1
        reg = row["metrics"]["reused_sha256"]
        reuse_bad = [p for p, h in reg.items() if fsha(os.path.join(ROOT, p)) != h]
        assert not reuse_bad, "reused SR 50 rollouts changed since the scene2_fix row: %s" % reuse_bad

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        s2_code_mismatch = [p for p, h in s2row[0]["metrics"]["files_sha256"].items()
                            if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        mine_mismatch = [p for p, h in row["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks = M["states"]["main"]
        RDIR, NDIR = s2.ROLL, ROLL
    T = {t["key"]: t for t in M["tasks"]}
    comps = s2.NEW + s2.KNOWN
    res = dict(states=ks, mapping=MAPPING, reuse_ok=True, scene2_code_mismatch=s2_code_mismatch, s2fix_code_mismatch=mine_mismatch,
               models={})
    for m, d in MODELS.items():
        A = {}
        for arm in d["reused"] + d["new"]:
            base = RDIR if arm in d["reused"] else NDIR
            base = base % m if "%s" in base else base
            files = {c: os.path.join(base, "%s_%s_%d" % (m, arm, T[c]["tid"])) for c in comps}
            miss = [f for f in files.values() if not (os.path.exists(f + ".json") and os.path.exists(f + "_track.npz"))]
            if miss:
                A[arm] = "MISSING %d of %d tasks" % (len(miss), len(files))
                continue
            succ, RATE, desc = {}, {}, {}
            for c in comps:
                rr = s2.stage_rows(files[c], T[c], ks)
                succ[c] = np.array([float(r["success"]) for r in rr])
                RATE[c] = {s: np.array([float(r["stage"] == s) for r in rr]) for s in PA.STAGES}
                o, tg = T[c]["object"], T[c]["target"]
                hp = s2.HABIT[o]
                op = [p for p in s2.PLATES if p != tg][0]
                dd = dict(success=float(succ[c].mean()), bound_named=float(np.mean([r["bound"] == o for r in rr])),
                          bound_other=float(np.mean([r["bound"] not in (None, o) for r in rr])),
                          success_given_named_bound=(float(np.mean([r["success"] for r in rr if r["bound"] == o]))
                                                     if any(r["bound"] == o for r in rr) else None),
                          on_trained_plate_instead=float(np.mean([s2.on_of(r["on"], o, hp) for r in rr])) if hp != tg else None,
                          on_other_plate=float(np.mean([s2.on_of(r["on"], o, op) for r in rr])),
                          stages={s: float(RATE[c][s].mean()) for s in PA.STAGES})
                Tr = s2.load_track(files[c] + "_track.npz")
                oi, ti = s2.MOVABLE.index(o), s2.MOVABLE.index(tg)
                dd["final_obj_tgt_dxy_median"] = float(np.median([np.linalg.norm(r_["final"][oi, :2] - r_["final"][ti, :2]) for r_ in Tr]))
                if arm in d["new"]:
                    H = json.load(open(files[c] + "_h.json", encoding="utf-8"))["eps"]
                    assert [h["trial"] for h in H] == ks
                    dd["handover"] = dict(
                        fired=float(np.mean([len(h["handovers"]) > 0 for h in H])),
                        oracle_held_at_first=float(np.mean([h["handovers"][0]["oracle_held"] for h in H if h["handovers"]])) if any(h["handovers"] for h in H) else None,
                        n_handovers_mean=float(np.mean([len(h["handovers"]) for h in H])),
                        aborted_any=float(np.mean([len(h["aborts"]) > 0 for h in H])),
                        committed=float(np.mean([h["committed"] is not None for h in H])),
                        phase_max=dict(zip(*[x.tolist() for x in np.unique([h["phase_max"] for h in H], return_counts=True)])),
                        detector_disagree_steps=int(sum(h["disagree"] for h in H)))
                desc[c] = dd
            new_m, kn_m = np.stack([succ[c] for c in s2.NEW]), np.stack([succ[c] for c in s2.KNOWN])
            A[arm] = dict(new=new_m, known=kn_m, succ=succ, row=dict(
                success_new=TF.boot1(new_m), success_known=TF.boot1(kn_m), per_task=desc, atlas=TF.atlas_est(RATE, s2.NEW, s2.PAIR, ks, PA)))
        V, D = {}, {}
        I = A.get("I")
        for arm in d["reused"][1:] + d["new"]:
            X = A.get(arm)
            if not isinstance(X, dict) or not isinstance(I, dict):
                continue
            xn = float(X["new"].mean())
            dj, ds = TF.boot2(X["new"], I["new"]), s2.boot2_states(X["new"], I["new"])
            kj, kss = TF.boot2(X["known"], I["known"]), s2.boot2_states(X["known"], I["known"])
            per = {}
            for c in s2.NEW:
                dc = s2.boot2_states(X["succ"][c][None], I["succ"][c][None])
                per[c] = dict(d=dc, x=float(X["succ"][c].mean()), verdict=rule(dc, float(X["succ"][c].mean())))
            for c in s2.KNOWN:
                per[c] = dict(d=s2.boot2_states(X["succ"][c][None], I["succ"][c][None]), x=float(X["succ"][c].mean()))
            D[arm] = dict(new_joint=dj, new_states=ds, known_joint=kj, known_states=kss, per_task=per, x_new=xn)
            V[arm] = dict(pooled=rule(dj, xn), pooled_states_rule=rule(ds, xn), N1=per["N1"]["verdict"], N2=per["N2"]["verdict"],
                          guard="NO COST" if kj[1] > -0.10 else "COST",
                          role=("PRIMARY" if (m, arm) == PRIMARY else ("arm (a), reused SR 50" if arm in d["reused"] else "SECONDARY"))
                          + (" (GR00T NOT TESTABLE, SR 50)" if m == "groot" else ""))
        res["models"][m] = dict(arms={k: (v if isinstance(v, str) else v["row"]) for k, v in A.items()}, deltas=D, verdicts=V)
        print("== %s" % m, flush=True)
        for arm, v in A.items():
            if isinstance(v, str):
                print("   %s: %s" % (arm, v), flush=True)
                continue
            r_ = v["row"]
            print("   %s: new %s known %s | per task %s | atlas %s cross %s" % (
                arm, np.round(r_["success_new"], 3).tolist(), np.round(r_["success_known"], 3).tolist(),
                {c: round(x["success"], 3) for c, x in r_["per_task"].items()}, r_["atlas"]["verdict"],
                np.round([r_["atlas"]["groups"]["cross"]["E"]] + r_["atlas"]["groups"]["cross"]["ci"], 3).tolist()), flush=True)
            for c, x in r_["per_task"].items():
                print("      %s %s" % (c, json.dumps({k: (round(y, 3) if isinstance(y, float) else y) for k, y in x.items() if k != "stages"})), flush=True)
        for arm, dd in D.items():
            print("   %s - I: new joint %s states %s | known joint %s states %s | per task %s | VERDICT %s" % (
                arm, np.round(dd["new_joint"], 3).tolist(), np.round(dd["new_states"], 3).tolist(), np.round(dd["known_joint"], 3).tolist(),
                np.round(dd["known_states"], 3).tolist(), {c: (np.round(x["d"], 3).tolist(), round(x["x"], 3), x.get("verdict")) for c, x in dd["per_task"].items()},
                json.dumps(V[arm])), flush=True)
    print("== code hashes: scene2 row mismatches %s | scene2_fix row mismatches %s" % (s2_code_mismatch, mine_mismatch), flush=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, out)
    print("[s2fixread] wrote %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("paintcheck", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--out-dir")
    ap.add_argument("--out", default=READ)
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    if A.mode == "paintcheck":
        paintcheck(A)
    elif A.mode == "summary":
        summary(A.dir)
    else:
        read(A)

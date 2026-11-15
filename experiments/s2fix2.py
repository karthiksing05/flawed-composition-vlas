"""S2FIX2 (docs/SCAFFOLD_RESULTS.md 62; m3_prereg tag s2_fix2): a scene-2 fix on frozen pi0.5 that keeps the trained tasks and fixes
the new pairings.  Scene 2 = LIBERO-10 LIVING_ROOM_SCENE5 (SR 50); tasks K1 (white mug -> left plate), K2 (yellow-and-white mug ->
right plate), N1 (white -> right), N2 (yellow-and-white -> left).

STEP 1, DIAGNOSIS (mode `diag`, descriptive, reads SR 54's existing outputs only: results/flawed/s2fix_roll and SR 50's
results/flawed/s2_roll): why did SR 54's hand-over cost K2?  Per task and arm (pi0.5 I, hI, htvp): the first mug moved ('bound'), the
final plate of each mug by bound, success by bound, the place policy's phases and aborts, and the first detector onset against the
oracle Held.  The SR 54 read already printed the aggregate (K2: success given the named mug moved first 1.00 under htvp; hI moved the
white mug first on every K2 episode and scored 0), which predicts the cause: the hand-over fires at the FIRST hold whatever mug is
held, and in K2 the VLA's trained two-step program grasps the white mug first, so P carries the WRONG mug to the instructed plate.

THE FIX OF THAT CAUSE (arms H and G): hand over only at a hold of the NAMED mug.  At a hold of any other mug the VLA keeps control (it
is running its trained program on another object; the instruction says nothing about where that object goes).

THE ROUTER (arm G only).  At a hold of the named mug, the router predicts the VLA's place plate by SR 61's MAJORITY RULE over three
binary cues of the held mug:
    NEARER  the plate nearer the mug's settled start position (s2cue2's rule: undefined iff |d_left - d_right| < .02 m);
    SALIENT the plate with >= 1.5x the other's agentview pixels AND a centroid >= 10 px nearer the image centre, from the simulator
            segmentation of the settled start render (s2cue4.salient_of, SR 60's rule, unchanged; undefined otherwise);
    TRAINED the held mug's plate in the training-task index (the LIBERO-10 task whose demos the VLA trained on: white -> plate_1,
            yellow-and-white -> plate_2; the red mug has none).
    PREDICTED = the plate backed by >= 2 defined cues; undefined otherwise (e.g. salient undefined and nearer != trained).
If PREDICTED equals the instructed plate, the VLA keeps control; otherwise (also when PREDICTED is undefined) the place policy P takes
over.  P is SR 54's object-blind place policy, unchanged (s2fix.Place: hand pose + the TARGET plate's simulator position, constants
fitted per plate), with SR 54's abort / hand-back rule.
Information used, stated plainly: the router adds the training-task index (which plate each mug was trained to) and reads simulator
state of the same kind SR 54 already used (SR 54's P read the target plate's simulator position; SR 54's htvp painted from simulator
segmentation masks of every mug and plate): the settled start positions of the mugs and plates, the simulator segmentation of the
agentview start render (salience), and, at the detector onset, the mugs' current simulator positions to name the held mug (the mug
nearest the hand in xy, within HELD_R).  It adds no task information: the instruction's object and target roles are what SR 54 used.
The router and the named-mug check are structural; neither reads the VLA's actions.

ARMS (pi0.5, frozen; pp_serve_purify --model pi05 --mode intact --comp= --seed 13; client seed 7, --bind track):
    I    intact (s2.py client, unchanged)
    H0   SR 54's hI, unchanged (s2fix.py client --arm hI): hand over at the first hold of any mug (SECONDARY: the cause, on new states)
    H    SR 54's hand-over with the step-1 fix: hand over at a hold of the named mug
    G    H + the router gate
LAYOUTS (new states, LIBERO sampler on K1's BDDL, seeds 890000 + k; k 0-3 smoke, 4-63 main; branches seed_base 36000000 + 1000 k):
    O    unmanipulated (PRIMARY): I, H0, H, G
    PD2  SR 61's PD2 layout on the new states (SECONDARY, router-discriminating): mugs' xy swapped (SR 57), plate_1 at (.10, -.10),
         plate_2 at (-.05, .31); I, H, G.  Here nearness and salience outvote the trained plate (SR 61: 153 of 153), so the router's
         prediction differs from an identity-only gate.

Modes (LIBERO venv, py3.8):
    diag [--out X]                    (CPU)  step 1, SR 54's outputs only
    states                            (EGL)  -> results/flawed/s2fix2/O/s2_living_<tid>.npy, branches_{smoke,main}.json
    build                             (EGL)  -> results/flawed/s2fix2/PD2/..., s2fix2_states.json (cues + router per state)
    client --arm H|G --layout O|PD2 <s2.py client args>
    ref --layout L --tid T --branches B --out-dir D   (CPU, C1)
    summary DIR [--layout L]          (smoke)
    read [DIR --selftest --out X]     (CPU)
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
import s2cue2                                                               # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2fix2")
ROLL = os.path.join(OUT, "s2fix2_roll")
READ = os.path.join(OUT, "s2fix2_read.json")
DIAG = os.path.join(OUT, "s2fix2_diag.json")
STATES_JSON = os.path.join(CD, "s2fix2_states.json")
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
BASE, BR_BASE, N_STATES = 890000, 36000000, 64
SMOKE = list(range(0, 4))
MAIN = list(range(4, 64))
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
LAYOUTS = ("O", "PD2")
PD2_PLATES = (0.10, -0.10, -0.05, 0.31)                                     # SR 61's PD2 pick (family D, D5)
TRAINED = {W: PL, Y: PR}                                                    # the training-task index (LIBERO-10 SCENE5's one task)
HELD_R = 0.08
RUNS = dict(O=("I", "H0", "H", "G"), PD2=("I", "H", "G"))
NI_MARGIN = -0.05
MAPPING = (
    "Frozen pi0.5 (pp_serve_purify --model pi05 --mode intact --comp= --seed 13), client seed 7, --bind track; NEW states (LIBERO "
    "sampler on K1's BDDL, seeds 890000 + k, none equal to an SR 50-61 state), main k = 4-63 (60), branches seed_base 36000000 + "
    "1000 k; tasks K1, K2, N1, N2.  Arms: I (s2.py client), H0 (SR 54's hI unchanged), H (hand over only at a hold of the named "
    "mug), G (H + router: at a hold of the named mug the VLA keeps control iff SR 61's majority plate of {nearer, salient, trained} "
    "equals the instructed plate).  Success = the BDDL goal (named mug On the named plate).  Matrices [task, state].  Paired "
    "bootstrap over states (tasks fixed; s2.boot2_states, B = 2000, numpy default_rng(0); CI = 2.5 / 97.5 percentiles).  PRIMARY "
    "(layout O): dN = G - I on N1+N2; dK = G - I on K1+K2; dGH = G - H on K1+K2.  LABEL: FIX WORKS iff dN lo > 0 AND dK lo > -.05 "
    "(NON-INFERIOR at margin -.05); PARTIAL iff exactly one of the two holds; FAILS iff neither.  Non-inferiority label on dK: "
    "NON-INFERIOR iff lo > -.05; INFERIOR iff hi < -.05; else NOT SHOWN.  dGH: ROUTER KEEPS MORE iff lo > 0; ROUTER COSTS iff hi < "
    "0; else NO CLEAR DIFFERENCE.  SECONDARY (O): H - H0 on K1+K2 (CAUSE FIXED iff lo > 0), H - I on N1+N2 and K1+K2 (same rules), G "
    "- H on N1+N2 (descriptive), SR 54's rule on G and H (FIXED iff lo > 0 and success >= .50).  SECONDARY (PD2): G - I, H - I, G - H "
    "pooled over the four tasks and per task, labels HELPS iff lo > 0, HURTS iff hi < 0, else NO CLEAR DIFFERENCE.  ROUTER "
    "AGREEMENT (held-out check of SR 61): on arm I, over placements (the first mug moved, 'bound', white or yellow-and-white, ends On "
    "exactly one plate) whose router prediction is defined, the share on the predicted plate (s2cue.boot_share, B = 2000, "
    "default_rng(0), counts per state over tasks); per layout, split by named / other mug.  C1: the reference planner "
    "(s2.plan_episode, unchanged) on the same main states; MET iff its N1-N2 success >= .80.  Descriptive: gate decisions per task "
    "(VLA kept: other mug; VLA kept: router agrees; place), held-mug identification vs 'bound', P phases and aborts, final plate of "
    "each mug, router configurations per layout.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def side_of(p):
    return None if p is None else ("L" if p == PL else "R")


def plate_of_side(s):
    return None if s is None else (PL if s == "L" else PR)


def router_row(xy, salient):
    """Settled start xy (name -> [x, y]) and the layout's salient side -> per mug: cues and the predicted plate (majority of defined)."""
    out = {}
    for m in s2.MUGS:
        p = np.array(xy[m])
        dl, dr = np.linalg.norm(p - np.array(xy[PL])), np.linalg.norm(p - np.array(xy[PR]))
        n = None if abs(dl - dr) < s2cue2.TOL_NEAR else ("L" if dl < dr else "R")
        t = side_of(TRAINED.get(m))
        votes = [v for v in (n, salient, t) if v is not None]
        pred = next((s_ for s_ in ("L", "R") if votes.count(s_) >= 2), None)
        cfg = None
        if None not in (n, salient, t):
            cfg = "unanimous" if n == salient == t else ("nearer odd" if salient == t else ("salient odd" if n == t else "trained odd"))
        out[m] = dict(nearer=n, salient=salient, trained=t, predicted=plate_of_side(pred), config=cfg, d_left=float(dl), d_right=float(dr))
    return out


# ---------------------------------------------------------------- step 1: diagnosis of SR 54's K2 cost (CPU, descriptive)
def diag(a):
    out = os.path.abspath(a.out or DIAG)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    ks = M_["states"]["main"]
    res = dict(note="DESCRIPTIVE: SR 54's existing outputs (results/flawed/s2_roll pi05_I, results/flawed/s2fix_roll pi05_hI / htvp)",
               states=ks, arms={})
    for arm, base in (("I", s2.ROLL), ("hI", os.path.join(OUT, "s2fix_roll")), ("htvp", os.path.join(OUT, "s2fix_roll"))):
        res["arms"][arm] = {}
        for c in s2.KNOWN + s2.NEW:
            t = T[c]
            stem = os.path.join(base, "pi05_%s_%d" % (arm, t["tid"]))
            rr = s2.stage_rows(stem, t, ks)
            Tr = s2.load_track(stem + "_track.npz")
            H = json.load(open(stem + "_h.json", encoding="utf-8"))["eps"] if arm != "I" else [None] * len(ks)
            if arm != "I":
                assert [h["trial"] for h in H] == ks
            o, tg = t["object"], t["target"]
            cells = {}
            for r, tr, h in zip(rr, Tr, H):
                b = r["bound"]
                key = "named" if b == o else ("other:%s" % (b.split("_")[0] if b else None))
                c_ = cells.setdefault(key, dict(n=0, success=0, white_on={}, yw_on={}, bound_on={}, phase_max={}, aborted=0,
                                                handovers=0, first_onset_minus_oracle=[]))
                c_["n"] += 1
                c_["success"] += int(r["success"])
                for nm, d_ in ((W, c_["white_on"]), (Y, c_["yw_on"])):
                    pp = str(s2cue.plate_of(r["on"], nm))
                    d_[pp] = d_.get(pp, 0) + 1
                if b in s2.MUGS:
                    pp = str(s2cue.plate_of(r["on"], b))
                    pp = "instructed" if pp == tg else pp
                    c_["bound_on"][pp] = c_["bound_on"].get(pp, 0) + 1
                if h is not None:
                    pm = str(h["phase_max"])
                    c_["phase_max"][pm] = c_["phase_max"].get(pm, 0) + 1
                    c_["aborted"] += int(len(h["aborts"]) > 0)
                    c_["handovers"] += int(len(h["handovers"]) > 0)
                    held = np.asarray(tr["held"], bool)
                    if h["onsets"] and held.any():
                        c_["first_onset_minus_oracle"].append(int(h["onsets"][0]) - int(np.argmax(held)))
            for c_ in cells.values():
                v = c_.pop("first_onset_minus_oracle")
                c_["first_onset_minus_oracle_held_median"] = float(np.median(v)) if v else None
                c_["n_with_oracle_held"] = len(v)
            res["arms"][arm][c] = dict(success=float(np.mean([r["success"] for r in rr])), by_bound=cells)
            print("[s2fix2 diag] %s %s success %.3f | %s" % (arm, c, res["arms"][arm][c]["success"], json.dumps(cells)), flush=True)
    # the cause, in one line per hand-over arm and K task: wrong mug first -> where it ended
    for arm in ("hI", "htvp"):
        for c in s2.KNOWN:
            cells = res["arms"][arm][c]["by_bound"]
            wrong = {k: v for k, v in cells.items() if k != "named"}
            nw = sum(v["n"] for v in wrong.values())
            ni = sum(v["bound_on"].get("instructed", 0) for v in wrong.values())
            sw = sum(v["success"] for v in wrong.values())
            nn = cells.get("named", dict(n=0, success=0))
            print("[s2fix2 diag] CAUSE %s %s: other mug moved first on %d of %d; of those, that mug ended On the INSTRUCTED plate %d, "
                  "success %d | named mug first %d, success %d" % (arm, c, nw, len(ks), ni, sw, nn["n"], nn["success"]), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(out + ".tmp", out)
    print("[s2fix2 diag] wrote %s" % out, flush=True)


# ---------------------------------------------------------------- states and build (LIBERO venv, EGL)
def states(a):
    import s2_tasks as T2
    od = os.path.join(CD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))

    def draw(k):                                                            # s2_tasks.build's rule with a new base
        np.random.seed(BASE + k)
        env.seed(BASE + k)
        env.reset()
        return env.sim.get_state().flatten()
    S = np.stack([draw(k) for k in range(N_STATES)])
    env.close()
    old = [np.load(os.path.join(s2.SD, "states", "%s_0.npy" % s2.SUITE))] + [
        np.load(os.path.join(OUT, d, "O", "%s_0.npy" % s2.SUITE)) for d in ("s2cue3", "s2cue4", "s2cue5")]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("same as an SR 50-61 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    for ph, ks in (("smoke", SMOKE), ("main", MAIN)):
        with open(os.path.join(CD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE + 1000 * k, at=0, variant="s2") for k in ks], fh, indent=1)
    print("[s2fix2 states] %d new states (seed %d + k), none equal to an SR 50-61 state; wrote %s" % (len(S), BASE, CD), flush=True)


def build(a):
    import s2cue3
    import s2cue4
    import s2cue5
    assert not os.path.exists(STATES_JSON), "exists: " + STATES_JSON
    pd = os.path.join(CD, "PD2")
    assert not os.path.exists(pd), "exists: " + pd
    SJ5 = json.load(open(s2cue5.STATES_JSON, encoding="utf-8"))
    assert tuple(SJ5["design"]["PD2"]["plates"]) == PD2_PLATES, SJ5["design"]["PD2"]
    M_ = s2.meta()
    S = np.load(os.path.join(CD, "O", "%s_0.npy" % s2.SUITE))
    sc = s2cue5.Scene5()
    orig = []
    for k in range(len(S)):
        snap, _ = sc.check(S[k])
        orig.append(s2cue4.annotate(snap))
    d0 = min(o["dmin"] for o in orig)
    lay = dict(O=dict(rows=[]), PD2=dict(rows=[], plates=list(PD2_PLATES)))
    A = []
    for k in range(len(S)):
        o = orig[k]
        lay["O"]["rows"].append(dict(k=k, xy=o["xy"], salient=o["salient"], sal_numbers=o["sal_numbers"], left_mug=o["left_mug"],
                                     router=router_row(o["xy"], o["salient"]), geom_ok=True))
        sp, placed = sc.make4(S[k], "PD2", PD2_PLATES)
        snap, _ = sc.check(sp)
        ok = s2cue3.geom_ok(snap, placed, o, d0)
        s2cue4.annotate(snap)
        lay["PD2"]["rows"].append(dict(k=k, xy=snap["xy"], salient=snap["salient"], sal_numbers=snap["sal_numbers"],
                                       left_mug=snap["left_mug"], router=router_row(snap["xy"], snap["salient"]), geom_ok=bool(ok)))
        A.append(sp)
    sc.env.close()
    os.makedirs(pd)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(pd, "%s_%d.npy" % (s2.SUITE, t["tid"])), np.stack(A))
    with open(STATES_JSON, "w", encoding="utf-8") as fh:
        json.dump(dict(d0=d0, pd2_plates=list(PD2_PLATES), held_r=HELD_R, trained=TRAINED, layouts=lay), fh, indent=1)
    for L, v in lay.items():
        rr = [r for r in v["rows"] if r["k"] in MAIN]
        pred = {m: {str(x): sum(r["router"][m]["predicted"] == x for r in rr) for x in (PL, PR, None)} for m in s2.MUGS}
        cfg = {m: {str(x): sum(r["router"][m]["config"] == x for r in rr) for x in
                   ("unanimous", "nearer odd", "salient odd", "trained odd", None)} for m in (W, Y)}
        print("[s2fix2 build] %s main %d: geom ok %d | salient %s | left mug %s | router predicted %s | configs %s" % (
            L, len(rr), sum(r["geom_ok"] for r in rr), {str(x): sum(r["salient"] == x for r in rr) for x in ("L", "R", None)},
            {str(x): sum(r["left_mug"] == x for r in rr) for x in (W, Y, None)}, json.dumps(pred), json.dumps(cfg)), flush=True)
    print("[s2fix2 build] wrote %s and %s" % (STATES_JSON, pd), flush=True)


# ---------------------------------------------------------------- client: arms H and G (LIBERO venv)
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
    layout = take("--layout")
    assert arm in ("H", "G") and layout in LAYOUTS, (arm, layout)
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
    ms = int(TASK["max_steps"])
    assert "--max-steps" not in sys.argv
    sys.argv += ["--max-steps", str(ms)]
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    KP = json.load(open(os.path.join(s2.SD, "keypoints.json"), encoding="utf-8"))["tasks"]
    KT = KP[SF.HALF_OF_PLATE[TASK["target"]]]
    assert KT["target"] == TASK["target"], (KT["target"], TASK["target"])
    RT = {r["k"]: r["router"] for r in json.load(open(STATES_JSON, encoding="utf-8"))["layouts"][layout]["rows"]}
    ORIG, RECS, SIDE = LC.OffScreenRenderEnv, [], []
    SIDEF = OUTF.replace(".json", "_h.json")

    class HEnv(s2.Rec2):
        def __init__(self, env):
            super().__init__(env, TASK["object"], TASK["target"])
            self.det, self.pl = None, None

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            i = len(RECS)
            self.k = int(br[i]["trial"]) if i < len(br) else None
            self.sb = int(br[i]["seed_base"]) if i < len(br) else 0
            self.route = RT[self.k] if self.k in RT else {}
            self.rng = np.random.default_rng(self.sb + 20)                 # SR 54's P rng rule
            self.det = ProprioHeld(X["tw"], X["lift"])
            self.pl, self.obs = None, obs
            self.h = dict(onsets=[], events=[], handovers=[], aborts=[], committed=None, p_steps=0, phase_max=-1)
            return obs

        def _tgt(self):
            return self._pos(TASK["target"])

        def _held(self, ee):
            d = {m: float(np.linalg.norm(self._pos(m)[:2] - np.asarray(ee, np.float64)[:2])) for m in s2.MUGS}
            m = min(d, key=d.get)
            return (m if d[m] < HELD_R else None), d

        def step(self, a):
            if self.pl is not None:
                ee = np.asarray(self.obs["robot0_eef_pos"], np.float64)
                a = self.pl.action(ee, T2.quat_mat(np.asarray(self.obs["robot0_eef_quat"], np.float64)), self._tgt()).tolist()
                self.h["p_steps"] += 1
            r = super().step(a)
            self.obs = r[0]
            if self.n < s2.WAIT:
                return r
            row = len(self.rows["held"]) - 1
            was = self.det.on
            on = bool(self.det.step(np.asarray(r[0]["robot0_gripper_qpos"], np.float64), float(a[6]), float(r[0]["robot0_eef_pos"][2])))
            if on and not was:
                self.h["onsets"].append(row)
            if self.pl is not None:
                res = self.pl.advance(on, r[0]["robot0_eef_pos"])
                self.h["phase_max"] = max(self.h["phase_max"], self.pl.phase)
                if self.pl.phase >= 3 and self.h["committed"] is None:
                    self.h["committed"] = row
                if res == "abort":
                    self.h["aborts"].append(dict(row=row, phase=self.pl.phase, log=self.pl.log["phases"]))
                    self.pl = None
            elif on and not was and self.h["committed"] is None:
                held, dd = self._held(r[0]["robot0_eef_pos"])
                rt = self.route.get(held, {}) if held else {}
                ev = dict(row=row, held=held, dist={m.split("_")[0]: round(v, 4) for m, v in dd.items()},
                          oracle_held=bool(self.rows["held"][-1]), predicted=rt.get("predicted"))
                if held != TASK["object"]:
                    ev["decision"] = "vla: other mug"
                elif arm == "G" and rt.get("predicted") == TASK["target"]:
                    ev["decision"] = "vla: router agrees"
                else:
                    ev["decision"] = "place"
                    pz = self._tgt()
                    self.pl = SF.Place(KT, self.rng, r[0]["robot0_eef_pos"],
                                       T2.quat_mat(np.asarray(r[0]["robot0_eef_quat"], np.float64)), float(pz[2]))
                    self.h["handovers"].append(dict(row=row, oracle_held=ev["oracle_held"]))
                self.h["events"].append(ev)
            return r

        def close(self):
            if self.n > 0:
                self.finish()
                RECS.append(self.data())
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
                i = len(SIDE)
                rec = dict(arm=arm, layout=layout, trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, tid=TID,
                           key=TASK["key"], route={m.split("_")[0]: v for m, v in self.route.items()})
                rec.update(self.h)
                rec["p_final_phase"] = None if self.pl is None else self.pl.phase
                rec["p_log"] = None if self.pl is None else self.pl.log["phases"]
                SIDE.append(rec)
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, layout=layout, tid=TID, key=TASK["key"], language=TASK["language"], lex=X, kt=KT,
                                   held_r=HELD_R, place="SR 54 object-blind P (target plate sim position)", eps=SIDE), fh, indent=1)
                os.replace(tmp, SIDEF)
                print("[s2fix2] %s %s %s trial %s: events %s | handovers %s aborts %d committed %s phase_max %d p_steps %d" % (
                    arm, layout, TASK["key"], rec["trial"], [(e["row"], e["held"] and e["held"].split("_")[0], e["decision"])
                                                             for e in rec["events"][:4]],
                    [h["row"] for h in rec["handovers"]], len(rec["aborts"]), rec["committed"], rec["phase_max"], rec["p_steps"]),
                    flush=True)
                self.pl = None
            self._e.close()

    LC.OffScreenRenderEnv = lambda *a, **k: HEnv(ORIG(*a, **k))
    print("[s2fix2] arm %s layout %s task %s %r object %s target %s max_steps %d lex %s kt %s held_r %.2f" % (
        arm, layout, TASK["key"], TASK["language"], TASK["object"], TASK["target"], ms, (X["tw"], X["lift"]),
        dict(carry_z=KT["carry_z_above_tgt"], release_off=KT["release_off"]), HELD_R), flush=True)
    LC.main()


# ---------------------------------------------------------------- reference planner (C1; LIBERO venv, CPU)
def ref(a):
    import s2cue3
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    assert t["tid"] == a.tid and a.layout in LAYOUTS
    os.makedirs(a.out_dir, exist_ok=True)
    fn = os.path.join(a.out_dir, "ref_%s_%d.json" % (a.layout, a.tid))
    assert not os.path.exists(fn), "exists: " + fn
    S = np.load(os.path.join(CD, a.layout, "%s_%d.npy" % (s2.SUITE, a.tid)))
    br = {int(b["trial"]): int(b["seed_base"]) for b in json.load(open(a.branches, encoding="utf-8"))}
    kp, eps = s2cue3.ref_eps(S, t, t["target"], sorted(br), lambda k: br[k])
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(layout=a.layout, tid=a.tid, key=t["key"], kp=kp, eps=eps), fh, indent=1)


# ---------------------------------------------------------------- read helpers (CPU)
def stem_of(RD, L, arm, tid):
    return os.path.join(RD, L, "pi05_%s_%d" % (arm, tid))


def load_run(RD, L, arm, ks):
    """-> dict(succ {task: [n_states]}, rows {task: stage rows}, side {task: _h.json eps or None}) or a MISSING string."""
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    out = dict(succ={}, rows={}, side={})
    for c in s2.KNOWN + s2.NEW:
        st = stem_of(RD, L, arm, T[c]["tid"])
        if not (os.path.exists(st + ".json") and os.path.exists(st + "_track.npz")):
            return "MISSING %s" % st
        rr = s2.stage_rows(st, T[c], ks)
        out["rows"][c] = rr
        out["succ"][c] = np.array([float(r["success"]) for r in rr])
        out["side"][c] = None
        if os.path.exists(st + "_h.json"):
            H = json.load(open(st + "_h.json", encoding="utf-8"))["eps"]
            assert [h["trial"] for h in H] == ks, st
            out["side"][c] = H
    return out


def mat(run, tasks):
    return np.stack([run["succ"][c] for c in tasks])


def ni_label(d):
    return "NON-INFERIOR" if d[1] > NI_MARGIN else ("INFERIOR" if d[2] < NI_MARGIN else "NOT SHOWN")


def diff_label(d, pos="HELPS", neg="HURTS"):
    return pos if d[1] > 0 else (neg if d[2] < 0 else "NO CLEAR DIFFERENCE")


def fix_label(dN, dK):
    a_, b_ = dN[1] > 0, dK[1] > NI_MARGIN
    return "FIX WORKS" if (a_ and b_) else ("PARTIAL" if (a_ or b_) else "FAILS")


def sr54_rule(d, x):
    return "FIXED" if (d[1] > 0 and x >= 0.50) else ("NOT FIXED" if d[2] < 0.10 else "PARTIAL")


def agreement(run, routes, ks):
    """Arm I: placements of the first mug moved vs the router's predicted plate for that mug."""
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    F, D = np.zeros(len(ks)), np.zeros(len(ks))
    split = dict(named=[0, 0], other=[0, 0])
    cfg, undefined, placements = {}, 0, 0
    for c, rr in run["rows"].items():
        for i, (k, r) in enumerate(zip(ks, rr)):
            h = r["bound"]
            if h not in (W, Y):
                continue
            dest = s2cue.plate_of(r["on"], h)
            if dest not in s2.PLATES:
                continue
            placements += 1
            rt = routes[k][h]
            if rt["predicted"] is None:
                undefined += 1
                continue
            hit = int(dest == rt["predicted"])
            F[i] += hit
            D[i] += 1
            g = "named" if h == T[c]["object"] else "other"
            split[g][0] += hit
            split[g][1] += 1
            cc = cfg.setdefault(str(rt["config"]), [0, 0])
            cc[0] += hit
            cc[1] += 1
    return dict(share=s2cue.boot_share(F, D), split=split, by_config=cfg, undefined=undefined, placements=placements)


def describe(run, routes, ks):
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    out = {}
    for c, rr in run["rows"].items():
        o, tg = T[c]["object"], T[c]["target"]
        d = dict(success=float(run["succ"][c].mean()),
                 bound={str(b and b.split("_")[0]): int(sum(r["bound"] == b for r in rr)) for b in (W, Y, R, None)},
                 success_given_named_bound=(float(np.mean([r["success"] for r in rr if r["bound"] == o]))
                                            if any(r["bound"] == o for r in rr) else None),
                 white_on={str(p): int(sum(s2cue.plate_of(r["on"], W) == p for r in rr)) for p in (PL, PR, None, "both")},
                 yw_on={str(p): int(sum(s2cue.plate_of(r["on"], Y) == p for r in rr)) for p in (PL, PR, None, "both")},
                 named_predicted={str(p): int(sum(routes[k][o]["predicted"] == p for k in ks)) for p in (PL, PR, None)})
        H = run["side"][c]
        if H is not None and "events" in H[0]:
            ev = [e for h in H for e in h["events"]]
            first = [h["events"][0] if h["events"] else None for h in H]
            d["gate"] = dict(
                events=len(ev), decisions={x: int(sum(e["decision"] == x for e in ev)) for x in
                                           ("vla: other mug", "vla: router agrees", "place")},
                episodes_with_place=int(sum(len(h["handovers"]) > 0 for h in H)),
                first_event_held_eq_bound=float(np.mean([f is not None and f["held"] == r["bound"] for f, r in zip(first, rr)])),
                first_event_held_none=int(sum(f is not None and f["held"] is None for f in first)),
                no_event=int(sum(f is None for f in first)),
                aborted_any=int(sum(len(h["aborts"]) > 0 for h in H)),
                committed=int(sum(h["committed"] is not None for h in H)),
                phase_max={str(p): int(sum(h["phase_max"] == p for h in H)) for p in (-1, 0, 1, 2, 3, 4)},
                success_given_place=(float(np.mean([r["success"] for h, r in zip(H, rr) if h["handovers"]]))
                                     if any(h["handovers"] for h in H) else None),
                success_given_no_place=(float(np.mean([r["success"] for h, r in zip(H, rr) if not h["handovers"]]))
                                        if any(not h["handovers"] for h in H) else None))
        elif H is not None:                                                 # SR 54's hI side file (H0)
            d["gate"] = dict(episodes_with_place=int(sum(len(h["handovers"]) > 0 for h in H)),
                             aborted_any=int(sum(len(h["aborts"]) > 0 for h in H)),
                             committed=int(sum(h["committed"] is not None for h in H)),
                             phase_max={str(p): int(sum(h["phase_max"] == p for h in H)) for p in (-1, 0, 1, 2, 3, 4)})
        out[c] = d
    return out


def summary(d, layout):
    SJ = json.load(open(STATES_JSON, encoding="utf-8"))
    M_ = s2.meta()
    for L in ([layout] if layout else LAYOUTS):
        routes = {r["k"]: r["router"] for r in SJ["layouts"][L]["rows"]}
        for arm in RUNS[L]:
            for tid in TIDS:
                st = stem_of(d, L, arm, tid)
                if not (os.path.exists(st + ".json") and os.path.exists(st + "_track.npz")):
                    continue
                t = M_["tasks"][tid]
                Rj = json.load(open(st + ".json", encoding="utf-8"))
                ks = [int(e["trial"]) for v in Rj["tasks"].values() for e in v["episodes"]]
                H = json.load(open(st + "_h.json", encoding="utf-8"))["eps"] if os.path.exists(st + "_h.json") else [None] * len(ks)
                for k, r, h in zip(ks, s2.stage_rows(st, t, ks), H):
                    on = {"%s>%s" % (m.split("_")[0][:5], p): 1 for m in s2.MUGS for p in s2.PLATES if s2.on_of(r["on"], m, p)}
                    ev = None if h is None else [(e["row"], e.get("held"), e.get("dist"), e.get("decision"), e.get("predicted"))
                                                 for e in h.get("events", [])] or [(x["row"],) for x in h.get("handovers", [])]
                    print("[s2fix2 smoke] %s %s %s k%d success %s bound %s on %s | named predicted %s | events %s | phase_max %s" % (
                        L, arm, t["key"], k, r["success"], r["bound"], on, routes[k][t["object"]]["predicted"], ev,
                        None if h is None else h["phase_max"]), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_") and x.endswith(".json")):
        Rj = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2fix2 smoke] %s: %s" % (f, [(e["trial"], e["success"]) for e in Rj["eps"]]), flush=True)


def read(a):
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    SJ = json.load(open(STATES_JSON, encoding="utf-8"))
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD, mism = SMOKE, a.dir, "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_fix2"]
        assert len(rows) == 1, "expected exactly one s2_fix2 prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD = MAIN, ROLL
    res = dict(states=ks, mapping=MAPPING, code_mismatch=mism, layouts={})
    for L in LAYOUTS:
        routes = {r["k"]: r["router"] for r in SJ["layouts"][L]["rows"]}
        runs = {arm: load_run(RD, L, arm, ks) for arm in RUNS[L]}
        R_ = dict(arms={}, deltas={}, verdicts={})
        for arm, run in runs.items():
            if isinstance(run, str):
                R_["arms"][arm] = run
                print("== %s %s: %s" % (L, arm, run), flush=True)
                continue
            R_["arms"][arm] = dict(success={c: float(v.mean()) for c, v in run["succ"].items()},
                                   new=float(mat(run, s2.NEW).mean()), known=float(mat(run, s2.KNOWN).mean()),
                                   describe=describe(run, routes, ks))
            print("== %s %s: per task %s | N1+N2 %.3f | K1+K2 %.3f" % (
                L, arm, {c: round(v, 3) for c, v in R_["arms"][arm]["success"].items()}, R_["arms"][arm]["new"],
                R_["arms"][arm]["known"]), flush=True)
            for c, x in R_["arms"][arm]["describe"].items():
                print("      %s %s" % (c, json.dumps(x)), flush=True)
        ok = {arm: not isinstance(v, str) for arm, v in runs.items()}

        def dd(x, y, tasks):
            return s2.boot2_states(mat(runs[x], tasks), mat(runs[y], tasks))
        if L == "O" and ok.get("G") and ok.get("I"):
            dN, dK = dd("G", "I", s2.NEW), dd("G", "I", s2.KNOWN)
            R_["deltas"]["G-I N"], R_["deltas"]["G-I K"] = dN, dK
            R_["verdicts"]["PRIMARY LABEL"] = fix_label(dN, dK)
            R_["verdicts"]["G-I K non-inferiority (margin -.05)"] = ni_label(dK)
            R_["verdicts"]["G-I N"] = diff_label(dN)
            if ok.get("H"):
                dGH = dd("G", "H", s2.KNOWN)
                R_["deltas"]["G-H K"] = dGH
                R_["verdicts"]["G-H K"] = diff_label(dGH, "ROUTER KEEPS MORE", "ROUTER COSTS")
                R_["deltas"]["G-H N"] = dd("G", "H", s2.NEW)
                dHN, dHK = dd("H", "I", s2.NEW), dd("H", "I", s2.KNOWN)
                R_["deltas"]["H-I N"], R_["deltas"]["H-I K"] = dHN, dHK
                R_["verdicts"]["H label (same rule as G)"] = fix_label(dHN, dHK)
                R_["verdicts"]["H-I K non-inferiority"] = ni_label(dHK)
                if ok.get("H0"):
                    dHH0 = dd("H", "H0", s2.KNOWN)
                    R_["deltas"]["H-H0 K"] = dHH0
                    R_["verdicts"]["H-H0 K (step-1 cause)"] = diff_label(dHH0, "CAUSE FIXED", "WORSE")
                    R_["deltas"]["H0-I K"] = dd("H0", "I", s2.KNOWN)
                    R_["deltas"]["H0-I N"] = dd("H0", "I", s2.NEW)
            for arm in ("G", "H", "H0"):
                if ok.get(arm):
                    d_ = dd(arm, "I", s2.NEW)
                    R_["verdicts"]["SR 54 rule %s" % arm] = sr54_rule(d_, float(mat(runs[arm], s2.NEW).mean()))
            per = {}
            for c in s2.NEW + s2.KNOWN:
                per[c] = {"%s-I" % arm: s2.boot2_states(runs[arm]["succ"][c][None], runs["I"]["succ"][c][None])
                          for arm in ("G", "H", "H0") if ok.get(arm)}
            R_["deltas"]["per task"] = per
        if L == "PD2" and ok.get("I"):
            ALLT = s2.KNOWN + s2.NEW
            for x, y in (("G", "I"), ("H", "I"), ("G", "H")):
                if ok.get(x) and ok.get(y):
                    d_ = dd(x, y, ALLT)
                    R_["deltas"]["%s-%s all" % (x, y)] = d_
                    R_["verdicts"]["%s-%s all" % (x, y)] = diff_label(d_)
                    R_["deltas"]["%s-%s per task" % (x, y)] = {c: s2.boot2_states(runs[x]["succ"][c][None], runs[y]["succ"][c][None])
                                                              for c in ALLT}
        if ok.get("I"):
            R_["agreement_I"] = agreement(runs["I"], routes, ks)
            print("== %s router agreement on arm I: %s" % (L, json.dumps(R_["agreement_I"])), flush=True)
        for k_, v in R_["deltas"].items():
            print("   %s %s" % (k_, json.dumps(v if not isinstance(v, list) else np.round(v, 3).tolist())), flush=True)
        print("   VERDICTS %s: %s" % (L, json.dumps(R_["verdicts"])), flush=True)
        ref_ = {}
        for tid in TIDS:
            f = os.path.join(RD, "ref_%s_%d.json" % (L, tid))
            if os.path.exists(f):
                ee = json.load(open(f, encoding="utf-8"))["eps"]
                assert [e["trial"] for e in ee] == ks, f
                ref_[s2.meta()["tasks"][tid]["key"]] = dict(success=float(np.mean([e["success"] for e in ee])),
                                                            errors=int(sum(e.get("error") is not None for e in ee)))
        nn = [ref_[c]["success"] for c in s2.NEW if c in ref_]
        R_["reference"] = dict(per_task=ref_, success_new=float(np.mean(nn)) if len(nn) == 2 else None,
                               C1=None if len(nn) < 2 else ("C1 MET" if np.mean(nn) >= 0.80 else "C1 UNMET"))
        print("== %s reference: %s" % (L, json.dumps(R_["reference"])), flush=True)
        res["layouts"][L] = R_
    print("== code hash mismatches: %s" % mism, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, out)
    print("[s2fix2 read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("diag", "states", "build", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--layout", choices=LAYOUTS)
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    if A.mode == "read" and A.out is None:
        A.out = READ
    dict(diag=diag, states=states, build=build, ref=ref, summary=lambda a: summary(a.dir, a.layout), read=read)[A.mode](A)

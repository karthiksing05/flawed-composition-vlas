"""S2RGT (docs/SCAFFOLD_RESULTS.md 69; m3_prereg tag s2_rgt): SR 68's scene-2 ReGuide port (experiments/s2rg.py; rg.py's controller
class, UNCHANGED) with (rgt) tracking of the held mug during the attachment, and (rgtp) tracking + table-plane depth instead of the
rendered depth, against I and a replication of SR 68's rgp, on NEW states, layout O, tasks K1 K2 N1 N2, frozen pi0.5 (PRIMARY) and
pi0 (SECONDARY).

Facts (SR 50's task table): K1 white -> left (plate_1), K2 yellow-and-white -> right (plate_2), N1 white -> right, N2 yellow-and-white
-> left.

THE ARMS (everything not named here is s2rg.py's client, line for line: entities, round declaration, statistics
results/flawed/s2rg/rg_stats_s2.json, registration at the first gated step, re-detection at ProprioHeld onsets, Hp's held-mug
identity rule (nearest perceived mug within HELD_R = .08 m of the hand), the interaction state "the instructed mug is held", the
hand-back handling; prompt and images never changed):
  - rgp   SR 68's arm: OWLv2 (s2fix3_det sidecar) + rendered depth + camera calibration (s2fix3.Percept, constants unchanged).  While
          attached, the instructed mug follows the hand rigidly from the offset fixed at the attachment (no tracking).  Run through
          this file's client with tracking off (the decision path is s2rg.py's; only logging is added).
  - rgt   rgp + TRACKING: while the instructed mug is attached, OWLv2 re-detects the mugs every TRACK_K control steps, from the
          attachment step on.  Each detected mug box is localised by s2fix3.Percept.points (rendered depth, inner-half median) +
          CAL_HELD_DEPTH (the held-mug constant, fitted in `pcheck` on development states); the measurement is the localised box
          nearest the current estimate in xy, accepted only within GATE_XY = 3 cm of it (innovation gate).  An accepted
          measurement replaces the estimate's xy; its z stays the rigid estimate's (hand height + the offset fixed at the
          attachment).  The hand-frame offset is re-anchored to the result, so between measurements the mug follows the hand from
          the last one, and the carry (rg.py, unchanged) steers the TRACKED centre to the plate.  No accepted candidate: the
          estimate is kept (logged).
          DEVELOPMENT REVISION (after the smoke on SR 68's smoke states, before the prereg row): the first version took the
          measurement's full 3D point and Hp's hand-centred rule (nearest the hand within HELD_R).  In the smoke the measured
          height ran away (z error -7 to -70 cm within a carry: a low estimate makes the carry lift the hand, the next detection is
          worse) and wrong boxes were taken (xy jumps 5-9 cm); rgtp's held plane from the demonstrated grasp offset was 3 cm low
          from the attachment on (the VLA grasps lower than the demos), giving 4-5 cm xy bias.  Hence xy-only updates, the 3 cm
          gate, and rgtp's plane height from the rigid z.
  - rgtp  rgt with the rendered depth REPLACED by table-plane depth (no depth render anywhere in the arm): a resting mug = the camera
          ray through its box's bottom-centre cut with the table plane z = Z_TABLE, + CAL_MUG_PLANE; a plate = PLATE_METHOD's ray
          (box centre, or inner side edge + the plate radius) cut with the same plane, + CAL_PLATE_PLANE; the held mug = the ray
          through its box centre cut with the horizontal plane at the mug's mid height, which comes from proprioception (the rigid
          estimate's z, i.e. hand height + the offset fixed at the attachment, + the mesh mid height), + CAL_HELD_PLANE's xy
          (fitted in `pcheck` with the plane from the demonstrated grasp offset, z error .33 cm there); same gate and update as rgt.  Camera intrinsics and extrinsics (sc_pu.cam_model) stay: the remaining
          assumption.  Z_TABLE is a one-off calibration on development states: the median simulator height of the mugs' bottoms
          (body z + mesh bottom) at the settled start.
  - Every constant above comes from `pcheck` on DEVELOPMENT states (SR 68's states, seeds 980000 + k, none in this probe's 60),
    before the prereg row: results/flawed/s2rgt/pcheck.json.  PLATE_METHOD is chosen there by a fixed rule (the lower split-half
    median xy error).  TRACK_K = 5 (one OWLv2 call per 5 control steps; cost, not tuned).
END ARMS.

New states: LIBERO's sampler on K1's BDDL, seeds 990000 + k, k 0-59 (all main; none equal to an SR 50-68 state); branches seed_base
48000000 + 1000 k.  Smoke on SR 68's smoke states (results/flawed/s2rg/O, k 0-3; development).  Servers: SR 63's.

Modes (LIBERO venv, py3.8):
    pcheck [--out X]                      (EGL + the OWLv2 sidecar) -> results/flawed/s2rgt/pcheck.json
    states                                (EGL)   -> results/flawed/s2rgt/O/s2_living_<tid>.npy, branches_main.json
    client --arm rgp|rgt|rgtp <s2.py client args>  (needs the OWLv2 sidecar on $S2FIX3_DET_PORT)
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

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2rgt")
ROLL = os.path.join(OUT, "s2rgt_roll")
READ = os.path.join(OUT, "s2rgt_read.json")
PCHECK = os.path.join(CD, "pcheck.json")
STATS = SRG.STATS                                                           # SR 68's statistics, unchanged
DEV_STATES = os.path.join(SRG.CD, "O", "s2_living_0.npy")                   # SR 68's states: development only
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
MUGS, PLATES = s2.MUGS, s2.PLATES
BASE, BR_BASE, N_STATES = 990000, 48000000, 60
MAIN = list(range(0, 60))
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
HELD_R = F3.HELD_R
NI_MARGIN = F3.NI_MARGIN
FLOOR = F3.FLOOR
BAR = 0.50
EQ_BAND = 0.10
TRACK_K = 5
GATE_XY = 0.03                                                              # set after the development smoke (see the docstring)
RUNS = dict(pi05=("I", "rgp", "rgt", "rgtp"), pi0=("I", "rgp", "rgt", "rgtp"))
PRIMARY_MODEL = "pi05"
MAPPING = (
    "NEW states (LIBERO sampler on K1's BDDL, seeds 990000 + k, k = 0-59, none equal to an SR 50-68 state), branches seed_base "
    "48000000 + 1000 k; layout O; tasks K1, K2, N1, N2; client seed 7, --bind track.  Models: frozen pi0.5 (PRIMARY; pp_serve_purify "
    "--model pi05 --mode intact --comp= --seed 13) and frozen pi0 (SECONDARY; serve_guided --mode ours --seed 13, "
    "BACKBONE_CONFIG=pi0_libero).  Arms per model: I (s2.py client), rgp (SR 68's arm: ReGuide on OWLv2 + rendered depth + "
    "calibration, rigid held-mug estimate), rgt (rgp + tracking of the held mug every 5 control steps, rendered depth), rgtp (rgt "
    "with table-plane depth instead of rendered depth; camera calibration kept).  Success = the BDDL goal.  Matrices [task, state].  "
    "Paired bootstrap over states, tasks fixed (s2.boot2_states, B = 2000, numpy default_rng(0); CI = 2.5 / 97.5).  Registered "
    "contrasts per model: (V1) rgt - rgp on N1: TRACKING HELPS iff lo > 0; TRACKING HURTS iff hi < 0; else NO CLEAR DIFFERENCE.  (V2, "
    "the de-privilege cost) rgtp - rgt on N1+N2 and, separately, on K1+K2: EQUIVALENT iff the CI lies inside [-.10, +.10]; else COST "
    "iff hi < 0; else GAIN iff lo > 0; else INCONCLUSIVE.  (V3) rgtp - I: dN on N1+N2, dK on K1+K2: FIX WORKS iff dN lo > 0 AND dK lo "
    "> -.05; PARTIAL iff exactly one holds; FAILS iff neither (SR 62-68's rule); BAR: MEETS BAR iff rgtp's N1+N2 >= .50 AND dN lo > "
    "0, else BELOW BAR; descriptive beside it: the state-bootstrap CI of rgtp's N1+N2.  PRIMARY: V1-V3 on pi0.5.  SR 50's floor: a "
    "model is TESTABLE iff its I K1+K2 >= .70 on these states; a NOT TESTABLE model's labels read NOT TESTABLE with the rule's "
    "outcome descriptive.  MECHANISM (registered, descriptive): (m1) per arm, the share of N1 episodes whose white mug ends on no "
    "plate (final LIBERO On predicates), with rgt - rgp and rgtp - rgp paired over states; (m2) per arm (rgp, rgt, rgtp), the "
    "held-mug estimate against the simulator at each release of the attached instructed mug (ProprioHeld off): xy error, median and "
    "90th percentile, on N1 and over all tasks, and the share of releases whose estimate vs whose simulator mug lies within the "
    "controller's release region (dest_xy, omega); (m3) tracking: measurements, the share accepted, the measurement's and the "
    "rigid estimate's xy error against the simulator at the same step.  SECONDARY: the same FIX / BAR labels for rgp and rgt; rgt - "
    "rgp and rgtp - rgp on N1+N2 and K1+K2 (GAIN iff lo > 0, COST iff hi < 0, else NO CLEAR DIFFERENCE).  Descriptive: registration "
    "error of each arm against the simulator (target plate, other plate, instructed mug, every mug; side right), onset held-mug "
    "identity, ReGuide's transports, episode length and time-outs.  C1: the reference planner (s2.plan_episode via s2cue3.ref_eps, "
    "unchanged) on the main states; MET iff its N1-N2 success >= .80.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


short = SRG.short


def nm(n):
    return n if n in PLATES else short(n)


def load_pc():
    return json.load(open(PCHECK, encoding="utf-8"))


# ---------------------------------------------------------------- perception (client side; numpy + the sidecar)
def ray_plane(cm, u, v, Wd, z0):
    """Pixel (u = column, v = row) in the policy view -> the camera ray cut with the horizontal plane z = z0 (world)."""
    f, Rm, c = cm
    import sc_rv as RV
    S = RV.SIZE
    ct = (Wd - 1) - u                                                       # policy view = top-down mirrored in columns
    d = Rm @ np.array([(ct - S / 2.0) / f, (v - S / 2.0) / f, 1.0])
    t = (z0 - c[2]) / d[2]
    return c + t * d


def plate_plane(cm, b, Wd, z0, method, plate_r):
    """A plate box -> raw point on the plane z0.  center: the box centre's ray.  edge: the side edge (centre row) nearer the scene's
    centre (smaller |y|), + the plate radius away from it (the plates sit at the image borders, cut on their outer side)."""
    x0, y0, x1, y1 = b
    cy = 0.5 * (y0 + y1)
    if method == "center":
        return ray_plane(cm, 0.5 * (x0 + x1), cy, Wd, z0)
    a_, b_ = ray_plane(cm, x0, cy, Wd, z0), ray_plane(cm, x1, cy, Wd, z0)
    inner = a_ if abs(a_[1]) < abs(b_[1]) else b_
    p = np.array(inner, copy=True)
    p[1] += math.copysign(plate_r, inner[1])
    return p


def mug_mid(G, oe):
    return 0.5 * (G[oe]["top"] + G[oe]["bottom"])


def off_z(S, oe):
    o = S["obj"].get(oe) or S["obj"][S["inherit"][oe]]
    return float(o["off"][2])                                               # hand minus mug body at the demonstrated close


class DepthTrack(F3.Percept):
    """rgp / rgt: s2fix3.Percept unchanged (registration, onsets); held(): the held-mug measurement (rgt)."""

    def __init__(self, pc):
        super().__init__()
        self.pc = pc

    def held(self, env, obs, ee, oe, S, G, ref):
        Wd, rep = self.detect(obs)
        mugs, _ = F3.assign(rep["boxes"], rep["scores"])
        if not mugs:
            return None, 0
        D = self.PU.depth_std(env, dict(hide=[]), F3.CAM)
        cm = self.PU.cam_model(env.env.sim, F3.CAM)
        C = [self.points(D, cm, b, Wd) + np.array(self.pc["cal_held_depth"]) for b, _ in mugs.values()]
        return gate_xy(C, ref), len(C)


class PlanePercept(F3.Percept):
    """rgtp: OWLv2 boxes + camera calibration + the table plane; never renders depth."""

    def __init__(self, pc):
        super().__init__()
        self.pc = pc

    def scene(self, env, obs, qm=None, method=None):
        Wd, rep = self.detect(obs, None if qm is None else list(qm) + [F3.QP])
        mugs, plates = F3.assign(rep["boxes"], rep["scores"], method)
        cm = self.PU.cam_model(env.env.sim, F3.CAM)
        zt = float(self.pc["z_table"])
        out = dict(mugs={}, plates=[], secs=rep.get("secs"))
        for m, (b, s_) in mugs.items():
            p = ray_plane(cm, 0.5 * (b[0] + b[2]), b[3], Wd, zt)
            out["mugs"][m] = dict(raw=p.tolist(), pos=(p + np.array(self.pc["cal_mug_plane"])).tolist(), box=[round(v, 1) for v in b],
                                  score=round(s_, 4))
        for b, s_ in plates:
            p = plate_plane(cm, b, Wd, zt, self.pc["plate_method"], F3.PLATE_R)
            out["plates"].append(dict(raw=p.tolist(), pos=(p + np.array(self.pc["cal_plate_plane"])).tolist(),
                                      box=[round(v, 1) for v in b], score=round(s_, 4), img_x=0.5 * (b[0] + b[2])))
        out["left"], out["right"] = F3.sides(out["plates"])
        return out

    def held(self, env, obs, ee, oe, S, G, ref):
        Wd, rep = self.detect(obs)
        mugs, _ = F3.assign(rep["boxes"], rep["scores"])
        if not mugs:
            return None, 0
        cm = self.PU.cam_model(env.env.sim, F3.CAM)
        zb = float(ref[2])                                                  # the mug body's height from proprioception: hand
        C = []                                                              # height + the offset fixed at the attachment
        cxy = np.array(self.pc["cal_held_plane"], np.float64)
        for b, _ in mugs.values():
            p = ray_plane(cm, 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]), Wd, zb + mug_mid(G, oe))
            p[:2] += cxy[:2]
            C.append(p)
        return gate_xy(C, ref), len(C)


def gate_xy(C, ref):
    """Tracking update (revised after the development smoke): the localised mug point nearest the current estimate (xy); accepted
    only within GATE_XY of it; returns the estimate with its xy replaced (z stays the rigid estimate's), else None."""
    if not C:
        return None
    ref = np.asarray(ref, np.float64)
    d = [float(np.linalg.norm(np.asarray(p)[:2] - ref[:2])) for p in C]
    i = int(np.argmin(d))
    if d[i] > GATE_XY:
        return None
    out = np.array(ref, copy=True)
    out[:2] = np.asarray(C[i], np.float64)[:2]
    return out


def pick_near(C, ee):
    """Hp's held rule on a list of localised mug points: the nearest to the hand (xy) within HELD_R, else None."""
    if not C:
        return None
    d = [float(np.linalg.norm(np.asarray(p)[:2] - np.asarray(ee, np.float64)[:2])) for p in C]
    i = int(np.argmin(d))
    return np.asarray(C[i], np.float64) if d[i] < HELD_R else None


# ---------------------------------------------------------------- development check (LIBERO venv, EGL, sidecar)
def pcheck(a):
    """On DEVELOPMENT states (SR 68's, k 0-63, settled start; scripted holds of the white and the yellow-and-white mug on k 0-31):
    Z_TABLE, the plane constants (mugs, plates by method, held mug) and the rendered-depth held-mug constant, each the median
    simulator-minus-raw residual over all samples, with split-half (even / odd k) xy errors; PLATE_METHOD = the lower mean split-half
    median xy error.  Also: the onset identity rule under table-plane positions at the scripted hold."""
    out = os.path.abspath(a.out or PCHECK)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    import fa1b_common as FB
    import s2_tasks as T2
    from fa1g_plan import ctrl
    M_ = s2.meta()
    KP = json.load(open(os.path.join(s2.SD, "keypoints.json"), encoding="utf-8"))["tasks"]
    S = np.load(DEV_STATES)
    ST = json.load(open(STATS, encoding="utf-8"))
    G = ST["geometry"]
    P = F3.Percept()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))
    env.seed(7)
    rest_m, rest_p, held, zb_list = [], [], [], []
    for k in range(len(S)):
        env.reset()
        obs = env.set_init_state(S[k])
        for _ in range(s2.WAIT):
            obs, _, _, _ = env.step(T2.DUMMY)
        sim = {n: T2.positions(env, [n])[0] for n in MUGS + PLATES}
        for m in MUGS:
            zb_list.append(float(sim[m][2] + G[m]["bottom"]))
        Wd, rep = P.detect(obs)
        mugs, plates = F3.assign(rep["boxes"], rep["scores"])
        D = P.PU.depth_std(env, dict(hide=[]), F3.CAM)
        cm = P.PU.cam_model(env.env.sim, F3.CAM)
        for m, (b, _) in mugs.items():
            pd = P.points(D, cm, b, Wd) + np.array(F3.CAL["mug"])
            n = min(MUGS, key=lambda x: np.linalg.norm(sim[x][:2] - pd[:2]))   # identity-free pairing (pcheck2's)
            rest_m.append(dict(k=k, cm=cm, b=list(b), Wd=Wd, sim=sim[n], depth=pd, label_ok=(n == m)))
        for b, _ in plates:
            pd = P.plate_point(D, cm, b, Wd) + np.array(F3.CAL["plate"])
            n = min(PLATES, key=lambda x: np.linalg.norm(sim[x][:2] - pd[:2]))
            rest_p.append(dict(k=k, cm=cm, b=list(b), Wd=Wd, sim=sim[n], depth=pd))
        if k < 32:
            for m, half in ((W, "K1"), (Y, "K2")):
                obs2, ok = F3.scripted_hold(env, S[k], m, KP[half], 1000 + k, T2, FB, ctrl)
                if not ok:
                    continue
                ee = np.asarray(obs2["robot0_eef_pos"], np.float64)
                sim2 = {n: T2.positions(env, [n])[0] for n in MUGS}
                Wd2, rep2 = P.detect(obs2)
                mugs2, _ = F3.assign(rep2["boxes"], rep2["scores"])
                D2 = P.PU.depth_std(env, dict(hide=[]), F3.CAM)
                cm2 = P.PU.cam_model(env.env.sim, F3.CAM)
                raw_d = [P.points(D2, cm2, b, Wd2) for b, _ in mugs2.values()]
                boxes = [list(b) for b, _ in mugs2.values()]
                held.append(dict(k=k, m=m, ee=ee, sim=sim2[m], sim_all=sim2, raw_d=raw_d, boxes=boxes, cm=cm2, Wd=Wd2))
        if k % 8 == 0:
            print("[s2rgt pcheck] k%d done" % k, flush=True)
    env.close()
    zt = float(np.median(zb_list))

    def xy_err(e):
        n_ = np.linalg.norm(np.asarray(e)[:, :2], axis=1)
        return dict(med_cm=round(100 * float(np.median(n_)), 2), p90_cm=round(100 * float(np.percentile(n_, 90)), 2),
                    max_cm=round(100 * float(np.max(n_)), 2), n=int(len(n_)))

    def fit(rows, raw_of):
        """-> (constant, all-sample errors after it, split-half report)."""
        R_ = np.array([r["sim"] - raw_of(r) for r in rows])
        kk = np.array([r["k"] for r in rows])
        c_ = np.median(R_, 0)
        half = {}
        for fi, te in ((0, 1), (1, 0)):
            cf = np.median(R_[kk % 2 == fi], 0)
            half["fit%d_test%d" % (fi, te)] = xy_err(R_[kk % 2 == te] - cf)
        half["mean_med_cm"] = round(0.5 * (half["fit0_test1"]["med_cm"] + half["fit1_test0"]["med_cm"]), 3)
        return c_, xy_err(R_ - c_), half
    rep_ = dict(z_table=zt, z_table_n=len(zb_list), dev_states=os.path.relpath(DEV_STATES, ROOT))
    rep_["mug_depth_rest_err"] = xy_err([r["sim"] - r["depth"] for r in rest_m])
    rep_["plate_depth_rest_err"] = xy_err([r["sim"] - r["depth"] for r in rest_p])
    rep_["mug_label_ok"] = [int(sum(r["label_ok"] for r in rest_m)), len(rest_m)]
    c_m, e_m, h_m = fit(rest_m, lambda r: ray_plane(r["cm"], 0.5 * (r["b"][0] + r["b"][2]), r["b"][3], r["Wd"], zt))
    rep_.update(cal_mug_plane=c_m.tolist(), mug_plane_err=e_m, mug_plane_split=h_m)
    PM = {}
    for meth in ("center", "edge"):
        c_p, e_p, h_p = fit(rest_p, lambda r, meth=meth: plate_plane(r["cm"], r["b"], r["Wd"], zt, meth, F3.PLATE_R))
        PM[meth] = dict(cal=c_p.tolist(), err=e_p, split=h_p)
    best = min(PM, key=lambda m_: PM[m_]["split"]["mean_med_cm"])
    rep_.update(plate_methods=PM, plate_method=best, cal_plate_plane=PM[best]["cal"])
    # held mug: rendered depth (rgt) and the proprioceptive plane (rgtp); the box = the localised point nearest the hand
    hd, hp, ident = [], [], [0, 0]
    for h in held:
        rd = [p + np.array(F3.CAL["mug"]) for p in h["raw_d"]]
        pk = pick_near(rd, h["ee"])
        if pk is not None:
            i = int(np.argmin([np.linalg.norm(np.asarray(p)[:2] - h["ee"][:2]) for p in rd]))
            hd.append(dict(k=h["k"], sim=h["sim"], raw=np.asarray(h["raw_d"][i])))
        zb = float(h["ee"][2]) - off_z(ST, h["m"])
        rp = []
        for b in h["boxes"]:
            p = ray_plane(h["cm"], 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]), h["Wd"], zb + mug_mid(G, h["m"]))
            p[2] = zb
            rp.append(p)
        pk2 = pick_near(rp, h["ee"])
        if pk2 is not None:
            hp.append(dict(k=h["k"], sim=h["sim"], raw=pk2))
        # onset identity under table-plane positions (rgtp's onset rule, at a 6 cm scripted lift: a stress case)
        tp = {}
        for b in h["boxes"]:
            p = ray_plane(h["cm"], 0.5 * (b[0] + b[2]), b[3], h["Wd"], zt) + c_m
            n = min(MUGS, key=lambda x: np.linalg.norm(h["sim_all"][x][:2] - p[:2]))
            tp[n] = p
        g_ = pick_near(list(tp.values()), h["ee"])
        got = None if g_ is None else [n for n, p in tp.items() if p is g_ or np.allclose(p, g_)][0]
        ident[0] += int(got == h["m"])
        ident[1] += 1
    rep_["held_n"] = len(held)
    rep_["held_depth_restcal_err"] = xy_err([r["sim"] - (r["raw"] + np.array(F3.CAL["mug"])) for r in hd]) if hd else None
    c_hd, e_hd, h_hd = fit(hd, lambda r: r["raw"])
    c_hp, e_hp, h_hp = fit(hp, lambda r: r["raw"])
    rep_.update(cal_held_depth=c_hd.tolist(), held_depth_err=e_hd, held_depth_split=h_hd, cal_held_plane=c_hp.tolist(),
                held_plane_err=e_hp, held_plane_split=h_hp, held_plane_z_abs_med_cm=round(100 * float(np.median(np.abs(
                    np.array([r["sim"][2] - r["raw"][2] for r in hp]) - c_hp[2]))), 2),
                onset_identity_plane=ident, track_k=TRACK_K)
    for k_, v in rep_.items():
        print("[s2rgt pcheck] %s %s" % (k_, json.dumps(v, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep_, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(out + ".tmp", out)
    print("[s2rgt pcheck] wrote %s" % out, flush=True)


# ---------------------------------------------------------------- states (LIBERO venv, EGL)
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
         ("s2fix5", "O"), ("s2fix6", "O"), ("s2rg", "O"))]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("same as an SR 50-68 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    with open(os.path.join(CD, "branches_main.json"), "w", encoding="utf-8") as fh:
        json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE + 1000 * k, at=0, variant="s2") for k in MAIN], fh, indent=1)
    print("[s2rgt states] %d new states (seed %d + k), none equal to an SR 50-68 state; wrote %s" % (len(S), BASE, CD), flush=True)


# ---------------------------------------------------------------- client: arms rgp / rgt / rgtp (LIBERO venv)
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
    assert arm in ("rgp", "rgt", "rgtp"), arm
    TRACK = arm in ("rgt", "rgtp")
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
    G = S["geometry"]
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    PC = load_pc()
    PER = PlanePercept(PC) if arm == "rgtp" else DepthTrack(PC)
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
            self.t_trk = None
            self.det = ProprioHeld(X["tw"], X["lift"])
            self.diag = dict(chunk_len=[], n_q=0, reg=None, onsets=[], attached_rows=[], inactive=False, track=[], rel=[])
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
            sc = PER.scene(self._e, obs)
            self.mem = {m: np.asarray(v["pos"], np.float64) for m, v in sc["mugs"].items()}
            self.plates = {}
            if sc["left"] is not None:
                self.plates[PL] = np.asarray(sc["left"], np.float64)
            if sc["right"] is not None:
                self.plates[PR] = np.asarray(sc["right"], np.float64)
            err = {nm(n): round(100 * float(np.linalg.norm((p - self._pos(n))[:2])), 3)          # plates by full name (s2rg.py
                   for n, p in list(self.mem.items()) + list(self.plates.items())}               # keyed both plates "plate")
            zerr = {nm(n): round(100 * float(p[2] - self._pos(n)[2]), 3) for n, p in list(self.mem.items()) + list(self.plates.items())}
            tp, op = self._pos(DE), self._pos([p for p in PLATES if p != DE][0])
            side_ok = None
            if DE in self.plates:
                side_ok = bool(np.linalg.norm(self.plates[DE][:2] - tp[:2]) < np.linalg.norm(self.plates[DE][:2] - op[:2]))
            self.inactive = OE not in self.mem or DE not in self.plates
            self.diag["inactive"] = self.inactive
            self.diag["reg"] = dict(row=len(self.rows["held"]) - 1, err_xy_cm=err, err_z_cm=zerr,
                                    found=sorted(short(n) for n in list(self.mem) + list(self.plates)),
                                    tgt_err_xy_cm=err.get(nm(DE)), tgt_side_ok=side_ok, obj_err_xy_cm=err.get(nm(OE)),
                                    scores={short(m): v["score"] for m, v in sc["mugs"].items()})
            self.reg = True

        def track(self, obs, ee, Rm, row):
            """rgt / rgtp: re-detect the held mug; its measured point re-anchors the hand-frame offset."""
            self.t_trk = row
            pred = ee + Rm @ self.att
            meas, nc = PER.held(self._e, obs, ee, OE, S, G, pred)
            sq = self._pos(OE)
            rec = dict(row=row, n=nc, ok=meas is not None, pred_xy_cm=round(100 * float(np.linalg.norm((pred - sq)[:2])), 3))
            if meas is not None:
                rec.update(meas_xy_cm=round(100 * float(np.linalg.norm((meas - sq)[:2])), 3),
                           meas_z_cm=round(100 * float(meas[2] - sq[2]), 3),
                           jump_xy_cm=round(100 * float(np.linalg.norm((meas - pred)[:2])), 3))
                self.att = Rm.T @ (meas - ee)
            self.diag["track"].append(rec)

        def log_release(self, row, q):
            sq = self._pos(OE)
            pxy, _ = self.rg.dest_xy(dict(pos=dict(self.plates), yaw=self._yaw0()))
            om = float(self.rg.ds["omega"])
            self.diag["rel"].append(dict(row=row, est=np.round(q, 4).tolist(), sim=np.round(sq, 4).tolist(),
                                         err_xy_cm=round(100 * float(np.linalg.norm((q - sq)[:2])), 3),
                                         err_z_cm=round(100 * float(q[2] - sq[2]), 3),
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
            on = bool(self.det.step(grip, self.cmd, ee[2]))
            onset = on and not was
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
                self.att = Rm.T @ (self.mem[OE] - ee)                        # rgp: rigid offset in the hand frame, fixed here
                self.diag["attached_rows"].append(row)
                self.t_trk = None
            if TRACK and held and self.att is not None and (self.t_trk is None or row - self.t_trk >= TRACK_K):
                self.track(obs, ee, Rm, row)                                 # rgt / rgtp: from the attachment step, every TRACK_K
            if not held and self.att is not None:
                q_ = ee + Rm @ self.att
                self.log_release(row, q_)
                self.mem[OE] = q_                                            # the released mug stays where the hand left it
                self.att = None
            q = ee + Rm @ self.att if self.att is not None else self.mem[OE]
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
                                   detector=PER.model, cal=F3.CAL, side_rule=F3.SIDE_RULE, pcheck=PC, track=TRACK, track_k=TRACK_K,
                                   eps=SIDE),
                              fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
                os.replace(tmp, SIDEF)
                tr = rec["track"]
                print("[s2rgt] %s %s trial %s: inactive %s reg %s | onsets %s | track %d/%d | rel %s | transports %d releases %d "
                      "steps_rg %d | events %s" % (
                          arm, TASK["key"], rec["trial"], rec["inactive"], None if rec["reg"] is None else rec["reg"].get("tgt_err_xy_cm"),
                          [(o_["row"], o_["sim_held"], o_["held"], o_["attached"]) for o_ in rec["onsets"][:4]],
                          sum(t_["ok"] for t_ in tr), len(tr), [(r_["row"], r_["err_xy_cm"], r_["sim_in"]) for r_ in rec["rel"][:4]],
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
    print("[s2rgt] arm %s track %s k %d task %s %r roles %s -> %s max_steps %d stats %s pcheck %s detector %s" % (
        arm, TRACK, TRACK_K, TASK["key"], TASK["language"], OE, DE, ms, STATS,
        json.dumps({k_: PC[k_] for k_ in ("z_table", "plate_method", "cal_mug_plane", "cal_plate_plane", "cal_held_depth", "cal_held_plane")}),
        PER.model), flush=True)
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
def eq_label(d):
    if d[1] >= -EQ_BAND and d[2] <= EQ_BAND:
        return "EQUIVALENT"
    if d[2] < 0:
        return "COST"
    if d[1] > 0:
        return "GAIN"
    return "INCONCLUSIVE"


def v1_label(d):
    return F3.diff_label(d, "TRACKING HELPS", "TRACKING HURTS")


def rel_stats(side, tasks):
    """Release events of the attached instructed mug: xy error of the estimate against the simulator; region membership."""
    E = [r for c in tasks for e in side[c] for r in e.get("rel", [])]
    last = [e["rel"][-1] for c in tasks for e in side[c] if e.get("rel")]
    if not E:
        return dict(n=0)
    x = np.array([r["err_xy_cm"] for r in E])
    xl = np.array([r["err_xy_cm"] for r in last])
    return dict(n=len(E), med_cm=float(np.median(x)), p90_cm=float(np.percentile(x, 90)), mean_cm=float(x.mean()),
                est_in=float(np.mean([r["est_in"] for r in E])), sim_in=float(np.mean([r["sim_in"] for r in E])),
                est_in_sim_out=int(sum(r["est_in"] and not r["sim_in"] for r in E)),
                last_n=len(last), last_med_cm=float(np.median(xl)), last_p90_cm=float(np.percentile(xl, 90)),
                last_sim_in=float(np.mean([r["sim_in"] for r in last])))


def track_stats(side, tasks):
    T_ = [t for c in tasks for e in side[c] for t in e.get("track", [])]
    if not T_:
        return dict(n=0)
    ok = [t for t in T_ if t["ok"]]
    return dict(n=len(T_), accepted=float(len(ok) / len(T_)), episodes_tracked=int(sum(bool(e.get("track")) for c in tasks for e in side[c])),
                meas_xy_med_cm=float(np.median([t["meas_xy_cm"] for t in ok])) if ok else None,
                meas_xy_p90_cm=float(np.percentile([t["meas_xy_cm"] for t in ok], 90)) if ok else None,
                pred_xy_med_cm=float(np.median([t["pred_xy_cm"] for t in T_])),
                pred_xy_p90_cm=float(np.percentile([t["pred_xy_cm"] for t in T_], 90)),
                jump_xy_med_cm=float(np.median([t["jump_xy_cm"] for t in ok])) if ok else None)


def reg_err(side):
    out = collections.defaultdict(list)
    for c, E in side.items():
        for e in E:
            rg_ = e.get("reg")
            if not rg_:
                continue
            for n, v in rg_["err_xy_cm"].items():
                out["all_mugs" if n in [short(m) for m in MUGS] else "all_plates"].append(v)
            if rg_.get("tgt_err_xy_cm") is not None:
                out["target_plate"].append(rg_["tgt_err_xy_cm"])
            if rg_.get("obj_err_xy_cm") is not None:
                out["instructed_mug"].append(rg_["obj_err_xy_cm"])
            out["side_ok"].append(float(bool(rg_.get("tgt_side_ok"))))
            for n, v in rg_.get("err_z_cm", {}).items():
                out["z_abs_mugs" if n in [short(m) for m in MUGS] else "z_abs_plates"].append(abs(v))
    res = {}
    for k_, v in out.items():
        v = np.array(v, float)
        res[k_] = dict(n=len(v), mean=float(v.mean())) if k_ == "side_ok" else dict(n=len(v), med=float(np.median(v)),
                                                                                       p90=float(np.percentile(v, 90)), max=float(v.max()))
    return res


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
                        tr = h.get("track", [])
                        x = dict(onsets=[(o_["row"], o_["sim_held"], o_["held"], o_["attached"]) for o_ in h["onsets"][:4]],
                                 starts=[(ev["t"], ev["what"], ev.get("why")) for ev in h["events"] if ev["kind"] == "start"][:6],
                                 reg=None if not h.get("reg") else h["reg"].get("err_xy_cm"),
                                 track=[sum(t_["ok"] for t_ in tr), len(tr),
                                        float(np.median([t_["meas_xy_cm"] for t_ in tr if t_["ok"]])) if any(t_["ok"] for t_ in tr) else None,
                                        float(np.median([t_["pred_xy_cm"] for t_ in tr])) if tr else None],
                                 rel=[(r_["row"], r_["err_xy_cm"], r_["est_in"], r_["sim_in"]) for r_ in h.get("rel", [])][:4])
                    print("[s2rgt summary] %s %s %s k%d success %s steps %s on %s | %s" % (
                        model, arm, t["key"], k, r["success"], e["steps"], on, json.dumps(x)), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_") and x.endswith(".json")):
        Rj = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2rgt summary] %s: %s" % (f, [(e["trial"], e["success"]) for e in Rj["eps"]]), flush=True)


def read(a):
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD, mism = [int(x) for x in a.ks.split(",")], a.dir, "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_rgt"]
        assert len(rows) == 1, "expected exactly one s2_rgt prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD = MAIN, ROLL
    res = dict(states=ks, mapping=MAPPING, code_mismatch=mism, models={})
    ALLT = s2.KNOWN + s2.NEW
    one = np.ones(len(ks))
    TRK = ("rgp", "rgt", "rgtp")
    for model, arms in RUNS.items():
        runs = {arm: F3.load_run(RD, model, arm, ks) for arm in arms}
        sides = {arm: SRG.load_rg(RD, model, arm, ks) for arm in arms if arm in TRK}
        ok = {arm: not isinstance(v, str) for arm, v in runs.items()}
        for arm in TRK:
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
            for arm in TRK:
                if not ok.get(arm):
                    continue
                dN, dK = dd(arm, "I", s2.NEW), dd(arm, "I", s2.KNOWN)
                R_["deltas"]["%s-I N" % arm], R_["deltas"]["%s-I K" % arm] = dN, dK
                lab = F3.fix_label(dN, dK)
                R_["verdicts"]["%s FIX label (rule)" % arm] = lab
                R_["verdicts"]["%s FIX label (registered)" % arm] = SRG.gate(lab, tst)
                R_["verdicts"]["%s-I K non-inferiority" % arm] = F3.ni_label(dK)
                xN = float(F3.mat(runs[arm], s2.NEW).mean())
                bar = "MEETS BAR" if (dN[1] > 0 and xN >= BAR) else "BELOW BAR"
                R_["verdicts"]["%s BAR (rule)" % arm] = bar
                R_["verdicts"]["%s BAR (registered)" % arm] = SRG.gate(bar, tst)
                ci = s2.boot2_states(F3.mat(runs[arm], s2.NEW), np.zeros((2, len(ks))))
                R_.setdefault("bar", {})[arm] = dict(new=xN, bar=BAR, new_ci=ci, lower_bound_ge_bar=bool(ci[1] >= BAR))
        if ok.get("rgt") and ok.get("rgp"):
            d1 = dd("rgt", "rgp", ("N1",))
            R_["deltas"]["V1 rgt-rgp N1"] = d1
            R_["verdicts"]["V1 rgt-rgp N1 (rule)"] = v1_label(d1)
            R_["verdicts"]["V1 rgt-rgp N1 (registered)"] = SRG.gate(v1_label(d1), bool(tst))
        if ok.get("rgtp") and ok.get("rgt"):
            for nm, tasks in (("N1+N2", s2.NEW), ("K1+K2", s2.KNOWN)):
                d2 = dd("rgtp", "rgt", tasks)
                R_["deltas"]["V2 rgtp-rgt %s" % nm] = d2
                R_["verdicts"]["V2 rgtp-rgt %s (rule)" % nm] = eq_label(d2)
                R_["verdicts"]["V2 rgtp-rgt %s (registered)" % nm] = SRG.gate(eq_label(d2), bool(tst))
        if "rgtp FIX label (registered)" in R_["verdicts"]:
            R_["verdicts"]["V3 rgtp-I (registered)"] = "%s; %s" % (R_["verdicts"]["rgtp FIX label (registered)"],
                                                                   R_["verdicts"]["rgtp BAR (registered)"])
        for x, y in (("rgt", "rgp"), ("rgtp", "rgp")):
            if ok.get(x) and ok.get(y):
                for nm, tasks in (("N", s2.NEW), ("K", s2.KNOWN)):
                    d_ = dd(x, y, tasks)
                    R_["deltas"]["%s-%s %s" % (x, y, nm)] = d_
                    R_["verdicts"]["%s-%s %s (secondary)" % (x, y, nm)] = F3.diff_label(d_, "GAIN", "COST")
                R_["deltas"]["%s-%s per task" % (x, y)] = {c: s2.boot2_states(runs[x]["succ"][c][None], runs[y]["succ"][c][None]) for c in ALLT}
        # mechanism (m1): N1 white mug on no plate
        NOP = {}
        for arm in arms:
            if ok.get(arm):
                ind = np.array([float(s2cue.plate_of(r["on"], W) is None) for r in runs[arm]["rows"]["N1"]])
                NOP[arm] = dict(share=s2cue.boot_share(ind, one), count=int(ind.sum()), ind=ind.tolist())
        for x in ("rgt", "rgtp"):
            if x in NOP and "rgp" in NOP:
                NOP["%s-rgp" % x] = s2.boot2_states(np.array([NOP[x]["ind"]]), np.array([NOP["rgp"]["ind"]]))
        R_["mechanism"]["N1_white_no_plate"] = NOP
        print("== %s MECHANISM N1 white on no plate: %s" % (model, json.dumps({k_: (v if not isinstance(v, dict) else
                                                                                     {kk: vv for kk, vv in v.items() if kk != "ind"})
                                                                                for k_, v in NOP.items()})), flush=True)
        # (m2) release error, (m3) tracking
        for arm in TRK:
            if ok.get(arm):
                R_["mechanism"]["release_%s" % arm] = dict(N1=rel_stats(sides[arm], ("N1",)), all=rel_stats(sides[arm], ALLT))
                R_["mechanism"]["track_%s" % arm] = dict(N1=track_stats(sides[arm], ("N1",)), all=track_stats(sides[arm], ALLT))
                R_["mechanism"]["reguide_%s" % arm] = SRG.rg_diag(sides[arm], runs[arm])
                R_["perception_%s" % arm] = SRG.perception(sides[arm])
                R_["perception_%s" % arm]["errors"] = reg_err(sides[arm])
                print("== %s %s release: %s" % (model, arm, json.dumps(R_["mechanism"]["release_%s" % arm])), flush=True)
                print("== %s %s tracking: %s" % (model, arm, json.dumps(R_["mechanism"]["track_%s" % arm])), flush=True)
                print("== %s %s ReGuide: %s" % (model, arm, json.dumps(R_["mechanism"]["reguide_%s" % arm])), flush=True)
                print("== %s %s perception: %s" % (model, arm, json.dumps(R_["perception_%s" % arm])), flush=True)
        for arm in arms:
            if ok.get(arm):
                st_ = {t["key"]: F6.ep_steps(RD, model, arm, t["tid"]) for t in s2.meta()["tasks"][:4]}
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
    print("[s2rgt read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pcheck", "states", "ref", "summary", "read"))
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
    dict(pcheck=pcheck, states=states, ref=ref, summary=lambda a: summary(a.dir), read=read)[A.mode](A)

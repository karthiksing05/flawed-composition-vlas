"""S2CUE3 (docs/SCAFFOLD_RESULTS.md 59; m3_prereg tag s2_cue3): in scene 2 (LIBERO-10 LIVING_ROOM_SCENE5) SR 57 and SR 58 found that
the place follows the held mug's START SIDE (swapped starts: 287/287 on pi0.5, .993 on pi0) when position is informative, and identity
when it is not (central starts).  In the trained layout the start side is also the side of the NEARER PLATE, and no edit to the mugs
alone separates the two (SR 58: with the plates where they are, "side" and "nearer plate" are the same function; mirroring the plates
renders pixel-identical).  This probe moves the PLATES, asymmetrically, so that the plate nearest the left-starting mug is the RIGHT
plate (it stands right of the workspace midline y = 0, the mug left of it).

Manipulations (initial state only; prompts, servers, client, launcher, client seed, --bind track as SR 57 / 58):
    Sm   both plates moved (shifted plates): the left plate (plate_1) to (XL, YL), back toward the camera on the left (YL < 0), the
         right plate (plate_2) to (XR, YR), in to just right of the midline (YR > 0); every state gets the same plate layout; the
         mugs stand where they were sampled.  The white mug (sampled left of centre) is then nearer the RIGHT plate.
         (A rigid shift of both plates toward one side was tried first: a plate shifted by >= .18 m leaves the table and the
         agentview image, which reaches only |y| ~ .30 at the plates' depth; see SR 59.)
    PSm  as Sm, after SR 57's swap of the white and the yellow-and-white mug's xy (each keeps its own height and orientation).
         The yellow-and-white mug then starts left of centre, nearer the RIGHT plate (its trained plate).
    So, for the left-starting (disagreeing) mug, START SIDE says left in both, NEARER PLATE says right in both, and IDENTITY (the
    mug's trained plate) says left under Sm and right under PSm.  The pair separates start side from nearer plate (each run) and both
    from identity (across the two runs).
(XL, YL, XR, YR) is chosen from CANDS by the rule in PICK_RULE, before any rollout (no policy is involved in the scan).

States.  NEW states (not SR 50 / 57 / 58's): LIBERO's sampler on K1's BDDL, np.random.seed = env.seed = 835000 + k, k = 0..43
(0-3 smoke, 4-43 main); branches trial = k, seed_base 27000000 + 1000 k.  Written to results/flawed/s2cue3/O (unmanipulated; not
rolled out) by `states`.

Modes (LIBERO venv, py3.8):
    states                 (EGL)  -> results/flawed/s2cue3/O/s2_living_<tid>.npy, branches_{smoke,main}.json
    probe                  (EGL)  table / camera geometry and a one-plate placement map (stdout)
    scan_orig              (EGL)  -> SCAN/{orig.json, frames_O.npz}: the unmanipulated new states, settled and rendered
    scan --ci I            (EGL)  -> SCAN/{geom_<ci>.json, states_<m>_<ci>.npy, frames_<ci>.npz}: one candidate, both manipulations
    scan_ref --m M --ci I --tid T (CPU) -> SCAN/ref_<m>_<ci>_<tid>.json (reference planner on states 0-7; ci -1 = unmanipulated O)
    scan_pick              (CPU)  -> SCAN/pick.json (PICK_RULE)
    build                  (CPU)  -> results/flawed/s2cue3/{Sm,PSm}/s2_living_<tid>.npy, s2cue3_states.json (the picked candidate)
    ref --manip M --tid T --branches B --out-dir D   (CPU)
    summary DIR            (smoke)
    read [DIR --selftest --out X --ref-dir D]        (CPU)
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
CD = os.path.join(OUT, "s2cue3")
ROLL = os.path.join(OUT, "s2cue3_roll")
READ = os.path.join(OUT, "s2cue3_read.json")
STATES_JSON = os.path.join(CD, "s2cue3_states.json")
SCAN = "/coc/flash10/ksingara3/.claude_tmp/pp/s2cue3_scan2"                # s2cue3_scan: the cancelled first scans (kept)
SHEET = "/coc/flash10/ksingara3/.claude_tmp/pp/s2cue3_sheet.png"
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
BASE3, BR_BASE3, N_STATES = 835000, 27000000, 44
SMOKE = list(range(0, 4))
MAIN = list(range(4, 44))
REF_SCAN_KS = list(range(0, 8))
MANIPS = ("Sm", "PSm")
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
# (XL, YL, XR, YR): world xy of plate_1 (left) and plate_2 (right); from the probe map (s2cue3_probe_4712585.out): the image shows a
# plate at |y| <= ~.30 for x in [-.10, .25]; a plate needs >= ~.12 m centre distance to a mug
# (the first grid, xl in (.15, .20, .25), xr in (.10, .13): xl = .15 or xr = .13 gave SEP on <= .34 of states; job 4712895)
CANDS = [(xl, yl, 0.10, yr) for xl in (0.20, 0.25) for yl in (-0.27, -0.24) for yr in (0.03, 0.06)]
VIS_MIN, CLEAR_SLACK = 0.60, 0.01
PICK_RULE = (
    "Candidate (XL, YL, XR, YR) in CANDS (plate_1 placed at (XL, YL), plate_2 at (XR, YR), each keeping its own height and "
    "orientation).  For each manipulation m in (Sm, PSm) and each of the 44 new states "
    "k: GEOM_OK iff as placed no plate touches a mug or the other plate; after the 10 settle steps the drift of every movable is "
    "< 1 cm in xy (SR 58's rule; the plates' z drop is at most 1 cm more than on the unmanipulated state), every mug's tilt < 10 deg, no mug is On a plate, no plate touches a mug, the smallest mug-plate centre distance is >= D0 - "
    ".01 m (D0 = the smallest mug-plate centre xy distance over the 44 unmanipulated states), each plate's agentview pixel count is "
    ">= .60 of its unmanipulated count on the same state, and the left plate's y < 0 < the right plate's y; SEP iff the mug that starts "
    "left of the midline (y < -.05) is nearer the right plate by >= .02 m.  FEASIBLE iff GEOM_OK on all 44 states for both m and the "
    "reference planner (s2.plan_episode, unchanged) succeeds on >= .50 of the 16 episodes per plate (states 0-7; tasks K1 + N2 for the "
    "left plate, K2 + N1 for the right) for both m.  Score = min over m of the share of the 44 states with SEP.  PICK = the feasible "
    "candidate with the highest score; ties (within .01) -> the larger mean SEP margin (d_left - d_right over the 44 states, min over "
    "m).  None feasible -> no pick (stop).")
# (model, manipulation, front-end arm, role); every run covers tids 0-3
RUNS = (("pi05", "Sm", "I", "PRIMARY"), ("pi05", "PSm", "I", "PRIMARY"), ("pi0", "Sm", "I", "PRIMARY"), ("pi0", "PSm", "I", "PRIMARY"),
        ("pi05", "Sm", "tv", "SECONDARY"), ("pi05", "PSm", "tv", "SECONDARY"), ("pi0", "Sm", "tv", "SECONDARY"),
        ("pi0", "PSm", "tv", "SECONDARY"))
H2H_MIN = 20
MAPPING = (
    "Models frozen: pi0.5 (pp_serve_purify --model pi05 --mode intact --comp= --seed 13) and pi0 (serve_guided --mode ours --seed 13, "
    "BACKBONE_CONFIG=pi0_libero), s2.py client unchanged run by s2_roll.sh unchanged (client seed 7, --bind track), FRESH = the "
    "manipulated NEW states (results/flawed/s2cue3/{Sm,PSm}; LIBERO sampler seeds 835000 + k), branches_main.json (k = 4-43, "
    "seed_base 27000000 + 1000 k).  Runs: pi0.5 Sm I, PSm I, pi0 Sm I, PSm I (PRIMARY); pi0.5 Sm tv, PSm tv, pi0 Sm tv, PSm tv "
    "(SECONDARY); tasks K1, K2, N1, N2 each (40 episodes).  Per episode: HELD mug = libero_client's 'bound' if it is the white or the "
    "yellow-and-white mug; PLACEMENT = the held mug ends On exactly one plate (final LIBERO predicates).  Plates are scored by SIDE (the "
    "plate with the smaller y = plate_1 = the left plate; the plates keep their order under Sm / PSm).  Start positions = xy after the "
    "10 settle steps (s2cue3_states.json).  Cues for the held mug's plate (s2cue2.cue_preds, unchanged): START SIDE = the side of the "
    "workspace midline y = 0 the mug starts on (undefined iff |y| < .05); NEARER PLATE = the plate nearer the mug's start (undefined "
    "iff |d_left - d_right| < .02 m); PLATE OBJECT / IDENTITY = the mug's trained plate (white -> left, yellow-and-white -> right); "
    "INSTRUCTION = the named side, defined only when the named mug is held; MUG LEFT/RIGHT ORDER; DEPTH ORDER (smaller x -> left); "
    "STEP = left.  Per run and cue (s2cue2.label_run, unchanged): share f over placements where the cue is defined, pooled over the 4 "
    "tasks, CI = 2.5 / 97.5 percentiles of B = 2000 bootstrap draws of the 40 states (default_rng(0)); ROUTES iff f >= .70 and lo > "
    ".50; RULED OUT iff f <= .30 and hi < .50; else UNRESOLVED; UNTESTABLE iff < 20 placements; a cue that disagrees with another "
    "co-defined cue (>= 10 co-defined) on < 10 placements is INSEPARABLE from it.  HEAD-TO-HEAD per run (the registered start-side vs "
    "nearer-plate test): over placements where START SIDE and NEARER PLATE are both defined and DISAGREE, s = share on the start-side "
    "plate (1 - s = share on the nearer plate), same bootstrap; START SIDE ROUTES iff s >= .70 and lo > .50; NEARER PLATE ROUTES iff "
    "s <= .30 and hi < .50; else UNRESOLVED; UNTESTABLE iff < 20 such placements.  PER MODEL (arm I; also reported for tv): Sm and PSm "
    "head-to-head both START SIDE ROUTES -> START SIDE ROUTES (nearer plate and identity ruled out as the router); both NEARER PLATE "
    "ROUTES -> NEARER PLATE ROUTES; Sm START SIDE and PSm NEARER PLATE -> IDENTITY (on the disagreeing mug, identity = start side under "
    "Sm and = nearer plate under PSm); Sm NEARER and PSm START -> MIXED (no single cue); otherwise UNRESOLVED.  By construction START "
    "SIDE = LEFT/RIGHT ORDER = DEPTH ORDER under both, and = IDENTITY under Sm.  Success: per task, the named mug On the named plate "
    "(= the BDDL goal; descriptive, new states, no pairing).  C1 per manipulation: the reference planner (s2.plan_episode, unchanged) "
    "on the same 40 states; MET iff its N1-N2 success >= .80.  Descriptive: first-held shares, held x destination tables, image-side "
    "check of the mugs' start (agentview pixel centroid against the image midline).")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def state_dir(m):
    return os.path.join(CD, m)


# ---------------------------------------------------------------- simulator tools (LIBERO venv)
class Scene:
    def __init__(self):
        import s2_tasks as T2
        self.T2 = T2
        M_ = s2.meta()
        self.env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))
        self.env.seed(7)
        self.E = self.env.env
        self.jn = {n: self.E.objects_dict[n].joints[-1] for n in s2.MOVABLE}
        self._g = {}

    def pos(self, n):
        return np.array(self.E.sim.data.body_xpos[self.E.obj_body_id[n]], np.float64)

    def geoms(self, n):
        if n not in self._g:
            mdl = self.E.sim.model
            b0 = self.E.obj_body_id[n]
            bodies = set()
            for b in range(mdl.nbody):
                c = b
                while c > 0 and c != b0:
                    c = mdl.body_parentid[c]
                if c == b0:
                    bodies.add(b)
            self._g[n] = set(int(g) for g in np.where(np.isin(mdl.geom_bodyid, list(bodies)))[0])
        return self._g[n]

    def touching(self, x, y):
        gx, gy = self.geoms(x), self.geoms(y)
        d = self.E.sim.data
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in gx and c.geom2 in gy) or (c.geom1 in gy and c.geom2 in gx):
                return True
        return False

    def plate_contacts(self):
        return [int(self.touching(p, n)) for p in (PL, PR) for n in (W, Y, R)] + [int(self.touching(PL, PR))]

    def qpos(self, n):
        return np.array(self.E.sim.data.get_joint_qpos(self.jn[n]))

    def setq(self, n, q):
        self.E.sim.data.set_joint_qpos(self.jn[n], q)
        self.E.sim.data.set_joint_qvel(self.jn[n], np.zeros(6))

    def load(self, s):
        self.env.reset()
        return self.env.set_init_state(s)

    def settle(self, obs):
        for _ in range(s2.WAIT):
            obs, _, _, _ = self.env.step(self.T2.DUMMY)
        return obs

    def tilt(self, n):
        w, x, y, z = self.E.sim.data.body_xquat[self.E.obj_body_id[n]]
        return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))

    def seg(self):
        import sc_rv as RV
        T = RV.entity_tables(self.env)
        ent = RV.seg_px(self.env, T, "agentview")
        names = list(T["names"])
        out = {}
        for n in s2.MOVABLE:
            msk = ent == names.index(n)
            c = int(msk.sum())
            out[n] = dict(px=c, col=float(np.where(msk)[1].mean()) if c else None)
        return out

    def make(self, s, m, cand):
        """Unmanipulated state s -> manipulated state (no settling): PSm swaps the two mugs' xy first; plate_1 -> (XL, YL), plate_2 ->
        (XR, YR)."""
        xl, yl, xr, yr = cand
        self.env.reset()
        self.env.set_init_state(s)
        if m == "PSm":
            qw, qy = self.qpos(W), self.qpos(Y)
            nw, ny = qw.copy(), qy.copy()
            nw[:2], ny[:2] = qy[:2], qw[:2]
            self.setq(W, nw)
            self.setq(Y, ny)
        for p, xy in ((PL, (xl, yl)), (PR, (xr, yr))):
            q = self.qpos(p)
            q[:2] = xy
            self.setq(p, q)
        self.E.sim.forward()
        placed = self.plate_contacts()
        return np.array(self.E.sim.get_state().flatten()), placed

    def check(self, sp, frame=False):
        obs = self.load(sp)
        a0 = {n: self.pos(n) for n in s2.MOVABLE}
        obs = self.settle(obs)
        p = {n: self.pos(n) for n in s2.MOVABLE}
        xy = {n: p[n][:2].tolist() for n in s2.MOVABLE}
        snap = dict(xy=xy, drift3=float(max(np.linalg.norm(p[n] - a0[n]) for n in s2.MOVABLE)),
                    drift_xy=float(max(np.linalg.norm((p[n] - a0[n])[:2]) for n in s2.MOVABLE)),
                    drift_z_plates=float(max(abs(p[n][2] - a0[n][2]) for n in (PL, PR))),
                    tilt=max(self.tilt(n) for n in s2.MUGS),
                    on=[int(bool(self.E._eval_predicate(["on", m_, p_]))) for m_ in s2.MUGS for p_ in s2.PLATES],
                    contacts_settled=self.plate_contacts(),
                    dmin=float(min(np.linalg.norm(p[a][:2] - p[b][:2]) for a in s2.MUGS for b in (PL, PR))),
                    seg=self.seg())
        img = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1]) if frame else None
        return snap, img


def sep_of(snap):
    """The mug that starts left of the midline (y < -.05): is it nearer the right plate by >= .02 m?  -> (mug, d_left - d_right)."""
    xy = {n: np.array(v) for n, v in snap["xy"].items()}
    left = [m for m in (W, Y) if xy[m][1] < -s2cue2.TOL_SIDE]
    if len(left) != 1:
        return None, None
    m = left[0]
    return m, float(np.linalg.norm(xy[m] - xy[PL]) - np.linalg.norm(xy[m] - xy[PR]))


def geom_ok(snap, placed, orig, d0):
    vis = {p: (snap["seg"][p]["px"] / max(orig["seg"][p]["px"], 1)) for p in (PL, PR)}
    snap["vis_ratio"] = vis
    return bool(not any(placed) and snap["drift_xy"] < 0.01 and snap["drift_z_plates"] < orig["drift_z_plates"] + 0.01 and snap["tilt"] < 10 and not any(snap["on"])
                and not any(snap["contacts_settled"]) and snap["dmin"] >= d0 - CLEAR_SLACK and min(vis.values()) >= VIS_MIN
                and snap["xy"][PL][1] < 0 < snap["xy"][PR][1])


# ---------------------------------------------------------------- states (new) and the feasibility scan
def states(a):
    import s2_tasks as T2
    od = os.path.join(CD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))

    def draw(k):                                                            # s2_tasks.build's rule with a new base
        np.random.seed(BASE3 + k)
        env.seed(BASE3 + k)
        env.reset()
        return env.sim.get_state().flatten()
    S = np.stack([draw(k) for k in range(N_STATES)])
    env.close()
    old = np.load(os.path.join(s2.SD, "states", "%s_0.npy" % s2.SUITE))
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for j in range(len(old)):
            assert not np.allclose(S[i], old[j]), ("same as an SR 50 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    for ph, ks in (("smoke", SMOKE), ("main", MAIN)):
        with open(os.path.join(CD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE3 + 1000 * k, at=0, variant="s2") for k in ks], fh, indent=1)
    print("[s2cue3 states] %d new states (seed %d + k), none equal to SR 50's %d; wrote %s" % (len(S), BASE3, len(old), CD), flush=True)


def scan_orig(a):
    os.makedirs(SCAN, exist_ok=True)
    oj = os.path.join(SCAN, "orig.json")
    assert not os.path.exists(oj), "exists: " + oj
    S = np.load(os.path.join(CD, "O", "%s_0.npy" % s2.SUITE))
    sc = Scene()
    orig, frames, names = [], [], []
    for k in range(len(S)):
        snap, img = sc.check(S[k], frame=k < 4)
        orig.append(snap)
        if k < 4:
            frames.append(img)
            names.append("O_k%d" % k)
    d0 = min(o["dmin"] for o in orig)
    # image-side check: does the workspace midline y = 0 split the mugs as the agentview image's midline does?
    side = []
    for o in orig:
        for m in (W, Y):
            c = o["seg"][m]["col"]
            if c is not None:
                side.append((o["xy"][m][1] < 0, c > 127.5))
    agree = float(np.mean([x == y for x, y in side]))
    print("[s2cue3 scan] D0 %.4f m; mug world side (y<0) vs image side (col>127.5) agreement %.3f of %d (1 or 0 = consistent)" % (
        d0, agree, len(side)), flush=True)
    print("[s2cue3 scan] unmanipulated settle: drift3 max %.4f, drift_xy max %.4f, plates' z drop max %.4f, tilt max %.2f" % (
        max(o["drift3"] for o in orig), max(o["drift_xy"] for o in orig), max(o["drift_z_plates"] for o in orig),
        max(o["tilt"] for o in orig)), flush=True)
    sc.env.close()
    with open(oj, "w", encoding="utf-8") as fh:
        json.dump(dict(d0=d0, image_side_agreement=agree, orig=orig), fh, indent=1)
    np.savez_compressed(os.path.join(SCAN, "frames_O.npz"), agent=np.stack(frames), names=np.array(names))


def scan(a):
    """One candidate (a.ci): both manipulations, every state -> SCAN/geom_<ci>.json, states_<m>_<ci>.npy, frames_<ci>.npz."""
    ci = a.ci
    cand = CANDS[ci]
    gj = os.path.join(SCAN, "geom_%d.json" % ci)
    assert not os.path.exists(gj), "exists: " + gj
    O_ = json.load(open(os.path.join(SCAN, "orig.json"), encoding="utf-8"))
    orig, d0 = O_["orig"], O_["d0"]
    S = np.load(os.path.join(CD, "O", "%s_0.npy" % s2.SUITE))
    sc = Scene()
    frames, names = [], []
    out = {}
    if True:
        for m in MANIPS:
            rows, A = [], []
            for k in range(len(S)):
                sp, placed = sc.make(S[k], m, cand)
                snap, img = sc.check(sp, frame=k < 4)
                ok = geom_ok(snap, placed, orig[k], d0)
                mug, dd = sep_of(snap)
                rows.append(dict(k=k, ok=ok, placed=placed, sep=bool(mug is not None and dd > s2cue2.TOL_NEAR), left_mug=mug,
                                 dl_minus_dr=dd, **{x: snap[x] for x in ("xy", "drift3", "drift_xy", "drift_z_plates", "tilt", "on", "contacts_settled", "dmin",
                                                                          "vis_ratio", "seg")}))
                A.append(sp)
                if k < 4:
                    frames.append(img)
                    names.append("%s_c%d_k%d" % (m, ci, k))
            np.save(os.path.join(SCAN, "states_%s_%d.npy" % (m, ci)), np.stack(A))
            g = [r["ok"] for r in rows]
            out[m] = dict(rows=rows, geom_ok_all=bool(all(g)), geom_ok_share=float(np.mean(g)),
                                      sep_share=float(np.mean([r["sep"] for r in rows])),
                                      fails={x: int(sum(1 for r in rows if not r["ok"] and f(r))) for x, f in (
                                          ("placed", lambda r: any(r["placed"])), ("drift", lambda r: r["drift_xy"] >= 0.01 or r["drift_z_plates"] >= orig[r["k"]]["drift_z_plates"] + 0.01),
                                          ("tilt", lambda r: r["tilt"] >= 10), ("on", lambda r: any(r["on"])),
                                          ("settled_contact", lambda r: any(r["contacts_settled"])),
                                          ("clearance", lambda r: r["dmin"] < d0 - CLEAR_SLACK),
                                          ("vis", lambda r: min(r["vis_ratio"].values()) < VIS_MIN))})
            v = out[m]
            v["sep_margin_mean"] = float(np.mean([r["dl_minus_dr"] for r in rows if r["dl_minus_dr"] is not None]))
            print("[s2cue3 scan] cand %d %s %s: geom ok %.3f (all %s) sep %.3f fails %s | min vis %.2f | dl-dr range "
                  "[%.3f, %.3f] | plate y L %.3f R %.3f | dmin min %.3f" % (
                      ci, cand, m, v["geom_ok_share"], v["geom_ok_all"], v["sep_share"], v["fails"],
                      min(min(r["vis_ratio"].values()) for r in rows),
                      min(r["dl_minus_dr"] for r in rows if r["dl_minus_dr"] is not None),
                      max(r["dl_minus_dr"] for r in rows if r["dl_minus_dr"] is not None),
                      float(np.mean([r["xy"][PL][1] for r in rows])), float(np.mean([r["xy"][PR][1] for r in rows])),
                      min(r["dmin"] for r in rows)), flush=True)
    sc.env.close()
    with open(gj, "w", encoding="utf-8") as fh:
        json.dump(dict(ci=ci, cand=list(cand), **out), fh, indent=1)
    np.savez_compressed(os.path.join(SCAN, "frames_%d.npz" % ci), agent=np.stack(frames), names=np.array(names))
    print("[s2cue3 scan] wrote %s" % gj, flush=True)


def probe(a):
    """Feasibility map (no policy): the table's and camera's geometry, and for one plate moved to each grid point on new state k
    (the other objects as sampled): placed contact, 3-D drift over the settle steps, agentview pixel share of its own count."""
    S = np.load(os.path.join(CD, "O", "%s_0.npy" % s2.SUITE))
    sc = Scene()
    mdl = sc.E.sim.model
    for i in range(mdl.ngeom):
        nm = mdl.geom_id2name(i) or ""
        if "table" in nm:
            print("[s2cue3 probe] geom %s type %d size %s pos(world) %s" % (nm, mdl.geom_type[i], np.round(mdl.geom_size[i], 3).tolist(),
                                                                       np.round(sc.E.sim.data.geom_xpos[i], 3).tolist()), flush=True)
    ci = mdl.camera_name2id("agentview")
    print("[s2cue3 probe] agentview pos %s fovy %.1f" % (np.round(sc.E.sim.data.cam_xpos[ci], 3).tolist(), mdl.cam_fovy[ci]), flush=True)
    for k in (0, 2):
        o, _ = sc.check(S[k])
        print("[s2cue3 probe] k %d start xy %s" % (k, json.dumps({n: np.round(v, 3).tolist() for n, v in o["xy"].items()})), flush=True)
        for p in (PL, PR):
            lines = []
            x0 = o["xy"][p][0]
            for x in x0 + np.arange(-0.10, 0.351, 0.05):
                for y in np.arange(-0.45, 0.451, 0.05):
                    sc.env.reset()
                    sc.env.set_init_state(S[k])
                    q = sc.qpos(p)
                    q[0], q[1] = x, y
                    sc.setq(p, q)
                    sc.E.sim.forward()
                    placed = sc.plate_contacts()
                    if any(placed):
                        lines.append("%+.2f,%+.2f:contact" % (x, y))
                        continue
                    sp = np.array(sc.E.sim.get_state().flatten())
                    snap, _ = sc.check(sp)
                    lines.append("%+.2f,%+.2f:d%.3f,v%.2f" % (x, y, np.linalg.norm(np.array(snap["xy"][p]) - [x, y]),
                                                            snap["seg"][p]["px"] / max(o["seg"][p]["px"], 1)))
            print("[s2cue3 probe] k %d plate %s map (x,y: xy drift, vis share): %s" % (k, p, " ".join(lines)), flush=True)
    sc.env.close()


def ref_eps(S, t, te, ks, br_of):
    import time
    kp = s2.kp_for(t["key"])
    eps = []
    for k in ks:
        rng = np.random.default_rng(int(br_of(k)) + 10)                     # s2.ref's rule
        t0 = time.time()
        try:
            e = s2.plan_episode(os.path.join(ROOT, t["bddl"]), S[k], kp, t["object"], te, rng, int(t["max_steps"]))
        except Exception as ex:                                             # noqa: BLE001
            e = dict(success=False, error=repr(ex))
        e.update(trial=int(k), secs=time.time() - t0, target_object=te)
        eps.append(e)
        print("[s2cue3 ref] %s trial %d success %s held %s phase %s on %s final dxy %s (%.0fs)" % (
            t["key"], k, e["success"], e.get("held_step"), e.get("phase"), e.get("on"), e.get("final_dxy"), e["secs"]), flush=True)
    return kp, eps


def scan_ref(a):
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    fn = os.path.join(SCAN, "ref_%s_%d_%d.json" % (a.m, a.ci, a.tid))
    if os.path.exists(fn):
        return
    if a.ci >= 0:
        G = json.load(open(os.path.join(SCAN, "geom_%d.json" % a.ci), encoding="utf-8"))
        if not all(G[m]["geom_ok_all"] for m in MANIPS):
            print("[s2cue3 scan_ref] cand %d not geometry-feasible: skipped" % a.ci, flush=True)
            return
        S = np.load(os.path.join(SCAN, "states_%s_%d.npy" % (a.m, a.ci)))
    else:
        S = np.load(os.path.join(CD, "O", "%s_0.npy" % s2.SUITE))
    kp, eps = ref_eps(S, t, t["target"], REF_SCAN_KS, lambda k: BR_BASE3 + 1000 * k)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(m=a.m, ci=a.ci, tid=a.tid, key=t["key"], eps=eps), fh, indent=1)


def scan_pick(a):
    G = {ci: json.load(open(os.path.join(SCAN, "geom_%d.json" % ci), encoding="utf-8")) for ci in range(len(CANDS))}
    M_ = s2.meta()

    def refrate(m, ci, tids):
        v = []
        for tid in tids:
            f = os.path.join(SCAN, "ref_%s_%d_%d.json" % (m, ci, tid))
            if not os.path.exists(f):
                return None
            v += [float(e["success"]) for e in json.load(open(f, encoding="utf-8"))["eps"]]
        return float(np.mean(v))
    print("[s2cue3 pick] unmanipulated reference (O, states 0-7): %s" % {M_["tasks"][t]["key"]: refrate("O", -1, [t]) for t in TIDS})
    table = []
    for ci, cand in enumerate(CANDS):
        c = G[ci]
        r = {m: dict(L=refrate(m, ci, (0, 3)), R=refrate(m, ci, (1, 2)), per={M_["tasks"][t]["key"]: refrate(m, ci, [t]) for t in TIDS})
             for m in MANIPS}
        geo = all(c[m]["geom_ok_all"] for m in MANIPS)
        reach = geo and all(r[m]["L"] is not None and r[m]["R"] is not None and r[m]["L"] >= 0.5 and r[m]["R"] >= 0.5 for m in MANIPS)
        score = min(c[m]["sep_share"] for m in MANIPS)
        margin = min(c[m]["sep_margin_mean"] for m in MANIPS)
        table.append(dict(ci=ci, cand=list(cand), geom=geo, reach=reach, feasible=bool(geo and reach), score=score, margin=margin, ref=r,
                          geom_share={m: c[m]["geom_ok_share"] for m in MANIPS}, fails={m: c[m]["fails"] for m in MANIPS},
                          sep={m: c[m]["sep_share"] for m in MANIPS}))
        print("[s2cue3 pick] cand %d %s: geom %s (%s; fails %s) sep %s margin %.3f ref %s -> feasible %s score %.3f" % (
            ci, cand, geo, table[-1]["geom_share"], table[-1]["fails"], table[-1]["sep"], margin, json.dumps(r), table[-1]["feasible"],
            score), flush=True)
    F = [x for x in table if x["feasible"]]
    pick = None
    if F:
        best = max(x["score"] for x in F)
        tied = [x for x in F if x["score"] >= best - 0.01]
        pick = max(tied, key=lambda x: (x["margin"], -x["ci"]))
    with open(os.path.join(SCAN, "pick.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(rule=PICK_RULE, table=table, pick=pick), fh, indent=1)
    print("[s2cue3 pick] PICK %s" % (None if pick is None else json.dumps({x: pick[x] for x in ("ci", "cand", "score", "margin", "sep", "ref")})),
          flush=True)


def build(a):
    for m in MANIPS:
        assert not os.path.exists(state_dir(m)), "exists: " + state_dir(m)
    assert not os.path.exists(STATES_JSON)
    P_ = json.load(open(os.path.join(SCAN, "pick.json"), encoding="utf-8"))
    assert P_["pick"] is not None, "no feasible candidate"
    ci = int(P_["pick"]["ci"])
    G = json.load(open(os.path.join(SCAN, "geom_%d.json" % ci), encoding="utf-8"))
    O_ = json.load(open(os.path.join(SCAN, "orig.json"), encoding="utf-8"))
    M_ = s2.meta()
    rows = [dict(k=k, orig=O_["orig"][k]) for k in range(N_STATES)]
    for m in MANIPS:
        A = np.load(os.path.join(SCAN, "states_%s_%d.npy" % (m, ci)))
        assert len(A) == N_STATES
        os.makedirs(state_dir(m))
        for t in M_["tasks"][:4]:
            np.save(os.path.join(state_dir(m), "%s_%d.npy" % (s2.SUITE, t["tid"])), A)
        for k, r in enumerate(G[m]["rows"]):
            assert r["k"] == k
            rows[k][m] = r
    with open(STATES_JSON, "w", encoding="utf-8") as fh:
        json.dump(dict(rule=PICK_RULE, pick=P_["pick"], plates=P_["pick"]["cand"], d0=O_["d0"],
                       image_side_agreement=O_["image_side_agreement"], source=os.path.join(CD, "O"), rows=rows), fh, indent=1)
    try:
        from PIL import Image
        z0, z = np.load(os.path.join(SCAN, "frames_O.npz")), np.load(os.path.join(SCAN, "frames_%d.npz" % ci))
        nm = [str(x) for x in z["names"]]
        ims = list(z0["agent"][:4]) + [z["agent"][nm.index("%s_c%d_k%d" % (m, ci, k))] for m in MANIPS for k in SMOKE]
        h, w_ = ims[0].shape[:2]
        sheet = np.full((3 * h, 4 * w_, 3), 255, np.uint8)
        for i, im in enumerate(ims):
            sheet[(i // 4) * h:(i // 4 + 1) * h, (i % 4) * w_:(i % 4 + 1) * w_] = im
        Image.fromarray(sheet).save(SHEET)
        print("[s2cue3 build] sheet rows O / Sm / PSm, k 0-3 -> %s" % SHEET, flush=True)
    except Exception as ex:                                                 # noqa: BLE001
        print("[s2cue3 build] no sheet: %r" % ex, flush=True)
    for m in MANIPS:
        rr = [rows[k][m] for k in range(N_STATES)]
        print("[s2cue3 build] %s (cand %d): ok %d/44, sep %d/44 (main %d/40), left mug %s, dl-dr [%.3f, %.3f], drift max %.4f, tilt max "
              "%.2f, vis min %.2f" % (m, ci, sum(r["ok"] for r in rr), sum(r["sep"] for r in rr), sum(r["sep"] for r in rr if r["k"] in MAIN),
                                      sorted(set(str(r["left_mug"]) for r in rr)), min(r["dl_minus_dr"] for r in rr),
                                      max(r["dl_minus_dr"] for r in rr), max(r["drift_xy"] for r in rr), max(r["tilt"] for r in rr),
                                      min(min(r["vis_ratio"].values()) for r in rr)), flush=True)
    print("[s2cue3 build] wrote %s" % CD, flush=True)


# ---------------------------------------------------------------- reference planner (C1; LIBERO venv, CPU)
def ref(a):
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    assert t["tid"] == a.tid and a.manip in MANIPS
    os.makedirs(a.out_dir, exist_ok=True)
    fn = os.path.join(a.out_dir, "ref_%s_%d.json" % (a.manip, a.tid))
    assert not os.path.exists(fn), "exists: " + fn
    S = np.load(os.path.join(state_dir(a.manip), "%s_%d.npy" % (s2.SUITE, a.tid)))
    br = {int(b["trial"]): int(b["seed_base"]) for b in json.load(open(a.branches, encoding="utf-8"))}
    kp, eps = ref_eps(S, t, t["target"], sorted(br), lambda k: br[k])
    for e in eps:
        e["success_side"] = bool(e["success"])                              # the plates keep their order: BDDL goal = named side
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(manip=a.manip, tid=a.tid, key=t["key"], kp=kp, target_object=t["target"], eps=eps), fh, indent=1)


# ---------------------------------------------------------------- scoring (CPU)
def h2h(E, ks):
    """Head-to-head on placements where start side and nearer plate are both defined and disagree: share on the start-side plate."""
    F, D = np.zeros(len(ks)), np.zeros(len(ks))
    by_mug = {W: [0, 0], Y: [0, 0]}
    for i, e in s2cue2.placements(E):
        a_, b_ = e["pred"]["start_side"], e["pred"]["nearer_plate"]
        if a_ is not None and b_ is not None and a_ != b_:
            D[i] += 1
            F[i] += float(e["dest_side"] == a_)
            by_mug[e["held"]][0] += int(e["dest_side"] == a_)
            by_mug[e["held"]][1] += 1
    sh = s2cue.boot_share(F, D)
    if sh[3] < H2H_MIN:
        v = "UNTESTABLE" if sh[3] else "NOT DEFINED"
    elif sh[0] >= 0.70 and sh[1] > 0.50:
        v = "START SIDE ROUTES"
    elif sh[0] <= 0.30 and sh[2] < 0.50:
        v = "NEARER PLATE ROUTES"
    else:
        v = "UNRESOLVED"
    return dict(share_start_side=sh, verdict=v, by_mug={str(m): x for m, x in by_mug.items()})


def model_verdict(vs, vp):
    if vs == vp == "START SIDE ROUTES":
        return "START SIDE ROUTES"
    if vs == vp == "NEARER PLATE ROUTES":
        return "NEARER PLATE ROUTES"
    if vs == "START SIDE ROUTES" and vp == "NEARER PLATE ROUTES":
        return "IDENTITY"
    if vs == "NEARER PLATE ROUTES" and vp == "START SIDE ROUTES":
        return "MIXED"
    return "UNRESOLVED"


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
            for e in s2cue2.score(stem, t, ks, manip, srows):
                print("[s2cue3 smoke] %s %s %s %s %s" % (model, manip, fe, t["key"], json.dumps(e)), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_")) if os.path.isdir(d) else []:
        R_ = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2cue3 smoke] %s: %s" % (f, [(e["trial"], e["success_side"], e.get("on")) for e in R_["eps"]]), flush=True)


def read(a):
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    M_ = s2.meta()
    T = {t["tid"]: t for t in M_["tasks"]}
    SJ = json.load(open(STATES_JSON, encoding="utf-8"))
    srows = {r["k"]: r for r in SJ["rows"]}
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD, mism = SMOKE, a.dir, "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_cue3"]
        assert len(rows) == 1, "expected exactly one s2_cue3 prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD = MAIN, ROLL
    res = dict(states=ks, mapping=MAPPING, design=dict(plates=SJ["plates"], pick=SJ["pick"]), code_mismatch=mism, runs={},
               reference={}, per_model={})
    # by-construction / descriptive geometry on the states read
    for m in MANIPS:
        rr = [srows[k][m] for k in ks]
        res.setdefault("geometry", {})[m] = dict(ok=int(sum(r["ok"] for r in rr)), sep=int(sum(r["sep"] for r in rr)), n=len(rr),
                                                 left_mug=sorted(set(str(r["left_mug"]) for r in rr)))
        print("== geometry %s: %s" % (m, json.dumps(res["geometry"][m])), flush=True)
    for model, manip, fe, role in RUNS:
        name = "%s %s %s" % (model, manip, fe)
        E, miss = {}, []
        for tid in TIDS:
            stem = os.path.join(RD, manip, "%s_%s_%d" % (model, fe, tid))
            if not (os.path.exists(stem + ".json") and os.path.exists(stem + "_track.npz")):
                miss.append(stem)
                continue
            E[T[tid]["key"]] = s2cue2.score(stem, T[tid], ks, manip, srows)
        if miss:
            res["runs"][name] = "MISSING %s" % miss
            print("== %s: MISSING %d" % (name, len(miss)), flush=True)
            continue
        row = dict(role=role, describe=s2cue2.describe(E), labels=s2cue2.label_run(E, ks), h2h=h2h(E, ks))
        res["runs"][name] = row
        L = row["labels"]
        print("== %s (%s): placements %d | routing %s | HEAD-TO-HEAD %s" % (name, role, L["n_placements"], L["routing"],
                                                                              json.dumps(row["h2h"])), flush=True)
        for c, v in L["cues"].items():
            print("   %-13s share %s -> %s" % (c, np.round([x if x is not None else np.nan for x in v["share"][:3]], 3).tolist()
                                              + [v["share"][3]], v["label"]), flush=True)
        for c, v in row["describe"].items():
            print("   %s %s" % (c, json.dumps(v)), flush=True)
    for model in ("pi05", "pi0"):
        for fe in ("I", "tv"):
            rs, rp = res["runs"].get("%s Sm %s" % (model, fe)), res["runs"].get("%s PSm %s" % (model, fe))
            if isinstance(rs, dict) and isinstance(rp, dict):
                vs, vp = rs["h2h"]["verdict"], rp["h2h"]["verdict"]
                res["per_model"]["%s %s" % (model, fe)] = dict(Sm=vs, PSm=vp, verdict=model_verdict(vs, vp),
                                                               role="PRIMARY" if fe == "I" else "SECONDARY")
                print("== PER MODEL %s %s: Sm %s | PSm %s -> %s" % (model, fe, vs, vp, model_verdict(vs, vp)), flush=True)
    RF = a.ref_dir or RD
    for manip in MANIPS:
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
    print("[s2cue3 read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("states", "probe", "scan_orig", "scan", "scan_ref", "scan_pick", "build", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--m", choices=MANIPS + ("O",))
    ap.add_argument("--ci", type=int)
    ap.add_argument("--manip", choices=MANIPS)
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--ref-dir")
    ap.add_argument("--out", default=READ)
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    dict(states=states, probe=probe, scan_orig=scan_orig, scan=scan, scan_ref=scan_ref, scan_pick=scan_pick, build=build, ref=ref, summary=lambda a: summary(a.dir),
         read=read)[A.mode](A)

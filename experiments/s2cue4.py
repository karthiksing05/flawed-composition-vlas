"""S2CUE4 (docs/SCAFFOLD_RESULTS.md 60; m3_prereg tag s2_cue4): in scene 2 (LIBERO-10 LIVING_ROOM_SCENE5) SR 59 found that pi0.5
carries the held mug to the NEARER plate (0 of 241 disagreeing placements on the start-side plate), and that no run ever placed on
the far plate (0 of 556).  In SR 59's layout (Sm) the nearer plate was also the SALIENT one: it stood in front of the mugs near the
image centre with ~2.7x the far plate's agentview pixels.  This probe separates NEARER PLATE from SALIENT PLATE, and runs frozen
GR00T on SR 59's Sm layout.

Item 1 (pi0.5 and pi0, arm I PRIMARY; pi0.5 tv SECONDARY).  Manipulations (initial state only; prompts, servers, client, launcher,
client seed 7 and --bind track as SR 57-59):
    Sx   both plates moved to fixed positions: the RIGHT plate (plate_2) to (XR, YR), in front of the mugs just right of the
         midline (large in the image, near its centre: SALIENT, and FAR from the left-starting mug); the LEFT plate (plate_1) to
         (XL, YL), at the left side, level with the mugs (small, near the image edge: NOT salient, and NEARER the left-starting
         mug).  The mugs stand where they were sampled; the white mug starts left.
    PSx  as Sx, after SR 57's swap of the white and the yellow-and-white mug's xy.  The yellow-and-white mug then starts left.
    For the left-starting (disagreeing) mug: NEARER PLATE says left in both, SALIENT PLATE says right in both, IDENTITY (its
    trained plate) says left under Sx and right under PSx.  START SIDE says left in both: it stays tied to the nearer plate here
    by construction (SR 59 separated those two: start side RULED OUT on pi0.5).
    SALIENT PLATE is measured from the agentview render of the settled start state (the simulator segmentation the policy's
    image is rendered from; 256 x 256, flipped like the policy's images): the plate with >= AREA_RATIO times the other plate's
    visible pixels AND whose pixel centroid is >= CENTRE_MARGIN px nearer the image centre; undefined otherwise.
    (XL, YL, XR, YR) is chosen from CANDS by PICK_RULE before any rollout (no policy anywhere in the scan).  If no feasible
    candidate separates the two cues on >= FEAS_MIN of the states for both manipulations, the layout is INFEASIBLE: a
    feasibility-only verdict is registered and nothing is rolled out.

Item 2 (frozen GR00T, arm I).  Sm = SR 59's picked layout (plate_1 at (.25, -.24), plate_2 at (.10, .03)) on THESE new states;
SR 59's start-side vs nearer-plate head-to-head (s2cue3.h2h, unchanged).  GR00T's testability floor as SR 50: TESTABLE iff its
intact K1-K2 success (BDDL) on the unmanipulated new states O is >= .70.

States.  NEW states (not SR 50 / 57 / 58 / 59's): LIBERO's sampler on K1's BDDL, np.random.seed = env.seed = 845000 + k,
k = 0..43 (0-3 smoke, 4-43 main); branches trial = k, seed_base 28000000 + 1000 k.  Written to results/flawed/s2cue4/O by `states`.

Modes (LIBERO venv, py3.8):
    states                  (EGL)  -> results/flawed/s2cue4/O/s2_living_<tid>.npy, branches_{smoke,main}.json
    scan_orig               (EGL)  -> SCAN/{orig.json, frames_O.npz}
    scan --ci I             (EGL)  -> SCAN/{geom_<ci>.json, states_<m>_<ci>.npy, frames_<ci>.npz}: candidate I, Sx and PSx
    scan_sm                 (EGL)  -> SCAN/{geom_sm.json, states_Sm.npy, frames_sm.npz}: SR 59's Sm layout on the new states
    scan_ref --m M --ci I --tid T (CPU) -> SCAN/ref_<m>_<ci>_<tid>.json (reference planner, states 0-7)
    scan_pick               (CPU)  -> SCAN/pick.json (PICK_RULE; FEASIBLE or INFEASIBLE)
    build                   (CPU)  -> results/flawed/s2cue4/{Sx,PSx,Sm}/s2_living_<tid>.npy, s2cue4_states.json, start-render sheet
    ref --manip M --tid T --branches B --out-dir D   (CPU)
    summary DIR             (smoke)
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
import s2cue3                                                               # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2cue4")
ROLL = os.path.join(OUT, "s2cue4_roll")
READ = os.path.join(OUT, "s2cue4_read.json")
STATES_JSON = os.path.join(CD, "s2cue4_states.json")
SCAN = "/coc/flash10/ksingara3/.claude_tmp/pp/s2cue4_scan"
SHEET = "/coc/flash10/ksingara3/.claude_tmp/pp/s2cue4_sheet.png"
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
BASE4, BR_BASE4, N_STATES = 845000, 28000000, 44
SMOKE = list(range(0, 4))
MAIN = list(range(4, 44))
REF_SCAN_KS = list(range(0, 8))
MANIPS = ("Sx", "PSx")                                                      # item 1
SM_PLATES = (0.25, -0.24, 0.10, 0.03)                                       # SR 59's pick (s2cue3_scan2/pick.json, cand 6); item 2
BUILT = MANIPS + ("Sm",)
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
# (XL, YL, XR, YR): plate_1 (left, the non-salient plate nearer the left-starting mug) at the left side, level with the mugs;
# plate_2 (right, the salient plate) in front of the mugs just right of the midline.  From SR 59's probe map
# (s2cue3_probe_4712585.out): a plate at x in [-.05, 0], |y| ~ .30 shows ~1.2-1.5x the sampled plate's pixels; at x .10-.15 near
# the midline ~2.6-3.2x; a plate needs >= ~.12 m centre distance to a mug.
CANDS = [(xl, yl, xr, 0.04) for xl in (-0.05, 0.0) for yl in (-0.31, -0.28) for xr in (0.10, 0.15)]
AREA_RATIO, CENTRE_MARGIN, FEAS_MIN = 1.5, 10.0, 0.75
PICK_RULE = (
    "Candidate (XL, YL, XR, YR) in CANDS (plate_1 placed at (XL, YL), plate_2 at (XR, YR), each keeping its own height and "
    "orientation).  For each manipulation m in (Sx, PSx) and each of the 44 new states k: GEOM_OK as SR 59 (s2cue3.geom_ok, "
    "unchanged: as placed no plate touches a mug or the other plate; after the 10 settle steps xy drift < 1 cm, plates' z drop "
    "<= 1 cm more than unmanipulated, mug tilt < 10 deg, no mug On a plate, no plate touching a mug, smallest mug-plate centre "
    "distance >= D0 - .01 m, each plate's agentview pixels >= .60 of its unmanipulated count, left plate y < 0 < right plate y).  "
    "NEARER(k) = the side of the plate nearer the mug that starts left of the midline (y < -.05; exactly one of the white and "
    "yellow-and-white mugs), defined iff |d_left - d_right| >= .02 m.  SALIENT(k) = the side of the plate whose agentview pixel "
    "count (simulator segmentation of the settled start render, 256 x 256) is >= 1.5 x the other plate's AND whose pixel centroid "
    "is >= 10 px nearer the image centre (127.5, 127.5); undefined otherwise.  SEP(k) iff NEARER and SALIENT are both defined and "
    "differ.  FEASIBLE iff GEOM_OK on all 44 states for both m and the reference planner (s2.plan_episode, unchanged) succeeds on "
    ">= .50 of the 16 episodes per plate (states 0-7; K1 + N2 for the left plate, K2 + N1 for the right) for both m.  Score = min "
    "over m of the share of the 44 states with SEP.  PICK = the feasible candidate with the highest score; ties (within .01) -> "
    "the larger mean |d_left - d_right| on SEP states (min over m), then the smaller ci.  LAYOUT FEASIBLE iff a pick exists with "
    "score >= .75; otherwise INFEASIBLE: register a feasibility-only verdict with the numbers and roll nothing out.")
# (model, manipulation, front-end arm, role, tids)
RUNS = (("pi05", "Sx", "I", "PRIMARY", TIDS), ("pi05", "PSx", "I", "PRIMARY", TIDS), ("pi0", "Sx", "I", "PRIMARY", TIDS),
        ("pi0", "PSx", "I", "PRIMARY", TIDS), ("groot", "Sm", "I", "PRIMARY", TIDS), ("groot", "O", "I", "FLOOR", (0, 1)),
        ("pi05", "Sx", "tv", "SECONDARY", TIDS), ("pi05", "PSx", "tv", "SECONDARY", TIDS))
CUES4 = s2cue2.CUES + ("salient",)
H2H_MIN, GROOT_FLOOR = 20, 0.70
MAPPING = (
    "Models frozen: pi0.5 (pp_serve_purify --model pi05 --mode intact --comp= --seed 13), pi0 (serve_guided --mode ours --seed 13, "
    "BACKBONE_CONFIG=pi0_libero), GR00T N1 (pp_serve_purify --model groot --mode intact --comp= --seed 13, ar0s/groot_libero), "
    "s2.py client unchanged run by s2_roll.sh unchanged (client seed 7, --bind track); states = the NEW states (LIBERO sampler "
    "seeds 845000 + k) under Sx, PSx, Sm or unmanipulated O (results/flawed/s2cue4/<m>), branches_main.json (k = 4-43, seed_base "
    "28000000 + 1000 k).  Runs: pi0.5 Sx I, PSx I, pi0 Sx I, PSx I (PRIMARY, item 1); GR00T Sm I (PRIMARY, item 2) and GR00T O I "
    "on K1, K2 (testability floor); pi0.5 Sx tv, PSx tv (SECONDARY).  Tasks K1, K2, N1, N2 (40 episodes each) unless stated.  Per "
    "episode: HELD mug = libero_client's 'bound' if it is the white or the yellow-and-white mug; PLACEMENT = the held mug ends On "
    "exactly one plate (final LIBERO predicates); plates scored by SIDE (plate_1 = the left plate; the plates keep their order).  "
    "Start positions and renders = the settled start state (s2cue4_states.json).  Cues (s2cue2.cue_preds unchanged, plus "
    "SALIENT): START SIDE, NEARER PLATE, PLATE OBJECT / IDENTITY, INSTRUCTION, MUG LEFT/RIGHT ORDER, DEPTH ORDER, STEP as SR 58 / "
    "59; SALIENT PLATE = the plate side with >= 1.5x the other plate's agentview pixels and a centroid >= 10 px nearer the image "
    "centre (undefined otherwise; per state, the same for either mug).  Per run and cue: s2cue2.label_run's rule unchanged "
    "(share over placements where the cue is defined, pooled over tasks, CI = 2.5 / 97.5 percentiles of B = 2000 bootstrap draws "
    "of the 40 states, default_rng(0); ROUTES iff f >= .70 and lo > .50; RULED OUT iff f <= .30 and hi < .50; UNTESTABLE iff < 20 "
    "placements; INSEPARABLE iff < 10 disagreements among >= 10 co-defined).  HEAD-TO-HEAD NS (item 1, the registered test) per "
    "Sx / PSx run: over placements where NEARER PLATE and SALIENT PLATE are both defined and DISAGREE, s = share on the salient "
    "plate, same bootstrap; SALIENT PLATE ROUTES iff s >= .70 and lo > .50; NEARER PLATE ROUTES iff s <= .30 and hi < .50; else "
    "UNRESOLVED; UNTESTABLE iff < 20 such placements.  PER MODEL (arm I; also reported for tv): Sx and PSx both NEARER -> NEARER "
    "PLATE ROUTES; both SALIENT -> SALIENT PLATE ROUTES; Sx NEARER and PSx SALIENT -> IDENTITY; Sx SALIENT and PSx NEARER -> "
    "MIXED; otherwise UNRESOLVED.  By construction START SIDE = NEARER PLATE on the disagreeing mug in both, IDENTITY = NEARER "
    "under Sx and = SALIENT under PSx.  HEAD-TO-HEAD SN (item 2) on GR00T Sm I: s2cue3.h2h unchanged (start side vs nearer plate; "
    "START SIDE ROUTES / NEARER PLATE ROUTES / UNRESOLVED / UNTESTABLE < 20).  GR00T FLOOR (SR 50's rule): TESTABLE iff GR00T's "
    "K1-K2 BDDL success on O (80 episodes) >= .70; if not, GR00T's Sm verdict is reported with NOT TESTABLE and does not count as "
    "a third-model point.  Success: per task, the named mug On the named plate (= the BDDL goal; descriptive).  C1 per "
    "manipulation (Sx, PSx, Sm): the reference planner (s2.plan_episode, unchanged) on the same 40 states; MET iff N1-N2 >= .80.  "
    "Descriptive: salience numbers per state, first-held shares, held x destination tables, wrist-camera pixels at the start.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def state_dir(m):
    return os.path.join(CD, m)


# ---------------------------------------------------------------- simulator tools (LIBERO venv)
class Scene4(s2cue3.Scene):
    """SR 59's Scene with a richer render record: per movable, agentview pixels, centroid (row, col), centroid distance to the
    image centre; the same for the wrist camera (descriptive)."""

    def seg(self):
        import sc_rv as RV
        T = RV.entity_tables(self.env)
        names = list(T["names"])
        out = {n: {} for n in s2.MOVABLE}
        for cam, pre in (("agentview", ""), ("robot0_eye_in_hand", "w_")):
            try:
                ent = RV.seg_px(self.env, T, cam)
            except Exception as ex:                                         # noqa: BLE001
                for n in s2.MOVABLE:
                    out[n][pre + "err"] = repr(ex)
                continue
            c0 = (ent.shape[0] - 1) / 2.0
            for n in s2.MOVABLE:
                msk = ent == names.index(n)
                c = int(msk.sum())
                rr, cc = np.where(msk)
                out[n][pre + "px"] = c
                out[n][pre + "col"] = float(cc.mean()) if c else None
                out[n][pre + "row"] = float(rr.mean()) if c else None
                out[n][pre + "cdist"] = float(np.hypot(rr.mean() - c0, cc.mean() - c0)) if c else None
        return out

    def make4(self, s, m, cand):
        return self.make(s, "PSm" if m == "PSx" else "Sm", cand)            # SR 59's make: PSm swaps the mugs' xy first


def salient_of(snap):
    seg = snap["seg"]
    a = {p: seg[p]["px"] for p in (PL, PR)}
    c = {p: seg[p]["cdist"] for p in (PL, PR)}
    if None in c.values():
        return None
    for s_, f_ in ((PL, PR), (PR, PL)):
        if a[s_] >= AREA_RATIO * a[f_] and c[s_] <= c[f_] - CENTRE_MARGIN:
            return "L" if s_ == PL else "R"
    return None


def nearer_of(snap):
    """The mug that starts left of the midline: (mug, side of its nearer plate or None, |d_left - d_right|)."""
    mug, dd = s2cue3.sep_of(snap)                                           # dd = d_left - d_right
    if mug is None:
        return None, None, None
    return mug, (None if abs(dd) < s2cue2.TOL_NEAR else ("L" if dd < 0 else "R")), abs(dd)


def annotate(snap):
    mug, near, marg = nearer_of(snap)
    sal = salient_of(snap)
    seg = snap["seg"]
    snap.update(left_mug=mug, nearer_left_mug=near, near_margin=marg, salient=sal,
                sep_ns=bool(near is not None and sal is not None and near != sal),
                sal_numbers={("L" if p == PL else "R"): dict(px=seg[p]["px"], cdist=seg[p]["cdist"], row=seg[p]["row"],
                                                             col=seg[p]["col"], w_px=seg[p].get("w_px")) for p in (PL, PR)})
    return snap


# ---------------------------------------------------------------- states (new) and the feasibility scan
def states(a):
    import s2_tasks as T2
    od = os.path.join(CD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))

    def draw(k):                                                            # s2_tasks.build's rule with a new base
        np.random.seed(BASE4 + k)
        env.seed(BASE4 + k)
        env.reset()
        return env.sim.get_state().flatten()
    S = np.stack([draw(k) for k in range(N_STATES)])
    env.close()
    old = [np.load(os.path.join(s2.SD, "states", "%s_0.npy" % s2.SUITE)),
           np.load(os.path.join(OUT, "s2cue3", "O", "%s_0.npy" % s2.SUITE))]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("same as an SR 50 / 59 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    for ph, ks in (("smoke", SMOKE), ("main", MAIN)):
        with open(os.path.join(CD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE4 + 1000 * k, at=0, variant="s2") for k in ks], fh, indent=1)
    print("[s2cue4 states] %d new states (seed %d + k), none equal to SR 50's or SR 59's; wrote %s" % (len(S), BASE4, CD), flush=True)


def scan_orig(a):
    os.makedirs(SCAN, exist_ok=True)
    oj = os.path.join(SCAN, "orig.json")
    assert not os.path.exists(oj), "exists: " + oj
    S = np.load(os.path.join(CD, "O", "%s_0.npy" % s2.SUITE))
    sc = Scene4()
    orig, frames, names = [], [], []
    for k in range(len(S)):
        snap, img = sc.check(S[k], frame=k < 4)
        orig.append(annotate(snap))
        if k < 4:
            frames.append(img)
            names.append("O_k%d" % k)
    d0 = min(o["dmin"] for o in orig)
    side = [(o["xy"][m][1] < 0, o["seg"][m]["col"] > 127.5) for o in orig for m in (W, Y) if o["seg"][m]["col"] is not None]
    agree = float(np.mean([x == y for x, y in side]))
    print("[s2cue4 scan] D0 %.4f m; mug world side vs image side agreement %.3f of %d" % (d0, agree, len(side)), flush=True)
    print("[s2cue4 scan] unmanipulated: left mug %s; plate px L %s R %s; cdist L %.1f R %.1f; salient %s; drift_xy max %.4f" % (
        sorted(set(str(o["left_mug"]) for o in orig)), np.percentile([o["seg"][PL]["px"] for o in orig], [0, 50, 100]).tolist(),
        np.percentile([o["seg"][PR]["px"] for o in orig], [0, 50, 100]).tolist(),
        np.mean([o["seg"][PL]["cdist"] for o in orig]), np.mean([o["seg"][PR]["cdist"] for o in orig]),
        {str(v): sum(o["salient"] == v for o in orig) for v in ("L", "R", None)}, max(o["drift_xy"] for o in orig)), flush=True)
    sc.env.close()
    with open(oj, "w", encoding="utf-8") as fh:
        json.dump(dict(d0=d0, image_side_agreement=agree, orig=orig), fh, indent=1)
    np.savez_compressed(os.path.join(SCAN, "frames_O.npz"), agent=np.stack(frames), names=np.array(names))


def _scan_layout(sc, S, m, cand, orig, d0, tag, frames, names):
    rows, A = [], []
    for k in range(len(S)):
        sp, placed = sc.make4(S[k], m, cand)
        snap, img = sc.check(sp, frame=k < 4)
        ok = s2cue3.geom_ok(snap, placed, orig[k], d0)
        annotate(snap)
        mug, dd = s2cue3.sep_of(snap)
        rows.append(dict(k=k, ok=ok, placed=placed, sep_sn=bool(mug is not None and dd > s2cue2.TOL_NEAR), dl_minus_dr=dd,
                         **{x: snap[x] for x in ("xy", "drift3", "drift_xy", "drift_z_plates", "tilt", "on", "contacts_settled", "dmin",
                                                  "vis_ratio", "seg", "left_mug", "nearer_left_mug", "near_margin", "salient", "sep_ns",
                                                  "sal_numbers")}))
        A.append(sp)
        if k < 4:
            frames.append(img)
            names.append("%s_%s_k%d" % (m, tag, k))
    g = [r["ok"] for r in rows]
    v = dict(rows=rows, geom_ok_all=bool(all(g)), geom_ok_share=float(np.mean(g)),
             sep_ns_share=float(np.mean([r["sep_ns"] for r in rows])), sep_sn_share=float(np.mean([r["sep_sn"] for r in rows])),
             salient={str(x): int(sum(r["salient"] == x for r in rows)) for x in ("L", "R", None)},
             nearer={str(x): int(sum(r["nearer_left_mug"] == x for r in rows)) for x in ("L", "R", None)},
             near_margin_sep=float(np.mean([r["near_margin"] for r in rows if r["sep_ns"]])) if any(r["sep_ns"] for r in rows) else 0.0,
             px_L=np.percentile([r["seg"][PL]["px"] for r in rows], [0, 50, 100]).tolist(),
             px_R=np.percentile([r["seg"][PR]["px"] for r in rows], [0, 50, 100]).tolist(),
             cd_L=np.percentile([r["seg"][PL]["cdist"] or np.nan for r in rows], [0, 50, 100]).tolist(),
             cd_R=np.percentile([r["seg"][PR]["cdist"] or np.nan for r in rows], [0, 50, 100]).tolist(),
             area_ratio_RL=np.percentile([r["seg"][PR]["px"] / max(r["seg"][PL]["px"], 1) for r in rows], [0, 50, 100]).tolist(),
             fails={x: int(sum(1 for r in rows if not r["ok"] and f(r))) for x, f in (
                 ("placed", lambda r: any(r["placed"])), ("drift", lambda r: r["drift_xy"] >= 0.01),
                 ("tilt", lambda r: r["tilt"] >= 10), ("on", lambda r: any(r["on"])),
                 ("settled_contact", lambda r: any(r["contacts_settled"])), ("clearance", lambda r: r["dmin"] < d0 - s2cue3.CLEAR_SLACK),
                 ("vis", lambda r: min(r["vis_ratio"].values()) < s2cue3.VIS_MIN))})
    print("[s2cue4 scan] %s %s %s: geom ok %.3f (all %s) fails %s | SEP(near vs salient) %.3f | salient %s nearer %s | px L %s R %s "
          "| cdist L %s R %s | area R/L %s | SEP(start vs near) %.3f | dmin min %.3f" % (
              tag, list(cand), m, v["geom_ok_share"], v["geom_ok_all"], v["fails"], v["sep_ns_share"], v["salient"], v["nearer"],
              np.round(v["px_L"]).tolist(), np.round(v["px_R"]).tolist(), np.round(v["cd_L"], 1).tolist(),
              np.round(v["cd_R"], 1).tolist(), np.round(v["area_ratio_RL"], 2).tolist(), v["sep_sn_share"],
              min(r["dmin"] for r in rows)), flush=True)
    return v, np.stack(A)


def scan(a):
    ci = a.ci
    cand = CANDS[ci]
    gj = os.path.join(SCAN, "geom_%d.json" % ci)
    assert not os.path.exists(gj), "exists: " + gj
    O_ = json.load(open(os.path.join(SCAN, "orig.json"), encoding="utf-8"))
    S = np.load(os.path.join(CD, "O", "%s_0.npy" % s2.SUITE))
    sc = Scene4()
    frames, names, out = [], [], {}
    for m in MANIPS:
        out[m], A = _scan_layout(sc, S, m, cand, O_["orig"], O_["d0"], "c%d" % ci, frames, names)
        np.save(os.path.join(SCAN, "states_%s_%d.npy" % (m, ci)), A)
    sc.env.close()
    with open(gj, "w", encoding="utf-8") as fh:
        json.dump(dict(ci=ci, cand=list(cand), **out), fh, indent=1)
    np.savez_compressed(os.path.join(SCAN, "frames_%d.npz" % ci), agent=np.stack(frames), names=np.array(names))


def scan_sm(a):
    gj = os.path.join(SCAN, "geom_sm.json")
    assert not os.path.exists(gj), "exists: " + gj
    O_ = json.load(open(os.path.join(SCAN, "orig.json"), encoding="utf-8"))
    S = np.load(os.path.join(CD, "O", "%s_0.npy" % s2.SUITE))
    sc = Scene4()
    frames, names = [], []
    v, A = _scan_layout(sc, S, "Sm", SM_PLATES, O_["orig"], O_["d0"], "sm", frames, names)
    np.save(os.path.join(SCAN, "states_Sm.npy"), A)
    sc.env.close()
    with open(gj, "w", encoding="utf-8") as fh:
        json.dump(dict(cand=list(SM_PLATES), Sm=v), fh, indent=1)
    np.savez_compressed(os.path.join(SCAN, "frames_sm.npz"), agent=np.stack(frames), names=np.array(names))


def scan_ref(a):
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    fn = os.path.join(SCAN, "ref_%s_%d_%d.json" % (a.m, a.ci, a.tid))
    if os.path.exists(fn):
        return
    G = json.load(open(os.path.join(SCAN, "geom_%d.json" % a.ci), encoding="utf-8"))
    if not all(G[m]["geom_ok_all"] for m in MANIPS):
        print("[s2cue4 scan_ref] cand %d not geometry-feasible: skipped" % a.ci, flush=True)
        return
    S = np.load(os.path.join(SCAN, "states_%s_%d.npy" % (a.m, a.ci)))
    kp, eps = s2cue3.ref_eps(S, t, t["target"], REF_SCAN_KS, lambda k: BR_BASE4 + 1000 * k)
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
    table = []
    for ci, cand in enumerate(CANDS):
        c = G[ci]
        r = {m: dict(L=refrate(m, ci, (0, 3)), R=refrate(m, ci, (1, 2)), per={M_["tasks"][t]["key"]: refrate(m, ci, [t]) for t in TIDS})
             for m in MANIPS}
        geo = all(c[m]["geom_ok_all"] for m in MANIPS)
        reach = geo and all(r[m]["L"] is not None and r[m]["R"] is not None and r[m]["L"] >= 0.5 and r[m]["R"] >= 0.5 for m in MANIPS)
        score = min(c[m]["sep_ns_share"] for m in MANIPS)
        margin = min(c[m]["near_margin_sep"] for m in MANIPS)
        table.append(dict(ci=ci, cand=list(cand), geom=geo, reach=reach, feasible=bool(geo and reach), score=score, margin=margin, ref=r,
                          geom_share={m: c[m]["geom_ok_share"] for m in MANIPS}, fails={m: c[m]["fails"] for m in MANIPS},
                          sep_ns={m: c[m]["sep_ns_share"] for m in MANIPS}, salient={m: c[m]["salient"] for m in MANIPS},
                          nearer={m: c[m]["nearer"] for m in MANIPS}, area_ratio_RL={m: c[m]["area_ratio_RL"] for m in MANIPS},
                          cd_L={m: c[m]["cd_L"] for m in MANIPS}, cd_R={m: c[m]["cd_R"] for m in MANIPS}))
        print("[s2cue4 pick] cand %d %s: geom %s (%s; fails %s) SEP %s salient %s nearer %s margin %.3f ref %s -> feasible %s score %.3f" % (
            ci, cand, geo, table[-1]["geom_share"], table[-1]["fails"], table[-1]["sep_ns"], table[-1]["salient"], table[-1]["nearer"],
            margin, json.dumps(r), table[-1]["feasible"], score), flush=True)
    F = [x for x in table if x["feasible"]]
    pick = None
    if F:
        best = max(x["score"] for x in F)
        tied = [x for x in F if x["score"] >= best - 0.01]
        pick = max(tied, key=lambda x: (x["margin"], -x["ci"]))
    verdict = "LAYOUT FEASIBLE" if pick is not None and pick["score"] >= FEAS_MIN else "LAYOUT INFEASIBLE"
    with open(os.path.join(SCAN, "pick.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(rule=PICK_RULE, table=table, pick=pick, verdict=verdict), fh, indent=1)
    print("[s2cue4 pick] PICK %s -> %s" % (None if pick is None else json.dumps({x: pick[x] for x in ("ci", "cand", "score", "margin", "sep_ns")}),
                                           verdict), flush=True)


def build(a):
    for m in BUILT:
        assert not os.path.exists(state_dir(m)), "exists: " + state_dir(m)
    assert not os.path.exists(STATES_JSON)
    P_ = json.load(open(os.path.join(SCAN, "pick.json"), encoding="utf-8"))
    assert P_["verdict"] == "LAYOUT FEASIBLE", "layout infeasible: feasibility-only verdict, nothing to build"
    ci = int(P_["pick"]["ci"])
    G = json.load(open(os.path.join(SCAN, "geom_%d.json" % ci), encoding="utf-8"))
    GS = json.load(open(os.path.join(SCAN, "geom_sm.json"), encoding="utf-8"))
    O_ = json.load(open(os.path.join(SCAN, "orig.json"), encoding="utf-8"))
    M_ = s2.meta()
    rows = [dict(k=k, orig=O_["orig"][k], O=O_["orig"][k]) for k in range(N_STATES)]
    for m in BUILT:
        A = np.load(os.path.join(SCAN, "states_%s_%d.npy" % (m, ci)) if m != "Sm" else os.path.join(SCAN, "states_Sm.npy"))
        rr = G[m]["rows"] if m != "Sm" else GS["Sm"]["rows"]
        assert len(A) == N_STATES == len(rr)
        os.makedirs(state_dir(m))
        for t in M_["tasks"][:4]:
            np.save(os.path.join(state_dir(m), "%s_%d.npy" % (s2.SUITE, t["tid"])), A)
        for k, r in enumerate(rr):
            assert r["k"] == k
            rows[k][m] = r
    with open(STATES_JSON, "w", encoding="utf-8") as fh:
        json.dump(dict(rule=PICK_RULE, pick=P_["pick"], plates=P_["pick"]["cand"], sm_plates=list(SM_PLATES), d0=O_["d0"],
                       image_side_agreement=O_["image_side_agreement"], source=os.path.join(CD, "O"), rows=rows), fh, indent=1)
    try:
        from PIL import Image
        z0, z, zs = (np.load(os.path.join(SCAN, f)) for f in ("frames_O.npz", "frames_%d.npz" % ci, "frames_sm.npz"))
        nm = [str(x) for x in z["names"]]
        ims = list(z0["agent"][:4]) + [z["agent"][nm.index("%s_c%d_k%d" % (m, ci, k))] for m in MANIPS for k in SMOKE] + list(zs["agent"][:4])
        h, w_ = ims[0].shape[:2]
        sheet = np.full((4 * h, 4 * w_, 3), 255, np.uint8)
        for i, im in enumerate(ims):
            sheet[(i // 4) * h:(i // 4 + 1) * h, (i % 4) * w_:(i % 4 + 1) * w_] = im
        Image.fromarray(sheet).save(SHEET)
        print("[s2cue4 build] sheet rows O / Sx / PSx / Sm, k 0-3 -> %s" % SHEET, flush=True)
    except Exception as ex:                                                 # noqa: BLE001
        print("[s2cue4 build] no sheet: %r" % ex, flush=True)
    for m in ("O",) + BUILT:
        rr = [rows[k][m] for k in MAIN]
        ok = "-" if m == "O" else sum(r["ok"] for r in rr)
        print("[s2cue4 build] %s main 40: geom ok %s | SEP near-vs-salient %d | SEP start-vs-near %s | salient %s | nearer %s | "
              "px L %s R %s | cdist L %s R %s | wrist px L %s R %s | near margin [%.3f, %.3f]" % (
                  m, ok, sum(r["sep_ns"] for r in rr), "-" if m == "O" else sum(r["sep_sn"] for r in rr),
                  {str(x): sum(r["salient"] == x for r in rr) for x in ("L", "R", None)},
                  {str(x): sum(r["nearer_left_mug"] == x for r in rr) for x in ("L", "R", None)},
                  np.percentile([r["seg"][PL]["px"] for r in rr], [0, 50, 100]).tolist(),
                  np.percentile([r["seg"][PR]["px"] for r in rr], [0, 50, 100]).tolist(),
                  np.round(np.percentile([r["seg"][PL]["cdist"] or np.nan for r in rr], [0, 50, 100]), 1).tolist(),
                  np.round(np.percentile([r["seg"][PR]["cdist"] or np.nan for r in rr], [0, 50, 100]), 1).tolist(),
                  np.percentile([r["seg"][PL].get("w_px", -1) for r in rr], [0, 50, 100]).tolist(),
                  np.percentile([r["seg"][PR].get("w_px", -1) for r in rr], [0, 50, 100]).tolist(),
                  min(r["near_margin"] or 0 for r in rr), max(r["near_margin"] or 0 for r in rr)), flush=True)
    print("[s2cue4 build] wrote %s" % CD, flush=True)


# ---------------------------------------------------------------- reference planner (C1; LIBERO venv, CPU)
def ref(a):
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    assert t["tid"] == a.tid and a.manip in BUILT
    os.makedirs(a.out_dir, exist_ok=True)
    fn = os.path.join(a.out_dir, "ref_%s_%d.json" % (a.manip, a.tid))
    assert not os.path.exists(fn), "exists: " + fn
    S = np.load(os.path.join(state_dir(a.manip), "%s_%d.npy" % (s2.SUITE, a.tid)))
    br = {int(b["trial"]): int(b["seed_base"]) for b in json.load(open(a.branches, encoding="utf-8"))}
    kp, eps = s2cue3.ref_eps(S, t, t["target"], sorted(br), lambda k: br[k])
    for e in eps:
        e["success_side"] = bool(e["success"])                              # the plates keep their order: BDDL goal = named side
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(manip=a.manip, tid=a.tid, key=t["key"], kp=kp, target_object=t["target"], eps=eps), fh, indent=1)


# ---------------------------------------------------------------- scoring (CPU)
def score4(stem, t, ks, manip, srows):
    E = s2cue2.score(stem, t, ks, manip, srows)
    for e in E:
        if e["pred"] is not None:
            e["pred"]["salient"] = srows[e["k"]][manip]["salient"]
    return E


def label_run4(E, ks):
    """s2cue2.label_run's rule unchanged, over CUES4 (SR 58's cues + salient plate)."""
    old = s2cue2.CUES
    try:
        s2cue2.CUES = CUES4
        return s2cue2.label_run(E, ks)
    finally:
        s2cue2.CUES = old


def h2h_ns(E, ks):
    """Head-to-head on placements where nearer plate and salient plate are both defined and disagree: share on the salient plate."""
    F, D = np.zeros(len(ks)), np.zeros(len(ks))
    by_mug = {W: [0, 0], Y: [0, 0]}
    for i, e in s2cue2.placements(E):
        a_, b_ = e["pred"]["salient"], e["pred"]["nearer_plate"]
        if a_ is not None and b_ is not None and a_ != b_:
            D[i] += 1
            F[i] += float(e["dest_side"] == a_)
            by_mug[e["held"]][0] += int(e["dest_side"] == a_)
            by_mug[e["held"]][1] += 1
    sh = s2cue.boot_share(F, D)
    if sh[3] < H2H_MIN:
        v = "UNTESTABLE" if sh[3] else "NOT DEFINED"
    elif sh[0] >= 0.70 and sh[1] > 0.50:
        v = "SALIENT PLATE ROUTES"
    elif sh[0] <= 0.30 and sh[2] < 0.50:
        v = "NEARER PLATE ROUTES"
    else:
        v = "UNRESOLVED"
    return dict(share_salient=sh, verdict=v, by_mug={str(m): x for m, x in by_mug.items()})


def model_verdict4(vs, vp):
    if vs == vp == "NEARER PLATE ROUTES":
        return "NEARER PLATE ROUTES"
    if vs == vp == "SALIENT PLATE ROUTES":
        return "SALIENT PLATE ROUTES"
    if vs == "NEARER PLATE ROUTES" and vp == "SALIENT PLATE ROUTES":
        return "IDENTITY"
    if vs == "SALIENT PLATE ROUTES" and vp == "NEARER PLATE ROUTES":
        return "MIXED"
    return "UNRESOLVED"


def run_stems(RD, model, manip, fe, tids):
    return {tid: os.path.join(RD, manip, "%s_%s_%d" % (model, fe, tid)) for tid in tids}


def summary(d):
    M_ = s2.meta()
    srows = {r["k"]: r for r in json.load(open(STATES_JSON, encoding="utf-8"))["rows"]}
    for model, manip, fe, _, tids in RUNS:
        for tid, stem in run_stems(d, model, manip, fe, tids).items():
            if not (os.path.exists(stem + ".json") and os.path.exists(stem + "_track.npz")):
                continue
            t = M_["tasks"][tid]
            R_ = json.load(open(stem + ".json", encoding="utf-8"))
            ks = [int(e["trial"]) for v in R_["tasks"].values() for e in v["episodes"]]
            for e in score4(stem, t, ks, manip, srows):
                print("[s2cue4 smoke] %s %s %s %s %s" % (model, manip, fe, t["key"], json.dumps(e)), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_")) if os.path.isdir(d) else []:
        R_ = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2cue4 smoke] %s: %s" % (f, [(e["trial"], e["success_side"], e.get("on")) for e in R_["eps"]]), flush=True)


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
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_cue4"]
        assert len(rows) == 1, "expected exactly one s2_cue4 prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD = MAIN, ROLL
    res = dict(states=ks, mapping=MAPPING, design=dict(plates=SJ["plates"], sm_plates=SJ["sm_plates"], pick=SJ["pick"]),
               code_mismatch=mism, runs={}, reference={}, per_model={}, salience={})
    for m in ("O",) + BUILT:
        rr = [srows[k][m] for k in ks]
        res["salience"][m] = dict(
            sep_ns=int(sum(r["sep_ns"] for r in rr)), n=len(rr), geom_ok=None if m == "O" else int(sum(r["ok"] for r in rr)),
            salient={str(x): int(sum(r["salient"] == x for r in rr)) for x in ("L", "R", None)},
            nearer_left_mug={str(x): int(sum(r["nearer_left_mug"] == x for r in rr)) for x in ("L", "R", None)},
            left_mug=sorted(set(str(r["left_mug"]) for r in rr)),
            px={s: np.percentile([r["seg"][p]["px"] for r in rr], [0, 50, 100]).tolist() for s, p in (("L", PL), ("R", PR))},
            cdist={s: np.percentile([r["seg"][p]["cdist"] or np.nan for r in rr], [0, 50, 100]).tolist() for s, p in (("L", PL), ("R", PR))},
            w_px={s: np.percentile([r["seg"][p].get("w_px", -1) for r in rr], [0, 50, 100]).tolist() for s, p in (("L", PL), ("R", PR))})
        print("== salience %s: %s" % (m, json.dumps(res["salience"][m])), flush=True)
    for model, manip, fe, role, tids in RUNS:
        name = "%s %s %s" % (model, manip, fe)
        E, miss = {}, []
        for tid, stem in run_stems(RD, model, manip, fe, tids).items():
            if not (os.path.exists(stem + ".json") and os.path.exists(stem + "_track.npz")):
                miss.append(stem)
                continue
            E[T[tid]["key"]] = score4(stem, T[tid], ks, manip, srows)
        if miss:
            res["runs"][name] = "MISSING %s" % miss
            print("== %s: MISSING %d" % (name, len(miss)), flush=True)
            continue
        row = dict(role=role, describe=s2cue2.describe(E))
        if role == "FLOOR":
            F_ = np.array([[float(E[T[t]["key"]][i]["success_bddl"]) for t in tids] for i in range(len(ks))])
            sh = s2cue.boot_share(F_.sum(1), np.full(len(ks), float(len(tids))))
            row["floor"] = dict(k1_k2=sh, testable=bool(sh[0] is not None and sh[0] >= GROOT_FLOOR))
            res["runs"][name] = row
            print("== %s (FLOOR): K1-K2 success %s -> %s" % (name, np.round(sh[:3], 3).tolist() + [sh[3]],
                                                            "TESTABLE" if row["floor"]["testable"] else "NOT TESTABLE"), flush=True)
            continue
        row.update(labels=label_run4(E, ks), h2h_sn=s2cue3.h2h(E, ks))
        if manip in MANIPS:
            row["h2h_ns"] = h2h_ns(E, ks)
        res["runs"][name] = row
        L = row["labels"]
        print("== %s (%s): placements %d | routing %s | H2H near-vs-salient %s | H2H start-vs-near %s" % (
            name, role, L["n_placements"], L["routing"], json.dumps(row.get("h2h_ns")), json.dumps(row["h2h_sn"])), flush=True)
        for c, v in L["cues"].items():
            print("   %-13s share %s -> %s" % (c, np.round([x if x is not None else np.nan for x in v["share"][:3]], 3).tolist()
                                              + [v["share"][3]], v["label"]), flush=True)
        for c, v in row["describe"].items():
            print("   %s %s" % (c, json.dumps(v)), flush=True)
    for model in ("pi05", "pi0"):
        for fe in ("I", "tv"):
            rs, rp = res["runs"].get("%s Sx %s" % (model, fe)), res["runs"].get("%s PSx %s" % (model, fe))
            if isinstance(rs, dict) and isinstance(rp, dict):
                vs, vp = rs["h2h_ns"]["verdict"], rp["h2h_ns"]["verdict"]
                res["per_model"]["%s %s" % (model, fe)] = dict(Sx=vs, PSx=vp, verdict=model_verdict4(vs, vp),
                                                               role="PRIMARY" if fe == "I" else "SECONDARY")
                print("== PER MODEL %s %s: Sx %s | PSx %s -> %s" % (model, fe, vs, vp, model_verdict4(vs, vp)), flush=True)
    g, gf = res["runs"].get("groot Sm I"), res["runs"].get("groot O I")
    if isinstance(g, dict) and isinstance(gf, dict):
        tst = gf["floor"]["testable"]
        res["per_model"]["groot I"] = dict(Sm=g["h2h_sn"]["verdict"], testable=tst,
                                           verdict=g["h2h_sn"]["verdict"] + ("" if tst else " (NOT TESTABLE: K1-K2 below .70)"))
        print("== PER MODEL groot I: %s" % json.dumps(res["per_model"]["groot I"]), flush=True)
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
    print("[s2cue4 read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("states", "scan_orig", "scan", "scan_sm", "scan_ref", "scan_pick", "build", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--m", choices=MANIPS)
    ap.add_argument("--ci", type=int)
    ap.add_argument("--manip", choices=BUILT)
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--ref-dir")
    ap.add_argument("--out", default=READ)
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    dict(states=states, scan_orig=scan_orig, scan=scan, scan_sm=scan_sm, scan_ref=scan_ref, scan_pick=scan_pick, build=build, ref=ref,
         summary=lambda a: summary(a.dir), read=read)[A.mode](A)

"""S2CUE5 (docs/SCAFFOLD_RESULTS.md 61; m3_prereg tag s2_cue5): a CONFIRMATORY test of the MAJORITY RULE on frozen pi0.5, arm I.

In scene 2 (LIBERO-10 LIVING_ROOM_SCENE5; two plates) three binary cues say where the held mug goes: NEARER plate (to the mug's
start), SALIENT plate (agentview pixel area AND centrality, from the settled start render), and the mug's TRAINED plate (white ->
left plate_1, yellow-and-white -> right plate_2).  A post-hoc reading of SR 59-60 (not registered) is the MAJORITY RULE: the
placement goes to the plate backed by 2 of the 3 cues.  Each odd-one-out configuration was seen once, on one layout each:
    NEARER ODD   (salient = trained != nearer):  SR 60 PSx
    SALIENT ODD  (nearer = trained != salient):  SR 60 Sx
    TRAINED ODD  (nearer = salient != trained):  SR 59 Sm (and PSm-free; SR 60 GR00T)
This probe registers the rule in advance and tests it on NEW states (seeds 875000 + k; none of SR 50-60's) and, per
configuration, on at least one NEW plate layout where feasible.  SR 59-60's layouts (Sx, PSx, Sm) are re-run on the new states as
REPLICATIONS, labelled that way.

pi0.5 holds the LEFT-starting mug first (SR 57-60: .925-1.00), so each layout family is designed around the left-starting mug.
For that mug, with (nearer, salient) fixed by the plates and trained fixed by which mug starts left (unswapped: white; P* = SR 57's
swap of the two mugs' xy first: yellow-and-white):
    family A  left plate at the left side (nearer), right plate in front near the midline (salient):
              A2 (white left)  -> SALIENT ODD;  PA2 (yellow-and-white left) -> NEARER ODD.   (Sx / PSx's family, new positions)
    family B  left plate in front near the midline (salient), right plate near the left mug (nearer):
              B2 (white left)  -> NEARER ODD;   PB2 (yellow-and-white left) -> SALIENT ODD.
    family C  right plate in front near the midline (nearer AND salient), left plate far forward-left:
              C2 (white left)  -> TRAINED ODD.                                             (Sm's family, new positions)
    family D  left plate in front near the midline (nearer AND salient), right plate at the right side:
              PD2 (yellow-and-white left) -> TRAINED ODD.                                  (the mirror of Sx)
Every candidate keeps each plate >= NEW_MIN from the same plate's position in SR 59's Sm and SR 60's Sx (asserted below).
Per family the layout is chosen from its candidates by PICK_RULE, before any policy run; a family whose pick fails the rule is
INFEASIBLE and is not built.

Per placement (the held mug ends On exactly one plate), the configuration is read from the held mug's own cues (whichever mug is
held): all three defined and not unanimous -> one of the three configurations, MAJORITY = the plate backed by two.  The registered
statistic and labels are in MAPPING.

States: LIBERO's sampler on K1's BDDL, np.random.seed = env.seed = 875000 + k, k = 0..43 (0-3 smoke, 4-43 main); branches trial = k,
seed_base 33000000 + 1000 k.  Prompts, server, client, launcher, client seed 7 and --bind track as SR 57-60.

Modes (LIBERO venv, py3.8):
    states                          (EGL) -> results/flawed/s2cue5/O/s2_living_<tid>.npy, branches_{smoke,main}.json
    scan_orig                       (EGL) -> SCAN/{orig.json, frames_O.npz}
    scan --fam F --ci I             (EGL) -> SCAN/{geom_<F><I>.json, states_<m>_<F><I>.npy, frames_<F><I>.npz}
    scan_rep                        (EGL) -> SCAN/{geom_rep.json, states_<m>_rep.npy, frames_rep.npz}: Sx, PSx, Sm on the new states
    scan_ref --fam F --ci I --m M --tid T (CPU) -> SCAN/ref_<m>_<F><I>_<tid>.json (reference planner, states 0-7)
    scan_pick                       (CPU) -> SCAN/pick.json
    build                           (CPU) -> results/flawed/s2cue5/<m>/s2_living_<tid>.npy, s2cue5_states.json, start-render sheet
    ref --manip M --tid T --branches B --out-dir D   (CPU)
    summary DIR                     (smoke)
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
import s2cue4                                                               # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2cue5")
ROLL = os.path.join(OUT, "s2cue5_roll")
READ = os.path.join(OUT, "s2cue5_read.json")
STATES_JSON = os.path.join(CD, "s2cue5_states.json")
SCAN = "/coc/flash10/ksingara3/.claude_tmp/pp/s2cue5_scan"
SHEET = "/coc/flash10/ksingara3/.claude_tmp/pp/s2cue5_sheet.png"
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
BASE5, BR_BASE5, N_STATES = 875000, 33000000, 44
SMOKE = list(range(0, 4))
MAIN = list(range(4, 44))
REF_SCAN_KS = list(range(0, 8))
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
SX_PLATES = (0.0, -0.28, 0.15, 0.04)                                        # SR 60's pick (s2cue4_scan/pick.json, c7)
SM_PLATES = tuple(s2cue4.SM_PLATES)                                         # SR 59's pick (.25, -.24, .10, .03)
EARLIER = dict(Sx=SX_PLATES, Sm=SM_PLATES)
NEW_MIN = 0.05
TRAINED = {W: "L", Y: "R"}
CONFIGS = ("nearer odd", "salient odd", "trained odd")


def _grid(ls, rs):
    return [(xl, yl, xr, yr) for xl, yl in ls for xr, yr in rs]


# Candidate positions from SR 59's probe map (s2cue3_probe_4712585.out: usable x in [-.10, .25], |y| <~ .30; a plate needs >= ~.12 m
# to a mug; the mugs start at x ~ -.07..-.20) and SR 60's scan (a plate at (.10, .04) shows 2136 px at 76 px from the image centre,
# at (.15, .04) 2574 px at 93 px, at the side (0, -.28) 1276 px at 116 px; the image centre lies near the mugs).
FAMILIES = dict(
    A=dict(manips=("A2", "PA2"), nearer="L", salient="R",
           cands=_grid(((-0.10, -0.30), (-0.05, -0.32)), ((0.10, 0.10), (0.15, 0.10), (0.20, 0.02), (0.05, 0.06)))),
    B=dict(manips=("B2", "PB2"), nearer="R", salient="L",
           cands=_grid(((0.15, -0.04), (0.20, -0.10)), ((-0.05, 0.04), (0.00, 0.03), (-0.10, 0.05)))),
    C=dict(manips=("C2",), nearer="R", salient="R",
           cands=_grid(((0.20, -0.28), (0.25, -0.18)), ((0.05, 0.04), (0.05, 0.08), (0.00, 0.06)))),
    D=dict(manips=("PD2",), nearer="L", salient="L",
           cands=_grid(((0.10, -0.04), (0.15, -0.04), (0.10, -0.10)), ((0.00, 0.28), (-0.05, 0.31)))))
REPS = dict(Sx=SX_PLATES, PSx=SX_PLATES, Sm=SM_PLATES)                      # REPLICATIONS on the new states
REP_DESIGN = dict(Sx=("L", "R"), PSx=("L", "R"), Sm=("R", "R"))             # (nearer, salient) for the left-starting mug
NEW_MANIPS = tuple(m for f in FAMILIES.values() for m in f["manips"])
ALL_MANIPS = NEW_MANIPS + tuple(REPS)
FEAS_MIN, CFG_MIN = 0.75, 20


def swapped(m):
    return m.startswith("P")                                                # PA2, PB2, PD2, PSx: SR 57's mug swap first


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


for _f, _v in FAMILIES.items():                                             # every candidate is a NEW layout
    for _c in _v["cands"]:
        for _e in EARLIER.values():
            assert dist(_c[:2], _e[:2]) >= NEW_MIN - 1e-9 and dist(_c[2:], _e[2:]) >= NEW_MIN - 1e-9, (_f, _c, _e)


def config_of(n, s, t):
    """(nearer, salient, trained) sides -> (configuration, majority side); (None, None) if a cue is undefined."""
    if n is None or s is None or t is None:
        return None, None
    if n == s == t:
        return "unanimous", n
    if s == t:
        return "nearer odd", s
    if n == t:
        return "salient odd", n
    return "trained odd", n


def design_config(m):
    n, s = REP_DESIGN[m] if m in REPS else next((f["nearer"], f["salient"]) for f in FAMILIES.values() if m in f["manips"])
    return config_of(n, s, TRAINED[Y if swapped(m) else W])[0]


PICK_RULE = (
    "Per family F (A, B, C, D) and candidate (XL, YL, XR, YR) in FAMILIES[F].cands (plate_1 at (XL, YL), plate_2 at (XR, YR), each "
    "keeping its height and orientation; every candidate keeps each plate >= .05 m from the same plate's position in SR 59's Sm "
    "(.25, -.24, .10, .03) and SR 60's Sx (0, -.28, .15, .04)), for each manipulation m of F (P* = SR 57's swap of the white and the "
    "yellow-and-white mug's xy first) and each of the 44 new states k: GEOM_OK as SR 59 (s2cue3.geom_ok, unchanged).  NEARER(k) = "
    "the side of the plate nearer the mug that starts left of the midline (y < -.05; exactly one of the two), defined iff "
    "|d_left - d_right| >= .02 m; SALIENT(k) = SR 60's rule unchanged (s2cue4.salient_of: >= 1.5x the other plate's agentview "
    "pixels AND centroid >= 10 px nearer the image centre, simulator segmentation of the settled start render, 256 x 256); "
    "TRAINED = that mug's trained plate (white -> left, yellow-and-white -> right).  CONFIG(k) iff all three are defined and form "
    "m's designed configuration (A2, PB2 salient odd; PA2, B2 nearer odd; C2, PD2 trained odd).  FEASIBLE iff GEOM_OK on all 44 "
    "states for every m of F and the reference planner (s2.plan_episode, unchanged) succeeds on >= .50 of the 16 episodes per plate "
    "(states 0-7; K1 + N2 for the left plate, K2 + N1 for the right) for every m of F.  Score = min over m of the share of the 44 "
    "states with CONFIG.  PICK = the feasible candidate with the highest score; ties (within .01) -> the larger min over m of the "
    "mean nearer margin |d_left - d_right| on CONFIG states, then the smaller ci.  F is NEW-LAYOUT FEASIBLE iff its pick scores "
    ">= .75; otherwise F is INFEASIBLE, reported with its numbers, and not built.")
MAPPING = (
    "Model frozen pi0.5 (pp_serve_purify --model pi05 --mode intact --comp= --seed 13), arm I only; s2.py client unchanged run by "
    "s2_roll.sh unchanged (client seed 7, --bind track); states = the NEW states (LIBERO sampler seeds 875000 + k) under each built "
    "manipulation (results/flawed/s2cue5/<m>), branches_main.json (k = 4-43, seed_base 33000000 + 1000 k).  Runs: pi0.5 <m> I for "
    "every built NEW-layout manipulation (PRIMARY) and for Sx, PSx, Sm on the new states (REPLICATION: SR 59-60's layouts); tasks "
    "K1, K2, N1, N2, 40 episodes each.  Per episode: HELD mug = libero_client's 'bound' if it is the white or the yellow-and-white "
    "mug; PLACEMENT = the held mug ends On exactly one plate (final LIBERO predicates); plates scored by SIDE (plate_1 = the left "
    "plate; the plates keep their order).  Cues of the HELD mug from the settled start state (s2cue5_states.json): NEARER PLATE = "
    "s2cue2.cue_preds' nearer_plate (undefined iff |d_left - d_right| < .02 m); SALIENT PLATE = SR 60's per-state salient side "
    "(s2cue4.salient_of); TRAINED PLATE = s2cue2.cue_preds' plate_object (white -> left, yellow-and-white -> right).  A placement is "
    "2-vs-1 iff all three are defined and not all equal; its CONFIGURATION is the odd cue (NEARER ODD: salient = trained != "
    "nearer; SALIENT ODD: nearer = trained != salient; TRAINED ODD: nearer = salient != trained) and its MAJORITY plate the side "
    "backed by the other two.  PRIMARY STATISTIC: the share of 2-vs-1 placements on the majority plate, per configuration and "
    "pooled over the three, with a PAIRED bootstrap CI: counts summed per state over runs and tasks, the same B = 2000 resamples of "
    "the 40 main states (s2cue.boot_share, default_rng(0)) for every configuration and the pool; CI = 2.5 / 97.5 percentiles.  "
    "SOURCES: for each configuration c, its placements are counted over the PRIMARY (NEW-layout) runs if any built NEW-layout "
    "manipulation is designed for c; otherwise over the REPLICATION runs, and c is labelled REPLICATION-ONLY.  The pool sums the "
    "three configurations' counts over their sources.  TESTABLE iff a configuration has >= 20 2-vs-1 placements.  LABEL: MAJORITY "
    "CONFIRMED iff the pooled lower CI >= .80 AND every configuration is TESTABLE with point estimate >= .70; REFUTED iff any "
    "TESTABLE configuration's upper CI < .50; otherwise PARTIAL (also when a configuration is untestable).  The same rule is "
    "reported, labelled REPLICATION, over the replication runs alone (secondary), and over all runs (descriptive).  C1 per built "
    "manipulation: the reference planner (s2.plan_episode, unchanged) on the same 40 states; MET iff its N1-N2 success >= .80.  "
    "Descriptive: salience numbers per layout, per-run cue labels (s2cue2.label_run's rule over SR 60's cues), first-held shares, "
    "held x destination, the majority share split by whether the instruction (named mug held) sides with the majority.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def state_dir(m):
    return os.path.join(CD, m)


# ---------------------------------------------------------------- simulator tools (LIBERO venv)
class Scene5(s2cue4.Scene4):
    """SR 60's Scene4; P* manipulations swap the two mugs' xy first (SR 59's PSm make), then place the plates."""

    def make4(self, s, m, cand):
        return self.make(s, "PSm" if swapped(m) else "Sm", cand)


def add_config(v, m):
    want = design_config(m)
    for r in v["rows"]:
        lm = r["left_mug"]
        r["config"] = config_of(r["nearer_left_mug"], r["salient"], TRAINED[lm])[0] if lm in TRAINED else None
        r["config_ok"] = bool(r["config"] == want)
    v["design_config"] = want
    v["config_share"] = float(np.mean([r["config_ok"] for r in v["rows"]]))
    ok = [r["near_margin"] for r in v["rows"] if r["config_ok"] and r["near_margin"] is not None]
    v["near_margin_cfg"] = float(np.mean(ok)) if ok else 0.0
    v["configs"] = {str(c): int(sum(r["config"] == c for r in v["rows"])) for c in CONFIGS + ("unanimous", None)}
    print("[s2cue5 scan] %s: design %s, CONFIG on %d of %d (%s), near margin on CONFIG %.3f" % (
        m, want, sum(r["config_ok"] for r in v["rows"]), len(v["rows"]), v["configs"], v["near_margin_cfg"]), flush=True)
    return v


# ---------------------------------------------------------------- states (new) and the feasibility scan
def states(a):
    import s2_tasks as T2
    od = os.path.join(CD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))

    def draw(k):                                                            # s2_tasks.build's rule with a new base
        np.random.seed(BASE5 + k)
        env.seed(BASE5 + k)
        env.reset()
        return env.sim.get_state().flatten()
    S = np.stack([draw(k) for k in range(N_STATES)])
    env.close()
    old = [np.load(os.path.join(s2.SD, "states", "%s_0.npy" % s2.SUITE))] + [
        np.load(os.path.join(OUT, d, "O", "%s_0.npy" % s2.SUITE)) for d in ("s2cue3", "s2cue4")]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("same as an SR 50 / 59 / 60 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    for ph, ks in (("smoke", SMOKE), ("main", MAIN)):
        with open(os.path.join(CD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE5 + 1000 * k, at=0, variant="s2") for k in ks], fh, indent=1)
    print("[s2cue5 states] %d new states (seed %d + k), none equal to SR 50's, 59's or 60's; wrote %s" % (len(S), BASE5, CD), flush=True)


def scan_orig(a):
    os.makedirs(SCAN, exist_ok=True)
    oj = os.path.join(SCAN, "orig.json")
    assert not os.path.exists(oj), "exists: " + oj
    S = np.load(os.path.join(CD, "O", "%s_0.npy" % s2.SUITE))
    sc = Scene5()
    orig, frames, names = [], [], []
    for k in range(len(S)):
        snap, img = sc.check(S[k], frame=k < 4)
        orig.append(s2cue4.annotate(snap))
        if k < 4:
            frames.append(img)
            names.append("O_k%d" % k)
    d0 = min(o["dmin"] for o in orig)
    side = [(o["xy"][m][1] < 0, o["seg"][m]["col"] > 127.5) for o in orig for m in (W, Y) if o["seg"][m]["col"] is not None]
    agree = float(np.mean([x == y for x, y in side]))
    print("[s2cue5 scan] D0 %.4f m; mug world side vs image side agreement %.3f of %d" % (d0, agree, len(side)), flush=True)
    print("[s2cue5 scan] unmanipulated: left mug %s; plate px L %s R %s; cdist L %.1f R %.1f; salient %s; drift_xy max %.4f; "
          "mug xy W %s Y %s" % (
              sorted(set(str(o["left_mug"]) for o in orig)), np.percentile([o["seg"][PL]["px"] for o in orig], [0, 50, 100]).tolist(),
              np.percentile([o["seg"][PR]["px"] for o in orig], [0, 50, 100]).tolist(),
              np.mean([o["seg"][PL]["cdist"] for o in orig]), np.mean([o["seg"][PR]["cdist"] for o in orig]),
              {str(v): sum(o["salient"] == v for o in orig) for v in ("L", "R", None)}, max(o["drift_xy"] for o in orig),
              np.round(np.percentile([o["xy"][W] for o in orig], [0, 50, 100], axis=0), 3).tolist(),
              np.round(np.percentile([o["xy"][Y] for o in orig], [0, 50, 100], axis=0), 3).tolist()), flush=True)
    sc.env.close()
    with open(oj, "w", encoding="utf-8") as fh:
        json.dump(dict(d0=d0, image_side_agreement=agree, orig=orig), fh, indent=1)
    np.savez_compressed(os.path.join(SCAN, "frames_O.npz"), agent=np.stack(frames), names=np.array(names))


def _scan(tag, layouts):
    """layouts: [(manip, cand)] -> SCAN/geom_<tag>.json, states_<m>_<tag>.npy, frames_<tag>.npz."""
    gj = os.path.join(SCAN, "geom_%s.json" % tag)
    assert not os.path.exists(gj), "exists: " + gj
    O_ = json.load(open(os.path.join(SCAN, "orig.json"), encoding="utf-8"))
    S = np.load(os.path.join(CD, "O", "%s_0.npy" % s2.SUITE))
    sc = Scene5()
    frames, names, out = [], [], {}
    for m, cand in layouts:
        v, A = s2cue4._scan_layout(sc, S, m, cand, O_["orig"], O_["d0"], tag, frames, names)   # prints "[s2cue4 scan] <tag> ..."
        out[m] = add_config(v, m)
        out[m]["cand"] = list(cand)
        np.save(os.path.join(SCAN, "states_%s_%s.npy" % (m, tag)), A)
    sc.env.close()
    with open(gj, "w", encoding="utf-8") as fh:
        json.dump(dict(tag=tag, **out), fh, indent=1)
    np.savez_compressed(os.path.join(SCAN, "frames_%s.npz" % tag), agent=np.stack(frames), names=np.array(names))


def scan(a):
    F = FAMILIES[a.fam]
    _scan("%s%d" % (a.fam, a.ci), [(m, F["cands"][a.ci]) for m in F["manips"]])


def scan_rep(a):
    _scan("rep", list(REPS.items()))


def scan_ref(a):
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    tag = "%s%d" % (a.fam, a.ci)
    fn = os.path.join(SCAN, "ref_%s_%s_%d.json" % (a.m, tag, a.tid))
    if os.path.exists(fn):
        return
    G = json.load(open(os.path.join(SCAN, "geom_%s.json" % tag), encoding="utf-8"))
    if not all(G[m]["geom_ok_all"] for m in FAMILIES[a.fam]["manips"]):
        print("[s2cue5 scan_ref] %s not geometry-feasible: skipped" % tag, flush=True)
        return
    S = np.load(os.path.join(SCAN, "states_%s_%s.npy" % (a.m, tag)))
    kp, eps = s2cue3.ref_eps(S, t, t["target"], REF_SCAN_KS, lambda k: BR_BASE5 + 1000 * k)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(m=a.m, tag=tag, tid=a.tid, key=t["key"], eps=eps), fh, indent=1)


def scan_pick(a):
    M_ = s2.meta()

    def refrate(m, tag, tids):
        v = []
        for tid in tids:
            f = os.path.join(SCAN, "ref_%s_%s_%d.json" % (m, tag, tid))
            if not os.path.exists(f):
                return None
            v += [float(e["success"]) for e in json.load(open(f, encoding="utf-8"))["eps"]]
        return float(np.mean(v))
    res = {}
    for fam, F in FAMILIES.items():
        table = []
        for ci, cand in enumerate(F["cands"]):
            tag = "%s%d" % (fam, ci)
            c = json.load(open(os.path.join(SCAN, "geom_%s.json" % tag), encoding="utf-8"))
            ms = F["manips"]
            r = {m: dict(L=refrate(m, tag, (0, 3)), R=refrate(m, tag, (1, 2)), per={M_["tasks"][t]["key"]: refrate(m, tag, [t]) for t in TIDS})
                 for m in ms}
            geo = all(c[m]["geom_ok_all"] for m in ms)
            reach = geo and all(r[m]["L"] is not None and r[m]["R"] is not None and r[m]["L"] >= 0.5 and r[m]["R"] >= 0.5 for m in ms)
            score = min(c[m]["config_share"] for m in ms)
            margin = min(c[m]["near_margin_cfg"] for m in ms)
            row = dict(ci=ci, tag=tag, cand=list(cand), geom=geo, reach=reach, feasible=bool(geo and reach), score=score, margin=margin,
                       ref=r, geom_share={m: c[m]["geom_ok_share"] for m in ms}, fails={m: c[m]["fails"] for m in ms},
                       configs={m: c[m]["configs"] for m in ms}, salient={m: c[m]["salient"] for m in ms},
                       nearer={m: c[m]["nearer"] for m in ms}, px_L={m: c[m]["px_L"] for m in ms}, px_R={m: c[m]["px_R"] for m in ms},
                       cd_L={m: c[m]["cd_L"] for m in ms}, cd_R={m: c[m]["cd_R"] for m in ms})
            table.append(row)
            print("[s2cue5 pick] %s %s: geom %s (%s; fails %s) configs %s salient %s nearer %s px L %s R %s cdist L %s R %s margin %.3f "
                  "ref %s -> feasible %s score %.3f" % (
                      tag, cand, geo, row["geom_share"], row["fails"], row["configs"], row["salient"], row["nearer"],
                      {m: v[1] for m, v in row["px_L"].items()}, {m: v[1] for m, v in row["px_R"].items()},
                      {m: round(v[1], 1) for m, v in row["cd_L"].items()}, {m: round(v[1], 1) for m, v in row["cd_R"].items()},
                      margin, json.dumps(r), row["feasible"], score), flush=True)
        Fe = [x for x in table if x["feasible"]]
        pick = None
        if Fe:
            best = max(x["score"] for x in Fe)
            tied = [x for x in Fe if x["score"] >= best - 0.01]
            pick = max(tied, key=lambda x: (x["margin"], -x["ci"]))
        verdict = "NEW-LAYOUT FEASIBLE" if pick is not None and pick["score"] >= FEAS_MIN else "INFEASIBLE"
        res[fam] = dict(table=table, pick=pick, verdict=verdict, manips=list(F["manips"]),
                        configs={m: design_config(m) for m in F["manips"]})
        print("[s2cue5 pick] FAMILY %s (%s): PICK %s -> %s" % (
            fam, {m: design_config(m) for m in F["manips"]},
            None if pick is None else json.dumps({x: pick[x] for x in ("tag", "cand", "score", "margin")}), verdict), flush=True)
    G = json.load(open(os.path.join(SCAN, "geom_rep.json"), encoding="utf-8"))
    for m in REPS:
        print("[s2cue5 pick] REPLICATION %s %s: geom ok %.3f, design %s, CONFIG %.3f, salient %s, nearer %s" % (
            m, G[m]["cand"], G[m]["geom_ok_share"], G[m]["design_config"], G[m]["config_share"], G[m]["salient"], G[m]["nearer"]), flush=True)
    with open(os.path.join(SCAN, "pick.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(rule=PICK_RULE, families=res), fh, indent=1)


def sources(built):
    """configuration -> (role of its source runs, manipulations)."""
    out = {}
    for c in CONFIGS:
        new = [m for m in built if m in NEW_MANIPS and design_config(m) == c]
        out[c] = ("PRIMARY", new) if new else ("REPLICATION-ONLY", [m for m in built if m in REPS])
    return out


def build(a):
    assert not os.path.exists(STATES_JSON)
    P_ = json.load(open(os.path.join(SCAN, "pick.json"), encoding="utf-8"))
    O_ = json.load(open(os.path.join(SCAN, "orig.json"), encoding="utf-8"))
    GR = json.load(open(os.path.join(SCAN, "geom_rep.json"), encoding="utf-8"))
    M_ = s2.meta()
    lay = []                                                                # (manip, tag, rows, cand, role, family)
    for fam, v in P_["families"].items():
        if v["verdict"] != "NEW-LAYOUT FEASIBLE":
            continue
        tag = v["pick"]["tag"]
        G = json.load(open(os.path.join(SCAN, "geom_%s.json" % tag), encoding="utf-8"))
        for m in v["manips"]:
            lay.append((m, tag, G[m]["rows"], v["pick"]["cand"], "PRIMARY", fam))
    for m, cand in REPS.items():
        lay.append((m, "rep", GR[m]["rows"], list(cand), "REPLICATION", "rep"))
    built = [x[0] for x in lay]
    for m in built:
        assert not os.path.exists(state_dir(m)), "exists: " + state_dir(m)
    rows = [dict(k=k, orig=O_["orig"][k], O=O_["orig"][k]) for k in range(N_STATES)]
    design = {}
    for m, tag, rr, cand, role, fam in lay:
        A = np.load(os.path.join(SCAN, "states_%s_%s.npy" % (m, tag)))
        assert len(A) == N_STATES == len(rr)
        os.makedirs(state_dir(m))
        for t in M_["tasks"][:4]:
            np.save(os.path.join(state_dir(m), "%s_%d.npy" % (s2.SUITE, t["tid"])), A)
        for k, r in enumerate(rr):
            assert r["k"] == k
            rows[k][m] = r
        design[m] = dict(tag=tag, plates=cand, role=role, family=fam, config=design_config(m), swapped=swapped(m))
    src = sources(built)
    with open(STATES_JSON, "w", encoding="utf-8") as fh:
        json.dump(dict(rule=PICK_RULE, families={f: dict(verdict=v["verdict"], pick=v["pick"]) for f, v in P_["families"].items()},
                       built=built, design=design, sources=src, sx_plates=list(SX_PLATES), sm_plates=list(SM_PLATES), d0=O_["d0"],
                       image_side_agreement=O_["image_side_agreement"], source=os.path.join(CD, "O"), rows=rows), fh, indent=1)
    try:
        from PIL import Image
        ims = list(np.load(os.path.join(SCAN, "frames_O.npz"))["agent"][:4])
        for m, tag, *_ in lay:
            z = np.load(os.path.join(SCAN, "frames_%s.npz" % tag))
            nm = [str(x) for x in z["names"]]
            ims += [z["agent"][nm.index("%s_%s_k%d" % (m, tag, k))] for k in SMOKE]
        h, w_ = ims[0].shape[:2]
        nr = len(ims) // 4
        sheet = np.full((nr * h, 4 * w_, 3), 255, np.uint8)
        for i, im in enumerate(ims):
            sheet[(i // 4) * h:(i // 4 + 1) * h, (i % 4) * w_:(i % 4 + 1) * w_] = im
        Image.fromarray(sheet).save(SHEET)
        print("[s2cue5 build] sheet rows O / %s, k 0-3 -> %s" % (" / ".join(built), SHEET), flush=True)
    except Exception as ex:                                                 # noqa: BLE001
        print("[s2cue5 build] no sheet: %r" % ex, flush=True)
    for m in ["O"] + built:
        rr = [rows[k][m] for k in MAIN]
        print("[s2cue5 build] %s %s main 40: geom ok %s | design %s | CONFIG %s | configs %s | salient %s | nearer %s | left mug %s | "
              "px L %s R %s | cdist L %s R %s | wrist px L %s R %s | near margin [%.3f, %.3f]" % (
                  m, "-" if m == "O" else design[m]["plates"], "-" if m == "O" else sum(r["ok"] for r in rr),
                  "-" if m == "O" else design[m]["config"], "-" if m == "O" else sum(r["config_ok"] for r in rr),
                  "-" if m == "O" else {str(c): sum(r["config"] == c for r in rr) for c in CONFIGS + ("unanimous", None)},
                  {str(x): sum(r["salient"] == x for r in rr) for x in ("L", "R", None)},
                  {str(x): sum(r["nearer_left_mug"] == x for r in rr) for x in ("L", "R", None)},
                  sorted(set(str(r["left_mug"]) for r in rr)),
                  np.percentile([r["seg"][PL]["px"] for r in rr], [0, 50, 100]).tolist(),
                  np.percentile([r["seg"][PR]["px"] for r in rr], [0, 50, 100]).tolist(),
                  np.round(np.percentile([r["seg"][PL]["cdist"] or np.nan for r in rr], [0, 50, 100]), 1).tolist(),
                  np.round(np.percentile([r["seg"][PR]["cdist"] or np.nan for r in rr], [0, 50, 100]), 1).tolist(),
                  np.percentile([r["seg"][PL].get("w_px", -1) for r in rr], [0, 50, 100]).tolist(),
                  np.percentile([r["seg"][PR].get("w_px", -1) for r in rr], [0, 50, 100]).tolist(),
                  min(r["near_margin"] or 0 for r in rr), max(r["near_margin"] or 0 for r in rr)), flush=True)
    print("[s2cue5 build] built %s; sources %s; wrote %s" % (built, json.dumps(src), CD), flush=True)


# ---------------------------------------------------------------- reference planner (C1; LIBERO venv, CPU)
def ref(a):
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    assert t["tid"] == a.tid and a.manip in ALL_MANIPS and os.path.isdir(state_dir(a.manip))
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
def score5(stem, t, ks, manip, srows):
    E = s2cue4.score4(stem, t, ks, manip, srows)                            # SR 60's scoring: SR 58's cues + salient plate
    for e in E:
        p = e["pred"]
        if p is not None:
            p["config"], p["majority"] = config_of(p["nearer_plate"], p["salient"], p["plate_object"])
    return E


def counts(Es, ks, cfgs):
    """Es: [E (task -> episodes)] -> per-state (on majority, 2-vs-1 placements) over configurations cfgs; + instruction split."""
    F, D = np.zeros(len(ks)), np.zeros(len(ks))
    ins = dict(agree=[0, 0], disagree=[0, 0], undefined=[0, 0])
    for E in Es:
        for i, e in s2cue2.placements(E):
            p = e["pred"]
            if p is None or p["config"] not in cfgs:
                continue
            hit = int(e["dest_side"] == p["majority"])
            D[i] += 1
            F[i] += hit
            key = "undefined" if p["instruction"] is None else ("agree" if p["instruction"] == p["majority"] else "disagree")
            ins[key][0] += hit
            ins[key][1] += 1
    return F, D, ins


def label5(per, pooled):
    tst = {c: per[c]["share"][3] >= CFG_MIN for c in CONFIGS}
    if any(tst[c] and per[c]["share"][2] < 0.50 for c in CONFIGS):
        return "REFUTED"
    if pooled["share"][1] is not None and pooled["share"][1] >= 0.80 and all(tst[c] and per[c]["share"][0] >= 0.70 for c in CONFIGS):
        return "MAJORITY CONFIRMED"
    return "PARTIAL"


def majority_test(Eruns, ks, src):
    """Eruns: manip -> E; src: configuration -> list of manips counted for it."""
    per, Ft, Dt = {}, np.zeros(len(ks)), np.zeros(len(ks))
    for c in CONFIGS:
        F, D, ins = counts([Eruns[m] for m in src[c] if m in Eruns], ks, (c,))
        Ft += F
        Dt += D
        sh = s2cue.boot_share(F, D)
        per[c] = dict(share=sh, testable=bool(sh[3] >= CFG_MIN), runs=list(src[c]), instruction=ins)
    pooled = dict(share=s2cue.boot_share(Ft, Dt))
    return dict(per_config=per, pooled=pooled, label=label5(per, pooled))


def run_stem(RD, m, tid):
    return os.path.join(RD, m, "pi05_I_%d" % tid)


def summary(d):
    M_ = s2.meta()
    SJ = json.load(open(STATES_JSON, encoding="utf-8"))
    srows = {r["k"]: r for r in SJ["rows"]}
    for m in SJ["built"]:
        for tid in TIDS:
            stem = run_stem(d, m, tid)
            if not (os.path.exists(stem + ".json") and os.path.exists(stem + "_track.npz")):
                continue
            t = M_["tasks"][tid]
            R_ = json.load(open(stem + ".json", encoding="utf-8"))
            ks = [int(e["trial"]) for v in R_["tasks"].values() for e in v["episodes"]]
            for e in score5(stem, t, ks, m, srows):
                p = e["pred"] or {}
                print("[s2cue5 smoke] %s %s k%d held %s dest %s | config %s majority %s | n %s s %s t %s | success %s" % (
                    m, t["key"], e["k"], e["held"], e["dest_side"], p.get("config"), p.get("majority"), p.get("nearer_plate"),
                    p.get("salient"), p.get("plate_object"), e["success_side"]), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_")) if os.path.isdir(d) else []:
        R_ = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2cue5 smoke] %s: %s" % (f, [(e["trial"], e["success_side"]) for e in R_["eps"]]), flush=True)


def read(a):
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    M_ = s2.meta()
    T = {t["tid"]: t for t in M_["tasks"]}
    SJ = json.load(open(STATES_JSON, encoding="utf-8"))
    srows = {r["k"]: r for r in SJ["rows"]}
    built, design = SJ["built"], SJ["design"]
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD, mism = SMOKE, a.dir, "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_cue5"]
        assert len(rows) == 1, "expected exactly one s2_cue5 prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD = MAIN, ROLL
    res = dict(states=ks, mapping=MAPPING, families=SJ["families"], design=design, sources=SJ["sources"], code_mismatch=mism,
               salience={}, runs={}, reference={})
    for m in ["O"] + built:
        rr = [srows[k][m] for k in ks]
        res["salience"][m] = dict(
            n=len(rr), geom_ok=None if m == "O" else int(sum(r["ok"] for r in rr)),
            config_ok=None if m == "O" else int(sum(r["config_ok"] for r in rr)),
            salient={str(x): int(sum(r["salient"] == x for r in rr)) for x in ("L", "R", None)},
            nearer_left_mug={str(x): int(sum(r["nearer_left_mug"] == x for r in rr)) for x in ("L", "R", None)},
            left_mug=sorted(set(str(r["left_mug"]) for r in rr)),
            px={s: np.percentile([r["seg"][p]["px"] for r in rr], [0, 50, 100]).tolist() for s, p in (("L", PL), ("R", PR))},
            cdist={s: np.percentile([r["seg"][p]["cdist"] or np.nan for r in rr], [0, 50, 100]).tolist() for s, p in (("L", PL), ("R", PR))},
            w_px={s: np.percentile([r["seg"][p].get("w_px", -1) for r in rr], [0, 50, 100]).tolist() for s, p in (("L", PL), ("R", PR))},
            near_margin=[float(min(r["near_margin"] or 0 for r in rr)), float(max(r["near_margin"] or 0 for r in rr))])
        print("== salience %s %s: %s" % (m, design.get(m, {}).get("plates", "sampled"), json.dumps(res["salience"][m])), flush=True)
    Eruns = {}
    for m in built:
        name = "pi05 %s I" % m
        E, miss = {}, []
        for tid in TIDS:
            stem = run_stem(RD, m, tid)
            if not (os.path.exists(stem + ".json") and os.path.exists(stem + "_track.npz")):
                miss.append(stem)
                continue
            E[T[tid]["key"]] = score5(stem, T[tid], ks, m, srows)
        if miss:
            res["runs"][name] = "MISSING %s" % miss
            print("== %s: MISSING %d" % (name, len(miss)), flush=True)
            continue
        Eruns[m] = E
        cfg = {}
        for c in CONFIGS + ("unanimous",):
            F, D, _ = counts([E], ks, (c,))
            cfg[c] = [int(F.sum()), int(D.sum())]
        undefined = sum(1 for _, e in s2cue2.placements(E) if e["pred"] is None or e["pred"]["config"] is None)
        row = dict(role=design[m]["role"], design_config=design[m]["config"], configs=cfg, undefined=undefined,
                   describe=s2cue2.describe(E), labels=s2cue4.label_run4(E, ks), h2h_ns=s2cue4.h2h_ns(E, ks))
        res["runs"][name] = row
        L = row["labels"]
        print("== %s (%s, design %s): placements %d | 2-vs-1 by config (on majority, n) %s | undefined %d | routing %s | H2H NS %s" % (
            name, row["role"], row["design_config"], L["n_placements"], json.dumps(cfg), undefined, L["routing"],
            json.dumps(row["h2h_ns"])), flush=True)
        for c, v in L["cues"].items():
            print("   %-13s share %s -> %s" % (c, np.round([x if x is not None else np.nan for x in v["share"][:3]], 3).tolist()
                                              + [v["share"][3]], v["label"]), flush=True)
        for c, v in row["describe"].items():
            print("   %s %s" % (c, json.dumps(v)), flush=True)
    src = {c: SJ["sources"][c][1] for c in CONFIGS}
    reps = [m for m in built if design[m]["role"] == "REPLICATION"]
    res["majority"] = dict(
        PRIMARY=dict(sources={c: SJ["sources"][c] for c in CONFIGS}, **majority_test(Eruns, ks, src)),
        REPLICATION=majority_test(Eruns, ks, {c: reps for c in CONFIGS}),
        ALL=majority_test(Eruns, ks, {c: built for c in CONFIGS}))
    for nm, v in res["majority"].items():
        print("== MAJORITY %s: label %s | pooled %s" % (nm, v["label"], np.round([x if x is not None else np.nan for x in v["pooled"]["share"][:3]], 3).tolist()
                                                     + [v["pooled"]["share"][3]]), flush=True)
        for c in CONFIGS:
            pc = v["per_config"][c]
            print("   %-12s %s testable %s runs %s instruction %s" % (
                c, np.round([x if x is not None else np.nan for x in pc["share"][:3]], 3).tolist() + [pc["share"][3]], pc["testable"],
                pc["runs"], json.dumps(pc["instruction"])), flush=True)
    RF = a.ref_dir or RD
    for m in built:
        per = {}
        for tid in TIDS:
            f = os.path.join(RF, "ref_%s_%d.json" % (m, tid))
            if os.path.exists(f):
                ee = json.load(open(f, encoding="utf-8"))["eps"]
                assert [e["trial"] for e in ee] == ks, (f, [e["trial"] for e in ee])
                per[T[tid]["key"]] = dict(success_side=float(np.mean([e["success_side"] for e in ee])),
                                          errors=int(sum(e.get("error") is not None for e in ee)))
        new = [per[c]["success_side"] for c in ("N1", "N2") if c in per]
        res["reference"][m] = dict(per_task=per, success_new=float(np.mean(new)) if len(new) == 2 else None,
                                   C1=(None if len(new) < 2 else ("C1 MET" if np.mean(new) >= 0.80 else "C1 UNMET")))
        print("== reference %s: %s" % (m, json.dumps(res["reference"][m])), flush=True)
    print("== code hash mismatches: %s" % mism, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, out)
    print("[s2cue5 read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("states", "scan_orig", "scan", "scan_rep", "scan_ref", "scan_pick", "build", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--fam", choices=tuple(FAMILIES))
    ap.add_argument("--ci", type=int)
    ap.add_argument("--m", choices=NEW_MANIPS)
    ap.add_argument("--manip", choices=ALL_MANIPS)
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--ref-dir")
    ap.add_argument("--out", default=READ)
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    dict(states=states, scan_orig=scan_orig, scan=scan, scan_rep=scan_rep, scan_ref=scan_ref, scan_pick=scan_pick, build=build, ref=ref,
         summary=lambda a: summary(a.dir), read=read)[A.mode](A)

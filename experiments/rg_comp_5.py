"""RVW5 COPY of rg_comp_z.py (SR 91) for the FIFTH fresh states (results/flawed/rg_comp_5): F7 candidates k = 860-939, F6 candidates k = 940-1019 (raaf_plan 7.1 recipe verbatim; docs/notes/plan_review_atlas.md 3.1).  Only the constants and the output names differ from rg_comp_z.py.  Original: SR 91 COPY of rg_comp_y.py (SR 90; rg_comp_x SR 87; rg_comp SR 70) for the FOURTH fresh states (results/flawed/rg_comp_5): F7 candidates k = 600-679, F6 candidates k = 700-779, first 4 accepted = smoke, next 60 = main per cell; every other line is rg_comp_y.py unchanged (docs/notes/sr91_plan.md).  Seeds: F7 sampler 820000 + k, layout 821000 + k; F6 810000 + k, plate 811000 + k.
"""
"""RG_COMP (docs/SCAFFOLD_RESULTS.md 70; m3_prereg tag rg_comp): does a composite RC (the F7R referent resolver + the F6 memory slot
+ ReGuide transport over a FROZEN VLA) beat ReGuide alone (rgp) and the intact VLA (I) on F7 (relational / negated language) and on
F6 (a target seen once, then covered), with no cost on trained tasks?  Scene 1 (LIBERO-Goal), the F7 / F6 harnesses of SR 45.

THE ARMS (per task; the policy is frozen and served as in SR 45 / 65; the prompt is the task's instruction unless stated)
  I     the harness alone (fx_client.py's / f6o2_client.py's env additions, restated here for the rgc suite: the hidden second bowl
        of the unique-name controls, f6o's covers, the track).
  rgp   ReGuide RG-perc (SR 52's controller, experiments/rg.py, unchanged; statistics results/flawed/rg_stats.json) given the
        instruction as it is.  Its referents come from its own noun parser (rg_roles: the first and the last LIBERO-Goal noun of the
        instruction, fa1_common.NOUN; equal to fa1_common.roles on every trained instruction).  Its perception is SR 52 / 55 / 65's:
        the learned U-Net (sc_seg) + the RENDERED depth + the camera calibration through sc_pu.Estimator, registered ONCE at the
        first policy query (paper C.6), the held object then rigid in the hand from ProprioHeld's onset.  The parser names "bowl";
        two identical bowls give the bowl class two instances, and the Estimator would average them, so rgp takes ONE instance the
        way a detector returns its top box for a noun: the bowl class's 8-connected components in the agentview at the first query
        (fx_r.components, >= fx_r.MIN_COMP px); if there are >= 2, rgp binds the LARGEST and registers the second as a separate,
        non-instructed movable bowl (both through the Estimator, each with the other's pixels removed: every bowl pixel is given to
        the nearer of the two components' 3D centres).  With one component (every one-bowl task) nothing is restricted, so rgp is
        exactly SR 65's arm there.  The binding is blind to the relation.
  rc    the composite.  (1) The F7R resolver (experiments/fx_r.py, unchanged: template parser + learned segmenter + depth) turns a
        relational instruction into one bowl instance; the VLA gets F7R's unique-name prompt and F7R's role view (the other bowl
        painted magenta in both cameras); an instruction off F7R's 4 templates passes through unchanged (no paint).  (2) ReGuide
        (rgp's perception and controller) binds the RESOLVED instance as the instructed object (registered with the other bowl's
        pixels removed) and the parsed target as the destination; the other bowl is a non-instructed movable, as in rgp.  Off
        template, ReGuide binds as rgp.  (3) The memory slot (SR 45.6's overlay, f6o2_tasks.Memory, unchanged paint rule) when the
        destination is the plate: at the first query it stores the plate's learned mask, sprite and depth-projected plane; it
        LATCHES at the first later step whose agentview plate component (fx_r.components on the learned segmenter) has fewer than
        LATCH x the stored mask's pixels, and from then on repaints the remembered plate in both cameras (occlusion-aware:
        12 cm height gate, gripper column, bowl pixels).  ReGuide's plate pose is the registration at the first query, held for the
        episode (C.6: no tracking): that registration IS the slot's pose memory, and it is shared with rgp.
  f7r   reference (F7 only): F7R alone (SR 45.2's arm: resolver, unique-name prompt, role view), no ReGuide.
  mem   reference (F6 only): the memory slot alone (rc's slot, latched trigger), no ReGuide.
END ARMS

Tasks (suite "rgc"; tid order):
  0 F7_SP_relL "put the left bowl on the plate"                   1 F7_SP_relR "put the right bowl on the plate"
  2 F7_NG_pos  "put the bowl that is on the plate on the stove"   3 F7_NG_neg  "put the bowl that is not on the plate on the stove"
  4 F7_SP_uniL / 5 F7_SP_uniR "put the bowl on the plate", 6 F7_NG_uniP "put the bowl on the stove": unique-name controls (the other
    bowl under the table, hidden from render from the first query), arm I only, for the testability gate.
  7 K2t = F7_NG_uniT "put the bowl on the stove", the bowl alone on its training spot: LIBERO-Goal K2 in its trained layout.
  8 F6_b  "put the bowl on the plate", displaced plate; cover 0 moves onto the plate at env step WAIT + HIDE_AT (f6o's b).
  9 F6_a  the same states, both covers parked (f6o's a; arm I only, the unhidden-plate ceiling and the F6 gate).
  10 K1t "put the bowl on the plate", the familiar (sampler) state, both covers parked: LIBERO-Goal K1 in its trained layout.
States: NEW.  F7: fx's two-bowl rule (fx_tasks.main's F7 loop, copied; sampler seed 820000 + k, layout seed 821000 + k) for k = 300,
301, ... (fx used k 0-23); first 4 accepted = smoke, next 40 = main.  F6: fx's F6 rule (sampler 810000 + k, plate displacement
811000 + k; f6o2_tasks.displace) for k = 400, 401, ... (fx / f6o2 used k 0-193), f6o's covers, park spots (830000 + k fam,
830000 + 100 + k disp) and render check (a, b; fam a); first 4 accepted = smoke, next 60 = main.  Branch seed_base 14000000 + 1000 k
(fx's rule; no seed reused from fx / f6o / f6o2).  Before drawing, both rules must reproduce fx's stored states k = 0-1.

Modes (LIBERO venv, py3.8, EGL, one GPU except read):
    python experiments/rg_comp.py build --step f7|f6|assemble    new states (f7, f6: resumable, cached), render check; assemble
                                                                 writes results/flawed/rg_comp/
    python experiments/rg_comp.py pcheck [--out X]               policy-free perception check on every row (resolver, rgp binding,
                                                                 registration error, memory slot record and latch)
    python experiments/rg_comp.py client --arm I|rgp|rc|f7r|mem <libero_client.py's arguments>     one rgc task per process
    python experiments/rg_comp.py read --phase smoke|main --roll DIR --out X.json [--models pi05+pi0]     (CPU)
"""
import argparse
import collections
import copy
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

OUT = os.path.join(ROOT, "results", "flawed", "rg_comp_5")
ST = os.path.join(OUT, "states")
META = os.path.join(OUT, "rg_comp_5_tasks.json")
PCHECK = os.path.join(OUT, "pcheck.json")
SUITE = "rgc"
B1, B2 = "akita_black_bowl_1", "akita_black_bowl_2"
WAIT, HIDE_AT, REPLAN = 10, 5, 5
F7_K0, F6_K0, N_CAND7, N_CAND6 = 860, 940, 80, 80     # RVW5 (plan_review_atlas 3.1, raaf_plan 7.1): fifth fresh k ranges (SR 91 used F7 600-679, F6 700-779; earlier F7 300-599, F6 400-699; fx / f6o2 0-193)
_OLD7, _OLD6 = list(range(0, 194)) + list(range(300, 680)), list(range(0, 194)) + list(range(400, 780))     # every k used by an earlier F7 / F6 set (SR 45-91)
_S7 = lambda ks: {b + k for k in ks for b in (820000, 821000)}                                   # noqa: E731  F7 sampler / layout seeds
_S6 = lambda ks: {b + k for k in ks for b in (810000, 811000, 830000, 830100)}                   # noqa: E731  F6 sampler / plate / park seeds
assert not (_S7(range(F7_K0, F7_K0 + N_CAND7)) & _S7(_OLD7)) and not (_S6(range(F6_K0, F6_K0 + N_CAND6)) & _S6(_OLD6)), "a new seed equals an earlier set's"
assert F7_K0 > max(_OLD7) and F6_K0 > max(_OLD6) and F7_K0 + N_CAND7 <= F6_K0
N_SMOKE, N7_MAIN, N6_MAIN = 4, 60, 60                       # SR 87: 60 main states per F7 cell (SR 70 had 40)
LATCH = 0.5
UGATE, AGATE, FLOOR, NI_MARGIN = 0.50, 0.50, 0.70, -0.05
B = 2000
LIFT = 0.03
MODELS, PRIMARY = ("pi05", "pi0"), "pi05"
ARMS = ("I", "rgp", "rc", "f7r", "mem")
# key, family, kind, fx task key (bddl, language), states kind, hide, cover cond, arms
TASKS = (
    ("F7_SP_relL", "SP", "f7", "F7_SP_relL", "SP_L", None, None, ("I", "rgp", "rc", "f7r")),
    ("F7_SP_relR", "SP", "f7", "F7_SP_relR", "SP_R", None, None, ("I", "rgp", "rc", "f7r")),
    ("F7_NG_pos", "NG", "f7", "F7_NG_pos", "NG_P", None, None, ("I", "rgp", "rc", "f7r")),
    ("F7_NG_neg", "NG", "f7", "F7_NG_neg", "NG_T", None, None, ("I", "rgp", "rc", "f7r")),
    ("F7_SP_uniL", "SP", "f7", "F7_SP_uniL", "SP_UL", B2, None, ("I",)),
    ("F7_SP_uniR", "SP", "f7", "F7_SP_uniR", "SP_UR", B2, None, ("I",)),
    ("F7_NG_uniP", "NG", "f7", "F7_NG_uniP", "NG_UP", B2, None, ("I",)),
    ("K2t", "K", "f7", "F7_NG_uniT", "NG_UT", B2, None, ("I", "rgp", "rc")),
    ("F6_b", "F6", "f6", "F6_fam_vis", "disp", None, "b", ("I", "rgp", "rc", "mem")),
    ("F6_a", "F6", "f6", "F6_fam_vis", "disp", None, "a", ("I",)),
    ("K1t", "K", "f6", "F6_fam_vis", "fam", None, "a", ("I", "rgp", "rc")),
)
REL = {"SP": (0, 1), "NG": (2, 3)}
UNI = {"SP": (4, 5), "NG": (6, 7)}
REL_UNI = {0: 4, 1: 5, 2: 6, 3: 7}                                         # each relational task's own unique-name control
NOUN_TARGET = {"plate": "plate_1", "stove": "flat_stove_1_cook_region"}

MAPPING = (
    "NEW states (F7: fx's two-bowl rule, k = 300+, 40 main; F6: fx's F6 rule + f6o's covers, k = 400+, 60 main), branches "
    "seed_base 14000000 + 1000 k; client seed 7, --bind track.  Models: frozen pi0.5 (PRIMARY; pp_serve_purify --model pi05 --mode "
    "intact --comp '' --seed 13, as SR 45) and frozen pi0 (SECONDARY; serve_guided --mode ours --seed 13, BACKBONE_CONFIG="
    "pi0_libero, as SR 45.3 / 65).  Success = the BDDL goal.  Bootstrap over states (B = 2000, numpy default_rng(0), percentile "
    "2.5 / 97.5); an F7 family = its two relational tasks on the same states, paired; K = the mean of K1t (F6 states) and K2t (F7 "
    "states), each task's states resampled separately in the same draw.  GATES per model: (floor, SR 50) TESTABLE iff I's K >= .70; "
    "(F7 family) TESTABLE iff U = the mean of I's two unique-name controls of the family >= .50 (fx's NO POWER rule, without the "
    "reference); (F6) TESTABLE iff I's F6_a >= .50 (f6o's NO POWER rule, without the reference).  A label on an untestable "
    "cell reads NOT TESTABLE with the rule's outcome descriptive.  REGISTERED CONTRASTS per model: (V1) RC - rgp on each F7 family "
    "(SP, NG): RC WINS iff lo > 0; RG WINS iff hi < 0; else NO CLEAR DIFFERENCE.  (V2) RC - rgp on F6_b: the same labels.  (V3) RC - "
    "I on each cell (SP, NG, F6_b): dF = RC - I on the cell, dK = RC - I on K: FIX WORKS iff dF lo > 0 AND dK lo > -.05; PARTIAL iff "
    "exactly one holds; FAILS iff neither (SR 62-69's rule); dK alone is labelled NON-INFERIOR iff lo > -.05, else NOT SHOWN.  "
    "PRIMARY: V1-V3 on pi0.5.  RECORDED (descriptive): the resolver's accuracy (rc, f7r: the resolved bowl is the referent, against "
    "the simulator); rgp's binding (the bound instance is the referent); registration error per entity (cm, against the simulator); "
    "the memory slot's record (mask IoU and centre xy error against the true plate), the latch step and painted pixels; ReGuide's "
    "plate-pose error held over the episode (registered vs true plate at the end); C = referent grasps among bowl grasps per arm and "
    "family (fx's C); per relational task, its own unique-name control (a task whose control is < .50 is flagged CEILING); rgp - I, "
    "f7r - I, mem - I, RC - f7r, RC - mem; ReGuide transports.  No reference planner is run (SR 55 / 65 ran none for ReGuide).")


# ---------------------------------------------------------------- the noun parser ReGuide uses (rgp; rc off template)
def rg_roles(instr):
    """-> (object entity, destination entity): the first and the last LIBERO-Goal noun in the instruction (fa1_common.NOUN)."""
    import fa1_common as FC
    s = " " + " ".join(instr.lower().strip().rstrip(".").split()) + " "
    hits = []
    for w, ent in FC.NOUN.items():
        i = s.find(" " + w + " ")
        while i >= 0:
            hits.append((i, w, ent))
            i = s.find(" " + w + " ", i + 1)
    hits.sort()
    assert len(hits) >= 2, instr
    oe, de = hits[0][2], hits[-1][2]
    try:
        r = FC.roles(instr)
        assert (oe, de) == r, (instr, (oe, de), r)
    except (ValueError, KeyError):  # template parse has no relational nouns ("left bowl"); cross-check only where it applies
        pass
    return oe, de


# ---------------------------------------------------------------- build (states)
def f7_one(env, k, C):
    """fx_tasks.main's F7 loop body for sampler index k (copied unchanged; fx_tasks.py writes files when run, so it cannot be called).
    -> {kind: state}; AssertionError where fx would assert."""
    FT = C["FT"]
    lo, hi, fixp, sf = C["lo"], C["hi"], C["fixp"], C["sf"]
    mov5 = [B1, B2, "cream_cheese_1", "wine_bottle_1", "plate_1"]
    others5 = [n for n in mov5 if n != B2]
    R5 = [FT.MOVR[n] for n in mov5]
    base = FT.sampled(env, FT.BASE7 + k)
    FT.load(env, base)
    P0 = FT.pos(env, mov5)
    _, _, rest = FT.settle(env, base, mov5)
    zrest = {B1: rest[B1][2], B2: rest[B1][2]}
    rng = np.random.default_rng(FT.LAY7 + k)
    done, n_try = False, 0
    for _ in range(100000):
        x0, y0, sep = rng.uniform(-0.16, -0.10), rng.uniform(-0.02, 0.04), rng.uniform(*FT.SP_SEP)
        L = np.array([x0 + rng.uniform(-0.01, 0.01), y0 + sep / 2])
        Rr = np.array([x0 + rng.uniform(-0.01, 0.01), y0 - sep / 2])
        P = np.array([L, Rr] + [P0[n][:2] for n in mov5[2:]])
        ok = all(FT.clear(P[i], i, P, R5, fixp, sf, lo, hi, among=(0, 1, 4)) for i in (0, 1))
        if not ok:
            continue
        for i in (2, 3):
            tries = 0
            while not FT.clear(P[i], i, P, R5, fixp, sf, lo, hi) and tries < 2000:
                P[i] = rng.uniform(lo, hi)
                tries += 1
        if not all(FT.clear(P[i], i, P, R5, fixp, sf, lo, hi) for i in range(4)):
            continue
        n_try += 1
        if n_try > 80:
            break
        s1 = FT.edit(env, base, xyz={n: P[i] for i, n in enumerate(mov5)})
        sL, a0, a1 = FT.settle(env, s1, mov5)
        if FT.xy_moved(a0, a1, mov5) > FT.SETTLE_MAX or not FT.seated(env, sL, [B1, B2], zrest):
            continue
        sL, a2, a3 = FT.settle(env, sL, mov5)
        if FT.xy_moved(a2, a3, mov5) > 0.005:
            continue
        sR = FT.edit(env, sL, swap=(B1, B2))
        _, c0, c1 = FT.settle(env, sR, mov5)
        if FT.xy_moved(c0, c1, mov5) <= FT.SETTLE_MAX and FT.seated(env, sR, [B1, B2], zrest):
            done = True
            break
    assert done, f"F7 state {k}: no SP layout ({n_try} physical tries)"
    sUL, sUR = FT.far_spot(env, sL, B2, others5)[0], FT.far_spot(env, sR, B2, others5)[0]
    pp = P0["plate_1"]
    s1 = FT.edit(env, base, xyz={B1: (pp[0], pp[1], P0[B1][2] + 0.03), B2: (P0[B1][0], P0[B1][1], P0[B1][2])})
    sP, b0, b1 = FT.settle(env, s1, mov5, steps=40)
    on_plate = float(np.linalg.norm(b1[B1][:2] - b1["plate_1"][:2]))
    assert on_plate < 0.03 and b1[B1][2] > b1[B2][2] + 0.005 and \
        FT.xy_moved(b0, b1, [B2, "cream_cheese_1", "wine_bottle_1", "plate_1"]) <= FT.SETTLE_MAX, \
        f"F7 state {k}: bowl not seated on the plate ({on_plate:.3f} m)"
    sP, _, _ = FT.settle(env, sP, mov5)
    sT = FT.edit(env, sP, swap=(B1, B2))
    sUP, sUT = FT.far_spot(env, sP, B2, others5)[0], FT.far_spot(env, sT, B2, others5)[0]
    FT.load(env, sL)
    q = FT.pos(env, mov5)
    return dict(SP_L=sL, SP_R=sR, SP_UL=sUL, SP_UR=sUR, NG_P=sP, NG_T=sT, NG_UP=sUP, NG_UT=sUT), \
        dict(sep=float(abs(q[B1][1] - q[B2][1])), SP={n: q[n].tolist() for n in mov5}, NG={n: b1[n].tolist() for n in mov5})


def render6(OT, env, s, park, zc, conds):
    """f6o's render check for the given conditions on one state (f6o2_tasks.render_check without d) -> violations."""
    S1 = np.asarray(s)[None]
    steps = WAIT + HIDE_AT + 3 * REPLAN
    bad, px = [], {}
    for cond in conds:
        q, extra, _ = OT.rollout_px(env, cond, [park], zc, S1, 0, steps)
        px[cond] = [[int(n), list(map(int, p))] for n, p, _ in q]
        for n, p, cv in q:
            hidden = cond == "b" and n != WAIT
            if hidden and max(p) > 0:
                bad.append((cond, n, "covered plate visible", p))
            if not hidden and p[0] <= 0:
                bad.append((cond, n, "plate not visible", p))
        if min(extra["cover1"]) < 0 or extra["cover1"][0] == 0:
            bad.append((cond, "decoy not visible in agentview", extra["cover1"]))
    return bad, px


CACHE7, CACHE6 = os.path.join(OUT, "cache_f7.pkl"), os.path.join(OUT, "cache_f6.pkl")
KINDS7 = ("SP_L", "SP_R", "SP_UL", "SP_UR", "NG_P", "NG_T", "NG_UP", "NG_UT")
FXKEY = dict(SP_L="F7_SP_relL", SP_R="F7_SP_relR", SP_UL="F7_SP_uniL", SP_UR="F7_SP_uniR", NG_P="F7_NG_pos", NG_T="F7_NG_neg",
             NG_UP="F7_NG_uniP", NG_UT="F7_NG_uniT")


def _pload(p):
    import pickle
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def _psave(p, obj):
    import pickle
    with open(p + ".tmp", "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(p + ".tmp", p)


def _setup():
    import fx_tasks as FT
    import f6o_tasks as OT
    import f6o2_tasks as O2
    for p in (META, os.path.join(ROOT, "results", "flawed", "rg_comp_5_read.json"), os.path.join(ROOT, "results", "flawed", "rg_comp_5_roll")):
        assert not os.path.exists(p), "exists: " + p
    os.makedirs(ST, exist_ok=True)
    fx = json.load(open(FT.META, encoding="utf-8"))
    f6 = json.load(open(OT.META, encoding="utf-8"))
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    mov4 = n2["movable"]
    c = n2["constraints"]
    return dict(FT=FT, OT=OT, O2=O2, fx=fx, f6=f6, TX={t["key"]: t for t in fx["tasks"]}, mov4=mov4,
                centre_p=np.array(n2["training"]["centre"])[mov4.index("plate_1")],
                lo=np.array(c["box"][0]) - FT.BOX_EXTRA, hi=np.array(c["box"][1]) + FT.BOX_EXTRA,
                fixp=np.array(c["fixture_xy"]), sf=c["stove_front_world"])


def build_f7(X):
    """F7 states, resumable: every candidate's outcome is cached (a resumed run continues on a fresh env; recorded)."""
    FT, TX = X["FT"], X["TX"]
    C = dict(FT=FT, lo=X["lo"], hi=X["hi"], fixp=X["fixp"], sf=X["sf"])
    cache = _pload(CACHE7) or dict(checks={}, acc=[], rej=[], next_k=F7_K0, resumed=0)
    env = FT.make_env(os.path.join(ROOT, TX["F7_SP_relL"]["bddl"]))
    env.seed(0)
    if not cache["checks"]:
        rep = {}
        for k in (0, 1):
            st, _ = f7_one(env, k, C)
            for kd in KINDS7:
                ref = np.load(os.path.join(FT.ST, "fx_%d.npy" % TX[FXKEY[kd]]["tid"]))[k]
                rep[f"{kd}_k{k}"] = float(np.abs(st[kd] - ref).max())
        print(f"[rgc] F7 reproduce fx k 0-1: max |diff| {max(rep.values()):.3g}", flush=True)
        assert max(rep.values()) <= 1e-6, rep
        cache["checks"]["reproduce_fx_F7_k0_1_max_abs_diff"] = rep
        _psave(CACHE7, cache)
    else:
        cache["resumed"] += 1
        print(f"[rgc] F7 resumed at k {cache['next_k']} with {len(cache['acc'])} accepted", flush=True)
    stored7 = {kd: np.load(os.path.join(FT.ST, "fx_%d.npy" % TX[FXKEY[kd]]["tid"])) for kd in KINDS7}
    acc7, rej7 = cache["acc"], cache["rej"]
    for k in range(cache["next_k"], F7_K0 + N_CAND7):
        if len(acc7) == N_SMOKE + N7_MAIN:
            break
        try:
            st, lay = f7_one(env, k, C)
            if any(np.allclose(st[kd], o) for kd in KINDS7 for o in stored7[kd]) or \
                    any(np.allclose(st["SP_L"], x["st"]["SP_L"]) for x in acc7):
                rej7.append([k, "duplicate"])
            else:
                acc7.append(dict(k=k, st=st, lay=lay))
                print(f"[rgc] F7 accepted k {k} ({len(acc7)}): sep {lay['sep']:.3f}", flush=True)
        except AssertionError as ex:
            rej7.append([k, str(ex)[:200]])
        cache["next_k"] = k + 1
        _psave(CACHE7, cache)
    env.close()
    assert len(acc7) == N_SMOKE + N7_MAIN, f"F7: only {len(acc7)} accepted; rejected {rej7}"
    cache["done"] = True
    _psave(CACHE7, cache)
    print("[rgc] F7 done", flush=True)


def build_f6(X):
    """F6 states (fam + disp), f6o's covers, parks and render check, resumable as build_f7."""
    FT, OT, O2, fx, f6, TX = X["FT"], X["OT"], X["O2"], X["fx"], X["f6"], X["TX"]
    mov4, centre_p, lo, hi, fixp, sf = X["mov4"], X["centre_p"], X["lo"], X["hi"], X["fixp"], X["sf"]
    bddl6 = os.path.join(ROOT, TX["F6_fam_vis"]["bddl"])
    fam0, disp0 = np.load(os.path.join(FT.ST, "fx_0.npy")), np.load(os.path.join(FT.ST, "fx_2.npy"))
    cache = _pload(CACHE6)
    ks6 = list(range(F6_K0, F6_K0 + N_CAND6))
    if cache is None:
        env = FT.make_env(bddl6)
        env.seed(0)
        assert fx["tasks"][0]["states"] == "F6_fam" and fx["tasks"][2]["states"] == "F6_disp"
        fr = [FT.sampled(env, FT.BASE6 + k) for k in range(FT.N)]            # fx's order: every fam state first, then disp
        dr = [O2.displace(env, fr[k], k, mov4, centre_p, lo, hi, fixp, sf)[0] for k in (0, 1)]
        rep6 = dict(fam=float(np.abs(np.stack(fr) - fam0).max()), disp=float(np.abs(np.stack(dr) - disp0[:2]).max()))
        print(f"[rgc] F6 reproduce fx k 0-1: {rep6}", flush=True)
        assert rep6["fam"] == 0.0 and rep6["disp"] <= 1e-6, rep6
        famN = {k: FT.sampled(env, FT.BASE6 + k) for k in ks6}
        dispN, plateN = {}, {}
        for k in ks6:
            dispN[k], plateN[k] = O2.displace(env, famN[k], k, mov4, centre_p, lo, hi, fixp, sf)
        env.close()
        cache = dict(checks=dict(reproduce_fx_F6_k0_1_max_abs_diff=rep6), famN=famN, dispN=dispN, plateN=plateN, acc=[], rej=[],
                     next_i=0, resumed=0)
        _psave(CACHE6, cache)
        print(f"[rgc] F6 drew {len(ks6)} candidates", flush=True)
    else:
        cache["resumed"] += 1
        print(f"[rgc] F6 resumed at candidate {cache['next_i']} with {len(cache['acc'])} accepted", flush=True)
    famN, dispN, plateN = cache["famN"], cache["dispN"], cache["plateN"]
    CV = f6["cover"]
    rc, zc, r_plate = CV["radius"], CV["zc"], f6["checks"]["plate_extent"]["r_xy_max"]
    OT.install(rc, CV["half_h"], CV["n"])
    env = FT.make_env(bddl6)
    env.seed(0)
    stored6 = [fam0, disp0, np.load(os.path.join(O2.ST, "f6o2_0.npy"))]
    acc6, rej6 = cache["acc"], cache["rej"]
    for i in range(cache["next_i"], len(ks6)):
        if len(acc6) == N_SMOKE + N6_MAIN:
            break
        k = ks6[i]
        cache["next_i"] = i + 1
        sd, sfm = dispN[k], famN[k]
        if sd is None:
            rej6.append([k, "no displaced plate"])
            continue
        if any(np.allclose(x, o) for x in (sd, sfm) for S_ in stored6 for o in S_) or \
                any(np.allclose(sd, y["disp"]) or np.allclose(sfm, y["fam"]) for y in acc6):
            rej6.append([k, "duplicate"])
            continue
        try:
            pd = OT.park_spots(env, sd, rc, r_plate, centre_p, mov4, lo, hi, fixp, sf, np.random.default_rng(OT.PARK_SEED + 100 + k))
            pf = OT.park_spots(env, sfm, rc, r_plate, centre_p, mov4, lo, hi, fixp, sf, np.random.default_rng(OT.PARK_SEED + k))
        except AssertionError as ex:
            rej6.append([k, "no park spots: " + str(ex)[:200]])
            continue
        bad_d, px_d = render6(OT, env, sd, pd, zc, ("a", "b"))
        bad_f, px_f = render6(OT, env, sfm, pf, zc, ("a",))
        if bad_d or bad_f:
            rej6.append([k, "render: " + str((bad_d + bad_f)[:3])[:300]])
        else:
            acc6.append(dict(k=k, disp=sd, fam=sfm, park_d=pd, park_f=pf, plate_xy=np.asarray(plateN[k]).tolist(),
                             shift=float(np.linalg.norm(np.asarray(plateN[k]) - centre_p)), px=dict(disp=px_d, fam=px_f)))
            print(f"[rgc] F6 accepted k {k} ({len(acc6)}): shift {acc6[-1]['shift']:.3f}", flush=True)
        _psave(CACHE6, cache)
    env.close()
    assert len(acc6) == N_SMOKE + N6_MAIN, f"F6: only {len(acc6)} accepted; rejected {rej6}"
    cache["done"] = True
    _psave(CACHE6, cache)
    print("[rgc] F6 done", flush=True)


def build(a):
    X = _setup()
    if a.step == "f7":
        return build_f7(X)
    if a.step == "f6":
        return build_f6(X)
    import imageio
    FT, OT, TX, f6 = X["FT"], X["OT"], X["TX"], X["f6"]
    c7, c6 = _pload(CACHE7), _pload(CACHE6)
    assert c7 and c7.get("done") and c6 and c6.get("done"), "run --step f7 and --step f6 first"
    acc7, rej7, acc6, rej6 = c7["acc"], c7["rej"], c6["acc"], c6["rej"]
    checks = dict(c7["checks"], **c6["checks"])
    checks["resumed"] = dict(f7=c7["resumed"], f6=c6["resumed"])
    CV = f6["cover"]
    # the flip pairs give the policy the same images (fx's check, smoke rows)
    env = FT.make_env(os.path.join(ROOT, TX["F7_SP_relL"]["bddl"]))
    env.seed(0)
    diffs, rows = {}, []
    for a_, b_ in (("SP_L", "SP_R"), ("NG_P", "NG_T")):
        d = []
        for x in acc7[:N_SMOKE]:
            ia, wa = FT.view(env, x["st"][a_])
            ib, wb = FT.view(env, x["st"][b_])
            d.append(float(max(np.abs(ia.astype(int) - ib.astype(int)).mean(), np.abs(wa.astype(int) - wb.astype(int)).mean())))
            if x is acc7[0]:
                rows.append(np.concatenate([ia, wa, ib, wb], axis=1))
        diffs[f"{a_}_vs_{b_}"] = d
    upx = []
    for u in ("SP_UL", "SP_UR", "NG_UP", "NG_UT"):
        ia, wa = FT.view(env, acc7[0]["st"][u], hide=B2)
        g2 = FT.subtree_geoms(env, B2)
        upx.append([FT.pixels(env, g2, "agentview"), FT.pixels(env, g2, "robot0_eye_in_hand")])
        rows.append(np.concatenate([ia, wa, ia * 0, wa * 0], axis=1))
    env.close()
    assert all(p == [0, 0] for p in upx), upx
    checks["F7_flip_image_mean_abs_diff_smoke"] = diffs
    checks["F7_uni_bowl2_pixels_hidden"] = upx
    imageio.imwrite(os.path.join(OUT, "check_F7.png"), np.concatenate(rows, axis=0))
    print(f"[rgc] F7 flip differences {diffs}", flush=True)

    checks["rejected_F7"], checks["rejected_F6"] = rej7, rej6

    # ---- write
    tasks, parks = [], {}
    for tid, (key, fam_, kind, fk, sk, hide, cond, arms) in enumerate(TASKS):
        t = TX[fk]
        lang = t["language"]
        goal = t["goal"]
        if key == "K1t":
            lang, goal = "put the bowl on the plate", ["On", B1, "plate_1"]
        if kind == "f7":
            S = np.stack([x["st"][sk] for x in acc7])
        else:
            S = np.stack([x[sk] for x in acc6])
            parks[str(tid)] = [x["park_d" if sk == "disp" else "park_f"] for x in acc6]
        np.save(os.path.join(ST, f"{SUITE}_{tid}.npy"), S)
        tasks.append(dict(tid=tid, key=key, family=fam_, kind=kind, fx_key=fk, language=lang, bddl=t["bddl"], goal=goal, object=B1,
                          target=goal[2], states=sk, hide=hide, hide_at=0 if hide else None, cover=cond, arms=list(arms), max_steps=300))
    br = lambda acc, rows: [dict(id=f"k{acc[r]['k']}", trial=r, seed_base=FT.SEED_BASE + 1000 * acc[r]["k"], at=0, variant="rgc")
                            for r in rows]                                    # noqa: E731
    rows7 = dict(smoke=list(range(N_SMOKE)), main=list(range(N_SMOKE, N_SMOKE + N7_MAIN)))
    rows6 = dict(smoke=list(range(N_SMOKE)), main=list(range(N_SMOKE, N_SMOKE + N6_MAIN)))
    for ph in ("smoke", "main"):
        for nm, acc, rows_ in (("br7", acc7, rows7), ("br6", acc6, rows6)):
            with open(os.path.join(OUT, f"{nm}_{ph}.json"), "w", encoding="utf-8") as fh:
                json.dump(br(acc, rows_[ph]), fh, indent=1)
    meta = dict(suite=SUITE, wait=WAIT, hide_at=HIDE_AT, replan=REPLAN, latch=LATCH, tasks=tasks, cover=CV, parks=parks,
                states=dict(f7=rows7, f6=rows6), sampler_k=dict(f7=[x["k"] for x in acc7], f6=[x["k"] for x in acc6]),
                f7_layouts=[x["lay"] for x in acc7], f6_plate=[dict(k=x["k"], plate_xy=x["plate_xy"], shift=x["shift"]) for x in acc6],
                f6_render_px={str(i): x["px"] for i, x in enumerate(acc6)}, f6o_meta=os.path.relpath(OT.META, ROOT),
                fx_meta=os.path.relpath(FT.META, ROOT), mapping=MAPPING, checks=checks,
                rule=("F7: fx_tasks.main's F7 loop, sampler 820000 + k, layout 821000 + k, k = 300+; F6: fx's F6 rule, sampler "
                      "810000 + k, displacement 811000 + k, k = 400+; parks 830000 + k (fam), 830000 + 100 + k (disp); "
                      "seed_base 14000000 + 1000 k"))
    with open(META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[rgc] wrote {META}: F7 k {meta['sampler_k']['f7'][0]}-{meta['sampler_k']['f7'][-1]} ({len(rej7)} rejected), "
          f"F6 k {meta['sampler_k']['f6'][0]}-{meta['sampler_k']['f6'][-1]} ({len(rej6)} rejected)", flush=True)


# ---------------------------------------------------------------- the episode modules (LIBERO venv)
class HideEnv:
    """fx_client's hide: from env step WAIT + hide_at on, the task's hide object sits in an unrendered geom group."""

    def __init__(self, env, hide, hide_at=0):
        self._e, self.hide, self.hide_at, self.n, self.hidden = env, hide, hide_at, 0, False

    def __getattr__(self, k):
        return getattr(self._e, k)

    def set_init_state(self, s):
        self.n, self.hidden = 0, False
        return self._e.set_init_state(s)

    def step(self, a):
        import fx_tasks as FT
        self.n += 1
        if not self.hidden and self.n >= WAIT + self.hide_at:
            FT.hide_geoms(self._e, self.hide)
            self.hidden = True
        return self._e.step(a)


def make_slot_class():
    import f6o2_tasks as O2

    class Slot(O2.Memory):
        """The memory slot: f6o2_tasks.Memory's record and paint, unchanged, with a latched trigger (module docstring, rc (3))."""

        def __init__(self, env, ckpt=""):
            super().__init__(env, ckpt)
            self.on = False
            self.rec.update(latched_at=None, latch_px=None)

        def step2(self, obs, n):
            if n == WAIT:
                self.record(obs)
                return obs
            if n < WAIT or self.m is None:
                return obs
            if not self.on:
                img = self.RV.flip(obs[O2.KEYS[0]])
                cls = self.seg.classes_of(img[None], [0], post_=False)[0]
                comps = self.XR.components(cls == self.cp, self.XR.MIN_PX)
                px = int(comps[0].sum()) if comps else 0
                ref = int(self.m["mask"].sum())
                if px >= LATCH * ref:
                    return obs
                self.on = True
                self.rec["latched_at"], self.rec["latch_px"] = int(n), [px, ref]
            out, npx = self.paint(obs)
            if (n - WAIT) % REPLAN == 0:
                self.rec["trace"].append([int(n), npx[0], npx[1]])
            return out
    return Slot


class Ctx:
    """Per process: what every episode needs (loaded once)."""

    def __init__(self, arm, task, meta):
        import rg as RG
        import fx_r as XR
        self.arm, self.task, self.meta, self.RG, self.XR = arm, task, meta, RG, XR
        self.parsed = XR.parse(task["language"]) if arm in ("rc", "f7r") else None
        self.front_on = self.parsed is not None
        self.slot_on = arm == "mem" or (arm == "rc" and rg_roles(task["language"])[1] == "plate_1")
        self.rg_on = arm in ("rgp", "rc")
        self.prompt = self.parsed["unique"] if self.parsed else task["language"]
        if self.parsed:
            self.oe, self.de = B1, NOUN_TARGET[self.parsed["target"]]
        else:
            self.oe, self.de = rg_roles(task["language"])
        self.Slot = make_slot_class() if self.slot_on else None
        self.P = None
        if self.rg_on:
            import sc_pu as PU
            import sc_rv as RV
            import sc_seg as SEG
            import fa1_common as FC
            from lifelong_ood.policy.parts_flow import ProprioHeld
            S = json.load(open(RG.STATS, encoding="utf-8"))
            S2 = dict(S)
            S2["geometry"] = dict(S["geometry"])
            S2["geometry"][B2] = S["geometry"][B1]
            S2["obj"] = dict(S["obj"])
            S2["obj"][B2] = S["obj"].get(B1) or S["obj"][S["inherit"][B1]]
            SEGM = SEG.load(os.environ.get("SC_SEG_CKPT") or None, "cuda")
            CACHE = {}

            def seg_std(env, E, cam):
                """rg.py's rgp seg_std (sc_hp's): the learned U-Net on an RGB render, in sc_pu.seg_std's convention."""
                sim = env.env.sim
                key = (id(sim), float(sim.data.time), cam)
                if key in CACHE:
                    return CACHE[key]
                img = RV.flip(sim.render(camera_name=cam, width=PU.SIZE, height=PU.SIZE, depth=False))
                p = SEGM.ent(img, cam, E["names"])
                out = np.ascontiguousarray(np.asarray(p)[:, ::-1])
                if len(CACHE) > 8:
                    CACHE.clear()
                CACHE[key] = out
                return out
            PU.seg_std = seg_std
            self.P = dict(PU=PU, RV=RV, FC=FC, X=json.load(open(RV.lex_path(""), encoding="utf-8")), C=PU.load_calib(), PH=ProprioHeld,
                          seg=SEGM, S=S2)
            assert self.de in S2["dst"], self.de


class RG2:
    """rg.ReGuide with extra non-instructed movables (the second bowl): only obj_referent's entity list is extended."""

    @staticmethod
    def make(RG, S, oe, de, extra):
        class _RG(RG.ReGuide):
            def obj_referent(self, s, qh):
                best = None
                cur = s["ee"]
                for e in tuple(RG.MOVABLE) + tuple(extra):
                    if e not in s["pos"]:
                        continue
                    g = self.G[e]
                    st = self.S["obj"].get(e) or self.S["obj"][self.S["inherit"][e]]
                    oh = RG.box_out(qh, s["pos"][e], s["yaw"][e], g)
                    oc = RG.box_out(cur, s["pos"][e], s["yaw"][e], g)
                    if int((oh > 0).sum()) > 1:
                        continue
                    ax = int(np.argmax(oh))
                    if oh[ax] < st["rho"] and (oh[ax] < oc[ax] or oc[ax] == 0.0):
                        dd = float(np.linalg.norm(qh - np.asarray(s["pos"][e])))
                        if best is None or dd < best[0]:
                            best = (dd, e)
                return None if best is None else best[1]
        return _RG(S, oe, de)


class Episode:
    """One episode's modules for one arm: front (F7R), slot (memory), ReGuide (+ its perception)."""

    def __init__(self, env, ctx):
        self.env, self.ctx, self.cmd = env, ctx, -1.0
        XR = ctx.XR
        self.front = XR.Front(env, "f7r", ctx.parsed) if ctx.front_on else None
        self.slot = ctx.Slot(env) if ctx.slot_on else None
        self.rg, self.reg, self.det, self.att = None, None, None, None
        self.keep, self.drop, self.inst = None, None, None
        self.dj = None
        if ctx.rg_on:
            self.rg = None                                                 # built at registration (it needs the extra entities)
        self.diag = dict(n_q=0, held_onset=None)

    # -- perception at the first query (env step WAIT)
    def instances(self, obs):
        """rgp's instance rule (module docstring): bowl-class components in the agentview at the first query; >= 2 -> keep the
        largest, the second is the other bowl.  rc with a resolved relation: keep = the resolver's referent, drop = the other."""
        XR, PU, RV = self.ctx.XR, self.ctx.P["PU"], self.ctx.P["RV"]
        seg = self.ctx.P["seg"]
        img = RV.flip(obs["agentview_image"])
        cls = seg.classes_of(img[None], [0], post_=False)[0]
        holder = types.SimpleNamespace(PU=PU, env=self.env, E=RV.entity_tables(self.env))
        P = XR.Front.cloud(holder, "agentview")
        comps = XR.components(cls == seg.classes.index(B1), XR.MIN_COMP)
        info = dict(n_comp=len(comps), px=[int(m.sum()) for m in comps[:3]])
        if self.front is not None and (self.front.res or {}).get("ok"):
            r = self.front.res
            self.keep, self.drop = np.asarray(r["centres"][r["used"]]), np.asarray(r["centres"][1 - r["used"]])
            info["rule"] = "resolver"
        elif len(comps) >= 2:
            self.keep, self.drop = XR.centre(comps[0], P), XR.centre(comps[1], P)
            info["rule"] = "largest"
        else:
            info["rule"] = "none (one instance)"
        inner = self.env.env
        sb = {n: np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]], np.float64) for n in (B1, B2) if n in inner.obj_body_id}
        if self.keep is not None:
            info["keep"], info["drop"] = self.keep.tolist(), self.drop.tolist()
            d = {n: float(np.linalg.norm(self.keep[:2] - p[:2])) for n, p in sb.items()}
            info["bound_body"] = min(d, key=d.get)
            info["bound_referent"] = bool(info["bound_body"] == B1)
            info["bound_xy_err"] = d[B1] if info["bound_referent"] else d.get(B2)
        self.inst = info

    def restricted(self, base, E, keep, drop):
        PU = self.ctx.P["PU"]
        ib = E["names"].index(B1)

        def f(env, E_, cam):
            s = np.array(base(env, E_, cam), copy=True)
            m = s == ib
            if m.any():
                d = PU.depth_std(env, E_, cam)
                fz, Rc, c = PU.cam_model(env.env.sim, cam)
                r, cc = np.nonzero(m)
                dd = d[r, cc]
                q = np.stack([(cc - PU.SIZE / 2.0) * dd / fz, (r - PU.SIZE / 2.0) * dd / fz, dd], 1) @ Rc.T + c
                far = np.linalg.norm(q - keep, axis=1) > np.linalg.norm(q - drop, axis=1)
                s[r[far], cc[far]] = -1
            return s
        return f

    def register(self, obs, ee, R):
        """C.6: registration once, at the first policy query (rg.py's register), restricted to one bowl instance when there are two."""
        P = self.ctx.P
        PU, RV, FC = P["PU"], P["RV"], P["FC"]
        E = RV.entity_tables(self.env)
        if self.dj is None:
            self.dj = FC.drawer_joint(self.env)
        sim = FC.scene(self.env, obs, self.dj)
        simpos = {n: sim["pos"][i].copy() for i, n in enumerate(FC.ENT)}

        def est_with(keep, drop):
            base = PU.seg_std
            if keep is not None:
                PU.seg_std = self.restricted(base, E, keep, drop)
            try:
                est = PU.Estimator(P["C"], self.ctx.oe)
                A, _ = PU.measure(self.env, E, PU.AGENT, PU.MOV)
                F = PU.measure(self.env, E, PU.AGENT, PU.fixture_names(E))[0]
                est.step(A, lambda: PU.measure(self.env, E, PU.WRIST, PU.MOV)[0], F, False, ee, R, -1.0)
            finally:
                PU.seg_std = base
            return est
        est = est_with(self.keep, self.drop)
        pos = {n: est.pos[i].copy() for i, n in enumerate(FC.ENT)}
        yaw = {n: float(est.yaw[i]) for i, n in enumerate(FC.ENT)}
        extra = ()
        if self.drop is not None:
            e2 = est_with(self.drop, self.keep)
            pos[B2], yaw[B2] = e2.pos[FC.ENT.index(B1)].copy(), float(e2.yaw[FC.ENT.index(B1)])
            extra = (B2,)
        err = {n: round(100 * float(np.linalg.norm(pos[n] - simpos[n])), 3) for n in FC.ENT}
        inner = self.env.env
        sb = {n: np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]], np.float64) for n in (B1, B2) if n in inner.obj_body_id}
        bound = min(sb, key=lambda n: float(np.linalg.norm(pos[B1][:2] - sb[n][:2])))
        diag = dict(src=dict(est.src), err_cm=err, bound_body=bound, bound_referent=bool(bound == B1),
                    bound_err_cm=round(100 * float(np.linalg.norm(pos[B1] - sb[bound])), 3), oe=self.ctx.oe, de=self.ctx.de,
                    de_err_cm=round(100 * float(np.linalg.norm(pos[self.ctx.de] - simpos[self.ctx.de])), 3),
                    plate_xy_err_cm=round(100 * float(np.linalg.norm(pos["plate_1"][:2] - simpos["plate_1"][:2])), 3))
        if extra:
            o = B2 if bound == B1 else B1
            diag["other_err_cm"] = round(100 * float(np.linalg.norm(pos[B2] - sb[o])), 3)
        self.reg = dict(pos=pos, yaw=yaw, last=None, diag=diag)
        self.rg = RG2.make(self.ctx.RG, P["S"], self.ctx.oe, self.ctx.de, extra)

    def state(self, obs):
        """rg.py's rgp state(): registered poses, ProprioHeld, the held object rigid in the hand from the onset."""
        P = self.ctx.P
        FC = P["FC"]
        ee = np.asarray(obs["robot0_eef_pos"], np.float64)
        R = FC._quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64))
        grip = np.asarray(obs["robot0_gripper_qpos"], np.float64)
        if self.reg is None:
            self.register(obs, ee, R)
        pos, yaw = {k: v.copy() for k, v in self.reg["pos"].items()}, dict(self.reg["yaw"])
        if self.det is None:
            self.det = P["PH"](P["X"]["tw"], P["X"]["lift"])
        held = bool(self.det.step(grip, self.cmd, ee[2]))
        oe = self.ctx.oe
        if held and self.att is None:
            o0 = pos[oe] if self.reg.get("last") is None else self.reg["last"]
            self.att = R.T @ (o0 - ee)
        if not held and self.att is not None:
            self.reg["last"] = ee + R @ self.att
            self.att = None
        q = ee + R @ self.att if self.att is not None else (pos[oe] if self.reg.get("last") is None else self.reg["last"])
        pos[oe] = q.copy()
        return dict(ee=ee, R=R, width=float(grip[0] - grip[1]), pos=pos, yaw=yaw, held=held, q=q, cmd=self.cmd)

    def see(self, obs, n):
        """Every observation the env returns (n = env steps so far): the slot, then the front; at n == WAIT the instance rule."""
        o = obs
        if self.slot is not None:
            o = self.slot.step2(o, n)
        if self.front is not None:
            o = self.front.step(o, n, WAIT)
        if n == WAIT and self.ctx.rg_on:
            self.instances(obs)
        return o

    def record(self):
        r = dict(diag=self.diag, inst=self.inst)
        if self.front is not None:
            fr = self.front.record()
            fr.pop("trace", None)
            r["front"] = fr
        if self.slot is not None:
            r["slot"] = dict(self.slot.rec, trace=self.slot.rec["trace"][:60])
        if self.reg is not None:
            r["reg"] = self.reg["diag"]
            inner = self.env.env
            pe = np.array(inner.sim.data.body_xpos[inner.obj_body_id["plate_1"]], np.float64)
            r["reg"]["plate_xy_err_end_cm"] = round(100 * float(np.linalg.norm(self.reg["pos"]["plate_1"][:2] - pe[:2])), 3)
        if self.rg is not None:
            r["rg"] = self.rg.record()
        return r


CUR = dict(env=None, plan=None, chunk=None, k=0)


class RCEnv:
    """The client's env (inner = raw / HideEnv / f6o's CoverEnv) with one arm's episode modules and the track."""

    def __init__(self, env, ctx, sink=None):
        self._e, self.ctx, self.sink = env, ctx, sink
        self.n, self.ep, self.obs, self.tr = 0, None, None, None
        CUR["env"] = self

    def __getattr__(self, k):
        return getattr(self._e, k)

    def _names(self):
        return list(self._e.env.objects_dict.keys())

    def _snap(self):
        inner = self._e.env
        return np.array([np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]]) for n in self._names()], np.float32)

    def set_init_state(self, s):
        obs = self._e.set_init_state(s)
        self.n, self.obs = 0, obs
        self.ep = Episode(self._e, self.ctx) if self.ctx.arm != "I" else None
        self.tr = dict(start=self._snap(), pos=[], ee=[], a6=[], hidden=[], t0=time.time())
        return obs

    def hidden_now(self):
        if hasattr(self._e, "covered") and not isinstance(self._e, HideEnv):
            try:
                return bool(self._e.covered)
            except Exception:                                              # noqa: BLE001
                return False
        return bool(getattr(self._e, "hidden", False)) if isinstance(self._e, HideEnv) else False

    def step(self, a):
        ep, hb = self.ep, False
        act = list(a)
        if ep is not None and self.ctx.rg_on and self.n >= WAIT:
            ch = CUR.get("chunk")
            rem = None
            if ch is not None:
                rem = ch[CUR["k"]:]
                CUR["k"] += 1
            s = ep.state(self.obs)
            s["rem"] = rem
            act, hb = ep.rg.decide(s, list(a), rem)
        r = self._e.step(act)
        self.n += 1
        obs = r[0]
        self.obs = obs
        if ep is not None:
            ep.cmd = float(act[6])
        if self.tr is not None:
            self.tr["pos"].append(self._snap())
            self.tr["ee"].append(np.asarray(obs["robot0_eef_pos"], np.float32))
            self.tr["a6"].append(float(act[6]))
            self.tr["hidden"].append(self.hidden_now())
        if ep is not None:
            obs = ep.see(obs, self.n)
        if hb:
            pl = CUR.get("plan")
            if pl is not None:
                pl.clear()
            CUR["chunk"] = None
        return (obs,) + tuple(r[1:])

    def close(self):
        if self.sink is not None and self.tr is not None and self.n > 0:
            self.sink(self)
        self.tr, self.ep = None, None
        self._e.close()


def wrap_inner(env, task, meta, states):
    if task["kind"] == "f6":
        import f6o_tasks as OT
        return OT.CoverEnv(env, task["cover"], meta["parks"][str(task["tid"])], meta["cover"]["zc"], states, track=False)
    if task["hide"]:
        return HideEnv(env, task["hide"], int(task["hide_at"] or 0))
    return env


# ---------------------------------------------------------------- client (LIBERO venv)
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
    import libero_client as LC
    from libero.libero import benchmark
    meta = json.load(open(META, encoding="utf-8"))
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = meta["tasks"][TID]
    assert TASK["tid"] == TID and arm in TASK["arms"], (TASK, arm)
    assert sys.argv[sys.argv.index("--suite") + 1] == SUITE and "," not in sys.argv[sys.argv.index("--tasks") + 1]
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "rgc episodes need --fresh and --branches"
    assert int(sys.argv[sys.argv.index("--wait") + 1]) == WAIT if "--wait" in sys.argv else True
    if TASK["kind"] == "f6":
        import f6o_tasks as OT
        CV = meta["cover"]
        OT.install(CV["radius"], CV["half_h"], CV["n"])
    ctx = Ctx(arm, TASK, meta)
    Task = collections.namedtuple("Task", "name language problem_folder bddl_file")

    class RgcSuite:
        def __init__(self):
            self.tasks = [Task(t["key"], ctx.prompt if t["tid"] == TID else t["language"], os.path.dirname(os.path.join(ROOT, t["bddl"])),
                               os.path.basename(t["bddl"])) for t in meta["tasks"]]
            self.n_tasks = len(self.tasks)

        def get_task(self, i):
            return self.tasks[i]

        def get_task_init_states(self, i):
            raise RuntimeError("rgc runs only from --fresh start states")
    _orig = benchmark.get_benchmark_dict

    def _bdict(*a, **k):
        d = dict(_orig(*a, **k))
        d[SUITE] = RgcSuite
        return d
    benchmark.get_benchmark_dict = _bdict
    LC.MAX_STEPS[SUITE] = int(TASK["max_steps"])
    STATES = np.load(os.path.join(ST, f"{SUITE}_{TID}.npy"))
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    RECS, SIDE = [], []
    SIDEF, TRF = OUTF.replace(".json", "_rc.json"), OUTF.replace(".json", "_track.npz")

    def sink(env):
        R = env.tr
        RECS.append(dict(start=R["start"], pos=np.stack(R["pos"]), ee=np.stack(R["ee"]), a6=np.array(R["a6"], np.float32),
                         hidden=np.array(R["hidden"], bool), success=np.array(bool(env._e.env._check_success()))))
        z = dict(n=np.array(len(RECS)), names=np.array(env._names()), wait=np.array(WAIT))
        for i, r in enumerate(RECS):
            z.update({f"e{i}_{k}": v for k, v in r.items()})
        np.savez_compressed(TRF, **z)
        i = len(SIDE)
        rec = env.ep.record() if env.ep is not None else {}
        rec.update(trial=int(br[i]["trial"]) if i < len(br) else None, steps=env.n, secs=time.time() - R["t0"])
        SIDE.append(rec)
        tmp = SIDEF + ".tmp%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], prompt=ctx.prompt, roles=[ctx.oe, ctx.de],
                           parsed=ctx.parsed, eps=SIDE), fh, indent=1)
        os.replace(tmp, SIDEF)
        f = rec.get("front", {}).get("resolve") or {}
        print("[rgc] %s %s trial %s: steps %d resolve %s inst %s reg %s slot latched %s rg %s" % (
            arm, TASK["key"], rec["trial"], env.n, f.get("resolver_correct"), (rec.get("inst") or {}).get("bound_referent"),
            {k: (rec.get("reg") or {}).get(k) for k in ("bound_referent", "bound_err_cm", "de_err_cm")},
            (rec.get("slot") or {}).get("latched_at"),
            None if "rg" not in rec else (rec["rg"]["transports"], [(e["t"], e["kind"], e.get("why")) for e in rec["rg"]["events"]][:5])),
            flush=True)

    ORIG = LC.OffScreenRenderEnv
    if ctx.rg_on:
        class PlanDeque(collections.deque):                                 # libero_client's action queue, with a handle (rg.py)
            def __init__(self, *a):
                super().__init__(*a)
                CUR["plan"] = self
        LC.collections = types.SimpleNamespace(deque=PlanDeque)
        Base = LC._wcp.WebsocketClientPolicy

        class RGClient(Base):
            def infer(self, el):
                env = CUR.get("env")
                ep = None if env is None else env.ep
                if ep is not None and ep.rg is not None and ep.rg.mode is not None:
                    CUR["chunk"] = None                                     # ReGuide steers: no query
                    return {"actions": np.zeros((REPLAN, 7))}
                out = super().infer(el)
                CUR["chunk"], CUR["k"] = np.asarray(out["actions"], np.float64), 0
                if ep is not None:
                    ep.diag["n_q"] += 1
                return out
        LC._wcp.WebsocketClientPolicy = RGClient
    LC.OffScreenRenderEnv = lambda *a, **k: RCEnv(wrap_inner(ORIG(*a, **k), TASK, meta, STATES), ctx, sink)
    if "--max-steps" not in sys.argv:
        sys.argv += ["--max-steps", str(TASK["max_steps"])]
    print("[rgc] arm %s task %s %r -> prompt %r roles %s -> %s front %s slot %s rg %s" % (
        arm, TASK["key"], TASK["language"], ctx.prompt, ctx.oe, ctx.de, ctx.front_on, ctx.slot_on, ctx.rg_on), flush=True)
    LC.main()


# ---------------------------------------------------------------- policy-free perception check
def pcheck(a):
    import fx_tasks as FT
    import f6o_tasks as OT
    meta = json.load(open(META, encoding="utf-8"))
    out = a.out or PCHECK
    assert not os.path.exists(out), "exists: " + out
    res = {}
    order = [0, 1, 2, 3, 7, 8, 10]                                         # F7 first: the covers are installed for F6 only
    for tid in order:
        task = meta["tasks"][tid]
        if task["kind"] == "f6":
            CV = meta["cover"]
            OT.install(CV["radius"], CV["half_h"], CV["n"])
        S = np.load(os.path.join(ST, f"{SUITE}_{tid}.npy"))
        env = FT.make_env(os.path.join(ROOT, task["bddl"]))
        steps = WAIT + HIDE_AT + 3 * REPLAN if task["kind"] == "f6" else WAIT + 2
        rows = []
        for arm in [x for x in ("rgp", "rc", "mem") if x in task["arms"]]:
            ctx = Ctx(arm, task, meta)
            for k in range(S.shape[0]):
                np.random.seed(0)
                env.seed(0)
                rce = RCEnv(wrap_inner(env, task, meta, S), ctx)
                rce.reset()
                rce.set_init_state(S[k])
                for _ in range(steps):
                    rce.step(FT.DUMMY)
                rec = rce.ep.record()
                rec.update(row=k, arm=arm)
                rows.append(rec)
                rce.ep = None
        env.close()
        res[task["key"]] = rows
        for arm in ("rgp", "rc", "mem"):
            R = [r for r in rows if r["arm"] == arm]
            if not R:
                continue
            s = dict(n=len(R))
            fr = [r["front"]["resolve"] for r in R if r.get("front")]
            if fr:
                s["resolver_correct"] = float(np.mean([bool(x.get("resolver_correct")) for x in fr]))
            rg = [r["reg"] for r in R if r.get("reg")]
            if rg:
                s["bound_referent"] = float(np.mean([x["bound_referent"] for x in rg]))
                s["bound_err_cm_median"] = float(np.median([x["bound_err_cm"] for x in rg]))
                s["de_err_cm_median"] = float(np.median([x["de_err_cm"] for x in rg]))
                s["plate_xy_err_cm_median"] = float(np.median([x["plate_xy_err_cm"] for x in rg]))
                s["other_err_cm_median"] = float(np.median([x["other_err_cm"] for x in rg if "other_err_cm" in x])) if any("other_err_cm" in x for x in rg) else None
            sl = [r["slot"] for r in R if r.get("slot")]
            if sl:
                s["slot_record_ok"] = float(np.mean([bool((x["record"] or {}).get("ok")) for x in sl]))
                s["slot_iou_median"] = float(np.median([x["record"]["iou"] for x in sl if (x["record"] or {}).get("ok")]))
                s["slot_centre_xy_err_median"] = float(np.median([x["record"]["centre_xy_err"] for x in sl if (x["record"] or {}).get("ok")]))
                s["slot_latched"] = float(np.mean([x["latched_at"] is not None for x in sl]))
                lat = [x["latched_at"] for x in sl if x["latched_at"] is not None]
                s["slot_latch_step_median"] = float(np.median(lat)) if lat else None
            print(f"[rgc pcheck] {task['key']} {arm}: {s}", flush=True)
            res.setdefault("_summary", {}).setdefault(task["key"], {})[arm] = s
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"[rgc pcheck] wrote {out}", flush=True)


# ---------------------------------------------------------------- read (CPU)
def load_run(roll, model, arm, tid, rows):
    fj = os.path.join(roll, f"{model}_{arm}_{tid}.json")
    if not os.path.exists(fj):
        return None
    eps = json.load(open(fj, encoding="utf-8"))["tasks"][str(tid)]["episodes"]
    fz = fj.replace(".json", "_track.npz")
    z = np.load(fz) if os.path.exists(fz) else None
    names = [str(x) for x in z["names"]] if z is not None else []
    fs = fj.replace(".json", "_rc.json")
    side = json.load(open(fs, encoding="utf-8"))["eps"] if os.path.exists(fs) else []
    out = {}
    for i, e in enumerate(eps):
        r = dict(success=bool(e["success"]), error=e.get("error"), steps=e.get("steps"), grasped=None, side=side[i] if i < len(side) else None)
        if z is not None and i < int(z["n"]):
            start, P = z[f"e{i}_start"], z[f"e{i}_pos"]
            lift = P[:, :, 2] - start[None, :, 2]
            first = [int(np.argmax(lift[:, j] >= LIFT)) if (lift[:, j] >= LIFT).any() else None for j in range(len(names))]
            cand = [(f, names[j]) for j, f in enumerate(first) if f is not None]
            r["grasped"] = min(cand)[1] if cand else "none"
        out[int(e["trial"])] = r
    return {k: out[k] for k in rows if k in out}


def bootm(fn, ns, seed=0):
    """-> [point, lo, hi]: fn(list of index arrays, one per state set), each state set resampled in the same draw."""
    rng = np.random.default_rng(seed)
    pt = fn([np.arange(n) for n in ns])
    v = []
    for _ in range(B):
        v.append(fn([rng.integers(0, n, n) for n in ns]))
    v = np.array(v, float)
    v = v[np.isfinite(v)]
    if not len(v) or not np.isfinite(pt):
        return [float(pt), None, None]
    return [float(pt), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def lab_win(ci):
    if ci[1] is None:
        return "MISSING"
    return "RC WINS" if ci[1] > 0 else ("RG WINS" if ci[2] < 0 else "NO CLEAR DIFFERENCE")


def read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    meta = json.load(open(META, encoding="utf-8"))
    T = meta["tasks"]
    rows7, rows6 = meta["states"]["f7"][a.phase], meta["states"]["f6"][a.phase]
    rows_of = lambda tid: rows7 if T[tid]["kind"] == "f7" else rows6        # noqa: E731
    n7, n6 = len(rows7), len(rows6)
    res = dict(phase=a.phase, roll=a.roll, mapping=MAPPING, B=B, rows=dict(f7=rows7, f6=rows6), models={})
    for m in a.models.split("+"):
        D, X = {}, {}
        for t in T:
            for arm in t["arms"]:
                D[(arm, t["tid"])] = load_run(a.roll, m, arm, t["tid"], rows_of(t["tid"]))
                d = D[(arm, t["tid"])]
                X[(arm, t["tid"])] = np.array([float(d[k]["success"]) if (d and k in d) else np.nan for k in rows_of(t["tid"])], float)
        missing = [f"{arm}/{T[tid]['key']}" for (arm, tid), d in D.items() if d is None or len(d) < len(rows_of(tid))]
        M = dict(missing=missing, success={}, errors={})
        mean = lambda x, i: float(np.nanmean(x[i]))                         # noqa: E731
        for (arm, tid), x in X.items():
            ns = [n7] if T[tid]["kind"] == "f7" else [n6]
            M["success"][f"{arm}/{T[tid]['key']}"] = bootm(lambda I, x=x: mean(x, I[0]), ns)
            d = D[(arm, tid)] or {}
            M["errors"][f"{arm}/{T[tid]['key']}"] = sum(1 for r in d.values() if r.get("error"))

        def fam(arm, F):
            a_, b_ = REL[F]
            return lambda I: (mean(X[(arm, a_)], I[0]) + mean(X[(arm, b_)], I[0])) / 2

        def kpool(arm):
            return lambda I: (mean(X[(arm, 10)], I[0]) + mean(X[(arm, 7)], I[1]))

        def kp(arm):
            return lambda I: kpool(arm)(I) / 2
        NK = [n6, n7]
        M["K"] = {arm: bootm(kp(arm), NK) for arm in ("I", "rgp", "rc")}
        floor = M["K"]["I"][0] >= FLOOR
        M["floor"] = dict(I_K=M["K"]["I"][0], label="TESTABLE" if floor else "NOT TESTABLE")
        gates = {}
        for F in ("SP", "NG"):
            u = (mean(X[("I", UNI[F][0])], np.arange(n7)) + mean(X[("I", UNI[F][1])], np.arange(n7))) / 2
            gates[F] = dict(U=u, label="TESTABLE" if (u >= UGATE and floor) else "NOT TESTABLE")
        a6 = mean(X[("I", 9)], np.arange(n6))
        gates["F6_b"] = dict(a=a6, label="TESTABLE" if (a6 >= AGATE and floor) else "NOT TESTABLE")
        M["gates"] = gates
        M["ceiling"] = {T[r]["key"]: dict(uni=T[u]["key"], I_uni=mean(X[("I", u)], np.arange(n7)),
                                          flag="CEILING" if mean(X[("I", u)], np.arange(n7)) < UGATE else "")
                        for r, u in REL_UNI.items()}
        # family / cell success per arm
        M["cell"] = {}
        for F in ("SP", "NG"):
            M["cell"][F] = {arm: bootm(fam(arm, F), [n7]) for arm in ("I", "rgp", "rc", "f7r")}
        M["cell"]["F6_b"] = {arm: bootm(lambda I, arm=arm: mean(X[(arm, 8)], I[0]), [n6]) for arm in ("I", "rgp", "rc", "mem")}
        M["cell"]["F6_b"]["a (I, unhidden)"] = bootm(lambda I: mean(X[("I", 9)], I[0]), [n6])
        cellfn = {F: (lambda arm, F=F: fam(arm, F)) for F in ("SP", "NG")}
        cellfn["F6_b"] = lambda arm: (lambda I, arm=arm: mean(X[(arm, 8)], I[0]))
        cellns = {"SP": [n7], "NG": [n7], "F6_b": [n6]}
        V = {}
        for cell in ("SP", "NG", "F6_b"):
            fr, fg = cellfn[cell]("rc"), cellfn[cell]("rgp")
            d = bootm(lambda I, fr=fr, fg=fg: fr(I) - fg(I), cellns[cell])
            lab = lab_win(d)
            V[("V1 " if cell != "F6_b" else "V2 ") + cell] = dict(ci=d, outcome=lab, label=lab if gates[cell]["label"] == "TESTABLE" else "NOT TESTABLE")
        dK = bootm(lambda I: kp("rc")(I) - kp("I")(I), NK)
        ni = "NON-INFERIOR" if (dK[1] is not None and dK[1] > NI_MARGIN) else "NOT SHOWN"
        for cell in ("SP", "NG", "F6_b"):
            fr, fi = cellfn[cell]("rc"), cellfn[cell]("I")
            dF = bootm(lambda I, fr=fr, fi=fi: fr(I) - fi(I), cellns[cell])
            h1 = dF[1] is not None and dF[1] > 0
            h2 = dK[1] is not None and dK[1] > NI_MARGIN
            lab = "FIX WORKS" if (h1 and h2) else ("PARTIAL" if (h1 or h2) else "FAILS")
            V["V3 " + cell] = dict(dF=dF, dK=dK, dK_label=ni, outcome=lab,
                                   label=lab if gates[cell]["label"] == "TESTABLE" else "NOT TESTABLE")
        M["V"] = V
        # descriptive differences
        Dd = {}
        for cell in ("SP", "NG", "F6_b"):
            others = (("rgp", "I"), ("f7r", "I"), ("rc", "f7r")) if cell != "F6_b" else (("rgp", "I"), ("mem", "I"), ("rc", "mem"))
            for x, y in others:
                fx_, fy_ = cellfn[cell](x), cellfn[cell](y)
                Dd[f"{cell}: {x} - {y}"] = bootm(lambda I, fx_=fx_, fy_=fy_: fx_(I) - fy_(I), cellns[cell])
        for arm in ("rgp", "rc"):
            Dd[f"K: {arm} - I"] = bootm(lambda I, arm=arm: kp(arm)(I) - kp("I")(I), NK)
            for tid in (7, 10):
                ns = [n7] if tid == 7 else [n6]
                Dd[f"{T[tid]['key']}: {arm} - I"] = bootm(lambda I, tid=tid, arm=arm: mean(X[(arm, tid)], I[0]) - mean(X[("I", tid)], I[0]), ns)
        M["diff"] = Dd
        # C: referent grasps among bowl grasps (fx's C), per family and arm
        Cc = {}
        for F in ("SP", "NG"):
            for arm in ("I", "rgp", "rc", "f7r"):
                hit = np.array([[((D[(arm, t)] or {}).get(k) or {}).get("grasped") == B1 for t in REL[F]] for k in rows7], float)
                bowl = np.array([[((D[(arm, t)] or {}).get(k) or {}).get("grasped") in (B1, B2) for t in REL[F]] for k in rows7], float)
                Cc[f"{F} {arm}"] = bootm(lambda I, hit=hit, bowl=bowl: float(hit[I[0]].sum() / bowl[I[0]].sum()) if bowl[I[0]].sum() else np.nan, [n7])
        M["C"] = Cc
        # mechanism (descriptive)
        mech = {}
        for (arm, tid), d in D.items():
            if not d or arm == "I":
                continue
            sd = [r["side"] for r in d.values() if r.get("side")]
            key = f"{arm}/{T[tid]['key']}"
            mm = dict(n=len(sd))
            fr = [s["front"]["resolve"] for s in sd if s.get("front") and s["front"].get("resolve")]
            if fr:
                mm["resolver_correct"] = f"{sum(bool(x.get('resolver_correct')) for x in fr)}/{len(fr)}"
            rg = [s["reg"] for s in sd if s.get("reg")]
            if rg:
                mm["bound_referent"] = f"{sum(bool(x['bound_referent']) for x in rg)}/{len(rg)}"
                mm["bound_err_cm_median"] = float(np.median([x["bound_err_cm"] for x in rg]))
                mm["de_err_cm_median"] = float(np.median([x["de_err_cm"] for x in rg]))
                mm["plate_xy_err_cm_median"] = float(np.median([x["plate_xy_err_cm"] for x in rg]))
                mm["plate_xy_err_end_cm_median"] = float(np.median([x["plate_xy_err_end_cm"] for x in rg]))
            rr = [s["rg"] for s in sd if s.get("rg")]
            if rr:
                mm["transports_mean"] = float(np.mean([x["transports"] for x in rr]))
                mm["any_transport"] = float(np.mean([x["transports"] > 0 for x in rr]))
                mm["steps_rg_mean"] = float(np.mean([x["steps_rg"] for x in rr]))
                why = collections.Counter(str(e.get("why")).split(" ->")[0] for x in rr for e in x["events"] if e["kind"] == "start")
                mm["starts"] = dict(why.most_common(8))
            sl = [s["slot"] for s in sd if s.get("slot")]
            if sl:
                okr = [x["record"] for x in sl if (x["record"] or {}).get("ok")]
                mm["slot_record_ok"] = f"{len(okr)}/{len(sl)}"
                if okr:
                    mm["slot_iou_median"] = float(np.median([x["iou"] for x in okr]))
                    mm["slot_centre_xy_err_cm_median"] = 100 * float(np.median([x["centre_xy_err"] for x in okr]))
                    mm["slot_centre_xy_err_cm_max"] = 100 * float(np.max([x["centre_xy_err"] for x in okr]))
                lat = [x["latched_at"] for x in sl if x["latched_at"] is not None]
                mm["slot_latched"] = f"{len(lat)}/{len(sl)}"
                mm["slot_latch_step_median"] = float(np.median(lat)) if lat else None
                tr = [t_ for x in sl for t_ in x["trace"]]
                if tr:
                    mm["slot_painted_px_median"] = [float(np.median([t_[1] for t_ in tr])), float(np.median([t_[2] for t_ in tr]))]
            ins = [s["inst"] for s in sd if s.get("inst")]
            if ins:
                mm["inst_rule"] = dict(collections.Counter(x.get("rule") for x in ins))
            mech[key] = mm
        M["mechanism"] = mech
        res["models"][m] = M
        # print
        print(f"==== {m}: floor {M['floor']}; gates {json.dumps(gates)}; missing {missing}", flush=True)
        for k_, v in M["success"].items():
            print(f"  success {k_:20s} {fmt(v)}  errors {M['errors'][k_]}", flush=True)
        for c_, v in M["cell"].items():
            print(f"  cell {c_}: " + "; ".join(f"{k_} {fmt(x)}" for k_, x in v.items()), flush=True)
        print(f"  K: " + "; ".join(f"{k_} {fmt(x)}" for k_, x in M["K"].items()), flush=True)
        for k_, v in V.items():
            print(f"  {k_}: {json.dumps(v)}", flush=True)
        for k_, v in Dd.items():
            print(f"  diff {k_}: {fmt(v)}", flush=True)
        for k_, v in Cc.items():
            print(f"  C {k_}: {fmt(v)}", flush=True)
        print(f"  ceiling {json.dumps(M['ceiling'])}", flush=True)
        for k_, v in mech.items():
            print(f"  mech {k_}: {json.dumps(v)}", flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"[rgc read] wrote {out}", flush=True)


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, list):
        return "%s [%s, %s]" % tuple("-" if x is None else "%.3f" % x for x in v)
    return "%.3f" % v if isinstance(v, float) else str(v)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        return client_main()
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "pcheck", "read"))
    ap.add_argument("--phase", default="main", choices=("smoke", "main"))
    ap.add_argument("--step", default="assemble", choices=("f7", "f6", "assemble"))
    ap.add_argument("--roll", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--models", default="pi05+pi0")
    a = ap.parse_args()
    {"build": build, "pcheck": pcheck, "read": read}[a.mode](a)


if __name__ == "__main__":
    main()

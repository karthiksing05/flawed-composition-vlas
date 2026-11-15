"""S2FIX3 (docs/SCAFFOLD_RESULTS.md 63; m3_prereg tag s2_fix3): SR 62's named-mug hand-over (arm H) on NEW states, layout O, tasks K1,
K2, N1, N2 of scene 2 (LIBERO-10 LIVING_ROOM_SCENE5).  Three gaps of SR 62:
  (a) privileged inputs  -> arm Hp on frozen pi0.5: H with perception in place of simulator state;
  (b) one model          -> arm H (SR 62's client, unchanged) and I on frozen pi0 and frozen GR00T (SR 50 / SR 56 servers);
  (c) N1 release          -> step-0 diagnosis (mode `diag`, SR 62's outputs only).  The optional arm Hplus is NOT run: the diagnosis
                             did not isolate a cause (see SR 63; carry limit never hit; the named mug's offset from the plate at
                             release is the same in N1 successes, .034 m, and failures, .037 m; the reference planner's relocation
                             rule would move it .037 -> .034 m).

ARM H (SR 62, unchanged; `s2fix2.py client --arm H --layout O`): ProprioHeld (sc_rv lex threshold) on finger qpos, executed gripper
command and hand height; at a detector onset the held mug is the mug nearest the hand in xy within HELD_R (.08 m), from SIMULATOR
positions; if it is the named mug, SR 54's object-blind place policy P takes over (target plate's SIMULATOR position every step,
constants of the trained half ending on that plate), else the VLA keeps control.  (s2fix2's client also loads SR 62's router table
keyed by state index; arm H never uses it, so the `route` field in its side file is SR 62's and is ignored here.)

ARM Hp (this file, `client --arm Hp`): H with every simulator read replaced.
  - Hold events: ProprioHeld, unchanged (proprioception only).
  - Detector: OWLv2 (where.OwlDetector's checkpoint google/owlv2-base-patch16-ensemble, the repo's off-the-shelf open-vocabulary
    detector), run in a sidecar (s2fix3_det.py, openpi venv) on the agentview RGB image the policy receives (256 px, before the
    224 resize), queries "a white mug", "a yellow and white mug", "a red mug", "a plate".  Mug boxes (ASSIGN = perm): the three best
    boxes by max mug-query score (>= MIN_SCORE, IoU < .3), then the query-to-box permutation maximising the sum of each query's
    score relative to its best of the three.  (pcheck's greedy assignment swapped the white and yellow-and-white mug; pcheck2
    chose perm on development states.)  Plates: the two best "a plate" boxes (IoU < .3).
  - 3-D: rg.py rgp's remaining privilege, unchanged in kind: the rendered agentview depth (sc_pu.depth_std) back-projected through the
    camera calibration (sc_pu.cam_model) -> the median world point of each box's inner half -> + a per-type constant (CAL, fitted by
    mode `pcheck` on DEVELOPMENT states only: SR 62's states k 0-63, never this probe's states).
  - Target plate: registered ONCE at the first gated step (n = WAIT), as rgp (paper C.6), no tracking afterwards.  "left" / "right"
    from the instruction picks the detected plate on that side by SIDE_RULE (fixed by `pcheck` on development states).
  - Held mug at an onset: OWLv2 re-run on that step's image; mug positions = this detection, falling back per mug to its last
    detected position (registration at n = WAIT, then every onset); the held mug = the nearest within HELD_R of the proprioceptive
    hand position.  Same gate rule as H.  P = SR 54's P with the registered target position in place of the simulator's.
  - The simulator oracle (positions, H's held-mug rule and decision on the same step) is LOGGED in the side file, never fed.
END Hp.

New states: LIBERO's sampler on K1's BDDL, seeds 920000 + k (none equal to an SR 50-62 state), k 0-3 smoke, 4-63 main; branches
seed_base 38000000 + 1000 k.  Servers / clients: SR 50's (pi0.5, GR00T: pp_serve_purify --mode intact --comp= --seed 13; pi0:
serve_guided --mode ours --seed 13 with BACKBONE_CONFIG=pi0_libero), client seed 7, --bind track.

Modes (LIBERO venv, py3.8):
    states                                (EGL)   -> results/flawed/s2fix3/O/s2_living_<tid>.npy, branches_{smoke,main}.json
    diag [--out X]                        (CPU)   step 0: N1 release, SR 62's existing outputs only (results/flawed/s2fix2_roll/O)
    pcheck [--out X]                      (EGL + sidecar on $S2FIX3_DET_PORT)  perception on development states (SR 62's k 0-63)
    client --arm Hp <s2.py client args>   (needs the sidecar on $S2FIX3_DET_PORT)
    ref --tid T --branches B --out-dir D  (CPU, C1)
    summary DIR                           (smoke)
    read [DIR --selftest --out X]         (CPU)
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import s2                                                                   # noqa: E402
import s2cue                                                                # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2fix3")
ROLL = os.path.join(OUT, "s2fix3_roll")
READ = os.path.join(OUT, "s2fix3_read.json")
DIAG = os.path.join(OUT, "s2fix3_diag.json")
PCHECK = os.path.join(OUT, "s2fix3_pcheck.json")
DEV_STATES = os.path.join(OUT, "s2fix2", "O", "s2_living_0.npy")         # SR 62's states: development only
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
BASE, BR_BASE, N_STATES = 920000, 38000000, 64
SMOKE = list(range(0, 4))
MAIN = list(range(4, 64))
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
HELD_R = 0.08
NI_MARGIN = -0.05
FLOOR = 0.70                                                                # SR 50's testability floor on intact K1-K2
RUNS = dict(pi05=("I", "H", "Hp"), pi0=("I", "H"), groot=("I", "H"))
AUTH = b"s2fix3-owlv2"
QM = {W: "a white mug", Y: "a yellow and white mug", R: "a red mug"}
QP = "a plate"
QUERIES = [QM[W], QM[Y], QM[R], QP]
MIN_SCORE, IOU_MAX, TOPK = 0.05, 0.3, 60
CAM = "agentview"
# ---- fixed by `pcheck` / `pcheck2` on development states (SR 62's k 0-63), before the predictions and the smoke (see SR 63 "seen")
CAL = dict(mug=[-0.0336, 0.0017, -0.0783], plate=[0.0072, 0.0, -0.0063])  # world-frame constant added to the raw point (plate: edge_radius's x, z)
SIDE_RULE = "world_y_min"                                                   # "left" = the detected plate with the SMALLER world y
ASSIGN = "perm"                                                             # pcheck2: q0 + perm 192/192 start, 64/64 scripted holds
PLATE_LOC = "edge_radius"                                                   # pcheck3 (job 4725886, after smoke 1): split-half xy .40-.45 cm
                                                                            # median (inner_median, smoke 1's rule: 2.2-2.4 cm)
PLATE_CAL = dict(inner_median=[0.0036, 0.0008, -0.0067])                    # per-method plate constant (pcheck3 fills the others)
PLATE_R = 0.0554                                                            # edge_radius: the plate's radius (pcheck3: fitted as the
                                                                            # median |y_sim - y_inner edge| on development states)
QSETS = dict(                                                               # pcheck2's candidate mug phrasings (s2.MUGS order)
    q0=("a white mug", "a yellow and white mug", "a red mug"),
    q1=("a plain white mug", "a mug with yellow", "a red mug"),
    q2=("a white mug", "a yellow mug", "a red mug"),
    q3=("a white cup", "a yellow cup", "a red cup"),
    q4=("an all white mug", "a white mug with yellow", "a red mug"),
    q5=("white", "yellow", "red"))
MAPPING = (
    "NEW states (LIBERO sampler on K1's BDDL, seeds 920000 + k, none equal to an SR 50-62 state), main k = 4-63 (60), branches "
    "seed_base 38000000 + 1000 k; layout O; tasks K1, K2, N1, N2; client seed 7, --bind track.  Models and arms: frozen pi0.5 "
    "(pp_serve_purify --model pi05 --mode intact --comp= --seed 13): I, H (SR 62's arm, s2fix2.py client --arm H unchanged), Hp (H "
    "with perception: OWLv2 boxes + rendered depth + camera calibration for the held mug's identity and the target plate's position; "
    "ProprioHeld for the hold); frozen pi0 (serve_guided --mode ours --seed 13, "
    "BACKBONE_CONFIG=pi0_libero) and frozen GR00T (pp_serve_purify --model groot --mode intact --comp= --seed 13): I and H.  Success "
    "= the BDDL goal.  Matrices [task, state].  Paired bootstrap over states, tasks fixed (s2.boot2_states, B = 2000, numpy "
    "default_rng(0); CI = 2.5 / 97.5).  Label rule (SR 62): for arm X against I, dN = X - I on N1+N2, dK = X - I on K1+K2; FIX WORKS "
    "iff dN lo > 0 AND dK lo > -.05 (NON-INFERIOR at margin -.05); PARTIAL iff exactly one holds; FAILS iff neither.  PRIMARY "
    "(pi0.5): the label of Hp.  PERCEPTION COST (pi0.5): Hp - H pooled over the four tasks, and on N1+N2 and K1+K2: PERCEPTION COSTS "
    "iff hi < 0; PERCEPTION GAINS iff lo > 0; else NO CLEAR DIFFERENCE.  CROSS-MODEL: per model (pi0.5, pi0, GR00T) the label of H; "
    "SR 50's floor: a model is TESTABLE iff its intact (I) K1+K2 success >= .70 on these states; a NOT TESTABLE model's label is "
    "reported as NOT TESTABLE with the rule's outcome descriptive.  SECONDARY: SR 54's rule for every hand-over arm (FIXED iff X - I on N1+N2 lo > 0 and X >= .50; NOT FIXED iff hi < .10; else PARTIAL).  PERCEPTION ACCURACY "
    "(Hp, against the simulator oracle, logged only): target plate registration error (xy, z; cm) and side correct (the registered "
    "point nearer the instructed plate than the other plate); mug identity at registration; at every detector onset, perceived held "
    "mug = the simulator rule's held mug (share, s2cue.boot_share over states) and gate decision = the decision H's rule makes on "
    "the same step; first-onset agreement per task.  C1: the reference planner (s2.plan_episode, unchanged) on the main states; "
    "MET iff its N1-N2 success >= .80.  Descriptive: gate decisions, P phases and aborts, final plate of each mug.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------- perception (client side; numpy + the sidecar)
def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def assign(boxes, scores, method=None):
    """Sidecar reply (queries = the 3 mug queries, then QP) -> (mugs {name: (box, score)}, plates [(box, score)] up to 2).
    greedy: one-to-one over (query, box) by score.  perm: the 3 best mug boxes (by max mug-query score, IoU < IOU_MAX), then the
    query-to-box permutation maximising the sum of each query's score relative to its best of the three."""
    import itertools
    method = method or ASSIGN
    S = np.asarray(scores, np.float64)
    mugs, used = {}, set()
    names = list(QM)
    if method == "greedy":
        pairs = sorted(((S[i, q], q, i) for i in range(len(boxes)) for q in range(len(names)) if S[i, q] >= MIN_SCORE), reverse=True)
        for s_, q, i in pairs:
            m = names[q]
            if m in mugs or i in used:
                continue
            if any(iou(boxes[i], b) >= IOU_MAX for b, _ in mugs.values()):
                continue
            mugs[m] = (list(boxes[i]), float(s_))
            used.add(i)
    else:
        cand = []
        for i in np.argsort(-S[:, :3].max(1)):
            i = int(i)
            if S[i, :3].max() < MIN_SCORE:
                break
            if any(iou(boxes[i], boxes[j]) >= IOU_MAX for j in cand):
                continue
            cand.append(i)
            if len(cand) == 3:
                break
        if cand:
            A = S[np.ix_(cand, [0, 1, 2])]
            Rn = A / np.maximum(A.max(0, keepdims=True), 1e-9)
            best = max(itertools.permutations(range(3), len(cand)), key=lambda p: sum(Rn[b, p[b]] for b in range(len(cand))))
            for b, q in enumerate(best):                                    # box b -> query q
                mugs[names[q]] = (list(boxes[cand[b]]), float(A[b, q]))
                used.add(cand[b])
    plates = []
    for i in np.argsort(-S[:, 3]):
        i = int(i)
        if S[i, 3] < MIN_SCORE or i in used:
            continue
        if any(iou(boxes[i], b) >= IOU_MAX for b, _ in plates):
            continue
        plates.append((list(boxes[i]), float(S[i, 3])))
        if len(plates) == 2:
            break
    return mugs, plates


class Percept:
    """OWLv2 (sidecar) on the policy's agentview image + rendered depth + camera calibration.  Reads, from the simulator, only the
    depth render and the camera model (rg.py rgp's remaining privilege); every object position comes from the detector."""

    def __init__(self, port=None):
        from multiprocessing.connection import Client
        import sc_pu as PU
        self.PU = PU
        port = int(port or os.environ["S2FIX3_DET_PORT"])
        self.conn = Client(("127.0.0.1", port), authkey=AUTH)
        self.conn.send("ping")
        self.model = self.conn.recv()["model"]

    def detect(self, obs, queries=None):
        raw = np.asarray(obs["agentview_image"])
        img = np.ascontiguousarray(raw[::-1, ::-1]).astype(np.uint8)       # the policy's view (libero_client's flip)
        self.conn.send(dict(img=img.tobytes(), shape=list(img.shape), queries=list(queries or QUERIES), topk=TOPK))
        rep = self.conn.recv()
        return img.shape[1], rep

    def points(self, D, cm, box, Wd):
        """Box in the policy view -> raw world point: median of the back-projected inner half of the box."""
        f, Rm, c = cm
        S = D.shape[0]
        x0, y0, x1, y1 = box
        cx, cy, hw, hh = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.25 * (x1 - x0), 0.25 * (y1 - y0)
        r0, r1 = int(max(0, math.floor(cy - hh))), int(min(S - 1, math.ceil(cy + hh)))
        c0, c1 = int(max(0, math.floor(cx - hw))), int(min(S - 1, math.ceil(cx + hw)))
        rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
        rr, cc = rr.ravel(), cc.ravel()
        ct = (Wd - 1) - cc                                                  # policy view = top-down mirrored in columns
        d = D[rr, ct]
        q = np.stack([(ct - S / 2.0) * d / f, (rr - S / 2.0) * d / f, d], 1)
        P = q @ Rm.T + c
        return np.median(P, 0)

    def plate_point(self, D, cm, box, Wd, method=None):
        """A plate box -> raw world point.  inner_median: as points().  ray_plane: the box centre's camera ray cut with the plane at
        the 25th percentile height of the box's inner-half points (mugs on or before the plate only raise the upper quantiles).
        low_extent: the box's points within 1 cm of that height, xy = mid of their 5th / 95th percentiles."""
        method = method or PLATE_LOC
        if method == "inner_median":
            return self.points(D, cm, box, Wd)
        f, Rm, c = cm
        S = D.shape[0]
        x0, y0, x1, y1 = box
        cx, cy, hw, hh = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0), 0.5 * (y1 - y0)

        def pts(fr):
            r0, r1 = int(max(0, math.floor(cy - fr * hh))), int(min(S - 1, math.ceil(cy + fr * hh)))
            c0, c1 = int(max(0, math.floor(cx - fr * hw))), int(min(S - 1, math.ceil(cx + fr * hw)))
            rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
            rr, cc = rr.ravel(), cc.ravel()
            ct = (Wd - 1) - cc
            d = D[rr, ct]
            return np.stack([(ct - S / 2.0) * d / f, (rr - S / 2.0) * d / f, d], 1) @ Rm.T + c
        z0 = float(np.percentile(pts(0.5)[:, 2], 25))
        if method == "ray_plane":
            ct = (Wd - 1) - cx
            v = Rm @ np.array([(ct - S / 2.0) / f, (cy - S / 2.0) / f, 1.0])
            t = (z0 - c[2]) / v[2]
            p = c + t * v
            p[2] = z0
            return p
        P = pts(1.0)
        P = P[np.abs(P[:, 2] - z0) < 0.01]
        lo, hi = np.percentile(P[:, :2], 5, axis=0), np.percentile(P[:, :2], 95, axis=0)
        if method == "edge_radius":                                        # the plates sit at the image's side borders, so their
            yin = lo[1] if abs(lo[1]) < abs(hi[1]) else hi[1]              # outer side is cut: centre y = the inner (scene-centre)
            return np.array([0.5 * (lo[0] + hi[0]), yin + math.copysign(PLATE_R, yin), z0])   # edge + the plate's radius
        return np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), z0])

    def scene(self, env, obs, qm=None, method=None):
        """-> dict(mugs {name: dict(pos, raw, box, score)}, plates [dict(pos, raw, box, score)], left, right, secs).  qm: mug
        queries in s2.MUGS order (default QM); method: assign's method (default ASSIGN)."""
        Wd, rep = self.detect(obs, None if qm is None else list(qm) + [QP])
        mugs, plates = assign(rep["boxes"], rep["scores"], method)
        D = self.PU.depth_std(env, dict(hide=[]), CAM)                     # this step's depth (top-down rows, robosuite columns)
        cm = self.PU.cam_model(env.env.sim, CAM)
        out = dict(mugs={}, plates=[], secs=rep.get("secs"))
        for m, (b, s_) in mugs.items():
            p = self.points(D, cm, b, Wd)
            out["mugs"][m] = dict(raw=p.tolist(), pos=(p + np.array(CAL["mug"])).tolist(), box=[round(v, 1) for v in b], score=round(s_, 4))
        for b, s_ in plates:
            p = self.plate_point(D, cm, b, Wd)
            out["plates"].append(dict(raw=p.tolist(), pos=(p + np.array(CAL["plate"])).tolist(), box=[round(v, 1) for v in b],
                                      score=round(s_, 4), img_x=0.5 * (b[0] + b[2])))
        out["left"], out["right"] = sides(out["plates"])
        return out


def sides(plates):
    """-> (left plate point, right plate point) by SIDE_RULE; None where fewer than two plates were detected."""
    if len(plates) < 2:
        return None, None
    a, b = plates[0], plates[1]
    assert SIDE_RULE == "world_y_min", SIDE_RULE
    l_, r_ = (a, b) if a["pos"][1] < b["pos"][1] else (b, a)
    return l_["pos"], r_["pos"]


def side_word(target):
    return "left" if target == PL else "right"


# ---------------------------------------------------------------- states (LIBERO venv, EGL)
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
        np.load(os.path.join(OUT, d, L, "%s_0.npy" % s2.SUITE)) for d, L in
        (("s2cue3", "O"), ("s2cue4", "O"), ("s2cue5", "O"), ("s2fix2", "O"), ("s2fix2", "PD2"))]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("same as an SR 50-62 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    for ph, ks in (("smoke", SMOKE), ("main", MAIN)):
        with open(os.path.join(CD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE + 1000 * k, at=0, variant="s2") for k in ks], fh, indent=1)
    print("[s2fix3 states] %d new states (seed %d + k), none equal to an SR 50-62 state; wrote %s" % (len(S), BASE, CD), flush=True)


# ---------------------------------------------------------------- step 0: why does P's N1 release fail? (CPU, descriptive)
def tilt_deg(q_wxyz):
    w, x, y, z = [float(v) for v in q_wxyz]
    zz = 1 - 2 * (x * x + y * y)                                            # body z axis . world z
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def diag(a):
    """SR 62's arm H (and G, which hands over identically on N) on layout O: for every episode where P took over, the carry's end
    (limit vs reached), the named mug's offset from the target plate when P opened, the live hand-minus-mug offset at the hand-over
    against the demonstrated grasp offsets, the final offset, height and tilt; what the reference planner's relocation rule would
    shift the release by."""
    out = os.path.abspath(a.out or DIAG)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    import s2fix as SF
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    KP = json.load(open(os.path.join(s2.SD, "keypoints.json"), encoding="utf-8"))["tasks"]
    ks = list(range(4, 64))
    RD = os.path.join(OUT, "s2fix2_roll", "O")
    res = dict(note="DESCRIPTIVE: SR 62's existing outputs only (results/flawed/s2fix2_roll/O, pi05 H and G)", kp={}, arms={})
    for k_, v in KP.items():
        res["kp"][k_] = {x: v.get(x) for x in ("grasp_off", "release_off", "carry_z_above_tgt")}
    dg = (np.array(KP["K1"]["grasp_off"]) - np.array(KP["K2"]["grasp_off"])).tolist()
    res["reloc_shift"] = dict(N1=dg, N2=[-v for v in dg])
    print("[s2fix3 diag] keypoints %s | relocation shift of the hand at release: N1 +%s, N2 -%s (K1.grasp_off - K2.grasp_off)" % (
        json.dumps(res["kp"]), np.round(dg, 4).tolist(), np.round(dg, 4).tolist()), flush=True)
    for arm in ("H", "G"):
        res["arms"][arm] = {}
        for c in ("K1", "K2", "N1", "N2"):
            t = T[c]
            stem = os.path.join(RD, "pi05_%s_%d" % (arm, t["tid"]))
            rr = s2.stage_rows(stem, t, ks)
            Tr = s2.load_track(stem + "_track.npz")
            Hs = json.load(open(stem + "_h.json", encoding="utf-8"))["eps"]
            assert [h["trial"] for h in Hs] == ks
            rows = []
            for k, r, tr, h in zip(ks, rr, Tr, Hs):
                if not h["handovers"]:
                    continue
                ho = int(h["handovers"][-1]["row"])
                obj, tgt, ee = tr["obj"].astype(np.float64), tr["tgt"].astype(np.float64), tr["ee"].astype(np.float64)
                cm = h["committed"]
                carry = [p for p in (h.get("p_log") or []) if p["phase"] == 1]
                d = dict(k=k, success=bool(r["success"]), phase_max=h["phase_max"], aborts=len(h["aborts"]),
                         live_goff=(ee[ho] - obj[ho]).tolist(),
                         carry_steps=carry[0]["steps"] if carry else None, carry_err=carry[0]["err"] if carry else None,
                         carry_hit_limit=bool(carry and carry[0]["steps"] >= SF.CARRY_STEPS and carry[0]["err"] >= 0.015))
                if cm is not None:
                    cm = int(cm)
                    d.update(open_dxy=(obj[cm] - tgt[cm])[:2].tolist(), open_dz=float(obj[cm, 2] - tgt[cm, 2]),
                             goff_at_open=(ee[cm] - obj[cm]).tolist(), slip=float(np.linalg.norm((ee[cm] - obj[cm]) - (ee[ho] - obj[ho]))),
                             hand_err_at_open=float(np.linalg.norm(ee[cm][:2] - (tgt[cm][:2] + np.array(KP[SF.HALF_OF_PLATE[t["target"]]]["release_off"])[:2]))))
                d.update(final_dxy=(obj[-1] - tgt[-1])[:2].tolist(), final_dz=float(obj[-1, 2] - tgt[-1, 2]), final_tilt=tilt_deg(tr["quat"][-1]))
                rows.append(d)

            def med(key, sel, i=None):
                v = [x[key] if i is None else x[key][i] for x in rows if sel(x) and x.get(key) is not None]
                return None if not v else float(np.median(v))
            summ = {}
            for nm, sel in (("success", lambda x: x["success"]), ("fail", lambda x: not x["success"])):
                summ[nm] = dict(n=sum(sel(x) for x in rows), committed=sum(sel(x) and x.get("open_dxy") is not None for x in rows),
                                carry_hit_limit=sum(sel(x) and x["carry_hit_limit"] for x in rows),
                                aborted=sum(sel(x) and x["aborts"] > 0 for x in rows),
                                open_dx=med("open_dxy", sel, 0), open_dy=med("open_dxy", sel, 1),
                                open_norm=None if not any(sel(x) and x.get("open_dxy") for x in rows) else float(np.median(
                                    [np.linalg.norm(x["open_dxy"]) for x in rows if sel(x) and x.get("open_dxy")])),
                                live_goff_x=med("live_goff", sel, 0), live_goff_y=med("live_goff", sel, 1), live_goff_z=med("live_goff", sel, 2),
                                slip=med("slip", sel), hand_err_at_open=med("hand_err_at_open", sel),
                                final_norm=float(np.median([np.linalg.norm(x["final_dxy"]) for x in rows if sel(x)])) if any(sel(x) for x in rows) else None,
                                final_dz=med("final_dz", sel), final_tilt=med("final_tilt", sel),
                                tipped=sum(sel(x) and x["final_tilt"] > 45 for x in rows))
            if c in ("N1", "N2"):                                           # the relocated release, applied to the observed open offsets
                sh = np.array(res["reloc_shift"][c][:2])
                op = [x for x in rows if x.get("open_dxy") is not None]
                summ["reloc_counterfactual"] = dict(
                    n=len(op), open_norm_now=float(np.median([np.linalg.norm(x["open_dxy"]) for x in op])) if op else None,
                    open_norm_reloc=float(np.median([np.linalg.norm(np.array(x["open_dxy"]) + sh) for x in op])) if op else None,
                    fail_open_norm_reloc=(float(np.median([np.linalg.norm(np.array(x["open_dxy"]) + sh) for x in op if not x["success"]]))
                                          if any(not x["success"] for x in op) else None))
            res["arms"][arm][c] = dict(rows=rows, summary=summ, success=float(np.mean([r["success"] for r in rr])))
            print("[s2fix3 diag] %s %s success %.3f | %s" % (arm, c, res["arms"][arm][c]["success"], json.dumps(summ)), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(out + ".tmp", out)
    print("[s2fix3 diag] wrote %s" % out, flush=True)


# ---------------------------------------------------------------- perception check on DEVELOPMENT states (EGL + sidecar)
def pcheck(a):
    """SR 62's states k 0-63 (development, not this probe's): settle (WAIT dummy steps), perceive, compare with the simulator.  Then
    a scripted grasp-and-lift (fa1g_plan.ctrl to the demonstrated grasp keypoints) of the white and the yellow-and-white mug on
    k 0-15, and the held-mug rule on the perceived scene at the oracle hold."""
    out = os.path.abspath(a.out or PCHECK)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    import fa1b_common as FB
    import s2_tasks as T2
    from fa1g_plan import ctrl
    M_ = s2.meta()
    KP = json.load(open(os.path.join(s2.SD, "keypoints.json"), encoding="utf-8"))["tasks"]
    S = np.load(DEV_STATES)
    P = Percept()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))
    env.seed(7)
    rows, holds = [], []
    for k in range(len(S)):
        env.reset()
        obs = env.set_init_state(S[k])
        for _ in range(s2.WAIT):
            obs, _, _, _ = env.step(T2.DUMMY)
        sim = {n: T2.positions(env, [n])[0].tolist() for n in s2.MOVABLE}
        sc = P.scene(env, obs)
        rows.append(dict(k=k, sim=sim, scene=sc))
        if k < 16:
            for m, half in ((W, "K1"), (Y, "K2")):
                env.reset()
                obs = env.set_init_state(S[k])
                for _ in range(s2.WAIT):
                    obs, _, _, _ = env.step(T2.DUMMY)
                kp = KP[half]
                rng = np.random.default_rng(1000 + k)
                con, hm = FB.Contact(env, [m]), FB.Held()
                goff, RG = np.array(kp["grasp_off"]), np.array(kp["grasp_rot"])
                ph, inph, held_at, z0 = 0, 0, None, None
                for st in range(200):
                    po = T2.positions(env, [m])[0]
                    s_ = dict(ee_pos=np.asarray(obs["robot0_eef_pos"], np.float64), ee_rot=T2.quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64)))
                    if ph == 0:
                        wp, g = po + goff + [0, 0, 0.10], -1.0
                    elif ph == 1:
                        wp, g = po + goff, -1.0
                    elif ph == 2:
                        wp, g = po + goff, 1.0
                    else:
                        wp, g = np.array([s_["ee_pos"][0], s_["ee_pos"][1], z0 + 0.06]), 1.0
                    act, err = ctrl(s_, wp, RG, g, rng)
                    obs, _, _, _ = env.step(act.tolist())
                    h, _ = hm.step(*con.touching(m), con.pos(m), obs["robot0_eef_pos"])
                    inph += 1
                    if ph in (0, 1) and (err < 0.01 or inph >= 40):
                        ph, inph = ph + 1, 0
                    elif ph == 2 and inph >= 12:
                        ph, inph, z0 = 3, 0, float(obs["robot0_eef_pos"][2])
                    if ph == 3 and h:
                        held_at = st
                        break
                ee = np.asarray(obs["robot0_eef_pos"], np.float64)
                simm = {n: T2.positions(env, [n])[0] for n in s2.MUGS}
                d_sim = {n: float(np.linalg.norm(simm[n][:2] - ee[:2])) for n in s2.MUGS}
                ms = min(d_sim, key=d_sim.get)
                sim_held = ms if d_sim[ms] < HELD_R else None
                if held_at is None:
                    holds.append(dict(k=k, mug=m, held=False))
                    continue
                sc2 = P.scene(env, obs)
                mem = {n: v["pos"] for n, v in rows[-1]["scene"]["mugs"].items()}
                det = {n: v["pos"] for n, v in sc2["mugs"].items()}
                res_ = {}
                for rule, pos in (("onset_detect", dict(mem, **det)), ("start_memory", mem), ("onset_only", det)):
                    dd = {n: float(np.linalg.norm(np.array(p)[:2] - ee[:2])) for n, p in pos.items()}
                    hm_ = min(dd, key=dd.get) if dd else None
                    res_[rule] = hm_ if (hm_ is not None and dd[hm_] < HELD_R) else None
                holds.append(dict(k=k, mug=m, held=True, sim_held=sim_held, rules=res_,
                                  det={n: dict(err_cm=round(100 * float(np.linalg.norm(np.array(v["pos"]) - simm[n])), 2), score=v["score"])
                                       for n, v in sc2["mugs"].items()}))
        print("[s2fix3 pcheck] k%d mugs %s plates %s" % (k, {n.split("_")[0]: (v["score"], np.round(v["pos"], 3).tolist())
                                                            for n, v in rows[-1]["scene"]["mugs"].items()},
                                                        [(p["score"], np.round(p["pos"], 3).tolist()) for p in rows[-1]["scene"]["plates"]]),
              flush=True)
    env.close()
    # ---- raw errors and the per-type constants (median sim - raw); split halves (even / odd k) for an honest residual
    rep = dict(model=P.model, n=len(rows), cal_in_use=CAL, side_rule_in_use=SIDE_RULE)
    for typ, names in (("mug", s2.MUGS), ("plate", s2.PLATES)):
        D = {0: [], 1: []}
        for r in rows:
            if typ == "mug":
                for n in names:
                    if n in r["scene"]["mugs"]:
                        D[r["k"] % 2].append(np.array(r["sim"][n]) - np.array(r["scene"]["mugs"][n]["raw"]))
            else:
                for p in r["scene"]["plates"]:                              # pair each detected plate with its nearest sim plate
                    n = min(names, key=lambda x: np.linalg.norm(np.array(r["sim"][x])[:2] - np.array(p["raw"])[:2]))
                    D[r["k"] % 2].append(np.array(r["sim"][n]) - np.array(p["raw"]))
        allD = np.array(D[0] + D[1])
        cal = np.median(allD, 0) if len(allD) else np.zeros(3)
        half = {}
        for fit, test in ((0, 1), (1, 0)):
            if len(D[fit]) and len(D[test]):
                c_ = np.median(np.array(D[fit]), 0)
                e = np.array(D[test]) - c_
                half["fit%d_test%d" % (fit, test)] = dict(xy_med_cm=round(100 * float(np.median(np.linalg.norm(e[:, :2], axis=1))), 2),
                                                          xy_p90_cm=round(100 * float(np.percentile(np.linalg.norm(e[:, :2], axis=1), 90)), 2),
                                                          z_med_abs_cm=round(100 * float(np.median(np.abs(e[:, 2]))), 2))
        rep[typ] = dict(n=int(len(allD)), cal_fit=np.round(cal, 4).tolist(),
                        raw_xy_med_cm=round(100 * float(np.median(np.linalg.norm(allD[:, :2], axis=1))), 2) if len(allD) else None,
                        split_half=half)
    # ---- identity and side
    n_ok = n_tot = n_miss = 0
    for r in rows:
        for n in s2.MUGS:
            if n not in r["scene"]["mugs"]:
                n_miss += 1
                continue
            p = np.array(r["scene"]["mugs"][n]["raw"]) + np.array(rep["mug"]["cal_fit"])
            near = min(s2.MUGS, key=lambda x: np.linalg.norm(np.array(r["sim"][x])[:2] - p[:2]))
            n_ok += int(near == n)
            n_tot += 1
    rep["mug_identity_at_start"] = dict(correct=n_ok, detected=n_tot, missed=n_miss)
    sd = dict(world_y=[0, 0], image_x=[0, 0], two_plates=0)
    for r in rows:
        ps = r["scene"]["plates"]
        if len(ps) < 2:
            continue
        sd["two_plates"] += 1
        for rule in ("world_y", "image_x"):
            a_, b_ = ps
            if rule == "world_y":
                l_ = a_ if a_["raw"][1] > b_["raw"][1] else b_
            else:
                l_ = a_ if a_["img_x"] < b_["img_x"] else b_
            near = min(s2.PLATES, key=lambda x: np.linalg.norm(np.array(r["sim"][x])[:2] - np.array(l_["raw"])[:2]))
            sd[rule][0] += int(near == PL)
            sd[rule][1] += 1
    rep["left_is_plate_1"] = sd
    rep["sim_plate_1_y_gt_plate_2_y"] = int(sum(r["sim"][PL][1] > r["sim"][PR][1] for r in rows))
    hs = [h for h in holds if h["held"]]
    rep["holds"] = dict(n_tried=len(holds), n_held=len(hs), sim_rule_correct=sum(h["sim_held"] == h["mug"] for h in hs),
                        rules={rule: sum(h["rules"][rule] == h["mug"] for h in hs) for rule in ("onset_detect", "start_memory", "onset_only")},
                        onset_detected={n: sum(n in h["det"] for h in hs if h["mug"] == n) for n in (W, Y)},
                        onset_err_cm_held=float(np.median([h["det"][h["mug"]]["err_cm"] for h in hs if h["mug"] in h["det"]])) if hs else None)
    secs = [r["scene"]["secs"] for r in rows if r["scene"].get("secs") is not None]
    rep["det_secs_median"] = float(np.median(secs)) if secs else None
    print("[s2fix3 pcheck] SUMMARY %s" % json.dumps(rep), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(report=rep, rows=rows, holds=holds), fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(out + ".tmp", out)
    print("[s2fix3 pcheck] wrote %s" % out, flush=True)


def scripted_hold(env, S_k, m, kp, seed, T2, FB, ctrl):
    """pcheck's scripted grasp-and-lift of mug m from state S_k (demonstrated grasp keypoints, fa1g_plan.ctrl) -> (obs, held?)."""
    env.reset()
    obs = env.set_init_state(S_k)
    for _ in range(s2.WAIT):
        obs, _, _, _ = env.step(T2.DUMMY)
    rng = np.random.default_rng(seed)
    con, hm = FB.Contact(env, [m]), FB.Held()
    goff, RG = np.array(kp["grasp_off"]), np.array(kp["grasp_rot"])
    ph, inph, z0 = 0, 0, None
    for _ in range(200):
        po = T2.positions(env, [m])[0]
        s_ = dict(ee_pos=np.asarray(obs["robot0_eef_pos"], np.float64), ee_rot=T2.quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64)))
        if ph == 0:
            wp, g = po + goff + [0, 0, 0.10], -1.0
        elif ph == 1:
            wp, g = po + goff, -1.0
        elif ph == 2:
            wp, g = po + goff, 1.0
        else:
            wp, g = np.array([s_["ee_pos"][0], s_["ee_pos"][1], z0 + 0.06]), 1.0
        act, err = ctrl(s_, wp, RG, g, rng)
        obs, _, _, _ = env.step(act.tolist())
        h, _ = hm.step(*con.touching(m), con.pos(m), obs["robot0_eef_pos"])
        inph += 1
        if ph in (0, 1) and (err < 0.01 or inph >= 40):
            ph, inph = ph + 1, 0
        elif ph == 2 and inph >= 12:
            ph, inph, z0 = 3, 0, float(obs["robot0_eef_pos"][2])
        if ph == 3 and h:
            return obs, True
    return obs, False


def pcheck2(a):
    """pcheck found OWLv2 swapping the white and the yellow-and-white mug under q0 + greedy.  On the same DEVELOPMENT states, every
    candidate phrasing (QSETS) x assignment (greedy, perm): mug identity at the settled start (k 0-63) and at a scripted hold of the
    white and the yellow-and-white mug (k 0-31), the held-mug rule on onset detection merged with the start memory; and the mug CAL
    by nearest-mug pairing (identity-free).  The best candidate is chosen on these numbers, before predictions and smoke."""
    out = os.path.abspath(a.out or PCHECK.replace(".json", "2.json"))
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    import fa1b_common as FB
    import s2_tasks as T2
    from fa1g_plan import ctrl
    M_ = s2.meta()
    KP = json.load(open(os.path.join(s2.SD, "keypoints.json"), encoding="utf-8"))["tasks"]
    S = np.load(DEV_STATES)
    P = Percept()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))
    env.seed(7)
    METHODS = ("greedy", "perm")
    C = {(q, mt): dict(start_ok=0, start_n=0, start_missed=0, hold_ok=0, hold_n=0, hold_white_ok=0, hold_yw_ok=0) for q in QSETS for mt in METHODS}
    raw_nearest = []

    def capture(obs):
        D = P.PU.depth_std(env, dict(hide=[]), CAM)
        cm = P.PU.cam_model(env.env.sim, CAM)
        res = {}
        for q, qm in QSETS.items():
            Wd, rep = P.detect(obs, list(qm) + [QP])
            for mt in METHODS:
                mugs, _ = assign(rep["boxes"], rep["scores"], mt)
                res[(q, mt)] = {n: P.points(D, cm, b, Wd) + np.array(CAL["mug"]) for n, (b, _) in mugs.items()}
        return res
    start = {}
    for k in range(len(S)):
        env.reset()
        obs = env.set_init_state(S[k])
        for _ in range(s2.WAIT):
            obs, _, _, _ = env.step(T2.DUMMY)
        sim = {n: T2.positions(env, [n])[0] for n in s2.MUGS}
        res = capture(obs)
        start[k] = res
        for key, pos in res.items():
            for n, p in pos.items():
                near = min(s2.MUGS, key=lambda x: np.linalg.norm(sim[x][:2] - p[:2]))
                C[key]["start_ok"] += int(near == n)
                C[key]["start_n"] += 1
                if key == ("q0", "greedy"):
                    raw_nearest.append(sim[near] - p)
            C[key]["start_missed"] += 3 - len(pos)
        if k < 32:
            for m, half in ((W, "K1"), (Y, "K2")):
                obs, held = scripted_hold(env, S[k], m, KP[half], 1000 + k, T2, FB, ctrl)
                if not held:
                    continue
                ee = np.asarray(obs["robot0_eef_pos"], np.float64)
                res2 = capture(obs)
                for key, pos in res2.items():
                    mem = dict(start[k][key])
                    mem.update(pos)
                    dd = {n: float(np.linalg.norm(p[:2] - ee[:2])) for n, p in mem.items()}
                    hm_ = min(dd, key=dd.get) if dd else None
                    got = hm_ if (hm_ is not None and dd[hm_] < HELD_R) else None
                    C[key]["hold_n"] += 1
                    C[key]["hold_ok"] += int(got == m)
                    C[key]["hold_white_ok" if m == W else "hold_yw_ok"] += int(got == m)
        if k % 8 == 0:
            print("[s2fix3 pcheck2] k%d done" % k, flush=True)
    env.close()
    rn = np.array(raw_nearest)
    rep = dict(cal_mug_nearest=np.round(np.median(rn, 0), 4).tolist(), cal_mug_in_use=CAL["mug"],
               nearest_xy_med_cm=round(100 * float(np.median(np.linalg.norm((rn - np.median(rn, 0))[:, :2], axis=1))), 2),
               candidates={"%s|%s" % k_: v for k_, v in C.items()}, qsets=QSETS)
    for k_, v in sorted(C.items(), key=lambda kv: -(kv[1]["start_ok"] + kv[1]["hold_ok"])):
        print("[s2fix3 pcheck2] %s %s %s" % (k_, QSETS[k_[0]], json.dumps(v)), flush=True)
    print("[s2fix3 pcheck2] SUMMARY %s" % json.dumps(dict(cal_mug_nearest=rep["cal_mug_nearest"], nearest_xy_med_cm=rep["nearest_xy_med_cm"])), flush=True)
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    os.replace(out + ".tmp", out)
    print("[s2fix3 pcheck2] wrote %s" % out, flush=True)


def pcheck3(a):
    """After the smoke (Hp K1 / K2 lost 4 of 8 to P's release with registration errors of 1.4-3.3 cm): the plate localisations of
    Percept.plate_point on the DEVELOPMENT states (k 0-63, settled start) against the simulator; per method the constant (median sim -
    raw, all states) and the split-half (even / odd k) residual."""
    out = os.path.abspath(a.out or PCHECK.replace(".json", "3.json"))
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    import s2_tasks as T2
    M_ = s2.meta()
    S = np.load(DEV_STATES)
    P = Percept()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))
    env.seed(7)
    global PLATE_R
    assert PLATE_R == 0.0, "pcheck3 fits PLATE_R from the inner edge"
    METH = ("inner_median", "ray_plane", "low_extent", "edge_radius")
    Dd = {m: {0: [], 1: []} for m in METH}
    for k in range(len(S)):
        env.reset()
        obs = env.set_init_state(S[k])
        for _ in range(s2.WAIT):
            obs, _, _, _ = env.step(T2.DUMMY)
        sim = {n: T2.positions(env, [n])[0] for n in s2.PLATES}
        Wd, rep = P.detect(obs)
        _, plates = assign(rep["boxes"], rep["scores"])
        D = P.PU.depth_std(env, dict(hide=[]), CAM)
        cm = P.PU.cam_model(env.env.sim, CAM)
        for b, _ in plates:
            p0 = P.plate_point(D, cm, b, Wd, "inner_median")
            n = min(s2.PLATES, key=lambda x: np.linalg.norm(sim[x][:2] - p0[:2]))
            for m in METH:
                q = P.plate_point(D, cm, b, Wd, m)
                d_ = sim[n] - q
                if m == "edge_radius":                                     # fold the side's sign into y: fit (cx, r, cz)
                    d_ = np.array([d_[0], math.copysign(1.0, q[1]) * d_[1], d_[2], math.copysign(1.0, q[1])])
                Dd[m][k % 2].append(d_)
    env.close()
    rep_ = {}

    def resid(Dt, c_, m):
        Dt = np.array(Dt)
        if m != "edge_radius":
            return Dt - c_
        s = Dt[:, 3]
        return np.stack([Dt[:, 0] - c_[0], s * (Dt[:, 1] - c_[1]), Dt[:, 2] - c_[2]], 1)
    for m in METH:
        allD = np.array(Dd[m][0] + Dd[m][1])
        cal = np.median(allD, 0)[:3]
        half = {}
        for fit, test in ((0, 1), (1, 0)):
            e = resid(Dd[m][test], np.median(np.array(Dd[m][fit]), 0)[:3], m)
            n_ = np.linalg.norm(e[:, :2], axis=1)
            half["fit%d_test%d" % (fit, test)] = dict(xy_med_cm=round(100 * float(np.median(n_)), 2),
                                                      xy_p90_cm=round(100 * float(np.percentile(n_, 90)), 2),
                                                      xy_max_cm=round(100 * float(np.max(n_)), 2),
                                                      z_med_abs_cm=round(100 * float(np.median(np.abs(e[:, 2]))), 2))
        rep_[m] = dict(n=len(allD), cal=np.round(cal, 4).tolist(), split_half=half)
        print("[s2fix3 pcheck3] %s %s" % (m, json.dumps(rep_[m])), flush=True)
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep_, fh, indent=1)
    os.replace(out + ".tmp", out)
    print("[s2fix3 pcheck3] wrote %s" % out, flush=True)


# ---------------------------------------------------------------- client: arm Hp (LIBERO venv)
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
    assert arm == "Hp", arm
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
    PER = Percept()
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
            self.rng = np.random.default_rng(self.sb + 20)                 # SR 54's P rng rule
            self.det = ProprioHeld(X["tw"], X["lift"])
            self.pl, self.obs = None, obs
            self.h = dict(onsets=[], events=[], handovers=[], aborts=[], committed=None, p_steps=0, phase_max=-1)
            self.reg, self.mem, self.tgt_p = None, {}, None
            return obs

        def _tgt(self):
            if arm == "Hp":
                return np.asarray(self.tgt_p, np.float64)
            return self._pos(TASK["target"])

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

        def _register(self, obs):
            sc = PER.scene(self._e, obs)
            self.mem = {m: v["pos"] for m, v in sc["mugs"].items()}
            side = side_word(TASK["target"])
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

        def step(self, a):
            if self.pl is not None:
                ee = np.asarray(self.obs["robot0_eef_pos"], np.float64)
                a = self.pl.action(ee, T2.quat_mat(np.asarray(self.obs["robot0_eef_quat"], np.float64)), self._tgt()).tolist()
                self.h["p_steps"] += 1
            r = super().step(a)
            self.obs = r[0]
            if self.n < s2.WAIT:
                return r
            if arm == "Hp" and self.reg is None:
                self._register(r[0])
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
                ee = r[0]["robot0_eef_pos"]
                sh, sd = self._sim_held(ee)
                ev = dict(row=row, oracle_held=bool(self.rows["held"][-1]))
                if arm == "Hp":
                    sc = PER.scene(self._e, r[0])
                    self.mem.update({m: v["pos"] for m, v in sc["mugs"].items()})
                    held, dd = self._per_held(ee)
                    ev.update(detected={m.split("_")[0]: dict(pos=np.round(v["pos"], 4).tolist(), score=v["score"])
                                        for m, v in sc["mugs"].items()})
                    tgt_ok = self.tgt_p is not None
                else:
                    held, dd = sh, sd
                    tgt_ok = True
                ev.update(held=held, dist={m.split("_")[0]: round(v, 4) for m, v in dd.items()},
                          sim_held=sh, sim_dist={m.split("_")[0]: round(v, 4) for m, v in sd.items()},
                          oracle_decision="place" if sh == TASK["object"] else "vla: other mug")
                if held != TASK["object"]:
                    ev["decision"] = "vla: other mug"
                elif not tgt_ok:
                    ev["decision"] = "vla: no target"
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
                rec = dict(arm=arm, layout="O", trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, tid=TID, key=TASK["key"])
                rec.update(self.h)
                rec["reg"] = self.reg
                rec["p_final_phase"] = None if self.pl is None else self.pl.phase
                rec["p_log"] = None if self.pl is None else self.pl.log["phases"]
                SIDE.append(rec)
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, layout="O", tid=TID, key=TASK["key"], language=TASK["language"], lex=X, kt=KT, held_r=HELD_R,
                                   cal=CAL, side_rule=SIDE_RULE, queries=QUERIES, detector=None if PER is None else PER.model,
                                   place="SR 54 object-blind P (%s target position)" % ("PERCEIVED" if arm == "Hp" else "simulator"),
                                   eps=SIDE), fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
                os.replace(tmp, SIDEF)
                print("[s2fix3] %s %s trial %s: reg %s | events %s | handovers %s aborts %d committed %s phase_max %d p_steps %d" % (
                    arm, TASK["key"], rec["trial"], None if self.reg is None else self.reg["oracle"].get("tgt_err_xy_cm"),
                    [(e["row"], e["held"] and e["held"].split("_")[0], e["sim_held"] and e["sim_held"].split("_")[0], e["decision"])
                     for e in rec["events"][:4]],
                    [h["row"] for h in rec["handovers"]], len(rec["aborts"]), rec["committed"], rec["phase_max"], rec["p_steps"]),
                    flush=True)
                self.pl = None
            self._e.close()

    LC.OffScreenRenderEnv = lambda *a, **k: HEnv(ORIG(*a, **k))
    print("[s2fix3] arm %s task %s %r object %s target %s max_steps %d lex %s kt %s held_r %.2f detector %s cal %s side %s" % (
        arm, TASK["key"], TASK["language"], TASK["object"], TASK["target"], ms, (X["tw"], X["lift"]),
        dict(carry_z=KT["carry_z_above_tgt"], release_off=KT["release_off"]), HELD_R, None if PER is None else PER.model, CAL, SIDE_RULE),
        flush=True)
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
def stem_of(RD, model, arm, tid):
    return os.path.join(RD, "O", "%s_%s_%d" % (model, arm, tid))


def load_run(RD, model, arm, ks):
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    out = dict(succ={}, rows={}, side={})
    for c in s2.KNOWN + s2.NEW:
        st = stem_of(RD, model, arm, T[c]["tid"])
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


def fix_label(dN, dK):
    a_, b_ = dN[1] > 0, dK[1] > NI_MARGIN
    return "FIX WORKS" if (a_ and b_) else ("PARTIAL" if (a_ or b_) else "FAILS")


def ni_label(d):
    return "NON-INFERIOR" if d[1] > NI_MARGIN else ("INFERIOR" if d[2] < NI_MARGIN else "NOT SHOWN")


def diff_label(d, pos="HELPS", neg="HURTS"):
    return pos if d[1] > 0 else (neg if d[2] < 0 else "NO CLEAR DIFFERENCE")


def sr54_rule(d, x):
    return "FIXED" if (d[1] > 0 and x >= 0.50) else ("NOT FIXED" if d[2] < 0.10 else "PARTIAL")


def accuracy(run, ks):
    """Hp side files -> perception accuracy against the logged simulator oracle."""
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    out = dict(per_task={})
    xy, z, side, mug_ok, mug_n, reg_missing = [], [], [0, 0], 0, 0, 0
    Fh, Dh, Fd, Dd, F1, D1 = (np.zeros(len(ks)) for _ in range(6))
    conf = {}
    for c, H in run["side"].items():
        if H is None:
            continue
        pt = dict(onsets=0, held_agree=0, decision_agree=0, first_agree=0, first_n=0, wrong_place=0, missed_place=0)
        for i, h in enumerate(H):
            rg = h.get("reg")
            if rg is None or rg["target"] is None:
                reg_missing += 1
            else:
                o = rg["oracle"]
                xy.append(o["tgt_err_xy_cm"])
                z.append(o["tgt_err_z_cm"])
                side[0] += int(o["side_ok"])
                side[1] += 1
            if rg is not None:
                mug_ok += sum(rg["oracle"]["mug_ok"].values())
                mug_n += len(rg["oracle"]["mug_ok"])
            for j, e in enumerate(h["events"]):
                ag = int(e["held"] == e["sim_held"])
                dg = int(e["decision"] == e["oracle_decision"])
                pt["onsets"] += 1
                pt["held_agree"] += ag
                pt["decision_agree"] += dg
                pt["wrong_place"] += int(e["decision"] == "place" and e["oracle_decision"] != "place")
                pt["missed_place"] += int(e["decision"] != "place" and e["oracle_decision"] == "place")
                key = "%s->%s" % (e["sim_held"] and e["sim_held"].split("_")[0], e["held"] and e["held"].split("_")[0])
                conf[key] = conf.get(key, 0) + 1
                Fh[i] += ag
                Dh[i] += 1
                Fd[i] += dg
                Dd[i] += 1
                if j == 0:
                    pt["first_agree"] += ag
                    pt["first_n"] += 1
                    F1[i] += ag
                    D1[i] += 1
        out["per_task"][c] = pt
    out.update(target_xy_cm=dict(median=float(np.median(xy)) if xy else None, p90=float(np.percentile(xy, 90)) if xy else None,
                                 max=float(np.max(xy)) if xy else None, n=len(xy)),
               target_z_cm=dict(median=float(np.median(z)) if z else None, med_abs=float(np.median(np.abs(z))) if z else None),
               target_side_ok=side, target_missing=reg_missing, mug_identity_at_registration=[mug_ok, mug_n],
               onset_held_agree=s2cue.boot_share(Fh, Dh), onset_decision_agree=s2cue.boot_share(Fd, Dd),
               first_onset_held_agree=s2cue.boot_share(F1, D1), confusion_sim_to_perceived=conf)
    return out


def summary(d):
    M_ = s2.meta()
    for model, arms in RUNS.items():
        for arm in arms:
            for tid in TIDS:
                st = stem_of(d, model, arm, tid)
                if not (os.path.exists(st + ".json") and os.path.exists(st + "_track.npz")):
                    continue
                t = M_["tasks"][tid]
                Rj = json.load(open(st + ".json", encoding="utf-8"))
                ks = [int(e["trial"]) for v in Rj["tasks"].values() for e in v["episodes"]]
                H = json.load(open(st + "_h.json", encoding="utf-8"))["eps"] if os.path.exists(st + "_h.json") else [None] * len(ks)
                for k, r, h in zip(ks, s2.stage_rows(st, t, ks), H):
                    on = {"%s>%s" % (m.split("_")[0][:5], p): 1 for m in s2.MUGS for p in s2.PLATES if s2.on_of(r["on"], m, p)}
                    ev = None if h is None else [(e["row"], e.get("held"), e.get("sim_held"), e.get("decision")) for e in h.get("events", [])]
                    reg = None if (h is None or not h.get("reg")) else h["reg"]["oracle"]
                    print("[s2fix3 smoke] %s %s %s k%d success %s bound %s on %s | events %s | phase_max %s | reg %s" % (
                        model, arm, t["key"], k, r["success"], r["bound"], on, ev, None if h is None else h["phase_max"],
                        None if reg is None else {x: reg.get(x) for x in ("tgt_err_xy_cm", "tgt_err_z_cm", "side_ok", "mug_ok")}), flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_") and x.endswith(".json")):
        Rj = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2fix3 smoke] %s: %s" % (f, [(e["trial"], e["success"]) for e in Rj["eps"]]), flush=True)


def read(a):
    from lifelong_ood import ledger
    import s2fix2
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD, mism = SMOKE, a.dir, "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_fix3"]
        assert len(rows) == 1, "expected exactly one s2_fix3 prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD = MAIN, ROLL
    res = dict(states=ks, mapping=MAPPING, code_mismatch=mism, models={})
    nulr = {k: {m: dict(predicted=None) for m in s2.MUGS} for k in ks}
    for model, arms in RUNS.items():
        runs = {arm: load_run(RD, model, arm, ks) for arm in arms}
        R_ = dict(arms={}, deltas={}, verdicts={})
        ok = {arm: not isinstance(v, str) for arm, v in runs.items()}
        for arm, run in runs.items():
            if isinstance(run, str):
                R_["arms"][arm] = run
                print("== %s %s: %s" % (model, arm, run), flush=True)
                continue
            R_["arms"][arm] = dict(success={c: float(v.mean()) for c, v in run["succ"].items()},
                                   new=float(mat(run, s2.NEW).mean()), known=float(mat(run, s2.KNOWN).mean()),
                                   describe=s2fix2.describe(run, nulr, ks))
            print("== %s %s: per task %s | N1+N2 %.3f | K1+K2 %.3f" % (
                model, arm, {c: round(v, 3) for c, v in R_["arms"][arm]["success"].items()}, R_["arms"][arm]["new"],
                R_["arms"][arm]["known"]), flush=True)
            for c, x in R_["arms"][arm]["describe"].items():
                print("      %s %s" % (c, json.dumps(x)), flush=True)

        def dd(x, y, tasks):
            return s2.boot2_states(mat(runs[x], tasks), mat(runs[y], tasks))
        if ok.get("I"):
            kI = float(mat(runs["I"], s2.KNOWN).mean())
            R_["testable"] = dict(intact_K=kI, floor=FLOOR, TESTABLE=bool(kI >= FLOOR))
            for arm in arms:
                if arm == "I" or not ok.get(arm):
                    continue
                dN, dK = dd(arm, "I", s2.NEW), dd(arm, "I", s2.KNOWN)
                R_["deltas"]["%s-I N" % arm], R_["deltas"]["%s-I K" % arm] = dN, dK
                lab = fix_label(dN, dK)
                R_["verdicts"]["%s label (rule)" % arm] = lab
                R_["verdicts"]["%s label (registered)" % arm] = lab if kI >= FLOOR else "NOT TESTABLE (rule: %s)" % lab
                R_["verdicts"]["%s-I K non-inferiority" % arm] = ni_label(dK)
                R_["verdicts"]["SR 54 rule %s" % arm] = sr54_rule(dN, float(mat(runs[arm], s2.NEW).mean()))
                R_["deltas"]["%s-I per task" % arm] = {c: s2.boot2_states(runs[arm]["succ"][c][None], runs["I"]["succ"][c][None])
                                                       for c in s2.KNOWN + s2.NEW}
        ALLT = s2.KNOWN + s2.NEW
        if model == "pi05" and ok.get("Hp") and ok.get("H"):
            for nm, tasks in (("all", ALLT), ("N", s2.NEW), ("K", s2.KNOWN)):
                d_ = dd("Hp", "H", tasks)
                R_["deltas"]["Hp-H %s" % nm] = d_
                R_["verdicts"]["Hp-H %s (cost of perception)" % nm] = diff_label(d_, "PERCEPTION GAINS", "PERCEPTION COSTS")
            R_["deltas"]["Hp-H per task"] = {c: s2.boot2_states(runs["Hp"]["succ"][c][None], runs["H"]["succ"][c][None]) for c in ALLT}
            R_["perception"] = accuracy(runs["Hp"], ks)
            print("== pi05 perception accuracy (Hp vs simulator oracle): %s" % json.dumps(R_["perception"]), flush=True)
        for k_, v in R_["deltas"].items():
            print("   %s %s %s" % (model, k_, json.dumps(v if not isinstance(v, list) else np.round(v, 3).tolist())), flush=True)
        print("   VERDICTS %s: %s | testable %s" % (model, json.dumps(R_["verdicts"]), json.dumps(R_.get("testable"))), flush=True)
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
    print("[s2fix3 read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("states", "diag", "pcheck", "pcheck2", "pcheck3", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    if A.mode == "read" and A.out is None:
        A.out = READ
    dict(states=states, diag=diag, pcheck=pcheck, pcheck2=pcheck2, pcheck3=pcheck3, ref=ref, summary=lambda a: summary(a.dir), read=read)[A.mode](A)

"""F6O2: F6O (experiments/f6o_tasks.py; docs/SCAFFOLD_RESULTS.md 45.5) on NEW displaced-layout states, plus arm M, a structural
memory overlay.  DATA, RENDER CHECK AND OVERLAY CHECK ONLY: no policy runs here.  LIBERO venv (py3.8), EGL + the segmenter on
one GPU.  f6o_tasks.py, fx_tasks.py, fx_r.py, sc_rv.py, sc_pu.py and sc_seg.py are imported, never edited.

1. New states.  fx's F6 rule, unchanged: the familiar state k = LIBERO's sampler on F6_fam_vis.bddl with np.random.seed =
   env.seed = 810000 + k (fx BASE6); the displaced state k = that state with only the plate moved to a point drawn by
   default_rng(811000 + k) (fx DISP6) on the 0.14-0.30 m ring around its training spot, clear of the objects, fixtures and
   stove front, and settling within 2 cm (fx_tasks.main's loop, copied: fx_tasks.py writes files when run, so its loop cannot be
   called).  Before drawing anything new the rule must reproduce fx's stored F6 states k = 0-23 (fam exactly; disp to 1e-6),
   drawn in fx's order (fam 0-23, then disp 0-23).  New candidates are k = 24, 25, ... (fam all drawn first, then disp, as fx).
   A candidate is rejected (recorded with its reason) when no displaced plate is found, it duplicates any stored F6 state
   (fx fam / disp k = 0-23) or another new state (np.allclose), f6o's beside spot search finds no spot, or f6o's render check
   (below) shows any violation.  The first N_SMOKE2 accepted states are the smoke set, the next N_MAIN2 the main set.
2. Covers, park spots, beside spots: f6o's, unchanged (cover size from results/flawed/f6o/f6o_tasks.json; park spots
   f6o_tasks.park_spots with default_rng(830000 + 100 + k), f6o's disp seed rule; beside spots f6o_tasks.beside_spot).
3. Render check, per new state, through f6o_tasks.CoverEnv with DUMMY actions for WAIT + HIDE_AT + 3 REPLAN steps (f6o's): plate
   pixels at every query in both cameras; a violation is a covered plate with any pixel (c; b after the first query), an
   uncovered plate with no agentview pixel, a decoy without agentview pixels, (d) keeping < .95 of a's plate pixels at any query
   in either camera or < .95 of a's bowl pixels.
4. Arm M (condition b + MemEnv below).  Memory overlay, policy frozen:
   - at the first query (env step WAIT) the learned segmenter (sc_seg, via fx_r.segmenter; classes_of on the agentview RGB the
     policy sees, no post-processing) gives the plate class; its largest 8-connected component (>= fx_r.MIN_PX px) is the plate
     mask.  Its 3D centre is fx_r.centre (median of the one-pixel-eroded mask's points back-projected through the rendered
     depth and the camera model, sc_pu.depth_std / cam_model: fx_r's rule; the depth render is the remaining privilege, as in
     SR 40 and 45.2).  Stored: the agentview RGB at that query (each pixel outside the mask takes its nearest mask pixel's
     colour, so sampling at the mask edge does not pull in table colour), the mask, the agentview camera model, and the plate
     plane z = the centre's z.
   - from the step cover 0 moves onto the plate (env step WAIT + HIDE_AT) to the end of the episode, every observation in BOTH
     cameras gets the remembered plate painted back at its remembered 3D position: each pixel's ray (the camera's current model)
     meets the plate plane at X; X is projected into the remembered agentview camera; if that pixel is in the remembered plate
     mask, the pixel takes the remembered colour there (bilinear).  For the static agentview this is the first-query sprite at
     its own pixels; for the moving wrist camera it is the agentview sprite reprojected through the plate's plane.
   - the overlay is static (it never moves with the scene) but occlusion-aware: a pixel is painted only where the surface the
     camera now sees (the rendered depth back-projected) lies within MEM_H = 12 cm above the remembered plane, is not part of
     the robot's own gripper column (within MEM_EE = 10 cm in xy of its proprioceptive eef position, obs robot0_eef_pos, and
     higher than MEM_EE_DZ = 3 cm below it: the gripper, the arm above it and a carried bowl stay in front of the memory), and
     the segmenter does not label it bowl (a bowl set down on the covered plate stays visible).  History, all before any
     rollout: MEM_H was 3 cm in the first build, whose overlay check painted nothing, because the cover's top lies about 8 cm
     above the plate plane measured from depth (plane z .910; cover centre z .979, half-height .014); the second build used
     MEM_H 12 cm and a 10 cm SPHERE around the eef, which also removed the cover's top under a descending gripper (descent
     check: wrist IoU .89 at 15 cm, .22 at 10 cm, 0 at 6 cm), so the exclusion became the column above.  The second build's
     f6o2_tasks.json checks.M / M_descent are that sphere version; f6o2_mcheck.py re-runs the overlay check for this rule.
   It adds no information beyond the robot's own first view: a memory slot holding one image patch and one 3D plane.
   The simulator is read only for diagnostics (segmenter IoU against the true plate, centre error), never for the paint.
5. Overlay check (policy-free), every accepted state: DUMMY rollouts in a and in b + M; at every query after the move, the
   painted mask in M against the plate's true pixels in a (same robot pose: the covers do not collide, physics identical) ->
   IoU per camera.  On the first smoke states: frames of a, b and M at a later query, and a scripted descent over the plate
   (f6o_tasks.descend, no policy) to 25 / 15 / 10 / 6 cm above the cover top, a vs b vs M, with the wrist IoU per height.
-> results/flawed/f6o2/{states/f6o2_<tid>.npy, f6o2_tasks.json, branches_*.json, refsrc/*, check_M*.png}
   states rows 0-23 = fx's stored disp states (f6o's; rows match f6o's parks / beside spots), rows 24.. = accepted new states;
   the trial index of an episode = its row.  Branch seed_base = 14000000 + 1000 k (fx's rule; k = the sampler index).

    python experiments/f6o2_tasks.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fx_tasks as FT                                                      # noqa: E402
import f6o_tasks as OT                                                     # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed", "f6o2")
ST = os.path.join(OUT, "states")
META = os.path.join(OUT, "f6o2_tasks.json")
REFSRC = os.path.join(OUT, "refsrc")
SUITE = "f6o2"
N_OLD = FT.N                                                               # fx's 24 stored states (rows 0-23)
N_SMOKE2, N_MAIN2 = 4, 120
N_SHARD = 4                                                                # main branches split in 4 shards of 30
N_CAND = 170                                                               # candidates k = 24 .. 24 + N_CAND - 1
WAIT, HIDE_AT, REPLAN = OT.WAIT, OT.HIDE_AT, OT.REPLAN
PLATE = OT.PLATE
CAMS = OT.CAMS
KEYS = ("agentview_image", "robot0_eye_in_hand_image")
MEM_H, MEM_EE, MEM_EE_DZ = 0.12, 0.10, 0.03
ARMS = (("a", "a", ""), ("b", "b", ""), ("c", "c", ""), ("d", "d", ""), ("M", "b", "M"))   # (name, cover condition, arm)


# ---------------------------------------------------------------- arm M: the memory overlay
def bilinear(img, r, c):
    S = img.shape[0]
    r = np.clip(r, 0, S - 1)
    c = np.clip(c, 0, S - 1)
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    r1, c1 = np.minimum(r0 + 1, S - 1), np.minimum(c0 + 1, S - 1)
    wr, wc = (r - r0)[..., None], (c - c0)[..., None]
    return (img[r0, c0] * (1 - wr) * (1 - wc) + img[r0, c1] * (1 - wr) * wc + img[r1, c0] * wr * (1 - wc) + img[r1, c1] * wr * wc)


class Memory:
    """One episode's memory slot (module docstring, 4).  step(obs, n, covered) -> obs, painted from the move on."""

    def __init__(self, env, ckpt=""):
        import fx_r as XR
        import sc_pu as PU
        import sc_rv as RV
        self.XR, self.PU, self.RV, self.env = XR, PU, RV, env
        self.seg = XR.segmenter(ckpt)
        self.cp, self.cb = self.seg.classes.index(PLATE), self.seg.classes.index(FT.B1)
        self.E = RV.entity_tables(env)
        self.m, self.cache, self.last = None, None, None
        self.rec = dict(record=None, trace=[], seg_ckpt=self.seg.path)

    def cloud(self, cam):
        return self.XR.Front.cloud(self, cam)                              # fx_r's back-projection (uses PU, env, E only)

    def record(self, obs):
        from scipy import ndimage
        RV, XR, PU = self.RV, self.XR, self.PU
        img = RV.flip(obs[KEYS[0]])
        cls = self.seg.classes_of(img[None], [0], post_=False)[0]
        comps = XR.components(cls == self.cp, XR.MIN_PX)
        g = np.asarray(RV.seg_px(self.env, self.E, CAMS[0]))               # diagnostics only
        true = g == self.E["names"].index(PLATE)
        if not comps:
            self.rec["record"] = dict(ok=False, seg_px=int((cls == self.cp).sum()), true_px=int(true.sum()))
            return
        m = comps[0]
        P = self.cloud(CAMS[0])
        cen = XR.centre(m, P)
        _, (ri, ci) = ndimage.distance_transform_edt(~m, return_indices=True)
        f, R, c = PU.cam_model(self.env.env.sim, CAMS[0])
        self.m = dict(fill=img[ri, ci].astype(np.float32), mask=m, zp=float(cen[2]), f=float(f), R=R, c=c)
        pxy = OT.plate_xy(self.env)
        self.rec["record"] = dict(ok=True, mask_px=int(m.sum()), true_px=int(true.sum()), n_comp=len(comps),
                                  iou=float((m & true).sum() / max(1, (m | true).sum())), centre=cen.tolist(),
                                  centre_xy_err=float(np.linalg.norm(cen[:2] - pxy)), zp=float(cen[2]))

    def sprite(self, cam):
        """-> (inside [S, S] bool, colour [S, S, 3] float) in the policy's flipped image convention: the remembered plate seen
        through this camera's current model, via the plate plane."""
        M, PU = self.m, self.PU
        S = PU.SIZE
        f, R, c = PU.cam_model(self.env.env.sim, cam)
        key = (cam, tuple(np.round(c, 6)), tuple(np.round(R.ravel(), 6)))
        if self.cache is not None and self.cache[0] == key:
            return self.cache[1], self.cache[2]
        rr, jj = np.mgrid[0:S, 0:S].astype(np.float64)
        cc = (S - 1) - jj                                                  # flipped column -> raw[::-1] column (fx_r.cloud)
        d = np.stack([(cc - S / 2.0) / f, (rr - S / 2.0) / f, np.ones_like(cc)], -1) @ R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (M["zp"] - c[2]) / d[..., 2]
            X = c + t[..., None] * d
            q = (X - M["c"]) @ M["R"]                                      # remembered camera frame
            r0 = M["f"] * q[..., 1] / q[..., 2] + S / 2.0
            j0 = (S - 1) - (M["f"] * q[..., 0] / q[..., 2] + S / 2.0)
        ok = np.isfinite(r0) & np.isfinite(j0) & (t > 0) & (q[..., 2] > 0) & (r0 > -0.5) & (r0 < S - 0.5) & (j0 > -0.5) & (j0 < S - 0.5)
        ri = np.clip(np.rint(np.nan_to_num(r0)), 0, S - 1).astype(int)
        ji = np.clip(np.rint(np.nan_to_num(j0)), 0, S - 1).astype(int)
        inside = ok & M["mask"][ri, ji]
        col = bilinear(M["fill"], np.nan_to_num(r0), np.nan_to_num(j0))
        if cam == CAMS[0]:
            self.cache = (key, inside, col)                                # the agentview camera is static
        return inside, col

    def paint(self, obs):
        RV = self.RV
        imgs = np.stack([RV.flip(obs[k]) for k in KEYS])
        cls = self.seg.classes_of(imgs, [0, 1], post_=False)
        out, npx, masks = dict(obs), [], []
        for j, (cam, key) in enumerate(zip(CAMS, KEYS)):
            inside, col = self.sprite(cam)
            if not inside.any():
                npx.append(0)
                masks.append(inside)
                continue
            P = self.cloud(cam)
            ee = np.asarray(obs["robot0_eef_pos"], np.float64)
            robot = (np.linalg.norm(P[..., :2] - ee[:2], axis=-1) < MEM_EE) & (P[..., 2] > ee[2] - MEM_EE_DZ)
            pm = inside & (P[..., 2] <= self.m["zp"] + MEM_H) & ~robot & (cls[j] != self.cb)
            im = imgs[j].copy()
            im[pm] = np.clip(np.rint(col[pm]), 0, 255).astype(np.uint8)
            out[key] = RV.flip(im)
            npx.append(int(pm.sum()))
            masks.append(pm)
        self.last = masks
        return out, npx

    def step(self, obs, n, covered):
        if n == WAIT:
            self.record(obs)
        if not covered or self.m is None:
            return obs
        out, npx = self.paint(obs)
        if (n - WAIT) % REPLAN == 0:
            self.rec["trace"].append([int(n), npx[0], npx[1]])
        return out


class MemEnv(OT.CoverEnv):
    """f6o_tasks.CoverEnv in condition b, plus the memory overlay on every observation from the move on."""

    def __init__(self, env, parks, zc, states, track=False, ckpt=""):
        super().__init__(env, "b", parks, zc, states, track=track)
        self.ckpt, self.mem = ckpt, None

    def set_init_state(self, s):
        self.mem = None
        obs = super().set_init_state(s)
        self.mem = Memory(self._e, self.ckpt)
        return obs

    def step(self, a):
        r = super().step(a)
        if self.mem is None:
            return r
        return (self.mem.step(r[0], self.n, self.moved),) + tuple(r[1:])


# ---------------------------------------------------------------- new states: fx's F6 rule
def displace(env, s_fam, k, mov4, centre_p, lo, hi, fixp, sf):
    """fx_tasks.main's F6 displacement loop for state k (copied unchanged; None where fx would assert)."""
    pi = mov4.index(PLATE)
    R4 = [FT.MOVR[n] for n in mov4]
    rng = np.random.default_rng(FT.DISP6 + k)
    FT.load(env, s_fam)
    P = np.array([FT.pos(env, mov4)[n][:2] for n in mov4])
    _, q0, q1 = FT.settle(env, s_fam, mov4, steps=10)
    dz_fam = np.array([q1[n][2] - q0[n][2] for n in mov4])
    tries = 0
    for _ in range(200000):
        p = rng.uniform(lo, hi)
        if not (FT.PLATE_RING[0] <= np.linalg.norm(p - centre_p) <= FT.PLATE_RING[1]):
            continue
        Q = P.copy()
        Q[pi] = p
        if not FT.clear(p, pi, Q, R4, fixp, sf, lo, hi):
            continue
        tries += 1
        if tries > 400:
            break
        s_new = FT.edit(env, s_fam, xyz={PLATE: p})
        _, a0, a1 = FT.settle(env, s_new, mov4, steps=10)
        d = max(FT.xy_moved(a0, a1, mov4), float(np.abs(np.array([a1[n][2] - a0[n][2] for n in mov4]) - dz_fam).max()))
        if d <= FT.SETTLE_MAX and np.allclose(a0[PLATE][:2], p, atol=1e-4):
            return s_new, p
    return None, None


def render_check(env, s, park, beside, zc):
    """f6o's render check on one state (module docstring, 3) -> (violations, per-condition query px)."""
    S1 = np.asarray(s)[None]
    steps = WAIT + HIDE_AT + 3 * REPLAN
    px, bad = {}, []
    for cond in OT.CONDS:
        q, extra, _ = OT.rollout_px(env, cond, [park], zc, S1, 0, steps, beside=[beside])
        px[cond] = dict(queries=[[int(n), list(map(int, p)), bool(cv)] for n, p, cv in q], **extra)
        for n, p, cv in q:
            hidden = cond == "c" or (cond == "b" and n != WAIT)
            if hidden and max(p) > 0:
                bad.append((cond, n, "covered plate visible", p))
            if not hidden and p[0] <= 0:
                bad.append((cond, n, "plate not visible", p))
        if min(extra["cover1"]) < 0 or extra["cover1"][0] == 0:
            bad.append((cond, "decoy not visible in agentview", extra["cover1"]))
    for (n, pa, _), (_, pd, cvd) in zip(px["a"]["queries"], px["d"]["queries"]):
        r = [1.0 if a_ == 0 else d_ / a_ for a_, d_ in zip(pa, pd)]
        if min(r) < OT.D_MIN or cvd:
            bad.append(("d", n, "plate px vs a", pa, pd))
    rb = min(1.0 if a_ == 0 else d_ / a_ for a_, d_ in zip(px["a"]["bowl"], px["d"]["bowl"]))
    if rb < OT.D_MIN:
        bad.append(("d", "bowl px vs a", px["a"]["bowl"], px["d"]["bowl"]))
    return bad, px


def flipped(o):
    return [np.ascontiguousarray(o[k][::-1, ::-1]) for k in KEYS]


def m_check(env, s, park, zc, frames_at=None):
    """Arm M on one state, DUMMY actions (module docstring, 5) -> (diag, frames {n: [a, b, M] x [agent, wrist]})."""
    import sc_rv as RV
    S1 = np.asarray(s)[None]
    steps = WAIT + HIDE_AT + 3 * REPLAN
    qn = [n for n in range(WAIT + HIDE_AT, steps + 1) if (n - WAIT) % REPLAN == 0]
    true, frames = {}, {}
    for name in ("a", "b", "M"):
        ce = MemEnv(env, [park], zc, S1) if name == "M" else OT.CoverEnv(env, name, [park], zc, S1)
        o = OT.fresh(ce, S1[0])
        E = RV.entity_tables(env)
        ip = E["names"].index(PLATE)
        for n in range(1, steps + 1):
            o, _, _, _ = ce.step(FT.DUMMY)
            if n in qn and name == "a":
                true[n] = [np.asarray(RV.seg_px(env, E, c)) == ip for c in CAMS]
            if n in qn and name == "M":
                pm = ce.mem.last if ce.mem.last is not None else [np.zeros((256, 256), bool)] * 2
                for j, c in enumerate(CAMS):
                    t_ = true[n][j]
                    frames.setdefault("iou", {}).setdefault(c, []).append(
                        [int(n), int((pm[j] & t_).sum()), int((pm[j] | t_).sum()), int(t_.sum()), int(pm[j].sum())])
            if frames_at is not None and n == frames_at:
                frames.setdefault("img", {})[name] = flipped(o)
        if name == "M":
            frames["record"] = ce.mem.rec["record"]
    return frames


def descent_check(env, s, park, zc, top):
    """A scripted end-effector descent over the remembered plate position (no policy), conditions a, b, M: per height the
    images and, for M, the wrist IoU of the painted plate against a's true plate pixels."""
    import sc_rv as RV
    S1 = np.asarray(s)[None]
    out, imgs, true = {}, {}, {}
    for name in ("a", "b", "M"):
        ce = MemEnv(env, [park], zc, S1) if name == "M" else OT.CoverEnv(env, name, [park], zc, S1)
        o = OT.fresh(ce, S1[0])
        E = RV.entity_tables(env)
        ip = E["names"].index(PLATE)
        for _ in range(WAIT):
            o, _, _, _ = ce.step(FT.DUMMY)
        pxy = OT.plate_xy(env)
        rows, ims = [], []
        for h in (0.25, 0.15, 0.10, 0.06):
            err, o = OT.descend(ce, o, [pxy[0], pxy[1], top + h])
            row = dict(h=h, err=round(err, 4))
            if name == "a":
                true[h] = [np.asarray(RV.seg_px(env, E, c)) == ip for c in CAMS]
                row["true_px"] = [int(x.sum()) for x in true[h]]
            if name == "M" and ce.mem.last is not None:
                pm = ce.mem.last
                row["iou"] = [float((pm[j] & true[h][j]).sum() / max(1, (pm[j] | true[h][j]).sum())) for j in range(2)]
                row["painted"] = [int(x.sum()) for x in pm]
            rows.append(row)
            a_, w_ = flipped(o)
            ims.append(np.concatenate([a_, w_], axis=0))
        out[name] = rows
        imgs[name] = np.concatenate(ims, axis=1)
    return out, imgs


# ---------------------------------------------------------------- main
def main():
    import imageio
    for p in (META, os.path.join(ROOT, "results", "flawed", "f6o2_read_main.json"), os.path.join(OUT, "roll")):
        assert not os.path.exists(p), "exists: " + p
    os.makedirs(ST, exist_ok=True)
    os.makedirs(REFSRC, exist_ok=True)
    f6 = json.load(open(OT.META, encoding="utf-8"))
    fx = json.load(open(FT.META, encoding="utf-8"))
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    mov4 = n2["movable"]
    centre_p = np.array(n2["training"]["centre"])[mov4.index(PLATE)]
    c = n2["constraints"]
    lo, hi = np.array(c["box"][0]) - FT.BOX_EXTRA, np.array(c["box"][1]) + FT.BOX_EXTRA
    fixp, sf = np.array(c["fixture_xy"]), c["stove_front_world"]
    t_disp = next(t for t in f6["tasks"] if t["layout"] == "disp" and t["cond"] == "a")
    bddl_rel = t_disp["bddl"]
    bddl = os.path.join(ROOT, bddl_rel)
    fam0, disp0 = np.load(os.path.join(FT.ST, "fx_0.npy")), np.load(os.path.join(FT.ST, "fx_2.npy"))
    assert fx["tasks"][0]["states"] == "F6_fam" and fx["tasks"][2]["states"] == "F6_disp"
    assert np.array_equal(np.load(os.path.join(OT.ST, "f6o_%d.npy" % t_disp["tid"])), disp0), "f6o's disp states are not fx's"
    checks = {}

    # 1. the rule reproduces fx's 24 stored states, in fx's order; then the new candidates (fam all, then disp, as fx)
    env = FT.make_env(bddl)
    env.seed(0)
    fam = [FT.sampled(env, FT.BASE6 + k) for k in range(N_OLD)]
    dfam = float(np.abs(np.stack(fam) - fam0).max())
    disp = [displace(env, fam[k], k, mov4, centre_p, lo, hi, fixp, sf)[0] for k in range(N_OLD)]
    assert all(x is not None for x in disp)
    ddisp = float(np.abs(np.stack(disp) - disp0).max())
    print(f"[f6o2] reproduce fx F6 k = 0-23: fam max |diff| {dfam:.3g}, disp max |diff| {ddisp:.3g}", flush=True)
    assert dfam == 0.0 and ddisp <= 1e-6, (dfam, ddisp)
    checks["reproduce_fx_k0_23"] = dict(fam_max_abs_diff=dfam, disp_max_abs_diff=ddisp)
    ks = list(range(N_OLD, N_OLD + N_CAND))
    famN = {k: FT.sampled(env, FT.BASE6 + k) for k in ks}
    dispN, plateN = {}, {}
    for k in ks:
        dispN[k], plateN[k] = displace(env, famN[k], k, mov4, centre_p, lo, hi, fixp, sf)
    env.close()
    print(f"[f6o2] drew {len(ks)} candidates; no displaced plate for {[k for k in ks if dispN[k] is None]}", flush=True)

    # 2. covers (f6o's size), park and beside spots, render check, overlay check
    CV = f6["cover"]
    rc, zc, r_plate = CV["radius"], CV["zc"], f6["checks"]["plate_extent"]["r_xy_max"]
    OT.install(rc, CV["half_h"], CV["n"])
    env = FT.make_env(bddl)
    env.seed(0)
    assert env.env.sim.get_state().flatten().shape[0] == disp0.shape[1], "the covers changed the state vector"
    stored = np.concatenate([fam0, disp0])
    acc, rej = [], []
    for k in ks:
        if len(acc) == N_SMOKE2 + N_MAIN2:
            break
        s = dispN[k]
        if s is None:
            rej.append([k, "no displaced plate"])
            continue
        if any(np.allclose(s, o) for o in stored) or any(np.allclose(s, a_["state"]) for a_ in acc):
            rej.append([k, "duplicate"])
            continue
        try:
            park = OT.park_spots(env, s, rc, r_plate, centre_p, mov4, lo, hi, fixp, sf, np.random.default_rng(OT.PARK_SEED + 100 + k))
            bxy, bst = OT.beside_spot(env, s, rc, r_plate, mov4, lo, hi, fixp, sf, park[1], zc)
        except AssertionError as ex:
            rej.append([k, "no park / beside spot: " + str(ex)[:200]])
            continue
        bad, px = render_check(env, s, park, bxy, zc)
        if bad:
            rej.append([k, "render: " + str(bad[:3])[:300]])
            continue
        acc.append(dict(k=k, state=s, park=park, beside=bxy, beside_stats=bst, plate_xy=np.asarray(plateN[k]).tolist(),
                        shift=float(np.linalg.norm(np.asarray(plateN[k]) - centre_p)), px=px))
        print(f"[f6o2] accepted k {k} ({len(acc)}): shift {acc[-1]['shift']:.3f} beside tier {bst['tier']}", flush=True)
    print(f"[f6o2] accepted {len(acc)}, rejected {len(rej)}: {rej}", flush=True)
    assert len(acc) == N_SMOKE2 + N_MAIN2, f"only {len(acc)} states accepted"
    old_xy = np.array(fx["f6"]["plate_xy"])
    checks["new_vs_stored_min_plate_xy_dist"] = float(min(np.linalg.norm(old_xy - np.array(a_["plate_xy"]), axis=1).min() for a_ in acc))
    checks["rejected"] = rej

    # 3. arm M on every accepted state; frames on the first smoke states
    mrows, frame_rows = [], []
    for i, a_ in enumerate(acc):
        fr = m_check(env, a_["state"], a_["park"], zc, frames_at=WAIT + HIDE_AT + 2 * REPLAN if i < N_SMOKE2 else None)
        tot = {cam: [sum(x[1] for x in fr["iou"][cam]), sum(x[2] for x in fr["iou"][cam])] for cam in CAMS}
        mrows.append(dict(k=a_["k"], record=fr["record"], iou_per_query=fr["iou"],
                          iou={cam: (None if not v[1] else v[0] / v[1]) for cam, v in tot.items()}))
        if "img" in fr:
            im = fr["img"]
            frame_rows.append(np.concatenate([im["a"][0], im["b"][0], im["M"][0], im["a"][1], im["b"][1], im["M"][1]], axis=1))
        if i < 8 or i % 20 == 0:
            print(f"[f6o2] M check k {a_['k']}: record {fr['record']} IoU {mrows[-1]['iou']}", flush=True)
    imageio.imwrite(os.path.join(OUT, "check_M.png"), np.concatenate(frame_rows, axis=0))
    ious = {cam: [r["iou"][cam] for r in mrows if r["iou"][cam] is not None] for cam in CAMS}
    checks["M"] = dict(per_state=mrows, record_ok=float(np.mean([bool((r["record"] or {}).get("ok")) for r in mrows])),
                       seg_iou_median=float(np.median([r["record"]["iou"] for r in mrows if (r["record"] or {}).get("ok")])),
                       centre_xy_err_median=float(np.median([r["record"]["centre_xy_err"] for r in mrows if (r["record"] or {}).get("ok")])),
                       paint_iou_vs_a_median={cam: (float(np.median(v)) if v else None) for cam, v in ious.items()},
                       paint_iou_vs_a_min={cam: (float(np.min(v)) if v else None) for cam, v in ious.items()})
    print(f"[f6o2] M: record ok {checks['M']['record_ok']:.3f} seg IoU median {checks['M']['seg_iou_median']:.3f} centre err median "
          f"{checks['M']['centre_xy_err_median']:.4f} paint IoU vs a median {checks['M']['paint_iou_vs_a_median']} min {checks['M']['paint_iou_vs_a_min']}", flush=True)
    desc, drows = {}, []
    top = zc + CV["half_h"]
    for i in range(2):
        a_ = acc[i]
        d_, im = descent_check(env, a_["state"], a_["park"], zc, top)
        desc[str(a_["k"])] = d_
        for name in ("a", "b", "M"):
            drows.append(im[name])
        print(f"[f6o2] descent k {a_['k']}: {d_}", flush=True)
    imageio.imwrite(os.path.join(OUT, "check_M_descent.png"), np.concatenate(drows, axis=0))
    checks["M_descent"] = desc
    env.close()

    # 4. write: states (rows 0-23 fx's, then the accepted), tasks, branches, the reference's inputs
    S_all = np.concatenate([disp0, np.stack([a_["state"] for a_ in acc])])
    parks = f6["parks"]["disp"] + [a_["park"] for a_ in acc]
    beside = f6["beside"]["disp"] + [a_["beside"] for a_ in acc]
    ksamp = list(range(N_OLD)) + [a_["k"] for a_ in acc]
    rows_smoke = list(range(N_OLD, N_OLD + N_SMOKE2))
    rows_main = list(range(N_OLD + N_SMOKE2, N_OLD + N_SMOKE2 + N_MAIN2))
    tasks = []
    for name, cond, arm in ARMS:
        tid = len(tasks)
        tasks.append(dict(tid=tid, key=f"F6O2_disp_{name}", layout="disp", cond=cond, arm=arm, language=t_disp["language"],
                          bddl=bddl_rel, goal=t_disp["goal"], fx_tid=2, fx_states="F6_disp", max_steps=300))
        np.save(os.path.join(ST, f"{SUITE}_{tid}.npy"), S_all)
    br = lambda rows: [dict(id=f"k{ksamp[r]}", trial=r, seed_base=FT.SEED_BASE + 1000 * ksamp[r], at=0, variant="fx") for r in rows]   # noqa: E731
    files = {"branches_smoke.json": br(rows_smoke), "branches_main.json": br(rows_main)}
    sh = N_MAIN2 // N_SHARD
    for j in range(N_SHARD):
        files[f"branches_main_s{j}.json"] = br(rows_main[j * sh:(j + 1) * sh])
    for fn, v in files.items():
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as fh:
            json.dump(v, fh, indent=1)
    # fx_ref.py's inputs for the new states (f6o2_ref.py points fx_tasks' META / OUT / ST here): fx's task table, fx tid 2's
    # states = S_all (trial = row), the branches as fx_branches_<phase>.json
    with open(os.path.join(REFSRC, "fx_tasks.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(fx, note="copy of results/flawed/fx/fx_tasks.json for f6o2_ref.py; fx tid 2's states are f6o2's rows"), fh, indent=1)
    np.save(os.path.join(REFSRC, "fx_2.npy"), S_all)
    for ph, rows in (("smoke", rows_smoke), ("main", rows_main)):
        with open(os.path.join(REFSRC, f"fx_branches_{ph}.json"), "w", encoding="utf-8") as fh:
            json.dump(br(rows), fh, indent=1)
    for a_ in acc:
        a_.pop("state")
    meta = dict(suite=SUITE, wait=WAIT, hide_at=HIDE_AT, replan=REPLAN, tasks=tasks, mem_h=MEM_H, mem_ee=MEM_EE,
                states=dict(smoke=rows_smoke, main=rows_main, f6o_main=list(range(FT.N_SMOKE, N_OLD))),
                sampler_k=ksamp, shards=N_SHARD, cover=CV, parks=parks, beside=beside, f6o_meta=os.path.relpath(OT.META, ROOT),
                rule="fx F6: fam np.random.seed = env.seed = 810000 + k; disp default_rng(811000 + k); seed_base 14000000 + 1000 k",
                accepted=[dict(k=a_["k"], row=N_OLD + i, plate_xy=a_["plate_xy"], shift=a_["shift"], beside_stats=a_["beside_stats"])
                          for i, a_ in enumerate(acc)],
                render_px={str(N_OLD + i): a_["px"] for i, a_ in enumerate(acc)}, checks=checks)
    with open(META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[f6o2] wrote {META}: rows {S_all.shape[0]} (smoke {rows_smoke[0]}-{rows_smoke[-1]}, main {rows_main[0]}-{rows_main[-1]})", flush=True)
    print("[f6o2] done", flush=True)


if __name__ == "__main__":
    main()

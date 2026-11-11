"""U from perceived poses: the parts policy U (sc_t4's pp_sc_U checkpoints) run with the movable objects' poses estimated from
instance masks + depth + camera calibration instead of read from the simulator.  LIBERO venv (python 3.8), one GPU (EGL rendering).

What U reads (lifelong_ood/policy/parts_flow.py with U's meta: aframe 0, hscale 0, tscale on, tnear off, lcomb off, rtransport on,
nmode 3, every size 0, pgate {tw, lift}, induced roles), per policy query:
  object term (weight 1 - held): the object's position pos[oi] (hand offset ee - o, the part's start-radius test) and its yaw
      yaw[oi] (the frame Q = rz(yaw); the slot's own yaw input is yaw - yaw0 = 0).  Never the target's or another object's pose.
  target term (weight held): pos[ti], pos[oi] (with z - zrest + pos[TABLE, 2]), cyaw (the frame), d0t (tscale), rot0 / ee_rot.
      The target's yaw is never read (the frame is cyaw).  drawer: only when ti or oi is the drawer slot (K7 / N7: not in this set).
  zrest, cyaw (fa1c_common.Canon) and d0t (fa1h_roll onset_d) are stepped EVERY simulator step from the object's and the target's
  positions, so this file perceives every step after the settle steps, not only at queries.  `off` is passed but PartsFlow never
  reads it.  The gate is ProprioHeld (proprioception only).
Perception (percep arm): agentview segmentation (sc_rv.seg_px: sites shrunk, the translucent rack geom hidden) and a depth render
  under the same hiding; each movable object's cleaned mask (sc_rv.clean_mask), eroded by one pixel when that leaves >= MINPX,
  back-projected with the camera's intrinsics (fovy) and extrinsics (cam_xpos / cam_xmat with the OpenCV axis flip; the image in
  robosuite's convention raw[::-1]).  Centre = per object the calib-chosen combination of the per-axis median ('med'), the midrange
  of the 5th / 95th percentiles ('mid') and, for z, the 95th percentile ('top'), plus a per-object constant world offset.
  Mask under MINPX px: the wrist camera's estimate (its own offset) if its mask has >= MINPX px, else the last estimate, else the
  object's training-demo median position.  Not held and partly hidden (mask under occl_frac of its first-step size; calib decides
  whether and at which share): the last estimate made while it was visible.  During the hold, the held object, per object by calib
  (mean error): 'mask' (masked points with the held rows' own offset), 'rigid' (hand pose composed with the offset measured at the
  last unoccluded pre-hold step, the hand at its grasp pose, plus a constant hand-frame correction) or 'hybrid' (mask while
  unoccluded, rigid while occluded: a slip shows in the mask).  Yaw: 'const' (training-demo circular mean) for the round bowl,
  bottle and plate (not identifiable); the cream cheese may use 'pca' (principal axis of the points on the table plane + a
  constant, mod pi, the branch nearest the constant) if calib finds it better.
  Target sites: 'const' (training-demo constant; demos.npz's site slots never vary) or 'mask' (the fixture's mask centre at the
  first step + a per-site offset), chosen on calib; the table slot is constant.  The drawer opening stays simulator-read (unused here).

    python experiments/sc_pu.py calib [--quick N]       demos 0-24 of K1-K6 -> results/flawed/sc_pu_calib.json
    python experiments/sc_pu.py check [--quick N]       demos 25-49 of K1-K6 -> results/flawed/sc_pu_check.json
    python experiments/sc_pu.py roll --arm oracle|percep|ref --index i [--record DIR [--nrec 1] [--maxtry K]]
    python experiments/sc_pu.py count
roll: combos() = sc_t4's U combos (seeds 10-12 x K1-K6, N1-N6 x familiar, new) -> results/flawed/sc_pu/<arm>_s<seed>_<tid>.json.
  ref = fa1h_roll.episode itself (the existing path), for the sanity comparison.  SC_SMOKE=1: n2_branches_smoke.json (first 2,
  as sc_t4), results under sc_pu_smoke.  --record (oracle only): successful episodes as sc_rv render-format .npz files, from
  initial states neither trial list uses (none exist in n2_states today: 24 states = smoke 0-3 + main 4-23, so it refuses);
  under SC_SMOKE=1 it records from the smoke states, as a format test only.
Measurements of the demo replays are cached under FC.DATA/pu/meas (one file per task and demo).
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import sc_rv as RV                                                          # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
SMOKE = bool(os.environ.get("SC_SMOKE"))
OD = os.path.join(OUT, "sc_pu_smoke" if SMOKE else "sc_pu")
SEEDS = (10, 11, 12)
COMPS = tuple("K%d" % i for i in range(1, 7)) + tuple("N%d" % i for i in range(1, 7))
LAYOUTS = ("familiar", "new")
CAL_TASKS = tuple("K%d" % i for i in range(1, 7))
CAL_DEMOS, CHK_DEMOS = tuple(range(0, 25)), tuple(range(25, 50))
CALIB = os.path.join(OUT, "sc_pu_calib.json")
CHECK = os.path.join(OUT, "sc_pu_check.json")
MEAS = os.path.join(FC.DATA, "pu", "meas")
MAX_STEPS, WAIT, REPLAN = 300, 10, 5                                        # fa1h_roll's
SIZE = RV.SIZE                                                              # 256
AGENT, WRIST = RV.CAMS
MINPX = 20                                                                  # a mask under this many pixels is "empty"
UNOCC = 0.5                                                                 # unoccluded: >= this share of the object's first mask
EL_MIN = 1.2                                                                # pca yaw: the footprint's axis ratio must reach this
MOV = list(FC.MOVABLE)
# round footprints (bowl, bottle, plate meshes): yaw is not identifiable from geometry, so only the cream cheese box may use 'pca'
# (the quick calib picked 'pca' for the bottle on a view-shaped axis: median 1.5 deg, p90 74 deg)
SYMMETRIC = ("akita_black_bowl_1", "wine_bottle_1", "plate_1")
FIX = list(range(4, 8))                                                     # site slots on fixtures (TABLE = 8 is constant)
TABLE = 8
XY, ZS = ("med", "mid"), ("med", "mid", "top")
HOLD_MODES = ("mask", "rigid", "hybrid")                                    # the held object during the hold (calib picks one)
NST = 12                                                                    # stats: n, med 3, p5 3, p95 3, pca angle, axis ratio


def wrap(a):
    return (np.asarray(a, np.float64) + math.pi) % (2 * math.pi) - math.pi


def ckpt_path(seed):
    return os.path.join(FC.DATA, "ckpt", "pp_sc_U_s%d" % seed, "P.npz")      # sc_t4.ckpt_of("U", seed), non-smoke


def u_pgate(seed=10):
    z = np.load(ckpt_path(seed), allow_pickle=False)
    return json.loads(str(z["__meta__"]))["pgate"]


def tasks():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def combos():
    n2 = tasks()
    out = []
    for s, c, l in itertools.product(SEEDS, COMPS, LAYOUTS):
        out.append((s, next(t for t in n2["tasks"] if t["compound"] == c and t["layout"] == l)["tid"]))
    return out


# ---------------------------------------------------------------- perception (simulator renders -> point statistics)
def make_env(bddl, camobs=False, seed=7):
    """FC.fresh_env with the offscreen renderer on (same seeding, controller and physics); camobs: 256 RGB camera observations, as
    sc_rv render's OffScreenRenderEnv (for --record)."""
    from libero.libero.envs.env_wrapper import ControlEnv
    np.random.seed(seed)
    env = ControlEnv(bddl_file_name=bddl, use_camera_obs=camobs, has_renderer=False, has_offscreen_renderer=True,
                     camera_heights=SIZE, camera_widths=SIZE)
    env.seed(seed)
    return env


def depth_std(env, E, cam):
    """Metric depth (m, along the optical axis) of one camera at SIZE, with seg_px's hiding (sites shrunk, faint geoms in group 5)
    so depth and mask see the same surfaces; image rows top-down (robosuite's convention raw[::-1])."""
    sim = env.env.sim
    m = sim.model
    old_site = np.array(m.site_size, copy=True)
    hid = E["hide"]
    old_grp = np.array(m.geom_group[hid], copy=True)
    try:
        m.site_size[:, :] = 1.0e-8
        if len(hid):
            m.geom_group[hid] = 5
        _, d = sim.render(camera_name=cam, width=SIZE, height=SIZE, depth=True)
    finally:
        m.site_size[:, :] = old_site
        if len(hid):
            m.geom_group[hid] = old_grp
    ext = float(m.stat.extent)
    near, far = float(m.vis.map.znear) * ext, float(m.vis.map.zfar) * ext
    d = near / (1.0 - np.asarray(d, np.float64) * (1.0 - near / far))
    return d[::-1]


def seg_std(env, E, cam):
    """sc_rv.seg_px (flipped [::-1, ::-1]) back to robosuite's top-down convention raw[::-1]."""
    return np.asarray(RV.seg_px(env, E, cam))[:, ::-1]


def cam_model(sim, cam):
    """-> (focal px, R camera-to-world in the OpenCV convention [x right, y down, z forward], camera position)."""
    cid = sim.model.camera_name2id(cam)
    f = 0.5 * SIZE / math.tan(float(sim.model.cam_fovy[cid]) * math.pi / 360.0)
    R = np.array(sim.data.cam_xmat[cid], np.float64).reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    return f, R, np.array(sim.data.cam_xpos[cid], np.float64)


def project(p, f, R, c):
    q = R.T @ (np.asarray(p, np.float64) - c)
    return f * q[1] / q[2] + SIZE / 2.0, f * q[0] / q[2] + SIZE / 2.0       # (row, col)


def pstats(P):
    """Points [n, 3] -> NST stats."""
    s = np.zeros(NST)
    if len(P) == 0:
        return s
    lo, hi = np.percentile(P, [5, 95], axis=0)
    xy = P[:, :2] - P[:, :2].mean(0)
    S = xy.T @ xy / len(P)
    ev = np.linalg.eigvalsh(S)
    s[0] = len(P)
    s[1:4], s[4:7], s[7:10] = np.median(P, 0), lo, hi
    s[10] = 0.5 * math.atan2(2 * S[0, 1], S[0, 0] - S[1, 1])
    s[11] = math.sqrt(max(ev[1], 0.0) / max(ev[0], 1e-12))
    return s


def mask_points(seg, depth, idx, f, R, c):
    """-> (mask pixel count, points of the eroded mask)."""
    from scipy import ndimage
    m = RV.clean_mask(seg == idx)
    n = int(m.sum())
    if n == 0:
        return 0, np.zeros((0, 3))
    me = ndimage.binary_erosion(m, structure=np.ones((3, 3), bool))
    if int(me.sum()) >= MINPX:
        m = me
    r, cc = np.nonzero(m)
    d = depth[r, cc]
    q = np.stack([(cc - SIZE / 2.0) * d / f, (r - SIZE / 2.0) * d / f, d], 1)
    return n, q @ R.T + c


def measure(env, E, cam, ents):
    """Per entity name in ents: NST stats of its back-projected mask in this camera (stats[0] = the cleaned mask's pixel count)."""
    sim = env.env.sim
    seg, depth = seg_std(env, E, cam), depth_std(env, E, cam)
    f, R, c = cam_model(sim, cam)
    out = np.zeros((len(ents), NST))
    for k, nm in enumerate(ents):
        if nm not in E["names"]:
            continue
        n, P = mask_points(seg, depth, E["names"].index(nm), f, R, c)
        out[k] = pstats(P)
        out[k, 0] = n
    return out, seg


def fixture_names(E):
    return [E["slots"].get(FC.ENT[i], "") for i in FIX]


def centre(s, xy, z):
    s = np.asarray(s, np.float64)
    med, mid = s[..., 1:4], 0.5 * (s[..., 4:7] + s[..., 7:10])
    out = np.array(med if xy == "med" else mid, copy=True)
    out[..., 2] = {"med": med[..., 2], "mid": mid[..., 2], "top": s[..., 9]}[z]
    return out


# ---------------------------------------------------------------- the estimator (shared by check and the rollouts)
class Estimator:
    """Stepped once per perceived step.  step(A, wrist_fn, F, held, ee_pos, ee_rot, cmd): A [4, NST] agentview stats of the movable
    objects, wrist_fn() -> [4, NST] (called only when needed), F [4, NST] the fixture sites' stats (read at the first step only),
    held = the policy's event, cmd = the last gripper command.  -> self.pos [9, 3], self.yaw [9], self.src {object: source}."""

    def __init__(self, C, oe, sites=None, hold=None, wrist=True, occl=None):
        self.C, self.oe, self.wrist = C, oe, wrist
        self.sites = sites or C["sites"]["mode"]
        self.hold = hold or (C["obj"][oe]["hold"] if oe in C["obj"] else "mask")
        self.occl = bool(C.get("occl", False)) if occl is None else bool(occl)
        self.uocc = float(C.get("occl_frac", UNOCC))                        # the occlusion rule's threshold (the anchor's is UNOCC)
        self.pos = np.zeros((len(FC.ENT), 3))
        self.yaw = np.zeros(len(FC.ENT))
        self.last, self.lyaw, self.ref, self.lgood = {}, {}, {}, {}
        self.good, self.grasp, self.offh, self.was, self.first = None, None, None, False, True
        self.src = {}

    def _sites(self, F):
        S = self.C["sites"]
        for i in range(4, len(FC.ENT)):
            nm = FC.ENT[i]
            self.pos[i] = np.asarray(S["const"][nm], np.float64)
            self.yaw[i] = float(S["yaw"][nm])
            if self.sites == "mask" and i in FIX and F is not None:
                s = F[FIX.index(i)]
                cfg = S["mask"][nm]
                if s[0] >= MINPX:
                    self.pos[i] = centre(s, cfg["xy"], cfg["z"]) + np.asarray(cfg["off"])

    def step(self, A, wrist_fn, F, held, ee_pos, ee_rot, cmd):
        if self.first:
            self._sites(F)
            self.first = False
        ee_pos, ee_rot = np.asarray(ee_pos, np.float64), np.asarray(ee_rot, np.float64)
        onset = held and not self.was
        if onset and self.good is not None:                                 # rigid attachment, fixed at the hold's onset
            p, hand = self.good
            if hand is None:
                hand = self.grasp if self.grasp is not None else (ee_pos, ee_rot)
            self.offh = hand[1].T @ (p - hand[0])
            self.good = None                                                # consumed: a later hold needs a fresh unoccluded view
        if not held:
            self.offh = None
            if float(cmd) > 0:
                if self.grasp is None:
                    self.grasp = (ee_pos.copy(), ee_rot.copy())             # the hand's grasp pose: the close's first step
            else:
                self.grasp = None
        W = None
        for k, nm in enumerate(MOV):
            cfg, s = self.C["obj"][nm], A[k]
            if nm not in self.ref and s[0] > 0:
                self.ref[nm] = float(s[0])
            hobj = held and nm == self.oe
            if hobj:
                self.lgood.pop(nm, None)                                    # it moves with the hand: its rest estimate is stale
            unocc = s[0] >= max(MINPX, UNOCC * self.ref.get(nm, 0.0))
            vis = s[0] >= max(MINPX, self.uocc * self.ref.get(nm, 0.0))
            rig = None if self.offh is None else ee_pos + ee_rot @ (self.offh + np.asarray(cfg.get("rcorr", np.zeros(3))))
            if hobj and self.hold == "rigid" and rig is not None:
                p, src = rig, "rigid"
            elif hobj and self.hold == "hybrid" and rig is not None and not unocc:   # hybrid: rigid only while the mask is occluded
                p, src = rig, "rigid"
            elif not hobj and self.occl and not vis and nm in self.lgood:     # partly hidden, not held: the last unoccluded estimate
                p, src = self.lgood[nm], "occl"
            elif s[0] >= MINPX:
                off = cfg.get("hoff", cfg["off"]) if hobj else cfg["off"]        # the held object's own offset during the hold
                p, src = centre(s, cfg["xy"], cfg["z"]) + np.asarray(off), "agent"
            elif hobj and rig is not None:
                p, src = rig, "rigid"
            else:
                if self.wrist and W is None:
                    W = wrist_fn()
                w = W[k] if W is not None else None
                if w is not None and w[0] >= MINPX:
                    p, src = centre(w, cfg["wxy"], cfg["wz"]) + np.asarray(cfg["woff"]), "wrist"
                elif nm in self.last:
                    p, src = self.last[nm], "last"
                else:
                    p, src = np.asarray(cfg["prior"], np.float64), "prior"
            p = np.asarray(p, np.float64)
            if not hobj and src == "agent" and vis:
                self.lgood[nm] = p.copy()
            if nm == self.oe and not held and unocc:                        # the rigid anchor: this step's own masked estimate
                self.good = (centre(s, cfg["xy"], cfg["z"]) + np.asarray(cfg["off"]), (ee_pos.copy(), ee_rot.copy()) if float(cmd) > 0 else None)
            y = cfg["yaw"]
            if y["mode"] == "pca" and src == "agent" and s[11] >= EL_MIN:
                y0 = float(wrap(s[10] + y["c"]))
                yy = min((y0, float(wrap(y0 + math.pi))), key=lambda v: abs(float(wrap(v - y["const"]))))
            elif y["mode"] == "pca" and nm in self.lyaw:
                yy = self.lyaw[nm]
            else:
                yy = float(y["const"])
            self.pos[FC.ENT.index(nm)], self.yaw[FC.ENT.index(nm)] = p, yy
            self.last[nm], self.lyaw[nm], self.src[nm] = p.copy(), yy, src
        self.was = held
        return self.pos, self.yaw


# ---------------------------------------------------------------- demo replays (calib / check)
def meas_path(task, j):
    return os.path.join(MEAS, "%s_d%02d.npz" % (task, j))


def measure_demos(task, demos):
    """Replay each demo's kept states (sc_rv.noop_keep) in the task's own env, re-placed per demo (hard reset, its own seed), and
    store per kept step: agentview and wrist stats of the movable objects, the true poses, proprioception, the last command; at
    the first kept step also the fixture sites' stats and where each object's true position projects (a convention check)."""
    todo = [j for j in demos if not os.path.exists(meas_path(task, j))]
    if not todo:
        return
    raw = np.load(RV.raw_path(RV.RV_ROOT, task))
    env = make_env(RV._bddl(task))
    os.makedirs(MEAS, exist_ok=True)
    for j in todo:
        t0 = time.time()
        np.random.seed(5000 + 100 * int(task[1:]) + j)
        env.reset()
        E = RV.entity_tables(env)
        dj = FC.drawer_joint(env)
        st, ac = raw["states_%d" % j], raw["actions_%d" % j]
        keep = np.flatnonzero(RV.noop_keep(ac))
        T = len(keep)
        R = dict(A=np.zeros((T, 4, NST)), W=np.zeros((T, 4, NST)), pos=np.zeros((T, 9, 3)), yaw=np.zeros((T, 9)), ee=np.zeros((T, 3)),
                 rot=np.zeros((T, 3, 3)), grip=np.zeros((T, 2)), cmd=np.zeros(T))
        for k, t in enumerate(keep):
            obs = env.set_init_state(st[t])
            sc = FC.scene(env, obs, dj)
            R["A"][k], seg = measure(env, E, AGENT, MOV)
            R["W"][k], _ = measure(env, E, WRIST, MOV)
            R["pos"][k], R["yaw"][k], R["ee"][k], R["rot"][k], R["grip"][k] = sc["pos"], sc["yaw"], sc["ee_pos"], sc["ee_rot"], sc["grip"]
            R["cmd"][k] = -1.0 if k == 0 else float(ac[keep[k - 1], 6])
            if k == 0:
                R["F"], _ = measure(env, E, AGENT, fixture_names(E))
                f, Rc, c = cam_model(env.env.sim, AGENT)
                hit = np.zeros((4, 2))
                for i, nm in enumerate(MOV):
                    r, cc = project(sc["pos"][i], f, Rc, c)
                    ri, ci = int(round(r)), int(round(cc))
                    idx = E["names"].index(nm)
                    inside = 0 <= ri < SIZE and 0 <= ci < SIZE
                    hit[i, 0] = float(inside and seg[ri, ci] == idx)
                    ys, xs = np.nonzero(seg == idx)
                    hit[i, 1] = float(np.sqrt(((ys - r) ** 2 + (xs - cc) ** 2).min())) if len(ys) else -1.0
                R["hit"] = hit
        RV.atomic_savez(meas_path(task, j), demo=j, task=task, keep=keep, **R)
        print("[pu] measured %s demo %d: %d steps (%.0fs)" % (task, j, T, time.time() - t0), flush=True)
    env.close()


def load_demos(demos, quick=0):
    """-> list of dict per demo (measurements + task object / target slots + the ProprioHeld event per row)."""
    by_c, _ = RV.roles_of()
    pg = u_pgate()
    from lifelong_ood.policy.parts_flow import ProprioHeld
    ds = list(demos)[:quick] if quick else list(demos)
    out = []
    for c in CAL_TASKS:
        measure_demos(c, ds)
        for j in ds:
            z = dict(np.load(meas_path(c, j)))
            det = ProprioHeld(pg["tw"], pg["lift"])
            z["H"] = np.array([det.step(z["grip"][k], z["cmd"][k], z["ee"][k][2]) for k in range(len(z["cmd"]))], bool)
            z["oe"], z["te"], z["c"], z["j"] = by_c[c][0], by_c[c][1], c, j
            out.append(z)
    return out


def first_px(n):
    """The Estimator's reference size: the first nonzero mask pixel count of an episode (0 if never seen)."""
    n = np.asarray(n)
    nz = np.flatnonzero(n > 0)
    return float(n[nz[0]]) if len(nz) else 0.0


def circmean(a):
    a = np.asarray(a, np.float64)
    return float(math.atan2(np.sin(a).mean(), np.cos(a).mean()))


def site_constants():
    D = np.load(os.path.join(FC.DATA, "demos.npz"))
    pos, yaw = D["pos"], D["yaw"]
    const = {FC.ENT[i]: np.median(pos[:, i], 0).tolist() for i in range(4, 9)}
    spread = {FC.ENT[i]: float(np.abs(pos[:, i] - np.median(pos[:, i], 0)).max()) for i in range(4, 9)}
    return const, {FC.ENT[i]: circmean(yaw[:, i]) for i in range(4, 9)}, spread


def run_est(z, C, hold=None, sites=None, wrist=True, occl=None):
    """The Estimator over one replayed demo -> (pos [T, 9, 3], yaw [T, 9], src [T] dicts)."""
    e = Estimator(C, z["oe"], sites=sites, hold=hold, wrist=wrist, occl=occl)
    T = len(z["cmd"])
    P, Y, S = np.zeros((T, 9, 3)), np.zeros((T, 9)), []
    for k in range(T):
        p, y = e.step(z["A"][k], lambda k=k: z["W"][k], z["F"] if k == 0 else None, bool(z["H"][k]), z["ee"][k], z["rot"][k], z["cmd"][k])
        P[k], Y[k] = p, y
        S.append(dict(e.src))
    return P, Y, S


def err_stats(e):
    e = np.asarray(e, np.float64) * 100.0
    if len(e) == 0:
        return dict(n=0)
    return dict(n=int(len(e)), med=round(float(np.median(e)), 3), p90=round(float(np.percentile(e, 90)), 3), max=round(float(e.max()), 3))


def calib(a):
    D = load_demos(CAL_DEMOS, a.quick)
    hits = np.array([z["hit"] for z in D])
    print("[pu] convention check: true position projects onto its own mask in %.3f of first frames; median distance to the mask %.2f px" % (
        float(hits[..., 0].mean()), float(np.median(hits[..., 1]))), flush=True)
    C = dict(minpx=MINPX, unocc=UNOCC, el_min=EL_MIN, demos=list(CAL_DEMOS)[:a.quick] if a.quick else list(CAL_DEMOS), tasks=list(CAL_TASKS),
             obj={}, sites={}, fit={})
    for i, nm in enumerate(MOV):
        A = np.concatenate([z["A"][:, i] for z in D])
        W = np.concatenate([z["W"][:, i] for z in D])
        T = np.concatenate([z["pos"][:, i] for z in D])
        Yt = np.concatenate([z["yaw"][:, i] for z in D])
        free = np.concatenate([~(z["H"] & (z["oe"] == nm)) for z in D])
        cfg, fit = {}, {}
        for cam, S, pre in (("agent", A, ""), ("wrist", W, "w")):
            ok = free & (S[:, 0] >= MINPX)
            best = None
            for xy, zz in itertools.product(XY, ZS):
                c = centre(S[ok], xy, zz)
                off = np.median(T[ok] - c, 0)
                e = np.linalg.norm(T[ok] - c - off, axis=1)
                fit["%s_%s_%s" % (cam, xy, zz)] = err_stats(e)
                if len(e) and (best is None or np.median(e) < best[0]):
                    best = (float(np.median(e)), xy, zz, off)
            if best is None:
                best = (None, "med", "med", np.zeros(3))
            cfg[pre + "xy"], cfg[pre + "z"], cfg[pre + "off"] = best[1], best[2], best[3].tolist()
        cfg["prior"] = np.median(T, 0).tolist()
        # yaw: the training-demo constant, or the footprint's principal axis + a constant (mod pi)
        pre_rows = free
        const = circmean(Yt[pre_rows])
        e_const = np.abs(wrap(Yt[pre_rows] - const))
        ok = pre_rows & (A[:, 0] >= MINPX) & (A[:, 11] >= EL_MIN)
        cy = 0.5 * circmean(2.0 * (Yt[ok] - A[ok, 10])) if ok.any() else 0.0
        if ok.any():
            y0 = wrap(A[ok, 10] + cy)
            y1 = wrap(y0 + math.pi)
            yy = np.where(np.abs(wrap(y0 - const)) <= np.abs(wrap(y1 - const)), y0, y1)
            e_pca = np.abs(wrap(yy - Yt[ok]))
            e_const_ok = np.abs(wrap(Yt[ok] - const))
        else:
            e_pca = e_const_ok = np.zeros(0)
        use_pca = bool(nm not in SYMMETRIC and len(e_pca) and np.median(e_pca) < np.median(e_const_ok) and ok.sum() >= 0.5 * pre_rows.sum())
        cfg["yaw"] = dict(mode="pca" if use_pca else "const", const=const, c=cy)
        fit["yaw"] = dict(const_deg=dict(med=float(np.degrees(np.median(e_const))), p90=float(np.degrees(np.percentile(e_const, 90)))),
                          pca_rows=int(ok.sum()), pre_rows=int(pre_rows.sum()),
                          pca_deg=None if not len(e_pca) else dict(med=float(np.degrees(np.median(e_pca))), p90=float(np.degrees(np.percentile(e_pca, 90)))),
                          const_on_pca_rows_deg=None if not len(e_const_ok) else float(np.degrees(np.median(e_const_ok))),
                          axis_ratio_med=float(np.median(A[pre_rows & (A[:, 0] >= MINPX), 11])) if (pre_rows & (A[:, 0] >= MINPX)).any() else None)
        cfg["hold"] = "mask"
        C["obj"][nm], C["fit"][nm] = cfg, fit
        print("[pu] calib %s: agent %s/%s off %s  wrist %s/%s  yaw %s  (%s)" % (nm, cfg["xy"], cfg["z"], np.round(cfg["off"], 4).tolist(),
              cfg["wxy"], cfg["wz"], cfg["yaw"]["mode"], json.dumps(fit["yaw"])), flush=True)
    # sites: the training-demo constant vs the fixture's mask centre at the first step + an offset
    const, syaw, spread = site_constants()
    C["sites"] = dict(const=const, yaw=syaw, demos_npz_spread=spread, mask={})
    sfit = {}
    for k, i in enumerate(FIX):
        nm = FC.ENT[i]
        F = np.array([z["F"][k] for z in D])
        Tt = np.array([z["pos"][0, i] for z in D])
        ok = F[:, 0] >= MINPX
        sfit[nm] = dict(const=err_stats(np.linalg.norm(Tt - np.asarray(const[nm]), axis=1)))
        best = None
        for xy, zz in itertools.product(XY, ZS):
            c = centre(F[ok], xy, zz)
            off = np.median(Tt[ok] - c, 0)
            e = np.linalg.norm(Tt[ok] - c - off, axis=1)
            sfit[nm]["mask_%s_%s" % (xy, zz)] = err_stats(e)
            if len(e) and (best is None or np.median(e) < best[0]):
                best = (float(np.median(e)), xy, zz, off)
        if best is None:                                                    # never visible: the constant stands (n_ok 0)
            best = (None, "med", "med", np.zeros(3))
        C["sites"]["mask"][nm] = dict(xy=best[1], z=best[2], off=best[3].tolist(), n_ok=int(ok.sum()), n=int(len(ok)))
    tg = sorted({z["te"] for z in D} & {FC.ENT[i] for i in FIX})               # the site targets of K1-K6 decide
    mc = np.median([sfit[t]["const"]["med"] for t in tg])
    mm = np.median([min(v["med"] for kk, v in sfit[t].items() if kk.startswith("mask_")) for t in tg])
    C["sites"]["mode"] = "mask" if mm < mc else "const"
    C["sites"]["decided_on"] = dict(targets=tg, const_med_cm=float(mc), mask_med_cm=float(mm))
    C["fit"]["sites"] = sfit
    print("[pu] calib sites: mode %s (median error over the site targets %s: const %.2f cm, mask %.2f cm)" % (C["sites"]["mode"], tg, mc, mm), flush=True)
    # the held object during the hold: masked points (with the held rows' own offset) vs rigid attachment (with a constant
    # hand-frame correction: the object settles in the grasp after the anchor), each fitted on the calib demos, then compared
    for nm in sorted({z["oe"] for z in D}):
        i, cfg = FC.ENT.index(nm), C["obj"][nm]
        Dn = [z for z in D if z["oe"] == nm]
        Ah = np.concatenate([z["A"][z["H"], MOV.index(nm)] for z in Dn])
        Th = np.concatenate([z["pos"][z["H"], i] for z in Dn])
        ref = np.concatenate([np.full(int(z["H"].sum()), first_px(z["A"][:, MOV.index(nm), 0])) for z in Dn])
        okh = Ah[:, 0] >= np.maximum(MINPX, UNOCC * ref)                    # fitted on the held rows where the mask is unoccluded
        cfg["hoff"] = (np.median(Th[okh] - centre(Ah[okh], cfg["xy"], cfg["z"]), 0) if okh.any() else np.asarray(cfg["off"])).tolist()
        cfg["rcorr"] = [0.0, 0.0, 0.0]
        dh = []
        for z in Dn:
            P, _, S = run_est(z, C, hold="rigid")
            for k in np.flatnonzero(z["H"]):
                if S[k][nm] == "rigid":
                    dh.append(z["rot"][k].T @ (z["pos"][k, i] - P[k, i]))
        cfg["rcorr"] = (np.median(np.array(dh), 0) if dh else np.zeros(3)).tolist()
        res = {}
        for mode in HOLD_MODES:
            e = []
            for z in Dn:
                P, _, _ = run_est(z, C, hold=mode)
                h = z["H"]
                e.append(np.linalg.norm(P[h, i] - z["pos"][h, i], axis=1))
            e = np.concatenate(e) if e else np.zeros(0)
            res[mode] = dict(err_stats(e), mean=round(float(e.mean()) * 100, 3) if len(e) else None)
        C["obj"][nm]["hold"] = min(HOLD_MODES, key=lambda m: res[m]["mean"] if res[m]["mean"] is not None else 1e9)   # mean: tails count
        C["fit"][nm]["hold"] = res
        print("[pu] calib hold %s: %s -> %s (held-row offset %s; rigid hand-frame correction %s)" % (
            nm, json.dumps(res), C["obj"][nm]["hold"], np.round(cfg["hoff"], 4).tolist(), np.round(cfg["rcorr"], 4).tolist()), flush=True)
    # occlusion rule for objects not held: keep the last unoccluded estimate while the mask is under UNOCC of its first size, or not
    occ = {}
    for flag, frac in ((False, UNOCC), (True, 0.5), (True, 0.7), (True, 0.85)):
        e = []
        C["occl_frac"] = frac
        for z in D:
            P, _, _ = run_est(z, C, occl=flag)
            for i, nm in enumerate(MOV):
                sel = ~(z["H"] & (z["oe"] == nm))
                e.append(np.linalg.norm(P[sel, i] - z["pos"][sel, i], axis=1))
        e = np.concatenate(e)
        occ["off" if not flag else "on_%.2f" % frac] = dict(err_stats(e), mean=round(float(e.mean()) * 100, 3), flag=flag, frac=frac)
    best = min(occ.values(), key=lambda v: v["mean"])
    C["occl"], C["occl_frac"] = bool(best["flag"]), float(best["frac"])
    C["fit"]["occl"] = occ
    print("[pu] calib occlusion rule (objects not held, all rows): %s -> occl %s frac %.2f" % (json.dumps(occ), C["occl"], C["occl_frac"]), flush=True)
    C["convention"] = dict(on_mask=float(hits[..., 0].mean()), med_dist_px=float(np.median(hits[..., 1])))
    C["pgate"] = u_pgate()
    if a.quick:
        print("[pu] quick calib: not written", flush=True)
        return C
    assert not os.path.exists(CALIB), CALIB + " exists"
    tmp = CALIB + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(C, fh, indent=1)
    os.replace(tmp, CALIB)
    print("[pu] calib -> %s" % CALIB, flush=True)
    return C


def load_calib():
    return json.load(open(CALIB, encoding="utf-8"))


def check(a):
    C = load_calib() if os.path.exists(CALIB) else calib(argparse.Namespace(quick=a.quick))
    D = load_demos(CHK_DEMOS, a.quick)
    rows = {}
    canon = dict(zrest=[], cyaw=[], d0t=[], onsets=0, demos=len(D))
    alt = {}
    site = {FC.ENT[i]: dict(const=[], mask=[]) for i in FIX}
    from fa1c_common import Canon
    for z in D:
        P, Y, S = run_est(z, C)
        H = z["H"]
        first = int(np.argmax(H)) if H.any() else len(H)
        before = np.arange(len(H)) < first
        for i, nm in enumerate(MOV):
            for ph, sel in (("pre", before), ("hold", H), ("post", ~H & ~before)):
                r = rows.setdefault((nm, ph), dict(err=[], xy=[], yaw=[], small=0, n=0, src={}, role_err=[]))
                e = np.linalg.norm(P[sel, i] - z["pos"][sel, i], axis=1)
                r["err"].append(e)
                r["xy"].append(np.linalg.norm(P[sel, i, :2] - z["pos"][sel, i, :2], axis=1))
                r["yaw"].append(np.abs(wrap(Y[sel, i] - z["yaw"][sel, i])))
                r["small"] += int((z["A"][sel, i, 0] < MINPX).sum())
                r["n"] += int(sel.sum())
                for k in np.flatnonzero(sel):
                    r["src"][S[k][nm]] = r["src"].get(S[k][nm], 0) + 1
                if nm == z["oe"]:
                    r["role_err"].append(e)
                    r.setdefault("role_yaw", []).append(np.abs(wrap(Y[sel, i] - z["yaw"][sel, i])))
        # the held object during the hold under the other mode
        i = FC.ENT.index(z["oe"])
        for mode in HOLD_MODES:
            Pm, _, _ = run_est(z, C, hold=mode)
            alt.setdefault((z["oe"], mode), []).append(np.linalg.norm(Pm[H, i] - z["pos"][H, i], axis=1))
        # Canon's zrest / cyaw and the onset distance d0t, true vs estimated
        ti = FC.ENT.index(z["te"])
        ct, ce = Canon(), Canon()
        was = False
        for k in range(len(H)):
            h = bool(H[k])
            zt, cyt = ct.step(h, z["pos"][k, i], lambda k=k: (z["pos"][k, ti], float(z["pos"][k, TABLE, 2])))
            ze, cye = ce.step(h, P[k, i], lambda k=k: (P[k, ti], float(P[k, TABLE, 2])))
            if h and not was:
                canon["onsets"] += 1
                canon["zrest"].append(abs(ze - zt))
                canon["cyaw"].append(abs(float(wrap(cye - cyt))))
                canon["d0t"].append(abs(np.linalg.norm(P[k, i, :2] - P[k, ti, :2]) - np.linalg.norm(z["pos"][k, i, :2] - z["pos"][k, ti, :2])))
                po = canon.setdefault("per_object", {}).setdefault(z["oe"], dict(zrest=[], zrest_signed=[], d0t=[], cyaw=[]))
                po["zrest"].append(canon["zrest"][-1])
                po["zrest_signed"].append(ze - zt)
                po["d0t"].append(canon["d0t"][-1])
                po["cyaw"].append(canon["cyaw"][-1])
            was = h
        for k, si in enumerate(FIX):
            nm = FC.ENT[si]
            site[nm]["const"].append(np.linalg.norm(np.asarray(C["sites"]["const"][nm]) - z["pos"][0, si]))
            cfg = C["sites"]["mask"][nm]
            Fk = z["F"][k]
            if Fk[0] >= MINPX:
                site[nm]["mask"].append(np.linalg.norm(centre(Fk, cfg["xy"], cfg["z"]) + np.asarray(cfg["off"]) - z["pos"][0, si]))
    R = dict(calib=CALIB, demos=list(CHK_DEMOS)[:a.quick] if a.quick else list(CHK_DEMOS), tasks=list(CAL_TASKS), minpx=MINPX,
             objects={}, held_modes={}, canon={}, sites={}, sites_mode=C["sites"]["mode"])
    print("\n%-20s %-5s %6s | %7s %7s %7s | %7s | %8s %8s | %6s  %s" % ("object", "phase", "n", "med cm", "p90 cm", "max cm", "xy med",
          "yaw med", "yaw p90", "<20px", "sources"))
    for (nm, ph), r in sorted(rows.items(), key=lambda kv: (MOV.index(kv[0][0]), kv[0][1] != "pre")):
        e, xy, y = np.concatenate(r["err"]), np.concatenate(r["xy"]), np.concatenate(r["yaw"])
        es = err_stats(e)
        ys = dict(med=float(np.degrees(np.median(y))), p90=float(np.degrees(np.percentile(y, 90)))) if len(y) else None
        R["objects"]["%s|%s" % (nm, ph)] = dict(pos_cm=es, xy_med_cm=float(np.median(xy) * 100) if len(xy) else None, yaw_deg=ys,
                                               yaw_used=ph == "pre", small_share=r["small"] / max(1, r["n"]), sources=r["src"],
                                               as_task_object=err_stats(np.concatenate(r["role_err"])) if r["role_err"] else None,
                                               yaw_as_task_object_deg=dict(
                                                   med=float(np.degrees(np.median(np.concatenate(r["role_yaw"])))),
                                                   p90=float(np.degrees(np.percentile(np.concatenate(r["role_yaw"]), 90))),
                                                   max=float(np.degrees(np.max(np.concatenate(r["role_yaw"])))))
                                               if r.get("role_yaw") and len(np.concatenate(r["role_yaw"])) else None)
        if es["n"]:
            print("%-20s %-5s %6d | %7.2f %7.2f %7.2f | %7.2f | %8.2f %8.2f | %6.3f  %s" % (nm, ph, es["n"], es["med"], es["p90"], es["max"],
                  float(np.median(xy) * 100), ys["med"], ys["p90"], r["small"] / max(1, r["n"]), json.dumps(r["src"])))
    print("")
    for (nm, mode), e in sorted(alt.items()):
        R["held_modes"]["%s|%s" % (nm, mode)] = err_stats(np.concatenate(e))
        print("[pu] held object during the hold, %-18s %-5s %s%s" % (nm, mode, err_stats(np.concatenate(e)),
              "  <- used" if C["obj"][nm]["hold"] == mode else ""))
    R["canon"] = dict(onsets=canon["onsets"], demos=canon["demos"], zrest_cm=err_stats(canon["zrest"]), d0t_cm=err_stats(canon["d0t"]),
                      cyaw_deg=dict(med=float(np.degrees(np.median(canon["cyaw"]))), p90=float(np.degrees(np.percentile(canon["cyaw"], 90))),
                                    max=float(np.degrees(np.max(canon["cyaw"])))) if canon["cyaw"] else None)
    print("[pu] Canon at the hold's onset (%d onsets in %d demos): zrest %s  d0t %s  cyaw (deg) %s" % (
        canon["onsets"], canon["demos"], R["canon"]["zrest_cm"], R["canon"]["d0t_cm"], R["canon"]["cyaw_deg"]))
    R["canon"]["per_object"] = {}
    for nm, po in sorted(canon.get("per_object", {}).items()):
        R["canon"]["per_object"][nm] = dict(zrest_cm=err_stats(po["zrest"]), zrest_signed_med_cm=float(np.median(po["zrest_signed"]) * 100),
                                            d0t_cm=err_stats(po["d0t"]), cyaw_med_deg=float(np.degrees(np.median(po["cyaw"]))),
                                            cyaw_max_deg=float(np.degrees(np.max(po["cyaw"]))))
        print("[pu]   %-20s %s" % (nm, json.dumps(R["canon"]["per_object"][nm])))
    for nm, v in site.items():
        R["sites"][nm] = dict(const=err_stats(v["const"]), mask=err_stats(v["mask"]))
        print("[pu] site %-32s const %s  mask %s%s" % (nm, R["sites"][nm]["const"], R["sites"][nm]["mask"],
              "" if nm not in {z["te"] for z in D} else "  (a K1-K6 target)"))
    if a.quick:
        print("[pu] quick check: not written", flush=True)
        return
    assert not os.path.exists(CHECK), CHECK + " exists"
    tmp = CHECK + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    os.replace(tmp, CHECK)
    print("[pu] check -> %s" % CHECK, flush=True)


# ---------------------------------------------------------------- rollouts
def episode(model, bddl, state, lang, seed_base, arm, C=None, rec=False):
    """fa1h_roll.episode (verb None) with the policy's entity poses from a pose source: arm "oracle" the simulator (FC.scene and
    the contact body positions, as fa1h_roll), "percep" the Estimator (movable objects) and calib constants / fixture masks (sites).
    The oracle instruments (the contact Held / Canon, bound, held_step, onset_canon, min_dxy, final) stay simulator-read in both.
    rec: also keep each policy step's frame (sc_rv render's fields) -> out["_frames"]."""
    import fa1b_common as FB
    import fa1c_common as FK
    import torch
    from lifelong_ood.policy import factored_flow as FF
    assert arm in ("oracle", "percep"), arm
    percep = arm == "percep"
    lex = getattr(model, "role_lex", None)
    if lex:
        from lifelong_ood.policy.parts_flow import role_parse
        parsed = role_parse(lex, lang)
        if parsed is None:
            return dict(success=False, steps=0, calls=0, secs=0.0, error="no induced roles", bound=None, bound_step=None, final={},
                        final_grip=None, last_cmd=-1.0, drawer=None, held_step=None, held_ee=None, held_segments=0, held_end=False,
                        onset_canon=None, min_dxy_held=None, roles=None,
                        pgate=dict(event_step=None, event_segments=0, event_end=False, steps=0, agree=0, event_only=0, oracle_only=0))
        oe, te = parsed
    else:
        oe, te = FC.roles(lang)
    oi, ti = FC.ENT.index(oe), FC.ENT.index(te)
    env = make_env(bddl, camobs=rec)
    env.reset()
    obs = env.set_init_state(state)
    dj = FC.drawer_joint(env)
    con = FB.Contact(env, [oe])
    hm, cn = FB.Held(), FK.Canon()
    pg = getattr(model, "pgate", None)
    if percep:
        assert pg and not pg.get("motion"), "percep needs the proprioceptive gate (no HandMoved)"
    if pg:
        from lifelong_ood.policy.parts_flow import ProprioHeld
        det, cnp = ProprioHeld(pg["tw"], pg["lift"]), FK.Canon()
        ptr = dict(on=False, step=None, segs=0, agree=0, n=0, ev_only=0, or_only=0)
        mv = None
    E = RV.entity_tables(env) if (percep or rec) else None
    est = Estimator(C, oe) if percep else None
    diag = dict(obj_pre=[], obj_held=[], tgt=[], yaw_pre=[], src={}, small=0, n=0, wrist_renders=0) if percep else None
    frames = dict(image=[], wrist=[], seg=[], wseg=[], state=[], actions=[]) if rec else None
    t, calls, done, err, plan = 0, 0, False, None, []
    prev, bound, bound_step, start_pos, t0 = -1.0, None, None, None, time.time()
    held_step, held_ee, onset_c, segs, h_on, off, zr, cy = None, None, None, 0, False, np.zeros(3), 0.0, 0.0
    min_dxy, rot0, d0t = None, None, None

    def tgt_sim():
        p = FC.scene(env, obs, dj)["pos"]
        return p[ti], float(p[FK.TABLE, 2])

    def tgt():                                                              # the policy's target reference
        if percep:
            return est.pos[ti].copy(), float(est.pos[TABLE, 2])
        return tgt_sim()

    def opos():                                                             # the policy's object position
        return est.pos[oi].copy() if percep else con.pos(oe)

    def perceive(held):
        sc = FC.scene(env, obs, dj)
        A, _ = measure(env, E, AGENT, MOV)
        F = measure(env, E, AGENT, fixture_names(E))[0] if est.first else None

        def wfn():
            diag["wrist_renders"] += 1
            return measure(env, E, WRIST, MOV)[0]
        est.step(A, wfn, F, held, sc["ee_pos"], sc["ee_rot"], prev)
        diag["n"] += 1
        diag["small"] += int(A[MOV.index(oe), 0] < MINPX)
        diag["src"][est.src[oe]] = diag["src"].get(est.src[oe], 0) + 1
        e = float(np.linalg.norm(est.pos[oi] - sc["pos"][oi]))
        diag["obj_held" if held else "obj_pre"].append(e)
        diag["tgt"].append(float(np.linalg.norm(est.pos[ti] - sc["pos"][ti])))
        if "sites0" not in diag:
            diag["sites0"] = {FC.ENT[i]: round(float(np.linalg.norm(est.pos[i] - sc["pos"][i])) * 100, 3) for i in range(4, 9)}

    def track():
        h, o = hm.step(*con.touching(oe), con.pos(oe), obs["robot0_eef_pos"])
        z, c = cn.step(h, con.pos(oe), tgt_sim)
        return h, o, z, c

    def policy_gate(h, z, c):
        """fa1h_roll's L1 gate: the policy's (held, zrest, cyaw) from the proprioceptive event; percep perceives this step's
        poses (with the event, for the rigid hold) before Canon reads them."""
        if not pg:
            return h, z, c
        sc = FC.scene(env, obs, dj)
        hp = det.step(sc["grip"], prev, sc["ee_pos"][2])
        if pg.get("motion"):                                                # oracle only (asserted above)
            nonlocal mv
            if mv is None:
                from lifelong_ood.policy.parts_flow import HandMoved
                mv = HandMoved(sc["pos"][oi])
            hp = bool(mv.step(sc["pos"][oi], prev)) or hp
        hp = bool(hp)
        if percep:
            perceive(hp)
        zp, cp = cnp.step(hp, opos(), tgt)
        ptr["n"] += 1
        ptr["agree"] += int(hp == h)
        ptr["ev_only"] += int(hp and not h)
        ptr["or_only"] += int(h and not hp)
        if hp and not ptr["on"]:
            ptr["segs"] += 1
            if ptr["step"] is None:
                ptr["step"] = t - WAIT
        ptr["on"] = hp
        return hp, zp, cp

    def onset_d(h, was, d0):
        if h and not was:
            return float(np.linalg.norm(np.asarray(opos())[:2] - np.asarray(tgt()[0])[:2]))
        return d0 if h else None

    def onset_rot(h, was, r0):
        if h and not was:
            return FC.scene(env, obs, dj)["ee_rot"].copy()
        return r0 if h else None

    try:
        while t < MAX_STEPS + WAIT:
            if t < WAIT:
                obs, _, done, _ = env.step(FC.DUMMY)
                t += 1
                if t == WAIT:
                    h_on, off, zr, cy = track()
                    p_on, p_zr, p_cy = policy_gate(h_on, zr, cy)
                    rot0 = onset_rot(p_on, False, None)
                    d0t = onset_d(p_on, False, None)
                    ee0 = FC.scene(env, obs, dj)["ee_pos"].copy()
                continue
            if not plan:
                sc = FC.scene(env, obs, dj)
                if percep:
                    sc["pos"], sc["yaw"] = est.pos.copy(), est.yaw.copy()
                    if not p_on:
                        diag["yaw_pre"].append(abs(float(wrap(est.yaw[oi] - FC.scene(env, obs, dj)["yaw"][oi]))))
                b = FF.batch_from_scene(sc, oi, ti, prev, held=float(p_on), off=off, zrest=p_zr, cyaw=p_cy, rot0=rot0)
                if float(getattr(model, "hscale", 0.0)) > 0:
                    b["ee0"] = torch.tensor(np.asarray(ee0), dtype=torch.float32)[None]
                if bool(getattr(model, "tscale", False)) and d0t is not None:
                    b["d0t"] = torch.tensor([d0t], dtype=torch.float32)
                g = torch.Generator().manual_seed(int(seed_base) + calls)
                ch = model.sample(b, generator=g)[0].numpy()
                plan = [np.clip(x, -1.0, 1.0) for x in ch[:REPLAN]]
                calls += 1
            act = plan.pop(0)
            if rec:
                frames["image"].append(RV.flip(obs["agentview_image"]))
                frames["wrist"].append(RV.flip(obs["robot0_eye_in_hand_image"]))
                frames["seg"].append(RV.seg_px(env, E, AGENT))
                frames["wseg"].append(RV.seg_px(env, E, WRIST))
                frames["state"].append(np.concatenate((obs["robot0_eef_pos"], RV.quat2axisangle(obs["robot0_eef_quat"]),
                                                       obs["robot0_gripper_qpos"])).astype(np.float32))
                frames["actions"].append(np.asarray(act, np.float32))
            obs, _, done, _ = env.step(act.tolist())
            prev = float(act[6])
            was, p_was = h_on, p_on
            h_on, off, zr, cy = track()
            p_on, p_zr, p_cy = policy_gate(h_on, zr, cy)
            rot0 = onset_rot(p_on, p_was, rot0)
            d0t = onset_d(p_on, p_was, d0t)
            if h_on and not was:
                segs += 1
                if held_step is None:
                    held_step, held_ee = t - WAIT, np.asarray(obs["robot0_eef_pos"]).tolist()
                    pt, ztab = tgt_sim()
                    p = con.pos(oe)
                    foot = np.array([p[0], p[1], p[2] - zr + ztab])
                    c, s = math.cos(cy), math.sin(cy)
                    QT = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
                    onset_c = dict(foot=(QT @ (foot - pt)).tolist(), ee=(QT @ (np.asarray(held_ee) - pt)).tolist(), cyaw=cy, zrest=zr)
            if h_on:
                dxy = float(np.linalg.norm(con.pos(oe)[:2] - np.asarray(tgt_sim()[0])[:2]))
                min_dxy = dxy if min_dxy is None else min(min_dxy, dxy)
            if bound is None:
                mn, mp = FC.movable_positions(env)
                if start_pos is None:
                    start_pos = mp
                elif act[6] > 0:
                    disp = np.linalg.norm(mp - start_pos, axis=-1)
                    if disp.max() > 0.03:
                        bound, bound_step = mn[int(np.argmax(disp))], t - WAIT
            if done:
                break
            t += 1
    except Exception as ex:                                                 # noqa: BLE001
        err = repr(ex)
    sc = FC.scene(env, obs, dj)
    names = None if E is None else (list(E["names"]), list(E["slots"].keys()), list(E["slots"].values()))
    env.close()
    out = dict(success=bool(done), steps=t - WAIT, calls=calls, secs=time.time() - t0, error=err, bound=bound, bound_step=bound_step,
               final={e: sc["pos"][i].tolist() for i, e in enumerate(FC.ENT)}, final_grip=sc["grip"].tolist(), last_cmd=prev,
               drawer=sc["drawer"], held_step=held_step, held_ee=held_ee, held_segments=segs, held_end=bool(h_on), onset_canon=onset_c,
               min_dxy_held=min_dxy)
    if pg:
        out["pgate"] = dict(event_step=ptr["step"], event_segments=ptr["segs"], event_end=bool(ptr["on"]), steps=ptr["n"],
                            agree=ptr["agree"], event_only=ptr["ev_only"], oracle_only=ptr["or_only"])
    if lex:
        out["roles"] = [oe, te]
    out["sites_true"] = {FC.ENT[i]: sc["pos"][i].tolist() for i in range(4, 9)}
    if percep:
        out["percep"] = dict(obj_pre_cm=err_stats(diag["obj_pre"]), obj_held_cm=err_stats(diag["obj_held"]), tgt_cm=err_stats(diag["tgt"]),
                             yaw_pre_deg=dict(med=float(np.degrees(np.median(diag["yaw_pre"]))), max=float(np.degrees(np.max(diag["yaw_pre"]))))
                             if diag["yaw_pre"] else None, obj_sources=diag["src"], obj_small_share=diag["small"] / max(1, diag["n"]),
                             wrist_renders=diag["wrist_renders"], sites0_cm=diag.get("sites0"), sites_mode=est.sites, hold_mode=est.hold)
    if rec:
        out["_frames"], out["_names"] = frames, names
    return out


def roll(a):
    import torch
    from pp_p_roll import load_any
    s, tid = combos()[a.index]
    n2 = tasks()
    t = n2["tasks"][tid]
    assert t["tid"] == tid
    S = np.load(os.path.join(OUT, "n2_states", "n2_goal_%d.npy" % tid))
    if a.record:
        return record(a, s, t, S)
    fn = os.path.join(OD, "%s_s%d_%d.json" % (a.arm, s, tid))
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    m, meta = load_any(ckpt_path(s))
    C = load_calib() if a.arm == "percep" else None
    br = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        if a.arm == "ref":
            from fa1h_roll import episode as ref_episode
            e = ref_episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        else:
            e = episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"], a.arm, C=C)
        d = dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], held_step=e["held_step"], held_segments=e["held_segments"],
                 roles=e.get("roles"), bound=e["bound"], steps=e["steps"], calls=e["calls"], secs=round(e["secs"], 1), pgate=e.get("pgate"))
        for k in ("percep", "sites_true"):
            if k in e:
                d[k] = e[k]
        eps.append(d)
        print("[pu]", a.arm, s, t["compound"], t["layout"], b["trial"], json.dumps({k: v for k, v in d.items() if k != "sites_true"}), flush=True)
    os.makedirs(OD, exist_ok=True)
    res = dict(arm=a.arm, seed=s, compound=t["compound"], layout=t["layout"], tid=tid, host=os.uname().nodename, ckpt=ckpt_path(s),
               calib=CALIB if a.arm == "percep" else None, smoke=SMOKE, eps=eps)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(fn + ".tmp", fn)


def record_states(S):
    """Initial-state indices for --record: those neither trial list uses (SC_SMOKE=1: the smoke list's, a format test only)."""
    main = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    smoke = json.load(open(os.path.join(OUT, "n2_branches_smoke.json"), encoding="utf-8"))
    used = {int(b["trial"]) for b in main} | {int(b["trial"]) for b in smoke}
    if SMOKE:
        out = [(int(b["trial"]), int(b["seed_base"])) for b in smoke]
    else:
        out = [(j, 13000000 + 1000 * j) for j in range(len(S)) if j not in used]
    assert not ({j for j, _ in out} & {int(b["trial"]) for b in main}), "recording must never use the main trial list"
    return out, sorted(used)


def record(a, s, t, S):
    import torch
    from pp_p_roll import load_any
    assert a.arm == "oracle", "--record records the oracle arm only"
    todo, used = record_states(S)
    if not todo:
        print("[pu] record: no free initial states for tid %d: the file holds %d states and the trial lists use %s; refusing" % (
            t["tid"], len(S), used), flush=True)
        sys.exit(2)
    torch.set_num_threads(1)
    m, _ = load_any(ckpt_path(s))
    os.makedirs(a.record, exist_ok=True)
    got, log = 0, []
    for j, sb in todo[: a.maxtry] if a.maxtry else todo:
        fn = os.path.join(a.record, "%s_%s_s%d_st%02d.npz" % (t["compound"], t["layout"], s, j))
        if os.path.exists(fn):
            print("[pu] record exists", fn, flush=True)
            got += 1
            if got >= a.nrec:
                break
            continue
        e = episode(m, os.path.join(ROOT, t["bddl"]), S[j], t["language"], sb, "oracle", rec=True)
        log.append(dict(state=j, seed_base=sb, success=bool(e["success"]), steps=e["steps"], error=e["error"]))
        print("[pu] record tid %d state %d: success %s steps %d" % (t["tid"], j, e["success"], e["steps"]), flush=True)
        if not e["success"]:
            continue
        F = e["_frames"]
        ac = np.stack(F["actions"])
        keep = RV.noop_keep(ac)
        names, sn, se = e["_names"]
        sel = lambda k: np.stack(F[k])[keep]                                # noqa: E731
        RV.atomic_savez(fn, image=sel("image").astype(np.uint8), wrist=sel("wrist").astype(np.uint8), seg=sel("seg"), wseg=sel("wseg"),
                        state=sel("state"), actions=ac[keep].astype(np.float32), kept=np.flatnonzero(keep), raw_len=len(ac), demo=j,
                        task=t["compound"], language=t["language"], instruction=t["language"], tid=t["tid"], layout=t["layout"], seed=s,
                        seed_base=sb, names=np.array(names), slot_names=np.array(sn), slot_ents=np.array(se))
        z = np.load(fn)
        print("[pu] record -> %s: %d of %d steps kept; keys %s; image %s seg %s state %s actions %s" % (
            fn, int(keep.sum()), len(ac), sorted(z.files), z["image"].shape, z["seg"].shape, z["state"].shape, z["actions"].shape), flush=True)
        got += 1
        if got >= a.nrec:
            break
    lf = os.path.join(a.record, "%s_%s_s%d_log_%d.json" % (t["compound"], t["layout"], s, int(time.time())))
    with open(lf, "w", encoding="utf-8") as fh:
        json.dump(dict(tid=t["tid"], seed=s, attempts=log, recorded=got, smoke=SMOKE), fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("calib", "check", "roll", "count"))
    ap.add_argument("--quick", type=int, default=0, help="calib / check: only the first N demos of each range, nothing written")
    ap.add_argument("--arm", choices=("oracle", "percep", "ref"))
    ap.add_argument("--index", type=int)
    ap.add_argument("--record", default="", help="roll, arm oracle: a directory for successful episodes as demonstrations")
    ap.add_argument("--nrec", type=int, default=1, help="--record: stop after this many successes")
    ap.add_argument("--maxtry", type=int, default=0, help="--record: at most this many attempts (0: every free state)")
    a = ap.parse_args()
    if a.what == "count":
        print(len(combos()))
    elif a.what == "calib":
        calib(a)
    elif a.what == "check":
        check(a)
    else:
        roll(a)

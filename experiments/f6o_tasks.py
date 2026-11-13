"""F6O: F6 (memory) with a physical occluder in EVERY condition (docs/SCAFFOLD_RESULTS.md 45.5), DATA AND RENDER CHECK ONLY.
No policy runs here.  LIBERO venv (py3.8), EGL on one GPU.

Why: fx's F6 (SR 45, 45.1) removed the plate from render after the first query; both VLAs then failed even in the familiar layout,
so a missing-plate view (out of distribution) was confounded with missing memory.  Here an occluder is in the scene in every
condition, and only whether it covers the plate at the moment the policy acts changes.

The occluder: two identical opaque covers (a squat cylinder, radius = the plate's horizontal extent + COVER_MARGIN, top COVER_TOP
above the plate's top), added to the model as static world bodies.  Their geoms do not collide (contype = conaffinity = 0) and add
no degree of freedom, so the simulator state, the physics, the success test and the scripted references are those of fx's F6
(fx's own start states load unchanged; checked).  A non-colliding cover is needed so the bowl can still be put on a covered
plate; it is rendered in both cameras like any LIBERO visual geom (group 1).
    cover 0 (the mover)  parked on a free table spot, or over the plate
    cover 1 (the decoy)  always parked on a second free spot, so a cover's presence alone does not say where the plate is
Conditions (same states, same seeds, same instruction "put the bowl on the plate" as fx F6):
    a  visible        both covers parked for the whole episode (the plate is never covered)
    b  occluded-late  as a until env step WAIT + HIDE_AT (fx's hide step: the first query sees the plate), then cover 0 moves onto
                      the plate and stays there
    c  never-visible  cover 0 over the plate from step 0 (b's view after the move, from the start)
    d  moved-beside   as b, but at env step WAIT + HIDE_AT cover 0 moves to a spot right beside the plate that leaves the plate
                      and the bowl fully visible (b's visual event, a disc appearing mid-table next to the plate, without hiding it).
                      The spot (per state, from beside_spot): on a ring just outside the plate (centre distance = cover radius +
                      plate radius + gap, the smallest gap in BESIDE_GAPS that works), clear of the movable objects, the fixtures,
                      the stove front and the decoy (where no ring spot is, the fixture / box clearances and then every clearance
                      but the bowl's are waived, recorded as the spot's tier), with plate and bowl pixels in both cameras >= BESIDE_MIN of their value with
                      cover 0 away, and cover 0 itself seen in agentview >= half as much as on the plate; among those, the spot whose
                      distance from the bowl is closest to the plate's (b's cover sits on the plate: the same distance from the bowl).
Layouts: fam = fx's F6_fam states (the plate on its training spot), disp = fx's F6_disp states (the plate moved 0.14-0.30 m).
Park spots: per state, seeded (PARK_SEED + k), clear of every movable object, fixture and the stove front, of the plate's training
spot and of each other; checked in render.

The render check (written to results/flawed/f6o/check_*.png and f6o_tasks.json "checks"), all policy-free:
    - plate / bowl / cover segmentation pixels in agentview and wrist at every query step of a DUMMY-action rollout through
      CoverEnv (the client's own wrapper) for every condition and every state (smoke + main);
    - a scripted end-effector descent over the covered plate (OSC position servo, no policy): wrist pixels of the plate at heights
      above the cover, covered vs uncovered;
    - physics identity: the same DUMMY rollout with the covers in different places gives identical simulator states.
States 0-3 = smoke, 4-23 = main (fx's), branches = fx's.
-> results/flawed/f6o/{states/f6o_<tid>.npy, f6o_tasks.json, check_*.png}

    python experiments/f6o_tasks.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fx_tasks as FT                                                      # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed", "f6o")
ST = os.path.join(OUT, "states")
META = os.path.join(OUT, "f6o_tasks.json")
SUITE = "f6o"
N_SMOKE, N = FT.N_SMOKE, FT.N
WAIT, HIDE_AT, REPLAN = FT.WAIT, FT.HIDE_AT, 5
PARK_SEED = 830000
PARK_EXTRA = 0.15                                                         # park spots may lie this far outside the object box
NCOV = 2
COVER_MARGIN, COVER_TOP = 0.015, 0.008
MARGINS = (0.015, 0.025, 0.035, 0.045, 0.06, 0.08)                          # the radius search (first that hides the plate)
RGBA = "0.42 0.36 0.30 1"                                                  # opaque, a dull brown
OFF_Z = -2.0                                                               # the default (pre-placement) pose: below the floor
PLATE = "plate_1"
CONDS = ("a", "b", "c", "d")
BESIDE_GAPS = (0.01, 0.02, 0.03, 0.045)                                    # (d): ring gap beyond cover radius + plate radius
BESIDE_MIN = 0.99                                                          # (d): plate / bowl px kept, vs cover 0 away
D_MIN = 0.95                                                               # (d) render check: plate px vs a, every query; bowl px vs a
LAYOUTS = (("fam", "F6_fam", 0), ("disp", "F6_disp", 2))                   # (layout, fx states, fx tid whose reference applies)
CAMS = ("agentview", "robot0_eye_in_hand")


# ---------------------------------------------------------------- the covers in the model
SIZE = {}                                                                  # radius, half_h, n of the installed covers


def install(radius, half_h, n=NCOV):
    """Every LIBERO model built after this call (env construction and every hard reset) carries n cover bodies (default pose below
    the floor).  A second call changes the size for models built afterwards (the render check's radius search only)."""
    import xml.etree.ElementTree as ET
    from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
    SIZE.update(radius=float(radius), half_h=float(half_h), n=int(n))
    if getattr(BDDLBaseDomain, "_f6o_installed", False):
        return
    orig = BDDLBaseDomain._load_model

    def _load_model(self):
        orig(self)
        wb = self.model.worldbody
        for i in range(SIZE["n"]):
            b = ET.SubElement(wb, "body", name=f"f6o_cover{i}", pos=f"0 0 {OFF_Z}")
            ET.SubElement(b, "geom", name=f"f6o_cover{i}_g", type="cylinder", size=f"{SIZE['radius']:.4f} {SIZE['half_h']:.4f}",
                          rgba=RGBA, contype="0", conaffinity="0", group="1")
    BDDLBaseDomain._load_model = _load_model
    BDDLBaseDomain._f6o_installed = True


def cover_ids(env):
    m = env.env.sim.model
    return [m.body_name2id(f"f6o_cover{i}") for i in range(NCOV)]


def cover_geoms(env, i):
    m = env.env.sim.model
    return [m.geom_name2id(f"f6o_cover{i}_g")]


def set_cover(env, i, xy, zc):
    """Move cover i (a static body: its pose is model.body_pos; kinematics picks it up at the next forward / step)."""
    env.env.sim.model.body_pos[cover_ids(env)[i]] = [float(xy[0]), float(xy[1]), float(zc)]


def plate_xy(env):
    """The centre of the plate's geometry (the midpoint of its geoms' world-frame bounding boxes in xy).  Not the body origin: the
    LIBERO plate's body origin is off its visual centre (the first build centred the covers on the body and left a rim visible)."""
    m, d = env.env.sim.model, env.env.sim.data
    W = []
    for g in FT.subtree_geoms(env, PLATE):
        ctr, half = np.array(m.geom_aabb[g][:3]), np.array(m.geom_aabb[g][3:])
        R = np.array(d.geom_xmat[g]).reshape(3, 3)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + ctr
        W.append(corners @ R.T + np.array(d.geom_xpos[g]))
    W = np.concatenate(W)
    return (W[:, :2].min(0) + W[:, :2].max(0)) / 2


class CoverEnv:
    """A LIBERO env with F6O's covers placed per condition.  Used by f6o_client.py (wrapping the client's env) and by the render
    check here.  spec = dict(cond, park0, park1, zc); park0 / park1 are per-state xy lists (the state index is matched by the
    start state itself, so the client's state order cannot misalign them).  beside: per-state xy of (d)'s spot.  `moved` = cover 0
    has left its park spot; `covered` = it is on the plate (b after the move, c)."""

    def __init__(self, env, cond, parks, zc, states, track=False, beside=None):
        self._e, self.cond, self.parks, self.zc, self.states = env, cond, parks, zc, states
        self.beside = beside
        assert cond != "d" or beside is not None, "condition d needs the beside spots"
        self.n, self.moved, self.k, self.track = 0, False, None, track
        self.rows = None

    @property
    def covered(self):
        return self.moved and self.cond in ("b", "c")

    def __getattr__(self, k):
        return getattr(self._e, k)

    def _which(self, s):
        d = np.abs(self.states - np.asarray(s)[None]).max(1)
        k = int(np.argmin(d))
        assert d[k] < 1e-9, f"start state not in the task's states (min diff {d[k]})"
        return k

    def place_start(self, s):
        """Covers for the episode's start (before the inner set_init_state renders its observation)."""
        k = self._which(s)
        self.k = k
        self._e.env.sim.set_state_from_flattened(np.asarray(s))
        self._e.env.sim.forward()
        p0, p1 = self.parks[k]
        set_cover(self._e, 0, plate_xy(self._e) if self.cond == "c" else p0, self.zc)
        set_cover(self._e, 1, p1, self.zc)
        self.n, self.moved = 0, self.cond == "c"

    def set_init_state(self, s):
        self.place_start(s)
        obs = self._e.set_init_state(s)
        if self.track:
            self.rows = dict(k=self.k, covered=[], px=[])
        return obs

    def plate_px(self):
        pg = FT.subtree_geoms(self._e, PLATE)
        return [FT.pixels(self._e, pg, c) for c in CAMS]

    def step(self, a):
        self.n += 1
        if self.cond in ("b", "d") and not self.moved and self.n >= WAIT + HIDE_AT:
            set_cover(self._e, 0, plate_xy(self._e) if self.cond == "b" else self.beside[self.k], self.zc)
            self.moved = True
        r = self._e.step(a)
        if self.rows is not None:
            self.rows["covered"].append(self.covered)
            q = self.n >= WAIT and (self.n - WAIT) % REPLAN == 0             # this observation feeds a policy query
            self.rows["px"].append(self.plate_px() if q else [-1, -1])
        return r


# ---------------------------------------------------------------- geometry
def plate_extent(env):
    """-> (horizontal radius about the plate body, z bottom, z top) from the plate's geom bounding boxes in world frame."""
    m, d = env.env.sim.model, env.env.sim.data
    c = plate_xy(env)
    r, zs = 0.0, []
    for g in FT.subtree_geoms(env, PLATE):
        ctr, half = np.array(m.geom_aabb[g][:3]), np.array(m.geom_aabb[g][3:])
        R = np.array(d.geom_xmat[g]).reshape(3, 3)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + ctr
        W = corners @ R.T + np.array(d.geom_xpos[g])
        zs += [W[:, 2].min(), W[:, 2].max()]
        gc = R @ ctr + np.array(d.geom_xpos[g])
        r = max(r, float(np.linalg.norm(gc[:2] - c)) + float(max(half[0], half[1])))
    return r, float(min(zs)), float(max(zs))


def park_spots(env, s, rc, r_plate, centre_plate, mov, lo, hi, fixp, sf, rng):
    """Two free xy spots for the covers on the state s: in the object box widened by PARK_EXTRA (visibility is checked in render),
    clear of the objects (fx's footprint radii; the plate's measured radius for the plate and its training spot)."""
    FT.load(env, s)
    P = FT.pos(env, mov)
    occ = [(plate_xy(env) if n == PLATE else P[n][:2], r_plate if n == PLATE else FT.MOVR[n]) for n in mov] + \
          [(np.asarray(centre_plate), r_plate + 0.03)]
    out = []
    for _ in range(200000):
        p = rng.uniform(lo - PARK_EXTRA, hi + PARK_EXTRA)
        ok = all(np.linalg.norm(p - q) >= rc + rq + FT.GAP for q, rq in occ)
        ok = ok and all(np.linalg.norm(p - q) >= 2 * rc + FT.GAP for q in out)
        ok = ok and all(np.linalg.norm(p - fixp[f]) >= rc + FT.FIXR[fn] + FT.GAP for f, fn in enumerate(FT.FIXR))
        cx, cy = np.clip(p[0], sf[0], sf[2]), np.clip(p[1], sf[1], sf[3])
        ok = ok and np.hypot(p[0] - cx, p[1] - cy) >= rc + FT.GAP
        if ok:
            out.append(p)
            if len(out) == NCOV:
                return [q.tolist() for q in out]
    raise AssertionError("no park spots")


def beside_spot(env, s, rc, r_plate, mov, lo, hi, fixp, sf, park1, zc):
    """(d)'s spot for cover 0 on the state s (see the module docstring) -> (xy, stats).  Measured at the start state with the decoy
    on its park spot, each cover placement followed by one DUMMY env step (a cover moved with sim.forward() alone is not picked up
    by the segmentation render: the first build of (d) measured that way passed spots that hid a third of the plate); the full
    rollout check in main() repeats it at every query."""
    FT.load(env, s)
    P = FT.pos(env, mov)
    pc, bowl = plate_xy(env), P[FT.B1][:2]
    d_plate = float(np.linalg.norm(bowl - pc))
    pg, bg, c0 = FT.subtree_geoms(env, PLATE), FT.subtree_geoms(env, FT.B1), cover_geoms(env, 0)
    set_cover(env, 1, park1, zc)

    def put0(xy, z):
        set_cover(env, 0, xy, z)
        env.step(FT.DUMMY)

    put0(pc, zc)
    cov_on_plate = FT.pixels(env, c0, "agentview")
    meas = lambda: [FT.pixels(env, g, c) for g in (pg, bg) for c in CAMS]    # noqa: E731  plate agent, wrist; bowl agent, wrist
    occ_b = [(bowl, FT.MOVR[FT.B1])]
    occ_o = [(P[n][:2], FT.MOVR[n]) for n in mov if n not in (PLATE, FT.B1)] + [(np.asarray(park1), rc)]
    tried, why = 0, {}
    # tiers: 0 = every clearance (bowl, other objects + decoy, fixtures, stove front, table box); 1 = fixtures / stove front / box
    # waived; 2 = only the bowl's clearance.  The render criteria (plate, bowl, cover 0 visible) hold in every tier.
    for tier in (0, 1, 2):
        for gap in BESIDE_GAPS:
            R = rc + r_plate + gap
            cands = []
            for ang in range(0, 360, 5):
                p = pc + R * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
                checks = [("bowl", all(np.linalg.norm(p - q) >= rc + rq + FT.GAP for q, rq in occ_b))]
                if tier <= 1:
                    checks.append(("objects", all(np.linalg.norm(p - q) >= rc + rq + FT.GAP for q, rq in occ_o)))
                if tier == 0:
                    cx, cy = np.clip(p[0], sf[0], sf[2]), np.clip(p[1], sf[1], sf[3])
                    checks += [("box", not (np.any(p < lo - PARK_EXTRA) or np.any(p > hi + PARK_EXTRA))),
                               ("fixtures", all(np.linalg.norm(p - fixp[f]) >= rc + FT.FIXR[fn] + FT.GAP for f, fn in enumerate(FT.FIXR))),
                               ("stove_front", np.hypot(p[0] - cx, p[1] - cy) >= rc + FT.GAP)]
                fail = [nm for nm, ok in checks if not ok]
                if fail:
                    why[f"t{tier}_{fail[0]}"] = why.get(f"t{tier}_{fail[0]}", 0) + 1
                    continue
                cands.append((abs(float(np.linalg.norm(p - bowl)) - d_plate), ang, p))
            for _, ang, p in sorted(cands, key=lambda x: (x[0], x[1])):
                tried += 1
                put0((0.0, 0.0), OFF_Z)
                base = meas()
                put0(p, zc)
                now = meas()
                ratio = [1.0 if b == 0 else n / b for n, b in zip(now, base)]
                cv = FT.pixels(env, c0, "agentview")
                if min(ratio) >= BESIDE_MIN and cv >= 0.5 * cov_on_plate:
                    seg = pc - bowl
                    tpar = float(np.clip(np.dot(p - bowl, seg) / max(1e-9, np.dot(seg, seg)), 0, 1))
                    return p.tolist(), dict(tier=tier, gap=gap, angle=ang, tried=tried, d_bowl_plate=d_plate,
                                            d_bowl_cover=float(np.linalg.norm(p - bowl)), d_cover_path=float(np.linalg.norm(p - (bowl + tpar * seg))),
                                            d_plate_centre=float(R), px_ratio_plate_bowl=[round(x, 3) for x in ratio], cover0_px_agent=[cv, cov_on_plate])
                why[f"t{tier}_render"] = why.get(f"t{tier}_render", 0) + 1
    raise AssertionError(f"no beside spot (tried {tried}; rejections {why})")


def rollout_px(env, cond, parks, zc, S, k, steps, beside=None):
    """DUMMY actions through CoverEnv: -> per query step [agent, wrist] plate pixels, covered flags, bowl / cover px at the end."""
    ce = CoverEnv(env, cond, parks, zc, S, track=True, beside=beside)
    fresh(ce, S[k])
    bowl = FT.subtree_geoms(env, FT.B1)
    for _ in range(steps):
        ce.step(FT.DUMMY)
    q = [(i + 1, px, cv) for i, (px, cv) in enumerate(zip(ce.rows["px"], ce.rows["covered"])) if px[0] >= 0]
    extra = dict(bowl=[FT.pixels(env, bowl, c) for c in CAMS], cover0=[FT.pixels(env, cover_geoms(env, 0), c) for c in CAMS],
                 cover1=[FT.pixels(env, cover_geoms(env, 1), c) for c in CAMS])
    return q, extra, env.env.sim.get_state().flatten()


def fresh(ce, s):
    """reset + set_init_state with the same seeding every time (the solver warm start depends on the reset's placement)."""
    np.random.seed(0)
    ce._e.seed(0)
    ce.reset()
    return ce.set_init_state(s)


def descend(env, obs, target, steps=150):
    """Drive the end effector to `target` with OSC delta actions (a proportional servo, gripper open); -> (error m, obs)."""
    err = None
    for _ in range(steps):
        e = np.asarray(target) - np.asarray(obs["robot0_eef_pos"])
        err = float(np.linalg.norm(e))
        if err < 0.01:
            break
        obs, _, _, _ = env.step(np.concatenate([np.clip(e * 20.0, -1, 1), [0, 0, 0, -1.0]]).tolist())
    return err, obs


def main():
    import imageio
    os.makedirs(ST, exist_ok=True)
    fx = json.load(open(FT.META, encoding="utf-8"))
    assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "f6o_read_main.json")), "main read exists"
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    mov4 = n2["movable"]
    centre = np.array(n2["training"]["centre"])[mov4.index(PLATE)]
    c = n2["constraints"]
    lo, hi = np.array(c["box"][0]) - FT.BOX_EXTRA, np.array(c["box"][1]) + FT.BOX_EXTRA
    fixp, sf = np.array(c["fixture_xy"]), c["stove_front_world"]
    bddl_rel = next(t["bddl"] for t in fx["tasks"] if t["key"] == "F6_fam_vis")
    bddl = os.path.join(ROOT, bddl_rel)
    S = {lay: np.load(os.path.join(FT.ST, f"fx_{fxtid}.npy")) for lay, _, fxtid in LAYOUTS}
    for lay, sname, fxtid in LAYOUTS:
        assert fx["tasks"][fxtid]["states"] == sname and fx["tasks"][fxtid]["key"] == f"F6_{lay}_vis"
    checks = {}

    # 1. the plate's size, measured without covers, fixes the cover size
    env = FT.make_env(bddl)
    env.seed(0)
    sd0 = env.env.sim.get_state().flatten().shape[0]
    ext = []
    for lay in ("fam", "disp"):
        for k in range(N):
            FT.load(env, S[lay][k])
            ext.append(plate_extent(env))
    env.close()
    ext = np.array(ext)
    r_plate, z_bot, z_top = float(ext[:, 0].max()), float(np.median(ext[:, 1])), float(np.median(ext[:, 2]))
    rc = r_plate + COVER_MARGIN
    half_h = (z_top + COVER_TOP - z_bot) / 2
    zc = z_bot + half_h - 0.002
    checks["plate_extent"] = dict(r_xy_max=r_plate, z_bottom=z_bot, z_top=z_top, z_spread=[float(np.ptp(ext[:, 1])), float(np.ptp(ext[:, 2]))])
    print(f"[f6o] plate r {r_plate:.4f} z [{z_bot:.4f}, {z_top:.4f}] -> cover r {rc:.4f} half-height {half_h:.4f} centre z {zc:.4f}", flush=True)

    # 2. with covers: same state size; park spots
    install(rc, half_h)
    env = FT.make_env(bddl)
    env.seed(0)
    assert env.env.sim.get_state().flatten().shape[0] == sd0 == S["fam"].shape[1], "the covers changed the state vector"
    m = env.env.sim.model
    for i in range(NCOV):
        g = cover_geoms(env, i)[0]
        assert int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0
    # 2b. the radius: the smallest margin over the measured plate radius at which the covered plate has no pixel in either camera
    #     at the first query, on every state of both layouts (the bounding-box radius alone left a rim visible in the first build)
    far = [[[5.0, 5.0], [5.0, 6.0]]] * N
    search = []
    for margin in MARGINS:
        install(r_plate + margin, half_h)
        worst = 0
        for lay, _, _ in LAYOUTS:
            for k in range(N):
                q, _, _ = rollout_px(env, "c", far, zc, S[lay], k, WAIT)
                worst = max(worst, max(q[0][1]))
        search.append([margin, worst])
        print(f"[f6o] radius search: margin {margin:.3f} -> max covered plate px {worst}", flush=True)
        if worst == 0:
            break
    checks["radius_search"] = search
    assert search[-1][1] == 0, f"no cover radius hides the plate: {search}"
    rc = r_plate + search[-1][0]
    print(f"[f6o] cover radius {rc:.4f}", flush=True)
    parks = {}
    for lay, _, _ in LAYOUTS:
        parks[lay] = [park_spots(env, S[lay][k], rc, r_plate, centre, mov4, lo, hi, fixp, sf, np.random.default_rng(PARK_SEED + 100 * (lay == "disp") + k))
                      for k in range(N)]
    beside, bstats = {}, {}
    for lay, _, _ in LAYOUTS:
        bs = [beside_spot(env, S[lay][k], rc, r_plate, mov4, lo, hi, fixp, sf, parks[lay][k][1], zc) for k in range(N)]
        beside[lay], bstats[lay] = [b[0] for b in bs], [b[1] for b in bs]
        print(f"[f6o] beside {lay}: tiers {[b[1]['tier'] for b in bs]} gaps {[b[1]['gap'] for b in bs]} |bowl-cover| - |bowl-plate| "
              f"{[round(b[1]['d_bowl_cover'] - b[1]['d_bowl_plate'], 3) for b in bs]} cover-path {[round(b[1]['d_cover_path'], 3) for b in bs]}", flush=True)
    checks["beside"] = bstats

    # 3. render check through CoverEnv, every state, every condition
    steps = WAIT + HIDE_AT + 3 * REPLAN
    px, rows, bad = {}, [], []
    for lay, _, _ in LAYOUTS:
        for cond in CONDS:
            per = []
            for k in range(N):
                q, extra, fin = rollout_px(env, cond, parks[lay], zc, S[lay], k, steps, beside=beside[lay])
                per.append(dict(k=k, queries=[[int(n), list(map(int, p)), bool(cv)] for n, p, cv in q], **extra))
                for n, p, cv in q:
                    first = n == WAIT
                    hidden_expected = cond == "c" or (cond == "b" and not first)
                    if hidden_expected and max(p) > 0:
                        bad.append((lay, cond, k, n, p))
                    if not hidden_expected and p[0] <= 0:
                        bad.append((lay, cond, k, n, p))
                if min(extra["cover1"]) < 0 or extra["cover1"][0] == 0:
                    bad.append((lay, cond, k, "decoy not visible in agentview", extra["cover1"]))
            px[f"{lay}_{cond}"] = per
            print(f"[f6o] {lay} {cond}: plate px [agent, wrist] at the queries of state 0: {per[0]['queries']}; bowl px {per[0]['bowl']}", flush=True)
    # the covers must not hide the bowl: bowl px (agentview) in c vs a, per state
    ratio = {lay: [px[f"{lay}_c"][k]["bowl"][0] / max(1, px[f"{lay}_a"][k]["bowl"][0]) for k in range(N)] for lay, _, _ in LAYOUTS}
    checks["bowl_px_agentview_ratio_c_over_a"] = {lay: [round(x, 3) for x in v] for lay, v in ratio.items()}
    # (d): the plate keeps >= D_MIN of a's pixels at every query in both cameras, and the bowl (both cameras, after the move) too
    dpl, dbo = {}, {}
    for lay, _, _ in LAYOUTS:
        wp, wb = [], []
        for k in range(N):
            qa, qd = px[f"{lay}_a"][k]["queries"], px[f"{lay}_d"][k]["queries"]
            assert [x[0] for x in qa] == [x[0] for x in qd]
            for (n, pa, _), (_, pd, cvd) in zip(qa, qd):
                r = [1.0 if a_ == 0 else d_ / a_ for a_, d_ in zip(pa, pd)]
                wp.append(min(r))
                if min(r) < D_MIN or cvd:
                    bad.append((lay, "d", k, n, "plate px vs a", pa, pd))
            ba, bd = px[f"{lay}_a"][k]["bowl"], px[f"{lay}_d"][k]["bowl"]
            rb = min(1.0 if a_ == 0 else d_ / a_ for a_, d_ in zip(ba, bd))
            wb.append(round(rb, 3))
            if rb < D_MIN:
                bad.append((lay, "d", k, "bowl px vs a", ba, bd))
        dpl[lay], dbo[lay] = round(float(min(wp)), 3), wb
    checks["d_min_plate_px_ratio_vs_a_any_query_any_cam"] = dpl
    checks["d_bowl_px_ratio_vs_a_min_cam"] = dbo
    checks["bowl_px_agentview_ratio_b_over_a"] = {lay: [round(px[f"{lay}_b"][k]["bowl"][0] / max(1, px[f"{lay}_a"][k]["bowl"][0]), 3)
                                                        for k in range(N)] for lay, _, _ in LAYOUTS}
    checks["cover0_px_agentview_end_b_vs_d"] = {lay: [[px[f"{lay}_b"][k]["cover0"][0], px[f"{lay}_d"][k]["cover0"][0]] for k in range(N)]
                                                for lay, _, _ in LAYOUTS}
    # the parked covers must not hide the plate in a: plate px with covers parked vs covers off-scene
    noc = {}
    for lay, _, _ in LAYOUTS:
        v = []
        for k in range(N):
            FT.load(env, S[lay][k])                                          # each placement followed by a DUMMY step (see beside_spot)
            for i in range(NCOV):
                set_cover(env, i, (0.0, 0.0), OFF_Z)
            env.step(FT.DUMMY)
            pg = FT.subtree_geoms(env, PLATE)
            off = FT.pixels(env, pg, "agentview")
            set_cover(env, 0, parks[lay][k][0], zc)
            set_cover(env, 1, parks[lay][k][1], zc)
            env.step(FT.DUMMY)
            v.append(round(FT.pixels(env, pg, "agentview") / max(1, off), 3))
        noc[lay] = v
    checks["plate_px_agentview_ratio_parked_over_no_covers"] = noc
    checks["render_violations"] = [list(map(str, b)) for b in bad]
    checks["query_px"] = px

    # 4. physics identity: a/c rollouts end in the same simulator state (covers anywhere)
    ident = []
    for lay, _, _ in LAYOUTS:
        for k in range(N_SMOKE):
            _, _, fa = rollout_px(env, "a", parks[lay], zc, S[lay], k, steps)
            _, _, fc = rollout_px(env, "c", parks[lay], zc, S[lay], k, steps)
            _, _, fd = rollout_px(env, "d", parks[lay], zc, S[lay], k, steps, beside=beside[lay])
            ident.append(float(max(np.abs(fa - fc).max(), np.abs(fa - fd).max())))
    checks["physics_max_abs_state_diff_a_vs_c"] = max(ident)

    # 5. images: per layout, state 0 and 1: a (agent, wrist), b at query 1 and 2, c
    for lay, _, _ in LAYOUTS:
        for k in (0, 1):
            row = []
            for cond, n_steps in (("a", WAIT), ("b", WAIT), ("b", WAIT + HIDE_AT), ("c", WAIT), ("d", WAIT + HIDE_AT)):
                ce = CoverEnv(env, cond, parks[lay], zc, S[lay], beside=beside[lay])
                o = fresh(ce, S[lay][k])
                for _ in range(n_steps):
                    o, _, _, _ = ce.step(FT.DUMMY)
                row.append(np.concatenate([o["agentview_image"][::-1, ::-1], o["robot0_eye_in_hand_image"][::-1, ::-1]], axis=1))
            rows.append(np.concatenate(row, axis=1))
        imageio.imwrite(os.path.join(OUT, f"check_{lay}.png"), np.concatenate(rows[-2:], axis=0))

    # 6. scripted descent over the plate (no policy): wrist plate px at heights above the cover, covered (c) vs parked (a)
    desc, drows = {}, []
    top = zc + half_h
    for lay, _, _ in LAYOUTS:
        for k in range(N_SMOKE):
            for cond in ("a", "c"):
                ce = CoverEnv(env, cond, parks[lay], zc, S[lay])
                o = fresh(ce, S[lay][k])
                for _ in range(WAIT):
                    o, _, _, _ = ce.step(FT.DUMMY)
                pxy = plate_xy(env)
                row, imgs = [], []
                for h in (0.25, 0.15, 0.10, 0.06):
                    err, o = descend(ce, o, [pxy[0], pxy[1], top + h])
                    row.append(dict(h_above_cover=h, err=round(err, 4), px=ce.plate_px(), cam_z=float(env.env.sim.data.cam_xpos[env.env.sim.model.camera_name2id("robot0_eye_in_hand")][2]) - top))
                    if k == 0:
                        imgs.append(np.concatenate([o["agentview_image"][::-1, ::-1], o["robot0_eye_in_hand_image"][::-1, ::-1]], axis=0))
                desc[f"{lay}_{cond}_{k}"] = row
                if k == 0:
                    drows.append(np.concatenate(imgs, axis=1))
                print(f"[f6o] descent {lay} {cond} state {k}: {row}", flush=True)
    imageio.imwrite(os.path.join(OUT, "check_descent.png"), np.concatenate(drows, axis=0))
    checks["descent"] = desc
    env.close()

    # write
    tasks = []
    for lay, sname, fxtid in LAYOUTS:
        for cond in CONDS:
            tid = len(tasks)
            tasks.append(dict(tid=tid, key=f"F6O_{lay}_{cond}", layout=lay, cond=cond, language="put the bowl on the plate",
                              bddl=bddl_rel, goal=fx["tasks"][fxtid]["goal"], fx_tid=fxtid, fx_states=sname, max_steps=300))
            np.save(os.path.join(ST, f"{SUITE}_{tid}.npy"), S[lay])
    meta = dict(suite=SUITE, wait=WAIT, hide_at=HIDE_AT, replan=REPLAN, tasks=tasks,
                states=dict(smoke=list(range(N_SMOKE)), main=list(range(N_SMOKE, N))),
                branches=dict(smoke=os.path.join("results", "flawed", "fx", "fx_branches_smoke.json"),
                              main=os.path.join("results", "flawed", "fx", "fx_branches_main.json")),
                cover=dict(radius=rc, half_h=half_h, zc=zc, n=NCOV, rgba=RGBA, margin=rc - r_plate, top_above_plate=COVER_TOP),
                parks=parks, beside=beside, plate_training_xy=centre.tolist(), park_seed=PARK_SEED, fx_meta=os.path.relpath(FT.META, ROOT), checks=checks)
    with open(META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[f6o] violations {len(bad)}: {bad[:20]}", flush=True)
    print(f"[f6o] bowl px ratio c/a {checks['bowl_px_agentview_ratio_c_over_a']}", flush=True)
    print(f"[f6o] d: min plate px ratio vs a (any query, any cam) {dpl}; bowl px ratio vs a {dbo}", flush=True)
    print(f"[f6o] cover 0 agentview px at the end, [b, d] {checks['cover0_px_agentview_end_b_vs_d']}", flush=True)
    print(f"[f6o] plate px ratio parked/no covers {noc}", flush=True)
    print(f"[f6o] physics identity max |diff| {checks['physics_max_abs_state_diff_a_vs_c']}", flush=True)
    print("[f6o] done", flush=True)


if __name__ == "__main__":
    main()

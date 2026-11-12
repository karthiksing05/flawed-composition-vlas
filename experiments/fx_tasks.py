"""FX diagnostics (docs/PLAN_ARCH_FAILURES.md section 4 item 5), DATA ONLY: tasks, BDDLs, start states and branch files for
F6 (memory) and F7 (relational and negated language).  No policy runs here.  LIBERO venv (py3.8), EGL on one GPU.

F6, memory: "put the bowl on the plate" (LIBERO-Goal K1, its own instruction in every condition) in LIBERO-Goal's scene.
  The needed fact is where the plate is.  FAMILIAR states = LIBERO's sampler (the plate on its training spot, where a habit
  suffices); DISPLACED = the same state with only the plate moved 0.14-0.30 m from its training spot (so a bowl dropped on the
  habitual spot misses it).  The fact is removed from the input by moving every geom of the plate to an unrendered geom group in
  BOTH cameras: physics, contacts and the success test are untouched; only pixels change.
    vis   the plate is rendered throughout (control)
    hid   the plate is rendered in every frame before policy step HIDE_AT (= the first query only), then never again
    abs   the plate is never rendered (no-information floor; no verdict)
  fam_vis / fam_hid check that hiding by itself (a missing plate in the image) does not break the VLA when no memory is needed.

F7, relational and negated language: LIBERO-Goal's scene with a SECOND identical akita black bowl.  The referent is always
  akita_black_bowl_1 (the goal atom names it); the two relational instructions of a family refer to the two bowls of ONE scene,
  built by swapping the two bowls' full poses, so the images are identical across the flip and only the instruction differs (an
  instruction-blind policy is right on exactly one of the two).
    SP  both bowls on the table, left / right (left = +y = the left of the policy's image and the robot's left); target the plate
        relL "put the left bowl on the plate"   relR "put the right bowl on the plate"
        uniL / uniR "put the bowl on the plate" with the other bowl removed (unique name; control)
        amb  "put the bowl on the plate" with both bowls (which one the policy takes unprompted; no verdict)
    NG  one bowl on the plate, one on the table at the bowl's training spot; target the stove
        pos "put the bowl that is on the plate on the stove"   neg "put the bowl that is not on the plate on the stove"
        uniP / uniT "put the bowl on the stove" with the other bowl removed;  amb "put the bowl on the stove" (both bowls)
  "Removed" = the other bowl at rest on the floor under the tabletop (out of reach; nothing else disturbed) and hidden from both
  cameras from the first frame (the client's hide, hide_at = 0).  A copy into LIBERO-Goal's one-bowl BDDL was tried first and
  rejected: the copied bowl did not stay put in that env (it slid or was thrown off the table), cause not established.
F7P, LIBERO-PRO reuse: LIBERO-PRO's only negated task (libero_spatial_task, "... not between the plate and the ramekin ...",
  goal akita_black_bowl_2 on the plate) on LIBERO-PRO's own init states 0-23, against the trained LIBERO-Spatial instruction on the
  SAME states (referent akita_black_bowl_1).  Both run in openpi's LIBERO package from the BDDL files on disk (read only).
  notmin = the trained instruction with "not" inserted; notpro = LIBERO-PRO's own wording.  The trained layout always went with
  the trained instruction, so this arm cannot separate negation from habit; it is reported with no verdict.

States 0-3 are the smoke set, 4-23 the main set.  Every task's episode k uses noise seed_base 14000000 + 1000 k.
-> results/flawed/fx/{bddl/*.bddl, states/fx_<tid>.npy, fx_tasks.json, fx_branches_{smoke,main}.json, check_*.png}

    python experiments/fx_tasks.py
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "results", "flawed", "fx")
BD, ST = os.path.join(OUT, "bddl"), os.path.join(OUT, "states")
META = os.path.join(OUT, "fx_tasks.json")
LIB = "/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/bddl_files"
TEMPLATE = os.path.join(LIB, "libero_goal", "put_the_cream_cheese_in_the_bowl.bddl")     # N2's template (same scene, same regions)
PRO_NAME = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
PRO_ROOT = "/coc/flash10/ksingara3/robot/ext/LIBERO-PRO/libero/libero"
PRO_BDDL = os.path.join(PRO_ROOT, "bddl_files", "libero_spatial_task", PRO_NAME + ".bddl")
PRO_INIT = os.path.join(PRO_ROOT, "init_files", "libero_spatial_task", PRO_NAME + ".pruned_init")
SPA_BDDL = os.path.join(LIB, "libero_spatial", PRO_NAME + ".bddl")
SUITE, N_SMOKE, N_MAIN = "fx", 4, 20
N = N_SMOKE + N_MAIN
BASE6, DISP6, BASE7, LAY7, SEED_BASE = 810000, 811000, 820000, 821000, 14000000
WAIT, HIDE_AT = 10, 5
DUMMY = [0.0] * 6 + [-1.0]
MOVR = {"akita_black_bowl_1": 0.06, "akita_black_bowl_2": 0.06, "cream_cheese_1": 0.045, "wine_bottle_1": 0.045, "plate_1": 0.10}
FIXR = {"wooden_cabinet_1": 0.13, "flat_stove_1": 0.11, "wine_rack_1": 0.10}        # fc_n2_tasks's footprints
GAP, SETTLE_MAX, BOX_EXTRA = 0.02, 0.02, 0.03
PLATE_RING = (0.14, 0.30)                     # F6: the displaced plate's distance from its training spot
SP_SEP = (0.17, 0.22)                         # F7 SP: the two bowls' left-right separation
FAR_XY = [(0.0, 0.0), (0.1, 0.1), (-0.1, -0.1), (0.1, -0.1), (0.0, 1.0), (0.0, -1.0), (0.0, 1.6), (1.6, 0.0)]   # under the tabletop first
GEOM = 5                                      # mujoco mjOBJ_GEOM (segmentation)
HIDDEN_GROUP = 5                              # a geom group the renderer does not draw (libero_client.background)
B1, B2 = "akita_black_bowl_1", "akita_black_bowl_2"


# ---------------------------------------------------------------- shared with fx_client.py / fx_ref.py
def subtree_geoms(env, name):
    """Every geom of the body `name` and of its descendants."""
    inner = env.env
    m = inner.sim.model
    root = int(inner.obj_body_id[name])
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        while b != root and b != 0 and int(m.body_parentid[b]) != b:
            b = int(m.body_parentid[b])
        if b == root:
            out.append(g)
    return out


def hide_geoms(env, name):
    """Stop rendering object `name` in every camera.  geom_group is read only by the renderer (collisions use contype /
    conaffinity), so physics and the success test do not change.  A hard reset rebuilds the model, which undoes it."""
    g = subtree_geoms(env, name)
    env.env.sim.model.geom_group[g] = HIDDEN_GROUP
    return g


# ---------------------------------------------------------------- builders
def make_env(path):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=path, camera_heights=256, camera_widths=256)


def bddl_of(text, lang, atom, obj):
    import fc_n2_tasks as N2
    return N2.bddl_of(text, lang, atom, obj)


def two_bowl(text):
    """The LIBERO-Goal template with akita_black_bowl_2 (LIBERO-Spatial's object) added on a free spot of the table."""
    reg = ("    (:regions\n      (bowl_two_region\n          (:target main_table)\n          (:ranges (\n"
           "              (0.09 0.17 0.11 0.19)\n            )\n          )\n      )\n")
    t = re.sub(r"\s*\(:regions\n", "\n" + reg, text, count=1)
    t = t.replace(f"{B1} - akita_black_bowl", f"{B1} {B2} - akita_black_bowl", 1)
    t = re.sub(rf"(\(On {B1} main_table_akita_black_bowl_region\))", rf"\1\n    (On {B2} main_table_bowl_two_region)", t, count=1)
    assert "bowl_two_region" in t and f"{B1} {B2} - akita_black_bowl" in t and f"(On {B2} main_table_bowl_two_region)" in t
    return t


def qadr(env, n):
    m = env.env.sim.model
    return int(m.jnt_qposadr[m.body_jntadr[env.env.obj_body_id[n]]])


def dadr(env, n):
    m = env.env.sim.model
    return int(m.jnt_dofadr[m.body_jntadr[env.env.obj_body_id[n]]])


def sampled(env, seed):
    np.random.seed(seed)
    env.seed(seed)
    env.reset()
    return env.env.sim.get_state().flatten()


def load(env, s):
    env.reset()
    return env.set_init_state(s)


def pos(env, names):
    sim, inner = env.env.sim, env.env
    return {n: np.array(sim.data.body_xpos[inner.obj_body_id[n]]) for n in names}


def edit(env, s, xyz=None, swap=None):
    """-> a new flattened state: bodies moved ({name: (x, y) or (x, y, z)}) and / or two bodies' full free-joint poses and
    velocities exchanged."""
    load(env, s)
    sim = env.env.sim
    for n, p in (xyz or {}).items():
        a = qadr(env, n)
        sim.data.qpos[a:a + len(p)] = p
        d = dadr(env, n)
        sim.data.qvel[d:d + 6] = 0.0
    if swap:
        a, b = (qadr(env, n) for n in swap)
        da, db = (dadr(env, n) for n in swap)
        qa, qb = sim.data.qpos[a:a + 7].copy(), sim.data.qpos[b:b + 7].copy()
        va, vb = sim.data.qvel[da:da + 6].copy(), sim.data.qvel[db:db + 6].copy()
        sim.data.qpos[a:a + 7], sim.data.qpos[b:b + 7] = qb, qa
        sim.data.qvel[da:da + 6], sim.data.qvel[db:db + 6] = vb, va
    sim.forward()
    return sim.get_state().flatten()


def settle(env, s, names, steps=20):
    """-> (settled state, start positions, settled positions)."""
    load(env, s)
    p0 = pos(env, names)
    for _ in range(steps):
        env.step(DUMMY)
    return env.env.sim.get_state().flatten(), p0, pos(env, names)


def seated(env, s, names, zrest, tol=0.01):
    """After 20 settle steps from s: every body in `names` upright (body z axis within ~18 deg of vertical) and at its resting
    height on the table (zrest[name] +- tol)."""
    s2, _, p1 = settle(env, s, names)
    inner = env.env
    up = [float(np.asarray(inner.sim.data.body_xmat[inner.obj_body_id[n]]).reshape(3, 3)[2, 2]) for n in names]
    return all(u > 0.95 for u in up) and all(abs(p1[n][2] - zrest[n]) < tol for n in names)


def xy_moved(p0, p1, names):
    return max(float(np.linalg.norm(p1[n][:2] - p0[n][:2])) for n in names)


def view(env, s, hide=None, steps=WAIT):
    """The policy's two images (flipped as the client flips them) after the client's settle steps."""
    load(env, s)
    if hide:
        hide_geoms(env, hide)
    obs = None
    for _ in range(steps):
        obs, _, _, _ = env.step(DUMMY)
    return (np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))


def pixels(env, geoms, cam):
    s = env.env.sim.render(camera_name=cam, width=256, height=256, depth=False, segmentation=True)
    ids = np.where(s[..., 0] == GEOM, s[..., 1], -1)
    return int(np.isin(ids, geoms).sum())


def clear(p, i, P, R, fixp, sf, lo, hi, among=None):
    """Is body i at p clear of the other bodies P (with radii R; only those in `among` if given), the fixtures, the
    stove-front region and inside the box?"""
    r = R[i]
    if np.any(p < lo) or np.any(p > hi):
        return False
    for j in (range(len(P)) if among is None else among):
        if j != i and np.linalg.norm(p - P[j]) < r + R[j] + GAP:
            return False
    for f, fn in enumerate(FIXR):
        if np.linalg.norm(p - fixp[f]) < r + FIXR[fn] + GAP:
            return False
    cx, cy = np.clip(p[0], sf[0], sf[2]), np.clip(p[1], sf[1], sf[3])
    return np.hypot(p[0] - cx, p[1] - cy) >= r + GAP


def far_spot(env, s, name, others):
    """Move `name` to rest on the floor where neither camera sees it -> (state, xy).  Checked: no pixel of it in the policy's
    two views after the settle steps, at rest, and no other object disturbed."""
    m = env.env.sim.model
    fl = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").lower().endswith("floor")]
    load(env, s)
    zf = float(env.env.sim.data.geom_xpos[fl[0]][2]) if fl else 0.0
    geoms = None
    for xy in FAR_XY:
        s1 = edit(env, s, xyz={name: (xy[0], xy[1], zf + 0.06)})
        s2, p0, p1 = settle(env, s1, [name] + others, steps=40)
        load(env, s2)
        z0 = pos(env, [name])[name][2]
        for _ in range(WAIT):
            env.step(DUMMY)
        geoms = subtree_geoms(env, name)
        still = abs(pos(env, [name])[name][2] - z0) < 1e-3 and pos(env, [name])[name][2] > zf - 0.05
        mv = xy_moved(p0, p1, others)
        if still and mv < 0.005:                       # visibility does not matter: the client hides it (checked in main)
            return s2, xy
        print(f"[fx] far spot {xy} rejected: z {pos(env, [name])[name][2]:.3f} still={still} px agent/wrist "
              f"{pixels(env, geoms, 'agentview')}/{pixels(env, geoms, 'robot0_eye_in_hand')} others moved {mv:.4f}", flush=True)
    raise AssertionError(f"no far spot for {name} (floor z {zf}, floor geoms {fl})")


def write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main():
    import imageio
    os.makedirs(BD, exist_ok=True)
    os.makedirs(ST, exist_ok=True)
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    centre = np.array(n2["training"]["centre"])                                # world xy of bowl, cheese, bottle, plate
    mov4 = n2["movable"]
    c = n2["constraints"]
    lo, hi = np.array(c["box"][0]) - BOX_EXTRA, np.array(c["box"][1]) + BOX_EXTRA
    fixp, sf = np.array(c["fixture_xy"]), c["stove_front_world"]
    text = open(TEMPLATE, encoding="utf-8").read()
    tasks, states, checks = [], {}, {}

    def add(key, diag, family, cond, lang, bddl, atom, referent, distractor, sname, hide=None, hide_at=None, ref=None, max_steps=300):
        g = atom.split()
        tasks.append(dict(tid=len(tasks), key=key, diag=diag, family=family, cond=cond, language=lang, bddl=bddl, goal=g,
                          object=g[1], target=g[2], referent=referent, distractor=distractor, states=sname, hide=hide,
                          hide_at=hide_at, ref=ref, max_steps=max_steps))

    # ------------------------------------------------------------ F6
    lang6, atom6 = "put the bowl on the plate", f"On {B1} plate_1"
    t6 = bddl_of(text, lang6, atom6, B1)
    conds6 = [("fam", "vis", None), ("fam", "hid", HIDE_AT), ("disp", "vis", None), ("disp", "hid", HIDE_AT), ("disp", "abs", 0)]
    for lay, cond, ha in conds6:
        p = os.path.join(BD, f"F6_{lay}_{cond}.bddl")
        write(p, t6)
        ref = dict(kp="K1", freeze=True) if cond == "vis" else None      # hid / abs share vis's states and physics
        add(f"F6_{lay}_{cond}", "F6", lay, cond, lang6, os.path.relpath(p, ROOT), atom6, B1, None, f"F6_{lay}",
            hide="plate_1" if ha is not None else None, hide_at=ha, ref=ref)
    env = make_env(os.path.join(BD, "F6_fam_vis.bddl"))
    env.seed(0)
    fam = np.stack([sampled(env, BASE6 + k) for k in range(N)])
    disp, plate_xy, plate_shift = [], [], []
    pi = mov4.index("plate_1")
    R4 = [MOVR[n] for n in mov4]
    for k in range(N):
        rng = np.random.default_rng(DISP6 + k)
        load(env, fam[k])
        P = np.array([pos(env, mov4)[n][:2] for n in mov4])
        _, q0, q1 = settle(env, fam[k], mov4, steps=10)
        dz_fam = np.array([q1[n][2] - q0[n][2] for n in mov4])
        done, tries, last = False, 0, None
        for _ in range(200000):
            p = rng.uniform(lo, hi)
            if not (PLATE_RING[0] <= np.linalg.norm(p - centre[pi]) <= PLATE_RING[1]):
                continue
            Q = P.copy()
            Q[pi] = p
            if not clear(p, pi, Q, R4, fixp, sf, lo, hi):
                continue
            tries += 1
            if tries > 400:
                break
            s_new = edit(env, fam[k], xyz={"plate_1": p})
            _, a0, a1 = settle(env, s_new, mov4, steps=10)
            d = max(xy_moved(a0, a1, mov4), float(np.abs(np.array([a1[n][2] - a0[n][2] for n in mov4]) - dz_fam).max()))
            last = d
            if d <= SETTLE_MAX and np.allclose(a0["plate_1"][:2], p, atol=1e-4):
                disp.append(s_new)
                plate_xy.append(p.tolist())
                plate_shift.append(float(np.linalg.norm(p - centre[pi])))
                done = True
                break
        assert done, f"F6 state {k}: no displaced plate ({tries} geometric candidates, last settle {last})"
    states["F6_fam"], states["F6_disp"] = fam, np.stack(disp)
    print(f"[fx] F6: plate shifts {np.round(plate_shift, 3).tolist()}", flush=True)
    # renders: what the policy sees in each condition (state 0, and after the hide)
    rows = []
    for lay in ("fam", "disp"):
        S = states[f"F6_{lay}"]
        a, w = view(env, S[0])
        ah, wh = view(env, S[0], hide="plate_1")
        rows.append(np.concatenate([a, w, ah, wh], axis=1))
    imageio.imwrite(os.path.join(OUT, "check_F6.png"), np.concatenate(rows, axis=0))
    load(env, fam[0])
    pg = subtree_geoms(env, "plate_1")
    vis_px = pixels(env, pg, "agentview")
    hide_geoms(env, "plate_1")
    env.env.sim.forward()
    hid_px = pixels(env, pg, "agentview")
    checks["F6_plate_pixels_agentview_visible_hidden"] = [vis_px, hid_px]
    m = env.env.sim.model
    checks["F6_plate_geoms"] = [dict(geom=int(g), contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g])) for g in pg]
    env.close()
    assert vis_px > 0 and hid_px == 0, (vis_px, hid_px)

    # ------------------------------------------------------------ F7 (two bowls in LIBERO-Goal's scene)
    t2 = two_bowl(text)
    F7 = [("SP", "relL", "put the left bowl on the plate", "SP_L"), ("SP", "relR", "put the right bowl on the plate", "SP_R"),
          ("SP", "uniL", "put the bowl on the plate", "SP_UL"), ("SP", "uniR", "put the bowl on the plate", "SP_UR"),
          ("SP", "amb", "put the bowl on the plate", "SP_L"),
          ("NG", "pos", "put the bowl that is on the plate on the stove", "NG_P"),
          ("NG", "neg", "put the bowl that is not on the plate on the stove", "NG_T"),
          ("NG", "uniP", "put the bowl on the stove", "NG_UP"), ("NG", "uniT", "put the bowl on the stove", "NG_UT"),
          ("NG", "amb", "put the bowl on the stove", "NG_T")]
    for fam_, cond, lang, sname in F7:
        tgt = "plate_1" if fam_ == "SP" else "flat_stove_1_cook_region"
        atom = f"On {B1} {tgt}"
        p = os.path.join(BD, f"F7_{fam_}_{cond}.bddl")
        write(p, bddl_of(t2, lang, atom, B1))
        ref = dict(kp="K1" if fam_ == "SP" else "K2", freeze=False) if cond != "amb" else None
        uni = cond.startswith("uni")
        add(f"F7_{fam_}_{cond}", "F7", fam_, cond, lang, os.path.relpath(p, ROOT), atom, B1, None if uni else B2, f"F7_{sname}",
            hide=B2 if uni else None, hide_at=0 if uni else None, ref=ref)
    env = make_env(os.path.join(BD, "F7_SP_relL.bddl"))
    env.seed(0)
    mov5 = [B1, B2, "cream_cheese_1", "wine_bottle_1", "plate_1"]
    others5 = [n for n in mov5 if n != B2]

    def one_bowl(s2):
        """akita_black_bowl_2 taken out of the scene: at rest on the floor under the tabletop (out of reach), and hidden from both
        cameras from the first frame by the client (task hide = bowl_2, hide_at = 0)."""
        s, xy = far_spot(env, s2, B2, others5)
        return s
    R5 = [MOVR[n] for n in mov5]
    S7 = {k_: [] for k_ in ("SP_L", "SP_R", "SP_UL", "SP_UR", "NG_P", "NG_T", "NG_UP", "NG_UT")}
    lay7 = []
    for k in range(N):
        base = sampled(env, BASE7 + k)
        load(env, base)
        P0 = pos(env, mov5)
        _, _, rest = settle(env, base, mov5)
        zrest = {B1: rest[B1][2], B2: rest[B1][2]}                          # a bowl resting on the table (the sampler's bowl_1)
        rng = np.random.default_rng(LAY7 + k)
        # SP: the two bowls side by side; the cheese and the bottle stay where the sampler put them unless they are in the way
        done, n_try = False, 0
        for _ in range(100000):
            x0, y0, sep = rng.uniform(-0.16, -0.10), rng.uniform(-0.02, 0.04), rng.uniform(*SP_SEP)
            L = np.array([x0 + rng.uniform(-0.01, 0.01), y0 + sep / 2])
            Rr = np.array([x0 + rng.uniform(-0.01, 0.01), y0 - sep / 2])
            P = np.array([L, Rr] + [P0[n][:2] for n in mov5[2:]])
            # the plate stays on its sampled (training) spot, which N2's conservative fixture radii would reject; only the
            # bodies placed here are checked
            ok = all(clear(P[i], i, P, R5, fixp, sf, lo, hi, among=(0, 1, 4)) for i in (0, 1))
            if not ok:
                continue
            for i in (2, 3):                                               # relocate an object in the way, inside the box
                tries = 0
                while not clear(P[i], i, P, R5, fixp, sf, lo, hi) and tries < 2000:
                    P[i] = rng.uniform(lo, hi)
                    tries += 1
            if not all(clear(P[i], i, P, R5, fixp, sf, lo, hi) for i in range(4)):
                continue
            n_try += 1
            if n_try > 80:
                break
            s1 = edit(env, base, xyz={n: P[i] for i, n in enumerate(mov5)})
            sL, a0, a1 = settle(env, s1, mov5)
            if xy_moved(a0, a1, mov5) > SETTLE_MAX or not seated(env, sL, [B1, B2], zrest):
                continue
            sL, a2, a3 = settle(env, sL, mov5)                              # at rest: a second settle moves nothing > 5 mm
            if xy_moved(a2, a3, mov5) > 0.005:
                continue
            sR = edit(env, sL, swap=(B1, B2))
            _, c0, c1 = settle(env, sR, mov5)
            if xy_moved(c0, c1, mov5) <= SETTLE_MAX and seated(env, sR, [B1, B2], zrest):
                done = True
                break
        assert done, f"F7 state {k}: no SP layout ({n_try} physical tries; zrest {zrest})"
        sUL, sUR = one_bowl(sL), one_bowl(sR)
        # NG: bowl_1 on the plate, bowl_2 on the bowl's training spot (the sampler's bowl_1 position); everything else as sampled
        pp = P0["plate_1"]
        s1 = edit(env, base, xyz={B1: (pp[0], pp[1], P0[B1][2] + 0.03), B2: (P0[B1][0], P0[B1][1], P0[B1][2])})
        sP, b0, b1 = settle(env, s1, mov5, steps=40)
        on_plate = float(np.linalg.norm(b1[B1][:2] - b1["plate_1"][:2]))
        assert on_plate < 0.03 and b1[B1][2] > b1[B2][2] + 0.005 and xy_moved(b0, b1, [B2, "cream_cheese_1", "wine_bottle_1", "plate_1"]) <= SETTLE_MAX, \
            f"F7 state {k}: bowl not seated on the plate ({on_plate:.3f} m, z {b1[B1][2]:.3f} vs {b1[B2][2]:.3f})"
        sP, _, _ = settle(env, sP, mov5)                                    # a second settle, so the state is at rest
        sT = edit(env, sP, swap=(B1, B2))
        sUP, sUT = one_bowl(sP), one_bowl(sT)
        for k_, v in (("SP_L", sL), ("SP_R", sR), ("SP_UL", sUL), ("SP_UR", sUR), ("NG_P", sP), ("NG_T", sT), ("NG_UP", sUP), ("NG_UT", sUT)):
            S7[k_].append(v)
        load(env, sL)
        q = pos(env, mov5)
        lay7.append(dict(SP={n: q[n].tolist() for n in mov5}, NG={n: b1[n].tolist() for n in mov5},
                         sep=float(abs(q[B1][1] - q[B2][1]))))
        print(f"[fx] F7 state {k}: SP sep {lay7[-1]['sep']:.3f}, L {np.round(q[B1][:2], 3).tolist()} R {np.round(q[B2][:2], 3).tolist()}", flush=True)
    for k_, v in S7.items():
        states[f"F7_{k_}"] = np.stack(v)
    # the flip pairs give the policy the same images; the removed bowl is out of view
    diffs, rows = {}, []
    for a_, b_ in (("SP_L", "SP_R"), ("NG_P", "NG_T")):
        d = []
        for k in range(N_SMOKE):
            ia, wa = view(env, states[f"F7_{a_}"][k])
            ib, wb = view(env, states[f"F7_{b_}"][k])
            d.append(float(max(np.abs(ia.astype(int) - ib.astype(int)).mean(), np.abs(wa.astype(int) - wb.astype(int)).mean())))
            if k == 0:
                rows.append(np.concatenate([ia, wa, ib, wb], axis=1))
        diffs[f"{a_}_vs_{b_}"] = d
    upx = []
    for u in ("SP_UL", "SP_UR", "NG_UP", "NG_UT"):
        ia, wa = view(env, states[f"F7_{u}"][0], hide=B2)
        g2 = subtree_geoms(env, B2)
        upx.append([pixels(env, g2, "agentview"), pixels(env, g2, "robot0_eye_in_hand")])
        rows.append(np.concatenate([ia, wa, ia * 0, wa * 0], axis=1))
    checks["F7_uni_bowl2_pixels_hidden"] = upx
    assert all(p == [0, 0] for p in upx), upx
    checks["F7_flip_image_mean_abs_diff_smoke"] = diffs
    imageio.imwrite(os.path.join(OUT, "check_F7.png"), np.concatenate(rows, axis=0))
    env.close()
    print(f"[fx] F7 flip image differences (mean |px|, 0-255): {diffs}", flush=True)

    # ------------------------------------------------------------ F7P (LIBERO-PRO's negated task; files read in place)
    import torch
    init = np.asarray(torch.load(PRO_INIT))[:N]
    for cond, lang, bddl, atom, ref_obj, other in (
            ("trained", "pick up the black bowl between the plate and the ramekin and place it on the plate", SPA_BDDL, f"On {B1} plate_1", B1, B2),
            ("notmin", "pick up the black bowl not between the plate and the ramekin and place it on the plate", PRO_BDDL, f"On {B2} plate_1", B2, B1),
            ("notpro", "Pick the akita black bowl not between the plate and the ramekin and place it on the plate", PRO_BDDL, f"On {B2} plate_1", B2, B1)):
        add(f"F7P_{cond}", "F7P", "PRO", cond, lang, bddl, atom, ref_obj, other, "F7P", ref=dict(kp="K1", freeze=False) if cond != "notpro" else None,
            max_steps=220)
    gtxt = open(PRO_BDDL, encoding="utf-8").read()
    assert f"(On {B2} plate_1)" in gtxt, "LIBERO-PRO's goal changed"
    rows = []
    for p in (SPA_BDDL, PRO_BDDL):
        e = make_env(p)
        e.seed(0)
        sd = e.env.sim.get_state().flatten().shape[0]
        assert init.shape[1] == sd, (p, init.shape, sd)
        mv = list(e.env.objects_dict.keys())
        moved = []
        for k in range(N):
            _, a0, a1 = settle(e, init[k], mv, steps=WAIT)
            moved.append(xy_moved(a0, a1, mv))
        checks[f"F7P_settle_max_xy_{os.path.basename(os.path.dirname(p))}"] = max(moved)
        ia, wa = view(e, init[0])
        rows.append(np.concatenate([ia, wa], axis=1))
        e.close()
    imageio.imwrite(os.path.join(OUT, "check_F7P.png"), np.concatenate(rows, axis=0))
    states["F7P"] = init

    # ------------------------------------------------------------ write
    for t in tasks:
        np.save(os.path.join(ST, f"{SUITE}_{t['tid']}.npy"), states[t["states"]])
    br = {ph: [dict(id=f"k{k}", trial=k, seed_base=SEED_BASE + 1000 * k, at=0, variant="fx") for k in ks]
          for ph, ks in (("smoke", range(N_SMOKE)), ("main", range(N_SMOKE, N)))}
    for ph, v in br.items():
        with open(os.path.join(OUT, f"fx_branches_{ph}.json"), "w", encoding="utf-8") as fh:
            json.dump(v, fh, indent=1)
    meta = dict(suite=SUITE, wait=WAIT, hide_at=HIDE_AT, tasks=tasks, states=dict(smoke=list(range(N_SMOKE)), main=list(range(N_SMOKE, N))),
                seeds=dict(f6_sampler=BASE6, f6_plate=DISP6, f7_sampler=BASE7, f7_layout=LAY7, seed_base=SEED_BASE),
                f6=dict(plate_training_xy=centre[pi].tolist(), plate_xy=plate_xy, plate_shift=plate_shift, ring=list(PLATE_RING)),
                f7=dict(layouts=lay7, left_is="+y (the left of the policy's flipped agentview image)"),
                f7p=dict(bddl_trained=SPA_BDDL, bddl_pro=PRO_BDDL, init=PRO_INIT),
                constraints=dict(box=[lo.tolist(), hi.tolist()], gap=GAP, settle_max=SETTLE_MAX), checks=checks)
    with open(META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    for t in tasks:
        print(f"[fx] {t['tid']:2d} {t['key']:14s} '{t['language']}' states={t['states']} hide={t['hide']}@{t['hide_at']}", flush=True)
    print("[fx] done", flush=True)


if __name__ == "__main__":
    main()

"""Rung R2 of the single-GPU ladder (docs/THEORY_COMPOSITION.md 3; docs/SCAFFOLD_RESULTS.md 34): role views.  The eight known
LIBERO-Goal tasks (K1-K8) are re-rendered from the RAW demonstrations, state by state, so every frame has RGB and an entity
segmentation from the same render.  Three LeRobot datasets come from those frames and differ only in what the policy sees:

    plain  every frame carries the task's whole instruction; images unchanged                       (R0)
    rsup   every frame carries the role phrase the hold event picks, as sc_t16.py under SC_SUP=1    (R1)
    rview  rsup's prompts; in BOTH cameras the entity filling the OTHER role is a flat magenta       (R2)
           silhouette (before the hold: the target; during it: the held object).  K7 and K8 stay whole and unpainted.
           Kept as the record of the first rule, which leaks the other role's filler (the painted target differs by task).
    rview2 the second rule (role_paint): rsup's prompts; only the four movable objects are paintable, and on every routed frame
           all of them are painted except the active filler (before the hold the task's object; during it the target if it is
           movable, otherwise all four).  Fixtures are never painted.  Exact masks, one magenta.
    rbox   rview2's rule with each painted object's axis-aligned bounding box in that camera, filled; the active filler's own
           mask pixels stay untouched.

Venvs: extract, lex, convert, norm, serve, check in openpi's (.venv, python 3.11); render and probe in LIBERO's
(examples/libero/.venv, python 3.8, PYTHONPATH=third_party/libero, MUJOCO_GL=egl).  This file must parse under python 3.8.

    python experiments/sc_rv.py extract --task K1 [--root R]                 raw hdf5 -> R/raw/K1.npz (every state and action)
    python experiments/sc_rv.py render  --task K1 [--demos 0+7] [--root R]   -> R/K1/demo_XX.npz, one per demo (skips existing)
    python experiments/sc_rv.py probe   --task K1 [--root R]                 robosuite's camera_segmentations="instance" vs ours
    python experiments/sc_rv.py lex     [--suffix S] [--root R]              the hold threshold, fitted as sc_t16.lex does
    python experiments/sc_rv.py convert --arm plain|rsup|rview|rview2|rbox [--suffix S] [--root R]   LeRobot dataset lol/libero_g8v_<arm><S>
    python experiments/sc_rv.py norm    --arm plain|rsup|rview|rview2|rbox [--suffix S]
    python experiments/sc_rv.py serve   --arm rsup|rview|rview2|rbox --ckpt DIR --port P [--seed 13] [--config NAME]
    python experiments/sc_rv.py check   [--arm rview|rview2|rbox] [--root R] [--n 2] [--suffix S]   per-task phase / paint numbers,
                                        contact sheets (rview: results/flawed/sc_rv_check; rview2, rbox: sc_rv_check2)

Segmentation: rendered from the simulator directly (sim.render(segmentation=True): per-pixel geom ids), each geom mapped to the
LIBERO object or fixture whose body tree holds it (objects_dict + fixtures_dict, as libero_client.seg_tables).  During that one
render, sites are shrunk and geoms with effective alpha under 0.5 (invisible or faint in RGB) are hidden, then restored, so the mask is the
entity's visible silhouette and the environment is otherwise untouched (robosuite's camera_segmentations shrinks every site for
good and adds observation keys; `probe` compares the two).  Role slots (fa1_common.ENT) map to entities: a movable object by its
body, a fixture target by the instance its site belongs to (flat_stove_1_cook_region -> flat_stove_1).
"""
import argparse
import glob
import json
import logging
import math
import os
import pathlib
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OPENPI = "/coc/flash10/ksingara3/robot/openpi"
LEROBOT_HOME = "/coc/flash10/ksingara3/robot/data/hf/lerobot"
RV_ROOT = "/coc/flash10/ksingara3/robot/data/fa1/rv"
OUT = os.path.join(ROOT, "results", "flawed")
CHECK_DIR = os.path.join(OUT, "sc_rv_check")
KNOWN = tuple("K%d" % i for i in range(1, 9))
ARMS = ("plain", "rsup", "rview", "rview2", "rbox", "rrel", "raa")
PAINTED = ("rview", "rview2", "rbox", "raa")
# SCAFFOLD_RESULTS 73 (RAA-min): the same frames as the base arm plus the role channel in the prompt text.  rrel = plain + channel
# (Rrel = R0 + channel), raa = rview2 + channel (RAA = R2 + channel).  The channel is on exactly the frames the base arm routes
# (sc_t16.given is not None: K1-K6); K7 and K8 stay whole with no channel.
BASE = {"rrel": "plain", "raa": "rview2"}
CHAN_ARMS = tuple(BASE)
# rview2 / rbox (the second rule): only the four movable objects are paintable; on every routed frame every one of them is painted
# except the active role's filler (before the hold the task's object; during it the task's target if movable, else none).
# rview2 paints exact masks; rbox paints each object's axis-aligned box in that camera, except the active filler's own pixels
MOVABLE_OBJ = ("akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1")
REPO = {k: "lol/libero_g8v_" + k for k in ARMS}
FULL = {k: "pi05_g8v_" + k for k in ARMS}                                   # assets dir name; the config is FULL + "_ae"
CAMS = ("agentview", "robot0_eye_in_hand")
GEOM = 5                                                                    # mujoco mjOBJ_GEOM
MAGENTA = np.array([255, 0, 255], np.uint8)
SIZE = 256
ALPHA_MIN = 0.5                                                             # effective alpha under which a geom is left out of masks


# ---------------------------------------------------------------- shared helpers (both venvs)
def n2():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def lang():
    return {t["compound"]: t["language"] for t in n2()["tasks"]}


def roles_of():
    """compound -> (object slot, target slot) and language -> the same, from the task definitions (n2_tasks.json)."""
    by_c, by_l = {}, {}
    for t in n2()["tasks"]:
        by_c.setdefault(t["compound"], (t["object"], t["target"]))
        by_l.setdefault(t["language"], (t["object"], t["target"]))
    return by_c, by_l


def noop_keep(actions, thr=1e-4):
    """OpenVLA's regenerate_libero_dataset.py rule (openpi's LIBERO data): drop a step whose first six action dims have norm under
    thr and whose gripper command equals the previous KEPT step's; the first step is dropped on the norm alone."""
    keep, prev = [], None
    for a in np.asarray(actions):
        small = float(np.linalg.norm(a[:-1])) < thr
        noop = small if prev is None else (small and a[-1] == prev[-1])
        keep.append(not noop)
        if not noop:
            prev = a
    return np.asarray(keep, bool)


def flip(x):
    return np.ascontiguousarray(np.asarray(x)[::-1, ::-1])


def quat2axisangle(quat):
    q = np.array(quat, dtype=np.float64)
    q[3] = min(max(q[3], -1.0), 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def chan_cm(rel):
    """Metres -> whole centimetres, rounded half away from zero (the role channel's 1 cm resolution)."""
    return [int(math.floor(abs(float(v)) * 100.0 + 0.5)) * (1 if float(v) >= 0 else -1) for v in rel]


def chan_text(base, rel, on):
    """SR 73: the role channel as a text suffix.  rel = the active role entity's position minus the end effector's, base-frame axes,
    metres; on = ProprioHeld (role target while held, else role object).  Used by training (convert) and by the test-time server."""
    c = chan_cm(rel)
    return "%s. rel: %+d %+d %+d cm, role %s" % (base, c[0], c[1], c[2], "target" if on else "object")


def paint(img, seg, idx):
    """The role view of one image: pixels of entity idx become MAGENTA.  -> (image, painted pixel count)."""
    if idx is None or idx < 0:
        return img, 0
    m = np.asarray(seg) == idx
    n = int(m.sum())
    if n == 0:
        return img, 0
    out = np.array(img, copy=True)
    out[m] = MAGENTA
    return out, n


def role_paint(obj, tgt, held):
    """The second rule -> (slots to paint, the active filler's slot or None).  Fixtures are never painted."""
    active = tgt if held else obj
    if active not in MOVABLE_OBJ:
        active = None
    return [m for m in MOVABLE_OBJ if m != active], active


MIN_PIECE = 4                                                               # px: smaller 8-connected pieces of a mask are dropped


def clean_mask(m):
    """One entity's mask without its pieces under MIN_PIECE pixels: the segmentation render's edge pixels sometimes decode to an
    unrelated geom (1-px strays, e.g. the plate or cheese next to the rack in K5), harmless for a mask but fatal for a box."""
    if not m.any():
        return m
    from scipy import ndimage
    lab, k = ndimage.label(m, structure=np.ones((3, 3), bool))
    if k == 0:
        return m
    keep = np.bincount(lab.ravel()) >= MIN_PIECE
    keep[0] = False
    return keep[lab]


def paint_many(img, seg, ids, active=None, box=False):
    """Paint entities ids MAGENTA: their exact masks, or (box) each one's axis-aligned bounding box in this image, which wins
    over everything except the active filler's own mask pixels.  -> (image, painted pixel count)."""
    seg = np.asarray(seg)
    ids = [int(i) for i in ids if i is not None and i >= 0]
    m = np.zeros(seg.shape, bool)
    for i in ids:
        mi = clean_mask(seg == i)
        if box:
            ys, xs = np.nonzero(mi)
            if len(ys):
                m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        else:
            m |= mi
    if box and active is not None and active >= 0:
        m &= seg != active
    n = int(m.sum())
    if n == 0:
        return img, 0
    out = np.array(img, copy=True)
    out[m] = MAGENTA
    return out, n


def slot_index(names, slot_names, slot_ents, slot):
    """A role slot -> the entity index in names (None when the slot has no entity in this scene)."""
    names, slot_names, slot_ents = [str(x) for x in names], [str(x) for x in slot_names], [str(x) for x in slot_ents]
    if slot not in slot_names:
        return None
    e = slot_ents[slot_names.index(slot)]
    return names.index(e) if e in names else None


# ---------------------------------------------------------------- simulator side (LIBERO venv)
def entity_tables(env):
    """-> dict(names [entities: objects then fixtures], geom_ent [ngeom] entity index or -1, hide [geom ids invisible in RGB],
    slots {fa1_common.ENT slot: entity name or ''}).  Rebuilt per environment (a hard reset rebuilds the simulator)."""
    import fa1_common as FC
    inner = env.env
    m = inner.sim.model
    names = list(inner.objects_dict.keys()) + list(inner.fixtures_dict.keys())
    root_of_body = {int(inner.obj_body_id[n]): i for i, n in enumerate(names) if n in inner.obj_body_id}
    body_ent = np.full(m.nbody, -1, np.int64)
    for b in range(m.nbody):
        x = b
        while True:
            if x in root_of_body:
                body_ent[b] = root_of_body[x]
                break
            p = int(m.body_parentid[x])
            if p == x or x == 0:
                break
            x = p
    geom_ent = body_ent[np.asarray(m.geom_bodyid)]
    rgba = np.asarray(m.geom_rgba, np.float32)
    matid = np.asarray(m.geom_matid)
    default = np.all(rgba == np.array([0.5, 0.5, 0.5, 1.0], np.float32), axis=1)    # MuJoCo uses the material's rgba then
    alpha = rgba[:, 3].copy()
    if int(m.nmat) > 0:
        use = (matid >= 0) & default
        alpha[use] = np.asarray(m.mat_rgba, np.float32)[matid[use], 3]
    slots = {}
    for s in FC.ENT:
        if s in names:
            slots[s] = s
            continue
        try:
            sid = m.site_name2id(s)
        except Exception:                                                   # noqa: BLE001
            slots[s] = ""
            continue
        e = int(body_ent[int(m.site_bodyid[sid])])
        slots[s] = names[e] if e >= 0 else ""
    # hidden in the mask: geoms that barely show in RGB.  The segmentation pass draws translucent geoms opaque; the one such
    # visual geom in LIBERO-Goal's scenes, wine_rack_1_g3 (material alpha 0.1), would otherwise paint a box over the rack's gaps
    return dict(names=names, geom_ent=geom_ent, hide=np.flatnonzero(alpha < ALPHA_MIN), slots=slots,
                missing=[n for n in names if n not in inner.obj_body_id])


def seg_px(env, E, cam, hide=True):
    """Per-pixel entity index (int8, -1 = robot, table, background) of one camera at 256, flipped [::-1, ::-1] like the policy's
    images.  Sites shrunk and invisible geoms hidden for this render only."""
    sim = env.env.sim
    m = sim.model
    old_site = np.array(m.site_size, copy=True)
    hid = E["hide"] if hide else np.zeros(0, np.int64)
    old_grp = np.array(m.geom_group[hid], copy=True)
    try:
        m.site_size[:, :] = 1.0e-8
        if len(hid):
            m.geom_group[hid] = 5
        s = sim.render(camera_name=cam, width=SIZE, height=SIZE, depth=False, segmentation=True)
    finally:
        m.site_size[:, :] = old_site
        if len(hid):
            m.geom_group[hid] = old_grp
    gid = np.where(s[..., 0] == GEOM, s[..., 1], -1)
    ent = np.where(gid >= 0, E["geom_ent"][np.clip(gid, 0, len(E["geom_ent"]) - 1)], -1)
    return flip(ent.astype(np.int8))


def client_payload(env):
    """fc_n2_client.py --seg: what the rview server needs at each query, in the client's orientation.  The raw 256 renders are
    sent so the server paints before the client's resize, as the training pipeline does."""
    E = entity_tables(env)
    sim = env.env.sim
    out = {"meta/rv_names": list(E["names"]), "meta/rv_slots": [[k, v] for k, v in E["slots"].items()]}
    for cam, k in zip(CAMS, ("agent", "wrist")):
        out["meta/rv_img_" + k] = flip(sim.render(camera_name=cam, width=SIZE, height=SIZE, depth=False))
        out["meta/rv_seg_" + k] = seg_px(env, E, cam)
    return out


def raw_path(root, task):
    return os.path.join(root, "raw", task + ".npz")


def demo_path(root, task, i):
    return os.path.join(root, task, "demo_%02d.npz" % int(i))


def demo_files(root, task):
    return sorted(glob.glob(os.path.join(root, task, "demo_*.npz")))


def atomic_savez(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrs)
    os.replace(tmp, path)


def tasks_arg(a):
    return list(KNOWN) if not a.task else [t for t in a.task.split("+") if t]


def extract(a):
    """openpi venv (h5py): every recorded state and action of every raw demo of the task, unfiltered."""
    import h5py
    import fa1_common as FC
    for c in tasks_arg(a):
        out = raw_path(a.root, c)
        if os.path.exists(out):
            print("[rv] extract %s exists" % out, flush=True)
            continue
        src = os.path.join(FC.RAW, FC.KNOWN[c] + "_demo.hdf5")
        arrs = {}
        with h5py.File(src, "r") as f:
            d = f["data"]
            demos = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
            for dm in demos:
                i = int(dm.split("_")[1])
                arrs["states_%d" % i] = np.asarray(d[dm]["states"])
                arrs["actions_%d" % i] = np.asarray(d[dm]["actions"])
            arrs["demos"] = np.array([int(s.split("_")[1]) for s in demos])
            bddl = d.attrs.get("bddl_file_name", "")
        atomic_savez(out, **arrs)
        print("[rv] extract %s: %d demos from %s (recorded bddl %s)" % (c, len(demos), src, bddl), flush=True)


def _bddl(task):
    import fa1_common as FC
    return os.path.join(FC.LIBBDDL, FC.KNOWN[task] + ".bddl")


def render(a):
    """LIBERO venv, GPU (EGL): each recorded state of each demo set in the task's own env, then both cameras and both
    segmentations rendered; no-op steps dropped (noop_keep)."""
    from libero.libero.envs import OffScreenRenderEnv
    L = lang()
    for c in tasks_arg(a):
        raw = np.load(raw_path(a.root, c))
        demos = [int(x) for x in raw["demos"]] if not a.demos else [int(x) for x in a.demos.split("+") if x]
        todo = [j for j in demos if not os.path.exists(demo_path(a.root, c, j))]
        print("[rv] render %s: %d demos, %d to do" % (c, len(demos), len(todo)), flush=True)
        if not todo:
            continue
        env = OffScreenRenderEnv(bddl_file_name=_bddl(c), camera_heights=SIZE, camera_widths=SIZE)
        env.seed(0)
        env.reset()
        E = entity_tables(env)
        print("[rv] %s entities %s; without a body id %s; slots %s; invisible geoms hidden in masks %d (per entity %s)" % (
            c, E["names"], E["missing"], json.dumps(E["slots"]), len(E["hide"]),
            {n: int(np.sum(E["geom_ent"][E["hide"]] == i)) for i, n in enumerate(E["names"])}), flush=True)
        nstate = env.get_sim_state().shape[0]
        for j in todo:
            st, ac = raw["states_%d" % j], raw["actions_%d" % j]
            assert st.shape[1] == nstate, (st.shape, nstate)
            keep = noop_keep(ac)
            R = {k: [] for k in ("image", "wrist", "seg", "wseg", "state")}
            for t in np.flatnonzero(keep):
                obs = env.set_init_state(st[t])
                R["image"].append(flip(obs["agentview_image"]))
                R["wrist"].append(flip(obs["robot0_eye_in_hand_image"]))
                R["seg"].append(seg_px(env, E, CAMS[0]))
                R["wseg"].append(seg_px(env, E, CAMS[1]))
                R["state"].append(np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                  obs["robot0_gripper_qpos"])).astype(np.float32))
            atomic_savez(demo_path(a.root, c, j), image=np.stack(R["image"]).astype(np.uint8), wrist=np.stack(R["wrist"]).astype(np.uint8),
                         seg=np.stack(R["seg"]), wseg=np.stack(R["wseg"]), state=np.stack(R["state"]),
                         actions=np.asarray(ac[keep], np.float32), kept=np.flatnonzero(keep), raw_len=len(ac), demo=j, task=c,
                         language=L[c], names=np.array(E["names"]), slot_names=np.array(list(E["slots"])),
                         slot_ents=np.array(list(E["slots"].values())))
            print("[rv] %s demo %d: %d of %d steps kept -> %s" % (c, j, int(keep.sum()), len(ac), demo_path(a.root, c, j)), flush=True)
        env.close()
    print("[rv] render done", flush=True)


def probe(a):
    """LIBERO venv: does LIBERO pass robosuite 1.4's camera_segmentations="instance" through, and does its instance map agree with
    ours?  Three states of demo 0 of the task; per camera and entity: our pixels, our pixels without hiding invisible geoms,
    robosuite's instance pixels, and the IoU."""
    from libero.libero.envs import OffScreenRenderEnv
    c = tasks_arg(a)[0]
    raw = np.load(raw_path(a.root, c))
    j = int(raw["demos"][0])
    st, ac = raw["states_%d" % j], raw["actions_%d" % j]
    keep = np.flatnonzero(noop_keep(ac))
    ts = [int(keep[0]), int(keep[len(keep) // 2]), int(keep[-1])]
    try:
        env = OffScreenRenderEnv(bddl_file_name=_bddl(c), camera_heights=SIZE, camera_widths=SIZE, camera_segmentations="instance")
        env.seed(0)
        env.reset()
        inst = list(env.env.model.instances_to_ids.keys())
    except Exception as ex:                                                 # noqa: BLE001
        print("[rv] probe: camera_segmentations='instance' through LIBERO failed: %r" % (ex,), flush=True)
        return
    print("[rv] probe %s: robosuite instances %s" % (c, inst), flush=True)
    E = entity_tables(env)
    m = env.env.sim.model
    for g in range(m.ngeom):                                                # every entity geom: what decides its visibility
        e = int(E["geom_ent"][g])
        if e < 0 or int(m.geom_group[g]) == 0:                             # group 0 (collision) is never rendered (vopt below)
            continue
        mid = int(m.geom_matid[g])
        print("[rv] probe geom %d %s %s type %d group %d contype %d conaff %d rgba %s mat %d mat_rgba %s size %s" % (
            g, E["names"][e], m.geom_id2name(g), int(m.geom_type[g]), int(m.geom_group[g]), int(m.geom_contype[g]),
            int(m.geom_conaffinity[g]), np.round(m.geom_rgba[g], 3).tolist(), mid,
            np.round(m.mat_rgba[mid], 3).tolist() if mid >= 0 else None, np.round(m.geom_size[g], 3).tolist()), flush=True)
    vo = env.env.sim._render_context_offscreen.vopt
    print("[rv] probe vopt geomgroup %s sitegroup %s" % (list(vo.geomgroup), list(vo.sitegroup)), flush=True)
    for t in ts:
        obs = env.set_init_state(st[t])
        for cam in CAMS:
            key = cam + "_segmentation_instance"
            if key not in obs:
                print("[rv] probe: no %s in obs (keys %s)" % (key, sorted(k for k in obs if "seg" in k)), flush=True)
                continue
            rs = flip(np.asarray(obs[key])[..., 0])
            mine, raw_m = seg_px(env, E, cam), seg_px(env, E, cam, hide=False)
            for i, n in enumerate(E["names"]):
                m = mine == i
                r = (rs == inst.index(n) + 1) if n in inst else np.zeros_like(m)
                iou = float((m & r).sum()) / max(1, int((m | r).sum()))
                print("[rv] probe t=%d %s %-28s ours %6d  ours-unhidden %6d  robosuite-instance %6d  IoU %.3f" % (
                    t, cam, n, int(m.sum()), int((raw_m == i).sum()), int(r.sum()), iou), flush=True)
    env.close()


# ---------------------------------------------------------------- dataset side (openpi venv)
def _openpi():
    if OPENPI + "/src" not in sys.path:
        sys.path.insert(0, OPENPI + "/src")


def t16():
    """sc_t16 in its supervised stage (SC_SUP=1): given(), held_rows(), the same routing as T16s."""
    os.environ["SC_SUP"] = "1"
    import sc_t16 as T16
    assert T16.SUP, "sc_t16 was imported before SC_SUP was set"
    return T16


def lex_path(suffix):
    return os.path.join(OUT, "sc_rv_lex%s.json" % suffix)


def lex(a):
    """ProprioHeld's threshold fitted on this render's own proprioception, as sc_t16.lex (all K1-K8 rows; grip = finger qpos,
    cmd = the previous action's gripper command).  Loaded if it exists."""
    p = lex_path(a.suffix)
    if os.path.exists(p):
        X = json.load(open(p, encoding="utf-8"))
        print("[rv] lex %s: tw %.6f lift %.3f over %d demos" % (p, X["tw"], X["lift"], X["demos"]), flush=True)
        return X
    from lifelong_ood.policy import parts_flow as PF
    grip, cmd, per = [], [], {}
    for c in KNOWN:
        fs = demo_files(a.root, c)
        per[c] = len(fs)
        for f in fs:
            z = np.load(f)
            grip.append(z["state"][:, 6:8])
            cmd.append(np.r_[-1.0, z["actions"][:-1, 6]])
    if not a.suffix:
        assert all(per[c] > 0 for c in KNOWN), per
    pg = PF.pgate_fit(np.concatenate(grip), np.concatenate(cmd))
    t16p = os.path.join(OUT, "sc_t16_lex.json")
    t16tw = json.load(open(t16p, encoding="utf-8"))["tw"] if os.path.exists(t16p) else None
    X = dict(tw=pg["tw"], lift=pg["lift"], fit=pg, demos=int(sum(per.values())), per_task=per, root=a.root, t16_tw=t16tw)
    tmp = p + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(X, fh, indent=1)
    os.replace(tmp, p)
    print("[rv] lex fitted -> %s: %s (sc_t16's tw on physical-intelligence/libero: %s)" % (p, json.dumps(pg), t16tw), flush=True)
    return X


def frames_of(z, c, arm, X, T16, roles, rel=None):
    """One rendered demo -> per frame (image, wrist, prompt, held, painted agent px, painted wrist px).  rel (SR 73, arms rrel / raa):
    [n, 2, 3] metres, per frame the role channel for the object role [:, 0] and the target role [:, 1]; the frame takes the one of its phase."""
    base_arm = BASE.get(arm, arm)
    assert (rel is not None) == (arm in CHAN_ARMS), (arm, rel is None)
    text = str(z["language"])
    P = T16.given(text)
    H = T16.held_rows(z["state"], z["actions"], X["tw"], X["lift"])
    ids = (None, None)
    if base_arm == "rview" and P is not None:
        obj, tgt = roles[c]
        ids = (slot_index(z["names"], z["slot_names"], z["slot_ents"], tgt),          # before the hold: the target is painted
               slot_index(z["names"], z["slot_names"], z["slot_ents"], obj))          # during it: the held object
        assert ids[0] is not None and ids[1] is not None, (c, roles[c], list(z["slot_ents"]))
    plan = None
    if base_arm in ("rview2", "rbox") and P is not None:
        obj, tgt = roles[c]
        plan = []
        for h in (False, True):                                             # per phase: (entity ids to paint, active filler id)
            slots, act = role_paint(obj, tgt, h)
            pid = [slot_index(z["names"], z["slot_names"], z["slot_ents"], s) for s in slots]
            assert all(x is not None for x in pid), (c, slots, list(z["names"]))
            plan.append((pid, None if act is None else slot_index(z["names"], z["slot_names"], z["slot_ents"], act)))
    for i in range(len(z["actions"])):
        h = bool(H[i])
        task = text if (base_arm == "plain" or P is None) else P[int(h)]
        if rel is not None and P is not None:
            task = chan_text(task, rel[i, int(h)], h)
        img, wimg = z["image"][i], z["wrist"][i]
        na = nw = 0
        if base_arm == "rview" and P is not None:
            img, na = paint(img, z["seg"][i], ids[int(h)])
            wimg, nw = paint(wimg, z["wseg"][i], ids[int(h)])
        elif plan is not None:
            pid, aid = plan[int(h)]
            img, na = paint_many(img, z["seg"][i], pid, aid, box=base_arm == "rbox")
            wimg, nw = paint_many(wimg, z["wseg"][i], pid, aid, box=base_arm == "rbox")
        yield img, wimg, task, h, na, nw


def chan_path(root, task, i):
    return os.path.join(root, task, "chan_%02d.npz" % int(i))


def chan(a):
    """SR 73, LIBERO venv (CPU is enough: no rendering).  Per rendered demo of K1-K6 (K7 and K8 are not routed, so carry no channel):
    each kept raw state is set in the task's own env and the nine role slots' positions (fa1_common.scene: movable bodies, fixture
    sites; the demo's own recorded scene) and the end effector are read -> <root>/<task>/chan_<demo>.npz.  The end effector must
    equal the rendered demo's state[:, :3]."""
    import fa1_common as FC
    for c in [t for t in tasks_arg(a) if t not in ("K7", "K8")]:
        raw = np.load(raw_path(a.root, c))
        todo = [f for f in demo_files(a.root, c) if not os.path.exists(chan_path(a.root, c, int(os.path.basename(f)[5:-4])))]
        print("[rv] chan %s: %d demos to do" % (c, len(todo)), flush=True)
        if not todo:
            continue
        env = FC.make_env(_bddl(c))
        env.seed(0)
        env.reset()
        dj = FC.drawer_joint(env)
        sim = env.env.sim
        try:
            bq = np.asarray(sim.data.body_xquat[sim.model.body_name2id("robot0_base")], np.float64).tolist()
            bp = np.asarray(sim.data.body_xpos[sim.model.body_name2id("robot0_base")], np.float64).tolist()
        except Exception as e:                                               # noqa: BLE001
            bq, bp = None, repr(e)
        print("[rv] %s robot0_base quat %s pos %s (identity quat: base axes = world axes, so rel is the base-frame offset)" % (c, bq, bp), flush=True)
        for f in todo:
            z = np.load(f)
            j = int(z["demo"])
            kept = np.asarray(z["kept"])
            st = raw["states_%d" % j]
            pos, ee = [], []
            for t in kept:
                obs = env.set_init_state(st[t])
                sc = FC.scene(env, obs, dj)
                pos.append(sc["pos"])
                ee.append(sc["ee_pos"])
            pos, ee = np.asarray(pos, np.float32), np.asarray(ee, np.float32)
            d = float(np.abs(ee - np.asarray(z["state"])[:, :3]).max())
            assert d < 2e-3, "%s demo %d: end effector differs from the rendered state by %.4g" % (c, j, d)
            atomic_savez(chan_path(a.root, c, j), pos=pos, ee=ee, kept=kept, ee_maxdiff=np.float32(d),
                         base_quat=np.asarray(bq if bq is not None else [np.nan] * 4, np.float64), ent=np.array(FC.ENT))
            print("[rv] chan %s demo %d: %d frames, ee max diff %.2g" % (c, j, len(kept), d), flush=True)
        env.close()
    print("[rv] chan done", flush=True)


def demo_rel(z, c, f, a, X, T16, roles, info):
    """SR 73 (openpi venv): one rendered demo -> rel [n, 2, 3], the role channel of each frame for both roles: slot position minus the
    end effector, metres.  The slots come from the demo's OWN contact events: pp_p_train.demo_sites (ladder L4 step 1) on the demo's
    recorded slot positions and ProprioHeld rows: g = the slot that moves farthest over the hold, r = the slot nearest g at the last
    held row.  A demo with no hold event takes the task definition's slots (counted in info['fallback']).  No instruction text is read."""
    import fa1_common as FC
    import pp_p_train as PP
    ch = np.load(chan_path(a.root, c, int(os.path.basename(f)[5:-4])))
    pos, ee = np.asarray(ch["pos"], np.float64), np.asarray(ch["ee"], np.float64)
    assert len(pos) == len(z["actions"]) and list(ch["ent"]) == list(FC.ENT), (f, pos.shape, len(z["actions"]))
    H = np.asarray(T16.held_rows(z["state"], z["actions"], X["tw"], X["lift"]), np.float64)
    D = dict(task=np.zeros(len(H), np.int32), demo=np.zeros(len(H), np.int32), pos=pos)
    s = PP.demo_sites(D, H).get((0, 0))
    info["demos"] += 1
    obj, tgt = FC.roles(str(z["language"]))                                  # fa1_common slot names (the task definition; fallback and a count only)
    if s is None:
        info["fallback"].append(os.path.basename(f) + ":" + c)
        g, r = obj, tgt
    else:
        g, r = s["g"], s["r"]
        info["g_ne_given"] += int(g != obj)
        info["r_ne_given"] += int(r != tgt)
    return np.stack([pos[:, FC.ENT.index(g)] - ee, pos[:, FC.ENT.index(r)] - ee], axis=1)


def convert(a):
    os.environ["HF_LEROBOT_HOME"] = LEROBOT_HOME
    _openpi()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    T16 = t16()
    X = lex(a)
    by_c, _ = roles_of()
    repo = REPO[a.arm] + a.suffix
    root = pathlib.Path(LEROBOT_HOME) / repo
    man_path = os.path.join(OUT, "sc_rv_%s%s_manifest.json" % (a.arm, a.suffix))   # written only after the last episode
    assert not (root.exists() and os.path.exists(man_path)), "%s exists and is complete (%s)" % (root, man_path)
    if root.exists():                                  # this job's own interrupted run (LeRobot writes meta/info.json at create)
        print("[rv] removing incomplete %s (no %s)" % (root, man_path), flush=True)
        shutil.rmtree(root)
    files = {c: demo_files(a.root, c)[:(a.ndemo or None)] for c in KNOWN}
    print("[rv] convert %s -> %s; demos per task %s" % (a.arm, repo, {c: len(v) for c, v in files.items()}), flush=True)
    if not a.suffix:
        assert all(len(files[c]) > 0 for c in KNOWN), "a task has no rendered demos"
    features = {"image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]}}
    ds = LeRobotDataset.create(repo_id=repo, root=root, robot_type="panda", fps=10, features=features, image_writer_threads=8,
                               image_writer_processes=4)
    k, counts, per_task, paint_stats = 0, {}, {}, {}
    chan_info = dict(demos=0, fallback=[], g_ne_given=0, r_ne_given=0)
    for c in KNOWN:
        for f in files[c]:
            z = dict(np.load(f))                                             # once: NpzFile decompresses on every access
            assert str(z["task"]) == c, (f, str(z["task"]))
            routed = T16.given(str(z["language"])) is not None                  # K7, K8 are not routed: no channel, no chan file
            rel = (demo_rel(z, c, f, a, X, T16, by_c, chan_info) if routed else np.zeros((len(z["actions"]), 2, 3))) if a.arm in CHAN_ARMS else None
            for i, (img, wimg, task, h, na, nw) in enumerate(frames_of(z, c, a.arm, X, T16, by_c, rel)):
                counts[task] = counts.get(task, 0) + 1
                s = paint_stats.setdefault("%s|%s" % (c, "held" if h else "pre"), dict(frames=0, agent_px=0, wrist_px=0, agent_empty=0,
                                                                                        wrist_empty=0))
                s["frames"] += 1
                s["agent_px"] += na
                s["wrist_px"] += nw
                s["agent_empty"] += int(na == 0)
                s["wrist_empty"] += int(nw == 0)
                ds.add_frame({"image": np.asarray(img), "wrist_image": np.asarray(wimg), "state": np.asarray(z["state"][i], np.float32),
                              "actions": np.asarray(z["actions"][i], np.float32), "task": task})
            ds.save_episode()
            per_task[c] = per_task.get(c, 0) + 1
            k += 1
            if k % 20 == 0:
                print("[rv] %d episodes, at %s %s" % (k, c, os.path.basename(f)), flush=True)
    ds.stop_image_writer()
    for s in paint_stats.values():
        s["agent_frac"] = s["agent_px"] / float(s["frames"] * SIZE * SIZE)
        s["wrist_frac"] = s["wrist_px"] / float(s["frames"] * SIZE * SIZE)
    man = dict(repo_id=repo, arm=a.arm, episodes=k, frames=int(ds.meta.total_frames), per_task=per_task,
               frames_per_prompt=counts if a.arm not in CHAN_ARMS else dict(distinct=len(counts), sample=dict(list(counts.items())[:12])),
               channel=chan_info if a.arm in CHAN_ARMS else None,
               paint=paint_stats if a.arm in PAINTED else None, tw=X["tw"], lift=X["lift"], root=a.root)
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print("[rv] done %s: %d episodes, %d frames, per task %s" % (repo, k, int(ds.meta.total_frames), json.dumps(per_task)), flush=True)
    print("[rv] frames per prompt %s" % json.dumps(counts if a.arm not in CHAN_ARMS else man["frames_per_prompt"]), flush=True)
    if a.arm in CHAN_ARMS:
        print("[rv] channel %s" % json.dumps(chan_info), flush=True)
    if a.arm in PAINTED:
        for key, s in sorted(paint_stats.items()):
            print("[rv] paint %-9s frames %6d  agent frac %.4f (empty %d)  wrist frac %.4f (empty %d)" % (
                key, s["frames"], s["agent_frac"], s["agent_empty"], s["wrist_frac"], s["wrist_empty"]), flush=True)


def norm(a):
    _openpi()
    import pyarrow.parquet as pq
    import openpi.shared.normalize as normalize
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import l90c_config as LC
    cfg = LC.train_config(FULL[a.arm] + "_ae")
    repo = REPO[a.arm] + a.suffix
    out = pathlib.Path(LC.L90C) / "assets" / (FULL[a.arm] + a.suffix) / repo
    if not a.suffix:
        assert out == pathlib.Path(cfg.data.assets.assets_dir) / cfg.data.assets.asset_id, (out, cfg.data.assets)
    files = sorted((pathlib.Path(LEROBOT_HOME) / repo / "data").rglob("episode_*.parquet"))
    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    n, tasks = 0, set()
    for f in files:
        t = pq.read_table(f, columns=["state", "actions", "task_index"]).to_pydict()
        st, ac = np.asarray(t["state"], np.float32), np.asarray(t["actions"], np.float32)
        idx, _ = D.chunk_indices(len(ac), cfg.model.action_horizon)
        stats["state"].update(st)
        stats["actions"].update(ac[idx])
        tasks.update(t["task_index"])
        n += len(ac)
    normalize.save(out, {k: s.get_statistics() for k, s in stats.items()})
    print("[rv] norm -> %s over %d frames of %d episodes; distinct per-frame task indices %d" % (out, n, len(files), len(tasks)), flush=True)


class RoleView:
    """sc_t16.Routed with the role view: the hold event picks the prompt at each query (object prompt until the hold, target prompt
    during it); with paint on, the entity filling the other role is painted magenta in both cameras from the client's segmentation
    (fc_n2_client.py --seg), on the raw 256 renders, then resized exactly as the client resizes.  Everything else is the policy's."""

    def __init__(self, policy, X, mode, replan=5):
        """mode: None (rsup: prompts only), "rview" (first rule), "rview2" or "rbox" (second rule: role_paint)."""
        from lifelong_ood.policy import parts_flow as PF
        assert mode in (None, "rview", "rview2", "rbox"), mode
        self.PF, self.p, self.X, self.paint, self.replan = PF, policy, X, mode, replan
        self.T16 = t16()
        self.roles = roles_of()[1]
        self.det, self.prev, self.metadata, self.checked = None, -1.0, getattr(policy, "metadata", {}), False

    def infer(self, obs):
        from openpi_client import image_tools
        o = dict(obs)
        rv = {k: o.pop(k) for k in list(o) if k.startswith("meta/rv_")}
        if o.get("meta/episode_start") or self.det is None:
            self.det, self.prev, self.checked = self.PF.ProprioHeld(self.X["tw"], self.X["lift"]), -1.0, False
        st = np.asarray(o["observation/state"])
        on = bool(self.det.step(st[6:8], self.prev, st[2]))
        text = str(o["prompt"])
        P = self.T16.given(text)
        if P is not None:
            o["prompt"] = P[int(on)]
        na = nw = 0
        if self.paint:
            if not rv:
                raise RuntimeError("the rview server needs the client's --seg maps (fc_n2_client.py --seg)")
            hw = np.asarray(o["observation/image"]).shape[:2]

            def rs(x):
                return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(x), hw[0], hw[1]))
            if not self.checked:                                             # the raw renders are the client's own images
                d = [int(np.abs(rs(rv["meta/rv_img_" + k]).astype(np.int32) - np.asarray(o[key]).astype(np.int32)).max())
                     for k, key in (("agent", "observation/image"), ("wrist", "observation/wrist_image"))]
                logging.info(f"[rv] episode check: max |resize(raw render) - client image| agent {d[0]} wrist {d[1]}")
                self.checked = True
            if P is not None:
                obj, tgt = self.roles[text]
                sl = [list(x) for x in rv["meta/rv_slots"]]

                def sidx(s):
                    return slot_index(rv["meta/rv_names"], [x for x, _ in sl], [e for _, e in sl], s)
                if self.paint == "rview":                                    # the first rule: the other role's filler
                    slot = obj if on else tgt
                    idx = sidx(slot)
                    if idx is None:
                        logging.warning(f"[rv] slot {slot} has no entity in {list(rv['meta/rv_names'])}")
                    ents = [slot]
                    img, na = paint(np.asarray(rv["meta/rv_img_agent"]), np.asarray(rv["meta/rv_seg_agent"]), idx)
                    wimg, nw = paint(np.asarray(rv["meta/rv_img_wrist"]), np.asarray(rv["meta/rv_seg_wrist"]), idx)
                else:                                                        # rview2 / rbox: every movable object but the active filler
                    ents, act = role_paint(obj, tgt, on)
                    pid, aid = [sidx(s) for s in ents], (None if act is None else sidx(act))
                    box = self.paint == "rbox"
                    img, na = paint_many(np.asarray(rv["meta/rv_img_agent"]), np.asarray(rv["meta/rv_seg_agent"]), pid, aid, box=box)
                    wimg, nw = paint_many(np.asarray(rv["meta/rv_img_wrist"]), np.asarray(rv["meta/rv_seg_wrist"]), pid, aid, box=box)
                if na:
                    o["observation/image"] = rs(img)
                if nw:
                    o["observation/wrist_image"] = rs(wimg)
        logging.info(f"[rv] t={o.get('meta/t')} held={on} prompt={o['prompt']!r} painted agent={na} wrist={nw}"
                     + (f" entities={ents}" if self.paint and P is not None else ""))
        r = self.p.infer(o)
        self.prev = float(np.asarray(r["actions"])[self.replan - 1, 6])
        return r


def serve_cmd(a):
    _openpi()
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import GuidedPolicy, serve, warm
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    assert a.arm in ("rsup",) + PAINTED, a.arm
    name = a.config or FULL[a.arm] + "_ae"
    X = json.load(open(lex_path(a.suffix), encoding="utf-8"))
    logging.info(f"[rv] serve arm {a.arm} config {name} ckpt {a.ckpt} tw {X['tw']:.6f} lift {X['lift']}")
    bk = PolicyBackbone(name, checkpoint_dir=a.ckpt, device="cuda")
    pol = GuidedPolicy(bk, hooks=None, num_steps=10, seed=a.seed)
    warm(pol)
    serve(RoleView(pol, X, mode=a.arm if a.arm in PAINTED else None), a.port)


def check(a):
    """Per task: frames per phase and painted fraction per camera per phase over the first n rendered demos; demo 0's agentview
    against the matching physical-intelligence/libero episode (same kept actions); a contact sheet per task."""
    _openpi()
    from PIL import Image, ImageDraw
    from lifelong_ood.policy import data as D
    T16 = t16()
    p = lex_path(a.suffix)
    if os.path.exists(p):
        X = json.load(open(p, encoding="utf-8"))
    else:
        from lifelong_ood.policy import parts_flow as PF
        g, cm = [], []
        for c in KNOWN:
            for f in demo_files(a.root, c)[: a.n]:
                z = np.load(f)
                g.append(z["state"][:, 6:8])
                cm.append(np.r_[-1.0, z["actions"][:-1, 6]])
        X = PF.pgate_fit(np.concatenate(g), np.concatenate(cm))
        print("[rv] check: no %s; threshold fitted on the checked demos: %s" % (p, json.dumps(X)), flush=True)
    by_c, _ = roles_of()
    tid = {(t["compound"], t["layout"]): t["tid"] for t in n2()["tasks"]}
    print("[rv] n2 tids: %s" % json.dumps({c: [tid.get((c, "familiar")), tid.get((c, "new"))] for c in KNOWN}), flush=True)
    arm = a.arm or "rview"                                                  # the first rule's sheets stay where they were
    assert arm in PAINTED, arm
    cdir = CHECK_DIR if arm == "rview" else CHECK_DIR + "2"
    tag = "" if arm == "rview" else "_" + arm
    os.makedirs(cdir, exist_ok=True)
    print("[rv] check arm %s -> %s" % (arm, cdir), flush=True)
    dm = D.Demos()
    for c in KNOWN:
        fs = demo_files(a.root, c)[: a.n]
        if not fs:
            continue
        st = {}
        for f in fs:
            z = dict(np.load(f))
            for img, wimg, task, h, na, nw in frames_of(z, c, arm, X, T16, by_c):
                s = st.setdefault("held" if h else "pre", [0, 0, 0, 0, 0])
                s[0] += 1
                s[1] += na
                s[2] += nw
                s[3] += int(na == 0)
                s[4] += int(nw == 0)
        z = dict(np.load(fs[0]))
        print("[rv] check %s %r roles %s; entities %s; slot map %s" % (c, str(z["language"]), by_c[c], list(z["names"]),
              dict(zip([str(x) for x in z["slot_names"]], [str(x) for x in z["slot_ents"]]))), flush=True)
        for ph, s in sorted(st.items()):
            print("[rv] check %s %-4s frames %5d  painted frac agent %.4f wrist %.4f  (frames with no painted px: agent %d wrist %d)" % (
                c, ph, s[0], s[1] / float(s[0] * SIZE * SIZE), s[2] / float(s[0] * SIZE * SIZE), s[3], s[4]), flush=True)
        # demo 0 against physical-intelligence/libero: the episode whose actions equal these kept actions
        ac = np.asarray(z["actions"], np.float32)
        match = None
        for e in dm.episodes(dm.task_of_text[str(z["language"])]):
            ep = dm.load(e)
            if len(ep["actions"]) == len(ac) and np.allclose(ep["actions"], ac, atol=1e-5):
                match = e
                break
        if match is None:
            lens = sorted(len(dm.load(e)["actions"]) for e in dm.episodes(dm.task_of_text[str(z["language"])]))
            print("[rv] check %s demo %d: not matched (our %d kept frames; theirs %s)" % (c, int(z["demo"]), len(ac), lens[:60]), flush=True)
        else:
            ep = dm.load(match, images=True)
            theirs = np.stack([D.Demos.decode(x) for x in ep["image"]]).astype(np.float32)
            ours = z["image"].astype(np.float32)
            d = float(np.abs(theirs - ours).mean())
            d_unflip = float(np.abs(theirs - ours[:, ::-1, ::-1]).mean())
            ds_ = np.abs(ep["state"] - z["state"]).max(0)
            wt, wo = ep["state"][:, 6] - ep["state"][:, 7], z["state"][:, 6] - z["state"][:, 7]
            Ht = T16.held_rows(ep["state"], ep["actions"], X["tw"], X["lift"])
            Ho = T16.held_rows(z["state"], z["actions"], X["tw"], X["lift"])
            print("[rv] check %s demo %d matched episode %d: mean |pixel diff| agentview %.2f (unflipped %.2f); max |state diff| per dim %s; "
                  "finger width min ours %.5f theirs %.5f; held rows ours %d theirs %d (same threshold), disagree %d" % (
                      c, int(z["demo"]), match, d, d_unflip, np.round(ds_, 4).tolist(), wo.min(), wt.min(), int(Ho.sum()), int(Ht.sum()),
                      int((Ho != Ht).sum())), flush=True)
        # the contact sheet: 6 frames (3 before the hold, 3 during it when there is one); columns agent raw | agent rview | wrist raw | wrist rview
        rows = list(frames_of(z, c, arm, X, T16, by_c))
        pre = [i for i, r in enumerate(rows) if not r[3]]
        hel = [i for i, r in enumerate(rows) if r[3]]
        pick = (list(np.array(pre)[np.linspace(0, len(pre) - 1, 3 if hel else 6).astype(int)]) if pre else []) + \
               (list(np.array(hel)[np.linspace(0, len(hel) - 1, 3).astype(int)]) if hel else [])
        sheet = Image.new("RGB", (4 * SIZE, len(pick) * SIZE), (0, 0, 0))
        dr = ImageDraw.Draw(sheet)
        for r, i in enumerate(pick):
            img, wimg, task, h, na, nw = rows[i]
            for col, x in enumerate((z["image"][i], img, z["wrist"][i], wimg)):
                sheet.paste(Image.fromarray(np.asarray(x, np.uint8)), (col * SIZE, r * SIZE))
            dr.rectangle([0, r * SIZE, 4 * SIZE, r * SIZE + 14], fill=(0, 0, 0))
            dr.text((3, r * SIZE + 1), "%s demo %d t=%d held=%d prompt=%r painted agent=%d wrist=%d" % (c, int(z["demo"]), i, int(h), task, na, nw),
                    fill=(255, 255, 255))
        out = os.path.join(cdir, "%s%s%s.png" % (c, tag, a.suffix))
        sheet.save(out)
        print("[rv] check %s contact sheet -> %s (frames %s)" % (c, out, [int(i) for i in pick]), flush=True)


def boxes(a):
    """Diagnostic: per entity, its pixel count, bounding box and connected pieces in both cameras at chosen frames of one demo.
    --task K --demos D (one demo number) --frames '0+210' (default: first, middle, last)."""
    from scipy import ndimage
    z = dict(np.load(demo_path(a.root, tasks_arg(a)[0], int(a.demos or 0))))
    T = len(z["actions"])
    fr = [int(x) for x in a.frames.split("+") if x] if a.frames else [0, T // 2, T - 1]
    for t in fr:
        for cam, key in (("agent", "seg"), ("wrist", "wseg")):
            seg = z[key][t]
            for i, n in enumerate(z["names"]):
                m = seg == i
                if not m.any():
                    continue
                ys, xs = np.nonzero(m)
                lab, k = ndimage.label(m)
                sizes = sorted(np.bincount(lab.ravel())[1:].tolist(), reverse=True)
                print("[rv] boxes t=%d %s %-20s px %6d box y %d-%d x %d-%d pieces %d sizes %s" % (
                    t, cam, n, int(m.sum()), ys.min(), ys.max(), xs.min(), xs.max(), k, sizes[:6]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("extract", "render", "probe", "lex", "convert", "norm", "serve", "check", "boxes", "chan"))
    ap.add_argument("--ndemo", type=int, default=0, help="convert: only the first N rendered demos per task (smoke datasets)")
    ap.add_argument("--frames", default="", help="boxes: frame indices, '+'-joined")
    ap.add_argument("--task", default="", help="K1..K8, '+'-joined (default: all)")
    ap.add_argument("--demos", default="", help="render: demo numbers, '+'-joined (default: all)")
    ap.add_argument("--root", default=RV_ROOT)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--suffix", default="", help="appended to dataset, norm and lex names (smoke: _smoke)")
    ap.add_argument("--n", type=int, default=2, help="check: demos per task")
    ap.add_argument("--port", type=int)
    ap.add_argument("--ckpt")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--config", default="", help="serve: config name override (default pi05_g8v_<arm>_ae)")
    a = ap.parse_args()
    if a.what == "serve":
        serve_cmd(a)
    else:
        dict(extract=extract, render=render, probe=probe, lex=lex, convert=convert, norm=norm, check=check, boxes=boxes, chan=chan)[a.what](a)

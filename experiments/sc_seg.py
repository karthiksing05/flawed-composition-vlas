"""A learned instance segmenter that replaces the simulator's privileged entity masks at test time (docs/THEORY_COMPOSITION.md 3, 5;
docs/SCAFFOLD_RESULTS.md 37, 38).  Supervised stage: trained on sc_rv.py's re-rendered K1-K8 demo frames (RGB plus the
simulator's per-pixel entity map), used at test time from RGB alone.

Classes (derived from sc_rv.entity_tables' names as stored in every rendered demo): background (sc_rv id -1: table, floor, walls
AND the robot, which sc_rv's maps do not separate), the four movable objects (sc_rv.MOVABLE_OBJ), each fixture some n2 task uses
as a target (wooden_cabinet_1, flat_stove_1, wine_rack_1 in LIBERO-Goal), and "other" (any further named entity; empty in this
scene).  One network for both cameras: input RGB plus a constant camera plane (agentview -1, wrist +1).

    python experiments/sc_seg.py data  [--root R] [--stride 2]              index + cache the rendered frames (openpi venv, CPU)
    python experiments/sc_seg.py gt    [--tids 0+1] [--states 0+1+2+3]      held-out frames from the rollout initial states (LIBERO venv)
    python experiments/sc_seg.py train [--tag main] [--steps N] ...         the U-Net, from scratch (openpi venv, one A40)
    python experiments/sc_seg.py eval  [--tag main]                         per-class IoU: val, gt familiar, gt new (openpi venv)
    python experiments/sc_seg.py pred  [--tag main]                         drop-in check of pred() against sc_rv.seg_px on gt files
    python experiments/sc_seg.py serve --arm rview2|rbox --ckpt DIR --port P [--seg-ckpt F] [--diag F]   R2 / R2b from learned masks
    python experiments/sc_seg.py pu    --index i[+j] [--seg-ckpt F] [--neps n]   PU's percep arm from learned masks (arm "learned")

data: demos 0-39 of each task train, 40-49 validate (split by demo); every --stride-th kept frame, both cameras ->
  DATA/{train,val}_{rgb,cls,cam,src}.npy and results/flawed/sc_seg/index.json (classes, counts, pixel frequencies).
gt: N1-N6 and K1-K6 (n2 tids 0-11, 16-27), familiar and new layouts, smoke states 0-3 only (states 4-23 are never indexed).  Per
  state: the env is reset and set to the state, 10 dummy steps settle it (the rollouts' wait), then a scripted, noisy reach to a
  movable object (approach, descend, close, lift; 40 steps) gives arm and gripper occlusions.  Every step is recorded with
  sc_rv's own segmentation (seg_px) -> DATA/gt/<tid>_st<j>.npz.  These frames never enter training.
pred(img, cam, names): a per-pixel entity index map in sc_rv.seg_px's convention (int8, -1 = background / robot, else the index
  into names; image orientation flipped [::-1, ::-1] like the policy's images), so it can replace seg_px / the client's maps.
  Post-processing (post=True): per movable class, connected pieces under max(4 px, 10% of its largest piece) become background.
serve: sc_rv.RoleView unchanged; a wrapper replaces meta/rv_seg_agent / meta/rv_seg_wrist with pred() on the client's raw renders
  (meta/rv_img_*) before RoleView paints.  The client's simulator maps, still sent by fc_n2_client.py --seg, are used for one
  thing only: an online IoU diagnostic written to --diag.  They never reach the paint.
pu: sc_pu.episode(arm "percep") with sc_pu.seg_std monkeypatched (in this process) to return pred() on an RGB render of the same
  camera; depth, calibration and the Estimator are sc_pu's own.  Results -> results/flawed/sc_seg/pu_learned[_smoke]/.
  SC_SMOKE=1: n2_branches_smoke.json's first two states (0, 1) only.

This file must parse under python 3.8 (the LIBERO venv runs gt and pu).
"""
import argparse
import glob
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import sc_rv as RV                                                          # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
SD = os.path.join(OUT, "sc_seg")
DATA = "/coc/flash10/ksingara3/robot/data/fa1/seg"
GT_DIR = os.path.join(DATA, "gt")
CK_DIR = os.path.join(DATA, "ckpt")
TRAIN_DEMOS, VAL_DEMOS = tuple(range(0, 40)), tuple(range(40, 50))
GT_COMPS = tuple("K%d" % i for i in range(1, 7)) + tuple("N%d" % i for i in range(1, 7))
GT_STATES = (0, 1, 2, 3)                                                    # the smoke states; 4-23 are the main trial list
SIZE = RV.SIZE
IGNORE = 255
MINPX = 20                                                                  # sc_pu's "empty mask" threshold, reused for detection
POST_FRAC = 0.1
WAIT = 10
MOV = list(RV.MOVABLE_OBJ)


def cam_id(cam):
    return {"agentview": 0, "agent": 0, 0: 0, "robot0_eye_in_hand": 1, "wrist": 1, 1: 1}[cam]


def dpath(name):
    return os.path.join(DATA, name + ".npy")


def index_path():
    return os.path.join(SD, "index.json")


def ckpt_of(tag):
    return os.path.join(CK_DIR, "unet_%s.npz" % tag)


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- classes and id maps (both venvs)
def derive_classes(names, slot_names, slot_ents):
    """background, the movable objects, each fixture an n2 task uses as a target (in names order), other."""
    names = [str(x) for x in names]
    fix = []
    for t in RV.n2()["tasks"]:
        if t["target"] in RV.MOVABLE_OBJ:
            continue
        e = RV.slot_index(names, slot_names, slot_ents, t["target"])
        if e is not None and names[e] not in RV.MOVABLE_OBJ and names[e] not in fix:
            fix.append(names[e])
    fix.sort(key=names.index)
    return ["background"] + MOV + fix + ["other"]


def lut_of(names, classes):
    """sc_rv entity index (+1) -> class id."""
    other = classes.index("other")
    lut = np.full(len(names) + 1, other, np.uint8)
    lut[0] = 0
    for i, n in enumerate(names):
        if str(n) in classes:
            lut[i + 1] = classes.index(str(n))
    return lut


def to_cls(seg, lut):
    return lut[np.asarray(seg, np.int64) + 1]


def inv_of(names, classes):
    """class id -> sc_rv entity index in names (-1 for background, other, and classes absent from names)."""
    names = [str(x) for x in names]
    inv = np.full(len(classes), -1, np.int8)
    for c, n in enumerate(classes):
        if n in names and n not in ("background", "other"):
            inv[c] = names.index(n)
    return inv


def post(cls_map, classes, frac=POST_FRAC):
    """Per movable class: connected pieces under max(MIN_PIECE, frac x its largest piece) become background."""
    from scipy import ndimage
    out = np.array(cls_map, copy=True)
    for nm in MOV:
        c = classes.index(nm)
        m = out == c
        if not m.any():
            continue
        lab, k = ndimage.label(m, structure=np.ones((3, 3), bool))
        if k <= 1:
            continue
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        keep = sz >= max(RV.MIN_PIECE, frac * sz.max())
        keep[0] = False
        out[m & ~keep[lab]] = 0
    return out


# ---------------------------------------------------------------- data (openpi venv, CPU)
def data(a):
    files = {c: RV.demo_files(a.root, c) for c in RV.KNOWN}
    assert all(files[c] for c in RV.KNOWN), {c: len(v) for c, v in files.items()}
    z0 = np.load(files[RV.KNOWN[0]][0])
    classes = derive_classes(z0["names"], [str(x) for x in z0["slot_names"]], [str(x) for x in z0["slot_ents"]])
    nc = len(classes)
    print("[seg] classes %s (names %s)" % (classes, [str(x) for x in z0["names"]]), flush=True)
    plan = {"train": [], "val": []}                                         # (task, file, demo, frame indices)
    for c in RV.KNOWN:
        for f in files[c]:
            j = int(os.path.basename(f)[5:7])
            split = "train" if j in TRAIN_DEMOS else ("val" if j in VAL_DEMOS else None)
            if split is None:
                continue
            T = int(np.load(f)["state"].shape[0])
            plan[split].append((c, f, j, list(range(0, T, a.stride)), T))
    os.makedirs(DATA, exist_ok=True)
    idx = dict(classes=classes, stride=a.stride, root=a.root, train_demos=list(TRAIN_DEMOS), val_demos=list(VAL_DEMOS), splits={})
    for split, rows in plan.items():
        n = sum(len(r[3]) for r in rows)
        M = 2 * n                                                           # agentview row then wrist row, per frame
        tmp = {k: dpath("%s_%s.tmp" % (split, k)) for k in ("rgb", "cls", "cam", "src")}
        X = np.lib.format.open_memmap(tmp["rgb"], mode="w+", dtype=np.uint8, shape=(M, SIZE, SIZE, 3))
        Y = np.lib.format.open_memmap(tmp["cls"], mode="w+", dtype=np.uint8, shape=(M, SIZE, SIZE))
        C = np.zeros(M, np.uint8)
        S = np.zeros((M, 3), np.int32)                                      # task number, demo, frame
        px = np.zeros(nc, np.int64)
        per = {}
        k, t0 = 0, time.time()
        for c, f, j, ts, T in rows:
            z = np.load(f)
            assert str(z["task"]) == c and int(z["demo"]) == j, (f, str(z["task"]), int(z["demo"]))
            lut = lut_of([str(x) for x in z["names"]], classes)
            img, wr, sg, ws = z["image"][ts], z["wrist"][ts], z["seg"][ts], z["wseg"][ts]
            for q in range(len(ts)):
                X[k], Y[k], C[k] = img[q], to_cls(sg[q], lut), 0
                X[k + 1], Y[k + 1], C[k + 1] = wr[q], to_cls(ws[q], lut), 1
                S[k] = S[k + 1] = (int(c[1:]), j, ts[q])
                k += 2
            px += np.bincount(np.asarray(Y[k - 2 * len(ts):k]).ravel(), minlength=nc)[:nc]
            per.setdefault(c, dict(demos=0, frames=0, kept_frames=0))
            per[c]["demos"] += 1
            per[c]["frames"] += len(ts)
            per[c]["kept_frames"] += T
        assert k == M, (k, M)
        X.flush()
        Y.flush()
        del X, Y
        np.save(tmp["cam"], C)
        np.save(tmp["src"], S)
        for key, p in tmp.items():
            os.replace(p, dpath("%s_%s" % (split, key)))
        idx["splits"][split] = dict(rows=M, frames=M // 2, per_task=per, px=px.tolist(),
                                    px_frac={classes[i]: float(px[i]) / float(px.sum()) for i in range(nc)})
        print("[seg] data %s: %d frames (%d rows) from %d demos in %.0fs; class px share %s" % (
            split, M // 2, M, len(rows), time.time() - t0, json.dumps({kk: round(v, 5) for kk, v in idx["splits"][split]["px_frac"].items()})),
            flush=True)
    atomic_json(index_path(), idx)
    print("[seg] index -> %s" % index_path(), flush=True)


# ---------------------------------------------------------------- gt (LIBERO venv, GPU for EGL)
def gt_path(tid, j):
    return os.path.join(GT_DIR, "%d_st%d.npz" % (int(tid), int(j)))


def gt_tasks(tids=""):
    ts = [t for t in RV.n2()["tasks"] if t["compound"] in GT_COMPS]
    if tids:
        want = {int(x) for x in tids.split("+") if x}
        ts = [t for t in ts if t["tid"] in want]
    return ts


def gt(a):
    from libero.libero.envs import OffScreenRenderEnv
    states = [int(x) for x in a.states.split("+") if x]
    assert states and all(s in GT_STATES for s in states), "gt renders smoke states 0-3 only, got %s" % states
    os.makedirs(GT_DIR, exist_ok=True)
    for t in gt_tasks(a.tids):
        todo = [j for j in states if not os.path.exists(gt_path(t["tid"], j))]
        print("[seg] gt tid %d %s %s: %d states to do" % (t["tid"], t["compound"], t["layout"], len(todo)), flush=True)
        if not todo:
            continue
        S = np.load(os.path.join(OUT, "n2_states", "n2_goal_%d.npy" % t["tid"]))
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(ROOT, t["bddl"]), camera_heights=SIZE, camera_widths=SIZE)
        env.seed(0)
        for j in todo:
            t0 = time.time()
            env.reset()
            obs = env.set_init_state(S[j])
            E = RV.entity_tables(env)
            for _ in range(WAIT):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            reach = t["object"] if j % 2 == 0 else MOV[(t["tid"] + j) % 4]
            body = env.env.obj_body_id[reach]
            rng = np.random.RandomState(100000 + 1000 * int(t["tid"]) + j)
            R = {k: [] for k in ("image", "wrist", "seg", "wseg", "ee")}
            n = a.steps
            for k in range(n + 1):
                R["image"].append(RV.flip(obs["agentview_image"]))
                R["wrist"].append(RV.flip(obs["robot0_eye_in_hand_image"]))
                R["seg"].append(RV.seg_px(env, E, RV.CAMS[0]))
                R["wseg"].append(RV.seg_px(env, E, RV.CAMS[1]))
                R["ee"].append(np.asarray(obs["robot0_eef_pos"], np.float32))
                if k == n:
                    break
                p = np.array(env.env.sim.data.body_xpos[body], np.float64)
                ee = np.asarray(obs["robot0_eef_pos"], np.float64)
                ph = k / float(n)                                               # approach, descend, close, lift
                if ph < 0.4:
                    goal, g = p + np.array([0.0, 0.0, 0.12]), -1.0
                elif ph < 0.65:
                    goal, g = p + np.array([0.0, 0.0, 0.01]), -1.0
                elif ph < 0.75:
                    goal, g = p + np.array([0.0, 0.0, 0.01]), 1.0
                else:
                    goal, g = p + np.array([0.0, 0.0, 0.20]), 1.0
                d = np.clip(8.0 * (goal - ee) + rng.normal(0.0, 0.15, 3), -1.0, 1.0)
                act = np.concatenate([d, rng.normal(0.0, 0.05, 3), [g]])
                obs, _, _, _ = env.step(act.tolist())
            RV.atomic_savez(gt_path(t["tid"], j), image=np.stack(R["image"]).astype(np.uint8), wrist=np.stack(R["wrist"]).astype(np.uint8),
                            seg=np.stack(R["seg"]), wseg=np.stack(R["wseg"]), ee=np.stack(R["ee"]), tid=int(t["tid"]),
                            compound=t["compound"], layout=t["layout"], state=j, reach=reach, names=np.array(E["names"]),
                            slot_names=np.array(list(E["slots"])), slot_ents=np.array(list(E["slots"].values())))
            print("[seg] gt tid %d state %d: %d frames, reach %s (%.0fs) -> %s" % (t["tid"], j, n + 1, reach, time.time() - t0,
                                                                                 gt_path(t["tid"], j)), flush=True)
        env.close()
    print("[seg] gt done", flush=True)


# ---------------------------------------------------------------- the network (both venvs: torch 2.7 and torch 1.11)
def build(nc, width=32, in_ch=4):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def gn(ch):
        return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)

    class Block(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.f = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), gn(o), nn.ReLU(inplace=True),
                                   nn.Conv2d(o, o, 3, padding=1, bias=False), gn(o), nn.ReLU(inplace=True))

        def forward(self, x):
            return self.f(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = [width, 2 * width, 4 * width, 8 * width, 8 * width]
            self.inc, self.d1, self.d2, self.d3, self.d4 = Block(in_ch, w[0]), Block(w[0], w[1]), Block(w[1], w[2]), Block(w[2], w[3]), Block(w[3], w[4])
            self.u3, self.u2, self.u1, self.u0 = Block(w[4] + w[3], w[3]), Block(w[3] + w[2], w[2]), Block(w[2] + w[1], w[1]), Block(w[1] + w[0], w[0])
            self.head = nn.Conv2d(w[0], nc, 1)

        def forward(self, x):
            def up(y, ref):
                return F.interpolate(y, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            x0 = self.inc(x)
            x1 = self.d1(F.max_pool2d(x0, 2))
            x2 = self.d2(F.max_pool2d(x1, 2))
            x3 = self.d3(F.max_pool2d(x2, 2))
            x4 = self.d4(F.max_pool2d(x3, 2))
            y = self.u3(torch.cat([up(x4, x3), x3], 1))
            y = self.u2(torch.cat([up(y, x2), x2], 1))
            y = self.u1(torch.cat([up(y, x1), x1], 1))
            y = self.u0(torch.cat([up(y, x0), x0], 1))
            return self.head(y)

    return UNet()


def prep(x, cam):
    """x float [B, 3, H, W] in 0..1, cam [B] 0 / 1 -> the network input [B, 4, H, W]."""
    import torch
    B, _, H, W = x.shape
    plane = (2.0 * cam.float() - 1.0).view(B, 1, 1, 1).expand(B, 1, H, W)
    return torch.cat([(x - 0.5) / 0.25, plane], 1)


def save_weights(net, path, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs = {"p:" + k: v.detach().cpu().float().numpy() for k, v in net.state_dict().items()}
    tmp = path + ".tmp%d.npz" % os.getpid()
    np.savez(tmp, __meta__=json.dumps(meta), **arrs)
    os.replace(tmp, path)


class Segmenter:
    """The trained network + its class table.  classes_of: RGB uint8 [N, 256, 256, 3] (policy orientation) and cams -> class ids;
    ent: one image -> sc_rv.seg_px's convention for the given entity names."""

    def __init__(self, path, device=None):
        import torch
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["__meta__"]))
        self.classes = list(self.meta["classes"])
        self.net = build(len(self.classes), int(self.meta["width"]))
        sd = {k[2:]: torch.from_numpy(np.array(z[k])) for k in z.files if k.startswith("p:")}
        self.net.load_state_dict(sd)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device).eval()
        self.path = path
        self._inv = {}

    def classes_of(self, imgs, cams, post_=True):
        import torch
        imgs = np.asarray(imgs)
        if imgs.ndim == 3:
            imgs = imgs[None]
        assert imgs.shape[1:] == (SIZE, SIZE, 3), imgs.shape
        cams = np.broadcast_to(np.asarray([cam_id(c) if not isinstance(c, (int, np.integer)) else int(c) for c in np.atleast_1d(cams)]),
                               (len(imgs),))
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(imgs)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            c = torch.from_numpy(np.ascontiguousarray(cams)).to(self.device)
            out = self.net(prep(x, c)).argmax(1).to(torch.uint8).cpu().numpy()
        if post_:
            out = np.stack([post(o, self.classes) for o in out])
        return out

    def ent(self, img, cam, names, post_=True):
        key = tuple(str(x) for x in names)
        if key not in self._inv:
            self._inv[key] = inv_of(key, self.classes)
        return self._inv[key][self.classes_of(img, [cam_id(cam)], post_=post_)[0]]


_SEG = {}


def default_ckpt():
    return os.environ.get("SC_SEG_CKPT") or ckpt_of("main")


def load(ckpt=None, device=None):
    key = (ckpt or default_ckpt(), device)
    if key not in _SEG:
        _SEG[key] = Segmenter(key[0], device)
    return _SEG[key]


def pred(img, cam, names, ckpt=None, device=None, post_=True):
    """Drop-in for sc_rv.seg_px: img = the camera's RGB render at 256, flipped [::-1, ::-1] like the policy's images; cam =
    'agentview' / 'robot0_eye_in_hand' (or 'agent' / 'wrist', 0 / 1); names = entity_tables(env)['names'].  -> int8 [256, 256]."""
    return load(ckpt, device).ent(img, cam, names, post_=post_)


# ---------------------------------------------------------------- train (openpi venv, one A40)
def augment(x, y, half, a):
    """On the GPU.  Copy-paste (a movable object cut by its mask from another frame of the same camera, shifted, pasted on top),
    a random affine (flip, rotation, scale, shift; label nearest, outside -> IGNORE), and mild photometric jitter."""
    import torch
    import torch.nn.functional as F
    B, _, H, W = x.shape
    dev = x.device
    if a.cp > 0:
        ar = torch.arange(B, device=dev)
        src = torch.where(ar < half, (ar + 1) % half, half + (ar - half + 1) % (B - half))
        xs, ys = x[src], y[src]
        mov = torch.arange(1, 1 + len(MOV), device=dev)
        pres = (ys[:, None] == mov[None, :, None, None]).flatten(2).sum(-1) >= 50
        do = (torch.rand(B, device=dev) < a.cp) & pres.any(1)
        ch = mov[(torch.rand(B, len(MOV), device=dev) * pres.float()).argmax(1)]
        for b in torch.nonzero(do).flatten().tolist():
            dy, dx = random.randint(-a.shift, a.shift), random.randint(-a.shift, a.shift)
            m = torch.roll(ys[b] == ch[b], (dy, dx), (0, 1))
            xr = torch.roll(xs[b], (dy, dx), (1, 2))
            xb = x[b]
            xb[:, m] = xr[:, m]
            yb = y[b]
            yb[m] = ch[b]
    if a.affine > 0:
        on = (torch.rand(B, device=dev) < a.affine).float()
        ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(a.rot) * on
        sc = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * a.scale * on
        fl = 1.0 - 2.0 * ((torch.rand(B, device=dev) < 0.5).float() * on)
        tx = (torch.rand(B, device=dev) * 2 - 1) * a.trans * on
        ty = (torch.rand(B, device=dev) * 2 - 1) * a.trans * on
        co, si = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([co * fl, -si, tx], 1), torch.stack([si * fl, co, ty], 1)], 1)
        grid = F.affine_grid(theta, [B, 3, H, W], align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        yy = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
        valid = F.grid_sample(torch.ones(B, 1, H, W, device=dev), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0] > 0.5
        y = torch.where(valid, yy, torch.full_like(yy, IGNORE))
    if a.color > 0:
        br = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * a.color
        gain = 1.0 + (torch.rand(B, 3, 1, 1, device=dev) * 2 - 1) * a.color * 0.5
        ct = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * a.color
        mu = x.mean((1, 2, 3), keepdim=True)
        x = ((x - mu) * ct + mu) * br * gain + torch.randn_like(x) * 0.015
        x = x.clamp(0.0, 1.0)
    return x, y


def dice_loss(logits, y, nc):
    import torch
    import torch.nn.functional as F
    p = F.softmax(logits, 1)
    valid = (y != IGNORE)
    yc = torch.where(valid, y, torch.zeros_like(y))
    oh = F.one_hot(yc, nc).permute(0, 3, 1, 2).float() * valid[:, None].float()
    p = p * valid[:, None].float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    present = oh.sum((0, 2, 3)) > 0
    present[0] = False
    d = 1.0 - (2 * inter + 1.0) / (den + 1.0)
    return d[present].mean() if bool(present.any()) else logits.sum() * 0.0


def conf_update(conf, p, g, nc):
    """p, g: torch long [...] (g may hold IGNORE) -> conf [nc, nc] (rows ground truth, columns prediction) += counts."""
    import torch
    v = g != IGNORE
    conf += torch.bincount((g[v] * nc + p[v]).flatten(), minlength=nc * nc).view(nc, nc).to(conf.dtype)


def iou_of(conf):
    conf = np.asarray(conf, np.float64)
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    gtpx = conf.sum(1)
    return [None if gtpx[i] == 0 else float(inter[i] / union[i]) for i in range(len(inter))]


def train(a):
    import torch
    import torch.nn.functional as F
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda"
    idx = json.load(open(index_path(), encoding="utf-8"))
    classes = idx["classes"]
    nc = len(classes)
    Xm, Ym, Cm = np.load(dpath("train_rgb"), mmap_mode="r"), np.load(dpath("train_cls"), mmap_mode="r"), np.load(dpath("train_cam"))
    M = len(Cm) if not a.limit else min(len(Cm), 2 * a.limit)
    t0 = time.time()
    X = torch.empty((M, SIZE, SIZE, 3), dtype=torch.uint8, device=dev)
    Y = torch.empty((M, SIZE, SIZE), dtype=torch.uint8, device=dev)
    for i in range(0, M, 1024):
        X[i:i + 1024] = torch.from_numpy(np.ascontiguousarray(Xm[i:min(M, i + 1024)])).to(dev)
        Y[i:i + 1024] = torch.from_numpy(np.ascontiguousarray(Ym[i:min(M, i + 1024)])).to(dev)
    cam = torch.from_numpy(np.ascontiguousarray(Cm[:M])).to(dev)
    rows = [torch.nonzero(cam == k).flatten() for k in (0, 1)]
    print("[seg] train: %d rows on the GPU in %.0fs (agent %d, wrist %d); classes %s" % (M, time.time() - t0, len(rows[0]), len(rows[1]), classes),
          flush=True)
    Xv, Yv, Cv = np.load(dpath("val_rgb"), mmap_mode="r"), np.load(dpath("val_cls"), mmap_mode="r"), np.load(dpath("val_cam"))
    vsel = np.arange(0, len(Cv), max(1, len(Cv) // 1024))[:1024]
    xv = torch.from_numpy(np.ascontiguousarray(Xv[vsel])).to(dev)
    yv = torch.from_numpy(np.ascontiguousarray(Yv[vsel])).to(dev)
    cv = torch.from_numpy(np.ascontiguousarray(Cv[vsel])).to(dev)
    px = np.asarray(idx["splits"]["train"]["px"], np.float64)
    freq = px / px.sum()
    pres = px > 0
    w = np.zeros(nc)
    w[pres] = np.clip(np.sqrt(np.median(freq[pres]) / freq[pres]), 0.25, 8.0)
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    print("[seg] CE class weights %s" % json.dumps({classes[i]: round(float(w[i]), 3) for i in range(nc)}), flush=True)
    net = build(nc, a.width).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        q = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * q)))

    state = os.path.join(CK_DIR, "%s_state.pt" % a.tag)
    start, log = 0, []
    if os.path.exists(state):
        st = torch.load(state, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        start, log = int(st["step"]), st["log"]
        print("[seg] resumed %s at step %d" % (state, start), flush=True)
    meta = dict(classes=classes, width=a.width, in_ch=4, params=int(nparam), tag=a.tag, steps=a.steps, batch=a.batch, lr=a.lr,
                aug=dict(cp=a.cp, shift=a.shift, affine=a.affine, rot=a.rot, scale=a.scale, trans=a.trans, color=a.color), dice=a.dice,
                ce_weights=w.tolist(), train_rows=int(M), stride=idx["stride"], seed=a.seed)
    print("[seg] U-Net width %d: %d params; %s" % (a.width, nparam, json.dumps(meta["aug"])), flush=True)

    def quick_val():
        net.eval()
        conf = torch.zeros(nc, nc, dtype=torch.float64, device=dev)
        with torch.no_grad():
            for i in range(0, len(cv), 64):
                x = xv[i:i + 64].permute(0, 3, 1, 2).float() / 255.0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = net(prep(x, cv[i:i + 64]))
                conf_update(conf, lo.argmax(1), yv[i:i + 64].long(), nc)
        net.train()
        iou = iou_of(conf.cpu().numpy())
        mv = [iou[classes.index(n)] for n in MOV if iou[classes.index(n)] is not None]
        return iou, float(np.mean(mv)) if mv else None

    net.train()
    half = a.batch // 2
    t1, s = time.time(), start
    for s in range(start, a.steps):
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(s)
        ii = torch.cat([rows[0][torch.randint(len(rows[0]), (half,), device=dev)], rows[1][torch.randint(len(rows[1]), (a.batch - half,), device=dev)]])
        x = X[ii].permute(0, 3, 1, 2).float() / 255.0
        y = Y[ii].long()
        x, y = augment(x, y, half, a)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = net(prep(x, cam[ii]))
        lo = lo.float()
        ce = F.cross_entropy(lo, y, weight=wt, ignore_index=IGNORE)
        dl = dice_loss(lo, y, nc) if a.dice > 0 else torch.zeros((), device=dev)
        loss = ce + a.dice * dl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if (s + 1) % 100 == 0:
            el = time.time() - t1
            print("[seg] step %d loss %.4f (ce %.4f dice %.4f) lr %.2e  %.2f steps/s" % (s + 1, float(loss), float(ce), float(dl), lr_at(s),
                                                                                    (s + 1 - start) / max(el, 1e-6)), flush=True)
        if (s + 1) % a.eval_every == 0 or s + 1 == a.steps:
            iou, mv = quick_val()
            log.append(dict(step=s + 1, val_movable_miou=mv, val_iou={classes[i]: iou[i] for i in range(nc)}, secs=time.time() - t1))
            print("[seg] val@%d (1024 rows) movable mIoU %s  %s" % (s + 1, None if mv is None else round(mv, 4),
                                                                   json.dumps({classes[i]: None if iou[i] is None else round(iou[i], 3) for i in range(nc)})),
                  flush=True)
        if (s + 1) % 1000 == 0 and s + 1 < a.steps:
            os.makedirs(CK_DIR, exist_ok=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=s + 1, log=log), state + ".tmp")
            os.replace(state + ".tmp", state)
    meta["train_secs_this_run"] = time.time() - t1
    meta["log"] = log
    save_weights(net, ckpt_of(a.tag), meta)
    atomic_json(os.path.join(SD, "train_%s.json" % a.tag), meta)
    print("[seg] weights -> %s" % ckpt_of(a.tag), flush=True)
    if os.path.exists(state):
        os.remove(state)


# ---------------------------------------------------------------- eval (openpi venv, one GPU)
def eval_cmd(a):
    import torch
    seg = Segmenter(ckpt_of(a.tag) if not a.seg_ckpt else a.seg_ckpt, "cuda")
    classes = seg.classes
    nc = len(classes)
    mov_ids = [classes.index(n) for n in MOV]
    G = {}                                                                  # group -> dict(raw conf, post conf, det counts)

    def grp(k):
        if k not in G:
            G[k] = dict(raw=np.zeros((nc, nc), np.float64), post=np.zeros((nc, nc), np.float64),
                        det=np.zeros((len(MOV), 2), np.int64), frames=0)
        return G[k]

    def add(keys, P, Pp, Gt):
        Pt, Ppt, Gtt = torch.from_numpy(P.astype(np.int64)), torch.from_numpy(Pp.astype(np.int64)), torch.from_numpy(Gt.astype(np.int64))
        craw = torch.zeros(nc, nc, dtype=torch.float64)
        cpost = torch.zeros(nc, nc, dtype=torch.float64)
        conf_update(craw, Pt, Gtt, nc)
        conf_update(cpost, Ppt, Gtt, nc)
        det = np.zeros((len(MOV), 2), np.int64)                             # frames with the object (>= MINPX gt px), of those IoU >= .5
        for q, c in enumerate(mov_ids):
            gm, pm = Gt == c, Pp == c
            n = gm.reshape(len(Gt), -1).sum(1)
            inter = (gm & pm).reshape(len(Gt), -1).sum(1)
            uni = (gm | pm).reshape(len(Gt), -1).sum(1)
            have = n >= MINPX
            det[q, 0] += int(have.sum())
            det[q, 1] += int((have & (inter >= 0.5 * np.maximum(uni, 1))).sum())
        for k in keys:
            g = grp(k)
            g["raw"] += craw.numpy()
            g["post"] += cpost.numpy()
            g["det"] += det
            g["frames"] += len(Gt)

    camn = ("agent", "wrist")
    # validation: demos 40-49 of K1-K8
    Xv, Yv, Cv = np.load(dpath("val_rgb"), mmap_mode="r"), np.load(dpath("val_cls"), mmap_mode="r"), np.load(dpath("val_cam"))
    t0 = time.time()
    for i in range(0, len(Cv), 128):
        x, y, c = np.asarray(Xv[i:i + 128]), np.asarray(Yv[i:i + 128]), np.asarray(Cv[i:i + 128])
        P = seg.classes_of(x, c, post_=False)
        Pp = np.stack([post(p, classes) for p in P])
        for k in (0, 1):
            s = c == k
            if s.any():
                add(["val|all", "val|" + camn[k]], P[s], Pp[s], y[s])
    print("[seg] eval val: %d rows in %.0fs" % (len(Cv), time.time() - t0), flush=True)
    # gt: N1-N6, K1-K6 smoke states, both layouts
    files = sorted(glob.glob(os.path.join(GT_DIR, "*_st*.npz")))
    t0 = time.time()
    for f in files:
        z = np.load(f)
        lay = "gt_familiar" if str(z["layout"]) == "familiar" else "gt_new"
        kn = "K" if str(z["compound"]).startswith("K") else "N"
        lut = lut_of([str(x) for x in z["names"]], classes)
        for k, (ik, sk) in enumerate((("image", "seg"), ("wrist", "wseg"))):
            x = np.asarray(z[ik])
            y = to_cls(z[sk], lut)
            P = seg.classes_of(x, [k] * len(x), post_=False)
            Pp = np.stack([post(p, classes) for p in P])
            add([lay + "|all", lay + "|" + camn[k], lay + "|all|" + kn, "gt_all|all"], P, Pp, y)
    print("[seg] eval gt: %d files in %.0fs" % (len(files), time.time() - t0), flush=True)
    R = dict(ckpt=seg.path, classes=classes, meta={k: v for k, v in seg.meta.items() if k != "log"}, groups={})
    for k, g in sorted(G.items()):
        ir, ip = iou_of(g["raw"]), iou_of(g["post"])
        cn = g["raw"] / np.maximum(g["raw"].sum(1, keepdims=True), 1)
        conf_top = {}
        for i in range(nc):
            if g["raw"][i].sum() == 0:
                continue
            row = [(classes[j], round(float(cn[i, j]), 4)) for j in np.argsort(-cn[i]) if j != i and cn[i, j] >= 0.01]
            conf_top[classes[i]] = row[:3]
        R["groups"][k] = dict(frames=g["frames"], iou_raw={classes[i]: ir[i] for i in range(nc)}, iou_post={classes[i]: ip[i] for i in range(nc)},
                              movable_miou_raw=float(np.mean([ir[i] for i in mov_ids if ir[i] is not None])),
                              movable_miou_post=float(np.mean([ip[i] for i in mov_ids if ip[i] is not None])),
                              miou_post=float(np.mean([v for v in ip if v is not None])),
                              det={MOV[q]: dict(frames=int(g["det"][q, 0]), iou_ge_half=int(g["det"][q, 1]),
                                                rate=None if g["det"][q, 0] == 0 else round(float(g["det"][q, 1]) / g["det"][q, 0], 4))
                                   for q in range(len(MOV))},
                              confusions_gt_row=conf_top, gt_px={classes[i]: float(g["raw"][i].sum()) for i in range(nc)})
    out = os.path.join(SD, "iou_%s.json" % a.tag)
    atomic_json(out, R)

    def f3(v):
        return "  -  " if v is None else "%.3f" % v
    cols = ["val|all", "gt_familiar|all", "gt_new|all"]
    print("\n| class | " + " | ".join(c.split("|")[0] + " (post)" for c in cols) + " | " + " | ".join(c.split("|")[0] + " (raw)" for c in cols) + " |")
    print("|---" * (1 + 2 * len(cols)) + "|")
    for i, n in enumerate(classes):
        print("| %s | " % n + " | ".join(f3(R["groups"][c]["iou_post"][n]) if c in R["groups"] else "n/a" for c in cols) + " | " +
              " | ".join(f3(R["groups"][c]["iou_raw"][n]) if c in R["groups"] else "n/a" for c in cols) + " |")
    for c in cols:
        if c in R["groups"]:
            g = R["groups"][c]
            print("[seg] %s: frames %d, movable mIoU post %.3f raw %.3f, all-class mIoU post %.3f; detection (IoU >= .5 | gt >= %d px) %s" % (
                c, g["frames"], g["movable_miou_post"], g["movable_miou_raw"], g["miou_post"], MINPX,
                json.dumps({m: v["rate"] for m, v in g["det"].items()})))
    print("\n| class | " + " | ".join(k for k in sorted(R["groups"]) if k not in cols) + " |")
    print("|---" * (1 + len([k for k in R["groups"] if k not in cols])) + "|")
    for n in classes:
        print("| %s | " % n + " | ".join(f3(R["groups"][k]["iou_post"][n]) for k in sorted(R["groups"]) if k not in cols) + " |")
    for c in ("gt_new|all", "gt_familiar|all"):
        if c in R["groups"]:
            print("[seg] %s confusions (gt row -> predicted share): %s" % (c, json.dumps(R["groups"][c]["confusions_gt_row"])))
    print("[seg] eval -> %s" % out, flush=True)
    if a.sheet:
        sheets(seg, files)


PALETTE = np.array([[0, 0, 0], [230, 25, 75], [255, 225, 25], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                    [70, 240, 240], [128, 128, 128]], np.uint8)


def sheets(seg, files):
    """Contact sheets (image | ground truth | prediction, both cameras) of gt frames: per layout, the 6 worst frames by movable
    IoU and 4 spread frames -> results/flawed/sc_seg/sheet_<tag>_<layout>.png."""
    from PIL import Image
    classes = seg.classes
    mov_ids = [classes.index(n) for n in MOV]
    for lay in ("new", "familiar"):
        rows = []
        for f in files:
            z = np.load(f)
            if str(z["layout"]) != lay:
                continue
            lut = lut_of([str(x) for x in z["names"]], classes)
            ya, yw = to_cls(z["seg"], lut), to_cls(z["wseg"], lut)
            pa = seg.classes_of(z["image"], [0] * len(ya))
            pw = seg.classes_of(z["wrist"], [1] * len(yw))
            for t in range(0, len(ya), 5):
                ious = []
                for c in mov_ids:
                    g, p = ya[t] == c, pa[t] == c
                    if g.sum() >= MINPX:
                        ious.append((g & p).sum() / float((g | p).sum()))
                rows.append((float(np.mean(ious)) if ious else 1.0, z["image"][t], ya[t], pa[t], z["wrist"][t], yw[t], pw[t]))
        if not rows:
            continue
        order = np.argsort([r[0] for r in rows])
        pick = list(order[:6]) + list(order[np.linspace(6, len(order) - 1, 4).astype(int)]) if len(order) > 10 else list(order)
        sheet = np.zeros((len(pick) * 128, 6 * 128, 3), np.uint8)
        for r, i in enumerate(pick):
            _, ia, ga, pa_, iw, gw, pw_ = rows[i]
            for col, im in enumerate((ia, PALETTE[ga], PALETTE[pa_], iw, PALETTE[gw], PALETTE[pw_])):
                sheet[r * 128:(r + 1) * 128, col * 128:(col + 1) * 128] = np.asarray(Image.fromarray(np.asarray(im, np.uint8)).resize((128, 128), Image.NEAREST))
        out = os.path.join(SD, "sheet_%s_%s.png" % (os.path.basename(seg.path)[5:-4], lay))
        Image.fromarray(sheet).save(out)
        print("[seg] sheet -> %s (worst movable IoU %s)" % (out, [round(rows[i][0], 3) for i in pick]), flush=True)


def pred_check(a):
    """pred() against sc_rv.seg_px on a few gt files: dtype, shape, value range, entity ids, per-entity IoU."""
    files = sorted(glob.glob(os.path.join(GT_DIR, "*_st*.npz")))[:: max(1, a.every)]
    seg = load(ckpt_of(a.tag) if not a.seg_ckpt else a.seg_ckpt)
    for f in files:
        z = np.load(f)
        names = [str(x) for x in z["names"]]
        for ik, sk, cam in (("image", "seg", "agentview"), ("wrist", "wseg", "robot0_eye_in_hand")):
            t = len(z[ik]) // 2
            p, g = pred(z[ik][t], cam, names, ckpt=seg.path), np.asarray(z[sk][t])
            assert p.dtype == g.dtype == np.int8 and p.shape == g.shape, (p.dtype, g.dtype, p.shape, g.shape)
            assert p.min() >= -1 and p.max() < len(names), (p.min(), p.max())
            iou = {}
            for i, n in enumerate(names):
                u = int(((p == i) | (g == i)).sum())
                iou[n] = None if u == 0 else round(float(((p == i) & (g == i)).sum()) / u, 3)
            print("[seg] pred %s t=%d %s %s ids pred %s true %s IoU %s" % (os.path.basename(f), t, str(z["layout"]), cam,
                  sorted(set(np.unique(p).tolist())), sorted(set(np.unique(g).tolist())), json.dumps(iou)), flush=True)


# ---------------------------------------------------------------- serve: R2 / R2b painted from learned masks (openpi venv)
class LearnedMasks:
    """In front of sc_rv.RoleView: meta/rv_seg_agent and meta/rv_seg_wrist are replaced by pred() on the client's raw renders
    (meta/rv_img_*).  The client's simulator maps are popped first and only feed an IoU diagnostic (diag)."""

    def __init__(self, inner, seg, diag=None):
        self.inner, self.seg, self.diag = inner, seg, diag
        self.metadata = getattr(inner, "metadata", {})
        self.acc = {k: {n: [0, 0, 0] for n in MOV} for k in ("agent", "wrist")}   # intersection, union, frames with the object
        self.n, self.secs = 0, 0.0

    def infer(self, obs):
        o = dict(obs)
        if "meta/rv_names" in o:
            names = [str(x) for x in o["meta/rv_names"]]
            t0 = time.time()
            msg = []
            for k, cam in (("agent", "agentview"), ("wrist", "robot0_eye_in_hand")):
                g = o.pop("meta/rv_seg_" + k, None)
                p = self.seg.ent(np.asarray(o["meta/rv_img_" + k]), cam, names)
                o["meta/rv_seg_" + k] = p
                if g is not None:
                    g = np.asarray(g)
                    for n in MOV:
                        if n not in names:
                            continue
                        i = names.index(n)
                        gm, pm = g == i, p == i
                        A = self.acc[k][n]
                        A[0] += int((gm & pm).sum())
                        A[1] += int((gm | pm).sum())
                        A[2] += int(gm.sum() >= MINPX)
                    msg.append("%s %s" % (k, {n[:6]: round(float(((g == names.index(n)) & (p == names.index(n))).sum()) /
                                                       max(1, int(((g == names.index(n)) | (p == names.index(n))).sum())), 2)
                                              for n in MOV if n in names}))
            self.secs += time.time() - t0
            self.n += 1
            logging.info("[seg] t=%s learned masks (%.0f ms); IoU vs simulator %s" % (o.get("meta/t"), 1000 * (time.time() - t0), "; ".join(msg)))
            if self.diag and self.n % 20 == 0:
                self.dump()
        return self.inner.infer(o)

    def dump(self):
        R = dict(queries=self.n, ms_per_query=1000 * self.secs / max(1, self.n), ckpt=self.seg.path,
                 iou={k: {n: (None if A[1] == 0 else A[0] / float(A[1])) for n, A in v.items()} for k, v in self.acc.items()},
                 frames_with_object={k: {n: A[2] for n, A in v.items()} for k, v in self.acc.items()})
        atomic_json(self.diag, R)


def serve_cmd(a):
    RV._openpi()
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import GuidedPolicy, serve, warm
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    assert a.arm in ("rview2", "rbox"), a.arm
    name = a.config or RV.FULL[a.arm] + "_ae"
    X = json.load(open(RV.lex_path(a.suffix), encoding="utf-8"))
    seg = load(a.seg_ckpt or None, "cuda")
    logging.info(f"[seg] serve arm {a.arm} (learned masks {seg.path}, classes {seg.classes}) config {name} ckpt {a.ckpt} "
                 f"tw {X['tw']:.6f} lift {X['lift']} diag {a.diag}")
    bk = PolicyBackbone(name, checkpoint_dir=a.ckpt, device="cuda")
    pol = GuidedPolicy(bk, hooks=None, num_steps=10, seed=a.seed)
    warm(pol)
    serve(LearnedMasks(RV.RoleView(pol, X, mode=a.arm), seg, diag=a.diag or None), a.port)


# ---------------------------------------------------------------- pu: PU's percep arm from learned masks (LIBERO venv)
def pu(a):
    import torch
    import sc_pu as PU
    from pp_p_roll import load_any
    smoke = bool(os.environ.get("SC_SMOKE"))
    od = os.path.join(SD, "pu_learned_smoke" if smoke else "pu_learned")
    seg = load(a.seg_ckpt or None, "cuda")
    diag = dict(acc={k: {n: [0, 0, 0] for n in MOV} for k in (RV.CAMS[0], RV.CAMS[1])}, n=0, secs=0.0)
    cache = {}

    def seg_std(env, E, cam):
        """sc_pu.seg_std's contract (entity ids, robosuite's top-down convention raw[::-1]) from pred() on an RGB render."""
        sim = env.env.sim
        key = (id(sim), float(sim.data.time), cam)
        if key in cache:
            return cache[key]
        t0 = time.time()
        img = RV.flip(sim.render(camera_name=cam, width=SIZE, height=SIZE, depth=False))
        p = seg.ent(img, cam, E["names"])
        diag["secs"] += time.time() - t0
        diag["n"] += 1
        g = RV.seg_px(env, E, cam)                                            # diagnostic only: never returned
        for n in MOV:
            if n in E["names"]:
                i = E["names"].index(n)
                A = diag["acc"][cam][n]
                A[0] += int(((g == i) & (p == i)).sum())
                A[1] += int(((g == i) | (p == i)).sum())
                A[2] += int((g == i).sum() >= MINPX)
        out = np.ascontiguousarray(np.asarray(p)[:, ::-1])
        if len(cache) > 8:
            cache.clear()
        cache[key] = out
        return out

    PU.seg_std = seg_std                                                    # this process only: measure() reads the module global
    n2 = PU.tasks()
    br_all = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if smoke else "n2_branches_main.json"), encoding="utf-8"))
    if smoke:
        br_all = br_all[:2]
        assert all(int(b["trial"]) in (0, 1) for b in br_all), br_all
    if a.neps:
        br_all = br_all[:a.neps]
    C = PU.load_calib()
    torch.set_num_threads(1)
    for ix in [int(x) for x in str(a.index).split("+") if x != ""]:
        s, tid = PU.combos()[ix]
        t = n2["tasks"][tid]
        assert t["tid"] == tid
        fn = os.path.join(od, "learned_s%d_%d.json" % (s, tid))
        if os.path.exists(fn):
            print("exists", fn, flush=True)
            continue
        S = np.load(os.path.join(OUT, "n2_states", "n2_goal_%d.npy" % tid))
        m, _ = load_any(PU.ckpt_path(s))
        eps = []
        for b in br_all:
            for v in diag["acc"].values():
                for A in v.values():
                    A[0] = A[1] = A[2] = 0
            n0, s0 = diag["n"], diag["secs"]
            e = PU.episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"], "percep", C=C)
            d = dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], held_step=e["held_step"], held_segments=e["held_segments"],
                     roles=e.get("roles"), bound=e["bound"], steps=e["steps"], calls=e["calls"], secs=round(e["secs"], 1), pgate=e.get("pgate"))
            for k in ("percep", "sites_true"):
                if k in e:
                    d[k] = e[k]
            d["seg"] = dict(preds=diag["n"] - n0, ms_per_pred=round(1000 * (diag["secs"] - s0) / max(1, diag["n"] - n0), 2),
                            iou={cam: {n: (None if A[1] == 0 else round(A[0] / float(A[1]), 4)) for n, A in v.items()} for cam, v in diag["acc"].items()},
                            frames_with_object={cam: {n: A[2] for n, A in v.items()} for cam, v in diag["acc"].items()})
            eps.append(d)
            print("[seg-pu] learned", s, t["compound"], t["layout"], b["trial"], json.dumps({k: v for k, v in d.items() if k != "sites_true"}), flush=True)
        res = dict(arm="learned", seed=s, compound=t["compound"], layout=t["layout"], tid=tid, host=os.uname().nodename, ckpt=PU.ckpt_path(s),
                   calib=PU.CALIB, seg_ckpt=seg.path, smoke=smoke, eps=eps)
        atomic_json(fn, res)
        print("[seg-pu] -> %s: %d of %d succeeded" % (fn, sum(e["success"] for e in eps), len(eps)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("data", "gt", "train", "eval", "pred", "serve", "pu"))
    ap.add_argument("--root", default=RV.RV_ROOT)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--tids", default="")
    ap.add_argument("--states", default="0+1+2+3")
    ap.add_argument("--steps", type=int, default=None, help="gt: scripted steps after the settle (40); train: optimiser steps (15000)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--dice", type=float, default=0.5)
    ap.add_argument("--cp", type=float, default=0.5, help="copy-paste probability per sample (0: off)")
    ap.add_argument("--shift", type=int, default=64)
    ap.add_argument("--affine", type=float, default=0.8, help="probability of a random affine per sample (0: off)")
    ap.add_argument("--rot", type=float, default=5.0)
    ap.add_argument("--scale", type=float, default=0.15)
    ap.add_argument("--trans", type=float, default=0.1)
    ap.add_argument("--color", type=float, default=0.15)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=2500)
    ap.add_argument("--limit", type=int, default=0, help="train: only the first N frames (smoke)")
    ap.add_argument("--sheet", type=int, default=1)
    ap.add_argument("--every", type=int, default=12, help="pred: every n-th gt file")
    ap.add_argument("--seg-ckpt", dest="seg_ckpt", default="")
    ap.add_argument("--arm", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--diag", default="")
    ap.add_argument("--index", default="")
    ap.add_argument("--neps", type=int, default=0)
    a = ap.parse_args()
    if a.steps is None:
        a.steps = 40 if a.what == "gt" else 15000
    if a.what == "serve":
        serve_cmd(a)
    else:
        dict(data=data, gt=gt, train=train, eval=eval_cmd, pred=pred_check, pu=pu)[a.what](a)

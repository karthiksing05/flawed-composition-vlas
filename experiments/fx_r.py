"""F7R: a structural front end for F7's relational language (docs/SCAFFOLD_RESULTS.md 45), in front of a FROZEN VLA.  It adds no
information beyond the image and the instruction; the VLA, its server (pp_serve_purify.py --mode intact, seed 13) and the fx
harness (experiments/fx_client.py, imported, never edited) are unchanged.

1. Parse.  A rule parser over F7's fixed templates (and nothing else; anything off-template is refused, the episode is not run):
       "put the left|right bowl on the plate|stove"                 -> relation left|right, landmark = the agentview image frame
       "put the bowl that is [not] on the plate on the stove"       -> relation on|not_on,  landmark = the plate
   and the unique-name instruction the VLA gets instead: "put the bowl on the <target>" (F7's unique-name controls' own wording).
   It is a symbolic resolver over these templates, not a language model; its scope is exactly the 4 relational F7 instructions.
2. Find the candidates.  sc_seg's learned U-Net (classes_of on the RGB the policy sees, no post-processing) gives a bowl class and a
   plate class.  Two identical bowls share the bowl class, so the bowl class in the agentview is split into 8-connected components
   (>= MIN_COMP px; the two largest are the candidates; if only one component exists, it is split by 2-means on its back-projected
   xy).  Each candidate and the plate get a 3D centre: the median of the one-pixel-eroded mask's points, back-projected through the
   camera's rendered depth and calibration (sc_pu.depth_std, sc_pu.cam_model: the same depth and camera model sc_pu's percep arm
   and sc_seg pu use; the depth render is the remaining privilege, as in SR 40).
3. Pick the referent.  left / right: the candidate whose mask's mean column is smaller / larger in the policy's (flipped) agentview
   image.  on / not_on: the candidate whose centre lies nearest the plate's centre in xy is the bowl on the plate (no plate pixels:
   the higher candidate); "not on" takes the other.  Done once, on the observation that feeds the first query (env step WAIT).
4. Hand the VLA two things, per arm:
       f7r     (a) the unique-name instruction and (b) a role view: in BOTH cameras, on every observation from step WAIT on, the
               distractor bowl's pixels are painted with sc_rv.paint_many (rview2's magenta and mask cleaning) from the learned mask.
       name    (a) only: the unique-name instruction on the two-bowl scene, images untouched (= the fx 'amb' wording on these states).
       oracle  f7r with the referent taken from the simulator (the candidate nearest akita_black_bowl_1's body), to separate the
               resolver's error from the policy's; masks, tracking and paint are f7r's.
   Which bowl pixels are the distractor's after step WAIT: every bowl-class pixel of each camera is back-projected (that camera's
   depth and pose) and labelled by the nearer of two 3D estimates, referent and distractor; a connected component >= 80% one label
   takes that label whole.  The agentview updates the estimates (the distractor's only when >= 60% of its first-step mask is
   visible and it moved < 5 cm; the referent's when >= 20 px and it moved < 8 cm), so a carried referent is followed.
   The simulator is read for diagnostics only (resolver accuracy, paint IoU at query steps), never for the paint or the prompt,
   except the oracle arm's referent choice.

Modes (LIBERO venv, py3.8, EGL, one GPU, except read and parse):
    python experiments/fx_r.py parse                                  the parser on every fx instruction + off-template refusals
    python experiments/fx_r.py check --phase smoke [--out DIR]        resolver + paint on the start states, no policy -> DIR/check.json, .png
    python experiments/fx_r.py client --arm f7r|name|oracle <fx_client.py's arguments>      one fx task per process
        -> fx_client's <out>.json, <out>_track.npz, and <out>_f7r.json (per episode: parse, resolve, paint diagnostics)
    python experiments/fx_r.py read --phase smoke|main --roll FXROLL --rroll FXRROLL [--fxjson results/flawed/fx_read_main.json]
        [--models pi05+groot] [--out X.json]                          (CPU) F7R next to fx_read's relational and unique-name rows

Proposed mapping (read), per model and family, f7r arm: R = mean success over the two relational tasks; C = fx_read's C (referent
grasps among bowl grasps); delta = R(f7r) - R(relational, fx's frozen run on the same states and seeds), bootstrapped paired over
states (fx_read.boot: B = 2,000, seed 0).  First matching row:
    NO POWER    U(unique-name control) < .50, or the fx reference < .60 (fx_read's own NO POWER gate)
    FIXED       delta > .10 with CI lower bound > 0, and C(f7r)'s CI lower bound > .75
    NOT FIXED   delta's CI upper bound < .10
    PARTIAL     otherwise
Reported with no verdict: name and oracle arms (success, C, their deltas), f7r - oracle (the resolver's cost), f7r - name (the role
view's share), resolver accuracy against the simulator, paint IoU and referent leak.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

B1, B2 = "akita_black_bowl_1", "akita_black_bowl_2"
PLATE = "plate_1"
ARMS = ("f7r", "name", "oracle")
FAMS = dict(SP=("relL", "relR", "uniL", "uniR"), NG=("pos", "neg", "uniP", "uniT"))
CAMS = ("agentview", "robot0_eye_in_hand")
MIN_COMP = 30              # px: a bowl candidate in the agentview at resolve time
MIN_PX = 20                # px: an estimate update needs this many points
MAJ = 0.8                  # a component >= this share one label takes that label whole
DIS_VIS, DIS_STEP, REF_STEP = 0.6, 0.05, 0.08
REPLAN = 5
FMIN = dict(EPS=0.10, C=0.75, U=0.50, REF=0.60)

# ---------------------------------------------------------------- 1. the parser (any venv)
TEMPLATES = (
    (re.compile(r"^put the (left|right) bowl on the (plate|stove)$"), lambda m: dict(relation=m.group(1), landmark="image", target=m.group(2))),
    (re.compile(r"^put the bowl that is (not )?on the (plate) on the (stove)$"),
     lambda m: dict(relation="not_on" if m.group(1) else "on", landmark=m.group(2), target=m.group(3))),
)


def parse(text):
    """-> dict(relation, landmark, target, unique) or None (refused: not one of the fixed templates)."""
    t = " ".join(str(text).lower().strip().rstrip(".").split())
    for rx, f in TEMPLATES:
        m = rx.match(t)
        if m:
            d = f(m)
            d["unique"] = "put the bowl on the %s" % d["target"]
            return d
    return None


def parse_cmd(a):
    import fx_tasks as FT
    meta = json.load(open(FT.META, encoding="utf-8"))
    rows = []
    for t in meta["tasks"]:
        p = parse(t["language"])
        rows.append(dict(key=t["key"], language=t["language"], parse=p))
        print("[f7r] %-14s %-58r -> %s" % (t["key"], t["language"], p), flush=True)
    for s in ("put the middle bowl on the plate", "put the bowl left of the plate on the stove", "put the bowl that is near the plate on the stove",
              "pick up the black bowl not between the plate and the ramekin and place it on the plate"):
        print("[f7r] off-template %-60r -> %s" % (s, parse(s)), flush=True)
    rel = {r["key"]: r["parse"] for r in rows if r["key"].split("_")[-1] in ("relL", "relR", "pos", "neg")}
    assert all(v is not None for v in rel.values()) and len(rel) == 4, rel
    want = dict(F7_SP_relL=("left", "plate"), F7_SP_relR=("right", "plate"), F7_NG_pos=("on", "stove"), F7_NG_neg=("not_on", "stove"))
    assert all((rel[k]["relation"], rel[k]["target"]) == v for k, v in want.items()), rel
    for k, v in rel.items():                                                 # the unique wording = the fx unique-name controls' own
        fam = k.split("_")[1]
        uni = [t["language"] for t in meta["tasks"] if t["key"] == "F7_%s_%s" % (fam, FAMS[fam][2])][0]
        assert v["unique"] == uni, (k, v["unique"], uni)
    print("[f7r] parser: 4/4 relational templates parsed, unique wording = the fx controls'", flush=True)


# ---------------------------------------------------------------- 2-4. perception, resolver, paint (LIBERO venv)
_SEG = {}


def segmenter(ckpt=""):
    import sc_seg as SG
    if "s" not in _SEG:
        _SEG["s"] = SG.load(ckpt or None, "cuda")
    return _SEG["s"]


def components(m, min_px):
    from scipy import ndimage
    lab, k = ndimage.label(m, structure=np.ones((3, 3), bool))
    if k == 0:
        return []
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    return [lab == i for i in np.argsort(-sz) if i > 0 and sz[i] >= min_px]


def centre(m, P):
    """Median of the one-pixel-eroded mask's 3D points (sc_pu.mask_points' erosion rule)."""
    from scipy import ndimage
    me = ndimage.binary_erosion(m, structure=np.ones((3, 3), bool))
    if int(me.sum()) >= MIN_PX:
        m = me
    return np.median(P[m], 0)


def split2(m, P, iters=20):
    """One mask holding both bowls -> two masks by 2-means on the points' xy (init: the extremes of the principal axis)."""
    r, c = np.nonzero(m)
    xy = P[r, c, :2]
    u = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))[1][:, -1]
    s = (xy - xy.mean(0)) @ u
    mu = np.stack([xy[s <= np.median(s)].mean(0), xy[s > np.median(s)].mean(0)])
    lab = np.zeros(len(xy), int)
    for _ in range(iters):
        lab = np.argmin(((xy[:, None] - mu[None]) ** 2).sum(-1), 1)
        mu = np.stack([xy[lab == j].mean(0) if (lab == j).any() else mu[j] for j in (0, 1)])
    out = []
    for j in (0, 1):
        mj = np.zeros_like(m)
        mj[r[lab == j], c[lab == j]] = True
        out.append(mj)
    return out


class Front:
    """Per episode: resolve at step WAIT, then the distractor's mask in both cameras at every step."""

    def __init__(self, env, arm, parsed, ckpt=""):
        import sc_rv as RV
        import sc_pu as PU
        self.RV, self.PU, self.env, self.arm, self.p = RV, PU, env, arm, parsed
        self.seg = segmenter(ckpt)
        self.cb, self.cp = self.seg.classes.index(B1), self.seg.classes.index(PLATE)
        self.E = RV.entity_tables(env)
        self.paint_on = arm in ("f7r", "oracle")
        self.res, self.ref, self.dis, self.dis_px0 = None, None, None, 0
        self.diag = {c: dict(inter=0, union=0, b1_painted=0, b1_px=0, b2_px=0, b2_seen=0, b2_missed=0, queries=0) for c in CAMS}
        self.trace = []

    # ---------------- geometry
    def cloud(self, cam):
        """World points of every pixel, in the policy's flipped image convention [S, S, 3]."""
        sim = self.env.env.sim
        f, R, c = self.PU.cam_model(sim, cam)
        d = self.PU.depth_std(self.env, self.E, cam)                          # robosuite's raw[::-1]
        S = d.shape[0]
        rr, cc = np.mgrid[0:S, 0:S].astype(np.float64)
        q = np.stack([(cc - S / 2.0) * d / f, (rr - S / 2.0) * d / f, d], -1)
        P = q.reshape(-1, 3) @ R.T + c
        return np.ascontiguousarray(P.reshape(S, S, 3)[:, ::-1])               # raw[::-1] -> flipped raw[::-1, ::-1]

    def sim_pos(self, n):
        inner = self.env.env
        return np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]], np.float64)

    # ---------------- 2-3: resolve
    def resolve(self, obs):
        img = self.RV.flip(obs["agentview_image"])
        cls = self.seg.classes_of(img[None], [0], post_=False)[0]
        P = self.cloud(CAMS[0])
        bowl = cls == self.cb
        comps, how = components(bowl, MIN_COMP), "components"
        n_comp = len(comps)
        if n_comp == 1:
            comps, how = split2(comps[0], P), "split"
        if len(comps) < 2:
            self.res = dict(ok=False, how="none", n_comp=n_comp, bowl_px=int(bowl.sum()))
            return self.res
        cand = comps[:2]
        cen = [centre(m, P) for m in cand]
        col = [float(np.nonzero(m)[1].mean()) for m in cand]
        pl = components(cls == self.cp, MIN_PX)
        pc = centre(pl[0], P) if pl else None
        rel = self.p["relation"]
        if rel in ("left", "right"):
            pick = int(np.argmin(col)) if rel == "left" else int(np.argmax(col))
            on_d = None
        else:
            if pc is not None:
                on_d = [float(np.linalg.norm(c[:2] - pc[:2])) for c in cen]
                on_i = int(np.argmin(on_d))
            else:
                on_d, on_i = None, int(np.argmax([c[2] for c in cen]))
            pick = on_i if rel == "on" else 1 - on_i
        b1, b2 = self.sim_pos(B1), self.sim_pos(B2)
        truth = int(np.argmin([np.linalg.norm(c[:2] - b1[:2]) for c in cen]))           # the candidate that is akita_black_bowl_1
        other = 1 - truth
        use = truth if self.arm == "oracle" else pick
        self.ref, self.dis = cen[use].copy(), cen[1 - use].copy()
        self.dis_px0 = int(cand[1 - use].sum())
        self.res = dict(ok=True, how=how, n_comp=n_comp, relation=rel, pick=pick, truth=truth, used=use, resolver_correct=bool(pick == truth),
                        used_correct=bool(use == truth), col=col, on_dist=on_d, plate_px=int(pl[0].sum()) if pl else 0,
                        cand_px=[int(m.sum()) for m in cand], centres=[c.tolist() for c in cen],
                        xy_err_b1=float(np.linalg.norm(cen[truth][:2] - b1[:2])), xy_err_b2=float(np.linalg.norm(cen[other][:2] - b2[:2])),
                        sim_b1=b1.tolist(), sim_b2=b2.tolist(),
                        # the truth assignment is unambiguous only when each candidate is nearest a different bowl
                        unambiguous=bool(int(np.argmin([np.linalg.norm(c[:2] - b2[:2]) for c in cen])) == other))
        return self.res

    # ---------------- 4: the distractor's mask in one camera
    def dmask(self, cls, P, update):
        bowl = cls == self.cb
        out = np.zeros(bowl.shape, bool)
        if not bowl.any():
            return out
        pts = P[bowl]
        lab = np.zeros(bowl.shape, np.int8)
        lab[bowl] = np.where(np.linalg.norm(pts - self.dis, axis=1) < np.linalg.norm(pts - self.ref, axis=1), 1, 2)
        for m in components(bowl, 1):
            sh = float((lab[m] == 1).mean())
            if sh >= MAJ:
                lab[m] = 1
            elif sh <= 1 - MAJ:
                lab[m] = 2
        out = lab == 1
        if update:
            if int(out.sum()) >= max(MIN_PX, DIS_VIS * self.dis_px0):
                c = centre(out, P)
                if np.linalg.norm(c - self.dis) < DIS_STEP:
                    self.dis = c
            r = lab == 2
            if int(r.sum()) >= MIN_PX:
                c = centre(r, P)
                if np.linalg.norm(c - self.ref) < REF_STEP:
                    self.ref = c
        return out

    def step(self, obs, n, wait):
        """Called on every observation the env returns; n = env steps so far.  -> obs (painted for f7r / oracle from step wait on)."""
        if n < wait:
            return obs
        if self.res is None:
            self.resolve(obs)
        if not self.paint_on or not self.res.get("ok"):
            return obs
        imgs = np.stack([self.RV.flip(obs["agentview_image"]), self.RV.flip(obs["robot0_eye_in_hand_image"])])
        cls = self.seg.classes_of(imgs, [0, 1], post_=False)
        query = (n - wait) % REPLAN == 0
        out = dict(obs)
        npx = []
        for j, (cam, key) in enumerate(zip(CAMS, ("agentview_image", "robot0_eye_in_hand_image"))):
            dm = self.dmask(cls[j], self.cloud(cam), update=(j == 0))
            seg = np.where(dm, 0, -1).astype(np.int8)
            painted, k = self.RV.paint_many(imgs[j], seg, [0])
            npx.append(k)
            if k:
                out[key] = self.RV.flip(painted)
            if query:
                self.score(cam, self.RV.clean_mask(dm))
        if query:
            self.trace.append([int(n), int(npx[0]), int(npx[1])])
        return out

    def score(self, cam, pm):
        """Diagnostic only: the painted mask against the simulator's masks of the two bowls in this camera."""
        g = np.asarray(self.RV.seg_px(self.env, self.E, cam))
        names = self.E["names"]
        t1, t2 = g == names.index(B1), g == names.index(B2)
        D = self.diag[cam]
        D["queries"] += 1
        D["inter"] += int((pm & t2).sum())
        D["union"] += int((pm | t2).sum())
        D["b1_painted"] += int((pm & t1).sum())
        D["b1_px"] += int(t1.sum())
        D["b2_px"] += int(t2.sum())
        if int(t2.sum()) >= MIN_PX:
            D["b2_seen"] += 1
            D["b2_missed"] += int((pm & t2).sum() < 0.5 * t2.sum())

    def record(self):
        return dict(arm=self.arm, parse=self.p, resolve=self.res, diag=self.diag, trace=self.trace,
                    final_ref=None if self.ref is None else self.ref.tolist(), final_dis=None if self.dis is None else self.dis.tolist())


# ---------------------------------------------------------------- client: fx_client.py wrapped (LIBERO venv)
def client_main():
    """fx_client.py's harness, imported: the arm's prompt replaces the task's language in fx_client's in-memory task table, and the
    env fx_client builds (FxEnv: track, hide) is wrapped once more so every observation passes through Front.step."""
    def take(name, default=None):
        if name in sys.argv:
            i = sys.argv.index(name)
            v = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return v
        return default
    sys.argv.pop(1)                                                           # "client"
    arm, ckpt = take("--arm"), take("--seg-ckpt", "")
    assert arm in ARMS, arm
    import fx_client as FC_                                                    # parses --tasks / --out; patches libero_client
    LC, TASK = FC_.LC, FC_.TASK
    parsed = parse(TASK["language"])
    assert parsed is not None, "F7R refuses an off-template instruction: %r" % TASK["language"]
    orig_lang = TASK["language"]
    FC_.FX["tasks"][FC_.TID]["language"] = parsed["unique"]                   # in memory only: the prompt the VLA gets
    br = json.load(open(FC_._arg("--branches"), encoding="utf-8"))
    side = FC_.OUTF.replace(".json", "_f7r.json")
    recs = []
    inner_factory = LC.OffScreenRenderEnv                                      # fx_client's: FxEnv(ORIG(...))
    import time

    class R7Env:
        def __init__(self, env):
            self._e, self.front, self.n, self.secs = env, None, 0, 0.0

        def __getattr__(self, k):
            return getattr(self._e, k)

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            self.n, self.secs = 0, 0.0
            self.front = Front(self._e, arm, parsed, ckpt)
            return obs

        def step(self, a):
            r = self._e.step(a)
            self.n += 1
            if self.front is None:
                return r
            t0 = time.time()
            obs = self.front.step(r[0], self.n, FC_.WAIT)
            self.secs += time.time() - t0
            return (obs,) + tuple(r[1:])

        def close(self):
            if self.front is not None and self.n > 0:
                i = len(recs)
                rec = self.front.record()
                rec.update(trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, secs_front=self.secs, language=orig_lang,
                           prompt=parsed["unique"], tid=FC_.TID, key=TASK["key"])
                recs.append(rec)
                tmp = side + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, tid=FC_.TID, key=TASK["key"], language=orig_lang, prompt=parsed["unique"],
                                   seg_ckpt=self.front.seg.path, eps=recs), fh, indent=1)
                os.replace(tmp, side)
                r = rec["resolve"] or {}
                print("[f7r] %s %s trial %s: resolved %s how=%s pick=%s truth=%s correct=%s; painted (agent, wrist) at queries %s" % (
                    arm, TASK["key"], rec["trial"], r.get("ok"), r.get("how"), r.get("pick"), r.get("truth"), r.get("resolver_correct"),
                    [t[1:] for t in rec["trace"][:6]]), flush=True)
                self.front = None
            self._e.close()

    LC.OffScreenRenderEnv = lambda *a, **k: R7Env(inner_factory(*a, **k))
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "fx episodes need --fresh and --branches"
    assert FC_._arg("--suite") == FC_.FX["suite"] and "," not in FC_._arg("--tasks"), "one fx task per process"
    if "--max-steps" not in sys.argv:
        sys.argv += ["--max-steps", str(TASK["max_steps"])]
    print("[f7r] arm %s task %s %r -> prompt %r, parse %s" % (arm, TASK["key"], orig_lang, parsed["unique"], parsed), flush=True)
    LC.main()


# ---------------------------------------------------------------- check: resolver + paint on the start states, no policy
def check(a):
    import imageio
    import fx_tasks as FT
    meta = json.load(open(FT.META, encoding="utf-8"))
    states = meta["states"][a.phase]
    od = a.out or os.path.join(FT.OUT, "f7r_check")
    os.makedirs(od, exist_ok=True)
    rows, R = [], []
    for t in meta["tasks"]:
        if t["diag"] != "F7" or t["cond"] not in ("relL", "relR", "pos", "neg"):
            continue
        p = parse(t["language"])
        S = np.load(os.path.join(FT.ST, "fx_%d.npy" % t["tid"]))
        env = FT.make_env(os.path.join(ROOT, t["bddl"]))
        env.seed(7)
        for k in states:
            FT.load(env, S[k])
            obs = None
            for _ in range(FT.WAIT):
                obs, _, _, _ = env.step(FT.DUMMY)
            out = {}
            for arm in ("f7r", "oracle"):
                fr = Front(env, arm, p, a.seg_ckpt)
                o2 = fr.step(obs, FT.WAIT, FT.WAIT)
                out[arm] = fr.record()
                if arm == "f7r":
                    a0, w0 = RVflip(obs["agentview_image"]), RVflip(obs["robot0_eye_in_hand_image"])
                    a1, w1 = RVflip(o2["agentview_image"]), RVflip(o2["robot0_eye_in_hand_image"])
                    rows.append(np.concatenate([a0, a1, w0, w1], 1))
            r = out["f7r"]["resolve"]
            R.append(dict(key=t["key"], state=k, parse=p, f7r=out["f7r"], oracle=out["oracle"]))
            print("[f7r] check %s state %d: ok=%s how=%s n_comp=%s pick=%s truth=%s correct=%s xy err b1 %.3f b2 %.3f; paint IoU agent %s wrist %s" % (
                t["key"], k, r.get("ok"), r.get("how"), r.get("n_comp"), r.get("pick"), r.get("truth"), r.get("resolver_correct"),
                r.get("xy_err_b1", np.nan), r.get("xy_err_b2", np.nan), iou(out["f7r"]["diag"][CAMS[0]]), iou(out["f7r"]["diag"][CAMS[1]])), flush=True)
        env.close()
    acc = [x["f7r"]["resolve"].get("resolver_correct", False) for x in R]
    S_ = dict(phase=a.phase, n=len(R), resolver_accuracy=float(np.mean(acc)), ok=float(np.mean([x["f7r"]["resolve"].get("ok", False) for x in R])),
              xy_err_b1_median=float(np.median([x["f7r"]["resolve"].get("xy_err_b1", np.nan) for x in R])),
              xy_err_b2_median=float(np.median([x["f7r"]["resolve"].get("xy_err_b2", np.nan) for x in R])),
              paint=paint_summary([x["f7r"] for x in R]), oracle_paint=paint_summary([x["oracle"] for x in R]), rows=R)
    with open(os.path.join(od, "check_%s.json" % a.phase), "w", encoding="utf-8") as fh:
        json.dump(S_, fh, indent=1)
    imageio.imwrite(os.path.join(od, "check_%s.png" % a.phase), np.concatenate(rows, 0))
    print("[f7r] check: resolver accuracy %.3f over %d (task, state); xy error b1 %.3f b2 %.3f m; paint %s" % (
        S_["resolver_accuracy"], len(R), S_["xy_err_b1_median"], S_["xy_err_b2_median"], S_["paint"]), flush=True)


def RVflip(x):
    return np.ascontiguousarray(np.asarray(x)[::-1, ::-1])


def iou(D):
    return None if not D["union"] else round(D["inter"] / D["union"], 3)


def paint_summary(recs):
    """Pooled over episodes and query steps: IoU(painted, true distractor), the share of the referent's pixels painted (leak), and
    the share of queries where the distractor was visible (>= 20 px) but under half of it was painted (miss)."""
    out = {}
    for c in CAMS:
        T = {k: sum(r["diag"][c][k] for r in recs if r and r.get("diag")) for k in ("inter", "union", "b1_painted", "b1_px", "b2_seen", "b2_missed", "queries")}
        out[c] = dict(iou=None if not T["union"] else T["inter"] / T["union"], leak=None if not T["b1_px"] else T["b1_painted"] / T["b1_px"],
                      miss=None if not T["b2_seen"] else T["b2_missed"] / T["b2_seen"], queries=T["queries"], b2_seen=T["b2_seen"])
    return out


# ---------------------------------------------------------------- read (CPU)
def read(a):
    import fx_read as FR
    import fx_tasks as FT
    meta = json.load(open(FT.META, encoding="utf-8"))
    states = meta["states"][a.phase]
    T = {t["key"]: t for t in meta["tasks"]}
    models = a.models.split("+")
    fxj = json.load(open(a.fxjson, encoding="utf-8")) if a.fxjson and os.path.exists(a.fxjson) else None
    n = len(states)
    mean = lambda x, i: float(np.nanmean(x[i]))                               # noqa: E731
    out = dict(phase=a.phase, states=states, fxjson=a.fxjson if fxj else None, roll=a.roll, rroll=a.rroll, B=FR.B, mapping=MAPPING, F7R={})
    for m in models:
        out["F7R"][m] = {}
        for F, (ra, rb, ua, ub) in FAMS.items():
            ks = ["F7_%s_%s" % (F, c) for c in (ra, rb, ua, ub)]
            rel = {k: FR.load_task(a.roll, m, T[k], states) for k in ks}
            arms = {arm: {k: FR.load_task(a.rroll, "%s_%s" % (m, arm), T[k], states) for k in ks[:2]} for arm in ARMS}
            fxrow = (fxj or {}).get("F7", {}).get(m, {}).get(F)
            row = dict(fx_read=None if not fxrow else {k: fxrow.get(k) for k in ("verdict", "U", "R", "C", "both", "delta", "ref", "success")})
            if any(rel[k] is None for k in ks):
                row["verdict"] = "MISSING (fx roll: %s)" % [k for k in ks if rel[k] is None]
                out["F7R"][m][F] = row
                continue
            s = {k: FR.col(rel[k], states) for k in ks}
            Rrel = lambda i: (mean(s[ks[0]], i) + mean(s[ks[1]], i)) / 2      # noqa: E731
            U = lambda i: (mean(s[ks[2]], i) + mean(s[ks[3]], i)) / 2         # noqa: E731
            row.update(U=FR.boot(U, n), R_rel=FR.boot(Rrel, n), C_rel=c_of(rel, ks[:2], states, FR), arms={})
            Rarm = {}
            for arm in ARMS:
                D = arms[arm]
                if any(D[k] is None for k in ks[:2]):
                    row["arms"][arm] = "MISSING"
                    continue
                sa = {k: FR.col(D[k], states) for k in ks[:2]}
                Rarm[arm] = (lambda sa_: (lambda i: (mean(sa_[ks[0]], i) + mean(sa_[ks[1]], i)) / 2))(sa)
                ra_ = Rarm[arm]
                side = [side_recs(a.rroll, "%s_%s" % (m, arm), T[k]) for k in ks[:2]]
                res = [r for sd in side for r in sd]
                row["arms"][arm] = dict(R=FR.boot(ra_, n), delta_vs_rel=FR.boot(lambda i, ra_=ra_: ra_(i) - Rrel(i), n), C=c_of(D, ks[:2], states, FR),
                                        success={k: mean(sa[k], np.arange(n)) for k in ks[:2]},
                                        grasped={k: {str(st): (D[k].get(st) or {}).get("grasped") for st in states} for k in ks[:2]},
                                        resolver=resolver_summary(res), paint=paint_summary(res) if arm != "name" else None)
            for x, y in (("f7r", "oracle"), ("f7r", "name")):
                if x in Rarm and y in Rarm:
                    row["%s_minus_%s" % (x, y)] = FR.boot(lambda i, x=x, y=y: Rarm[x](i) - Rarm[y](i), n)
            f = row["arms"].get("f7r")
            ref = (fxrow or {}).get("ref")
            if not isinstance(f, dict):
                v = "MISSING"
            elif row["U"][0] < FMIN["U"] or (ref is not None and ref < FMIN["REF"]):
                v = "NO POWER"
            elif f["delta_vs_rel"][1] is not None and f["delta_vs_rel"][0] > FMIN["EPS"] and f["delta_vs_rel"][1] > 0 and \
                    f["C"][1] is not None and f["C"][1] > FMIN["C"]:
                v = "FIXED"
            elif f["delta_vs_rel"][2] is not None and f["delta_vs_rel"][2] < FMIN["EPS"]:
                v = "NOT FIXED"
            else:
                v = "PARTIAL"
            row["verdict"] = v
            out["F7R"][m][F] = row
    js = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(js)
    for m in models:
        for F in FAMS:
            r = out["F7R"][m][F]
            fx = r.get("fx_read") or {}
            print("F7R %s %s: %s | fx_read: %s U %s R %s C %s" % (m, F, r.get("verdict"), fx.get("verdict"), fmt(fx.get("U")), fmt(fx.get("R")), fmt(fx.get("C"))))
            if "U" not in r:
                continue
            print("    recomputed from the fx roll: U %s R(rel) %s C(rel) %s" % (fmt(r["U"]), fmt(r["R_rel"]), fmt(r["C_rel"])))
            for arm in ARMS:
                x = r["arms"].get(arm)
                if not isinstance(x, dict):
                    print("    %-6s %s" % (arm, x))
                    continue
                print("    %-6s R %s  R-R(rel) %s  C %s  resolver %s  paint %s" % (arm, fmt(x["R"]), fmt(x["delta_vs_rel"]), fmt(x["C"]),
                                                                              x["resolver"], short_paint(x["paint"])))
            for k in ("f7r_minus_oracle", "f7r_minus_name"):
                if k in r:
                    print("    %s %s" % (k, fmt(r[k])))


MAPPING = ("per model and family, f7r arm; delta = R(f7r) - R(relational) paired over states; first match: NO POWER iff U < .50 or fx ref "
           "< .60; FIXED iff delta > .10, delta CI lo > 0 and C(f7r) CI lo > .75; NOT FIXED iff delta CI hi < .10; else PARTIAL")


def c_of(D, ks, states, FR):
    """fx_read's C (referent grasps among bowl grasps, over the two relational tasks) and both-right, bootstrapped over states."""
    def gr(k, st):
        e = D[k].get(st)
        return None if e is None else e["grasped"]
    hit = np.array([[gr(k, st) == B1 for k in ks] for st in states], float)
    bowl = np.array([[gr(k, st) in (B1, B2) for k in ks] for st in states], float)
    n = len(states)
    C = FR.boot(lambda i: float(hit[i].sum() / bowl[i].sum()) if bowl[i].sum() else np.nan, n)
    return C


def side_recs(rroll, model, t):
    fn = os.path.join(rroll, "%s_%d_f7r.json" % (model, t["tid"]))
    if not os.path.exists(fn):
        return []
    return json.load(open(fn, encoding="utf-8"))["eps"]


def resolver_summary(recs):
    r = [x["resolve"] for x in recs if x.get("resolve")]
    if not r:
        return None
    ok = [x for x in r if x.get("ok")]
    return dict(n=len(r), ok=len(ok), correct=sum(bool(x.get("resolver_correct")) for x in ok),
                accuracy=float(np.mean([bool(x.get("resolver_correct")) for x in r])) if r else None,
                split=sum(x.get("how") == "split" for x in ok), unambiguous=sum(bool(x.get("unambiguous")) for x in ok),
                xy_err_b1_median=float(np.median([x["xy_err_b1"] for x in ok])) if ok else None,
                xy_err_b2_median=float(np.median([x["xy_err_b2"] for x in ok])) if ok else None)


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, list):
        return "[" + ", ".join("-" if x is None else "%.3f" % x for x in v) + "]"
    return "%.3f" % v if isinstance(v, float) else str(v)


def short_paint(p):
    if not p:
        return "-"
    return " ".join("%s IoU %s leak %s miss %s" % (c[:5], fmt(p[c]["iou"]), fmt(p[c]["leak"]), fmt(p[c]["miss"])) for c in CAMS)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        return client_main()
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("parse", "check", "read"))
    ap.add_argument("--phase", default="smoke", choices=("smoke", "main"))
    ap.add_argument("--out", default="")
    ap.add_argument("--seg-ckpt", dest="seg_ckpt", default="")
    ap.add_argument("--roll", default="")
    ap.add_argument("--rroll", default="")
    ap.add_argument("--fxjson", default=os.path.join(ROOT, "results", "flawed", "fx_read_main.json"))
    ap.add_argument("--models", default="pi05+groot")
    a = ap.parse_args()
    {"parse": parse_cmd, "check": check, "read": read}[a.mode](a)


if __name__ == "__main__":
    main()

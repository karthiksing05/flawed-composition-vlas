"""F1m-CUE: which input makes frozen pi0.5 start K4's high carry ("put the wine bottle on top of the cabinet") when it holds the wine
bottle on new pairings N4 (bottle -> plate) and N6 (bottle -> bowl), although tvp (SR 46-47) already paints the held bottle as a flat
magenta silhouette in the target phase?  (m3_prereg tag f1m_cue; docs/SCAFFOLD_RESULTS.md 53; ARCH_FAILURES F1m, SR 51.)

Every arm is tvp (tv_frozen's role views + R1's role phrases, the same ProprioHeld sequencer, the same learned segmenter) with ONE
input changed in the target phase only (detector on); no arm adds information.
    sb   shape -> box: the held object (the task's object) is painted as the axis-aligned bounding box of its learned mask (after
         sc_rv.clean_mask) instead of its silhouette, in both cameras.  The target's own mask pixels stay visible (sc_rv rbox rule).
         Removes the outline; keeps the extent.
    sf   shape and size -> fixed box: a box of fixed size per camera (BOX_FIXED, set from the smoke) centred on the mask's bounding
         box centre, shifted to stay inside the image, unioned with the bounding box (so no held pixel is ever shown), target pixels
         visible.  Removes outline and (unless the bbox exceeds the fixed box; logged per query) extent.
    g    gripper proprio: tvp's images and phrases, and while the detector is on the two gripper-finger qpos entries of the
         observation (observation/state[6:8] in libero_client: state = eef_pos 3, axis-angle 3, gripper_qpos 2) are replaced by
         G_HELD, the mean finger qpos recorded while the cream cheese is held on K6 (the trained cheese task) in the smoke.
    sg, h  NOT rolled out.  pi0.5's LIBERO checkpoint (openpi config pi05_libero: Pi0Config(pi05=True, discrete_state_input=False);
         pi0_pytorch.embed_suffix adds no state token when pi05) never reads observation/state, and its output transform does not
         use it (extra_delta_transform=False; PolicyBackbone.unnormalize: "zeros are exact for pi0.5").  `statecheck` measures this
         on the served policy (GuidedPolicy, seed rule unchanged): actions under state swaps are compared bit for bit.  So on pi0.5
         g = tvp, sg = sf (or sb) and h (hand height / arm state) = tvp by construction; g is still rolled out on N4 / N6 as the
         empirical check.
tvp (the base) is reused: fresh = results/flawed/tv_fresh/pi05_tvp_<tid> (trials 24-43), main = results/flawed/tv_frozen/pi05_tvp_<tid>
(main states 4-23), both hashed in the prereg row.

Modes:
    LIBERO venv:  python experiments/f1m_cue.py client --arm sb|sf|g <pp_atlas_client.py's arguments>     one n2 task per process
                  -> <out>.json, <out>_track.npz, <out>_tv.json (per episode: detector, paint, box sizes, held finger qpos)
                  python experiments/f1m_cue.py summary DIR [DIR ...]                                       smoke table
                  python experiments/f1m_cue.py read [--out results/flawed/f1m_cue_read.json]               CPU
    openpi venv:  python experiments/f1m_cue.py statecheck --out X.json                                     GPU

Registered mapping (read): pi0.5; primary cells = N4 and N6 on fresh trials 24-43 and main states 4-23 pooled (success matrix
[2 tasks, 40 states], fresh columns then main); for each arm X in (sb, sf, g): d = X - tvp, paired, tv_frozen.boot2 (B = 2000,
numpy default_rng(0), tasks and states drawn jointly), CI 2.5 / 97.5.  CUE FOUND iff CI lo > 0; NOT THIS CUE iff CI hi < .10; else
UNRESOLVED.  Descriptive: high-carry share (n46_phase.episode's lift_held > .22 m among grasped episodes) per arm on N4 / N6 and its
change against tvp; success after high and low carries; K1-K4, K6 success (fresh; K4 also main) and arm - tvp; N1-N3, N5 (fresh) and
arm - tvp; box-size and exceed rates; g's per-episode identity with tvp.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import tv_frozen as TF                                                      # noqa: E402

ARMS = ("sb", "sf", "g")
MODEL = "pi05"
CUT = 0.22                                                                  # n46_lift.py: high carry = lift while held > .22 m
EPS = 0.10
OUT = TF.OUT
D_CUE = os.path.join(OUT, "f1m_cue")
READ_OUT = os.path.join(OUT, "f1m_cue_read.json")
TIDS = dict(N1=16, N2=18, N3=20, N4=22, N5=24, N6=26, K1=0, K2=2, K3=4, K4=6, K6=10)
# Set from smoke job 4706934 (disclosed in the prereg row's "seen"): BOX_FIXED = the max held-object bbox over bottle (N4, K4) and
# cheese (K6) target-phase queries, per dimension, rounded up (agent: bottle h <= 56, w <= 27, cheese 20 x 18; wrist: bottle h <= 157,
# w <= 66, cheese 72 x 78), so bottle and cheese get the same painted box.  G_HELD = K6's mean finger qpos while the cheese is held
# (n 13 queries; the bottle is held at ~(.008, -.008)).
BOX_FIXED = {"agentview": (56, 28), "robot0_eye_in_hand": (160, 80)}      # (h, w) px on the 256 render
G_HELD = (0.0198, -0.0223)                                                  # finger qpos (left, right) while the cheese is held, K6


class CueFront(TF.Front):
    """tv_frozen.Front with arm 'tv' underneath (the same detector, segmenter and paint), one target-phase input changed per cue."""

    def __init__(self, env, cue, obj, tgt, X, seg, sheet=False):
        super().__init__(env, "tv", obj, tgt, X, seg, sheet=sheet)
        self.cue = cue
        self.boxlog = {c: [] for c in TF.CAMS}                              # held queries: [bbox h, bbox w, mask px, exceeds fixed]
        self.qheld = []                                                     # held queries: raw finger qpos
        self.g_steps = 0

    def step(self, obs, n, cmd):
        if self.cue == "g":
            out = super().step(obs, n, cmd)
            if n >= TF.WAIT:
                if self.on and (n - TF.WAIT) % TF.REPLAN == 0:
                    self.qheld.append([float(x) for x in obs["robot0_gripper_qpos"]])
                if self.on:
                    out = dict(out)
                    q = np.asarray(obs["robot0_gripper_qpos"])
                    out["robot0_gripper_qpos"] = np.asarray(G_HELD, q.dtype).reshape(q.shape)
                    self.g_steps += 1
            return out
        if n < TF.WAIT:
            return obs
        t0 = time.time()
        RV = self.RV
        if self.E is None:
            self.E = RV.entity_tables(self.env)
            self.names = list(self.E["names"])
        was = self.on                                                       # tv_frozen.Front.step's sequencer lines, unchanged
        self.on = bool(self.det.step(np.asarray(obs["robot0_gripper_qpos"], np.float64), cmd, float(obs["robot0_eef_pos"][2])))
        row = n - TF.WAIT
        if self.on and not was:
            self.onsets.append(row)
        elif was and not self.on:
            self.offsets.append(row)
        query = row % TF.REPLAN == 0
        if query:
            self.queries += 1
            self.held_queries += int(self.on)
            if self.on:
                self.qheld.append([float(x) for x in obs["robot0_gripper_qpos"]])
        out = dict(obs)
        raw = [TF._flip(obs[k]) for k in TF.KEYS]
        shown = list(raw)
        ents, act = RV.role_paint(self.obj, self.tgt, self.on)
        ids = [self.names.index(e) for e in ents if e in self.names]
        hid = self.names.index(self.obj) if (self.on and self.obj in self.names) else None
        aid = self.names.index(act) if (act is not None and act in self.names) else None
        for j, (cam, key) in enumerate(zip(TF.CAMS, TF.KEYS)):
            p = np.asarray(self.seg.ent(raw[j], cam, self.names))
            img, k = RV.paint_many(raw[j], p, [i for i in ids if i != hid])
            if hid is not None:
                img, kb, st = self.box(img, p, hid, aid, cam)
                k += kb
                if query and st is not None:
                    self.boxlog[cam].append(st)
            if k:
                out[key] = TF._flip(img)
                shown[j] = img
                self.painted_q[j] += int(query)
            if query:
                self.score(cam, p, ids, act)
        if self.sheet and query:
            tags = [f[0] for f in self.frames]
            tag = None
            if row == 0:
                tag = "start"
            elif self.on and "held" not in tags:
                tag = "held"
            elif self.on and "held" in tags and "held+25" not in tags and row >= self.onsets[0] + 25:
                tag = "held+25"
            if tag:
                self.frames.append((tag, np.concatenate([raw[0], shown[0], raw[1], shown[1]], 1)))
        self.secs += time.time() - t0
        return out

    def box(self, img, p, hid, aid, cam):
        """The held object's box (sb: its bounding box; sf: BOX_FIXED unioned with it), target pixels left visible."""
        m = self.RV.clean_mask(p == hid)
        if not m.any():
            return img, 0, None
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        bh, bw = y1 - y0 + 1, x1 - x0 + 1
        M = np.zeros(p.shape, bool)
        M[y0:y1 + 1, x0:x1 + 1] = True
        exc = None
        if self.cue == "sf":
            H, W = BOX_FIXED[cam]
            S0, S1 = p.shape
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            fy, fx = min(max(cy - H // 2, 0), S0 - H), min(max(cx - W // 2, 0), S1 - W)
            M[fy:fy + H, fx:fx + W] = True
            exc = bool(bh > H or bw > W)
        if aid is not None:
            M &= p != aid
        k = int(M.sum())
        if not k:
            return img, 0, [bh, bw, int(m.sum()), exc]
        out = np.array(img, copy=True)
        out[M] = self.RV.MAGENTA
        return out, k, [bh, bw, int(m.sum()), exc]

    def record(self, held):
        r = super().record(held)
        r.update(arm=self.cue, box=self.boxlog, qpos_held=self.qheld, g_steps=self.g_steps)
        return r


# ---------------------------------------------------------------- client (LIBERO venv)
def client_main():
    """tv_frozen.client_main with CueFront and the tvp phrase wrapper (every arm is tvp plus one change)."""
    def take(name, default=None):
        if name in sys.argv:
            i = sys.argv.index(name)
            v = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return v
        return default
    sys.argv.pop(1)
    arm = take("--arm")
    assert arm in ARMS, arm
    import fc_n2_client as FN
    import pp_atlas as PA
    import sc_rv as RV
    import sc_seg as SG
    LC = FN.LC
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = FN.N2["tasks"][TID]
    assert TASK["tid"] == TID and TASK["layout"] == "familiar", TASK
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    seg = SG.load(None, "cuda")
    P = RV.t16().given(TASK["language"])
    assert P is not None, TASK
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    ORIG, RECS, SIDE, CUR = LC.OffScreenRenderEnv, [], [], {}
    SIDEF = OUTF.replace(".json", "_tv.json")
    sheet = os.environ.get("TV_SHEET") == "1"

    class TVEnv(PA.Rec):
        def __init__(self, env):
            super().__init__(env, TASK["object"], TASK["target"])
            self.front = None
            CUR["env"] = self

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            self.front = CueFront(self._e, arm, TASK["object"], TASK["target"], X, seg, sheet=sheet and len(SIDE) < 2)
            self.front.prompts = dict(object=0, target=0)
            return obs

        def step(self, a):
            r = super().step(a)
            if self.front is None:
                return r
            obs = self.front.step(r[0], self.n, float(a[6]))
            return (obs,) + tuple(r[1:])

        def close(self):
            if self.n > 0 and self.front is not None:
                RECS.append(self.data())
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
                i = len(SIDE)
                rec = self.front.record(self.rows["held"])
                rec.update(trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, tid=TID, key=TASK["key"],
                           prompts=self.front.prompts)
                SIDE.append(rec)
                if self.front.frames:
                    import imageio
                    imageio.imwrite(OUTF.replace(".json", "_sheet_%s.png" % rec["trial"]), np.concatenate([f[1] for f in self.front.frames], 0))
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], phrases=P, lex=X, seg_ckpt=seg.path,
                                   box_fixed=BOX_FIXED, g_held=G_HELD, eps=SIDE), fh, indent=1)
                os.replace(tmp, SIDEF)
                print("[f1m] %s %s trial %s: onsets %s (oracle first held %s) held queries %d/%d painted %s g_steps %d" % (
                    arm, TASK["key"], rec["trial"], rec["onsets"][:4], rec["oracle_first_held"], rec["held_queries"], rec["queries"],
                    rec["painted_queries"], rec["g_steps"]), flush=True)
                self.front = None
            self._e.close()

    Base = LC._wcp.WebsocketClientPolicy

    class PClient(Base):                                                    # tv_frozen's tvp wrapper, unchanged
        def infer(self, el):
            env = CUR.get("env")
            f = None if env is None else env.front
            assert str(el["prompt"]) == TASK["language"], el["prompt"]
            el = dict(el)
            on = bool(f is not None and f.on)
            el["prompt"] = P[int(on)]
            if f is not None:
                f.prompts["target" if on else "object"] += 1
            return super().infer(el)
    LC._wcp.WebsocketClientPolicy = PClient
    LC.OffScreenRenderEnv = lambda *a, **k: TVEnv(ORIG(*a, **k))
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    print("[f1m] arm %s task %s %r object %s target %s phrases %s box_fixed %s g_held %s seg %s" % (
        arm, TASK["key"], TASK["language"], TASK["object"], TASK["target"], P, BOX_FIXED, G_HELD, seg.path), flush=True)
    LC.main()


# ---------------------------------------------------------------- state check (openpi venv, GPU)
def statecheck(a):
    """Does the served pi0.5 read observation/state?  Same GuidedPolicy and seed rule as pp_serve_purify --mode intact --seed 13;
    each call is an episode start of a branch with a fixed seed_base, so every call draws the same noise.  Bit-for-bit comparison of
    the action chunks under state swaps; image and prompt swaps are the sensitivity controls."""
    sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
    from lifelong_ood.policy.serve import GuidedPolicy
    from pp_f1_mix import Pi05
    M = Pi05()
    pol = GuidedPolicy(M.bk, hooks=None, num_steps=M.bk.num_steps, seed=13)
    rng = np.random.default_rng(0)
    base = np.array([-0.06, 0.02, 1.05, 3.10, 0.05, -0.08, 0.012, -0.012], np.float32)
    var = {"grip_cheese": np.r_[base[:6], np.asarray(G_HELD, np.float32)], "grip_open": np.r_[base[:6], [0.039, -0.039]],
           "eef_z_plus_0.20": base + np.array([0, 0, 0.2, 0, 0, 0, 0, 0], np.float32),
           "all_random": (base + rng.normal(0, 1.0, 8)).astype(np.float32), "zeros": np.zeros(8, np.float32)}
    prompts = ("put the wine bottle on the plate", "target: the plate", "target: the bowl")
    res = dict(config=M.bk.train_config.name, discrete_state_input=bool(getattr(M.bk.train_config.model, "discrete_state_input", None)),
               pi05=bool(getattr(M.bk.train_config.model, "pi05", None)), cases=[])

    def run(img, wimg, st, prompt, sb):
        o = {"observation/image": img, "observation/wrist_image": wimg, "observation/state": st, "prompt": prompt,
             "meta/episode_start": True, "meta/seed_base": sb, "meta/branch_at": 0, "meta/variant": "n2"}
        return np.asarray(pol.infer(o)["actions"], np.float64)
    for s in range(3):
        img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        wimg = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        for pr in prompts:
            sb = 13000000 + 1000 * s
            a0 = run(img, wimg, base, pr, sb)
            a0b = run(img, wimg, base, pr, sb)
            row = dict(image_seed=s, prompt=pr, repeat_maxabs=float(np.abs(a0 - a0b).max()),
                       state_maxabs={k: float(np.abs(run(img, wimg, v, pr, sb) - a0).max()) for k, v in var.items()},
                       image_maxabs=float(np.abs(run(img2, wimg, base, pr, sb) - a0).max()),
                       prompt_maxabs=float(np.abs(run(img, wimg, base, prompts[(prompts.index(pr) + 1) % 3], sb) - a0).max()),
                       action_absmean=float(np.abs(a0).mean()))
            res["cases"].append(row)
            print("[statecheck] %s" % json.dumps(row), flush=True)
    res["state_invariant"] = all(v == 0.0 for c in res["cases"] for v in c["state_maxabs"].values())
    res["repeat_exact"] = all(c["repeat_maxabs"] == 0.0 for c in res["cases"])
    res["image_sensitive"] = all(c["image_maxabs"] > 1e-3 for c in res["cases"])
    res["prompt_sensitive"] = all(c["prompt_maxabs"] > 1e-3 for c in res["cases"])
    print("[statecheck] config %s pi05 %s discrete_state_input %s | state invariant %s repeat exact %s image sensitive %s prompt sensitive %s" % (
        res["config"], res["pi05"], res["discrete_state_input"], res["state_invariant"], res["repeat_exact"], res["image_sensitive"],
        res["prompt_sensitive"]), flush=True)
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), out
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out, flush=True)


# ---------------------------------------------------------------- smoke summary (LIBERO venv)
def summary(dirs):
    for d in dirs:
        TF.summary(d)
    for d in dirs:
        for f in sorted(x for x in os.listdir(d) if x.endswith("_tv.json")):
            S = json.load(open(os.path.join(d, f), encoding="utf-8"))
            for c in TF.CAMS:
                b = [x for e in S["eps"] for x in (e.get("box") or {}).get(c, [])]
                if b:
                    B = np.asarray([x[:3] for x in b], float)
                    print("== %s %s box %s: n %d h p50/p90/max %s w p50/p90/max %s px p50 %d exceed %s" % (
                        f, S["key"], c, len(b), np.percentile(B[:, 0], [50, 90, 100]).round(0).tolist(),
                        np.percentile(B[:, 1], [50, 90, 100]).round(0).tolist(), int(np.median(B[:, 2])),
                        None if b[0][3] is None else round(float(np.mean([x[3] for x in b])), 3)), flush=True)
            q = [x for e in S["eps"] for x in (e.get("qpos_held") or [])]
            if q:
                Q = np.asarray(q, float)
                print("== %s %s qpos held: n %d mean %s p10 %s p90 %s" % (f, S["key"], len(q), Q.mean(0).round(5).tolist(),
                                                                         np.percentile(Q, 10, 0).round(5).tolist(),
                                                                         np.percentile(Q, 90, 0).round(5).tolist()), flush=True)


# ---------------------------------------------------------------- read (CPU, LIBERO venv)
def label(d):
    if d[1] > 0:
        return "CUE FOUND"
    if d[2] < EPS:
        return "NOT THIS CUE"
    return "UNRESOLVED"


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read(a):
    import pp_atlas as PA
    import n46_phase as NP
    import tv_fresh as TFR
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "f1m_cue"]
    assert len(rows) == 1, len(rows)
    reg = rows[0]["metrics"]["files_sha256"]
    n2 = PA.tasks()
    T = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar"}
    KS = dict(fresh=list(TFR.FRESH_TRIALS), main=list(n2["states"]["main"]))
    base = dict(fresh=(TFR.DIR, MODEL + "_tvp_{tid}"), main=(TF.MAIN_DIR, MODEL + "_tvp_{tid}"))
    plan = dict(sb=dict(fresh=list(TIDS), main=["N4", "N6", "K4"]), sf=dict(fresh=list(TIDS), main=["N4", "N6", "K4"]),
                g=dict(fresh=["N4", "N6"], main=["N4", "N6"]))
    hash_ok = {k: sha(os.path.join(ROOT, k)) == v for k, v in reg.items() if k.startswith("results/flawed/tv_")}

    def stem(arm, st, c):
        if arm == "tvp":
            d, s = base[st]
        else:
            d, s = os.path.join(os.path.abspath(a.dir), st), MODEL + "_" + arm + "_{tid}"
        return os.path.join(d, s.format(tid=T[c]["tid"]))

    def have(f):
        return os.path.exists(f + ".json") and os.path.exists(f + "_track.npz")
    EP, SU = {}, {}
    for arm in ("tvp",) + ARMS:
        for st in ("fresh", "main"):
            for c in (list(TIDS) if arm == "tvp" else plan[arm][st]):
                f = stem(arm, st, c)
                if arm == "tvp" and st == "main" and c not in ("N4", "N6", "K4"):
                    continue
                if not have(f):
                    print("MISSING", arm, st, c, f, flush=True)
                    continue
                r = NP.load(f, T[c], KS[st])
                EP[(arm, st, c)] = r
                SU[(arm, st, c)] = np.array([float(x["success"]) for x in r])
    res = dict(model=MODEL, mapping=MAPPING, states=KS, dir=os.path.abspath(a.dir), reused_tvp_hashes_match=hash_ok,
               arms={}, deltas={}, verdicts={}, carry={}, harness={}, g_identity=None)

    def mat(arm, comps, sets):
        try:
            return np.stack([np.concatenate([SU[(arm, st, c)] for st in sets]) for c in comps])
        except KeyError:
            return None

    def carry(arm, comps, sets):
        R = [r for st in sets for c in comps for r in EP.get((arm, st, c), [])]
        g = [r for r in R if r.get("lift_held") is not None]
        hi = [r for r in g if r["lift_held"] > CUT]
        lo = [r for r in g if r["lift_held"] <= CUT]
        sr = lambda v: (sum(r["success"] for r in v) / float(len(v))) if v else None
        return dict(n=len(R), grasped=len(g), high=len(hi), high_share=(len(hi) / float(len(g))) if g else None,
                    success_high=sr(hi), success_low=sr(lo), lift_med=float(np.median([r["lift_held"] for r in g])) if g else None)
    PRIM = ("N4", "N6")
    for arm in ("tvp",) + ARMS:
        A = dict()
        Xp = mat(arm, PRIM, ("fresh", "main"))
        A["primary_success"] = None if Xp is None else TF.boot1(Xp)
        A["per_set_task"] = {"%s|%s" % (st, c): float(SU[(arm, st, c)].mean()) for (x, st, c) in SU if x == arm}
        Kf = mat(arm, ("K1", "K2", "K3", "K4", "K6"), ("fresh",))
        A["known_fresh"] = None if Kf is None else TF.boot1(Kf)
        Of = mat(arm, ("N1", "N2", "N3", "N5"), ("fresh",))
        A["other_new_fresh"] = None if Of is None else TF.boot1(Of)
        res["carry"][arm] = {"N4+N6 both sets": carry(arm, PRIM, ("fresh", "main")), "N4+N6 fresh": carry(arm, PRIM, ("fresh",)),
                             "N4+N6 main": carry(arm, PRIM, ("main",)), "N4": carry(arm, ("N4",), ("fresh", "main")),
                             "N6": carry(arm, ("N6",), ("fresh", "main")), "K4 both sets": carry(arm, ("K4",), ("fresh", "main"))}
        res["arms"][arm] = A
    Xb = mat("tvp", PRIM, ("fresh", "main"))
    for arm in ARMS:
        X = mat(arm, PRIM, ("fresh", "main"))
        D = dict(primary=None if X is None or Xb is None else TF.boot2(X, Xb))
        for nm, comps, sets in (("known_fresh", ("K1", "K2", "K3", "K4", "K6"), ("fresh",)), ("K4_both", ("K4",), ("fresh", "main")),
                                ("other_new_fresh", ("N1", "N2", "N3", "N5"), ("fresh",)), ("N4N6_fresh", PRIM, ("fresh",)),
                                ("N4N6_main", PRIM, ("main",))):
            x, y = mat(arm, comps, sets), mat("tvp", comps, sets)
            D[nm] = None if x is None or y is None else TF.boot2(x, y)
        c0, c1 = res["carry"]["tvp"]["N4+N6 both sets"], res["carry"][arm]["N4+N6 both sets"]
        D["high_share_change"] = None if c0["high_share"] is None or c1["high_share"] is None else c1["high_share"] - c0["high_share"]
        res["deltas"][arm] = D
        res["verdicts"][arm] = None if D["primary"] is None else label(D["primary"])
        H = {}
        for (x, st, c) in SU:
            if x == arm:
                diff, ok = TFR.harness(MODEL, stem("tvp", st, c), stem(arm, st, c))
                H["%s|%s" % (st, c)] = dict(args_diff=diff, episodes_match=ok)
        res["harness"][arm] = H
        side = [e for (x, st, c) in SU if x == arm for e in json.load(open(stem(arm, st, c) + "_tv.json", encoding="utf-8"))["eps"]]
        bx = {}
        for cam in TF.CAMS:
            b = [v for e in side for v in (e.get("box") or {}).get(cam, [])]
            bx[cam] = dict(n=len(b), exceed=float(np.mean([v[3] for v in b])) if (b and b[0][3] is not None) else None,
                           h_med=float(np.median([v[0] for v in b])) if b else None, w_med=float(np.median([v[1] for v in b])) if b else None)
        res["arms"][arm]["side"] = dict(TF.side_summary([stem(arm, st, c) + "_tv.json" for (x, st, c) in SU if x == arm]) or {},
                                        box=bx, g_steps_mean=float(np.mean([e.get("g_steps", 0) for e in side])) if side else None)
    if all(("g", st, c) in EP and ("tvp", st, c) in EP for st in ("fresh", "main") for c in PRIM):
        same = []
        for st in ("fresh", "main"):
            for c in PRIM:
                Tg, Tt = PA.load_track(stem("g", st, c) + "_track.npz"), PA.load_track(stem("tvp", st, c) + "_track.npz")
                for rg, rt, eg, et in zip(Tg, Tt, EP[("g", st, c)], EP[("tvp", st, c)]):
                    same.append(dict(success_same=eg["success"] == et["success"], steps_same=len(rg["obj"]) == len(rt["obj"]),
                                     obj_maxabs=float(np.abs(rg["obj"] - rt["obj"]).max()) if len(rg["obj"]) == len(rt["obj"]) else None))
        res["g_identity"] = dict(n=len(same), success_same=float(np.mean([s["success_same"] for s in same])),
                                 traj_identical=float(np.mean([s["obj_maxabs"] is not None and s["obj_maxabs"] < 1e-6 for s in same])),
                                 obj_maxabs_med=float(np.median([s["obj_maxabs"] for s in same if s["obj_maxabs"] is not None] or [np.nan])))
    print("== f1m_cue read | reused tvp hashes match: %s of %d" % (sum(hash_ok.values()), len(hash_ok)), flush=True)
    for arm in ("tvp",) + ARMS:
        A, C = res["arms"][arm], res["carry"][arm]
        print("   %-3s N4+N6 %s | known(fresh) %s | N1-3,5(fresh) %s" % (arm, TF._f(A["primary_success"]), TF._f(A["known_fresh"]),
                                                                       TF._f(A["other_new_fresh"])), flush=True)
        print("        per set/task %s" % json.dumps({k: round(v, 3) for k, v in sorted(A["per_set_task"].items())}), flush=True)
        for k, v in C.items():
            print("        carry %-16s %s" % (k, json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})), flush=True)
        if "side" in A:
            s = A["side"]
            print("        side %s" % json.dumps({k: (TF._r(v) if isinstance(v, float) else v) for k, v in s.items() if k not in ("segmenter_iou", "paint")}), flush=True)
            if s.get("paint"):
                print("        paint %s" % json.dumps(s["paint"]), flush=True)
    for arm in ARMS:
        D = res["deltas"][arm]
        print("   %s - tvp: PRIMARY %s -> %s | high share change %s" % (arm, TF._f(D["primary"]), res["verdicts"][arm],
                                                                       None if D["high_share_change"] is None else round(D["high_share_change"], 3)), flush=True)
        print("        %s" % {k: TF._f(v) for k, v in D.items() if k not in ("primary", "high_share_change")}, flush=True)
        bad = {k: h for k, h in res["harness"][arm].items() if h["args_diff"] or not h["episodes_match"]}
        print("        harness vs tvp: %d cells, mismatches %s" % (len(res["harness"][arm]), json.dumps(bad)), flush=True)
    print("   g identity with tvp: %s" % json.dumps(res["g_identity"]), flush=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    os.replace(tmp, out)
    print("wrote", out, flush=True)


MAPPING = ("pi0.5; primary cells N4, N6 on fresh trials 24-43 + main states 4-23 pooled ([2, 40]); per arm X in sb, sf, g: d = X - tvp "
           "paired, tv_frozen.boot2 B=2000 rng 0 (tasks and states drawn jointly), CI 2.5/97.5; CUE FOUND iff lo > 0; NOT THIS CUE iff "
           "hi < .10; else UNRESOLVED. Descriptive: high-carry share (lift_held > .22 m among grasped) per arm and change vs tvp; known "
           "tasks and N1-N3, N5 vs tvp; box sizes; g identity with tvp. sg and h not rolled out (state is not an input of pi05_libero).")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        return client_main()
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        return summary(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("read", "statecheck"))
    ap.add_argument("--dir", default=D_CUE)
    ap.add_argument("--out", default=READ_OUT)
    a = ap.parse_args()
    return statecheck(a) if a.mode == "statecheck" else read(a)


if __name__ == "__main__":
    main()

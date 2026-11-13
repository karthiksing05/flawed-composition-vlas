"""TV-F: the proposal's role views (docs/ARCHITECTURE_PROPOSAL.md 2, parts 1-3) applied at TEST TIME to FROZEN pi0.5 and GR00T, with
no retraining (docs/SCAFFOLD_RESULTS.md 46).  Question: does hiding the non-active objects' identity remove F1 (the held object picks
the destination; the atlas's cross-chunk break at HANDOFF / TRANSPORT / RELEASE) on new pairings, and is that separable from an
AC-VLA-style whole-wrist mask?

The harness is the break atlas's, unchanged (pp_atlas_client.py: fc_n2_client -> libero_client, pp_atlas.Rec's oracle track as the
read's instrument; the VLA served intact by pp_serve_purify.py --mode intact, seed 13; client seed 7; results/flawed/n2_states;
n2_branches_main.json; --bind track).  This file wraps the env once more so every observation from step WAIT on passes through a
front end, and (arm tvp only) wraps the websocket client so the prompt follows the role sequencer.  No other file is edited.

Role sequencer.  lifelong_ood.policy.parts_flow.ProprioHeld with sc_rv's lex threshold (results/flawed/sc_rv_lex.json: the detector
R1 / R2 / R2w use at test), stepped on EVERY env step from the WAIT-th (as ProprioHeld is meant to be stepped and as sc_hp.py steps
it) on the finger qpos, the last executed gripper command and the hand height.  It reads no object pose, contact or identity.
The view follows the detector's current state, as R2's RoleView does: held -> target phase; not held -> object phase (it returns
to the object phase if the hold ends, e.g. after a failed lift).

Arms (prompt = the task's instruction, unchanged, except tvp):
    tv   role views (R2's rule, sc_rv.role_paint): object phase -> every movable object (sc_rv.MOVABLE_OBJ: bowl, cream cheese,
         wine bottle, plate) except the task's object is painted flat magenta; target phase -> every movable except the target,
         the held object included (all four when the target is a fixture).  Fixtures are never painted.  Masks from sc_seg's
         learned U-Net (Segmenter.ent, post-processing on, as sc_seg.py serve / R2l), in BOTH cameras, painted by sc_rv.paint_many
         on the 256 renders (the client then flips and resizes exactly as for the intact arm).
    tvp  tv plus R1's role phrases: the prompt is sc_t16.given(instruction)[held] ("object: the cream cheese" / "target: the plate").
    w    AC-VLA-style control (sc_rv_w.FILL): on every observation where the detector is on, the whole wrist image is replaced by
         grey 128; the agentview is untouched; no silhouettes; prompt unchanged.
    I    (not run here) the atlas's intact rollouts results/flawed/pp_atlas/<model>_<tid>.json: same states, branches, seeds,
         server mode and client flags.
The simulator is read for diagnostics only (segmenter IoU, paint IoU, active-object leak, hold onset against the oracle Held), never
for the paint, the prompt or the hold signal.  The entity list (names) for the class -> entity map comes from the env, as in R2l.

Modes (LIBERO venv, py3.8, EGL, one GPU for client):
    python experiments/tv_frozen.py client --arm tv|tvp|w <pp_atlas_client.py's arguments>       one n2 task per process
        -> <out>.json (libero_client), <out>_track.npz (pp_atlas.Rec), <out>_tv.json (per episode: hold, IoU, paint)
    python experiments/tv_frozen.py summary DIR                                                  per-episode table (smoke)
    python experiments/tv_frozen.py read --phase main|smoke --dir D [--models pi05+groot] [--out X.json]     (CPU)

Registered mapping (read; m3_prereg tag tv_frozen), per model, familiar layout, main states 4-23:
    success matrices [task, state]; deltas paired over the same (task, state) cells; bootstrap B = 2000, numpy default_rng(0),
    tasks and states drawn jointly with replacement as the atlas's estimator draws them; CI = 2.5 / 97.5 percentiles.
    PRIMARY  d = success(tv) - success(I) on N1-N6:  TV WINS iff d > .10 and CI lower bound > 0;  TV LOSES iff CI upper bound < -.10;
             else NO CLEAR DIFFERENCE.
    AC-VLA   d = success(tv) - success(w) on N1-N6, the same rule: SEPARATED FROM AC-VLA iff it is a win; TV LOSES TO W iff the upper
             bound < -.10; else NO CLEAR DIFFERENCE.
Reported with no verdict: tvp - I, tvp - tv, w - I on N1-N6; K1-K4, K6 success under every arm and each arm - I there (the cost on
trained tasks); the atlas's within- and cross-chunk excess (pp_atlas.py's estimator, partners and stage rule, unchanged) and its
verdict per arm; stage shares; segmenter IoU against the simulator per camera and object; paint IoU and active-object leak; hold
onsets against the oracle Held.  The I arm's excess must reproduce results/flawed/pp_atlas_read.json (checked and reported).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

ARMS = ("tv", "tvp", "w")
CAMS = ("agentview", "robot0_eye_in_hand")
KEYS = ("agentview_image", "robot0_eye_in_hand_image")
WAIT, REPLAN = 10, 5
EPS = 0.10
OUT = os.path.join(ROOT, "results", "flawed")
MAIN_DIR = os.path.join(OUT, "tv_frozen")
MAIN_READ = os.path.join(OUT, "tv_frozen_read.json")


def _flip(x):
    return np.ascontiguousarray(np.asarray(x)[::-1, ::-1])


# ---------------------------------------------------------------- the front end (LIBERO venv)
class Front:
    """Per episode: the role sequencer stepped every env step from WAIT, and the arm's view of each observation."""

    def __init__(self, env, arm, obj, tgt, X, seg, sheet=False):
        import sc_rv as RV
        import sc_rv_w as RW
        from lifelong_ood.policy.parts_flow import ProprioHeld
        self.RV, self.fill, self.env, self.arm, self.obj, self.tgt, self.seg = RV, RW.FILL, env, arm, obj, tgt, seg
        self.det = ProprioHeld(X["tw"], X["lift"])
        self.on, self.E, self.names = False, None, None
        self.onsets, self.offsets, self.queries, self.held_queries, self.painted_q = [], [], 0, 0, [0, 0]
        self.seg_acc = {c: {n: [0, 0] for n in RV.MOVABLE_OBJ} for c in CAMS}
        self.paint = {c: dict(inter=0, union=0, act_px=0, act_painted=0) for c in CAMS}
        self.secs, self.sheet, self.frames = 0.0, sheet, []

    def step(self, obs, n, cmd):
        if n < WAIT:
            return obs
        t0 = time.time()
        RV = self.RV
        if self.E is None:
            self.E = RV.entity_tables(self.env)
            self.names = list(self.E["names"])
        was = self.on
        self.on = bool(self.det.step(np.asarray(obs["robot0_gripper_qpos"], np.float64), cmd, float(obs["robot0_eef_pos"][2])))
        row = n - WAIT                                                      # pp_atlas.Rec's row of this observation
        if self.on and not was:
            self.onsets.append(row)
        elif was and not self.on:
            self.offsets.append(row)
        query = row % REPLAN == 0
        if query:
            self.queries += 1
            self.held_queries += int(self.on)
        out = dict(obs)
        raw = [_flip(obs[k]) for k in KEYS]                                 # the policy's orientation
        shown = list(raw)
        if self.arm == "w":
            if self.on:
                w = np.asarray(obs[KEYS[1]])
                out[KEYS[1]] = np.full(w.shape, self.fill, w.dtype)
                shown[1] = np.full(raw[1].shape, self.fill, raw[1].dtype)
                self.painted_q[1] += int(query)
        else:
            ents, act = RV.role_paint(self.obj, self.tgt, self.on)
            ids = [self.names.index(e) for e in ents if e in self.names]
            for j, (cam, key) in enumerate(zip(CAMS, KEYS)):
                p = np.asarray(self.seg.ent(raw[j], cam, self.names))
                img, k = RV.paint_many(raw[j], p, ids)
                if k:
                    out[key] = _flip(img)
                    shown[j] = img
                    self.painted_q[j] += int(query)
                if query:
                    self.score(cam, p, ids, act)
        if self.sheet and query:                                            # smoke contact sheet: raw | shown, agent then wrist
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

    def score(self, cam, p, ids, act):
        """Diagnostic only: learned masks against the simulator's entity map of the same render state."""
        RV = self.RV
        g = np.asarray(RV.seg_px(self.env, self.E, cam))
        for nme in RV.MOVABLE_OBJ:
            if nme not in self.names:
                continue
            i = self.names.index(nme)
            A = self.seg_acc[cam][nme]
            A[0] += int(((g == i) & (p == i)).sum())
            A[1] += int(((g == i) | (p == i)).sum())
        pm = np.zeros(p.shape, bool)
        for i in ids:
            pm |= RV.clean_mask(p == i)
        tm = np.isin(g, ids) if ids else np.zeros(g.shape, bool)
        D = self.paint[cam]
        D["inter"] += int((pm & tm).sum())
        D["union"] += int((pm | tm).sum())
        if act is not None and act in self.names:
            am = g == self.names.index(act)
            D["act_px"] += int(am.sum())
            D["act_painted"] += int((pm & am).sum())

    def record(self, held):
        held = np.asarray(held, bool)
        ofh = int(np.argmax(held)) if held.any() else None
        f0 = self.onsets[0] if self.onsets else None
        return dict(arm=self.arm, onsets=self.onsets, offsets=self.offsets, first_onset=f0, oracle_first_held=ofh,
                    oracle_held_at_onset=None if f0 is None or f0 >= len(held) else bool(held[f0]),
                    queries=self.queries, held_queries=self.held_queries, painted_queries=dict(zip(("agent", "wrist"), self.painted_q)),
                    seg_acc=self.seg_acc if self.arm != "w" else None, paint=self.paint if self.arm != "w" else None, secs_front=self.secs)


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
    import fc_n2_client as FN
    import pp_atlas as PA
    import sc_rv as RV
    LC = FN.LC
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = FN.N2["tasks"][TID]
    assert TASK["tid"] == TID and TASK["layout"] == "familiar", TASK
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    seg = None
    if arm != "w":
        import sc_seg as SG
        seg = SG.load(None, "cuda")
    P = None
    if arm == "tvp":
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
            self.front = Front(self._e, arm, TASK["object"], TASK["target"], X, seg, sheet=sheet and len(SIDE) < 2)
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
                    json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], phrases=P, lex=X,
                                   seg_ckpt=None if seg is None else seg.path, eps=SIDE), fh, indent=1)
                os.replace(tmp, SIDEF)
                print("[tv] %s %s trial %s: onsets %s (oracle first held %s) held queries %d/%d painted %s" % (
                    arm, TASK["key"], rec["trial"], rec["onsets"][:4], rec["oracle_first_held"], rec["held_queries"], rec["queries"],
                    rec["painted_queries"]), flush=True)
                self.front = None
            self._e.close()

    if arm == "tvp":
        Base = LC._wcp.WebsocketClientPolicy

        class PClient(Base):
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
    print("[tv] arm %s task %s %r object %s target %s lex tw %.6f lift %s phrases %s seg %s" % (
        arm, TASK["key"], TASK["language"], TASK["object"], TASK["target"], X["tw"], X["lift"], P, None if seg is None else seg.path), flush=True)
    LC.main()


# ---------------------------------------------------------------- summary (smoke)
def summary(d):
    import pp_atlas as PA
    T = {t["tid"]: t for t in PA.tasks()["tasks"]}
    for f in sorted(x for x in os.listdir(d) if x.endswith("_tv.json")):
        S = json.load(open(os.path.join(d, f), encoding="utf-8"))
        stem = os.path.join(d, f[:-len("_tv.json")])
        if not os.path.exists(stem + ".json"):
            print("== %s: no client json yet" % f, flush=True)
            continue
        R = json.load(open(stem + ".json", encoding="utf-8"))
        E = [e for v in R["tasks"].values() for e in v["episodes"]]
        Tr = PA.load_track(stem + "_track.npz")
        t = T[S["tid"]]
        print("== %s %s %r" % (f, t["key"], t["language"]), flush=True)
        for e, r, s in zip(E, Tr, S["eps"]):
            dd = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
            st = PA.classify(r["held"], dd, bool(e["success"]), e.get("bound"), t["object"])
            print(json.dumps(dict(trial=e["trial"], success=e["success"], stage=st[0], sub=st[1], bound=e.get("bound"), onsets=s["onsets"][:4],
                                  oracle_first_held=s["oracle_first_held"], oracle_held_at_onset=s["oracle_held_at_onset"],
                                  held_q=s["held_queries"], q=s["queries"], painted=s["painted_queries"], prompts=s.get("prompts"),
                                  paint_iou={c: _r(_ratio(s["paint"][c]["inter"], s["paint"][c]["union"])) for c in CAMS} if s["paint"] else None,
                                  leak={c: _r(_ratio(s["paint"][c]["act_painted"], s["paint"][c]["act_px"])) for c in CAMS} if s["paint"] else None,
                                  ms=round(1000 * s["secs_front"] / max(1, s["steps"]), 1))), flush=True)


def _ratio(a, b):
    return None if not b else a / float(b)


def _r(x):
    return None if x is None else round(x, 3)


# ---------------------------------------------------------------- read (CPU)
def boot2(X, Y, B=2000):
    """Paired difference of two [task, state] matrices: point, and the 2.5 / 97.5 percentiles over B joint (task, state) draws."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    rng = np.random.default_rng(0)
    nt, ns = X.shape
    d = float((X - Y).mean())
    v = []
    for _ in range(B):
        ti, si = rng.choice(nt, nt), rng.choice(ns, ns)
        v.append(float((X[ti][:, si] - Y[ti][:, si]).mean()))
    return [d, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot1(X, B=2000):
    X = np.asarray(X, float)
    rng = np.random.default_rng(0)
    nt, ns = X.shape
    v = [float(X[rng.choice(nt, nt)][:, rng.choice(ns, ns)].mean()) for _ in range(B)]
    return [float(X.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def label(d, win, lose):
    if d[0] > EPS and d[1] > 0:
        return win
    if d[2] < -EPS:
        return lose
    return "NO CLEAR DIFFERENCE"


def atlas_est(RATE, NEW, PAIR, ks, PA):
    """pp_atlas.read's estimator, unchanged (copied from pp_atlas.py / pp_atlas_models_read.py, which are not importable as functions)."""
    def excess(names, ti, si):
        v = []
        for i in ti:
            c = NEW[i]
            ko, kt = PAIR[c]
            v.append(sum(RATE[c][n][si] - 0.5 * (RATE[ko][n][si] + RATE[kt][n][si]) for n in names).mean())
        return float(np.mean(v))

    def est(names):
        rng = np.random.default_rng(0)
        e = excess(names, np.arange(len(NEW)), np.arange(len(ks)))
        v = [excess(names, rng.choice(len(NEW), len(NEW)), rng.choice(len(ks), len(ks))) for _ in range(PA.B)]
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        return dict(E=e, ci=[lo, hi], breaks=bool(lo > 0.0 and e > 0.10))
    r = dict(stages={s: est((s,)) for s in PA.STAGES[1:]}, groups={g: est(n) for g, n in PA.GROUP.items()})
    r["verdict"] = {(True, False): "WITHIN-CHUNK", (False, True): "CROSS-CHUNK", (True, True): "BOTH", (False, False): "NEITHER"}[
        (r["groups"]["within"]["breaks"], r["groups"]["cross"]["breaks"])]
    return r


def side_summary(files):
    eps = []
    for f in files:
        if os.path.exists(f):
            eps += json.load(open(f, encoding="utf-8"))["eps"]
    if not eps:
        return None
    out = dict(episodes=len(eps), fired=float(np.mean([e["first_onset"] is not None for e in eps])),
               onset_oracle_held=_ratio(sum(bool(e["oracle_held_at_onset"]) for e in eps), sum(e["oracle_held_at_onset"] is not None for e in eps)),
               onset_minus_oracle_median=None, multiple_onsets=float(np.mean([len(e["onsets"]) > 1 for e in eps])),
               held_query_share=_ratio(sum(e["held_queries"] for e in eps), sum(e["queries"] for e in eps)))
    dl = [e["first_onset"] - e["oracle_first_held"] for e in eps if e["first_onset"] is not None and e["oracle_first_held"] is not None]
    out["onset_minus_oracle_median"] = float(np.median(dl)) if dl else None
    if eps[0].get("seg_acc"):
        seg, paint = {}, {}
        for c in CAMS:
            seg[c] = {}
            for n in eps[0]["seg_acc"][c]:
                a = sum(e["seg_acc"][c][n][0] for e in eps)
                b = sum(e["seg_acc"][c][n][1] for e in eps)
                seg[c][n] = _ratio(a, b)
            T = {k: sum(e["paint"][c][k] for e in eps) for k in ("inter", "union", "act_px", "act_painted")}
            paint[c] = dict(iou=_ratio(T["inter"], T["union"]), active_leak=_ratio(T["act_painted"], T["act_px"]))
        out["segmenter_iou"], out["paint"] = seg, paint
    return out


def read(a):
    import pp_atlas as PA
    n2 = PA.tasks()
    ks = n2["states"][a.phase]
    T = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar"}
    NEW, KNOWN, PAIR = PA.NEW, PA.KNOWN, PA.PAIR
    comps = NEW + KNOWN
    models = a.models.split("+")
    ref = json.load(open(os.path.join(OUT, "pp_atlas_read.json"), encoding="utf-8")) if a.phase == "main" else {}
    res = dict(phase=a.phase, states=ks, dir=a.dir, B=PA.B, eps=EPS, mapping=MAPPING, models={})
    for m in models:
        stems = {"I": (PA.OD, m + "_{tid}")} if a.phase == "main" else {}           # the atlas ran I on main states only
        stems.update({arm: (a.dir, "%s_%s_{tid}" % (m, arm)) for arm in ARMS})
        A = {}
        for arm, (d, st) in stems.items():
            files = [os.path.join(d, st.format(tid=T[c]["tid"])) for c in comps]
            miss = [f for f in files if not (os.path.exists(f + ".json") and os.path.exists(f + "_track.npz"))]
            if miss:
                A[arm] = "MISSING %d of %d tasks" % (len(miss), len(files))
                continue
            RATE, succ, per = {}, {}, {}
            for c, f in zip(comps, files):
                S, X, G, SU = PA.stage_table(f, T[c], ks)
                RATE[c] = {s: np.array([float(x == s) for x in S]) for s in PA.STAGES}
                succ[c] = np.array(SU, float)
                per[c] = {s: float(RATE[c][s].mean()) for s in PA.STAGES}
            atl = atlas_est(RATE, NEW, PAIR, ks, PA)
            new_m = np.stack([succ[c] for c in NEW])
            kn_m = np.stack([succ[c] for c in KNOWN])
            grasp_new = 1.0 - float(np.mean([per[c]["GRASP"] for c in NEW]))
            cross_new = float(np.mean([per[c]["HANDOFF"] + per[c]["TRANSPORT"] for c in NEW]))
            A[arm] = dict(new=new_m, known=kn_m, row=dict(
                success_new=boot1(new_m), success_known=boot1(kn_m), per_task_success={c: float(succ[c].mean()) for c in comps},
                stage_shares_new={s: float(np.mean([per[c][s] for c in NEW])) for s in PA.STAGES},
                grasp_ok_new=grasp_new, cross_stage_share_new=cross_new, per_task_stages=per, atlas=atl,
                side=None if arm == "I" else side_summary([os.path.join(d, st.format(tid=T[c]["tid"])) + "_tv.json" for c in comps])))
        row = {arm: (v if isinstance(v, str) else v["row"]) for arm, v in A.items()}
        D = {}

        def dl(x, y, key="new"):
            if isinstance(A.get(x), dict) and isinstance(A.get(y), dict):
                return boot2(A[x][key], A[y][key])
            return None
        D["tv_minus_I"] = dl("tv", "I")
        D["tv_minus_w"] = dl("tv", "w")
        V = dict(primary=None if D["tv_minus_I"] is None else label(D["tv_minus_I"], "TV WINS", "TV LOSES"),
                 acvla=None if D["tv_minus_w"] is None else label(D["tv_minus_w"], "SEPARATED FROM AC-VLA", "TV LOSES TO W"))
        D["tvp_minus_I"], D["tvp_minus_tv"], D["w_minus_I"] = dl("tvp", "I"), dl("tvp", "tv"), dl("w", "I")
        D["known"] = {"%s_minus_I" % x: dl(x, "I", "known") for x in ARMS}
        chk = None
        if isinstance(A.get("I"), dict) and m in ref:
            g0, g1 = ref[m]["familiar"]["groups"], A["I"]["row"]["atlas"]["groups"]
            chk = all(abs(g0[g]["E"] - g1[g]["E"]) < 1e-9 and all(abs(u - v) < 1e-9 for u, v in zip(g0[g]["ci"], g1[g]["ci"])) for g in g0)
        res["models"][m] = dict(arms=row, deltas=D, verdicts=V, I_reproduces_pp_atlas_read=chk)
        print("== %s: primary %s | AC-VLA %s | I reproduces pp_atlas_read: %s" % (m, V["primary"], V["acvla"], chk), flush=True)
        for arm in ("I",) + ARMS:
            r = row.get(arm)
            if isinstance(r, str) or r is None:
                print("   %-4s %s" % (arm, r), flush=True)
                continue
            g = r["atlas"]["groups"]
            print("   %-4s new %s known %s | grasp ok %.3f cross-stage %.3f | within E %s cross E %s -> %s" % (
                arm, _f(r["success_new"]), _f(r["success_known"]), r["grasp_ok_new"], r["cross_stage_share_new"],
                _f([g["within"]["E"]] + g["within"]["ci"]), _f([g["cross"]["E"]] + g["cross"]["ci"]), r["atlas"]["verdict"]), flush=True)
            if r.get("side"):
                print("        side %s" % json.dumps({k: (_r(v) if isinstance(v, float) else v) for k, v in r["side"].items()
                                                      if k not in ("segmenter_iou", "paint")}), flush=True)
                if r["side"].get("paint"):
                    print("        paint %s seg %s" % (json.dumps(r["side"]["paint"]), json.dumps(r["side"]["segmenter_iou"])), flush=True)
        for k, v in D.items():
            if k != "known":
                print("   %s %s" % (k, _f(v)), flush=True)
        print("   known %s" % {k: _f(v) for k, v in D["known"].items()}, flush=True)
    if a.out:
        out = os.path.abspath(a.out)
        assert not os.path.exists(out), "never overwrite a result file: %s" % out
        tmp = out + ".tmp%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        os.replace(tmp, out)
        print("wrote", out, flush=True)


MAPPING = ("per model, familiar layout; d = success(tv) - success(I) on N1-N6 paired over (task, state), bootstrap B=2000 rng 0, tasks "
           "and states drawn jointly; TV WINS iff d > .10 and CI lo > 0; TV LOSES iff CI hi < -.10; else NO CLEAR DIFFERENCE. "
           "d = tv - w on N1-N6, same rule: SEPARATED FROM AC-VLA iff win; TV LOSES TO W iff hi < -.10; else NO CLEAR DIFFERENCE.")


def _f(v):
    if v is None:
        return "-"
    return "[" + ", ".join("%+.3f" % x for x in v) + "]"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        return client_main()
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        return summary(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("read",))
    ap.add_argument("--phase", default="main", choices=("smoke", "main"))
    ap.add_argument("--dir", default=MAIN_DIR)
    ap.add_argument("--models", default="pi05+groot")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    read(a)


if __name__ == "__main__":
    main()

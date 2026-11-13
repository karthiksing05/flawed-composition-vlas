"""F7RA: F7R (experiments/fx_r.py, SR 45.2-45.3) with the role view as docs/ARCHITECTURE_PROPOSAL.md specifies it: EVERY
non-referent movable object painted as a silhouette in both cameras, not only the distractor bowl.  Frozen VLAs, the fx harness
(fx_client.py) and F7R's resolver, tracking and distractor-bowl paint (fx_r.Front) are imported unchanged; nothing hashed in an
earlier prereg is edited.

Why.  On NG ("put the bowl that is [not] on the plate on the stove") every model fails the unique-name plate-bowl control: pi0,
SmolVLA and GR00T grasp the cream cheese, so F7R (which paints only the other bowl) caps at ~.50.  The proposal's role view
paints every non-active movable object (sc_rv's rview2 rule), the cream cheese included.  This tests whether that removes the habit.

Arms (one fx task per client process):
    f7rA  on the relational tasks (SP relL / relR = tids 5, 6; NG pos / neg = 10, 11): F7R exactly (template parser, learned-
          segmenter resolver at step WAIT, unique-name prompt, the distractor bowl's tracked mask painted), PLUS, in both cameras on
          every observation from step WAIT on, every pixel the learned segmenter (sc_seg, classes_of, post_=False as F7R) labels
          cream_cheese_1, wine_bottle_1 or plate_1 is painted with sc_rv.paint_many (rview2's magenta and mask cleaning).
    uniA  on F7's unique-name controls (SP uniL / uniR = 7, 8; NG uniP / uniT = 12, 13: the other bowl removed and hidden, prompt
          = the control's own wording, which is F7R's unique wording): the same object paint; the bowl class is never painted.
          No resolver (one bowl).  This measures whether paint-all lifts the named-referent ceiling that capped F7R.
  The movable objects of the scene are sc_rv.MOVABLE_OBJ (bowl, cream cheese, wine bottle, plate) plus the second bowl.  The
  referent bowl is never painted.  Target rule: the plate is painted unless it is the instruction's target (SP: "... on the
  plate"), because the unique-name prompt names it as the place role and rview2 leaves a movable target unpainted while it is
  the active role; without F7R's lack of a role sequencer that is the whole episode.  On NG the target is the stove (a fixture,
  never painted), so on NG this is literally every movable object except the referent.
  Diagnostics only (never the paint or the prompt): per object, IoU of the painted class mask against the simulator's mask at
  query steps, and the referent's painted share (leak), as fx_r.Front.score does for the distractor bowl.

Modes (LIBERO venv, py3.8, EGL, one GPU, except read):
    python experiments/fx_ra.py client --arm f7rA|uniA <fx_client.py's arguments>
        -> fx_client's <out>.json, <out>_track.npz, and <out>_fxa.json (per episode: parse, resolve, paint diagnostics)
    python experiments/fx_ra.py read --phase smoke|main [--models pi05+groot+pi0+xvla+smolvla] [--out X.json]
        [--roll D --rroll D --fxjson F --aroll D]   (smoke: one path each, applied to every model; main: per-model defaults)

Mapping (read), per model and family (cells: NG for pi05, groot, pi0, xvla, smolvla; SP for pi0, groot).  R = mean success over
the two relational tasks, U = over the two unique-name tasks.  delta = R(f7rA) - R(relational, the fx / fx_more frozen run on the
same states and seeds), bootstrapped paired over states (fx_read.boot: B = 2,000, seed 0).  First matching row:
    NO POWER    the fx reference < .60 (the U gate of fx_r is dropped: the ceiling is measured by uniA, not gated)
    FIXED       delta > .10 with CI lower bound > 0, and C(f7rA)'s CI lower bound > .75
    NOT FIXED   delta's CI upper bound < .10
    PARTIAL     otherwise
Ceiling, per cell: ceil = U(uniA) - U(original unique-name, fx / fx_more), paired over states:
    CEILING LIFTED  ceil > .10 with CI lower bound > 0;   CEILING HOLDS  ceil's CI upper bound < .10;   else UNRESOLVED
Reported with no verdict: f7rA - F7R (paint-all's share), per-task success, what was grasped, resolver accuracy, paint IoU / leak.
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
import fx_r as F7R                                                          # noqa: E402

B1, B2, PLATE = F7R.B1, F7R.B2, F7R.PLATE
CAMS = F7R.CAMS
ARMS = ("f7rA", "uniA")
PAINT_OBJ = ("cream_cheese_1", "wine_bottle_1", PLATE)
CELLS = dict(pi05=("NG",), groot=("NG", "SP"), pi0=("NG", "SP"), xvla=("NG",), smolvla=("NG",))
EPS, CMIN, REFMIN = 0.10, 0.75, 0.60
FL = os.path.join(ROOT, "results", "flawed")
MAIN_DIRS = {m: (os.path.join(FL, "fx", "roll"), os.path.join(FL, "fx", "roll_r"), os.path.join(FL, "fx_read_main.json"))
             for m in ("pi05", "groot")}
MAIN_DIRS.update({m: (os.path.join(FL, "fx_more", "roll"), os.path.join(FL, "fx_more", "roll_r"), os.path.join(FL, "fx_more_read_main.json"))
                  for m in ("pi0", "xvla", "smolvla")})
AROLL_MAIN = os.path.join(FL, "fx_ra", "roll")
UNI_RX = re.compile(r"^put the bowl on the (plate|stove)$")


class FrontA(F7R.Front):
    """fx_r.Front (resolve, distractor tracking, distractor score) plus the object paint."""

    def __init__(self, env, arm, parsed, ckpt=""):
        super().__init__(env, "f7r" if arm == "f7rA" else "uni", parsed, ckpt)
        self.arm, self.paint_on = arm, True
        self.objs = [n for n in PAINT_OBJ if not (n == PLATE and parsed["target"] == "plate")]
        self.cobj = [self.seg.classes.index(n) for n in self.objs]
        self.diagA = {c: dict({n: dict(inter=0, union=0, true_px=0, painted_px=0) for n in self.objs},
                              ref_painted=0, ref_px=0, queries=0) for c in CAMS}

    def step(self, obs, n, wait):
        if n < wait:
            return obs
        if self.res is None:
            if self.arm == "uniA":
                self.res = dict(ok=True, how="unique", relation="unique")
            else:
                self.resolve(obs)
        if not self.res.get("ok"):                                           # as F7R: an unresolved episode runs unpainted
            return obs
        RV = self.RV
        imgs = np.stack([RV.flip(obs["agentview_image"]), RV.flip(obs["robot0_eye_in_hand_image"])])
        cls = self.seg.classes_of(imgs, [0, 1], post_=False)
        query = (n - wait) % F7R.REPLAN == 0
        out = dict(obs)
        npx = []
        for j, (cam, key) in enumerate(zip(CAMS, ("agentview_image", "robot0_eye_in_hand_image"))):
            seg = np.full(cls[j].shape, -1, np.int8)
            for k, c in enumerate(self.cobj):
                seg[cls[j] == c] = k + 1
            dm = None
            if self.arm == "f7rA":
                dm = self.dmask(cls[j], self.cloud(cam), update=(j == 0))
                seg[dm] = 0
            painted, k = RV.paint_many(imgs[j], seg, list(range(len(self.cobj) + 1)))
            npx.append(k)
            if k:
                out[key] = RV.flip(painted)
            if query:
                if dm is not None:
                    self.score(cam, RV.clean_mask(dm))
                self.scoreA(cam, seg, dm)
        if query:
            self.trace.append([int(n), int(npx[0]), int(npx[1])])
        return out

    def scoreA(self, cam, seg, dm):
        """Diagnostic only: each painted object's cleaned class mask against the simulator's mask; the referent's painted share."""
        g = np.asarray(self.RV.seg_px(self.env, self.E, cam))
        names = self.E["names"]
        D = self.diagA[cam]
        D["queries"] += 1
        allp = np.zeros(seg.shape, bool)
        for k, nm in enumerate(self.objs):
            pm = self.RV.clean_mask(seg == k + 1)
            allp |= pm
            t = g == names.index(nm)
            d = D[nm]
            d["inter"] += int((pm & t).sum())
            d["union"] += int((pm | t).sum())
            d["true_px"] += int(t.sum())
            d["painted_px"] += int(pm.sum())
        if dm is not None:
            allp |= self.RV.clean_mask(dm)
        t1 = g == names.index(B1)
        D["ref_painted"] += int((allp & t1).sum())
        D["ref_px"] += int(t1.sum())

    def record(self):
        r = super().record()
        r.update(arm=self.arm, objs=self.objs, diagA=self.diagA)
        return r


# ---------------------------------------------------------------- client: fx_r.client_main's wrapping, arm-aware
def client_main():
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
    import fx_client as FC_
    import time
    LC, TASK = FC_.LC, FC_.TASK
    orig_lang = TASK["language"]
    if arm == "f7rA":
        assert TASK["diag"] == "F7" and TASK["cond"] in ("relL", "relR", "pos", "neg"), TASK["key"]
        parsed = F7R.parse(orig_lang)
        assert parsed is not None, "F7R refuses an off-template instruction: %r" % orig_lang
    else:
        assert TASK["diag"] == "F7" and TASK["cond"] in ("uniL", "uniR", "uniP", "uniT"), TASK["key"]
        mm = UNI_RX.match(" ".join(orig_lang.lower().strip().rstrip(".").split()))
        assert mm, orig_lang
        parsed = dict(relation="unique", landmark=None, target=mm.group(1), unique=orig_lang)
    FC_.FX["tasks"][FC_.TID]["language"] = parsed["unique"]                   # in memory only (uniA: unchanged)
    br = json.load(open(FC_._arg("--branches"), encoding="utf-8"))
    side = FC_.OUTF.replace(".json", "_fxa.json")
    recs = []
    inner_factory = LC.OffScreenRenderEnv

    class RAEnv:
        def __init__(self, env):
            self._e, self.front, self.n, self.secs = env, None, 0, 0.0

        def __getattr__(self, k):
            return getattr(self._e, k)

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            self.n, self.secs = 0, 0.0
            self.front = FrontA(self._e, arm, parsed, ckpt)
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
                                   objs=self.front.objs, seg_ckpt=self.front.seg.path, eps=recs), fh, indent=1)
                os.replace(tmp, side)
                r = rec["resolve"] or {}
                print("[fxa] %s %s trial %s: resolved %s how=%s correct=%s; painted (agent, wrist) at queries %s" % (
                    arm, TASK["key"], rec["trial"], r.get("ok"), r.get("how"), r.get("resolver_correct"),
                    [t[1:] for t in rec["trace"][:6]]), flush=True)
                self.front = None
            self._e.close()

    LC.OffScreenRenderEnv = lambda *a, **k: RAEnv(inner_factory(*a, **k))
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "fx episodes need --fresh and --branches"
    assert FC_._arg("--suite") == FC_.FX["suite"] and "," not in FC_._arg("--tasks"), "one fx task per process"
    if "--max-steps" not in sys.argv:
        sys.argv += ["--max-steps", str(TASK["max_steps"])]
    print("[fxa] arm %s task %s %r -> prompt %r, parse %s, painted objects %s" % (
        arm, TASK["key"], orig_lang, parsed["unique"], parsed, [n for n in PAINT_OBJ if not (n == PLATE and parsed["target"] == "plate")]),
        flush=True)
    LC.main()


# ---------------------------------------------------------------- read (CPU)
def side_recs(d, model, t):
    fn = os.path.join(d, "%s_%d_fxa.json" % (model, t["tid"]))
    return json.load(open(fn, encoding="utf-8"))["eps"] if os.path.exists(fn) else []


def paintA_summary(recs):
    out = {}
    for c in CAMS:
        rs = [r["diagA"][c] for r in recs if r and r.get("diagA")]
        if not rs:
            out[c] = None
            continue
        objs = [k for k in rs[0] if isinstance(rs[0][k], dict)]
        o = {}
        for nm in objs:
            T = {k: sum(r[nm][k] for r in rs) for k in ("inter", "union", "true_px", "painted_px")}
            o[nm] = dict(iou=None if not T["union"] else T["inter"] / T["union"],
                         recall=None if not T["true_px"] else T["inter"] / T["true_px"])
        rp, rx = sum(r["ref_painted"] for r in rs), sum(r["ref_px"] for r in rs)
        o["referent_leak"] = None if not rx else rp / rx
        o["queries"] = sum(r["queries"] for r in rs)
        out[c] = o
    return out


def grasp_counts(D, ks, states):
    c = {}
    for k in ks:
        for st in states:
            e = (D.get(k) or {}).get(st)
            g = "missing" if e is None else str(e.get("grasped"))
            c[g] = c.get(g, 0) + 1
    return c


def read(a):
    import fx_read as FR
    import fx_tasks as FT
    meta = json.load(open(FT.META, encoding="utf-8"))
    states = meta["states"][a.phase]
    n = len(states)
    T = {t["key"]: t for t in meta["tasks"]}
    models = a.models.split("+")
    mean = lambda x, i: float(np.nanmean(x[i]))                               # noqa: E731
    out = dict(phase=a.phase, states=states, B=FR.B, mapping=MAPPING, cells={})
    for m in models:
        if a.phase == "main":
            roll, rroll, fxjson = MAIN_DIRS[m]
            aroll = a.aroll or AROLL_MAIN
        else:
            roll, rroll, fxjson, aroll = a.roll, a.rroll, a.fxjson, a.aroll
        fxj = json.load(open(fxjson, encoding="utf-8")) if fxjson and os.path.exists(fxjson) else {}
        out["cells"][m] = {}
        for F in CELLS[m]:
            ra, rb, ua, ub = F7R.FAMS[F]
            kr = ["F7_%s_%s" % (F, c) for c in (ra, rb)]
            ku = ["F7_%s_%s" % (F, c) for c in (ua, ub)]
            fxrow = fxj.get("F7", {}).get(m, {}).get(F) or {}
            ref = fxrow.get("ref")
            row = dict(sources=dict(roll=roll, rroll=rroll, fxjson=fxjson, aroll=aroll), ref=ref,
                       fx_read={k: fxrow.get(k) for k in ("verdict", "U", "R", "C")})
            L = dict(rel={k: FR.load_task(roll, m, T[k], states) for k in kr},
                     uni={k: FR.load_task(roll, m, T[k], states) for k in ku},
                     f7r={k: FR.load_task(rroll, m + "_f7r", T[k], states) for k in kr},
                     f7rA={k: FR.load_task(aroll, m + "_f7rA", T[k], states) for k in kr},
                     uniA={k: FR.load_task(aroll, m + "_uniA", T[k], states) for k in ku})
            miss = ["%s:%s" % (arm, k) for arm, D in L.items() for k, v in D.items() if v is None]
            row["missing"] = miss
            S = {arm: {k: FR.col(v, states) for k, v in D.items() if v is not None} for arm, D in L.items()}

            def two(arm):
                s = S[arm]
                if len(s) < 2:
                    return None
                ka, kb = sorted(s)
                return lambda i: (mean(s[ka], i) + mean(s[kb], i)) / 2
            f = {arm: two(arm) for arm in L}
            for arm in L:
                if f[arm] is not None:
                    row[arm] = dict(mean=FR.boot(f[arm], n), success={k: mean(v, np.arange(n)) for k, v in S[arm].items()},
                                    grasped=grasp_counts(L[arm], list(L[arm]), states))
            for arm in ("rel", "f7r", "f7rA"):
                if f[arm] is not None:
                    row[arm]["C"] = F7R.c_of(L[arm], kr, states, FR)
            pairs = dict(delta=("f7rA", "rel"), ceil=("uniA", "uni"), f7rA_minus_f7r=("f7rA", "f7r"), f7r_minus_rel=("f7r", "rel"))
            for nm, (x, y) in pairs.items():
                if f[x] is not None and f[y] is not None:
                    row[nm] = FR.boot(lambda i, x=x, y=y: f[x](i) - f[y](i), n)
            sides = {arm: [r for k in (kr if arm == "f7rA" else ku) for r in side_recs(aroll, "%s_%s" % (m, arm), T[k])] for arm in ARMS}
            row["resolver_f7rA"] = F7R.resolver_summary(sides["f7rA"])
            row["paint_bowl_f7rA"] = F7R.paint_summary(sides["f7rA"]) if sides["f7rA"] else None
            row["paint_objs"] = {arm: paintA_summary(sides[arm]) for arm in ARMS}
            # verdicts
            d, C = row.get("delta"), (row.get("f7rA") or {}).get("C")
            if ref is None or ref < REFMIN:
                v = "NO POWER"
            elif d is None or C is None:
                v = "MISSING"
            elif d[1] is not None and d[0] > EPS and d[1] > 0 and C[1] is not None and C[1] > CMIN:
                v = "FIXED"
            elif d[2] is not None and d[2] < EPS:
                v = "NOT FIXED"
            else:
                v = "PARTIAL"
            row["verdict"] = v
            c = row.get("ceil")
            if c is None:
                cv = "MISSING"
            elif c[1] is not None and c[0] > EPS and c[1] > 0:
                cv = "CEILING LIFTED"
            elif c[2] is not None and c[2] < EPS:
                cv = "CEILING HOLDS"
            else:
                cv = "UNRESOLVED"
            row["ceiling_verdict"] = cv
            out["cells"][m][F] = row
    js = json.dumps(out, indent=1)
    if a.out:
        assert not os.path.exists(a.out), "refusing to overwrite %s" % a.out
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(js)
    for m in models:
        for F in CELLS[m]:
            r = out["cells"][m][F]
            g = lambda arm: F7R.fmt((r.get(arm) or {}).get("mean"))       # noqa: E731
            print("FXA %s %s: %s | %s | ref %s | missing %s" % (m, F, r["verdict"], r["ceiling_verdict"], r["ref"], r["missing"]))
            print("    U %s  uniA %s  R(rel) %s  F7R %s  f7rA %s" % (g("uni"), g("uniA"), g("rel"), g("f7r"), g("f7rA")))
            print("    delta %s  C(f7rA) %s  ceil %s  f7rA-F7R %s" % (F7R.fmt(r.get("delta")), F7R.fmt((r.get("f7rA") or {}).get("C")),
                                                                  F7R.fmt(r.get("ceil")), F7R.fmt(r.get("f7rA_minus_f7r"))))
            for arm in ("uni", "uniA", "f7r", "f7rA"):
                if r.get(arm):
                    print("    %-5s success %s grasped %s" % (arm, {k: round(v, 3) for k, v in r[arm]["success"].items()}, r[arm]["grasped"]))
            print("    resolver %s" % r["resolver_f7rA"])
            print("    paint %s" % json.dumps(r["paint_objs"], default=str)[:1500])


MAPPING = ("per model and family (NG: pi05 groot pi0 xvla smolvla; SP: pi0 groot), arm f7rA; delta = R(f7rA) - R(relational, fx/fx_more) "
           "paired over states, fx_read.boot B=2000 seed 0; first match: NO POWER iff fx ref < .60; FIXED iff delta > .10, delta CI lo > 0 "
           "and C(f7rA) CI lo > .75; NOT FIXED iff delta CI hi < .10; else PARTIAL.  Ceiling: ceil = U(uniA) - U(fx unique-name) paired; "
           "CEILING LIFTED iff ceil > .10 and CI lo > 0; CEILING HOLDS iff CI hi < .10; else UNRESOLVED.  f7rA - F7R: no verdict.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        return client_main()
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("read",))
    ap.add_argument("--phase", default="main", choices=("smoke", "main"))
    ap.add_argument("--out", default="")
    ap.add_argument("--roll", default="")
    ap.add_argument("--rroll", default="")
    ap.add_argument("--fxjson", default="")
    ap.add_argument("--aroll", default="")
    ap.add_argument("--models", default="pi05+groot+pi0+xvla+smolvla")
    a = ap.parse_args()
    read(a)


if __name__ == "__main__":
    main()

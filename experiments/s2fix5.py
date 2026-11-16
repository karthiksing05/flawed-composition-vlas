"""S2FIX5 (docs/SCAFFOLD_RESULTS.md 66; m3_prereg tag s2_fix5): SR 50's role views in front of SR 63's named-mug hand-over with
perception (arm Hp), on NEW states, layout O, tasks K1, K2, N1, N2 of scene 2 (LIBERO-10 LIVING_ROOM_SCENE5), on frozen pi0.5
(PRIMARY) and pi0 (SECONDARY).

Why: under Hp the N2 limit is the VLA's first grasp: it takes the white mug, which N2 does not name, and after SR 64's put-back it
takes the same mug again (240 of 240).  SR 50's role views (tvp) moved pi0.5's first grasp to the named mug on .533 of N2, but the
VLA then carried it to its trained plate (N2 0).  Hypothesis: the views fix the grasp and Hp fixes the place, so views + Hp lifts N2.

ARM Hp: SR 63's arm, run through SR 63's client unchanged (`s2fix3.py client --arm Hp`).

ARM Hv (this file, `client --arm Hv`; PRIMARY): SR 50's tvp front end in front of Hp.
  - The env is SR 63's Hp env (s2fix3.client_main, run UNCHANGED: this client only wraps the env class it installs and the websocket
    client), wrapped once more by VEnv.  Hp's env sees the RAW observation: its OWLv2 perception, hold detector, gate rule, target
    registration and P are Hp's, byte for byte.
  - VEnv passes every observation the env returns through tv_frozen.Front (SR 50's front end, unchanged: ProprioHeld at sc_rv_lex's
    threshold, R2's role_paint with this scene's movables = the three mugs and two plates), stepped on the EXECUTED gripper command
    (Rec2's logged a6: P's command while P acts, the VLA's otherwise).  Object phase (no hold): every movable except the task's mug is
    painted magenta in both cameras; target phase (hold): every movable except the target plate.  The VLA receives only the painted
    observation; the detector and P never see it.
  - Masks: the SIMULATOR's entity maps (s2.SimSeg, as SR 50; PRIVILEGED: the learned segmenter has no mug class, SR 50).
  - Phrases (tvp): the prompt the VLA receives is "object: <the task's mug>" while the view detector is off and "target: <the target
    plate>" while it is on (s2_tasks PHRASE; SR 50's format).  It is the only change to the prompt.
  - While P acts the VLA's actions are discarded (Hp's rule), so the views matter only before the hand-over and after a P abort.
ARM Hv0 (SECONDARY, pi0.5 only): Hv without the phrases (SR 50's tv in front of Hp; prompt unchanged).
END Hv.

Which views: SR 50's variant with the higher N2 named-mug-first share on pi0.5 is tvp (.533; tv .500).  tvp also cost K2 more in SR 50
(.767 vs .933), so tv + Hp runs as Hv0, secondary.
What Hv reads beyond Hp: the simulator entity masks of the five movables, in both cameras, at every step (privileged), and the role
phrases (built from the task's object and target, which Hp already reads).  The simulator oracle is logged, never fed.
Not run: Hvp (OWLv2-perceived views).  OWLv2 gives boxes, not silhouettes; a box paint is a different front end from SR 50's and
s2fix3_det returns no masks.  GR00T (optional; NOT TESTABLE on every earlier scene-2 state set, SR 50/63/64).

New states: LIBERO's sampler on K1's BDDL, seeds 960000 + k (none equal to an SR 50-64 state), k 0-3 smoke, 4-63 main; branches
seed_base 42000000 + 1000 k.  Servers / clients: SR 63's (pi0.5: pp_serve_purify --mode intact --comp= --seed 13; pi0:
serve_guided --mode ours --seed 13 with BACKBONE_CONFIG=pi0_libero), client seed 7, --bind track.

Modes (LIBERO venv, py3.8):
    states                                     (EGL)   -> results/flawed/s2fix5/O/s2_living_<tid>.npy, branches_{smoke,main}.json
    client --arm Hv|Hv0 <s2.py client args>    (needs the OWLv2 sidecar on $S2FIX3_DET_PORT)
    ref --tid T --branches B --out-dir D       (CPU, C1)
    summary DIR                                (smoke)
    read [DIR --selftest --out X]              (CPU)
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import s2                                                                   # noqa: E402
import s2cue                                                                # noqa: E402
import s2fix3 as F3                                                         # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
CD = os.path.join(OUT, "s2fix5")
ROLL = os.path.join(OUT, "s2fix5_roll")
READ = os.path.join(OUT, "s2fix5_read.json")
W, Y, R = s2.MUGS
PL, PR = s2.PLATES
BASE, BR_BASE, N_STATES = 960000, 42000000, 64
SMOKE = list(range(0, 4))
MAIN = list(range(4, 64))
TIDS = (0, 1, 2, 3)                                                         # 0 K1, 1 K2, 2 N1, 3 N2
NI_MARGIN = F3.NI_MARGIN
FLOOR = F3.FLOOR
BAR = 0.50                                                                  # SR 54's bar on N1+N2
RUNS = dict(pi05=("I", "Hp", "Hv", "Hv0"), pi0=("I", "Hp", "Hv"))
PRIMARY_MODEL = "pi05"
VIEW_ARMS = dict(Hv=True, Hv0=False)                                        # arm -> phrases on
MAPPING = (
    "NEW states (LIBERO sampler on K1's BDDL, seeds 960000 + k, none equal to an SR 50-64 state), main k = 4-63 (60), branches "
    "seed_base 42000000 + 1000 k; layout O; tasks K1, K2, N1, N2; client seed 7, --bind track.  Models: frozen pi0.5 (PRIMARY; "
    "pp_serve_purify --model pi05 --mode intact --comp= --seed 13) and frozen pi0 (SECONDARY; serve_guided --mode ours --seed 13, "
    "BACKBONE_CONFIG=pi0_libero).  Arms: I (s2.py client), Hp (SR 63's arm, s2fix3.py client --arm Hp, unchanged), Hv (SR 50's tvp "
    "front end, simulator masks, in front of Hp: s2fix5.py client --arm Hv), and on pi0.5 only Hv0 (SR 50's tv, no phrases, in front "
    "of Hp; SECONDARY).  Success = the BDDL goal.  Matrices [task, state].  Paired bootstrap over states, tasks fixed "
    "(s2.boot2_states, B = 2000, numpy default_rng(0); CI = 2.5 / 97.5).  Per model, registered labels: (V1) Hv - Hp on N2: VIEWS "
    "HELP iff lo > 0; VIEWS HURT iff hi < 0; else NO CLEAR DIFFERENCE.  (V2, SR 62's rule) dN = Hv - I on N1+N2, dK = Hv - I on "
    "K1+K2: FIX WORKS iff dN lo > 0 AND dK lo > -.05 (NON-INFERIOR at margin -.05); PARTIAL iff exactly one holds; FAILS iff "
    "neither.  (V3) BAR: MEETS BAR iff Hv's N1+N2 success >= .50 AND dN lo > 0 (SR 54's FIXED); else BELOW BAR.  PRIMARY: V1, V2, V3 "
    "on pi0.5.  SR 50's floor on every model: TESTABLE iff its I K1+K2 success >= .70 on these states; a NOT TESTABLE model's labels "
    "read NOT TESTABLE with the rule's outcome descriptive.  MECHANISM (registered read, descriptive label): on N2, the share of "
    "episodes whose FIRST detector onset holds the named (yellow-and-white) mug by the logged simulator rule (events[0].sim_held; an "
    "episode with no onset counts as not named), Hv vs Hp, with s2cue.boot_share CIs and the paired state-bootstrap difference; also "
    "the same share on K2, the share by the perceived held mug, and N2 success given a named / unnamed first hold.  SECONDARY: Hp's "
    "FIX label per model (SR 63 replication); Hv - Hp on N1+N2, K1+K2 and K2 (VIEWS GAIN iff lo > 0, VIEWS COST iff hi < 0, else NO "
    "CLEAR DIFFERENCE); on pi0.5 V1-V3 for Hv0 and Hv - Hv0 on N2 and on K1+K2.  Descriptive: perception accuracy (SR 63's "
    "accuracy() on the Hp / Hv / Hv0 side files), front-end diagnostics (onsets, painted queries), final plate of each mug (s2fix2."
    "describe), episode length.  C1: the reference planner (s2.plan_episode via s2cue3.ref_eps, unchanged) on the main states; MET "
    "iff its N1-N2 success >= .80.")


def fsha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------- states (LIBERO venv, EGL)
def states(a):
    import s2_tasks as T2
    od = os.path.join(CD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))

    def draw(k):                                                            # s2_tasks.build's rule with a new base
        np.random.seed(BASE + k)
        env.seed(BASE + k)
        env.reset()
        return env.sim.get_state().flatten()
    S = np.stack([draw(k) for k in range(N_STATES)])
    env.close()
    old = [np.load(os.path.join(s2.SD, "states", "%s_0.npy" % s2.SUITE))] + [
        np.load(os.path.join(OUT, d, L, "%s_0.npy" % s2.SUITE)) for d, L in
        (("s2cue3", "O"), ("s2cue4", "O"), ("s2cue5", "O"), ("s2fix2", "O"), ("s2fix2", "PD2"), ("s2fix3", "O"), ("s2fix4", "O"))]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("same as an SR 50-64 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    for ph, ks in (("smoke", SMOKE), ("main", MAIN)):
        with open(os.path.join(CD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE + 1000 * k, at=0, variant="s2") for k in ks], fh, indent=1)
    print("[s2fix5 states] %d new states (seed %d + k), none equal to an SR 50-64 state; wrote %s" % (len(S), BASE, CD), flush=True)


# ---------------------------------------------------------------- client: arms Hv / Hv0 (LIBERO venv)
def client_main():
    def take(name, default=None):
        if name in sys.argv:
            i = sys.argv.index(name)
            v = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return v
        return default
    assert sys.argv[1] == "client"
    arm = take("--arm")
    assert arm in VIEW_ARMS, arm
    import libero_client as LC
    import sc_rv as RV
    import tv_frozen as TF
    assert TF.WAIT == s2.WAIT
    M = s2.meta()
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = M["tasks"][TID]
    assert TASK["tid"] == TID and len(TASK["atoms"]) == 1
    RV.MOVABLE_OBJ = s2.MOVABLE                                             # this scene's movables (role_paint and Front read it)
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    P = list(TASK["phrases"]) if VIEW_ARMS[arm] else None
    SIDEF = OUTF.replace(".json", "_tv.json")
    SIDE, CUR = [], {}
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))

    class VEnv:
        """SR 50's front end around SR 63's Hp env: the inner env (perception, gate, P) sees the raw observation; the VLA sees the
        painted one."""

        def __init__(self, inner):
            self._i, self.front = inner, None
            CUR["env"] = self

        def __getattr__(self, k):
            return getattr(self._i, k)

        def set_init_state(self, s):
            obs = self._i.set_init_state(s)
            self.front = TF.Front(self._i._e, "tv", TASK["object"], TASK["target"], X, s2.SimSeg(self._i._e))
            self.front.prompts = dict(object=0, target=0)
            CUR["env"] = self
            return obs

        def step(self, a):
            r = self._i.step(a)
            if self.front is None:
                return r
            n = self._i.n
            cmd = float(self._i.rows["a6"][-1]) if n >= s2.WAIT else float(a[6])   # the EXECUTED gripper command (P's while P acts)
            obs = self.front.step(r[0], n, cmd)
            return (obs,) + tuple(r[1:])

        def close(self):
            if self._i.n > 0 and self.front is not None:
                i = len(SIDE)
                rec = self.front.record(self._i.rows["held"])
                rec.update(arm=arm, trial=int(br[i]["trial"]) if i < len(br) else None, steps=self._i.n, tid=TID, key=TASK["key"],
                           prompts=self.front.prompts, p_steps=self._i.h.get("p_steps"))
                SIDE.append(rec)
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], phrases=P, lex=X, masks="simulator",
                                   movable=list(s2.MOVABLE), inner="s2fix3 Hp client (unchanged)", eps=SIDE), fh, indent=1)
                os.replace(tmp, SIDEF)
                print("[s2fix5] %s %s trial %s: view onsets %s (oracle first held %s) painted %s prompts %s p_steps %s" % (
                    arm, TASK["key"], rec["trial"], rec["onsets"][:4], rec["oracle_first_held"], rec["painted_queries"], rec["prompts"],
                    rec["p_steps"]), flush=True)
            self.front = None
            self._i.close()

    if P is not None:
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

    orig_main = LC.main

    def main_v():                                                           # s2fix3.client_main has just installed its Hp env
        H_ = LC.OffScreenRenderEnv
        LC.OffScreenRenderEnv = lambda *a, **k: VEnv(H_(*a, **k))
        return orig_main()
    LC.main = main_v
    print("[s2fix5] arm %s = SR 50 %s front end (simulator masks) around SR 63's Hp client; task %s %r phrases %s lex %s" % (
        arm, "tvp" if P else "tv", TASK["key"], TASK["language"], P, (X["tw"], X["lift"])), flush=True)
    sys.argv += ["--arm", "Hp"]
    F3.client_main()


# ---------------------------------------------------------------- reference planner (C1; LIBERO venv, CPU)
def ref(a):
    import s2cue3
    M_ = s2.meta()
    t = M_["tasks"][a.tid]
    assert t["tid"] == a.tid
    os.makedirs(a.out_dir, exist_ok=True)
    fn = os.path.join(a.out_dir, "ref_O_%d.json" % a.tid)
    assert not os.path.exists(fn), "exists: " + fn
    S = np.load(os.path.join(CD, "O", "%s_%d.npy" % (s2.SUITE, a.tid)))
    br = {int(b["trial"]): int(b["seed_base"]) for b in json.load(open(a.branches, encoding="utf-8"))}
    kp, eps = s2cue3.ref_eps(S, t, t["target"], sorted(br), lambda k: br[k])
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(layout="O", tid=a.tid, key=t["key"], kp=kp, eps=eps), fh, indent=1)


# ---------------------------------------------------------------- read helpers (CPU)
def ep_steps(RD, model, arm, tid):
    Rj = json.load(open(F3.stem_of(RD, model, arm, tid) + ".json", encoding="utf-8"))
    return [int(e["steps"]) for v in Rj["tasks"].values() for e in v["episodes"]]


def first_hold(run, c, obj):
    """-> (sim_named [n], perceived_named [n], has_onset [n]) for task c: first detector onset's held mug == the task's mug."""
    H = run["side"][c]
    if H is None:
        return None
    f = [h["events"][0] if h.get("events") else None for h in H]
    return (np.array([float(e is not None and e["sim_held"] == obj) for e in f]),
            np.array([float(e is not None and e["held"] == obj) for e in f]),
            np.array([float(e is not None) for e in f]))


def mechanism(runs, arms):
    M_ = s2.meta()
    T = {t["key"]: t for t in M_["tasks"]}
    out = {}
    for c in ("N2", "K2", "N1", "K1"):
        o = T[c]["object"]
        d = {}
        ind = {}
        for arm in arms:
            run = runs.get(arm)
            if run is None or isinstance(run, str) or run["side"][c] is None:
                continue
            sn, pn, ho = first_hold(run, c, o)
            ind[arm] = sn
            succ = run["succ"][c]
            one = np.ones(len(sn))
            d[arm] = dict(first_hold_named_sim=s2cue.boot_share(sn, one), first_hold_named_perceived=s2cue.boot_share(pn, one),
                          episodes_with_onset=int(ho.sum()),
                          success_given_named_first=float(succ[sn > 0].mean()) if (sn > 0).any() else None,
                          success_given_unnamed_first=float(succ[sn == 0].mean()) if (sn == 0).any() else None,
                          n_named_first=int(sn.sum()))
        for x, y in (("Hv", "Hp"), ("Hv0", "Hp"), ("Hv", "Hv0")):
            if x in ind and y in ind:
                d["%s-%s first_hold_named_sim" % (x, y)] = s2.boot2_states(ind[x][None], ind[y][None])
        out[c] = d
    return out


def view_stats(RD, model, arm, ks):
    out = {}
    for tid in TIDS:
        st = F3.stem_of(RD, model, arm, tid)
        if not os.path.exists(st + "_tv.json"):
            continue
        E = json.load(open(st + "_tv.json", encoding="utf-8"))["eps"]
        assert [e["trial"] for e in E] == ks, st
        key = s2.meta()["tasks"][tid]["key"]
        out[key] = dict(episodes=len(E), any_onset=int(sum(bool(e["onsets"]) for e in E)),
                        multi_onset=int(sum(len(e["onsets"]) > 1 for e in E)),
                        oracle_held_at_first_onset=float(np.mean([bool(e["oracle_held_at_onset"]) for e in E])),
                        painted_agent=int(sum(e["painted_queries"]["agent"] for e in E)), queries=int(sum(e["queries"] for e in E)),
                        prompts=dict(object=int(sum(e["prompts"]["object"] for e in E)), target=int(sum(e["prompts"]["target"] for e in E))),
                        front_secs_median=float(np.median([e["secs_front"] for e in E])))
    return out


def summary(d):
    M_ = s2.meta()
    for model, arms in RUNS.items():
        for arm in arms:
            for tid in TIDS:
                st = F3.stem_of(d, model, arm, tid)
                if not (os.path.exists(st + ".json") and os.path.exists(st + "_track.npz")):
                    continue
                t = M_["tasks"][tid]
                Rj = json.load(open(st + ".json", encoding="utf-8"))
                E = [e for v in Rj["tasks"].values() for e in v["episodes"]]
                ks = [int(e["trial"]) for e in E]
                H = json.load(open(st + "_h.json", encoding="utf-8"))["eps"] if os.path.exists(st + "_h.json") else [None] * len(ks)
                V = json.load(open(st + "_tv.json", encoding="utf-8"))["eps"] if os.path.exists(st + "_tv.json") else [None] * len(ks)
                for k, e, r, h, v in zip(ks, E, s2.stage_rows(st, t, ks), H, V):
                    on = {"%s>%s" % (m.split("_")[0][:5], p): 1 for m in s2.MUGS for p in s2.PLATES if s2.on_of(r["on"], m, p)}
                    ev = None if h is None else [(x["row"], x.get("held") and x["held"].split("_")[0],
                                                  x.get("sim_held") and x["sim_held"].split("_")[0], x.get("decision")) for x in h.get("events", [])]
                    vv = None if v is None else (v["onsets"][:4], v["prompts"], v["painted_queries"])
                    print("[s2fix5 summary] %s %s %s k%d success %s steps %s bound %s on %s | events %s | phase_max %s | view %s" % (
                        model, arm, t["key"], k, r["success"], e["steps"], r["bound"], on, ev, None if h is None else h["phase_max"], vv),
                        flush=True)
    for f in sorted(x for x in os.listdir(d) if x.startswith("ref_") and x.endswith(".json")):
        Rj = json.load(open(os.path.join(d, f), encoding="utf-8"))
        print("[s2fix5 summary] %s: %s" % (f, [(e["trial"], e["success"]) for e in Rj["eps"]]), flush=True)


def gate(lab, testable):
    return lab if testable else "NOT TESTABLE (rule: %s)" % lab


def read(a):
    from lifelong_ood import ledger
    import s2fix2
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    if a.selftest:
        assert not out.startswith(OUT), "selftest output must stay outside results/"
        ks, RD, mism = SMOKE, a.dir, "selftest"
    else:
        rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "s2_fix5"]
        assert len(rows) == 1, "expected exactly one s2_fix5 prereg row, found %d" % len(rows)

        def ap_(p):
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
        mism = [p for p, h in rows[0]["metrics"]["files_sha256"].items() if not os.path.exists(ap_(p)) or fsha(ap_(p)) != h]
        ks, RD = MAIN, ROLL
    res = dict(states=ks, mapping=MAPPING, code_mismatch=mism, models={})
    nulr = {k: {m: dict(predicted=None) for m in s2.MUGS} for k in ks}
    ALLT = s2.KNOWN + s2.NEW
    for model, arms in RUNS.items():
        runs = {arm: F3.load_run(RD, model, arm, ks) for arm in arms}
        R_ = dict(arms={}, deltas={}, verdicts={})
        ok = {arm: not isinstance(v, str) for arm, v in runs.items()}
        for arm, run in runs.items():
            if isinstance(run, str):
                R_["arms"][arm] = run
                print("== %s %s: %s" % (model, arm, run), flush=True)
                continue
            R_["arms"][arm] = dict(success={c: float(v.mean()) for c, v in run["succ"].items()},
                                   new=float(F3.mat(run, s2.NEW).mean()), known=float(F3.mat(run, s2.KNOWN).mean()),
                                   describe=s2fix2.describe(run, nulr, ks))
            print("== %s %s: per task %s | N1+N2 %.3f | K1+K2 %.3f" % (
                model, arm, {c: round(v, 3) for c, v in R_["arms"][arm]["success"].items()}, R_["arms"][arm]["new"],
                R_["arms"][arm]["known"]), flush=True)
            for c, x in R_["arms"][arm]["describe"].items():
                print("      %s %s" % (c, json.dumps(x)), flush=True)

        def dd(x, y, tasks):
            return s2.boot2_states(F3.mat(runs[x], tasks), F3.mat(runs[y], tasks))
        if ok.get("I"):
            kI = float(F3.mat(runs["I"], s2.KNOWN).mean())
            tst = bool(kI >= FLOOR)
            R_["testable"] = dict(intact_K=kI, floor=FLOOR, TESTABLE=tst)
            for arm in ("Hp", "Hv", "Hv0"):
                if not ok.get(arm):
                    continue
                dN, dK = dd(arm, "I", s2.NEW), dd(arm, "I", s2.KNOWN)
                R_["deltas"]["%s-I N" % arm], R_["deltas"]["%s-I K" % arm] = dN, dK
                lab = F3.fix_label(dN, dK)
                R_["verdicts"]["%s FIX label (rule)" % arm] = lab
                R_["verdicts"]["%s FIX label (registered)" % arm] = gate(lab, tst)
                R_["verdicts"]["%s-I K non-inferiority" % arm] = F3.ni_label(dK)
                R_["deltas"]["%s-I per task" % arm] = {c: s2.boot2_states(runs[arm]["succ"][c][None], runs["I"]["succ"][c][None]) for c in ALLT}
                xN = float(F3.mat(runs[arm], s2.NEW).mean())
                bar = "MEETS BAR" if (dN[1] > 0 and xN >= BAR) else "BELOW BAR"
                R_["verdicts"]["%s BAR (rule)" % arm] = bar
                R_["verdicts"]["%s BAR (registered)" % arm] = gate(bar, tst)
                R_.setdefault("bar", {})[arm] = dict(new=xN, bar=BAR, sr54_rule=F3.sr54_rule(dN, xN))
            for x, y in (("Hv", "Hp"), ("Hv0", "Hp"), ("Hv", "Hv0")):
                if not (ok.get(x) and ok.get(y)):
                    continue
                d2 = s2.boot2_states(runs[x]["succ"]["N2"][None], runs[y]["succ"]["N2"][None])
                R_["deltas"]["%s-%s N2" % (x, y)] = d2
                lab = F3.diff_label(d2, "VIEWS HELP", "VIEWS HURT") if y == "Hp" else F3.diff_label(d2, "PHRASES HELP", "PHRASES HURT")
                R_["verdicts"]["%s-%s N2 (rule)" % (x, y)] = lab
                R_["verdicts"]["%s-%s N2 (registered)" % (x, y)] = gate(lab, tst)
                for nm, tasks in (("N", s2.NEW), ("K", s2.KNOWN), ("K2", ("K2",)), ("all", ALLT)):
                    d_ = dd(x, y, list(tasks))
                    R_["deltas"]["%s-%s %s" % (x, y, nm)] = d_
                    R_["verdicts"]["%s-%s %s (secondary)" % (x, y, nm)] = F3.diff_label(d_, "GAIN", "COST")
                R_["deltas"]["%s-%s per task" % (x, y)] = {c: s2.boot2_states(runs[x]["succ"][c][None], runs[y]["succ"][c][None]) for c in ALLT}
        R_["mechanism"] = mechanism(runs, [x for x in arms if x != "I"])
        print("== %s MECHANISM (first onset holds the named mug): %s" % (model, json.dumps(R_["mechanism"])), flush=True)
        for arm in arms:
            if arm != "I" and ok.get(arm):
                R_["perception_%s" % arm] = F3.accuracy(runs[arm], ks)
                print("== %s %s perception accuracy (vs simulator oracle): %s" % (model, arm, json.dumps(R_["perception_%s" % arm])), flush=True)
            if arm in VIEW_ARMS and ok.get(arm):
                R_["views_%s" % arm] = view_stats(RD, model, arm, ks)
                print("== %s %s front end: %s" % (model, arm, json.dumps(R_["views_%s" % arm])), flush=True)
            if ok.get(arm):
                R_.setdefault("steps", {})[arm] = {c: float(np.median(ep_steps(RD, model, arm, T_["tid"])))
                                                   for c, T_ in ((t["key"], t) for t in s2.meta()["tasks"][:4])}
        for k_, v in R_["deltas"].items():
            print("   %s %s %s" % (model, k_, json.dumps(v if not isinstance(v, list) else np.round(v, 3).tolist())), flush=True)
        print("   VERDICTS %s: %s | testable %s | bar %s | median steps %s" % (model, json.dumps(R_["verdicts"]), json.dumps(R_.get("testable")),
                                                                         json.dumps(R_.get("bar")), json.dumps(R_.get("steps"))), flush=True)
        res["models"][model] = R_
    ref_ = {}
    for tid in TIDS:
        f = os.path.join(RD, "ref_O_%d.json" % tid)
        if os.path.exists(f):
            ee = json.load(open(f, encoding="utf-8"))["eps"]
            assert [e["trial"] for e in ee] == ks, f
            ref_[s2.meta()["tasks"][tid]["key"]] = dict(success=float(np.mean([e["success"] for e in ee])),
                                                        errors=int(sum(e.get("error") is not None for e in ee)))
    nn = [ref_[c]["success"] for c in s2.NEW if c in ref_]
    res["reference"] = dict(per_task=ref_, success_new=float(np.mean(nn)) if len(nn) == 2 else None,
                            C1=None if len(nn) < 2 else ("C1 MET" if np.mean(nn) >= 0.80 else "C1 UNMET"))
    print("== reference: %s" % json.dumps(res["reference"]), flush=True)
    print("== code hash mismatches: %s" % mism, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, out)
    print("[s2fix5 read] wrote %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("states", "ref", "summary", "read"))
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--tid", type=int)
    ap.add_argument("--branches")
    ap.add_argument("--out-dir")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    if A.mode == "read" and A.out is None:
        A.out = READ
    dict(states=states, ref=ref, summary=lambda a: summary(a.dir), read=read)[A.mode](A)

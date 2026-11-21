"""WA-s2r (SR 89 STEP 2, docs/notes/scene2_plan.md section 2 step 2): wa_s2.py (unchanged) plus the scene-2 analogues of the kitchen WA modules, as a separate file.
ADDED here (everything else is wa_s2.py's client / server):
  R-S2  the resolver analogue.  The two plates are identical and named by position, so the client parses "left plate" / "right plate" from the instruction and picks the plate INSTANCE by its
        base-frame position (mask + depth + calibration, first query; the axis and sign = results/flawed/sr89/plate_axis.json, from the BDDL's own plate_1 = left).  The chosen plate is the channel's
        target entity and the role paint's active filler (Front.tgt).  noR = the plate chosen WITHOUT the parse: the larger agentview mask component (ties: plate_1), as wa.py's noR
        (rgp's "largest component" rule); the prompt stays the instruction's role phrase.
  M-S2  the pose-memory analogue.  The estimator's hold rule (occluded mask: the last estimate seen with >= half of its first mask; src hold / none) is the memory.  noM = the channel is withheld
        (all-zero state) whenever the ACTIVE role entity's estimate is a held one (the view paint of that query is NOT removed: the images come from the shared Front; deviation from wa.py's noM).
  --chan maskoff (registered primary): mask + depth positions with the hold rule (as --chan mask), but the in-hand offset at the hold's onset is the simulator's (PRIVILEGED, the value the training labels
        carry), because the unprivileged onset rule gave 3-4 cm held-target channel error in the SR 89 probe (docs/notes/sr89_build.md).  --chan mask / sim as wa_s2.py.
  serve --mu_json F   the W5 release-point table of the scene-2 plates (lifelong_ood/policy/raasls2_mu.json) replaces the kitchen table of PolicyBackboneW (backbone_w.py untouched).
Arms (client --arm): full, noR, noM, noC (noC = prompt + views + all-zero state).  RAA-sL-S2 = served with --where_own 0, W5-S2 = --where_own 5 --mu_json, step-0 = the kitchen checkpoint with --where_own 0.

Original wa_s2.py docstring follows.

WA-s2 (SR 89, docs/notes/scene2_plan.md, docs/notes/sr89_build.md): the scene-2 (LIBERO-10 LIVING_ROOM_SCENE5) client for STATE-TOKEN channel models
(RAA-sL family: pi05_g8v_raas_l_ae, pi05_g8v_raas_ls2_ae), a scene-2 copy of wa_s.py's channel path on s2.py's tvp client.  wa_s.py, wa_sw5.py,
s2.py and backbone_w.py are unchanged.

What the client hands the model (per query), the same as SR 82's kitchen runs and as s2_rv.py's training data:
  prompt   the role phrase, "object: the white mug" until ProprioHeld turns on, then "target: the left plate"      (s2_tasks.TASKS phrases)
  views    both cameras, every movable (3 mugs, 2 plates) painted magenta except the active filler (sc_rv.role_paint), exact SIMULATOR masks
           (s2.SimSeg + tv_frozen.Front, as SR 50's tvp)
  state    the continuous channel [dx, dy, dz (m), role id] (sc_rv.chan_state): role 1 = object - eef; role 2 (held) = target - eef - offset,
           offset = the object estimate minus the eef at the hold's onset, fixed while held.  The 8-dim eef state is never sent.
Channel positions (--chan):  mask  (default) = SIMULATOR masks + rendered depth + camera calibration: the median of the entity's back-projected
                                   agentview mask points plus a per-entity constant fitted at rest on calibration states (mode `calib`), a hold
                                   rule for an occluded mask (the last estimate made while >= half of the first mask was visible);
                             sim   = the simulator's body positions (privileged, error 0).
Arms (client --arm):  full = prompt + views + channel;  noC = prompt + views + all-zero state (the localisation ablation).
Server (mode serve): the RAA-s family checkpoint behind GuidedPolicy (PolicyBackbone.sample, servo on); --where_own 0 = RAA-sL as is, 5 = W5 (backbone_w
mode 5; "target: the left plate" / "target: the right plate" are not in raasw5_mu.json, so mu = 0 = the channel's own rel_xy, W2c-style).

The channel diag (--chan-diag on the client): per query, the estimate against the simulator truth (cm), the hold, the source.  Never fed to the model.

    python experiments/wa_s2.py states                    (LIBERO venv, EGL) the probe states (seeds PROBE_BASE + k) and calibration states
    python experiments/wa_s2.py calib                     (LIBERO venv, EGL) -> results/flawed/sr89/probe/calib.json
    python experiments/wa_s2.py serve --ckpt D --port P [--where_own 0|5] [--config pi05_g8v_raas_l_ae]    (openpi venv, GPU)
    python experiments/wa_s2.py client --arm full|noC [--chan mask|sim] <libero_client.py's arguments>     (LIBERO venv)
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
PD = os.path.join(OUT, "sr89", "probe")
PROBE_BASE, PROBE_N, PROBE_BR = 1090000, 20, 47000000            # SR 89 step 0 (exploratory): states seed 1,090,000 + k, k 0-19; branches seed_base 47,000,000 + 1000 k
CAL_BASE, CAL_N = 1080000, 10                                    # calibration states (rest poses): seed 1,080,000 + k, k 0-9
ARMS = ("full", "noR", "noM", "noC")
MINPX = 20
PLATES = ("plate_1", "plate_2")
AXIS_JSON = os.path.join(OUT, "sr89", "plate_axis.json")
MU_S2 = os.path.join(ROOT, "lifelong_ood", "policy", "raasls2_mu.json")


# ---------------------------------------------------------------- states / calibration (LIBERO venv, EGL)
def _old_states(s2):
    od = [np.load(os.path.join(s2.SD, "states", "%s_0.npy" % s2.SUITE))]
    for d, L in (("s2cue3", "O"), ("s2cue4", "O"), ("s2cue5", "O"), ("s2fix2", "O"), ("s2fix2", "PD2"), ("s2fix3", "O"), ("s2fix4", "O"),
                 ("s2fix5", "O"), ("s2fix6", "O"), ("s2rg", "O")):
        od.append(np.load(os.path.join(OUT, d, L, "%s_0.npy" % s2.SUITE)))
    return od


def states(a):
    import s2
    import s2_tasks as T2
    od = os.path.join(PD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))

    def draw(k, base):                                               # s2_tasks.build's / s2rg.states' rule with a new base
        np.random.seed(base + k)
        env.seed(base + k)
        env.reset()
        return env.sim.get_state().flatten()
    S = np.stack([draw(k, PROBE_BASE) for k in range(PROBE_N)])
    C = np.stack([draw(k, CAL_BASE) for k in range(CAL_N)])
    env.close()
    old = _old_states(s2)
    for X in (S, C):
        for i in range(len(X)):
            for O_ in old:
                for j in range(len(O_)):
                    assert not (O_[j].shape == X[i].shape and np.allclose(X[i], O_[j])), ("equals an earlier scene-2 state", i, j)
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for j in range(len(C)):
            assert not np.allclose(S[i], C[j]), (i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    np.save(os.path.join(PD, "calib_states.npy"), C)
    with open(os.path.join(PD, "branches_probe.json"), "w", encoding="utf-8") as fh:
        json.dump([dict(id="k%d" % k, trial=k, seed_base=PROBE_BR + 1000 * k, at=0, variant="s2") for k in range(PROBE_N)], fh, indent=1)
    with open(os.path.join(PD, "ranges_used.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(note="SR 89 step 0 (EXPLORATORY probe): scene-2 states, LIBERO sampler on K1's BDDL (s2rg.states' rule); never used by the registered SR 89 rollouts",
                       probe_states=dict(seed_base=PROBE_BASE, k=[0, PROBE_N - 1]), calibration_states=dict(seed_base=CAL_BASE, k=[0, CAL_N - 1]),
                       probe_branches=dict(seed_base=PROBE_BR, step=1000, k=[0, PROBE_N - 1]),
                       reserved_for_registered_main=dict(seed_base=1100000, k=[0, 63], note="plan: k 0-3 smoke, 4-63 main; NOT used here"),
                       earlier=dict(SR50=830000, SR68_72=980000)), fh, indent=1)
    print("[wa-s2r] %d probe states (seed %d + k), %d calibration states (seed %d + k), none equal to an earlier scene-2 state; wrote %s" % (
        PROBE_N, PROBE_BASE, CAL_N, CAL_BASE, PD), flush=True)


def calib(a):
    """Per movable: the constant that maps the back-projected mask's median to the body position, fitted at rest on the calibration states."""
    import s2
    import s2_tasks as T2
    import sc_pu as PU
    import sc_rv as RV
    p = os.path.join(PD, "calib.json")
    assert not os.path.exists(p), "exists: " + p
    C = np.load(os.path.join(PD, "calib_states.npy"))
    M_ = s2.meta()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))
    env.seed(7)
    env.reset()
    E = RV.entity_tables(env)
    rows = {n: [] for n in s2.MOVABLE}
    for k in range(len(C)):
        env.set_init_state(C[k])
        for _ in range(s2.WAIT):
            env.step(T2.DUMMY)
        A, _ = PU.measure(env, E, PU.AGENT, list(s2.MOVABLE))
        truth = T2.positions(env, list(s2.MOVABLE))
        for i, n in enumerate(s2.MOVABLE):
            if A[i, 0] >= 100:
                rows[n].append(np.concatenate([truth[i] - A[i, 1:4], [A[i, 0]]]))
    env.close()
    res = {}
    for n, r in rows.items():
        r = np.asarray(r)
        res[n] = dict(corr=np.median(r[:, :3], 0).tolist(), n=int(len(r)), px_median=float(np.median(r[:, 3])) if len(r) else 0.0,
                      resid_cm_median=float(100 * np.median(np.linalg.norm(r[:, :3] - np.median(r[:, :3], 0), axis=1))) if len(r) else None)
        print("[wa-s2r] calib %s n %d corr_cm %s resid_cm %.2f px %.0f" % (n, len(r), np.round(100 * np.asarray(res[n]["corr"]), 2).tolist(),
                                                                         res[n]["resid_cm_median"] or -1, res[n]["px_median"]), flush=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(dict(calib_seed_base=CAL_BASE, n_states=int(len(C)), corr=res), fh, indent=1)
    print("[wa-s2r] calib -> %s" % p, flush=True)


# ---------------------------------------------------------------- server (openpi venv)
def serve_cmd(a):
    import sc_rv as RV
    RV._openpi()
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import GuidedPolicy, StripMeta, serve
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    name = a.config or "pi05_g8v_raas_l_ae"
    logging.info(f"[wa-s2r] serve config {name} ckpt {a.ckpt} where_own {a.where_own} where_log {a.where_log or None}")
    if a.where_own:
        from lifelong_ood.policy.backbone_w import PolicyBackboneW
        bk = PolicyBackboneW(name, checkpoint_dir=a.ckpt, device="cuda", where_own=a.where_own, log_path=a.where_log or None)
        if a.mu_json:                                                       # SR 89 step 2: the scene-2 plates' release points (training cycles only) replace the kitchen table
            tab = json.load(open(a.mu_json, encoding="utf-8"))["mu_m"]
            bk._mu_table = {k: [float(v[0]), float(v[1])] for k, v in tab.items()}
            logging.info(f"[wa-s2r] mu table {a.mu_json}: {bk._mu_table}")
    else:
        bk = PolicyBackbone(name, checkpoint_dir=a.ckpt, device="cuda")
    pol = GuidedPolicy(bk, hooks=None, num_steps=10, seed=a.seed)        # PolicyBackbone.sample adds the servo when servo_k != 0
    from openpi.policies import libero_policy
    for _ in range(2):                                                      # the norm stats are 4-dim: warm up with a 4-dim state
        ex = libero_policy.make_libero_example()
        ex["observation/state"] = np.zeros(RV.CHAN_DIM, np.float32)
        ex["prompt"] = "object: the white mug"
        pol.infer(ex)
    serve(StripMeta(pol), a.port)


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
    chan_mode = take("--chan", "mask")
    calib_path = take("--calib", os.path.join(PD, "calib.json"))
    assert arm in ARMS and chan_mode in ("mask", "sim", "maskoff"), (arm, chan_mode)
    import pp_atlas as PA
    import libero_client as LC
    import s2
    import sc_pu as PU
    import sc_rv as RV
    import tv_frozen as TF
    M = s2.register_suite(LC)
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = M["tasks"][TID]
    assert TASK["tid"] == TID and len(TASK["atoms"]) == 1, "the role sequencer handles one pick-and-place"
    ms = int(TASK["max_steps"])
    if "--max-steps" in sys.argv:
        assert int(sys.argv[sys.argv.index("--max-steps") + 1]) == ms
    else:
        sys.argv += ["--max-steps", str(ms)]
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "scene-2 episodes need --fresh and --branches"
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    RV.MOVABLE_OBJ = s2.MOVABLE                                             # this scene's movables (role_paint and Front read it)
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    P = list(TASK["phrases"])
    OE, TE = TASK["object"], TASK["target"]
    corr = {n: np.asarray(v["corr"], np.float64) for n, v in json.load(open(calib_path, encoding="utf-8"))["corr"].items()}
    AX = json.load(open(AXIS_JSON, encoding="utf-8"))
    LANG = TASK["language"]
    SIDE_WORD = "left" if "left plate" in LANG else ("right" if "right plate" in LANG else None)
    assert SIDE_WORD is not None, LANG
    ORIG, RECS, SIDE, CUR = LC.OffScreenRenderEnv, [], [], {}
    SIDEF = OUTF.replace(".json", "_wa.json")
    S = dict(est=None, was=False, off=None, toff=None, pre=None, q=[], res=None, te=TE)

    def resolve(raw_env, front):
        """R-S2 (full / noM / noC) or the noR rule, on the first query's agentview: -> the plate entity the channel and the paint's filler use."""
        import sc_pu as PU_
        E_ = front.E if front.E is not None else RV.entity_tables(raw_env)
        A, _ = PU_.measure(raw_env, E_, PU_.AGENT, list(PLATES))
        pos = [A[i, 1:4] + corr[n] for i, n in enumerate(PLATES)]
        px = [float(A[i, 0]) for i in range(2)]
        if arm == "noR":
            ch = PLATES[int(px[1] > px[0])]                                  # the larger component; ties -> plate_1
            how = "largest"
        else:
            c0, c1 = pos[0][AX["axis"]], pos[1][AX["axis"]]
            greater = PLATES[0] if c0 > c1 else PLATES[1]
            lesser = PLATES[1] if c0 > c1 else PLATES[0]
            want_greater = (SIDE_WORD == "left") == bool(AX["left_is_greater"])
            ch = greater if want_greater else lesser
            how = "parse"
        return dict(how=how, chosen=ch, truth=TE, correct=bool(ch == TE), px=px, pos=[p.tolist() for p in pos])

    class Est:
        """Mask + depth estimates (sim masks) of the instructed mug and plate, with the hold rule; `sim` = body positions."""

        def __init__(self, env):
            self.env, self.E = env, RV.entity_tables(env)
            self.ref, self.good, self.src = {}, {}, {}

        def truth(self, n):
            inner = self.env.env
            return np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]], np.float64)

        def step(self):
            if chan_mode == "sim":
                return {OE: self.truth(OE), S["te"]: self.truth(S["te"])}, {OE: "sim", S["te"]: "sim"}
            A, _ = PU.measure(self.env, self.E, PU.AGENT, [OE, S["te"]])
            out = {}
            for k, n in enumerate((OE, S["te"])):
                s = A[k]
                px = float(s[0])
                if n not in self.ref and px > 0:
                    self.ref[n] = px
                p = s[1:4] + corr[n]
                if px >= max(MINPX, 0.5 * self.ref.get(n, 0.0)):
                    self.good[n], src = p.copy(), "mask"
                elif n in self.good:
                    p, src = self.good[n], "hold"
                elif px >= MINPX:
                    self.good[n], src = p.copy(), "mask_partial"
                else:
                    p, src = None, "none"
                out[n], self.src[n] = (None if p is None else np.asarray(p, np.float64)), src
            return out, dict(self.src)

    class WEnv(s2.Rec2):
        def __init__(self, env):
            super().__init__(env, OE, TE)
            self.front = None
            CUR["env"] = self

        def set_init_state(self, s):
            obs = self._e.set_init_state(s)
            self.front = TF.Front(self._e, "tv", OE, TE, X, s2.SimSeg(self._e))
            self.front.prompts = dict(object=0, target=0)
            S.update(est=None, was=False, off=None, toff=None, pre=None, q=[], res=None, te=TE)
            return obs

        def step(self, a):
            r = super().step(a)
            if self.front is None:
                return r
            if S["res"] is None and self.n >= s2.WAIT:                       # R-S2 / noR: the plate instance, before the first painted observation
                S["res"] = resolve(self._e, self.front)
                S["te"] = S["res"]["chosen"]
                self.front.tgt = S["te"]
            obs = self.front.step(r[0], self.n, float(a[6]))
            return (obs,) + tuple(r[1:])

        def close(self):
            if self.n > 0:
                self.finish()
                RECS.append(self.data())
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
                i = len(SIDE)
                rec = self.front.record(self.rows["held"])
                rec.update(arm=arm, trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, tid=TID, key=TASK["key"],
                           prompts=self.front.prompts, queries_diag=S["q"], res=S["res"], chan=chan_mode)
                SIDE.append(rec)
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, chan=chan_mode, tid=TID, key=TASK["key"], language=TASK["language"], phrases=P, lex=X,
                                   masks="simulator", movable=list(s2.MOVABLE), eps=SIDE), fh)
                os.replace(tmp, SIDEF)
                q = S["q"]
                e_o = [x["err_obj_cm"] for x in q if x["role"] == "object" and x["err_obj_cm"] is not None]
                e_t = [x["err_chan_cm"] for x in q if x["role"] == "target" and x["err_chan_cm"] is not None]
                print("[wa-s2r] %s %s trial %s: steps %d held queries %d/%d chan err obj median %s cm, held-target channel median %s cm" % (
                    arm, TASK["key"], rec["trial"], self.n, rec["held_queries"], rec["queries"],
                    None if not e_o else round(float(np.median(e_o)), 2), None if not e_t else round(float(np.median(e_t)), 2)), flush=True)
                self.front = None
            self._e.close()

    def channel(el):
        env = CUR["env"]
        f = env.front
        on = bool(f is not None and f.on)
        ee = np.asarray(el["observation/state"], np.float64)[:3]
        assert S["res"] is not None, "the plate resolution happens at the first painted step (n >= WAIT), before the first query"
        if arm == "noC":
            return on, RV.chan_state(None, False, False), None
        if S["est"] is None:
            S["est"] = Est(env._e)
        tc = S["te"]                                                         # the plate the resolver (or noR's size rule) chose
        est, src = S["est"].step()
        tobj, ttgt = S["est"].truth(OE), S["est"].truth(tc)
        if on and not S["was"]:                                              # the hold's onset: the offset = the object estimate before the hold minus the eef (mask); maskoff / sim: the simulator's
            S["off"] = None if S["pre"] is None else (S["pre"] - ee)
            S["toff"] = tobj - ee
            if chan_mode == "maskoff":
                S["off"] = S["toff"].copy()
        if not on:
            S["off"] = S["toff"] = None
            if est[OE] is not None:
                S["pre"] = est[OE].copy()
        S["was"] = on
        role = tc if on else OE
        p = est[role]
        rel = None
        if p is not None:
            rel = p - ee
            if on:
                rel = rel - (S["off"] if S["off"] is not None else np.zeros(3))
        true = (ttgt if on else tobj) - ee - (S["toff"] if (on and S["toff"] is not None) else 0.0)
        held_est = str(src[role]) in ("hold", "none")                        # M-S2: the active role entity's estimate is a held one (not a measurement of it now)
        withheld = bool(arm == "noM" and held_est)
        rec = dict(n=env.n, role="target" if on else "object", on=on, src=str(src[role]), has=rel is not None, held_est=bool(held_est), withheld=withheld,
                   err_obj_cm=None if est[OE] is None else float(100 * np.linalg.norm(est[OE] - tobj)),
                   err_tgt_cm=None if est[tc] is None else float(100 * np.linalg.norm(est[tc] - ttgt)),
                   err_chan_cm=None if rel is None else float(100 * np.linalg.norm(rel - true)),
                   rel_cm=None if rel is None else RV.chan_cm(rel), true_cm=RV.chan_cm(true))
        S["q"].append(rec)
        if withheld:
            return on, RV.chan_state(None, False, False), rec
        return on, RV.chan_state(rel, on, rel is not None), rec

    Base = LC._wcp.WebsocketClientPolicy

    class WClient(Base):
        def infer(self, el):
            env = CUR.get("env")
            assert str(el["prompt"]) == TASK["language"], el["prompt"]
            el = dict(el)
            on, st, _ = channel(el)
            el["prompt"] = P[int(on)]
            el["observation/state"] = st
            if env.front is not None:
                env.front.prompts["target" if on else "object"] += 1
            return super().infer(el)
    LC._wcp.WebsocketClientPolicy = WClient
    LC.OffScreenRenderEnv = lambda *a, **k: WEnv(ORIG(*a, **k))
    print("[wa-s2r] arm %s chan %s task %s %r object %s target %s max_steps %d phrases %s" % (arm, chan_mode, TASK["key"], TASK["language"], OE, TE, ms, P), flush=True)
    LC.main()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "client":
        client_main()
    elif what == "states":
        states(None)
    elif what == "calib":
        calib(None)
    elif what == "serve":
        ap = argparse.ArgumentParser()
        ap.add_argument("what")
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--port", type=int, required=True)
        ap.add_argument("--seed", type=int, default=13)
        ap.add_argument("--config", default="")
        ap.add_argument("--where_own", type=int, default=0, choices=(0, 1, 2, 3, 4, 5, 6, 7))
        ap.add_argument("--where_log", default="")
        ap.add_argument("--mu_json", default="")
        serve_cmd(ap.parse_args())
    else:
        raise SystemExit(__doc__)

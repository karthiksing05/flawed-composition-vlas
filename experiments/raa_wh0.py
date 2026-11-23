"""NPH COPY of raa_wh.py (docs/notes/plan_necessity_power.md 3.2 step 8): adds pi0_g8v_raah_l_ae -> PolicyBackboneH0 (raah0_model, uncentred H*); the only change vs raa_wh.py.  RVH COPY of raa_w.py: serves pi05_g8v_raah_l_ae through PolicyBackboneH (where_own stays 0); the only change (docs/notes/plan_review_atlas.md 6.5).  Original: RAA-min (docs/SCAFFOLD_RESULTS.md 73; docs/PLAN_PAPER.md 2b): the test-time half of the two role-channel arms.

Training side: experiments/sc_rv.py (arms rrel = R0's inputs + channel, raa = R2's inputs + channel; `chan` computes each demo's slot
positions, `convert` writes the datasets).  The channel is a text suffix on the prompt of every routed frame (K1-K6, N1-N6; K7, K8, N7,
N8 stay whole):  "<prompt>. rel: +12 -3 +5 cm, role target"  =  the active role entity's position minus the end effector's, base-frame
axes (the robot base has the identity orientation), rounded to 1 cm, and the role id (object until ProprioHeld fires, target while
held).  pi0.5 here runs with discrete_state_input=False, so the checkpoint's state vector is never read; the prompt is the only
language-side input the model has.

Test time, the pointer and the position:
  words     the contact-induced lexicon (parts_flow.role_lexicon / role_parse over results/flawed/sc_lex.json's situations, which come
            from the demos' own contact events) parses the instruction into (object slot, target slot); no given role is read.
  location  learned masks (sc_seg U-Net) + the rendered depth + the camera calibration -> sc_pu.Estimator, stepped at every policy
            query on the client (the depth render and the calibration are privileged in simulation, as ReGuide-perc).  The Estimator
            holds the last unoccluded estimate when its entity is hidden (the F6 memory).  The simulator's positions are read ONLY for
            the per-query error log (meta/raa_t*), never fed to the model.
  server    RAAServer pops the client's estimates, lets sc_rv.RoleView choose the role prompt and paint (arm raa: learned masks via
            sc_seg.LearnedMasks), then appends the channel.  Arm rrel: whole instruction + channel, images untouched.

    python experiments/raa.py serve  --arm rrel|raa --ckpt DIR --port P [--config NAME] [--seed 13] [--chan-diag F.jsonl] [--diag F.json]
    python experiments/raa.py client --arm rrel|raa <fc_n2_client.py's arguments>       (LIBERO venv, py3.8, GPU for the U-Net)
    python experiments/raa.py summarize F.jsonl [F2.jsonl ...]                          per-query channel error vs simulator truth
    python experiments/raa.py selftest                                                  the text format and rounding
"""
import argparse
import json
import logging
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
LEX = os.path.join(OUT, "sc_lex.json")
ARMS = ("rrel", "raa", "raao", "raac")                                       # raao = RAA-o (SR 76): the target role is measured from the held object   # raac = RAA-c (SR 77): raao's channel as a continuous state token, no text suffix
PAINT = ("raa", "raao", "raac")


def lexicon():
    """The contact-induced lexicon of results/flawed/sc_lex.json (situations from pp_p_train.demo_sites on K1-K8's demos)."""
    from lifelong_ood.policy import parts_flow as PF
    sit = json.load(open(LEX, encoding="utf-8"))["situations"]
    return PF.role_lexicon(sit), PF


# ---------------------------------------------------------------- server (openpi venv)
class _Hook:
    """The policy RoleView calls last: RAAServer.final writes the channel into the prompt, then the policy answers."""

    def __init__(self, pol, srv):
        self.p, self.srv = pol, srv
        self.metadata = getattr(pol, "metadata", {})

    def infer(self, o):
        return self.srv.final(o, self.p)


class RAAServer:
    def __init__(self, policy, X, arm, diag=None, replan=5):
        import sc_rv as RV
        assert arm in ARMS, arm
        self.RV, self.arm, self.diag = RV, arm, diag
        self.T16 = RV.t16()
        self.metadata = getattr(policy, "metadata", {})
        self.rv = RV.RoleView(_Hook(policy, self), X, mode="rview2" if arm in PAINT else None, replan=replan)
        self.cur, self.n, self.noch = None, 0, 0

    def infer(self, obs):
        o = dict(obs)
        raa = {k[len("meta/raa_"):]: o.pop(k) for k in list(o) if k.startswith("meta/raa_")}
        self.cur = dict(text=str(o["prompt"]), raa=raa, t=o.get("meta/t"), ep=o.get("meta/episode"))
        return self.rv.infer(o)

    def final(self, o, pol):
        c = self.cur
        P = self.T16.given(c["text"])
        if P is not None:                                                   # routed: K1-K6, N1-N6
            on = str(o["prompt"]) == P[1]
            base = str(o["prompt"]) if self.arm in PAINT else c["text"]
            r = c["raa"]
            if r.get("ok"):
                ee = np.asarray(o["observation/state"], np.float64)[:3]
                role = "tgt" if on else "obj"
                rel = np.asarray(r[role], np.float64) - ee
                off = toff = np.zeros(3)
                if self.arm in ("raao", "raac") and on:                               # RAA-o: target minus the held object = target - eef - offset (fixed at the onset)
                    if r.get("off_ok"):
                        off, toff = np.asarray(r["off"], np.float64), np.asarray(r["toff"], np.float64)
                    else:                                                   # server says held, client has no offset yet: eef-relative, counted
                        self.nooff = getattr(self, "nooff", 0) + 1
                        logging.warning(f"[raao] held but no offset at ep {c['ep']} t {c['t']}")
                    rel = rel - off
                if self.arm == "raac":                                      # SR 77: the channel is the continuous state token, the prompt stays the role phrase
                    o["prompt"] = base
                    o["observation/state"] = self.RV.chan_state(rel, on)
                else:
                    o["prompt"] = self.RV.chan_text(base, rel, on)
                self.n += 1
                if self.diag and "t" + role in r:
                    true = np.asarray(r["t" + role], np.float64) - ee - toff
                    d = rel - true
                    rec = dict(ep=c["ep"], t=c["t"], text=c["text"], role="target" if on else "object", rel_cm=self.RV.chan_cm(rel),
                               true_cm=self.RV.chan_cm(true), err_cm=float(100 * np.linalg.norm(d)),
                               axis_err_cm=[float(100 * abs(v)) for v in d], src=str(r.get("src_" + role)),
                               roles=r.get("roles"), prompt=o["prompt"],
                               **({"off_cm": self.RV.chan_cm(off), "toff_cm": self.RV.chan_cm(toff), "off_ok": bool(r.get("off_ok"))}
                                  if self.arm in ("raao", "raac") else {}))
                    with open(self.diag, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
            else:                                                           # the lexicon refused: no channel, the base prompt
                o["prompt"] = base
                self.noch += 1
                logging.warning(f"[raa] no channel at ep {c['ep']} t {c['t']}: {r.get('why')}")
        if self.arm == "raac" and np.asarray(o["observation/state"]).shape[-1] != self.RV.CHAN_DIM:   # unrouted (K7, K8, ...) or no channel: zeros, never the 8-dim eef state
            o["observation/state"] = self.RV.chan_state(None, False, False)
        logging.info(f"[raa] ep={c['ep']} t={c['t']} prompt={o['prompt']!r}" + (f" state={np.round(np.asarray(o['observation/state']), 4).tolist()}" if self.arm == "raac" else ""))
        return pol.infer(o)


def _make_bk(PolicyBackbone, name, a):
    """SR 84 (copy of raa.py / wa_s.py with this one change): --where_own 0 (default) builds exactly PolicyBackbone(name, checkpoint_dir, device) as before;
    1 | 2 build lifelong_ood.policy.backbone_w.PolicyBackboneW (the guarded branch after the servo)."""
    if not getattr(a, "where_own", 0):
        if name == "pi05_g8v_raah_l_ae":                                   # RVH (plan_review_atlas 6.5): the learned where-head; the ONLY change vs the source file
            from lifelong_ood.policy.backbone_h import PolicyBackboneH
            logging.info("[rvh] PolicyBackboneH")
            return PolicyBackboneH(name, checkpoint_dir=a.ckpt, device="cuda", log_path=getattr(a, "where_log", None) or None)
        if name == "pi0_g8v_raah_l_ae":                                    # NPH: the goal-directed learned where-head on pi0; the ONLY change vs raa_wh.py
            from lifelong_ood.policy.backbone_h0 import PolicyBackboneH0
            logging.info("[nph] PolicyBackboneH0")
            return PolicyBackboneH0(name, checkpoint_dir=a.ckpt, device="cuda", log_path=getattr(a, "where_log", None) or None)
        return PolicyBackbone(name, checkpoint_dir=a.ckpt, device="cuda")
    from lifelong_ood.policy.backbone_w import PolicyBackboneW
    logging.info(f"[sr84] PolicyBackboneW where_own {a.where_own} log {a.where_log or None}")
    return PolicyBackboneW(name, checkpoint_dir=a.ckpt, device="cuda", where_own=a.where_own, log_path=a.where_log or None)


def serve_cmd(a):
    import sc_rv as RV
    RV._openpi()
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import GuidedPolicy, serve, warm
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    name = a.config or RV.FULL[a.arm] + "_ae"
    X = json.load(open(RV.lex_path(a.suffix), encoding="utf-8"))
    logging.info(f"[raa] serve arm {a.arm} config {name} ckpt {a.ckpt} tw {X['tw']:.6f} lift {X['lift']} chan_diag {a.chan_diag}")
    bk = _make_bk(PolicyBackbone, name, a)
    pol = GuidedPolicy(bk, hooks=None, num_steps=10, seed=a.seed)
    if a.arm == "raac":                                                      # the norm stats are 4-dim: warm up with a 4-dim state, not openpi's 8-dim example
        from openpi.policies import libero_policy
        for _ in range(2):
            ex = libero_policy.make_libero_example()
            ex["observation/state"] = np.zeros(RV.CHAN_DIM, np.float32)
            pol.infer(ex)
    else:
        warm(pol)
    srv = RAAServer(pol, X, a.arm, diag=a.chan_diag or None)
    if a.arm in PAINT:
        import sc_seg as SEG
        seg = SEG.load(a.seg_ckpt or None, "cuda")
        logging.info(f"[raa] learned masks {seg.path}")
        srv = SEG.LearnedMasks(srv, seg, diag=a.diag or None)
    serve(srv, a.port)


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
    import fa1_common as FC
    import sc_pu as PU
    import sc_rv as RV
    import sc_seg as SEG
    from lifelong_ood.policy.parts_flow import ProprioHeld
    LC = FN.LC
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    Cal = PU.load_calib()
    SEGM = SEG.load(os.environ.get("SC_SEG_CKPT") or None, "cuda")
    lex, PF = lexicon()
    if arm in PAINT:                                                        # the server paints from the client's raw renders (sc_rv.RoleView)
        FN._seg_px()
    CACHE = {}

    def seg_std(env, E, cam):
        """sc_pu.seg_std's contract (entity ids, robosuite's top-down convention) from the learned U-Net on an RGB render."""
        sim = env.env.sim
        key = (id(sim), float(sim.data.time), cam)
        if key in CACHE:
            return CACHE[key]
        img = RV.flip(sim.render(camera_name=cam, width=PU.SIZE, height=PU.SIZE, depth=False))
        p = SEGM.ent(img, cam, E["names"])
        out = np.ascontiguousarray(np.asarray(p)[:, ::-1])
        if len(CACHE) > 8:
            CACHE.clear()
        CACHE[key] = out
        return out
    PU.seg_std = seg_std
    CUR = {}
    S = dict(est=None, det=None, E=None, roles=None, first=True)
    prev_ctor = LC.OffScreenRenderEnv

    class PEnv:
        """Passes everything through; remembers the last observation and the last gripper command (the server's ProprioHeld inputs)."""

        def __init__(self, e):
            self.__dict__["_e"] = e
            self.__dict__["obs"] = None
            self.__dict__["cmd"] = -1.0
            CUR["env"] = self

        def __getattr__(self, n):
            return getattr(self.__dict__["_e"], n)

        def reset(self, *a, **k):
            o = self._e.reset(*a, **k)
            self.__dict__["obs"], self.__dict__["cmd"] = o, -1.0
            return o

        def set_init_state(self, s):
            o = self._e.set_init_state(s)
            self.__dict__["obs"], self.__dict__["cmd"] = o, -1.0
            return o

        def step(self, a):
            r = self._e.step(a)
            self.__dict__["obs"], self.__dict__["cmd"] = r[0], float(a[6])
            return r
    LC.OffScreenRenderEnv = lambda *a, **k: PEnv(prev_ctor(*a, **k))

    def truth(raw, slot):
        inner, sim = raw.env, raw.env.sim
        if slot in FC.MOVABLE:
            return np.array(sim.data.body_xpos[inner.obj_body_id[slot]], np.float64)
        return np.array(sim.data.site_xpos[sim.model.site_name2id(slot)], np.float64)

    def channel(el):
        env = CUR["env"]
        raw = env._e
        text = str(el["prompt"])
        if el.get("meta/episode_start") or S["det"] is None:
            S["roles"] = PF.role_parse(lex, text)
            S["det"], S["E"], S["first"] = ProprioHeld(X["tw"], X["lift"]), RV.entity_tables(raw), True
            S["was"], S["off"], S["toff"] = False, None, None                   # RAA-o: the in-hand offset, fixed at each hold onset
            S["est"] = PU.Estimator(Cal, S["roles"][0]) if S["roles"] else None
            try:
                given = FC.roles(text)
            except Exception:                                               # noqa: BLE001
                given = None
            S["agree"] = None if given is None or S["roles"] is None else (tuple(S["roles"]) == tuple(given))
        if S["roles"] is None:
            return {"meta/raa_ok": False, "meta/raa_why": "no induced roles"}
        obs = env.obs
        ee = np.asarray(obs["robot0_eef_pos"], np.float64)
        R = FC._quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64))
        held = bool(S["det"].step(np.asarray(obs["robot0_gripper_qpos"], np.float64), env.cmd, ee[2]))
        E = S["E"]
        A, _ = PU.measure(raw, E, PU.AGENT, PU.MOV)
        F = PU.measure(raw, E, PU.AGENT, PU.fixture_names(E))[0] if S["first"] else None
        S["est"].step(A, lambda: PU.measure(raw, E, PU.WRIST, PU.MOV)[0], F, held, ee, R, env.cmd)
        S["first"] = False
        oe, te = S["roles"]
        pos = S["est"].pos
        src = S["est"].src
        extra = {}
        if arm in ("raao", "raac"):                                                   # SR 76: offset = the Estimator's object estimate minus the eef at the onset
            if held and not S["was"]:                                       # (this query's estimate), then held fixed; truth only for the error log
                S["off"] = (pos[FC.ENT.index(oe)] - ee).tolist()
                S["toff"] = (truth(raw, oe) - ee).tolist()
            if not held:
                S["off"] = S["toff"] = None
            S["was"] = held
            extra = {"meta/raa_off_ok": S["off"] is not None, "meta/raa_off": S["off"] or [0.0, 0.0, 0.0],
                     "meta/raa_toff": S["toff"] or [0.0, 0.0, 0.0]}
        return {**extra, "meta/raa_ok": True, "meta/raa_obj": pos[FC.ENT.index(oe)].tolist(), "meta/raa_tgt": pos[FC.ENT.index(te)].tolist(),
                "meta/raa_tobj": truth(raw, oe).tolist(), "meta/raa_ttgt": truth(raw, te).tolist(),
                "meta/raa_src_obj": str(src.get(oe, "site")), "meta/raa_src_tgt": str(src.get(te, "site")),
                "meta/raa_roles": [oe, te, S["agree"]]}

    Base = LC._wcp.WebsocketClientPolicy

    class RAAClient(Base):
        def infer(self, el):
            el = dict(el)
            try:
                el.update(channel(el))
            except Exception as e:                                          # noqa: BLE001  (a perception failure is logged, not hidden)
                logging.exception("[raa] channel failed")
                el.update({"meta/raa_ok": False, "meta/raa_why": "client error: %r" % (e,)})
            return super().infer(el)
    LC._wcp.WebsocketClientPolicy = RAAClient
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "N2 episodes need --fresh and --branches"
    print("[raa] client arm %s; lexicon %s" % (arm, LEX), flush=True)
    LC.main()


# ---------------------------------------------------------------- summaries and self test
def summarize(paths):
    rows = []
    for p in paths:
        rows += [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    out = dict(files=list(paths), queries=len(rows))
    for key in ("all", "object", "target"):
        e = np.array([r["err_cm"] for r in rows if key == "all" or r["role"] == key], np.float64)
        if len(e):
            out[key] = dict(n=int(len(e)), mean_cm=float(e.mean()), median_cm=float(np.median(e)), p90_cm=float(np.percentile(e, 90)),
                            max_cm=float(e.max()), within_1cm=float((e <= 1.0).mean()), within_2cm=float((e <= 2.0).mean()))
    by = {}
    for r in rows:
        by.setdefault(r["text"] + "|" + r["role"], []).append(r["err_cm"])
    out["by_instruction_role"] = {k: dict(n=len(v), mean_cm=float(np.mean(v)), p90_cm=float(np.percentile(v, 90))) for k, v in sorted(by.items())}
    src = {}
    for r in rows:
        src[r["src"]] = src.get(r["src"], 0) + 1
    out["sources"] = src
    out["roles_agree_with_given"] = sorted({json.dumps(r["roles"]) for r in rows})
    print(json.dumps(out, indent=1), flush=True)
    return out


def selftest():
    import sc_rv as RV
    assert RV.chan_cm([0.124, -0.0251, 0.006]) == [12, -3, 1], RV.chan_cm([0.124, -0.0251, 0.006])
    assert RV.chan_cm([-0.006, 0.0049, -0.0]) == [-1, 0, 0], RV.chan_cm([-0.006, 0.0049, -0.0])
    t = RV.chan_text("object: the bowl", [0.12, -0.03, 0.05], False)
    assert t == "object: the bowl. rel: +12 -3 +5 cm, role object", t
    t = RV.chan_text("put the bowl on the plate", [-0.4, 0.0, 0.001], True)
    assert t == "put the bowl on the plate. rel: -40 +0 +0 cm, role target", t
    lex, PF = lexicon()
    for s, want in (("put the bowl on the plate", ("akita_black_bowl_1", "plate_1")),
                    ("put the cream cheese on the plate", ("cream_cheese_1", "plate_1")),
                    ("put the wine bottle in the bowl", ("wine_bottle_1", "akita_black_bowl_1"))):
        print("[raa] role_parse %r -> %s" % (s, PF.role_parse(lex, s)), flush=True)
    print("[raa] selftest ok", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    elif len(sys.argv) > 1 and sys.argv[1] == "serve":
        ap = argparse.ArgumentParser()
        ap.add_argument("what")
        ap.add_argument("--arm", choices=ARMS, required=True)
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--port", type=int, required=True)
        ap.add_argument("--seed", type=int, default=13)
        ap.add_argument("--config", default="")
        ap.add_argument("--suffix", default="")
        ap.add_argument("--seg-ckpt", dest="seg_ckpt", default="")
        ap.add_argument("--diag", default="", help="raa: the learned masks' IoU log (sc_seg.LearnedMasks)")
        ap.add_argument("--chan-diag", dest="chan_diag", default="", help="per-query channel error vs simulator truth (jsonl)")
        ap.add_argument("--where_own", type=int, default=0, choices=(0, 1, 2, 3, 4, 5, 6, 7), help="SR 84 / 85 / 87: 0 = plain PolicyBackbone (unchanged path); 1 = W1; 2 = W2; 3 = W3; 4 = W4; 5 = W5 (W4 + C1 release-point aim + C2 handoff radius); 6 = W5c2 (W4 + C2 only); 7 = W5c1 (W4 + C1 only)")
        ap.add_argument("--where_log", default="", help="per-sample jsonl of the SR 84 branch")
        serve_cmd(ap.parse_args())
    else:
        raise SystemExit(__doc__)

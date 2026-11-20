"""SR 83 (A2 distributedness probe), capture client: a copy of steer_probe.py (SR 74; unchanged) whose SteerClient is replaced by the capture rule below.  LIBERO venv (py3.8), one task per process.

CHANGES against steer_probe.py: (1) probes N and K tasks (not only K); (2) only the CAP_Q-th (3rd) place-phase (held) query of each episode, one side query per swap
candidate (the target phrase swapped, no absent / nonsense words) then the real query, each carrying meta/a2_id for a2_capture.py; (3) the record keeps the
geometry (eef, the object, the trained target, every candidate: simulator positions, measurement only); (4) --ra-arm raac is accepted.  The original docstring follows.

SR 74: the steerability probe (docs/PLAN_PAPER.md 2c), client side.

An existing arm's client (fc_n2_client for I / rungs, tv_frozen tv / tvp, f1m_cue sb = tvpb, groot_box tvb, s2.py I / tvp) runs UNCHANGED, with
three layers added around it (nothing the policy sees or the episode executes changes):
  RawTap       innermost env wrapper: counts steps, keeps the last raw observation, steps a ProprioHeld exactly as tv_frozen.Front does
               (sc_rv_lex.json's threshold, every step from the WAIT-th, on the finger qpos, the last executed gripper command and the hand
               height).  Phase of a query: grasp = before the first hold; place = while held; nothing after a release.
  SteerClient  outermost client: before each real query of a trained task in a phase, up to four SIDE queries (meta/side, never executed):
                 cf        the counterfactual instruction (grasp: object phrase -> another movable object; place: target phrase -> another
                           target present), one partner per query, cycled over the candidates
                 absent    an object / target word that is not in the scene (the validity control)
                 nonsense  a nonsense word (the language-sensitivity baseline)
               and then the real query, unchanged, whose chunk is executed.  The server restores its counters and detector, so the side
               chunks are drawn with the real chunk's noise (steer_serve.py).
  front ends   tv / tvp / tvpb / tvb: the side query goes through the front end as deployed: a copy of the episode's Front (its mask network,
               its paint rule, its box rule) is re-run on the SAME raw observation with the counterfactual object / target, its detector
               replaced by the real front end's `on`; tvp / tvpb also write the counterfactual role phrases.  Servers of role-phrase rungs do
               the same inside the server.
Records (one JSON line per probed query, <out>_steer.jsonl): chunk displacement d (sum of the first 5 / 8 xyz deltas; for X-VLA's absolute
targets the n-th target minus the hand) of the true, cf, absent and nonsense chunks; the hand; p_old, p_new (simulator positions of the true and
the swapped-in entity: measurement only).  The scores are computed in steer_read.py.

    python experiments/steer_probe.py client --arm I|IR|tv|tvp|tvpb|tvb|RA [--ra-arm rrel|raa] --scene 1|2 --steer-out X_steer.jsonl [--seg] <libero_client args>
      IR = I against a role-phrase rung server (absent / nonsense words travel as meta/side_slot + meta/side_phrase).
"""
import copy
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from openpi_client import websocket_client_policy as W                      # noqa: E402  (before any arm patches the class)
TRUE_INFER = W.WebsocketClientPolicy.infer
import steer_text as ST                                                      # noqa: E402

WAIT, REPLAN = 10, 5
OUT = os.path.join(ROOT, "results", "flawed")
LEX = os.path.join(OUT, "sc_rv_lex.json")
CFG = dict(arm=None, scene=1, out=None, resize=224, tasks=None, task=None)
CAP_Q = 3                                                                    # SR 83: the 3rd place-phase (held) query of an episode (>= 10 steps after the hold onset)
ST_ = dict(cyc={"grasp": 0, "place": 0}, absent={"grasp": 0, "place": 0}, shadows={}, front_id=None, n_q=0, n_side=0, pl=0, done=False)


class FRONT:
    cur = None


class StubDet:
    """The real front end's hold signal for this observation (tap.on), in place of a second ProprioHeld."""

    def __init__(self, tap):
        self.tap = tap

    def step(self, *a):
        return self.tap.on


class RawTap:
    cur = None

    def __init__(self, env):
        from lifelong_ood.policy.parts_flow import ProprioHeld
        X = json.load(open(LEX, encoding="utf-8"))
        self._e, self.n, self.raw, self.cmd = env, 0, None, 0.0
        self.det, self.on, self.ever_on, self.q = ProprioHeld(X["tw"], X["lift"]), False, False, None
        RawTap.cur = self

    def __getattr__(self, k):
        return getattr(self._e, k)

    def step(self, a):
        r = self._e.step(a)
        self.n += 1
        obs = r[0]
        self.raw, self.cmd = obs, float(a[6])
        if self.n >= WAIT:
            self.on = bool(self.det.step(np.asarray(obs["robot0_gripper_qpos"], np.float64), self.cmd, float(obs["robot0_eef_pos"][2])))
            self.ever_on = self.ever_on or self.on
            row = self.n - WAIT
            self.q = None
            if row % REPLAN == 0:
                phase = "grasp" if not self.ever_on else ("place" if self.on else None)
                self.q = dict(row=row, on=self.on, phase=phase, eef=np.asarray(obs["robot0_eef_pos"], np.float64).copy())
        return r


def make_shadow(front, tap, obj, tgt):
    d = {}
    for k, v in front.__dict__.items():
        d[k] = copy.deepcopy(v) if isinstance(v, (list, dict)) and k not in ("E",) else v
    sh = object.__new__(type(front))
    sh.__dict__.update(d)
    sh.obj, sh.tgt, sh.det, sh.sheet = obj, tgt, StubDet(tap), False
    sh.score = lambda *a, **k: None
    return sh


def install(arm, scene, phrases):
    import libero_client as LC
    orig_env = LC.OffScreenRenderEnv
    LC.OffScreenRenderEnv = lambda *a, **k: RawTap(orig_env(*a, **k))
    if arm in ("tv", "tvp", "tvpb", "tvb"):
        import tv_frozen as TF
        init0 = TF.Front.__init__

        def init(self, *a, **k):
            init0(self, *a, **k)
            FRONT.cur = self
        TF.Front.__init__ = init
    orig_main = LC.main

    def main():
        Cur = LC._wcp.WebsocketClientPolicy
        task = CFG["task"]
        kind = "front" if arm in ("tv", "tvp", "tvpb", "tvb") else ("rung" if arm == "IR" else ("ra" if arm == "RA" else "plain"))

        class SteerClient(Cur):
            absolute = None

            def side(self, e):
                if kind == "ra":                                          # below RAAClient (no channel step), above the --seg payload layer
                    return Cur.__mro__[1].infer(self, e)
                return TRUE_INFER(self, e) if kind == "front" else Cur.infer(self, e)

            def infer(self, el):
                tap = RawTap.cur
                if el.get("meta/episode_start"):
                    ST_["pl"], ST_["done"] = 0, False                       # per-episode: place-phase query counter, capture done
                q = None if tap is None else tap.q
                if q is None or q["phase"] != "place" or "meta/episode" not in el:
                    return Cur.infer(self, el)
                ST_["pl"] += 1
                if ST_["done"] or ST_["pl"] != CAP_Q:                       # SR 83 rule: the CAP_Q-th place-phase (held) query of each episode
                    return Cur.infer(self, el)
                ST_["done"] = True
                if self.absolute is None:
                    self.absolute = (self.get_server_metadata() or {}).get("control") == "absolute"
                return self.probe(el, tap, q, Cur)

            def probe(self, el, tap, q, Cur):
                """SR 83 capture: at the CAP_Q-th held query, one side query per swap candidate (cf instruction = the target phrase swapped, as
                SR 74's place probe), then the real query (executed, unchanged).  Every one carries meta/a2_id: the server (a2_capture.py) stores the
                dict the model is given.  The record holds the geometry the open-loop read needs (simulator positions: measurement only)."""
                sc, ph = CFG["scene"], "place"
                o, t = task["object"], task["target"]
                raw_env = tap._e
                cl = ST.cands(sc, ph, o, t)
                trial = int(el["meta/episode"])
                base = "%s_t%d_q%d" % (task["key"], trial, q["row"])
                if kind == "ra":
                    RAP = ra_peek(Cur, el, tap)
                chunks = {}
                for c in cl:
                    text = ST.cf_text(sc, ph, o, t, c)
                    e = dict(el)
                    e["meta/side"] = True
                    e["meta/a2_id"] = "%s_cf_%s" % (base, c)
                    if kind == "ra":
                        e.update(ra_side(RAP, CFG["ra_arm"], "cf", text, el, q, ph, tap, None))
                    else:
                        e["prompt"] = text
                    chunks[c] = np.asarray(self.side(e)["actions"], np.float64)
                    ST_["n_side"] += 1
                e2 = dict(el)
                e2["meta/a2_id"] = base + "_true"
                r = Cur.infer(self, e2)                                    # the real query: executed
                if kind == "ra":
                    ra_check(RAP)
                chunks["true"] = np.asarray(r["actions"], np.float64)
                ST_["n_q"] += 1
                rec = dict(arm=arm, tid=task["tid"], key=task["key"], trial=trial, row=q["row"], base=base, object=o, target=t, cands=cl,
                           eef=[float(x) for x in q["eef"]], p_obj=[float(x) for x in ST.entity_pos(raw_env, o)],
                           p_tgt=[float(x) for x in ST.entity_pos(raw_env, t)],
                           p_cand={c: [float(x) for x in ST.entity_pos(raw_env, c)] for c in cl},
                           d5={k: [float(x) for x in ST.chunk_delta(v, q["eef"], self.absolute, 5)] for k, v in chunks.items()})
                with open(CFG["out"], "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                return r

            def shadow(self, tap, obj, tgt):
                fr = FRONT.cur
                if ST_["front_id"] != id(fr):
                    ST_["shadows"], ST_["front_id"] = {}, id(fr)
                k = (obj, tgt)
                if k not in ST_["shadows"]:
                    ST_["shadows"][k] = make_shadow(fr, tap, obj, tgt)
                return ST_["shadows"][k]

        LC._wcp = types.SimpleNamespace(WebsocketClientPolicy=SteerClient)
        print("[steer] arm %s scene %s kind %s task %s %r phrases %s out %s" % (arm, CFG["scene"], kind, task["key"], task["language"],
                                                                              phrases, CFG["out"]), flush=True)
        orig_main()
        print("[steer] queries probed %d side queries %d ra_peek_max_diff %s" % (ST_["n_q"], ST_["n_side"], ST_.get("ra_diff")), flush=True)
    LC.main = main


def _free(fn):
    return dict(zip(fn.__code__.co_freevars, [c.cell_contents for c in (fn.__closure__ or ())]))


def ra_peek(Cur, el, tap):
    """STEER-CONF adapter, arm RA (Rrel / RAA).  The estimate of every entity at THIS query, computed on COPIES of raa.py's client state
    (detector, Estimator) with exactly the steps raa.py's `channel` takes, so a side query never steps the client's Estimator.  The real
    query that follows steps the originals once, as an unprobed episode would; ra_check compares the two."""
    g = _free(_free(Cur.infer)["channel"])
    S, CUR, PF, lex, X, Cal, RV, FC, PU, PH = (g[k] for k in ("S", "CUR", "PF", "lex", "X", "Cal", "RV", "FC", "PU", "ProprioHeld"))
    env = CUR["env"]
    raw = env._e
    text = str(el["prompt"])
    if el.get("meta/episode_start") or S["det"] is None:
        roles = PF.role_parse(lex, text)
        det, E, first = PH(X["tw"], X["lift"]), RV.entity_tables(raw), True
        est = PU.Estimator(Cal, roles[0]) if roles else None
    else:
        roles, det, E, first, est = S["roles"], copy.deepcopy(S["det"]), S["E"], S["first"], copy.deepcopy(S["est"])
    out = dict(S=S, roles=roles, ok=roles is not None, FC=FC, RV=RV, PF=PF, lex=lex, pos=None)
    if roles is None:
        return out
    obs = env.obs
    ee = np.asarray(obs["robot0_eef_pos"], np.float64)
    R = FC._quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64))
    held = bool(det.step(np.asarray(obs["robot0_gripper_qpos"], np.float64), env.cmd, ee[2]))
    A, _ = PU.measure(raw, E, PU.AGENT, PU.MOV)
    F = PU.measure(raw, E, PU.AGENT, PU.fixture_names(E))[0] if first else None
    est.step(A, lambda: PU.measure(raw, E, PU.WRIST, PU.MOV)[0], F, held, ee, R, env.cmd)
    out["pos"] = np.asarray(est.pos, np.float64).copy()
    return out


def ra_side(P, arm, kd, text, el, q, ph, tap, w):
    """The side query's fields.  Channel rule (registered): cf = the lexicon parse of the swapped instruction and the estimate of the
    swapped-in entity minus the end effector; absent / nonsense words keep the true instruction's channel.  Rrel reads the whole
    instruction, so its prompt carries the suffix from here; RAA's server appends it from meta/raa_obj / meta/raa_tgt and paints the
    views (the absent / nonsense word goes in as a slot phrase, as for the role-phrase rungs)."""
    roles = P["roles"]
    if kd == "cf" and P["ok"]:
        roles = P["PF"].role_parse(P["lex"], text)
    ok = bool(P["ok"] and roles is not None)
    ent = P["FC"].ENT
    on = bool(q["on"])
    out = {"meta/raa_ok": ok}
    if ok:
        out["meta/raa_obj"] = [float(x) for x in P["pos"][ent.index(roles[0])]]
        out["meta/raa_tgt"] = [float(x) for x in P["pos"][ent.index(roles[1])]]
    else:
        out["meta/raa_why"] = "side query: no roles"
    if kd == "cf":
        out["prompt"] = text
    else:
        out["prompt"] = el["prompt"]
        if arm == "rrel":                                                   # whole-instruction prompt: the server swaps the text, keeps the suffix
            out["meta/side_text"] = text
            return out
        out["meta/side_slot"] = "object" if ph == "grasp" else "target"
        out["meta/side_phrase"] = w if ph == "grasp" else strip_prep(w)
    return out


def ra_check(P):
    if P["pos"] is None:
        return
    d = float(np.abs(P["S"]["est"].pos - P["pos"]).max())
    ST_["ra_diff"] = max(ST_.get("ra_diff", 0.0), d)
    if d > 1e-9:
        print("[steer] RA peek differs from the real estimator by %.3g m" % d, flush=True)


def strip_prep(w):
    for pre in ("on top of ", "on ", "in "):
        if w.startswith(pre):
            return w[len(pre):]
    return w



def take(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


def client_main():
    sys.argv.pop(1)                                                         # "client"
    arm = take("--arm")
    assert arm in ("I", "IR", "tv", "tvp", "tvpb", "tvb", "RA"), arm
    CFG["ra_arm"] = take("--ra-arm") if arm == "RA" else None
    scene = int(take("--scene", "1"))
    CFG.update(arm=arm, scene=scene, out=take("--steer-out"))
    assert CFG["out"], "--steer-out"
    if "--resize" in sys.argv:
        CFG["resize"] = int(sys.argv[sys.argv.index("--resize") + 1])
    tid = int(sys.argv[sys.argv.index("--tasks") + 1])
    tj = os.path.join(OUT, "n2_tasks.json") if scene == 1 else os.path.join(OUT, "s2", "s2_tasks.json")
    task = json.load(open(tj, encoding="utf-8"))["tasks"][tid]
    assert task["tid"] == tid
    assert ST.lang(scene, task["object"], task["target"]) == task["language"] or not task["key"][0] in "KN", (task["language"],)
    CFG["task"] = task
    seg = "--seg" in sys.argv
    if seg:
        sys.argv.remove("--seg")
    phrases = arm in ("tvp", "tvpb")
    install(arm, scene, phrases)
    if scene == 1:
        if arm == "RA":                                                     # STEER-CONF: Rrel / RAA through raa.py's client
            assert CFG["ra_arm"] in ("rrel", "raa", "raac"), CFG["ra_arm"]
            import raa as RAA
            sys.argv[1:1] = ["client", "--arm", CFG["ra_arm"]]
            RAA.client_main()
        elif arm in ("I", "IR"):
            import fc_n2_client as FN
            if seg:
                FN._seg_px()
            assert "--fresh" in sys.argv and "--branches" in sys.argv
            FN.LC.main()
        elif arm in ("tv", "tvp"):
            import tv_frozen as TF
            sys.argv[1:1] = ["client", "--arm", arm]
            TF.client_main()
        elif arm == "tvpb":
            import f1m_cue as FM
            sys.argv[1:1] = ["client", "--arm", "sb"]
            FM.client_main()
        else:
            import groot_box as GB
            sys.argv[1:1] = ["client", "--arm", "tvb"]
            GB.client_main()
    else:
        import s2
        assert arm in ("I", "tv", "tvp"), arm
        sys.argv[1:1] = ["client", "--arm", arm]
        s2.client_main()


if __name__ == "__main__":
    assert len(sys.argv) > 1 and sys.argv[1] == "client"
    client_main()

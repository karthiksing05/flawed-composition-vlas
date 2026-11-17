"""SR 74: the steerability probe (docs/PLAN_PAPER.md 2c), client side.  LIBERO venv (py3.8), one task per process.

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

    python experiments/steer_probe.py client --arm I|IR|tv|tvp|tvpb|tvb --scene 1|2 --steer-out X_steer.jsonl [--seg] <libero_client args>
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
ST_ = dict(cyc={"grasp": 0, "place": 0}, absent={"grasp": 0, "place": 0}, shadows={}, front_id=None, n_q=0, n_side=0)


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
        kind = "front" if arm in ("tv", "tvp", "tvpb", "tvb") else ("rung" if arm == "IR" else "plain")

        class SteerClient(Cur):
            absolute = None

            def side(self, e):
                return TRUE_INFER(self, e) if kind == "front" else Cur.infer(self, e)

            def infer(self, el):
                tap = RawTap.cur
                q = None if tap is None else tap.q
                if q is None or q["phase"] is None or "meta/episode" not in el or not task["key"].startswith("K"):
                    return Cur.infer(self, el)
                if self.absolute is None:
                    self.absolute = (self.get_server_metadata() or {}).get("control") == "absolute"
                return self.probe(el, tap, q, Cur)

            def probe(self, el, tap, q, Cur):
                S, sc, ph = ST.SCENES[CFG["scene"]], CFG["scene"], q["phase"]
                o, t = task["object"], task["target"]
                raw_env = tap._e
                cl = ST.cands(sc, ph, o, t)
                if not cl:
                    return Cur.infer(self, el)
                i = ST_["cyc"][ph]
                ST_["cyc"][ph] += 1
                cand = cl[i % len(cl)]
                j = ST_["absent"][ph]
                ST_["absent"][ph] += 1
                kinds = [("cf", cand), ("absent", None), ("nonsense", None)]
                chunks = {}
                words = {}
                for kd, c in kinds:
                    e = dict(el)
                    e["meta/side"] = True
                    if kd == "cf":
                        text = ST.cf_text(sc, ph, o, t, c)
                        obj2, tgt2 = (c, t) if ph == "grasp" else (o, c)
                        P2 = ST.phrases(sc, obj2, tgt2) if phrases else None
                    else:
                        text, w = ST.word_text(sc, ph, o, t, kd, j)
                        words[kd] = w
                        obj2, tgt2 = (("absent:" + w), t) if ph == "grasp" else (o, "absent:" + w)
                        P2 = ST.phrases_word(sc, ph, o, t, w) if phrases else None
                    if kind == "front":
                        sh = self.shadow(tap, obj2, tgt2)
                        out = sh.step(tap.raw, tap.n, tap.cmd)
                        from openpi_client import image_tools
                        ims = []
                        for key in ("agentview_image", "robot0_eye_in_hand_image"):
                            im = np.ascontiguousarray(np.asarray(out[key])[::-1, ::-1])
                            ims.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(im, CFG["resize"], CFG["resize"])))
                        e["observation/image"], e["observation/wrist_image"] = ims
                        e["prompt"] = P2[int(q["on"])] if phrases else text
                    elif kind == "rung" and kd != "cf":
                        e["prompt"] = el["prompt"]
                        e["meta/side_slot"] = "object" if ph == "grasp" else "target"
                        e["meta/side_phrase"] = (w if ph == "grasp" else strip_prep(w))
                    else:
                        e["prompt"] = text
                    chunks[kd] = np.asarray(self.side(e)["actions"], np.float64)
                    ST_["n_side"] += 1
                r = Cur.infer(self, el)                                    # the real query: executed
                chunks["true"] = np.asarray(r["actions"], np.float64)
                p_old = ST.entity_pos(raw_env, o if ph == "grasp" else t)
                p_new = ST.entity_pos(raw_env, cand)
                ST_["n_q"] += 1
                rec = dict(scene=sc, arm=arm, tid=task["tid"], key=task["key"], trial=int(el["meta/episode"]), row=q["row"], phase=ph,
                           on=bool(q["on"]), cyc=i, cand=cand, words=words, eef=[float(x) for x in q["eef"]],
                           p_old=[float(x) for x in p_old], p_new=[float(x) for x in p_new], absolute=bool(self.absolute),
                           d5={k: [float(x) for x in ST.chunk_delta(v, q["eef"], self.absolute, 5)] for k, v in chunks.items()},
                           d8={k: [float(x) for x in ST.chunk_delta(v, q["eef"], self.absolute, 8)] for k, v in chunks.items()})
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
        print("[steer] queries probed %d side queries %d" % (ST_["n_q"], ST_["n_side"]), flush=True)
    LC.main = main


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
    assert arm in ("I", "IR", "tv", "tvp", "tvpb", "tvb"), arm
    scene = int(take("--scene", "1"))
    CFG.update(arm=arm, scene=scene, out=take("--steer-out"))
    assert CFG["out"], "--steer-out"
    if "--resize" in sys.argv:
        CFG["resize"] = int(sys.argv[sys.argv.index("--resize") + 1])
    tid = int(sys.argv[sys.argv.index("--tasks") + 1])
    tj = os.path.join(OUT, "n2_tasks.json") if scene == 1 else os.path.join(OUT, "s2", "s2_tasks.json")
    task = json.load(open(tj, encoding="utf-8"))["tasks"][tid]
    assert task["tid"] == tid
    assert ST.lang(scene, task["object"], task["target"]) == task["language"] or not task["key"].startswith("K"), (task["language"],)
    CFG["task"] = task
    seg = "--seg" in sys.argv
    if seg:
        sys.argv.remove("--seg")
    phrases = arm in ("tvp", "tvpb")
    install(arm, scene, phrases)
    if scene == 1:
        if arm in ("I", "IR"):
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

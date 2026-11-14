"""GROOT-BOX: the refined front end (role views + the held object painted as its bounding box in the target phase) on FROZEN GR00T,
and whether GR00T's state input is a cue for F1m (the held bottle starts K4's high carry on N4 / N6).  (m3_prereg tag groot_box;
docs/SCAFFOLD_RESULTS.md 56; SR 46, 49, 51, 53, 55.)

No arm uses role phrases: phrases hurt GR00T (SR 46: tvp .167), so every arm keeps the task's own instruction (asserted per query).
Every arm is tv_frozen's arm tv underneath (the ProprioHeld sequencer, the learned segmenter, silhouette role views) with one or two
target-phase changes (detector on); no arm adds task information.
    tvb   tv + the held object painted as the bounding box of its learned mask in the target phase, target pixels visible
          (f1m_cue.CueFront with cue sb, unchanged: SR 53's sb without the phrases).
    tvg   tv + while the detector is on, the two gripper-finger entries of the observation (robot0_gripper_qpos, which libero_client
          sends as observation/state[6:8]; state = eef_pos 3, axis-angle 3, gripper_qpos 2) are replaced by G_GROOT, the mean finger
          qpos recorded while GR00T holds the cream cheese on K6 (its trained cheese task) in the smoke.  The detector reads the real
          qpos (it is stepped before the swap).
    tvbg  tvb and tvg together (descriptive).
    tv, I run by existing clients unchanged (tv_frozen.py client --arm tv; pp_atlas_client.py).
GR00T reads the state (unlike pi05_libero, SR 53): GrootBackbone.prefix min-max normalises the 8-D state and its state encoder's
token heads the DiT's sequence (kv["sf"]).  `statecheck` measures this on the served policy (GuidedPolicy, the server's seed rule):
action chunks under state swaps against image and prompt swaps.

Modes:
    LIBERO venv:  python experiments/groot_box.py client --arm tvb|tvg|tvbg <pp_atlas_client.py's arguments>    one task per process
                  -> <out>.json, <out>_track.npz, <out>_tv.json (per episode: detector, paint, box sizes, held finger qpos, the
                     finger entries of observation/state actually sent at held queries)
                  python experiments/groot_box.py summary DIR [DIR ...]                                        smoke table
    GR00T venv:   python experiments/groot_box.py statecheck --out X.json                                      GPU
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import tv_frozen as TF                                                      # noqa: E402

ARMS = ("tvb", "tvg", "tvbg")
MODEL = "groot"
G_PROV = (0.0198, -0.0223)                                                  # f1m_cue.G_HELD (pi0.5's K6 cheese hold): smoke only
# Set from smoke job 4710084 (disclosed in the prereg row's "seen"): GR00T's mean finger qpos while it holds the cream cheese on K6
# under tvb (n 14 held queries: .02048, -.02168; tvg's raw K6 qpos .02045, -.02171), rounded.  GR00T's bottle on K4 is held at
# ~(.0072, -.0077); on N4 / N6 the held qpos in the smoke averaged ~(.018, -.020).
G_GROOT = (0.0205, -0.0217)
G_BOTTLE = (0.008, -0.008)                                                  # SR 53: the bottle is held at about this


def g_value():
    return tuple(G_GROOT) if G_GROOT is not None else tuple(G_PROV)


def make_front(env, arm, obj, tgt, X, seg, g, sheet=False):
    import f1m_cue as F1

    class GFront(F1.CueFront):
        """f1m_cue.CueFront (tv underneath) with cue sb for tvb / tvbg; tvg is tv itself; tvg / tvbg swap the finger qpos when on."""

        def __init__(self):
            super().__init__(env, "sb" if arm in ("tvb", "tvbg") else "g", obj, tgt, X, seg, sheet=sheet)
            self.garm, self.g, self.sent = arm, tuple(float(x) for x in g), []

        def step(self, obs, n, cmd):
            if self.garm == "tvg":
                out = TF.Front.step(self, obs, n, cmd)                      # arm tv, unchanged
                if n >= TF.WAIT and self.on and (n - TF.WAIT) % TF.REPLAN == 0:
                    self.qheld.append([float(x) for x in obs["robot0_gripper_qpos"]])
            else:
                out = F1.CueFront.step(self, obs, n, cmd)                   # cue sb: tv + held-object box (logs qheld itself)
            if self.garm in ("tvg", "tvbg") and n >= TF.WAIT and self.on:
                out = dict(out)
                q = np.asarray(obs["robot0_gripper_qpos"])
                out["robot0_gripper_qpos"] = np.asarray(self.g, q.dtype).reshape(q.shape)
                self.g_steps += 1
            return out

        def record(self, held):
            r = F1.CueFront.record(self, held)
            r.update(arm=self.garm, g_used=list(self.g) if self.garm in ("tvg", "tvbg") else None, sent_held=self.sent)
            return r
    return GFront()


# ---------------------------------------------------------------- client (LIBERO venv)
def client_main():
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
    if os.path.abspath(OUTF).startswith(os.path.join(ROOT, "results") + os.sep):
        assert G_GROOT is not None, "main runs need G_GROOT (set from the smoke before the prereg row)"
    G = g_value()
    TASK = FN.N2["tasks"][TID]
    assert TASK["tid"] == TID and TASK["layout"] == "familiar", TASK
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    seg = SG.load(None, "cuda")
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
            self.front = make_front(self._e, arm, TASK["object"], TASK["target"], X, seg, G, sheet=sheet and len(SIDE) < 2)
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
                    json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], phrases=None, lex=X, seg_ckpt=seg.path,
                                   g=list(G), g_groot_set=G_GROOT is not None, eps=SIDE), fh, indent=1)
                os.replace(tmp, SIDEF)
                sh = rec["sent_held"]
                print("[gbox] %s %s trial %s: onsets %s (oracle first held %s) held queries %d/%d painted %s g_steps %d sent_held %d%s" % (
                    arm, TASK["key"], rec["trial"], rec["onsets"][:4], rec["oracle_first_held"], rec["held_queries"], rec["queries"],
                    rec["painted_queries"], rec["g_steps"], len(sh), (" first %s" % sh[0]) if sh else ""), flush=True)
                self.front = None
            self._e.close()

    Base = LC._wcp.WebsocketClientPolicy

    class SClient(Base):                                                    # no phrases: the prompt is asserted unchanged
        def infer(self, el):
            env = CUR.get("env")
            f = None if env is None else env.front
            assert str(el["prompt"]) == TASK["language"], el["prompt"]
            if f is not None:
                on = bool(f.on)
                f.prompts["target" if on else "object"] += 1
                if on:
                    f.sent.append([float(x) for x in np.asarray(el["observation/state"])[6:8]])
            return super().infer(el)
    LC._wcp.WebsocketClientPolicy = SClient
    LC.OffScreenRenderEnv = lambda *a, **k: TVEnv(ORIG(*a, **k))
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    print("[gbox] arm %s task %s %r object %s target %s g %s (G_GROOT set %s) seg %s" % (
        arm, TASK["key"], TASK["language"], TASK["object"], TASK["target"], G, G_GROOT is not None, seg.path), flush=True)
    LC.main()


# ---------------------------------------------------------------- state check (GR00T venv, GPU)
def statecheck(a):
    """Does the served GR00T read observation/state, and do the finger entries move its actions?  Same GuidedPolicy and seed rule as
    pp_serve_purify --model groot --mode intact --seed 13; each call is an episode start with a fixed seed_base, so every call draws
    the same noise.  Action chunks under state swaps, against image and prompt swaps (the sensitivity scale), as f1m_cue.statecheck."""
    from lifelong_ood.policy.serve import GuidedPolicy
    from pp_f1_mix import Groot
    M = Groot()
    bk = M.bk
    pol = GuidedPolicy(bk, hooks=None, num_steps=bk.num_steps, seed=13)
    rng = np.random.default_rng(0)
    base = np.array([-0.06, 0.02, 1.05, 3.10, 0.05, -0.08, G_BOTTLE[0], G_BOTTLE[1]], np.float32)   # bottle-held fingers
    G = np.asarray(g_value(), np.float32)
    var = {"grip_cheese": np.r_[base[:6], G], "grip_open": np.r_[base[:6], [0.039, -0.039]],
           "grip_closed": np.r_[base[:6], [0.0, 0.0]], "eef_z_plus_0.20": base + np.array([0, 0, 0.2, 0, 0, 0, 0, 0], np.float32),
           "all_random": (base + rng.normal(0, 1.0, 8)).astype(np.float32), "zeros": np.zeros(8, np.float32)}
    prompts = ("put the wine bottle on the plate", "put the wine bottle in the bowl", "put the wine bottle on top of the cabinet")
    sd = bk.policy.state_dict() if hasattr(bk, "policy") else {}
    res = dict(model=MODEL, describe=bk.describe(), g=G.tolist(), g_groot_set=G_GROOT is not None, base_state=base.tolist(),
               state_layout="libero_client observation/state = eef_pos(3) + quat2axisangle(eef_quat)(3) + robot0_gripper_qpos(2)",
               state_encoder_params=int(sum(p.numel() for p in bk.head.state_encoder.parameters())),
               state_keys=[k for k in sd if "state_encoder" in k][:6], cases=[])
    try:                                                                    # the pre-processor's steps (diagnostics only)
        res["pre_steps"] = [type(s).__name__ for s in bk.pre.steps]
    except Exception as e:
        res["pre_steps_error"] = repr(e)
    try:
        o0 = bk.observation({"observation/image": np.zeros((224, 224, 3), np.uint8), "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
                             "observation/state": base, "prompt": prompts[0]})
        o1 = bk.observation({"observation/image": np.zeros((224, 224, 3), np.uint8), "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
                             "observation/state": np.r_[base[:6], G], "prompt": prompts[0]})
        s0, s1 = o0.state.float().cpu().numpy().reshape(-1), o1.state.float().cpu().numpy().reshape(-1)
        res["normalised_state_base_first8"] = s0[:8].tolist()
        res["normalised_state_cheese_first8"] = s1[:8].tolist()
        res["normalised_state_diff_dims"] = [int(i) for i in np.flatnonzero(np.abs(s0 - s1) > 0)]
    except Exception as e:
        res["normalise_error"] = repr(e)

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
            ac = run(img, wimg, var["grip_cheese"], pr, sb)
            row["grip_cheese_meanabs"] = float(np.abs(ac - a0).mean())
            row["grip_cheese_dz_mean"] = float((ac[:, 2] - a0[:, 2]).mean())         # action dim 2 = delta z
            row["grip_cheese_grip_cmd"] = [float(a0[:, 6].mean()), float(ac[:, 6].mean())]
            res["cases"].append(row)
            print("[statecheck] %s" % json.dumps(row), flush=True)
    C = res["cases"]
    res["state_invariant"] = all(v == 0.0 for c in C for v in c["state_maxabs"].values())
    res["finger_swap_moves_actions"] = all(c["state_maxabs"]["grip_cheese"] > 1e-3 for c in C)
    res["repeat_exact"] = all(c["repeat_maxabs"] == 0.0 for c in C)
    res["image_sensitive"] = all(c["image_maxabs"] > 1e-3 for c in C)
    res["prompt_sensitive"] = all(c["prompt_maxabs"] > 1e-3 for c in C)
    res["median"] = {k: float(np.median([c["state_maxabs"][k] for c in C])) for k in var}
    res["median"].update(image=float(np.median([c["image_maxabs"] for c in C])), prompt=float(np.median([c["prompt_maxabs"] for c in C])))
    print("[statecheck] %s | state invariant %s finger swap moves actions %s repeat exact %s image sensitive %s prompt sensitive %s" % (
        res["describe"], res["state_invariant"], res["finger_swap_moves_actions"], res["repeat_exact"], res["image_sensitive"],
        res["prompt_sensitive"]), flush=True)
    print("[statecheck] median max|d action| %s" % json.dumps({k: round(v, 4) for k, v in res["median"].items()}), flush=True)
    print("[statecheck] normalised state diff dims %s base %s cheese %s" % (res.get("normalised_state_diff_dims"),
          res.get("normalised_state_base_first8"), res.get("normalised_state_cheese_first8")), flush=True)
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), out
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out, flush=True)


# ---------------------------------------------------------------- smoke summary (LIBERO venv)
def summary(dirs):
    import f1m_cue as F1
    F1.summary(dirs)
    for d in dirs:
        for f in sorted(x for x in os.listdir(d) if x.endswith("_tv.json")):
            S = json.load(open(os.path.join(d, f), encoding="utf-8"))
            s = [x for e in S["eps"] for x in (e.get("sent_held") or [])]
            if s:
                Q = np.asarray(s, float)
                print("== %s %s sent state[6:8] at held queries: n %d mean %s min %s max %s (g %s)" % (
                    f, S["key"], len(s), Q.mean(0).round(5).tolist(), Q.min(0).round(5).tolist(), Q.max(0).round(5).tolist(),
                    S.get("g")), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        return client_main()
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        return summary(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("statecheck",))
    ap.add_argument("--out", required=True)
    statecheck(ap.parse_args())


if __name__ == "__main__":
    main()

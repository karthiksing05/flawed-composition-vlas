"""WA-s (SR 82, docs/notes/raasl_plan.md; m3_prereg tag raasl): a COPY of wa.py (SR 79 / 80, unchanged) for the RAA-s family (RAA-sL = pi05_g8v_raas_l_ae).
The ONLY changes against wa.py (everything else, the arms, cells, resolver, pose memory, rollouts and the log record, is the same text):
  (1) the role channel goes to the model as RAA-c's continuous state token chan_state(rel, on) = [dx, dy, dz (m), role id] (the prompt stays the role phrase), as
      raa.py --arm raac / raas does, instead of RAA's text suffix; held target: rel = target - eef - offset, the offset = the Estimator's object estimate minus the eef
      at the held onset, fixed while held (raa.py's raao / raac rule; the client sends meta/raa_off_ok / off / toff as raa.py's client does);
  (2) the noM skip (the active role entity has a held estimate) sends NO channel: the base prompt and the all-zero state (the unrouted K7 / K8 convention);
      a frame with no channel (unrouted, lexicon refused) is the all-zero state, never the 8-dim eef state;
  (3) the server warms up with a 4-dim state (the norm stats are 4-dim), as raa.py.
Sampling is unchanged: wa.py already serves GuidedPolicy(PolicyBackbone, hooks=None) = PolicyBackbone.sample, which adds the servo when servo_k != 0 (SR 78b fix).
Original docstring follows.

WA (docs/notes/wa_plan.md; m3_prereg tag wa): a module-removal (double dissociation) test of the WHERE architecture, on the
EXISTING RAA checkpoint (SR 73: pi05_g8v_raa_ae_tl6000), no training, on rg_comp.py's rgc harness (F7: tids 0-3; F6: tids 8, 9; known
tids 7, 10; the main states of results/flawed/rg_comp, unchanged).

THE TWO MODULES
  R  referent resolver.  fx_r.py's resolver (template parser + learned U-Net + depth; imported, unchanged) picks the bowl INSTANCE the
     instruction names ("the left bowl", "the bowl that is not on the plate").  The resolved instance (i) fills RAA's object slot: the
     Estimator is run on that instance's pixels only (every bowl pixel goes to the nearer of two tracked 3D centres, rg_comp's
     Episode.restricted rule), so the channel's object position is that bowl's; (ii) is the instance left natural in the role view: the
     other bowl is painted magenta (fx_r.Front's distractor mask, in both cameras, the server paints it on the raw renders) like every
     other non-filler movable; (iii) gives the prompt: F7R's unique-name instruction ("put the bowl on the plate|stove"), whose roles are
     (akita_black_bowl_1, the parsed target).  Off the 4 templates (every K and F6 cell) R is not engaged: the lexicon's roles.
  M  pose memory.  sc_pu.Estimator's hold: when an entity's mask is empty in both cameras the estimate is the last one made while it was
     visible (src "last"; "occl" when the mask is partly hidden: the last good pose replaces the biased partial-mask pose).  RAA already
     has both (calib: occl on).  A held estimate = src in (occl, last, prior) for a movable; fixtures (the stove site) never are.

THE ARMS (the server is the same for all; the client decides)
  full  R + M.
  noR   M, no R: roles from RAA's lexicon (role_parse -> the n2 task whose (object, target) it is -> that task's instruction, the text
        RAA's role phrases are keyed by), original image; RAA's Estimator AVERAGES the two identical bowls (the learned map has one bowl
        class, sc_pu takes the median over all its pixels), so noR binds ONE instance by rgp's rule (rg_comp.Episode.instances: the
        bowl class's 8-connected components >= fx_r.MIN_COMP at the first query, the LARGEST is the object, the second is the other
        bowl; tracked with fx_r.Front's update rule; no paint, the prompt is the lexicon's).  Where the scene has one bowl component
        (every K and F6 cell, K2t's second bowl is unrendered) nothing is restricted, so noR IS RAA there and R is not engaged in
        full either: on those cells noR and full are the same arm; noR is rolled there too, as a replicate (a noise floor for D3a and DK).
  noM   R, no M.  Whenever the ACTIVE role entity (object until ProprioHeld, then target; the server's ProprioHeld decides) has a HELD
        estimate (Estimator src occl, last or prior: not a measurement of it now), the channel suffix is omitted (the prompt is the role
        phrase alone, "target: the plate") and the role paint of that query is absent (the server paints nothing; the client's own image,
        which already carries R's distractor paint, goes through).  Every query with a measured role entity is exactly `full`'s.
  noMe  descriptive variant of noM: only an EMPTY-mask hold (src last, prior) is withheld; the occlusion rule's hold (src occl: the mask
        is partly visible, the pose is the last good one) is kept.

    python experiments/wa.py serve  --ckpt DIR --port P [--seed 13] [--seg-ckpt F] [--chan-diag F.jsonl] [--diag F.json]     (openpi venv)
    python experiments/wa.py client --arm full|noR|noM <libero_client.py's arguments>                                       (LIBERO venv)
    python experiments/wa.py roles                                                      the lexicon's roles for every rgc instruction
    python experiments/wa.py selftest                                                   CPU: cells, prompt mapping
"""
import argparse
import collections
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
LEX = os.path.join(OUT, "sc_lex.json")
B1, B2 = "akita_black_bowl_1", "akita_black_bowl_2"
WAIT, REPLAN = 10, 5
ARMS = ("full", "noR", "noM", "noMe")
CELLS = {"full": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), "noR": (0, 1, 2, 3, 7, 8, 10), "noM": (0, 1, 2, 3, 8), "noMe": (0, 1, 2, 3, 8)}
# (arm -> rgc tids).  R is not engaged on tids 7-10 (no relational template, one bowl), so noR IS full there by construction: rolled as a replicate
# (same code path, no front end), which makes D3a / DK measure rollout noise rather than an identity.
STALE, EMPTY = ("occl", "last", "prior"), ("last", "prior")                  # estimator sources that are not a measurement of the entity now (occl = the occlusion rule's hold)
MAGENTA = np.array([255, 0, 255], np.uint8)


def lexicon():
    from lifelong_ood.policy import parts_flow as PF
    sit = json.load(open(LEX, encoding="utf-8"))["situations"]
    return PF.role_lexicon(sit), PF


def n2_prompt(roles):
    """(object slot, target slot) -> the instruction of the first routed n2 task with those roles (the text RAA's role phrases are keyed by)."""
    tasks = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))["tasks"]
    for t in tasks:
        if t["compound"] in ("K7", "K8", "N7", "N8"):
            continue
        if (t["object"], t["target"]) == tuple(roles):
            return t["language"]
    return None


# ---------------------------------------------------------------- server (openpi venv)
def _server_classes():
    import sc_rv as RV
    from lifelong_ood.policy import parts_flow as PF                            # noqa: F401

    class WARoleView(RV.RoleView):
        """sc_rv.RoleView.infer (mode rview2), with (a) the noM skip: no role paint on a query whose active role entity is hidden when the
        client says nomem, (b) R's distractor mask painted magenta on the raw renders (meta/wa_dm_*)."""
        wa = {}
        last_skip, last_on = False, False

        def infer(self, obs):
            from openpi_client import image_tools
            o = dict(obs)
            rv = {k: o.pop(k) for k in list(o) if k.startswith("meta/rv_")}
            if o.get("meta/episode_start") or self.det is None:
                self.det, self.prev, self.checked = self.PF.ProprioHeld(self.X["tw"], self.X["lift"]), -1.0, False
            st = np.asarray(o["observation/state"])
            on = bool(self.det.step(st[6:8], self.prev, st[2]))
            text = str(o["prompt"])
            P = self.T16.given(text)
            if P is not None:
                o["prompt"] = P[int(on)]
            wa = self.wa
            skip = bool(wa.get("nomem")) and not bool(wa.get("vis_" + ("tgt" if on else "obj"), True))
            self.last_skip, self.last_on = skip, on
            na = nw = 0
            if not rv:
                raise RuntimeError("the WA server needs the client's --seg payload")
            hw = np.asarray(o["observation/image"]).shape[:2]

            def rs(x):
                return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(x), hw[0], hw[1]))
            if P is not None and not skip:
                obj, tgt = self.roles[text]
                sl = [list(x) for x in rv["meta/rv_slots"]]

                def sidx(s):
                    return RV.slot_index(rv["meta/rv_names"], [x for x, _ in sl], [e for _, e in sl], s)
                ents, act = RV.role_paint(obj, tgt, on)
                pid, aid = [sidx(s) for s in ents], (None if act is None else sidx(act))
                img, na = RV.paint_many(np.asarray(rv["meta/rv_img_agent"]), np.asarray(rv["meta/rv_seg_agent"]), pid, aid)
                wimg, nw = RV.paint_many(np.asarray(rv["meta/rv_img_wrist"]), np.asarray(rv["meta/rv_seg_wrist"]), pid, aid)
                dm = [wa.get("dm_agent"), wa.get("dm_wrist")]
                if dm[0] is not None and np.asarray(dm[0]).any():
                    img = np.array(img, copy=True)
                    img[np.asarray(dm[0], bool)] = MAGENTA
                    na += int(np.asarray(dm[0]).sum())
                if dm[1] is not None and np.asarray(dm[1]).any():
                    wimg = np.array(wimg, copy=True)
                    wimg[np.asarray(dm[1], bool)] = MAGENTA
                    nw += int(np.asarray(dm[1]).sum())
                if na:
                    o["observation/image"] = rs(img)
                if nw:
                    o["observation/wrist_image"] = rs(wimg)
            logging.info(f"[wa] t={o.get('meta/t')} held={on} skip={skip} prompt={o['prompt']!r} painted agent={na} wrist={nw}")
            r = self.p.infer(o)
            self.prev = float(np.asarray(r["actions"])[self.replan - 1, 6])
            return r

    class _Hook:
        def __init__(self, pol, srv):
            self.p, self.srv = pol, srv
            self.metadata = getattr(pol, "metadata", {})

        def infer(self, o):
            return self.srv.final(o, self.p)

    class WAServer:
        """raa.RAAServer for arm raa (SR 73's rule: prompt = role phrase + "rel: dx dy dz cm, role r"), with the WA skip and a per-query log."""

        def __init__(self, policy, X, diag=None, replan=5):
            self.RV, self.diag = RV, diag
            self.T16 = RV.t16()
            self.metadata = getattr(policy, "metadata", {})
            self.rv = WARoleView(_Hook(policy, self), X, mode="rview2", replan=replan)
            self.cur, self.n, self.noch, self.nskip = None, 0, 0, 0

        def infer(self, obs):
            o = dict(obs)
            raa = {k[len("meta/raa_"):]: o.pop(k) for k in list(o) if k.startswith("meta/raa_")}
            wa = {k[len("meta/wa_"):]: o.pop(k) for k in list(o) if k.startswith("meta/wa_")}
            self.cur = dict(text=str(o["prompt"]), raa=raa, wa=wa, t=o.get("meta/t"), ep=o.get("meta/episode"))
            self.rv.wa = wa
            return self.rv.infer(o)

        def final(self, o, pol):
            c, wa = self.cur, self.cur["wa"]
            P = self.T16.given(c["text"])
            if P is not None:
                on = str(o["prompt"]) == P[1]
                base = str(o["prompt"])
                r = c["raa"]
                skip = bool(self.rv.last_skip)
                role = "tgt" if on else "obj"
                if r.get("ok"):
                    ee = np.asarray(o["observation/state"], np.float64)[:3]
                    rel = np.asarray(r[role], np.float64) - ee
                    off = toff = np.zeros(3)
                    if on:                                                  # SR 82: held target = target - eef - offset (raa.py raao / raac)
                        if r.get("off_ok"):
                            off, toff = np.asarray(r["off"], np.float64), np.asarray(r["toff"], np.float64)
                        else:
                            self.nooff = getattr(self, "nooff", 0) + 1
                            logging.warning(f"[wa-s] held but no offset at ep {c['ep']} t {c['t']}")
                        rel = rel - off
                    if skip:
                        o["prompt"] = base
                        o["observation/state"] = self.RV.chan_state(None, False, False)
                        self.nskip += 1
                    else:
                        o["prompt"] = base
                        o["observation/state"] = self.RV.chan_state(rel, on)
                        self.n += 1
                    if self.diag and "t" + role in r:
                        true = np.asarray(r["t" + role], np.float64) - ee - toff
                        d = rel - true
                        rec = dict(arm=str(wa.get("arm")), tid=int(wa.get("tid", -1)), ep=c["ep"], t=c["t"], text=c["text"],
                                   role="target" if on else "object", omitted=skip, rel_cm=self.RV.chan_cm(rel),
                                   true_cm=self.RV.chan_cm(true), err_cm=float(100 * np.linalg.norm(d)), src=str(r.get("src_" + role)),
                                   vis=bool(wa.get("vis_" + role, True)), hidden_truth=bool(wa.get("hidden", False)),
                                   roles=r.get("roles"), prompt=o["prompt"])
                        with open(self.diag, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(rec) + "\n")
                else:
                    o["prompt"] = base
                    self.noch += 1
                    logging.warning(f"[wa] no channel at ep {c['ep']} t {c['t']}: {r.get('why')}")
            if np.asarray(o["observation/state"]).shape[-1] != self.RV.CHAN_DIM:   # unrouted or no channel: zeros, never the 8-dim eef state
                o["observation/state"] = self.RV.chan_state(None, False, False)
            logging.info(f"[wa] ep={c['ep']} t={c['t']} prompt={o['prompt']!r} state={np.round(np.asarray(o['observation/state']), 4).tolist()}")
            return pol.infer(o)
    return WAServer


def _make_bk(PolicyBackbone, name, a):
    """SR 84 (copy of raa.py / wa_s.py with this one change): --where_own 0 (default) builds exactly PolicyBackbone(name, checkpoint_dir, device) as before;
    1 | 2 build lifelong_ood.policy.backbone_w.PolicyBackboneW (the guarded branch after the servo)."""
    if not getattr(a, "where_own", 0):
        return PolicyBackbone(name, checkpoint_dir=a.ckpt, device="cuda")
    from lifelong_ood.policy.backbone_w import PolicyBackboneW
    logging.info(f"[sr84] PolicyBackboneW where_own {a.where_own} log {a.where_log or None}")
    return PolicyBackboneW(name, checkpoint_dir=a.ckpt, device="cuda", where_own=a.where_own, log_path=a.where_log or None)


def serve_cmd(a):
    import sc_rv as RV
    RV._openpi()
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import GuidedPolicy, serve, warm
    import sc_seg as SEG
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    name = a.config or RV.FULL["raa"] + "_ae"
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    logging.info(f"[wa-s] serve (RAA-s family, SR 82; state-token channel) config {name} ckpt {a.ckpt} tw {X['tw']:.6f} lift {X['lift']} chan_diag {a.chan_diag}")
    bk = _make_bk(PolicyBackbone, name, a)
    pol = GuidedPolicy(bk, hooks=None, num_steps=10, seed=a.seed)        # PolicyBackbone.sample adds the servo when servo_k != 0
    from openpi.policies import libero_policy
    for _ in range(2):                                                      # the norm stats are 4-dim: warm up with a 4-dim state, as raa.py --arm raac
        ex = libero_policy.make_libero_example()
        ex["observation/state"] = np.zeros(RV.CHAN_DIM, np.float32)
        pol.infer(ex)
    srv = _server_classes()(pol, X, diag=a.chan_diag or None)
    seg = SEG.load(a.seg_ckpt or None, "cuda")
    logging.info(f"[wa] learned masks {seg.path}")
    serve(SEG.LearnedMasks(srv, seg, diag=a.diag or None), a.port)


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
    import libero_client as LC
    from libero.libero import benchmark
    import fa1_common as FC
    import fx_r as XR
    import rg_comp as RGC
    import sc_pu as PU
    import sc_rv as RV
    import sc_seg as SEG
    from lifelong_ood.policy.parts_flow import ProprioHeld
    meta = json.load(open(RGC.META, encoding="utf-8"))
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = meta["tasks"][TID]
    assert TASK["tid"] == TID and TID in CELLS[arm], (TASK, arm)
    assert sys.argv[sys.argv.index("--suite") + 1] == RGC.SUITE and "," not in sys.argv[sys.argv.index("--tasks") + 1]
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "rgc episodes need --fresh and --branches"
    if TASK["kind"] == "f6":
        import f6o_tasks as OT
        CV = meta["cover"]
        OT.install(CV["radius"], CV["half_h"], CV["n"])
    X = json.load(open(RV.lex_path(""), encoding="utf-8"))
    Cal = PU.load_calib()
    SEGM = SEG.load(os.environ.get("SC_SEG_CKPT") or None, "cuda")
    lex, PF = lexicon()
    # ---- what the arm hands the VLA: prompt and roles
    parsed = XR.parse(TASK["language"]) if arm in ("full", "noM", "noMe") else None
    if parsed is not None:                                                  # R engaged (the 4 relational templates)
        prompt, roles = parsed["unique"], (B1, RGC.NOUN_TARGET[parsed["target"]])
        assert tuple(PF.role_parse(lex, prompt)) == roles, (prompt, PF.role_parse(lex, prompt), roles)
    else:                                                                   # RAA's lexicon, as RAA does today
        lr = PF.role_parse(lex, TASK["language"])
        assert lr is not None, TASK["language"]
        roles = tuple(lr)
        prompt = n2_prompt(roles)
        assert prompt is not None, (TASK["language"], roles)
    oe, te = roles
    front_mode = "resolver" if parsed is not None else ("largest" if (arm == "noR" and TID <= 3) else None)
    print("[wa] arm %s task %s %r -> prompt %r roles %s front %s" % (arm, TASK["key"], TASK["language"], prompt, roles, front_mode), flush=True)

    class WFront(XR.Front):
        """fx_r.Front with the per-step distractor masks kept (for the server's paint) and a track-only mode (noR: the largest component)."""

        def __init__(self, env, parsed_, mode):
            super().__init__(env, "f7r", parsed_)
            self.mode, self.dms, self.dm = mode, [], {}

        def resolve(self, obs):
            if self.mode == "resolver":
                return super().resolve(obs)
            img = RV.flip(obs["agentview_image"])
            cls = self.seg.classes_of(img[None], [0], post_=False)[0]
            P = self.cloud(XR.CAMS[0])
            comps = XR.components(cls == self.cb, XR.MIN_COMP)
            if len(comps) < 2:                                              # one component: nothing is restricted (RAA as is)
                self.res = dict(ok=False, how="none (one instance)", n_comp=len(comps))
                return self.res
            cen = [XR.centre(m, P) for m in comps[:2]]
            b1, b2 = self.sim_pos(B1), self.sim_pos(B2)
            truth = int(np.argmin([np.linalg.norm(c[:2] - b1[:2]) for c in cen]))
            self.ref, self.dis, self.dis_px0 = cen[0].copy(), cen[1].copy(), int(comps[1].sum())
            self.res = dict(ok=True, how="largest", n_comp=len(comps), used=0, truth=truth, used_correct=bool(truth == 0),
                            resolver_correct=bool(truth == 0), cand_px=[int(m.sum()) for m in comps[:2]], centres=[c.tolist() for c in cen],
                            xy_err_b1=float(np.linalg.norm(cen[truth][:2] - b1[:2])), sim_b1=b1.tolist(), sim_b2=b2.tolist())
            return self.res

        def dmask(self, cls, P, update):
            out = super().dmask(cls, P, update)
            self.dms.append(out)
            return out

        def step(self, obs, n, wait):
            self.dms = []
            out = super().step(obs, n, wait)
            self.dm = dict(zip(XR.CAMS, self.dms)) if len(self.dms) == 2 else {}
            return obs if self.mode != "resolver" else out

    CUR = dict(env=None, front=None)
    CACHE = {}

    def seg_std(env, E, cam):
        """sc_pu.seg_std's contract from the learned U-Net on an RGB render; with a tracked pair of bowls, the bowl class keeps only the
        pixels nearer the object instance's centre than the other's (rg_comp.Episode.restricted's rule, centres from fx_r.Front's tracking)."""
        sim = env.env.sim
        key = (id(sim), float(sim.data.time), cam)
        if key not in CACHE:
            img = RV.flip(sim.render(camera_name=cam, width=PU.SIZE, height=PU.SIZE, depth=False))
            p = SEGM.ent(img, cam, E["names"])
            if len(CACHE) > 8:
                CACHE.clear()
            CACHE[key] = np.ascontiguousarray(np.asarray(p)[:, ::-1])
        s = np.array(CACHE[key], copy=True)
        fr = CUR["front"]
        if fr is not None and (fr.res or {}).get("ok") and fr.ref is not None and fr.dis is not None:
            m = s == E["names"].index(B1)
            if m.any():
                d = PU.depth_std(env, E, cam)
                fz, Rc, c = PU.cam_model(sim, cam)
                r, cc = np.nonzero(m)
                dd = d[r, cc]
                q = np.stack([(cc - PU.SIZE / 2.0) * dd / fz, (r - PU.SIZE / 2.0) * dd / fz, dd], 1) @ Rc.T + c
                far = np.linalg.norm(q - fr.ref, axis=1) > np.linalg.norm(q - fr.dis, axis=1)
                s[r[far], cc[far]] = -1
        return s
    PU.seg_std = seg_std

    Task = collections.namedtuple("Task", "name language problem_folder bddl_file")

    class RgcSuite:
        def __init__(self):
            self.tasks = [Task(t["key"], prompt if t["tid"] == TID else t["language"], os.path.dirname(os.path.join(ROOT, t["bddl"])),
                               os.path.basename(t["bddl"])) for t in meta["tasks"]]
            self.n_tasks = len(self.tasks)

        def get_task(self, i):
            return self.tasks[i]

        def get_task_init_states(self, i):
            raise RuntimeError("wa runs only from --fresh start states")
    _orig = benchmark.get_benchmark_dict

    def _bdict(*a, **k):
        d = dict(_orig(*a, **k))
        d[RGC.SUITE] = RgcSuite
        return d
    benchmark.get_benchmark_dict = _bdict
    LC.MAX_STEPS[RGC.SUITE] = int(TASK["max_steps"])
    STATES = np.load(os.path.join(RGC.ST, f"{RGC.SUITE}_{TID}.npy"))
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    RECS, SIDE = [], []
    SIDEF, TRF = OUTF.replace(".json", "_wa.json"), OUTF.replace(".json", "_track.npz")
    S = dict(det=None, E=None, est=None, first=True, q=[], was=False, off=None, toff=None)

    def truth(raw, slot):
        inner, sim = raw.env, raw.env.sim
        if slot in FC.MOVABLE:
            return np.array(sim.data.body_xpos[inner.obj_body_id[slot]], np.float64)
        return np.array(sim.data.site_xpos[sim.model.site_name2id(slot)], np.float64)

    class WAEnv:
        """The client's env (inner = raw / HideEnv / CoverEnv) with R's tracker, the last observation and gripper command, and the track."""

        def __init__(self, env):
            self._e = env
            self.n, self.obs, self.cmd, self.tr, self.front = 0, None, -1.0, None, None
            CUR["env"] = self

        def __getattr__(self, k):
            return getattr(self._e, k)

        def _names(self):
            return list(self._e.env.objects_dict.keys())

        def _snap(self):
            inner = self._e.env
            return np.array([np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]]) for n in self._names()], np.float32)

        def hidden_now(self):
            if hasattr(self._e, "covered") and not isinstance(self._e, RGC.HideEnv):
                try:
                    return bool(self._e.covered)
                except Exception:                                          # noqa: BLE001
                    return False
            return False

        def reset(self, *a, **k):
            o = self._e.reset(*a, **k)
            self.obs, self.cmd = o, -1.0
            return o

        def set_init_state(self, s):
            o = self._e.set_init_state(s)
            self.n, self.obs, self.cmd = 0, o, -1.0
            self.front = WFront(self._e, parsed if parsed is not None else {"relation": "-", "target": "-"}, front_mode) if front_mode else None
            CUR["front"] = self.front
            S.update(det=None, E=None, est=None, first=True, q=[])
            self.tr = dict(start=self._snap(), pos=[], ee=[], a6=[], hidden=[], t0=time.time())
            return o

        def step(self, a):
            r = self._e.step(a)
            self.n += 1
            obs = r[0]
            self.obs, self.cmd = obs, float(a[6])
            if self.tr is not None:
                self.tr["pos"].append(self._snap())
                self.tr["ee"].append(np.asarray(obs["robot0_eef_pos"], np.float32))
                self.tr["a6"].append(float(a[6]))
                self.tr["hidden"].append(self.hidden_now())
            if self.front is not None:
                obs = self.front.step(obs, self.n, WAIT)
            return (obs,) + tuple(r[1:])

        def close(self):
            if self.tr is not None and self.n > 0:
                sink(self)
            self.tr = None
            self._e.close()

    def sink(env):
        R = env.tr
        RECS.append(dict(start=R["start"], pos=np.stack(R["pos"]), ee=np.stack(R["ee"]), a6=np.array(R["a6"], np.float32),
                         hidden=np.array(R["hidden"], bool), success=np.array(bool(env._e.env._check_success()))))
        z = dict(n=np.array(len(RECS)), names=np.array(env._names()), wait=np.array(WAIT))
        for i, r in enumerate(RECS):
            z.update({f"e{i}_{k}": v for k, v in r.items()})
        np.savez_compressed(TRF, **z)
        i = len(SIDE)
        q = S["q"]
        rec = dict(trial=int(br[i]["trial"]) if i < len(br) else None, steps=env.n, secs=time.time() - R["t0"], queries=q,
                   front=None if env.front is None else dict(res=env.front.res, final_ref=None if env.front.ref is None else env.front.ref.tolist(),
                                                              final_dis=None if env.front.dis is None else env.front.dis.tolist(),
                                                              diag=env.front.diag))
        SIDE.append(rec)
        tmp = SIDEF + ".tmp%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], prompt=prompt, roles=list(roles), front=front_mode,
                           eps=SIDE), fh)
        os.replace(tmp, SIDEF)
        f = (rec["front"] or {}).get("res") or {}
        print("[wa] %s %s trial %s: steps %d front %s ok %s correct %s queries %d hidden_obj %d hidden_tgt %d" % (
            arm, TASK["key"], rec["trial"], env.n, front_mode, f.get("ok"), f.get("resolver_correct"), len(q),
            sum(1 for x in q if not x["vis_obj"]), sum(1 for x in q if not x["vis_tgt"])), flush=True)

    def channel(el):
        env = CUR["env"]
        raw = env._e
        if el.get("meta/episode_start") or S["det"] is None:
            S["det"], S["E"], S["first"] = ProprioHeld(X["tw"], X["lift"]), RV.entity_tables(raw), True
            S["est"] = PU.Estimator(Cal, oe)
        obs = env.obs
        ee = np.asarray(obs["robot0_eef_pos"], np.float64)
        R = FC._quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64))
        held = bool(S["det"].step(np.asarray(obs["robot0_gripper_qpos"], np.float64), env.cmd, ee[2]))
        E = S["E"]
        A, _ = PU.measure(raw, E, PU.AGENT, PU.MOV)
        F = PU.measure(raw, E, PU.AGENT, PU.fixture_names(E))[0] if S["first"] else None
        S["est"].step(A, lambda: PU.measure(raw, E, PU.WRIST, PU.MOV)[0], F, held, ee, R, env.cmd)
        S["first"] = False
        pos, src = S["est"].pos, S["est"].src
        gone = EMPTY if arm == "noMe" else STALE                            # noM: any held estimate is withheld; noMe: only an empty-mask one
        vis = {e: (str(src.get(e, "site")) not in gone) for e in (oe, te)}
        tobj, ttgt = truth(raw, oe), truth(raw, te)
        o_i, t_i = FC.ENT.index(oe), FC.ENT.index(te)
        if el.get("meta/episode_start"):                                    # SR 82: raa.py's raao / raac offset rule (fresh per episode)
            S["was"], S["off"], S["toff"] = False, None, None
        if held and not S["was"]:                                           # offset = the Estimator's object estimate minus the eef at the onset, then held fixed
            S["off"] = (pos[o_i] - ee).tolist()
            S["toff"] = (tobj - ee).tolist()
        if not held:
            S["off"] = S["toff"] = None
        S["was"] = held
        offx = {"meta/raa_off_ok": S["off"] is not None, "meta/raa_off": S["off"] or [0.0, 0.0, 0.0], "meta/raa_toff": S["toff"] or [0.0, 0.0, 0.0]}
        S["q"].append(dict(n=env.n, held=held, src_obj=str(src.get(oe, "site")), src_tgt=str(src.get(te, "site")), vis_obj=bool(vis[oe]),
                           vis_tgt=bool(vis[te]), hidden_truth=bool(env.hidden_now()),
                           err_obj_cm=float(100 * np.linalg.norm(pos[o_i] - tobj)), err_tgt_cm=float(100 * np.linalg.norm(pos[t_i] - ttgt))))
        fr = env.front
        dm = (fr.dm if fr is not None and fr.mode == "resolver" else {})
        z = np.zeros((PU.SIZE, PU.SIZE), np.uint8)
        return {**offx, "meta/raa_ok": True, "meta/raa_obj": pos[o_i].tolist(), "meta/raa_tgt": pos[t_i].tolist(),
                "meta/raa_tobj": tobj.tolist(), "meta/raa_ttgt": ttgt.tolist(),
                "meta/raa_src_obj": str(src.get(oe, "site")), "meta/raa_src_tgt": str(src.get(te, "site")), "meta/raa_roles": [oe, te, None],
                "meta/wa_arm": arm, "meta/wa_tid": TID, "meta/wa_nomem": arm in ("noM", "noMe"), "meta/wa_vis_obj": bool(vis[oe]),
                "meta/wa_vis_tgt": bool(vis[te]), "meta/wa_hidden": bool(env.hidden_now()),
                "meta/wa_dm_agent": np.asarray(dm.get("agentview", z), np.uint8), "meta/wa_dm_wrist": np.asarray(dm.get("robot0_eye_in_hand", z), np.uint8)}

    ORIG = LC.OffScreenRenderEnv
    LC.OffScreenRenderEnv = lambda *a, **k: WAEnv(RGC.wrap_inner(ORIG(*a, **k), TASK, meta, STATES))
    Base = LC._wcp.WebsocketClientPolicy

    class WAClient(Base):
        def infer(self, el):
            el = dict(el)
            try:
                el.update(channel(el))
                el.update(RV.client_payload(CUR["env"]._e))
            except Exception as e:                                          # noqa: BLE001  (a perception failure is logged, not hidden)
                logging.exception("[wa] channel failed")
                raise
            return super().infer(el)
    LC._wcp.WebsocketClientPolicy = WAClient
    if "--max-steps" not in sys.argv:
        sys.argv += ["--max-steps", str(TASK["max_steps"])]
    LC.main()


# ---------------------------------------------------------------- roles / selftest
def roles_cmd():
    lex, PF = lexicon()
    import fx_r as XR
    meta = json.load(open(os.path.join(OUT, "rg_comp", "rg_comp_tasks.json"), encoding="utf-8"))
    for t in meta["tasks"]:
        p = XR.parse(t["language"])
        lr = PF.role_parse(lex, t["language"])
        lp = PF.role_parse(lex, p["unique"]) if p else None
        print("[wa] tid %2d %-11s %-52r lexicon roles %s -> n2 prompt %r | R parse %s -> unique %r roles %s" % (
            t["tid"], t["key"], t["language"], lr, n2_prompt(lr) if lr else None, None if p is None else (p["relation"], p["target"]),
            None if p is None else p["unique"], lp), flush=True)


def selftest():
    assert set(CELLS) == set(ARMS) and CELLS["noM"] == (0, 1, 2, 3, 8)
    assert n2_prompt(("akita_black_bowl_1", "plate_1")) == "put the bowl on the plate"
    assert n2_prompt(("akita_black_bowl_1", "flat_stove_1_cook_region")) == "put the bowl on the stove"
    assert n2_prompt(("akita_black_bowl_1", "wine_rack_1_top_region")) is None
    print("[wa] selftest ok", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "roles":
        roles_cmd()
    elif len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    elif len(sys.argv) > 1 and sys.argv[1] == "serve":
        ap = argparse.ArgumentParser()
        ap.add_argument("what")
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--port", type=int, required=True)
        ap.add_argument("--seed", type=int, default=13)
        ap.add_argument("--config", default="")
        ap.add_argument("--seg-ckpt", dest="seg_ckpt", default="")
        ap.add_argument("--diag", default="")
        ap.add_argument("--chan-diag", dest="chan_diag", default="")
        ap.add_argument("--where_own", type=int, default=0, choices=(0, 1, 2, 3, 4), help="SR 84 / 85: 0 = plain PolicyBackbone (unchanged path); 1 = W1; 2 = W2; 3 = W3; 4 = W4")
        ap.add_argument("--where_log", default="", help="per-sample jsonl of the SR 84 branch")
        serve_cmd(ap.parse_args())
    else:
        raise SystemExit(__doc__)

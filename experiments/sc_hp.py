"""Arm HP: the hand-over (docs/HAND_OVER.md 2, 6.1) with no simulator privilege in the policy loop at test time.  LIBERO venv (py3.8).

pp_hand_client.py's harness (fc_n2_client -> libero_client, pp_atlas.Rec's oracle track as the read's instrument, the same VLA
server, seeds, start states and prompt), with three changes:
  place policy  U (sc_pu's pp_sc_U_s10, the checkpoint PU / PU-L ran) instead of P_min, run as sc_pu.episode's percep arm runs it:
                roles from U's lexicon (role_parse on the prompt), ProprioHeld gate, Canon / d0t / rot0 / ee0 stepped from the
                episode's start, noise seed seed_base + calls (calls counted from the hand-over, as pp_hand_client counts P_min's).
  poses         sc_pu.Estimator fed by sc_seg's learned U-Net masks (sc_pu.seg_std monkeypatched in this process, as sc_seg.py pu
                does) plus the depth render, with sc_pu's calibration.  Stepped every simulator step from the WAIT-th, so the
                estimator's memory (last estimates, rigid anchor, occlusion rule) is built while the VLA acts.
  hold signal   the hand-over fires at the first step where U's ProprioHeld (gripper width, last command, hand height) turns on,
                not at FB.Held's contact onset.  It does not pass back.
What the policy sees after the hand-over: proprioception (ee_pos, ee_rot from obs quat, finger qpos, last command), the estimator's
pos / yaw, Canon's zrest / cyaw on estimated poses, d0t on estimated poses.  b["off"] is zeros (PartsFlow never reads it) and
b["drawer"] is 0.0 (PartsFlow reads it only for slot 7, which no N1-N6 / K1-K6 role uses; asserted).
Simulator reads that remain, none of which reach the policy: the atlas Rec's contact Held / positions (the read's instrument),
LIBERO's success flag, libero_client's bind-track, and this file's logged pose errors / segmenter IoU diagnostic.

    client:   python experiments/sc_hp.py --port P --suite n2_goal --tasks TID --trials 1 --out results/flawed/sc_hp/pi05_s10_TID.json \
                  --fresh results/flawed/n2_states --branches B.json --videos 0 --bind track     (via .claude_tmp/pp/hp_roll.sh)
    summary:  python experiments/sc_hp.py summary DIR [--cmp H_DIR]   per-episode table of a run directory (smoke); --cmp: the
                  pre-hand-over end-effector rows against H's tracks on the same tids / states (a determinism check)
Output: the client's JSON (episodes in libero_client's format; each also carries "hp": hand-over step, trigger, pose errors,
segmenter diagnostics), "hp_complete": true once merged; <stem>_track.npz as pp_hand_client's, plus per-row est_obj, est_tgt, phold.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

U_SEED = 10
REPLAN = 5


def _cm(v):
    v = np.asarray(v, np.float64) * 100.0
    if not len(v):
        return None
    return dict(n=int(len(v)), med=round(float(np.median(v)), 3), p90=round(float(np.percentile(v, 90)), 3), max=round(float(v.max()), 3))


# ---------------------------------------------------------------- summary (no simulator, no GPU)
def summary(argv):
    import pp_atlas as PA
    d = argv[0]
    cmp_dir = argv[argv.index("--cmp") + 1] if "--cmp" in argv else None
    files = sorted(f for f in os.listdir(d) if f.endswith(".json") and "_s%d_" % U_SEED in f and not f.startswith("server"))
    allr = []
    for f in files:
        R = json.load(open(os.path.join(d, f), encoding="utf-8"))
        E = [e for v in R["tasks"].values() for e in v["episodes"]]
        tid = int(next(iter(R["tasks"])))
        tr = os.path.join(d, f.replace(".json", "_track.npz"))
        T = PA.load_track(tr) if os.path.exists(tr) else [None] * len(E)
        Hz = None
        if cmp_dir:
            model = f.split("_s%d_" % U_SEED)[0]
            hp_ = os.path.join(cmp_dir, "%s_H_%d_track.npz" % (model, tid))
            hj = os.path.join(cmp_dir, "%s_H_%d.json" % (model, tid))
            if os.path.exists(hp_) and os.path.exists(hj):
                HE = [e for v in json.load(open(hj, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
                Hz = (np.load(hp_), {int(e["trial"]): i for i, e in enumerate(HE)}, HE)
        print("== %s  complete=%s" % (f, R.get("hp_complete")), flush=True)
        for j, e in enumerate(E):
            h = e.get("hp", {})
            r = T[j]
            d_end = None if r is None else round(float(np.linalg.norm((r["obj"] - r["tgt"])[-1, :2])), 3)
            stage = None if r is None else PA.classify(r["held"], np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1), bool(e["success"]),
                                                       e.get("bound"), h.get("oracle_object"))[0]
            row = dict(tid=tid, trial=e["trial"], success=e["success"], stage=stage, steps=e["steps"], error=e["error"],
                       handed=h.get("handed"), fired=h.get("trigger_fired"), oracle_first_held=h.get("oracle_first_held"),
                       oracle_held_at_handover=h.get("oracle_held_at_handover"), d_end=d_end, u_calls=h.get("u_calls"),
                       err=h.get("pose_err_cm"), at_handover=h.get("at_handover"), src=h.get("obj_sources"),
                       seg_ms=h.get("seg", {}).get("ms_per_pred"))
            hd = int(h.get("handed", -1))
            if r is not None and hd >= 0:                                   # the instrument's view of the hand-over row
                row["at_handover_sim"] = dict(obj_lift_cm=round(100 * float(r["obj"][hd, 2] - r["obj"][0, 2]), 2),
                                              ee_obj_cm=round(100 * float(np.linalg.norm(r["ee"][hd] - r["obj"][hd])), 2),
                                              a6_last5=[round(float(x), 2) for x in r["a6"][max(0, hd - 4): hd + 1]],
                                              obj_lift_10_later_cm=round(100 * float(r["obj"][min(hd + 10, len(r["obj"]) - 1), 2] - r["obj"][0, 2]), 2))
            if Hz is not None and r is not None and int(e["trial"]) in Hz[1]:
                k = Hz[1][int(e["trial"])]
                zH = Hz[0]
                hH = int(zH["e%d_handed" % k])
                eeH = zH["e%d_ee" % k]
                cand = [x for x in (hH, int(h.get("handed", -1))) if x >= 0]  # rows 0..first hand-over: VLA actions in both arms
                n = min(min(cand) + 1 if cand else len(eeH), len(eeH), len(r["ee"]))
                row["vs_H"] = dict(H_handed=hH, H_success=bool(Hz[2][k]["success"]), rows_compared=int(n),
                                   max_ee_diff_m=float(np.abs(eeH[:n] - r["ee"][:n]).max()) if n else None)
            allr.append(row)
            print(json.dumps(row), flush=True)
    if allr:
        s = np.array([r["success"] for r in allr], float)
        fired = np.array([bool(r["fired"]) for r in allr], float)
        print("TOTAL episodes %d success %.3f fired %.3f" % (len(allr), s.mean(), fired.mean()), flush=True)


# ---------------------------------------------------------------- the client
def client():
    import torch
    import fc_n2_client as FN
    import pp_atlas as PA
    import fa1_common as FC
    import fa1c_common as FK
    import sc_pu as PU
    import sc_rv as RV
    import sc_seg as SEG
    from lifelong_ood.policy import factored_flow as FF
    from lifelong_ood.policy.parts_flow import ProprioHeld, role_parse
    from pp_p_roll import load_any

    LC = FN.LC
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TRACK = OUTF[:-len(".json")] + "_track.npz"
    TASK = FN.N2["tasks"][TID]
    assert TASK["tid"] == TID
    torch.set_num_threads(1)
    M, _ = load_any(PU.ckpt_path(U_SEED))
    PG = getattr(M, "pgate", None)
    assert PG and not PG.get("motion"), ("U needs the proprioceptive gate", PG)
    LEX = getattr(M, "role_lex", None)
    assert LEX, "U reads its roles through its lexicon"
    parsed = role_parse(LEX, TASK["language"])
    assert parsed is not None, ("no induced roles", TASK["language"])
    OE_U, TE_U = parsed                                                     # the policy's roles: from the prompt only
    OE, TE = TASK["object"], TASK["target"]                                 # the atlas instrument's roles (n2_tasks)
    OI, TI = FC.ENT.index(OE_U), FC.ENT.index(TE_U)
    assert FF.DRAWER not in (OI, TI), "drawer slot: b['drawer'] would be read"
    C = PU.load_calib()
    SEGM = SEG.load(os.environ.get("SC_SEG_CKPT") or None, "cuda")
    ORIG, RECS, HPS, CUR, CACHE = LC.OffScreenRenderEnv, [], [], {}, {}
    SEGD = dict(n=0, secs=0.0, acc=None)

    def seg_std(env, E, cam):
        """sc_seg.py pu's seg_std: pred() on an RGB render of the camera, in sc_pu.seg_std's convention.  The simulator's map is
        read only for the logged IoU diagnostic, never returned."""
        sim = env.env.sim
        key = (id(sim), float(sim.data.time), cam)
        if key in CACHE:
            return CACHE[key]
        t0 = time.time()
        img = RV.flip(sim.render(camera_name=cam, width=PU.SIZE, height=PU.SIZE, depth=False))
        p = SEGM.ent(img, cam, E["names"])
        SEGD["secs"] += time.time() - t0
        SEGD["n"] += 1
        g = RV.seg_px(env, E, cam)                                          # diagnostic only
        for n in PU.MOV:
            if n in E["names"]:
                i = E["names"].index(n)
                A = SEGD["acc"][cam][n]
                A[0] += int(((g == i) & (p == i)).sum())
                A[1] += int(((g == i) | (p == i)).sum())
        out = np.ascontiguousarray(np.asarray(p)[:, ::-1])
        if len(CACHE) > 8:
            CACHE.clear()
        CACHE[key] = out
        return out

    PU.seg_std = seg_std                                                    # this process only: PU.measure reads the module global

    def proprio(obs):
        return dict(ee_pos=np.asarray(obs["robot0_eef_pos"], np.float64), ee_rot=FC._quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64)),
                    grip=np.asarray(obs["robot0_gripper_qpos"], np.float64))

    class HPRec(PA.Rec):
        def __init__(self, *a):
            super().__init__(*a)
            self.on, self.handed, self.calls, self.plan = False, -1, 0, []
            self.obs, self.prev = None, -1.0
            self.p_on, self.zr, self.cy, self.rot0, self.d0t, self.ee0 = False, 0.0, 0.0, None, None, None
            self.est, self.E, self.det, self.cnp, self.cno = None, None, None, None, None
            self.erows = dict(obj=[], tgt=[], phold=[])
            self.dg = dict(obj_pre_vla=[], obj_pre_u=[], obj_held=[], tgt=[], yaw_pre=[], src={}, wrist=0, n=0, at_handover=None,
                           onsets=[], sites0=None)
            CACHE.clear()
            SEGD["n"], SEGD["secs"] = 0, 0.0
            SEGD["acc"] = {c: {n: [0, 0] for n in PU.MOV} for c in RV.CAMS}
            CUR["env"] = self

        def _tgt_est(self):
            return self.est.pos[TI].copy(), float(self.est.pos[PU.TABLE, 2])

        def _perceive(self, hp, pr, sim_pos, sim_yaw):
            env, E = self._e, self.E
            A, _ = PU.measure(env, E, PU.AGENT, PU.MOV)
            F = PU.measure(env, E, PU.AGENT, PU.fixture_names(E))[0] if self.est.first else None

            def wfn():
                self.dg["wrist"] += 1
                return PU.measure(env, E, PU.WRIST, PU.MOV)[0]
            self.est.step(A, wfn, F, hp, pr["ee_pos"], pr["ee_rot"], self.prev)
            # logged only: the estimate against the simulator
            dg = self.dg
            dg["n"] += 1
            s = self.est.src[OE_U]
            dg["src"][s] = dg["src"].get(s, 0) + 1
            e = float(np.linalg.norm(self.est.pos[OI] - sim_pos[OI]))
            dg["obj_held" if hp else ("obj_pre_u" if self.on else "obj_pre_vla")].append(e)
            dg["tgt"].append(float(np.linalg.norm(self.est.pos[TI] - sim_pos[TI])))
            if not hp:
                dg["yaw_pre"].append(abs(float(PU.wrap(self.est.yaw[OI] - sim_yaw[OI]))))
            if dg["sites0"] is None:
                dg["sites0"] = {FC.ENT[i]: round(float(np.linalg.norm(self.est.pos[i] - sim_pos[i])) * 100, 3) for i in range(4, 9)}

        def _track(self):
            """sc_pu.episode's policy_gate, onset_rot, onset_d (percep arm), stepped on every step from the WAIT-th."""
            first = self.est is None
            pr = proprio(self.obs)
            if first:
                self.E = RV.entity_tables(self._e)
                self.est = PU.Estimator(C, OE_U)
                self.det, self.cnp, self.cno = ProprioHeld(PG["tw"], PG["lift"]), FK.Canon(), FK.Canon()
                self.ee0 = pr["ee_pos"].copy()
            was = self.p_on
            sim = FC.scene(self._e, self.obs, self._dj)                    # logged only (errors); never reaches the policy
            hp = bool(self.det.step(pr["grip"], self.prev, pr["ee_pos"][2]))
            self._perceive(hp, pr, sim["pos"], sim["yaw"])
            zp, cp = self.cnp.step(hp, self.est.pos[OI].copy(), self._tgt_est)
            zo, co = self.cno.step(hp, sim["pos"][OI], lambda: (sim["pos"][TI], float(sim["pos"][PU.TABLE, 2])))   # logged only
            self.p_on, self.zr, self.cy = hp, zp, cp
            self.rot0 = pr["ee_rot"].copy() if hp and not was else (self.rot0 if hp else None)
            if hp and not was:
                self.d0t = float(np.linalg.norm(self.est.pos[OI, :2] - self.est.pos[TI, :2]))
                d0s = float(np.linalg.norm(sim["pos"][OI, :2] - sim["pos"][TI, :2]))
                self.dg["onsets"].append(dict(row=len(self.rows["held"]) - 1, by="vla" if not self.on else "u",
                                              obj_cm=round(100 * float(np.linalg.norm(self.est.pos[OI] - sim["pos"][OI])), 3),
                                              tgt_cm=round(100 * float(np.linalg.norm(self.est.pos[TI] - sim["pos"][TI])), 3),
                                              d0t_err_cm=round(100 * abs(self.d0t - d0s), 3), zrest_err_cm=round(100 * abs(zp - zo), 3),
                                              cyaw_err_deg=round(float(np.degrees(abs(PU.wrap(cp - co)))), 3),
                                              oracle_held=bool(self.rows["held"][-1])))
            elif not hp:
                self.d0t = None
            self.erows["obj"].append(self.est.pos[OI].copy())
            self.erows["tgt"].append(self.est.pos[TI].copy())
            self.erows["phold"].append(hp)

        def _part_action(self):
            if not self.plan:
                pr = proprio(self.obs)
                sc = dict(pr, pos=self.est.pos.copy(), yaw=self.est.yaw.copy(), drawer=0.0)
                b = FF.batch_from_scene(sc, OI, TI, self.prev, held=float(self.p_on), off=np.zeros(3), zrest=self.zr, cyaw=self.cy, rot0=self.rot0)
                if float(getattr(M, "hscale", 0.0)) > 0:
                    b["ee0"] = torch.tensor(np.asarray(self.ee0), dtype=torch.float32)[None]
                if bool(getattr(M, "tscale", False)) and self.d0t is not None:
                    b["d0t"] = torch.tensor([self.d0t], dtype=torch.float32)
                g = torch.Generator().manual_seed(int(CUR["seed_base"]) + self.calls)
                ch = M.sample(b, generator=g)[0].numpy()
                self.plan = [np.clip(x, -1.0, 1.0) for x in ch[:REPLAN]]
                self.calls += 1
            return self.plan.pop(0).tolist()

        def step(self, a):
            if self.on:
                a = self._part_action()
            r = super().step(a)
            if self.n < PA.WAIT:
                return r
            self.obs = r[0]
            self.prev = float(a[6]) if self.est is not None else -1.0
            self._track()
            if not self.on and self.p_on:
                self.on, self.handed = True, len(self.rows["held"]) - 1
                self.dg["at_handover"] = self.dg["onsets"][-1] if self.dg["onsets"] else None
            return r

        def close(self):
            if self.n > 0:
                d = self.data()
                d["handed"] = np.array(self.handed)
                for k in ("obj", "tgt"):
                    d["est_" + k] = np.array(self.erows[k], np.float32).reshape(-1, 3)
                d["phold"] = np.array(self.erows["phold"], bool)
                RECS.append(d)
                PA.save(TRACK, RECS)
                held = np.asarray(self.rows["held"], bool)
                dg = self.dg
                HPS.append(dict(
                    handed=int(self.handed), trigger_fired=bool(self.handed >= 0), trigger="ProprioHeld", place_policy="U pp_sc_U_s%d" % U_SEED,
                    roles_u=[OE_U, TE_U], oracle_object=OE, roles_match=bool((OE_U, TE_U) == (OE, TE)),
                    oracle_first_held=int(np.argmax(held)) if held.any() else None,
                    oracle_held_at_handover=bool(held[self.handed]) if self.handed >= 0 else None,
                    u_calls=int(self.calls), onsets=dg["onsets"], at_handover=dg["at_handover"],
                    pose_err_cm=dict(obj_pre_vla=_cm(dg["obj_pre_vla"]), obj_pre_u=_cm(dg["obj_pre_u"]), obj_held=_cm(dg["obj_held"]),
                                     tgt=_cm(dg["tgt"]), sites0=dg["sites0"]),
                    yaw_pre_deg=None if not dg["yaw_pre"] else dict(med=float(np.degrees(np.median(dg["yaw_pre"]))),
                                                                     max=float(np.degrees(np.max(dg["yaw_pre"])))),
                    obj_sources=dg["src"], wrist_renders=dg["wrist"], perceived_steps=dg["n"],
                    sites_mode=None if self.est is None else self.est.sites, hold_mode=None if self.est is None else self.est.hold,
                    seg=dict(preds=SEGD["n"], ms_per_pred=round(1000 * SEGD["secs"] / max(1, SEGD["n"]), 2),
                             iou={c: {n: (None if A[1] == 0 else round(A[0] / float(A[1]), 4)) for n, A in v.items()} for c, v in SEGD["acc"].items()})))
            self._e.close()

    class HPClient(LC._wcp.WebsocketClientPolicy):
        def infer(self, el):
            CUR["seed_base"] = el["meta/seed_base"]
            if CUR["env"].on:                                               # the env ignores this plan after the hand-over
                return {"actions": np.zeros((REPLAN, 7))}
            return super().infer(el)

    LC.OffScreenRenderEnv = lambda *a, **k: HPRec(ORIG(*a, **k), OE, TE)
    LC._wcp.WebsocketClientPolicy = HPClient
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    LC.main()
    R = json.load(open(OUTF, encoding="utf-8"))
    E = [e for v in R["tasks"].values() for e in v["episodes"]]
    assert len(E) == len(HPS) == len(RECS), (len(E), len(HPS), len(RECS))
    for e, h in zip(E, HPS):
        e["hp"] = h
    R["hp_complete"] = True
    R["hp_meta"] = dict(u_ckpt=PU.ckpt_path(U_SEED), calib=PU.CALIB, seg_ckpt=SEGM.path, pgate=PG, host=os.uname().nodename)
    tmp = OUTF + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    os.replace(tmp, OUTF)
    print("[hp] -> %s: %d of %d succeeded; trigger fired in %d" % (OUTF, sum(e["success"] for e in E), len(E), sum(h["trigger_fired"] for h in HPS)), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary(sys.argv[2:])
    else:
        client()

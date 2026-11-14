"""N46-PG: the place phase alone.  Does a frozen VLA place the wine bottle on the plate (N4) or in the bowl (N6) when it starts with the
bottle already grasped and held just above the target?  (docs/SCAFFOLD_RESULTS.md 51; m3_prereg tag n46.)

Why.  SR 46-49: N4 and N6 stay near the intact rate on frozen pi0.5 and near 0 on frozen GR00T under every front-end arm.  The phase
breakdown of the existing rollouts (experiments/n46_phase.py, results/flawed/n46_phase.json) puts nearly every N4 / N6 failure after
the grasp, with the bottle carried at K4's cabinet height (~.30 m lift, against ~.15 m in successes) and released high or never.
P_min (oracle state) and the hand-over arm H (the VLA's own grasp, P_min after it; HAND_OVER 6.1) place it at .80-1.00, so the task
is feasible.  Left open: is the VLA's place skill for a bottle on a low movable target present but not selected (routing), or absent
(a motor data gap: no LIBERO-Goal demo puts the bottle on anything below the cabinet top)?  This probe removes the grasp and carry.

Episode (the atlas's harness: fc_n2_client -> libero_client, pp_atlas.Rec track, main states 4-23, n2_branches_main.json, client seed
7, --bind track; --max-steps 450 so the VLA keeps >= 200 steps after the prefix):
  prefix (scripted, the v2 servo's phases from pp_n4d.plan_episode(servo=True), unchanged waypoints, kp = pp_l1_floor.kp_for(task),
  rng = default_rng(seed_base + 10) as the atlas's servo; reach timeouts 120 steps instead of 40, as HAND_OVER 7.4's Read G):
     0 above the grasp point, 1 to it, 2 close (12 steps), 3 lift to carry height, 4 over the target at carry height,
     5 down to the release point + HZ = 0.06 m (servo's phase-5 waypoint raised 6 cm), gripper closed.
  hand-over at the end of phase 5 (err < .008 or 120 steps), or at 300 prefix steps (prefix timeout).  Then the arm acts:
     vla    the frozen VLA (pp_serve_purify --mode intact --comp= --seed 13), task instruction unchanged, from its next query.
            While the prefix runs the client returns hold actions and the server is not queried.
     servo  scripted reference: the servo's phase 5 (to the release point), 6 (open, 12 steps), 7 (up 0.10 m).
     open   scripted null: open the gripper where it is (12 steps), then up 0.10 m.  What release-on-arrival alone buys.
The simulator is read by the prefix and the scripted arms (they are planners); never by the VLA arm after the hand-over.

Outputs: <out>.json (libero_client), <out>_track.npz (pp_atlas.Rec + 'handed' row), <out>_pg.json (per episode: hand-over row, held at
hand-over (oracle), prefix steps, timeout, success before hand-over, bottle height / horizontal offset / tilt at hand-over).

Registered mapping (read; per model, arm vla, pooled N4 + N6, main states; the unit is an episode HELD at hand-over and not already
successful; cluster bootstrap over states, B = 2000, numpy default_rng(0), CI 2.5 / 97.5):
    PLACE PRESENT iff the CI lower bound of placement success >= .40;  PLACE ABSENT iff the CI upper bound < .30;  else MIXED.
    Reported with no verdict: per task (N4, N6, N5 control) success for vla / servo / open; held-at-hand-over share (LOW POWER flag if
    < .60 on N4 or N6); the post-hand-over phase (pp_atlas.classify on the post-hand-over rows), lift after hand-over, final tilt,
    nearest entity at the end; vla - open and vla - servo on the same held cells.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

ARMS = ("vla", "servo", "open")
HZ = 0.06
REACH_T, PREFIX_CAP, HOLD_T, REPLAN = 120, 300, 12, 5
HOLD_ACT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
OUT = os.path.join(ROOT, "results", "flawed")
DIR = os.path.join(OUT, "n46_pg")
READ = os.path.join(OUT, "n46_pg_read.json")
TASKS = ("N4", "N6", "N5")


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
    import fa1_common as FC
    from fa1g_plan import ctrl
    from pp_l1_floor import kp_for
    LC = FN.LC
    TID = int(sys.argv[sys.argv.index("--tasks") + 1])
    OUTF = sys.argv[sys.argv.index("--out") + 1]
    TASK = FN.N2["tasks"][TID]
    assert TASK["tid"] == TID and TASK["layout"] == "familiar", TASK
    br = json.load(open(sys.argv[sys.argv.index("--branches") + 1], encoding="utf-8"))
    assert all(int(b["at"]) == 0 for b in br), "branches with a recorded prefix (at > 0) would replay actions after the hand-over"
    OE, TE = TASK["object"], TASK["target"]
    OI, TI = FC.ENT.index(OE), FC.ENT.index(TE)
    KP = kp_for(TASK["compound"])
    ORIG, RECS, SIDE, CUR = LC.OffScreenRenderEnv, [], [], {}
    SIDEF = OUTF.replace(".json", "_pg.json")

    class PGRec(PA.Rec):
        def __init__(self, env):
            super().__init__(env, OE, TE)
            self.reset_pg()
            CUR["env"] = self

        def reset_pg(self):
            self.started, self.prefix, self.post = False, True, None
            self.phase, self.in_phase, self.hold, self.psteps, self.err = 0, 0, 0, 0, 1.0
            self.obs, self.dj, self.rng, self.zref = None, None, None, None
            self.handed, self.info, self.rel_up = -1, {}, None

        def set_init_state(self, s):
            self.reset_pg()
            obs = self._e.set_init_state(s)
            self.obs = obs
            return obs

        def _setup(self):
            self.dj = FC.drawer_joint(self._e)
            self.rng = np.random.default_rng(int(CUR["seed_base"]) + 10)
            jit = self.rng.normal(0, 0.01, (2, 3))
            self.cz = KP["carry_z_above_ref"] + self.rng.normal(0, 0.02)
            self.goff, self.roff = np.array(KP["grasp_off"]) + jit[0], np.array(KP["release_off"]) + jit[1]
            self.RG, self.RR = np.array(KP["grasp_rot"]), np.array(KP["release_rot"])
            self.zref = float(FC.scene(self._e, self.obs, self.dj)["pos"][8, 2])

        def _scripted(self):
            """-> action.  The prefix's phases 0-5 and, after the hand-over, the scripted arms' own phases."""
            sc = FC.scene(self._e, self.obs, self.dj)
            po, pt = sc["pos"][OI], sc["pos"][TI]
            p = self.phase
            if p == 0:
                wp, Rt, g = po + self.goff + [0, 0, 0.10], self.RG, -1.0
            elif p == 1:
                wp, Rt, g = po + self.goff, self.RG, -1.0
            elif p == 2:
                wp, Rt, g = po + self.goff, self.RG, 1.0
            elif p == 3:
                wp, Rt, g = np.array([sc["ee_pos"][0], sc["ee_pos"][1], self.zref + self.cz]), self.RG, 1.0
            elif p == 4:
                wp, Rt, g = np.array([pt[0] + self.roff[0], pt[1] + self.roff[1], self.zref + self.cz]), self.RR, 1.0
            elif p == 5:
                wp, Rt, g = pt + self.roff + [0, 0, HZ], self.RR, 1.0
            elif p == 15:                                                   # servo arm: down to the release point
                wp, Rt, g = pt + self.roff, self.RR, 1.0
            elif p == 16:                                                   # servo arm: open there
                wp, Rt, g = pt + self.roff, self.RR, -1.0
            elif p == 26:                                                   # open arm: open where the hand is
                wp, Rt, g = self.hand_pos, self.RR, -1.0
            else:                                                           # 17 / 27: up 0.10 m, open
                wp, Rt, g = self.rel_up, self.RR, -1.0
            a, self.err = ctrl(sc, wp, Rt, g, self.rng)
            return a.tolist()

        def _advance(self):
            p = self.phase
            self.in_phase += 1
            if p in (2, 16, 26):
                self.hold += 1
                if self.hold >= HOLD_T:
                    self.phase, self.hold, self.in_phase = p + 1, 0, 0
                    if self.phase in (17, 27):
                        self.rel_up = np.asarray(self.obs["robot0_eef_pos"], np.float64) + [0, 0, 0.10]
                return
            if p in (0, 3, 4) and self.err < 0.015 or p == 1 and self.err < 0.008 or p in (0, 1) and self.in_phase >= REACH_T \
                    or p in (3, 4) and self.in_phase >= 40:
                self.phase, self.in_phase = p + 1, 0
            elif p == 5 and (self.err < 0.008 or self.in_phase >= REACH_T):
                self.hand_over(timeout=False)
            elif p == 15 and (self.err < 0.008 or self.in_phase >= 40):
                self.phase, self.in_phase = 16, 0

        def hand_over(self, timeout):
            self.prefix, self.handed = False, len(self.rows["held"]) - 1
            sc = FC.scene(self._e, self.obs, self.dj)
            po, pt = sc["pos"][OI], sc["pos"][TI]
            q = self._e.env.sim.data.body_xquat[self._e.env.obj_body_id[OE]]
            self.info = dict(handed_row=self.handed, prefix_steps=self.psteps, prefix_timeout=bool(timeout), end_phase=self.phase,
                             held_at_handover=bool(self.rows["held"][-1]) if self.rows["held"] else False,
                             d_xy=float(np.linalg.norm((po - pt)[:2])), dz_obj_tgt=float(po[2] - pt[2]), tilt=PA.tilt_deg(q),
                             ee=np.asarray(self.obs["robot0_eef_pos"], np.float64).tolist())
            self.hand_pos = np.asarray(self.obs["robot0_eef_pos"], np.float64).copy()
            self.phase, self.in_phase, self.hold = {"vla": 99, "servo": 15, "open": 26}[arm], 0, 0

        def step(self, a):
            scripted = self.started and (self.prefix or arm != "vla")
            if scripted:
                if self.dj is None:
                    self._setup()
                a = self._scripted()
            r = super().step(a)
            self.obs = r[0]
            if scripted:
                if self.prefix:
                    self.psteps += 1
                    self._advance()
                    if self.prefix and self.psteps >= PREFIX_CAP:
                        self.hand_over(timeout=True)
                elif arm != "vla":
                    self._advance()
            return r

        def close(self):
            if self.n > 0:
                d = self.data()
                d["handed"] = np.array(self.handed)
                RECS.append(d)
                PA.save(OUTF.replace(".json", "_track.npz"), RECS)
                i = len(SIDE)
                rec = dict(self.info, trial=int(br[i]["trial"]) if i < len(br) else None, steps=self.n, arm=arm, tid=TID, key=TASK["key"],
                           handed=self.handed)
                SIDE.append(rec)
                tmp = SIDEF + ".tmp%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(dict(arm=arm, tid=TID, key=TASK["key"], language=TASK["language"], hz=HZ, kp=KP, eps=SIDE), fh, indent=1)
                os.replace(tmp, SIDEF)
                print("[pg] %s %s trial %s: handed row %s held %s prefix %s steps timeout %s d_xy %s dz %s" % (
                    arm, TASK["key"], rec["trial"], self.handed, rec.get("held_at_handover"), rec.get("prefix_steps"),
                    rec.get("prefix_timeout"), _r(rec.get("d_xy")), _r(rec.get("dz_obj_tgt"))), flush=True)
            self._e.close()

    Base = LC._wcp.WebsocketClientPolicy

    class PGClient(Base):
        def infer(self, el):
            env = CUR["env"]
            CUR["seed_base"] = el["meta/seed_base"]
            env.started = True
            if env.prefix or arm != "vla":                                  # scripted steps: the env overrides these actions
                return {"actions": np.tile(np.asarray(HOLD_ACT, np.float32), (REPLAN, 1))}
            return super().infer(el)

    LC._wcp.WebsocketClientPolicy = PGClient
    LC.OffScreenRenderEnv = lambda *a, **k: PGRec(ORIG(*a, **k))
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    print("[pg] arm %s task %s %r object %s target %s hz %.3f kp %s" % (arm, TASK["key"], TASK["language"], OE, TE, HZ, json.dumps(KP)), flush=True)
    LC.main()


def _r(x):
    return None if x is None else round(float(x), 3)


# ---------------------------------------------------------------- read (CPU)
def post_phase(PA, r, e, t, handed):
    """pp_atlas.classify on the rows after the hand-over (the place phase only)."""
    held = np.asarray(r["held"], bool)[handed:]
    d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)[handed:]
    return PA.classify(held, d, bool(e["success"]), e.get("bound"), t["object"])


def read(out, d):
    import pp_atlas as PA
    from lifelong_ood import ledger
    out = os.path.abspath(out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "n46"]
    assert len(rows) == 1, "expected one n46 prereg row, found %d" % len(rows)
    n2 = PA.tasks()
    ks = list(n2["states"]["main"])
    T = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar"}
    ENT = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side",
           "wine_rack_1_top_region", "wooden_cabinet_1_top_region", "main_table_stove_front_region"]
    E = {}
    for stem_arm, arm in (("pi05_vla", "pi05"), ("groot_vla", "groot"), ("pi05_servo", "servo"), ("pi05_open", "open")):
        for c in TASKS:
            f = os.path.join(d, "%s_%d" % (stem_arm, T[c]["tid"]))
            if not (os.path.exists(f + ".json") and os.path.exists(f + "_track.npz") and os.path.exists(f + "_pg.json")):
                continue
            R = json.load(open(f + ".json", encoding="utf-8"))
            Ep = [e for v in R["tasks"].values() for e in v["episodes"]]
            Tr = PA.load_track(f + "_track.npz")
            z = np.load(f + "_track.npz")
            S = json.load(open(f + "_pg.json", encoding="utf-8"))["eps"]
            assert len(Ep) == len(Tr) == len(S) == len(ks), (f, len(Ep), len(Tr), len(S))
            L = []
            for i, (k, e, r, s) in enumerate(zip(ks, Ep, Tr, S)):
                assert int(e["trial"]) == k == int(s["trial"]) and e.get("error") is None, (f, k, e.get("error"))
                hd = int(z["e%d_handed" % i])
                pre_success = bool(e["success"]) and hd < 0
                st = post_phase(PA, r, e, T[c], max(hd, 0)) if hd >= 0 else ("prefix_only", "", None)
                O = r["obj"]
                fo = r["final"][ENT.index(T[c]["object"])]
                near = min(((n, float(np.linalg.norm((fo - r["final"][j])[:2]))) for j, n in enumerate(ENT) if n != T[c]["object"]), key=lambda x: x[1])
                L.append(dict(trial=k, success=bool(e["success"]), handed=hd, held=bool(s.get("held_at_handover")), timeout=s.get("prefix_timeout"),
                              pre_success=pre_success, stage=st[0], sub=st[1], d_xy=s.get("d_xy"), dz=s.get("dz_obj_tgt"), tilt_hand=s.get("tilt"),
                              lift_after=float(O[hd:, 2].max() - O[hd, 2]) if hd >= 0 and len(O) > hd else None,
                              final_tilt=PA.tilt_deg(r["quat"][-1]), final_nearest=near[0], final_nearest_d=near[1],
                              steps_after=int(len(r["held"]) - hd) if hd >= 0 else None))
            E["%s|%s" % (arm, c)] = L
    usable = {k: [x for x in v if x["held"] and not x["pre_success"] and x["handed"] >= 0] for k, v in E.items()}

    def cluster_boot(arm, comps):
        cells = {k: [x for x in usable.get("%s|%s" % (arm, c), []) if x["trial"] == k] for c in comps for k in ks}
        by_state = {k: [x for c in comps for x in usable.get("%s|%s" % (arm, c), []) if x["trial"] == k] for k in ks}
        allx = [x for v in by_state.values() for x in v]
        if not allx:
            return None
        pt = float(np.mean([x["success"] for x in allx]))
        rng = np.random.default_rng(0)
        v = []
        for _ in range(2000):
            dr = rng.choice(len(ks), len(ks))
            xs = [x for j in dr for x in by_state[ks[j]]]
            if xs:
                v.append(float(np.mean([x["success"] for x in xs])))
        del cells
        return [pt, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(allx)]

    def verdict(b):
        if b is None:
            return None
        if b[1] >= 0.40:
            return "PLACE PRESENT"
        if b[2] < 0.30:
            return "PLACE ABSENT"
        return "MIXED"

    res = dict(dir=d, states=ks, hz=HZ, mapping=MAPPING, per_task={}, primary={}, verdicts={})
    for k, v in E.items():
        u = usable[k]
        ph = {}
        for x in u:
            key = x["stage"] + ("/" + x["sub"] if x["sub"] else "")
            ph[key] = ph.get(key, 0) + 1
        nr = {}
        for x in u:
            if not x["success"]:
                nr[x["final_nearest"]] = nr.get(x["final_nearest"], 0) + 1
        res["per_task"][k] = dict(n=len(v), held_share=float(np.mean([x["held"] for x in v])), timeouts=int(sum(bool(x["timeout"]) for x in v)),
                                  pre_success=int(sum(x["pre_success"] for x in v)), n_usable=len(u),
                                  success_usable=float(np.mean([x["success"] for x in u])) if u else None,
                                  success_all=float(np.mean([x["success"] for x in v])), post_phase_usable=ph,
                                  lift_after_med=float(np.median([x["lift_after"] for x in u if x["lift_after"] is not None])) if u else None,
                                  final_tilt_gt45_fail=float(np.mean([x["final_tilt"] > 45 for x in u if not x["success"]])) if any(not x["success"] for x in u) else None,
                                  final_nearest_fail=nr, d_xy_med=float(np.median([x["d_xy"] for x in v if x["d_xy"] is not None])) if v else None,
                                  dz_med=float(np.median([x["dz"] for x in v if x["dz"] is not None])) if v else None)
    for arm in ("pi05", "groot", "servo", "open"):
        b = cluster_boot(arm, ("N4", "N6"))
        res["primary"][arm] = b
        if arm in ("pi05", "groot"):
            res["verdicts"][arm] = verdict(b)
    low = {c: res["per_task"].get("pi05|%s" % c, {}).get("held_share") for c in ("N4", "N6")}
    res["low_power"] = {arm: any((res["per_task"].get("%s|%s" % (arm, c), {}).get("held_share") or 0) < 0.60 for c in ("N4", "N6")) for arm in ("pi05", "groot")}
    # paired, same held cells: vla - open and vla - servo (pooled N4+N6), point only and counts
    pairs = {}
    for m in ("pi05", "groot"):
        for ref in ("open", "servo"):
            a_ = {(c, x["trial"]): x["success"] for c in ("N4", "N6") for x in usable.get("%s|%s" % (m, c), [])}
            b_ = {(c, x["trial"]): x["success"] for c in ("N4", "N6") for x in usable.get("%s|%s" % (ref, c), [])}
            kk = sorted(set(a_) & set(b_))
            pairs["%s_minus_%s" % (m, ref)] = None if not kk else [float(np.mean([a_[q] - b_[q] for q in kk])), len(kk)]
    res["paired"] = pairs
    del low
    print("== N46-PG verdicts %s | primary %s | low power %s" % (res["verdicts"], json.dumps(res["primary"]), res["low_power"]), flush=True)
    for k, v in res["per_task"].items():
        print("   %-12s %s" % (k, json.dumps(v)), flush=True)
    print("   paired %s" % json.dumps(pairs), flush=True)
    for k, v in E.items():
        print("   eps %-12s %s" % (k, " ".join("%d:%s%s:%s/%s" % (x["trial"], "S" if x["success"] else "f", "" if x["held"] else "(nh)", x["stage"][:6], x["sub"][:5]) for x in v)), flush=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(res, episodes=E), fh, indent=1)
    os.replace(tmp, out)
    print("wrote", out, flush=True)


MAPPING = ("per model, arm vla, pooled N4 + N6, main states 4-23; unit = episode held (oracle) at the hand-over and not successful before "
           "it; placement success; cluster bootstrap over states B=2000 rng 0, CI 2.5/97.5. PLACE PRESENT iff CI lo >= .40; PLACE ABSENT iff "
           "CI hi < .30; else MIXED. Descriptive: per task N4/N6/N5 for vla/servo/open, held share (LOW POWER if < .60 on N4 or N6), "
           "post-hand-over stage, lift after hand-over, final tilt, final nearest entity, vla - open and vla - servo on shared held cells.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "read":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("mode")
        ap.add_argument("--out", default=READ)
        ap.add_argument("--dir", default=DIR)
        A = ap.parse_args()
        read(A.out, os.path.abspath(A.dir))
    else:
        print(__doc__)

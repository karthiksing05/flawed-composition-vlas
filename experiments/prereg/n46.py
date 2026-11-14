"""PRE-REGISTRATION ROW of N46-PG (SCAFFOLD_RESULTS 51): the place phase alone.  Frozen pi0.5 and GR00T start with the wine bottle
already grasped (scripted servo prefix) and held 6 cm above the release point over the plate (N4) or the bowl (N6); N5 (stove) is the
bottle control.  Scripted references from the same hand-over: servo (lower and release) and open (release where it is).
BEFORE any main rollout.

    python experiments/prereg/n46.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "n46" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "n46_pg")), "n46_pg rollouts exist"
assert not os.path.exists(os.path.join(FL, "n46_pg_read.json")), "n46_pg read exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
FILES = (["results/flawed/n2_states/n2_goal_%d.npy" % t for t in (22, 26, 24)]
         + ["results/flawed/n2_branches_main.json", "results/flawed/n2_tasks.json", "results/flawed/a1g_keypoints.json",
            "results/flawed/n46_phase.json"])
CODE = ["experiments/n46_pg.py", "experiments/n46_pg_roll.sh", "experiments/n46_phase.py", "experiments/n46_lift.py",
        "experiments/pp_atlas.py", "experiments/fc_n2_client.py", "experiments/libero_client.py", "experiments/pp_serve_purify.py",
        "experiments/pp_n4d.py", "experiments/fa1g_plan.py", "experiments/pp_l1_floor.py", PP + "n46_pg_job.sh"]
SEEN = ("When this was written: SR 46-49 (N4 / N6 per task under every arm), the descriptive phase breakdown of all existing N4 / N6 "
        "rollouts (experiments/n46_phase.py, job 4704946, results/flawed/n46_phase.json) and the carry-height split (experiments/n46_lift.py, "
        "job 4705215): N4 / N6 failures are post-grasp on pi0.5 and GR00T; pooled over all sets and arms, pi0.5 N4+N6 grasped episodes with a "
        "high carry (lift > .22 m, K4's cabinet carry) succeed .04 (n 139) and low carries .66 (n 221); GR00T carries high in 556 of 582 "
        "(success .01); the servo carries high and still places (N4 1.00, N6 .77 of grasped); P_min .80 / .80; HAND_OVER 6.1's H (VLA grasp, "
        "P_min after) .80-1.00. The front end works on N4 / N6 (detector fires ~1.00, median onset 0 from oracle, paint IoU ~.98). "
        "Code is new, so a smoke ran before this row: job 4705214, pi0.5, arms vla / servo / open, N4 and N6, smoke states 0-3 "
        "(n2_branches_smoke.json), output outside results/ (.claude_tmp/pp/n46_smoke). Smoke outcomes are in the row's 'smoke' field.")
SMOKE = ("job 4705214 (tail-lab), pi0.5, N4 + N6, smoke states 0-3, output .claude_tmp/pp/n46_smoke: every prefix held the bottle at "
         "hand-over (16/16), no timeout, prefix 93-291 steps, d_xy .013-.035 m, bottle z .04-.06 m (N4) / .11-.14 m (N6) above the target "
         "origin; vla N4 4/4, N6 4/4; servo N4 4/4, N6 4/4; open N4 3/4, N6 3/4. Nothing changed after the smoke.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="n46", seed=0,
                    command="pre-registration: N46-PG, place phase alone on frozen pi0.5 / GR00T (bottle pre-grasped, held 6 cm above the "
                            "release point over the target), with scripted servo / open references (docs/SCAFFOLD_RESULTS.md 51)",
                    metrics={"status": "PRE-REGISTERED N46-PG",
                             "arms": "experiments/n46_pg.py client. Prefix: the v2 servo's waypoints (pp_n4d.plan_episode servo=True), "
                                     "kp = pp_l1_floor.kp_for(task), rng default_rng(seed_base + 10), reach timeouts 120 (Read G's), "
                                     "phases 0-4 then down to release point + 0.06 m, gripper closed; hand-over at err < .008 or 120 "
                                     "steps (cap 300 prefix steps). vla = frozen model (pp_serve_purify --model M --mode intact --comp= "
                                     "--seed 13) from its next query, task instruction unchanged, server not queried during the prefix; "
                                     "servo = lower to the release point, open 12 steps, up 0.10 m; open = open where it is 12 steps, up "
                                     "0.10 m. Launcher experiments/n46_pg_roll.sh; client seed 7; --bind track; --max-steps 450.",
                             "tasks": "familiar n2 tids 22 (N4 bottle -> plate), 26 (N6 bottle -> bowl), 24 (N5 bottle -> stove, "
                                      "control); main states 4-23 with n2_branches_main.json (all at = 0, asserted); vla on pi05 and "
                                      "groot; servo and open run once (pi05 job, scripted, model-independent).",
                             "mapping": "experiments/n46_pg.py read as hashed. Per model, arm vla, pooled N4 + N6; unit = episode held "
                                        "(oracle) at the hand-over and not successful before it; placement success; cluster bootstrap "
                                        "over states, B = 2000, numpy default_rng(0), CI 2.5/97.5. PLACE PRESENT iff CI lo >= .40; "
                                        "PLACE ABSENT iff CI hi < .30; else MIXED. Descriptive: per task N4 / N6 / N5 for vla / servo / "
                                        "open; held share (LOW POWER if < .60 on N4 or N6); post-hand-over stage; lift after hand-over; "
                                        "final tilt and nearest entity; vla - open and vla - servo on shared held cells.",
                             "interpretation": "PLACE PRESENT on a model: its place skill for the bottle on a low movable target exists, "
                                               "so its N4 / N6 residual sits upstream, in which carry program the held bottle selects "
                                               "(routing; with P_min / H as the feasibility reference, an architecture-rooted candidate). "
                                               "PLACE ABSENT: the model cannot put the bottle down on the plate / in the bowl even from 6 "
                                               "cm above, a motor gap no LIBERO-Goal demo covers (data gap as measured). Servo and open "
                                               "give the feasible ceiling and the release-only floor from the same states.",
                             "prediction": "(after the smoke, which showed pi0.5 vla 8/8, servo 8/8, open 6/8 on states 0-3) pi0.5 PLACE "
                                           "PRESENT 80% (MIXED 15%, ABSENT 5%), point ~.85. GR00T: PLACE PRESENT 35%, MIXED 35%, ABSENT 30%, "
                                           "point ~.45; its failures re-lift (lift after hand-over > .10 m) or never let go. servo >= .80 on "
                                           "N4 and N6; open ~.6-.8 on both (the smoke's floor: release alone nearly suffices from 6 cm, so "
                                           "vla - open is the closer read of any skill beyond releasing). N5 control: pi0.5 >= .7. Held at "
                                           "hand-over >= .80 per task.",
                             "seen": SEEN, "smoke": SMOKE,
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "code_sha256": {p if not p.startswith(PP) else os.path.basename(p): fsha(p) for p in CODE}},
                    bytes=Z, config={"hz": 0.06, "reach_timeout": 120, "prefix_cap": 300, "max_steps": 450}, script="experiments/prereg/n46.py"))

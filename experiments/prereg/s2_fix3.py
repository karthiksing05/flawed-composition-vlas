"""PRE-REGISTRATION ROW of S2FIX3 (SCAFFOLD_RESULTS 63): SR 62's named-mug hand-over on NEW states (seeds 920000 + k), layout O, tasks
K1 K2 N1 N2.  (a) arm Hp on frozen pi0.5: H with perception (OWLv2 + rendered depth + camera calibration; ProprioHeld for holds) in
place of simulator state; (b) arm H (SR 62's client, unchanged) and I on frozen pi0 and frozen GR00T; (c) the optional N1-release arm
is not run (step-0 diagnosis not clear).  BEFORE any main (k = 4-63) rollout or reference run.

    python experiments/prereg/s2_fix3.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2fix3                                                                # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_fix3" for r in rows), "already registered"
assert len([r for r in rows if r.get("tag") == "s2_fix2"]) == 1
assert not os.path.exists(os.path.join(FL, "s2fix3_read.json")), "s2fix3 read exists"
assert not os.path.exists(os.path.join(FL, "s2fix3_roll")), "s2fix3 rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2fix3/O/s2_living_%d.npy" % t for t in (0, 1, 2, 3)]
CODE = ["experiments/s2fix3.py", "experiments/s2fix3_det.py", "experiments/s2fix3_roll.sh", "experiments/s2fix2.py", "experiments/s2fix.py",
        "experiments/s2.py", "experiments/s2_tasks.py", "experiments/s2cue.py", "experiments/s2cue3.py", "experiments/sc_pu.py",
        "experiments/sc_rv.py", "experiments/libero_client.py", "experiments/pp_serve_purify.py", "experiments/serve_guided.py",
        "experiments/fa1g_plan.py", "experiments/fa1b_common.py", "experiments/fa1_common.py", "experiments/pp_atlas.py",
        "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/where.py", "lifelong_ood/policy/serve.py",
        "lifelong_ood/policy/backbone.py",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json", "results/flawed/s2fix2/s2fix2_states.json",
        "results/flawed/s2fix3_diag.json", "results/flawed/s2fix3_pcheck.json", "results/flawed/s2fix3_pcheck2.json",
        "results/flawed/s2fix3_pcheck3.json", "results/flawed/s2fix3_pcheck3b.json",
        "results/flawed/s2fix3/branches_main.json", "results/flawed/s2fix3/branches_smoke.json",
        "experiments/prereg/s2_fix3_pred.txt", "experiments/prereg/s2_fix3_seen.txt"] + STATES + [
        PP + "s2fix3_prep.sh", PP + "s2fix3_pcheck2.sh", PP + "s2fix3_smoke.sh", PP + "s2fix3_job.sh", PP + "s2fix3_ref.sh", PP + "s2fix3_read.sh"]
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_fix3_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_fix3_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
RUNS = [[m, "O", arm, "PRIMARY" if (m, arm) == ("pi05", "Hp") else "SECONDARY", [0, 1, 2, 3]] for m, arms in s2fix3.RUNS.items() for arm in arms]
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_fix3", seed=0,
                    command="pre-registration: S2FIX3, scene 2 (LIVING_ROOM_SCENE5) on NEW states, layout O: SR 62's named-mug hand-over "
                            "de-privileged (Hp, frozen pi0.5: OWLv2 + depth + calibration for the held mug's identity and the target "
                            "plate; ProprioHeld for holds) and across models (H and I on frozen pi0 and GR00T, SR 50's floor) "
                            "(docs/SCAFFOLD_RESULTS.md 63)",
                    metrics={"status": "PRE-REGISTERED S2FIX3",
                             "runs": RUNS,
                             "design": dict(base=s2fix3.BASE, br_base=s2fix3.BR_BASE, main=s2fix3.MAIN, smoke=s2fix3.SMOKE,
                                            held_r=s2fix3.HELD_R, ni_margin=s2fix3.NI_MARGIN, floor=s2fix3.FLOOR, cal=s2fix3.CAL,
                                            side_rule=s2fix3.SIDE_RULE, queries=s2fix3.QUERIES, min_score=s2fix3.MIN_SCORE,
                                            iou_max=s2fix3.IOU_MAX, topk=s2fix3.TOPK, assign=s2fix3.ASSIGN,
                                            plate_loc=s2fix3.PLATE_LOC, plate_r=s2fix3.PLATE_R),
                             "perception": s2fix3.__doc__.split("ARM Hp")[1].split("END Hp")[0].strip(),
                             "step0_n1_release": "DESCRIPTIVE, SR 62's outputs only (results/flawed/s2fix3_diag.json): arm H N1, "
                                                 "36 successes, 24 failures (19 after a completed release, 6 with a P abort, 3 "
                                                 "tipped); P's carry never hit its step limit; the named mug's offset from the plate "
                                                 "centre when P opened was .034 m (successes) and .037 m (failures), against .015 / "
                                                 ".009 m on K1 / K2; the reference planner's relocation rule would move the failures' "
                                                 "median to .034 m.  No clear cause -> the optional release arm is not run.",
                             "information": "Hp reads from the simulator only the agentview depth render and the camera model (rg.py "
                                            "rgp's remaining privilege, SR 52); every position comes from OWLv2 boxes on the "
                                            "policy's RGB image; holds from proprioception.  CAL and SIDE_RULE were fixed on "
                                            "development states (SR 62's k 0-63).  The simulator oracle is logged, never fed.",
                             "labels": s2fix3.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "deviations_planned": "Layout O only.  Client seed 7 and server seed 13 kept from SR 50-62 (state and "
                                                   "branch seeds are new).  H runs SR 62's client unchanged (its side file's `route` "
                                                   "field is SR 62's router table and is ignored).  Depth and camera calibration stay "
                                                   "privileged in Hp (as rgp).  Ledger tree dirty (other agents' files).",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main (k = 4-63) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_fix3.py"))

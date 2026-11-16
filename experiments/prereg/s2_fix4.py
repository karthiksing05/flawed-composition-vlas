"""PRE-REGISTRATION ROW of S2FIX4 (SCAFFOLD_RESULTS 64): SR 63's named-mug hand-over with perception (Hp) plus PUT-BACK (Hb: a hold of
a mug the instruction does not name is undone by a scripted put-back to the grasp point, cap 2 per episode, then the VLA acts again
with the same instruction), arms I / Hp / Hb on frozen pi0.5 (PRIMARY), pi0 and GR00T, NEW states (seeds 950000 + k), layout O, tasks
K1 K2 N1 N2.  BEFORE any main (k = 4-63) rollout or reference run.  Committed before it runs, so the row is named after the commit
that contains this file and the code it hashes.

    python experiments/prereg/s2_fix4.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2fix4                                                                # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_fix4" for r in rows), "already registered"
r63 = [r for r in rows if r.get("tag") == "s2_fix3"]
assert len(r63) == 1
assert not os.path.exists(os.path.join(FL, "s2fix4_read.json")), "s2fix4 read exists"
assert not os.path.exists(os.path.join(FL, "s2fix4_roll")), "s2fix4 rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2fix4/O/s2_living_%d.npy" % t for t in (0, 1, 2, 3)]
CODE = ["experiments/s2fix4.py", "experiments/s2fix4_roll.sh", "experiments/s2fix3.py", "experiments/s2fix3_det.py",
        "experiments/s2fix2.py", "experiments/s2fix.py", "experiments/s2.py", "experiments/s2_tasks.py", "experiments/s2cue.py",
        "experiments/s2cue3.py", "experiments/sc_pu.py", "experiments/sc_rv.py", "experiments/libero_client.py",
        "experiments/pp_serve_purify.py", "experiments/serve_guided.py", "experiments/fa1g_plan.py", "experiments/fa1b_common.py",
        "experiments/fa1_common.py", "experiments/pp_atlas.py",
        "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/where.py", "lifelong_ood/policy/serve.py",
        "lifelong_ood/policy/backbone.py",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json",
        "results/flawed/s2fix4/branches_main.json", "results/flawed/s2fix4/branches_smoke.json",
        "results/flawed/s2fix4/branches_dev.json",
        "experiments/prereg/s2_fix4.py", "experiments/prereg/s2_fix4_pred.txt", "experiments/prereg/s2_fix4_seen.txt"] + STATES + [
        PP + "s2fix4_prep.sh", PP + "s2fix4_smoke.sh", PP + "s2fix4_job.sh", PP + "s2fix4_ref.sh", PP + "s2fix4_read.sh"]
H = {p: fsha(p) for p in CODE}
same63 = {p: (H[p] == r63[0]["metrics"]["files_sha256"].get(p)) for p in CODE if p in r63[0]["metrics"]["files_sha256"]}
print("[s2_fix4 prereg] files shared with the s2_fix3 row, sha256 equal: %s" % same63, flush=True)
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_fix4_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_fix4_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
RUNS = [[m, "O", arm, "PRIMARY" if m == s2fix4.PRIMARY_MODEL else "SECONDARY", [0, 1, 2, 3]] for m, arms in s2fix4.RUNS.items() for arm in arms]
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_fix4", seed=0,
                    command="pre-registration: S2FIX4, scene 2 (LIVING_ROOM_SCENE5) on NEW states, layout O: SR 63's named-mug "
                            "hand-over with perception (Hp) plus a put-back of any held mug the instruction does not name (Hb), on "
                            "frozen pi0.5 (PRIMARY), pi0 and GR00T (docs/SCAFFOLD_RESULTS.md 64)",
                    metrics={"status": "PRE-REGISTERED S2FIX4",
                             "runs": RUNS,
                             "design": dict(base=s2fix4.BASE, br_base=s2fix4.BR_BASE, main=s2fix4.MAIN, smoke=s2fix4.SMOKE,
                                            dev=s2fix4.DEV, held_r=s2fix4.HELD_R, ni_margin=s2fix4.NI_MARGIN, floor=s2fix4.FLOOR,
                                            bar=s2fix4.BAR, pb_cap=s2fix4.PB_CAP, pb_clear=s2fix4.PB_CLEAR, pb_dz=s2fix4.PB_DZ,
                                            pb_up=s2fix4.PB_UP, pb_carry=s2fix4.PB_CARRY, pb_desc=s2fix4.PB_DESC,
                                            pb_retreat=s2fix4.PB_RETREAT, open_steps=s2fix4.OPEN_STEPS, off_abort=s2fix4.OFF_ABORT,
                                            name_phrases=s2fix4.NAME_PHRASES, hp_constants=dict(cal=s2fix4.F3.CAL,
                                                                                                side_rule=s2fix4.F3.SIDE_RULE,
                                                                                                queries=s2fix4.F3.QUERIES,
                                                                                                assign=s2fix4.F3.ASSIGN,
                                                                                                plate_loc=s2fix4.F3.PLATE_LOC,
                                                                                                plate_r=s2fix4.F3.PLATE_R)),
                             "arm_Hb": s2fix4.__doc__.split("ARM Hb")[1].split("END Hb")[0].strip(),
                             "information": "Hb reads, beyond Hp, (a) the instruction text, parsed for the mug names it contains (the "
                                            "VLA receives the same text, and Hp already reads the named mug as the task's object), "
                                            "and (b) the hand's position at its lowest point since the gripper closed "
                                            "(proprioception, the z_min ProprioHeld already tracks) as the set-down point.  The "
                                            "put-back reads no object pose.  No simulator read is added: as in Hp, only the "
                                            "agentview depth render and the camera model (rgp's remaining privilege).  The "
                                            "simulator oracle is logged, never fed.",
                             "k2_note": "K1 and K2 are single-step instructions that name one mug each; on K2 the rule fires at the "
                                        "trained program's first hold (the white mug).  Only the compounds KC / NC name both mugs, so "
                                        "only there the rule never fires by construction; KC / NC are not run.",
                             "labels": s2fix4.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "same_sha_as_s2_fix3_row": same63,
                             "deviations_planned": "Layout O only.  Client seed 7 and server seed 13 kept from SR 50-63 (state and "
                                                   "branch seeds are new).  Hp runs SR 63's client unchanged; Hp and Hb share "
                                                   "SR 63's perception constants (fitted on SR 62's states).  pi0 and GR00T run Hp "
                                                   "(SR 63 ran H on them).  Depth and camera calibration stay privileged.  Ledger "
                                                   "tree dirty (other agents' files).",
                             "files_sha256": H,
                             "written_before": "any main (k = 4-63) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_fix4.py"))

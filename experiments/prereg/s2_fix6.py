"""PRE-REGISTRATION ROW of S2FIX6 (SCAFFOLD_RESULTS 67): SR 63's named-mug hand-over with perception (Hp) plus KEEP THE TARGET CLEAR
(arm Hc: at a hold of a mug the instruction does not name, SR 54's target-only place P carries it to the plate the instruction does
not name, fallback the grasp spot, cap 1, then the VLA acts again), against I and Hp on frozen pi0.5 (PRIMARY) and pi0 (SECONDARY),
NEW states (seeds 970000 + k), layout O, tasks K1 K2 N1 N2.  BEFORE any main (k = 4-63) rollout or reference run.  Committed before
it runs, so the row is named after the commit that contains this file and the code it hashes.

    python experiments/prereg/s2_fix6.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2fix6                                                                # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_fix6" for r in rows), "already registered"
r63 = [r for r in rows if r.get("tag") == "s2_fix3"]
assert len(r63) == 1
assert not os.path.exists(os.path.join(FL, "s2fix6_read.json")), "s2fix6 read exists"
assert not os.path.exists(os.path.join(FL, "s2fix6_roll")), "s2fix6 rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2fix6/O/s2_living_%d.npy" % t for t in (0, 1, 2, 3)]
CODE = ["experiments/s2fix6.py", "experiments/s2fix6_roll.sh", "experiments/s2fix4.py", "experiments/s2fix3.py",
        "experiments/s2fix3_det.py", "experiments/s2fix2.py", "experiments/s2fix.py", "experiments/s2.py", "experiments/s2_tasks.py",
        "experiments/s2cue.py", "experiments/s2cue3.py", "experiments/sc_rv.py", "experiments/sc_rv_w.py", "experiments/sc_pu.py",
        "experiments/libero_client.py", "experiments/pp_serve_purify.py", "experiments/serve_guided.py", "experiments/fa1g_plan.py",
        "experiments/fa1b_common.py", "experiments/fa1_common.py", "experiments/pp_atlas.py",
        "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/where.py", "lifelong_ood/policy/serve.py",
        "lifelong_ood/policy/backbone.py",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json", "results/flawed/sc_rv_lex.json",
        "results/flawed/s2fix6/branches_main.json", "results/flawed/s2fix6/branches_smoke.json", "results/flawed/s2fix6/hist.json",
        "experiments/prereg/s2_fix6.py", "experiments/prereg/s2_fix6_pred.txt", "experiments/prereg/s2_fix6_seen.txt"] + STATES + [
        PP + "s2fix6_prep.sh", PP + "s2fix6_smoke.sh", PP + "s2fix6_job.sh", PP + "s2fix6_ref.sh", PP + "s2fix6_read.sh"]
H = {p: fsha(p) for p in CODE}
same63 = {p: (H[p] == r63[0]["metrics"]["files_sha256"].get(p)) for p in CODE if p in r63[0]["metrics"]["files_sha256"]}
print("[s2_fix6 prereg] files shared with the s2_fix3 row, sha256 equal: %s" % same63, flush=True)
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_fix6_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_fix6_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
RUNS = [[m, "O", arm, "PRIMARY" if m == s2fix6.PRIMARY_MODEL else "SECONDARY", [0, 1, 2, 3]]
        for m, arms in s2fix6.RUNS.items() for arm in arms]
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_fix6", seed=0,
                    command="pre-registration: S2FIX6, scene 2 (LIVING_ROOM_SCENE5) on NEW states, layout O: SR 63's named-mug "
                            "hand-over with perception (Hp) plus KEEP THE TARGET CLEAR (Hc: an unnamed held mug is carried by P to "
                            "the plate the instruction does not name, cap 1), against I and Hp, on frozen pi0.5 (PRIMARY) and pi0 "
                            "(docs/SCAFFOLD_RESULTS.md 67)",
                    metrics={"status": "PRE-REGISTERED S2FIX6",
                             "runs": RUNS,
                             "design": dict(base=s2fix6.BASE, br_base=s2fix6.BR_BASE, main=s2fix6.MAIN, smoke=s2fix6.SMOKE,
                                            ni_margin=s2fix6.NI_MARGIN, floor=s2fix6.FLOOR, bar=s2fix6.BAR, cl_cap=s2fix6.CL_CAP,
                                            cl_retreat=s2fix6.CL_RETREAT, plate_phrases=s2fix6.PLATE_PHRASES,
                                            hp_constants=dict(cal=s2fix6.F3.CAL, side_rule=s2fix6.F3.SIDE_RULE, queries=s2fix6.F3.QUERIES,
                                                              assign=s2fix6.F3.ASSIGN, plate_loc=s2fix6.F3.PLATE_LOC,
                                                              plate_r=s2fix6.F3.PLATE_R, held_r=s2fix6.F3.HELD_R)),
                             "arm_Hc": s2fix6.__doc__.split("ARM Hc (")[1].split("END Hc")[0].strip(),
                             "information": "Hc reads, beyond Hp, the instruction's plate words (the text the VLA receives; Hp "
                                            "already reads the target plate) and the second plate of Hp's own registration scene "
                                            "(OWLv2 + rendered depth + camera calibration, the same call).  No lookup of trained "
                                            "pairings; no new simulator read.  Depth and calibration stay privileged as in Hp.  "
                                            "The simulator oracle is logged, never fed.",
                             "labels": s2fix6.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "same_sha_as_s2_fix3_row": same63,
                             "deviations_planned": "Layout O only.  Client seed 7 and server seed 13 kept from SR 50-66 (state and "
                                                   "branch seeds are new).  Hp runs SR 63's client unchanged; Hc is SR 64's Hb "
                                                   "client with the put-back replaced by P to the unnamed plate (put-back kept as "
                                                   "the fallback) and the cap set to 1.  GR00T not run (NOT TESTABLE on every "
                                                   "earlier scene-2 state set).  Depth and camera calibration stay privileged.  "
                                                   "Ledger tree dirty (other agents' files).",
                             "files_sha256": H,
                             "written_before": "any main (k = 4-63) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_fix6.py"))

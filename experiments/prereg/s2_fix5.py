"""PRE-REGISTRATION ROW of S2FIX5 (SCAFFOLD_RESULTS 66): SR 50's role views (simulator masks; tvp = views + role phrases) in front of
SR 63's named-mug hand-over with perception (Hp), arm Hv, against I and Hp on frozen pi0.5 (PRIMARY; also Hv0 = tv views without
phrases + Hp, secondary) and pi0 (SECONDARY), NEW states (seeds 960000 + k), layout O, tasks K1 K2 N1 N2.  BEFORE any main (k = 4-63)
rollout or reference run.  Committed before it runs, so the row is named after the commit that contains this file and the code it
hashes.

    python experiments/prereg/s2_fix5.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2fix5                                                                # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_fix5" for r in rows), "already registered"
r63 = [r for r in rows if r.get("tag") == "s2_fix3"]
assert len(r63) == 1
assert not os.path.exists(os.path.join(FL, "s2fix5_read.json")), "s2fix5 read exists"
assert not os.path.exists(os.path.join(FL, "s2fix5_roll")), "s2fix5 rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2fix5/O/s2_living_%d.npy" % t for t in (0, 1, 2, 3)]
CODE = ["experiments/s2fix5.py", "experiments/s2fix5_roll.sh", "experiments/s2fix3.py", "experiments/s2fix3_det.py",
        "experiments/s2fix2.py", "experiments/s2fix.py", "experiments/s2.py", "experiments/s2_tasks.py", "experiments/s2cue.py",
        "experiments/s2cue3.py", "experiments/tv_frozen.py", "experiments/sc_rv.py", "experiments/sc_rv_w.py", "experiments/sc_pu.py",
        "experiments/libero_client.py", "experiments/pp_serve_purify.py", "experiments/serve_guided.py", "experiments/fa1g_plan.py",
        "experiments/fa1b_common.py", "experiments/fa1_common.py", "experiments/pp_atlas.py",
        "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/where.py", "lifelong_ood/policy/serve.py",
        "lifelong_ood/policy/backbone.py",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json", "results/flawed/sc_rv_lex.json",
        "results/flawed/s2fix5/branches_main.json", "results/flawed/s2fix5/branches_smoke.json",
        "experiments/prereg/s2_fix5.py", "experiments/prereg/s2_fix5_pred.txt", "experiments/prereg/s2_fix5_seen.txt"] + STATES + [
        PP + "s2fix5_prep.sh", PP + "s2fix5_smoke.sh", PP + "s2fix5_job.sh", PP + "s2fix5_ref.sh", PP + "s2fix5_read.sh"]
H = {p: fsha(p) for p in CODE}
same63 = {p: (H[p] == r63[0]["metrics"]["files_sha256"].get(p)) for p in CODE if p in r63[0]["metrics"]["files_sha256"]}
print("[s2_fix5 prereg] files shared with the s2_fix3 row, sha256 equal: %s" % same63, flush=True)
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_fix5_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_fix5_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
RUNS = [[m, "O", arm, "PRIMARY" if (m == s2fix5.PRIMARY_MODEL and arm != "Hv0") else "SECONDARY", [0, 1, 2, 3]]
        for m, arms in s2fix5.RUNS.items() for arm in arms]
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_fix5", seed=0,
                    command="pre-registration: S2FIX5, scene 2 (LIVING_ROOM_SCENE5) on NEW states, layout O: SR 50's role views "
                            "(simulator masks, tvp) in front of SR 63's named-mug hand-over with perception (Hv = views + Hp), "
                            "against I and Hp, on frozen pi0.5 (PRIMARY) and pi0 (docs/SCAFFOLD_RESULTS.md 66)",
                    metrics={"status": "PRE-REGISTERED S2FIX5",
                             "runs": RUNS,
                             "design": dict(base=s2fix5.BASE, br_base=s2fix5.BR_BASE, main=s2fix5.MAIN, smoke=s2fix5.SMOKE,
                                            ni_margin=s2fix5.NI_MARGIN, floor=s2fix5.FLOOR, bar=s2fix5.BAR,
                                            view_arms=s2fix5.VIEW_ARMS,
                                            hp_constants=dict(cal=s2fix5.F3.CAL, side_rule=s2fix5.F3.SIDE_RULE, queries=s2fix5.F3.QUERIES,
                                                              assign=s2fix5.F3.ASSIGN, plate_loc=s2fix5.F3.PLATE_LOC,
                                                              plate_r=s2fix5.F3.PLATE_R, held_r=s2fix5.F3.HELD_R)),
                             "arm_Hv": s2fix5.__doc__.split("ARM Hv (")[1].split("END Hv")[0].strip(),
                             "information": "Hv reads, beyond Hp, the SIMULATOR entity masks of the five movables in both cameras "
                                            "(PRIVILEGED, as SR 50's tv / tvp) and shows the VLA the role phrases built from the "
                                            "task's object and target (which Hp already reads).  The hand-over's perception "
                                            "(OWLv2 + rendered depth + camera calibration), hold detector, gate and P are Hp's, "
                                            "on the RAW observation.  The simulator oracle is logged, never fed.",
                             "labels": s2fix5.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "same_sha_as_s2_fix3_row": same63,
                             "deviations_planned": "Layout O only.  Client seed 7 and server seed 13 kept from SR 50-64 (state and "
                                                   "branch seeds are new).  Hp runs SR 63's client unchanged; Hv / Hv0 run it "
                                                   "unchanged inside a view wrapper (its side file's arm field reads Hp).  Views "
                                                   "use simulator masks (privileged).  Hvp (OWLv2-perceived views) and GR00T are "
                                                   "not run.  Hv0 added as a secondary pi0.5 arm (tv without phrases).  Depth and "
                                                   "camera calibration stay privileged.  Ledger tree dirty (other agents' files).",
                             "files_sha256": H,
                             "written_before": "any main (k = 4-63) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_fix5.py"))

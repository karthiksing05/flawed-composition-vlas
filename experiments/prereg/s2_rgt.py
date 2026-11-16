"""PRE-REGISTRATION ROW of S2RGT (SCAFFOLD_RESULTS 69): SR 68's scene-2 ReGuide port with tracking of the held mug (rgt, rendered
depth) and with tracking + table-plane depth (rgtp, no depth render), against I and a replication of SR 68's rgp, on frozen pi0.5
(PRIMARY) and pi0 (SECONDARY), NEW states (seeds 990000 + k, k 0-59), layout O, tasks K1 K2 N1 N2.  BEFORE any main rollout or
reference run.  Committed before it runs, so the row is named after the commit that contains this file and the code it hashes.

    python experiments/prereg/s2_rgt.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2rgt                                                                 # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_rgt" for r in rows), "already registered"
r68 = [r for r in rows if r.get("tag") == "s2_rg"]
assert len(r68) == 1
assert not os.path.exists(os.path.join(FL, "s2rgt_read.json")), "s2rgt read exists"
assert not os.path.exists(os.path.join(FL, "s2rgt_roll")), "s2rgt rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2rgt/O/s2_living_%d.npy" % t for t in (0, 1, 2, 3)]
CODE = ["experiments/s2rgt.py", "experiments/s2rgt_roll.sh", "experiments/s2rg.py", "experiments/rg.py", "experiments/rg_stats.py",
        "experiments/s2fix6.py", "experiments/s2fix4.py", "experiments/s2fix3.py", "experiments/s2fix3_det.py", "experiments/s2fix2.py",
        "experiments/s2fix.py", "experiments/s2.py", "experiments/s2_tasks.py", "experiments/s2cue.py", "experiments/s2cue3.py",
        "experiments/sc_rv.py", "experiments/sc_rv_w.py", "experiments/sc_pu.py", "experiments/libero_client.py",
        "experiments/pp_serve_purify.py", "experiments/serve_guided.py", "experiments/fa1g_plan.py", "experiments/fa1b_common.py",
        "experiments/fa1_common.py", "experiments/pp_atlas.py", "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/where.py",
        "lifelong_ood/policy/serve.py", "lifelong_ood/policy/backbone.py",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json", "results/flawed/sc_rv_lex.json",
        "results/flawed/s2rg/rg_stats_s2.json", "results/flawed/s2rgt/pcheck.json", "results/flawed/s2rgt/branches_main.json",
        "experiments/prereg/s2_rgt.py", "experiments/prereg/s2_rgt_pred.txt", "experiments/prereg/s2_rgt_seen.txt"] + STATES + [
        PP + "s2rgt_prep.sh", PP + "s2rgt_smoke.sh", PP + "s2rgt_job.sh", PP + "s2rgt_ref.sh", PP + "s2rgt_read.sh"]
H = {p: fsha(p) for p in CODE}
same68 = {p: (H[p] == r68[0]["metrics"]["files_sha256"].get(p)) for p in CODE if p in r68[0]["metrics"]["files_sha256"]}
print("[s2_rgt prereg] files shared with the s2_rg row, sha256 equal: %s" % same68, flush=True)
PC = json.load(open(s2rgt.PCHECK, encoding="utf-8"))
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_rgt_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_rgt_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
RUNS = [[m, "O", arm, "PRIMARY" if (m == s2rgt.PRIMARY_MODEL and arm in ("rgt", "rgtp")) else "SECONDARY", [0, 1, 2, 3]]
        for m, arms in s2rgt.RUNS.items() for arm in arms]
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_rgt", seed=0,
                    command="pre-registration: S2RGT, scene 2 (LIVING_ROOM_SCENE5) on NEW states, layout O: SR 68's ReGuide port "
                            "with tracking of the held mug (rgt, rendered depth) and with tracking + table-plane depth (rgtp), "
                            "against I and SR 68's rgp, on frozen pi0.5 (PRIMARY) and pi0 (docs/SCAFFOLD_RESULTS.md 69)",
                    metrics={"status": "PRE-REGISTERED S2RGT",
                             "runs": RUNS,
                             "design": dict(base=s2rgt.BASE, br_base=s2rgt.BR_BASE, main=s2rgt.MAIN, ni_margin=s2rgt.NI_MARGIN,
                                            floor=s2rgt.FLOOR, bar=s2rgt.BAR, eq_band=s2rgt.EQ_BAND, track_k=s2rgt.TRACK_K,
                                            stats=os.path.relpath(s2rgt.STATS, ROOT), held_r=s2rgt.HELD_R,
                                            pcheck={k: PC[k] for k in ("z_table", "plate_method", "cal_mug_plane", "cal_plate_plane",
                                                                       "cal_held_depth", "cal_held_plane")},
                                            hp_constants=dict(cal=s2rgt.F3.CAL, side_rule=s2rgt.F3.SIDE_RULE, plate_loc=s2rgt.F3.PLATE_LOC,
                                                              plate_r=s2rgt.F3.PLATE_R)),
                             "arms": s2rgt.__doc__.split("THE ARMS (")[1].split("END ARMS")[0].strip(),
                             "information": "rgp and rgt read OWLv2 boxes on the policy's agentview image, the rendered depth and "
                                            "the camera calibration, ProprioHeld, and ReGuide's demonstration statistics (SR 68).  "
                                            "rgtp reads OWLv2 boxes, the camera calibration, a table-plane height fitted once on "
                                            "development states, proprioception (hand height) and the same statistics; it renders "
                                            "no depth.  Every arm's constants were fitted on development states (SR 68's), none "
                                            "in this probe's 60.  The simulator oracle is logged, never fed.",
                             "labels": s2rgt.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "same_sha_as_s2_rg_row": same68,
                             "deviations_planned": "Layout O only.  Client seed 7 and server seed 13 kept from SR 50-68 (state and "
                                                   "branch seeds new).  rgp runs through s2rgt.py's client with tracking off (the "
                                                   "decision path is s2rg.py's; logging added; registration-error keys fixed for "
                                                   "the plates).  TRACK_K = 5 chosen for cost, not tuned.  Development constants "
                                                   "(Z_TABLE, plane constants, plate method, held-mug constants) fitted on SR 68's "
                                                   "states before this row.  Camera calibration stays privileged in every arm.  Hp "
                                                   "and rgo not rerun (SR 68).  GR00T not run.  Ledger tree dirty (other agents' "
                                                   "files).",
                             "files_sha256": H,
                             "written_before": "any main (k = 0-59) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_rgt.py"))

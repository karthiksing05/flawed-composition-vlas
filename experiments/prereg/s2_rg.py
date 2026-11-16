"""PRE-REGISTRATION ROW of S2RG (SCAFFOLD_RESULTS 68): SR 52's ReGuide reimplementation (experiments/rg.py's controller, unchanged)
ported to scene 2 (rgp: SR 63's OWLv2 + depth + calibration poses; rgo: simulator poses, PRIVILEGED), against I and SR 63's
named-mug hand-over Hp, on frozen pi0.5 (PRIMARY) and pi0 (SECONDARY), NEW states (seeds 980000 + k), layout O, tasks K1 K2 N1 N2.
BEFORE any main (k = 4-63) rollout or reference run.  Committed before it runs, so the row is named after the commit that contains
this file and the code it hashes.

    python experiments/prereg/s2_rg.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2rg                                                                  # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_rg" for r in rows), "already registered"
r63 = [r for r in rows if r.get("tag") == "s2_fix3"]
r52 = [r for r in rows if r.get("tag") == "reguide"]
assert len(r63) == 1 and len(r52) == 1
assert not os.path.exists(os.path.join(FL, "s2rg_read.json")), "s2rg read exists"
assert not os.path.exists(os.path.join(FL, "s2rg_roll")), "s2rg rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2rg/O/s2_living_%d.npy" % t for t in (0, 1, 2, 3)]
CODE = ["experiments/s2rg.py", "experiments/s2rg_roll.sh", "experiments/rg.py", "experiments/rg_stats.py", "experiments/s2fix6.py",
        "experiments/s2fix4.py", "experiments/s2fix3.py", "experiments/s2fix3_det.py", "experiments/s2fix2.py", "experiments/s2fix.py",
        "experiments/s2.py", "experiments/s2_tasks.py", "experiments/s2cue.py", "experiments/s2cue3.py", "experiments/sc_rv.py",
        "experiments/sc_rv_w.py", "experiments/sc_pu.py", "experiments/libero_client.py", "experiments/pp_serve_purify.py",
        "experiments/serve_guided.py", "experiments/fa1g_plan.py", "experiments/fa1b_common.py", "experiments/fa1_common.py",
        "experiments/pp_atlas.py", "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/where.py", "lifelong_ood/policy/serve.py",
        "lifelong_ood/policy/backbone.py",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json", "results/flawed/sc_rv_lex.json",
        "results/flawed/s2rg/rg_stats_s2.json", "results/flawed/s2rg/branches_main.json", "results/flawed/s2rg/branches_smoke.json",
        "experiments/prereg/s2_rg.py", "experiments/prereg/s2_rg_pred.txt", "experiments/prereg/s2_rg_seen.txt"] + STATES + [
        PP + "s2rg_prep.sh", PP + "s2rg_smoke.sh", PP + "s2rg_job.sh", PP + "s2rg_ref.sh", PP + "s2rg_read.sh"]
H = {p: fsha(p) for p in CODE}
same63 = {p: (H[p] == r63[0]["metrics"]["files_sha256"].get(p)) for p in CODE if p in r63[0]["metrics"]["files_sha256"]}
same52 = {p: (H[p] == r52[0]["metrics"]["files_sha256"].get(p)) for p in CODE if p in r52[0]["metrics"].get("files_sha256", {})}
print("[s2_rg prereg] files shared with the s2_fix3 row, sha256 equal: %s" % same63, flush=True)
print("[s2_rg prereg] files shared with the reguide row, sha256 equal: %s" % same52, flush=True)
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_rg_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_rg_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
RUNS = [[m, "O", arm, "PRIMARY" if (m == s2rg.PRIMARY_MODEL and arm == "rgp") else "SECONDARY", [0, 1, 2, 3]]
        for m, arms in s2rg.RUNS.items() for arm in arms]
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_rg", seed=0,
                    command="pre-registration: S2RG, scene 2 (LIVING_ROOM_SCENE5) on NEW states, layout O: SR 52's ReGuide "
                            "(rg.py's controller, unchanged) ported to scene 2 (rgp: OWLv2 + depth + calibration poses as SR 63; "
                            "rgo: simulator poses), against I and SR 63's Hp, on frozen pi0.5 (PRIMARY) and pi0 "
                            "(docs/SCAFFOLD_RESULTS.md 68)",
                    metrics={"status": "PRE-REGISTERED S2RG",
                             "runs": RUNS,
                             "design": dict(base=s2rg.BASE, br_base=s2rg.BR_BASE, main=s2rg.MAIN, smoke=s2rg.SMOKE,
                                            ni_margin=s2rg.NI_MARGIN, floor=s2rg.FLOOR, bar=s2rg.BAR, eq_band=s2rg.EQ_BAND,
                                            stats=os.path.relpath(s2rg.STATS, ROOT), held_r=s2rg.HELD_R,
                                            hp_constants=dict(cal=s2rg.F3.CAL, side_rule=s2rg.F3.SIDE_RULE, queries=s2rg.F3.QUERIES,
                                                              assign=s2rg.F3.ASSIGN, plate_loc=s2rg.F3.PLATE_LOC,
                                                              plate_r=s2rg.F3.PLATE_R)),
                             "port": s2rg.__doc__.split("THE PORT (")[1].split("END PORT")[0].strip(),
                             "information": "rgp reads the same inputs as Hp (OWLv2 boxes on the policy's agentview image, the "
                                            "rendered depth and the camera calibration, ProprioHeld) plus ReGuide's demonstration "
                                            "statistics, mined from scene 2's only training demos (the LIBERO-10 compound's raw "
                                            "demos, which Hp's P constants also come from).  rgo reads simulator poses every step "
                                            "(PRIVILEGED).  Neither reads any N1 / N2 demo (none exists) or any lookup of trained "
                                            "pairings beyond the statistics.  The simulator oracle is logged, never fed to rgp.",
                             "labels": s2rg.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "same_sha_as_s2_fix3_row": same63,
                             "same_sha_as_reguide_row": same52,
                             "deviations_planned": "Layout O only.  Client seed 7 and server seed 13 kept from SR 50-67 (state and "
                                                   "branch seeds are new).  Hp runs SR 63's client unchanged.  ReGuide's controller "
                                                   "class is rg.py's, unchanged; the port changes its entity sets (mugs; plates; no "
                                                   "masked class), declares every entity round (no perceived yaw), mines the "
                                                   "statistics from the compound's per-mug cycles, uses SR 63's OWLv2 perception with "
                                                   "re-detection at hold onsets and Hp's held-mug identity rule, and keeps SR 52's 18 "
                                                   "deviations (experiments/prereg/reguide_choices.txt) otherwise.  GR00T not run "
                                                   "(NOT TESTABLE on every earlier scene-2 state set).  Depth and camera calibration "
                                                   "stay privileged.  Ledger tree dirty (other agents' files).",
                             "files_sha256": H,
                             "written_before": "any main (k = 4-63) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_rg.py"))

"""PRE-REGISTRATION ROW of S2RGS (SCAFFOLD_RESULTS 72): which pose source carries the scene-2 N1 gap of ReGuide.  SR 68's ReGuide port on
frozen pi0.5 with perception everywhere (rgp), simulator poses everywhere (rgo), the held-mug pose from the simulator (rgoH) and the
target-plate pose from the simulator (rgoT), against I, on NEW states (seeds 1000000 + k, k 0-59), layout O, tasks K1 K2 N1 N2.  BEFORE
any main rollout or reference run.  Committed before it runs, so the row is named after the commit that contains this file and the code
it hashes.

    python experiments/prereg/s2_rgs.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2rgs                                                                 # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_rgs" for r in rows), "already registered"
r68 = [r for r in rows if r.get("tag") == "s2_rg"]
assert len(r68) == 1
assert not os.path.exists(os.path.join(FL, "s2rgs_read.json")), "s2rgs read exists"
assert not os.path.exists(os.path.join(FL, "s2rgs_roll")), "s2rgs rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2rgs/O/s2_living_%d.npy" % t for t in (0, 1, 2, 3)]
CODE = ["experiments/s2rgs.py", "experiments/s2rgs_roll.sh", "experiments/s2rgt.py", "experiments/s2rg.py", "experiments/rg.py",
        "experiments/rg_stats.py", "experiments/s2fix6.py", "experiments/s2fix4.py", "experiments/s2fix3.py", "experiments/s2fix3_det.py",
        "experiments/s2fix2.py", "experiments/s2fix.py", "experiments/s2.py", "experiments/s2_tasks.py", "experiments/s2cue.py",
        "experiments/s2cue3.py", "experiments/sc_rv.py", "experiments/sc_rv_w.py", "experiments/sc_pu.py", "experiments/libero_client.py",
        "experiments/pp_serve_purify.py", "experiments/serve_guided.py", "experiments/fa1g_plan.py", "experiments/fa1b_common.py",
        "experiments/fa1_common.py", "experiments/pp_atlas.py", "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/where.py",
        "lifelong_ood/policy/serve.py", "lifelong_ood/policy/backbone.py",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json", "results/flawed/sc_rv_lex.json",
        "results/flawed/s2rg/rg_stats_s2.json", "results/flawed/s2rgs/branches_main.json",
        "experiments/prereg/s2_rgs.py", "experiments/prereg/s2_rgs_pred.txt", "experiments/prereg/s2_rgs_seen.txt"] + STATES + [
        PP + "s2rgs_prep.sh", PP + "s2rgs_smoke.sh", PP + "s2rgs_job.sh", PP + "s2rgs_ref.sh", PP + "s2rgs_read.sh"]
H = {p: fsha(p) for p in CODE}
same68 = {p: (H[p] == r68[0]["metrics"]["files_sha256"].get(p)) for p in CODE if p in r68[0]["metrics"]["files_sha256"]}
print("[s2_rgs prereg] files shared with the s2_rg row, sha256 equal: %s" % same68, flush=True)
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_rgs_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_rgs_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
RUNS = [[m, "O", arm, "PRIMARY" if arm in ("rgoH", "rgoT") else "SECONDARY", [0, 1, 2, 3]]
        for m, arms in s2rgs.RUNS.items() for arm in arms]
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_rgs", seed=0,
                    command="pre-registration: S2RGS, scene 2 (LIVING_ROOM_SCENE5) on NEW states, layout O: SR 68's ReGuide port with "
                            "perception everywhere (rgp), simulator poses everywhere (rgo), the held-mug pose from the simulator "
                            "(rgoH) and the target-plate pose from the simulator (rgoT), against I, on frozen pi0.5 "
                            "(docs/SCAFFOLD_RESULTS.md 72)",
                    metrics={"status": "PRE-REGISTERED S2RGS",
                             "runs": RUNS,
                             "design": dict(base=s2rgs.BASE, br_base=s2rgs.BR_BASE, main=s2rgs.MAIN, ni_margin=s2rgs.NI_MARGIN,
                                            floor=s2rgs.FLOOR, bar=s2rgs.BAR, eq_band=s2rgs.EQ_BAND,
                                            stats=os.path.relpath(s2rgs.STATS, ROOT), held_r=s2rgs.HELD_R,
                                            bootstrap="paired over states, B = 2000, numpy default_rng(0)"),
                             "arms": s2rgs.__doc__.split("THE ARMS (")[1].split("END ARMS")[0].strip(),
                             "information": "rgp reads OWLv2 boxes on the policy's agentview image, the rendered depth and the camera "
                                            "calibration, ProprioHeld, and ReGuide's demonstration statistics (SR 68).  rgo reads the "
                                            "simulator's body positions of every mug and plate (privileged).  rgoH is rgp with the held "
                                            "mug's pose from the simulator between the attachment and the release; rgoT is rgp with the "
                                            "target plate's pose from the simulator.  No constant was fitted for this probe.  The "
                                            "simulator oracle is also logged, never fed, in rgp.",
                             "labels": s2rgs.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "same_sha_as_s2_rg_row": same68,
                             "deviations_planned": "Layout O only; frozen pi0.5 only.  Client seed 7 and server seed 13 kept from SR 50-69 "
                                                   "(state and branch seeds new).  rgp and rgo run through s2rgs.py's client (the decision "
                                                   "path is s2rg.py's; logging as s2rgt.py's, with plates keyed by name).  rgoH keeps "
                                                   "the perceived hold event and identity; rgoT makes the instructed target plate "
                                                   "always 'perceived'.  Camera calibration stays privileged in the perceiving arms.  "
                                                   "GR00T, pi0 and SmolVLA not run.  Ledger tree dirty (other agents' files).",
                             "files_sha256": H,
                             "written_before": "any main (k = 0-59) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_rgs.py"))

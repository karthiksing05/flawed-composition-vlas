"""PRE-REGISTRATION ROW of S2CUE4 (SCAFFOLD_RESULTS 60): in scene 2, (1) with the plates placed so that the plate NEARER the held
mug is the LESS salient one (Sx; PSx = SR 57's mug swap + the same plates), does the place follow the NEARER PLATE or the SALIENT
PLATE, on frozen pi0.5 and pi0?  (2) Does SR 59's nearer-plate routing hold on frozen GR00T (Sm, new states), and is GR00T
testable by SR 50's floor?  BEFORE any main (k = 4-43 of the new states) rollout or reference run of this probe.

    python experiments/prereg/s2_cue4.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2cue4                                                                # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_cue4" for r in rows), "already registered"
assert len([r for r in rows if r.get("tag") == "s2_cue3"]) == 1
assert not os.path.exists(os.path.join(FL, "s2cue4_read.json")), "s2cue4 read exists"
assert not os.path.exists(os.path.join(FL, "s2cue4_roll")), "s2cue4 rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2cue4/%s/s2_living_%d.npy" % (m, t) for m in ("O",) + s2cue4.BUILT for t in (0, 1, 2, 3)]
CODE = ["experiments/s2cue4.py", "experiments/s2cue3.py", "experiments/s2cue2.py", "experiments/s2cue.py", "experiments/s2.py",
        "experiments/s2_tasks.py", "experiments/s2_roll.sh", "experiments/libero_client.py", "experiments/tv_frozen.py",
        "experiments/sc_rv.py", "experiments/pp_atlas.py", "experiments/pp_serve_purify.py", "experiments/serve_guided.py",
        "experiments/pp_f1_mix.py", "experiments/fa1g_plan.py", "experiments/fa1b_common.py", "experiments/fa1_common.py",
        "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/serve.py", "lifelong_ood/policy/backbone.py",
        "results/flawed/sc_rv_lex.json", "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json",
        "results/flawed/s2cue4/branches_main.json", "results/flawed/s2cue4/branches_smoke.json",
        "results/flawed/s2cue4/s2cue4_states.json"] + STATES + [
        PP + "s2cue4_scan/pick.json", PP + "s2cue4_scan.sh", PP + "s2cue4_build.sh", PP + "s2cue4_smoke.sh", PP + "s2cue4_job.sh",
        PP + "s2cue4_ref.sh", PP + "s2cue4_read.sh"]
SJ = json.load(open(os.path.join(FL, "s2cue4", "s2cue4_states.json"), encoding="utf-8"))
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_cue4_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_cue4_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_cue4", seed=0,
                    command="pre-registration: S2CUE4, scene 2 (LIVING_ROOM_SCENE5) on NEW states: (1) nearer plate vs salient plate "
                            "(Sx, PSx) on frozen pi0.5 and pi0, arm I PRIMARY, pi0.5 tv SECONDARY; (2) frozen GR00T on SR 59's Sm "
                            "layout (start side vs nearer plate) with SR 50's testability floor (docs/SCAFFOLD_RESULTS.md 60)",
                    metrics={"status": "PRE-REGISTERED S2CUE4",
                             "runs": [[r[0], r[1], r[2], r[3], list(r[4])] for r in s2cue4.RUNS],
                             "design": dict(plates=SJ["plates"], sm_plates=SJ["sm_plates"], pick=SJ["pick"], rule=s2cue4.PICK_RULE,
                                            cands=s2cue4.CANDS, area_ratio=s2cue4.AREA_RATIO, centre_margin_px=s2cue4.CENTRE_MARGIN,
                                            feas_min=s2cue4.FEAS_MIN),
                             "arms": "pi0.5 and pi0: Sx I, PSx I (PRIMARY); pi0.5 Sx tv, PSx tv (SECONDARY); GR00T Sm I (PRIMARY, "
                                     "item 2) and GR00T O I on K1, K2 (testability floor, SR 50's rule: TESTABLE iff >= .70).  Tasks "
                                     "K1, K2, N1, N2 (tids 0-3), 40 new states each (seeds 845000 + k, k = 4-43; branches "
                                     "seed_base 28000000 + 1000 k).  Reference planner (C1) = s2.plan_episode unchanged from Sx, PSx "
                                     "and Sm.",
                             "mapping": s2cue4.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "deviations_planned": "pi0 tv not run (SR 59: pi0 tv made 1 and 4 placements).  PSx (swap + the same "
                                                   "plates) beside Sx so that identity, which equals the nearer plate under Sx, is "
                                                   "separated across runs.  Start side stays tied to the nearer plate on the "
                                                   "disagreeing mug by construction (SR 59 separated them).  GR00T runs Sm only "
                                                   "(no PSm), so its label is per run.  Salience = agentview pixels + centroid "
                                                   "distance of the start render from simulator segmentation; the wrist view is "
                                                   "descriptive only.  New states: no per-episode pairing to SR 50 / 59.  Ledger "
                                                   "tree dirty (other agents' files).",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main (k = 4-43) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_cue4.py"))

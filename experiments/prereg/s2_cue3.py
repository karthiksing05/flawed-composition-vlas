"""PRE-REGISTRATION ROW of S2CUE3 (SCAFFOLD_RESULTS 59): in scene 2, with the plates shifted asymmetrically (Sm; PSm = SR 57's
mug swap + the same shift), does the place follow the held mug's START SIDE or the NEARER PLATE, on frozen pi0.5 and pi0?  BEFORE any
main (k = 4-43 of the new states) rollout or reference run of this probe.

    python experiments/prereg/s2_cue3.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2cue3                                                                # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_cue3" for r in rows), "already registered"
assert len([r for r in rows if r.get("tag") == "s2_cue2"]) == 1
assert not os.path.exists(os.path.join(FL, "s2cue3_read.json")), "s2cue3 read exists"
assert not os.path.exists(os.path.join(FL, "s2cue3_roll")), "s2cue3 rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2cue3/%s/s2_living_%d.npy" % (m, t) for m in ("O",) + s2cue3.MANIPS for t in (0, 1, 2, 3)]
CODE = ["experiments/s2cue3.py", "experiments/s2cue2.py", "experiments/s2cue.py", "experiments/s2.py", "experiments/s2_tasks.py",
        "experiments/s2_roll.sh", "experiments/libero_client.py", "experiments/tv_frozen.py", "experiments/sc_rv.py",
        "experiments/pp_atlas.py", "experiments/pp_serve_purify.py", "experiments/serve_guided.py", "experiments/pp_f1_mix.py",
        "experiments/fa1g_plan.py", "experiments/fa1b_common.py", "experiments/fa1_common.py", "lifelong_ood/policy/parts_flow.py",
        "lifelong_ood/policy/serve.py", "lifelong_ood/policy/backbone.py", "results/flawed/sc_rv_lex.json",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json", "results/flawed/s2cue3/branches_main.json",
        "results/flawed/s2cue3/branches_smoke.json", "results/flawed/s2cue3/s2cue3_states.json"] + STATES + [
        PP + "s2cue3_scan2/pick.json", PP + "s2cue3_scan.sh", PP + "s2cue3_build.sh", PP + "s2cue3_smoke.sh", PP + "s2cue3_job.sh",
        PP + "s2cue3_ref.sh", PP + "s2cue3_read.sh"]
SJ = json.load(open(os.path.join(FL, "s2cue3", "s2cue3_states.json"), encoding="utf-8"))
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_cue3_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_cue3_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_cue3", seed=0,
                    command="pre-registration: S2CUE3, scene 2 (LIBERO-10 LIVING_ROOM_SCENE5): plates shifted asymmetrically (Sm) and "
                            "SR 57's mug swap + the same shift (PSm) on NEW states; frozen pi0.5 and pi0, arms I (PRIMARY) and tv; "
                            "start side vs nearer plate (docs/SCAFFOLD_RESULTS.md 59)",
                    metrics={"status": "PRE-REGISTERED S2CUE3",
                             "runs": [list(r) for r in s2cue3.RUNS],
                             "design": dict(plates=SJ["plates"], pick=SJ["pick"], rule=s2cue3.PICK_RULE, cands=s2cue3.CANDS),
                             "arms": "pi0.5 and pi0: Sm I, PSm I (PRIMARY); Sm tv, PSm tv (SECONDARY).  Tasks K1, K2, N1, N2 (tids 0-3), "
                                     "40 new states each (seeds 835000 + k, k = 4-43; branches seed_base 27000000 + 1000 k).  GR00T "
                                     "not run (NOT TESTABLE per SR 50).  Reference planner (C1) = s2.plan_episode unchanged from Sm "
                                     "and PSm.",
                             "mapping": s2cue3.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "deviations_planned": "tv added on both models and both manipulations (secondary; the brief allows one "
                                                   "secondary tv arm if cheap).  PSm (swap + shift) added beside the shift alone so "
                                                   "that identity, which equals start side under Sm, is separated across runs.  New "
                                                   "states, so no per-episode pairing to SR 50 (success is descriptive).  Start side "
                                                   "stays inseparable from mug left/right order and depth order by construction.  "
                                                   "Ledger tree dirty (other agents' files).",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main (k = 4-43) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_cue3.py"))

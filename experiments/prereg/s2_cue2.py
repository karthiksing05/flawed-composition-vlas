"""PRE-REGISTRATION ROW of S2CUE2 (SCAFFOLD_RESULTS 58): in scene 2, does the place follow the start side when neither plate is nearer
(C, C'), the plate object (M), and does SR 57's position routing hold on pi0 (P)?  BEFORE any main (k = 4-43) rollout or reference
run of this probe.

    python experiments/prereg/s2_cue2.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2cue2                                                                # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_cue2" for r in rows), "already registered"
assert len([r for r in rows if r.get("tag") == "s2_cue"]) == 1
assert not os.path.exists(os.path.join(FL, "s2cue2_read.json")), "s2cue2 read exists"
assert not os.path.exists(os.path.join(FL, "s2cue2_roll")), "s2cue2 rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2cue2/%s/s2_living_%d.npy" % (m, t) for m in s2cue2.BUILT for t in (0, 1, 2, 3)]
STATES += ["results/flawed/s2cue/P/s2_living_%d.npy" % t for t in (0, 1, 2, 3)]
CODE = ["experiments/s2cue2.py", "experiments/s2cue.py", "experiments/s2.py", "experiments/s2_tasks.py", "experiments/s2_roll.sh",
        "experiments/libero_client.py", "experiments/tv_frozen.py", "experiments/sc_rv.py", "experiments/pp_atlas.py",
        "experiments/pp_serve_purify.py", "experiments/serve_guided.py", "experiments/pp_f1_mix.py", "experiments/fa1g_plan.py",
        "experiments/fa1b_common.py", "experiments/fa1_common.py", "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/serve.py",
        "lifelong_ood/policy/backbone.py", "results/flawed/sc_rv_lex.json", "results/flawed/s2/keypoints.json",
        "results/flawed/s2/s2_tasks.json", "results/flawed/s2/branches_main.json", "results/flawed/s2cue/branches_40.json",
        "results/flawed/s2cue2/s2cue2_states.json"] + STATES + [
        PP + "s2cue2_smoke.sh", PP + "s2cue2_job.sh", PP + "s2cue2_ref.sh", PP + "s2cue2_read.sh"]
SR50 = [os.path.relpath(os.path.join(s2cue2.s2.ROLL, "%s_%s_%d%s" % (m, fe, t, ext)), ROOT)
        for m, fe in (("pi05", "I"), ("pi05", "tv"), ("pi0", "I")) for t in (0, 1, 2, 3) for ext in (".json", "_track.npz")]
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_cue2_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_cue2_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_cue2", seed=0,
                    command="pre-registration: S2CUE2, scene 2 (LIBERO-10 LIVING_ROOM_SCENE5): central starts (C, C'), mirrored plates "
                            "(M) on frozen pi0.5; SR 57's swapped starts (P) and M on frozen pi0 (docs/SCAFFOLD_RESULTS.md 58)",
                    metrics={"status": "PRE-REGISTERED S2CUE2",
                             "runs": [list(r) for r in s2cue2.RUNS],
                             "arms": "pi0.5: C I (PRIMARY), C tv, C' I (PRIMARY), C' tv, M I (PRIMARY); pi0: P I (PRIMARY), P tv, M I "
                                     "(PRIMARY).  Tasks K1, K2, N1, N2 (tids 0-3), 40 states each (SR 50's k = 4-43, SR 57's "
                                     "branches_40.json).  GR00T not run (NOT TESTABLE per SR 50, K1-K2 .508).  Reference planner (C1) "
                                     "= s2.plan_episode unchanged from C, C', M (target = the plate at the named side); P's reference "
                                     "is SR 57's.",
                             "mapping": s2cue2.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "sr50_reused_sha256": {p: fsha(p) for p in SR50},
                             "deviations_planned": "Added secondary tv arms beyond the brief (pi0.5 C' tv, pi0 P tv) so that STEP can "
                                                   "be separated from the position cues (SR 57: arm I picks the left-hand mug first). "
                                                   "C and C' remove start side and nearer plate TOGETHER; no manipulation here splits "
                                                   "them.  M is scored by side (the BDDL's object goal is also reported).  "
                                                   "INSTRUCTION is defined only when the named mug is held (SR 57 counted other-mug "
                                                   "placements as not following it).  40 of SR 50's 60 main states.  Ledger tree "
                                                   "dirty (other agents' files).",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main (k = 4-43) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_cue2.py"))

"""PRE-REGISTRATION ROW of S2CUE (SCAFFOLD_RESULTS 57): which input cue routes the place in scene 2 (LIBERO-10 LIVING_ROOM_SCENE5)?
BEFORE any main (k = 4-43) rollout or reference run from the manipulated states.

    python experiments/prereg/s2_cue.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2cue                                                                 # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_cue" for r in rows), "already registered"
assert len([r for r in rows if r.get("tag") == "scene2"]) == 1
assert not os.path.exists(os.path.join(FL, "s2cue_read.json")), "s2cue read exists"
assert not os.path.exists(os.path.join(FL, "s2cue_roll")), "s2cue rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2cue/%s/s2_living_%d.npy" % (m, t) for m in ("P", "Q") for t in (0, 1, 2, 3)]
CODE = ["experiments/s2cue.py", "experiments/s2.py", "experiments/s2_tasks.py", "experiments/s2_roll.sh", "experiments/libero_client.py",
        "experiments/tv_frozen.py", "experiments/sc_rv.py", "experiments/pp_atlas.py", "experiments/pp_serve_purify.py",
        "experiments/pp_f1_mix.py", "experiments/fa1g_plan.py", "experiments/fa1b_common.py", "experiments/fa1_common.py",
        "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/serve.py", "results/flawed/sc_rv_lex.json",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json", "results/flawed/s2/branches_main.json",
        "results/flawed/s2cue/branches_40.json", "results/flawed/s2cue/s2cue_states.json"] + STATES + [
        PP + "s2cue_smoke.sh", PP + "s2cue_job.sh", PP + "s2cue_ref.sh", PP + "s2cue_read.sh"]
SR50 = [os.path.relpath(os.path.join(s2cue.s2.ROLL, "pi05_%s_%d%s" % (fe, t, ext)), ROOT)
        for fe in ("I", "tv") for t in (0, 1, 2, 3) for ext in (".json", "_track.npz")]
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_cue_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_cue_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_cue", seed=0,
                    command="pre-registration: S2CUE, which input cue routes the place in scene 2 (LIBERO-10 LIVING_ROOM_SCENE5): "
                            "start-position swap (P), step-1-done start (Q), state read check (R) on frozen pi0.5 "
                            "(docs/SCAFFOLD_RESULTS.md 57)",
                    metrics={"status": "PRE-REGISTERED S2CUE",
                             "arms": "pi0.5 only (GR00T NOT TESTABLE per SR 50, K1-K2 .508).  Front-end arms tv (PRIMARY) and I "
                                     "(SECONDARY), s2.py client unchanged, run by s2_roll.sh unchanged with FRESH = the manipulated "
                                     "states.  P: white / yellow-and-white mugs' start xy swapped, tids 0 K1, 1 K2, 2 N1, 3 N2.  Q: white "
                                     "mug starts on the left plate, tids 1 K2, 2 N1, 3 N2 (K1 is satisfied at t = 0).  SR 50's main "
                                     "states k = 4-43 (40) and branch seeds (branches_40.json).  Reference planner (C1) = s2.plan_episode "
                                     "unchanged from the same manipulated states.  R: statecheck (bit-for-bit action comparison under "
                                     "state swaps, scene-2 prompts and renders), no rollout arm.",
                             "mapping": s2cue.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "sr50_reused_sha256": {p: fsha(p) for p in SR50},
                             "deviations_planned": "Q runs K2, N1, N2 only (K1 is already satisfied in Q's start state).  Under Q the "
                                                   "instructed N2 placement puts a second mug on an occupied plate; the reference "
                                                   "planner measures whether it is reachable (C1).  40 of SR 50's 60 main states.  "
                                                   "Ledger tree dirty (other agents' files).",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main (k = 4-43) rollout or reference run from the manipulated states"},
                    bytes=Z, config={}, script="experiments/prereg/s2_cue.py"))

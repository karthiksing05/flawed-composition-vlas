"""PRE-REGISTRATION ROW of TV-F (SCAFFOLD_RESULTS 46): test-time role views (role sequencer + learned-mask silhouettes) on FROZEN
pi0.5 and GR00T, no retraining; arms tv, tvp, w against the atlas's intact rollouts I, n2 familiar layout, main states 4-23.
BEFORE any main-state tv_frozen rollout.

    python experiments/prereg/tv_frozen.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "tv_frozen" for r in rows), "already registered"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "tv_frozen_read.json")), "main read exists"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "tv_frozen")), "main rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="tv_frozen", seed=0,
                    command="pre-registration: TV-F, test-time role views on frozen VLAs (docs/SCAFFOLD_RESULTS.md 46)",
                    metrics={"status": "PRE-REGISTERED TV-F",
                             "arms": "I = results/flawed/pp_atlas/<model>_<tid> (atlas intact, same states / branches / seeds / server); "
                                     "tv = role views from sc_seg learned masks in both cameras, sc_rv.role_paint rule (object phase: "
                                     "every movable but the object painted; target phase: every movable but the target, held object "
                                     "included), switched by ProprioHeld (sc_rv_lex.json tw/lift) stepped every env step, current state "
                                     "(reverts if the hold ends), prompt unchanged; tvp = tv + R1 role phrases (sc_t16.given); "
                                     "w = whole wrist image grey 128 while ProprioHeld is on, no silhouettes, prompt unchanged.",
                             "tasks": "familiar n2 tids 16,18,20,22,24,26 (N1-N6) and 0,2,4,6,10 (K1-K4, K6); states 4-23; "
                                      "models pi05, groot (pp_serve_purify --mode intact --seed 13).",
                             "mapping": "experiments/tv_frozen.py read as hashed, per model.  Success matrices [task, state]; paired "
                                        "bootstrap B=2000, numpy default_rng(0), tasks and states drawn jointly; CI 2.5/97.5.  PRIMARY "
                                        "d = tv - I on N1-N6: TV WINS iff d > .10 and CI lo > 0; TV LOSES iff CI hi < -.10; else NO "
                                        "CLEAR DIFFERENCE.  AC-VLA d = tv - w on N1-N6, same rule: SEPARATED FROM AC-VLA iff win; TV "
                                        "LOSES TO W iff hi < -.10; else NO CLEAR DIFFERENCE.  No verdict: tvp - I, tvp - tv, w - I; "
                                        "K-task success per arm and arm - I (cost on trained tasks); the atlas's within/cross-chunk "
                                        "excess (pp_atlas estimator, partners, stage rule unchanged) per arm; stage shares; "
                                        "segmenter IoU, paint IoU, active-object leak; hold onsets vs oracle Held.  I's excess must "
                                        "reproduce pp_atlas_read.json (reported).",
                             "prediction": "pi0.5 (I .50 on N1-N6): NO CLEAR DIFFERENCE 50%, TV WINS 30%, TV LOSES 20%. GR00T (I "
                                           ".167): TV WINS 40%, NO CLEAR DIFFERENCE 45%, TV LOSES 15%. tv - w: NO CLEAR DIFFERENCE "
                                           "on both models (70%). K-task success under tv within .10 of I on pi0.5, a larger drop "
                                           "possible on GR00T. tv lowers the cross-chunk excess against I on both models. Paint IoU "
                                           ">= .90 agentview, segmenter weakest on the bottle in the wrist.",
                             "seen": "Smoke before registration and before the predictions: pi0.5 only, smoke states 0-3 "
                                     "(n2_branches_smoke.json), N1 / N4 / K1 familiar, job 4677916. Successes of 4: tv N1 2, N4 3, "
                                     "K1 4; tvp N1 4, N4 2, K1 4; w N1 4, N4 0, K1 4. ProprioHeld first onset within 0-2 steps of "
                                     "the oracle Held on most episodes; paint IoU .94-.98 agentview, .96-.995 wrist; active leak "
                                     "<= .007. Contact sheets inspected: the rule paints as intended. No gate, threshold, rule or "
                                     "code changed after the smoke. GR00T was not smoked. The intact arm I is not run here; it reuses "
                                     "results/flawed/pp_atlas (seen in earlier registered reads: pi0.5 .50, GR00T .167 on N1-N6).",
                             "files_sha256": {p: fsha(p) for p in ["experiments/tv_frozen.py", "experiments/pp_atlas.py", "experiments/fc_n2_client.py",
                                                                   "experiments/libero_client.py", "experiments/sc_rv.py", "experiments/sc_rv_w.py",
                                                                   "experiments/sc_seg.py", "experiments/sc_t16.py", "experiments/pp_serve_purify.py",
                                                                   "results/flawed/sc_rv_lex.json", PP + "tv_roll.sh", PP + "tv_job.sh", PP + "tv_read.sh"]},
                             "written_before": "any main-state tv_frozen rollout"}, bytes=Z, config={}, script="experiments/prereg/tv_frozen.py"))

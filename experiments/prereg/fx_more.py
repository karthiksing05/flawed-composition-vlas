"""PRE-REGISTRATION ROW of FX_MORE (SCAFFOLD_RESULTS 45.3): F7 (relational language) and F7R (the structural front end) on three
more frozen VLAs: pi0 (openpi pi0_libero, PyTorch), X-VLA (lerobot/xvla-libero) and SmolVLA (lerobot/smolvla_libero), served as
experiments/pp_atlas_more_roll.sh serves them; LIBERO-Goal scene, main states 4-23.  BEFORE any main-state fx_more rollout.

    python experiments/prereg/fx_more.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
PP = "/coc/flash10/ksingara3/.claude_tmp/pp"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "fx_more" for r in rows), "already registered"
assert any(r.get("tag") == "fx" for r in rows) and any(r.get("tag") == "fx_r" for r in rows), "fx / fx_r registrations missing"
for p in ("results/flawed/fx_more_read_main.json", "results/flawed/fx_more_r_read_main.json", "results/flawed/fx_more"):
    assert not os.path.exists(os.path.join(ROOT, p)), "main output exists: %s" % p


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["experiments/fx_tasks.py", "experiments/fx_client.py", "experiments/fx_read.py", "experiments/fx_r.py",
         "experiments/serve_guided.py", "experiments/libero_client.py", "experiments/pp_atlas_more_roll.sh",
         PP + "/fx_roll_more.sh", PP + "/fxr_roll_more.sh", PP + "/fxm_job.sh", PP + "/fxm_readjob.sh"]

Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="fx_more", seed=0,
                    command="pre-registration: F7 + F7R on pi0, X-VLA, SmolVLA (docs/SCAFFOLD_RESULTS.md 45.3)",
                    metrics={"status": "PRE-REGISTERED FX_MORE",
                             "models": "pi0 (BACKBONE_CONFIG=pi0_libero), xvla (lerobot/xvla-libero, absolute eef control), smolvla "
                                       "(lerobot/smolvla_libero); each served by experiments/serve_guided.py --mode ours --seed 13 as in "
                                       "experiments/pp_atlas_more_roll.sh; clients fx_client.py (F7 tasks 5-14) and fx_r.py client (arms "
                                       "f7r, name, oracle on tasks 5, 6, 10, 11), main states 4-23, fx_branches_main.json.",
                             "mapping": "F7, per model and family (SP left/right; NG on/not on the plate): experiments/fx_read.py as hashed "
                                        "(the fx mapping, unchanged): NO POWER iff U < .50 or reference < .60; RELATION IGNORED iff "
                                        "delta = U - R > .10 with CI lower bound > 0 and C's CI upper bound < .75; RELATION USED iff C's "
                                        "lower bound > .75; else PARTIAL.  The reference is fx's own scripted servo / P_min main "
                                        "references (results/flawed/fx/ref; model-independent, not re-run).  F7R, per model and family: "
                                        "experiments/fx_r.py read as hashed (the fx_r mapping, unchanged), arm f7r, delta = R(F7R) - "
                                        "R(relational) paired over states, B = 2000 seed 0: NO POWER iff U < .50 or fx reference < .60; "
                                        "FIXED iff delta > .10 with CI lower bound > 0 and C(F7R) CI lower bound > .75; NOT FIXED iff "
                                        "delta CI upper bound < .10; else PARTIAL.  Aggregates of this extension, over the 6 new "
                                        "model-family cells: F7 EXTENDS iff RELATION IGNORED on >= 4 of 6; F7R EXTENDS iff FIXED on >= 4 "
                                        "of 6.  name and oracle arms, F7R - oracle, F7R - name, resolver accuracy, paint IoU, the "
                                        "ambiguous-choice match: reported with no verdict.  F6 and F7P not run.",
                             "prediction": "F7 EXTENDS (5 of 6): RELATION IGNORED on pi0 SP, pi0 NG, SmolVLA SP, SmolVLA NG and "
                                           "X-VLA NG; X-VLA SP NO POWER (its unique-name SP controls fail).  C near .5 and the "
                                           "relational choice equal to the ambiguous choice on >= 80% of states on every powered cell.  "
                                           "F7R EXTENDS with exactly 4 of 6: FIXED on pi0 SP, pi0 NG, SmolVLA SP, SmolVLA NG; X-VLA SP "
                                           "NO POWER, X-VLA NG PARTIAL or NOT FIXED (smoke: it takes the table bowl whatever it sees).  "
                                           "Low confidence on the SmolVLA NG cell, so F7R EXTENDS is a coin-flip-level call.  Resolver "
                                           "correct >= 95% of episodes; F7R - oracle within .05 everywhere.",
                             "seen": "Smoke on states 0-3 for X-VLA only (job 4676175; read 4676176, "
                                     ".claude_tmp/pp/fxm_smoke/fx_read_smoke.json and fxr_read_smoke.json): F7 SP NO POWER (uniL .25, "
                                     "uniR 0, U .125; R 0, C .50); F7 NG RELATION IGNORED (uniP 0, uniT 1.0, U .50 exactly at the "
                                     "gate; R 0, C .50, rel = amb 1.0; X-VLA grasps the table bowl in every NG episode).  F7R: SP NO "
                                     "POWER (f7r = name = oracle = .125); NG PARTIAL (f7r = name = oracle = .50, C(F7R) .50).  "
                                     "Resolver 16/16, paint IoU .94-.99.  pi0 and SmolVLA were not smoked.  No gate, mapping or "
                                     "threshold was changed after the smoke.",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any main-state fx_more rollout"},
                    bytes=Z, config={}, script="experiments/prereg/fx_more.py"))

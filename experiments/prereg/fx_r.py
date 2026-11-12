"""PRE-REGISTRATION ROW of F7R (SCAFFOLD_RESULTS 45): referent resolver + unique-name prompt + distractor role view on frozen
pi0.5 and GR00T, LIBERO-Goal scene, main states 4-23.  BEFORE any main-state fx_r rollout.

    python experiments/prereg/fx_r.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "fx_r" for r in rows), "already registered"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "fx_r_read_main.json")), "main read exists"
assert not os.path.isdir(os.path.join(ROOT, "results", "flawed", "fx", "roll_r")), "main rollouts exist"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="fx_r", seed=0, command="pre-registration: F7R (docs/SCAFFOLD_RESULTS.md 45)",
                    metrics={"status": "PRE-REGISTERED F7R",
                             "mapping": "experiments/fx_r.py read as hashed, per model and family (SP left/right; NG on/not on), arm f7r. "
                                        "delta = R(F7R) - R(relational), paired over states, fx_read.boot B=2000 seed 0. First match: "
                                        "NO POWER iff U < .50 or fx reference < .60; FIXED iff delta > .10 with CI lower bound > 0 and "
                                        "C(F7R) CI lower bound > .75; NOT FIXED iff delta CI upper bound < .10; else PARTIAL. "
                                        "F7R GENERAL iff FIXED on >= 3 of 4 model-family cells. name and oracle arms, F7R - oracle, "
                                        "F7R - name, resolver accuracy and paint IoU reported with no verdict.",
                             "prediction": "FIXED on pi0.5 SP, pi0.5 NG, GR00T SP; GR00T NG NOT FIXED or NO POWER (its unique-name "
                                           "control grasps the cheese, U near .5). F7R - oracle within .05 everywhere. Resolver correct "
                                           ">= 95% of episodes.",
                             "seen": "Smoke on states 0-3 (fxr_read_smoke.json): FIXED on 3 of 4 cells; GR00T NG U = .50, delta 0. "
                                     "Resolver 16/16. The agent suggested moving the NO POWER gate to U <= .50 after the smoke; "
                                     "kept at < .50 (the gate as built before the smoke).",
                             "files_sha256": {p: fsha(p) for p in ["experiments/fx_r.py", "experiments/fx_client.py",
                                                                   "experiments/fx_tasks.py", "experiments/fx_read.py"]},
                             "written_before": "any main-state fx_r rollout"}, bytes=Z, config={}, script="experiments/prereg/fx_r.py"))

"""PRE-REGISTRATION ROW of FX (SCAFFOLD_RESULTS 45): F6 (memory, descriptive) and F7 (relational and negated language) on frozen
pi0.5 and GR00T, LIBERO-Goal scene, main states.  BEFORE any main-state fx rollout.

    python experiments/prereg/fx.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "fx" for r in rows), "already registered"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "fx_read_main.json")), "main read exists"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="fx", seed=0, command="pre-registration: FX F6/F7 (docs/SCAFFOLD_RESULTS.md 45)",
                    metrics={"status": "PRE-REGISTERED FX",
                             "mapping": "experiments/fx_read.py as hashed. F7 per model and family (SP left/right; NG on/not on the plate): "
                                        "NO POWER iff U < .50 or reference < .60; RELATION IGNORED iff delta = U - R > .10 with CI lower bound > 0 "
                                        "and C's CI upper bound < .75; RELATION USED iff C's lower bound > .75; else PARTIAL. F7 GENERAL iff "
                                        "RELATION IGNORED on at least 3 of the 4 model-family cells. F6: DESCRIPTIVE ONLY; fx_read's F6 labels "
                                        "are not read as verdicts, because hiding the plate breaks both VLAs even in the familiar layout "
                                        "(smoke fam_hid 0/4), which confounds memory with an out-of-distribution view; report vis/hid/abs "
                                        "success and where the object ends. F7P (LIBERO-PRO negation) reported with no verdict.",
                             "prediction": "F7 GENERAL: pi0.5 RELATION IGNORED on SP and NG; GR00T RELATION IGNORED on SP, PARTIAL or NO POWER "
                                           "on NG (its unique-name plate control grasps the cheese). C near .5 everywhere and the relational "
                                           "choice equal to the ambiguous condition's choice on >= 80% of states. F6: fam_vis >= .9, fam_hid and "
                                           "disp_hid near 0 for both models.",
                             "seen": "Smoke on states 0-3 (fx_read_smoke.json): F7 C = .50 and both = .00 for both models and families; "
                                     "references SP .67-.83, NG 1.00; F6 fam_vis 1.0, fam_hid 0, disp_vis pi0.5 1.0 / GR00T .5, reference "
                                     "P_min .75. The reference floor (.80 in the agent's draft) was set to .60 after seeing these.",
                             "files_sha256": {p: fsha(p) for p in ["experiments/fx_tasks.py", "experiments/fx_client.py", "experiments/fx_ref.py",
                                                                   "experiments/fx_read.py"]},
                             "written_before": "any main-state fx rollout"}, bytes=Z, config={}, script="experiments/prereg/fx.py"))

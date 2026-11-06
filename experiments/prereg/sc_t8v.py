"""PRE-REGISTRATION ROW of T8b (docs/SCAFFOLD_RESULTS.md 30).  BEFORE training starts.

    python experiments/prereg/sc_t8v.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t8v_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t8v" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t8v", seed=0, command="pre-registration: T8b (docs/SCAFFOLD_RESULTS.md 30)",
                    metrics={"status": "PRE-REGISTERED T8b",
                             "labelled": "development read on N2 main states 4-23, which have shaped U; the VLA reads pixels, U oracle positions; 335 vs 400 demos; one VLA seed against three; the user asked to keep iterating overnight, and PLAN_SCAFFOLD T8 was approved in the plan",
                             "mapping": "new pairings N1-N6, familiar, VLA - U: VLA MATCHES iff CI lower >= -.10; RUNG HOLDS iff CI upper < -.10; else UNRESOLVED. Known K1-K6 upper < -.10 turns RUNG HOLDS into UNDERFIT",
                             "prediction": "RUNG HOLDS; VLA near .95 known, near .45 new pairings against U .894; new layout near .15 new pairings against U .381",
                             "states": "dev only: N2 main states 4-23; no smoke rollouts",
                             "training": "pi05_goal8 from pi05_base, batch 64, 3000 steps, read at step 3000",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t8v.py", "experiments/c_train_pi05.py", "lifelong_ood/policy/l90c_config.py", "results/flawed/sc_t8v_manifest.json"]},
                             "written_before": "T8b training"}, bytes=Z, config={}, script="experiments/prereg/sc_t8v.py"))

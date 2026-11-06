"""PRE-REGISTRATION ROW of T16 (docs/SCAFFOLD_RESULTS.md 33).  BEFORE training starts.

    python experiments/prereg/sc_t16.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t16_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t16" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t16", seed=0, command="pre-registration: T16 (docs/SCAFFOLD_RESULTS.md 33)",
                    metrics={"status": "PRE-REGISTERED T16",
                             "labelled": "development read on N2 main states 4-23; both VLAs read pixels and proprioception; one seed per arm; the user set this comparison on 2026-11-06 and has not reviewed this design",
                             "mapping": "S - T8b's VLA. New pairings N1-N6, familiar: STRUCTURE WINS iff CI lower > +.10; STRUCTURE LOSES iff CI upper < -.10; else NO CLEAR DIFFERENCE. Known K1-K6, familiar: NO COST iff CI lower >= -.10; else COST",
                             "prediction": "STRUCTURE WINS; S near .75 new pairings against VLA near .45; NO COST, S near .90 known; new layout S near .25 against VLA near .15",
                             "states": "dev only: N2 main states 4-23",
                             "training": "pi05_goal8_roles from pi05_base, batch 64, 3000 steps, read at step 3000, as T8b",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t16.py", "experiments/c_train_pi05.py", "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/parts_flow.py", "results/flawed/sc_t16_lex.json", "results/flawed/sc_t16_manifest.json"]},
                             "written_before": "T16 training"}, bytes=Z, config={}, script="experiments/prereg/sc_t16.py"))

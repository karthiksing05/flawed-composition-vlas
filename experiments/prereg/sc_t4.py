"""PRE-REGISTRATION ROW of T4 (docs/SCAFFOLD_RESULTS.md 10).  BEFORE any non-smoke build or rollout.

    python experiments/prereg/sc_t4.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t4_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t4" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t4", seed=0, command="pre-registration: T4, the unsupervised rebuild (docs/SCAFFOLD_RESULTS.md 10)",
                    metrics={"status": "PRE-REGISTERED T4",
                             "labelled": "development read on N2 main states 4-23, which have shaped earlier work; the user approved T4 and asked for unsupervised reusable parts on 2026-11-05, and has not reviewed these rules",
                             "mapping_V1": "U minus L4, familiar known and new cells, first match: COSTS iff either CI lies wholly below -.10; HOLDS iff both lower bounds >= -.10; else UNRESOLVED",
                             "mapping_V2": "each of Unm, Unp, Unt, Unr, Ur8, U0 minus U: KEEP iff in any of 4 cells the difference < -.10 and its CI upper < 0; else DROP",
                             "mapping_V3": "created part minus full U, k = 1, familiar: TRANSFERS iff the lower bound >= -.10 on both atoms; else NOT YET",
                             "prediction": "V1 HOLDS; KEEP on Unm, Unr, U0, Unt; DROP on Unp, Ur8; V3 TRANSFERS on the plate, NOT YET on the bottle, so NOT YET overall",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1, with 50-step builds, read before this row",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t4.py", "experiments/pp_p_train.py", "experiments/fa1h_roll.py", "lifelong_ood/policy/parts_flow.py"]},
                             "written_before": "any non-smoke build or rollout of T4"}, bytes=Z, config={}, script="experiments/prereg/sc_t4.py"))

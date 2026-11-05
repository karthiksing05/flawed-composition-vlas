"""PRE-REGISTRATION ROW of T0 and T3 (docs/SCAFFOLD_RESULTS.md 1-2).  BEFORE T0's read and before any T3 run on dev states.

    python experiments/prereg/sc_t03.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
for res in ("sc_t0.json", "sc_t3_read.json"):
    assert not os.path.exists(os.path.join(ROOT, "results", "flawed", res)), "read exists"
assert not any(r.get("tag") == "sc_t03" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t03", seed=0, command="pre-registration: T0 and T3 (docs/SCAFFOLD_RESULTS.md 1-2)",
                    metrics={"status": "PRE-REGISTERED T0 and T3",
                             "labelled": "development reads on N2 main states 4-23, which have shaped earlier work; changes no earlier verdict; the user approved the plan on 2026-11-05 and has not reviewed these rules",
                             "mapping_T0_A": "per system and layout, first match: TOO FEW iff under 10 events in the fatal or control set; BOUNDARY iff CI lower bound of delta above 0 and delta > 0.10; else NOT AT BOUNDARIES",
                             "mapping_T0_B": "per system: LONG iff long share >= 2/3 and CI lower >= 0.5; SHORT iff share <= 1/3 and CI upper <= 0.5; else MIXED",
                             "mapping_T3": "first match: INSTRUMENT FAILS iff familiar hold in A or familiar success in B < 0.70; CONTACT FAILS iff either contrast's CI lies wholly below -0.10; CONTACT INVARIANT iff both CI lower bounds >= -0.10; else UNRESOLVED",
                             "gate": "sc_t3 equiv: mode none disagrees with the atlas on more than 1 of 8 episodes stops T3",
                             "prediction": "T0 A NOT AT BOUNDARIES on both VLAs; T0 B MIXED on P_min, long share .4-.7; T3 CONTACT FAILS on A through near-fixture descents, D_B inside the margin, familiar rates over the floor",
                             "states": "dev only: N2 main states 4-23; smoke states 0-1 read before this row",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "docs/PLAN_SCAFFOLD.md", "experiments/sc_t0.py", "experiments/sc_t3.py"]},
                             "written_before": "T0's read and any T3 run outside the smoke"}, bytes=Z, config={}, script="experiments/prereg/sc_t03.py"))

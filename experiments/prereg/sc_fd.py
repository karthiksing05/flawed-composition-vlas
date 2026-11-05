"""PRE-REGISTRATION ROWS of F and D (docs/SCAFFOLD_RESULTS.md 7-8).  BEFORE any non-smoke run of either.

    python experiments/prereg/sc_fd.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
for tag, res in (("sc_feas", "sc_feas_read.json"), ("sc_disc", "sc_disc_read.json")):
    assert not os.path.exists(os.path.join(ROOT, "results", "flawed", res)), "read exists"
    assert not any(r.get("tag") == tag for r in rows), f"{tag} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_feas", seed=0, command="pre-registration: F, physical feasibility (docs/SCAFFOLD_RESULTS.md 7)",
                    metrics={"status": "PRE-REGISTERED F",
                             "labelled": "development read, no training; the user approved this test on 2026-11-05 and has not reviewed these rules; a hold means held_lift or success, set after the smoke",
                             "mapping": "per state, an option counts if it holds in at least 2 of 3 noise seeds. First match: INSTRUMENT FAILS iff familiar hold_any < .90 on any object; GAIN iff some object's new-layout hold_any - hold_base >= .10 with CI lower bound above 0; else NO GAIN",
                             "prediction": "GAIN on the cheese, hold_any about .75 against about .45; no gain on the bowl; familiar over the floor, bottle at risk",
                             "states": "dev only: N2 main states 4-23; smoke state 0 read before this row",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_feas.py", "experiments/fa1l_cgen.py"]},
                             "written_before": "any F run outside the smoke"}, bytes=Z, config={}, script="experiments/prereg/sc_fd.py"))
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_disc", seed=0, command="pre-registration: D, part discovery (docs/SCAFFOLD_RESULTS.md 8)",
                    metrics={"status": "PRE-REGISTERED D",
                             "labelled": "development read, no rollouts; demos 40-49 held out from build U40; these demos and instructions shaped earlier design; the user asked for unsupervised reusable parts on 2026-11-05 and has not reviewed these rules",
                             "mapping": "split iff a part with several tasks has refit gain above the seed's noise floor (largest gain among single-task parts) and CI lower bound above 0; a split stands iff 2 of 3 seeds agree. First match: UNDER-SPLITS iff no bottle object split stands; OVER-SPLITS iff at least 2 splits stand on anything other than the bottle; else RECOVERS. Merges reported with no verdict",
                             "prediction": "RECOVERS with the K5 bottle split at low margin; few merges among objects, some among fixture targets",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_disc.py", "experiments/pp_p_train.py", "lifelong_ood/policy/parts_flow.py"]},
                             "written_before": "any non-smoke build or score of D"}, bytes=Z, config={}, script="experiments/prereg/sc_fd.py"))

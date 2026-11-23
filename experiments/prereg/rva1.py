"""PRE-REGISTRATION ROW tag `rva1` (experiments/prereg/rva1_pred.txt; docs/notes/plan_review_atlas.md 2, 7.1): the failure-mode split of held-out-pairing failures from EXISTING rollouts.
Written AFTER the geometry job (geom_dev.json) and the reader's selftest, BEFORE the classifier reads any arm (results/flawed/rva1/rva1_*.json must not exist).  sbatch (CPU) only."""
import glob
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "rva1" for r in rows), "already registered"
assert not glob.glob(os.path.join(ROOT, "results", "flawed", "rva1", "rva1_*.json")), "classifier output exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


CODE = ["experiments/prereg/rva1_pred.txt", "experiments/rva1_geom.py", "experiments/rva1_read.py", "results/flawed/rva1/geom_dev.json", "docs/notes/plan_review_atlas.md"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "rva1_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-23 after rva1_geom.py ran (geom_dev.json) and rva1_read.py's selftest passed on synthetic sequences; before the classifier (or its validity gate) read any real arm's chan log.  "
        "Success rates of every arm were seen earlier (pzdev, raaf_sel); no failure class of a trained arm has been computed.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="rva1", seed=0,
                    command="pre-registration: RVA1: failure-mode split (G / W / O / H / M) of N1-N6 failures from existing development-state rollouts; rules P / R / MIXED; validity gate",
                    metrics={"status": "PRE-REGISTERED RVA1", "arm": "pi0-sL, pi0-A0, RAA-sL, A, A0 (free extras RAA-sL s43, A+0, B); Phase A N1-N6 dev states 4-63; Phase B tracks F6_b / F7 pool",
                             "mapping": "docs/notes/plan_review_atlas.md section 2", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "none at registration", "files_sha256": {p: fsha(p) for p in CODE}, "written_before": "any real-arm classification"},
                    bytes=Z, config={}, script="experiments/prereg/rva1.py"))

"""PRE-REGISTRATION ROW of the break atlas on three more VLAs (docs/ARCH_FAILURES.md; BREAK_ATLAS's design unchanged): pi0
(pi0_libero), X-VLA (xvla-libero) and SmolVLA (smolvla_libero), all LIBERO-trained including LIBERO-Goal.  BEFORE any main-state run.

    python experiments/prereg/pp_atlas_more.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "pp_atlas_more" for r in rows), "already registered"
for m in ("pi0", "xvla", "smolvla"):
    for tid in range(32):
        assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_atlas", "%s_%d.json" % (m, tid))), (m, tid)


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_atlas_more", seed=0, command="pre-registration: break atlas on pi0, X-VLA, SmolVLA",
                    metrics={"status": "PRE-REGISTERED ATLAS-MORE",
                             "mapping": "BREAK_ATLAS's registered estimator and mapping (pp_atlas read via experiments/pp_atlas_models_read.py, "
                                        "which reproduces pp_atlas_read.json exactly for pi05 and groot), per model and layout: within-chunk and "
                                        "cross-chunk excess E with CIs and the labels BOTH / WITHIN-CHUNK / CROSS-CHUNK / NEITHER. For ARCH_FAILURES "
                                        "F1 criterion (2): F1 GENERALISES iff the familiar layout's cross-chunk E has CI lower bound > 0 and E > .10 "
                                        "on at least 2 of the 3 new models (so on >= 4 of 5 models with pi0.5 and GR00T).",
                             "prediction": "Familiar: pi0 BOTH, X-VLA CROSS-CHUNK or BOTH, SmolVLA WITHIN-CHUNK or BOTH (its smoke failed at the "
                                           "grasp). F1 GENERALISES (pi0 and X-VLA). New layout: NEITHER on all three.",
                             "seen": "Smokes on states 0-3, familiar: known 20/20 (pi0), 20/20 (X-VLA), 18/20 (SmolVLA); N1 0/4 each, stages pi0 "
                                     "GRASP x3 RELEASE x1, X-VLA RELEASE x4, SmolVLA GRASP x4. Registered atlas results for pi0.5 and GR00T.",
                             "files_sha256": {p: fsha(p) for p in ["experiments/pp_atlas_more_roll.sh", "experiments/pp_atlas_models_read.py"]},
                             "written_before": "any main-state run of pi0, X-VLA or SmolVLA"}, bytes=Z, config={},
                    script="experiments/prereg/pp_atlas_more.py"))

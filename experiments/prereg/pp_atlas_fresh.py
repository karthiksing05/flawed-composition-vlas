"""PRE-REGISTRATION ROW of the five-model break atlas's FRESH-STATE replication (docs/ARCH_FAILURES.md "Update 2026-11-12";
docs/BREAK_ATLAS.md's design unchanged): pi0.5, GR00T N1, pi0, X-VLA and SmolVLA, familiar layout, on 20 fresh start states
k = 36..55 (experiments/fc_n2_fresh.py).  BEFORE any rollout on a fresh state.

    python experiments/prereg/pp_atlas_fresh.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "pp_atlas_fresh" for r in rows), "already registered"
assert os.path.exists(os.path.join(ROOT, "results", "flawed", "n2_states_fresh", "states.npy")), "fresh states not drawn yet"
for d in ("pp_atlas_fresh", "pp_atlas_fresh_smoke"):
    for m in ("pi05", "groot", "pi0", "xvla", "smolvla"):
        for tid in range(32):
            assert not os.path.exists(os.path.join(ROOT, "results", "flawed", d, "%s_%d.json" % (m, tid))), (d, m, tid)
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_atlas_fresh_read.json"))


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["experiments/fc_n2_fresh.py", "experiments/fc_n2_fresh.sh", "experiments/pp_atlas_fresh_roll.sh",
         "experiments/pp_atlas_fresh_read.py", "experiments/pp_atlas_models_read.py", "experiments/pp_atlas.py",
         "experiments/pp_atlas_client.py", "results/flawed/n2_states_fresh/states.npy", "results/flawed/n2_branches_fresh.json"]
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_atlas_fresh", seed=0,
                    command="pre-registration: five-model break atlas, fresh familiar-layout states k = 36-55",
                    metrics={"status": "PRE-REGISTERED ATLAS-FRESH",
                             "mapping": "BREAK_ATLAS's registered estimator and labels (pp_atlas.py's stage rule, partners, bootstrap B = 2000 "
                                        "rng seed 0; read through experiments/pp_atlas_fresh_read.py, which runs pp_atlas_models_read.py's "
                                        "read_models unchanged except output dir, state list and familiar-only layout loop), per model "
                                        "(pi05, groot, pi0, xvla, smolvla), familiar layout, on the 20 fresh states k = 36-55 (harness "
                                        "trials 24-43, seed_base 13000000 + 1000 k), 11 tasks (K1, K2, K3, K4, K6, N1-N6). Replication "
                                        "verdicts: F1 REPLICATES iff the cross-chunk break (group 'cross' CI lower bound > 0 and E > .10) "
                                        "holds on at least 3 of the 4 models where it held on the development states (pi0.5, GR00T, pi0, "
                                        "X-VLA). F2 REPLICATES iff the within-chunk GRASP excess (stage GRASP, same bar: CI lower bound > 0 "
                                        "and E > .10) holds on both pi0 and SmolVLA.",
                             "prediction": "F1 REPLICATES and F2 REPLICATES; the five familiar-layout verdicts match the development read on "
                                           "at least 4 of 5 models.",
                             "seen": "development-state five-model read (results/flawed/pp_atlas_more_read.json); no fresh state rolled "
                                     "before this row",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any rollout on a fresh state (smoke included) of any model"}, bytes=Z, config={},
                    script="experiments/prereg/pp_atlas_fresh.py"))

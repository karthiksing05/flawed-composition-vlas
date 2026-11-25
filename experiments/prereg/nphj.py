"""PRE-REGISTRATION ROW tag `nphj` (experiments/prereg/nphj_pred.txt): pi0-A0 trained 24k steps (seeds 42, 43) and a jointly trained where-branch J on pi0 (seeds 42-45), both on the existing 7th set k 880-939.
Written BEFORE any 24k-step or J training and any rollout of the new arms.  Copy of nphs.py.  sbatch (CPU) only."""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "nphj" for r in rows), "already registered"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "nphj7", "a_pzA24_s42")) and not os.path.exists(os.path.join(ROOT, "results", "flawed", "nphj7", "a_pzJ_s42")), "rollouts exist"
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c"
for exp in ("t24k_s42", "t24k_s43"):
    assert not os.path.exists(CK + "/jax/pi0_g8v_raac_l_ae/" + exp), "training exists " + exp
for s in (42, 43, 44, 45):
    assert not os.path.exists(CK + "/jax/pi0_g8v_raaj_l_ae/tj_s%d" % s), "training exists"
assert os.path.isdir(os.path.join(ROOT, "results/flawed/n2_states_w7")) and os.path.isdir(os.path.join(ROOT, "results/flawed/nphs7")), "7th set / comparators missing"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


J = "docs/notes/nphj_jobs/"
CODE = ["experiments/prereg/nphj_pred.txt", "experiments/prereg/nphj_seen.txt", "experiments/prereg/nphj.py", "experiments/nphj_read.py", "experiments/nphj_e1.py", "experiments/nphs_read.py", "experiments/nph_read.py",
        "experiments/c_train_nphj.py", "experiments/raaj_export.py", "experiments/raaj_check.py", "experiments/raaj_cfgcheck.py", "experiments/raa_wj.py", "experiments/raa.py",
        "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raaj_model.py", "lifelong_ood/policy/backbone_j.py", "lifelong_ood/policy/raah0_head.npz"] + \
       [J + f for f in ("nphj_train.sh", "nphj_chain.sh", "nphj_export.sh", "nphj_check.sh", "nphj_roll.sh", "nphj_gated.sh", "nphj_pipeline.sh")]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "nphj_pred.txt"), encoding="utf-8").read()
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "nphj_seen.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="nphj", seed=0,
                    command="pre-registration: NPHJ: pi0-A0 at 24k steps (u1 under-training, u2) and a jointly trained where-branch J (j1 PRIMARY J - A0 on N, j2, jK), 7th set k 880-939, four-seed nphs comparators",
                    metrics={"status": "PRE-REGISTERED NPHJ", "arm": "pi0_g8v_raac_l_ae exp t24k_s42/43 (24,001 steps); pi0_g8v_raaj_l_ae exp tj_s42-45 (6,001 steps); Phase A N1-N6 / K1-K6 (60 states, k 880-939)",
                             "mapping": "docs/notes/review_atlas_adv2.md items 2 and 3 (experiments 6 and 7); docs/notes/nphs_results.md", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "hard cap 110 GPU-h; everything on overcap; the check job (identity, smoke, export parity) was first run once as a code debug before this row (no rollout, no result)",
                             "files_sha256": {p: fsha(p) for p in CODE}, "written_before": "any 24k-step or J training and any rollout of the new arms"},
                    bytes=Z, config={}, script="experiments/prereg/nphj.py"))

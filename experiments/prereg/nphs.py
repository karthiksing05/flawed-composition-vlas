"""PRE-REGISTRATION ROW tag `nphs` (experiments/prereg/nphs_pred.txt): multi-seed NPH (pi0-H*, pi0-A0, pi0-sL; seeds 42-45), 7th fresh state set k 880-939.
Written BEFORE any seed-43/44/45 training and BEFORE the 7th set is built.  Copy of nph.py.  sbatch (CPU) only."""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "nphs" for r in rows), "already registered"
for p in ("results/flawed/n2_states_w7", "results/flawed/n2_branches_w7.json", "results/flawed/nphs7"):
    assert not os.path.exists(os.path.join(ROOT, p)), "exists: " + p
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c"
for cfg in ("pi0_g8v_raah_l_ae", "pi0_g8v_raac_l_ae", "pi0_g8v_raas_l_ae"):
    assert os.path.isdir(CK + "/pytorch/%s_tl6000" % cfg), "seed-42 export missing: " + cfg
    for s in (43, 44, 45):
        assert not os.path.exists(CK + "/jax/%s/tl_s%d" % (cfg, s)), "training exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


J = "docs/notes/nphs_jobs/"
CODE = ["experiments/prereg/nphs_pred.txt", "experiments/prereg/nphs_seen.txt", "experiments/prereg/nphs.py", "experiments/nphs_read.py", "experiments/nphs_e1.py", "experiments/nphs_disjoint.py",
        "experiments/nphs_geom.py", "experiments/fc_n2_w7.py", "experiments/c_train_nphs.py", "experiments/nph_read.py", "experiments/raa_wh0.py", "experiments/raa.py",
        "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raah0_head.npz", "docs/notes/plan_necessity_power.md"] + \
       [J + f for f in ("nphs_train.sh", "nphs_chain.sh", "nphs_export.sh", "nphs_check.sh", "nphs_roll.sh", "nphs_states.sh", "nphs_gated.sh", "nphs_pipeline.sh")]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "nphs_pred.txt"), encoding="utf-8").read()
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "nphs_seen.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="nphs", seed=0,
                    command="pre-registration: NPHS: multi-seed NPH; n1s (primary, pi0-H* - pi0-A0 on N, hierarchical bootstrap), n2s, d1s, nKs, seed SD, E1",
                    metrics={"status": "PRE-REGISTERED NPHS", "arm": "pi0_g8v_raah_l_ae / raac_l_ae / raas_l_ae, seeds 42 (existing tl6000), 43, 44, 45 (exp tl_s<seed>, tag s<seed>_6000); Phase A N1-N6 / K1-K6 (60 states, k 880-939)",
                             "mapping": "docs/notes/plan_necessity_power.md section 3; docs/notes/nph_results.md", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "hard cap 130 GPU-h; pcheck of new pi0-H* seeds runs on overcap (tail-lab keeps 4 training chains)",
                             "files_sha256": {p: fsha(p) for p in CODE}, "written_before": "seed 43-45 training and building the 7th set / any rollout on it"},
                    bytes=Z, config={}, script="experiments/prereg/nphs.py"))

"""PRE-REGISTRATION ROW tag `nphp` (experiments/prereg/nphp_pred.txt): pi0.5 multi-seed NPH (H*, A0, RAA-sL; seeds 42-45), 8th fresh state set k 940-999.
Written BEFORE any new-seed / H* pi0.5 training (smoke excepted) and BEFORE the 8th set is built.  Copy of nphs.py.  sbatch (CPU) only."""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "nphp" for r in rows), "already registered"
for p in ("results/flawed/n2_states_w8", "results/flawed/n2_branches_w8.json", "results/flawed/nphp8"):
    assert not os.path.exists(os.path.join(ROOT, p)), "exists: " + p
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c"
for d in ("pi05_g8v_raac_l_ae_tl6000", "pi05_g8v_raas_l_ae_main6000", "pi05_g8v_raas_l_ae_s43_6000"):
    assert os.path.isdir(CK + "/pytorch/" + d), "existing export missing: " + d
for cfg, seeds in (("pi05_g8v_raah0_l_ae", (42, 43, 44, 45)), ("pi05_g8v_raac_l_ae", (43, 44, 45)), ("pi05_g8v_raas_l_ae", (44, 45))):
    for s in seeds:
        assert not os.path.exists(CK + "/jax/%s/tl_s%d" % (cfg, s)), "training exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


J = "docs/notes/nphp_jobs/"
CODE = ["experiments/prereg/nphp_pred.txt", "experiments/prereg/nphp_seen.txt", "experiments/prereg/nphp.py", "experiments/nphp_read.py", "experiments/nphp_e1.py", "experiments/nphp_disjoint.py",
        "experiments/nphp_geom.py", "experiments/fc_n2_w8.py", "experiments/c_train_nphp.py", "experiments/nph_read.py", "experiments/raa_wh0p.py", "experiments/raa.py",
        "experiments/raah0p_check.py", "experiments/raah0p_cfgcheck.py", "lifelong_ood/policy/raah0p_model.py", "lifelong_ood/policy/raah0_model.py", "lifelong_ood/policy/backbone_h0.py",
        "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raah0_head.npz", "docs/notes/plan_necessity_power.md"] + \
       [J + f for f in ("nphp_train.sh", "nphp_chain.sh", "nphp_export.sh", "nphp_check.sh", "nphp_roll.sh", "nphp_states.sh", "nphp_gated.sh", "nphp_pipeline.sh", "nphp_checks.sh")]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "nphp_pred.txt"), encoding="utf-8").read()
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "nphp_seen.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="nphp", seed=0,
                    command="pre-registration: NPHP: pi0.5 multi-seed NPH; p1 (primary, H* - A0 on N, hierarchical bootstrap), p2, pd, pK, seed SD, E1",
                    metrics={"status": "PRE-REGISTERED NPHP", "arm": "pi05_g8v_raah0_l_ae / raac_l_ae / raas_l_ae, seeds 42-45 (existing: A0 42, sL 42, 43; new exp tl_s<seed>, tag s<seed>_6000); Phase A N1-N6 / K1-K6 (60 states, k 940-999)",
                             "mapping": "docs/notes/nphs_results.md; docs/notes/plan_necessity_power.md section 3", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "hard cap 150 GPU-h; H* head = raah0_head.npz as fit for pi0 (no refit; raw-unit head, same RAA-c norm stats); real H* pcheck jobs run on overcap",
                             "files_sha256": {p: fsha(p) for p in CODE}, "written_before": "any non-smoke pi0.5 H* / seed 43-45 training and building the 8th set / any rollout on it"},
                    bytes=Z, config={}, script="experiments/prereg/nphp.py"))

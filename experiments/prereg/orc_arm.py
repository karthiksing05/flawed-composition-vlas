"""PRE-REGISTRATION ROW tag `orc_arm` (docs/notes/plan_oracle_pi0.md 4.3): the conditional oracle arm ORC = pi05_g8v_raac_lo_ae.  Written AFTER the Stage 1a gate fired (A+0 phi PARTIAL, orc_read_Ap0.json)
and the ORC data build + smoke, BEFORE any main / tl training job.  Predictions: experiments/prereg/orc_pred.txt (o6-o9).  Run through sbatch (CPU) only."""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

JAXD = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "orc_arm" for r in rows), "already registered"
for e in ("main", "tl"):
    assert not os.path.isdir(os.path.join(JAXD, "pi05_g8v_raac_lo_ae", e)), "training has started"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
LOGS = [x for x in os.environ.get("ORC_LOGS", "").split("~") if x]
CODE = ["docs/notes/plan_oracle_pi0.md", "experiments/prereg/orc_pred.txt", "lifelong_ood/policy/l90c_config.py", "experiments/c_train_x.py", "results/flawed/raaf/raafp_rv_manifest_raaco.json",
        "results/flawed/orc/orc_read_Ap0.json", "results/flawed/orc/gen_pairs_summary.json", PP + "pz_train.sh", PP + "pz_chain.sh"] + LOGS
PRED = open(os.path.join(ROOT, "experiments", "prereg", "orc_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-22 after the Stage 1a gate fired (A+0 phi .771 [.605, .887] PARTIAL; A0 -.001, A+ phi_net -.068, RAA-sL -.006, A .662).  The pair yield was 17 / 50 (bowl) and 22 / 50 (cream cheese) new demos, "
        "below the plan's 70-of-100 expectation (k3) and its risk-6 line of 50 per object; the coordinator instructed training from the pairs that exist (k3).  ORC therefore adds 39 paired demos to A+0's 100, not 100.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="orc_arm", seed=0,
                    command="pre-registration: ORC arm pi05_g8v_raac_lo_ae (docs/notes/plan_oracle_pi0.md 2.3): A+0 exactly, data lol/libero_g8v_raacpo (A+0's planner demos + 39 paired-spot demos)",
                    metrics={"status": "PRE-REGISTERED ORC arm (conditional on the gate)", "arm": "ORC pi05_g8v_raac_lo_ae, servo_k 0, 6,001 steps, batch 32, lr 5e-5, frozen VLM, exp tl",
                             "mapping": "docs/notes/plan_oracle_pi0.md sections 2, 4.1, 4.3", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "pairs available: 17 bowl + 22 cheese (plan: 100); the 15 cm rule was not relaxed",
                             "files_sha256": {p: fsha(p) for p in CODE}, "written_before": "any ORC main / tl training"},
                    bytes=Z, config={}, script="experiments/prereg/orc_arm.py"))

"""PRE-REGISTRATION ROW of the RAA-sL SECOND SEED (docs/notes/sr82_s43.md; SR 82 = seed 1).  BEFORE any seed-43 training job (exp smoke43 / s43 / s43tl), probe or rollout.
Predictions + replication rule: experiments/prereg/raaslS2_pred.txt.  Run through sbatch only."""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

JAXD = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raaslS2" for r in rows), "already registered"
d = os.path.join(JAXD, "pi05_g8v_raas_l_ae")
for e in ("s43", "s43tl", "smoke43"):
    assert not os.path.isdir(os.path.join(d, e)), "training has started: " + e


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["docs/notes/sr82.md", "experiments/prereg/raaslS2_pred.txt", "experiments/c_train_pi05.py", "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raas_model.py",
        "experiments/raasl_read.py", "experiments/wa_s.py", "experiments/raa.py", PP + "raaslS2_train.sh", PP + "raaslS2_chain.sh", PP + "raa_roll.sh", PP + "wasl_roll.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raaslS2_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-22 AFTER seed 1's results (SR 82: N1-N6 .878, K1-K6 .936, D1 +.319, D2 -.367 REVERSED, probe in.tok -.007) were seen; "
        "no seed-43 checkpoint, probe or rollout exists.  Only the seed (--seed 43) changes.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raaslS2", seed=1,
                    command="pre-registration: RAA-sL second training seed (pi05_g8v_raas_l_ae, --seed 43, exp s43 / s43tl), replication of SR 82's N1-N6 / K1-K6, D1 / D2 and probe in.tok / servo shares",
                    metrics={"status": "PRE-REGISTERED RAA-sL seed 2", "arm": "pi05_g8v_raas_l_ae exactly as SR 82 (6,001 steps, batch 32, lr 5e-5, frozen VLM, pi05_libero init), TrainConfig.seed 43 instead of 42",
                             "mapping": "docs/notes/sr82.md; docs/notes/sr82_s43.md", "prediction": PRED, "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any seed-43 training job (smoke43 / s43 / s43tl), probe and rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raaslS2.py"))

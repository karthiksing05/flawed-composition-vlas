"""PRE-REGISTRATION ROW of RAA-F Route A (docs/notes/raaf_plan.md, commit 6a0ee159): relabelled-goal training of the servo-free state-token model.  BEFORE any main training job
(exp main / tl) and any rollout or probe.  Predictions: experiments/prereg/raaf_pred.txt.  Run through sbatch only."""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

JAX = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raac_lf_ae"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raaf" for r in rows), "already registered"
assert not any(os.path.isdir(os.path.join(JAX, e)) for e in ("main", "tl")), "main training has started"
assert set(os.listdir(JAX) if os.path.isdir(JAX) else []) <= {"smoke"}


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["docs/notes/raaf_plan.md", "experiments/prereg/raaf_pred.txt", "experiments/raaf_prep.py", "experiments/raaf_geom.py", "experiments/raaf_null.py", "experiments/raaf_qcheck.py",
        "experiments/raaf_tokmax.py", "experiments/raaf_rv.py", "experiments/sc_rv.py", "experiments/c_train_pi05.py", "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raas_model.py",
        "lifelong_ood/policy/raac_model.py", "results/flawed/raaf_delta.json", "results/flawed/raaf/step0/s0_1_3_geom.json", "results/flawed/raaf/step0/s0_3_quantile_baseline.json",
        "results/flawed/raaf/step0/s0_4_null.json", "results/flawed/raaf/step0/s0_3_token_normmax.json", "results/flawed/raas_gate.json",
        "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json", PP + "raaf_train.sh", PP + "raaf_chain.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raaf_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-20.  Known when written: Step 0 CPU checks and the dataset build / 20-step smoke (plumbing only); no Route A checkpoint, probe or rollout exists.  Dataset lol/libero_g8v_raacf built and verified (400 -> 972 episodes, 52,181 -> 76,105 frames; 572 copies, every copy checked equal to its source rows + K delta); 20-step smoke (exp smoke, tail-lab job 4818884): step-0 loss .0529, state_proj in the parameter list, RAA-c norm stats loaded.  DEVIATION (orchestrator decision): the 2.4 "
        "2% quantile STOP read as excess over the demos' own baseline (18.0% vs 18.5%): clears; state-token x-dim widening 7.9% -> 25.2% accepted; RAA-c norm stats kept.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raaf", seed=0,
                    command="pre-registration: RAA-F Route A (docs/notes/raaf_plan.md): RAA-c's state-token model (servo_k = 0) from pi05_libero on lol/libero_g8v_raacf (RAA-c data + relabelled held-carry copies); selection by the A2 probe",
                    metrics={"status": "PRE-REGISTERED RAA-F Route A", "arm": "pi05_g8v_raac_lf_ae: RAA-c model, weights from pi05_libero, RAA-c norm stats / assets, data lol/libero_g8v_raacf, recipe of RAA-sL (6,001 steps, batch 32, lr 5e-5, frozen VLM), exp main / tl",
                             "mapping": "docs/notes/raaf_plan.md sections 6-7", "prediction": PRED, "seen": SEEN, "deviation_notes": "quantile STOP excess reading (orchestrator); see raaf_pred.txt NOTES",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main Route A training job (exp main / tl), probe and rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raaf.py"))

"""PRE-REGISTRATION ROW of RAA-F A0 (docs/notes/raaf_plan.md, "ORCHESTRATOR ERROR AND CORRECTION: Route B's servo"): Route A's model and config on the base data lol/libero_g8v_raac.
BEFORE any main A0 training job (exp main / tl) and any A0 probe or rollout.  Predictions: experiments/prereg/raaf0_pred.txt.  Run through sbatch only."""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

JAX = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raac_l_ae"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raaf0" for r in rows), "already registered"
assert not any(os.path.isdir(os.path.join(JAX, e)) for e in ("main", "tl")), "main training has started"
assert set(os.listdir(JAX) if os.path.isdir(JAX) else []) <= {"smoke"}


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["docs/notes/raaf_plan.md", "experiments/prereg/raaf0_pred.txt", "experiments/c_train_pi05.py", "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raas_model.py",
        "lifelong_ood/policy/raac_model.py",
        "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json", PP + "raaf0_train.sh", PP + "raaf0_chain.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raaf0_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-21.  Known when written: the 20-step A0 smoke (plumbing only; step-0 loss in the smoke log of the A0 launch record); Route A and B are training, no probe or rollout of any route checkpoint exists.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raaf0", seed=0,
                    command="pre-registration: RAA-F A0 (docs/notes/raaf_plan.md): Route A's model and config (state token, servo_k = 0, pi05_libero init, RAA-c norm stats) on the base data lol/libero_g8v_raac; servo-matched comparator for A and B",
                    metrics={"status": "PRE-REGISTERED RAA-F A0", "arm": "pi05_g8v_raac_l_ae: RAA-c model, weights from pi05_libero, RAA-c norm stats / assets, data lol/libero_g8v_raac, recipe of Route A (6,001 steps, batch 32, lr 5e-5, 1,000 warmup, no EMA, frozen VLM, one seed), exp main / tl",
                             "mapping": "docs/notes/raaf_plan.md sections 1, 6.1 and the correction section", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "ORCHESTRATOR ERROR: the plan registered B as RAA-sL's model (token + servo) plus the crop; the resume message to the B builder said servo off, so B (pi05_g8v_raac_lb_ae) differs from RAA-sL by crop AND servo removal.  A0 is therefore trained unconditionally (plan section 1 had it conditional on A passing) as the servo-matched comparator: B - A0 isolates the crop, A - A0 the relabel.",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main A0 training job (exp main / tl), probe and rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raaf0.py"))

"""PRE-REGISTRATION ROW of RAA-F A+0, the servo-free A+ (docs/notes/raaf_plan.md 3; docs/notes/raafp_diag.md).  BEFORE any training job (exp smoke / main / tl) of pi05_g8v_raac_lp_ae
and any probe or rollout.  Predictions: experiments/prereg/raafp0_pred.txt.  Post-hoc addition: written after A+'s probe and dev read were seen.  Run through sbatch only."""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

JAXD = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raafp0" for r in rows), "already registered"
d = os.path.join(JAXD, "pi05_g8v_raac_lp_ae")
assert not os.path.isdir(d), "training has started: " + d


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["docs/notes/raaf_plan.md", "docs/notes/raafp_diag.md", "docs/notes/raaf_sel.md", "experiments/prereg/raafp0_pred.txt", "experiments/c_train_pi05.py", "lifelong_ood/policy/l90c_config.py",
        "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py", "results/flawed/raaf/raafp_rv_manifest_raac.json",
        "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json", PP + "raafp0_train.sh", PP + "raafp0_chain.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raafp0_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-21 AFTER A+'s probe and dev read were seen (docs/notes/raafp_diag.md, docs/notes/raaf_sel.md A+ section): in.tok where-share RAA-sL -.007, A0 -.001, A .274, A+ -.007; A+ dev F6_a .417.  "
        "No A+0 checkpoint, probe or rollout exists.  The config differs from pi05_g8v_raas_lp_ae only in RS.servo_fields(k=0.0).  Post-hoc addition to the selection set; only confirmatory test = the unbuilt 5th fresh set (N/K k 760-819, F7 860-939, F6 940-1019).")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raafp0", seed=0,
                    command="pre-registration: RAA-F A+0 (docs/notes/raaf_plan.md 3): A+ exactly (RAA-sL token, planner table-spot data lol/libero_g8v_raacp, pi05_libero weights) with servo_k = 0 (servo_fields(k=0.0)); post-hoc addition to the selection set",
                    metrics={"status": "PRE-REGISTERED RAA-F A+0 (post-hoc addition)", "arm": "A+0 pi05_g8v_raac_lp_ae (data lol/libero_g8v_raacp, RAA-c norm stats, servo_k 0, 6,001 steps, batch 32, lr 5e-5, frozen VLM), exp main / tl",
                             "mapping": "docs/notes/raaf_plan.md sections 3, 6-7; docs/notes/raaf_sel.md", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "added after A+ probe and dev read were seen; confirmatory test = 5th fresh set only; A0 is the servo-matched prefix reference (registered .089 kept and reported alongside)",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any A+0 training job (exp smoke / main / tl), probe and rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raafp0.py"))

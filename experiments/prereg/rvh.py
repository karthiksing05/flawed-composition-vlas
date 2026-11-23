"""PRE-REGISTRATION ROW tag `rvh` (experiments/prereg/rvh_pred.txt; docs/notes/plan_review_atlas.md 6, 7.2): the learned where-head arm H = pi05_g8v_raah_l_ae.  Written AFTER raah_fit, raah_check and the 20-step
smoke, BEFORE the main training chain (jax/pi05_g8v_raah_l_ae/tl must not exist) and before the 5th set.  sbatch (CPU) only.  Args: sha-checked files are listed in CODE; the check / smoke logs are named in argv[1:]."""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "rvh" for r in rows), "already registered"
for d in ("main", "tl"):
    assert not os.path.exists("/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raah_l_ae/" + d), "training exists: " + d


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
LOGS = sys.argv[1:]
CODE = ["experiments/prereg/rvh_pred.txt", "experiments/raah_fit.py", "lifelong_ood/policy/raah_head.npz", "results/flawed/raah/fit.json", "lifelong_ood/policy/raah_model.py",
        "lifelong_ood/policy/backbone_h.py", "lifelong_ood/policy/l90c_config.py", "experiments/raah_check.py", "experiments/c_train_h.py", "experiments/raah_cfgcheck.py",
        PP + "rvh_train.sh", PP + "rvh_chain.sh", PP + "rvh_roll.sh", PP + "rvh_wa.sh"] + LOGS
PRED = open(os.path.join(ROOT, "experiments", "prereg", "rvh_pred.txt"), encoding="utf-8").read()
fit = json.load(open(os.path.join(ROOT, "results", "flawed", "raah", "fit.json"), encoding="utf-8"))
SEEN = ("Written 2026-11-23 after the head fit: validation R2 head %.4f vs linear-saturating refit %.4f (K %.3f vs gate 2.851; head - linear %+.4f), npz sha256 %s, best step %d; "
        "raah_check (items 1-3) and the 20-step smoke ran (logs hashed); no H checkpoint exists." % (
            fit["r2_val"]["head"], fit["r2_val"]["linear_refit"], fit["K_refit_train"], fit["r2_val"]["head_minus_linear"], fit["npz_sha256"], fit["best_step"]))
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="rvh", seed=0,
                    command="pre-registration: RVH: pi05_g8v_raah_l_ae (RAA-sL with the servo replaced by a frozen learned MLP where-head), one seed, one chain",
                    metrics={"status": "PRE-REGISTERED RVH", "arm": "pi05_g8v_raah_l_ae (H); training 6,001 steps batch 32 lr 5e-5 seed 42 on lol/libero_g8v_raac, init pi05_libero",
                             "mapping": "docs/notes/plan_review_atlas.md section 6", "prediction": PRED, "seen": SEEN, "head_fit": {k: fit[k] for k in ("r2_val", "r2_train", "K_refit_train", "best_step", "val_loss", "npz_sha256", "magnitude_along_x")},
                             "deviation_notes": "none at registration", "files_sha256": {p: fsha(p) for p in CODE}, "written_before": "H main training / 5th set / any H rollout"},
                    bytes=Z, config={}, script="experiments/prereg/rvh.py"))

"""PRE-REGISTRATION ROW tag `pz` (docs/notes/plan_oracle_pi0.md 3, 4.2): RAA-sL vs A0 on openpi pi0 (pi0_g8v_raas_l_ae, pi0_g8v_raac_l_ae).  Written AFTER the pz smokes, pz_check and the
serving smoke, BEFORE any main / tl training job of pi0_g8v_* and before any probe or rollout of these arms.  Predictions: experiments/prereg/pz_pred.txt.  Run through sbatch (CPU) only."""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

JAXD = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "pz" for r in rows), "already registered"
for n in ("pi0_g8v_raas_l_ae", "pi0_g8v_raac_l_ae"):
    for e in ("main", "tl"):
        assert not os.path.isdir(os.path.join(JAXD, n, e)), "training has started: %s/%s" % (n, e)


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
LOGS = os.environ["PZ_LOGS"].split("~")                                       # the pz_check log(s), smoke train logs, serving-smoke log (paths)
CODE = ["docs/notes/plan_oracle_pi0.md", "experiments/prereg/pz_pred.txt", "lifelong_ood/policy/raas0_model.py", "lifelong_ood/policy/l90c_config.py", "experiments/c_train_x.py",
        "experiments/pi0z_export.py", "experiments/pz_check.py", "experiments/pz_cfgcheck.py", "experiments/a2fx_capture.sh",
        "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json",
        PP + "pz_train.sh", PP + "pz_chain.sh", PP + "pz_cpu.sh", PP + "pz_check.sh", PP + "pz_export.sh"] + LOGS
PRED = open(os.path.join(ROOT, "experiments", "prereg", "pz_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-22 before any pi0_g8v_* training beyond the 20-step smokes.  Seen before writing: the pi0.5 results (RAA-sL in.tok where-share -.007, A0 -.001, A .274, A+ -.007; raaf_sel.md), "
        "the smoke step-0 losses and pz_check.  No pi0_g8v_* checkpoint beyond the smoke, probe or rollout exists.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="pz", seed=0,
                    command="pre-registration: PZ (docs/notes/plan_oracle_pi0.md 3): RAA-sL vs A0 on openpi pi0 (native trained state_proj, token in dims 0-3, quantile norm forced, raw actions, horizon 10, pi0_libero weights)",
                    metrics={"status": "PRE-REGISTERED PZ", "arm": "pi0-sL pi0_g8v_raas_l_ae (servo kept), pi0-A0 pi0_g8v_raac_l_ae (servo off); data lol/libero_g8v_raac, RAA-c norm stats, 6,001 steps, batch 32, lr 5e-5, frozen VLM, exp tl",
                             "mapping": "docs/notes/plan_oracle_pi0.md sections 3, 4.2, 7", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "pi0 labels are raw LIBERO actions (pi0_libero used a - s), horizon 10 (pi0_libero 50), quantile norm forced (pi0 default z-score); stated in the plan",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any pi0_g8v_* main / tl training job, probe and rollout"},
                    bytes=Z, config={}, script="experiments/prereg/pz.py"))

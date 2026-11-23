"""PRE-REGISTRATION ROW tag `pzdev`: full development-state rollouts of the two pi0 arms (experiments/prereg/pzdev_pred.txt).  Written BEFORE any pi0 dev rollout
(results/flawed/raaf/dev_pzsL and dev_pzA0 must not exist).  Run through sbatch (CPU) only."""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "pzdev" for r in rows), "already registered"
for r in ("pzsL", "pzA0"):
    assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "raaf", "dev_" + r)), "rollouts exist for " + r


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/prereg/pzdev_pred.txt", "experiments/pzdev_read.py", "experiments/raaf_dev_read.py", "docs/notes/oracle_pi0_results.md",
        "results/flawed/pz/a2/raaf_probe_pzsL.json", "results/flawed/pz/a2/raaf_probe_pzA0.json", PP + "pzdev_roll.sh", PP + "pzdev_wa.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "pzdev_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-22 after the PZ probe reads and the probe-capture success numbers of both pi0 arms (4 tasks x 20 states) were seen; before any full dev-state rollout of a pi0 arm.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="pzdev", seed=0,
                    command="pre-registration: PZDEV: full development-state rollouts of pi0-sL and pi0-A0 (raaf_dev_read pipeline); contrasts d1-d3, K non-inferiority",
                    metrics={"status": "PRE-REGISTERED PZDEV", "arm": "pi0_g8v_raas_l_ae tl6000, pi0_g8v_raac_l_ae tl6000; Phase A N1-N6 / K1-K6 (60 states), Phase B F7 / F6_b / F6_a",
                             "mapping": "docs/notes/oracle_pi0_results.md section 4", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "rule 'read but not in control' = d1 CI upper < 0 and S_off CI lower >= .30 and K non-inferiority",
                             "files_sha256": {p: fsha(p) for p in CODE}, "written_before": "any pi0 dev rollout"},
                    bytes=Z, config={}, script="experiments/prereg/pzdev.py"))

"""PRE-REGISTRATION ROW tag `orc` (docs/notes/plan_oracle_pi0.md 2, 4.1): the oracle readout of existing checkpoints on held-out table-spot frames (phi).  Written BEFORE orc_probe runs on any
checkpoint (results/flawed/orc/ holds no orc_read_*.json and no probe_* dir).  Predictions: experiments/prereg/orc_pred.txt.  Run through sbatch (CPU) only."""
import glob
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "orc" for r in rows), "already registered"
OD = os.path.join(ROOT, "results", "flawed", "orc")
assert not glob.glob(os.path.join(OD, "orc_read_*.json")) and not glob.glob(os.path.join(OD, "probe_*")), "a read or probe exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


GEN = "/coc/flash10/ksingara3/.claude_tmp/raaf/gen_orc_test/pairs.json"
CODE = ["docs/notes/plan_oracle_pi0.md", "experiments/prereg/orc_pred.txt", "experiments/orc_gen.py", "experiments/orc_obs.py", "experiments/orc_probe.py", "experiments/orc_read.py",
        "results/flawed/orc/obs_test/selection.json", GEN, os.environ.get("ORC_SELFTEST_LOG", "")]
CODE = [c for c in CODE if c]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "orc_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-22 before any model was run on table-spot frames.  Seen before writing: the N + K in.tok shares of RAA-sL, A0, A, A+, A+0 (raaf_sel.md), the test-set selection counts (obs_test/selection.json), "
        "the orc_read selftest (1 / 0 / .5).  Deviations at this point: test set generation hit the registered cap of 250 attempts per object with fewer than 20 complete pairs (see the results note).")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="orc", seed=0,
                    command="pre-registration: ORC (docs/notes/plan_oracle_pi0.md 2): follow share phi of existing checkpoints on held-out paired table-spot frames; conditional oracle arm pi05_g8v_raac_lo_ae",
                    metrics={"status": "PRE-REGISTERED ORC readout", "arm": "A+0 pi05_g8v_raac_lp_ae, A+ pi05_g8v_raas_lp_ae, A0 pi05_g8v_raac_l_ae, A pi05_g8v_raac_lf_ae, RAA-sL pi05_g8v_raas_l_ae (readout only)",
                             "mapping": "docs/notes/plan_oracle_pi0.md sections 2.4, 2.5, 4.1, 7", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "test set smaller than the 20 pairs per object target (cap 250 attempts); ORC arm data yield per object < 50 (see results note)",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "orc_probe on any checkpoint; any ORC training"},
                    bytes=Z, config={}, script="experiments/prereg/orc.py"))

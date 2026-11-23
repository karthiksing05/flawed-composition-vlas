"""PRE-REGISTRATION ROW tag `rvw5` (experiments/prereg/rvw5_pred.txt; docs/notes/plan_review_atlas.md 3-6, 7.3): the 5th fresh state set takes over for E2 / E3 / E4.  Written BEFORE the set is built
(results/flawed/n2_states_w5, n2_branches_w5.json, results/flawed/rg_comp_5 must not exist) and after the rvh row.  sbatch (CPU) only."""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "rvw5" for r in rows), "already registered"
assert any(r.get("tag") == "rvh" for r in rows), "the rvh row must be written first"
for p in ("results/flawed/n2_states_w5", "results/flawed/n2_branches_w5.json", "results/flawed/rg_comp_5"):
    assert not os.path.exists(os.path.join(ROOT, p)), "exists: " + p


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/prereg/rvw5_pred.txt", "docs/notes/plan_review_atlas.md", "experiments/fc_n2_w5.py", "experiments/rg_comp_5.py", "experiments/wa_sw5_5.py", "experiments/raa_wh.py",
        "experiments/wa_sh5.py", "experiments/rvw5_read.py", "experiments/rvw5_disjoint.py", "experiments/rvw5_nowcheck.py", "experiments/rvw5_geom.py", "lifelong_ood/policy/raah_head.npz",
        PP + "rvw5_roll.sh", PP + "rvw5_wa.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "rvw5_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-23 after the rvh row and before the set is built.  raaf_plan.md 7.1's recipe is used verbatim (N/K k 760-819, F7 860-939, F6 940-1019); the confirmatory raafc row of raaf_plan "
        "is VOID and will never be written (no RAA-F route passed, raaf_sel.md); this row takes the 5th set over.  The E1 failure split (rva1) of the development-state arms was seen before this row "
        "(pi0-A0 failures mostly class O, rule MIXED); every arm's development-state success was seen except H.  No arm has been rolled on these states.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="rvw5", seed=0,
                    command="pre-registration: RVW5: the 5th fresh state set; contrasts d1' (primary), r1 (primary), h1 (primary), Holm; E2 read-but-not-in-control, E3 where-module removal, E4 learned where-head",
                    metrics={"status": "PRE-REGISTERED RVW5", "arm": "pi0-sL, pi0-A0, RAA-sL, A0, A, W5, noW, H (+ options: pi0 arms F6_b / F6_a cells, RAA-sL seed 2 Phase A)",
                             "mapping": "docs/notes/plan_review_atlas.md sections 3-6", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": ("scope fixed by the orchestrator: core plus the pi0 arms' F6 cells (tids 8, 9) and RAA-sL seed 2 (pi05_g8v_raas_l_ae_s43_6000, arm raasl43) Phase A on the 5th set; "
                                                 "NOT run: H seed 2, the H probe; E1-opt only if the E1 gate fails.  The options carry no registered contrast (descriptive)."),
                             "files_sha256": {p: fsha(p) for p in CODE}, "written_before": "building the 5th set / any rollout on it"},
                    bytes=Z, config={}, script="experiments/prereg/rvw5.py"))

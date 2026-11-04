"""PRE-REGISTRATION ROW of PLAN_PARTS round 15 (docs/PARTS_RESULTS.md section 47: the backbone-alone path reads no atom-specific slot
attribute -- the object's hand-set size is 0 on backbone-only rows).  Written after the round's builds and creations (training only),
BEFORE its S1 read and any of its rollouts.  The Layer-2 GO mapping is pp_l2's, unchanged, with S1 restated per PLAN_PARTS 10.5.

    python experiments/prereg/pp_r15.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG, SECTION = "pp_r15", 47
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_r15_roll", "pp_r15_read.json", "pp_r15vrtr_read.json", "pp_r15k5_read.json", "pp_s1_mix_full_nsz_m0.json",
          "pp_s1_mix_full_nsz_m1.json", "pp_s1_mix_full_nsz_m2.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PARTS_RESULTS.md"), encoding="utf-8").read()
m = re.search(rf"## {SECTION}\. Round .*?(?=\n## {SECTION + 1}\.|\Z)", doc, flags=re.S)
assert m, f"section {SECTION} missing"
FILES = ["lifelong_ood/policy/parts_flow.py", "experiments/pp_p_train.py", "experiments/pp_p_roll.py", "experiments/pp_s1_mix.py",
         "experiments/pp_l2_read.py", "experiments/pp_l2r32_read.py", "experiments/pp_round_read.py", "experiments/fa1h_roll.py",
         "experiments/fa1g_read.py", "docs/PLAN_PARTS.md"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command=f"pre-registration: PLAN_PARTS round 15, backbone-alone path reads no size (docs/PARTS_RESULTS.md section {SECTION})",
                    metrics={"status": f"PRE-REGISTERED PLAN_PARTS round 15 (docs/PARTS_RESULTS.md section {SECTION})",
                             "section_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "arm": "P_nsz: P_nom's flags + --bbnosize; object term retrained, target term reused from <cond>_tdc; seeds 10-12; "
                                    "builds full, nobottle, noplate and the S3 creations",
                             "reference": "P_nom, seeds 10-12 (pp_r14_roll)",
                             "metric": "bottle_hold_new: new-layout hold pooled over K4, K5, N4, N5, N6, seeds pooled; P_nsz - P_nom, "
                                       "2,000-draw state bootstrap (pp_round_read.py --tag r15 --metric bottle_hold_new --ref-s2 pp_r14_roll:P2nom)",
                             "round_verdict": "improves iff delta > 0.10 and the CI lower bound > 0",
                             "reported_no_verdict": "bottle_hold_new vs P_rtr (tag r15vrtr, ref pp_r13_roll:P2rtr); k5_hold_new vs P_nom (tag r15k5); "
                                                    "familiar non-regression vs P_nom ('regresses' iff CI upper bound < 0); per-pairing 3F; "
                                                    "pp_far_diag.py bb on seed 10",
                             "s1_restated": "S1 holds iff, with every target part's m fixed to v for each v in the inventory, both swap tests and "
                                            "every target-into-grasp component are exactly 0 for every seed",
                             "layer2_go": "pp_l2's mapping, unchanged (S1 as restated; S2 3F + OM in both layouts; S3 ratio >= 0.8, LB >= 0.6, "
                                          "both atoms, both layouts, k in {1, 5, 10})",
                             "states": "dev: N2 main states 4-23, N2 seeds",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "the round's S1 read and any of its rollouts"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_r15.py"))

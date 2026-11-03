"""PRE-REGISTRATION ROW of PLAN_PARTS round 13 (docs/PARTS_RESULTS.md section 41: role-level transport, per PLAN_PARTS 10.6).
Written after the round's builds and creations (training only), BEFORE its S1 read and any of its rollouts.  The Layer-2 GO
mapping is pp_l2's, unchanged, with S1 restated per PLAN_PARTS 10.5 (m held fixed at each inventory value).

    python experiments/prereg/pp_r13.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG, SECTION = "pp_r13", 41
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_r13_roll", "pp_r13_read.json", "pp_s1_mix_full_rtr_m0.json", "pp_s1_mix_full_rtr_m1.json", "pp_s1_mix_full_rtr_m2.json"):
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
         "experiments/fa1g_read.py", "experiments/pp_far_diag.py", "docs/PLAN_PARTS.md"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command=f"pre-registration: PLAN_PARTS round 13, role-level transport (docs/PARTS_RESULTS.md section {SECTION})",
                    metrics={"status": f"PRE-REGISTERED PLAN_PARTS round 13 (docs/PARTS_RESULTS.md section {SECTION})",
                             "section_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "arm": "P_rtr: P_imd's flags without --rclip, plus --rtransport; object term retrained, target term reused "
                                    "from <cond>_tdc; seeds 10-12; builds full, nobottle, noplate and the S3 creations",
                             "reference": "P_imd, seeds 10-12 (pp_r11_roll)",
                             "metric": "bottle_hold_new: new-layout hold pooled over K4, K5, N4, N5, N6, seeds pooled; P_rtr - P_imd, "
                                       "2,000-draw state bootstrap",
                             "round_verdict": "improves iff delta > 0.10 and the CI lower bound > 0",
                             "familiar_nonregression": "pooled familiar known-pairing success K1-K6, P_rtr - P_imd, same bootstrap; "
                                                       "'regresses' iff the CI upper bound < 0 (reported, no verdict)",
                             "s1_restated": "S1 holds iff, with every target part's m fixed to v for each v in the inventory, both swap tests and "
                                            "every target-into-grasp component are exactly 0 for every seed",
                             "layer2_go": "pp_l2's mapping, unchanged (S1 as restated; S2 3F + OM in both layouts; S3 ratio >= 0.8, LB >= 0.6, "
                                          "both atoms, both layouts, k in {1, 5, 10})",
                             "reported_no_verdict": "r_s per part and r_role per build; share of pre-hold calls served by the backbone alone "
                                                    "(pp_far_diag, seed 10)",
                             "states": "dev: N2 main states 4-23, N2 seeds",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "the round's S1 read and any of its rollouts"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_r13.py"))

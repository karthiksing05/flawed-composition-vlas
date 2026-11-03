"""PRE-REGISTRATION ROW of PLAN_PARTS round 10 (docs/PARTS_RESULTS.md section 35: one training per role).  Written after the round's
builds and creations (training only), BEFORE its checkpoint comparisons, its S1 read and any of its rollouts.  The round verdict is
section 35's (metric decouple; experiments/pp_decouple_check.py); the Layer-2 GO mapping is pp_l2's, unchanged.

    python experiments/prereg/pp_r10.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG, SECTION = "pp_r10", 35
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_r10_roll", "pp_r10_read.json", "pp_r10_decouple.json", "pp_s1_mix_full_tdc.json"):
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
         "experiments/fa1g_read.py", "experiments/pp_decouple_check.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command=f"pre-registration: PLAN_PARTS round 10, one training per role (docs/PARTS_RESULTS.md section {SECTION})",
                    metrics={"status": f"PRE-REGISTERED PLAN_PARTS round 10 (docs/PARTS_RESULTS.md section {SECTION})",
                             "section_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "arm": "P_tdc: P_ts's flags (rank 32, aframe, rclip, rtclip, psupport, tscale) + --decouple; seeds 10-12; "
                                    "builds full, nobottle, noplate and the S3 creations",
                             "reference": "P_pdc: P_pp's flags + --decouple, full, seeds 10-12 (same object-term design as P_tdc)",
                             "metric": "decouple",
                             "round_verdict": "(a) for each seed 10-12 every object-term tensor (obj.*, rclip, rclip_p, home) of full_pdc equals "
                                              "full_tdc's bitwise; (b) every paired S2 episode's hold flag agrees between P2pdc and P2tdc "
                                              "(all tasks, both layouts, seeds 10-12). decoupled iff (a) and (b); 'decoupled in training only' "
                                              "iff (a) and not (b); 'not decoupled' iff not (a). Success for the failure streak iff decoupled.",
                             "reported_no_verdict": "(c) repeat full_tdcr vs full_tdc seed 10, whole state bitwise; (d) reuse full_pdcu "
                                                    "(--reuse obj:full_tdc) vs full_pdc seed 10, whole state bitwise; (e) per-object factor "
                                                    "variance across seeds under P_tdc vs the between-arm range of rounds 6-9; (f) P_tdc GO read",
                             "layer2_go": "pp_l2's mapping, unchanged (S1 exact; S2 3F + OM in both layouts; S3 ratio >= 0.8, LB >= 0.6, "
                                          "both atoms, both layouts, k in {1, 5, 10})",
                             "states": "dev: N2 main states 4-23, N2 seeds",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "the round's checkpoint comparisons, its S1 read and any of its rollouts"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_r10.py"))

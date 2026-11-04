"""PRE-REGISTRATION ROW of the PLAN_PARTS consolidation phase (PLAN_PARTS 10.9; docs/PARTS_RESULTS.md section 52: leave-one-out
removal of P_nsz's stacked options under one keep/drop rule, then one combined minimal arm).  Written BEFORE any build, check,
S1 run, rollout or read of this phase.  The Layer-2 GO mapping is pp_l2's, unchanged, with S1 restated per PLAN_PARTS 10.5.

    python experiments/prereg/pp_cons.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG, SECTION = "pp_cons", 52
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_cons_roll", "pp_cons_rtclip_check.json", "pp_cons_rtc_read.json", "pp_cons_bbs_read.json", "pp_cons_nts_read.json",
          "pp_cons_naf_read.json", "pp_cons_nnm_read.json", "pp_cons_min_read.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PARTS_RESULTS.md"), encoding="utf-8").read()
m = re.search(rf"## {SECTION}\. Consolidation .*?(?=\n## {SECTION + 1}\.|\Z)", doc, flags=re.S)
assert m, f"section {SECTION} missing"
FILES = ["lifelong_ood/policy/parts_flow.py", "experiments/pp_p_train.py", "experiments/pp_p_roll.py", "experiments/pp_s1_mix.py",
         "experiments/pp_l2_read.py", "experiments/pp_l2r32_read.py", "experiments/pp_round_read.py", "experiments/pp_cons_read.py",
         "experiments/pp_cons_check.py", "experiments/pp_assemble.py", "experiments/fa1h_roll.py", "experiments/fa1g_read.py",
         "docs/PLAN_PARTS.md"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command=f"pre-registration: PLAN_PARTS consolidation (docs/PARTS_RESULTS.md section {SECTION})",
                    metrics={"status": f"PRE-REGISTERED PLAN_PARTS consolidation (docs/PARTS_RESULTS.md section {SECTION})",
                             "section_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "reference": "P_nsz: --rank 32 --aframe --rtclip --psupport --tscale --decouple --imode --rtransport --bbnom "
                                          "--bbnosize; seeds 10-12; pp_r15_roll (P2nsz, Pnsz_<cond>_k<k>); S1 pp_s1_mix_full_nsz_m0-2",
                             "ablations": {"rtc": "rtclip removed: exactness check pp_cons_check.py (no GPU); own metric n4_fam",
                                           "bbs": "bbnosize removed = P_nom (pp_r14_roll; re-read of existing rollouts); own metric bottle_hold_new",
                                           "nts": "tscale removed (target term retrained; object term reused from <cond>_nsz; full assembled "
                                                  "from full_pdc's target term after one bitwise verification build); own metric n4_fam",
                                           "naf": "aframe removed (object term retrained; target term reused); own metric s_known_new",
                                           "nnm": "bbnom removed (object term retrained; target term reused); own metric k5_hold_new"},
                             "not_ablated": "psupport, imode, rtransport, rank 32 (budget); decouple kept without ablation (measurement fix)",
                             "rule": "removing an option WORSENS iff (a) its own metric or (b) familiar non-regression (pooled familiar K1-K6 "
                                     "success) has delta = arm - P_nsz < -0.10 with paired-state-bootstrap CI upper bound < 0, or (c) a GO "
                                     "item passing for P_nsz fails for the arm (items: S1 restated; OM per layout; 3F per layout and pairing "
                                     "N1-N6; S3 per atom and layout, passing iff all k in {1, 5, 10} cells meet ratio >= 0.8, LB >= 0.6). "
                                     "DROP iff not worsens, else KEEP",
                             "dependencies": "rtclip's drop holds only with tscale on; if tscale is dropped, rtclip stays (it switches the "
                                             "per-part radial clip on). psupport stays in every arm",
                             "combined": "every DROP removed at once; assembled from leave-one-out terms where possible, else its object term "
                                         "built once; read with the same rule (own metrics of every dropped option)",
                             "tiebreak": "if the combined arm worsens: the single-option DROP arm with the most passing GO items, then higher "
                                         "pooled new-layout s new, then step-2 order (nts, naf, nnm); plus rtclip's exact drop where allowed; "
                                         "none: P_nsz",
                             "stop": "if the minimal arm passes Layer-2 GO on dev (pp_l2 mapping, both layouts): stop and report; no "
                                     "confirmatory read in this phase",
                             "s1_restated": "S1 holds iff, with every target part's m fixed to v for each v in the inventory, both swap tests and "
                                            "every target-into-grasp component are exactly 0 for every seed",
                             "states": "dev: N2 main states 4-23, N2 seeds; S3 k in {1, 5, 10}",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any build, check, S1 run, rollout or read of this phase"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_cons.py"))

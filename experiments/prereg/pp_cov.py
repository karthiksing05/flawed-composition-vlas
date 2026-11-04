"""PRE-REGISTRATION ROW of the coverage arm (PLAN_PARTS 10.11; docs/COVERAGE_RESULTS.md sections 1-2: W's generated demos added to
P_min's training data, a labelled ablation testing the condition-S account).  Written after P_cov's builds and creations and BEFORE
any S1 run, rollout or read of the arm, and before P_cov_h's builds.  The GO estimators are pp_l2's, unchanged (through
pp_cons_read.components); the mapping is COVERAGE_RESULTS 1.8, unchanged.

    python experiments/prereg/pp_cov.py
"""
import hashlib
import json
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_cov"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_cov_roll", "pp_cov_read.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(os.path.basename(p).startswith("pp_s1_mix_cov_") for p in os.listdir(F)), "an S1 file of the arm exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "COVERAGE_RESULTS.md"), encoding="utf-8").read()
m = re.search(r"## 1\. Design note.*?(?=\n## 3\.|\Z)", doc, flags=re.S)
assert m and "## 2. Pre-registration" in m.group(0), "sections 1-2 missing"
K = {json.load(open(os.path.join(F, f"pp_train_cov_full_s{s}.json"), encoding="utf-8"))["interface"]["K"] for s in (10, 11, 12)}
assert len(K) == 1, ("inventory size differs across seeds", K)
K = K.pop()
S1 = [f"pp_s1_mix_cov_full_m{v}.json" for v in range(K)]
gen = {t: json.load(open(os.path.join(F, f"pp_cov_gen_{t}.json"), encoding="utf-8"))["sha256"] for t in ("K1", "K2", "K3", "K4", "K5", "K6")}
FILES = ["lifelong_ood/policy/parts_flow.py", "experiments/pp_p_train.py", "experiments/pp_p_roll.py", "experiments/fa1h_roll.py",
         "experiments/pp_s1_mix.py", "experiments/pp_l2_read.py", "experiments/pp_l2r32_read.py", "experiments/pp_round_read.py",
         "experiments/pp_cons_read.py", "experiments/pp_cov_gen.py", "experiments/pp_cov_train.py", "experiments/pp_cov_read.py",
         "experiments/pp_n4d.py", "experiments/fa1g_plan.py", "docs/PLAN_PARTS.md"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: coverage arm (docs/COVERAGE_RESULTS.md sections 1-2; PLAN_PARTS 10.11)",
                    metrics={"status": "PRE-REGISTERED coverage arm (docs/COVERAGE_RESULTS.md sections 1-2)",
                             "section_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "labelled": "generated-data ablation (PLAN_PARTS 4.1), post hoc (PLAN_PARTS 10.11); not the structural claim; "
                                         "does not change the main arm's GO",
                             "arm": "P_cov: P_min's flags (--rank 32 --psupport --tscale --decouple --imode --rtransport --bbnom --bbnosize), "
                                    "human demos + W's generated demos (50 per K1-K6 task), seeds 10-12, both roles trained; S3 created from "
                                    "k in {1, 5, 10} human demos only",
                             "reference": "P_min: pp_cons_roll (P2naf, Pnaf_<cond>_k<k>); S1 pp_s1_mix_full_naf_m0-2",
                             "candidate": {"s2": "pp_cov_roll:P2cov", "s3": "pp_cov_roll:Pcov", "s1": S1},
                             "primary": "P_cov's verdict is the arm's verdict; P_cov_h never replaces it",
                             "secondary": {"arm": "P_cov_h: as P_cov, but the grasp-mode inventory and each target's m fitted on the human rows "
                                                  "only (P_min's interface); object term retrained, target term reused from cov_<cond>; added "
                                                  "after seeing the builds' fitted interfaces, before any read",
                                           "s2": "pp_cov_roll:P2covh", "s3": "pp_cov_roll:Pcovh",
                                           "s1": [f"pp_s1_mix_covh_full_m{v}.json" for v in range(3)], "mapping": "the same five clauses",
                                           "output": "pp_cov_h_read.json"},
                             "mapping": "SUPPORTED iff (a) S1 restated exact; (b) new-layout 3F passes on every pairing N1-N6; (c) new-layout "
                                        "OM passes; (d) new-layout S3 passes for each atom (all k in {1, 5, 10}: ratio >= 0.8, LB >= 0.6); "
                                        "(e) every familiar GO item passing for P_min passes, and pooled familiar K1-K6 success delta "
                                        "(P_cov - P_min, paired state bootstrap) not < -0.10 with CI upper < 0. Else NOT SUPPORTED",
                             "prediction": "NOT SUPPORTED, failing (d) for the plate only; s known new >= 0.65; familiar within 0.05",
                             "states": "dev only: N2 main states 4-23, N2 seeds; no confirmatory or sealed read",
                             "generated_sha256": gen,
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any S1 run, rollout or read of this arm"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_cov.py"))

"""PRE-REGISTRATION ROW of ladder rung L4 (PLAN_PARTS 5, 10.10 and 10.12; docs/LADDER_RESULTS.md sections 5-6: the given role assignment
replaced by roles induced from contact events, on top of L1).  Written BEFORE any L4 rollout, S1 run or read.

    python experiments/prereg/pp_L4.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_L4"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_L4_roll", "pp_ladder_L4_read.json", "pp_L4_smoke_rollagree.json", "pp_s1_mix_full_L4_m0.json", "pp_s1_mix_full_L4_m1.json",
          "pp_s1_mix_full_L4_m2.json", "pp_s1_mix_full_L4_m3.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "LADDER_RESULTS.md"), encoding="utf-8").read()
m = re.search(r"## 5\. L4 design note.*?(?=\n## 7\.|\Z)", doc, flags=re.S)
assert m and "## 6. L4 protocol" in m.group(0), "sections 5-6 missing"
FILES = ["lifelong_ood/policy/parts_flow.py", "experiments/pp_p_train.py", "experiments/pp_p_roll.py", "experiments/fa1h_roll.py",
         "experiments/pp_s1_mix.py", "experiments/pp_l2_read.py", "experiments/pp_l2r32_read.py", "experiments/pp_round_read.py",
         "experiments/pp_ladder_read.py", "experiments/pp_L4_smoke.py", "experiments/fa1g_read.py", "docs/PLAN_PARTS.md",
         "results/flawed/pp_L4_smoke_roles.json"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: ladder rung L4 (docs/LADDER_RESULTS.md sections 5-6)",
                    metrics={"status": "PRE-REGISTERED ladder rung L4 (docs/LADDER_RESULTS.md sections 5-6)",
                             "sections_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "L0": "P_min: pp_cons_roll P2naf / Pnaf_<cond>_k<k>; S1 pp_s1_mix_full_naf_m0-2",
                             "arm": "L1's flags + --iroles (--rank 32 --rtclip --psupport --tscale --decouple --imode --rtransport --bbnom "
                                    "--bbnosize --pgate --iroles); object term retrained, target term reused from <cond>_L1 under the "
                                    "asserted precondition; creations k = 1, 5, 10; rollouts pp_L4_roll P2L4 / PL4_<cond>_k<k>; S1 "
                                    "pp_s1_mix_full_L4_m<v> for every v of the inventory",
                             "replaced": "FC.roles (template parse + noun table) in training rows, creation and the rollout, by the lexicon of "
                                         "LADDER_RESULTS 5.2; a task trains iff it has a contact event and an induced parse. The read's "
                                         "n2_tasks roles stay the instrument",
                             "s1": "holds iff the model's swap tests and target-into-grasp components are exactly 0 with m fixed at every "
                                   "inventory value on every seed AND parse locality holds on K1-K6, N1-N6 with the full lexicon",
                             "measures": "pp_ladder_read.py --rung L4 (as pp_L1): OM, 3F D3 N1-N6, S3 per atom (mean over k in {1, 5, 10}); "
                                         "delta = L4 - L0, 2,000-draw paired state bootstrap",
                             "counted": "familiar only: S1, OM, 3F N1 N2 N3 N4 N6 (N5 excluded: L0 fails it), S3 bottle, S3 plate",
                             "rule": "a counted measure BREAKS iff |delta| > 0.10 and its CI excludes 0; L4 HOLDS iff S1 holds and no counted "
                                     "measure breaks, else L4 BREAKS (broken measures named); new layout and competence reported with no verdict",
                             "stop": "L4 breaks: ladder stops for the user's decision. L4 holds: stop and report; L5 and L2/L3 are not started "
                                     "here. Cap: the track's 15 A40-h",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any L4 rollout, S1 run or read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_L4.py"))

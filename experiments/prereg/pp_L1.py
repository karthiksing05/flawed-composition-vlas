"""PRE-REGISTRATION ROW of ladder rung L1 (PLAN_PARTS 5 and 10.10; docs/LADDER_RESULTS.md sections 1-2: the sim-contact phase gate
replaced by a gripper-width + proprioception event).  Written BEFORE any L1 rollout, S1 run or read.

    python experiments/prereg/pp_L1.py
"""
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_L1"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_L1_roll", "pp_ladder_L1_read.json", "pp_L1_smoke_rollagree.json", "pp_s1_mix_full_L1_m0.json", "pp_s1_mix_full_L1_m1.json",
          "pp_s1_mix_full_L1_m2.json", "pp_s1_mix_full_L1_m3.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "LADDER_RESULTS.md"), encoding="utf-8").read()
m = re.search(r"## 1\. L1 design note.*?(?=\n## 3\.|\Z)", doc, flags=re.S)
assert m and "## 2. L1 protocol" in m.group(0), "sections 1-2 missing"
FILES = ["lifelong_ood/policy/parts_flow.py", "experiments/pp_p_train.py", "experiments/pp_p_roll.py", "experiments/fa1h_roll.py",
         "experiments/pp_s1_mix.py", "experiments/pp_l2_read.py", "experiments/pp_l2r32_read.py", "experiments/pp_round_read.py",
         "experiments/pp_ladder_read.py", "experiments/pp_L1_smoke.py", "experiments/fa1g_read.py", "docs/PLAN_PARTS.md"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: ladder rung L1 (docs/LADDER_RESULTS.md sections 1-2)",
                    metrics={"status": "PRE-REGISTERED ladder rung L1 (docs/LADDER_RESULTS.md sections 1-2)",
                             "sections_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "L0": "P_min = A_naf: pp_{full,nobottle,noplate}_naf_s10-12, created P_<atom>_k{1,5,10}; pp_cons_roll P2naf / "
                                   "Pnaf_<cond>_k<k>; S1 pp_s1_mix_full_naf_m0-2; read pp_cons_min_read.json",
                             "arm": "--rank 32 --psupport --tscale --decouple --imode --rtransport --bbnom --bbnosize --pgate; both roles "
                                    "rebuilt (no reuse: the held labels split both roles' rows); creations k = 1, 5, 10; rollouts "
                                    "pp_L1_roll P2L1 / PL1_<cond>_k<k>; S1 pp_s1_mix_full_L1_m<v> for every v of the inventory",
                             "replaced": "sim-contact held (fa1b_common.Held) in training rows and the policy, by ProprioHeld (finger width, "
                                         "last gripper command, hand height): on when closing, width >= tau_w and the hand rose LIFT = "
                                         "0.01 m above its lowest height since the close began; off when the command opens or width < "
                                         "tau_w; tau_w = (min width + 1st pct of close-commanded width) / 2 over the build's demo rows; "
                                         "creation uses the library's tau_w. The read's instrument (held_step, min_dxy_held) stays the "
                                         "oracle signal",
                             "measures": "per layout: S1 (restated, m fixed); OM; 3F D3 per new pairing N1-N6; S3 per atom = mean over k "
                                         "in {1, 5, 10} of the transfer ratio; delta = L1 - L0 with a 2,000-draw paired state bootstrap "
                                         "(seeds pooled, rng seed 0)",
                             "counted": "familiar layout only: S1, OM, 3F N1 N2 N3 N4 N6 (N5 excluded: L0 fails it), S3 bottle, S3 plate",
                             "rule": "a counted measure BREAKS iff |delta| > 0.10 and its CI excludes 0; L1 HOLDS iff S1 holds and no "
                                     "counted measure breaks, else L1 BREAKS (broken measures named); new layout and competence "
                                     "reported with no verdict",
                             "stop": "L1 breaks: ladder stops for the user's decision; L1 holds: L2 designed next (sources checked first; "
                                     "write up options if it needs large new infrastructure or a plan-level choice); 15 A40-h GPU cap",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any L1 rollout, S1 run or read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_L1.py"))

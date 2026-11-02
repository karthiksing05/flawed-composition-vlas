"""PRE-REGISTRATION ROW of the PLAN_PARTS Layer-2 CONFIRMATORY read with rank-32 parts (docs/PARTS_RESULTS.md section 6).

POST-HOC RE-REGISTRATION.  Rank 32 was chosen AFTER seeing pp_l2's rank-8 (primary) and rank-32 (secondary) results on the
development data (seeds 0-2, N2 states 4-23).  The fresh-seed read (user approval 2026-11-01, this read only) is the plan's S-conf
(section 6: one fresh-seed read with no changes, 50 states).  Written after the builds and creations (training only), BEFORE the S1
read and before any rollout of this read.

    python experiments/prereg/pp_l2r32.py
"""
import hashlib
import json
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_s2c_roll", "pp_s3c_roll", "pp_s4c_old", "pp_l2r32_read.json", "pp_s1_mix_full_r32c.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert any(r.get("tag") == "pp_l2" for r in ledger.load("m3_prereg")), "pp_l2 missing"
assert not any(r.get("tag") == "pp_l2r32" for r in ledger.load("m3_prereg")), "pp_l2r32 already registered"
conf = json.load(open(os.path.join(F, "pp_conf_states.json"), encoding="utf-8"))
assert conf["n"] == 50 and conf["min_abs_diff_to_n2_states"] > 0


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PARTS_RESULTS.md"), encoding="utf-8").read()
m = re.search(r"## 6\. Layer 2 confirmatory protocol.*?(?=\n## 7\.|\Z)", doc, flags=re.S)
assert m, "section 6 missing"
FILES = ["lifelong_ood/policy/parts_flow.py", "experiments/pp_p_train.py", "experiments/pp_baselines.py", "experiments/pp_p_roll.py",
         "experiments/pp_s1_mix.py", "experiments/pp_l2_read.py", "experiments/pp_l2r32_read.py", "experiments/fa1h_roll.py",
         "experiments/fa1g_read.py", "experiments/pp_conf_states.py", "results/flawed/pp_conf_states.json",
         "results/flawed/pp_conf_branches.json"]
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_l2r32", seed=10,
                    command="pre-registration: PLAN_PARTS Layer 2 confirmatory read, rank-32 parts (POST-HOC re-registration)",
                    metrics={"status": "PRE-REGISTERED PLAN_PARTS Layer 2 confirmatory read (docs/PARTS_RESULTS.md section 6); "
                                       "POST-HOC: rank 32 chosen after seeing pp_l2's rank-8 and rank-32 development results",
                             "section_6_protocol_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "architecture": "P with rank-32 parts (parts_flow.PartsFlow rank=32); build recipe and creation recipe of pp_l2 "
                                             "(30,000 steps AdamW 3e-4 batch 512 EMA 0.999; creation 4,000 steps AdamW 3e-4 batch 256 no EMA)",
                             "held_out": "bottle (obj, created from K4), plate (tgt, created from K1); k in {1, 5, 10, 50}",
                             "seeds": [10, 11, 12],
                             "states": "pp_conf_states: 50 fresh start states per layout (familiar seeds 9100000+k, new 9200000+k, noise "
                                       "23000000+1000k), never used by any earlier read",
                             "S1": "pp_l2's: holds iff swap tests exactly 0 (every seed) and every tg component exactly 0 on every frame",
                             "S2": "pp_l2's: holds iff in both layouts 3F (every N1-N6 D3 CI contains 0) and object-matched ratio >= 0.8 "
                                   "with CI lower bound >= 0.6",
                             "S3_layer3_go": "pp_l2's: S1 and S2 hold and, for bottle and plate, both layouts, every k in {1, 5, 10}: "
                                             "transfer ratio >= 0.8 with CI lower bound >= 0.6",
                             "S4": "descriptive: HF, HL, RF, LIB exactly as pp_l2 (rank 8), at seeds 10-12 on the same fresh states",
                             "multiplicity": "95% CIs as pp_l2; Layer-3 GO is an intersection-union test (every cell must pass), so no "
                                             "Holm adjustment is applied",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "the S1 read and any rollout of this read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_l2r32.py"))

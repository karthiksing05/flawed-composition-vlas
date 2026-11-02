"""PRE-REGISTRATION ROW of one PLAN_PARTS improvement round (docs/PARTS_RESULTS.md, the round's design-note section).  Written after
the round's builds and creations (training only), BEFORE its S1 read and any of its rollouts.  The Layer-2 GO mapping is pp_l2's,
unchanged; the round metric is pp_round_read.py's (improves iff delta > 0.10 with state-bootstrap CI lower bound > 0).

    python experiments/prereg/pp_round.py --tag pp_r2 --section 12 --metric n4_fam --outdir pp_r2_roll \
        --arm "P_sz: rank 32, target term reads the held object's size; seeds 10-12" --ref "P rank 32, seeds 10-12"
"""
import argparse
import hashlib
import os
import re
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

ap = argparse.ArgumentParser()
for k in ("--tag", "--section", "--metric", "--outdir", "--arm", "--ref"):
    ap.add_argument(k, required=True)
ap.add_argument("--files", default="")
a = ap.parse_args()
F = os.path.join(ROOT, "results", "flawed")
short = a.tag.replace("pp_", "")
for x in (a.outdir, f"pp_{short}_read.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == a.tag for r in ledger.load("m3_prereg")), f"{a.tag} already registered"


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "PARTS_RESULTS.md"), encoding="utf-8").read()
m = re.search(rf"## {a.section}\. Round .*?(?=\n## {int(a.section) + 1}\.|\Z)", doc, flags=re.S)
assert m, f"section {a.section} missing"
FILES = ["lifelong_ood/policy/parts_flow.py", "experiments/pp_p_train.py", "experiments/pp_p_roll.py", "experiments/pp_s1_mix.py",
         "experiments/pp_l2_read.py", "experiments/pp_l2r32_read.py", "experiments/pp_round_read.py", "experiments/fa1h_roll.py",
         "experiments/fa1g_read.py"] + [x for x in a.files.split("+") if x]
print(ledger.record("m3_prereg", bench="libero_goal", tag=a.tag, seed=0,
                    command=f"pre-registration: PLAN_PARTS improvement round {a.tag} (docs/PARTS_RESULTS.md section {a.section})",
                    metrics={"status": f"PRE-REGISTERED PLAN_PARTS improvement round {a.tag} (docs/PARTS_RESULTS.md section {a.section})",
                             "section_sha256": hashlib.sha256(m.group(0).encode("utf-8")).hexdigest(),
                             "arm": a.arm, "reference": a.ref, "metric": a.metric,
                             "round_verdict": "improves iff candidate - reference > 0.10 with state-bootstrap CI lower bound > 0",
                             "layer2_go": "pp_l2's mapping, unchanged (S1 exact; S2 3F + OM in both layouts; S3 ratio >= 0.8, LB >= 0.6, "
                                          "both atoms, both layouts, k in {1, 5, 10})",
                             "states": "dev: N2 main states 4-23, N2 seeds",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "the round's S1 read and any of its rollouts"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_round.py"))

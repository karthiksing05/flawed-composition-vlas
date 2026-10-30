"""PRE-REGISTRATION ROW of N1 (docs/FLAWED_COMPOSITION.md section 4.9): input novelty against compound novelty as explanations of
per-task held-out failure.  Written after the scene renders and the novelty measures (results/flawed/n1_novelty.json, computed
without any outcome) and BEFORE any outcome is joined to them (asserted: no n1_read.json exists).

    python experiments/prereg/pn1.py
"""
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

F = os.path.join(ROOT, "results", "flawed")
assert not os.path.exists(os.path.join(F, "n1_read.json")), "the N1 read exists"
assert not any(r.get("tag") == "pn1" for r in ledger.load("m3_prereg")), "pn1 already registered"
nov = json.load(open(os.path.join(F, "n1_novelty.json"), encoding="utf-8"))


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


print(ledger.record("m3_prereg", bench="libero_90_c", tag="pn1", seed=0, command="pre-registration: FLAWED_COMPOSITION N1",
                    metrics={"status": "PRE-REGISTERED N1 (FLAWED_COMPOSITION 4.9)",
                             "novelty": {b: {k: {c: r.get(c) for c in ("d_a", "d_b", "d_c", "d_d", "input_composite", "score", "u")}
                                             for k, r in D.items()} for b, D in nov.items()},
                             "predictions": {
                                 "N1a": "pooled stratified Spearman(input composite, failure) > 0 with task-identity bootstrap 95% CI "
                                        "excluding 0; falsified otherwise",
                                 "N1b": "partial(input | score, u, block) CI excludes 0 (> 0) AND partial(compound | input, block) CI includes "
                                        "0 -> 'where, not what'; falsified if the first CI includes 0; both exclude 0 -> both matter",
                                 "N1c": "main2 / C1: Spearman(D1 per-task dev excess, input composite) > 0 with CI excluding 0"},
                             "secondary": "components d_a/d_b/d_c (d_d BindVLA); pool without BindVLA blocks; per block; BindVLA continuous "
                                          "outcomes: named-contact failure and closest approach to the named object",
                             "sealed": "sealed LIBERO-90-C task identities were inferred from split.json's salted hashes and used only to exclude",
                             "deviation": "Li's tasks use set-a practice start states, not the scoring set e",
                             "files_sha256": {p: fsha(p) for p in ["docs/FLAWED_COMPOSITION.md", "experiments/fc_n1_tasklist.py",
                                                                   "experiments/fc_n1_render.py", "experiments/fc_n1_novelty.py",
                                                                   "experiments/fc_n1_read.py", "experiments/fc_d2_read.py",
                                                                   "results/flawed/n1_tasks.json", "results/flawed/n1_novelty.json"]},
                             "written_before": "any outcome joined to the novelty measures"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pn1.py"))

"""PRE-REGISTRATION ROW of T37-L (SCAFFOLD_RESULTS 43): R2's checkpoint served with LEARNED masks (sc_seg.py serve: the segmenter's
pred() on the client's renders paints the role views; simulator maps used only for an IoU log).  BEFORE any r2l rollout.

    python experiments/prereg/sc_t37l.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "sc_t37l" for r in rows), "already registered"
for tid in range(32):
    assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t37", f"r2l_main_{tid}.json")), tid


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t37l", seed=0, command="pre-registration: T37-L (docs/SCAFFOLD_RESULTS.md 43)",
                    metrics={"status": "PRE-REGISTERED T37-L",
                             "mapping": "sc_t34.py read --dir sc_t37 --rungs r0 r2 r2l --primary r2l:r2 --tag l; N1-N6 familiar. MASKS HOLD iff "
                                        "r2l - r2 CI lower bound >= -.10; MASKS COST iff upper bound < -.10; else UNRESOLVED. Also reported: r2l - r0 "
                                        "with sc_t34's labels (STRUCTURE WINS iff lower bound > +.10), and the online IoU logs.",
                             "prediction": "UNRESOLVED (50%), MASKS HOLD (35%), MASKS COST (15%): agentview masks are good (.97+ on held-out "
                                           "frames) but wrist masks of the bottle are poor (.68 new layout; .10 in a smoke rollout), and R2 paints "
                                           "both cameras. r2l - r0: STRUCTURE WINS (point near +.30).",
                             "seen": "SR 37.1 (R0 .025, R2 .425) and the segmenter's IoU tables and smokes (SR 40).",
                             "files_sha256": {p: fsha(p) for p in ["experiments/sc_seg.py", "experiments/sc_rv.py", "experiments/sc_t34.py"]},
                             "written_before": "any r2l rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t37l.py"))

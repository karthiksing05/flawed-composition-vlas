"""PRE-REGISTRATION ROW of Q2-N (SCAFFOLD_RESULTS 42): the operator parses one demo of each NEW pairing (N1-N6) on each ladder rung.
BEFORE any sc_q2_demo.py run.

    python experiments/prereg/sc_q2n.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "sc_q2n" for r in rows), "already registered"
for r in ("r0", "r1", "r2", "r2b"):
    assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_q2", r, "demo_new.json")), r


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


REC = "/coc/flash10/ksingara3/robot/data/fa1/pu/rec_new"
dem = {}
for k in range(1, 7):
    p = os.path.join(REC, "N%d_familiar_s10_st24.npz" % k)
    with open(p, "rb") as fh:
        dem[p] = hashlib.sha256(fh.read()).hexdigest()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_q2n", seed=0, command="pre-registration: Q2-N (docs/SCAFFOLD_RESULTS.md 42)",
                    metrics={"status": "PRE-REGISTERED Q2-N",
                             "mapping": "Per rung, at its picked Q2 setting (results/flawed/sc_q2/<R>/picked.json), over the 6 new-pairing demos: "
                                        "joint correct count J (object and target phrase both right). PARSES NEW iff J >= 5; FAILS NEW iff J <= 2; "
                                        "else MIXED. The flow-loss baseline's count on the same rows is reported beside it, with no verdict (6 demos "
                                        "cannot separate the two). r0's candidates are the six trained K1-K6 instructions, so its joint is 0 by "
                                        "construction; its object and target matches are reported only.",
                             "prediction": "r1, r2, r2b: MIXED or PARSES NEW (J 3-6), equal to flow loss within one demo; the cheese demos "
                                           "(N1-N3) parse their target worse than the bottle demos if the rung stores the pair (P1). r0: object "
                                           "match on about half (the instruction sharing the object), joint 0.",
                             "seen": "The six demos were recorded by U (oracle) from extra state 24 and succeeded on the first attempt (74-94 steps); "
                                     "their frames were not inspected. Q2's search/read and every T37 rollout are unread at registration.",
                             "demos_sha256": dem,
                             "files_sha256": {p: fsha(p) for p in ["experiments/sc_q2_demo.py", "experiments/sc_q2.py", "experiments/sc_pu_record.py",
                                                                   "experiments/fc_n2_extra.py"]},
                             "written_before": "any Q2-N run"}, bytes=Z, config={}, script="experiments/prereg/sc_q2n.py"))

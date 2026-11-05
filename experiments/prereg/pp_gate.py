"""PRE-REGISTRATION ROW of the hold gate (docs/HOLD_GATE.md).  After the instruments' smoke and BEFORE any main run or read.

    python experiments/prereg/pp_gate.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_gate"
F = os.path.join(ROOT, "results", "flawed")
for x in ("pp_gate", "pp_gate_read.json"):
    assert not os.path.exists(os.path.join(F, x)), f"{x} exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/HOLD_GATE.md", "experiments/pp_atlas.py", "experiments/pp_gate_client.py", "experiments/pp_gate_roll.sh", "experiments/pp_gate_read.py", "experiments/libero_client.py", "experiments/fc_n2_client.py", "experiments/pp_serve_purify.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: the hold gate (docs/HOLD_GATE.md)",
                    metrics={"status": "PRE-REGISTERED the hold gate",
                             "labelled": "post hoc; development read; no training; oracle signal; changes no earlier verdict; not reviewed by the user, who is away and asked to keep iterating",
                             "mapping": "per model, familiar, pooled N1-N6, success: GATE RECOVERS iff s(G) - s(intact) CI lower > 0 and > 0.10; PROMPT ADDS iff s(GA) - s(G) CI lower > 0 and > 0.10. "
                                        "yes/yes TIMING AND BINDING; yes/no TIMING; no/yes BINDING NEEDS THE GATE; no/no NOT TIMING. Known tasks losing > 0.05 under G: GATE DAMAGES KNOWN TASKS",
                             "prediction": "pi0.5 TIMING; GR00T NOT TIMING",
                             "states": "dev only: N2 main states 4-23, N2 seeds; no confirmatory or sealed read; demos never executed",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any main gate run or read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_gate.py"))

"""PRE-REGISTRATION ROW of the break atlas sensitivity read (docs/BREAK_ATLAS.md 6).  After the frozen read and BEFORE this read.

    python experiments/prereg/pp_atlas_sens.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

TAG = "pp_atlas_sens"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "pp_atlas_sens.json")), "read exists"
assert not any(r.get("tag") == TAG for r in ledger.load("m3_prereg")), f"{TAG} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/BREAK_ATLAS.md", "experiments/pp_atlas.py", "experiments/pp_atlas_sens.py"]
print(ledger.record("m3_prereg", bench="libero_goal", tag=TAG, seed=0,
                    command="pre-registration: break atlas sensitivity read (docs/BREAK_ATLAS.md 6)",
                    metrics={"status": "PRE-REGISTERED the atlas sensitivity read",
                             "labelled": "post hoc, after seeing the frozen read; development read; no new episodes; the frozen verdict stays on record",
                             "variants": "S1: hold = Held, or after onset a6 > 0 and ee-object gap <= the episode's median held gap + 0.02. "
                                         "S2: a frozen HANDOFF or TRANSPORT whose object ends within 0.10 m becomes RELEASE",
                             "mapping": "per VLA and layout: ROBUST iff S1's and S2's verdicts equal the frozen verdict under the frozen E rule, else INSTRUMENT-DEPENDENT",
                             "prediction": "pi0.5 familiar INSTRUMENT-DEPENDENT, with S2 giving WITHIN-CHUNK and S1 no change; GR00T familiar ROBUST; both new-layout rows ROBUST",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "this read"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pp_atlas_sens.py"))

"""PRE-REGISTRATION ROW of T34 (docs/SCAFFOLD_RESULTS.md 34).  BEFORE any rollout.

    python experiments/prereg/sc_t34.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t34_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t34" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t34", seed=0, command="pre-registration: T34 (docs/SCAFFOLD_RESULTS.md 34)",
                    metrics={"status": "PRE-REGISTERED T34",
                             "labelled": "development read on N2 main states 4-23; one seed per rung; pi0.5 with frozen SigLIP and PaliGemma LLM, action expert trained, 6000 steps, batch 32; R0 whole instructions, R1 role phrases switched by ProprioHeld",
                             "mapping": "Fit gate: R0 mean success on K1-K6 familiar under .80 -> UNDERFIT. R1 - R0, paired bootstrap over tasks and states (2000 draws, seed 0). New pairings N1-N6, familiar: STRUCTURE WINS iff CI lower > +.10; STRUCTURE LOSES iff CI upper < -.10; else NO CLEAR DIFFERENCE. Known K1-K6, familiar: NO COST iff CI lower >= -.10; else COST",
                             "prediction": "NO CLEAR DIFFERENCE and NO COST. R0 near .45 on new pairings, with N1 and N2 at or under .25 and N4, N5 above .70. R1 within .10 of R0. Both at or above .90 on known",
                             "states": "dev only: N2 main states 4-23",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_t34.py", "experiments/sc_t16.py", "experiments/fc_n2_client.py", "lifelong_ood/policy/l90c_config.py", "experiments/c_train_pi05.py"]},
                             "written_before": "any T34 rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t34.py"))

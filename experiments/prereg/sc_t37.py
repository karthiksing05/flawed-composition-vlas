"""PRE-REGISTRATION ROW of T37, the matched ladder on re-rendered frames (docs/SCAFFOLD_RESULTS.md 37).  BEFORE any rollout.

    python experiments/prereg/sc_t37.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t37_read.json")), "read exists"
assert not any(r.get("tag") == "sc_t37" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
FILES = ["docs/SCAFFOLD_RESULTS.md", "docs/THEORY_COMPOSITION.md", "experiments/sc_t34.py", "experiments/sc_rv.py", "experiments/sc_t16.py",
         "experiments/fc_n2_client.py", "lifelong_ood/policy/l90c_config.py", "experiments/c_train_pi05.py",
         "results/flawed/sc_rv_plain_manifest.json", "results/flawed/sc_rv_rsup_manifest.json"]
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t37", seed=0, command="pre-registration: T37 matched ladder (docs/SCAFFOLD_RESULTS.md 37)",
                    metrics={"status": "PRE-REGISTERED T37",
                             "labelled": "development read on N2 main states 4-23; one seed per rung; pi0.5 with frozen SigLIP and PaliGemma LLM, action expert trained, 6000 steps, batch 32, on 52,181 frames re-rendered from all 400 raw demos of K1-K8 (experiments/sc_rv.py). r0 whole instructions; r1 given role phrases switched by ProprioHeld; r2 r1 plus every movable object except the active role's filler painted magenta by its exact mask, fixtures never painted; r2b the same with bounding rectangles. Masks come from the simulator at test time (supervised stage). Read by experiments/sc_t34.py read --dir sc_t37 --rungs r0 r1 r2 r2b --primary r2:r0",
                             "mapping": "Fit gate: r0 mean success on K1-K6 familiar under .80 -> UNDERFIT. Each difference: paired bootstrap over tasks and states (2000 draws, seed 0), new pairings N1-N6, familiar: STRUCTURE WINS iff CI lower > +.10; STRUCTURE LOSES iff CI upper < -.10; else NO CLEAR DIFFERENCE. Primary: r2 - r0. Also r2 - r1, r1 - r0, r2b - r2 with the same labels. Cost on K1-K6 familiar for r1, r2, r2b against r0: NO COST iff CI lower >= -.10; else COST",
                             "prediction": "r0 near .45 on new pairings and at or above .90 on known. r1 - r0: NO CLEAR DIFFERENCE. r2 - r0: STRUCTURE WINS with r2 near .70, held at a little better than even. r2b - r2: NO CLEAR DIFFERENCE, r2b a few points higher. NO COST for r1; r2 and r2b lose under .10 on known",
                             "against_the_theory": "r2 and r2b both within .10 of r0 while r0 passes the fit gate",
                             "seen_before_registration": "smoke rollouts on smoke states with an unrelated 500-step checkpoint to test the servers; contact sheets of the painted frames; a first painting rule (dataset rview) was built and dropped before any training. No rung of this ladder has been rolled out",
                             "states": "dev only: N2 main states 4-23",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any T37 rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t37.py"))

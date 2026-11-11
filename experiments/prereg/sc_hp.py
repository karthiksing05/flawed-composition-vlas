"""PRE-REGISTRATION ROW of HP (SCAFFOLD_RESULTS 44): the hand-over hybrid with no simulator state at test time reaching either
policy's decisions beyond those listed in SR 44 (depth render, camera matrices, entity list).  Frozen VLA (pi0.5, GR00T) from the start;
at U's first ProprioHeld onset control passes to U (pp_sc_U_s10) reading poses estimated from LEARNED masks + depth (sc_seg pred,
sc_pu Estimator).  BEFORE any main-state HP rollout.

    python experiments/prereg/sc_hp.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "sc_hp" for r in rows), "already registered"
od = os.path.join(ROOT, "results", "flawed", "sc_hp")
assert not os.path.isdir(od) or not os.listdir(od), "main HP results exist"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_hp", seed=0, command="pre-registration: HP (docs/SCAFFOLD_RESULTS.md 44)",
                    metrics={"status": "PRE-REGISTERED HP",
                             "mapping": "experiments/sc_hp_read.py; per model, N1-N6 familiar, N2 main states, bootstrap over tasks and states "
                                        "(2,000 draws). (1) PRIMARY HP - intact VLA: BEATS THE VLA iff CI lower bound > +.10; else NOT SHOWN. "
                                        "(2) HP - H (the simulator-state hand-over): NO PRIVILEGE COST iff lower bound >= -.10; PRIVILEGE COST iff "
                                        "upper bound < -.10; else UNRESOLVED. (3) Known K1,K2,K3,K4,K6 familiar, HP - intact: NO KNOWN COST iff "
                                        "lower bound >= -.10. Reported with no verdict: the new layout, hand-over share, early hand-overs (trigger "
                                        "before the oracle's first hold), HP - U alone, break E per arm, pose errors.",
                             "prediction": "pi0.5: BEATS THE VLA (HP near .88 vs .50), NO PRIVILEGE COST, NO KNOWN COST. GR00T: BEATS THE VLA "
                                           "(HP near .80 vs .17); HP - H UNRESOLVED, from the bottle's 3-4 cm held-object error in rigid mode. "
                                           "Early hand-overs on the cheese in at least a third of its episodes.",
                             "seen": "Smokes on smoke states 0-1, N1 and N4 familiar: pi0.5 4/4, GR00T 2/4; ProprioHeld fired early on failed "
                                     "cheese lifts in 3 of 4 cheese episodes, and no debounce was added after seeing that. sc_hp_read --selftest "
                                     "reproduces HAND_OVER 6.1. Known: H .96 / .93, intact .50 / .17, U on learned masks .881 (SR 40.1), SR 37.1.",
                             "files_sha256": {p: fsha(p) for p in ["experiments/sc_hp.py", "experiments/sc_hp_read.py", "experiments/sc_seg.py",
                                                                   "experiments/sc_pu.py", "results/flawed/sc_pu_calib.json"]},
                             "written_before": "any main-state HP rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_hp.py"))

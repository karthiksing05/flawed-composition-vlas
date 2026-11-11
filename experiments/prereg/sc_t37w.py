"""PRE-REGISTRATION ROW of T37w (SCAFFOLD_RESULTS 41): the AC-VLA-style control for the matched ladder.  R2w = R1's role phrases +
the WHOLE wrist image filled with grey 128 on held frames (experiments/sc_rv_w.py, arm rwrist; config pi05_g8v_rwrist_ae, the same
recipe as every T37 rung).  Two test-time variants of one checkpoint: r2w (filled at test, as trained) and r2w0 (served unfilled:
AC-VLA's rule masks at training only).  BEFORE the R2w training starts and before any R2w rollout.

    python experiments/prereg/sc_t37w.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "sc_t37w" for r in rows), "already registered"
for nm in ("r2w", "r2w0"):
    for tid in range(32):
        assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_t37", f"{nm}_main_{tid}.json")), (nm, tid)
assert not os.path.isdir("/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_rwrist_ae/tl"), "R2w training already started"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["docs/SCAFFOLD_RESULTS.md", "experiments/sc_rv_w.py", "experiments/c_train_pi05_w.py", "experiments/sc_rv.py", "experiments/sc_t34.py",
         "lifelong_ood/policy/l90c_config.py", "experiments/c_train_pi05.py", "results/flawed/sc_rv_rwrist_manifest.json",
         "results/flawed/sc_rv_rsup_manifest.json"]
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_t37w", seed=0, command="pre-registration: T37w (docs/SCAFFOLD_RESULTS.md 41)",
                    metrics={"status": "PRE-REGISTERED T37w",
                             "arms": "r2w: rwrist checkpoint served with the fill (ProprioHeld, the gate that switches the prompt); r2w0: the same "
                                     "checkpoint served without the fill. Rollouts as T37: all 16 compounds x 2 layouts, N2 main states, "
                                     "results/flawed/sc_t37/<arm>_main_<tid>.json",
                             "mapping": "sc_t34.py read --dir sc_t37, T37's estimator and margins, on N1-N6 familiar. (1) PRIMARY r2 minus r2w "
                                        "(--tag w): ENTITY VIEW WINS iff CI lower bound > +.10 (sc_t34's STRUCTURE WINS); WHOLE-VIEW MASK WINS iff "
                                        "upper bound < -.10 (STRUCTURE LOSES); else NO CLEAR DIFFERENCE. (2) r2w minus r1 (--tag w2) and (3) r2w0 "
                                        "minus r1 (--tag w3): same labels, read as whether whole-wrist masking adds to role phrases. (4) Cost on "
                                        "known K1-K6 familiar, r2w vs r0: NO COST iff lower bound >= -.10. The T37 fit gate applies: if R0 is "
                                        "UNDERFIT, nothing here is read as a statement about composition.",
                             "prediction": "r2w - r1: NO CLEAR DIFFERENCE (agentview still shows the held object in every carry frame, so "
                                           "hiding the wrist does not remove the shortcut). r2w0 - r1: NO CLEAR DIFFERENCE. r2 - r2w: ENTITY VIEW "
                                           "WINS if and only if T37's r2 - r0 is STRUCTURE WINS; I put ENTITY VIEW WINS at 45%, NO CLEAR DIFFERENCE "
                                           "50%, WHOLE-VIEW MASK WINS 5%. r2w NO COST on known tasks.",
                             "seen_before_registration": "No T37 or R2w rollout on main states has been read (T37's trainings were at steps "
                                                         "2,150-2,550 of 6,000). R2w: dataset check (13,327 of 16,240 held frames filled, K1-K6; "
                                                         "every other frame equals rsup's), norm stats identical to rsup's, a 31-step smoke "
                                                         "training, and an 8-episode smoke rollout on smoke states (0 of 8, as expected).",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "the R2w training and any R2w rollout"}, bytes=Z, config={}, script="experiments/prereg/sc_t37w.py"))

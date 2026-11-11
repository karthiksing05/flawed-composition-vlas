"""PRE-REGISTRATION ROW of Q2 (docs/THEORY_COMPOSITION.md 4b): the prototype-discovery operator parsing one demo into a pi0.5 rung's role
phrases (R0-R2b of SCAFFOLD_RESULTS 37).  BEFORE any search or read (only experiments/sc_q2.py smoke has run, on the 500-step
pi05_goal8_rsup_smoke500 checkpoint, not on any rung).

    python experiments/prereg/sc_q2.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

RUNGS = ("r0", "r1", "r2", "r2b")
rows = ledger.load("m3_prereg")
for r in RUNGS:
    assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_q2", r, "read.json")), f"read of {r} exists"
    assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_q2", r, "picked.json")), f"search of {r} already picked"
assert not any(r.get("tag") == "sc_q2" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_q2", seed=0, command="pre-registration: Q2 (docs/THEORY_COMPOSITION.md 4b)",
                    metrics={"status": "PRE-REGISTERED Q2",
                             "question": "from one rendered demo of a known task (K1-K6) and no instruction, does prototype discovery over a pi0.5 rung's "
                                         "action field pick the role phrases the demo is made of: the object phrase on 6 pre-hold rows, the target phrase on "
                                         "6 held rows (sc_t16.held_rows at the g8v threshold, results/flawed/sc_rv_lex.json)? Rungs r0 pi05_g8v_plain_ae "
                                         "(candidates: the six K1-K6 instructions over all 12 rows), r1 pi05_g8v_rsup_ae, r2 pi05_g8v_rview2_ae and r2b "
                                         "pi05_g8v_rbox_ae (candidates: 'object: the bowl' / 'the wine bottle' / 'the cream cheese' and 'target: the plate' / "
                                         "'the stove' / 'the cabinet' / 'the rack' / 'the bowl'; on r2 and r2b each candidate on its own painted view)",
                             "mapping": "Per rung, from read.json (read split, demos 25-34 of K1-K6, 60 demos, bootstrap over tasks and demos), on JOINT "
                                        "accuracy (object and target both right). (1) PARSING: the picked setting's operator joint accuracy: PARSES iff the CI "
                                        "lower bound >= .50; FAILS iff the upper bound < .50; else UNRESOLVED. (2) AGAINST RETRIEVAL, picked_minus_fm joint: "
                                        "OPERATOR BEATS RETRIEVAL iff lower bound > +.10; RETRIEVAL BEATS OPERATOR iff upper bound < -.10; else NO CLEAR "
                                        "DIFFERENCE. (3) SEARCH, picked_minus_0 joint: SEARCH MATTERS iff lower bound > +.10, else NOT SHOWN (when picked = 0: "
                                        "PICKED SETTING 0). Comparisons across rungs (r2 vs r0 parsing) are reported with no verdict. Each rung is labelled "
                                        "only if its read completes; a rung that fails the T37 fit gate is still labelled, with that noted.",
                             "selection": "10 settings (sc_q2.py settings): 0 = sc_q0 setting 0 (plan v4); 1 = sc_q0 setting 31 (3 levels .05-.4, 4 starts "
                                          "hybrid, 50 Adam steps of .02, K 5, tau .05, capped, marginal, floor 1e-2); 2-9 = setting 1 with one change each: "
                                          "levels (.1,.7); levels (.3,.9); 16 starts; floor 1e-3; floor 1e-1; extent posterior; score pooled; K 1. "
                                          "rows_per_phase 6, probes 4, dedup .05. Picked per rung = highest search-split joint accuracy, ties to the lower index",
                             "prediction": "r1, r2, r2b: PARSES (joint .60-.85), with a picked setting whose levels reach .5 or above (setting 3, "
                                           "levels .3-.9, most likely), since the smoke showed role phrases move the field only at sigma >= .5. r0: "
                                           "UNRESOLVED (joint .35-.60), because the scene, not the whole instruction, sets the mode (plan v4's failure). "
                                           "Against retrieval: NO CLEAR DIFFERENCE on every rung, as on U (Q0: both 1.000 or near). Search: SEARCH "
                                           "MATTERS on r1-r2b, NOT SHOWN on r0. If this holds, the operator is a parser of equal power to retrieval and "
                                           "not the contribution; the contribution would be the role channel that makes parsing possible.",
                             "seen_before_registration": "Only sc_q2 smoke on the 500-step checkpoint pi05_goal8_rsup_smoke500 (r1 candidates, "
                                                         "demo 0 of K1 and K5): setting 0 got 2 of 4 decisions right, setting 1 got 1 of 4, "
                                                         "flow loss 3 of 4 and 16 of 24 rows; an r2-path smoke on K1 at setting 1 (both wrong, "
                                                         "untrained on painted views); Tweedie sanity (the true phrase lowest at sigma .5 and .8). "
                                                         "Plus a re-smoke (job 4643641) of the same command after the shared-batch change: "
                                                         "same counts (setting 0 2/4, setting 1 1/4, flow loss 16/24 rows), two "
                                                         "wrong picks changed name (bf16 batch effects). No "
                                                         "rung checkpoint (pi05_g8v_*_ae) existed at registration; Q0's results (SR 36.1) are known.",
                             "split": "per task K1-K6, rendered demo index: 0-4 SEARCH, 25-34 READ; the read runs the picked setting and setting 0 once",
                             "baselines": "(a) flow-matching loss ranking (openpi t .1/.3/.5, 8 noise draws, same noise per candidate, 70 real dims); (b) chance",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "docs/THEORY_COMPOSITION.md", "experiments/sc_q2.py",
                                                                   "experiments/sc_q0.py", "experiments/sc_rv.py", "experiments/sc_t16.py",
                                                                   "lifelong_ood/policy/backbone.py"]},
                             "written_before": "any Q2 search or read"}, bytes=Z, config={}, script="experiments/prereg/sc_q2.py"))

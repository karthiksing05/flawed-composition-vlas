"""PRE-REGISTRATION ROW of Q0 (docs/THEORY_COMPOSITION.md 4b): the prototype-discovery operator parsing one demo into U's parts.
BEFORE any search or read (only experiments/sc_q0.py smoke has run: search-split demo 0 of K1 and K5, setting 0, seed 10).

    python experiments/prereg/sc_q0.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "sc_q0_read.json")), "read exists"
assert not any(r.get("tag") == "sc_q0" for r in rows), "already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="sc_q0", seed=0, command="pre-registration: Q0 (docs/THEORY_COMPOSITION.md 4b)",
                    metrics={"status": "PRE-REGISTERED Q0",
                             "question": "from one demo of a known task and no instruction, does prototype discovery over U's part fields (pp_sc_U_s10-12) "
                                         "recover the object role (movable entities, pre-hold rows) and the target role (target slots other than the object, "
                                         "held rows; U's proprioceptive hold event)?",
                             "mapping": "picked setting, read split, pooled joint accuracy (object right and target right given the inferred object), "
                                        "bootstrap 95% CI over demos (2,000 draws, seed 0, demos resampled jointly across the three checkpoints, stratified by task): "
                                        "OPERATOR PARSES iff CI lower > 0.80; OPERATOR FAILS iff CI upper < 0.50; otherwise PARTIAL. "
                                        "Paired difference picked minus setting 0 on joint accuracy: SEARCH MATTERS iff CI lower > +0.10; otherwise SEARCH DOES NOT MATTER.",
                             "selection": "50 settings (sc_q0.py settings): 0 = plan v4's (score uncapped, extent deconv, var_floor 1e-4); "
                                          "1 = the paper-like setting (15 levels .05-.4, 128 starts, hybrid, 150 iters, lr .1, K 3, tau .5, score pooled, extent posterior, var_floor 1e-4); "
                                          "2-49 = 48 distinct draws with numpy default_rng(0) from the grid levels {3,6,12} x sigma range {(.05,.4),(.1,.7),(.3,.9)} x n_starts {4,16,64} "
                                          "x init {query,hybrid} x iters {50,150} x lr {.02,.1} x K {1,3,5} x tau {.05,.5} x score {uncapped,capped,pooled} "
                                          "x extent {posterior,marginal} x var_floor {1e-4,1e-3,1e-2}, a draw redrawn if it repeats or levels*n_starts*iters > 12*16*150; "
                                          "rows_per_phase 6, probes 4, dedup .05 for all. Picked = highest search-split joint accuracy, ties to lower n_starts*levels*iters, then lower index. "
                                          "The read also reports the picked setting's pools scored by the other two score rules, without verdict",
                             "prediction": "written before the smoke run: OPERATOR PARSES at or above .90, SEARCH DOES NOT MATTER, retrieval rule also above .90. Revised after the smoke run and before any search: OPERATOR PARSES at the picked setting at or above .90; SEARCH MATTERS, plan v4's setting near .5 joint because of the variance floor; the picked setting uses floor 1e-3 or 1e-2 or pooled scoring; the retrieval rule matches the picked setting within .05",
                             "seen_before_registration": "smoke runs on two search-split demos (K1 and K5, demo 0, seed 10): setting 0 got 3 of 6 decisions, the retrieval rule and the motion heuristic 6 of 6; 23-31% of variances at the 1e-4 floor; floor 1e-3, the marginal extent or pooled scoring each fixed the misses; settings 1, 25 and 31 got 6 of 6. The fields score, extent and var_floor and the 50-setting list were added after that smoke and before any search run. No read-split demo has been run",
                             "split": "per task K1-K6, demo index: 0-9 SEARCH (seed-10 checkpoint only), 25-49 READ (seeds 10, 11, 12), 10-24 unused; "
                                      "smoke read before this row: search demo 0 of K1 and K5, seed 10",
                             "baselines": "(a) flow-matching loss ranking (t_U in .5/.7/.9, 8 draws); (b) pp_p_train.demo_sites motion heuristic; (c) chance",
                             "files_sha256": {p: fsha(p) for p in ["docs/SCAFFOLD_RESULTS.md", "docs/THEORY_COMPOSITION.md", "experiments/sc_q0.py",
                                                                   "lifelong_ood/policy/parts_flow.py"]},
                             "written_before": "any Q0 search or read"}, bytes=Z, config={}, script="experiments/prereg/sc_q0.py"))

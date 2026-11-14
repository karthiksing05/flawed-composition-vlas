"""PRE-REGISTRATION ROW of TV-F fresh (SCAFFOLD_RESULTS 47): the confirmatory read of test-time role views plus R1's role phrases on
frozen pi0.5 (and the secondary tv arms on pi0.5 and GR00T) on the atlas replication's FRESH familiar-layout states (k = 36-55 =
harness trials 24-43), against the replication's intact rollouts I.  BEFORE any fresh-state TV rollout.

    python experiments/prereg/tv_fresh.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
TIDS = (16, 18, 20, 22, 24, 26, 0, 2, 4, 6, 10)
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "tv_fresh" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "tv_fresh_read.json")), "fresh read exists"
assert not os.path.exists(os.path.join(FL, "tv_fresh")), "fresh TV rollouts exist"
for m in ("pi05", "groot"):
    for t in TIDS:
        for suf in (".json", "_track.npz"):
            assert os.path.exists(os.path.join(FL, "pp_atlas_fresh", "%s_%d%s" % (m, t, suf))), ("intact I missing", m, t, suf)


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATE_FILES = (["results/flawed/n2_states_fresh/states.npy"] + ["results/flawed/n2_states_fresh/n2_goal_%d.npy" % t for t in TIDS]
               + ["results/flawed/n2_branches_fresh.json", "results/flawed/n2_tasks.json", "results/flawed/pp_atlas_fresh_read.json"])
CODE = ["experiments/tv_fresh.py", "experiments/tv_fresh_roll.sh", "experiments/tv_frozen.py", "experiments/pp_atlas.py",
        "experiments/pp_atlas_fresh_roll.sh", "experiments/fc_n2_client.py", "experiments/libero_client.py", "experiments/sc_rv.py",
        "experiments/sc_rv_w.py", "experiments/sc_seg.py", "experiments/sc_t16.py", "experiments/pp_serve_purify.py",
        "results/flawed/sc_rv_lex.json", PP + "tv_fresh_job.sh", PP + "tv_fresh_read.sh"]
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="tv_fresh", seed=0,
                    command="pre-registration: TV-F fresh, confirmatory test-time role views + role phrases on frozen VLAs, fresh states "
                            "(docs/SCAFFOLD_RESULTS.md 47)",
                    metrics={"status": "PRE-REGISTERED TV-F FRESH (confirmatory)",
                             "arms": "I = results/flawed/pp_atlas_fresh/<model>_<tid> (atlas fresh replication, intact, pp_serve_purify "
                                     "--mode intact --comp= --seed 13, client seed 7, n2_states_fresh, n2_branches_fresh.json, --bind "
                                     "track); tv and tvp exactly as tv_frozen (SR 46): experiments/tv_frozen.py client, unchanged, run by "
                                     "experiments/tv_fresh_roll.sh with --fresh results/flawed/n2_states_fresh --branches "
                                     "results/flawed/n2_branches_fresh.json and the same server command. Runs: pi05 tvp, pi05 tv, groot tv.",
                             "tasks": "familiar n2 tids 16,18,20,22,24,26 (N1-N6) and 0,2,4,6,10 (K1-K4, K6); fresh states k = 36-55 "
                                      "= harness trials 24-43 (one episode per task and state).",
                             "mapping": "experiments/tv_fresh.py read as hashed.  Per model, success matrices [task, state]; paired "
                                        "bootstrap B=2000, numpy default_rng(0), tasks and states drawn jointly; CI 2.5/97.5.  PRIMARY "
                                        "pi0.5 d = tvp - I on N1-N6: CONFIRMED iff d > .10 and CI lo > 0; REFUTED iff CI hi < .10; else "
                                        "UNRESOLVED.  GUARD pi0.5 tvp - I on K1-K4, K6: NO COST iff CI lo > -.10 (else COST NOT "
                                        "EXCLUDED).  SECONDARY GR00T tv - I on N1-N6 and pi0.5 tv - I on N1-N6: the primary's rule.  "
                                        "Descriptive only: the atlas's within / cross-chunk excess per arm (pp_atlas estimator, partners, "
                                        "stage rule unchanged); pi0.5 tvp - tv; tv K-task cost; stage shares; segmenter / paint / detector "
                                        "diagnostics; pi0.5 tvp against R2-retrain's main-state N1-N6 success .425 (SR 37.1; R2 was never "
                                        "run on fresh states, so unpaired and descriptive).  Harness checks reported: arm client args = "
                                        "I's except out/port/videos, per-episode trial/branch/variant/at = I's, I's excess reproduces "
                                        "pp_atlas_fresh_read.json, state and branch files match the hashes below.",
                             "prediction": "PRIMARY pi0.5 tvp - I: CONFIRMED 65%, UNRESOLVED 30%, REFUTED 5%; point estimate +.25 to "
                                           "+.40 (regression from +.425).  GUARD NO COST 80%.  SECONDARY GR00T tv - I: CONFIRMED 35%, "
                                           "UNRESOLVED 50%, REFUTED 15%; pi0.5 tv - I: CONFIRMED 35%, UNRESOLVED 55%, REFUTED 10%.  "
                                           "pi0.5 tvp cross-chunk excess falls to near 0 (from I's +.450 on fresh states).",
                             "seen": "When this was written the main-state SR 46 results were known: pi0.5 I .500, tv .725, tvp .925 "
                                     "(tvp - I +.425 [+.133, +.717], no registered verdict), tvp known-task cost -.01, tvp cross-chunk "
                                     "excess +.325 -> -.013; GR00T tv - I +.233 [+.025, +.467] TV WINS, tvp hurt GR00T (.167 new, .67 "
                                     "known).  The fresh-state atlas replication's excess table (ARCH_FAILURES.md: pi0.5 cross +.450, "
                                     "GR00T cross +.500, verdicts) was known; the fresh I success matrices were not inspected by the "
                                     "registrant.  No TV arm has ever run on these states.  No smoke on fresh states: the code path "
                                     "(tv_frozen.py client) is unchanged; only the --fresh / --branches arguments move, as for I.  The "
                                     "new files are the launcher tv_fresh_roll.sh (tv_roll.sh with the fresh states, branches and "
                                     "pp_atlas_fresh_roll.sh's server command) and the read tv_fresh.py (imports tv_frozen / pp_atlas).",
                             "files_sha256": {p: fsha(p) for p in CODE + STATE_FILES},
                             "written_before": "any fresh-state TV rollout"}, bytes=Z, config={}, script="experiments/prereg/tv_fresh.py"))

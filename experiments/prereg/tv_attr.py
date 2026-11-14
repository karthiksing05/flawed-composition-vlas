"""PRE-REGISTRATION ROW of TV-A (SCAFFOLD_RESULTS 48): attribute the confirmed TV+P gain on frozen pi0.5 (SR 47) between R1's role
phrases (arm p, phrases only) and the silhouette role views, and separate the views from an AC-VLA-style whole-wrist grey at a matched
prompt (arm wp = SR 46's w plus the same phrases).  Fresh familiar-layout states k = 36-55 (harness trials 24-43).  BEFORE any main
p or wp rollout.

    python experiments/prereg/tv_attr.py
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
assert not any(r.get("tag") == "tv_attr" for r in rows), "already registered"
assert sum(r.get("tag") == "tv_fresh" for r in rows) == 1, "tv_fresh row missing"
assert not os.path.exists(os.path.join(FL, "tv_attr_read.json")), "tv_attr read exists"
assert not os.path.exists(os.path.join(FL, "tv_attr")), "tv_attr rollouts exist"
REUSED = []
for t in TIDS:
    for suf in (".json", "_track.npz"):
        p = os.path.join(FL, "pp_atlas_fresh", "pi05_%d%s" % (t, suf))
        assert os.path.exists(p), ("intact I missing", p)
        for arm in ("tv", "tvp"):
            p = "results/flawed/tv_fresh/pi05_%s_%d%s" % (arm, t, suf)
            assert os.path.exists(os.path.join(ROOT, p)), ("reused arm missing", p)
            REUSED.append(p)


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATE_FILES = (["results/flawed/n2_states_fresh/states.npy"] + ["results/flawed/n2_states_fresh/n2_goal_%d.npy" % t for t in TIDS]
               + ["results/flawed/n2_branches_fresh.json", "results/flawed/n2_tasks.json", "results/flawed/pp_atlas_fresh_read.json",
                  "results/flawed/tv_fresh_read.json"])
CODE = ["experiments/tv_attr.py", "experiments/tv_attr_roll.sh", "experiments/tv_fresh.py", "experiments/tv_fresh_roll.sh",
        "experiments/tv_frozen.py", "experiments/pp_atlas.py", "experiments/fc_n2_client.py", "experiments/libero_client.py",
        "experiments/sc_rv.py", "experiments/sc_rv_w.py", "experiments/sc_t16.py", "experiments/pp_serve_purify.py",
        "results/flawed/sc_rv_lex.json", PP + "tv_attr_job.sh", PP + "tv_attr_read.sh"]
SEEN = ("When this was written the SR 46 (main states) and SR 47 (fresh states) results were known: fresh pi0.5 I .450, tv .700, tvp "
        ".817, tvp - I +.367 [+.092, +.634] CONFIRMED, tvp - tv +.117 [-.017, +.283], tvp K cost NO COST, tvp per task N1-N6 "
        ".95/1.00/1.00/.50/1.00/.45, I .20/.25/.60/.55/.75/.35, cross-chunk excess I +.450, tv +.125, tvp -.008; main-state pi0.5 I "
        ".500, tv .725, tvp .925, w .542 (w - I +.042, w K cost -.13, tv - w +.183 NO CLEAR DIFFERENCE).  No p or wp arm has ever run "
        "on pi0.5 on any state.  Code changed (new client tv_attr.py, launcher tv_attr_roll.sh), so a smoke ran before this row: job "
        "4701216 (first attempt 4701214 failed on a launcher state-dir check, fixed: the check now tests the directory), pi0.5, arms p "
        "and wp, N1 familiar only, smoke states 0-1 (results/flawed/n2_states, first two rows of n2_branches_smoke.json), output "
        "outside results/.  Smoke outcomes: p 1 of 2 (trial 1 RELEASE away), wp 1 of 2 (trial 1 RELEASE never_let_go); p painted "
        "nothing and switched prompts on the detector; wp greyed the wrist on every held query.  Nothing changed after the smoke.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="tv_attr", seed=0,
                    command="pre-registration: TV-A, attribution of the TV+P gain on frozen pi0.5 between role phrases and silhouette "
                            "role views, and views vs AC-VLA wrist grey at a matched prompt, fresh states (docs/SCAFFOLD_RESULTS.md 48)",
                    metrics={"status": "PRE-REGISTERED TV-A (attribution)",
                             "arms": "New: p = R1's role phrases only (sc_t16.given(instruction)[held], switched by tv_frozen's "
                                     "ProprioHeld sequencer, no paint), wp = tv_frozen's w (whole wrist image grey 128 while the "
                                     "detector is on, no silhouettes) plus the same phrases. experiments/tv_attr.py client (imports "
                                     "tv_frozen.Front; tv_frozen.py unchanged), run by experiments/tv_attr_roll.sh = tv_fresh_roll.sh "
                                     "with only the client script moved: server pp_serve_purify --mode intact --comp= --seed 13, client "
                                     "seed 7, --fresh n2_states_fresh, --branches n2_branches_fresh.json, --bind track. Reused: I = "
                                     "results/flawed/pp_atlas_fresh/pi05_<tid>; tv, tvp = results/flawed/tv_fresh/pi05_<arm>_<tid> "
                                     "(hashed below; the read verifies args / episodes against I and the hashes). pi0.5 only.",
                             "tasks": "familiar n2 tids 16,18,20,22,24,26 (N1-N6) and 0,2,4,6,10 (K1-K4, K6); fresh states k = 36-55 "
                                      "= harness trials 24-43 (one episode per task and state).",
                             "mapping": "experiments/tv_attr.py read as hashed. pi0.5, success matrices [task, state]; paired "
                                        "bootstrap B=2000, numpy default_rng(0), tasks and states drawn jointly (tv_frozen.boot2); CI "
                                        "2.5/97.5; on N1-N6.  VIEWS d = tvp - p: VIEWS ADD iff d > .10 and CI lo > 0; VIEWS NOT NEEDED "
                                        "iff CI hi < .10; else UNRESOLVED.  SEPARATION d = tvp - wp: SEPARATED iff d > .10 and CI lo > "
                                        "0; W+P EQUIVALENT iff CI hi < .10; else UNRESOLVED.  Descriptive only: p - I, wp - I, wp - p, "
                                        "p - tv, tvp - tv on N1-N6; K1-K4, K6 success per arm and arm - I (known-task cost); the "
                                        "atlas's within / cross-chunk excess per arm (pp_atlas estimator via tv_frozen.atlas_est); stage "
                                        "shares; detector diagnostics.  Checks reported: arm client args = I's except out/port/videos, "
                                        "per-episode trial/branch/variant/at = I's, I reproduces pp_atlas_fresh_read.json, tv / tvp "
                                        "reproduce tv_fresh_read.json, state files match tv_fresh's row, reused rollouts match this row.",
                             "prediction": "p N1-N6 success .60-.75.  VIEWS (tvp - p): VIEWS ADD 25%, UNRESOLVED 65%, VIEWS NOT NEEDED "
                                           "10%.  SEPARATION (tvp - wp): SEPARATED 35%, UNRESOLVED 60%, W+P EQUIVALENT 5%; wp at or "
                                           "below p on new pairings and with a known-task cost (w cost -.13 on main states).  p "
                                           "cross-chunk excess below I's +.450 but above tvp's -.008.",
                             "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE + STATE_FILES + REUSED},
                             "written_before": "any main (fresh-state) p or wp rollout"},
                    bytes=Z, config={}, script="experiments/prereg/tv_attr.py"))

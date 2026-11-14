"""PRE-REGISTRATION ROW of TV-G (SCAFFOLD_RESULTS 49): do test-time silhouette role views (tv) help FROZEN GR00T N1 on new pairings
N1-N6, and do they beat an AC-VLA-style whole-wrist grey (w) at a matched prompt (no role phrases in any arm)?  Arms I, tv, w, all
run fresh on a second fresh batch of 60 familiar-layout states k = 56-115 (experiments/fc_n2_fresh2.py; harness trials 24-83).
BEFORE any main rollout.

    python experiments/prereg/tv_g.py
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
assert not any(r.get("tag") == "tv_g" for r in rows), "already registered"
assert sum(r.get("tag") == "tv_fresh" for r in rows) == 1, "tv_fresh row missing"
assert not os.path.exists(os.path.join(FL, "tv_g_read.json")), "tv_g read exists"
assert not os.path.exists(os.path.join(FL, "tv_g")), "tv_g rollouts exist"
REUSED = []
for t in TIDS:
    for suf in (".json", "_track.npz"):
        for p in ("results/flawed/pp_atlas_fresh/groot_%d%s" % (t, suf), "results/flawed/tv_fresh/groot_tv_%d%s" % (t, suf)):
            assert os.path.exists(os.path.join(ROOT, p)), ("SR 47 rollout missing", p)
            REUSED.append(p)


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATE_FILES = (["results/flawed/n2_states_fresh2/states.npy", "results/flawed/n2_states_fresh2/meta.json"]
               + ["results/flawed/n2_states_fresh2/n2_goal_%d.npy" % t for t in TIDS]
               + ["results/flawed/n2_branches_fresh2.json", "results/flawed/n2_tasks.json", "results/flawed/tv_fresh_read.json",
                  "results/flawed/pp_atlas_fresh_read.json"])
CODE = ["experiments/tv_g.py", "experiments/tv_g_roll.sh", "experiments/fc_n2_fresh2.py", "experiments/tv_fresh.py",
        "experiments/tv_frozen.py", "experiments/pp_atlas.py", "experiments/pp_atlas_client.py", "experiments/fc_n2_client.py",
        "experiments/libero_client.py", "experiments/sc_rv.py", "experiments/sc_seg.py", "experiments/pp_serve_purify.py",
        "results/flawed/sc_rv_lex.json", PP + "tv_g_job.sh", PP + "tv_g_read.sh", PP + "tv_g_states.sh"]
SEEN = ("When this was written the SR 46-48 results were known: main-state GR00T I .167, tv .400 (tv - I +.233 [+.025, +.467] TV "
        "WINS), tvp .167, w .158 (tv - w +.242 [-.025, +.550]; w K cost -.26; tv K cost -.03); SR 47 fresh k 36-55 GR00T I .183, tv "
        ".367, tv - I +.183 [-.008, +.425] UNRESOLVED, K I .99 tv .99, cross E I +.500 tv +.296; SR 48 pi0.5 tvp - wp +.442 SEPARATED. "
        "The new states k = 56-115 were drawn by fc_n2_fresh2.py (job 4702533; a first attempt 4702520 died on a print-format bug "
        "before drawing anything); no policy has run on them.  Launcher tv_g_roll.sh is new code, so a smoke ran before this row: "
        "job 4702532, GR00T, arms I, tv, w, N1 familiar only, stored smoke states 0-1 (results/flawed/n2_states, first two rows of "
        "n2_branches_smoke.json), output outside results/ (.claude_tmp/pp/tv_g_smoke).  Smoke outcomes: I 0 of 2, tv 1 of 2 (trial 0 "
        "RELEASE away), w 0 of 2 (RELEASE never_let_go, RELEASE away; 3-4 detector onsets per episode); tv paint IoU .975-.989, leak "
        "<= .001.  Nothing changed after the smoke.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="tv_g", seed=0,
                    command="pre-registration: TV-G, test-time silhouette role views (tv) vs intact (I) and vs AC-VLA-style wrist "
                            "grey (w) on frozen GR00T N1, no phrases, 60 new fresh states (docs/SCAFFOLD_RESULTS.md 49)",
                    metrics={"status": "PRE-REGISTERED TV-G",
                             "arms": "GR00T only, all three arms run new: I = experiments/pp_atlas_client.py (intact, unchanged); tv = "
                                     "tv_frozen.py client --arm tv (R2's role paint with sc_seg masks in both cameras, ProprioHeld "
                                     "detector, task prompt unchanged); w = tv_frozen.py client --arm w (whole wrist image grey 128 "
                                     "while the detector is on, no silhouettes, prompt unchanged). No role phrases. Launcher "
                                     "experiments/tv_g_roll.sh = tv_fresh_roll.sh with FRESH/BR moved to n2_states_fresh2 / "
                                     "n2_branches_fresh2.json and arm I added; server pp_serve_purify --model groot --mode intact "
                                     "--comp= --seed 13; client seed 7; --bind track; --videos 0.",
                             "tasks": "familiar n2 tids 16,18,20,22,24,26 (N1-N6) and 0,2,4,6,10 (K1-K4, K6); states k = 56-115 = "
                                      "harness trials 24-83 (k = trial + 32), one episode per task and state, branch seed_base "
                                      "13000000 + 1000 k. Distinct (not allclose) from each other, from N2 familiar k = 0-55 and "
                                      "from pp_conf familiar states (checked by fc_n2_fresh2.py).",
                             "mapping": "experiments/tv_g.py read as hashed. GR00T, success matrices [task, state] on N1-N6; paired "
                                        "bootstrap B=2000, numpy default_rng(0), tasks and states drawn jointly (tv_frozen.boot2); CI "
                                        "2.5/97.5.  PRIMARY d = tv - I: TV WINS iff CI lo > 0; else NO CLEAR DIFFERENCE.  SECONDARY "
                                        "d = tv - w: SEPARATED iff CI lo > 0; W EQUIVALENT iff CI lo > -.10 and CI hi < .10; else "
                                        "NOT SEPARATED.  Descriptive only: arm success, w - I, K1-K4,K6 success and arm - I (known-task "
                                        "cost), atlas within / cross-chunk excess per arm (tv_frozen.atlas_est), stage shares, "
                                        "detector / segmenter / paint diagnostics, tv - I pooled with SR 47's 20 fresh states (80).  "
                                        "Checks reported: arm args = I's except out/port/videos; per-episode trial/branch/variant/at "
                                        "= I's; branches address trials 24-83 with the seed rule; state files and SR 47 rollouts match "
                                        "this row; SR 47 rollouts reproduce tv_fresh_read.json; server command the same in every log.",
                             "prediction": "Point: I ~.18, tv ~.37, w ~.16 on N1-N6.  PRIMARY: TV WINS 60%, NO CLEAR DIFFERENCE 40%.  "
                                           "SECONDARY: SEPARATED 45%, NOT SEPARATED 50%, W EQUIVALENT 5%.  K cost: tv within .05 of "
                                           "I; w costs .15-.30.  Cross-chunk excess: I ~+.45 (BOTH), tv lower (~+.30) but still > 0; "
                                           "w below I mainly because episodes fail before the handoff.",
                             "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE + STATE_FILES + REUSED},
                             "written_before": "any main (60-state) I, tv or w rollout"},
                    bytes=Z, config={}, script="experiments/prereg/tv_g.py"))

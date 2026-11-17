"""PRE-REGISTRATION ROW of GROOT-RG (SCAFFOLD_RESULTS 71): does ReGuide transport (rg.py, SR 52 / 55 / 65) fix frozen GR00T N1.5's new
pairings in scene 1, and is GR00T's high carry (F1m, SR 51 / 56) still a limit under it?  Arms I, rgp (RG-perc, primary) and rgo
(RG-oracle, reference), N1-N6 and K1-K4, K6, on 40 NEW fresh familiar-layout states k = 500-539 (experiments/fc_n2_gx.py; harness trials
24-63) that no arm of any model has run on.  BEFORE any main rollout.

    python experiments/prereg/groot_rg.py
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
assert not any(r.get("tag") == "groot_rg" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "groot_rg")), "groot_rg rollouts exist"
assert not os.path.exists(os.path.join(FL, "groot_rg_read.json")), "groot_rg read exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ReGuide's code and statistics must be the hashed SR 52 version.
REG = [r for r in rows if r.get("tag") == "reguide"]
assert len(REG) == 1, len(REG)
H = REG[0]["metrics"]["files_sha256"]
for p in ("experiments/rg.py", "experiments/rg_stats.py", "experiments/sc_pu.py", "results/flawed/rg_stats.json",
          "results/flawed/sc_pu_calib.json"):
    assert H[p] == fsha(p), ("ReGuide file differs from its reguide-row hash", p)
print("[groot_rg] ReGuide files match the reguide row", flush=True)

PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATE_FILES = (["results/flawed/n2_states_gx/states.npy", "results/flawed/n2_states_gx/meta.json"]
               + ["results/flawed/n2_states_gx/n2_goal_%d.npy" % t for t in TIDS]
               + ["results/flawed/n2_branches_gx.json", "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json",
                  "results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json"])
CODE = ["experiments/groot_rg_roll.sh", "experiments/groot_rg_read.py", "experiments/fc_n2_gx.py", "experiments/rg.py",
        "experiments/rg_stats.py", "experiments/rg_read.py", "experiments/tv_frozen.py", "experiments/tv_fresh.py",
        "experiments/n46_phase.py", "experiments/pp_atlas.py", "experiments/pp_atlas_client.py", "experiments/fc_n2_client.py",
        "experiments/libero_client.py", "experiments/fa1_common.py", "experiments/sc_pu.py", "experiments/sc_rv.py",
        "experiments/sc_seg.py", "experiments/sc_t16.py", "experiments/pp_serve_purify.py", "lifelong_ood/policy/backbone.py",
        "lifelong_ood/policy/serve.py", "experiments/prereg/reguide_choices.txt", "experiments/prereg/groot_rg_pred.txt",
        "experiments/prereg/groot_rg_seen.txt", PP + "gx_job.sh", PP + "gx_readjob.sh", PP + "gx_smoke.sh", PP + "gx_states.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "groot_rg_pred.txt"), encoding="utf-8").read()
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "groot_rg_seen.txt"), encoding="utf-8").read()
MAP = open(os.path.join(ROOT, "experiments", "groot_rg_read.py"), encoding="utf-8").read()
MAP = MAP[MAP.index("MAPPING = ("):MAP.index("def v1(")]
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="groot_rg", seed=0,
                    command="pre-registration: GROOT-RG, ReGuide transport (RG-perc primary, RG-oracle reference) against the intact "
                            "frozen GR00T N1.5, 40 new fresh states k 500-539 (docs/SCAFFOLD_RESULTS.md 71)",
                    metrics={"status": "PRE-REGISTERED GROOT-RG",
                             "arms": "All new on these states, GR00T N1.5 served as in SR 49 / 56 (pp_serve_purify.py --model groot "
                                     "--mode intact --comp= --seed 13; the server returns LIBERO delta eef commands as pi0.5's does), "
                                     "run by experiments/groot_rg_roll.sh with existing clients unchanged; client seed 7, --fresh "
                                     "n2_states_gx, --branches n2_branches_gx.json, --bind track.  I = pp_atlas_client.py.  rgp = rg.py "
                                     "client --arm rgp (ReGuide RG-perc, SR 52 / 55 / 65).  rgo = rg.py client --arm rgo (RG-oracle, "
                                     "simulator poses, privileged reference).  ReGuide files = their reguide-row hashes, asserted "
                                     "here.  rg.py is model-agnostic (observation, chunk remainder and delta commands); no code "
                                     "change was needed for GR00T.",
                             "states": "k = 500-539 (40), drawn by experiments/fc_n2_gx.py = fc_n2_s1x.py's sampler rule unchanged "
                                       "(LIBERO's sampler on the familiar BDDL, seed 600000 + k); k 276-499 skipped; the rule "
                                       "reproduced stored k = 0, 1, 23, 24, 35, 36, 55, 56, 115, 116, 155, 156, 195, 236, 275 exactly; "
                                       "every new state differs from every other and from all 286 stored familiar states.  Harness "
                                       "trials 24-63 (k = trial + 476), branch seed_base 13000000 + 1000 k.  No model has run on them.",
                             "tasks": "familiar n2 tids 16,18,20,22,24,26 (N1-N6) and 0,2,4,6,10 (K1-K4, K6); one episode per task "
                                      "and state; 3 arms x 11 tasks x 40 states = 1320 episodes.",
                             "mapping": MAP,
                             "prediction": PRED,
                             "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE + STATE_FILES},
                             "written_before": "any main (k 500-539) rollout of any arm"},
                    bytes=Z, config={}, script="experiments/prereg/groot_rg.py"))

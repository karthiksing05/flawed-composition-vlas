"""PRE-REGISTRATION ROW of TVPB-RG (SCAFFOLD_RESULTS 55): a confirmatory head-to-head on frozen pi0.5 between our refined test-time
front end tvpb (tvp = role sequencer + silhouette role views + role phrases, plus the held object painted as its bounding box in the
target phase: arm sb of SR 53) and ReGuide RG-perc (SR 52), with I, tvp and RG-oracle (privileged, descriptive), on N1-N6 and
K1-K4, K6, on 40 NEW fresh familiar-layout states k = 116-155 (experiments/fc_n2_fresh3.py; harness trials 24-63) that no earlier
read or rollout has used.  BEFORE any main rollout.

    python experiments/prereg/tvpb_rg.py
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
assert not any(r.get("tag") == "tvpb_rg" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "tvpb_rg")), "tvpb_rg rollouts exist"
assert not os.path.exists(os.path.join(FL, "tvpb_rg_read.json")), "tvpb_rg read exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ReGuide's code and statistics must be the hashed SR 52 version; the box arm must be the hashed SR 53 version.
REG = {t: [r for r in rows if r.get("tag") == t] for t in ("reguide", "f1m_cue")}
assert all(len(v) == 1 for v in REG.values()), {k: len(v) for k, v in REG.items()}
H_RG, H_F1M = REG["reguide"][0]["metrics"]["files_sha256"], REG["f1m_cue"][0]["metrics"]["files_sha256"]
for p in ("experiments/rg.py", "experiments/rg_stats.py", "experiments/sc_pu.py", "results/flawed/rg_stats.json",
          "results/flawed/sc_pu_calib.json"):
    assert H_RG[p] == fsha(p), ("ReGuide file differs from its reguide-row hash", p)
assert H_F1M["experiments/f1m_cue.py"] == fsha("experiments/f1m_cue.py"), "f1m_cue.py differs from its f1m_cue-row hash"
SHARED = {p: dict(reguide=(H_RG.get(p) == fsha(p)) if p in H_RG else None, f1m_cue=(H_F1M.get(p) == fsha(p)) if p in H_F1M else None)
          for p in sorted(set(H_RG) | set(H_F1M)) if p.startswith("experiments/") and os.path.exists(os.path.join(ROOT, p))}
print("[tvpb_rg] shared code vs earlier rows (True = unchanged):", SHARED, flush=True)

PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATE_FILES = (["results/flawed/n2_states_fresh3/states.npy", "results/flawed/n2_states_fresh3/meta.json"]
               + ["results/flawed/n2_states_fresh3/n2_goal_%d.npy" % t for t in TIDS]
               + ["results/flawed/n2_branches_fresh3.json", "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json",
                  "results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json"])
CODE = ["experiments/tvpb_roll.sh", "experiments/tvpb_read.py", "experiments/fc_n2_fresh3.py", "experiments/rg.py",
        "experiments/rg_stats.py", "experiments/rg_read.py", "experiments/f1m_cue.py", "experiments/tv_frozen.py",
        "experiments/tv_fresh.py", "experiments/n46_phase.py", "experiments/pp_atlas.py", "experiments/pp_atlas_client.py",
        "experiments/fc_n2_client.py", "experiments/libero_client.py", "experiments/fa1_common.py", "experiments/sc_pu.py",
        "experiments/sc_rv.py", "experiments/sc_seg.py", "experiments/sc_t16.py", "experiments/pp_serve_purify.py",
        "experiments/prereg/reguide_choices.txt", PP + "tvpb_job.sh", PP + "tvpb_readjob.sh", PP + "tvpb_smoke.sh",
        PP + "tvpb_states.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "tvpb_rg_pred.txt"), encoding="utf-8").read()
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "tvpb_rg_seen.txt"), encoding="utf-8").read()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="tvpb_rg", seed=0,
                    command="pre-registration: TVPB-RG, frozen pi0.5 head-to-head of the refined front end tvpb (tvp + held-object "
                            "box) against ReGuide RG-perc on 40 new fresh states k 116-155 (docs/SCAFFOLD_RESULTS.md 55)",
                    metrics={"status": "PRE-REGISTERED TVPB-RG",
                             "arms": "All new on these states, pi0.5, run by experiments/tvpb_roll.sh with existing clients "
                                     "unchanged; server pp_serve_purify --model pi05 --mode intact --comp= --seed 13, client seed 7, "
                                     "--fresh n2_states_fresh3, --branches n2_branches_fresh3.json, --bind track.  I = "
                                     "pp_atlas_client.py (intact).  tvp = tv_frozen.py client --arm tvp (silhouette role views + role "
                                     "phrases, ProprioHeld sequencer, learned segmenter; SR 46-47).  tvpb = f1m_cue.py client --arm sb "
                                     "(tvp with the held object painted as the bounding box of its learned mask in the target phase, "
                                     "target pixels visible; SR 53; f1m_cue.py = its f1m_cue-row hash).  rgp = rg.py client --arm rgp "
                                     "(ReGuide RG-perc; rg.py, rg_stats.py, sc_pu.py, rg_stats.json, sc_pu_calib.json = their "
                                     "reguide-row hashes, asserted here).  rgo = rg.py client --arm rgo (ReGuide RG-oracle, "
                                     "PRIVILEGED, descriptive only).",
                             "states": "k = 116-155 (40), drawn by experiments/fc_n2_fresh3.py = fc_n2_fresh2.py's sampler rule "
                                       "unchanged (LIBERO's sampler on the familiar BDDL, seed 600000 + k); the rule reproduced stored "
                                       "k = 0, 1, 23, 24, 35, 36, 55, 56, 115 exactly; every new state differs from every other and "
                                       "from all 166 stored familiar states (k 0-115 and pp_conf's 50).  Harness trials 24-63 (k = "
                                       "trial + 92), branch seed_base 13000000 + 1000 k.  No earlier rollout or read used them.",
                             "tasks": "familiar n2 tids 16,18,20,22,24,26 (N1-N6) and 0,2,4,6,10 (K1-K4, K6); one episode per task "
                                      "and state; 5 arms x 11 tasks x 40 states = 2200 episodes.",
                             "mapping": "experiments/tvpb_read.py as hashed.  Success matrices [task, state]; paired bootstrap B=2000, "
                                        "numpy default_rng(0), tasks and states drawn jointly (tv_frozen.boot2, as SR 52); CI "
                                        "2.5/97.5.  V1: d = tvpb - I on N1-N6: FRONT END HELPS iff CI lo > 0; else NO CLEAR "
                                        "DIFFERENCE.  V2: d = tvpb - rgp on N1-N6: FRONT END WINS iff CI lo > 0; RG WINS iff CI hi < "
                                        "0; EQUIVALENT iff the CI lies within [-.10, +.10]; else NO CLEAR DIFFERENCE (a win label "
                                        "takes precedence; the equivalence flag is reported beside it).  V3: d = tvpb - tvp on N4+N6 "
                                        "([2, 40]): BOX ADDS iff CI lo > 0; else NO CLEAR DIFFERENCE (replication of SR 53).  "
                                        "Descriptive only: trained-task K1-K4, K6 success per arm and arm - I; cross-chunk and "
                                        "within-chunk excess per arm (tv_frozen.atlas_est); tvp - I, rgp - I, rgo - I, tvp - rgp, "
                                        "tvpb - rgo, rgo - rgp, per-task success, high-carry share on N4/N6, detector and RG "
                                        "diagnostics.  Checks reported: every arm's client args = I's except out/port/videos and "
                                        "per-episode trial/branch/variant/at = I's (tv_fresh.harness, SR 48/52); every episode's trial "
                                        "is registered; no episode errors; state files and code match this row; ReGuide files match "
                                        "the reguide row; f1m_cue.py matches the f1m_cue row; one server command, seed 13.",
                             "prediction": PRED,
                             "seen": SEEN,
                             "shared_code_unchanged_vs_earlier_rows": SHARED,
                             "files_sha256": {p: fsha(p) for p in CODE + STATE_FILES},
                             "written_before": "any main (k 116-155) rollout of any arm"},
                    bytes=Z, config={}, script="experiments/prereg/tvpb_rg.py"))

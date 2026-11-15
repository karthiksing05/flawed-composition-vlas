"""PRE-REGISTRATION ROW of S1-CROSS (SCAFFOLD_RESULTS 65): does the scene-1 front end carry to a third model?  Frozen pi0 (openpi
pi0_libero) and frozen SmolVLA (lerobot/smolvla_libero), arms I, tv, tvpb (SR 55's refined front end) and rgp (ReGuide RG-perc), on
N1-N6 and K1-K4, K6, on 40 NEW fresh familiar-layout states k = 236-275 (experiments/fc_n2_s1x.py; harness trials 24-63) that no model
has run on.  BEFORE any main rollout.

    python experiments/prereg/s1_cross.py
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
assert not any(r.get("tag") == "s1_cross" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "s1_cross")), "s1_cross rollouts exist"
assert not os.path.exists(os.path.join(FL, "s1_cross_read.json")), "s1_cross read exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ReGuide's code and statistics must be the hashed SR 52 version, the box arm the hashed SR 53 version, tv the hashed SR 55 version.
REG = {t: [r for r in rows if r.get("tag") == t] for t in ("reguide", "f1m_cue", "tvpb_rg")}
assert all(len(v) == 1 for v in REG.values()), {k: len(v) for k, v in REG.items()}
H = {k: v[0]["metrics"]["files_sha256"] for k, v in REG.items()}
for p in ("experiments/rg.py", "experiments/rg_stats.py", "experiments/sc_pu.py", "results/flawed/rg_stats.json",
          "results/flawed/sc_pu_calib.json"):
    assert H["reguide"][p] == fsha(p), ("ReGuide file differs from its reguide-row hash", p)
assert H["f1m_cue"]["experiments/f1m_cue.py"] == fsha("experiments/f1m_cue.py"), "f1m_cue.py differs from its f1m_cue-row hash"
assert H["tvpb_rg"]["experiments/tv_frozen.py"] == fsha("experiments/tv_frozen.py"), "tv_frozen.py differs from its tvpb_rg-row hash"
SHARED = {p: {t: (H[t].get(p) == fsha(p)) if p in H[t] else None for t in H}
          for p in sorted(set(H["reguide"]) | set(H["f1m_cue"]) | set(H["tvpb_rg"]))
          if p.startswith("experiments/") and os.path.exists(os.path.join(ROOT, p))}
print("[s1_cross] shared code vs earlier rows (True = unchanged):", SHARED, flush=True)

PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATE_FILES = (["results/flawed/n2_states_s1x/states.npy", "results/flawed/n2_states_s1x/meta.json"]
               + ["results/flawed/n2_states_s1x/n2_goal_%d.npy" % t for t in TIDS]
               + ["results/flawed/n2_branches_s1x.json", "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json",
                  "results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json"])
CODE = ["experiments/s1_cross_roll.sh", "experiments/s1_cross_read.py", "experiments/fc_n2_s1x.py", "experiments/rg.py",
        "experiments/rg_stats.py", "experiments/rg_read.py", "experiments/f1m_cue.py", "experiments/tv_frozen.py",
        "experiments/tv_fresh.py", "experiments/n46_phase.py", "experiments/pp_atlas.py", "experiments/pp_atlas_client.py",
        "experiments/fc_n2_client.py", "experiments/libero_client.py", "experiments/fa1_common.py", "experiments/sc_pu.py",
        "experiments/sc_rv.py", "experiments/sc_seg.py", "experiments/sc_t16.py", "experiments/serve_guided.py",
        "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/backbone_smolvla.py", "lifelong_ood/policy/serve.py",
        "experiments/prereg/reguide_choices.txt", "experiments/prereg/s1_cross_pred.txt", "experiments/prereg/s1_cross_seen.txt",
        PP + "s1x_job.sh", PP + "s1x_readjob.sh", PP + "s1x_smoke.sh", PP + "s1x_states.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s1_cross_pred.txt"), encoding="utf-8").read()
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s1_cross_seen.txt"), encoding="utf-8").read()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="s1_cross", seed=0,
                    command="pre-registration: S1-CROSS, the scene-1 refined front end tvpb against I, tv and ReGuide RG-perc on "
                            "frozen pi0 and frozen SmolVLA, 40 new fresh states k 236-275 (docs/SCAFFOLD_RESULTS.md 65)",
                    metrics={"status": "PRE-REGISTERED S1-CROSS",
                             "arms": "All new on these states, per model (pi0 = openpi pi0_libero, PyTorch; smolvla = "
                                     "lerobot/smolvla_libero), run by experiments/s1_cross_roll.sh with existing clients unchanged; "
                                     "server serve_guided.py --mode ours --seed 13 as the five-model atlas served these models (pi0's "
                                     "output transform adds the state back, AbsoluteActions, so the clients receive LIBERO delta eef "
                                     "commands); client seed 7, --fresh n2_states_s1x, --branches n2_branches_s1x.json, --bind track.  "
                                     "I = pp_atlas_client.py (intact).  tv = tv_frozen.py client --arm tv (silhouette role views, "
                                     "ProprioHeld sequencer, learned segmenter; SR 46 / 49).  tvpb = f1m_cue.py client --arm sb (tv + "
                                     "role phrases + the held object painted as the bounding box of its learned mask in the target "
                                     "phase; SR 53 / 55).  rgp = rg.py client --arm rgp (ReGuide RG-perc, SR 52 / 55; ReGuide files = "
                                     "their reguide-row hashes, asserted here).  X-VLA is not run: its server sends absolute eef targets "
                                     "and the client switches the OSC controller to absolute mode, while rg.py's transports issue delta "
                                     "commands, so rgp cannot run unchanged on X-VLA.",
                             "states": "k = 236-275 (40), drawn by experiments/fc_n2_s1x.py = fc_n2_fresh4.py's sampler rule "
                                       "unchanged (LIBERO's sampler on the familiar BDDL, seed 600000 + k); k 196-235 skipped so a "
                                       "concurrent batch cannot coincide; the rule reproduced stored k = 0, 1, 23, 24, 35, 36, 55, 56, "
                                       "115, 116, 155, 156, 195 exactly; every new state differs from every other and from all stored "
                                       "familiar states (k 0-195 and pp_conf's 50).  Harness trials 24-63 (k = trial + 212), branch "
                                       "seed_base 13000000 + 1000 k.  Seeds 600236-600275 and 13236000-13275000 are distinct from every "
                                       "seed of SR 47-56.  No model has run on them.",
                             "tasks": "familiar n2 tids 16,18,20,22,24,26 (N1-N6) and 0,2,4,6,10 (K1-K4, K6); one episode per task "
                                      "and state; 2 models x 4 arms x 11 tasks x 40 states = 3520 episodes.",
                             "mapping": "experiments/s1_cross_read.py as hashed.  Per model: success matrices [task, state]; paired "
                                        "bootstrap B=2000, numpy default_rng(0), tasks and states drawn jointly (tv_frozen.boot2, as SR "
                                        "52 / 55); CI 2.5/97.5.  F (testability floor, SR 50 style): TESTABLE iff I's success on "
                                        "K1-K4, K6 >= .70.  V1: d = tvpb - I on N1-N6: FRONT END HELPS iff CI lo > 0; else NO CLEAR "
                                        "DIFFERENCE.  V2: d = tvpb - rgp on N1-N6: FRONT END WINS iff CI lo > 0; RG WINS iff CI hi < "
                                        "0; EQUIVALENT iff the CI lies within [-.10, +.10]; else NO CLEAR DIFFERENCE (a win label "
                                        "takes precedence; the equivalence flag is reported beside it).  V3 (trained-task cost): d = "
                                        "tvpb - I on K1-K4, K6: COST iff CI hi < 0; NO COST iff CI lo >= -.05; else UNRESOLVED.  "
                                        "AGGREGATE over TESTABLE models: CARRIES iff V1 = FRONT END HELPS and V3 != COST on every "
                                        "testable model; PARTIAL iff on at least one but not all; DOES NOT CARRY iff on none; NOT "
                                        "TESTED iff no model is testable.  Descriptive only: tv - I, tvpb - tv, rgp - I, tv - rgp on "
                                        "N1-N6; tvpb - tv, tvpb - rgp on N4+N6; K success per arm and arm - I; per-task success; atlas "
                                        "within / cross-chunk excess per arm; stage shares; high-carry share on N4 / N6 / K4; detector "
                                        "and RG diagnostics.  No reference planner (C1) is run: SR 55 reported none.  Checks reported: "
                                        "every arm's client args = I's except out/port/videos and per-episode trial/branch/variant/at "
                                        "= I's (tv_fresh.harness); every episode's trial is registered; no episode errors; state files "
                                        "and code match this row; ReGuide files match the reguide row; f1m_cue.py the f1m_cue row; "
                                        "tv_frozen.py the tvpb_rg row; one server command per model, seed 13.",
                             "prediction": PRED,
                             "seen": SEEN,
                             "shared_code_unchanged_vs_earlier_rows": SHARED,
                             "files_sha256": {p: fsha(p) for p in CODE + STATE_FILES},
                             "written_before": "any main (k 236-275) rollout of any arm"},
                    bytes=Z, config={}, script="experiments/prereg/s1_cross.py"))

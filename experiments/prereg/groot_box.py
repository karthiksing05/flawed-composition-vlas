"""PRE-REGISTRATION ROW of GROOT-BOX (SCAFFOLD_RESULTS 56): the refined front end on FROZEN GR00T without phrases (tvb = role views +
the held object painted as its bounding box in the target phase) and whether GR00T's state input is a cue for F1m (tvg = role views +
the finger qpos replaced by GR00T's K6 cheese hold while the detector is on), with I, tv and tvbg, on N1-N6 and K1-K4, K6, on 40 NEW
fresh familiar-layout states k = 156-195 (experiments/fc_n2_fresh4.py; harness trials 24-63) that no model has run on.  BEFORE any
main rollout.

    python experiments/prereg/groot_box.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import groot_box as GB                                                       # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
TIDS = (16, 18, 20, 22, 24, 26, 0, 2, 4, 6, 10)
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "groot_box" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "groot_box")), "groot_box rollouts exist"
assert not os.path.exists(os.path.join(FL, "groot_box_read.json")), "groot_box read exists"
assert GB.G_GROOT is not None, "G_GROOT must be set from the smoke before registration"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# The box arm is f1m_cue.CueFront (hashed in the f1m_cue row); tv is tv_frozen.py (hashed in the tvpb_rg row).
REG = {t: [r for r in rows if r.get("tag") == t] for t in ("f1m_cue", "tvpb_rg", "tv_g")}
assert all(len(v) == 1 for v in REG.values()), {k: len(v) for k, v in REG.items()}
H = {k: v[0]["metrics"]["files_sha256"] for k, v in REG.items()}
assert H["f1m_cue"]["experiments/f1m_cue.py"] == fsha("experiments/f1m_cue.py"), "f1m_cue.py differs from its f1m_cue-row hash"
assert H["tvpb_rg"]["experiments/tv_frozen.py"] == fsha("experiments/tv_frozen.py"), "tv_frozen.py differs from its tvpb_rg-row hash"
SHARED = {p: {t: (H[t].get(p) == fsha(p)) if p in H[t] else None for t in H}
          for p in sorted(set(H["f1m_cue"]) | set(H["tvpb_rg"]) | set(H["tv_g"]))
          if p.startswith("experiments/") and os.path.exists(os.path.join(ROOT, p))}
print("[groot_box] shared code vs earlier rows (True = unchanged):", SHARED, flush=True)

PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATE_FILES = (["results/flawed/n2_states_fresh4/states.npy", "results/flawed/n2_states_fresh4/meta.json"]
               + ["results/flawed/n2_states_fresh4/n2_goal_%d.npy" % t for t in TIDS]
               + ["results/flawed/n2_branches_fresh4.json", "results/flawed/n2_tasks.json", "results/flawed/sc_rv_lex.json"])
CODE = ["experiments/groot_box.py", "experiments/groot_box_read.py", "experiments/groot_box_roll.sh", "experiments/fc_n2_fresh4.py",
        "experiments/f1m_cue.py", "experiments/tv_frozen.py", "experiments/tv_fresh.py", "experiments/n46_phase.py",
        "experiments/pp_atlas.py", "experiments/pp_atlas_client.py", "experiments/fc_n2_client.py", "experiments/libero_client.py",
        "experiments/fa1_common.py", "experiments/sc_rv.py", "experiments/sc_seg.py", "experiments/pp_serve_purify.py",
        "experiments/pp_f1_mix.py", "lifelong_ood/policy/backbone_groot.py", "lifelong_ood/policy/serve.py",
        PP + "gbox_job.sh", PP + "gbox_readjob.sh", PP + "gbox_smoke.sh", PP + "gbox_states.sh", PP + "gbox_statecheck.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "groot_box_pred.txt"), encoding="utf-8").read()
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "groot_box_seen.txt"), encoding="utf-8").read()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="groot_box", seed=0,
                    command="pre-registration: GROOT-BOX, frozen GR00T, no phrases: the refined front end tvb (role views + held-object "
                            "box) and the finger-state cue swap tvg against tv and I on 40 new fresh states k 156-195 "
                            "(docs/SCAFFOLD_RESULTS.md 56)",
                    metrics={"status": "PRE-REGISTERED GROOT-BOX",
                             "arms": "All new on these states, GR00T (ar0s/groot_libero), run by experiments/groot_box_roll.sh; server "
                                     "pp_serve_purify --model groot --mode intact --comp= --seed 13, client seed 7, --fresh "
                                     "n2_states_fresh4, --branches n2_branches_fresh4.json, --bind track; no arm changes the prompt "
                                     "(asserted per query in groot_box.py).  I = pp_atlas_client.py (intact).  tv = tv_frozen.py client "
                                     "--arm tv (silhouette role views, ProprioHeld sequencer, learned segmenter; SR 46 / 49).  tvb = "
                                     "groot_box.py client --arm tvb (tv + the held object painted as the bounding box of its learned "
                                     "mask in the target phase, target pixels visible: f1m_cue.CueFront cue sb, SR 53, without phrases).  "
                                     "tvg = groot_box.py client --arm tvg (tv + while the detector is on, robot0_gripper_qpos = "
                                     "observation/state[6:8] replaced by G_GROOT = %s, GR00T's mean finger qpos while holding the cream "
                                     "cheese on K6 in the smoke).  tvbg = both (descriptive)." % (list(GB.G_GROOT),),
                             "g_groot": list(GB.G_GROOT),
                             "states": "k = 156-195 (40), drawn by experiments/fc_n2_fresh4.py = fc_n2_fresh3.py's sampler rule "
                                       "unchanged (LIBERO's sampler on the familiar BDDL, seed 600000 + k); the rule reproduced stored "
                                       "k = 0, 1, 23, 24, 35, 36, 55, 56, 115, 116, 155 exactly (job 4710073); every new state differs "
                                       "from every other and from all 206 stored familiar states (k 0-155 and pp_conf's 50).  Harness "
                                       "trials 24-63 (k = trial + 132), branch seed_base 13000000 + 1000 k.  No model has run on them.",
                             "tasks": "familiar n2 tids 16,18,20,22,24,26 (N1-N6) and 0,2,4,6,10 (K1-K4, K6); one episode per task "
                                      "and state; 5 arms x 11 tasks x 40 states = 2200 episodes.",
                             "mapping": "experiments/groot_box_read.py as hashed.  Success matrices [task, state]; paired bootstrap "
                                        "B=2000, numpy default_rng(0), tasks and states drawn jointly (tv_frozen.boot2, as SR 52 / 55); "
                                        "CI 2.5/97.5.  V1: d = tvb - I on N1-N6: FRONT END HELPS iff CI lo > 0; else NO CLEAR "
                                        "DIFFERENCE.  V2: d = tvb - tv on N4+N6 ([2, 40]): BOX ADDS iff CI lo > 0; NOT THIS CUE iff CI "
                                        "hi < .10; else UNRESOLVED.  V3: d = tvg - tv on N4+N6: CUE FOUND iff CI lo > 0; NOT THIS CUE "
                                        "iff CI hi < .10; else UNRESOLVED.  (lo > 0 takes precedence.)  Descriptive only: trained-task "
                                        "K1-K4, K6 success per arm and arm - I; tv - I, tvg - I, tvbg - I, tvb - tv on N1-N6; tvbg vs "
                                        "tv / tvb / tvg on N4+N6; per-task success; atlas within / cross-chunk excess per arm; "
                                        "high-carry share (n46_phase lift_held > .22 m among grasped) on N4 / N6 / N5 / K4 and success "
                                        "after high / low carries; detector stats; finger entries actually sent at held queries.  "
                                        "Checks reported: every arm's client args = I's except out/port/videos and per-episode "
                                        "trial/branch/variant/at = I's (tv_fresh.harness); every episode's trial is registered; no "
                                        "episode errors; state files and code match this row; f1m_cue.py matches the f1m_cue row; "
                                        "tv_frozen.py matches the tvpb_rg row; one server command, seed 13.",
                             "prediction": PRED,
                             "seen": SEEN,
                             "shared_code_unchanged_vs_earlier_rows": SHARED,
                             "files_sha256": {p: fsha(p) for p in CODE + STATE_FILES},
                             "written_before": "any main (k 156-195) rollout of any arm"},
                    bytes=Z, config={}, script="experiments/prereg/groot_box.py"))

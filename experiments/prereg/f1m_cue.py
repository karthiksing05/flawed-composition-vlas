"""PRE-REGISTRATION ROW of F1m-CUE (SCAFFOLD_RESULTS 53): which input makes frozen pi0.5 start K4's high carry when it holds the wine
bottle on N4 / N6 under tvp?  Arms on top of tvp, each removing one cue without adding information: sb (held object painted as its
bounding box), sf (fixed-size box), g (held finger qpos replaced by the cheese-held value).  sg and h are not rolled out: the served
pi0.5 does not read observation/state (statecheck, smoke job 4706934).  BEFORE any main rollout.

    python experiments/prereg/f1m_cue.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
TIDS = (16, 18, 20, 22, 24, 26, 0, 2, 4, 6, 10)
MAIN_TIDS = (22, 26, 6)
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "f1m_cue" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "f1m_cue")), "f1m_cue rollouts exist"
assert not os.path.exists(os.path.join(FL, "f1m_cue_read.json")), "f1m_cue read exists"
REUSED = []
for t in TIDS:
    for suf in (".json", "_track.npz", "_tv.json"):
        REUSED.append("results/flawed/tv_fresh/pi05_tvp_%d%s" % (t, suf))
for t in MAIN_TIDS:
    for suf in (".json", "_track.npz", "_tv.json"):
        REUSED.append("results/flawed/tv_frozen/pi05_tvp_%d%s" % (t, suf))
for p in REUSED:
    assert os.path.exists(os.path.join(ROOT, p)), ("reused tvp missing", p)


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATE_FILES = (["results/flawed/n2_states_fresh/states.npy"] + ["results/flawed/n2_states_fresh/n2_goal_%d.npy" % t for t in TIDS]
               + ["results/flawed/n2_states/n2_goal_%d.npy" % t for t in MAIN_TIDS]
               + ["results/flawed/n2_branches_fresh.json", "results/flawed/n2_branches_main.json", "results/flawed/n2_tasks.json"])
CODE = ["experiments/f1m_cue.py", "experiments/f1m_cue_roll.sh", "experiments/tv_frozen.py", "experiments/tv_fresh.py",
        "experiments/n46_phase.py", "experiments/pp_atlas.py", "experiments/fc_n2_client.py", "experiments/libero_client.py",
        "experiments/sc_rv.py", "experiments/sc_seg.py", "experiments/sc_t16.py", "experiments/pp_serve_purify.py",
        "results/flawed/sc_rv_lex.json", PP + "f1m_job.sh", PP + "f1m_read.sh", PP + "f1m_smoke.sh",
        PP + "f1m_smoke_4706934/statecheck.json"]
SEEN = ("When this was written: SR 46-49 and SR 51 were known (pi0.5 tvp N4 / N6: main .80 / .75, fresh .50 / .45; high-carry share "
        "tvp main .00 / .10, fresh .25 / .55; pooled pi0.5 N4+N6 success .04 after a high carry, .66 after a low one; N46-PG PLACE "
        "PRESENT 26/26).  New code, so a smoke ran first: job 4706934 (overcap), output outside results/ "
        "(.claude_tmp/pp/f1m_smoke_4706934).  (1) statecheck on the served pi0.5 (GuidedPolicy, pp_serve_purify's seed rule): config "
        "pi05_libero, pi05 True, discrete_state_input False; over 3 image draws x 3 prompts, the action chunk under 5 state swaps "
        "(gripper set to (.02, -.02) [the provisional G_HELD] or (.039, -.039), eef z + .20, random N(0, 1) on all 8 dims, all zeros) "
        "differs from the base by exactly 0.0 in every case; a repeat is exact; image swaps move actions by .008-.104 and prompt swaps "
        "by .17-2.0 (max abs).  So pi0.5 never reads observation/state.  (2) arms sb, sf (provisional BOX_FIXED agent 40x40, wrist "
        "120x120), g (provisional G_HELD .02 / -.02) on N4, K4, K6, smoke states 0-1 (main n2_states): N4 sb 2/2, sf 2/2, g 0/2 "
        "(both RELEASE away, two detector onsets); K4 2/2 and K6 2/2 under every arm.  Held-object bbox (px on 256): agent bottle "
        "h <= 56, w <= 27, cheese 20 x 18; wrist bottle h <= 157, w <= 66, cheese 72 x 78.  Held finger qpos: cheese (K6) mean "
        "(.0198, -.0223), bottle (N4, K4) ~(.008, -.008).  After the smoke the two constants were set from these numbers (BOX_FIXED "
        "agent 56x28, wrist 160x80; G_HELD (.0198, -.0223)); no other code changed.  I have not seen tvp's per-episode N4 / N6 rows on "
        "the smoke states.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="f1m_cue", seed=0,
                    command="pre-registration: F1m-CUE, which input cue makes frozen pi0.5 start K4's high carry when holding the "
                            "wine bottle on N4 / N6 under tvp (docs/SCAFFOLD_RESULTS.md 53)",
                    metrics={"status": "PRE-REGISTERED F1m-CUE",
                             "arms": "All on top of tvp (tv_frozen role views + R1 phrases, same ProprioHeld detector and learned "
                                     "segmenter), one target-phase input changed: sb = the held object painted as the bounding box of "
                                     "its learned mask (target pixels visible); sf = a fixed-size box (agent 56x28, wrist 160x80 px) "
                                     "centred on that bbox, unioned with it, target pixels visible; g = while the detector is on, the "
                                     "two finger qpos entries of observation/state replaced by (.0198, -.0223), K6's cheese-held mean. "
                                     "sg and h NOT rolled out: pi05_libero does not read observation/state (statecheck), so sg = sf / "
                                     "sb and h = tvp by construction; g is rolled out on N4 / N6 as the empirical check. "
                                     "experiments/f1m_cue.py client via f1m_cue_roll.sh (= tv_attr_roll.sh with the client and state "
                                     "set moved): server pp_serve_purify --mode intact --comp= --seed 13, client seed 7, --bind track.",
                             "tasks": "fresh (trials 24-43, n2_states_fresh, n2_branches_fresh.json): sb and sf on N1-N6 + K1-K4, K6; "
                                      "g on N4, N6.  main (states 4-23, n2_states, n2_branches_main.json): sb and sf on N4, N6, K4; "
                                      "g on N4, N6.  Base tvp reused: tv_fresh/pi05_tvp_<tid>, tv_frozen/pi05_tvp_<tid> (hashed).",
                             "mapping": "experiments/f1m_cue.py read as hashed.  Primary cells N4, N6 x (fresh 20 + main 20) = [2, 40]; "
                                        "for each arm X in sb, sf, g: d = X - tvp paired, tv_frozen.boot2 (B=2000, numpy "
                                        "default_rng(0), tasks and states drawn jointly), CI 2.5/97.5.  CUE FOUND iff CI lo > 0; NOT "
                                        "THIS CUE iff CI hi < .10; else UNRESOLVED.  sg and h take sf's / g's outcome by construction "
                                        "(state not an input).  Descriptive: high-carry share (n46_phase lift_held > .22 m among "
                                        "grasped) per arm and its change vs tvp; success after high / low carries; K1-K4, K6 fresh and "
                                        "K4 both sets vs tvp; N1-N3, N5 fresh vs tvp; box sizes and exceed rate; g's per-episode "
                                        "identity with tvp; harness (args, episodes) vs tvp; reused hashes.",
                             "prediction": "g: identical to tvp episode for episode (85%) -> d 0 [0, 0], NOT THIS CUE (90% overall; "
                                           "UNRESOLVED 10% if the harness is not bit-reproducible).  sb: CUE FOUND 15%, UNRESOLVED 50%, "
                                           "NOT THIS CUE 35%; point d ~ +.05.  sf: CUE FOUND 20%, UNRESOLVED 50%, NOT THIS CUE 30%; "
                                           "point d ~ +.07.  High-carry share (tvp ~.22 pooled) changes by less than .10 under every "
                                           "arm.  K4 stays >= .90 under sb and sf (it needs the high carry); known-task cost within "
                                           ".05; N1-N3, N5 within .05 of tvp.  My guess for the real cue: the object phase (the "
                                           "unpainted bottle and 'object: the wine bottle' at the grasp, which K4's grasp-and-lift "
                                           "chunk starts from), which no target-phase arm reaches.",
                             "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE + STATE_FILES + REUSED},
                             "written_before": "any main f1m_cue rollout"},
                    bytes=Z, config={}, script="experiments/prereg/f1m_cue.py"))

"""PRE-REGISTRATION ROW of S2FIX (SCAFFOLD_RESULTS 54; written to docs/sr54_scene2_fix.md): does any structural change that adds no
task information fix F1 in scene 2 (LIBERO-10 LIVING_ROOM_SCENE5)?  BEFORE any main (k = 4-63) hand-over rollout.

    python experiments/prereg/scene2_fix.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2fix                                                                 # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "scene2_fix" for r in rows), "already registered"
assert len([r for r in rows if r.get("tag") == "scene2"]) == 1
assert not os.path.exists(os.path.join(FL, "s2fix_read.json")), "s2fix read exists"
assert not os.path.exists(os.path.join(FL, "s2fix_roll")), "s2fix rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/s2fix.py", "experiments/s2fix_roll.sh", "experiments/s2.py", "experiments/s2_tasks.py", "experiments/libero_client.py",
        "experiments/tv_frozen.py", "experiments/sc_rv.py", "experiments/pp_atlas.py", "experiments/pp_serve_purify.py",
        "experiments/fa1g_plan.py", "experiments/fa1b_common.py", "lifelong_ood/policy/parts_flow.py", "results/flawed/sc_rv_lex.json",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json", "results/flawed/s2/branches_main.json",
        PP + "s2fix_smoke.sh", PP + "s2fix_job.sh", PP + "s2fix_read.sh", PP + "s2fix_selftest.sh"]
REUSED = {os.path.relpath(p, ROOT): fsha(p) for p in s2fix.reused_files()}
assert len(REUSED) == 2 * 4 * 5, len(REUSED)
SEEN = (
    "Written after smokes on smoke states k = 0-3 only (results/flawed/s2/branches_smoke.json), outputs outside results/ "
    "(.claude_tmp/pp/s2fix_smoke/{pi05,groot}); jobs 4707564 (pi0.5: paintcheck, htvp and hI on tids 0-3) and 4707565 (GR00T hI on "
    "tids 0-3).  PAINTCHECK (states 0-1, tids 0-3): this scene has no fixtures (both plates are objects); tv's target phase paints "
    "every movable except the target plate, i.e. all three mugs (the held one included) AND the non-target plate, every pixel of it "
    "in both cameras; the target plate is never painted in the target phase.  So tv already is the proposed tvx, and arm (a) is SR "
    "50's tv / tvp, reused.  "
    "SMOKES (4 states per task): pi0.5 htvp K1 4/4, K2 4/4, N1 4/4, N2 4/4 (the named mug bound first on all 16; hand-over fired on "
    "all, no aborts; detector agrees with tv's on every step).  pi0.5 hI K1 4/4, K2 1/4 (the white mug is grasped first on all four "
    "K2 episodes and P places it on the right plate), N1 3/4 (one RELEASE miss), N2 0/4 (the white mug grasped first on all four, "
    "placed on the left plate).  GR00T hI K1 3/4 (one failed grasp: hand-over, abort, no regrasp), K2 0/4 (white or red mug "
    "grasped), N1 4/4, N2 0/4 (white mug grasped on all four).  Read selftest job 4707665 (s2fix.py read --selftest on smoke "
    "states with SR 50's smoke I / tv / tvp) ran clean; on 2 tasks the joint bootstrap's pooled lower bound equals the smaller "
    "task, so pooled FIXED needs both tasks above 0 (noted, rule unchanged).  "
    "Nothing in s2fix.py, s2fix_roll.sh or the read changed after the smokes.  Known when written: SR 50 (all numbers, incl. per "
    "task: pi0.5 I grasps the named mug first on N1 1.00 and on N2 .00; tvp on N2 .533; GR00T I N1 .917, N2 .067; the reference "
    "planner N1 .333, N2 .967), SR 51, SR 52, and s2_inspect outputs of SR 50's read.")
PRED = (
    "Arm (a), reused tv / tvp (pi0.5) and tv (GR00T): NOT FIXED pooled and per task (99%; SR 50 gives 0 with CI [0, 0]).  "
    "pi0.5 htvp (PRIMARY): N1 ~.90, N2 ~.50 (bounded by tvp's named-mug-first rate, .533 on SR 50's main states; the smoke's 4/4 is "
    "easier states), pooled ~.70; pooled FIXED 75%, PARTIAL 23%, NOT FIXED 2%; N1 FIXED 90%; N2 FIXED 50%, PARTIAL 48%; guard on "
    "K1+K2 COST 70% (K2 ~.65: tvp grasps the white mug first on .467 of K2 in SR 50, and P then places the wrong mug), point ~-.17.  "
    "pi0.5 hI (SECONDARY): N1 ~.85, N2 ~.02 (I grasps the white mug first on 100% of N2), pooled ~.43; pooled PARTIAL 80% (the "
    "joint lower bound sits at N2's ~0), FIXED 10%, NOT FIXED 10%; N1 FIXED 90%; N2 NOT FIXED 95%; guard COST 95% (K2 ~.15: I grasps "
    "the white mug first on K2 too, and the hand-over stops the trained two-step program that rescued K2 under I), point ~-.42.  "
    "GR00T hI (SECONDARY, not testable): N1 ~.75, N2 ~.03, pooled ~.39; PARTIAL 75%; N1 FIXED 85%; N2 NOT FIXED 85%; guard COST 70% "
    "(K2 ~.05).  Mechanism expected: with the hand-over, the carry and release never read the held mug; every remaining N failure is "
    "the wrong mug grasped first (F2) or a P drop / release miss, and habit placements (named mug on its trained plate) go to ~0.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_10", tag="scene2_fix", seed=0,
                    command="pre-registration: S2FIX, structural fixes for F1 in scene 2 (LIBERO-10 LIVING_ROOM_SCENE5): target-phase "
                            "views (reused SR 50 tv / tvp) and the hand-over to an object-blind place policy (docs/SCAFFOLD_RESULTS.md 54)",
                    metrics={"status": "PRE-REGISTERED S2FIX",
                             "arms": "pi0.5: htvp (PRIMARY; tvp's object phase + hand-over, chosen before any read because pi0.5 I "
                                     "grasps the wrong mug first on 100% of N2 in SR 50), hI (SECONDARY); GR00T: hI (SECONDARY; GR00T is "
                                     "NOT TESTABLE per SR 50, K1-K2 .508).  Arm (a) = SR 50's pi0.5 tv, tvp and GR00T tv, reused (tv "
                                     "already paints the non-target plate in the target phase; paintcheck).  Arm (c) not run: the "
                                     "target-only phrase 'target: the right plate' is already tvp's target-phase phrase.  Tasks tid "
                                     "0 K1, 1 K2, 2 N1, 3 N2; main states k = 4-63; branches_main.json; servers and client flags as "
                                     "SR 50.  P: object-blind place policy (s2fix.Place) reading only proprioception and the TARGET "
                                     "plate's simulator position (privileged label), constants from the target plate's own trained "
                                     "half (keypoints.json).",
                             "mapping": s2fix.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "reused_sha256": REUSED,
                             "deviations_planned": "P reads the target plate's simulator position (privileged; HP in SR 44 used "
                                                   "learned perception).  Two new pairings, so the joint bootstrap resamples 2 tasks; "
                                                   "state-only beside it.  Ledger tree dirty (other agents' files).",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main (k = 4-63) hand-over rollout"},
                    bytes=Z, config={}, script="experiments/prereg/scene2_fix.py"))

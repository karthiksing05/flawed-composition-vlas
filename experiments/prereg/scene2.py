"""PRE-REGISTRATION ROW of SCENE 2 (SCAFFOLD_RESULTS 50): do F1 (the held object picks the destination; the atlas's cross-chunk break)
and the test-time role-view front end replicate in a second trained scene, LIBERO-10's LIVING_ROOM_SCENE5 (two mugs x two plates)?
BEFORE any main (k = 4-63) rollout or reference run.

    python experiments/prereg/scene2.py
"""
import glob
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2                                                                    # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "scene2" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "s2_read.json")), "s2 read exists"
assert not os.path.exists(os.path.join(FL, "s2_roll")), "s2 rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
DATA = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(FL, "s2", "**", "*"), recursive=True) if os.path.isfile(p))
assert len(DATA) >= 6 + 6 + 4, DATA
CODE = ["experiments/s2.py", "experiments/s2_tasks.py", "experiments/s2_roll.sh", "experiments/libero_client.py",
        "experiments/tv_frozen.py", "experiments/sc_rv.py", "experiments/pp_atlas.py", "experiments/pp_serve_purify.py",
        "experiments/serve_guided.py", "experiments/fa1g_plan.py", "experiments/fa1b_common.py", "experiments/fa1_common.py",
        "results/flawed/sc_rv_lex.json", PP + "s2_job.sh", PP + "s2_ref.sh", PP + "s2_read.sh", PP + "s2_prep.sh", PP + "s2_kp.sh",
        PP + "s2_smoke.sh", PP + "s2_refsmoke.sh"]
SEEN = (
    "Written after a feasibility pass and smokes, all on smoke states k = 0-3 (results/flawed/s2/branches_smoke.json), outputs "
    "outside results/ (.claude_tmp/pp/s2_smoke, s2_check).  Jobs: 4704506 (build states / BDDLs, sc_seg check; its keypoint step "
    "failed: no h5py in the LIBERO venv, and the LIBERO-90 single-step raw demos are absent), 4704525 (keypoints from the LIBERO-10 "
    "compound's own raw demos), 4704509-4704512 (VLA smokes), 4704654 and 4704674 (reference re-runs).  "
    "sc_seg does NOT transfer: its classes have no mug and one plate; on agentview the white mug's pixels read background .77 / "
    "bowl .20, the yellow-white mug background .60 / wine bottle .36, the right plate background .98; so tv / tvp use SIMULATOR "
    "masks (privileged).  "
    "VLA smoke (4 states each): pi0.5 I: K1 4/4, K2 4/4 (but it first moves the white mug to the left plate, i.e. it runs the "
    "trained compound), N1 0/4, N2 0/4, KC 4/4, NC 0/4; in every N and NC episode the final scene is the TRAINED compound (white "
    "on left, yellow-white on right).  pi0.5 tv: K1 4/4, K2 4/4, N1 0/4, N2 0/4 (habit placements persist).  pi0.5 tvp: K1 4/4, "
    "K2 4/4 (now the named mug is grasped first), N1 0/4 (white mug grasped, ends on the left plate 3/4), N2 0/4 (yellow-white "
    "grasped, ends on the right plate 3/4).  GR00T I: K1 3/4, K2 2/4 (two GRASP other_object), N1 0/4 (HANDOFF), N2 0/4 (2 HANDOFF, "
    "2 GRASP other_object).  GR00T tv: K1 4/4, K2 0/4 (all grasp the painted white mug), N1 0/4, N2 0/4 (all grasp the painted "
    "white mug).  pi0 I: K1 3/4, N1 0/4.  Detector onsets within 1-3 steps of the oracle Held on held episodes.  "
    "Reference planner: first version (fitted offsets, 40-step carry limit) K1 4/4, K2 2/4, N1 1/4, N2 1/4; then two changes, "
    "both before this row and both on smoke states only: the carry phase limit raised to 120 steps (CARRY_STEPS) and the release "
    "made object-centred (the hand goes where the held mug's centre lies over the plate's centre, from the live hand-minus-mug "
    "offset).  Final smoke: K1 4/4, K2 3/4, N1 3/4, N2 3/4 (misses: one grasp miss on state 3 for the yellow-white mug in K2 and N2, "
    "one drop during the long N1 carry).  Nothing in the VLA client, front end or read changed after the VLA smokes.  "
    "Earlier results known when written: SR 46-49 and the atlas tables in docs/ARCH_FAILURES.md.")
PRED = (
    "pi0.5 I: N1-N2 success ~.03, K1-K2 ~.95, TESTABLE; atlas cross E ~+.8, BREAKS (90%); verdict CROSS-CHUNK 60% / BOTH 35%; "
    "N1 white mug on the left plate instead ~.8.  GR00T I: N ~.02, K ~.65; TESTABLE 55%; if testable, cross BREAKS 80%.  "
    "pi0 I: N ~.05, K ~.8, cross BREAKS 75%.  SCENE-2 F1: REPLICATES 45%, PARTIAL 45%, NOT REPLICATED 7%, NOT TESTABLE 3%.  "
    "C1 (reference on N1-N2 >= .80): MET 55% (point ~.80).  "
    "Front end: pi0.5 PRIMARY tvp - I: REFUTED 70%, UNRESOLVED 25%, CONFIRMED 5% (point ~+.03); pi0.5 tv - I REFUTED 75%; GR00T "
    "PRIMARY tv - I: NO CLEAR DIFFERENCE 92%, TV WINS 8%.  SCENE-2 FIX: NOT REPLICATED 85%, PARTIAL 13%, REPLICATES 2%.  GUARD: "
    "pi0.5 tvp NO COST 85%, pi0.5 tv NO COST 85%, GR00T tv COST 70% (it grasps the painted white mug on K2).  KC: pi0.5 ~.95, "
    "GR00T ~.7, pi0 ~.8; NC ~0 on all three, with the TRAINED atoms (white-left, yellow-white-right) reached at about KC's rate.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_10", tag="scene2", seed=0,
                    command="pre-registration: SCENE 2, the F1 atlas and the test-time role-view front end on LIBERO-10's "
                            "LIVING_ROOM_SCENE5 (docs/SCAFFOLD_RESULTS.md 50)",
                    metrics={"status": "PRE-REGISTERED SCENE 2",
                             "scene": "LIBERO-10 LIVING_ROOM_SCENE5: white (porcelain) mug, yellow-and-white mug, red mug, left plate "
                                      "(plate_1), right plate (plate_2). The only scene among the 40 training tasks with two movable "
                                      "objects bound to two targets (one compound task: white -> left, yellow-white -> right). New "
                                      "pairings N1 white -> right, N2 yellow-white -> left. The type pairing mug -> plate is trained "
                                      "(also LIBERO-10 scene 6), so what is new is which plate.",
                             "tasks": "s2_tasks.py: tid 0 K1, 1 K2, 2 N1, 3 N2 (400 steps), 4 KC (training instruction verbatim), "
                                      "5 NC (swapped compound) (520 steps). States k = 4-63 (60) from LIBERO's sampler, seed 830000 + k; "
                                      "branch seed_base 23000000 + 1000 k; one episode per task and state.",
                             "arms": "pi0.5: I (tids 0-5), tv, tvp (tids 0-3). GR00T: I (0-5), tv (0-3). pi0: I (0-5). Reference "
                                     "planner (C1) on tids 0-3. Servers: pp_serve_purify --mode intact --comp= --seed 13 (pi0.5, GR00T), "
                                     "serve_guided --mode ours --seed 13 (pi0). Client seed 7, --bind track. tv / tvp: tv_frozen.Front "
                                     "unchanged with SIMULATOR masks (sc_seg does not transfer); tvp phrases 'object: the white mug' / "
                                     "'target: the right plate' etc.",
                             "mapping": s2.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "deviations_planned": "Two new pairings only, so the joint (task, state) bootstrap resamples 2 tasks; the "
                                                   "state-only bootstrap is reported beside it as descriptive. The ledger tree is dirty "
                                                   "(other agents' uncommitted files, incl. serve_guided.py and backbone.py; their diffs "
                                                   "do not touch mode ours or pi05_libero / pi0_libero).",
                             "files_sha256": {p: fsha(p) for p in CODE + DATA},
                             "written_before": "any main (k = 4-63) VLA rollout or reference run"},
                    bytes=Z, config={}, script="experiments/prereg/scene2.py"))

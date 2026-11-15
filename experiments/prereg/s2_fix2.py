"""PRE-REGISTRATION ROW of S2FIX2 (SCAFFOLD_RESULTS 62): a scene-2 fix on frozen pi0.5 that keeps the trained tasks and fixes the new
pairings.  SR 54's hand-over with the step-1 cause fixed (hand over only at a hold of the NAMED mug; arm H) and the GATED hand-over
(arm G: at that hold, a router predicts the VLA's place plate by SR 61's majority rule over {nearer, salient, trained}; the VLA keeps
control iff the prediction equals the instructed plate), against I and SR 54's hI unchanged (H0), on NEW states (seeds 890000 + k),
layout O (PRIMARY) and SR 61's PD2 layout on the new states (SECONDARY).  BEFORE any main (k = 4-63) rollout or reference run.

    python experiments/prereg/s2_fix2.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2fix2                                                                # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_fix2" for r in rows), "already registered"
assert len([r for r in rows if r.get("tag") == "s2_cue5"]) == 1 and len([r for r in rows if r.get("tag") == "scene2_fix"]) == 1
assert not os.path.exists(os.path.join(FL, "s2fix2_read.json")), "s2fix2 read exists"
assert not os.path.exists(os.path.join(FL, "s2fix2_roll")), "s2fix2 rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/s2fix2/%s/s2_living_%d.npy" % (L, t) for L in s2fix2.LAYOUTS for t in (0, 1, 2, 3)]
CODE = ["experiments/s2fix2.py", "experiments/s2fix2_roll.sh", "experiments/s2fix.py", "experiments/s2.py", "experiments/s2_tasks.py",
        "experiments/s2cue.py", "experiments/s2cue2.py", "experiments/s2cue3.py", "experiments/s2cue4.py", "experiments/s2cue5.py",
        "experiments/libero_client.py", "experiments/sc_rv.py", "experiments/pp_serve_purify.py", "experiments/fa1g_plan.py",
        "experiments/fa1b_common.py", "experiments/fa1_common.py", "experiments/pp_atlas.py", "lifelong_ood/policy/parts_flow.py",
        "lifelong_ood/policy/serve.py", "lifelong_ood/policy/backbone.py",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json", "results/flawed/s2fix2_diag.json",
        "results/flawed/s2fix2/branches_main.json", "results/flawed/s2fix2/branches_smoke.json",
        "results/flawed/s2fix2/s2fix2_states.json", "experiments/prereg/s2_fix2_pred.txt", "experiments/prereg/s2_fix2_seen.txt"] + STATES + [
        PP + "s2fix2_diag.sh", PP + "s2fix2_states.sh", PP + "s2fix2_smoke.sh", PP + "s2fix2_job.sh", PP + "s2fix2_ref.sh",
        PP + "s2fix2_read.sh"]
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_fix2_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_fix2_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
RUNS = [["pi05", L, arm, "PRIMARY" if L == "O" else "SECONDARY", [0, 1, 2, 3]] for L in s2fix2.LAYOUTS for arm in s2fix2.RUNS[L]]
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_fix2", seed=0,
                    command="pre-registration: S2FIX2, scene 2 (LIVING_ROOM_SCENE5) on NEW states: frozen pi0.5 I vs SR 54's hand-over "
                            "(H0, unchanged), the hand-over with the step-1 cause fixed (H: hand over only at a hold of the named mug) "
                            "and the GATED hand-over (G: SR 61's majority-rule router keeps the VLA when its predicted place plate is "
                            "the instructed one); layout O PRIMARY, SR 61's PD2 on the new states SECONDARY (docs/SCAFFOLD_RESULTS.md 62)",
                    metrics={"status": "PRE-REGISTERED S2FIX2",
                             "runs": RUNS,
                             "design": dict(base=s2fix2.BASE, br_base=s2fix2.BR_BASE, main=s2fix2.MAIN, smoke=s2fix2.SMOKE,
                                            pd2_plates=s2fix2.PD2_PLATES, held_r=s2fix2.HELD_R, trained=s2fix2.TRAINED,
                                            ni_margin=s2fix2.NI_MARGIN),
                             "router": "At a detector onset (ProprioHeld, SR 54's threshold) the held mug is the mug nearest the hand "
                                       "in xy (simulator positions) within .08 m.  If it is not the named mug, the VLA keeps control "
                                       "(H and G).  If it is: H hands over to SR 54's P; G predicts the VLA's place plate as the "
                                       "plate backed by >= 2 defined cues of {NEARER (settled start positions, .02 m tolerance), "
                                       "SALIENT (SR 60's rule on the simulator segmentation of the agentview start render), TRAINED "
                                       "(training-task index: white -> plate_1, yellow-and-white -> plate_2)}, and hands over iff the "
                                       "prediction differs from the instructed plate or is undefined.",
                             "information": "Beyond SR 54: the training-task index (which plate each mug was trained to).  Simulator "
                                            "state of the kind SR 54 already read (SR 54's P: target plate position; SR 54's htvp: "
                                            "simulator masks of all mugs and plates), here: mugs' and plates' settled start "
                                            "positions, the start render's segmentation, and the mugs' positions at the onset.  No "
                                            "task information beyond the instruction's object and target roles.",
                             "labels": s2fix2.MAPPING,
                             "step1_diagnosis": "DESCRIPTIVE, run before this row on SR 54's existing outputs only (job 4720867, "
                                                "results/flawed/s2fix2_diag.json): under hI the white mug was moved first on 60 of 60 "
                                                "K2 episodes, P carried it to the instructed right plate on 39, success 0; under htvp "
                                                "the named mug was moved first on 32 (success 32 of 32) and the white mug on 28 "
                                                "(success 1).  P succeeded on every K episode where it carried the named mug (K1 120 "
                                                "of 120, K2 32 of 32); first onset vs oracle Held median 0 steps on K1.  Cause: the "
                                                "hand-over fires at a hold of the wrong mug.",
                             "prediction": PRED,
                             "seen": SEEN,
                             "deviations_planned": "pi0.5 only.  Client seed 7 and server seed 13 kept from SR 50-61 (state and "
                                                   "branch seeds are new).  H, G and H0 run on I's front end (no views): SR 54's "
                                                   "primary htvp needed simulator masks and cost K2 under tvp alone (.767).  PD2 is "
                                                   "SR 61's layout re-used on new states (router-discriminating, SECONDARY).  Scene 1 "
                                                   "(optional item 4) not run.  Ledger tree dirty (other agents' files).",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main (k = 4-63) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_fix2.py"))

"""PRE-REGISTRATION ROW of F6O2 (SCAFFOLD_RESULTS 45.6): F6O (45.5) on NEW displaced-layout states, to settle the registered
equivalence bound, plus arm M, a structural memory overlay, on frozen pi0.5 and GR00T.  BEFORE any main-state f6o2 rollout.

    python experiments/prereg/f6o2.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "f6o2" for r in rows), "already registered"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "f6o2", "roll")), "main rollouts exist"
for fn in ("servo_2_main.json", "pmin_2_main_s10.json"):
    assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "f6o2", "ref", fn)), "main reference exists"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "f6o2_read_main.json")), "main read exists"
meta = json.load(open(os.path.join(ROOT, "results", "flawed", "f6o2", "f6o2_tasks.json"), encoding="utf-8"))
assert len(meta["states"]["main"]) == 120 and len(meta["states"]["smoke"]) == 4


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SEEN = ("Smoke, pi0.5 only, new smoke rows 24-27 (job 4701598; read job 4701837, .claude_tmp/pp/f6o2_read_smoke.json), read after "
        "the prediction was written; nothing in the mapping changed after it: a 1.00, b .50, c .75, d 1.00, M .75; reference 1.00 "
        "(P_min 1.00, servo .00 on these 4 states). Failed b / c / M bowls end near the plate. Plate visible at later queries: b .64, "
        "c .38, M .42 (true plate pixels, the occluder leak). Smoke reference also run (jobs 4701599, 4701600). GR00T not smoked. "
        "Also seen: f6o's n = 20 result (SR 45.5).")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="f6o2", seed=0,
                    command="pre-registration: F6O2, F6 on new displaced states + arm M memory overlay (docs/SCAFFOLD_RESULTS.md 45.6)",
                    metrics={"status": "PRE-REGISTERED F6O2",
                             "design": "f6o's covers, conditions a/b/c/d, park and beside rules, frozen policies and servers, on 124 NEW "
                                       "displaced-layout states drawn by fx's F6 rule (fam np.random.seed = env.seed = 810000 + k, disp "
                                       "default_rng(811000 + k), k = 24..147; the rule reproduces fx's stored k = 0-23 exactly; none "
                                       "duplicates a stored F6 state; 0 of 124 rejected by f6o's render check). Rows 24-27 = smoke, "
                                       "rows 28-147 = main (120 states). Branch seed_base 14000000 + 1000 k. Arm M = condition b + a "
                                       "memory overlay: at the first query the learned segmenter's plate mask (agentview) and its "
                                       "depth back-projected 3D centre (fx_r.centre) are stored with the RGB sprite; from the cover's "
                                       "move on, both cameras get the remembered plate painted at its remembered position (the wrist "
                                       "view by reprojecting the agentview sprite through the plate plane with the camera model), "
                                       "except where the visible surface is > 12 cm above the plane, in the robot's own gripper column "
                                       "(within 10 cm in xy of its proprioceptive eef position and higher than 3 cm below it), or "
                                       "segmented as bowl. Overlay check (f6o2_mcheck.py, DUMMY rollouts, painted vs a's true plate "
                                       "pixels at queries after the move, 124 states): agentview IoU median .987 (min .86); wrist at "
                                       "the home pose median .47 (.56 where the plate has >= 200 wrist px; the plate is a sliver at "
                                       "the wrist image edge there); scripted descent over the plate, wrist IoU .88-.90 at 25/15/10/6 "
                                       "cm above the cover. Segmenter plate IoU at the first query median .987, centre xy error 1.7 "
                                       "mm. Reference = fx_ref (servo seeds 10-12, P_min s10-s12) on "
                                       "the new states, better system.",
                             "mapping": "experiments/f6o2_read.py as hashed, on the NEW main states alone (120). Per model, first "
                                        "matching row: NO POWER iff s(a) < .50 or reference < .60; MEMORY ABSENT iff s(d) - s(b) > .10 "
                                        "with CI lower bound > 0 and s(b) - s(c) CI upper bound < .10; MEMORY USED iff s(b) - s(c) > .10 "
                                        "with lower bound > 0; else PARTIAL (f6o_read.verdict). Fix: NO POWER if F6 row is NO POWER; "
                                        "MEMORY RESTORED iff s(M) - s(b) > .10 with lower bound > 0 and s(M)'s CI lower bound >= s(b)'s "
                                        "CI upper bound; else NO VERDICT (M - b reported). Also reported: M - d, M - c, a - d, the F6 "
                                        "rows pooled with f6o's states 4-23 (descriptive). Paired over states, bootstrap B = 2,000, "
                                        "seed 0. F6 HOLDS iff MEMORY ABSENT for both pi0.5 and GR00T; F6 FIX iff MEMORY RESTORED for both.",
                             "power_note": "At n = 120 the b - c CI half-width is about +-.11 for pi0.5 (b, c near .5) and +-.07 for GR00T "
                                           "(b, c near .1), so MEMORY ABSENT on pi0.5 needs a b - c point estimate at or below about "
                                           "-.01.",
                             "caveat": "The occluder leaks: in f6o the plate had pixels at 36-56% of later b/c queries (the policy "
                                       "disturbs the plate or the wrist reaches the cover). Kept as is. The cover's top lies about 8 cm "
                                       "above the plate plane measured from depth. M's occlusion rule changed twice before any "
                                       "rollout, both times after a policy-free check: height gate 3 cm -> 12 cm (the first build "
                                       "painted nothing), and a 10 cm eef sphere -> the gripper column (the sphere removed the overlay "
                                       "under a descending gripper). f6o2_tasks.json checks.M / M_descent hold the sphere version; "
                                       "results/flawed/f6o2/mcheck.json holds the registered rule. The wrist sprite is the agentview "
                                       "patch upsampled (a blockier plate rim than the real one).",
                             "prediction": "F6 does not HOLD: GR00T MEMORY ABSENT, pi0.5 PARTIAL (its b - c upper bound stays above .10). "
                                           "F6 FIX: MEMORY RESTORED on both models (pi0.5 M near d, about .9; GR00T M about .5 against "
                                           "b about .1).",
                             "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in ["experiments/f6o2_tasks.py", "experiments/f6o2_client.py",
                                                                   "experiments/f6o2_read.py", "experiments/f6o2_ref.py",
                                                                   "experiments/f6o2_mcheck.py", "results/flawed/f6o2/f6o2_tasks.json",
                                                                   "results/flawed/f6o2/mcheck.json"]},
                             "written_before": "any main-state f6o2 rollout and the main-state reference"},
                    bytes=Z, config={}, script="experiments/prereg/f6o2.py"))

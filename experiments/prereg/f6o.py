"""PRE-REGISTRATION ROW of F6O (SCAFFOLD_RESULTS 45.5): F6 (VLA memory) with a physical occluder in every condition, on frozen
pi0.5 and GR00T, LIBERO-Goal scene, fx's F6 states (fam, disp) and branch seeds.  BEFORE any main-state f6o rollout.

    python experiments/prereg/f6o.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "f6o" for r in rows), "already registered"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "f6o", "roll")), "main rollouts exist"
assert not os.path.exists(os.path.join(ROOT, "results", "flawed", "f6o_read_main.json")), "main read exists"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="f6o", seed=0,
                    command="pre-registration: F6O, F6 memory with a physical occluder (docs/SCAFFOLD_RESULTS.md 45.5)",
                    metrics={"status": "PRE-REGISTERED F6O",
                             "design": "Two identical opaque non-colliding cylinder covers (radius .1135 m, top 8 mm above the plate), "
                                       "in the scene in every condition. a visible: both parked. b occluded-late: cover 0 moves onto the "
                                       "plate at env step WAIT+HIDE_AT (after the first query sees the plate). c never-visible: cover 0 on "
                                       "the plate from step 0. d moved-beside: cover 0 moves at b's step to a spot right beside the plate "
                                       "(ring just outside it, about as far from the bowl as the plate) that leaves plate and bowl visible "
                                       "(render check job 4697889: 0 violations; d keeps >= .99 of a's plate pixels at every query in "
                                       "both cameras and 1.0 of the bowl's). Physics identical across conditions. Reference = fx's "
                                       "frozen-plate servo / P_min (better system, fx tid 0 fam, 2 disp).",
                             "mapping": "experiments/f6o_read.py as hashed. Per model and layout, first matching row: NO POWER iff s(a) < "
                                        ".50 or reference < .60; MEMORY ABSENT iff s(d) - s(b) > .10 with CI lower bound > 0 and s(b) - "
                                        "s(c) CI upper bound < .10; MEMORY USED iff s(b) - s(c) > .10 with CI lower bound > 0; else "
                                        "PARTIAL. Paired over states, bootstrap B = 2,000, seed 0. s(a) - s(d) reported as the "
                                        "disc-appearance cost, no verdict. F6 HOLDS iff MEMORY ABSENT on disp for both pi0.5 and GR00T.",
                             "headline_layout": "disp. In fam the cover on the plate also hides most of the bowl (bowl agentview px c/a "
                                                ".07-.49), so fam is descriptive only (same mapping, not read as a verdict).",
                             "prediction": "F6 HOLDS (written before the smoke was read): on disp both VLAs succeed in a and d "
                                           "(pi0.5 near 1.0, GR00T near fx's disp_vis .65), fail in b about as often as in c, "
                                           "and carry the bowl to the plate's training spot in failed b/c episodes; disc cost "
                                           "a - d within .10. fam (descriptive): b and c low as the bowl is hidden too.",
                             "seen": "Smoke on states 0-3 (read job 4698476, .claude_tmp/pp/f6o_read_smoke.json), read after the "
                                     "prediction was written; nothing in the mapping changed after it. disp: pi0.5 a 1.0 b .25 "
                                     "c .25 d 1.0 (MEMORY ABSENT); GR00T a .50 b 0 c 0 d .25 (PARTIAL); reference .75 (P_min). "
                                     "fam: pi0.5 a 1.0 b/c/d .75; GR00T a 1.0 b 0 c 0 d 1.0. Failed b/c bowls end near the true "
                                     "plate, never at the training spot. The client's in-rollout track shows the plate with "
                                     "pixels at 16-55% of later b/c queries (the policy disturbs the plate or the wrist camera "
                                     "reaches the cover; the cover does not follow the plate): a leak toward b, reported, no "
                                     "design change.",
                             "files_sha256": {p: fsha(p) for p in ["experiments/f6o_tasks.py", "experiments/f6o_client.py",
                                                                   "experiments/f6o_read.py"]},
                             "written_before": "any main-state f6o rollout"}, bytes=Z, config={}, script="experiments/prereg/f6o.py"))

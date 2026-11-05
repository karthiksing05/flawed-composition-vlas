"""PRE-REGISTRATION ROWS of arm LR (pp_hand4) and the figure-ground plate (pp_loc3); docs/LOCATOR.md 11-12.  BEFORE any non-smoke run of either.

    python experiments/prereg/pp_hand4.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
for tag, res in (("pp_hand4", "pp_hand4_read.json"), ("pp_loc3", "pp_loc3.json")):
    assert not os.path.exists(os.path.join(ROOT, "results", "flawed", res)), "read exists"
    assert not any(r.get("tag") == tag for r in rows), f"{tag} already registered"


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_hand4", seed=0, command="pre-registration: arm LR (docs/LOCATOR.md 11)",
                    metrics={"status": "PRE-REGISTERED arm LR",
                             "labelled": "post hoc; development read; no training; the 20 states shaped the locator; changes no earlier verdict; the user asked to fix the estimate and keep iterating, details not reviewed",
                             "mapping": "per VLA and layout, N2 N3 N5 pooled over locator seeds: RAISES iff s(LR)-s(intact) has CI lower bound above 0 and exceeds 0.10; PERCEPTION HOLDS iff RAISES and s(H1)-s(LR) <= 0.10; PERCEPTION COSTS iff RAISES and s(H1)-s(LR) > 0.10; NO FIX iff not RAISES. KNOWN COST qualifies iff K4 and K6 lose more than 0.05 against H1",
                             "prediction": "LR recovers most of LS's loss in the new layout and on K6; PERCEPTION HOLDS on both familiar cases; NO FIX in the new layout",
                             "states": "dev only: N2 main states 4-23",
                             "files_sha256": {p: fsha(p) for p in ["docs/LOCATOR.md", "experiments/pp_hand4_client.py", "experiments/pp_hand3_client.py", "experiments/pp_handx_roll.sh", "experiments/pp_hand4_read.py"]},
                             "written_before": "any rollout outside the smoke"}, bytes=Z, config={}, script="experiments/prereg/pp_hand4.py"))
print(ledger.record("m3_prereg", bench="libero_goal", tag="pp_loc3", seed=0, command="pre-registration: figure-ground plate (docs/LOCATOR.md 12)",
                    metrics={"status": "PRE-REGISTERED figure-ground plate",
                             "labelled": "post hoc; development read; no rollouts; the 20 states shaped this design; changes no earlier verdict; the user asked to keep iterating, details not reviewed",
                             "mapping": "on arm CP3 alone, LOCATOR 5's mapping, first match: WITHIN BUDGET iff every head has e <= 0.02 m in both layouts; BEATS A TABLE, OVER BUDGET iff every head has e <= T/2 in the new layout; otherwise NO BETTER THAN A TABLE. AP3 takes the label as a report",
                             "prediction": "CP3 BEATS A TABLE, OVER BUDGET; the plate within 0.05 m; the bowl's shift and the familiar bottle still over budget",
                             "states": "dev only: N2 main states 4-23, never in training; training on K1-K6 demos 0-39, familiar layout",
                             "files_sha256": {p: fsha(p) for p in ["docs/LOCATOR.md", "experiments/pp_loc3.py", "experiments/pp_loc2.py", "experiments/pp_loc.py"]},
                             "written_before": "any non-smoke training of CP3 or AP3"}, bytes=Z, config={}, script="experiments/prereg/pp_hand4.py"))

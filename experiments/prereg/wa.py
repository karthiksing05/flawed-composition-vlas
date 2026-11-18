"""PRE-REGISTRATION ROW of WA (SR 79; docs/notes/wa_plan.md): module-removal test (R referent resolver, M pose memory) on the existing RAA checkpoint, rg_comp's rgc
main states.  BEFORE any main rollout.  Predictions: experiments/prereg/wa_pred.txt.

    python experiments/prereg/wa.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "wa" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "wa")), "wa rollouts exist"
assert not os.path.exists(os.path.join(FL, "wa_read.json")), "wa read exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


RGC = [r for r in rows if r.get("tag") == "rg_comp"]
assert len(RGC) == 1
H0 = RGC[0]["metrics"]["files_sha256"]
STATE = [p for p in H0 if p.startswith("results/flawed/rg_comp/") and (p.endswith(".npy") or "br" in os.path.basename(p) or p.endswith("tasks.json"))]
for p in STATE + ["results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json", "experiments/fx_r.py", "experiments/rg_comp.py", "experiments/f6o2_tasks.py"]:
    assert H0[p] == fsha(p), ("differs from the rg_comp row", p)
print("[wa] rgc states, branches, tasks, calibration, lexicon, fx_r, rg_comp match the rg_comp row", flush=True)
PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/wa.py", "experiments/wa_read.py", "experiments/prereg/wa.py", "experiments/prereg/wa_pred.txt", "docs/notes/wa_plan.md", "experiments/fx_r.py",
        "experiments/rg_comp.py", "experiments/sc_rv.py", "experiments/sc_pu.py", "experiments/sc_seg.py", "experiments/sc_t16.py", "experiments/fa1_common.py",
        "experiments/libero_client.py", "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/serve.py",
        "lifelong_ood/policy/l90c_config.py", "results/flawed/sc_lex.json", "results/flawed/n2_tasks.json", PP + "wa_roll.sh"] + STATE
PRED = open(os.path.join(ROOT, "experiments", "prereg", "wa_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-18.  Known when written: SR 70, SR 73, the smoke (4 states per cell, jobs 4790728 / 4790729 / 4790846, outputs outside results/) summarized in "
        "wa_pred.txt.  wa_read selftest (job 4790804) reproduces SR 70's rc - I on F6_b and rc - rgp on SP / NG.  Smoke changes before this row: noM defined over all held "
        "estimates (occl, last, prior), noMe added, noR rolled on K / F6 cells as a replicate, unique-name gate cells added.  No code, arm, mapping or prediction changed after "
        "this row.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="wa", seed=0,
                    command="pre-registration: WA (SR 79), R / M removal on RAA's checkpoint, rgc main states (docs/notes/wa_plan.md)",
                    metrics={"status": "PRE-REGISTERED WA",
                             "arm": "RAA checkpoint pi05_g8v_raa_ae_tl6000, no training; arms full, noR, noM, noMe (docs/notes/wa_plan.md)",
                             "tasks": "rgc tids 0-3 (F7), 4-6 (gate), 7 K2t, 8 F6_b, 9 F6_a, 10 K1t; main states of results/flawed/rg_comp",
                             "mapping": "experiments/wa_read.py docstring (D1, D2, D3a, D3b, DK, primary, gates)",
                             "prediction": PRED, "seen": SEEN, "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main rollout of WA"},
                    bytes=Z, config={}, script="experiments/prereg/wa.py"))

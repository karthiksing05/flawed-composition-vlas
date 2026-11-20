"""PRE-REGISTRATION ROW of A2 (SR 83; docs/notes/a2_plan.md): the distributedness probe, open-loop activation patching and band ablation on recorded model inputs of frozen pi0.5 and
RAA-sL, no training.  BEFORE any main capture or main probe.  Predictions: experiments/prereg/a2_pred.txt.

    python experiments/prereg/a2.py
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
assert not any(r.get("tag") == "a2" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "a2", "a2_read_pi05.json")), "a2 read exists"
assert not any(n.startswith("cap_") for n in os.listdir(os.path.join(FL, "a2"))), "a2 main captures exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATES = ["results/flawed/a2/br7_a2.json", "results/flawed/n2_branches_main.json", "results/flawed/n2_tasks.json", "results/flawed/rg_comp/rg_comp_tasks.json",
          "results/flawed/rg_comp/cache_f7.pkl", "results/flawed/sc_lex.json"]
CODE = ["experiments/a2_capture.py", "experiments/a2_cap_client.py", "experiments/a2_probe.py", "experiments/a2_read.py", "experiments/prereg/a2.py", "experiments/prereg/a2_pred.txt",
        "docs/notes/a2_plan.md", "experiments/steer_serve.py", "experiments/steer_conf_serve.py", "experiments/steer_probe.py", "experiments/steer_text.py", "experiments/raa.py",
        "experiments/wa_s.py", "experiments/rg_comp.py", "experiments/pp_serve_purify.py", "experiments/libero_client.py", "experiments/fc_n2_client.py",
        "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/serve.py", "lifelong_ood/policy/raas_model.py", PP + "a2_cap.sh", PP + "gpu.sh", PP + "cpu.sh"] + STATES
PRED = open(os.path.join(ROOT, "experiments", "prereg", "a2_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-19.  Known when written: SR 74 / 75 / 82 (steerability, role channel, F7), and the plumbing smoke: a 4-episode capture per model per route (jobs 4808120-4808123, "
        "results outside results/ under .claude_tmp/pp/a2_smoke) and a probe of 5 observations per model (jobs 4808141, 4808166), read for plumbing numbers only (replay vs the server .002-.006, "
        "identity patches .001-.006, finite chunks); the effect statistics of the smoke were not computed.  a2_read selftest (synthetic, including the full analysis) passed.  No code, "
        "set, model, component, band, statistic or prediction changed after this row.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="a2", seed=0,
                    command="pre-registration: A2 (SR 83), distributedness probe, pi0.5 vs RAA-sL, activation patching + band ablation, F7 / N / K (docs/notes/a2_plan.md)",
                    metrics={"status": "PRE-REGISTERED A2",
                             "models": "pi05 = frozen pi05_libero (pp_serve_purify intact), raasl = pi05_g8v_raas_l_ae main6000; no training",
                             "sets": "F7 (rgc tids 0 / 1, trials 4-23, row 0), N (tids 16 22 24 26), K (tids 0 2 6 10): 3rd held query, first 12 per task with an equal-token swap candidate",
                             "mapping": "experiments/a2_read.py docstring and docs/notes/a2_plan.md (statistics, labels, bootstrap B = 2000, rng 0)",
                             "prediction": PRED, "seen": SEEN, "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main capture or main probe of A2"},
                    bytes=Z, config={}, script="experiments/prereg/a2.py"))

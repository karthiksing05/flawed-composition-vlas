"""PRE-REGISTRATION ROW of RAA-sL-W5 / W5c2 / W5c1 (SR 87; docs/notes/raasw5_plan.md): confirmatory run, on FRESH start states, of test-time-only variants of the RAA-sL checkpoint (SR 82):
W5 = W4 + C1 (aim at the demos' release point of the target class) + C2 (the module owns xy only while |rel - mu| > 3 cm; release gate from the release point, radius 3 cm); W5c2 = W4 + C2; W5c1 = W4 + C1.  No training.
BEFORE any W5 / W5c2 / W5c1 / fresh-state baseline rollout.  Predictions: experiments/prereg/raasw5_pred.txt (committed before this row).

    python experiments/prereg/raasw5.py        (login-free: run through sbatch)
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CKPT = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_g8v_raas_l_ae_main6000"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raasw5" for r in rows), "already registered"
assert sum(1 for r in rows if r.get("tag") == "raasl") == 1, "SR 82 row missing"
assert sum(1 for r in rows if r.get("tag") == "raasw3") == 1, "SR 85 row missing"
for d in ("raasw5", "wasw5", "raasw5_read.json", "raasw5_wa_gated.json"):
    assert not os.path.exists(os.path.join(FL, d)), d + " exists"
assert os.path.isdir(CKPT), CKPT
for p in ("n2_states_wx/states.npy", "n2_states_wx/meta.json", "n2_branches_wx.json", "n2_states_wx/n2_goal_0.npy"):
    assert os.path.exists(os.path.join(FL, p)), p


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


L = [r for r in rows if r.get("tag") == "raasl"][0]["metrics"]["files_sha256"]
for p in ("experiments/raa.py", "experiments/wa_s.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py",
          "lifelong_ood/policy/serve.py", "results/flawed/raas_gate.json", "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json", "results/flawed/sc_lex.json"):
    if p in L:
        assert L[p] == fsha(p), ("differs from the SR 82 (raasl) row's hash", p)
print("[raasw5] serving code and gate constants match the SR 82 row", flush=True)
CODE = ["experiments/raa.py", "experiments/wa_s.py", "experiments/raa_w.py", "experiments/wa_sw.py", "experiments/wa_sw5.py", "experiments/rg_comp.py", "experiments/rg_comp_x.py", "experiments/fc_n2_wx.py",
        "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/backbone_w.py", "lifelong_ood/policy/raasw5_mu.json", "experiments/raasw5_mu.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py",
        "lifelong_ood/policy/serve.py", "experiments/raasw5_test.py", "experiments/raasw5_read.py", "experiments/raa_read.py", "experiments/wa_read.py", "experiments/prereg/raasw5_pred.txt",
        "docs/notes/raasw5_plan.md", "docs/notes/sr86_diag.md", "results/flawed/raas_gate.json", "results/flawed/n2_states_wx/states.npy", "results/flawed/n2_branches_wx.json",
        PP + "raaw5_roll.sh", PP + "wasw5_roll.sh", PP + "rg5_roll.sh", PP + "rgcx_roll.sh", PP + "w5_states.sh", PP + "raasw5_test.sh", PP + "raasw5_prereg.sh"]
TESTLOG = [f for f in sorted(os.listdir(PP)) if f.startswith("raasw5_test_") and f.endswith(".out")]
assert TESTLOG, "run experiments/raasw5_test.py first"
tl = open(PP + TESTLOG[-1], encoding="utf-8", errors="replace").read()
assert "OVERALL PASS" in tl, "the bit-identity test did not pass"
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raasw5_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-20.  Known when written: SR 82 / 84 / 85 results on the OLD states (incl. per-cell K / N / F tables), the SR 86 exploratory diagnostic (docs/notes/sr86_diag.md; instrumented K5 / F7 runs on the old states) and the training demos' release points "
        "(SR 86 q1d; raasw5_mu.json, computed once from the training demos only before any W5 episode).  The gate radius (3 cm = R, not 2 cm) and R = 3 cm were fixed in the plan from SR 86's numbers, before any W5 rollout.  Fresh states: Phase A k = 580-639 "
        "(drawn and checked before this row; hashes below); Phase B rg_comp_x.py (F7 k = 380+, F6 k = 500+; the rule and ranges are fixed in the committed file, the build was still running when this row was written, so its state files are not hashed here).  "
        "Checks before the row, plumbing and not results: the bit-identity / semantics test (%s: OVERALL PASS; modes 0-4 equal to the SR 85 code, modes 5 / 6 / 7 as specified).  No smoke rollouts were run; no W5 episode exists." % TESTLOG[-1])
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raasw5", seed=0,
                    command="pre-registration: RAA-sL-W5 / W5c2 / W5c1 (SR 87), test-time-only serving of the RAA-sL checkpoint, confirmatory run on FRESH start states (N1-N6 / K1-K6 on n2 familiar states k = 580-639; the WA arms (wa_sw5.py) on rg_comp_x's fresh rgc states), "
                            "baselines RAA-sL, W4, rgp, I re-rolled on the same states (docs/notes/raasw5_plan.md)",
                    metrics={"status": "PRE-REGISTERED RAA-sL-W5 / W5c2 / W5c1",
                             "arm": "W5 = the SR 82 checkpoint pi05_g8v_raas_l_ae main/6000 (%s), NO training; serving only: while a target is held and c = -dz >= 3 cm and |rel_xy - mu_class| > 3 cm the action's xy = clip(K_w * (rel_xy - mu_class), +-servo_max) "
                                    "(K_w = 46.875 / m, no expert xy), otherwise RAA-sL's chunk; mu_class = the training demos' median channel rel xy at the release frame (lifelong_ood/policy/raasw5_mu.json); an open gripper command is replaced by closed "
                                    "while held and |rel_xy - mu_class| > 3 cm.  W5c2: mu = 0.  W5c1: no handoff radius, gate 2 cm from mu_class." % CKPT,
                             "states": "phase A fresh n2 familiar states k = 580-639 (harness trials 0-59); phase B rg_comp_x fresh rgc main states (F7 60, F6 60 per cell)",
                             "tasks": "phase A n2 tids (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26); phase B rgc tids 0-10",
                             "mapping": "docs/notes/raasw5_plan.md 'Reads' (read: experiments/raasw5_read.py)",
                             "prediction": PRED, "seen": SEEN, "test_log_tail": tl[-1800:],
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main W5 / W5c2 / W5c1 / fresh-state baseline rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raasw5.py"))

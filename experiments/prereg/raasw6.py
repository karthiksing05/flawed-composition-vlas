"""PRE-REGISTRATION ROW of RAA-sL-W6 / W6nl (SR 90; docs/notes/raasw6_plan.md): confirmatory run, on the THIRD fresh start-state set, of test-time-only variants of the RAA-sL checkpoint (SR 82):
W6 = W5c1 (aim at the demos' release point of the target class; W4's CLEAR) + release-envelope hand-off (the module owns xy and the release gate while c > h_class or |rel - mu| > r_class) + target latch; W6nl = W6 without the latch.
No training.  BEFORE any W6 / W6nl / third-fresh-state baseline rollout.  Predictions: experiments/prereg/raasw6_pred.txt (committed before this row).

    python experiments/prereg/raasw6.py        (login-free: run through sbatch)
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CKPT = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_g8v_raas_l_ae_main6000"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raasw6" for r in rows), "already registered"
for t in ("raasl", "raasw3", "raasw5"):
    assert sum(1 for r in rows if r.get("tag") == t) == 1, t + " row missing"
for d in ("raasw6", "raasw6_read.json", "raasw6_wa_gated.json"):
    assert not os.path.exists(os.path.join(FL, d)), d + " exists"
assert os.path.isdir(CKPT), CKPT
for p in ("n2_states_wy/states.npy", "n2_states_wy/meta.json", "n2_branches_wy.json", "n2_states_wy/n2_goal_0.npy"):
    assert os.path.exists(os.path.join(FL, p)), p


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


L = [r for r in rows if r.get("tag") == "raasl"][0]["metrics"]["files_sha256"]
for p in ("experiments/raa.py", "experiments/wa_s.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py",
          "lifelong_ood/policy/serve.py", "results/flawed/raas_gate.json", "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json", "results/flawed/sc_lex.json"):
    if p in L:
        assert L[p] == fsha(p), ("differs from the SR 82 (raasl) row's hash", p)
W5 = [r for r in rows if r.get("tag") == "raasw5"][0]["metrics"]["files_sha256"]
for p in ("experiments/raa_w.py", "experiments/wa_sw.py", "experiments/wa_sw5.py", "experiments/rg_comp_x.py", "lifelong_ood/policy/raasw5_mu.json", "experiments/fc_n2_wx.py"):
    assert W5[p] == fsha(p), ("differs from the SR 87 (raasw5) row's hash", p)
print("[raasw6] serving code, SR 87 files and gate constants match the SR 82 / SR 87 rows", flush=True)
env = json.load(open(os.path.join(ROOT, "lifelong_ood", "policy", "raasw6_env.json"), encoding="utf-8"))
CODE = ["experiments/raa.py", "experiments/wa_s.py", "experiments/raa_w.py", "experiments/wa_sw.py", "experiments/wa_sw5.py", "experiments/raa_w6.py", "experiments/wa_sw6.py", "experiments/rg_comp.py", "experiments/rg_comp_x.py",
        "experiments/rg_comp_y.py", "experiments/fc_n2_wx.py", "experiments/fc_n2_wy.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/backbone_w.py", "lifelong_ood/policy/w6_latch.py", "lifelong_ood/policy/raasw5_mu.json",
        "lifelong_ood/policy/raasw6_env.json", "experiments/raasw5_mu.py", "experiments/raasw6_env.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py", "lifelong_ood/policy/serve.py",
        "experiments/raasw6_test.py", "experiments/raasw6_read.py", "experiments/raa_read.py", "experiments/wa_read.py", "experiments/prereg/raasw6_pred.txt", "docs/notes/raasw6_plan.md", "docs/notes/sr88_diag.md",
        "results/flawed/raas_gate.json", "results/flawed/n2_states_wy/states.npy", "results/flawed/n2_branches_wy.json",
        PP + "raaw6_roll.sh", PP + "wasw6_roll.sh", PP + "rg6_roll.sh", PP + "rgcy_roll.sh", PP + "w6_states.sh", PP + "raasw6_test.sh", PP + "raasw6_prereg.sh"]
TESTLOG = [f for f in sorted(os.listdir(PP)) if f.startswith("raasw6_test_") and f.endswith(".out")]
assert TESTLOG, "run experiments/raasw6_test.py first"
tl = open(PP + TESTLOG[-1], encoding="utf-8", errors="replace").read()
assert "OVERALL PASS" in tl, "the bit-identity test did not pass"
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raasw6_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-20.  Known when written: SR 82 / 84 / 85 / 87 results (incl. per-cell tables on the second fresh set), the SR 88 exploratory diagnostic (instrumented replays of SR 87 episodes on development states) and the training demos' "
        "release statistics (raasw6_env.json: h_class / r_class / tau, computed once from the training demos only, committed before any W6 episode).  Third fresh states: Phase A k = 640-699 (drawn and checked before this row; hashes below); "
        "Phase B rg_comp_y.py (F7 candidates k = 520-599, F6 k = 620-699; the rule and ranges are fixed in the committed file; the build was still running when this row was written, so its state files are not hashed here).  Checks before the row, plumbing and not results: the bit-identity / semantics test (%s: OVERALL PASS; modes 0-7 equal to the SR 85 / SR 87 code, "
        "modes 8 / 9 and the latch as specified).  No smoke rollouts were run; no W6 / W6nl episode exists." % TESTLOG[-1])
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raasw6", seed=0,
                    command="pre-registration: RAA-sL-W6 / W6nl (SR 90), test-time-only serving of the RAA-sL checkpoint, confirmatory run on the THIRD fresh start-state set (N1-N6 / K1-K6 on n2 familiar states k = 640-699; the WA arms (wa_sw6.py) on rg_comp_y's fresh rgc states), "
                            "baselines W5, W5c1, RAA-sL, rgp, I re-rolled on the same states (docs/notes/raasw6_plan.md)",
                    metrics={"status": "PRE-REGISTERED RAA-sL-W6 / W6nl",
                             "arm": "W6 = the SR 82 checkpoint pi05_g8v_raas_l_ae main/6000 (%s), NO training; serving only: while a target is held and c = -dz >= 3 cm and the object is outside the class's demo release envelope (c > h_class or |rel_xy - mu_class| > r_class) the "
                                    "action's xy = clip(K_w * (rel_xy - mu_class), +-servo_max) (K_w = 46.875 / m, no expert xy), otherwise RAA-sL's chunk; an open gripper command is replaced by closed while held and outside the envelope; the target estimate is latched at carry start "
                                    "and a later estimate is accepted only if it moves <= tau (server side).  W6nl: no latch.  Constants (lifelong_ood/policy/raasw6_env.json): %s" % (CKPT, json.dumps({k: env[k] for k in ("h_class_m", "r_class_m", "tau_target_m")})),
                             "states": "phase A third fresh n2 familiar states k = 640-699 (harness trials 0-59); phase B rg_comp_y third fresh rgc main states (F7 60, F6 60 per cell)",
                             "tasks": "phase A n2 tids (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26); phase B rgc tids 0-10",
                             "mapping": "docs/notes/raasw6_plan.md 'Registered reads' (read: experiments/raasw6_read.py)",
                             "prediction": PRED, "seen": SEEN, "test_log_tail": tl[-1800:],
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main W6 / W6nl / third-fresh-state baseline rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raasw6.py"))

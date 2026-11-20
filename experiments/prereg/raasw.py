"""PRE-REGISTRATION ROW of RAA-sL-W (SR 84; docs/notes/raasw_plan.md): test-time-only variants W1 / W2 of the RAA-sL checkpoint (SR 82) in which the xy of transport is owned by the
role module (saturating servo, K_w = servo_max / 0.02 m), W2 adding a release gate.  No training.  BEFORE any main W1 / W2 rollout (smoke <= 4 episodes per arm was allowed and is not read).
Predictions: experiments/prereg/raasw_pred.txt (committed before this row).

    python experiments/prereg/raasw.py        (login-free: run through sbatch)
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
assert not any(r.get("tag") == "raasw" for r in rows), "already registered"
assert sum(1 for r in rows if r.get("tag") == "raasl") == 1, "SR 82 row missing"
assert not any(f.startswith("raasw") for f in os.listdir(os.path.join(FL, "raa"))), "raasw rollouts exist"
for d in ("wasw1", "wasw2"):
    assert not os.path.exists(os.path.join(FL, d)), d + " rollouts exist"
assert not any(f.startswith("raasw") for f in os.listdir(FL)), "raasw read exists"
assert os.path.isdir(CKPT), CKPT


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


L = [r for r in rows if r.get("tag") == "raasl"][0]["metrics"]["files_sha256"]
for p in ("experiments/raa.py", "experiments/wa_s.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py",
          "lifelong_ood/policy/serve.py", "results/flawed/raas_gate.json", "results/flawed/n2_branches_rx.json", "results/flawed/n2_branches_main.json",
          "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json", "results/flawed/sc_lex.json", "results/flawed/n2_states_rx/states.npy"):
    if p in L:
        assert L[p] == fsha(p), ("differs from the SR 82 (raasl) row's hash", p)
print("[raasw] serving code, gate constants and phase A / B state files match the SR 82 row", flush=True)
CODE = ["experiments/raa.py", "experiments/wa_s.py", "experiments/raa_w.py", "experiments/wa_sw.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/backbone_w.py",
        "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py", "lifelong_ood/policy/serve.py", "experiments/raasw_test.py", "experiments/raasw_read.py",
        "experiments/raasl_read.py", "experiments/raa_read.py", "experiments/wa_read.py", "experiments/prereg/raasw_pred.txt", "docs/notes/raasw_plan.md", "results/flawed/raas_gate.json",
        PP + "raaw_roll.sh", PP + "wasw_roll.sh", PP + "raasw_test.sh", PP + "raasw_prereg.sh", PP + "raa_roll.sh", PP + "wasl_roll.sh"]
TESTLOG = [f for f in sorted(os.listdir(PP)) if f.startswith("raasw_test_") and f.endswith(".out")]
assert TESTLOG, "run experiments/raasw_test.py first"
tl = open(PP + TESTLOG[-1], encoding="utf-8", errors="replace").read()
assert "OVERALL PASS" in tl, "the bit-identity test did not pass"
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raasw_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-19.  Known when written: SR 82 and its diagnostic (RAA-sL N1-N6 .878, F6_b full .017, noM .383; expert xy opposes the servo near the covered target).  Checks before the row, "
        "plumbing and not results: the bit-identity / semantics test (%s: OVERALL PASS) and a serving smoke of at most 4 episodes per arm outside results/ (never read for success).  "
        "K_w = .9375 / .02 m was fixed in the plan before any W rollout; no W cell was run with any other value." % TESTLOG[-1])
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raasw", seed=0,
                    command="pre-registration: RAA-sL-W (SR 84), test-time-only W1 / W2 serving of the RAA-sL checkpoint (xy of transport owned by the module; W2 + release gate), "
                            "N1-N6 / K1-K6 on SR 73's states and the WA arms (wa_sw.py) on rg_comp's rgc main states (docs/notes/raasw_plan.md)",
                    metrics={"status": "PRE-REGISTERED RAA-sL-W",
                             "arm": "W1 / W2 = the SR 82 checkpoint pi05_g8v_raas_l_ae main/6000 (%s), config pi05_g8v_raas_l_ae, NO training; serving only: while a target is held (role id 2) the action's xy "
                                    "= clip(K_w * rel_xy, +-servo_max) with K_w = servo_max / 0.02 m = 46.875 per m and no expert xy; z, rotation, gripper as RAA-sL; W2 adds: while held and |rel_xy| > 2 cm an open "
                                    "gripper command is replaced by the closed command (raw +1)" % CKPT,
                             "states": "phase A as SR 73 / SR 82 (dev trials 4-23 + new k 540-579); phase B rg_comp rgc main states",
                             "tasks": "phase A n2 tids (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26); phase B rgc tids 0-10",
                             "mapping": "docs/notes/raasw_plan.md 'Registered reads' (read: experiments/raasw_read.py)",
                             "prediction": PRED, "seen": SEEN, "test_log_tail": tl[-1800:],
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main W1 / W2 rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raasw.py"))

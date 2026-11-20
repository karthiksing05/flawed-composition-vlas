"""PRE-REGISTRATION ROW of RAA-sL-W3 / W4 (SR 85; docs/notes/raasw3_plan.md): test-time-only variants of the RAA-sL checkpoint (SR 82) in which the module owns the xy of a held-target transport only once the
held object is >= CLEAR = 3 cm above the target's reference height (W3), W4 adding SR 84's release gate.  No training.  BEFORE any main W3 / W4 rollout.
Predictions: experiments/prereg/raasw3_pred.txt (committed before this row).

    python experiments/prereg/raasw3.py        (login-free: run through sbatch)
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
assert not any(r.get("tag") == "raasw3" for r in rows), "already registered"
assert sum(1 for r in rows if r.get("tag") == "raasl") == 1, "SR 82 row missing"
assert sum(1 for r in rows if r.get("tag") == "raasw") == 1, "SR 84 row missing"
assert not any(f.startswith(("raasw3", "raasw4")) for f in os.listdir(os.path.join(FL, "raa"))), "raasw3 / raasw4 rollouts exist"
for d in ("wasw3", "wasw4"):
    assert not os.path.exists(os.path.join(FL, d)), d + " rollouts exist"
assert not any(f.startswith("raasw3") for f in os.listdir(FL)), "raasw3 read exists"
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
print("[raasw3] serving code, gate constants and phase A / B state files match the SR 82 row", flush=True)
CODE = ["experiments/raa.py", "experiments/wa_s.py", "experiments/raa_w.py", "experiments/wa_sw.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/backbone_w.py",
        "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py", "lifelong_ood/policy/serve.py", "experiments/raasw3_test.py", "experiments/raasw3_read.py",
        "experiments/raasw_read.py", "experiments/raasl_read.py", "experiments/raa_read.py", "experiments/wa_read.py", "experiments/prereg/raasw3_pred.txt", "docs/notes/raasw3_plan.md",
        "docs/notes/sr85_diag.md", "experiments/sr85_diag.py", "experiments/sr85_clear.py", "results/flawed/sr85_clear.json", "results/flawed/raas_gate.json",
        PP + "raaw3_roll.sh", PP + "wasw3_roll.sh", PP + "raasw3_test.sh", PP + "raasw3_prereg.sh"]
TESTLOG = [f for f in sorted(os.listdir(PP)) if f.startswith("raasw3_test_") and f.endswith(".out")]
assert TESTLOG, "run experiments/raasw3_test.py first"
tl = open(PP + TESTLOG[-1], encoding="utf-8", errors="replace").read()
assert "OVERALL PASS" in tl, "the bit-identity test did not pass"
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raasw3_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-20.  Known when written: SR 82, SR 84 (W1 / W2 results incl. per-cell K / N tables) and the SR 85 diagnostic (instrumented W1 vs RAA-sL rollouts on K3 / K5, 20 of the registered states per cell; sr85_diag.md) "
        "and the training demos' clearance distribution (sr85_clear.json).  CLEAR = 3 cm was fixed from the demos' release heights and the channel error before any W3 rollout, not from the diagnostic rollouts.  "
        "Checks before the row, plumbing and not results: the bit-identity / semantics test (%s: OVERALL PASS; modes 1 / 2 equal to SR 84's code, modes 3 / 4 as specified).  No smoke rollouts were run; no W3 / W4 episode exists." % TESTLOG[-1])
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raasw3", seed=0,
                    command="pre-registration: RAA-sL-W3 / W4 (SR 85), test-time-only serving of the RAA-sL checkpoint (xy of transport owned by the module only once the held object is >= 3 cm above the target's "
                            "reference height; W4 + release gate), N1-N6 / K1-K6 on SR 73's states and the WA arms (wa_sw.py) on rg_comp's rgc main states (docs/notes/raasw3_plan.md)",
                    metrics={"status": "PRE-REGISTERED RAA-sL-W3 / W4",
                             "arm": "W3 / W4 = the SR 82 checkpoint pi05_g8v_raas_l_ae main/6000 (%s), config pi05_g8v_raas_l_ae, NO training; serving only: while a target is held (role id 2) and c = -dz >= CLEAR = 0.03 m the action's xy "
                                    "= clip(K_w * rel_xy, +-servo_max) with K_w = 46.875 per m and no expert xy; otherwise RAA-sL's chunk; z, rotation, gripper as RAA-sL; W4 adds SR 84's gate: while held and |rel_xy| > 2 cm an open "
                                    "gripper command is replaced by the closed command (raw +1)" % CKPT,
                             "states": "phase A as SR 73 / SR 82 / SR 84 (dev trials 4-23 + new k 540-579); phase B rg_comp rgc main states",
                             "tasks": "phase A n2 tids (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26); phase B rgc tids 0-10",
                             "mapping": "docs/notes/raasw3_plan.md 'Reads' (read: experiments/raasw3_read.py)",
                             "prediction": PRED, "seen": SEEN, "test_log_tail": tl[-1800:],
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main W3 / W4 rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raasw3.py"))

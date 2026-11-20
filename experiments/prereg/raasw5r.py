"""PRE-REGISTRATION ROW of the SR 91 replication of RAA-sL-W5 (docs/notes/sr91_plan.md): W5 (where_own 5, the SR 87 design, NO code change) on a FOURTH disjoint fresh start-state set (N1-N6 / K1-K6 on n2 familiar states k = 700-759;
the WA arms on rg_comp_z's fresh rgc states), baselines RAA-sL, rgp, frozen I re-rolled on the same states; SECONDARY pooled analysis over SR 87 / SR 90 / SR 91.  No training.  BEFORE any SR 91 rollout.
Predictions: experiments/prereg/sr91_pred.txt (committed before this row).

    python experiments/prereg/raasw5r.py        (login-free: run through sbatch)
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
assert not any(r.get("tag") == "raasw5r" for r in rows), "already registered"
for t in ("raasl", "raasw3", "raasw5", "raasw6"):
    assert sum(1 for r in rows if r.get("tag") == t) == 1, t + " row missing"
for d in ("raasw5r", "raasw5r_read.json", "raasw5r_wa_gated.json", "raasw5r_pooled.json"):
    assert not os.path.exists(os.path.join(FL, d)), d + " exists"
assert os.path.isdir(CKPT), CKPT
for p in ("n2_states_wz/states.npy", "n2_states_wz/meta.json", "n2_branches_wz.json", "n2_states_wz/n2_goal_0.npy", "rg_comp_z/rg_comp_z_tasks.json", "rg_comp_z/br7_main.json", "rg_comp_z/br6_main.json"):
    assert os.path.exists(os.path.join(FL, p)), p


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


W5 = [r for r in rows if r.get("tag") == "raasw5"][0]["metrics"]["files_sha256"]
W6 = [r for r in rows if r.get("tag") == "raasw6"][0]["metrics"]["files_sha256"]
# W5's code path = the files of the SR 87 row; none may differ from SR 87 / SR 90 (backbone_w.py is SR 90's, whose identity test showed modes 1-7 equal to SR 87's)
for p in ("experiments/raa_w.py", "experiments/wa_sw.py", "experiments/wa_sw5.py", "experiments/rg_comp_x.py", "lifelong_ood/policy/raasw5_mu.json"):
    assert W5[p] == fsha(p), ("differs from the SR 87 (raasw5) row's hash", p)
for p in ("lifelong_ood/policy/backbone_w.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/raasw5_mu.json", "experiments/raa_w.py", "experiments/wa_sw5.py", "experiments/rg_comp_y.py", "experiments/raa_read.py", "experiments/wa_read.py",
          "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py", "lifelong_ood/policy/serve.py", "results/flawed/raas_gate.json"):
    assert W6[p] == fsha(p), ("differs from the SR 90 (raasw6) row's hash", p)
print("[raasw5r] W5's code path (raa_w.py, wa_sw5.py, backbone_w.py, mu table, serving files) matches the SR 87 / SR 90 rows", flush=True)
CODE = ["experiments/raa.py", "experiments/wa_s.py", "experiments/raa_w.py", "experiments/wa_sw.py", "experiments/wa_sw5.py", "experiments/wa_sw5z.py", "experiments/rg_comp.py", "experiments/rg_comp_x.py", "experiments/rg_comp_y.py", "experiments/rg_comp_z.py",
        "experiments/fc_n2_wx.py", "experiments/fc_n2_wy.py", "experiments/fc_n2_wz.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/backbone_w.py", "lifelong_ood/policy/raasw5_mu.json", "lifelong_ood/policy/raas_model.py",
        "lifelong_ood/policy/raac_model.py", "lifelong_ood/policy/serve.py", "experiments/raasw5r_read.py", "experiments/raasw5r_disjoint.py", "experiments/sr91_step0.py", "experiments/raa_read.py", "experiments/wa_read.py",
        "experiments/prereg/sr91_pred.txt", "docs/notes/sr91_plan.md", "results/flawed/raas_gate.json", "results/flawed/n2_states_wz/states.npy", "results/flawed/n2_branches_wz.json", "results/flawed/rg_comp_z/rg_comp_z_tasks.json",
        "results/flawed/rg_comp_z/br7_main.json", "results/flawed/rg_comp_z/br6_main.json",
        PP + "raaw5r_roll.sh", PP + "wasw5r_roll.sh", PP + "rg5r_roll.sh", PP + "rgcz_roll.sh", PP + "w5r_states.sh", PP + "raasw5r_prereg.sh"]
DISJ = [f for f in sorted(os.listdir(PP)) if f.startswith("cpu_raasw5r_disj_") and f.endswith(".out")]
assert DISJ, "run experiments/raasw5r_disjoint.py first"
dl = open(PP + DISJ[-1], encoding="utf-8", errors="replace").read()
assert "TOTAL EQUAL ROWS 0" in dl, "disjointness check failed"
TL = [f for f in sorted(os.listdir(PP)) if f.startswith("raasw6_test_") and f.endswith(".out") and "OVERALL PASS" in open(PP + f, encoding="utf-8", errors="replace").read()]
assert TL, "SR 90 identity test log with OVERALL PASS not found"
PRED = open(os.path.join(ROOT, "experiments", "prereg", "sr91_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-20.  Known when written: SR 87 (W5 on the second fresh set, all per-cell tables), SR 90 (the W5, RAA-sL, rgp, I arms on the third fresh set) and the Step 0 exploratory paired bootstrap on the SR 90 rollouts "
        "(experiments/sr91_step0.py; W5 - rgp N1-N6 +.047 [-.017, +.125], K -.008 [-.067, +.050]; W5 - RAA-sL N +.078 [+.011, +.161], K +.025 [-.067, +.150]); a dry run of the pooled routine on SR 87 + SR 90 (seen).  Fourth fresh states built and "
        "checked disjoint from SR 70-90 before this row (disjointness log %s: TOTAL EQUAL ROWS 0; states hashed below).  Identity of W5's code path to SR 87 / SR 90 asserted by hash; SR 90's identity test (%s: OVERALL PASS) covers modes 1-7.  "
        "No SR 91 W5 / RAA-sL / rgp / I episode on the fourth set exists; no smoke rollouts." % (DISJ[-1], TL[-1]))
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raasw5r", seed=0,
                    command="pre-registration: SR 91 replication of RAA-sL-W5 (where_own 5, SR 87 design, no code change), test-time-only serving of the RAA-sL checkpoint on a FOURTH fresh start-state set (N1-N6 / K1-K6 on n2 familiar states k = 700-759; the WA arms (wa_sw5z.py) on rg_comp_z's fresh rgc states), "
                            "baselines RAA-sL, rgp, I re-rolled on the same states; SECONDARY pooled analysis over SR 87 / SR 90 / SR 91 (docs/notes/sr91_plan.md)",
                    metrics={"status": "PRE-REGISTERED SR 91 W5 replication",
                             "arm": "W5 = the SR 82 checkpoint pi05_g8v_raas_l_ae main/6000 (%s), NO training; serving only (where_own 5): while a target is held and c = -dz >= 3 cm and |rel_xy - mu_class| > R = 3 cm the action's xy = clip(K_w * (rel_xy - mu_class), +-servo_max), otherwise RAA-sL's chunk; "
                                    "an open gripper command is replaced by closed while held and |rel_xy - mu_class| > 3 cm; mu = lifelong_ood/policy/raasw5_mu.json (class medians of the training demos' release points); unchanged from SR 87" % CKPT,
                             "states": "phase A fourth fresh n2 familiar states k = 700-759 (harness trials 0-59); phase B rg_comp_z fourth fresh rgc main states (F7 60, F6 60 per cell)",
                             "tasks": "phase A n2 tids (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26); phase B rgc tids 0-10",
                             "mapping": "docs/notes/sr91_plan.md 'Registered contrasts' and 'SECONDARY' (read: experiments/raasw5r_read.py)",
                             "prediction": PRED, "seen": SEEN, "disjointness_log_tail": dl[-1500:],
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any SR 91 main rollout (W5 / RAA-sL / rgp / I on the fourth fresh set)"},
                    bytes=Z, config={}, script="experiments/prereg/raasw5r.py"))

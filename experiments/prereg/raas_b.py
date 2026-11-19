"""PRE-REGISTRATION ROW of RAA-s RE-SERVE (SR 78b): the SAME RAA-s checkpoint (pi05_g8v_raas_ae_tl6000, no retraining) served WITH its servo term, against RAA-o, RAA, RAA-c and
ReGuide-perc (rgp) on N1-N6 and K1-K6, the SAME 60 states as SR 73 / 76 / 77 / 78.  Contrasts S1a / S1b / S2 / S3a / S3b / SK and X1-X6 unchanged from SR 78 (experiments/prereg/raas.py);
X7 / X8 (78b minus 78) added unlabelled.  BEFORE any SR 78b rollout.  SR 78's registered read (RAA-s served WITHOUT servo) stands and is not touched.
Predictions: experiments/prereg/raas_b_pred.txt; plan docs/notes/raas_b_plan.md.

    python experiments/prereg/raas_b.py        (login-free: run through sbatch)
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
TIDS = (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26)
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raas_b" for r in rows), "already registered"
assert any(r.get("tag") == "raas" for r in rows), "the SR 78 row is missing"
assert not any(f.startswith("raas_b") for f in os.listdir(os.path.join(FL, "raa"))), "raas_b rollouts exist"
assert not os.path.exists(os.path.join(FL, "raas_b_read.json")), "raas_b read exists"
assert os.path.exists(os.path.join(FL, "raas_read.json")), "SR 78's read must exist (and stays)"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


RAA = [r for r in rows if r.get("tag") == "raa"]
assert len(RAA) == 1, len(RAA)
H = RAA[0]["metrics"]["files_sha256"]
for p in ("results/flawed/n2_branches_rx.json", "results/flawed/n2_branches_main.json", "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json",
          "results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json", "results/flawed/sc_lex.json", "results/flawed/n2_states_rx/states.npy",
          "results/flawed/n2_states_rx/meta.json"):
    assert H[p] == fsha(p), ("state / calibration file differs from the raa row's hash", p)
print("[raas_b] states, branches, calibration and lexicon match the raa row", flush=True)

PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CKPT = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_g8v_raas_ae_tl6000/model.safetensors"
CODE = ["experiments/raas_b_test.py", "experiments/raas_diag.py", "experiments/raas_b_read.py", "experiments/raas_read.py", "experiments/raac_read.py", "experiments/raao_read.py",
        "experiments/raa_read.py", "experiments/raa.py", "experiments/sc_rv.py", "experiments/fc_n2_client.py", "experiments/libero_client.py",
        "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raac_model.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/backbone.py",
        "lifelong_ood/policy/serve.py", "experiments/prereg/raas_b_pred.txt", "experiments/prereg/raas_pred.txt", "docs/notes/raas_b_plan.md", "docs/notes/raas_plan.md",
        "docs/notes/sr78.md", "docs/notes/sr78_diag.md", "results/flawed/raas_read.json", "results/flawed/raas_gate.json", PP + "raa_job.sh", PP + "raa_roll.sh",
        PP + "raasb_test.sh", CKPT]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raas_b_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-19.  Known when written: SR 78's read (RAA-s served WITHOUT servo: N1-N6 .028, K1-K6 .092, NOT SUPPORTED), SR 78's diagnosis (PolicyBackbone.sample never added the "
        "servo; docs/notes/sr78_diag.md), SR 77 (RAA-c N1-N6 .503, K1-K6 .814), SR 76 (RAA-o .594, RAA .656, rgp .897).  The fix (PolicyBackbone.sample adds servo_norm_torch to dims 0-2 of "
        "every chunk step when the config has a nonzero servo_k) and its tests were done BEFORE this row, none a rollout: experiments/raas_b_test.py job 4799556 (PASS): the pre-change and "
        "patched samples are np.array_equal with max|diff| 0 on fixed noise for pi05_libero (frozen pi0.5), pi05_g8v_plain_ae and pi05_g8v_raac_ae (default, full, 4 steps); for raas the "
        "patched backbone.sample equals PI0PytorchS.sample_actions on the same noise (max|diff| 0), and GuidedPolicy.infer equals unnormalize of it (5.9e-8 action units); RAA-c's state_proj "
        "token is read on the served path (backbone.sample equals PI0PytorchC.sample_actions, max|diff| 0, and zeroing the prefix state changes the chunk by up to .37).  raas_diag.py rerun "
        "(job 4799549): mean |xyz - demo| over 15 held-target demo frames, served RAA-s now .176 (was .297 without the servo; RAA-c .167).  No smoke and no rollout of SR 78b had run.  "
        "No registered quantity, state, bootstrap setting or label rule changed from SR 78; the checkpoint is the SR 78 export, unchanged.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raas_b", seed=0,
                    command="pre-registration: RAA-s re-serve (SR 78b), the SR 78 checkpoint served with its servo, against RAA-o, RAA, RAA-c and ReGuide-perc on N1-N6 and K1-K6, same states "
                            "and contrasts as SR 78 (docs/notes/sr78b.md)",
                    metrics={"status": "PRE-REGISTERED RAA-S RE-SERVE",
                             "arm": "RAA-s = pi05_g8v_raas_ae exported at step 6,000 (tl), UNCHANGED (SR 78's checkpoint, no retraining); ONE change from SR 78: serving.  PolicyBackbone.sample adds the "
                                    "fixed servo (servo_norm_torch) to dims 0-2 of every step of the normalised chunk when the config has a nonzero servo_k, so the served chunk equals "
                                    "PI0PytorchS.sample_actions; other configs are bit-identical.  SR 78 = the same checkpoint served without the servo (bug), 78b = as designed.",
                             "gate": "as SR 78 (offline gate K = 2.851, M = .9375); R0 fit gate on K1-K6 >= .80",
                             "states": "as SR 73 / 76 / 77 / 78: dev trials 4-23 of n2_states + new k 540-579 (harness trials 24-63)",
                             "tasks": "familiar n2 tids %s (K1-K6, N1-N6)" % (TIDS,),
                             "mapping": "docs/notes/raas_b_plan.md and raas_plan.md 'Registered contrasts' (read: experiments/raas_b_read.py, a copy of raas_read.py with arm raasb = raas_b_* rollouts)",
                             "prediction": PRED, "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any SR 78b rollout (raas_b_* files) and any smoke of the patched server"},
                    bytes=Z, config={}, script="experiments/prereg/raas_b.py"))

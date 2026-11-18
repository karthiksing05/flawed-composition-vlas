"""PRE-REGISTRATION ROW of RAA-s (SCAFFOLD_RESULTS 78): RAA-c's continuous role channel plus a servo residual action head (translation dims; K, max_step from the offline
phase-0 gate on the training demos), against RAA-o, RAA, RAA-c and ReGuide-perc (rgp) on N1-N6 and K1-K6, the SAME 60 states as SR 73 / 76 / 77.  BEFORE any main RAA-s
training job and any RAA-s rollout.  Predictions: experiments/prereg/raas_pred.txt (committed 92b50f8e before this row, with the gate).

    python experiments/prereg/raas.py        (login-free: run through sbatch)
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
JAX = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raas_ae"
TIDS = (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26)
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raas" for r in rows), "already registered"
assert not any(f.startswith("raas_") for f in os.listdir(os.path.join(FL, "raa"))), "raas rollouts exist"
assert not os.path.exists(os.path.join(FL, "raas_read.json")), "raas read exists"
assert not any(os.path.isdir(os.path.join(JAX, e)) for e in ("main", "tl")), "main RAA-s training has started"
smoke_dirs = sorted(os.listdir(JAX)) if os.path.isdir(JAX) else []
assert set(smoke_dirs) <= {"smoke", "smoketl"}, smoke_dirs                   # the quick plumbing checkpoint (exp smoketl) is the only allowed training so far


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


RAA = [r for r in rows if r.get("tag") == "raa"]
RAAC = [r for r in rows if r.get("tag") == "raac"]
assert len(RAA) == 1 and len(RAAC) == 1, (len(RAA), len(RAAC))
H = RAA[0]["metrics"]["files_sha256"]
for p in ("results/flawed/n2_branches_rx.json", "results/flawed/n2_branches_main.json", "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json",
          "results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json", "results/flawed/sc_lex.json", "results/flawed/n2_states_rx/states.npy",
          "results/flawed/n2_states_rx/meta.json"):
    assert H[p] == fsha(p), ("state / calibration file differs from the raa row's hash", p)
print("[raas] states, branches, calibration and lexicon match the raa row", flush=True)

PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/raas_gate.py", "experiments/raas_check.py", "experiments/raas_read.py", "experiments/raac_read.py", "experiments/raao_read.py", "experiments/raa_read.py",
        "experiments/raa.py", "experiments/raac_export.py", "experiments/sc_rv.py", "experiments/c_train_pi05.py", "experiments/pp_p_train.py", "experiments/fa1_common.py",
        "experiments/fc_n2_client.py", "experiments/libero_client.py", "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raac_model.py",
        "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/serve.py",
        "experiments/prereg/raas_pred.txt", "docs/notes/raas_plan.md", "results/flawed/raas_gate.json", "results/flawed/sc_rv_raac_manifest.json",
        "results/flawed/raac_read.json", PP + "raa_job.sh", PP + "raas_train.sh", PP + "raas_chain.sh", PP + "raas_check.sh", PP + "raa_roll.sh", PP + "t16_export.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raas_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-18.  Known when written: SR 77's read (RAA-c N1-N6 .503, K1-K6 .814, U1 NOT SUPPORTED; failed-release median 11.8 / 11.0 cm on N4 / N6) and SR 76's read "
        "(RAA-o .594, RAA .656, rgp .897, K1-K6 .881).  Predictions and the phase-0 gate committed (92b50f8e) before this row, any main training job and any RAA-s rollout.  "
        "Phase-0 gate (experiments/raas_gate.py, job 4790531, offline on the 400 RAA-c training demos, no model, no rollout): K = 2.851, max_step = .9375, 13,327 held rows, pooled "
        "chunk-consistent R^2 .361 and residual variance ratio .573 (per step R^2 .142, ratio .752): PASS (K > 0, ratio < .9).  Checks before the row, all plumbing and not results: "
        "model check (experiments/raas_check.py, job 4790553, one A40, random init): the raas config has the same 53 parameter tensors and the same frozen set as raac, the raac "
        "config is untouched, the same data / norm stats / recipe; the JAX and PyTorch servo equal a numpy reference on real RAA-c states (role 2 on, role 1 and zeros off, clip binds); "
        "with servo_k = 0 the raas model reproduces the raac model's loss and samples exactly (max |diff| 0), and with the real K the sample equals the raac sample plus the servo and the "
        "loss equals the raac loss on (action - servo); a 20-step quick checkpoint (exp smoketl on tail-lab, job 4790596, step-0 loss .0991 against raac's .0987; the overcap smoke 4790555 "
        "was cancelled before it started) was exported (job 4790759) and the exported PyTorch model checked (experiments/raas_check.py --torch-ckpt, job 4790781: PI0PytorchS minus PI0PytorchC on the same weights and noise equals the servo for a held-target state, and is zero for an object-role state and for zeros, dims 3+ untouched).  No rollout smoke was run.  No code, arm, mapping or prediction "
        "changed after the smoke.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raas", seed=0,
                    command="pre-registration: RAA-s, RAA-c plus a servo residual action head (translation = clip(K * target-relative xyz) while the target channel is active + expert), "
                            "against RAA-o, RAA, RAA-c and ReGuide-perc on N1-N6 and K1-K6, same states as SR 73 / 76 / 77 (docs/SCAFFOLD_RESULTS.md 78)",
                    metrics={"status": "PRE-REGISTERED RAA-S",
                             "arm": "RAA-s = pi05_g8v_raas_ae: RAA-c's dataset (lol/libero_g8v_raac), norm stats, state token (state_proj), frames, role phrases, painted views, recipe (frozen "
                                    "VLM, action expert + state_proj, batch 32, 6,001 steps, lr 5e-5, 1,000 warmup, no EMA, one seed, one A40, exp main on overcap and exp tl on tail-lab, the first to "
                                    "reach 6,000 exported); ONE change: the action head adds a fixed servo to the translation dims, action_xyz = clip(K * rel_xyz, -M, M) * m + expert_xyz, with "
                                    "K = 2.851 and M = .9375 fitted offline on the training demos (results/flawed/raas_gate.json, never from a rollout), rel_xyz = the channel's dx dy dz (m), "
                                    "m = 1 iff the channel's role id is 2 (target, held); rotation and gripper unchanged; the flow-matching loss is on (action - servo) in normalised action "
                                    "space and sampling adds the servo back; no trained parameter added beyond RAA-c's state_proj.",
                             "gate": "K > 0 and pooled residual variance ratio < .9 (results/flawed/raas_gate.json): K 2.851, ratio .573, PASS",
                             "states": "as SR 73 / 76 / 77: dev trials 4-23 of n2_states + new k 540-579 (harness trials 24-63)",
                             "tasks": "familiar n2 tids %s (K1-K6, N1-N6)" % (TIDS,),
                             "mapping": "docs/notes/raas_plan.md 'Registered contrasts' (read: experiments/raas_read.py, extends raac_read.py / raao_read.py / raa_read.py)",
                             "prediction": PRED, "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main RAA-s training job (exp main / tl) and any RAA-s rollout in the read directory"},
                    bytes=Z, config={}, script="experiments/prereg/raas.py"))

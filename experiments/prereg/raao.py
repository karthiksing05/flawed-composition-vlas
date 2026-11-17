"""PRE-REGISTRATION ROW of RAA-o (SCAFFOLD_RESULTS 76): RAA with the target-role channel measured from the held object (target - (eef + the in-hand
offset fixed at the grasp event)), one change; against RAA, ReGuide-perc (rgp) on N1-N6 and K1-K6, the SAME 60 states as SR 73.  BEFORE any RAA-o
training job and any RAA-o rollout.  Predictions: experiments/prereg/raao_pred.txt (committed 4e1386cd before this row).

    python experiments/prereg/raao.py
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
assert not any(r.get("tag") == "raao" for r in rows), "already registered"
assert not any(f.startswith("raao_") for f in os.listdir(os.path.join(FL, "raa"))), "raao rollouts exist"
assert not os.path.exists(os.path.join(FL, "raao_read.json")), "raao read exists"
assert not os.path.exists("/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raao_ae"), "raao training has started"


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
print("[raao] states, branches, calibration and lexicon match the raa row", flush=True)

PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/raa.py", "experiments/raao_read.py", "experiments/raao_check.py", "experiments/raa_read.py", "experiments/sc_rv.py",
        "experiments/sc_pu.py", "experiments/sc_seg.py", "experiments/sc_t16.py", "experiments/fa1_common.py", "experiments/fc_n2_client.py",
        "experiments/libero_client.py", "experiments/c_train_pi05.py", "experiments/pp_p_train.py", "lifelong_ood/policy/l90c_config.py",
        "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/serve.py", "experiments/prereg/raao_pred.txt",
        "results/flawed/sc_rv_raao_manifest.json", "results/flawed/raa_read.json", PP + "raa_job.sh", PP + "raao_train.sh", PP + "raa_conv.sh",
        PP + "raa_roll.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raao_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-17.  Known when written: SR 73's read (RAA N1-N6 .656, bottles .467, cheese .844, K1-K6 .839, rgp .897) and the exploratory place-step "
        "diagnostic (commit 88858efb; docs/notes/raa_instr_results.md).  Predictions, code, read and channel check committed (4e1386cd) before this row, any RAA-o "
        "training job and any RAA-o rollout.  Channel unit check (job 4771244, experiments/raao_check.py, K5 and K1 demo 0; the object role and every not-held frame "
        "equal RAA's, a held frame's target channel = target - eef - offset); raao_read selftest (job 4771245) reproduces SR 73's V1a and V1b exactly.  Conversion job "
        "4771246 (arm raao, same demos and renders as RAA).  PLUMBING SMOKE, not a result: job 4771248 served RAA's own checkpoint (pi05_g8v_raa_ae_tl6000, never "
        "trained with the offset channel) in offset mode on N4, smoke states 0-1 (not in the 60 read states), output outside results/: 1 of 2 episodes succeeded, 49 "
        "target-role queries all carried an offset.  No code, arm, mapping or prediction changed after the smoke.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raao", seed=0,
                    command="pre-registration: RAA-o, RAA with the target-role channel measured from the held object (one change), against RAA and "
                            "ReGuide-perc on N1-N6 and K1-K6, same states as SR 73 (docs/SCAFFOLD_RESULTS.md 76)",
                    metrics={"status": "PRE-REGISTERED RAA-O",
                             "arm": "RAA-o = pi05_g8v_raao_ae: RAA's frames, role phrases, painted views, recipe (frozen VLM, action expert, batch 32, 6,000 steps, "
                                    "lr 5e-5, 1,000 warmup, no EMA, one seed, one A40, exp main on overcap and exp tl on tail-lab, the first to reach 6,000 exported) "
                                    "and text-suffix channel format and resolution; ONE change: while held, the channel is target - held object, held object = eef + "
                                    "offset, offset = the grasped object's position minus the eef at the hold's onset (training: the demo's own slot positions and "
                                    "ProprioHeld rows via pp_p_train.demo_sites; test: sc_pu.Estimator's object estimate minus the eef when ProprioHeld fires), fixed "
                                    "until the hold ends.  The object role is unchanged.  No release gate, no controller logic.",
                             "states": "as SR 73: dev trials 4-23 of n2_states + new k 540-579 (harness trials 24-63)",
                             "tasks": "familiar n2 tids %s (K1-K6, N1-N6)" % (TIDS,),
                             "mapping": "docs/notes/raao_plan.md 'Registered contrasts' (read: experiments/raao_read.py, extends raa_read.py)",
                             "prediction": PRED, "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any RAA-o training job and any RAA-o rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raao.py"))

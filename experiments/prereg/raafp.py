"""PRE-REGISTRATION ROW of RAA-F Route A+ and A+ctrl (docs/notes/raaf_plan.md 3; docs/notes/raafp_diag.md).  BEFORE any main training job (exp main / tl) of pi05_g8v_raas_lp_ae / pi05_g8v_rview2_lp_ae
and any probe or rollout.  Predictions: experiments/prereg/raafp_pred.txt.  Run through sbatch only."""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

JAXD = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raafp" for r in rows), "already registered"
for cfg in ("pi05_g8v_raas_lp_ae", "pi05_g8v_rview2_lp_ae"):
    d = os.path.join(JAXD, cfg)
    assert not any(os.path.isdir(os.path.join(d, e)) for e in ("main", "tl")), "main training has started: " + cfg
    assert set(os.listdir(d) if os.path.isdir(d) else []) <= {"smoke"}, cfg


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["docs/notes/raaf_plan.md", "docs/notes/raafp_diag.md", "experiments/prereg/raafp_pred.txt", "experiments/raafp_gen.py", "experiments/raafp_rv.py", "experiments/sc_rv.py",
        "experiments/c_train_pi05.py", "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py", "results/flawed/raas_gate.json",
        "results/flawed/raaf/raafp_rv_manifest_raac.json", "results/flawed/raaf/raafp_rv_manifest_rview2.json",
        "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json", PP + "raafp_train.sh", PP + "raafp_chain.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raafp_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-20.  Known when written: planner dry run and rule h_j (docs/notes/raafp_diag.md); 100 kept demos (bowl 50 of 79 attempts, cream cheese 50 of 163; seeds 94,000,000 + 1000 obj + k, layout redraw + 50,000,000 j); replay check: restoring each stored state reproduces its end effector to 0.000 mm (100 of 100 demos), action replay does not (one-step median 15 mm, as the genuine K1 demos' 10-12 mm; open-loop median 21 mm); "
        "datasets lol/libero_g8v_raacp and lol/libero_g8v_rview2p (500 episodes = 400 base + 100 planner, 61,649 frames, 9,468 planner frames, 5,306 with the held prompt 'target: the table'), both checks passed (base episodes identical to lol/libero_g8v_raac / rview2 on state and actions); "
        "20-step smokes (tail-lab): A+ step-0 loss .0486 (job 4819964), A+ctrl .0475 (job 4819990); no A+ / A+ctrl checkpoint, probe or rollout exists.  DEVIATIONS (orchestrator): spot rule h_j; wine bottle dropped by plan k1.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raafp", seed=0,
                    command="pre-registration: RAA-F Route A+ and A+ctrl (docs/notes/raaf_plan.md 3): RAA-sL's model (token + servo) and the data-only rview2 control, both from pi05_libero, on the K1-K8 base episodes + 100 planner table-spot demos (bowl, cream cheese); selection by the A2 probe",
                    metrics={"status": "PRE-REGISTERED RAA-F Route A+ / A+ctrl", "arm": "A+ pi05_g8v_raas_lp_ae (data lol/libero_g8v_raacp, RAA-c norm stats, servo constants, 6,001 steps, batch 32, lr 5e-5, frozen VLM) and A+ctrl pi05_g8v_rview2_lp_ae (data lol/libero_g8v_rview2p, rview2 stats), exp main / tl",
                             "mapping": "docs/notes/raaf_plan.md sections 3, 6-7", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "spot rule h_j, bottle dropped (k1), orchestrator-approved 2026-11-20; docs/notes/raafp_diag.md",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main A+ / A+ctrl training job (exp main / tl), probe and rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raafp.py"))

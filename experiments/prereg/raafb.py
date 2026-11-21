"""PRE-REGISTRATION ROW of RAA-F Route B (docs/notes/raaf_plan.md section 4, docs/notes/raafb_build.md): the target crop in the third image slot, servo-free RAA-c base.  BEFORE any
main training job (exp main / tl) and any rollout or probe.  Predictions: experiments/prereg/raafb_pred.txt.  Usage (sbatch only): raafb.py "<smoke step-0 loss text>"."""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

SMOKE = sys.argv[1]
JAX = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raac_lb_ae"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raafb" for r in rows), "already registered"
assert set(os.listdir(JAX) if os.path.isdir(JAX) else []) <= {"smoke"}, "main training has started"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["docs/notes/raaf_plan.md", "docs/notes/raafb_build.md", "experiments/prereg/raafb_pred.txt", "experiments/raafb_crop.py", "experiments/raafb_proj.py", "experiments/raafb_lift.py",
        "experiments/raafb_rv.py", "experiments/raafb_check.py", "experiments/raafb_batch_check.py", "experiments/raafb_smoke_test.py", "experiments/wa_sb.py", "experiments/raa_b.py",
        "experiments/sc_rv.py", "experiments/c_train_pi05.py", "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raafb_inputs.py", "lifelong_ood/policy/raas_model.py",
        "lifelong_ood/policy/raac_model.py", "results/flawed/raafb_lift.json", "results/flawed/raafb_camKE.npz", "results/flawed/raafb_s_b1.json", "results/flawed/raafb_rv_manifest.json",
        "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json", PP + "raafb_train.sh", PP + "raafb_chain.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raafb_pred.txt"), encoding="utf-8").read()
man = json.load(open(os.path.join(ROOT, "results", "flawed", "raafb_rv_manifest.json"), encoding="utf-8"))
SEEN = ("Written 2026-11-20.  Known when written: S-B1 on rendered demos with the class-offset reference (movable median 3.71 px, p90 8.35; body-origin reference was 9.54 / 16.5 and is recorded as the "
        "first read; K / E vs sc_pu.cam_model agree to 1e-13 px); dataset lol/libero_g8v_raacb built (%d episodes, %d frames; raafb_check.py PASS, crop slot unmasked in the real data pipeline per "
        "raafb_batch_check.py); 20-step smoke (exp smoke): %s.  No Route B checkpoint, probe or rollout exists." % (man["episodes"], man["frames"], SMOKE))
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raafb", seed=0,
                    command="pre-registration: RAA-F Route B (docs/notes/raaf_plan.md 4): RAA-c state-token model (servo_k = 0) from pi05_libero on lol/libero_g8v_raacb (RAA-c frames + target-centred crop in the third image slot); selection by the A2 probe",
                    metrics={"status": "PRE-REGISTERED RAA-F Route B", "arm": "pi05_g8v_raac_lb_ae: RAA-c model, weights from pi05_libero, RAA-c norm stats / assets, data lol/libero_g8v_raacb, recipe of RAA-sL (6,001 steps, batch 32, lr 5e-5, frozen VLM), exp main / tl",
                             "mapping": "docs/notes/raaf_plan.md sections 4, 6-7", "prediction": PRED, "seen": SEEN,
                             "deviation_notes": "(a) crop centre = visible geom centre (class offset from the simulator's geoms), S-B1 reference change; (b) servo-free base instead of RAA-sL's model; both orchestrator decisions, see raafb_pred.txt NOTES",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main Route B training job (exp main / tl), probe and rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raafb.py"))

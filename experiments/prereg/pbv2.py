"""PRE-REGISTRATION ROW of the PHASE-2 BindVLA dev read (docs/PLAN_BINDVLA.md section 11): main2 vs the matched vanilla_frozen
comparator, closed loop on the 29 dev tasks, plus the offline dev slot-hit diagnostic.  Written after training, the training-task
parity reads and the leakage check, BEFORE any phase-2 dev episode or dev slot score (asserted).

    python experiments/prereg/pbv2.py --main results/bindvla/main2/ckpt/step_060000.pt --vanilla results/bindvla/vanilla/ckpt/step_060000.pt
"""
import argparse
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

ap = argparse.ArgumentParser()
ap.add_argument("--main", required=True)
ap.add_argument("--vanilla", required=True)
a = ap.parse_args()
B = os.path.join(ROOT, "results", "bindvla")
assert not any("main2" in str(r.get("tag")) or "vanilla" in str(r.get("tag")) for r in ledger.load("bv_dev")), "a phase-2 dev row exists"
assert not glob.glob(os.path.join(B, "eval", "dev_bindvla_main2*")) and not glob.glob(os.path.join(B, "eval", "dev_bindvla_vanilla*"))
assert not os.path.exists(os.path.join(B, "devdiag.json")), "the dev slot diagnostic must not have run"
assert not any(r.get("tag") == "pbv2" for r in ledger.load("m3_prereg"))
leak = json.load(open(os.path.join(B, "leakcheck.json"), encoding="utf-8"))
assert leak["PASS"] and "main2" in leak["4_runs"] and "vanilla" in leak["4_runs"], "re-run bv_leakcheck with both runs' configs"


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def last_eval(exp):
    log = [json.loads(x) for x in open(os.path.join(B, exp, "train_log.jsonl"), encoding="utf-8")]
    ev = [r for r in log if "val_loss" in r]
    return ev[-1] if ev else None


par = json.load(open(os.path.join(B, "eval", "bv_eval.json"), encoding="utf-8"))
print(ledger.record("m3_prereg", bench="libero_90_c", tag="pbv2", seed=7, command="pre-registration: PLAN_BINDVLA phase-2 dev read",
                    metrics={"status": "PRE-REGISTERED the phase-2 BindVLA dev read (PLAN_BINDVLA section 11)",
                             "models": {"main2": dict(ckpt=os.path.relpath(os.path.abspath(a.main), ROOT), sha256=fsha(a.main), last_eval=last_eval("main2")),
                                        "vanilla": dict(ckpt=os.path.relpath(os.path.abspath(a.vanilla), ROOT), sha256=fsha(a.vanilla), last_eval=last_eval("vanilla"))},
                             "parity_training_tasks": {k: v.get("parity") for k, v in par.items()},
                             "bars": {"S1": "held-out-training-episode final-level slot hit >= 0.6 for object AND goal; depth-2 and depth-3 lower 95% bound > 0.070",
                                      "S2": "training-task parity >= 0.65",
                                      "S3": "dev success (29 tasks x 10 episodes, MODE=ours, init 0-9) of main2 minus vanilla: paired task-cluster bootstrap 95% CI excludes 0, AND main2 absolute dev success >= 0.30"},
                             "success": "BindVLA succeeds iff S1 and S2 and S3",
                             "dev_slot_diagnostic": "descriptive: experiments/bv_devdiag.py on main2 and phase-1 main, final-level hit vs the dev demos' contact-event referents (scored only), by CD bin / role / depth, task-cluster CIs; vanilla has no slots",
                             "leakcheck": {k: leak[k] for k in ("PASS", "3_heldout_episodes", "4_runs")},
                             "files_sha256": sha(["docs/PLAN_BINDVLA.md", "lifelong_ood/policy/bindvla.py", "lifelong_ood/policy/backbone_bindvla.py",
                                                  "experiments/bv_train.py", "experiments/bv_eval.py", "experiments/bv_refs.py",
                                                  "experiments/bv_devrefs.py", "experiments/bv_devdiag.py", "experiments/bv_leakcheck.py",
                                                  "results/bindvla/devrefs.json"]),
                             "written_before": "any phase-2 dev episode and any dev slot score"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pbv2.py"))

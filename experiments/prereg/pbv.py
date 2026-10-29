"""PRE-REGISTRATION ROW of the BindVLA dev read (docs/PLAN_BINDVLA.md section 6), written after training, the smoke and the parity
read on TRAINING tasks only, and BEFORE any dev episode is rolled out (asserted).

    python experiments/prereg/pbv.py --ckpt results/bindvla/main/ckpt/step_NNNNNN.pt
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
ap.add_argument("--ckpt", required=True)
a = ap.parse_args()
assert not ledger.load("bv_dev"), "a dev row exists: the pre-registration must come first"
assert not glob.glob(os.path.join(ROOT, "results", "bindvla", "eval", "dev_*.json"))
assert not any(r.get("tag") == "pbv" for r in ledger.load("m3_prereg"))


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


B = os.path.join(ROOT, "results", "bindvla")
ck = os.path.abspath(a.ckpt)
exp = ck.split("/results/bindvla/")[1].split("/")[0]
cfg = json.load(open(os.path.join(B, exp, "config.json"), encoding="utf-8"))
log = [json.loads(x) for x in open(os.path.join(B, exp, "train_log.jsonl"), encoding="utf-8")]
evals = [r for r in log if "val_loss" in r]
parity = json.load(open(os.path.join(B, "eval", "bv_eval.json"), encoding="utf-8"))
posfree = json.load(open(os.path.join(B, "posfree.json"), encoding="utf-8"))
h = hashlib.sha256()
with open(ck, "rb") as fh:
    for blk in iter(lambda: fh.read(1 << 24), b""):
        h.update(blk)
print(ledger.record("m3_prereg", bench="libero_90_c", tag="pbv", seed=7, command="pre-registration: PLAN_BINDVLA dev read",
                    metrics={"status": "PRE-REGISTERED the BindVLA dev read (PLAN_BINDVLA section 6)",
                             "model": dict(setting=cfg["setting"], cfg=cfg["cfg"], params=cfg["params"], ckpt=os.path.relpath(ck, ROOT),
                                           ckpt_sha256=h.hexdigest()),
                             "training_evidence": dict(last_eval=evals[-1] if evals else None, n_evals=len(evals)),
                             "parity_training_tasks": {k: v.get("parity") for k, v in parity.items()},
                             "posfree": {k: posfree[k] for k in ("linear_r2_before", "linear_r2_after", "mlp_r2_before", "mlp_r2_after", "removed_dims")},
                             "dev_set": "the split's 29 dev tasks (CD bins 0.1 / 0.3 / 0.5: 5 each; part: 14), 10 episodes (init 0-9), MODE=ours",
                             "models": "BindVLA (this checkpoint) and lerobot/smolvla_libero (public reference, CONTAMINATED: trained on libero_object ketchup, libero_goal 3 / 7, LIBERO-10 two-step tests); vanilla_frozen and the ablations are NOT run (need approval)",
                             "predictions": "PLAN_BINDVLA section 4 (directions per ablation) and the hierarchy predictions (a)-(c); in this read only the bindvla row and the contaminated reference exist, so the read is DESCRIPTIVE: success per CD bin and per part role with task-bootstrap CIs; no comparator claim",
                             "files_sha256": sha(["docs/PLAN_BINDVLA.md", "lifelong_ood/policy/bindvla.py", "lifelong_ood/policy/backbone_bindvla.py",
                                                  "experiments/bv_train.py", "experiments/bv_eval.py", "experiments/bv_data.py"]),
                             "written_before": "any BindVLA dev episode"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pbv.py"))

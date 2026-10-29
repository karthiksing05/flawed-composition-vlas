"""PRE-REGISTRATION ROW of PLAN_BINDER B0 (docs/PLAN_BINDER.md section 3 and "B0 pre-registration"), written 2026-10-27 after a
smoke on TRAINING units only (results/bind/b0/smoke/) and BEFORE any dev unit is predicted or scored (asserted)."""
import glob
import hashlib
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OD = os.path.join(ROOT, "results", "bind", "b0")
assert not glob.glob(os.path.join(OD, "*__*.npz")) and not os.path.exists(os.path.join(ROOT, "results", "bind", "b0.json"))
assert not ledger.load("b0_vlm")


def sha(ps):
    return {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in ps}


print(ledger.record("m3_prereg", bench="libero_90_c", tag="pb0", seed=0, command="pre-registration: PLAN_BINDER B0",
                    metrics={"status": "PRE-REGISTERED PLAN_BINDER B0 (pretrained VLM backbones as binders, offline screen)",
                             "arms": "primary: pi05_base backbone detect_phrase; Qwen2.5-VL-7B-Instruct bbox_phrase; Molmo-7B-D-0924 point_phrase. secondary: segment and instruction-context variants. supplementary: pi05_l90c@5000 backbone detect_phrase",
                             "not_run": "PaliGemma mix (google/paligemma-3b-mix-224, not the policy base) and PaliGemma pt-224 (google/paligemma-3b-pt-224, the policy base; coordinator amendment): gated, no local weights",
                             "smoke": "6 TRAINING units: pi05_base emits fluent captions / yes-answers, no loc token, under all 4 prompts (recorded as found, no change); Qwen and Molmo parse 6/6 (Molmo context 5/6)",
                             "prompts": "experiments/b0_vlm.py docstring (phrase verbatim from S1 labels; context variant adds the full instruction)",
                             "conversion": "box -> centre-token peak, coverage>=0.5 mask (B1a rule); segment -> decoded VQ mask, max-coverage peak; points -> S3b rule; unparseable -> miss",
                             "units": "S1's 240 dev units (results/bind/s1/labels.json), S1 hit metric (s1_read.hit), task-cluster bootstrap (d4_read.rate)",
                             "bar": "per (arm, config): held-out depth-1 hit >= 0.70 AND all depth-2 seen-body hit >= 0.70 AND depth-0 hit >= GroundingDINO (0.786); B0 PASS iff any primary config passes; caddy reported separately",
                             "power": "(i) 3 tasks, 15 units; (ii) 4 tasks, 20 units",
                             "pi05_rule": "if the pi05 backbone emits no parseable <loc> output it is recorded as a finding (no prompt or decoding changes)",
                             "files_sha256": sha(["docs/PLAN_BINDER.md", "experiments/b0_vlm.py", "experiments/b0_read.py",
                                                  "results/bind/s1/labels.json"]),
                             "written_before": "any B0 dev prediction or score"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pb0.py"))

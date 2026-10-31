"""PRE-REGISTRATION ROW of prototype A, stage 1b (docs/FLAWED_COMPOSITION.md section 5.6): the contact-gated factored motor with the
grasp offset (Ab), against Hb and Rb (3 seeds each) and three seed-0 ablations, on ORACLE state, N2's 16 familiar-layout tasks.
Criteria C1-C4 and the verdict order of section 5.6 / experiments/fa1b_read.py.  Written after the contact labels, the training of
every model and the swap test (training states only), and BEFORE any stage-1b rollout (asserted: results/flawed/a1b_roll does not
exist).  Hashes: the section-5.6 protocol text, every script, the tasks / states / branches, the calibration, the training data and
the twelve checkpoints.

    python experiments/prereg/pa1b.py
"""
import hashlib
import json
import os
import re
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

F = os.path.join(ROOT, "results", "flawed")
DATA = "/coc/flash10/ksingara3/robot/data/fa1"
RUNS = [(m, s) for m in ("Hb", "Rb", "Ab") for s in (0, 1, 2)] + [(m, 0) for m in ("Ab_cmd", "Ab_nooff", "Ab_nosum")]
assert not os.path.exists(os.path.join(F, "a1b_roll")), "stage-1b rollouts exist"
assert not os.path.exists(os.path.join(F, "a1b_read.json")), "a stage-1b read exists"
assert not any(r.get("tag") == "pa1b" for r in ledger.load("m3_prereg")), "pa1b already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
swap = json.load(open(os.path.join(F, "a1b_swap.json"), encoding="utf-8"))
held = json.load(open(os.path.join(F, "a1b_held.json"), encoding="utf-8"))
train = {s: json.load(open(os.path.join(F, f"a1b_train_b_s{s}.json"), encoding="utf-8")) for s in (0, 1, 2)}


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.6 What \"compositional, with success guaranteed\" means.*?(?=\n\*\*Results|\n---\n)", doc, flags=re.S)
assert m, "section 5.6's protocol is missing"
section_sha = hashlib.sha256(m.group(0).encode("utf-8")).hexdigest()
fam = [t for t in tasks["tasks"] if t["layout"] == "familiar"]
FILES = ["experiments/fa1_common.py", "experiments/fa1_train.py", "experiments/fa1_read.py", "experiments/fa1b_common.py",
         "experiments/fa1b_data.py", "experiments/fa1b_train.py", "experiments/fa1b_roll.py", "experiments/fa1b_read.py",
         "experiments/fa1b_swap.py", "lifelong_ood/policy/factored_flow.py", "results/flawed/n2_tasks.json",
         "results/flawed/n2_branches_main.json", "results/flawed/a1_calib.json", "results/flawed/a1b_held.json",
         "results/flawed/a1b_swap.json"] + [f"results/flawed/a1b_train_b_s{s}.json" for s in (0, 1, 2)] + \
        sorted({t["bddl"] for t in fam}) + [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in fam]
CKPT = {f"{a}_s{s}": os.path.join(DATA, "ckpt", f"b_s{s}", f"{a}.npz") for a, s in RUNS}
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1b", seed=0,
                    command="pre-registration: FLAWED_COMPOSITION 5.6 prototype A stage 1b",
                    metrics={"status": "PRE-REGISTERED A1b (FLAWED_COMPOSITION 5.6, prototype A stage 1b, oracle state, familiar layout)",
                             "section_5_6_protocol_sha256": section_sha,
                             "runs": {f"{a}_s{s}": dict(params=train[s]["arms"][a]["params"]) for a, s in RUNS},
                             "training": {k: train[0][k] for k in ("steps", "batch", "lr", "horizon", "n_train")},
                             "changes": "contact gate (held machine: both fingers + lift >= 1 cm on, either finger keeps, neither off); "
                                        "grasp offset in the target term (target frame, no identity); held flag + offset in Hb / Rb "
                                        "proprio; seeds 0-2 for Hb, Rb, Ab; ablations Ab_cmd, Ab_nooff, Ab_nosum seed 0",
                             "episodes": "N2's 16 familiar-layout tasks x main states 4-23 (320 per model), stage 1's rollout protocol, "
                                         "N2's noise seed bases, identical across arms and seeds",
                             "primary_class": "pick-and-place K1-K6 (known), N1-N6 (new)",
                             "descriptive_class": "K7 / N7 (drawer opened before the place), K8 / N8 (push, no grasp): per compound only",
                             "criteria": {
                                 "floor": "s_Hb(known) < 0.5 -> uninformative",
                                 "C1": "s_Ab(known) - s_Hb(known) >= -0.10 (point; CI reported)",
                                 "C2": "s_Ab(new) / s_Ab(known) >= 0.8 (point) and CI lower bound >= 0.6",
                                 "C3": "for each new pairing (o, t): CI of s_Ab(o, t) - g_Ab(o) * p_Ab(t) contains 0 (g_Ab(o) pooled g over "
                                       "known pick-place tasks with object o; p_Ab(t) pooled s given g over known pick-place tasks with "
                                       "target t; start states resampled only)",
                                 "C4": "s_Ab(new) - s_Hb(new) > 0 and s_Ab(new) - s_Rb(new) > 0, both CIs excluding 0",
                                 "verdict_order": "floor -> 'uninformative'; C1-C4 all -> 'compositional'; else 'not compositional "
                                                  "(fails: <failing criteria in order C1, C2, C3, C4>)'"},
                             "predictions": "Proposition 3 (ii): Ab's first held step and end-effector position there identical across "
                                            "tasks sharing object and start state; C2 expected to fail since the support condition "
                                            "(Supp) fails for every new pairing; C1 and C3 open",
                             "ci": "2,000-draw percentile bootstrap, seed 0, compounds resampled within status (6 + 6) and start "
                                   "states (20) resampled, jointly for every arm and seed; seeds pooled, not resampled",
                             "swap_test_before_rollouts": swap["tests"],
                             "held_labels": {k: v["held_machine"] for k, v in held["tasks"].items()},
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {x: fsha(os.path.join(DATA, x)) for x in ("demos.npz", "demos_b.npz")},
                             "checkpoints_sha256": {k: fsha(p) for k, p in CKPT.items()},
                             "written_before": "any stage-1b rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1b.py"))

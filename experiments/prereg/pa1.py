"""PRE-REGISTRATION ROW of prototype A, stage 1 (docs/FLAWED_COMPOSITION.md section 5.5): the factored, frame-equivariant, phase-gated
flow motor against a holistic (H) and an OA-WAM-like role-routed (R) head, and three ablations, all on ORACLE state, trained on N2's
8 known LIBERO-Goal tasks and evaluated on N2's 32 tasks.  Written after the data, the relaxed-predicate calibration, the recipe
development on held-out demos (results/flawed/a1_dev) and the final training, and BEFORE any smoke or main rollout (asserted:
results/flawed/a1_roll does not exist).  Hashes: the protocol section, every script, the tasks / states / branches, the calibration,
the training set and the six final checkpoints.

    python experiments/prereg/pa1.py
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
ARMS = ("H", "R", "A", "A_noframe", "A_nogate", "A_nosum")
assert not os.path.exists(os.path.join(F, "a1_roll")), "A1 rollouts exist"
assert not os.path.exists(os.path.join(F, "a1_read.json")) and not os.path.exists(os.path.join(F, "a1_smoke.json")), "an A1 read exists"
assert not any(r.get("tag") == "pa1" for r in ledger.load("m3_prereg")), "pa1 already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
train = json.load(open(os.path.join(F, "a1_train_final.json"), encoding="utf-8"))


def fsha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


doc = open(os.path.join(ROOT, "docs", "FLAWED_COMPOSITION.md"), encoding="utf-8").read()
m = re.search(r"### 5\.5 Prototype A, stage 1 \(oracle state\).*?(?=\n\*\*Results)", doc, flags=re.S)
assert m, "section 5.5's protocol is missing"
section_sha = hashlib.sha256(m.group(0).encode("utf-8")).hexdigest()
FILES = ["experiments/fa1_common.py", "experiments/fa1_data.py", "experiments/fa1_train.py", "experiments/fa1_roll.py",
         "experiments/fa1_read.py", "lifelong_ood/policy/factored_flow.py", "results/flawed/n2_tasks.json",
         "results/flawed/n2_branches_smoke.json", "results/flawed/n2_branches_main.json", "results/flawed/a1_calib.json",
         "results/flawed/a1_equiv.json", "results/flawed/a1_train_final.json"] + \
        sorted({t["bddl"] for t in tasks["tasks"]}) + [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in tasks["tasks"]]
CKPT = {a: os.path.join(DATA, "ckpt", "final", f"{a}.npz") for a in ARMS}
print(ledger.record("m3_prereg", bench="libero_goal", tag="pa1", seed=0, command="pre-registration: FLAWED_COMPOSITION 5.5 prototype A stage 1",
                    metrics={"status": "PRE-REGISTERED A1 (FLAWED_COMPOSITION 5.5, prototype A stage 1, oracle state)",
                             "section_5_5_protocol_sha256": section_sha,
                             "arms": {a: dict(params=train["arms"][a]["params"]) for a in ARMS},
                             "training": {k: train[k] for k in ("steps", "batch", "lr", "horizon", "seed", "holdout", "n_train")},
                             "oracle": "slots = simulator poses of the 4 movable objects and 5 target sites (+ identity one-hot, hand-set "
                                       "size, top-drawer opening); roles from a template parse of the prompt; same for every arm",
                             "episodes": "N2's 32 tasks; smoke: states 0-3 on the 8 known / familiar tasks; main: states 4-23 on all 32; "
                                         "fresh re-seeded env per episode, N2's noise seed bases, identical across arms",
                             "gate": "after the smoke: H's known / familiar smoke success < 0.5 -> stop, fix the recipe on held-out "
                                     "demos only, re-register; no main run",
                             "predictions": {
                                 "A1": "s_A(known, fam) - s_H(known, fam) >= -0.10 (point)",
                                 "A2": "p_rel,A(new, fam) - p_rel,H(new, fam) > 0 and p_rel,A(new, fam) - p_rel,R(new, fam) > 0, both 95% "
                                       "CIs excluding 0",
                                 "A3": "L_g(H) - L_g(A) > 0 and L_g(R) - L_g(A) > 0, both CIs excluding 0; L_g = mean_c [g(c, fam) - "
                                       "g(c, new)] over all 16 compounds",
                                 "A4": "for H and for R: p_rel(known, fam) - p_rel(new, fam) > 0, CI excluding 0",
                                 "A5": "descriptive: per ablation, p_rel,A(new, fam) - p_rel,Y(new, fam) and L_g(Y) - L_g(A) with CIs",
                                 "verdict_order": "floor (s_H(known, fam) < 0.5 on main) -> uninformative; not A1 -> 'A1 fails: A is a "
                                                  "worse policy'; A2 and A3 -> confirmed; A2 only -> 'placement confirmed, layout "
                                                  "not'; else -> 'killed: the factored motor does not fix placement'; A4 reported "
                                                  "beside, never changes the verdict",
                                 "listed_failures": "K7 / N7 (articulated drawer opened before the place) and K8 / N8 (no gripper "
                                                    "closure, A's target term never runs) violate A's factorisation"},
                             "ci": "2,000-draw percentile bootstrap, seed 0, compounds resampled within status and start states "
                                   "resampled, jointly for every arm",
                             "relaxed": "s, or the object's end position within max(RAD, dxy_rest + 0.03) of the target reference and "
                                        "dz in [dz_rest - 0.04, dz_rest + 0.06] (a1_calib.json); drawer target strict only",
                             "secondary": "strict p; pick-and-place pairs only (K1-K6, N1-N6); success effects; per-compound table; "
                                          "habit table (instructed / habit / other / elsewhere on new-compound g = 1, s = 0)",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "data_sha256": {"demos.npz": fsha(os.path.join(DATA, "demos.npz"))},
                             "checkpoints_sha256": {a: fsha(p) for a, p in CKPT.items()},
                             "written_before": "any A1 smoke or main rollout"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pa1.py"))

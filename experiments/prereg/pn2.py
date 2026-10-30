"""PRE-REGISTRATION ROW of N2 (docs/FLAWED_COMPOSITION.md section 4.11): the compound x layout factorial inside one scene, on
pi0.5 and GR00T N1.5.  Written after the tasks, start states, renders and novelty measures (results/flawed/n2_tasks.json,
n2_novelty.json; no policy run) and BEFORE any rollout, smoke included (asserted: results/flawed/n2_roll does not exist).

    python experiments/prereg/pn2.py
"""
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

F = os.path.join(ROOT, "results", "flawed")
assert not os.path.exists(os.path.join(F, "n2_roll")), "N2 rollouts exist"
assert not os.path.exists(os.path.join(F, "n2_read.json")) and not os.path.exists(os.path.join(F, "n2_smoke.json")), "an N2 read exists"
assert not any(r.get("tag") == "pn2" for r in ledger.load("m3_prereg")), "pn2 already registered"
tasks = json.load(open(os.path.join(F, "n2_tasks.json"), encoding="utf-8"))
nov = json.load(open(os.path.join(F, "n2_novelty.json"), encoding="utf-8"))


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


FILES = ["docs/FLAWED_COMPOSITION.md", "experiments/fc_n2_tasks.py", "experiments/fc_n2_client.py", "experiments/fc_n2_roll.sh",
         "experiments/fc_n2_novelty.py", "experiments/fc_n2_read.py", "experiments/libero_client.py", "experiments/serve_guided.py",
         "results/flawed/n2_tasks.json", "results/flawed/n2_novelty.json", "results/flawed/n2_branches_smoke.json",
         "results/flawed/n2_branches_main.json"] + [t["bddl"] for t in tasks["tasks"] if t["layout"] == "familiar"] + \
        [f"results/flawed/n2_states/n2_goal_{t['tid']}.npy" for t in tasks["tasks"]]
print(ledger.record("m3_prereg", bench="libero_goal", tag="pn2", seed=0, command="pre-registration: FLAWED_COMPOSITION N2",
                    metrics={"status": "PRE-REGISTERED N2 (FLAWED_COMPOSITION 4.11)",
                             "design": {t["key"]: dict(language=t["language"], status=t["status"], layout=t["layout"], goal=t["goal"])
                                        for t in tasks["tasks"]},
                             "compound_check": nov["compounds"],
                             "novelty_cells": nov["cells"],
                             "novelty_tasks": {k: {c: v[c] for c in ("d_a", "d_b", "d_c", "input_composite")} for k, v in nov["tasks"].items()},
                             "models": {"pi05": "openpi pi05_libero (PyTorch), serve_guided mode ours",
                                        "groot": "ar0s/groot_libero (GR00T N1.5), serve_guided mode ours"},
                             "episodes": "smoke: states 0-3; main: states 4-23; every cell and model on the same states and noise seeds",
                             "gate": "after the smoke: a model whose known-compound / familiar-layout smoke success is < 0.20 is floored; "
                                     "stop and report, no main run",
                             "predictions": {
                                 "P1": "GR00T layout effect L_groot = mean_c [s(c, fam) - s(c, new)] > 0, 95% CI excluding 0",
                                 "P2": "pi0.5 compound effect C_pi05 = mean_known sbar - mean_new sbar > 0, 95% CI excluding 0",
                                 "P3": "DD = (L_groot - C_groot) - (L_pi05 - C_pi05) > 0, 95% CI excluding 0",
                                 "double_dissociation": "P1, P2, P3, and L_groot > C_groot and C_pi05 > L_pi05 (points)",
                                 "partial": "P3 without P1 or P2, or P1 and P2 without P3",
                                 "reversed": "DD < 0 with CI excluding 0",
                                 "kill_two_axes": "DD CI includes 0 and in both models L and C both > 0 (CIs exclude 0) with L - C CI "
                                                  "including 0: both factors hurt equally in both models -> the two-axes story is dropped",
                                 "uninformative": "a model with success <= 0.05 in all three non-baseline cells is uninformative; both -> "
                                                  "the experiment is uninformative",
                                 "otherwise": "inconclusive"},
                             "ci": "2,000-draw percentile bootstrap, seed 0, compounds resampled within status and start states "
                                   "resampled, jointly for both models",
                             "secondary": "g (grasped the named object, or success) and p (success given g); log-odds scale; pick-and-place "
                                          "pairs only (K1-K6, N1-N6); interaction I per model; per-cell N1 input novelty, the manipulation "
                                          "check new - familiar on d_a / d_b / d_c, and Spearman of the task composite with failure",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any N2 rollout, smoke included"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pn2.py"))

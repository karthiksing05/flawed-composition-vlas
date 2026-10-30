"""PRE-REGISTRATION ROW of N3 (docs/FLAWED_COMPOSITION.md section 4.12): the confirmatory test of N2's phase dissociation (a new
layout breaks the grasp, a new compound breaks the placement) in LIBERO-Spatial's scene, on pi0.5 and GR00T N1.5 (+ X-VLA under the
rule below).  Written after the tasks, start states, feasibility check and compound check (results/flawed/n3_tasks.json,
n3_novelty.json; no policy run) and BEFORE any rollout, smoke included (asserted: results/flawed/n3_roll does not exist).

    python experiments/prereg/pn3.py
"""
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

F = os.path.join(ROOT, "results", "flawed")
assert not os.path.exists(os.path.join(F, "n3_roll")), "N3 rollouts exist"
assert not os.path.exists(os.path.join(F, "n3_read.json")) and not os.path.exists(os.path.join(F, "n3_smoke.json")), "an N3 read exists"
assert not any(r.get("tag") == "pn3" for r in ledger.load("m3_prereg")), "pn3 already registered"
tasks = json.load(open(os.path.join(F, "n3_tasks.json"), encoding="utf-8"))
nov = json.load(open(os.path.join(F, "n3_novelty.json"), encoding="utf-8"))


def fsha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


FILES = ["docs/FLAWED_COMPOSITION.md", "experiments/fc_n3_tasks.py", "experiments/fc_n3_client.py", "experiments/fc_n3_roll.sh",
         "experiments/fc_n3_novelty.py", "experiments/fc_n3_read.py", "experiments/fc_n3_record.py", "experiments/libero_client.py",
         "experiments/serve_guided.py", "lifelong_ood/policy/factors.py", "results/flawed/n3_tasks.json", "results/flawed/n3_novelty.json",
         "results/flawed/n3_branches_smoke.json", "results/flawed/n3_branches_main.json"] + \
        sorted({t["bddl"] for t in tasks["tasks"]}) + [f"results/flawed/n3_states/n3_spatial_{t['tid']}.npy" for t in tasks["tasks"]]
print(ledger.record("m3_prereg", bench="libero_spatial", tag="pn3", seed=0, command="pre-registration: FLAWED_COMPOSITION N3",
                    metrics={"status": "PRE-REGISTERED N3 (FLAWED_COMPOSITION 4.12)",
                             "design": {t["key"]: dict(language=t["language"], status=t["status"], layout=t["layout"], goal=t["goal"],
                                                       pair=t["pair"]) for t in tasks["tasks"]},
                             "compound_check": nov["compounds"],
                             "feasibility": {p["new"]["key"]: p["feasibility"] for p in tasks["pairs"]},
                             "dz_rest": tasks["dz_rest"],
                             "models": {"pi05": "openpi pi05_libero (PyTorch), serve_guided mode ours",
                                        "groot": "ar0s/groot_libero (GR00T N1.5), serve_guided mode ours",
                                        "xvla": "lerobot/xvla-libero, serve_guided mode ours (BACKBONE_CONFIG=xvla, SmolVLA venv)"},
                             "xvla_rule": "X-VLA's smoke runs with the others; its main runs iff its smoke has no episode errors, passes the "
                                          "gate, and the projected total cost (measured so far + its main at its smoke rate) is <= 6 A40-h. "
                                          "Decided from errors, the gate and cost only; the verdict covers every model whose main ran",
                             "episodes": "smoke: states 0-3; main: states 4-23; every cell and model on the same states and noise seeds; "
                                         "300 steps",
                             "gate": "after the smoke: a model whose known / familiar smoke success is < 0.20 has no main run",
                             "outcomes": {"s": "LIBERO strict success",
                                          "g": "s, or the first object moved > 3 cm while the gripper is commanded closed is bowl_1",
                                          "r": "relaxed placement: s, or bowl_1's end position within the instructed target's footprint "
                                               "(horizontal distance to plate body / stove cook site / cabinet top-side site <= 0.08 / 0.09 / "
                                               "0.10 m, height above it within dz_rest - 0.04 .. dz_rest + 0.06), stable or not",
                                          "p": "pooled sum(x g) / sum(g), x = s (strict) or r (relaxed, primary)"},
                             "hypotheses": {
                                 "H1": "L_g - C_g > 0, CI excluding 0; L_g = mean_c [g(c,fam) - g(c,new)], C_g = mean_i [gbar(K_i) - gbar(N_i)]",
                                 "H2": "C_p - L_p > 0 (relaxed), CI excluding 0; L_p = p(all,fam) - p(all,new), C_p = p(known) - p(new)",
                                 "H3": "(L_g - C_g) - (L_p - C_p) > 0 (relaxed), CI excluding 0",
                                 "H4": "secondary: new-compound episodes with g = 1, s = 0, end position classified in order instructed / "
                                       "habit (plate) / other (remaining fixture target) / elsewhere; rate(habit) - rate(other) > 0, CI "
                                       "excluding 0"},
                             "verdict_mapping": {
                                 "per_model": "H3 holds iff estimate > 0 and CI lower bound > 0",
                                 "confirmed": "H3 holds in every model whose main ran, and H1 and H2 hold in every such model",
                                 "confirmed on the interaction": "H3 holds in every model, H1 or H2 fails in at least one",
                                 "mixed": "H3 holds in at least one model and fails in at least one",
                                 "killed": "H3 fails in every model",
                                 "floored": "no model ran its main (all below the gate)",
                                 "note": "complete and exclusive; H4 and the strict-p versions are reported, never enter the verdict"},
                             "ci": "2,000-draw percentile bootstrap, seed 0, resampling pairs (K_i with N_i) and start states, jointly for "
                                   "all models",
                             "secondary": "strict p; success L and C; cell table; per pair; H1-H4 without pair 2 (table centre); habit "
                                          "classes over all new-compound failures",
                             "weaknesses": "one scene; 8 pairs; the compound is a (referent, target) re-pairing, not an object-target one; "
                                           "pair 2's referent ('from table center') is only approximately kept under translation; "
                                           "fixture-resting objects (pair 4's bowl in the drawer) do not move with the layout",
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any N3 rollout, smoke included"},
                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                    script="experiments/prereg/pn3.py"))

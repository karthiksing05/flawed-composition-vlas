"""PRE-REGISTRATION ROW of STEER (SCAFFOLD_RESULTS 74): the steerability failure predictor of docs/PLAN_PAPER.md 2c.  BEFORE any main probe
rollout.  Committed before it runs, so the row is named after the commit that contains this file and the code it hashes.

    python experiments/prereg/steer.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import steer_read                                                            # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "steer" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "steer_read.json")), "steer read exists"
assert not os.path.isdir(os.path.join(FL, "steer", "s1_pi05")), "main probe rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/steer_probe.py", "experiments/steer_serve.py", "experiments/steer_text.py", "experiments/steer_read.py",
        "experiments/steer_targets.py", "experiments/prereg/steer.py", "experiments/prereg/steer_pred.txt", "experiments/prereg/steer_seen.txt",
        "experiments/pp_serve_purify.py", "experiments/serve_guided.py", "experiments/sc_rv.py", "experiments/sc_rv_w.py", "experiments/sc_seg.py",
        "experiments/sc_t16.py", "experiments/tv_frozen.py", "experiments/f1m_cue.py", "experiments/groot_box.py", "experiments/s2.py",
        "experiments/s2_tasks.py", "experiments/fc_n2_client.py", "experiments/libero_client.py", "experiments/pp_atlas.py",
        "lifelong_ood/policy/serve.py", "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/backbone.py",
        "results/flawed/steer/targets.json", "results/flawed/steer/branches_s1_main.json", "results/flawed/steer/branches_s2_main.json",
        "results/flawed/sc_rv_lex.json", "results/flawed/n2_tasks.json", "results/flawed/s2/s2_tasks.json", "reports/predictor_targets.md",
        PP + "steer_roll.sh", PP + "steer_cpu.sh"]
H = {p: fsha(p) for p in CODE}
T = json.load(open(os.path.join(FL, "steer", "targets.json"), encoding="utf-8"))
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "steer_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "steer_pred.txt"), encoding="utf-8").read().strip()
POL1 = {"pi05_I": "frozen pi0.5", "groot_I": "frozen GR00T", "pi0_I": "frozen pi0", "smolvla_I": "frozen SmolVLA", "xvla_I": "frozen X-VLA (no target)",
        "pi05_tv": "tv", "pi05_tvp": "tvp", "pi05_tvpb": "tvpb", "groot_tv": "tv on GR00T", "groot_tvb": "tvb on GR00T",
        "r0": "R0", "r1": "R1", "r2": "R2", "r2b": "R2b", "r2w": "R2w", "r2l": "r2l"}
POL2 = {"pi05_I": "frozen pi0.5", "pi0_I": "frozen pi0", "pi05_tvp": "tvp on pi0.5"}
SKIPPED = {"rgp, rgo (ReGuide), Hp / H (hand-over), RC": "their controllers take over the episode (attach / hand-back state, no query while ReGuide steers; "
           "H replaces the policy by a scripted planner), so a side query cannot pass through the wrapper's own logic without changing or "
           "bypassing the executed episode; they inherit steerability by design (PLAN_PAPER 2c) and would appear only in the full set",
           "Rrel, RAA": "not trained yet (SR 73, another agent's files); probed later",
           "pi0 / SmolVLA front ends tv, tvpb": "no existing front-end rollouts to use as targets and not in the requested list (front ends: pi0.5 tv, tvp, "
           "tvpb; GR00T tv, tvb)"}
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="steer", seed=0,
                    command="pre-registration: STEER, the steerability failure predictor (docs/PLAN_PAPER.md 2c): counterfactual-instruction "
                            "side queries inside closed-loop rollouts of trained tasks; s_hat = g_s(o) * p_s(t); T0-T3 (docs/SCAFFOLD_RESULTS.md 74)",
                    metrics={"status": "PRE-REGISTERED STEER",
                             "policies_scene1": POL1, "policies_scene2": POL2, "skipped": SKIPPED,
                             "states": "10 trained-task states per task per policy: scene 1 n2 main states k = 4-13 (results/flawed/n2_states, "
                                       "steer/branches_s1_main.json = the first 10 of n2_branches_main.json), tasks K1 K2 K3 K4 K6 (tids 0 2 4 6 10); "
                                       "scene 2 s2 main states k = 4-13 (steer/branches_s2_main.json), tasks K1 K2; the same states for every policy",
                             "probe": "docs/PLAN_PAPER 2c and the user's fixed definition: side query per policy query, never executed; grasp = before the "
                                      "first ProprioHeld hold, place = during the hold; swap = object phrase (grasp) / target phrase (place) with another "
                                      "movable object / target present; one partner per query, cycled; d = summed xyz of the first 5 steps (the executed "
                                      "chunk; 8 steps descriptive); steer = nearer p_new than p_old by angle; true-prompt chunk and absent-word chunk logged "
                                      "with the same p_old, p_new; nonsense-word chunk logged for the baseline",
                             "target_rule": T["rule"],
                             "cells": [[c["policy"], c["scene"], c["task"], round(c["success"], 4), c["state_set"], c["n_states"], c["sr"]] for c in T["cells"]],
                             "no_target": T["no_target"],
                             "tests": steer_read.__doc__.split("    python")[0].strip(),
                             "prediction": PRED,
                             "seen": SEEN,
                             "deviations_planned": "Swap candidates are every other movable object / target present, whether or not the composed "
                                                   "instruction is a task of the suites (role-phrase servers get the extra compositions in their lookup "
                                                   "tables); in scene 2 the objects are the mugs.  The side queries draw the real query's noise (server "
                                                   "counters restored).  Phase labels use a per-step ProprioHeld in the client; a role-phrase server's own "
                                                   "detector steps per query (absent / nonsense words reach rung servers as a phrase override in the slot "
                                                   "the client's phase names; mismatches are counted).  For absent / nonsense words a rung's front end keeps "
                                                   "the true task's routing and painting; front ends on frozen policies receive the word as an unknown "
                                                   "entity (every movable object painted).  X-VLA targets: none.  Trained-task K5 is not probed (the K set "
                                                   "of Tables A and B).  The probe runs through the real arm clients unchanged apart from three layers.",
                             "files_sha256": H,
                             "written_before": "any main probe rollout"},
                    bytes=Z, config={}, script="experiments/prereg/steer.py"))

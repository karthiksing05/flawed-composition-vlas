"""PRE-REGISTRATION ROW of STEER-CONF: the confirmatory test of the exploratory predictor k x a (trained-task success x floor-corrected
steerability, steer_v2_explore.py's "k*a", no refitting) on the two policies of SR 73 that did not exist when SR 74 was run: Rrel and RAA.
Written BEFORE their checkpoints are exported and before any probe of them.  The row is named after the commit that contains this file, the
read script and the predictions note, all of which it hashes.

    python experiments/prereg/steer_conf.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "steer_conf" for r in rows), "already registered"
for p in ("steer_conf_read.json",):
    assert not os.path.exists(os.path.join(FL, p)), p + " exists"
for p in ("s1_rrel", "s1_raa"):
    assert not os.path.isdir(os.path.join(FL, "steer", p)), "probe rollouts of " + p + " exist"
assert not os.path.exists(os.path.join(FL, "raa_read.json")), "SR 73 read exists: the labels would already be known"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/steer_v2_explore.py", "experiments/steer_read.py", "experiments/steer_conf_read.py", "experiments/prereg/steer_conf.py",
        "experiments/steer_probe.py", "experiments/steer_serve.py", "experiments/steer_text.py", "experiments/steer_targets.py",
        "experiments/raa.py", "experiments/raa_read.py", "experiments/prereg/raa_pred.txt", "experiments/prereg/steer_pred.txt",
        "results/flawed/steer/targets.json", "results/flawed/steer/branches_s1_main.json", "results/flawed/steer_v2_explore.json",
        "results/flawed/steer_read.json", "docs/notes/steer_conf_predictions.md", PP + "steer_roll.sh", PP + "steer_conf_cpu.sh"]
H = {p: fsha(p) for p in CODE}
NOTE = open(os.path.join(ROOT, "docs", "notes", "steer_conf_predictions.md"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="steer_conf", seed=0,
                    command="pre-registration: STEER-CONF, the confirmatory test of k x a (steer_v2_explore.py, no refitting) on Rrel and RAA "
                            "(docs/PLAN_PAPER.md 2c; docs/SCAFFOLD_RESULTS.md 73 and 74); C1-C3 in docs/notes/steer_conf_predictions.md",
                    metrics={"status": "PRE-REGISTERED STEER-CONF",
                             "score": "k x a = K(policy) * a_grasp(o) * a_place(t) exactly as steer_v2_explore.py 'k*a' (variant a: (mean steer(cf) - mean "
                                      "steer(true)) / (1 - floor) per policy x phase x swapped-in entity, clipped to [0,1]; floor = policy x phase mean "
                                      "steer(true); K = success share of the probe's own K1-K4, K6 episodes); frozen by the sha256 of steer_v2_explore.py "
                                      "and steer_read.py below, the verbatim re-statement in steer_conf_read.ka(), and a selftest that reproduces the "
                                      "k*a all-90 row of results/flawed/steer_v2_explore.json to 1e-9; no refitting, no entity normalisation",
                             "probe": "identical to SR 74: scene 1, tasks K1 K2 K3 K4 K6 (tids 0 2 4 6 10), the first 10 states of n2_branches_main "
                                      "(steer/branches_s1_main.json, k 4-13), d = summed xyz of the first 5 steps, one partner per side query cycled, "
                                      "steer = nearer p_new than p_old by angle; bootstrap B = 2000, numpy default_rng(0), tasks and policies jointly",
                             "channel_rule": "Rrel and RAA read a prompt suffix with the role entity's position relative to the gripper.  A side query "
                                             "carries the channel the deployed system would compute for the COUNTERFACTUAL instruction: the lexicon "
                                             "parse of the swapped instruction names the entity, the estimate of that entity minus the end effector is the "
                                             "suffix; for an absent or nonsense word the true instruction's channel is kept.  Side queries never step "
                                             "the client's Estimator and the executed episode is the unprobed one.  Simulator positions are used for the "
                                             "steer score only, never fed to the model.",
                             "cells": "Rrel and RAA x N1-N6, scene 1, 12 new cells; the 90 old cells are those of SR 74 (targets.json)",
                             "labels": "SR 73 measured success >= .5 per (policy, task), the mean over the 60 states of raa_read (dev trials 4-23 + new "
                                       "trials 24-63, familiar layout), computed by raa_read.grid from the rollout files",
                             "tests": "C1 AUC of k x a on the 12 new cells >= .75; C2 pooled AUC over 90 + 12 cells, score frozen, PASS iff the lower bound "
                                      "(joint bootstrap) > .5; C3 sign test, k x a ranks Rrel against RAA in the order of their mean N1-N6 success.  "
                                      "Fallback: an undefined AUC (fewer than 12 scoreable cells or one class missing) is NOT TESTABLE, never a pass.",
                             "predictions_and_runbook": NOTE,
                             "files_sha256": H,
                             "written_before": "any probe of Rrel or RAA; their checkpoints do not exist"},
                    bytes=Z, config={}, script="experiments/prereg/steer_conf.py"))

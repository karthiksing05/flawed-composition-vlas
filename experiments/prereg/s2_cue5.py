"""PRE-REGISTRATION ROW of S2CUE5 (SCAFFOLD_RESULTS 61): a confirmatory test of the MAJORITY RULE in scene 2 on frozen pi0.5, arm I.
With two plates, three binary cues say where the held mug goes (NEARER plate, SALIENT plate from the agentview start render, the mug's
TRAINED plate); the rule says the placement goes to the plate backed by 2 of the 3.  NEW states (seeds 875000 + k); NEW plate layouts
per odd-one-out configuration where feasible (families A-D, registered pick rule, policy-free scan); SR 59-60's layouts (Sx, PSx, Sm)
re-run on the new states as REPLICATIONS.  BEFORE any main (k = 4-43 of the new states) rollout or reference run of this probe.

    python experiments/prereg/s2_cue5.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import s2cue5                                                                # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2_cue5" for r in rows), "already registered"
assert len([r for r in rows if r.get("tag") == "s2_cue4"]) == 1
assert not os.path.exists(os.path.join(FL, "s2cue5_read.json")), "s2cue5 read exists"
assert not os.path.exists(os.path.join(FL, "s2cue5_roll")), "s2cue5 rollouts exist"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
SJ = json.load(open(os.path.join(FL, "s2cue5", "s2cue5_states.json"), encoding="utf-8"))
BUILT = SJ["built"]
STATES = ["results/flawed/s2cue5/%s/s2_living_%d.npy" % (m, t) for m in ["O"] + BUILT for t in (0, 1, 2, 3)]
CODE = ["experiments/s2cue5.py", "experiments/s2cue4.py", "experiments/s2cue3.py", "experiments/s2cue2.py", "experiments/s2cue.py",
        "experiments/s2.py", "experiments/s2_tasks.py", "experiments/s2_roll.sh", "experiments/libero_client.py",
        "experiments/sc_rv.py", "experiments/pp_serve_purify.py", "experiments/fa1g_plan.py", "experiments/fa1b_common.py",
        "experiments/fa1_common.py", "lifelong_ood/policy/serve.py", "lifelong_ood/policy/backbone.py",
        "results/flawed/s2/keypoints.json", "results/flawed/s2/s2_tasks.json",
        "results/flawed/s2cue5/branches_main.json", "results/flawed/s2cue5/branches_smoke.json",
        "results/flawed/s2cue5/s2cue5_states.json", "experiments/prereg/s2_cue5_pred.txt", "experiments/prereg/s2_cue5_seen.txt"] + STATES + [
        PP + "s2cue5_scan/pick.json", PP + "s2cue5_scan.sh", PP + "s2cue5_build.sh", PP + "s2cue5_smoke.sh", PP + "s2cue5_job.sh",
        PP + "s2cue5_ref.sh", PP + "s2cue5_read.sh"]
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "s2_cue5_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2_cue5_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
RUNS = [["pi05", m, "I", SJ["design"][m]["role"], [0, 1, 2, 3]] for m in BUILT]
print(ledger.record("m3_prereg", bench="libero_10", tag="s2_cue5", seed=0,
                    command="pre-registration: S2CUE5, scene 2 (LIVING_ROOM_SCENE5) on NEW states: confirmatory test of the MAJORITY "
                            "RULE (the held mug goes to the plate backed by 2 of {nearer, salient, trained}) on frozen pi0.5 arm I, "
                            "NEW plate layouts per odd-one-out configuration (PRIMARY) and SR 59-60's layouts as REPLICATIONS "
                            "(docs/SCAFFOLD_RESULTS.md 61)",
                    metrics={"status": "PRE-REGISTERED S2CUE5",
                             "runs": RUNS,
                             "design": dict(built=BUILT, design=SJ["design"], sources=SJ["sources"], families=SJ["families"],
                                            rule=s2cue5.PICK_RULE, family_cands={f: v["cands"] for f, v in s2cue5.FAMILIES.items()},
                                            new_min=s2cue5.NEW_MIN, feas_min=s2cue5.FEAS_MIN, cfg_min=s2cue5.CFG_MIN,
                                            sx_plates=SJ["sx_plates"], sm_plates=SJ["sm_plates"]),
                             "statistic": "share of 2-vs-1 placements on the majority plate, per configuration (nearer odd, salient "
                                          "odd, trained odd) and pooled; paired bootstrap over the 40 main states, B = 2000, "
                                          "default_rng(0) (s2cue.boot_share), the same resamples for every configuration and the pool",
                             "labels": "MAJORITY CONFIRMED iff pooled lower CI >= .80 and every configuration TESTABLE (>= 20 2-vs-1 "
                                       "placements) with point estimate >= .70; REFUTED iff any TESTABLE configuration's upper CI < "
                                       ".50; otherwise PARTIAL.  Computed on the PRIMARY sources (NEW layouts; a configuration with no "
                                       "built NEW layout falls back to the replication runs, labelled REPLICATION-ONLY).  The same "
                                       "rule over the replication runs alone is reported as REPLICATION (secondary).",
                             "mapping": s2cue5.MAPPING,
                             "prediction": PRED,
                             "seen": SEEN,
                             "deviations_planned": "pi0.5 arm I only (no pi0, GR00T or tv).  Client seed 7 and server seed 13 kept "
                                                   "from SR 57-60 (the state and branch seeds are new).  Salience = SR 60's rule "
                                                   "(agentview start render), a fixed property of each layout.  New states: no "
                                                   "per-episode pairing to SR 50-60.  Ledger tree dirty (other agents' files).",
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main (k = 4-43) rollout or reference run of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/s2_cue5.py"))

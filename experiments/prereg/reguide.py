"""PRE-REGISTRATION ROW of RG (SCAFFOLD_RESULTS 52): a reimplementation of ReGuide (Zhang et al., arXiv 2609.38616; training-free
wrapper on frozen pi0.5: rebinds the instructed referent and steers the end effector with object poses plus training-demo
statistics, then hands back to the frozen policy) on our new pairings N1-N6 and trained tasks K1-K4, K6, fresh familiar-layout
states k = 36-55 (harness trials 24-43), against our frozen front end TV+P (SR 47).  BEFORE any main rgo or rgp rollout.

    python experiments/prereg/reguide.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
TIDS = (16, 18, 20, 22, 24, 26, 0, 2, 4, 6, 10)
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "reguide" for r in rows), "already registered"
assert sum(r.get("tag") == "tv_fresh" for r in rows) == 1, "tv_fresh row missing"
assert not os.path.exists(os.path.join(FL, "rg_read.json")), "rg read exists"
assert not os.path.exists(os.path.join(FL, "rg")), "rg rollouts exist"
assert os.path.exists(os.path.join(FL, "rg_stats.json")), "rg_stats.json missing"
with open(os.path.join(FL, "rg_stats.json"), "rb") as _a, open("/coc/flash10/ksingara3/.claude_tmp/pp/rg/rg_stats_try2.json", "rb") as _b:
    assert hashlib.sha256(_a.read()).digest() == hashlib.sha256(_b.read()).digest(), "smoke stats differ from rg_stats.json"
REUSED = []
for t in TIDS:
    for suf in (".json", "_track.npz"):
        for p in ("results/flawed/pp_atlas_fresh/pi05_%d%s" % (t, suf), "results/flawed/tv_fresh/pi05_tvp_%d%s" % (t, suf)):
            assert os.path.exists(os.path.join(ROOT, p)), ("reused arm missing", p)
            REUSED.append(p)


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATE_FILES = (["results/flawed/n2_states_fresh/states.npy"] + ["results/flawed/n2_states_fresh/n2_goal_%d.npy" % t for t in TIDS]
               + ["results/flawed/n2_branches_fresh.json", "results/flawed/n2_tasks.json", "results/flawed/pp_atlas_fresh_read.json",
                  "results/flawed/tv_fresh_read.json", "results/flawed/rg_stats.json", "results/flawed/sc_pu_calib.json",
                  "results/flawed/sc_rv_lex.json"])
CODE = ["experiments/rg.py", "experiments/rg_roll.sh", "experiments/rg_read.py", "experiments/rg_stats.py", "experiments/tv_fresh.py",
        "experiments/tv_frozen.py", "experiments/pp_atlas.py", "experiments/fc_n2_client.py", "experiments/libero_client.py",
        "experiments/fa1_common.py", "experiments/sc_pu.py", "experiments/sc_rv.py", "experiments/sc_seg.py",
        "experiments/pp_serve_purify.py", "experiments/prereg/reguide_choices.txt", PP + "rg_job.sh", PP + "rg_readjob.sh"]
CHOICES = open(os.path.join(ROOT, "experiments", "prereg", "reguide_choices.txt"), encoding="utf-8").read()
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "reguide_seen.txt"), encoding="utf-8").read()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "reguide_pred.txt"), encoding="utf-8").read()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="reguide", seed=0,
                    command="pre-registration: RG, ReGuide (arXiv 2609.38616) reimplemented on frozen pi0.5, oracle and perceived "
                            "poses, against intact I and our frozen front end TV+P on fresh states (docs/SCAFFOLD_RESULTS.md 52)",
                    metrics={"status": "PRE-REGISTERED RG (ReGuide baseline)",
                             "arms": "New: rgo = ReGuide with simulator entity poses every step (PRIVILEGED; paper App. B oracle "
                                     "interface); rgp = ReGuide with poses registered once at the first query from sc_seg's learned "
                                     "U-Net masks + rendered depth back-projection (sc_pu.Estimator, sc_hp's seg_std; paper C.6: no "
                                     "tracking afterwards), hold event ProprioHeld. experiments/rg.py client, run by "
                                     "experiments/rg_roll.sh = tv_attr_roll.sh with only the client moved: server pp_serve_purify "
                                     "--mode intact --comp= --seed 13, client seed 7, --fresh n2_states_fresh, --branches "
                                     "n2_branches_fresh.json, --bind track. Statistics results/flawed/rg_stats.json from "
                                     "experiments/rg_stats.py on the trained K1-K6 LIBERO-Goal demos only (never N1-N6). Reused "
                                     "(hashed): I = results/flawed/pp_atlas_fresh/pi05_<tid>; TV+P = results/flawed/tv_fresh/"
                                     "pi05_tvp_<tid>. pi0.5 only.",
                             "implementation_choices": CHOICES,
                             "tasks": "familiar n2 tids 16,18,20,22,24,26 (N1-N6) and 0,2,4,6,10 (K1-K4, K6); fresh states k = 36-55 "
                                      "= harness trials 24-43 (one episode per task and state).",
                             "mapping": "experiments/rg_read.py as hashed. pi0.5, success matrices [task, state]; paired bootstrap "
                                        "B=2000, numpy default_rng(0), tasks and states drawn jointly (tv_frozen.boot2); CI 2.5/97.5; "
                                        "on N1-N6.  A: d = rgp - I: RG HELPS iff CI lo > 0; RG HURTS iff CI hi < 0; else NO CLEAR "
                                        "DIFFERENCE.  B: d = tvp - rgp: FRONT END WINS iff CI lo > 0; RG WINS iff CI hi < 0; else NO "
                                        "CLEAR DIFFERENCE.  Descriptive only: rgo - I, tvp - rgo, rgo - rgp on N1-N6; K1-K4, K6 success "
                                        "per arm and arm - I (trained-task cost); the atlas's within / cross-chunk excess per arm "
                                        "(pp_atlas estimator via tv_frozen.atlas_est); stage shares; ReGuide diagnostics.  Checks "
                                        "reported: arm client args = I's except out/port/videos, per-episode trial/branch/variant/at = "
                                        "I's, I reproduces pp_atlas_fresh_read.json, tvp reproduces tv_fresh_read.json, state files "
                                        "match tv_fresh's row, reused rollouts and rg_stats.json match this row.",
                             "prediction": PRED,
                             "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE + STATE_FILES + REUSED},
                             "written_before": "any main (fresh-state) rgo or rgp rollout"},
                    bytes=Z, config={}, script="experiments/prereg/reguide.py"))

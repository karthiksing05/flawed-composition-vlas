"""PRE-REGISTRATION ROW of RGPR (docs/notes/rgpr.md): ReGuide (rgp) given our F7R referent resolver, on the F7 states of SR 70, frozen pi0.5, no
training.  BEFORE any main rollout of the arm rgpr.  Predictions: experiments/prereg/rgpr_pred.txt.  Committed before it runs, so the row is
named after the commit that contains this file and the code it hashes.  Run through sbatch only (the ledger records the job):

    python experiments/prereg/rgpr.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import rg_comp as RC                                                         # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "rgpr" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "rgpr_roll")), "rgpr rollouts exist"
assert any(r.get("tag") == "rg_comp" for r in rows), "no rg_comp row"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/rgpr.py", "experiments/rgpr_roll.sh", "experiments/rg_comp.py", "experiments/rg_comp_roll.sh", "experiments/rg.py",
        "experiments/fx_r.py", "experiments/sc_pu.py", "experiments/sc_rv.py", "experiments/sc_seg.py", "experiments/fa1_common.py",
        "experiments/libero_client.py", "experiments/pp_serve_purify.py", "lifelong_ood/policy/parts_flow.py",
        "results/flawed/rg_stats.json", "results/flawed/rg_comp/rg_comp_tasks.json",
        "results/flawed/rg_comp/br7_main.json", "results/flawed/rg_comp/br7_smoke.json",
        "results/flawed/rg_comp/states/rgc_0.npy", "results/flawed/rg_comp/states/rgc_1.npy",
        "results/flawed/rg_comp/states/rgc_2.npy", "results/flawed/rg_comp/states/rgc_3.npy",
        "experiments/prereg/rgpr.py", "experiments/prereg/rgpr_pred.txt", PP + "rgpr_job.sh", PP + "rgpr_read.sh"]
H = {p: fsha(p) for p in CODE}
PRED = open(os.path.join(ROOT, "experiments", "prereg", "rgpr_pred.txt"), encoding="utf-8").read().strip()
SEEN = ("rgp and rc on these exact states and branches were read in SR 70 (pi0.5 F7 pool rgp .356, rc .963).  The arm rgpr is new: no rgpr "
        "rollout has been read.  A smoke of rgpr on the 4 smoke states per task (job 4841110) ran before this row; its logs were read for "
        "crashes and resolver / binding only, and no success rate of it informs the predictions (written before the smoke finished).")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="rgpr", seed=0,
                    command="pre-registration: RGPR, ReGuide (rgp) given the F7R referent resolver, F7 states of SR 70, frozen pi0.5, evaluation only "
                            "(docs/notes/rgpr.md)",
                    metrics={"status": "PRE-REGISTERED RGPR",
                             "arm": "rgp is SR 70's ReGuide RG-perc (rg_comp.py docstring).  RGPR = rgp with the bowl instance from the F7R resolver (R's pick; the other bowl a non-instructed movable, "
                                      "as in rc); the VLA gets the task's original instruction and unmodified images (no unique-name prompt, no role-view "
                                      "paint, no memory slot).  Everything else is rgp's.",
                             "design": dict(tasks=[0, 1, 2, 3], keys=[t[0] for t in RC.TASKS[:4]], states="rg_comp main F7 rows (40 per task), same branches br7_main.json as rgp and rc",
                                            B=RC.B, seed=0, eq_margin=0.10, model="pi05 (frozen, pp_serve_purify intact, server seed 13), client seed 7"),
                             "contrasts": "r1 = rgpr - rgp and r2 = rc - rgpr on the F7 pool, paired, clustered by state, B=2000 (pred file)",
                             "prediction": PRED,
                             "seen": SEEN,
                             "deviations_planned": "none at registration.  rg_comp.py is not edited; experiments/rgpr.py patches rg_comp.Ctx and fx_r.Front "
                                                   "in-process (the resolver without paint; the original prompt; no slot).  rgpr runs under the rc cell of "
                                                   "rg_comp's client with those patches, so its _rc.json side files carry arm 'rc'.",
                             "files_sha256": H,
                             "written_before": "any main-state rollout of the arm rgpr"},
                    bytes=Z, config={}, script="experiments/prereg/rgpr.py"))

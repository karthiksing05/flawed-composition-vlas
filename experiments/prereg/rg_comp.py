"""PRE-REGISTRATION ROW of RG_COMP (SCAFFOLD_RESULTS 70): the composite RC (the F7R referent resolver + the F6 memory slot + ReGuide
transport over a frozen VLA) against ReGuide alone (rgp) and the intact VLA (I) on F7 (relational / negated language) and F6 (a target
seen once, then covered), with trained-task controls K1t / K2t, on frozen pi0.5 (PRIMARY) and pi0 (SECONDARY), NEW states.  BEFORE
any main rollout.  Committed before it runs, so the row is named after the commit that contains this file and the code it hashes.

    python experiments/prereg/rg_comp.py
"""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood import ledger                                              # noqa: E402
import rg_comp as RC                                                         # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "rg_comp" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "rg_comp_read.json")), "rg_comp read exists"
assert not os.path.exists(os.path.join(FL, "rg_comp_roll")), "rg_comp rollouts exist"
prior = {t: [r for r in rows if r.get("tag") == t] for t in ("reguide", "fx_r", "f6o2", "s1_cross")}
assert all(len(v) == 1 for v in prior.values()), {k: len(v) for k, v in prior.items()}


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
META = json.load(open(RC.META, encoding="utf-8"))
STATES = ["results/flawed/rg_comp/states/rgc_%d.npy" % t["tid"] for t in META["tasks"]]
DATA = ["results/flawed/rg_comp/rg_comp_tasks.json", "results/flawed/rg_comp/pcheck.json"] + \
       ["results/flawed/rg_comp/%s_%s.json" % (b, ph) for b in ("br7", "br6") for ph in ("smoke", "main")]
CODE = ["experiments/rg_comp.py", "experiments/rg_comp_roll.sh", "experiments/rg.py", "experiments/rg_stats.py", "experiments/fx_r.py",
        "experiments/fx_tasks.py", "experiments/fx_client.py", "experiments/f6o_tasks.py", "experiments/f6o2_tasks.py",
        "experiments/f6o2_client.py", "experiments/sc_pu.py", "experiments/sc_rv.py", "experiments/sc_seg.py", "experiments/fa1_common.py",
        "experiments/fa1b_common.py", "experiments/libero_client.py", "experiments/pp_serve_purify.py", "experiments/serve_guided.py",
        "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/serve.py",
        "results/flawed/rg_stats.json", "results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json",
        "results/flawed/fx/fx_tasks.json", "results/flawed/f6o/f6o_tasks.json",
        "experiments/prereg/rg_comp.py", "experiments/prereg/rg_comp_pred.txt", "experiments/prereg/rg_comp_seen.txt"] + \
       STATES + DATA + [PP + "rgc_build.sh", PP + "rgc_job.sh", PP + "rgc_read.sh"]
H = {p: fsha(p) for p in CODE}
same = {}
for tag, r in prior.items():
    fs = r[0]["metrics"].get("files_sha256", {})
    same[tag] = {p: (H[p] == fs[p]) for p in CODE if p in fs}
print("[rg_comp prereg] files shared with earlier rows, sha256 equal: %s" % json.dumps(same), flush=True)
PC = json.load(open(RC.PCHECK, encoding="utf-8")).get("_summary", {})
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "rg_comp_seen.txt"), encoding="utf-8").read().strip()
PRED = open(os.path.join(ROOT, "experiments", "prereg", "rg_comp_pred.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
RUNS = [[m, t["key"], arm, "PRIMARY" if m == RC.PRIMARY else "SECONDARY", t["tid"]] for m in RC.MODELS for t in META["tasks"] for arm in t["arms"]]
print(ledger.record("m3_prereg", bench="libero_goal", tag="rg_comp", seed=0,
                    command="pre-registration: RG_COMP, the composite RC (F7R resolver + memory slot + ReGuide transport) against "
                            "ReGuide alone (rgp) and the intact VLA on F7 (relational / negated) and F6 (seen once, then covered) "
                            "with trained-task controls, frozen pi0.5 (PRIMARY) and pi0, NEW states (docs/SCAFFOLD_RESULTS.md 70)",
                    metrics={"status": "PRE-REGISTERED RG_COMP",
                             "runs": RUNS,
                             "design": dict(f7_k=META["sampler_k"]["f7"], f6_k=META["sampler_k"]["f6"], states=META["states"],
                                            latch=RC.LATCH, ugate=RC.UGATE, agate=RC.AGATE, floor=RC.FLOOR, ni_margin=RC.NI_MARGIN,
                                            B=RC.B, rule=META["rule"]),
                             "arms": RC.__doc__.split("THE ARMS (")[1].split("END ARMS")[0].strip(),
                             "information": "rgp and rc read the learned U-Net masks, the rendered depth and the camera calibration "
                                            "(sc_pu.Estimator, SR 52 / 55 / 65), ProprioHeld, and ReGuide's demonstration statistics "
                                            "(rg_stats.json, trained-task demos).  rc and f7r also read F7R's template parser over the "
                                            "instruction (fx_r.py) with the learned masks and depth; the memory slot reads the "
                                            "learned plate mask, the RGB at the first query, the depth and the camera model "
                                            "(f6o2_tasks.Memory).  The depth render and the calibration are privileged in simulation "
                                            "in every non-I arm.  The simulator is read for diagnostics only (logged, never fed).",
                             "labels": RC.MAPPING,
                             "pcheck_summary": PC,
                             "prediction": PRED,
                             "seen": SEEN,
                             "same_sha_as_prior_rows": same,
                             "deviations_planned": "rgp's instance rule (the largest bowl component; the second bowl a non-instructed "
                                                   "movable) is ours: the scene-1 rgp averages a two-instance class.  rc's ReGuide binds "
                                                   "the resolver's instance through the same restriction.  The memory slot's trigger is "
                                                   "a latch on the learned plate mask (< LATCH of its first-query size), not SR 45.6's "
                                                   "cover-move step; the paint rule is SR 45.6's.  ReGuide's one-shot registration (C.6) "
                                                   "is the slot's pose memory in rc and is also present in rgp.  No reference planner and "
                                                   "no fx reference gate (the F7 / F6 gates use I's own controls).  Client seed 7 and "
                                                   "server seed 13 kept from SR 45.  GR00T, X-VLA and SmolVLA not run.  Ledger tree dirty "
                                                   "(other agents' files).",
                             "files_sha256": H,
                             "written_before": "any main-state rollout of this probe"},
                    bytes=Z, config={}, script="experiments/prereg/rg_comp.py"))

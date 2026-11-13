"""PRE-REGISTRATION ROW of FX_RA (SCAFFOLD_RESULTS 45.4): F7R with the role view the proposal specifies (every non-referent movable
object painted, experiments/fx_ra.py) on NG for pi0.5, GR00T, pi0, X-VLA, SmolVLA and on SP for pi0 and GR00T; LIBERO-Goal scene,
main states 4-23.  BEFORE any main-state fx_ra rollout.

    python experiments/prereg/fx_ra.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
PP = "/coc/flash10/ksingara3/.claude_tmp/pp"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "fx_ra" for r in rows), "already registered"
assert all(any(r.get("tag") == t for r in rows) for t in ("fx", "fx_r", "fx_more")), "fx / fx_r / fx_more registrations missing"
for p in ("results/flawed/fx_ra", "results/flawed/fx_ra_read_main.json"):
    assert not os.path.exists(os.path.join(ROOT, p)), "main output exists: %s" % p


def fsha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


FILES = ["experiments/fx_ra.py", "experiments/fx_r.py", "experiments/fx_client.py", "experiments/fx_tasks.py", "experiments/fx_read.py",
         "experiments/sc_rv.py", "experiments/sc_seg.py", "experiments/libero_client.py", "experiments/pp_serve_purify.py",
         "experiments/serve_guided.py", PP + "/fxa_roll.sh", PP + "/fxa_job.sh", PP + "/fxa_readjob.sh"]

Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="fx_ra", seed=0,
                    command="pre-registration: F7R with every distractor painted (docs/SCAFFOLD_RESULTS.md 45.4)",
                    metrics={"status": "PRE-REGISTERED FX_RA",
                             "arms": "f7rA: F7R (fx_r.Front: parser, learned-segmenter resolver, unique-name prompt, the distractor "
                                     "bowl's tracked mask painted) plus every pixel sc_seg labels cream_cheese_1, wine_bottle_1 or "
                                     "plate_1 painted magenta (sc_rv.paint_many) in both cameras from step WAIT on; the plate is left "
                                     "unpainted only when it is the instruction's target (SP), so on NG every movable object except the "
                                     "referent is painted.  uniA: F7's unique-name controls (other bowl removed and hidden, their own "
                                     "prompt) plus the same object paint.  Models served as in fx (pi05, groot: pp_serve_purify.py "
                                     "intact seed 13) and fx_more (pi0, xvla, smolvla: serve_guided.py --mode ours seed 13); "
                                     "fx_branches_main.json, main states 4-23.  Cells: NG x {pi05, groot, pi0, xvla, smolvla}; "
                                     "SP x {pi0, groot}.",
                             "mapping": "experiments/fx_ra.py read as hashed, per cell, arm f7rA.  delta = R(f7rA) - R(relational, the "
                                        "fx / fx_more frozen rollouts on the same states and seeds), paired over states, fx_read.boot "
                                        "B=2000 seed 0.  First match: NO POWER iff the fx reference < .60 (U is not gated: the "
                                        "ceiling is the measured uniA arm); FIXED iff delta > .10 with CI lower bound > 0 and C(f7rA) "
                                        "CI lower bound > .75; NOT FIXED iff delta CI upper bound < .10; else PARTIAL.  Ceiling: ceil "
                                        "= U(uniA) - U(original unique-name control, fx / fx_more), paired; CEILING LIFTED iff ceil > "
                                        ".10 with CI lower bound > 0; CEILING HOLDS iff ceil CI upper bound < .10; else UNRESOLVED.  "
                                        "f7rA - F7R (paint-all's share), per-task success, grasp counts, resolver accuracy, paint IoU "
                                        "and referent leak: reported with no verdict.",
                             "prediction": ("Written before the smoke read was seen.  FIXED on pi0.5 NG and GR00T SP (F7R already FIXED there; "
                                           "paint-all should not hurt); PARTIAL on pi0 SP; GR00T NG PARTIAL; pi0, X-VLA, SmolVLA NG "
                                           "PARTIAL (the plate-bowl failure is a location habit, so painting the cheese moves the "
                                           "grasp off the cheese but not onto the plate bowl; C(f7rA) stays below .75).  Ceiling: "
                                           "UNRESOLVED or HOLDS on every NG cell except GR00T NG (LIFTED, low confidence); HOLDS on "
                                           "the SP cells.  f7rA - F7R within +-.10 everywhere.  Low confidence overall."),
                             "seen": ("Smoke on states 0-3, GR00T NG only (job 4677930 on tail-lab, read 4677931; "
                                     ".claude_tmp/pp/fxa_smoke/fxa_read_smoke.json), seen AFTER the prediction above was written: "
                                     "NG NOT FIXED, CEILING HOLDS on n=4 (f7rA = F7R = R = U = uniA = .50; uniP 0/4 under uniA, "
                                     "grasping none 2, cheese 2, against cheese 4/4 unpainted; C(f7rA) .80).  Resolver 8/8.  "
                                     "Object paint IoU .95-.98 except the wrist wine-bottle class under f7rA (IoU .07, recall .99: "
                                     "the segmenter over-labels wine bottle in the wrist view; uniA .93); referent leak <= .009.  "
                                     "No design, gate, mapping or threshold was changed after the smoke.  Before the smoke the "
                                     "overcap smoke job 4677741 was cancelled unrun (queue wait) and resubmitted on tail-lab."),
                             "files_sha256": {p: fsha(p) for p in FILES},
                             "written_before": "any main-state fx_ra rollout"},
                    bytes=Z, config={}, script="experiments/prereg/fx_ra.py"))

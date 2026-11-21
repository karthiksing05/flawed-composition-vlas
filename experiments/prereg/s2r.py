"""PRE-REGISTRATION ROW of the SR 89 step-2 read (scene 2, LIVING_ROOM_SCENE5; docs/notes/scene2_plan.md section 3, docs/notes/sr89_build.md): RAA-sL-S2 (pi05_g8v_raas_ls2_ae), W5-S2 (serving only, scene-2 plate mu table),
the resolver / pose-memory / channel ablations noR / noM / noC, the kitchen RAA-sL zero-shot (step0), and I / rgp, on FRESH states (seed 1,100,000 + k, k 4-63).  BEFORE any registered rollout.
Predictions: experiments/prereg/s2r_pred.txt (committed before this row).    python experiments/prereg/s2r.py        (run through sbatch)
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CKPT = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_g8v_raas_ls2_ae_tl6000"
KCK = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_g8v_raas_l_ae_main6000"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "s2r" for r in rows), "already registered"
for d in ("sr89/main/raasls2", "sr89/main/w5", "sr89/main/w5_noR", "sr89/main/w5_noM", "sr89/main/w5_noC", "sr89/main/step0", "sr89/main/ref", "sr89/s2r_read.json"):
    assert not os.path.exists(os.path.join(FL, d)), d + " exists"
assert os.path.isdir(CKPT) and os.path.isdir(KCK)
for p in ("sr89/main/O/s2_living_0.npy", "sr89/main/branches_main.json", "sr89/plate_axis.json"):
    assert os.path.exists(os.path.join(FL, p)), p


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


CODE = ["experiments/wa_s2.py", "experiments/wa_s2r.py", "experiments/s2r_prep.py", "experiments/s2r_read.py", "experiments/raasls2_mu.py", "lifelong_ood/policy/raasls2_mu.json", "lifelong_ood/policy/backbone_w.py",
        "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/raasw5_mu.json", "experiments/s2.py", "experiments/s2rg.py", "experiments/s2rg_roll.sh", "experiments/prereg/s2r_pred.txt",
        "docs/notes/scene2_plan.md", "docs/notes/sr89_build.md", "results/flawed/sr89/plate_axis.json", "results/flawed/sr89/probe/calib.json", "results/flawed/sr89/main/branches_main.json", "results/flawed/sr89/main/O/s2_living_0.npy",
        PP + "sr89r_roll.sh", PP + "s2r_prereg.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "s2r_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-20.  Known when written: SR 50 / 66 / 68 / 69 / 72 (scene-2 results on other states), the SR 89 step-0 probe (20 states, seed 1,090,000 + k: kitchen RAA-sL .775 K / .525 N) and the SR 89 K fit check of RAA-sL-S2 (channels mask, maskoff, sim; W5-S2 and RAA-sL-S2) "
        "on those 20 probe states (not on the registered range), the 100 training cycles' release points (raasls2_mu.json, computed once before any rollout).  The registered states (seed 1,100,000 + k) were drawn and checked unequal to every earlier "
        "scene-2 state (results/flawed/sr89/main); k 0-3 are the smoke (plumbing, not read).  Channel: --chan sim (privileged positions + onset offset), chosen after the probe-state fit check (maskoff K1+K2 .65 / .475 < .70; sim .70), see experiments/prereg/s2r_pred.txt.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_10", tag="s2r", seed=0,
                    command="pre-registration: SR 89 step 2, scene 2: RAA-sL-S2 / W5-S2 / noR / noM / noC / kitchen RAA-sL zero-shot / I / rgp on FRESH scene-2 states (docs/notes/scene2_plan.md section 3)",
                    metrics={"status": "PRE-REGISTERED SR 89 step 2", "arm": "RAA-sL-S2 = %s (where_own 0); W5-S2 = the same checkpoint, where_own 5 with the scene-2 plate mu table; step0 = %s" % (CKPT, KCK),
                             "states": "scene 2 (s2_living) tids 0-3 (K1 K2 N1 N2), states seed 1,100,000 + k, k 4-63 (60)", "mapping": "experiments/prereg/s2r_pred.txt (read: experiments/s2r_read.py)",
                             "prediction": PRED, "seen": SEEN, "files_sha256": {p: fsha(p) for p in CODE}, "written_before": "any registered SR 89 step-2 rollout"},
                    bytes=Z, config={}, script="experiments/prereg/s2r.py"))

SR 78b plan: re-serve of the SR 78 RAA-s checkpoint WITH its servo.  Written 2026-11-19 before the prereg row and before any SR 78b rollout.

Why.  SR 78 served pi05_g8v_raas_ae_tl6000 through PolicyBackbone.sample, whose own Euler loop never added the servo term (only PI0PytorchS.sample_actions adds it).  The SR 78
read (N1-N6 .028, K1-K6 .092, NOT SUPPORTED) therefore stands as "RAA-s served WITHOUT servo" and is NOT overwritten or edited; its results/flawed/raas_* files stay.  The SR 78
diagnosis (docs/notes/sr78_diag.md) located the cause.  SR 78b serves the SAME checkpoint (no retraining, same export) with the servo as designed.

Fix (serving only).  lifelong_ood/policy/backbone.py, PolicyBackbone.sample: after the Euler loop, when the model config has a nonzero servo_k, add
servo_norm_torch(model.config, prefix.state, A)[..., :3] to dims 0-2 of every step of the normalised chunk (the same term and space as PI0PytorchS.sample_actions).  Configs
without servo_k never reach the branch.  sample_composed and sample_poe are not touched: raas is served with hooks None (raa.py serve), which goes only through
bk.sample.  Test (sbatch, experiments/raas_b_test.py): the pre-change sample (git HEAD copy) and the patched one are torch/numpy array_equal on fixed noise for the frozen pi0.5
(pi05_libero), pi05_g8v_plain_ae and pi05_g8v_raac_ae; for raas the patched chunk equals PI0PytorchS.sample_actions on the same noise and GuidedPolicy.infer equals
unnormalize of it; RAA-c's state_proj is read on the served path.  experiments/raas_diag.py is rerun unchanged (its "served as-is" row is now the servo-added chunk; its
"+ servo" row now double-counts and is not used).

Registered design.  Hypothesis, contrasts S1a / S1b / S1 / S2 / S3a / S3b / SK, X1-X6, states (dev trials 4-23 + new k 540-579, 60 per task, 12 tasks), bootstrap (B = 2000,
rng 0), margins and label rules are SR 78's, unchanged (experiments/prereg/raas.py, raas_pred.txt, raas_plan.md).  Added unlabelled rows X7 / X8 = SR 78b minus SR 78.
Read: experiments/raas_b_read.py (a copy of raas_read.py: arm raasb = files raas_b_{main,rx}_<tid>.json; raas_read.py untouched).  Prereg: experiments/prereg/raas_b.py
(tag raas_b), predictions experiments/prereg/raas_b_pred.txt.

Run.  4-episode smoke (raao_smoke_branches.json, tasks 0 and 22, smoke states 0-1) with the patched server; then raa_roll.sh unchanged with ARM=raac
CONFIG=pi05_g8v_raas_ae CKPT=.../pytorch/pi05_g8v_raas_ae_tl6000 NAME=raas_b on the main and rx states / branches, 12 tasks, overcap with the full exclude list (four
jobs as SR 78); read via sbatch (selftest first); exploratory raa_instr (N1 / N4 / N6 / K5 x 20, trials 24-43) NAME=raas_b, read by experiments/raas_b_instr_read.py.
Nothing is overwritten: all new result names carry raas_b.

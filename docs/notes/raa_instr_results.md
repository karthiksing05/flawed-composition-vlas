# RAA place-step instrumentation: results (EXPLORATORY diagnostic; plan in raa_instr_plan.md)

Logging: every control step, `experiments/raa_instr.py`. Analysis: `experiments/raa_instr_read.py` -> `results/flawed/raa_instr.json`
(jobs: smoke 4770692 RAA, 4770693 rgp; main 4770730 RAA N1+N4, 4770732 RAA N6+K5, 4770731 rgp N1+N4, 4770728 rgp N6+K5; unlogged rgp
control 4770950; read job 4771002). Tasks N1, N4, N6, K5; trials 24-43 (n2_states_rx); pi05_g8v_raa_ae_tl6000 and frozen pi0.5 + rgp.
Per-step npz logs are in results/flawed/raa_instr/{raa,rgp} (not committed). raa_instr_prelim*.json are superseded intermediate reads.

Identity: RAA 80/80 episodes equal the unlogged record in (steps, success). rgp 71/80; divergences N1 2, N4 4, N6 3, K5 0. An unlogged
re-run of rgp (control, N1/N4/N6, 60 episodes) differs from the original record in 4/60 and from the logged run in 5/60, so rgp is not
exactly reproducible by itself (unlogged vs original 93% equal, logged vs original 85% equal; not separable at n=60).
The recomputed On() terms equal the simulator's `on_full` at every step of all 160 episodes; 0 snapshot errors.
Success (of 20): RAA N1 11, N4 5, N6 6, K5 7; rgp 19, 18, 16, 14.

Caveat on categories: a wine bottle ends lying (tilt change 30-90 deg) in successful placements, so tilt is a failure category only for
the cream cheese (N1); a first read that called bottle tilt "TIPPED" was wrong and is superseded.

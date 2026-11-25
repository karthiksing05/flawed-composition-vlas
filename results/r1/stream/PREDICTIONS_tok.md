# Slot-token R1 policy and idea B inside the slot token, registered 2026-11-24 before any job was launched

Follows PREDICTIONS_bc.md (BC-P0 falsified at 4/9 with input-side adapters on the single-state-token base). One principle: a part can only change what flows through the slot it owns.
Code: experiments/r1/r1tok.py. Outputs: results/r1/tok/.

## Base (slot tokens)
TokPolicy = r1s FlowPolicy with the state_in token replaced by 4 slot tokens (object dims 0-14, obstacle 14-28, goal 28-45, proprio 45-77; each its own Linear-GELU-Linear embed
plus a learned slot embedding), then task and time tokens and the 16 action tokens; full attention across all tokens. The 16-dim indicator stays stripped (free cond, one task embedding).
Trained exactly as the r1 base: same 23 base cells, 2300 episodes, same normaliser (reused from free/base.pt), 60k steps, lr 3e-4, bs 256, one base shared by all seeds.
## Reference arm (labelled routed parts)
Per seed 0-2, rank-1 LoRA per task (HB OW TC) on the new base, 3000 steps, lr 2e-3, r1s demo draws, smax, labelled routing, r1s eval (same groups, episode counts, eval seed).
- TK-P1: mean over 3 seeds slot-token routed pairs >= .45 (current single-token base .489) AND panel >= .80 (current .867). Falsifier: pairs < .40 -> stop and report (the base is worse, not B's fault); .40-.45 or panel < .80 with pairs >= .45 = inconclusive, B still runs and is read with that caveat.
## Idea B on the slot-token base (run only if TK-P1 is not falsified)
Hypothesis f in {object, obstacle, goal}: a new part owning slot f = zero-init adapter MLP (hidden 64) on that slot's raw normalised input, added INSIDE that slot's token embedding;
base and everything else frozen; AdamW lr 1e-3, 1000 steps, bs 256. Score = 2-fold held-out flow loss over the task's own 16 demos (r1s.val_loss, fixed noise). Same task draws (r1s seeds 0-2)
and same null (16 demos from 3 of the 4 panel cells, 4 triples x 3 seeds = 12 draws) as PREDICTIONS_bc.md. Labels only score.
- BC-P0: lowest-loss hypothesis = true slot on >= 8/9 task draws (object for HB, obstacle for OW, goal for TC). Falsifier <= 6/9 (7/9 inconclusive).
- BC-P1: margin (second best minus best) > 95th percentile of the null margin on >= 8/9 draws.
- BC-P4: base flow loss on the task demos > null 95th percentile on >= 8/9 draws (previous base: 6/9, OW inside null).
- BC-P2 (create call, reported only): max gain over base > null 95th percentile on 9/9.
Compute: base and parts and B fits on one GPU each (A40 if it starts within a few hours, else any >= 24GB GPU; choice stated in the results); eval on CPU as for the FD jobs.

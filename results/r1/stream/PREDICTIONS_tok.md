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

## Results (2026-11-24): TK-P1 FALSIFIED, B not run (stop rule)
Compute: base trained on an A40 (job 4857096, 60k steps, final loss ~.282; started within 1 minute), parts on A40 (4857104/4857105/4857106 chain; 3 x 3000 steps, ~4 min), eval on CPU (24 jobs, 4857107-4857130 range), report 4858528; smoke chain 4857084-86 on CPU.
Slot-token base, labelled routed smax parts (r1s eval counts):
| seed | pairs | HBOW | HBTC | OWTC | singles | panel | triple |
|---|---|---|---|---|---|---|---|
| 0 | .233 | .300 | .200 | .200 | .533 | .70 | .20 |
| 1 | .311 | .233 | .433 | .267 | .504 | .70 | .00 |
| 2 | .289 | .167 | .300 | .400 | .570 | .70 | .00 |
| mean | .278 | | | | .536 | .70 | .067 |
Reference (single-token base): pairs .489, singles .615, panel .867. TK-P1 needed pairs >= .45 and panel >= .80; pairs < .40 falsifies it.
The panel is served by the base alone (no part on), so panel .70 vs .867 says the slot-token base itself is worse on held-out base combinations; the routed parts then add less on singles (.536 vs .615) and pairs. Per the registered rule: the base is worse, not B's fault; the BC-P0/P1/P4 rerun on this base was not launched (code for it is in r1tok.py, commands bc and bcreport, smoke-tested).

## TK-D1 diagnostic (registered 2026-11-24 before any job; user-directed, then stop)
Question: is the slot-token base's deficit (panel .70, pairs .278) a training/capacity issue or the token split itself?
Changes, one principle (each slot must be usable on its own, which composition needs):
- Base: TokPolicyD (null token per slot), 120k steps (2x), slot-token dropout p = .15 per slot token per sample in base training only (token replaced by that slot's learned null); otherwise as TokPolicy (lr 3e-4, bs 256, same data and normaliser). Dropout is off in parts, B fits and eval.
- Parts: rank 8 instead of 1 (3000 steps, lr 2e-3, same draws), labelled routed smax, r1s eval counts, seeds 0-2.
- Control (cheap): the existing single-token r1s base (results/r1/free/base.pt) with rank-8 parts, same draws, same eval.
- TK-D1: panel >= .80 AND labelled routed pairs >= .45 (3-seed means). Falsifier: panel < .75 -> the token split is the problem; stop. Between: inconclusive, B not run.
- If TK-D1 passes, run B (bc, bcreport) on this base with the earlier registered predictions: BC-P0 >= 8/9 (falsifier <= 6/9), BC-P1 margin > null p95 on >= 8/9, BC-P4 base loss > null p95 on >= 8/9.
Code: r1tok.py flags --v2 (results/r1/tok2) and --ctl (results/r1/tokctl), --rank, --slot_drop.

### TK-D1 results (2026-11-24): INCONCLUSIVE (panel .767 in [.75 .80), pairs .396 < .45); B not run (rule: B only on a pass)
Compute: smoke chain on CPU; base 120k steps on an L40S (job 4859463; the A40 queue estimate was ~1 h and later >1 day, so L40S, 48 GB, started in ~3 min); parts rank 8 on RTX 6000 (24 GB; jobs 4860345-47); eval CPU; reports 4860420 (TK-D1) and 4860421 (control).
Labelled routed smax, rank-8 parts, r1s eval counts:
| arm | seed | pairs | singles | panel | triple |
|---|---|---|---|---|---|
| slot-token, dropout .15, 120k (TK-D1) | 0 | .467 | .548 | .70 | .40 |
| | 1 | .478 | .578 | .80 | .80 |
| | 2 | .244 | .585 | .80 | .00 |
| | mean | .396 | .570 | .767 | .400 |
| single-token base, rank 8 (control) | 0 | .656 | .607 | .85 | .20 |
| | 1 | .578 | .541 | .95 | .50 |
| | 2 | .478 | .593 | .85 | .20 |
| | mean | .571 | .580 | .883 | .300 |
Reference (rank 1): single-token pairs .489 panel .867; slot-token (60k, no dropout) pairs .278 panel .70.
Reading: 2x steps plus slot dropout lifts the slot-token panel .70 -> .767 and pairs .278 -> .396 (rank 8 included), but it stays below the single-token base on the same parts (panel .883, pairs .571). Rank 8 helps the single-token pairs (.489 -> .571), so part of the earlier gap is rank, and the rest is the token split. Panel is the base alone, so the base remains worse (.767 vs .883); seed spread is large (pairs .244-.478).

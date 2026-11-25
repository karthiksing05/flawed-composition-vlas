# Idea B (explaining away by behaviour cloning), gate 1 (P4-style), registered 2026-11-24 before any job was launched

Plan: compositional-architecture/docs/PLAN_PARTWISE_POLICY.md, section 3 item 2 and P4. Code: experiments/r1/r1bc.py. No labels enter the pick; labels (HB = object slot,
OW = obstacle slot, TC = goal slot) score it only.

## Hypotheses (how they are formed)
The R1 base (results/r1/free/base.pt) has ONE state token (state_in), so true per-slot tokens would need a retrained base. Instead each hypothesis f in {object, obstacle, goal}
adds a new part owning slot f: a zero-initialised adapter MLP on the normalised state slice of slot f only (object dims 0-14, obstacle 14-28, goal 28-45; hidden 64) whose output
is added to the state token. The base, every existing part and all other slots stay frozen ("known values"); only the new adapter trains (AdamW, lr 1e-3, 1000 steps, bs 256).
Proprio and the relative dims are never an owned slot. Slot boundaries are the simulator observation spec (the plan's stated perception prior). The existing rank-1 parts are not reused
(they carry no slot ownership, so they cannot distinguish hypotheses). Score = held-out flow-matching loss (r1s.val_loss, fixed noise and time, 8 reps), 2-fold over the task's own
16 demos (fit on 8, score the other 8, swap, average); no extra demos. Also logged: base loss with no fit (P4, B-lite).
Draws: the stream's own 16-demo draws (r1s.sources/draw, seeds 0-2, tasks HB OW TC) = 9 task draws.
Null: the same procedure on 16 demos drawn from 3 of the 4 held-out panel cells (known values, held-out combinations; 4 triples x seeds 0-2 = 12 null draws), taken as the "held-out base
demos" (the base was trained on all train cells without a held-out split, so panel cells are the only unseen base-value demos; the plan's 10% held-out retrain is not done).
gain_f = base val loss - fitted val loss of hypothesis f. Create call (for step 2 only): create iff max_f gain_f exceeds the 95th percentile of the null max_f gain.

## Predictions
- BC-P0: the lowest-val-loss hypothesis is the true changed slot on >= 8/9 task draws (3 tasks x seeds 0-2). Falsifier: <= 6/9 -> stop (7/9 = inconclusive).
- BC-P1: the margin (second-best minus best val loss) exceeds the 95th percentile of the null margin distribution (12 panel draws) on >= 8/9 task draws.
- BC-P4 (diagnostic only): base flow loss on the 9 task draws exceeds the null 95th percentile of base loss on panel draws on >= 8/9 draws.
- BC-P2 (create call, reported only): max gain exceeds the null 95th percentile on 9/9 new-value draws.
Honest caveat registered in advance: the adapters are input-side and low-capacity; if every hypothesis fits the same amount, BC-P0 fails and that says input-side slot parts cannot
explain away dynamics-borne values, not that B fails with per-slot tokens in a retrained base.

## Results (2026-11-24; jobs: runs 4856692-4856712 (21 CPU jobs, one per task or null draw), report 4857027; results/r1/stream/bc/report_bc.json)
Deviation: the A40 queue estimated a start next day, so fits ran on CPU (same b1_cpu.sbatch, overcap/short, same --exclude); runs split per draw. Hypotheses, steps and scoring as registered.
All three adapters cut val loss by nearly the same amount on every draw (gains agree within .001-.003), so the pick is decided at the 1e-4 level.
| seed | task | truth | pick | margin (null p95 .00056) | base loss (null p95 .318) | max gain (null p95 .0163) |
|---|---|---|---|---|---|---|
| 0 | HB | object | obstacle | .00027 | .360 | .057 |
| 0 | OW | obstacle | goal | .00020 | .311 | .015 |
| 0 | TC | goal | goal | .00062 | .347 | .036 |
| 1 | HB | object | goal | .00041 | .361 | .065 |
| 1 | OW | obstacle | goal | .00034 | .315 | .017 |
| 1 | TC | goal | goal | .00009 | .347 | .036 |
| 2 | HB | object | object | .00009 | .354 | .058 |
| 2 | OW | obstacle | object | .00058 | .315 | .015 |
| 2 | TC | goal | goal | .00099 | .342 | .033 |
- BC-P0: 4/9 hits (goal 3/3, object 1/3, obstacle 0/3). FALSIFIED (<= 6/9). Null picks (panel draws): obstacle 10/12, goal 2/12, i.e. the pick is a slot bias, not evidence.
- BC-P1: margin > null p95 on 3/9. FAIL.
- BC-P4: base loss > null p95 on 6/9 (HB and TC 6/6, OW 0/3). FAIL as registered. The base loss separates HB and TC (new object, new objective) from known values; OW (ObjectWall) sits inside the null.
- BC-P2: max gain > null p95 on 7/9 (OW creates on 1/3). FAIL.
Reading: the input-side slot adapter explains away the shift as a generic "fit these demos" gain, not as slot-specific. Hollowbox and Trashcan show up as a large gain over base, but which slot fits best is noise. The stop rule applies: no step 2 (stream design for the compose/create call) is proposed on this evidence. This does not test B with per-slot tokens in a retrained base.

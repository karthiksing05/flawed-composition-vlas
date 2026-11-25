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

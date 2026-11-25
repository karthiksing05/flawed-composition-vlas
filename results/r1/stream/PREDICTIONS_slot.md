# Slot-keyed parts, Gate 0 (offline, no training, no rollouts). Registered 2026-11-24 before any Gate 0 job ran.

Spec: compositional-architecture/docs/PLAN_PARTWISE_POLICY.md (sections 4, 7 P0, 8). Code: experiments/r1/r1slot.py. Output: results/r1/stream/slot_gate0.json.

## Data and features (fixed now)
- Stored episodes only (data/composuite_expert, success-filtered as r1.load_cell, indicator stripped). Feature source: the t = 0 observation of each episode.
- Four slots, world-frame pose only (relative dims excluded): object = dims 0-7, obstacle = 14-21, goal = 28-35, proprio = 45-77.
- Bank: all t = 0 states of all base-train cells (r1.make_split(0)["train"], 18 cells). Each slot's features are standardised by the bank mean and std (std floored at 1e-3).
- Task draws: the 9 draws of r1s.sources(seed) x r1s.draw (tasks HB, OW, TC; seeds 0, 1, 2; rng = RandomState(seed), draw order HB, OW, TC; 16 demos each). Labels (HB = object slot, OW = obstacle, TC = goal) are used only to score.

## Statistic and threshold (fixed now)
- Per slot, energy distance between the 16 task t = 0 features and the bank: E = 2 mean|x-y| - mean|x-x'| - mean|y-y'| (Euclidean, standardised features).
- Null: 1000 draws that mimic a task draw (3 random base cells, 16 episodes split 6/5/5), each compared with the bank minus those 3 cells (the whole cells are left out, as a new value comes from unseen cells). Null seed 20261124.
- A slot clears when its permutation-style p = (1 + #{null >= observed}) / 1001 is < .01 (alpha = .01 per slot per task, no multiplicity correction). Slots that clear are the task's owned slots; no slot clearing = compose.
- Layout check: obstacle slot exactly zero on every None cell (all dims 14-28), and non-zero on non-None cells.

## Routing (fixed now)
- Per seed, each part owns the slots that cleared and stores its 16 demos' t = 0 features there. Slot bank = base states plus the stored features of the parts that own that slot.
- A part is on for a query state iff, in every slot it owns, the nearest neighbour (Euclidean, standardised) of the query's feature is a state of that part.
- Scored on every t = 0 state of every cell in r1s.all_cells() that is in the groups single_HB, single_OW, single_TC, pair_*, triple and panel (dropped cells removed), excluding the 48 demo episodes used as keys. Truth = r1s.newvals(cell). Metric: exact-set agreement (all three parts' on/off match), per seed and pooled over seeds; also per part.

## Predictions
- P0a: the owned slot is named correctly on 9 of 9 task draws (HB: object, OW: obstacle, TC: goal) and no other slot clears (proprio never).
- P0b: routing exact-set agreement >= 95% of scored t = 0 states (pooled over seeds; per seed and per cell reported).
- Falsifier: fewer than 8 of 9 draws correct, or routing < 90%. Between (8/9 correct counts as the lower bound; 90-95% routing) = marginal, reported as such. On falsification: the 8-step fallback (not run in Gate 0), then idea B.
- Not run here: B-lite (P4), rollouts, Gate 1.

## Errata (2026-11-24, before the final run; no statistic, threshold or criterion changed)
- The base bank has 23 train cells (27 base cells minus 4 panel), not 18.
- The layout check covers dims 14-21 only (obstacle world pose, exactly zero on None cells). Dims 21-28 (obstacle_to_eef) are non-zero on None cells and were never used.
- Job 4856600 stopped on a layout assertion (made a logged check), 4856633 was cancelled (slow null; replaced by an exact-sum null, same statistic), 4856741 is the reported run.

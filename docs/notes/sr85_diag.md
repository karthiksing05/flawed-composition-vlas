# SR 85 diagnostic: why W1 (module-owned xy) fails on elevated targets (EXPLORATORY, not registered)

Written 2026-11-20, before any W3 code.  Script experiments/sr85_diag.py (output results/flawed/sr85_diag.json, job 4810569); demo clearances experiments/sr85_clear.py (results/flawed/sr85_clear.json, job 4810571).
Hypothesis (SR 84 limits): the W1 servo drags the held object sideways into the cabinet / rack before the expert has lifted it clear.

## Runs
Instrumented rollouts (raa_instr.py logs every control step: object, target, end effector, executed action, contact flag), RAA-sL checkpoint, 20 states per cell (the rx states, trials 24-43, a subset of the 60 registered states).
W1 = `--where_own 1`, jobs 4810437 (K3, tid 4), 4810439 (K5, tid 8); RAA-sL control (same states, no module-owned xy), jobs 4810438, 4810440.  Directories results/flawed/raa_instr/raasw1K and raaslK.
Not part of any registered read; the success rates here are 20 states each and are not used for a label.

## Findings (table is medians over episodes unless stated)
| | K3 W1 | K3 RAA-sL | K5 W1 | K5 RAA-sL |
|---|---|---|---|---|
| success | 0 / 20 | 20 / 20 | 0 / 20 | 16 / 20 |
| target reference height above the object's start | 15.7 cm | 15.7 | 16.7 | 16.7 |
| object rise above its start when the object has moved 3 cm sideways | **5.7 cm** | 11.8 | **6.4 cm** | 15.5 |
| object z minus target z at that moment | **-9.9 cm** | -3.8 | **-10.4 cm** | -1.3 |
| steps from first 2 cm lift to 3 cm sideways | 4.5 | 7 | 4.5 | 11 |
| episodes with target contact while > 5 cm (xy) from the target | **20 / 20** | 9 / 20 | 2 / 20 | 18 / 20 |
| episodes with >= 5 steps of full-speed xy command and no object motion (stall), > 5 cm out | **20 / 20** (10 of them with target contact) | 0 | **16 / 20** | 4 / 20 |
| final object-target xy distance | 20.1 cm | 4.9 | 9.8 | 9.7 |

(Contact flag = LIBERO contact(target, object); for K5 the blocker may be another fixture, which this log does not name.)

## Reading
The hypothesis holds.  Under W1 the module commands xy at the demos' clip from the first held step: the object moves 3 cm sideways 4-5 steps after it leaves the table, when it has risen only ~6 cm and is ~10 cm below the target's
reference height, so it pushes into the cabinet side (K3: every episode touches the target while still > 5 cm away, every episode stalls with the xy command saturated) or stalls on the rack (K5: 16 / 20).  RAA-sL (expert xy)
lifts 12-15 cm first.  The cause is the xy ownership timing, not the channel.

## What the demos say about "clear" (training demos, 300 episodes with a held phase; channel coordinates, c = object height above the target reference = -dz)
- Elevated targets (target > 5 cm above the grasp, n = 150): c at the start of xy transport (3 cm of progress) median -7.5 cm, p75 -3.4 cm; c at the last held frame (the release) median -0.4 cm, p90 +1.3 cm, p5 -3.8 cm.  Only 1.3 % of these demos are at c >= 3 cm at their last held frame.
- Non-elevated targets (n = 150): c at xy onset median +10 cm; c at release median +2.4 cm, p90 +7.9 cm.
- So the demos' own xy transport toward an elevated target starts while the object is still below the target's reference height and ends at release height c about 0 (+1.3 cm at p90): the expert, not the module, knows how to start sideways motion
  early enough to arrive at the right height.  This is why "the module owns xy only once c >= CLEAR" has to leave the final approach to the expert as well (the gate is an instantaneous function of c; see the plan).
- Consequence for the principle: the expert's xy while c < CLEAR is exactly RAA-sL's; the module owns xy only in the part of the carry that is above the target's release height by CLEAR.

Limits: 20 states per cell, one trial; target contact is the single contact flag; no claim about which obstacle the K5 stall is against; the 20 states are within (not disjoint from) the 60 registered states, so CLEAR is NOT taken from them (it is fixed from the demos and the lead's 3 cm; see raasw3_plan.md).

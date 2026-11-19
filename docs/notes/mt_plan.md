# MT (SR 81): moved-target test (prereg tag `mt`)

Written 2026-11-18, before the prereg row and any main rollout.  Code: experiments/mt_tasks.py (states), experiments/mt_client.py (clients),
experiments/mt_read.py (read, selftest), experiments/prereg/mt.py (row), experiments/prereg/mt_pred.txt (predictions with probabilities, copied into the row);
launch scripts /coc/flash10/ksingara3/.claude_tmp/pp/mt_build.sh and mt_roll.sh.  No training; no existing file edited.

## Claim and test
Claim: "a VLA learns an object's trained position rather than the object; if the target is moved to new coordinates, the arm still goes to the trained coordinates."
Test: move the target (the plate) and, separately, the object (the bowl) to positions outside the demos' range, at graded distances, and score where the held
object is released.

## Scene and demo ranges
N2's scene 1 (LIBERO-Goal; results/flawed/n2_bddl/K1.bddl "put the bowl on the plate", K2.bddl "put the bowl on the stove"; one scene, instruction and goal differ).
The stove is a fixture and cannot move, so K2 tests the object only.  LIBERO's sampler puts each object in a 2 cm x 2 cm region.  Demo range (radius around the
trained centre = the mean of the stored sampler states, results/flawed/n2_tasks.json training): spread .0273 m (n2's 50 stored states).  The 50 official init states
of K1 (the demos' start states, read in the build) lie within .018 m of the plate's centre and .021 m of the bowl's (mt_tasks.json official_init).
Trained positions (world xy): plate (.0510, -.0212), bowl (-.0893, -.0007).

## Cells (bins by the moved entity's distance d from its trained centre; 4 smoke + 40 main states each)
| cell | task | moved | d (m) |
|---|---|---|---|
| K1_in | K1 tid 0 | nothing (sampler) | <= .027 (control) |
| T_near | K1 | plate | .04-.07 (1.5-2.6 x spread; just outside) |
| T_far | K1 | plate | .14-.30 (5-11 x spread; F6's ring) |
| O1_near, O1_far | K1 | bowl | .04-.07, .14-.30 |
| K2_in | K2 tid 2 | nothing | control |
| O2_near, O2_far | K2 | bowl | .04-.07, .14-.30 |

Seed rule (no seed of fx / f6o / f6o2 / rg_comp / steer / s2rg / cov_gen / pp_n4d is reused): candidate j of cell cid draws its start state with LIBERO's sampler,
np.random.seed = env.seed = 1050000 + 100 cid + j; a moved entity's xy from default_rng(1060000 + 100 cid + j) (uniform angle, uniform radius in the bin, first draw that
passes); state id k = 100 cid + a (a = accepted index; the first 4 are smoke), branch seed_base = 17000000 + 1000 k, trial = row of the state in results/flawed/mt/states/n2_goal_<tid>.npy.
Checks per candidate: inside n2's box; >= 0.75 (r + r') from every other movable, every fixture footprint and the stove-front region; the state settles (nothing else moves > 2 cm,
the moved entity stays within 1e-4 of its target); render check = agentview pixels of every movable >= .5 x its median in the control cell (the moved one >= .4 x).
Deviation from f6o2 / rg_comp: no cover decoys and no hidden second bowl (plain N2 scene), and the clearance rule is 0.75 of fx's (the trained plate-bowl gap, .141 m,
is inside fx's own 0.16 + gap rule).

## Arms (existing checkpoints, no training; server seed 13, client seed 7, one trial per state)
- I: frozen pi0.5 (pp_serve_purify.py --model pi05 --mode intact --comp= --seed 13, as SR 70), pp_atlas recording.
- R2: pi05_g8v_rview2_ae_tl6000 (sc_rv.py serve --arm rview2; exact masks, client --seg; SR 73's R2 row).
- RAA: pi05_g8v_raa_ae_tl6000 (raa.py serve/client --arm raa; learned masks + depth + calibration, the channel text; SR 73).
- rgp: ReGuide-perc (rg.py client --arm rgp, the reference that reads poses), on the I server.
- pi0 frozen: NOT run.  SR 70 / 65's pi0 fails the K gate (.017 on F6_a), so every contrast on it would be NOT TESTABLE.
All four arms record the oracle track (pp_atlas.Rec; the simulator is read for the read only, never fed to a policy).

## Measures
Success (the BDDL goal).  Endpoint attribution for the target-moved cells (T_near, T_far; K1_in for the control): the held object's xy at the last held step L
(end of the carry; >= 5 held steps) against the target's actual xy at L and against its trained xy.  A TRAINED RELEASE is d_act - d_tr > 3 cm.  share_all = trained
releases / all 40 states (no carry counts 0), share_carry = over carry episodes.  lam = 0 at the actual target, 1 at the trained one.  Reach attribution on the object-moved
cells: the eef xy at the first close command within 10 cm of the object's start height against the object's actual and trained start positions (> 3 cm = trained attempt).
Rule property, stated before the data: a displacement <= 3 cm cannot give a trained release, so the in-range share is 0 by construction and M1 means "the far-out share is
> 0" (about >= 4 of 40 episodes).

## Registered contrasts (paired by state within a bin; bootstrap B = 2000, default_rng(0), percentile 2.5 / 97.5; states resampled)
- M1: I's share_all on T_far minus on K1_in; SUPPORTED iff lower bound > 0 (the position-bias claim for frozen pi0.5).
- M2: RAA - I success on T_far; SUPPORTED iff lower bound > 0.   M3: RAA - I share_all on T_far; SUPPORTED iff upper bound < 0.
- MK: in-range success of every arm (K1_in, K2_in), FIT iff >= .70.
- Testability annotations (labels are kept): M1 needs I's K1_in success >= .70; M2 needs I's T_far success <= .85; M3 needs I's T_far share_all > .10.
- Descriptive: rgp and R2 against I per cell, every arm x cell, the object-moved cells, lam, the release error to the actual target.

## Predictions (experiments/prereg/mt_pred.txt has the full list)
SR 70 found I at .967 on a displaced, uncovered plate, so I expect M1 NOT SUPPORTED (85%; SUPPORTED 12%), M2 NOT SUPPORTED 60% / REVERSED 35%, M3 NOT SUPPORTED 92%;
I FIT 95%, RAA 80%, R2 60%, rgp 95%.  P(claim holds for frozen pi0.5) = 12%.

## Procedure
1. mt_tasks.py (job 4794579) builds the states and the render check (images results/flawed/mt/check_<cell>.png).
2. Smoke: 4 states per arm per cell, outputs outside results/ (OUTD under .claude_tmp); read for plumbing only (the run finishes, the track is written, prompts and states are the intended ones).
3. Selftest of mt_read.py, then the prereg row (prereg/mt.py), then main rollouts, then one registered read -> results/flawed/mt_read.json.

## Build and smoke (done before the row)
Build job 4794579 (exit 0): 44 states per cell accepted (rejected: O1_far 2, O2_far 3, all for low cream-cheese pixels = the moved bowl occluded it in agentview).  Moved-entity
distances: T_near .040-.069, T_far .141-.289, O*_near .040-.069, O*_far .142-.296 m; in-range cells .001-.021 m.  Official K1 init states: bowl <= .021, plate <= .018 m from the n2 centre.
Render check images: results/flawed/mt/check_<cell>.png (first 4 states; agentview above, wrist below, policy orientation).  Some displaced bowls sit against the stove's edge
(the footprint rule is a fraction of fx's); they settle without moving (checked), so the scenes are physically valid, but a bowl-by-the-stove layout is part of the far bins.
mt_read selftest (job 4794822) passed.  Smoke: 4 states per arm per cell (jobs 4795016 I, 4795017 r2, 4795018 raa, 4795019 rgp; read job 4795141; all 32 files per
arm-cell exist, exit 0, tracks written, identity check 0 bad, arm-to-arm start pairing 0 bad).  It was meant for plumbing; the read's table printed the n = 4 outcomes (see the SEEN
paragraph in mt_pred.txt: I and rgp succeed in every T cell; r2 released at the trained plate spot in 3 of 4 T_far episodes; raa 0 of 4).  No code, cell or prediction changed after it.
Wall time: about 10-15 s per episode, so a main cell of 40 episodes takes about 5-10 min per arm.

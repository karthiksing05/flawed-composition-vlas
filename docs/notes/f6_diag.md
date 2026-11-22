# F6 diagnosis: what is the F6_b failure? (2026-11-21)

Question from the coordinator: my 20 display rollouts of F6_b (pi0.5) failed 9 times, every failure releasing within 3 cm of the plate. Is the
cost of covering (DRAFT 4.5, d - b = .38, overlay .58 -> .84) a lost target, or physical interference by the cover?
Everything below is CPU analysis of tracks already on disk (registered dirs read only). Script: experiments/f6_diag.py (sbatch via fig_cpu.sh,
last run job 4829901), full output results/flawed/fig_frames/f6_diag.json (not committed; medians and IQRs of every measure per condition).

## Data used

| source | what | n |
|---|---|---|
| results/flawed/f6o2/roll/pi05-s{0-3}_{0..4}_track.npz, groot-s*_* | SR 45.6 registered rollouts, displaced layout, tid 0..4 = cond a, b, c, d, M (overlay) | 120 states per cond, per model |
| results/flawed/rg_comp_roll/pi05_{I_8,I_9,rc_8,mem_8,rgp_8}_track.npz | rg_comp (SR 70): F6_b, F6_a, RC, memory slot only, ReGuide-perc | 60 each |
| results/flawed/fig_frames/F6{,_more}/pi05_{I_8,I_9,rc_8}_track.npz | my 20 display rollouts (same harness as rg_comp) | 20 / 10 / 20 |

Per episode, from the track (pos of every object, ee, commanded gripper a6; rows = every env step): held = closed command and the bowl lifted
> 2 cm over its start and within 10 cm of the EE; release = last held row; plate position = its start position (the plate never moves).
The registered `_track.npz` files hold no bowl orientation, so "tipped" cannot be read directly; bowl height is used instead (below).

## 0. What the harness's success check is, and what the cover is

- The task goal is On(akita_black_bowl_1, plate_1). LIBERO's `check_ontop` (libero/envs/object_states/base_object_states.py:78) requires
  plate z <= bowl z, **bowl-plate contact, and the bowl's body centre within 0.03 m (xy) of the plate's body centre**. The episode ends at the
  first step where it holds; a failure is an episode where it never holds in 300 steps (T = 310 rows in the tracks).
  So the 3 cm in my selection signature is the predicate's own tolerance on the *bowl's resting position*, not a property of the release.
- The cover is physically inert: f6o_tasks.py docstring and install(): the covers are static world bodies with `contype = conaffinity = 0`,
  no degree of freedom, "so the simulator state, the physics, the success test ... are those of fx's F6". The bowl cannot land on, be knocked by,
  or collide with the cylinder. Physical interference is excluded by construction, and the tracks agree (section 1).
- Cover geometry (f6o2_tasks.json): radius .1135 m, half height .0138 m, centre z .979 (top about 8 cm above the plate plane), centred on the
  plate in b and c, parked far away in a, moved beside the plate in d.

## 1. pi0.5, F6_b (f6o2, 120 states): all 50 failures, classified

Success .583 (70/120). Failures 50. `rel` = bowl xy at the last held row against the plate centre.

| measure (medians) | b success (70) | b failure (50) | a success (113) | d success (116) |
|---|---|---|---|---|
| release xy distance to plate centre (= to the cylinder centre in b) | 1.16 cm | 1.78 cm | 2.47 cm | 3.37 cm |
| in cylinder radii (.1135 m) | .10 | .16 | | |
| EE z above plate plane at release | 6.18 cm | 6.22 cm | 5.72 cm | 5.90 cm |
| bowl z above plate plane at release | 2.68 cm | 2.57 cm | 2.57 cm | 2.48 cm |
| release step | 101 | 100 | 92 | 96 |
| bowl xy, plate centre, 5 / 10 / 40 steps after release | 1.68 / 2.03 / 2.07 cm | 3.20 / 3.88 / 4.05 cm | 1.97 / 2.47 / 2.35 cm | |
| bowl xy shift in the 10 steps after release | 1.80 cm | 2.63 cm | 1.96 cm | 2.46 cm |
| bowl end height vs success end height (end_dz) | -5.90 cm | -5.88 cm | -5.80 cm | -5.76 cm |
| held at the last row / hold segments | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 |

Classes of the 50 failures:
- **44 released within 3 cm of the plate centre**, 6 released 3-10 cm away, 0 farther, 0 never grasped, 0 held to timeout, 0 re-grasps.
  So **12% of failures (6/50) are a mislocated release; 88% (44/50) are a release inside the tolerance whose bowl then ended outside it.**
- **When it fails:** at release the arm is where it is for successes (step 100 vs 101, EE z 6.22 vs 6.18 cm, bowl height 2.57 vs 2.68 cm).
  The failure is fixed in the 5-10 steps after release: the bowl slides 2.6 cm (success 1.8 cm), in the -x direction, and comes to rest 4.05 cm
  from the plate centre (IQR 3.6 - 6.1 cm; all 50 are > 3 cm, success 2.07 cm). The bowl's height after the settle equals that of successes
  (end_dz -5.88 vs -5.90 cm), so it sits on the plate (plate radius is about 10 cm), just 1-3 cm beyond the predicate's 3 cm disc.
  In 44 of 50 failures the bowl passed inside 3 cm at some step after release (median closest approach 1.78 cm) without the predicate becoming
  true: it fell through the disc before it was in contact and settled outside it.
- **Why the slide differs: arrival motion at release (3-step mean EE velocity).** b: speed 2.2 mm/step success, 2.8 failure; x velocity
  -1.3 / -1.9 mm/step; descent -12.5 / -10.9 mm/step. a: speed 4.3, x velocity +3.0, descent -9.5. d: speed 4.7, x velocity +1.2. Bowl x velocity in the 5 steps after
  release: b -2.1 (success) / -2.8 (failure) mm/step, a -0.9, d -1.0. With the plate covered the policy comes in slower, steeper, and moving -x;
  the bowl follows -x after release, and the 1.4 cm longer slide in failures decides the 3 cm test. The release point in failures is also
  1.4 cm further -x than in successes (rel_dx -1.54 vs -0.15 cm), so release offset and slide both push the same way.
- **Plate visibility.** Plate pixels at later policy queries (agentview, median): a 1603, b 1.3 (success) / 0.3 (failure), c 1.2 / 0.3, d 1604.
  The plate is invisible in every b and c episode; the policy never sees it, so it is not that failures saw the plate less.

## 2. Same measures in the other conditions (pi0.5, f6o2, 120 states each)

| cond | success | failures | within 3 cm release | 3-10 cm | > 10 cm | release xy, success / failure (cm) | end xy, success / failure (cm) |
|---|---|---|---|---|---|---|---|
| a visible | .942 | 7 | 0 | 6 | 1 | 2.47 / 4.17 | 2.34 / 3.10 |
| b covered after the first query | .583 | 50 | 44 | 6 | 0 | 1.16 / 1.78 | 2.07 / 4.05 |
| c covered from the start | .533 | 56 | 48 | 7 | 1 | 1.04 / 1.88 | 2.15 / 4.42 |
| d cover beside the plate | .967 | 4 | 1 | 2 | 1 | 3.37 / 5.66 | 2.08 / 3.28 |
| M b + memory overlay | .842 | 19 | 12 | 7 | 0 | 2.14 / 2.43 | 2.11 / 5.42 |

(Failure counts reproduce the registered `failed_bowl_ends` of f6o2_read_main.json: near_plate 7, 50, 54 (+2 elsewhere), 4, 19.)

- b, c and M failures have the same signature as b: release inside 3 cm, bowl slides out. a's and d's few failures release farther (3-10 cm).
- rg_comp (60 states): I on F6_b .517, 27/29 failures release within 3 cm; I on F6_a (plate visible) .967, 2 failures; RC (resolver + memory slot
  + ReGuide transport) .800, 12 failures: 4 within 3 cm, 3 at 3-10 cm, 2 far, 3 never grasped; slot alone `mem` .817, 11 failures: 9 within 3 cm.
  RC succeeds by releasing 3.1 cm from the centre but with lower EE (4.8 cm vs 5.75 cm above plane) and smaller slide.
- My 20 display trials (fig_frames): I_8 .55, 9 failures all released within 3 cm; I_9 (a) 10/10; rc_8 .85, 3 failures (1 never grasped).
  They agree with the registered 60 and 120.

GR00T for contrast (f6o2): b .142, 103 failures: 19 within 3 cm, 66 at 3-10 cm, 12 beyond 10 cm, 6 held to the timeout; failed bowls end 21 cm
from the plate (median). GR00T mislocates and loses the bowl; pi0.5 does not.

## 3. Plate position against the trained spot

The disp layout displaces the plate .14-.30 m from the trained spot (plate_training_xy = [.0510, -.0212], f6o_tasks.json). Measured in the tracks:
distance plate - trained spot, quartiles .16 / .20 / .25 m (b successes), .18 / .22 / .24 m (b failures); the 20 display trials .15 / .20 / .24 m.
So a release within 1-2 cm of the plate is 14-30 cm from where the trained policy would put it: the arm is following the actual plate position,
not its trained habit. In this sense the 3 cm number is meaningful against the trained spot, but it is not a failure marker (section 4).

## 4. Verdict

**Registered numbers (SR 45.6, pi0.5, 120 states): a .942, b .583, c .533, d .967, M .842; d - b +.383 [.283, .475], b - c +.05 [-.05, +.158],
a - b +.358, M - b +.258, M - d -.125.**

- **Physical interference: ruled out.** The cover is non-colliding by construction, bowl height, release height, hold segments (1 each) and end
  height are the same in failures as in successes, no failure is held at the timeout, none is re-grasped. The d condition puts the same
  cover in the scene and success is .967: a cover's presence, appearance or motion costs nothing (a - d = -.025 [-.075, +.025]).
- **Lost target (the arm does not know where the plate is): not supported for pi0.5.** Release xy error under cover (1.1-1.8 cm) is *smaller*
  than with the plate visible (2.5 cm); 88% of b failures release inside the 3 cm tolerance and the other 12% within 10 cm. The cylinder is centred
  on the plate (radius .1135 m), so it is itself a visible marker of the plate's position; c (the plate never seen, the cover on it from step 0)
  scores .533, equal to b (.583; b - c = +.05 with a CI that contains 0). The first look at the plate contributes nothing, i.e. there is no memory in use,
  which is DRAFT 4.5's "MEMORY ABSENT/PARTIAL" reading; but the effect of hiding is not a lost xy target either, because the arm still reaches
  the actual plate position from the cover alone.
  GR00T is the lost-target case: 76% of its b failures release 3-10 cm or farther, 6 never release, and the failed bowls end 21 cm away.
- **What the cost is (pi0.5): a placement-precision loss at the success tolerance.** With the plate hidden the policy releases at the right place but
  arrives slower, steeper and moving -x; after release the bowl slides 2.6 cm instead of 1.8 cm and rests 4.05 cm from the centre against the
  predicate's 3 cm disc. The baseline is tight even when the plate is visible: successes in a end 2.34 cm from the centre, 78% of the tolerance. A
  1-2 cm shift moves the whole distribution across the line, hence a .38 cost from a small geometric change. Quantified: of the b failure rate
  .417, .367 (44/120) is a within-tolerance release that slid out and .050 (6/120) is a mislocated release (3-10 cm); a/d failure rates are .058 and .033.
- **Overlay (M) .58 -> .84:** repainting the plate back onto the cover restores the policy's normal approach (release dx and slide closer to a's), and
  the 19 remaining failures are the same slide-out type (12 within 3 cm, 7 at 3-10 cm). It repairs the appearance the policy uses to time and aim its
  release; it does not show that a lost position was restored. DRAFT's phrase "a target seen once and then hidden leaves little trace" is
  not what the f6o2 pi0.5 data show: the trace is not missing, the cylinder supplies it; what changes is how precisely the bowl is put down.
- **Does the 3 cm "released away" signature fit?** No. (i) 3 cm is the success predicate's tolerance on the final bowl position; a release
  more than 3 cm from the plate is ordinary in successes (a: upper quartile of successful releases 3.2 cm; d: median 3.4 cm). (ii) It flags only 6 of 50 b failures; 44 pass it.
  (iii) It was written for a mislocated release, which is the GR00T pattern and the a/d minority, not pi0.5's b failure. A signature that fits the
  harness is the predicate itself: the bowl's final xy more than 3 cm from the plate centre, which all 50 b failures meet (50/50 end > 3 cm) and 1/70
  b successes do. With it the failing pi0.5 trials qualify as failures by definition (the 9 failures among my 20 have final bowl xy 3.5-6.6 cm), so the F6_b gallery row could have a failing arm (the first 20 trials
  contain 9 failures; whether RC succeeds on the same trial was not re-checked). I did not change the selection rule or the gallery.

## Caveats

- The tracks carry no bowl orientation; "slide" is inferred from bowl xy and z. A direct check needs a quaternion log (a new rollout).
- "Within-tolerance release that slid out" rests on the last held row being the release; the held test uses the closed command and 10 cm proximity.
- x-direction statements are in the world frame; the plate-to-trained-spot axis is almost constant across states (dx -.09, dy -.19 median), so
  "toward the trained spot" and "-x" are not separable here; the y component of the release does not move toward the trained spot (-0.6 vs -0.7 cm).
- The mapping of d/a failures to classes uses the same rule, n is small (4 and 7).

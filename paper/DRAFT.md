# Routed by Cues: Why Vision-Language-Action Models Fail New Pairings of Known Parts, and a Role-Addressed Repair

<!-- Title alternatives:
1. The Instruction's Roles Do Not Choose the Sub-Goal: Architecture-Rooted Failures of Vision-Language-Action Models
2. Cues Choose the Sub-Goal: A Failure Atlas, a Failure Predictor and a Role Channel for Vision-Language-Action Models
-->

Draft of 2026-11-19. Plan: docs/PLAN_PAPER.md. Numbers come from docs/SCAFFOLD_RESULTS.md (SR), docs/ARCH_FAILURES.md (AF) and docs/ARCHITECTURE_PROPOSAL.md; each claim names its sections. The previous draft (a different project) is at docs/history/DRAFT_2026-10-21_units.md. Prose follows docs/PAPER_STYLE.md of the diffusion-composition paper: active voice, one idea per paragraph, no hedged contrasts.

## Abstract

Vision-language-action models (VLAs) succeed on trained tasks and fail when a new instruction pairs a known object with a known target. We trace the failure to one routing defect. The action expert picks its sub-goal from visual cues that covaried with the role in training, and the instruction's roles do not pick it. We make three contributions. First, we build a failure atlas of architecture-rooted failures across five models: pi0.5, GR00T, pi0, X-VLA and SmolVLA. A failure is architecture-rooted when the input holds the needed information and a reference planner succeeds, when the failure persists across models, and when a structural change that adds no information removes it. A moved-plate test shows that frozen pi0.5 follows a moved plate and that a narrow retrain acquires the position bias. Second, we build a steerability predictor of new-pairing failure from trained tasks alone. Its first form failed its pre-registered test. A corrected form, frozen in advance, ranked 12 policy-pairing cells at AUC .944 on two policies trained afterwards. Third, we locate the failures in identifiable parts of the architecture. Removing the referent resolver from a role-addressed policy restores the relational-language failure, with success on relational instructions falling by .263 [.213, .313]. The matching test for a pose memory and the hidden-target failure stays open. Repairs built on this account reach .656 success on held-out pairings with role views plus a role channel (RAA) and .717 with an added servo term (RAA-s), against .389 for the same retrain with role views and no channel (.042 for a plain retrain). A test-time wrapper, ReGuide, reaches .897. All results come from LIBERO, two scenes and pick-place tasks.

## 1 Introduction

A household robot that stacks a cup on a plate and a bowl on a shelf should also put the cup on the shelf. Instructions combine a small set of objects and targets in many ways, and a trained policy sees few of those combinations. Vision-language-action models (VLAs) inherit this problem. They reach .98 to 1.00 success on the tasks they trained on and fail on new pairings of the same objects and targets (CRAFT, LIBERO-PRO, LIBERO-CF).

Prior work names this failure or patches it. Benchmarks show that VLAs follow habit and ignore language. Fixes add counterfactual data, wrap the frozen policy in a grounding module, or mask the camera view. None of them says which part of the architecture produces the failure, so none predicts where a given fix works or where it breaks.

We give that account. A VLA picks its sub-goal from cues that covaried with the sub-goal in training. After a correct grasp, the held object's outline or the nearer plate steers the carry and the release, and the instruction's target role has little influence. In one scene, a pre-registered test on new layouts confirms a majority rule on pi0.5. The place goes to the plate backed by two of three cues: nearer, more salient, trained (.912 of 445 placements).

The account needs a test that separates architecture from task difficulty. We call a failure architecture-rooted when three criteria hold.
- C1. The input holds the information the task needs, and a reference planner succeeds on the same states.
- C2. The failure persists across models.
- C3. A structural change that adds no information removes the failure.

A failure that misses C1 belongs to the task, and we report it as a limit. New layouts fall in that group, because every system fails there, including the reference planners.

The criteria also predict where a fix works. An edit to the images or the prompt can remove a cue that the task does not need. It cannot remove a cue that the grasp or the carry needs. Our experiments confirm this: role views lift pi0.5 from .450 to .817 on new pairings in scene 1, and they give .000 in scene 2, where the cue is position.

If cues route the sub-goal, a policy that addresses its actions by role should remove the failures, and trained tasks should reveal whether a policy follows roles. Both ideas lead to tests. A policy that follows roles should switch its sub-goal when a trained task's object or target word changes. A policy that addresses actions by role needs geometry addressed by role.

**Contributions.**
1. A failure atlas of five failures on five frozen VLAs (pi0.5, GR00T, pi0, X-VLA, SmolVLA) in two LIBERO scenes: the place follows a cue, the held outline picks the carry, the habitual object is grasped, relational language is ignored, and a hidden target leaves little trace. Each failure carries its C1, C2 and C3 evidence. We also report the fixes that work, the cases where they fail, and a moved-plate test in which frozen pi0.5 tracks the plate and a narrow retrain releases at the trained spot.
2. A steerability predictor of new-pairing failure that uses trained tasks only. Its first form failed its pre-registered test. A corrected form, frozen in code before we probed two new policies, ranks their 12 cells at AUC .944.
3. A localisation of the failures in the architecture. Module removal on a role-addressed policy restores the relational-language failure (F7) when we remove the referent resolver, and the test for the pose memory and the hidden-target failure (F6) stays open. The repairs are secondary: role views plus a role channel (RAA) reach .656 on held-out pairings against .389 for role views alone and .042 for a plain retrain, and an added servo term (RAA-s) reaches .717. A test-time wrapper, ReGuide, reaches .897, so the repairs narrow the gap and leave it open.

**Scope.** All results come from LIBERO, two scenes (one kitchen scene with 11 tasks, one living-room scene with one trained task) and pick-place tasks. Each trained policy has one seed. The role channel is tested on one backbone (pi0.5) in one scene. The predictor is confirmed on two policies that descend from one base. GR00T misses the testability floor in scene 2, and X-VLA has no transport-level fix. The reference planners use simulator state or privileged depth and calibration, as do the perceived-pose runs of ReGuide. We make no claim about real robots or about tasks beyond pick-place.

## 2 Related work

**Compositional generalisation in VLAs.** Benchmarks show that VLAs follow habit and ignore the instruction. LIBERO-PRO (2510.03827) reports collapse under perturbed objects and instructions, and LIBERO-CF (2602.17659) reports that pi0.5 completes 13.2% of faithful counterfactual tasks. LIBERO-Plus (2510.13626) also perturbs instructions and scenes. Robust Skills, Brittle Grounding (2602.24143) separates reliable skills from brittle grounding on held-out object-location pairings in SmolVLA and pi0.5. It stops at the grasp, and we extend the stage-wise view to the carry and the release across five models. CRAFT (2610.00524) reports 8.4%, 2.8% and 1.1% on held-out pairings for pi0.5, pi0 and GR00T after plain fine-tuning. Its authors fix the failure with counterfactual training pairs.

**Fixes for held-out pairings.** Data fixes retrain the model. CRAFT adds counterfactual pairs, ACT-VLA (2607.00351) augments demonstrations from the model's latent task representations, and AC-VLA (2607.15714) adds an LLM task decomposer and masks the wrist view while the gripper is closed. Test-time wrappers edit a frozen policy. ReGuide (2609.38616) grounds poses and steers the arm to a demonstration-supported pre-contact configuration before the policy resumes. LiLo-VLA (2602.21531) reaches with a planner and lets an object-centric wrist-only policy interact. Harness VLA (2607.08448) and VP-VLA (2603.22003) bind referents or switch the prompt at gripper events. w2VLA (2606.21496) splits the instruction into a skill string and an object string and reports 91.7 against 38.2 on unseen skill-object pairings for pi0.5. We reimplement ReGuide as our wrapper reference and claim no priority for hand-over, masking or phase switching. Our contribution is the criteria that tell which cue each fix can remove.

**Object-centric and role-addressed architectures.** TAX-Pose (2211.09325) and SPOT (2411.00965) place actions in an object-relative frame, and FOCI (2609.08743) extends the idea to relative motion between task-relevant objects. OA-WAM (2605.06481) gives an action expert object slots with address and content vectors, and POT-VLA (2607.18016) feeds role-indexed 3D object records to an action expert. RAA combines an input restriction with one role-addressed geometry channel, and each part has a precedent here. We claim the decomposition that the failure account derives, and its test against a plain retrain at matched data. We read FOCI, OA-WAM and POT-VLA in full text and w2VLA by abstract. OA-WAM already biases attention by the relative pose between each object slot and the robot slot, the closest precedent for our channel.

**Language sensitivity and mechanisms.** STAGE (2609.13458) and LIBERO-PRO measure how much actions change when the instruction changes. Encoded but Not in Control (2610.06235) reports that the instruction's target is linearly decodable and goes unused. Grant et al. (2603.19233) inject activations at action-expert layers and redirect pi0.5 in 99.6% of 1,968 episodes, which gives the strongest mechanistic evidence for goal selection in the action expert. Our evidence for the routing rule comes from behavioural tests, and Grant et al. supply the probe of the attention path. Wang et al. (2607.29687) state a coverage law for partner count in a small custom policy, and we did not reproduce it on frozen pi0.5 and GR00T.

**Performance prediction.** Okawa et al. (2310.09336) show that composed-task performance rises multiplicatively with the performance on its constituents in diffusion models, and our predictor takes that product form. Agreement-on-the-Line (2404.01542) predicts out-of-distribution accuracy of classifiers from model agreement. World-model evaluators, such as the Veo simulator for Gemini Robotics policies (2512.10675), rank policies with closed-loop rollouts inside a learned simulator. Our probe needs only trained-task rollouts, and we compare cost and make no accuracy comparison.

## 3 Setting and criteria

SR denotes docs/SCAFFOLD_RESULTS.md. AF denotes docs/ARCH_FAILURES.md. Section numbers after a number name the record that holds it. Intervals are 95% intervals.

**Models.** We test five frozen VLAs trained on LIBERO's 40 tasks. They are pi0.5 (openpi pi05_libero), pi0 (pi0_libero), GR00T N1 (ar0s/groot_libero), X-VLA (xvla-libero) and SmolVLA (smolvla_libero) (SR §50, §65). We never retrain them in the atlas. Each model enters a read only where the atlas says so. X-VLA enters the atlas and the F7 reads. It misses the transport reads because its server sends absolute end-effector targets and our ReGuide transports issue delta commands (SR §65).

**Scenes.** Scene 1 descends from LIBERO-Goal's kitchen and has 11 tasks (SR §55). Five are trained tasks that we call K1-K4 and K6. Six are new pairings, N1-N6, of the cheese or the wine bottle with the plate, stove, cabinet or bowl. Scene 1 has two layouts. The familiar layout keeps the trained start regions. The new layout shifts them. Scene 2 is LIBERO-10's LIVING_ROOM_SCENE5, which holds a white mug, a yellow-and-white mug, a red mug, a left plate and a right plate. Its one trained task puts the white mug on the left plate and the yellow-and-white mug on the right plate (SR §50). It is the only LIBERO-10 scene with two movable objects and two targets (SR §50).

**Terms.** A trained task is an instruction the model saw in training. A role is the object role or the target role of an instruction. A new pairing combines an object role from one trained task with a target role from another. Its known partners are the trained tasks that share its object and its target. In scene 2 the partners K1 and K2 are the two halves of the trained compound, and the new pairings N1 and N2 re-pair the mugs and plates (SR §50). A cue is any visual feature that covaried with a sub-goal in training. The testability floor says that a model is testable on a scene only if it scores at least .70 on the trained tasks without any edit (SR §50, §65). Scene-2 GR00T misses the floor at .508 [.333, .667] (SR §50).

**States.** Every read draws start states from LIBERO's sampler. Before it draws a new set of states, the rule reproduces the stored states exactly (SR §65). Scene 1 uses 20 development states, 20 fresh states and later blocks of 40 to 60 new states (SR §47, §65, §71). Scene 2 uses 60 states (SR §50).

**Reference planner.** A scripted planner reads simulator state. In scene 1 the parts policy U scores .894 on new pairings from simulator poses, .883 from camera poses and .881 from a learned segmenter (SR §38.1, §40.1). In scene 2 the planner fits its keypoints to the compound's raw demos and executes none. It scores .967 on N2 and .333 on N1, which pools to .65 [.25, 1.00] (SR §50).

**Criteria.** A failure is architecture-rooted when three criteria hold.
- C1. The input holds the information the task needs, and the reference planner succeeds on the same states. We call C1 met when the planner reaches .80 (SR §50).
- C2. The failure persists across models. We test models, chunk length and prompting. We test no data ladder, so C2 on more data stays open (AF §3, F1).
- C3. A structural change that adds no information removes the failure.

A failure that misses C1 belongs to the task. We report it as a limit.

**Registration and statistics.** Each confirmatory read has a ledger row (m3_prereg) written before any main rollout. The row fixes the arms, states, predictions and verdict rules. Verdicts appear in capitals, such as FIX WORKS. We compute every contrast by paired bootstrap over (task, state) pairs, with 2,000 draws and tasks and states drawn jointly (SR §50, §65). The atlas on development states and the cue reads that followed it were development reads. We mark them as such. Every trained rung has one seed.

## 4 The failure atlas

The routing defect has one form. A VLA picks a sub-goal by a cue that covaried with that sub-goal in training, and the instruction's roles do not pick it. The cue differs by failure and by scene. Table 1 lists the measured break for each failure and model. The subsections give the claim, the C1, C2 and C3 evidence, and the scope.

**Table 1.** Measured break by failure and model. "E" is the excess failure over known partners. Brackets give 95% intervals. Blank cells mean no read.

| failure | pi0.5 | pi0 | GR00T N1 | X-VLA | SmolVLA |
|---|---|---|---|---|---|
| F1, scene 1, cross-chunk E, fresh states (AF five-model atlas) | +.450 [.28, .61] | +.225 [.05, .43] | +.500 [.26, .74] | +.379 [.04, .73] | +.054 [-.00, .14], no break |
| F1, scene 2, cross-chunk E (SR §50) | +1.000 [1.00, 1.00] | +.908 [.792, 1.00] | +.533 [.167, .858], below floor | | |
| F1m, N4 and N6 (SR §51) | succeeds .04 after a high carry, .66 after a low one | high carry on .67-1.00 of grasps | high carry on at least .78 of grasps | high carry on 1.00 | |
| F2, grasp E, fresh states (AF five-model atlas) | +.000 [.00, .00] | +.650 [.39, .88] | +.017 [.00, .07] | +.192 [.00, .53] | +.833 [.67, .96] |
| F7, relational choice equals ambiguous choice (SR §45.1, §45.3) | .925 (SP), 1.00 (NG) | ignored on 4 of 6 cells across pi0, X-VLA and SmolVLA, with agreement 75-100% | 1.00 (SP), 1.00 (NG, PARTIAL) | see pi0 | see pi0 |
| F6, cost of covering the plate after one view, d - b (SR §45.6) | .38 [.28, .48] | not testable, gate .017 (SR §70) | .60 [.51, .68] | | |
| New layout, atlas verdict (AF five-model atlas) | NEITHER | NEITHER | NEITHER | NEITHER | NEITHER |

### 4.1 F1. The place follows a cue

**Claim.** After a correct grasp, the carry and release follow a cue and ignore the instruction's target role.

**C1.** Trained tasks run at .98-1.00 on pi0.5 and GR00T in scene 1 (AF §3, F1). The reference planner reaches the new pairings in scene 1 (SR §38.1, §40.1). In scene 2 it reaches N2 at .967 and N1 at .333 (SR §50). C1 therefore holds in scene 1 and on N2 in scene 2.

**C2.** In scene 1 the cross-chunk break clears the bar on pi0.5, pi0, GR00T and X-VLA, as Table 1 shows (AF five-model atlas). The four verdicts replicate from the development states to the fresh states (AF fresh-state replication). SmolVLA fails inside the chunk, with a within-chunk E of +.862 [.73, .96] (AF fresh-state replication). In scene 2 every model scores 0 of 720 episodes on the two new pairings, while pi0.5 scores .99 and pi0 .81 on the trained halves (SR §50). The swapped two-step task scores 0 of 180 and ends in the trained arrangement on .83-.98 of episodes (SR §50). Chunk length of 2 and 10 actions per query changes pi0.5 by -.050 [-.250, +.150] and -.042 [-.200, +.100], and GR00T by -.008 [-.117, +.100] and +.083 [-.025, +.233] (SR §32.1). We read these as no large effect, because the intervals allow up to +.15 on pi0.5 and +.23 on GR00T (SR §32.1).

**The scene-2 cue.** A series of reads on pi0.5 identifies the cue.
- With the two mugs' start positions swapped, 287 of 287 placements go to the plate on the mug's start side and none go to its trained plate (SR §57).
- pi0 replicates the routing, at .993 of 145 placements (SR §58).
- With the plates moved, 0 of 116 and 0 of 125 disagreeing placements go to the start-side plate. The nearer plate routes the place (SR §59).
- In a third layout the nearer plate is the less salient one. Then 158 of 158 placements go to the small nearer plate when it is the mug's trained plate. Then .859 of 149 go to the large far plate when that plate is the trained one (SR §60).
- A pre-registered test on new layouts confirms a majority rule. The place goes to the plate backed by two of {nearer, salient, trained}. Pooled, .912 [.883, .942] of 445 two-against-one placements go there (SR §61).
- By configuration, the trained plate is outvoted 153 of 153 times, the salient plate .974 of 151 times and the nearer plate .752 [.660, .843] of 141 times (SR §61). The earlier layouts replicate at .970 of 427 (SR §61).

The instruction is unresolved in every run (SR §61). The reference planner meets C1 on three layouts (.825-.838) and misses it on three others (.725-.788) (SR §61).

**C3.** In scene 1, the front end tvpb lifts frozen pi0.5 from .454 to .938 with no trained-task cost (SR §55). The hand-over to a parts policy lifts GR00T from .17 to .84 and pi0.5 from .50 to .85 [+.017, +.675], which reads NOT SHOWN for pi0.5 (SR §44.1). In scene 2 a reimplemented ReGuide lifts pi0.5 to N1+N2 .708 [.650, .775] and pi0 to .700, with trained tasks held (SR §68). The fixes appear below.

**Scope.** The majority rule holds on pi0.5 only. pi0's labels are unresolved on two of the layouts, and GR00T matches the nearer-plate routing below its floor (SR §59, §60). The record tests consequences of the cue and has no probe of the attention path. Grant et al. hold the mechanistic evidence (docs/NOVELTY.md).

### 4.2 F1m. The held object's outline picks the carry

**Claim.** The held bottle's outline starts K4's high carry toward the cabinet, whatever the instruction's target.

**C1.** P_min scores .80 on N4 and .80 on N6 (SR §51). From a pre-grasped pose 6 cm above the target, the servo scores 25 of 26 and a bare release 21 of 26 (SR §51). Given the bottle already held above the target, pi0.5 places 26 of 26 (SR §51).

**C2.** The bottle starts the high carry in at least .78 of GR00T's grasped N4 and N6 episodes under every front end we tried. It does so in 1.00 of X-VLA's and .67-1.00 of pi0's (SR §51). pi0.5 succeeds .04 after a high carry and .66 after a low one (SR §51).

**C3.** On pi0.5, painting the held bottle as its box in the target phase lowers the high-carry share from .225 to .062. It raises N4+N6 from .625 to .812, +.188 [+.075, +.325], with no known-task cost (SR §53). The state input is ruled out on pi0.5, because the served model reads no state (SR §53). On GR00T the same box adds +.025 [-.025, +.100], which reads UNRESOLVED, and costs -.070 on trained tasks (SR §56). The cheese-finger swap on GR00T also reads UNRESOLVED (SR §56). ReGuide's transport takes GR00T's N4 high-carry share from 1.00 to 0 of 38. N6 still carries high on .821 and reaches .200 (SR §71).

**Scope.** The cue on pi0.5 is the outline, and it does not transfer to GR00T. GR00T's N6 places 0 of 14 even from above the bowl (SR §51). We treat that part as a model-specific motor gap.

### 4.3 F2. The habitual object is grasped

**Claim.** The scene picks the object to grasp, and the instruction's object role loses.

**C1.** Frozen pi0.5 and GR00T grasp the named object at .950-.992 in every chunk setting (SR §32.1). U reaches .894 on new pairings (SR §38.1).

**C2.** The failure is the larger one on the weaker models. On pi0 and SmolVLA, .679 and .829 of new-pairing episodes end at the grasp (SR §65). The grasp excess on fresh states is +.650 [.39, .88] on pi0 and +.833 [.67, .96] on SmolVLA (AF five-model atlas). It reads +.192 [.00, .53] on X-VLA, which misses the bar. In scene 2, pi0.5 puts back a held mug the instruction does not name, and it grasps the same mug again 240 of 240 times (SR §64). After a clear-away carry, it takes the named mug on .167 of N2 episodes. After its own trained white-mug carry, it takes it on .967 (SR §67).

**C3.** ReGuide's approach transport lifts pi0 from .021 to .608 and SmolVLA from .021 to .604 on new pairings. Trained tasks change by +.020 [.000, +.060] and +.030 [-.020, +.120] (SR §65). In scene 2 it moves pi0.5's first hold to the named mug on 60 of 60 N2 episodes, against 0 for the hand-over alone (SR §68). Role views alone lift pi0 by +.183 [+.021, +.412] at no cost and do nothing on SmolVLA (SR §65).

**Scope.** F2 is architecture-rooted on pi0 and SmolVLA. Frozen pi0.5 and GR00T grasp correctly in scene 1. In scene 2, pi0.5 grasps the wrong mug first on 1.00 of N2 episodes (SR §50).

### 4.4 F7. Relational and negated language is ignored

**Claim.** Relations such as "left" and "not on the plate" leave the choice of referent unchanged.

**C1.** Two identical bowls share the scene, so only the relation picks the referent. With the other bowl removed, the models score .825-1.00, and the scripted references score .93-1.00 (SR §45.1).

**C2.** On pi0.5 and GR00T, the relational choice equals the ambiguous choice on 92.5-100% of states (SR §45.1). The referent share sits at .475-.50 (SR §45.1). On pi0, X-VLA and SmolVLA the relation is ignored on 4 of 6 cells, and the choice agrees with the ambiguous one on 75-100% of states (SR §45.3). X-VLA and SmolVLA on the SP family have no power (SR §45.3). F7 therefore holds on all five VLAs on at least one family (SR §45.3). A negated LIBERO-PRO instruction scores 0 of 20 on pi0.5 and GR00T against .95 for the trained one. We report that as descriptive (SR §45.1).

**C3.** A resolver turns the relation into one entity, and a role view paints the other bowl. It fixes 3 of 4 cells with the resolver exact on 160 of 160 (SR §45.2). On pi0.5 relational success goes from .40 to 1.00 (SP) and from .20 to .775 (NG). On GR00T SP it goes from .30 to .85 (SR §45.2).

**Scope.** The resolver covers fixed templates on two relation families and reads a parsed scene. It is a routing fix and makes no language-understanding claim. On pi0, X-VLA and SmolVLA it fixes 0 of 6 cells, and the resolver is exact on 240 of 240 (SR §45.3). Those models fail even the named plate bowl on 0 of 20 episodes (SR §45.3). GR00T's NG cell is capped at .50 by its own unique-name control (SR §45.1, §45.3).

### 4.5 F6. A target seen once and then hidden leaves little trace

**Claim.** After one view of the plate, a cover over it costs success, and the action path keeps almost no trace of the view.

**Design.** The plate appears at a displaced position and a cylinder then covers it. Condition a never covers it. Condition b covers it after the first query. Condition c covers it from the start. Condition d moves the cover beside the plate, which keeps the plate visible and gives b's visual event (SR §45.5).

**C1.** The reference reaches .839 on 120 new states (SR §45.6). Condition d keeps pi0.5 at .967 and GR00T at .742 (SR §45.6).

**C2.** Covering costs pi0.5 .38 [.28, .48] and GR00T .60 [.51, .68] (d - b) (SR §45.6). Success under b stays within .05 of the never-seen control c, at +.05 [-.05, +.158] and +.042 [-.033, +.117] (SR §45.6). The registered verdict is PARTIAL on both models, because the equivalence bound misses .10 by .058 and .017 (SR §45.6). The pattern is consistent with a missing memory.

**C3.** A memory overlay repaints the plate seen at the first query over the cover in both cameras. It raises pi0.5 from .58 to .84, +.26 [+.15, +.35], and GR00T from .14 to .28, +.14 [+.04, +.24] (SR §45.6). It recovers 67% and 24% of the cost of covering (SR §45.6).

**Scope.** The test uses one occluder task. The plate leaks back into view on 32-49% of later queries, so late views may carry part of the gain (AF §1, F6). GR00T sits at the rule's boundary, and the overlay still trails condition d by -.458 [-.558, -.358] there (SR §45.6). pi0 is not testable, with a gate of .017 (SR §70).

### 4.6 The new-layout limit

**Claim.** New layouts fail every system, including the reference planners.

**Evidence.** The atlas finds no pairing effect in the new layout on any of the five models (AF five-model atlas). Success on N1-N6 is .30 on pi0.5 and .08 on GR00T (AF §3, F3). The scripted servo scores .31, P_min .37 and U .378 (SR §38.1; AF §3, F3). All four systems fail on the same states. The demo's fixed top-down grasp holds on 0 of 10 cheese states and 0 of 10 bottle states (AF §3, F3).

**Scope.** C1 fails, so we report a limit of the task and make no claim about VLAs. Actions per query of 2 and 10 leave new-layout success unchanged on both models (SR §32.1).

### 4.7 A moved target: position bias comes from narrow fine-tuning

**Claim.** Frozen pi0.5 follows a moved plate, and a narrow retrain releases at the trained plate position.

**Design.** F1 invites a simpler reading, in which a VLA learns an object's trained coordinates and the arm goes there when the target moves. We test it in scene 1 on K1 ("put the bowl on the plate", plate movable) and K2 ("put the bowl on the stove", stove fixed). The demonstrations place the plate within .027 m of one centre. We move the plate to .040-.069 m (near) or .141-.296 m (far) from that centre, which is 5 to 11 times the demonstration range, with 40 states per cell. We also move the bowl by the same two distances. Four systems run on the same states: frozen pi0.5, a narrow retrain on the eight known tasks from role-painted views (R2), RAA (§6) and ReGuide with perceived poses. A release counts as trained when it lies more than 3 cm closer to the trained plate position than to the actual plate (SR §81, docs/notes/sr81.md).

**Target moved.** Frozen pi0.5 succeeds on .975 [.925, 1.000] of the far-plate states. It releases at the trained position on 0 of 40, and its median release lies 2.0 cm from the actual plate. The registered prediction that frozen pi0.5 shows a trained-release share above 0 reads NOT SUPPORTED (M1), as predicted at 85%. R2 behaves as the position account says. On the far plate it releases at the trained position on 31 of 40 episodes (.775 [.625, .900]), with a median 20.2 cm from the actual plate, and succeeds on 0 of 40. That finding is descriptive and not registered, and R2 sits below the .70 floor in range at .675 [.525, .825] (SR §81). RAA neither releases at the trained position (.025) nor reaches the moved plate (.100 success), so RAA minus frozen pi0.5 reads -.875 [-.975, -.750] and REVERSED against the prediction that RAA would follow the plate better (M2) (SR §81).

**Object moved.** The weakness of frozen pi0.5 sits in the grasp. With the bowl .14-.30 m away, it succeeds on .375 [.225, .525] of K1 states and .400 [.250, .550] of K2 states, against .850 and .925 for ReGuide with perceived poses. The paired differences are +.475 [+.275, +.650] and +.525 [+.375, +.675] (SR §81). The record does not say where the arm goes when the grasp fails.

**Scope.** In this test, narrow fine-tuning induces the position bias and the broad VLA does not carry it. The test has one scene, two tasks, one seed per state and one checkpoint per retrained system. M1 is weak by construction. The in-range share of trained releases is 0 by the 3 cm rule, so M1 asks only whether the far share exceeds 0. The reach attribution is confounded by the approach path, so we read it as descriptive (SR §81). The result agrees with the nearer-plate routing of §4.1, where the cue is a feature of the scene.

### 4.8 Causes with no large effect, and pending reads

We found no large effect from chunk length (SR §32.1) or from release timing, where pi0.5's lower bound sits at 0 (AF §2.3). We did not reproduce the partner-count law on two frozen VLAs (SR §35.1). Wang et al. state that law for a different policy class (docs/NOVELTY.md).

[A2] A mechanistic probe per failure is to run. This section will carry its result.

## 5 What fixes them

The account predicts where an edit works. An input-only edit hides or paints something in the images or the prompt. It can remove a cue that the task does not need. It cannot remove a cue that the grasp or the carry needs. We report the fixes in four groups.

### 5.1 Input-only fixes

The front end works where the cue is task-irrelevant. In scene 1 on frozen pi0.5, role views plus role phrases lift new pairings from .450 to .817, +.367 [+.092, +.634], with trained tasks at -.010 [-.08, +.04] (SR §47). Silhouettes beat a whole-wrist grey proxy at a matched prompt by +.442 [+.208, +.675] (SR §48). Phrases alone add +.025 [-.15, +.19] (SR §48). With the held-object box, the front end tvpb reaches .938 against .454 for the intact model (SR §55). On GR00T the views alone add +.186 [+.033, +.361] (SR §49).

The front end fails where the cue is task-relevant. In scene 2 the cue is position, and the grasp needs it. Views give .000 on pi0.5 and GR00T, and views plus phrases give .000 on pi0.5. All use simulator masks (SR §50). Views move pi0.5's first grasp to the named mug on about half of N2 episodes, and the mug still reaches its trained plate (SR §50). Placed before a hand-over, the views lift N1+N2 to .508 and cost the trained tasks -.108 [-.158, -.058] (SR §66).

The front end also fails where the failure is the grasp habit. On pi0, tvpb costs the trained tasks -.485 [-.725, -.220]. On SmolVLA it costs -.935 [-1.00, -.810] (SR §65). ReGuide beats it by .317 and .496 (SR §65). The hold detector fires early on those models, so the cost may come from the phrases, the detector or the box. The read cannot separate them (SR §65). On GR00T the box adds nothing, as §4.2 shows (SR §56).

Prior work edits frozen policies at test time. ReGuide, LiLo-VLA and AC-VLA overlap this front end, and we claim no priority for it (docs/NOVELTY.md). The contribution is the test that separates cues task by task.

### 5.2 Transport-level fixes

A transport-level fix sets the grasp target and the carry from perception and returns control to the VLA. ReGuide is the training-free example. We reimplement it from its paper, with our segmenter for poses, and we omit its post-hand-back escort. Our results are on our states and say nothing about its published benchmark (SR §52, §55).

The fix works on four models. In scene 1, our reimplementation on pi0.5 scores .929 against tvpb's .938. The difference is +.008 [-.083, +.092], EQUIVALENT (SR §55). On pi0 it reaches .608 and on SmolVLA .604, with trained tasks unchanged (SR §65). On GR00T it lifts N1-N6 from .212 to .838, +.625 [+.350, +.879], with trained tasks at .000 [-.035, +.035] (SR §71). In scene 2 it reaches N1+N2 .708 [.650, .775] on pi0.5 and .700 on pi0, with trained tasks held. It beats our hand-over by +.350 [+.242, +.458] on pi0.5 (SR §68). Without rendered depth, it keeps .700 [.633, .775] on pi0.5 and reaches .608 on pi0 (SR §69). X-VLA is untested.

Our own hand-over binds only the place. It reaches .442 with simulator inputs and .450 with perception on pi0.5 (SR §62, §63). A scripted put-back does not lift N2 (SR §64), and the clear-away rule lowers it (SR §67). No hand-over variant of ours clears the .50 bar and holds the trained tasks. The hand-over idea has prior art in ReGuide, Harness VLA and LiLo-VLA, so we claim no priority for it (docs/NOVELTY.md).

### 5.3 The resolver and the memory slot

ReGuide's transport leaves F7 and F6 open. On pi0.5 it adds +.000 on relational and +.012 on negated language over the intact model (SR §70). Its bound instance is the referent on only 6 to 34 of 40 episodes across the cells (SR §70).

The composite RC joins the resolver, the memory slot and ReGuide's transport. On pi0.5 it beats ReGuide alone on F7 by +.550 [+.488, +.625] (SP) and +.663 [+.587, +.738] (NG). The trained tasks move by -.008 [-.025, .000] (SR §70). The resolver supplies the gain, and the transport adds +.087 [+.037, +.150] on NG (SR §70). On F6, the slot alone lifts pi0.5 from .517 to .817, +.300 [+.167, +.433]. RC beats ReGuide alone by +.167 [+.050, +.284] and equals the slot, at -.017 [-.100, +.067] (SR §70). ReGuide's pose registration shows no measured memory, at +.117 [-.033, +.267] (SR §70).

On pi0 the composite beats ReGuide on the SP family by +.375 and costs trained task K1t -.100 [-.183, -.033], which reads PARTIAL. Negated language and F6 are not testable there (SR §70). We compare the slot with no trained-memory method. MEM and pi0.7 add memory by training (docs/NOVELTY.md).

### 5.4 The N1 residual

In scene 2, N1 stays at .417-.500 under ReGuide's perceived poses, and it reaches .833-.900 under simulator poses (SR §68, §72). At the N1 release, the rigid held-mug estimate sits 5.65 cm off (SR §69). A box tracker did not shrink that error, at 5.71 cm. N1 moved by +.133 [-.033, +.283], NO CLEAR DIFFERENCE, and N2 fell by .583 (SR §69). SR §72 swaps one pose source at a time on pi0.5. The simulator's held-mug pose gives N1 .567 against .500, +.067 [-.067, +.200]. The simulator's target-plate pose gives .450, -.050 [-.183, +.084]. Both read NO CLEAR DIFFERENCE (SR §72). The oracle variant reaches .900, and the contrast with the held-mug swap is +.333 [+.183, +.467] (SR §72). That contrast also switches to the oracle interaction rule, so it is not pose-only. The registered held-mug account did not hold. The held-mug interval does not exclude an effect up to +.20 (SR §72). The remainder is consistent with how the hold and release events are read. SR §72 did not swap that rule, so we leave the reading untested.


## 6 The failures sit in identifiable parts of the architecture

The atlas says that a VLA chooses its sub-goals by cues. If that account holds, each failure should sit in one identifiable part of the architecture, and removing that part should restore exactly that failure. Every fix that works, as the previous section reports, binds a role to an entity and moves the arm relative to that entity. This section builds that binding into a policy and tests the claim with module removals. Removing the referent resolver restores the relational-language failure (F7). The matching test for the pose memory and the hidden-target failure (F6) stays open. The repair numbers are secondary, and we report them as an honest gap to ReGuide: the best architectural variant reaches .717 on held-out pairings against .897.

### 6.1 The role channel

We first train the binding into the policy on the same demonstrations as a plain retrain, and we test whether it closes the place gap.

**Diagnosis.** A ladder of retrains on LIBERO-Goal's eight known tasks isolates the step that still fails. R0 retrains pi0.5's action expert on the whole instructions and frames. R2 restricts the input to the active role's entity, with role phrases and painted views. R2 fixes the grasp, with a right-grasp rate of .972 on the new pairings N1-N6. It places only .400 of those right grasps (SR §73). Input purity therefore repairs the grasp, and the place step still follows cues (§4.1).

**The edit.** The theory predicts that the place step needs geometry addressed by role. We add one input, the position of the active role entity relative to the gripper. It has three numbers in whole centimetres and a role id. The entity is the object while the gripper is open and the target while it holds. The channel carries no information that the images lack (C3).

In training, the entity comes from each demonstration's own contact events. The grasped entity is the one under the gripper at the close, and the target is the one under the held object at release. All 300 routed demonstrations had a hold event, and the contact sites equal the task definition's slots (SR §73).

At test time, a contact-induced lexicon parses the instruction into the two roles. Learned masks and an estimator with rendered depth and calibration locate the entity, and the estimator holds its last estimate while the entity is occluded. The channel stays within 1 cm of the simulator for 69% of object-role queries and 89% of target-role queries, with medians of 0.30 cm and 0.06 cm (SR §73). The served pi0.5 reads no state input, so the channel enters as a short text suffix on the prompt. It adds no parameters.

**Variants.** All four retrains share the 52,181 frames, the frozen VLM, the action expert, 6,001 steps and one seed (SR §73).
- R0 uses whole instructions and frames.
- R2 uses role phrases and painted views.
- Rrel is R0 plus the channel. It tests whether geometry addressing suffices.
- RAA is R2 plus the channel.

ReGuide (rgp) serves as the external reference. The registered primary contrast is RAA minus R2 on N1-N6. A ledger row fixed the contrasts and the predictions, including 45% for SUPPORTED, before any main rollout (SR §73).

**Table 2.** Matched-data results on 60 states per task in scene 1. The columns give success on the six new pairings N1-N6, the right-grasp rate, the share of right grasps that end placed, and success on the trained tasks K1-K6 (SR §73).

| variant | N1-N6 | right grasp | placed given right grasp | K1-K6 |
|---|---|---|---|---|
| R0 | .042 | .417 | .100 | .853 |
| R2 | .389 | .972 | .400 | .753 |
| Rrel | .278 | .861 | .323 | .911 |
| RAA | .656 | 1.000 | .656 | .839 |
| RAA-s | .717 | .967 | .741 | .844 |
| rgp (ReGuide) | .897 | .989 | .907 | .950 |

**V1: the channel closes part of the place gap.** RAA beats R2 on N1-N6 by +.267 [+.092, +.467] and on placed given right grasp by +.256 [+.077, +.461]. Its right-grasp rate rises by .028, so the place step supplies almost all of the gain. The registered verdict is SUPPORTED (SR §73). On the trained tasks, RAA exceeds R2 by +.086 [+.011, +.172], which meets the -.05 non-inferiority margin. The gain holds for the cheese pairings (+.322 [+.005, +.717]) and the bottle pairings (+.211 [+.078, +.367]).

**V2: the channel needs pure input.** Rrel gains +.236 [-.006, +.553] over R0, and the interval includes 0. RAA beats Rrel by +.378 [+.133, +.658]. The channel therefore acts on top of the restricted input. On N4 and N6, Rrel grasps the right object on only .733 and .567 of episodes, so the bottle's grasp itself fails without the restriction (SR §73). The registered prediction of a clear Rrel gain (60%) missed.

**V3: ReGuide still places better.** RAA trails rgp by -.242 [-.464, -.033], which misses the -.10 non-inferiority margin. Its placed-given-right-grasp rate of .656 compares with .907 for rgp (SR §73). ReGuide uses demonstration statistics at test time, and RAA uses none. The channel narrows the gap between the trained retrains and ReGuide without closing it.

**Where the remaining gap sits.** Instrumented rollouts trace RAA's misses to the release (exploratory diagnostic, 160 episodes; docs/notes/raa_instr_results.md). In 47 of 51 failures RAA carries the right object to the right target and lets go outside LIBERO's placement tolerance. ReGuide's median release error is 0.9 to 1.7 cm; RAA's failed releases miss by 3.7 to 7.9 cm. On the bottle pairings that error tracks the held object's offset from the gripper (r = .57 on N4), because the channel measures the target from the gripper.

**An object-relative channel removes that dependence and does not raise success.** RAA-o measures the target from the held object, with the in-hand offset fixed at the grasp. It uses the same data, recipe and views as RAA. The release error stops tracking the in-hand offset (r = .14 on N4, .03 on N6). The error itself barely shrinks (N4 6.5 cm against 7.0), and new-pairing success shows no clear difference from RAA: -.061 [-.264, +.108]. The registered verdict is NOT SUPPORTED (SR §76). RAA-o trails ReGuide by -.303 [-.481, -.111]. The remaining miss is aim. The policy reads a target offset in centimetres and still steers to within about 3 to 7 cm. ReGuide closes that last few centimetres with a pose controller. A trained action expert reading the same geometry as text does not.

**Entering the offset as numbers, not text, does not help either.** RAA-c feeds the same object-relative offset to the action expert as a continuous state token, through one new linear layer of 33,792 parameters, and drops the text suffix. Everything else matches RAA-o. The channel reaches the model intact (median error of the fed target .94 cm). New-pairing success falls to .503 and shows no clear difference from RAA-o: -.092 [-.306, +.072]. The registered verdict is NOT SUPPORTED (SR §77, docs/notes/sr77.md). RAA-c trails ReGuide by -.394 [-.608, -.158]. Known-task success is .814, against .881 for RAA-o: -.067 [-.233, +.061], so non-inferiority is not shown. In an exploratory check, failed releases miss by more, not less (N4 11.8 cm against 6.5 for RAA-o). The format of the number does not limit the place step. A behaviour-cloned action expert given exact target geometry still does not convert it into centimetre-level releases, and the pose controller in ReGuide does.

**A fixed servo term gives the best architectural variant and leaves the gap to ReGuide open.** RAA-s adds a fixed servo residual on the three translation dimensions of each chunk to RAA-c, with the same data and recipe (SR §78b, docs/notes/sr78b.md). Our first serve of the checkpoint omitted the servo term, which is a serving bug, and its collapse to .028 on N1-N6 (SR §78) carries no evidence about the servo. We report the re-serve of the same checkpoint with the servo, with no retraining. RAA-s reaches .717 on N1-N6, against .656 for RAA, .594 for RAA-o and .503 for RAA-c, with .844 on the trained tasks and a right-grasp rate of .967. The registered primary contrast against RAA-o reads PARTIAL. Success rises by +.122 [-.019, +.264], and placed given right grasp rises by +.140 [+.008, +.280]. The gain concentrates on the bottle pairings (+.261 [+.133, +.400] over RAA-o), and the cheese pairings show no difference (-.017 [-.133, +.067]). RAA-s beats RAA-c by +.214 [+.075, +.381] and shows no clear difference from RAA (+.061 [-.117, +.253]). In an exploratory check, failed bottle releases land 6.5 cm (N4) and 3.8 cm (N6) from the target, against 11.8 and 11.0 cm for RAA-c. ReGuide still leads by -.181 [-.300, -.055], and trained-task non-inferiority against RAA-o is NOT SHOWN (-.036 [-.178, +.075]).

<!-- SR 82 RAA-sL pending -->

### 6.2 Module removal

The resolver R turns a relational instruction such as "the bowl on the left" into one entity. The pose memory M keeps the last estimate of an entity that a cover hides or a view loses. R serves F7, and M serves F6. A removal test deletes one module and checks that exactly one failure returns. We run three arms on the same states, with the RAA checkpoint as the policy: full (R and M), noR (M with no R, where ReGuide's largest-component rule binds one bowl) and noM (R with no M, where the channel suffix is omitted whenever its estimate is held). The registered contrasts are D1 (full minus noR on F7), D2 (full minus noM on F6 after one view of the plate), D3a and D3b (the cross terms, which should show no difference) and DK (trained tasks). A gate marks a cell testable when full scores at least .50 on its unhidden control (SR §79, docs/notes/sr79.md).

**Narrow initialisation leaves the gates untestable.** On the RAA checkpoint, full scores .888 on the trained tasks. It scores .083 on the unmoved-plate control of F6 and .108 on the unique-name F7 controls, where frozen pi0.5 scores .967 on the F6 control. Narrow retraining on the eight known tasks destroyed the competence outside the suite. D1 reads SUPPORTED at +.213 [+.131, +.300] and D2 reads NOT SUPPORTED at +.050 [.000, +.117]. The primary contrast reads NOT SHOWN. Because the policy fails the cell, a removal on it cannot show that M is needed, and we draw no module conclusion from this checkpoint (SR §79).

**A broad initialisation restores the gates.** RAA-L trains the RAA recipe from openpi's LIBERO-fine-tuned pi0.5, with the same data, frozen VLM, 6,000 steps and one seed. Held-out pairings do not improve, at .575 against .656 for RAA (-.081 [-.181, +.022]), and the registered L1 reads NOT SUPPORTED. The loss concentrates on the bottle pairings (-.133 [-.233, -.022]). The broad initialisation restores the competence outside the suite: F6 control .967 (RAA .083), unique-name controls .558 (RAA .108), trained tasks .958. All three gates are testable (SR §80, docs/notes/sr80.md).

**The resolver restores F7.** Removing R lowers success on the pooled relational cells from .675 to .413, so D1 reads SUPPORTED at +.263 [+.213, +.313]. The full policy beats frozen pi0.5 (.350) and ReGuide (.356) on the same cells. The gain sits on "right" (+.850 [+.725, +.950]) and "left" (+.200 [.000, +.400]), and the server log shows that R binds the correct referent on every template (SR §79). Removing R leaves the F6 cell identical on all 60 states (D3a EQUIVALENT) and leaves the trained tasks unchanged (DK NON-INFERIOR). Both comparisons hold by construction, because R does not engage on those cells (SR §79, §80). The failure of F7 therefore sits in the missing resolver.

**The pose memory stays open.** Removing M lowers F6 success from .333 to .217, a difference of +.117 [-.017, +.250] that reads NOT SUPPORTED. Full sits below frozen pi0.5 on that cell (.517) and loses .633 [.500, .767] against its own unmoved-plate control. The channel's median error on the hidden plate is 5.1 cm, so the channel limits that cell. Removing M leaves the F7 pool within the equivalence bound (D3b, -.031 [-.069, +.006]). The registered primary reads NOT SHOWN, and this time the gates make it a test of M. We report the F6 localisation as open (SR §80).

**Limits of the removal test.** Both removals use one seed, one trial per state and 40 to 60 states per cell. The relational "position" template scores 0 for every non-composite arm, so it carries no information on the modules. Comparisons with frozen pi0.5 and ReGuide cross policies and isolate no module.

### 6.3 Prior art and limits

**Prior art.** Each component exists. FOCI predicts motion in an object-relative frame. OA-WAM and POT-VLA feed object tokens or role-indexed records to an action expert, and w2VLA injects a language-conditioned location through FiLM. AC-VLA and LiLo-VLA restrict the input by gripper phase (docs/NOVELTY.md; reports/novelty_raa.md). We read FOCI, OA-WAM and POT-VLA in full text through a summarising fetch, and the others by abstract. OA-WAM already adds a relative-pose bias between object slots and the robot slot, which comes closest to our channel. We claim the decomposition that the theory derives, which pairs input restriction with one role-addressed geometry channel, and its test against a plain retrain at matched data on held-out pairings. We claim no priority for any single component.

**Limits.** The tests use one scene, one backbone (pi0.5) and one seed per variant. The place gap shrinks and stays open, with RAA-s at .717 against .897 for ReGuide. The channel enters through the prompt or a state token, and its depth and calibration inputs are privileged, as they are for ReGuide's perceived-pose runs. Scene 2, where position is the cue, remains to be run for the channel and the module removals.

## 7 Predicting new-pairing failure from trained tasks

A practitioner who holds a new policy wants to know whether it will fail a new pairing without running one. The atlas suggests a route. A policy that follows the instruction's roles should switch its sub-goal when the instruction's object or target word changes. A trained task supplies the test, because its scene and success are known. This section builds that probe, reports its pre-registered failure, and reports a corrected score that a later, pre-registered test confirms on two new policies.

**The probe.** At every query of a closed-loop rollout of a trained task, the client sends a side query on the same observation with one word swapped. In the grasp phase, before the first hold, the object word changes to another movable object in the scene. In the place phase, during the hold, the target word changes to another target. The side chunk is never executed. A query scores 1 when the first five steps of the chunk point nearer the swapped-in entity than the trained one. The grasp steerability g_s(o) and the place steerability p_s(t) average that score. The prediction is ŝ(o, t) = g_s(o) · p_s(t), with no fitted weight. The product form follows Okawa et al. (docs/NOVELTY.md). We probed 16 policies in scene 1 and 3 in scene 2 with 10 states per trained task (SR §74).

**The pre-registered headline failed.** The headline set holds the wrapper-free policies, which are intact pi0.5, GR00T, pi0 and SmolVLA, on 24 cells. It gives AUC .550 [.000, 1.000] and Spearman .227 [-.440, .812], and the verdict is NOT SUPPORTED. Across 15 policies, the policy-level Spearman ρ with success on N1-N6 is .193 [-.424, .736], also NOT SUPPORTED (T1). On the four wrapper-free policies, trained-task success alone ranks better, with AUC .800. Scene 2's place-steerability test (T3) also failed its bar (SR §74).

**Table 3.** Ranking of cells by measured new-pairing success, where a cell is a policy and a new pairing and the label is success of at least .5. AUC is the area under the ROC curve. Brackets give 95% intervals.

| score | cells | AUC | stage |
|---|---|---|---|
| ŝ, wrapper-free policies | 24 | .550 [.000, 1.000] | registered headline |
| ŝ, scene 1 | 90 | .742 [.526, .907] | registered, missed the .75 bar by .008 |
| trained-task success alone | 90 | .645 [.419, .846] | baseline |
| nonsense-word sensitivity | 90 | .462 [.240, .684] | baseline |
| k × a (success times floor-corrected steering) | 90 | .787 | post hoc |
| k × a, Rrel and RAA | 12 | .944 [.667, 1.000] | pre-registered, new policies |
| k × a, all policies | 102 | .805 [.603, .940] | pre-registered, new policies |
| trained-task success alone, Rrel and RAA | 12 | .500 | baseline |

On all 90 scene-1 cells, ŝ passes the lower bound and beats both baselines (SR §74). That result carries no headline credit, because the registered rule named the wrapper-free set.

**Two problems explain the failure.** First, the angle rule has a floor. Under the true prompt, the chunk points nearer the swapped-in entity on .22 to .28 of scene-1 queries, because swapped-in entities lie near the hand's path. A steerability of .5 therefore sits above chance. Intact policies also steer in the place phase at .33 to .55, so F1 does not appear as place steerability near 0. Second, the entity decides most of the score. Swaps toward the wine bottle score .13 to .47, and swaps to the bowl target score .00 to .46 on every policy. The product partly ranks pairings by geometry (SR §74).

**Exploratory corrections.** After the failure, we explored variants on the stored probe outputs. These changes are post hoc, and the intervals of the registered score contain them. Subtracting the true-prompt floor raises the 90-cell AUC from .742 to .770. Multiplying the floor-corrected product by trained-task success, which we call k × a, reaches .787 and Spearman .579. Normalising by entity lowers the AUC to about .60 to .65, which indicates that the pooled signal is largely entity difficulty. On the wrapper-free four, no variant beats trained-task success at .800 (PLAN_PAPER §2c).

**Confirmation.** We froze k × a in code and wrote a ledger row before probing two policies that did not exist when we chose the variant. These are Rrel and RAA from the role-channel test. Success on the 12 new cells is the label (SR §75). Trained-task success alone cannot separate them, because both policies have it at .94. The score ranks the 12 cells at AUC .944, which clears the .75 bar. The descriptive interval's lower end of .667 sits below the bar, so the pass is a point estimate. Pooled over 102 cells and 17 policies, the AUC is .805 [.603, .940] and Spearman is .575 [.254, .756]. The score ranks RAA above Rrel (mean .364 against .301), as their success does (.656 against .278). One cell fails clearly. Rrel on N1 scores .393 and succeeds on 0, because it grasps the right object and never places it.

**Scope.** The confirmation covers two policies, six pairings and one scene, all pick-place tasks. Both policies descend from one base, so the evidence says that the score ranks variants of one base better than unrelated frozen models. Within the GR00T variants the cell-level Spearman is .922 over 18 cells, and the wrapper-free policies give no usable ranking (SR §74). Steerability tracks which entities a policy can be steered toward. A claim that it predicts which unrelated policies generalise has no support here. ReGuide, Hp and RC lie outside the probe, because their controllers take over the episode.

**Prior art.** Okawa et al. state the multiplicative form for compositional skills. STAGE and LIBERO-PRO measure action sensitivity to instruction changes as a diagnostic, and none of them validates a forecast across policies. World-model evaluators such as the Veo simulator for Gemini Robotics policies rank policies, and they need closed-loop rollouts of the held-out condition in a learned simulator. Our probe needs only trained-task rollouts. We compare cost only (reports/novelty_predictor.md).

## 8 Conclusion and limitations

- Two LIBERO scenes.
- GR00T below its floor in scene 2.
- X-VLA has no fix.
- The fixes still need depth and calibration.
- The role channel and the module removals run in scene 1 on pi0.5 with one seed. The place gap stays open (RAA-s .717 against ReGuide .897), and the pose memory's role in F6 is open.
- The moved-plate test covers one scene, two tasks and one checkpoint per retrain.
- The failure predictor is confirmed on two policies that descend from one base.

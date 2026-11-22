# Adversarial ICML 2027 review: "Routed by Cues" (paper/DRAFT.md, draft of 2026-11-21)

Reviewer role: hostile but fair, robot learning / VLA interpretability. Written 2026-11-22.
Inputs: the full paper/DRAFT.md (494 lines), paper/figs/captions.md, the four reports/ novelty notes (read as claims to check), and about a dozen web searches and arXiv abstract fetches done for this review. Web reads are at abstract level unless noted. No code was run and no results files were opened.

---

## 0. One-paragraph verdict

The draft is a careful, honest and heavily pre-registered lab record. It is not yet a paper with one contribution. The headline diagnosis, that a VLA keeps its local skill and lets scene cues choose the referent, is published by at least five groups in 2026. The proposed architecture (RAA-sL-W5) loses to ReGuide on the main metric twice under registration. Its "where vs how" division of labour is ReGuide's own title ("Correcting WHERE, Preserving HOW"). The "localisation" result is a lesion study of modules the authors built to fix the very failures they then restore, and two of its cross terms hold by construction (the draft says so in §6.2). The one in-network localisation, activation patching in §4.8, fails its registered criterion and finds no selective band. As it stands this is a clear reject at ICML. A narrower paper exists inside it, built from the negative and mechanistic results that nobody else has. Section 3 says what that paper is.

**Score as it stands: 3 / 10 (reject).** After the fixes in Section 4: **5 / 10 (borderline), and 6 / 10 only if the behaviour-cloning shortcut result replicates on a second backbone and a second seed.**

---

## 1. Closest prior work and what each one already shows

ids below were resolved during this review or by the authors' notes. "(abs)" means abstract-level reading.

| Work | What it already shows that overlaps this draft |
|---|---|
| **ReGuide**, arXiv 2609.38616, "Correcting WHERE, Preserving HOW" (abs) | VLAs "fail at global grounding but retain local manipulation skills". It is a training-free wrapper that rebinds the referent from grounded poses and steers the end-effector into *demonstration-supported configurations*, then hands back to the frozen policy. Up to +56.8 pts sim and +75.0 real, across several backbones. **This pre-empts the thesis of §6.3 almost word for word**: "the module owns where the object goes and when it is let go, and the expert keeps how to move". W5's release-point table mu (median release offset per target class from the training demonstrations) is a demonstration-supported configuration in ReGuide's sense. W5 = ReGuide's where-correction moved inside the serving loop of a retrained policy, and it still loses (-.022 twice). |
| **CRAFT**, 2610.00524 | Same setting (LIBERO-Goal held-out object-target pairings, pi0.5 / pi0 / GR00T). Table A.4: pi0.5 completes a *share-object* trained task on 38.4% of held-out rollouts, which means it grasps right and goes to the trained partner. That is F1 as an outcome class. Its fix reaches pi0.5 84.2 on held-out pick-place. |
| **Encoded but Not in Control**, 2610.06235 (abs, Oct 2026) | The instruction's target is linearly decodable, the policy still approaches the scene-favoured object, and target-token attention stays on the original object. This is the routing claim with attention-level evidence. The draft's own evidence for routing is behavioural (§4.1 Scope says so). |
| **Grant et al.**, 2603.19233 (abs) | Goal selection sits in the VLM pathway and the motor program in the expert. Expert-layer injection redirects pi0.5 in 99.6% of 1,968 episodes, and the program is "bound to absolute workspace coordinates". Two problems for the draft. (i) The mechanistic story of goal selection is theirs. (ii) A search hit reports that an ICLR 2026 version of the study gives 0% cross-task transfer. **Check which version you cite.** The moved-plate result (§4.7, frozen pi0.5 follows the plate) is in tension with "absolute workspace coordinates", and the paper should say so, because that tension is one of its few genuinely new findings. |
| **Wulff & Cangelosi**, 2610.10178, "Do VLAs Understand Instructions? A Mechanistic Interpretability Study on Language Grounding" (abs, Oct 2026) | Activation and attribution patching on the residual stream of the **pi0.5 and GR00T N1.7 action experts** under instruction corruptions on LIBERO. They find pi0.5's sensitivity "distributed across the earliest layers and selected later layers" and GR00T's concentrated in early cross-attention. **They also find both models react strongly to directional language.** That pre-empts the method of §4.8 (write-level patching of pi0.5 under a counterfactual instruction) and the qualitative finding ("distributed, no single site"). It also conflicts with §4.4 / §4.8, where swapping left/right moves pi0.5's chunk by .004. The draft does not cite this paper. |
| **Robust Skills, Brittle Grounding**, 2602.24143 | Held-out object-location pairings in SmolVLA and pi0.5. Skill is intact and target selection is brittle. Stage-wise metrics. 10x more data gives no gain. F2 (habitual grasp on SmolVLA) is their result. |
| **LIBERO-CF**, 2602.17659; **LIBERO-PRO**, 2510.03827; **LIBERO-Plus**, 2510.13626 | Vision overrides language (pi0.5 13.2% of faithful counterfactuals). Collapse under object and instruction perturbations. Invariance to nonsense and negated instructions. F7's negation half ("0 of 20 on a negated LIBERO-PRO instruction") is their observation. |
| **ACT-VLA**, 2607.00351; **AC-VLA**, 2607.15714 | The same recomposition protocol (object from one task, target from another) on LIBERO-Goal/Spatial-OOD with pi0.5. ACT-VLA reaches 95.6 on Goal-OOD from 46.6. AC-VLA masks the wrist view by gripper phase (73.3). These are data or training fixes that reach or beat .878 / .892 without privileged depth or a hand-set servo. The splits differ, and a reviewer will line the numbers up anyway. |
| **w2VLA**, 2606.21496 | An architectural where/what split ("where-what-VLA") with a held-out skill-object transfer test against plain pi0.5: 91.7 vs 38.2. It is real-robot and small, but it is the same claim at the architecture level: separate "where" from "what" and new pairings work. The draft read it by abstract only (§2), and it has to be read in full. |
| **OA-WAM** 2605.06481, **POT-VLA** 2607.18016, **LiLo-VLA** 2602.21531, **FOCI** 2609.08743 | Relative-pose channels and role-indexed object records in the action head. POT-VLA feeds a role-indexed end-effector-to-object offset to a GR00T action expert, which is the RAA channel. LiLo-VLA has a relative frame and non-target masking. The draft already concedes "no priority for any single component" (§6.5). |
| **Nouns, Not Modifiers** (ICML 2026, OpenVLA) | OpenVLA parses nouns and fails spatial modifiers (left 15%, right 80%). That is F7 on another VLA family. |
| **RoboMME** (ICML 2026), **LIBERO-Mem** (AAAI 2026), **MEM** 2603.03596 | Memory and occlusion benchmarks for pi0.5-class policies, and memory-augmented pi0.5 variants. F6 and the pose memory M sit in a populated area. F6 also turns out not to be a memory failure on frozen pi0.5 (f6_diag). |
| **Causal confusion in imitation learning** (de Haan et al., NeurIPS 2019); copycat / shortcut literature (Wen et al. 2020; Geirhos et al. 2020) | §6.4's thesis is that a BC network ignores an input when another path already explains the actions. That is the causal-confusion / shortcut phenomenon, and it must be cited and positioned against. |
| **Benchmark audit**, 2606.04233 (CoRL 2026 oral) | A 0.09B DINO+MLP probe nearly solves LIBERO, and only 19.8% of LIBERO SOTA claims are significant. Reviewers will use it to discount any LIBERO-only paper. This draft's statistics (paired bootstrap, fresh states, registration) are a strength against that criticism and should be advertised as one. |

Search gaps: OpenReview ICLR 2027 submissions were not searchable. LIBERO-Gen (ICML 2026) has no public full text. ReGuide's controller and demo-statistics details were not verified beyond its abstract.

---

## 2. Contribution by contribution

I read the draft as claiming six things: C-A the routing diagnosis and the C1-C3 criteria; C-B the five-failure atlas; C-C the moved-plate test; C-D the steerability predictor; C-E the architectural localisation by module removal; C-F the repairs (RAA ladder, servo, W5), plus the RAA-F data routes (§6.4) as a seventh, unlisted result.

### C-A. "One routing defect: cues, not roles, pick the sub-goal", plus the C1-C3 criteria
- **Status: already known (diagnosis), incremental (criteria).**
- Strongest prior: ReGuide ("fail at global grounding, retain local skills"), Encoded but Not in Control (target encoded, scene-favoured object approached, attention stays on it), CRAFT Table A.4, LIBERO-CF.
- What is new: the pre-registered majority rule in scene 2 (place goes to the plate backed by two of {nearer, salient, trained}, .912 of 445). That is a concrete, falsifiable cue law, which no prior paper states.
- Too weak: the majority rule is pi0.5 only, one LIBERO-10 scene with two plates, and "salient" is the authors' label. C3 ("a structural change that adds no information removes the failure") is contestable for every fix used. They use rendered depth plus calibration, simulator masks (scene-2 views), a parsed scene (resolver), and simulator state (SR 89). A 3-D target offset from depth and calibration is not information that "the images" contain in any usable sense for an RGB policy. C2 tests models, chunk length and prompting, and no data scale (the draft admits this). The criteria are sensible bookkeeping and do not count as a contribution at ICML.

### C-B. The failure atlas (F1, F1m, F2, F7, F6, new-layout limit) on five VLAs
- **Status: mostly known, partly incremental.**
- F1 (grasp right, place wrong): CRAFT A.4, ReGuide. F2 (habitual grasp on pi0/SmolVLA): Brittle Grounding, CRAFT's pi0 share-target class. F7 (relations and negation ignored): LIBERO-PRO, Nouns-Not-Modifiers, and contradicted in part by Wulff & Cangelosi's directional-language sensitivity. F6: re-read post hoc as placement precision at LIBERO's 3 cm On() tolerance on frozen pi0.5, because the occluder marks the plate. So for the model the atlas is mainly about, F6 is not a memory failure, and the atlas row "a covered target costs ... the target's location in GR00T" rests on GR00T alone. New-layout limit: every system fails, including the planners, so it carries no claim about VLAs.
- New: the excess-over-known-partners measure E, the five-model coverage including X-VLA, the reference planner on the same states, and the F1m outline cue (paint the bottle as a box: +.188 on pi0.5, which does not transfer to GR00T).
- Too weak: F1m has no intervals (caption says so). F7 on X-VLA/SmolVLA SP "has no power". GR00T misses the testability floor in scene 2. The atlas runs on development states, and only F1/F2 replicate on fresh states. Scene 2 has one trained task, so "new pairings" there are swaps of a single compound demo, and a 0/720 failure is close to expected by construction. The planner pools to only .65 on N1+N2, so C1 fails for N1.

### C-C. Moved plate: frozen pi0.5 follows the plate, and narrow retraining induces the position bias
- **Status: possibly novel and interesting.** It conflicts with Grant et al.'s "absolute workspace coordinates" reading, and that is worth stating directly.
- Too weak: one scene, two tasks, one checkpoint per retrain. The R2 result is "descriptive and not registered", and R2 sits below the .70 floor. M1 is "weak by construction" (the in-range share is 0 by the rule). **And the proposed architecture fails the test**: RAA reaches the moved plate on .100 and reads REVERSED (-.875). A reviewer will read §4.7 as evidence that the authors' narrow retrains, RAA included, are less robust than the frozen VLA they set out to fix.

### C-D. Steerability predictor of new-pairing failure
- **Status: incremental at best. It failed its registered test.**
- Strongest prior: Okawa et al. (product form), STAGE/SAT-Bench and LIBERO-PRO (instruction-sensitivity probes), and world-model evaluators (Veo 2512.10675) for ranking policies.
- Too weak: the registered headline gives AUC .550 [.000, 1.000] and policy-level Spearman .193. The corrected score k x a is post hoc, and multiplying by trained-task success brings in the baseline that alone gave .800 on the wrapper-free set. The "confirmation" is 12 cells from **two policies of one base** (Rrel, RAA). Its lower bound of .667 sits below the .75 bar, and the comparison baseline is set to .500 only because both policies sit at .94 trained success. Steering scores are dominated by entity geometry (entity-normalised AUC about .60-.65). An ICML reviewer will call this a failed predictor with a post hoc rescue. **Cut it or move it to the appendix.** It costs a contribution slot and credibility.

### C-E. "The failures sit in identifiable parts of the architecture" (module-removal dissociation)
- **Status: the strongest-looking result, and conceptually the weakest. As framed, it is close to circular.**
  1. R and M are modules the authors added to fix F7 and F6. Removing a resolver that is the only thing resolving "left" restores the left/right failure. Removing a memory that is the only thing holding a hidden target's position restores the hidden-target failure. That shows the modules work as built. It does not show that the *VLA's* failures "sit" anywhere.
  2. The cross terms hold by construction. The draft says so itself in §6.2 for RAA-L ("Both comparisons hold by construction, because R does not engage on those cells"). D3a is "identical on all 60 states" and DK is "0 [0, 0]" on every set. Those are constants, not tests. "Equivalent cross terms" should not be presented as replicated evidence.
  3. The F6 half measures a failure the RAA design created. Frozen pi0.5 scores .517 on F6_b, RAA-sL scores .017, and the RAA-sL failure is the expert braking short under the cover. M then "restores" a location failure that, in the models where it is a location failure, the authors' own channel and servo design induced. On frozen pi0.5, f6_diag shows that F6 is not a lost location at all.
  4. The development history is a garden of forking paths. D2 reads NOT SUPPORTED (RAA-L), then REVERSED (RAA-sL), then SUPPORTED with gates failing (W1-W4), then SUPPORTED on fresh states after redesign. The fresh-state replications are a genuine strength. The *design* of the module they test was still selected on the outcome it is tested on. D2's first fresh lower bound is +.033.
  5. **The in-network localisation is null.** In §4.8 the registered "distributed" criterion is not met (PR .155 < .2), the effect is not localised, no band is selective, and band ablations hurt working tasks more than failing ones. The "read at a narrow KV-14 site" observation is unregistered and descriptive. On RAA-sL the localisation sits "in a component built to carry it, a fixed function of the channel" (the servo carries .878). So the title's claim, "where in the architecture they sit", has a hand-coded servo as its answer.
- Strongest prior against the framing: Wulff & Cangelosi 2610.10178 (patching pi0.5 / GR00T action experts, distributed sensitivity). Modular lesion logic is ordinary in modular robot stacks.
- Honest reframe: "a modular decomposition whose parts are separable: each module carries exactly one failure class, on three fresh state sets". That is a property of the system the authors built, and it is a modest contribution.

### C-F. The repairs: RAA ladder, servo, RAA-sL, W5 / W6
- **Status: dominated.** On the main metric it does not beat the external baseline in either registered test (-.022 [-.114, +.047], -.022 [-.147, +.069]). The pooled +.001 uses two sets that were seen before registration.
- Strongest prior: ReGuide (same where/how split, training-free, wins). ACT-VLA, CRAFT and AC-VLA reach comparable or higher held-out success with data or training fixes and no privileged depth. POT-VLA, OA-WAM and LiLo-VLA hold the channel components.
- What survives: V1, the clean matched-data ladder (R0 .042, R2 .389, Rrel .278, RAA .656; RAA - R2 = +.267 [+.092, +.467], registered SUPPORTED), and V2 (the channel helps only on top of restricted input). Those are controlled ablations at matched data, and few prior papers have them. Beyond V1 and V2:
  - RAA-sL's jump to .878 comes mostly from the initialisation (RAA-sL vs RAA-s: NOT SUPPORTED, lower bound -.017).
  - W5 needs a hand-set gain, clearance, hand-off radius and per-class release table, and "a target class absent from the demonstrations cannot be aimed".
  - W5 keeps the unique-name controls below RAA-sL on every set.
  - It fails the moved plate (C-C).
  - The ReGuide reimplementation omits its post-hand-back escort. A reviewer will say the baseline was weakened and still won.
- Wins on F7 and F6 over ReGuide are real but cheap. ReGuide has no relational resolver, and the F7 resolver is a fixed-template rule on a parsed scene, "a routing fix" with "no language-understanding claim" (the draft's words). Bolting a template parser onto ReGuide would erase that margin. The F6 advantage over *frozen pi0.5* holds on two of three sets.

### C-G (unlisted, §6.4). Behaviour cloning does not take up an exact where-signal
- **Status: the most interesting result in the draft, and under-sold.** Every attempt to make the trained expert *use* exact geometry fails:
  - text offset (RAA-o): no gain;
  - continuous state token (RAA-c): patching share -.007, so the network does not read it;
  - target crop (B): .022;
  - in-support relabel (A): .274, below the .70 bar;
  - large planner dose with the servo kept (A+): -.007.
- The grid shows a clean pattern: the network reads the where-input only when the servo is off and the data demand it.
- Strongest prior: causal confusion (de Haan 2019), copycat (Wen 2020), and general shortcut learning. None of them shows it for VLA action experts with a patching-based uptake measure.
- Too weak:
  - **The confound a reviewer will raise first:** the served pi0.5 LIBERO checkpoint was trained with no state input. The token enters through a *new* 33,792-parameter linear layer trained for 6k steps with the VLM frozen. "The network does not read the token" may be an optimisation or initialisation artefact (an under-trained new layer, small gradient into it), not a BC shortcut. pi0, which does read state, is the obvious control.
  - A+0 is pending and post hoc.
  - One seed, one checkpoint per route.
  - Development states only. No route passed, so no confirmatory read exists.
  - The prefix half of the pass rule "fails every servo-free route by construction" (the draft concedes this).

---

## 3. "If the architecture does not beat ReGuide, is there a paper?"

**Not the paper that is drafted.** The current title, abstract and contribution 3 promise an architectural explanation plus a fix. The fix loses to a training-free wrapper whose title states the same where/how thesis. The explanation's in-network half is null, and its module half is a lesion of hand-built parts. An ICML reviewer who reads ReGuide's title next to §6.3 will stop there.

**Framing that gets rejected (the current one):** "We locate VLA failures in the architecture and propose a de-distributed architecture (RAA-sL-W5) with a where-module, resolver and pose memory." The reviewer's reasons:
- (i) the diagnosis is known (CRAFT, ReGuide, Encoded-but-Not-in-Control, LIBERO-CF, Grant);
- (ii) the architecture is ReGuide inside the policy plus a template parser and a pose cache, and it loses to ReGuide;
- (iii) the dissociation is by construction;
- (iv) one seed, one simulator, one backbone, privileged depth and calibration, a hand-set servo;
- (v) the predictor failed its registration;
- (vi) the paper reads as a lab notebook (SR numbers, capitalised verdicts, a W1 to W6 chronology).

**Framing under which a reviewer could accept it (an analysis paper, not a methods paper):**

> *"Behaviour-cloned VLA action experts do not take up explicit geometry, so the 'where' of a placement must be owned outside the learned network: a pre-registered study."*

The evidence, in order:
1. The routing failure is reproduced across five VLAs with a same-state reference planner and an excess-over-partners measure. Keep this to one figure and one table, and cite CRAFT, ReGuide and Encoded-but-Not-in-Control as the established diagnosis. Do not claim the diagnosis.
2. Four ways of handing the expert exact target geometry (text, state token, crop, relabel) are measured by causal patching. All leave its where-share at or near zero unless the servo is removed and the data demand it, and even then the share stays at .27. Every claim here runs at matched data.
3. Only an explicit module that owns the where (W5) makes the geometry effective, and it lands level with ReGuide, the external module of the same kind. This turns the loss to ReGuide into *corroboration*: the two explicit-where systems, one inside the serving loop and one outside the policy, converge on the same number (.89-.92), and no learned route gets there.
4. The moved-plate result: broad pre-training does not carry the position bias and narrow fine-tuning induces it, in tension with Grant et al.
5. Separable modules: each added module carries exactly one failure class on three fresh sets. Present this as a property of the decomposition, with the by-construction cross terms labelled as such.

This framing makes the ReGuide comparison a consistency check rather than a race. It puts the draft's unusual assets (registration, fresh state sets, patching-based uptake measures) at the centre, and it makes an ML claim (BC shortcut in a large pretrained policy) that ICML reviewers care about. Its ceiling at ICML is still a borderline accept, because the scope is one simulator and pick-place. Unless the shortcut result replicates on a second backbone, it belongs at CoRL 2027 or RA-L. The draft's own PLAN_PAPER lists CoRL 2027 as the fallback, and I would treat that fallback as the likely home.

---

## 4. Most likely reviewer objections, ranked, with the minimum fix

Cost estimates assume the existing harnesses and that the second-seed run now training is one of the retrains below.

**O1 (fatal as framed). "Known diagnosis, and the method loses to ReGuide, which already states the where/how split."**
- Minimum fix (rewrite, 0 GPU-h): retitle and reframe as in Section 3.
  - Drop "we propose an architecture" as a contribution.
  - Cite ReGuide's title in the introduction and concede the where/how framing to it. State the delta: ReGuide shows an external where-correction works; this paper shows why a learned expert cannot absorb one (C-G) and that an internal module converges to the same ceiling.
  - Add and discuss Encoded but Not in Control (2610.06235), Wulff & Cangelosi (2610.10178), Nouns-Not-Modifiers, RoboMME / LIBERO-Mem, and causal confusion (de Haan 2019).
  - Read w2VLA in full.
  - Resolve the Grant et al. version conflict (99.6% vs 0%).

**O2 (fatal for the "where in the architecture" claim). "The dissociation is circular. Removing a module you added restores the failure you added it for, the cross terms are constants, F6 in RAA models is a failure your design induced, and the in-network patching is null."**
- Minimum fix:
  - (a) Rewrite (0 GPU-h): rename §6 "a separable modular decomposition", label D3a/DK as holding by construction, and state that the F6 location failure is induced by the channel and servo design (RAA-sL .017 vs frozen .517). Drop "the failures sit in identifiable parts of the architecture" from the title and abstract.
  - (b) If you want any in-network localisation claim, run **one** targeted causal test on frozen pi0.5, reusing the SR 83 open-loop replay harness: patch layer-14 prefix keys and values restricted to (i) instruction-token positions and (ii) image tokens of the cue entity, on N1-N6 versus K tasks. Test whether instruction-token KV-14 patching flips the place target and image-cue KV-14 patching carries the habitual target. This is open-loop, roughly 10-20 GPU-h, and turns the KV-14 cliff, the one positive mechanistic observation, into a registered, selective result. Register it before running.
  - Otherwise, call §4.8 a negative result in one paragraph.

**O3. "One seed, one backbone, one simulator, two scenes, development-state design selection, dozens of registered tests with no multiplicity correction, intervals that resample only six tasks."**
- Minimum fix:
  - Second seed **only** for the claims kept in the reframed headline: R2 and RAA (V1), and A0 and A (the where-share grid). That is 4 retrains, with the one in training counted.
  - Holm-correct the registered family that the headline uses, and report it.
  - Report per-task results next to task-resampled intervals.
  - Do not spend compute on more W variants. W6 already lost, and more variants make the forking-paths objection worse.

**O4. "The 'network ignores the token' result is an artefact: pi0.5-LIBERO was trained with no state input, the token enters through a new 34k-parameter layer trained 6k steps with the VLM frozen. Show it on a model that natively reads state, or show the layer is trained."**
- This is the objection that decides whether C-G is a finding. Minimum fix, two cheap checks:
  - (i) Report the gradient norm and weight change of the new token layer, plus an oracle test: a version where the token is the *only* source of the target (views and phrase uninformative). The network must read it there. If it does not, the result is an optimisation failure.
  - (ii) Repeat RAA-c vs RAA-s (token share, servo on and off) on **pi0**, which has a trained state pathway in pi0_libero. That is two retrains plus the probe.
- If pi0 also shows a share near 0 with the servo on, and greater than 0 without it, C-G becomes a two-backbone result and the paper's best claim. Prioritise this over A+0.

**O5. "Privileged inputs and baseline fairness. Rendered depth, calibration and simulator masks; a parsed-scene template resolver; a privileged scene-2 channel; a ReGuide reimplementation without its escort; no comparison on CRAFT / ACT-VLA / AC-VLA splits."**
- Minimum fix:
  - (a) Rewrite C3 to "adds no task information beyond a calibrated RGB-D sensor", and say that ReGuide gets the same perception.
  - (b) State explicitly that the F7 margin over ReGuide comes from a template parser that ReGuide could adopt, or run ReGuide + resolver as an arm (eval-only, about 1 job).
  - (c) If the ACT-VLA / AC-VLA Goal-OOD re-pairing list or the CRAFT LIBERO-Goal list can be recovered, run frozen pi0.5, ReGuide and W5 on it (eval-only) and report against the published numbers. Otherwise state plainly that numbers are not comparable across splits.
  - (d) Implement or justify omitting ReGuide's escort.

Also cut, in the same rewrite (0 GPU-h):
- The predictor (§7): appendix, one paragraph.
- The W1-W6 chronology (§6.3): one table and the final design.
- The SR pointers and capitalised verdict words in the main text.

The draft is about three times the 8-page budget.

---

## 5. Scores

| | Score (ICML 1-10) | Reason |
|---|---|---|
| As it stands | **3 (reject)** | Known diagnosis. The repair loses to ReGuide, which owns the where/how framing. The localisation is a by-construction lesion of hand-built modules, and the in-network patching is null. The predictor failed registration. One seed, one simulator. Notebook presentation. Rigour and honesty are real strengths and do not rescue it. |
| After O1, O2(a), O3, O5(a, b) and the cuts | **5 (borderline)** | A coherent analysis paper with a clean matched-data ladder, a new moved-plate finding, the ReGuide convergence as corroboration, and the BC-shortcut grid. Reviewers will still cite scope (LIBERO, pick-place). |
| Plus O4(ii) replicating on pi0, and O2(b) giving a selective, registered KV-14 result | **6 (weak accept)** | The two-backbone BC-shortcut result is a general ML claim with a causal measure. That is the version that can survive ICML. Without it, aim for CoRL 2027. |

---

## Sources (web, read for this review)
- ReGuide: https://arxiv.org/abs/2609.38616
- Encoded but Not in Control: https://arxiv.org/abs/2610.06235
- Wulff & Cangelosi, language-grounding patching in pi0.5 / GR00T N1.7: https://arxiv.org/abs/2610.10178
- Grant et al.: https://arxiv.org/abs/2603.19233 ; ICLR 2026 page https://iclr.cc/virtual/2026/10013199 (search hit reporting 0% cross-task transfer; verify)
- Do VLAs Really Follow Instructions? (How-level): https://arxiv.org/abs/2610.21251
- w2VLA: https://arxiv.org/pdf/2606.21496 , https://tsagkas.github.io/w2vla/index.html
- Robust Skills, Brittle Grounding: https://arxiv.org/html/2602.24143
- CRAFT: https://arxiv.org/pdf/2610.00524 ; ACT-VLA: https://arxiv.org/pdf/2607.00351 ; Primitive subspace: https://arxiv.org/html/2605.30695 ; LIBERO-PRO: https://arxiv.org/html/2510.03827v2
- Nouns, Not Modifiers (ICML 2026): https://icml.cc/virtual/2026/69337
- RoboMME (ICML 2026): https://icml.cc/virtual/2026/poster/65933 ; LIBERO-Mem (AAAI 2026): https://ojs.aaai.org/index.php/AAAI/article/download/37337/41299 ; LIBERO-Occ: https://arxiv.org/html/2606.10862v2
- Residual / anchor refinement of VLA actions (adjacent to the servo): https://arxiv.org/html/2604.21391 , https://arxiv.org/html/2604.17787
- Authors' notes checked: reports/novelty_raa.md, reports/novelty_check_failure_atlas.md, reports/novelty_predictor.md, "reports/CoRL 2026 VLA literature review.md". Their component-level verdicts hold up. They miss 2610.10178 and the ReGuide-title overlap with §6.3, and they treat 2610.06235 as motivation when it overlaps the routing claim.

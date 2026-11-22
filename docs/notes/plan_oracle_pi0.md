# Plan: the oracle arm (Exp 1) and the pi0 replication (Exp 2) for objection O4

Written 2026-11-22.  Design only: nothing here was run, no file other than this one was written.  For the executor agent.
Inputs read: docs/notes/novelty_review_adv.md (O4), docs/notes/stateproj_diag.md, paper/DRAFT.md 6.3-6.4, docs/notes/raaf_sel.md, docs/notes/raafp_diag.md,
lifelong_ood/policy/l90c_config.py, raac_model.py, raas_model.py, backbone.py, experiments/c_train_pi05.py, raac_export.py, a2_probe.py, a2_probe_f.py, a2_read.py,
a2f_capture.py / .sh, raa.py (serve), raafp_gen.py, raafp_rv.py, experiments/prereg/raafp0.py / raafp0_pred.txt, and openpi's models/pi0.py, pi0_config.py,
training/config.py (pi0_libero, pi05_libero, LeRobotLiberoDataConfig), transforms.py, examples/convert_jax_model_to_pytorch.py.

## 0. The objection and what each experiment answers

O4: "the action expert ignores the where-token" may be an artefact.  The token enters pi0.5 through a NEW random-init `state_proj` (Linear 32 -> 1024).  The layer barely
moves from init in the no-dose arms (1-3 %) and moves 4-9 % in A, A+0 and A+ (stateproj_diag).  Two versions of the artefact reading need separate answers:

| version | claim | answered by |
|---|---|---|
| capacity / optimisation | the new layer cannot be learned in 6,001 steps at lr 5e-5, so no data could make the expert read it | Exp 1 (oracle): train or find a model for which the token is the only source of the target and measure whether the expert follows it |
| inductive bias | the layer could be learned, but a random-init pathway is disfavoured, so a network with a trained state pathway would read the token from ordinary data | Exp 2 (pi0): pi0_libero has a trained state token that its expert necessarily reads; replicate RAA-sL vs A0 on it |

The paper claim under test (DRAFT 6.4): "the network reads the where-input only when no other path supplies the where-correction", and the framing built on it, "the
where must be owned outside the learned network".  Section 7 says which results would refute that framing.

## 1. Facts this design rests on (checked in code, 2026-11-22)

1. **A+0 is already a partial oracle arm.**  raafp_rv.py renders the planner table-spot demos with the target = main_table (a fixture: nothing is painted for the target),
   and the held prompt is the constant "target: the table".  On the 5,306 held planner frames (8.6 % of 61,649), the views and prompt carry no information about the spot.
   The token is the only source, with one exception: each of the 100 layouts has ONE spot, so a network can memorise layout -> spot from the images instead of reading the token.
   A+0's existing checkpoint (pytorch/pi05_g8v_raac_lp_ae_tl6000) has never been measured on table frames; its .124 share is on N + K frames, where the prefix also carries the target.
2. **The probe's effect measure is a projection** (a2_read.py): s_cf(chunk) = d5 . u_cf, with d5 = the sum of the first 5 xyz deltas of the unnormalised chunk and u_cf the
   unit vector from the end effector to the counterfactual entity.  share = mean E(in.tok) / mean T.  A change that does not point toward the counterfactual target projects to about 0.
3. **pi0's state token is trained and read.**  openpi Pi0 with pi05=False embeds obs.state through its own `state_proj` into suffix position 0.  The action tokens attend to it and the prefix does not.
   This is the same topology as raac_model.Pi0C.  pi0_libero is fine-tuned with `extra_delta_transform=True` (labels a[0:6] - s[0:6]), so its expert MUST read the state token to produce its labels.
   The cache holds `openpi-assets/checkpoints/pi0_libero/{params,assets}` and `pi0_libero_pytorch`, but no pi0_base.  pi0_libero's norm stats cover the 8-dim LIBERO proprio (eef xyz, axis-angle, gripper) padded to 32.
4. **Pi0C cannot be used with pi05=False.**  Pi0C.__init__ assigns a NEW random `self.state_proj` over the native one, and its embed_suffix prepends a second token to the
   native one.  The pi0 arms need a new servo subclass of openpi's plain Pi0 (section 4.2).
5. **pi0 uses z-score normalisation.**  openpi sets `use_quantile_norm = model_type != PI0`.  raas_model.servo_norm_{jax,torch} invert the QUANTILE normalisation, and
   backbone.PolicyBackbone.sample and a2_probe.Patcher.run hard-import servo_norm_torch.  A z-scored pi0 arm with the servo kept would therefore get a wrong servo at
   training, serving and probing.  Decision: force `use_quantile_norm=True` for the pi0 arms (section 4.1).
6. **openpi's converter detects pi05 by the substring "pi05" in the checkpoint path** (convert_jax_model_to_pytorch.py lines 293, 348 and 372; the projection key list is chosen from model_config.pi05).
   pi0 config names and exp names must not contain "pi05".  The converter already handles pi0's state_proj (it is in the pi0 key list), so raac_export.py's extra-tensor step is not needed.
7. **Every probe and serving path ports to pi0 unchanged.**  PolicyBackbone (kind "pi0", 3 images x 256 tokens + prompt, denoise_step -> embed_suffix), Patcher (18 VLM and 18
   expert layers, 8 heads x 256; pi0's gemma_300m expert has the same shape), a2_probe_f (`--config` / `--ckpt`, any label) and raa.py serve (`--arm raac --config NAME`)
   contain no pi05-only code on these paths.  pi0_libero PyTorch already served in this repo (experiments/prereg/g3_crossmodel_pi0.py).
8. Capture rollouts already record per-task success (`<POLICY>_nk_<tid>.json` -> success_rate, 20 trials per task).  That gives a fit gate at no extra cost.
9. The running seed-2 RAA-sL chain imports c_train_pi05.py and l90c_config.py at every 4 h restart.  c_train_pi05.py is NOT edited (a new wrapper is used instead).
   l90c_config.py gets add-only edits, verified at once by a CPU import job (section 6).

## 2. Experiment 1: the oracle arm

### 2.1 Hypothesis and claim
H1: the random-init state_proj is learnable within the shared recipe (6,001 steps, lr 5e-5, frozen VLM, batch 32).  Where the token is the only source of the target, the trained expert follows it.
Under H1 the near-zero shares on N + K are a choice of the network (the BC shortcut), not an inability.
Paper claim tested: DRAFT 6.4's shortcut reading, and the stateproj_diag caveat "this diagnostic cannot say how much the share would rise ...".

### 2.2 Design choices, with reasons
- **Data: planner table-spot episodes whose spot only the token determines, mixed with the base data, servo off.**  Rejected alternatives:
  (a) Instructions that do not name an entity target.  The trained object-target pairing still supplies the target (this is the shortcut under study), and the target stays visible, so the token is not the only source.
  (b) Oracle-only training (no base episodes).  K competence collapses, so the N + K probe and the capture gate become meaningless, and the arm is no longer a one-variable change from A+0.
  (c) Servo kept.  A+ shows that the servo zeroes the share (-.007), and with a servo the token is not the only route to the spot.
- **Decorrelate spot from layout (the new ingredient over A+0).**  Each of A+0's 100 layouts gets a SECOND planner demo to a different spot, at least 15 cm from the first.
  Within a pair, the images up to the lift are the same layout, so layout memorisation can at best predict the midpoint of the pair.  The held oracle frames double (about 10.6 k, about 15 % of frames).
  The ORC - A+0 difference therefore isolates "paired spots, 2x the oracle frames" with everything else equal.
- **One arm (ORC), gated.**  A+0 is measured first on held-out oracle frames (Stage 1a, about 2 GPU-h).  If A+0 already follows the token (phi lower bound >= .70, section 2.4), the oracle question is answered by an existing checkpoint.
  ORC is then not trained, and its GPU-h go to the pi0 dose arm (section 3.6).  Otherwise ORC is trained.
- **No state_proj learning-rate multiplier and no warm start in this round.**  An lr multiplier changes the recipe shared by every arm, and the question is whether THAT recipe can learn the layer.
  It is also not cheap (optax multi_transform in openpi's optimizer path).  It is the follow-up only if ORC fails (NOT LEARNED); then it separates optimisation from capacity.
  A warm start has no principled source.  pi0's trained state_proj projects into a different expert's residual basis (no adaRMS, different weights).

### 2.3 Arms and config diffs
| arm | config (new names in l90c_config.py) | diff |
|---|---|---|
| ORC (conditional) | `pi05_g8v_raac_lo_ae` | = `pi05_g8v_raac_lp_ae` (A+0) exactly: RS.Pi0ConfigS(pi05=True, servo_fields(k=0.0)), RaacLoader(LIBERO_PARAMS), RAA-c norm stats / assets (`AE[...] = ("pi05_g8v_raac", "lol/libero_g8v_raac")`), frozen VLM, no EMA, 6,001 steps, batch 32, lr 5e-5.  ONLY the data repo changes: `lol/libero_g8v_raacp` -> `lol/libero_g8v_raacpo` |
| readout set (no training) | existing exports A+0 (raac_lp_ae_tl6000), A+ (raas_lp_ae_main6000), A0 (raac_l_ae_tl6000), A (raac_lf_ae_tl6000), RAA-sL (raas_l_ae_main6000) | none |

### 2.4 Registered measure on oracle frames: the follow share phi
Why the standard share is degenerate here: on an oracle frame only the token can carry a counterfactual target.  Images and prompt are equal by construction, so T = E(in.tok) and the share is 1 for every model.
The measure must therefore say how far the expert follows the token relative to a reference.  The reference is an ideal follower that keeps its clean speed.
It uses the probe's own projection (fact 2), restricted to xy because both spots sit at the object's resting height.

For an oracle observation i (a frame of test demo i, commanded spot s_i) with paired spot s_j (same layout):
- clean row: the observation as rendered (token tau_i = spot s_i at the object's resting height - hand - in-hand offset, as raafp_rv's planner_rel / sc_rv.held_target_rel).
- cf row (in.tok): the token tau_i + (s_j - s_i, 0) in raw metres, i.e. exactly the token the front end would give for spot s_j.  Images, prompt, noise seed and (for servo models) the servo input stay clean.
- d5 = the xy sum of the first 5 unnormalised chunk rows; u_j = unit_xy(s_j - p_obj), u_i = unit_xy(s_i - p_obj) (p_obj = the held object's sim position, measurement only).
- num_i = d5(cf) . u_j - d5(clean) . u_j;   den_i = |d5(clean)| - d5(clean) . u_j   (the gain an ideal follower would get by rotating its clean steer onto u_j at the same speed).
- **phi = sum_i num_i / sum_i den_i** (ratio of sums).  95 % CI by clustered bootstrap, clusters = layouts, B = 2000, numpy default_rng(0).  phi = 0 means the token is ignored; phi = 1 means full following.

Observation set (model-independent, fixed by the demo labels before any model is run): held frames of a test demo where (i) the object is >= 3 cm above its resting
height, (ii) it is <= 6 cm (xy) from its start, so the images before transit are near-identical within a pair, (iii) the demo's own label has |d5_xy| >= 1.0 action units, which excludes the pure-lift frames, and (iv) angle(u_i, u_j) >= 30 deg.
Up to 3 frames per demo, spaced >= 3 frames apart.  Both directions of a pair are used.  Target: 20 layouts x 2 objects x 2 spots x 3 frames = 240 observations, 40 clusters.

Also registered, reported for every model:
- c_clean = mean over observations of d5(clean) . u_i / |d5(clean)|.  It asks whether the clean prediction points at the commanded spot.
- abl.tok (the token set to the unrouted zero convention): phi_zero and c_zero.
- sigma_demo (descriptive, secondary).  This is the same numerator, divided by the demonstrated difference sum(d5_label_j . u_j - d5_label_i . u_j) at the first frame that meets the observation rule in both demos of the pair.
- For servo-kept models (A+, RAA-sL): phi_net (servo clean, the registered number) and phi_total (token and servo both from cf, descriptive).
- UNTESTABLE rule: if fewer than 120 observations meet the rule, or sum den <= 0 for a model, that model's phi is UNTESTABLE.

Decision labels (the paper's bars): **LEARNED** iff phi CI lower >= .70; **NOT LEARNED** iff phi CI upper < .30; **PARTIAL** otherwise.
Gate for training ORC: ORC is trained iff A+0 is not LEARNED.
Contrasts (cluster-paired on the same observations): phi(ORC) - phi(A+0); phi(A+0) - phi(A0); phi_net(A+) - phi(A+0) (the servo effect on oracle frames).

Also registered for ORC only if ORC's phi point >= .30: the standard N + K probe (a2_probe_f, rows full), so that phi and the N + K share come from the same network.
If ORC's N + K share CI lower >= .70, ORC dev rollouts (Phase A N/K, raaf_roll copy) are added, because only then can behaviour decide section 7's data route.

### 2.5 What the outcomes mean
| oracle model (A+0, or ORC if trained) | its N + K share | reading for the paper |
|---|---|---|
| phi LEARNED | CI upper < .30 (A+0: .124 [.089, .164] already) | **Within-model dissociation.**  The same layer and network follow the token where only the token informs, and ignore it where the prefix suffices.  The capacity / optimisation artefact is answered, and C-G's shortcut reading stands.  Strongest outcome for the paper |
| phi LEARNED | CI lower >= .70 | Once some frames force the token, the network takes it up everywhere.  The "data can replace the module" route opens; go to section 7 |
| phi PARTIAL | any | The layer is learnable in part within the recipe.  Report as a limit; the lr-multiplier follow-up is the next test |
| phi NOT LEARNED (ORC) | not probed | Not learned even when needed.  The artefact (optimisation / capacity) reading gains weight; C-G must be restricted to "through a newly added pathway at this recipe", and Exp 2 carries the claim |

### 2.6 Data build (Stage 1a test set always; Stage 1b ORC training pairs, built in parallel and used only if the gate opens)
All generation and conversion is CPU in the LIBERO / openpi venvs via sbatch.  Rendering needs a GPU (EGL).
1. `experiments/orc_gen.py` (new; imports experiments/raafp_gen.py unchanged and monkeypatches its module-level `spot`).
   - **Seed reproduction**: attempt(...) derives seed = BASE + 1000*oix + k + 50e6*j.  Reproduce an existing demo's layout by setting raafp_gen.BASE and passing
     k = seed - BASE - 1000*oix, j = 0, so the derived seed equals the stored `seed` of the existing demo npz.
   - **Paired spot**: the patched `spot` draws from its own stream numpy default_rng([seed, 7, s, r]) (s = spot index, r = retry).  It uses the h_j free mask (raafp_gen.free_mask)
     and rejects candidates < 15 cm (xy) from every spot already kept for this layout.  It consumes r extra normals from the attempt's rng, so each retry gets a fresh jitter.  At most 6 retries per (layout, spot).
   - Mode `pairs`: for each of the 100 kept A+ demos (bowl 50, cream cheese 50; /coc/flash10/ksingara3/.claude_tmp/raaf/gen_aplus/<obj>/demo_<k>.npz), add 1 new spot.
     Out: /coc/flash10/ksingara3/.claude_tmp/raaf/gen_orc_pairs/<obj>/demo_<k>.npz (the same per-step state format; `pair_of` = the source demo file).
   - Mode `test`: fresh layouts with BASE = 96,000,000.  First check that no summary in results/flawed/raaf/planner/gen_*.json uses 96M (head / grep on those named files only).
     Keep a layout only when BOTH spots yield kept demos; continue until 20 complete pairs per object (cap 250 attempts per object).
     Out: /coc/flash10/ksingara3/.claude_tmp/raaf/gen_orc_test/<obj>/demo_<k>.npz plus a pairs.json.
   - Cost (raafp_diag 5, per kept demo 17 s bowl / 33 s cheese, retries included): pairs about 0.8 CPU-h; test about 0.6 CPU-h.  One sbatch each (-c 4, <= 2 h, CPU).
2. Render with the existing CLI: `raafp_rv.py render --gen <gen dir> --rvp <new rvp dir> --keep 0`.  It runs replay check k3, and the gen dir basename keys the replay_check json, so the names must be distinct.
   - pairs -> /coc/flash10/ksingara3/robot/data/fa1/rvp_orc_new ; test -> /coc/flash10/ksingara3/robot/data/fa1/rvp_orc_test.  Never write into the existing rvp/.
   - Then build /coc/flash10/ksingara3/robot/data/fa1/rvp_orc/{K1,K6}/ with symlinks: demo_00-49 / chan_00-49 -> the existing rvp files, and demo_50-99 / chan_50-99 -> rvp_orc_new's 00-49.
     render indexes j from 0 per gen dir, hence the renumbering.
3. ORC dataset: `raafp_rv.py convert --arm raac --suffix o --rvp .../rvp_orc` -> lol/libero_g8v_raacpo (400 base + 200 planner episodes), then `check --arm raac --suffix o`.
   check verifies base episodes identical, role 2 exactly on held frames, finite, |a| <= 1.  The base loop's `nd = ... or 50` refers to base demos per task and is unaffected.
4. Test observations: `experiments/orc_obs.py` (new, openpi venv, CPU).  It mirrors raafp_rv.convert's planner loop (planner_rel, RV.frames_of with roles (obj, "main_table"), RV.chan_state) on rvp_orc_test.
   It does not write a LeRobot set; it writes one npz per observation in a2's capture format (image, wrist, state [raw 4-dim], prompt, seed = a fixed hash of the id, actions = the demo's label chunk).
   It adds `state_cf`, spot_i, spot_j, p_obj, u_i, u_j, layout id (cluster), object, and the label d5 of both demos.  Out: results/flawed/orc/obs_test/ plus selection.json (counts per rule).

### 2.7 Readout implementation
- `experiments/orc_probe.py` (new, GPU).  Imports a2_probe unchanged (Patcher) and lifelong_ood.policy.backbone.PolicyBackbone.  For each observation it runs one batch of rows:
  ref (clean), tok_cf (pre = clean, tok = state_cf, srv = clean), abl.tok (tok = zeros), and for servo models chan_cf (tok = srv = state_cf).
  It saves the unnormalised chunks per observation (skip-if-exists, so a requeued job resumes).  Args: --label --config --ckpt --obs DIR --out DIR.  Runs on A40, about 6-12 min per model.
- `experiments/orc_read.py` (new, CPU).  Computes phi, c_clean, phi_zero, sigma_demo, phi_net/phi_total, the contrasts and the labels of 2.4.  Clustered bootstrap B = 2000, rng 0.
  It writes results/flawed/orc/orc_read_<label>.json and orc_contrast_<a>_<b>.json with open(..., "x") (it refuses to overwrite).  It also has a `selftest` on synthetic chunks: a perfect follower gives 1, an ignorer 0, half-rotation about .5.

## 3. Experiment 2: the second backbone (pi0)

### 3.1 Hypothesis and claim
H2: the no-dose null is not specific to a new pathway.  With pi0's trained, already-read state token, the expert still leaves the where-token unread on base data with the servo kept (pi0-sL, mirrors RAA-sL -.007) and with it off (pi0-A0, mirrors A0 -.001).
Paper claim tested: C-G / DRAFT 6.4 on a second backbone (review O4(ii); the "6/10 only if it replicates on a second backbone" condition).

### 3.2 How the where-token enters pi0 (decision: trained columns, proprio replaced)
**Chosen: state = [dx, dy, dz, role] in dims 0-3 (zero-padded to 32), i.e. exactly the RAA-c state, on the same dataset lol/libero_g8v_raac with the same norm stats.**
The token enters through pi0_libero's trained state_proj.  Columns 0-2 were trained on eef xyz, column 3 on axis-angle x, and the downstream attention to the token is trained and in use (fact 3).
Reasons:
- It is the trained pathway the objection asks for, and the normalised vector reaching state_proj is identical to RAA-sL's and A0's (same data, same stats, quantile norm forced, see 3.3).
  The two backbones then differ in the pathway, not the input.
- The servo (raas_model, which reads state[:4]), the server (raa.py --arm raac writes the 4-dim chan_state), the capture, the probe and the data repo all port unchanged.  No new dataset and no new norm stats are needed.
- Dropping proprio matches the pi0.5 arms, which read no proprio either.
**Rejected: proprio kept in dims 0-7 and the token appended in pad dims 8-11.**  pi0_libero's columns 8-31 got zero input during its LIBERO fine-tune, so they were never trained on LIBERO (they keep pi0_base's cross-embodiment values or init; not verified).
That rebuilds the untrained-column confound.  It would also need a new 12-dim dataset and norm stats, and a servo that reads dims 8-11.

### 3.3 Arms and config diffs (names must NOT contain "pi05", fact 6)
| arm | new config | = | diff |
|---|---|---|---|
| pi0-sL (servo kept) | `pi0_g8v_raas_l_ae` | `pi05_g8v_raas_l_ae` (RAA-sL) | model RS.Pi0ConfigS(pi05=True) -> **Pi0ConfigZS(pi05=False, action_horizon=10, **RS.servo_fields())** (native trained state token, same servo constants K 2.851, max .9375, the same RAA-c quantile stats); weights RaacLoader(pi05_libero) -> **CheckpointWeightLoader(pi0_libero/params)** (nothing missing); data = the same repo lol/libero_g8v_raac and assets, with **use_quantile_norm forced True** (pi0's default is z-score) and extra_delta_transform False (inherited from pi05_l90c, assert it); max_token_len 48 (pi0 default) |
| pi0-A0 (servo off) | `pi0_g8v_raac_l_ae` | pi0-sL | ONLY `RS.servo_fields(k=0.0)` (mirrors A0 vs RAA-sL) |
| pi0-A (conditional, 3.6) | `pi0_g8v_raac_lf_ae` | pi0-A0 | ONLY data repo `lol/libero_g8v_raacf` (Route A's relabel, existing) |
Unchanged from the pi0.5 arms: frozen VLM regex (img + llm except `_1`), so the expert, state_proj, action_in/out_proj and action_time_mlp_* train; no EMA; batch 32; 6,001 steps; cosine lr 5e-5 with 1k warmup; AdamW clip 1; seed 42; save 250 / keep 1000.

Why quantile: (i) the servo, PolicyBackbone.sample and Patcher.run all use the quantile servo_norm_torch (fact 5), so z-score would need edits to existing files.
(ii) Inputs become identical to the pi0.5 arms.  (iii) The cost is small: quantile [-1, 1] and z-score have similar scales, and the label semantics change anyway (iv).
(iv) Deviations from pi0_libero's own recipe, stated in the paper: labels are raw LIBERO actions, not a - s; horizon 10, not 50 (it matches every pi0.5 arm and the 5-step replan).
Because of (iv), pi0's step-0 loss is not comparable to pi0.5's .047-.056, and the smoke kill rule below is pi0-specific.

### 3.4 Implementation steps (additive only)
1. `lifelong_ood/policy/raas0_model.py` (new).
   - Pi0ZS(openpi.models.pi0.Pi0): a copy of raas_model.Pi0S's three methods.  It reuses RS.servo_norm_jax, keeps servo_cfg = config, and with servo_k = 0 is openpi's Pi0 exactly.
   - Pi0ConfigZS(openpi Pi0Config): the same six servo fields as Pi0ConfigS; create -> Pi0ZS; load_pytorch -> PI0PytorchZS.
   - PI0PytorchZS(openpi PI0Pytorch): sample_actions adds RS.servo_norm_torch, a copy of PI0PytorchS.
   - No new parameters, and no import of raac_model.
2. `lifelong_ood/policy/l90c_config.py`, add-only.
   - PI0_LIBERO_PARAMS = f"{CACHE}/pi0_libero/params"; RAAFO_REPO = "lol/libero_g8v_raacpo".
   - The 4 names appended to NAMES, with AE entries ("pi05_g8v_raac", "lol/libero_g8v_raac") (the assets dir name contains "pi05", which is harmless: the converter sniffs the checkpoint path only).
   - A helper `_quantile_data(C, data)`: a LeRobotLiberoDataConfig subclass whose create() returns dataclasses.replace(dc, use_quantile_norm=True), in the pattern of _crop_data.
   - Four new `if name == ...` blocks inside the AE branch, before its final generic return.  Imports stay lazy inside each block, as in the existing blocks.
3. `experiments/c_train_x.py` (new).  It imports c_train_pi05 as CT and adds `--name FULLNAME` (one of the new names).  It sets CT.run_name = lambda a: a.name and calls CT.train(a), which reads run_name at call time.
   Export: Pi0ConfigC-family models -> experiments/raac_export.py (ORC); Pi0ConfigZS -> experiments/pi0z_export.py.  It logs to the same results/l90c/train_log.jsonl.  c_train_pi05.py is NOT edited (fact 9).
4. `experiments/pi0z_export.py` (new, CPU).
   - It runs LC.register(), asserts "pi05" not in the src path, and calls CV.convert_pi0_checkpoint(src, "bfloat16", dst, LC.train_config(name).model).
   - It asserts "state_proj.weight" is in model.safetensors and |torch - jax| is within bf16 rounding (raac_export's check), then copies the assets.  dst = pytorch/<name>_<tag>; check with ls that it does not exist.
5. `experiments/pz_check.py` (new, a copy of raas_check.py items 1-4 retargeted).  It checks:
   - the Pi0ZS(k=0) loss and samples equal openpi Pi0's on one batch;
   - with the real K, loss(a) = Pi0.loss(a - servo);
   - the parameter tree equals pi0_libero's (no missing or extra keys), and state_proj at step 0 equals pi0_libero's kernel (norm and max |diff| = 0);
   - the data config has use_quantile_norm True and no DeltaActions;
   - the torch servo (item 4) on the exported smoke checkpoint.
6. `experiments/a2fx_capture.sh` (new, a copy of a2f_capture.sh): a generic `*)` case (SERVE=raa.py, CONFIG and CKPT required, A2F_RAA=raa) so labels pzsL / pzA0 / pzA / orc can be used.
   The probe (`a2_probe_f.py --model <label> --config --ckpt`) and the reader (`raaf_probe_read.py read --route <label>`) already accept any label.
7. `experiments/stateproj_diag_pz.py` (new, a copy of stateproj_diag.py, CPU).  pi0's init IS saved (pi0_libero/params), so it reports the exact change of state_proj from init: per live column, Frobenius, and the pad columns 4-31 as a control.  Descriptive.
8. Prereg files (section 5) and job-script copies in /coc/flash10/ksingara3/.claude_tmp/pp/: pz_train.sh / pz_chain.sh copied from raafp0_train.sh / raafp0_chain.sh,
   calling `experiments/c_train_x.py --name~<cfg>~--exp~tl...`; the same pattern for orc_*; cfgcheck.sh (CPU import check).

### 3.5 Pipeline after training (per arm, mirrors raaf_sel / the runbook)
- Export (CPU): c_train_x.py --name <cfg> --exp tl --export --step 6000 --tag tl6000 -> pytorch/<cfg>_tl6000.
- Serving smoke (a2fx_capture.sh PHASE=smoke, OUTD under .claude_tmp, one K and one N task, 2 trials).  It must show role-1 and role-2 state tokens in the server log (kill k1 of the raaf rows) and finite chunks.
- Captures (A40, the SR 83 design: K tids 0+2+6+10, N tids 16+22+24+26, trials 4-23, 3rd held query): results/flawed/pz/a2/cap_<label>, about 0.5 GPU-h per K / N job.
- Probe: a2_probe_f.py --rows full, K and N jobs (about 0.6 GPU-h per arm), then raaf_probe_read.py read --route <label> --ref-dirs results/flawed/a2/probe_raasl_{N,K}.
- Contrasts: experiments/raaf_probe_contrast.py (cluster-paired; same dev states for every model): pzA0 - pzsL, pzA0 - A0 (pi0.5), pzsL - RAA-sL (pi0.5).
- Fit gate and behaviour from the capture json files (success_rate, 20 trials per task): K pool (K1, K2, K4, K6) and N pool (N1, N4, N5, N6).
  The same numbers for A0 and RAA-sL come from their existing captures (cap_A0; SR 83's raasl captures).  Descriptive, with a paired-by-state bootstrap.
- stateproj_diag_pz.py on both arms.

### 3.6 Is a pi0 small-relabel arm (mirror of A) needed?
**Not for O4.**  The objection is about the NULL: a near-zero share at no dose.  pi0-sL and pi0-A0 test exactly that, with a trained pathway.
The positive half of the grid (a dose raises the share, A .274) is a separate claim.  pi0-A would make it two-backbone, and it adds no evidence on the artefact question.
Run pi0-A only if (a) the ORC gate closed (A+0 LEARNED), which frees about 11 GPU-h, and (b) R1 below holds (if pi0 already reads the token at no dose, a dose arm says nothing about the shortcut).
Its data already exist (lol/libero_g8v_raacf), so it costs training plus the probe.

### 3.7 Registered quantities and decision rules (Exp 2)
- S_kept = pooled N + K in.tok share of pzsL; S_off = the same for pzA0 (95 % CI, clustered, B = 2000, rng 0).  The probe pass bar (.70) and failure line (.30) are unchanged.
- Gate (per arm): K-pool capture success >= .70 AND the T CI lower bound > 0.  An arm that fails is UNTESTABLE.  The pi0 recipe then failed, and no reading claim is made from that arm.
- The prefix half of the old pass rule is NOT used (raaf_sel documented its by-construction defect).  in.prefix is reported only.
- **R1 SHORTCUT REPLICATES** iff both gates hold, S_kept CI upper < .30 and S_off CI upper < .30.
- **R2 NO-DOSE READ (artefact for the null)** iff S_off CI lower >= .30.  The strong form is S_off CI lower >= .70 (the pass bar).
- **R3 SERVO SUPPRESSES on pi0** iff (S_off - S_kept) CI lower > 0.
- **R4 SERVO FAILS TO SUPPRESS** iff S_kept CI lower >= .30.
- Reported: in.servo share of pzsL (RAA-sL .878), abl.tok drops, top1 / PRn, cross-backbone contrasts, state_proj change from init, and K / N capture success against A0 / RAA-sL.

| pi0 result | reading |
|---|---|
| R1 | The no-dose null holds through a trained, already-read state pathway.  The inductive-bias artefact fails for the null, and C-G's necessity half is a two-backbone result |
| R2 (and R1 false) | pi0 reads the token from ordinary data when nothing else supplies the correction.  The pi0.5 A0 null was pathway-specific; C-G becomes "a new input pathway is not taken up", and the "only when data demand it" half fails on pi0 |
| R3 with R2 | Servo suppression replicates (the servo half of the grid), and the dose half does not |
| R4 | The servo does not suppress a trained pathway.  The servo-shortcut reading fails on pi0 |
| gate fails | UNTESTABLE.  Report the recipe failure and the step-0 / final losses |

## 4. Pre-registration predictions (format of experiments/prereg/raafp0_pred.txt; probabilities are the designer's, for the orchestrator to adjust before writing)

### 4.1 experiments/prereg/orc_pred.txt
```
ORC (docs/notes/plan_oracle_pi0.md 2): oracle readout of existing checkpoints on held-out table-spot frames, and the conditional oracle arm ORC = pi05_g8v_raac_lo_ae
(= A+0 exactly; ONLY data lol/libero_g8v_raacp -> lol/libero_g8v_raacpo: each of A+0's 100 layouts gets a second planner demo to a spot >= 15 cm from the first).
Principle: if the new state_proj can be learned in the shared recipe, a servo-free expert follows the token where the token is the only source of the target.
Measure: follow share phi (plan 2.4) on 20 held-out layouts x 2 objects x 2 paired spots (BASE 96,000,000), clusters = layouts, B = 2000, rng 0.
Labels: LEARNED phi CI lower >= .70; NOT LEARNED CI upper < .30; PARTIAL otherwise.  ORC is trained iff A+0 is not LEARNED.
STATUS: A+0's, A+'s, A's, A0's and RAA-sL's N + K shares were seen (raaf_sel.md); no model has been run on table-spot frames before this row.
Predictions:
 (o1) A0 phi CI upper < .30: 90%.     (o2) RAA-sL phi_net CI upper < .30: 90%.     (o3) A+ phi_net CI upper < .30: 70%.
 (o4) A+0 phi point .50 (plausible range .15-.85); A+0 LEARNED 30%; A+0 NOT LEARNED 25%.
 (o5) A+0 c_clean > A0 c_clean (CI lower of the difference > 0): 75%.
 (o6) ORC (if trained) phi point .75; ORC LEARNED 55%; ORC NOT LEARNED 10%.
 (o7) phi(ORC) - phi(A+0) CI lower > 0: 65%.
 (o8) ORC N + K in.tok share (probed iff ORC phi point >= .30): point .20; CI upper < .30: 55%; CI lower >= .70: 7%.
 (o9) Within-model dissociation (the oracle model LEARNED and its N + K share CI upper < .30), for A+0 or ORC: 45%.
Kill: (k1) orc_obs selection < 120 observations (UNTESTABLE); (k2) ORC training loss NaN or step-0 loss > 2 x RAA-sL's .0556; (k3) orc_gen pairs yield < 70 of 100 new demos per object
after 6 retries (then ORC is built from the pairs that exist and the deviation is noted, never by relaxing the 15 cm rule after seeing data).
```

### 4.2 experiments/prereg/pz_pred.txt
```
PZ (docs/notes/plan_oracle_pi0.md 3): RAA-sL vs A0 on openpi pi0 (Pi0Config pi05=False, the native trained state token of pi0_libero).
pi0-sL = pi0_g8v_raas_l_ae (servo kept), pi0-A0 = pi0_g8v_raac_l_ae (servo off); the token [dx, dy, dz, role] replaces proprio in dims 0-3 (trained columns);
data lol/libero_g8v_raac, RAA-c norm stats, quantile norm forced, raw (non-delta) actions, horizon 10, weights pi0_libero, frozen VLM, 6,001 steps, batch 32, lr 5e-5.
Principle (C-G): a BC expert does not take up an exact where-signal when the prompt and views already determine the target; the pi0.5 nulls (RAA-sL -.007, A0 -.001) are not an artefact of a new random-init layer.
Measure: pooled N + K in.tok share (a2_probe_f rows full, SR 83 design), B = 2000, rng 0, clustered; gate K capture success >= .70 and T CI lower > 0.
Rules: R1 both CI upper < .30; R2 S_off CI lower >= .30; R3 (S_off - S_kept) CI lower > 0; R4 S_kept CI lower >= .30.  The prefix half of the old rule is not used.
Predictions:
 (z1) gates: pi0-sL passes 85%, pi0-A0 passes 75%.
 (z2) S_kept point .00 (range -.05 to .20); CI upper < .30: 80%.
 (z3) S_off point .08 (range -.05 to .45); CI upper < .30: 60%; CI lower >= .30: 20%; CI lower >= .70: 5%.
 (z4) R1: 50%.   (z5) R3: 30%.   (z6) R4: 10%.
 (z7) pi0-sL in.servo share >= .70: 75%.
 (z8) pi0-sL K capture success within +-.10 of RAA-sL's on the same states: 55%; pi0-A0 N capture success below pi0-sL's: 70%.
 (z9) state_proj relative change from init (live columns, Frobenius) larger in pi0-A0 than in pi0-sL: 55% (descriptive).
Kill: (k1) the smoke shows no role token in the server log, or the loss is NaN; (k2) pz_check fails (k = 0 not equal to Pi0, or state_proj not loaded from pi0_libero);
(k3) step-0 loss > 10 x pi0.5 RAA-sL's .0556 (pi0's labels change from delta to raw, so the 2x rule of the pi0.5 rows does not apply).
```

### 4.3 Ledger rows
Two scripts copied from experiments/prereg/raafp0.py:
- experiments/prereg/orc.py (tag `orc`): written before orc_probe runs on any checkpoint.  It hashes this plan, orc_pred.txt, orc_gen.py, orc_obs.py, orc_probe.py, orc_read.py and the test pairs.json,
  and asserts that results/flawed/orc/ holds no orc_read_*.json.
- A second row `orc_arm` before ORC's main training.  It hashes the config, the raacpo manifest and check log, and the job scripts, and asserts jax/pi05_g8v_raac_lo_ae/{main,tl} are absent.
- experiments/prereg/pz.py (tag `pz`): after the pz smokes and pz_check, before main / tl training.  It hashes raas0_model.py, l90c_config.py, c_train_x.py, pi0z_export.py, pz_check log, pz_pred.txt,
  the norm stats and the job scripts, and asserts jax/pi0_g8v_*/{main,tl} are absent.  Run each through sbatch (CPU).

## 5. Order of work for tonight
1. Write the new files of 2.6-2.7 and 3.4.  Make the add-only l90c_config edit right after a seed-2 chain link has started (squeue), then at once submit cfgcheck.sh.
   It is a CPU job that imports l90c_config, calls register(), and constructs train_config(n) for every name in NAMES.  If it fails, revert the edit before the seed-2 chain's next restart.
2. Submit in parallel:
   - (a) pz smokes, 20 steps each, exp smoke, on tail-lab (2 GPUs; tail-lab then holds 3 of 4), plus pz_check;
   - (b) orc_gen test and orc_gen pairs, CPU jobs.
3. Once the smokes and pz_check pass: write prereg `pz`, then launch the two pz chains, ONE chain each (tail-lab exp tl, 6 x 4 h afterany).  Do not run duplicate overcap chains: they cost budget.
4. When orc_gen test finishes: render on overcap (A40 or any EGL GPU), run orc_obs (CPU), then orc_read selftest, then write prereg `orc`.
   Then run orc_probe for A+0, A+, A0, A and RAA-sL (one overcap GPU job, sequential), then orc_read and the contrasts.  This is the gate.
5. When orc_gen pairs finishes, render, link, convert and check whatever the gate says (CPU / GPU about 1.4 h), so that ORC is ready to launch.
   If the gate opens: ORC smoke, prereg `orc_arm`, then the ORC chain on the 4th tail-lab slot (or on overcap if the slot is busy).
6. After step 6000: export, serving smoke, captures, probe, read, contrasts and stateproj_diag_pz for each pz arm.  For ORC: export, orc_probe, orc_read, and the conditional N + K probe.
7. Append results to a new note (docs/notes/plan_oracle_pi0_results.md); label every prediction.  Never open results/l90c/split_sealed.json or key files.  Python never runs on the login node.

## 6. Smoke checks (each must pass before the step after it)
| step | check |
|---|---|
| cfgcheck | every name in NAMES constructs; the 4 new configs print model class, pi05 flag, max_token_len, weight loader path, use_quantile_norm, and the absence of DeltaActions |
| pz smoke (20 steps) | finite loss; step-0 loss recorded (k3); 0 tokenizer truncation warnings (max_token_len 48); state_proj in the trainable set; the checkpoint path has no "pi05" |
| pz_check | k = 0 equals openpi Pi0; the servo formula matches; the parameter tree equals pi0_libero's; state_proj at init equals pi0_libero's; quantile norm on; on the smoke export the torch servo is correct |
| pi0z_export | state_proj in safetensors, bf16-close to JAX; PolicyBackbone(<cfg>, ckpt).describe() loads; H = 10 |
| pz serving smoke | role 1 and 2 tokens in the server log; replay-vs-server max abs diff < .01 on the A40 in a 2-observation probe (`--limit 2 --ident`); ident rows about 0 |
| orc_gen | for 5 pair layouts the regenerated init object positions equal the source demo's within 1 mm; every pair is >= 15 cm apart; yields logged per object |
| render | the replay check restores the stored states to 0.000 mm (as for A+) |
| convert / check | CHECK PASSED; held-frame count about 2x A+0's 5,306; planner episodes 200 |
| orc_obs | >= 120 observations; for every observation state_cf - state equals (s_j - s_i, 0, 0) exactly in raw dims 0-1; images and prompt identical; role 2 |
| orc_probe | the clean row reproduces bk.sample with the same seed (max abs diff < 1e-3); abl.tok runs; orc_read selftest passes (1 / 0 / .5) |
| ORC smoke | step-0 loss <= 2 x .0556 (A+0 was .0474); state_proj trainable; assets raac |

## 7. Which results would refute "the where must be owned outside the network"
- **Refuting (Exp 2):** pi0-A0, with ordinary data, no dose and no servo, reads the token at the pass bar (S_off CI lower >= .70), AND its N capture success on the same states is >= RAA-sL's - .05.
  A learned network with a trained state pathway would then absorb the exact where-signal from ordinary BC data and aim as well as the explicit servo.
  The pi0.5 result would be a property of a newly added pathway, not of behaviour cloning.  Weaker version: R2 alone (S_off CI lower >= .30) forces the framing down to "a newly added input pathway is not taken up".
- **Refuting (Exp 1):** ORC (or A+0) is LEARNED AND its N + K share reaches the pass bar (CI lower >= .70) AND its dev N (conditional rollouts) is >= RAA-sL's - .05.
  Training data that force the token on a minority of frames would then replace the explicit module, which is exactly DRAFT 6.4's question.
- **Not refuting, but weakening the evidence:** ORC NOT LEARNED.  The shortcut reading then cannot be told apart from an optimisation failure on pi0.5, so C-G rests on Exp 2 alone.
- **Supporting:** a within-model dissociation (2.5, row 1) together with R1.  Then the network reads the token exactly where it must, ignores it where the prefix suffices, and does so on two backbones, one of which has a trained pathway.

## 8. GPU-hour estimate (A40; 5.4 s/it at batch 32, 6,001 steps = 9 h, plus about 0.5 h of restarts / compile per arm)
| item | GPU-h | notes |
|---|---|---|
| Exp 1a: render test set (80 demos) | 1.1 | raafp_diag: about 1.4 per 100 |
| Exp 1a: orc_probe, 5 existing models x 240 obs | 0.8 | 3-4 rows per batch, 10 denoising steps |
| Exp 2: smokes + pz_check + serving smokes | 0.6 | |
| Exp 2: train pi0-sL, pi0-A0 | 19 | pi0's 48-token prompt (pi0.5 pads to 200) may make it about 10 % faster |
| Exp 2: captures (2 arms x K + N) | 2.0 | about 80 episodes per job |
| Exp 2: probes rows full (2 arms x K + N) | 2.4 | `--rows inputs` cuts this to about 0.4 |
| **fixed subtotal** | **about 26** | CPU work (orc_gen 1.4 CPU-h, convert, exports, reads, prereg) not counted |
| Exp 1b (iff the gate opens, p about .70): render 100 pair demos 1.4 + smoke 0.1 + train 9.5 + orc_probe 0.2 | +11.2 | |
| Exp 1b conditional N + K capture + probe (iff ORC phi point >= .30) | +2.2 | |
| pi0-A (iff the gate closed and R1 holds) | +11.5 | |
| **expected total** | **about 26 (gate closed, no pi0-A) to about 37-39 (gate open)** | |

**Budget: if the gate opens, the plan needs about 35-39 GPU-h, over the 30 budget by 5-9.**  Cuts, in order, before asking:
1. pz probes `--rows inputs` (-2.0; top1 / PRn of the writes are then not reported, and the share is unchanged);
2. the ORC N + K probe only if ORC is LEARNED (not at point >= .30) (-2.2 in most cases);
3. the descriptive readout models A and RAA-sL dropped (-0.3).
After these cuts the gate-open total is about 33.  The remaining overrun (about 3 GPU-h) needs the orchestrator's approval.  Never cut: the pz-sL / pz-A0 training and probes, the A+0 / A0 oracle readout, and ORC when the gate opens.
pi0-A runs only within budget.  Wall clock: the pz chains take about 10-11 h from launch.  ORC finishes about 6 h after them (it needs the gate and its data).  tail-lab peaks at 4 GPUs (seed-2 + 2 pz + ORC); every short job goes to overcap.

## 9. Risks
1. **pi0's init is less matched than pi05_libero's** (the delta -> raw label change and horizon 50 -> 10).  The pi0 arms may miss the K gate (UNTESTABLE).  No fix tonight.  The gate keeps a failed recipe from being read as evidence.
2. **Residual reading of the old token meaning.**  pi0_libero's expert reads state heavily at init (it subtracted it).  Leftover dependence that does not point at the counterfactual target projects to about 0 (fact 2), and the kept arm carries the same residual.
   stateproj_diag_pz reports the change from init.  A share that is nonzero in both pz arms equally reads as residue, not use.
3. **Quantile forced on pi0** is a deviation from pi0's shipped z-score norm.  It must be stated.  If a reviewer objects, a z-score variant needs a z-score servo and edits to the probe and serving (not planned).
4. **The converter's "pi05" path sniffing** silently mis-converts a pi0 checkpoint whose path contains "pi05".  The naming rule and the assert in pi0z_export guard it.
5. **l90c_config edits under the running seed-2 chain**: an error breaks its next restart.  Mitigation: edit right after a link starts, then cfgcheck at once; add-only; lazy imports.
6. **Paired-spot yield.**  Free area for a second spot >= 15 cm away may be scarce (h_j spot-exists 70-92 %).  k3 keeps the 15 cm rule and builds from the pairs that exist.  Below 50 pairs per object, ORC is not trained (deviation recorded).
7. **phi depends on the clean steer.**  A model that stalls at transit onset gives a small denominator.  The label-based observation rule is model-independent, and the ratio of sums is stable; UNTESTABLE when sum den <= 0.
   The speed-preserving ideal follower is an assumption, stated with phi.  sigma_demo is the label-referenced check.
8. **Memorisation inside A+0** makes A+0 a conservative oracle.  The test set uses new layouts, so any phi > 0 there comes from the token.
9. **The table-spot data lower F6_a** (A+, A+ctrl, A+0; raaf_sel).  This is irrelevant to the oracle measure, and ORC's dev rollouts are conditional.  Do not read ORC's F6 cells as part of this test.
10. **One seed per arm, development states only**, as for every RAA-F route.  Exp 2's read uses the SR 83 dev set, the same set as A0 and RAA-sL.  A confirmatory read would need the sealed 5th fresh set and is not planned.
11. **Overcap preemption**: chains are requeue-safe (resumable checkpoints every 250 steps).  Captures and probes skip existing files.

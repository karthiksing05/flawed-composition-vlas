# Plan BINDVLA: a SmolVLA with an explicit binding bottleneck (2026-10-27)

Follows [THEORY_FRAMING.md](THEORY_FRAMING.md) (P1-P5) and [PLAN_BINDER.md](PLAN_BINDER.md) (T1-T4, sections 6-12). User
approval (2026-10-27): "use SmolVLA public but train BindVLA". This file is written and committed BEFORE any BindVLA training
step. Nothing in it is a result.

## 0. Why this model

- The B1 series (PLAN_BINDER 8-12) showed three things about external binders fitted on the training cells:
  - ViT keys carry absolute position (linear R^2 0.72 / 0.52 for SigLIP row / col; B1), so "left" became a remembered location.
  - Removing position linearly removed that memory (held-out depth 1 0.867 -> 0.467; B1.1).
  - Any selection criterion scored by FIT on training layouts picks a layout shortcut (B1.2: nouns moved to a layout code and
    held-out depth 0 fell 0.86 -> 0.30).
- BindVLA moves the binding inside the policy. It is trained end to end by behaviour cloning only, with no binding
  supervision. Each theory condition is an architectural constraint that a single config flag can switch off:
  - the policy can reach the scene only through per-role slots (T1 says a holistic path is not systematic);
  - each slot is built from its own role's phrase (T2: atom experts that see only their atom);
  - binding keys carry no absolute position (the B1 lesson);
  - an invariance penalty across layouts (P4).
- The selection criterion is BC loss, which is also a fit criterion. The architecture, not the criterion, is what blocks the
  layout shortcut, and the ablations test exactly that.

## 1. Weights, data, contamination

**Base weights (on disk only; nothing is downloaded).**
- `lerobot/smolvla_base` is NOT on disk. A cluster search (job 4288962) found it in no HF cache under /coc/flash10/ksingara3 or
  ~/.cache. It is public (not gated), but the run uses on-disk weights only, so it is not fetched.
- On disk:
  - `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ 7b375e1 (SmolVLA's VLM);
  - `lerobot/smolvla_libero` @ 31d453f (fine-tuned from smolvla_base on lerobot/libero, 25k steps x 32, VLM unfrozen per its
    train_config.json).
- **BindVLA's initialisation:**
  - VLM: SmolVLM2-500M-Video-Instruct, truncated to its first 16 text layers as SmolVLA does. It stays FROZEN (vision tower,
    connector, text layers, embeddings).
  - The SmolVLA defaults (train_expert_only = freeze_vision_encoder = True) mean that smolvla_base's VLM should be these same
    weights. This cannot be checked without the base checkpoint.
  - Action expert: initialised from SmolVLA's config (width 0.75, 16 layers, cross-attention, self-attention every 2 layers) with
    random weights, because smolvla_base's pretrained expert is absent.
  - Not initialised from smolvla_libero: it was trained on held-out material of this split (below).
  - **Deviation from the brief** ("initialised from the public SmolVLA base weights"): only the VLM half of smolvla_base is
    available. If the user prefers, smolvla_base can be fetched (public) and the main run repeated with its expert. That needs
    approval because it is a download.
- **Contamination of the public reference.** `lerobot/smolvla_libero` was fine-tuned on lerobot/libero, the 40 LIBERO-Spatial /
  Object / Goal / 10 tasks. That set contains material this split holds out:
  - libero_object's ketchup task (ketchup is the split's held-out OBJECT part atom);
  - libero_goal 3 and 7;
  - the LIBERO-10 two-step tests (`two_step_test`).
  - LIBERO-10's scenes are LIBERO-90 scenes, and its steps duplicate LIBERO-90 tasks (SPLIT.md G0).
  - **Any held-out comparison against smolvla_libero is therefore contaminated. It is a reference, not a comparator.**
- **The fair comparator (recommended, NOT run without approval):** `vanilla_frozen`. It is the same frozen SmolVLM2 with a
  fresh expert, trained on the same cells, sampler, steps and recipe as BindVLA, with every binder flag off. It is one flag
  setting of the same code (section 3), at the same cost as the main run (about 8-10 A40-h).

**Data.** `lol/libero_l90c` (results/l90c/data_manifest.json) is the frozen split's 76 TRAINING tasks: 44 LIBERO-90 cells and 32
suite additions. Every episode the ports hold: 3,359 episodes, 495,313 frames. This is pi05_l90c's dataset, made by
experiments/c_data.py, which refuses any non-training task.
- pi05_libero conventions: 8-D state, 7-D actions with gripper -1 open / +1 close, 256 px frames already in the policy's
  orientation, fps 10.
- The rollout client (experiments/libero_client.py) sends 224 px frames (resize 224). Training therefore resizes 256 -> 224 ->
  SmolVLA's 512 pad-resize, so training and serving see the same pixels.
- The training code reads only the training split. split_sealed.json is never opened.
- **Held-out training episodes** (validation and diagnostics): 2 episodes per training task, chosen by a fixed hash among the
  episodes that have contact-event referents in results/bind/b1/demos (152 episodes, about 4.5%). They are excluded from
  BindVLA training. They are used for (a) the validation flow loss, under fixed (t, noise) draws, and (b) the slot-mask referent
  diagnostic (section 5).

## 2. Architecture (lifelong_ood/policy/bindvla.py)

Frozen (no gradient, bf16):
- the SmolVLM2 vision tower (SigLIP, 512 px -> 32 x 32 patches, 768-d) and connector (pixel shuffle -> 64 tokens x 960 per
  camera);
- the token embeddings;
- the 16 text layers.

Binding features:
- The SigLIP last hidden state, 2 x 2 average-pooled to a 16 x 16 grid (256 tokens, 768-d) per camera.
- The grid is aligned with the b1 demo token grid (16 x 16 on the 224 policy image), so the diagnostic needs no remapping.

**1. Role slots** (flag `parser`, default `rule`; flag `cross_role`, default off).
- Parse:
  - Steps come from primitives.steps_of. Each step gives (object phrase, goal phrase) by s1_labels.split_step, and the verb by
    the b1 VERB rule. These are the same rules S1 / B1 used. "A of B" splits at the first "of" (b1_bind.parse).
  - Each step gives three roles: verb, object, goal. At most 2 steps, so at most 6 role slots.
- Each phrase is tokenised by SmolVLA's tokenizer and embedded by the FROZEN SmolVLM2 token embeddings (context-free).
- Query = a 2-layer transformer (width 384, 4 heads) over that phrase's tokens ONLY, plus learned role, step and sub-phrase
  (A / B) embeddings, mean-pooled.
  - With cross_role off, phrases are separate sequences, so no attention crosses roles.
  - `cross_role=on`: one sequence holds every role's tokens (role-tagged). Each role still pools its own tokens. This is the
    RoboGround-style setting: queries see the whole instruction.
- The verb role produces a token but binds nothing. Verb and goal preposition carry the operator identity (P2: operators
  compose).
- `parser`:
  - `rule`: the parse above. It is an oracle on LIBERO's templates, so it is disclosed as one.
  - `noisy`: rule parse with training-time span corruption. With p = 0.3 per phrase, the phrase boundary moves 1-2 words into
    the neighbouring role, or one word is dropped.
  - `learned`: a soft tagger. A 2-layer transformer over the whole instruction's frozen embeddings gives p(role | token), and
    each role query pools tokens weighted by p. It is trained only by the BC loss.
  - `generic`: K learned slot queries, independent of language (Pelican-style). Language enters through one pooled instruction
    token.
  - `token`: one query per content token of the instruction, with no role grouping (OTTER-style per-token selection).
    Positions enter only in the values (absolute). The per-token slots are pooled into K tokens by K learned attention-pooling
    queries.
  - `generic` and `token` use K = 6, the same token budget as BindVLA's 6 role slots.

**2. Soft mask + two-hop** (flag `two_hop`, default on).
- Appearance logits: a_j = <q, W_k k_j> / sqrt(d) x a learned temperature, over the 256 agentview tokens. m1 = softmax(a).
- Relative geometry:
  - With p_j the token's (row, col) and (mu, sigma) the m1-weighted mean and spread (the soft "sibling set" of the phrase's
    appearance match), r_j = (p_j - mu) / max(sigma, sigma_min).
  - Final logits = a_j + g(q)^T phi(r_j), with phi = (r, r^2, r_row r_col) and g a ZERO-INITIALISED linear map of the query.
  - So "left" can bind only relative to the phrase's own sibling set. No convention is coded in: LIBERO's mirrored left / right
    (PLAN_BINDER 6.1) must be learned from the actions.
  - m = softmax(final logits).
- "A of B" with two_hop on:
  - B's slot is resolved first (its own query, sub-phrase embedding B).
  - A's logits get + beta log(eps + m_B / max m_B) (beta learned, init 1), so B's mask restricts A's attention.
  - A's relative geometry is computed relative to B's (mu, sigma).
  - With two_hop off, A and B tokens form one phrase and one hop.
- Slot feature s = sum_j m_j W_v k_j. With `wrist_slots` (default on) the same query also attends the 256 wrist tokens (raw
  keys: the wrist frame is egocentric), giving s_w and the wrist mask centroid.

**3. Position-free keys** (flag `pos_keys`, default on).
- k_j = P f_j. P is a fixed linear projection that removes the position subspace by iterative nullspace projection (B1.1's
  method). It is fitted on agentview tokens of TRAINING frames only, until a fresh ridge probe's held-out R^2 for row and col is
  < 0.05.
- **A nonlinear probe is also reported after removal:** a 2-layer MLP (256 hidden), held-out frames. If its R^2 stays high, the
  keys still carry position nonlinearly, and the report says so. (SigLIP's position embedding is added at the input and mixed
  nonlinearly, so features "before the positional embedding" do not exist for a ViT. Projection is the available route.)
- Values use the same P.
- `pos_keys=off`: raw SigLIP features as keys and values (position available implicitly).
- The OTTER-style `token` setting appends absolute (row, col) to its values.

**4. Bottleneck** (flag `bottleneck`, default on).
- The prefix the action expert reads (through the frozen text layers' K / V, as in SmolVLA) holds:
  - the 6 role tokens. A binding token is MLP([s, s_w, pose, wrist pose, relative geometry, role / step embeddings]) -> 960-d,
    RMS-normalised and scaled to the language-embedding norm. Absent roles are masked.
  - one global context token: a projection of the mean connector token of each camera (low-res gist);
  - the state token (SmolVLA's state_proj).
- Pose:
  - the mask-weighted (row, col) and its spread: the operator's parameter, which is absolute by necessity (P1: competence is
    pose-parameterised);
  - relative geometry: the centroid relative to the parent (two-hop) and the sibling rank, i.e. the m1-mass fraction left of /
    above the centroid.
  - **No 3D point:** the dataset stores no depth, so the camera-frame 3D pose of the brief is not available (deviation).
- `bottleneck=off`: the expert reads SmolVLA's full prefix (64 connector tokens per camera + the 48 instruction tokens + state).
  If any binder module is on, the role tokens are appended (RoboGround-style).
- `binder=off` + `bottleneck=off` is vanilla SmolVLA with a frozen VLM: `vanilla_frozen`.

**5. Invariance penalty** (flag `inv`, weight lambda; default chosen in the smoke, section 4).
- Environments: the task's scene (factors.Task.scene; e.g. KITCHEN_SCENE3 is shared by several LIBERO-90 cells). For
  LIBERO-Spatial, whose tasks share a scene name but differ in layout, the environment is the task.
- **Env-balanced sampler** (used for EVERY config, inv on or off, so the flag changes only the loss):
  - each batch of 32 = 4 environments x 8 frames;
  - environments are drawn uniformly, frames uniformly within an environment.
- **t matched within the batch:** the 8 frames of every environment get the SAME 8 stratified flow times t (one per 1/8 bin)
  and the same noise draws (broadcast over environments).
- **Per-t standardisation:** each sample's loss is divided by a running mean (EMA 0.99) of the loss in its t bin, computed over
  all environments. This removes the t-dependence of the flow-matching loss before the variance is taken. (Dividing by the
  initial model's per-environment loss is meaningless here: the expert starts random.)
- Penalty = lambda x Var_e(mean standardised loss of environment e), V-REx (Krueger et al. 2021), on the 4 environments of the
  batch. The EMA is detached.
- **Known weakness:** layouts that are just harder (more action variance) still raise the variance. V-REx is weak when
  environments barely differ (Pareto-IRM 2206.07766), and LIBERO's layouts are few and similar. The run therefore logs:
  - the per-environment loss spread;
  - the penalty's size relative to the BC loss;
  - the ratio of its gradient norm to the BC gradient norm on the binder parameters, every 100 steps.
  - The report states whether the term was active (gradient ratio > 0.05 over most of training) or idle.
- Layout diversity is a known limitation. The sim layout generator being built for B1.3 belongs to another agent and is not
  used here.

**Trainable:** the action expert, action / time projections, state_proj, and the binder (phrase encoder, key / value / geometry
projections, slot MLP, global-token projection). The count is printed by the model and reported (section 7).

## 3. Named settings (each is a flag setting of one code path)

| setting | binder | parser | cross_role | two_hop | pos_keys | bottleneck | inv | what it tests |
|---|---|---|---|---|---|---|---|---|
| **bindvla** (main) | on | rule | off | on | on | on | lambda* | the full theory |
| vanilla_frozen (fair comparator) | off | - | - | - | - | off | 0 | the holistic baseline (T1) |
| -bottleneck | on | rule | off | on | on | **off** | lambda* | T1: a holistic path is available |
| -pos_keys | on | rule | off | on | **off** | on | lambda* | the B1 lesson inside a VLA |
| -two_hop | on | rule | off | **off** | on | on | lambda* | depth: parent resolution |
| +cross_role | on | rule | **on** | on | on | on | lambda* | T2: atom experts see only their atom |
| -inv (ERM, same sampler) | on | rule | off | on | on | on | **0** | P4; the DomainBed lesson: beat a tuned ERM |
| parser=noisy / learned | on | **noisy / learned** | off | on | on | on | lambda* | the oracle-parse objection |
| roboground | on | rule | **on** | on | on | **off** | 0 | RoboGround 2504.21530 at equal budget |
| pelican | on | **generic (K=6)** | - | - | on | on | 0 | Pelican-VLA 0.5 2607.06655, matched K |
| otter | on | **token (K=6)** | - | - | **off** (abs. pos in values) | on | 0 | OTTER 2503.03734 |

Only **bindvla** is trained within the current cap. Every other row is one flag and about 8-10 A40-h each, and needs approval.

## 4. Predicted outcomes (written before training; each ablation's predicted direction and split)

Splits:
- TR: closed-loop success on the training tasks;
- DEV: the dev held-out tasks, i.e. the CD bins 0.1 / 0.3 / 0.5 (5 dev tasks each) and the part set (14 dev tasks);
- HT: held-out training episodes (mask diagnostic, validation loss);
- LAT: dev tasks with a lateral / ordinal instance word (:16, :60, :38, caddy compartments);
- D2: depth-2 phrases (drawer / shelf / compartment).

| setting vs bindvla | TR | DEV | finer prediction | theory |
|---|---|---|---|---|
| vanilla_frozen | equal or higher (more capacity to fit) | LOWER, gap growing with CD bin | its error on DEV rises with CD; bindvla's is flatter | T1 vs T2 |
| -bottleneck | equal or higher | LOWER on DEV (any bin) | slot-mask referent hit on HT LOWER (slots get less gradient once a holistic path exists) | T1 |
| -pos_keys | equal | LOWER on LAT and caddy; ~equal on depth-0 DEV | HT mask hit equal or HIGHER (training layouts reward location memory) | B1 / B1.1 lesson, P4 identifiability |
| -two_hop | ~equal (few depth-2 training tasks) | LOWER on D2 | drawer / compartment masks hit the body, not the part | P3 depth |
| +cross_role | equal | LOWER on recombination bins (0.3 / 0.5) | role-swap errors (object slot on the goal entity) up | T2 invariance |
| -inv | equal | ~equal (predicted: the term is weak on LIBERO layouts) | if the gradient-ratio log shows the term idle, no difference is expected or claimed | P4, with a known weakness |
| parser=noisy | slightly lower | ~equal; better on paraphrases | | robustness of the parse |
| roboground / pelican / otter | equal or higher | LOWER than bindvla on CD 0.3 / 0.5 and part | otter, like -pos_keys, fails on LAT | prior-art contrasts |

The **paraphrase test** (planned, not run in this cap): for each training and dev instruction, 3 non-templated paraphrases,
written once by rule-free human-style rewriting (e.g. "grab the ketchup and drop it into the basket"). They are frozen in a
file before any read. Scored closed loop for bindvla, parser=noisy / learned, and vanilla_frozen. Prediction: rule-parser
bindvla degrades the most, learned the least.

## 5. Training recipe (experiments/bv_train.py)

- Loss: SmolVLA's flow-matching BC loss. The action chunk is 50; the loss is the MSE of the velocity over the 7 real dims, with
  padded steps masked. Plus the inv penalty. No binding supervision of any kind.
- Optimiser: AdamW (lr 1e-4, betas 0.9 / 0.95, wd 1e-10; SmolVLA's preset), grad clip 10, 1k warmup, cosine to 2.5e-6, bf16
  autocast. Normalisation: MEAN_STD for state and action, computed on the training episodes only.
- Batch 32 (4 environments x 8), 30k steps (960k samples, about 2 epochs; smolvla_libero used 25k x 32). Checkpoints every
  1k steps (trainable weights + optimiser + EMA stats + RNG + sampler state). The latest 2 are kept, plus every 10k.
  - The job is `--requeue` on overcap. On start it resumes from the newest complete checkpoint (written to a temp file and
    renamed atomically).
- Lambda for inv: the smoke runs 1.5k steps at lambda in {0, 0.1, 1} and chooses the value with the lowest HT validation flow
  loss. Ties within 2% go to the smaller lambda. A 1.5k-step selection is weak, and that is disclosed.
- Logged every 100 steps:
  - BC loss and per-environment loss spread;
  - inv penalty and gradient ratio;
  - learned temperatures, beta, and |g| (does geometry turn on?).
- Logged every 2k steps:
  - HT validation flow loss under fixed (t, noise);
  - **slot-mask referent diagnostic** on HT frames (frame 0 of the held-out episodes, from results/bind/b1/demos):
    - object-slot hit: the mask's argmax token within Chebyshev distance 1 of the grasp referent token;
    - goal-slot hit: the same for the release referent;
    - also the mask mass in that 3 x 3 window and the uniform chance level.
  - For the 61 sim-rendered matched demos (results/bind/b1/demo), the mask mass on the named body's sim mask is also reported.
    It is used for scoring only.

## 6. Evaluation plan (pre-registered before any dev read)

- **Parity (training tasks):** closed-loop success, MODE=ours (the plain Euler sampler, 10 steps), replan 5, init states 0-9
  (10 episodes per task).
  - BindVLA runs on its 76 training tasks.
  - Reference: smolvla_libero on its own 40 fine-tuning tasks, the same harness (c_parity's pattern).
  - Parity bar (descriptive): BindVLA's mean training success >= 0.8 x smolvla_libero's own-40 mean.
- **Dev read (held-out):** the 29 dev tasks (15 CD-bin + 14 part), 10 episodes each, for BindVLA and the contaminated
  smolvla_libero reference.
  - Reported per CD bin and per part role, with two-level task bootstrap CIs.
  - The predictions of section 4 are rows: the vanilla_frozen and ablation rows are filled only if approved.
  - experiments/prereg/pbv.py writes the m3_prereg row (tag pbv) BEFORE the first dev episode. It holds:
    - the checkpoint hash;
    - the smoke and parity numbers;
    - the bars;
    - the file hashes.
  - Ledger rows: bv_parity, bv_dev.
- **No sealed set and no confirmatory read** (they need the user's approval).

## 7. Budget (cap 30 A40-h; stop and report before exceeding it)

| item | A40-h (estimate) |
|---|---|
| data cache (CPU), unit tests (CPU) | 0 |
| smoke: position fit + probes, timing, 300-step runs of bindvla and vanilla_frozen, lambda mini-sweep 3 x 1.5k | <= 1.5 |
| bindvla main run, 30k x 32 (to be re-estimated from the smoke timing; steps cut if > 12 A40-h) | 8-12 |
| parity: bindvla 760 episodes + smolvla_libero 400 episodes (about 0.26 A40-h per 100 episodes at 2 instances per A40) | about 3 |
| dev: 2 x 290 episodes | about 1.5 |
| total | about 14-18 |

Fair comparator (vanilla_frozen) and ablations: about 8-10 A40-h each, recommended, not run.

## 8. Rules

SLURM only (overcap / tail-lab, explicit -p / -A, the exclude list), job scripts under .claude_tmp. Demos are fitted only, never
replayed. split_sealed.json is never opened. Audits stay at 0 / 0 after result rows are added. Other agents' files (b1_3*,
prereg/pb1_3.py, the layout generator) are not touched.

## 9. Prior art to cite and contrast (novelty check, research_notes/BindVLA novelty/)

- RoboGround 2504.21530: object + placement queries under masks from a GLaMM-style VLM.
  - Its binding is supervised and reads the whole instruction.
  - It has no per-role phrase queries and no bottleneck.
  - Our `roboground` setting matches its budget.
- SlotVLA 2511.06754: slot-attention object slots plus relation queries, with language reaching the decoder.
- Pelican-VLA 0.5 2607.06655: a hard 32-slot generic bottleneck. Our `pelican` setting uses matched K.
- OTTER 2503.03734: the closest on components 2 + 3 + 4.
  - Unsupervised per-text-token cosine selection over frozen CLIP patches, with absolute position only in values, pooled into the
    policy.
  - Our `otter` setting.
- POT-VLA 2607.18016: role-typed TARGET / DESTINATION slots from a planner + SAM3 + depth into GR00T; no bottleneck.
- ControlVLA 2506.16211: GroundingDINO + SAM2 object tokens via zero-init K / V.
- Invariant Slot Attention 2302.04973: slot-relative frames, the precedent for relative geometry.
- POCR: what / where split, the precedent for separating content from position.
- IRM in imitation, 2308.09189 / 2311.01489: invariance across environments in IL.
- INGRESS / MAttNet: two-stage relational grounding, the precedent for two-hop.
- **Our claim** is the theory-driven assembly plus ablations predicted to fail in stated directions on stated splits. It is not
  any single module.

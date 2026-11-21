# RAA-F (SR 92 candidate): make the where-signal NECESSARY in the training data. Three routes, one selection rule, one confirmatory read

Written 2026-11-20 as a DESIGN, before any build, training job, prereg row or rollout.  Nothing here has been run except the read-only audit
listed in the appendix (file reads on the login node, `sacct` for past job durations, no python).  The plan commits to the selection rule and
the confirmatory contrasts below.  The probability lines in section 8 are copied into experiments/prereg/raaf_pred.txt when the row is written.
No existing file is edited by this plan.  The SR 91 agent's files and jobs (raasw5r_*, fc_n2_wz.py, rg_comp_z.py, wa_sw5z.py, sr91_*) are not
touched, read for writing, or relied on except as listed ranges.

## 0. Diagnosis being tested, and what is already known

- Paper (paper/DRAFT.md 4.8, 6.2, 6.3): pi0.5 fails new pairings of known parts.  The fine-tuned RAA arms add a where-to-go channel: the 4-dim
  continuous state token [dx, dy, dz, role] (sc_rv.chan_state: target - eef - held offset while held, role 2; object - eef before, role 1), and
  RAA-s / RAA-sL add a hand-set servo (raas_model.py: action_xyz = clip(K rel, -M, M) m + expert, K 2.851 /m, M .9375 from
  results/flawed/raas_gate.json; the expert is trained on (action - servo), the servo computed from the chunk's FIRST frame and broadcast over
  the 10-step chunk).
- SR 83 (docs/notes/sr83.md): in RAA-sL the channel share is .871, but it is the servo's (.878); the state token's share is **-.007** (token
  ablation -.01 on every set).  The learned network does not read the token.  SR 82: the expert pushes against the servo near the target.
  SR 84-90: test-time overrides W1-W6 (lifelong_ood/policy/backbone_w.py) help only where the override is itself competent.  Module removal
  reproduces the failure (dissociation SHOWN in SR 87 and SR 90).
- RAA-o / RAA-c (translation-only target frame; channel as text, then as a state token) were NOT SUPPORTED.  A frame change does not change
  a LIBERO label (delta end-effector commands), so it adds no pressure.
- Diagnosis: a behaviour-cloning shortcut.  In K1-K6 the role phrase, the painted views and the held object predict the target exactly, so
  nothing in the loss rewards reading the token.

**Principle under test (falsifiable).**  A where-signal is used by the learned network if and only if the training data make it
necessary, i.e. some held-phase labels have a destination that the non-channel inputs (prompt, views, held object) do not determine and the
channel does.  Operational dose: the fraction of held-phase training frames whose destination is not determined by the prefix, and the size of
the label change those frames carry (both reported per route, section 1).  The prediction is that the learned-token share rises only in routes
with a dose above 0 (A, A+), and stays near 0 in the route with dose 0 (B).  "If" is falsified if A+ (large dose) fails the probe.  "Only
if" is falsified if B passes it.  A failure of A alone (small dose) is read as a dose result, not a falsification.

### 0.1 A derivation that fixes Route A's architecture (to verify in Step 0, check S0-4)
For a held chunk starting at frame t (role 2 at t, so m = 1) whose 10 frames are all held and unclipped, shift the channel by delta
(rel' = rel + delta).  Relabel every frame servo-consistently, a'_{t+j} = a_{t+j} + K delta.  Then the expert's target is

    a'_{t+j} - clip(K rel'_t) = a_{t+j} + K delta - K rel_t - K delta = a_{t+j} - K rel_t,

which is **identical to RAA-sL's target**.  So a servo-consistent relabel lies in the servo's null space.  It gives the learned network no
pressure; worse, the token now varies while the target does not, which teaches the expert to ignore the token.  Any other relabel (for
example an open-loop correction that integrates to delta over the remaining carry) leaves the expert a target term (c - K delta).  The expert
can only fit that term by reading the token AGAINST the servo, which is the SR 82 pathology written into the labels.  Therefore **Route A is
trained servo-free** (RAA-c's model, servo_k = 0, from pi05_libero).  The labels then carry K delta, and only the token can supply it.  The
servo-kept variant ("A-s") is not trained.  Its null is checked offline (S0-4) at zero GPU cost.

## 1. Arms and the shared comparator

Shared recipe for every trained arm unless a stated change: pi05_libero init (`/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi05_libero/params`,
RaacLoader with `missing_regex=".*lora.*|.*state_proj.*"`), frozen SigLIP + PaliGemma (the `_ae` recipe), action expert + state_proj trained, no
EMA, batch 32, lr 5e-5, 1,000 warmup, **6,001 steps**, one seed, one A40, rview2 role views and role-phrase prompts as RAA-c / RAA-sL,
RAA-c's norm stats / assets dir `pi05_g8v_raac` kept for every arm (the SR 89 precedent: a changed data repo, the same normaliser and servo
constants), checkpoint step 6000, export by `t16_export.sh` / raac_export.py.

| arm | config (new, additive) | model | data repo (new) | stated change vs RAA-sL |
|---|---|---|---|---|
| **RAA-sL** (comparator) | pi05_g8v_raas_l_ae (exists) | token + servo | lol/libero_g8v_raac | none; checkpoint main/6000 exists; dev rollouts exist (SR 82) |
| **A** relabel | pi05_g8v_raac_lf_ae | token, servo_k = 0 | lol/libero_g8v_raacf | servo removed (0.1), data = base + relabelled held copies |
| A0 (conditional) | pi05_g8v_raac_l_ae | token, servo_k = 0 | lol/libero_g8v_raac | servo removed, base data: trained ONLY if A passes the probe, to attribute the rise to the data rather than to removing the servo |
| **A+** planner | pi05_g8v_raas_lp_ae | token + servo | lol/libero_g8v_raacp | data = base + planner demos to table spots |
| A+ctrl (data-only) | pi05_g8v_rview2_lp_ae | no channel (pi0.5 reads no state) | lol/libero_g8v_rview2p | same frames, prompts and views as A+, no token / servo |
| **B** target crop | pi05_g8v_raas_lb_ae | token + servo + crop in the 3rd image slot | lol/libero_g8v_raacb | one extra image input |

A+ keeps the servo because its labels come from executed planner actions (fa1g's controller, gain 1/0.02 m, saturating), not from the servo
law.  The expert's target a - K rel on table-spot frames therefore depends on the spot, and only the token carries the spot.  The 0.1 null
does not apply.  B keeps RAA-sL's model so that the only change is the crop.

Dose per route (computed in each build, reported before training): fraction of held-phase frames with a prefix-undetermined destination;
median |label change| on those frames in action units; for A also the share of the xyz label variance that the K delta term explains.

## 2. Route A: in-support counterfactual goal relabel

### 2.1 Relabel rule (exact)
For each rendered K1-K6 demo d (K7 / K8 carry no channel and are unchanged), with ProprioHeld rows H (sc_t16.held_rows, the convert's own
rule), hold onset h0 = first held row, release r = first row at or after h0 whose gripper command is "open" (actions[:, 6] < 0, LIBERO sign:
-1 open), and the demo's own channel rel_t (sc_rv.demo_rel + held_target_rel, base-frame axes, metres):

1. Draw delta = (dx, dy, 0) (section 2.3).
2. Make a NEW episode d' from frames h0 .. r - 1 of d (the held carry and descent, before the open command), with the same images (agentview +
   wrist, rview2-painted exactly as in d), the same role-phrase prompt, and:
   - state token: chan_state(rel_t + delta, role 2) on every frame;
   - actions: a'_t[0:3] = a_t[0:3] + K delta, with K = 2.851209548740752 (raas_gate.json "K", the demos' own least-squares feedback gain,
     action units per metre in the channel's axes; the channel and the OSC delta command share the robot base axes, so no rotation);
     a'_t[3:6] = a_t[3:6]; a'_t[6] = a_t[6] (closed on every frame of d', since d' ends before the open command).
3. The original d stays in the data unchanged (delta = 0).  Each eligible demo gets **M = 2** copies with independent delta.

Why K delta.  It is the closed-loop counterfactual label under the demos' own fitted proportional law, a = K rel + (residual the image
explains): at observation o_t, aiming at T + delta changes the law's output by exactly K delta.  It is the "servo-consistent" option, applied
to a servo-free model, so the network itself must produce K delta from the token.  Rejected alternative: a per-frame open-loop correction
c_t = delta / (g N_rem) that integrates to delta under OSC's position scaling (goal = current + 0.05 a, so g <= .05 m per unit per step)
over the N_rem frames left before release.  It depends on time-to-release, which no input reveals.  It also diverges near release, and
it does not follow from the demos' feedback law.

### 2.2 Which frames, and the release timing
- Relabelled: held rows h0 .. r - 1 only (role 2; the target channel).  Object-role frames (role 1, before the hold) and post-release frames
  are never shifted.  The F7 grasp-phase binding (prefix-borne, SR 83) is not addressed by Route A.
- Release frames are excluded from the copies: d' ends at r - 1.  openpi's LeRobot loader (src/openpi/training/data_loader.py, delta_timestamps
  over action_horizon) clamps a chunk that runs past an episode's end to the episode's last frame.  So the chunks near the end of d' repeat the
  last relabelled action: gripper closed, still moving toward T + delta.  The copies thus teach "do not open while the token says delta
  away", and the release itself is learned only from the originals, where token and views agree.  This puts the token in charge of the
  release timing too.  Build check S-A3 asserts the clamping on one d' (the loader's returned chunk at the last frame equals the repeated
  last action, and no frame of d is drawn into d').
- Chunks of d that cross into the release are unchanged (they belong to d).

### 2.3 delta distribution, bounded by the real success geometry
Bounds come from LIBERO's own success predicates (third_party/libero/libero/libero), not from visual footprints.  Read during this audit:
- Movable targets (plate, bowl: K1, K6; tests N1, N4, N6): `On(obj, tgt)` = `tgt.check_ontop(obj)` (envs/object_states/base_object_states.py,
  ObjectState.check_ontop): contact, z order, and **|obj_xy - tgt_xy| < 0.03 m**.  The plate's physical rim (about 5 cm, the
  stable_scanned_objects/plate collision boxes) is wider than the predicate.  The predicate binds.
- Fixture sites (envs/objects/site_object.py, SiteObject.under): the object's body position, in the site frame, satisfies
  |dxy| < site half-size and size_z - .005 < dz < size_z + .10, plus contact with the parent.  Half-sizes from the asset XMLs:
  flat_stove `cook_region` 0.075 x 0.075 (assets/articulated_objects/flat_stove.xml:22); wooden_cabinet `top_side` 0.12534 x 0.09438
  (wooden_cabinet.xml:27); wine_rack `top_region` 0.10087 x 0.02200, site tilted about x by 62.4 deg (quat .855, -.519;
  turbosquid_objects/wine_rack/wine_rack.xml:15).

Per-demo feasible set (registered rule).  Let p_d = the carried object's settled position at the demo's last frame (success fired there), and
let tgt_d be the target body / site pose at that frame (read from the demo's own state in the task env).  F_d = { delta in R^2, delta_z = 0 :
the predicate holds for p_d + delta with every half-size shrunk by a margin of 1.0 cm (the 3 cm disk becomes 2 cm) }.  It is evaluated with
LIBERO's own function on a 1 mm grid over |dx|, |dy| <= 15 cm (pure geometry, no physics).  delta ~ Uniform(F_d minus {|delta| < 1.0 cm}).
1 cm is above the channel's median test error (.18 / .63 cm, SR 82) and gives a label change of at least .029 action units.  Seed:
np.random.default_rng(92_000_000 + 1000 tid + 10 demo + copy).  A demo whose set is empty gets no copy, and the count is reported.

Physics validation (S-A2, on 10 demos per class): teleport the object to p_d + delta at the last frame, step 50 dummy actions, and require
the predicate to still hold.  If more than 10% of a class fails, that class's margin is raised to 2 cm and the grid is redone (recorded).

Indicative bounds before the per-demo computation (half-size - margin - typical release offset; the demos' xy release radius r_class from
lifelong_ood/policy/raasw6_env.json: plate 1.64, bowl 2.67, stove 2.82, cabinet 4.44, rack 4.42 cm):

| target class (K task) | predicate region | indicative max |delta| | expected eligible share |
|---|---|---|---|
| plate (K1) | disk r 3 cm -> 2 cm | about 2-3.5 cm toward the centre side, about 1-2 cm typical | high but small delta |
| bowl (K6) | disk r 3 cm -> 2 cm | about 1-2 cm; many demos settle 1-2.5 cm off-centre | low-medium |
| stove (K2) | 7.5 x 7.5 cm box -> 6.5 | about 4-6 cm | high |
| cabinet top (K3, K4) | 12.5 x 9.4 cm box -> 11.5 x 8.4 | about 5-9 cm | high |
| rack (K5) | 10.1 cm along, 2.2 cm across (tilted) | about 3-6 cm along the rack only, < 1 cm across | medium; physics check decides |

So Route A's dose sits on the site targets (stove, cabinet, rack).  The plate and bowl, the targets of the weak test cells N4 and N6, get
1-3 cm at most.  This is a hard limit of an in-support relabel and is reported as such.  The larger shifts are Route A+'s job.

### 2.4 Clip check (registered, S-A1)
The xyz actions of the demos lie in [-.9375, +.9375] (raas_gate max_step; norm-stats q01 / q99 -.922 / .937, -.767 / .905, -.938 / .937).
For each copy, every relabelled frame must satisfy |a'_t[k]| <= .9375 on k = 0, 1, 2.  If a draw violates this on any frame, delta is redrawn
from F_d (up to 20 draws).  After 20 failed draws the copy is dropped, and the count is reported.  Also reported: the fraction of relabelled
frames outside the per-dim [q01, q99] (the normalised value outside [-1, 1]; allowed, since openpi's quantile Normalize does not clip, but if
it exceeds 2% the build stops for a decision), and the fraction of shifted state tokens outside the state q01 / q99 (state q01 / q99: dx
-.156 / .176, dy -.240 / .174, dz -.279 / .201 m).

### 2.5 Route A build (code that exists vs to write)
Exists: sc_rv.py extract / render / chan / convert / norm (the raac arm; demo_rel, held_target_rel, chan_state); chan_<demo>.npz per demo
(slot positions pos [n, 9, 3], ee); fc_n2_tasks.PAIRS; raac_model.Pi0C, raas_model.Pi0S (servo_k = 0 reproduces Pi0C exactly, per its
docstring and experiments/raas_check.py); RaacLoader; c_train_pi05.py; raac_export.py; raa.py --arm raac and wa_s.py serving for a
state-token model (RAA-c was served this way); a2_capture.py / a2_cap_client.py / a2_probe.py / a2_read.py.
To write (new files only):
- experiments/raaf_geom.py (LIBERO venv, CPU, sbatch): per K1-K6 demo, F_d, the physics validation, the delta draws and the clip check
  -> /coc/flash10/ksingara3/.claude_tmp/raaf/geom/ during the dry run; the registered copy goes to results/flawed/raaf_delta.json after the
  plan is committed.
- experiments/raaf_rv.py (copy of sc_rv.py convert with the relabel; sc_rv.py untouched) -> lol/libero_g8v_raacf.
- additive config entries in l90c_config.py (pi05_g8v_raac_lf_ae, and pi05_g8v_raac_l_ae for A0) and `--g8v raac_lf / raac_l` choices in
  c_train_pi05.py (the additive-edit practice of SR 77-89).  This is the only edit of existing files, made in the build commit, not by this plan.
- experiments/a2_probe_f.py: a2_probe.py imported unchanged, with the model registry extended to the new configs (a2_probe.py's
  `--model` accepts only pi05 / raasl).
- job scripts /coc/flash10/ksingara3/.claude_tmp/pp/raaf_*.sh (sed copies of raasl_train.sh / raasl_chain.sh / raa_roll.sh / wasl_roll.sh).

## 3. Route A+: large relabel through planner-generated demos

### 3.1 Audit: a scripted planner exists
`grep -rIl -i -E "planner|mimicgen|scripted|waypoint" experiments lifelong_ood` (rg is not installed on the login node; bounded grep used)
finds a working oracle-state planner:
- **experiments/fa1g_plan.py `gen`** (stage 1g): N2's familiar sampler + relocation of the four movables in N2's layout box
  (`layout()`); phases approach / descend / close / lift to carry height / carry over the release point / descend / open / retreat;
  `ctrl()` = proportional OSC delta command clip(dp / 0.02) + N(0, .03) noise, rotation clip(rotvec / 0.2); keypoints FITTED from demo
  states (results/flawed/a1g_keypoints.json); kept only if LIBERO's success predicate fires.  Yields (results/flawed/a1g_gen_K*.json):
  K1 50 / 85, K2 50 / 68, K3 50 / 56, K4 50 / 142, K5 50 / 91, K6 50 / 75 attempts, mean 119-220 steps.
- experiments/pp_n4d.py `gen`: the same planner with keypoints COMPOSED across tasks (grasp from one task, release offset from another), for
  a new pairing.  This shows the planner handles a target its keypoints were not fitted on.
- experiments/fa1l_cgen.py, fa1p_sgen.py: variants (grasp-segment acceptance, clearance-stratified layouts).  experiments/p0_retarget.py: a
  MimicGen-style segment retargeter (T_new = T_ref,new T_ref,src^-1 T_ee,src) with replay / track / retarget arms.
Missing for pi0.5 training: fa1g records scene vectors (ee, object poses, actions), not simulator states or images.  sc_rv.py's
render / chan / convert need the per-step flattened sim state, so the generator must also save `env.env.sim.get_state().flatten()` per step,
plus the initial state, in the raw-demo layout that sc_rv.extract produces.  No other planner or generator (mimicgen, "v2 gen") exists in
experiments/ or lifelong_ood/.

### 3.2 What the planner generates (registered)
- Scenes: the LIBERO-Goal kitchen (the K tasks' BDDL), fa1g's `layout()` sampler.  Seeds 93_000_000 + 1000 obj + k (no earlier generator
  used them; fa1g uses 900000+, pp_n4d 950000+).
- Objects: the three movables of the test pairings (bowl, cream cheese, wine bottle).  Grasp keypoints from their K tasks (bowl K1, cheese K6,
  bottle K4), as pp_n4d composes them.
- **Destination = a free TABLE SPOT** (target entity `main_table`).  It is in no test pairing N1-N8: N8's stove_front_region (BDDL
  -0.09..-0.01, 0.17..0.25), expanded by 5 cm, is excluded.  The spot is sampled uniformly in N2's layout box with: >= 20 cm from the
  object's start and from every K target of that object (the size of the real failure, per the SR 82 release-error table); >= 12 cm clear of
  every fixture / movable footprint (fa1g MOVR / FIXR); not in the top drawer.  Release keypoint = the object's K-task release rotation and
  release height above the table (from a1g_keypoints.json).
- Prompt: object phase "object: the <object>", held phase "**target: the table**" (one new lexicon entry, sc_t16 PHRASE; LIBERO's own phrase
  for main_table).  The phrase names the entity, not the spot, so on these frames only the token says where: the dose.  Views: the rview2 rule
  (role_paint; the table is a fixture and is not painted).
- Success filter (no LIBERO predicate exists for a table spot): after the open command and the retreat, 20 settle steps; keep the episode iff
  the object's xy is within 3 cm of the spot (the On predicate's radius), its tilt is < 20 deg from upright, it moved < 1 cm over the settle
  steps, no other movable moved > 2 cm, and both fingers are clear.
- Size: 50 kept per object (150 demos), about the size of one K task's demos, about 1/5 of the 8-task data.
- Dose: all held frames of the 150 demos (about 1/6 of all held frames); label change on them = the full carry (>= 20 cm).

### 3.3 Data-only control (A+ctrl)
The same frames (base + 150 table demos), the same prompts and the same painted views, converted in the rview2 format (no token; pi0.5 reads no
state) and trained from pi05_libero with the same recipe.  On table frames the control cannot know the spot, so it learns a spot average.
A+ - A+ctrl (architecture given the data) is the registered dev contrast of this route (paired, N1-N6 and K1-K6, labelled as raa_read).  The
data-alone effect has no matched no-channel base model here (an R2-from-pi05_libero model is not trained), so A+ctrl is compared only
descriptively, with RAA-sL and RAA-L.  A+ctrl gets no probe (it has no token) and its dev rollouts run through sc_rv serve (rview2 mode), the
R2 arm's path.

### 3.4 Route A+ build and cost
To write: experiments/raaf_gen.py (copy of fa1g_plan.py `gen` with table spots, composed keypoints, the success filter, and per-step sim
states saved to DATA/gen_raaf/<obj>/demo_<k>.npz in sc_rv's raw layout); experiments/raaf_rv.py (shared with A; `--source gen` renders and
converts the generated demos twice: raac format -> lol/libero_g8v_raacp, rview2 format -> lol/libero_g8v_rview2p, base frames copied from
the existing repos' sources); the lexicon entry; configs pi05_g8v_raas_lp_ae, pi05_g8v_rview2_lp_ae.  Cost: generation about 300 attempts
x about 30 s = 2.5 CPU-h; rendering 150 demos (2 cameras + segmentation, sc_rv render's path) about 2 GPU-h; chan / convert CPU.
Smallest fallback if composed keypoints fail for a table spot (yield < 30% after 3 x 50 attempts for an object): reuse the object's K-task
release keypoints unchanged with the spot as the reference; if that also fails, Route A+ is killed for that object (K1).

## 4. Route B: target-centred view

### 4.1 Audit
- Camera model: experiments/sc_pu.py `cam_model(sim, cam)` / `project(p, f, R, c)` (fovy intrinsics; cam_xpos / cam_xmat extrinsics with the
  OpenCV axis flip; image rows top-down, robosuite's raw[::-1]); robosuite `camera_utils.get_camera_intrinsic_matrix` /
  `get_camera_extrinsic_matrix` used in experiments/b1_render.py:82-83.  LIBERO's cameras are fixed per task, so K / E are read once per
  task (one LIBERO-venv sbatch job).
- Training crop centre: the demo ground truth (chan_<demo>.npz `pos` of the active role's slot: the object before the hold, the target
  while held; the same slot as the token).  It is projected into the stored, already flipped agentview frame, and the crop is taken from the
  rview2-PAINTED agentview so the paint conventions carry.  A crop 96 x 96 px of the 256 px view (about 35 cm of table) is resized to 224
  with openpi's resize_with_pad.  Train-time centre jitter is drawn from N(0, sigma) with sigma = the test estimator's error (SR 82: target
  median .63 cm, about 2 px), so train and test centres match in distribution.
- Test crop centre: the module's estimate (the sc_pu.Estimator the wa_s.py / raa.py client already runs, with the fx_r resolver in the rgc
  cells) projected with the same cam_model.  The client sends the pixel centre as meta/crop_uv, and the server crops the painted
  meta/rv_img_agent it already receives.  When the entity is occluded, the crop follows the Estimator's held / pose-memory estimate (the M
  component).
- Entry into pi0.5: the THIRD image slot `right_wrist_0_rgb`.  openpi's LiberoInputs (src/openpi/policies/libero_policy.py:62-68) fills
  it with zeros and image_mask False for pi0.5, so the slot exists in the model.  B fills it with the crop, mask True.  "Instead of the wrist"
  is not built: the wrist view carries the grasp, which K and F7 rely on.
- Pretrained-input shift (stated honestly): pi05_libero never saw this slot unmasked.  pi05_base saw a right-wrist camera in multi-robot
  pretraining, but never a zoomed third-person crop.  The VLM is frozen in the `_ae` recipe, so the crop's tokens come from frozen
  SigLIP / PaliGemma features that were never fine-tuned to this content, and the expert must learn to read them from 6,001 steps.  The
  crop also adds 256 prefix tokens, which shifts every later position's attention statistics.  The F7 / F6 cells (relational language,
  occlusion), which RAA-sL gets from the prefix, may move either way.  Expected risk: K degradation of a few points and a higher step-0 loss.
- Necessity: the crop is centred on the target in every training frame, the original views still predict the target, and no label changes.
  The route's dose is 0, so the principle predicts the network will not rely on it.  B is the test of the "only if" direction.
To write: experiments/raaf_crop.py (K / E per task, LIBERO venv; crop builder shared by convert and server); experiments/raaf_rv.py
`--crop` -> lol/libero_g8v_raacb (feature crop_image 224 x 224 x 3); a LiberoInputs variant mapping crop_image -> right_wrist_0_rgb with
mask True (lifelong_ood/policy/, new file); config pi05_g8v_raas_lb_ae; serving copies raa_b.py (of raa.py) and wa_sb.py (of wa_s.py) that add
meta/crop_uv; a2_probe_f.py patch group `in.crop` (the crop image's prefix positions).  Projection check S-B1 (GPU, LIBERO venv, 20 dev
frames per task): the projected centre must lie within 8 px of the segmentation-mask centroid of the role entity, or the route stops.

## 5. Step 0 (CPU, before any GPU job; outputs outside results/ during the dry run, /coc/flash10/ksingara3/.claude_tmp/raaf/)
- S0-1 geometry: F_d per demo, eligible counts per class, delta histograms (raaf_geom.py).
- S0-2 physics validation of F_d (S-A2).
- S0-3 clip check (S-A1) and the dose of A (variance share of the K delta term in the xyz labels of held frames).
- S0-4 the 0.1 null: compute the servo-kept expert targets for relabelled vs original chunks; expect equality (|diff| < 1e-6) on every chunk
  fully inside [h0, r - 10] and differences only on truncated tail chunks.  If it fails, the derivation is wrong and the plan is revised
  before any training.
- S0-5 A+ generator dry run: 10 attempts per object, yield and step counts.
All via sbatch (`-p overcap -A overcap` or `-p tail-lab -A tail-lab`, `--exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax`).

## 6. Selection on dev states (pre-registered)

### 6.1 PRIMARY: the learned where-share (A2 probe, SR 83 design unchanged)
On each route's checkpoint (A, A+, B; A0 if trained; RAA-sL's existing probe files reused): a2_capture.py + a2_cap_client.py on the SR 83 dev
states (n2_goal, n2_branches_main trials 4-23; tasks K1, K2, K4, K6, N1, N4, N5, N6; the 3rd held query; swap candidates as SR 83), then
a2_probe_f.py and a2_read.py.  Statistic = **learned where-share** on the pooled N + K sets: the effect T carried by patching only the
learned where-input from the counterfactual run, divided by T.  The learned where-input is `in.tok` for A / A0 / A+ and `in.tok + in.crop`
for B.  The servo is NOT part of it (it is outside the learned network).  B = 2000, rng 0, SR 83's clustered bootstrap.
- **PASS** iff the share's 95% CI lower bound >= .70 AND the pooled N + K `in.prefix` share point is below RAA-sL's, recomputed by a2_read.py
  from results/flawed/a2/probe_raasl_{N,K} (no new RAA-sL capture), AND the gate T lower bound > 0.
- Reported always: the share per set (N, K), token-ablation drop, prefix share, top1 / PRn of the 72 writes.
- RAA-sL reference: token share -.007.  A route whose share CI upper bound is < .30 is FAILED for the probe.

### 6.2 SECONDARY: behaviour on existing DEV states (no new states; seen sets, selection only)
- N1-N6, K1-K6: SR 73's 60 states (n2_states trials 4-23 + n2_states_rx k 540-579; branches n2_branches_main.json / n2_branches_rx.json),
  raa_roll.sh as SR 82 (4 jobs per arm).  RAA-sL's SR 82 rollouts on these states are the comparator (results/flawed/raa raasl_*).
- F7 / F6: rg_comp main states (SR 70; F7 tids 0-3 k 300-343, 40 per cell; F6_b tid 8 and F6_a tid 9, k 400-463, 60), wa_s.py full arm on
  tids 0-10 (6 jobs per arm; controls tids 4-6, K1t / K2t tids 10 / 7).  RAA-sL's SR 82 Phase B (results/flawed/wasl) is the comparator.
  noR / noM arms are not rolled on dev.
- A+ctrl through sc_rv serve (rview2) on the same Phase A states; Phase B full cells only.

### 6.3 Winner rule
1. Eligible = PASS on 6.1 AND dev K1-K6 >= .70 AND (route - RAA-sL on K1-K6, point) >= -.05.  The paired CI-based non-inferiority on K at
   -.05 is computed and reported, but it is not an eligibility condition: W5 itself fails it (SR 87, SR 90 AK NOT SHOWN, point +.008) at 60
   states per task, so as a gate it would only measure power.
2. Winner = the eligible route with the highest dev N1-N6 success.
3. Tie-break (dev N within .025): higher dev F6_b + F7-pool mean, then higher where-share point, then the cheaper route (A < A+ < B).
4. If A passes and A0 (trained only then) also passes, the winner is still the route with the best dev N.  The paper's attribution of the
   rise then goes to the servo removal, not the data.
5. **If no route passes the probe**: there is no winner and no confirmatory read.  The 5th fresh set stays unbuilt / sealed for a later
   design.  The reportable result is "the shortcut survives the data at these doses": share per route, dose per route, dev behaviour.  A
   route that beats RAA-sL on dev N (lower bound > 0) without passing the probe is reported as a behavioural observation only, never
   confirmed under this plan.

## 7. Confirmatory read of the winner only, once, on a NEW 5th fresh state set

### 7.1 Fresh states (disjoint by index range and by np.allclose, job-checked)
Used so far: N / K n2 familiar k 0-23, 24-195, 236-275, 500-539, 540-579 (rx), 580-639 (wx, SR 87), 640-699 (wy, SR 90), 700-759 (wz, SR 91);
F7 rg_comp candidates 300-379, rg_comp_x 380-519 (main 384-444), rg_comp_y 520-599 (main 524-583), rg_comp_z 600-679 (main 604-663); F6 rg_comp
400-498, rg_comp_x 500-619 (main 504-563), rg_comp_y 620-699 (main 624-683), rg_comp_z 700-779 (main 704-763); fx / f6o2 0-193; scene 2
1090000 + k (from docs/notes/sr91_plan.md, which lists every range above).
New:
- N1-N6, K1-K6: n2 familiar **k = 760-819** (60), trial t = k - 760, sampler seed 600000 + k, branch seed_base 13000000 + 1000 k
  (experiments/fc_n2_w5.py = a copy of fc_n2_wz.py with K0 = 760 and wz added to the refusal list; fc_n2_wz.py is read, not edited).
- F7: rg_comp_5.py (copy of rg_comp_z.py) candidates **k = 860-939**; F6: candidates **k = 940-1019**.  These skip k 680-859 on purpose: they
  keep every F7 branch seed base (14000000 + 1000 k) clear of SR 91's F6 k 700-763, and every F6 one clear of all earlier F7 / F6 k.  The
  first 4 accepted of each are smoke rows (unused), the next 60 are main.  Seeds sampler 820000 + k / layout 821000 + k (F7), 810000 + k /
  plate 811000 + k (F6).  The build asserts that no seed equals any earlier set's seed.
- Disjointness job (copy of raasw5r_disjoint.py, run as a new file): np.allclose of every new row against every stored n2_states_* and
  rg_comp* row, which must give 0 equal rows.
- Built only after the winner is named and the confirmatory prereg row (tag raafc) is written.  Nobody rolls on these states before that.

### 7.2 Arms (all re-rolled on the new states)
Phase A: winner (served plain: where_own 0, its own config), W5 (raa_w.py --where_own 5, the SR 87 checkpoint and mu table, unchanged), RAA-sL
(where_own 0), rgp (ReGuide-perc, rg5r_roll.sh's recipe copied to a new script).  Phase B: winner full (tids 0-10), winner noR (0-3, 7, 8, 10),
winner noM (0-3, 8), W5 full, RAA-sL full (0-10), frozen I (0-3, 8) and rgp (0-3, 8) through rg_comp_5.py.

### 7.3 Registered contrasts (paired bootstrap B = 2000, rng 0, tasks then states resampled by raa_read.diff; Phase B by wa_read.bootm)
- **C1 PRIMARY**: winner - rgp, N1-N6: NON-INFERIOR iff lower > -.10 (COST iff upper < -.10, else NOT SHOWN).
- C2: winner - RAA-sL, K1-K6: NON-INFERIOR iff lower > -.05 (COST iff upper < 0, else NOT SHOWN).
- C3: winner - RAA-sL, N1-N6: SUPPORTED iff lower > 0, REVERSED iff upper < 0.
- C4: winner - W5, N1-N6: NON-INFERIOR iff lower > -.10.  Does the learned use replace the hand override?
- Dissociation (winner's full / noR / noM, gates and labels as SR 87 / SR 90 / SR 91): D1' full - noR on the F7 pool (tids 0-3) SUPPORTED iff
  lower > 0; D2' full - noM on F6_b (tid 8) SUPPORTED iff lower > 0; D3a noR - full on F6_b and D3b noM - full on the F7 pool EQUIVALENT iff
  the CI lies inside +-.10; DK full - noR on K1t + K2t NON-INFERIOR iff lower > -.05.  Gates on the full winner arm: K floor (K1t + K2t)
  >= .70, unique-name controls (tids 4-6) >= .50, F6_a (tid 9) >= .50.  Gated: DISSOCIATION SHOWN iff D1', D2' SUPPORTED, D3a, D3b
  EQUIVALENT and all gates TESTABLE; UNTESTABLE if the four hold and a gate fails; else wa_read's PARTIAL / NOT SHOWN.
- B1 winner full - RAA-sL full on F6_b SUPPORTED iff lower > 0; B2 winner full - frozen I on F6_b SUPPORTED iff lower > 0, REVERSED iff upper < 0.
- Descriptive: per-cell tables of every arm; K5 alone; bottle / cheese pairings; placed given right grasp; winner - W5 on K and F6_b.
- What counts: the route repairs the pairings by learning iff C1 NON-INFERIOR and C2 not COST.  A learned use that replaces the override
  needs C1 and C4 both NON-INFERIOR.

## 8. Predictions (probabilities; copied into raaf_pred.txt with the row)
Probe (6.1):
1. A passes: 18%.  Point share about .35 (range .05-.80); K share above N share 65%.
2. A+ passes: 32%.  Point share about .55 (.10-.90).
3. B passes: 6%.  B's crop + token share point < .30: 80%.
4. A0 (if trained) passes: 5%.
5. No route passes: 52%.  If A+ fails while A+ctrl - RAA-sL on dev N is >= 0, the reading is "the data help without the channel": 20%.
Dev behaviour (6.2; RAA-sL dev N .878, K .936, F6_b .017):
6. A (servo-free) dev N1-N6 below RAA-sL: 75% (point .74; RAA-c .50, RAA-L .58 suggest servo-free models lose release aim).  A dev K >= .70: 85%.
7. A+ dev N1-N6 within +-.05 of RAA-sL: 55%; above by > .05: 20%.  A+ dev K point >= RAA-sL - .05: 70%.
8. B dev K point >= RAA-sL - .05: 55% (input-shift risk); B dev N within +-.05 of RAA-sL: 60%.
9. A+ctrl dev N above R2-type levels (> .70): 45%.
Confirmatory (7.3), conditional on a winner existing (48%):
10. C1 NON-INFERIOR: 50%; C2 NON-INFERIOR: 30% (NOT SHOWN 62%, COST 8%); C3 SUPPORTED: 22%; C4 NON-INFERIOR: 45%.
11. Gated dissociation SHOWN: 35%; D1' SUPPORTED 90%; D2' SUPPORTED 55% (RAA-sL's own D2 was REVERSED on dev, SR 82).
12. B1 SUPPORTED: 45%; B2 SUPPORTED: 15%.
13. Joint "repairs by learning" (C1 NI and C2 not COST): 45% given a winner, about 22% unconditionally.
Counter-evidence rule (written now): if A+ passes the probe but its dev N1-N6 is more than .10 below RAA-sL, "used" and "useful" have
come apart.  This is reported as the main finding.  A+ can still be selected under 6.3 (only the K conditions exclude it), and the
confirmatory read then tests it as written.

## 9. Kill criteria per route
- A: (k1) after S0-1 / S0-2 fewer than 30% of K1-K6 held frames have an eligible copy, or the K delta term explains < 0.5% of the held xyz
  label variance (dose too small to train on): A is not trained, reported as infeasible at in-support scale.  (k2) S0-4 fails: stop and
  revise.  (k3) more than 5% of copies dropped by the clip check.  (k4) training smoke: step-0 loss NaN or > 2x RAA-sL's .0556, or no state_proj
  in the parameter list.  (k5) probe share CI upper < .30: A closed, A0 not trained.
- A+: (k1) S0-5 yield < 30% for any object after the keypoint fallback; that object is dropped, and with fewer than 2 objects A+ is
  killed.  (k2) fewer than 100 kept demos after 6 CPU-h.  (k3) render / convert mismatch (sc_rv check fails, or the ee max diff between the
  stored and replayed state is > 1 mm).  (k4) probe share CI upper < .30.  A+ctrl is trained with A+ (the same data build), not killed
  separately.
- B: (k1) projection check S-B1 > 8 px median error.  (k2) training smoke step-0 loss > 2x RAA-sL's, or the crop slot is not unmasked in the
  batch (assert on image_mask).  (k3) dev K point below RAA-sL - .10 (input shift too costly).  (k4) probe share CI upper < .30.
- Plan-level: if A and A+ both fail with doses (frames x label change) above the S0 thresholds, the "if" direction of the principle is
  falsified at the tested doses.  If B passes, the "only if" direction is falsified.

## 10. Costs and build order
Unit costs from sacct of past jobs (A40, overcap): training 6,001 steps x about 5.4 s = about 9.0 h per model (RAA-sL: chain links of 4 h,
step 6000 at about 10.3 h wall).  Practice is two redundant 6 x 4 h chains, the second cancelled when the first reaches 6000, so the
budget is about 20 GPU-h per model.  Phase A rollouts about .5 GPU-h per job (4 per arm); Phase B .3-1.0 h per job (about 4 GPU-h per
arm); a2 capture about .25 h x 3 and probe about .25 h x 3 per model.

| item | GPU-h | CPU-h |
|---|---|---|
| Step 0 (S0-1 ... S0-5) | 0.5 (S-B1 projection check) | about 4 |
| A: convert, smoke, train (2 chains), export, probe, dev rollouts | 20 + 1.5 + 6 = about 28 | 2 |
| A0 (conditional on A PASS) | about 28 | 0 |
| A+: generation, render, convert x2 | 2 | 4 |
| A+ train + A+ctrl train (2 chains each) | 40 | |
| A+ probe + dev rollouts (A+ and A+ctrl) | 1.5 + 12 | |
| B: K / E, convert, smoke, train, export, probe, dev rollouts | 0.3 + 20 + 1.5 + 6 = about 28 | 2 |
| Confirmatory: state builds | about 3 | |
| Confirmatory: Phase A (4 arms x 4 jobs) + Phase B (about 33 jobs) | 8 + 20 = 28 | |
| **Total** | **about 168 without A0, about 196 with A0** (about 130 / 150 with single training chains) | about 16 |

Build order (the cheapest decisive step first):
1. Step 0 (CPU): S0-4 (the 0.1 null) and S0-1 ... S0-3 decide whether A is trainable; S0-5 decides A+'s generator.
2. Route A: raaf_rv.py relabel -> lol/libero_g8v_raacf, config, smoke, prereg row tag raaf (written before any main training job), chains.
3. While A trains: A+ generation + rendering (CPU / small GPU), Route B code + S-B1.
4. B training, then A+ and A+ctrl training (tail-lab 4-GPU cap: at most two models' chains at once).
5. Probes as checkpoints land (A first).  If A's probe PASSES, train A0.  If it FAILS, A is closed and its dev rollouts still run (behaviour
   is reported).
6. Dev rollouts for every trained route, then the selection read (experiments/raaf_select.py, one job, once), then the confirmatory row
   raafc, the 5th set build, the confirmatory rollouts, read, tables and note.

Hard rules kept by every step: no python on the login node; every job by sbatch with the exclude list of section 5; no training before the
prereg row; nothing written to results/ before the plan commit; never opening results/l90c/split_sealed.json; SR 91's files and jobs untouched.

## 11. Limits stated in advance
One seed, one checkpoint per route, 60 states per cell.  Route A's shifts are bounded by the predicates (plate / bowl 1-3 cm), so they cannot
reach the 20-40 cm scale of a real mis-pairing.  Route A also changes the architecture (servo removed); A0 exists to separate that.  A+'s
table spots use a new phrase ("the table") and planner actions whose style (saturating P-control, noise .03) differs from human teleop.
B's crop enters a slot pi05_libero never saw unmasked, under a frozen VLM.  The probe is open loop on captured inputs, and its patches
follow SR 83's caveats (super-additive expert patches, bf16 noise .002-.007).  The dev sets are seen; only the confirmatory set is fresh.

## Appendix: audit record (read-only, 2026-11-20)
- Read: paper/DRAFT.md headings; docs/notes/raasl_plan.md, sr82.md, sr83.md, sr87.md, sr90.md, sr91_plan.md; raas_model.py; l90c_config.py
  (AE table, servo configs); c_train_pi05.py (g8v choices); sc_rv.py (chan_state, demo_rel, held_target_rel, convert); fc_n2_tasks.PAIRS;
  fa1g_plan.py (ctrl, layout, gen); pp_n4d.py, fa1l_plan.py, fa1p_sgen.py, p0_retarget.py, b1_demos.py, b1_render.py, sc_pu.py, wa_s.py
  headers; a2_probe.py arguments; openpi libero_policy.py (image slots) and data_loader.py (delta_timestamps); LIBERO base_object_states.py,
  base_predicates.py, site_object.py; flat_stove.xml, wooden_cabinet.xml, wine_rack.xml, plate.xml, put_the_bowl_on_the_plate.bddl;
  norm_stats.json of pi05_g8v_raac; raas_gate.json; raasw6_env.json; a1g_gen_K*.json.
- sacct: 4801561 (train link 4 h), 4806984 / 4814115 / 4814131 (Phase A .45-.56 h), 4806988 / 4816515 / 4816540 (Phase B .34-1.05 h),
  4808253 / 4808368 / 4808369 (a2 capture / probe .24-.26 h).
- Deviation: one audit command included `python3 --version` on the login node (it printed nothing that was used; no script ran).  Recorded
  so the ledger is complete.
- No sbatch job was submitted for this plan.  The geometry numbers above come from the asset XMLs and LIBERO's predicate code.  The per-demo
  bounds are Step 0's job.

## LAUNCH RECORD (Route A, 2026-11-20)
- Step 0 (commit 1ac82916, summaries in results/flawed/raaf/step0/): eligible held rows 90.6% (floor 30%), K delta variance share 3.36% (floor 0.5%), clip drops 4 of 576 = 0.7% (limit 5%), S0-4 servo null passed (interior unclipped chunks max diff 2e-16), physics validation 10 demos per class (bowl and rack raised to 2 cm margin, others 1 cm).  S0-5 planner dry run: 1 kept of 30 (A+ not in this launch).
- **DEVIATION (orchestrator decision, not mine):** the 2.4 "more than 2% outside q01 / q99 -> stop" rule fired as written (18.0% of relabelled action rows, any dim).  The orchestrator read it as the EXCESS over the demos' own baseline on the same rows: originals 18.5% on those rows, excess -0.5%, so it clears (the rule was miscalibrated: held carry rows are 18.5% outside before any relabel; z dim 13.3% is unchanged by the xy shift).  The state token widens (x dim 7.9% -> 25.2% outside state q01 / q99, any dim 16.0% -> 30.9%; normalised max |value| originals 1.45 / 1.33 / 1.14, copies 2.15 / 1.53 / 1.14); accepted since openpi's quantile Normalize does not clip and state_proj is linear.  RAA-c's norm stats are kept as registered, not recomputed.  Recorded in the prereg row notes.
- Ambiguities resolved conservatively: Kdelta variance share taken on the pooled held labels (originals + copies), not copies only (4.9%); the copy ends at min(release, first non-held row) (7 demos); tid in the seeds = K index - 1; last frame for the geometry = raw final state if the success predicate holds there, else the last kept row (11 demos fail both and get no copy).
- Dataset lol/libero_g8v_raacf: 400 -> 972 episodes, 52,181 -> 76,105 frames (572 copies, 23,924 rows); every copy verified equal to its source rows + K delta (job 4818883).  lol/libero_g8v_raac untouched.
- Config pi05_g8v_raac_lf_ae, `--g8v raac_lf` (additive).  Smoke (job 4818884, tail-lab, exp smoke, 20 steps): step-0 loss .0529 (limit 2 x .0556), state_proj present.
- Prereg row tag raaf: ledger/m3_prereg/m3_prereg__libero_goal__raaf__s0__16fc469f.json (job 4818948), predictions experiments/prereg/raaf_pred.txt, written before the chains.
- Chains (raaf_chain.sh, copies of raasl_*): main1-6 on overcap = 4818954-4818959; tl1-6 on tail-lab = 4818960-4818965 (user's tail-lab GPUs running at launch: ctx_full.sh, raasls2_tl2; with the tl chain 3 of 4).  Cancel the second chain when one reaches step 6000.

## ORCHESTRATOR CLARIFICATION OF THE PROBE PASS RULE (2026-11-20, written after the pipeline smoke on RAA-sL and a 20-step B smoke, BEFORE any probe of a trained route checkpoint)
1. Route B: the module's signal is the joint patch `in.tokcrop` (token + crop patched together). The single-patch sum is descriptive only.
2. Route B: the crop image carries the module's target, so the "visual share falls" half of the pass rule uses `in.prefix_nocrop`, the scene and wrist cameras with the crop left clean, against RAA-sL's `in.prefix` .089 (N+K pool). The literal `in.prefix` (crop included) is reported as `pass_literal`, descriptive only. Reason: counting the module's own channel as visual input would fail B by construction, which is not the purpose of the rule.
3. in.crop patches the input crop image (third image block, prefix positions 512-767). Counterfactual side queries carry `meta/raa_crop_uv_*`.
4. All captures for the selection probes run on A40 nodes (`--gres=gpu:a40:1`), as SR 83 did. The B smoke ran on an L40S (replay vs server .012-.020, against .003-.006 on the A40).
Reference values from the pipeline validation (a730dbd9, 2f03d3a8): RAA-sL in.tok -.007 [-.008, -.006]; in.prefix .132 on all rows and .089 on the N+K pool; T 4.475 on the N+K pool.
5. Mapping onto experiments/raaf_probe_read.py (unchanged): for Route B the selection PASS is its `pass_alt_nocrop` field; its `PASS` field is the literal reading (descriptive). For A / A0 / A+ the selection PASS is `PASS`.

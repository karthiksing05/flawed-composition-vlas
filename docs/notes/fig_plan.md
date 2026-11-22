# Figure plan for the ICML 2027 submission (2026-11-21)

Source: user request 2026-11-21, "a lot of illustrations for each of the failures as well as architecture and other stuff" (docs/PLAN_PAPER.md, "Venue"). Style chosen by the user: real LIBERO sim frames (start / grasp / release) with the end-effector path drawn over the frame, plus a small schematic of the cue that drives the wrong choice, shown next to a model or arm that succeeds on the same state.

Planning only. Nothing here is rendered or submitted. Section references: DRAFT = paper/DRAFT.md, SR = docs/SCAFFOLD_RESULTS.md, AP = docs/ARCHITECTURE_PROPOSAL.md.

## 0. Constraints that shape the plan

- ICML two-column. Single column (figure) = 3.25 in wide. Full width (figure*) = 6.75 in. 8 pages of main text, appendix unlimited. Figures must be drawn at final size so type stays at or above 6.5 pt (the existing four use 8 pt base, 6.5 pt smallest).
- DRAFT is far over 8 pages of text already (489 lines). The figure budget below is therefore capped at about 2.6 pages in the main text. Anything not in the main set goes to the appendix.
- All python and all rendering run on compute nodes via sbatch (`-p overcap -A overcap`, or `--partition=tail-lab --account=tail-lab --qos=short`; explicit -p/-A, `--exclude=sky1`, copy the header from experiments/a2f_capture.sh). The login node does edits, git, sbatch/squeue and log reads only. Frame renders need a GPU (EGL); matplotlib composition is CPU-only but also goes through sbatch.
- Never use the sealed split (results/l90c/split_sealed.json) or the sealed fifth fresh-state set named in DRAFT §6.4 (A+0 row). Use the development states (results/flawed/n2_states, n2_branches_main.json) and, where the registered reads used them, fresh sets 1 to 4. Label every shown episode as a development-state episode.

## 1. Page budget

Numbering below is the new numbering. Old figures: fig_atlas (old Fig 1), fig_moved (old Fig 2), fig_arms (old Fig 3), fig_removal (old Fig 4). All are in paper/figs/ and made by experiments/paper_figs.py (captions in paper/figs/captions.md).

### Main text (6 figures, about 2.6 pages)

| # | Figure | Width | Height | Pages | Contents |
|---|---|---|---|---|---|
| 1 | Teaser | figure* | 2.3 in | .34 | One failure (F1, scene 1, frozen pi0.5, cream cheese on the plate) in frames: start, grasp, release with the EE path, the failing release beside the same-state success. Right third: the cue schematic (held object's identity picks the destination) and a thumbnail of the architecture with the failure's site marked (the "where" step). |
| 2 | Failure gallery | figure* | 4.4 in | .65 | Five rows, F1, F1m, F2, F7, F6. Columns: start / grasp / release (fails) / release (succeeds) / cue schematic. Crop frames to the table region (about 3:2) so a tile is 1.2 in wide. The new-layout limit and the moved target go to the appendix (rows A-NL, A-MV). |
| 3 | Atlas data (old Fig 1, fig_atlas) | figure* | 2.7 in | .40 | Existing, unchanged. Table 1 and this figure carry the same numbers. If the page count overruns, drop Table 1 or this figure, not both. |
| 4 | Architecture | figure* | 3.0 in | .45 | Left, the standard distributed pi0.5 layout. Right, the de-distributed proposal. Failures marked on both. Spec in section 4. |
| 5 | Module removal and where-module (old Fig 4, fig_removal, re-cut) | figure* | 2.6 in | .38 | Re-cut fig_removal from 2x2 at 5.0 in tall to 1x2 at 2.6 in (D1 resolver R, D2 memory M, left RAA-L, right RAA-sL). A third panel adds the SR 91 W5 where-module read (success with and without the servo/where-module, on the N1-N6 pool and K1-K6). All numbers already in results/flawed; no new rollouts. |
| 6 | Probe and dose by servo grid | figure* | 2.5 in | .37 | Left (60%): probe diagram, section 4. Right (40%): the 2 x 3 dose-by-servo grid of DRAFT §6.4 with the A+0 cell hatched and labelled "pending". Table 6.4a stays in text or moves to the appendix. |

Totals: about 2.6 pages. If the text needs more room, demote Fig 3 (atlas) to the appendix and keep Table 1; that saves .40.

### Appendix (unlimited; plan for about 18 pages of figures)

| # | Figure | Width | Contents |
|---|---|---|---|
| A1 | Scenes and task set | figure* | Scene 1 (LIBERO-Goal kitchen) and scene 2 (LIVING_ROOM_SCENE5) as sim renders with labelled entities, the K1-K8 / N1-N8 table as a K x N pairing grid with trained and new cells shaded. Source frames: results/flawed/n2_render/*.npz, results/flawed/n1_render/*.npz, pp_f1_frames.npz; or fresh renders. |
| A2 to A9 | One full-size figure per failure: A2 F1 scene 1, A3 F1 scene 2, A4 F1m, A5 F2, A6 F7, A7 F6, A8 new layout, A9 moved target | figure* each (A3 may be two stacked) | Per-failure spec in section 2: 6 to 10 columns of frames (start, approach, grasp, lift, mid-carry, release, end), two to three arms stacked, EE path overlaid, an outcome stamp, the cue schematic and a 3-panel EE trace (x, y, z against step with phase bands). |
| A10 | Cue majority rule, scene 2 | figure (single col) or figure* | Schematic of {nearer, salient, trained} and the plate chosen in each configuration, with the §4.1 numbers (SR 57-61): .912 pooled, trained plate outvoted 153 of 153, salient .974, nearer .752. Pure vector; the data are in DRAFT §4.1. |
| A11 | Arms of the fix (old Fig 3, fig_arms) | figure* | Existing. Demoted from main. |
| A12 | Moved target numbers (old Fig 2, fig_moved) | figure* | Existing. Demoted; A9 is its frames companion. |
| A13 | Probe profile | figure* | Layer profile of the residual-stream patch for pi0.5 and RAA-sL (effect 1.66 to 1.53 at layers 0-12, 1.21 at 13, .10 at 14), plus the write-level bars of the 72 writes and the band-by-set cells of DRAFT §4.8. Data: results/flawed/a2/a2_read_pi05.json, a2_read_raasl.json, br7_a2.json. No new compute. |
| A14 | Where-module trace | figure* | EE path with and without the servo (RAA-sL against RAA-sL-W5), release error histograms from sr82_diag and SR 87/91 releases. Frames from section 3 jobs R9/R10. |
| A15 | Route table as bars | figure (single col) | Table 6.4a and 6.4b as a dot plot (where-share and N1-N6 success). Existing json under results/flawed/raaf. |
| A16 | Frame gallery, extra seeds | figure* | Three further episodes per failure, selected by the same rule as the main figure (section 3), as a thinner grid. Guards against a cherry-picked look. |

Each appendix figure gets its own caption in paper/figs/captions.md (extend that file; do not create another).

## 2. One entry per failure

General rules for every entry. Frames are fresh rollouts that also log their outcome (section 3). The overlay draws the EE path from the logged `ee` rows. Reach segment (gripper open) is dashed, carry segment (held flag on) is solid, and the carry colour follows the outcome (red for a failure, green for a success). Grasp = circle, release = square, both at the EE position of the first held row and the last held row. A thin dashed ring marks the instruction's target region and a thin dotted ring marks the cue-implied (wrong) region. The tags "FAIL" and "OK" go on the release tiles. Selection rule (fixed before looking): take the lowest-index trial in the entry's state list for which the failing arm fails with the failure's own signature (below) and the comparator arm succeeds, from the same fresh run. Report the registered rate of the cell in the caption.

Task ids: scene 1 uses suite n2_goal (results/flawed/n2_tasks.json: K1 bowl on plate = 0/1, K2 bowl on stove = 2/3, K3 bowl on cabinet = 4/5, K4 bottle on cabinet = 6/7, K5 bottle on rack = 8/9, K6 cheese in bowl = 10/11, N1 cheese on plate = 16/17, N2 cheese on stove = 18/19, N3 cheese on cabinet = 20/21, N4 bottle on plate = 22/23, N5 bottle on stove = 24/25, N6 bottle in bowl = 26/27; even = familiar layout, odd = new layout). Scene 2 uses results/flawed/s2/s2_tasks.json (0 K1 white mug on the left plate, 1 K2 yellow-and-white mug on the right plate, 2 N1 white mug on the right plate, 3 N2 yellow-and-white mug on the left plate, 4 KC, 5 NC).

### F1. The place follows a cue (DRAFT §4.1)

- Scene 1. Task N1_fam (tid 16, "put the cream cheese on the plate"), secondary N2_fam (tid 18). Frozen pi0.5 (openpi pi05_libero) grasps the cheese, then carries it to the bowl (its trained target) and releases there. The cue is the held object's identity. Registered: atlas E +.450 [.28, .61] (DRAFT Table 1). Signature: held flag on and release within 3 cm of the bowl instead of the plate.
- Comparator A (same weights, cue removed): pi0.5 with the front end tvpb (role views + held-object box; .938 vs .454, SR 55), or the ReGuide reimplementation with perceived poses (.929). Use ReGuide (rgp) in the main row because it is also the comparator in F2, and tvpb in A2. Comparator B in A2: GR00T N1 (breaks too, +.500).
- Frames: start (cheese, plate, bowl all visible, instruction text strip), grasp (circle), mid-carry, release (square). Overlay: EE path plus the two rings (plate = instructed, bowl = cue-implied).
- Cue schematic F1: a held-object icon (cheese) with an arrow to the "trained target" (bowl) and a crossed-out arrow to the instructed target (plate); caption chip "held object -> trained destination".
- Episode source: pp_atlas_roll.sh harness (MODEL=pi05, TIDS=16+18, pp_serve_purify.py --mode intact --seed 13 and pp_atlas_client.py --suite n2_goal --fresh results/flawed/n2_states --branches results/flawed/n2_branches_main.json --videos 0 --bind track); rgp arm from rg_roll.sh / rg_comp_roll.sh (SR 52/55 harness, experiments/rg.py). First 10 trials in the branches file for each tid.

### F1, scene 2 (DRAFT §4.1 "The scene-2 cue"; appendix A3, one row in main Fig 2 only if the page budget allows)

- Scene 2, N1 (tid 2, "put the white mug on the right plate") on frozen pi0.5; 0 of 720 new-pairing episodes succeed. Signature: first grasp correct (or the wrong mug on N2) then place on the nearer / start-side plate. Comparator: ReGuide with perceived poses on pi0.5 (.708 on N1+N2, SR 68; N2 1.00) from s2rg_roll.sh. Second row: pi0 (start-side rule, .993 of 145 placements).
- A3 layout: (a) the standard layout, (b) the two mugs swapped (287 of 287 placements go to the start-side plate), (c) the plates moved (nearer plate wins). Three rows, same four frame columns, with the majority-rule schematic (A10) at the right.
- Source: s2_roll.sh (MODEL pi05, pi0; ARMS I, tv, tvp; TIDS 2+3), s2rg_roll.sh for ReGuide; swapped and moved layouts from the s2cue*_roll.sh scripts (results/flawed/s2cue2_roll .. s2cue5_roll for the layouts).

### F1m. The held object's outline picks the carry (DRAFT §4.2)

- Scene 1, tid 22 (N4, bottle on the plate) and tid 26 (N6, bottle in the bowl), frozen pi0.5. The bottle's outline starts K4's high carry toward the cabinet, whatever the target. Signature: EE height above the trained-carry threshold during the carry, then a miss. Registered: high-carry share .225 and N4+N6 .625 (arm g). pi0.5 succeeds .04 after a high carry, .66 after a low one.
- Comparator: the same weights with the held bottle painted as its box (arm sb): high-carry share .062 and N4+N6 .812 (SR 53). Optional third row: GR00T high carry on at least .78 of grasps, fixes UNRESOLVED (shown only in A4).
- Frames: start, grasp, mid-carry (shows the high arc), release. Overlay: EE path with z-profile inset (z against step with the carry threshold), since this failure is about height; the side-view schematic of a bottle-shaped outline against K4's outline carries the cue.
- Cue schematic F1m: two silhouettes (K4 bottle outline in the trained task, the held bottle in N4/N6) with a "high arc" arrow; below, the same bottle painted as a box with a "low arc" arrow.
- Source: f1m_cue_roll.sh (SET=main or fresh; TIDS 22+26; ARMS g+sb+sf), client experiments/f1m_cue.py; results/flawed/f1m_cue/{main,fresh}/pi05_g_22 etc. already hold the tracked outcomes (track.npz has ee and held rows) to pre-screen states.

### F2. The habitual object is grasped (DRAFT §4.3)

- Scene 1 new pairings, pi0 (openpi pi0_libero, delta actions; serve_guided.py --mode ours with BACKBONE_CONFIG=pi0_libero) and SmolVLA (lerobot/smolvla_libero; same server with BACKBONE_CONFIG=smolvla_libero, its own venv, lifelong_ood/policy/backbone_smolvla.py). Tasks N1 (tid 16, cheese on plate) and N4 (tid 22, bottle on plate): the model grasps the bowl (the habitual object of the scene's most common task) or nothing, and the episode ends at the grasp. Registered: grasp excess +.650 [.39, .88] (pi0) and +.833 [.67, .96] (SmolVLA); .679 and .829 of new-pairing episodes end at the grasp (SR 65). Signature: the first held object is not the instructed object.
- Comparator: ReGuide approach transport on the same two models (.608 and .604, SR 65), from s1_cross rollouts (results/flawed/s1_cross). For a no-extra-parts comparator use frozen pi0.5 (grasps the named object at .950-.992).
- Frames: start, approach (the path heads to the wrong object), grasp (wrong object, circle on it), release. Overlay: the path plus an outline around the instructed object and a dashed outline around the grasped object.
- Cue schematic F2: scene thumbnail with the grasp attractor (bowl) highlighted and the instructed object (cheese) outlined; the label "scene picks the object".
- Source: pp_atlas_more_roll.sh (pi0 / SmolVLA in the five-model atlas) and rg_roll.sh / s1_cross jobs for ReGuide.

### F7. Relational and negated language is ignored (DRAFT §4.4)

- Scene 1 with two identical bowls (suite built by experiments/fx_tasks.py; rgc suite for the composite). Instruction "put the left bowl on the plate" (relL) or "put the bowl that is not on the plate on the stove" (neg). Frozen pi0.5 picks the same bowl as for the ambiguous instruction "put the bowl on the plate" on 92.5-100% of states, so it is right on about half. GR00T N1 shows the same (1.00 SP). Signature: the grasped bowl is the one the ambiguous run picks and not the referent. Registered success: pi0.5 .40 (SP) and .20 (NG); with the resolver 1.00 and .775 (SR 45.2).
- Comparator: the composite RC (resolver F7R + role view + ReGuide transport) from rg_comp_roll.sh (arm rc, tids 0-3 F7 states, BR7 branches), or f7r alone (the resolver and paint, no ReGuide). The sim frame on the comparator shows the other bowl painted magenta in the model's views, so show both the sim render and the painted view (two tiles).
- Frames: start (two bowls, left one outlined), grasp, release. Overlay: path plus the referent outline (instructed bowl) and the grasped-bowl outline.
- Cue schematic F7: token strip "put the [left] bowl on the plate" with the word "left" in the cue colour, feeding a resolver box that selects one of two bowls, and a second strip with the model's real choice (the ambiguous choice) and a "no effect" bar. For negation: "not on the plate" with the bowl on the plate crossed.
- Source: rg_comp_roll.sh MODEL=pi05, ARMS I+rgp+rc, TIDS 0+1+2+3 (and pi0 from SR 45.3 if wanted for the appendix: fx_more).

### F6. A target seen once and then hidden (DRAFT §4.5)

- Scene 1 K1 "put the bowl on the plate" with the plate displaced, harness f6o2 (experiments/f6o2_tasks.py, rg_comp tids 8-10 for the F6 states). Conditions: a = plate never covered, b = covered after the first query, c = covered from the start, d = cover beside the plate. Frozen pi0.5: .38 [.28, .48] cost of covering (d minus b); GR00T .60. Signature: the plate is covered after the first query and the bowl is released away from the plate (the pose is lost).
- Comparator: (i) the same pi0.5 on condition a (the plate stays in view), and (ii) RC with the pose memory slot (SR 70: .800 against .517 intact, slot alone .817). Three tiles per row: release, release with cover, release with cover and memory.
- Frames: start (plate visible), the first query (plate visible, cover not yet on it), mid-carry (cover on the plate, plate hidden), release. Overlay: path plus a ghost of the plate at its remembered position in the memory arm (drawn in the sim render, not in the model views, and labelled as the memory overlay).
- Cue schematic F6: a timeline with the plate seen at t0, hidden at t1, and the model's target estimate decaying, against the slot holding it constant.
- Source: rg_comp_roll.sh MODEL=pi05, ARMS I+rc (+ I with the a condition from f6o2 client), TIDS 8+9 for the covered F6 states (BR6), plus f6o2_roll for condition a.

### The new-layout limit (DRAFT §4.6; appendix A8)

- Scene 1, tid 17 (N1_new) and tid 27 (N6_new) on pi0.5, with the scripted servo and P_min running the same states: all four systems fail on the same states, which is the point of the figure (C1 fails, so the failure belongs to the task). The demo's fixed top-down grasp holds on 0 of 10 cheese and 0 of 10 bottle states.
- Frames: start in the new layout, pi0.5 at its grasp attempt (the object slips or is missed), the servo at its grasp attempt, and the demo grasp pose drawn as a ghost gripper. Overlay: the EE paths of both arms (grey for pi0.5, black for the servo) and a failure stamp on each.
- Cue schematic NL: the two start regions (familiar, new) as rectangles on the table with the object's start inside the new one and the trained region marked.
- Source: pp_atlas.py modes `pmin --seed 10 --tid 17` and `servo --tid 17` (these already log held, obj, tgt, ee, quat; so EE paths exist for the scripted arms in results/flawed/pp_atlas*), pp_atlas_roll.sh for pi0.5 on odd tids.

### A moved target (DRAFT §4.7; appendix A9)

- Scene 1, K1 "put the bowl on the plate" with the plate .141-.296 m from the trained centre (cell T_far; 40 states, mt_tasks.py). Rows: frozen pi0.5 (follows the plate, .975 success, 0 of 40 release at the trained position), R2 (the narrow retrain pi05_g8v_rview2_ae_tl6000; releases at the trained plate position on 31 of 40, median 20.2 cm off), RAA (neither, .100), ReGuide with perceived poses (the transport follows the plate). Signature for the failing row: the release lies more than 3 cm closer to the trained plate position than to the actual plate (SR 81's rule).
- Frames: start with the plate drawn at its displaced position and a ghost ring at the trained position (from mt_tasks.py's training.centre), grasp, release. Overlay: path, with the release point marked on every row and the line from the release point to the trained position.
- Cue schematic MV: the 2 cm demo box on the table with the plate far outside it, two arrows (follow the plate: frozen; go to the trained spot: R2).
- Source: mt_client.py (`--arm I|r2|raa|rgp`; results/flawed/mt_roll holds the tracked outcomes `{I,r2,raa,rgp}_T_far_main*_track.npz` with ee, held; use for pre-screening). R2 server: sc_rv.py serve --arm rview2 --ckpt /coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_g8v_rview2_ae_tl6000 (mt_plan.md, line 39); client with --seg.

## 3. Tooling inventory

Bounded searches only (ls on specific dirs and `grep -rI --exclude-dir=...`); no data dirs searched.

### Existing rollout code that already logs positions or frames

| Script | What it saves |
|---|---|
| experiments/libero_client.py | `--videos N` writes an mp4 of the agentview frames for the first N episodes per task (imageio, at 224 px, `task<tid>_trial<k>_<success|failure>.mp4`); `--record` (and `--light` for no images) writes `<out>_rec/task<t>_trial<k>.npz` with policy-input frames, 8-D states (eef pos, axis-angle, gripper), actions and outcome; `--objects` / `--all-objects` add object poses. Policy env renders at 256 px. |
| experiments/pp_atlas.py (class `Rec`) and pp_atlas_client.py | After every step from WAIT on: `held` flag, object xyz, target xyz, **ee xyz**, object quat, gripper action. Saved to `<out>_track.npz`. No frames. This is the logging used by the break atlas, scene 2 (s2.py), rg_comp, f1m_cue, mt_client, wa_*/raasw5 clients. |
| experiments/pp_f1_frames.py | Renders frames from stored init states with the agentview projection of every entity (camera matrix from robosuite, function `project`), no policy. Output results/flawed/pp_f1_frames.npz and pp_f1_frames_check.png. Reuse `project` to put the EE path on a frame. |
| experiments/fig_strips.py | Builds a filmstrip from the client's saved mp4s using PyAV and matplotlib (openpi venv). The closest existing precedent. |
| experiments/fc_d1_frames.py, fc_d2_frames.py, fc_n1_render.py, pl0_render.py, s1_render.py, b1_render.py | State renders for the F6 / F7 / N1 harness checks (no policy). |
| experiments/sc_rv.py, sc_rv_w.py, raa_b.py, raa_w.py, wa_s*.py, wa_sw5*.py, raafp_rv.py | Servers and role-view wrappers (RAA family, W-arms). They log per-query channel diagnostics (`--chan-diag` jsonl) but save no frames. launch_pack.py only forwards `--record/--light` to the client. |

No script has a "save_frames" option; the only frame savers are libero_client.py's `--videos` / `--record` and the render scripts above.

### What results already hold (ls only)

- No saved frames or videos for any rollout of the atlas, scene 2, rg_comp, f1m_cue, mt or raasw5 runs. They were run with `--videos 0`. The result dirs there contain per-episode `.json` and `_track.npz` (EE, held, object and target positions, gripper action) but no images. Examples: results/flawed/pp_atlas (516 files), results/flawed/mt_roll, results/flawed/f1m_cue/{fresh,main}, results/flawed/raasw5/*.
- `*_rec` dirs with image arrays exist only under results/robot (at least 10 seen, older "practice" runs such as b_k1_practice_52_role_rec); they belong to the earlier BindVLA line, not to the atlas states, and they are not usable for this paper.
- Frame-like assets that do exist: results/flawed/pp_f1_frames.npz (+ _check.png), results/flawed/n2_render/*.npz, results/flawed/n1_render/*.npz, results/flawed/d1_frames.json, d2_frames.json, f6o2/check_M2.png, n2_render/check_K1_fam_new.png. These are no-policy state renders (scene layouts, F6 cover checks) and are good for Fig A1 and the start tiles. Their init states do not give the rollout's later frames.

### What a new render script needs

Name: experiments/fig_roll_client.py (LIBERO venv, py3.8, `PYTHONPATH=$ROBOT/third_party/libero`, `MUJOCO_GL=egl`). It must be a thin wrapper that leaves the policy's inputs untouched, in the style of pp_atlas_client.py: import the arm's client module (fc_n2_client, f1m_cue, mt_client, rg_comp client, s2.py), wrap `OffScreenRenderEnv` in a subclass of `pp_atlas.Rec`, and add:
1. a display render at every REC_EVERY steps (5) and always at the held-onset row, the held-end row and the final step: `env.env.sim.render(camera_name="agentview", height=512, width=512)` flipped as the client flips; this reads the simulator without touching the 256 px policy view;
2. the `ee` and `held` rows already in `Rec`, plus the EE pixel (row, col) from `pp_f1_frames.project` for each stored frame (store the camera matrix once);
3. the outcome (success flag, step count, and the failure signature computed from the track: release-to-instructed-target and release-to-cue-region distances, EE z-peak during carry, grasped-object name);
4. a compact npz per episode: frames uint8 (about 20 per episode, about 16 MB), ee, held, obj, tgt, gripper action, step of each frame, outcome, config (arm, model, tid, trial, seeds, checkpoint).
Output under results/flawed/fig_frames/<failure>/<arm>_<tid>_<trial>.npz (new dir, no overlap with any registered result dir). Skip finished outputs, as the roll scripts do.

Roll scripts to copy: experiments/pp_atlas_roll.sh (server start, port retry, loop over TIDS; pi05 and groot), rg_comp_roll.sh (rg arms, F7/F6 harness), s2_roll.sh (scene 2), f1m_cue_roll.sh (arms g, sb, sf), and the mt arm lines in mt_plan.md. New wrapper: experiments/fig_roll.sh that takes FAIL (F1 | F1s2 | F1m | F2 | F7 | F6 | NL | MV), MODEL, ARM, TIDS and a TRIALS list (first 10 trials of the branches file), serves the model exactly as the source script does and runs fig_roll_client.py in place of the old client. Each job is one server with several (arm, tid) clients, as in the existing scripts. Header from experiments/a2f_capture.sh: `#SBATCH -p overcap -A overcap --gres=gpu:a40:1 -c 8 --mem=64G` plus the repo's `--exclude` list.

Gotcha (memory: LIBERO session state): an episode is not reproducible from the init state alone, because MuJoCo's solver warm-start and fixture state depend on the resets since `env.seed` (libero_client.py comments on r18 branches). A re-run of a registered state can give a different outcome from the registered rollout. So frames must come from fresh rollouts that log their own outcome, and the shown episode is chosen by that outcome and not by the old json. Use the old `_track.npz` files only to pre-screen which states fail often. Each panel's caption carries the cell's registered rate, and the appendix A16 shows further episodes chosen by the same rule, not the best-looking ones. Keep the client's seed (7) and the server seed (13) so the draw matches the registered one up to that state effect.

### Servable models (checkpoints and export paths as the scripts reference them)

| Model | Server and checkpoint |
|---|---|
| pi0.5 frozen | openpi pi05_libero; `pp_serve_purify.py --model pi05 --mode intact --comp '' --seed 13` or `serve_guided.py --mode ours --seed 13`; PyTorch checkpoint /coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi05_libero_pytorch (JAX params in .../pi05_libero/params). Python: /coc/flash10/ksingara3/robot/openpi/.venv/bin/python. |
| pi0 | openpi pi0_libero (delta actions, extra_delta_transform); `serve_guided.py --mode ours --seed 13` with `BACKBONE_CONFIG=pi0_libero`. |
| GR00T N1 | ar0s/groot_libero (GR00T-N1.5-3B, lifelong_ood/policy/backbone_groot.py); `pp_serve_purify.py --model groot`; venv /coc/flash10/ksingara3/robot/groot/.venv/bin/python (see groot_*_roll.sh). |
| X-VLA | lerobot/xvla-libero (backbone_xvla.py); `serve_guided.py` with `BACKBONE_CONFIG=xvla...`; the server sends absolute EE targets (metadata `{"control": "absolute"}`), so ReGuide transports cannot drive it (DRAFT §3). Atlas rows only. |
| SmolVLA | lerobot/smolvla_libero (backbone_smolvla.py); `serve_guided.py` with `BACKBONE_CONFIG=smolvla_libero`; own venv. |
| RAA family | /coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/: pi05_g8v_raa_ae_tl6000 (RAA), pi05_g8v_raao_ae_tl6000 (RAA-o), pi05_g8v_raac_ae_tl6000 (RAA-c), pi05_g8v_raas_ae_tl6000 (RAA-s), pi05_g8v_raa_l_ae_tl6000 (RAA-L), pi05_g8v_raas_l_ae_main6000 (RAA-sL), pi05_g8v_raas_lp_ae_main6000, pi05_g8v_raas_ls2_ae_tl6000, pi05_g8v_raac_l{b,f}_ae_tl6000 / main6000 (RAA-F routes), R0/R2: pi05_g8v_plain_ae_tl6000, pi05_g8v_rview2_ae_tl6000 (R2), pi05_g8v_rsup/rbox/rrel/rwrist_ae_tl6000. Serve with `sc_rv.py serve --arm <rview2|rsup|rbox|rview> --ckpt DIR`, `raa.py serve --arm rrel|raa --ckpt DIR` (raa_w.py / raa_b.py copies), `wa.py serve` (wa_s.py, wa_sw5.py for the W-arms). |
| W5 (RAA-sL-W5, where-module) | same checkpoint as RAA-sL, serving only (wa_sw5.py); registered reads in results/flawed/raasw5, raasw5r. |
| ReGuide with perceived poses | experiments/rg.py controller over any of the above, learned U-Net segmenter (sc_seg) and rendered depth; arms `rgp` (rg_comp_roll.sh, rg_roll.sh, mt_client.py `--arm rgp`). |

Python venvs: client = /coc/flash10/ksingara3/robot/openpi/examples/libero/.venv/bin/python (py3.8, LIBERO, imageio); servers = openpi .venv; matplotlib and PyAV exist in the openpi .venv (fig_strips.py runs there). Composition/plotting scripts run in the openpi .venv through sbatch, same as paper_figs.py.

## 4. Schematics and diagram tooling

### Pipeline

One pipeline: **matplotlib vector drawn in scripts, run through sbatch on a compute node**, in the style of experiments/paper_figs.py (same `rcParams`, same `save(fig, name)` writing PDF and PNG into paper/figs/, fonttype 42). Reasons: the four existing figures already live there; every number is read from results/flawed json; no LaTeX toolchain to install on the cluster; the PDFs drop into the ICML LaTeX port unchanged. TikZ is not chosen because the sim frames are rasters that must share a coordinate system with vector overlays (paths, rings), which matplotlib does in one figure. Raster frames are embedded at 220 dpi (the existing `savefig.dpi`), target under 1.5 MB per figure.

Shared module (step 0, before the two work packages start, owned by nobody afterwards): experiments/fig_style.py with the rcParams, palette, `save`, `frame_tile(ax, img, crop)`, `draw_path(ax, ee_px, held, outcome)`, `stamp(ax, "FAIL"|"OK")`, `box(ax, xy, w, h, label, kind)` and `arrow`. Both packages import it and never edit it. The cue schematics sit behind one interface: `draw_cue(key, ax, compact=False)` in experiments/fig_cues.py (package b), with keys `F1, F1s2, F1m, F2, F7, F6, NL, MV`, each drawing into an axis whose limits are (0,1) x (0,1) with equal aspect. Package a imports `draw_cue` (and uses a grey placeholder box until it lands).

### Palette and type

- Type: DejaVu Sans (matplotlib default; the existing figures use it), 8 pt base, 7 pt tick and legend, 6.5 pt smallest (annotations), panel letters 9 pt bold; line widths 0.8 pt (paths 1.6 pt with a 3 pt white halo).
- Models (as paper_figs.py `MODEL`): pi0.5 #8c8c8c, pi0 #E69F00, GR00T N1 #009E73, X-VLA #CC79A7, SmolVLA #D55E00. Our arms: the blue family of paper_figs.py `ARM` (R0 #c6dbef, R2 #6baed6, RAA #2171b5, RAA-o #08519c, RAA-c #4292c6, RAA-s / RAA-sL #08306b); ReGuide black #000000.
- Outcome: success #0ca30c, failure #d03b3b (the values in fig_strips.py), ink #0b0b0b, muted ink #52514e, grey #898781, surface #fcfcfb.
- Cue accent (new, used for the cue and only the cue): #7b3fa0 (purple). Target ring = dashed ink, cue-implied ring = dotted purple. Frame borders carry the model colour (2 pt); the path colour carries the outcome, so the two never share a hue meaning. Everything must still read in greyscale (dash and marker shapes differ: reach dashed, carry solid, grasp circle, release square).
- Module boxes in the architecture: VLM prefix / frozen weights light grey #e6e6e6, action expert #c6dbef, new modules (R, M, role channel, where-module) white with a #08306b outline; failure markers are red #d03b3b numbered tags (F1, F1m, F2, F6, F7).

### Architecture figure (Fig 4) spec

Two panels, same scale, shared legend. Use the modules named in DRAFT §5.3 and §6.1-§6.3 and AP.

- Left, **standard distributed pi0.5**: images (agentview and wrist) -> SigLIP -> VLM prefix (18 layers) together with the language tokens, the prefix KV cache -> the flow-matching action expert (18 layers) that attends to all of it, producing the 50-step chunk of 7-D actions, 5 steps executed per query. Draw the served pi0.5's state as a dashed, greyed input labelled "not read in the served model" (DRAFT §4.2, SR 53). Mark the layer-14 keys and values band as the narrow read site (DRAFT §4.8). Failure tags: F2 and F1/F1m on the action expert's output (nothing in the net owns where the arm goes), F7 on the language path into the prefix, F6 on the image path (no memory across queries).
- Right, **de-distributed proposal**: the same frozen network, plus: the parse step -> resolver R (relation to one entity: "the bowl on the left" to an entity id), the pose memory slot M (last estimate of an entity a cover or a view loses), the role channel (target offset = target - (eef + in-hand offset), entered as a state token, RAA-c), and the where-module (the servo that owns the xy of the carry and the release gate while the role is the target, RAA-sL-W5, SR 87), with the closed-gripper command W2. Show the network keeping the grasp and the motor skill. Marks on the right panel: removing R returns F7, removing M returns F6, the where-module owns F1/F1m (these come from SR 79, SR 82, SR 84-91). Note that the trained network does not read the state token (patching share -.007; DRAFT §6.3), so draw the token's arrow into the network dotted.
- A small "ReGuide transport" inset is not drawn; it is prior art.
- A column of three arrows labelled "(a) grasp, (b) carry, (c) release" ties each tag to the phase of the episode used in Figs 1 and 2.

### Probe diagram (Fig 6, left) spec

Activation patching in DRAFT §4.8 and §6.3. Two runs side by side on the same recorded inputs: the clean run (instruction names target X) and the counterfactual run (instruction names target Y). A row of 72 write sites (attention and MLP in each of 18 VLM layers and 18 action-expert layers) drawn as a 2 x 36 grid of small squares. One site is highlighted; an arrow takes its activation from the counterfactual run into the clean run, and the output chunk displacement toward Y is the effect. Beside the grid: the share of the positive effect per band (bars: early VLM, late VLM, early expert, late expert) as in §4.8, and for RAA-sL the channel inputs (servo input + state token) at .871 [.834, .914]. Mark the layer-14 cliff in the residual-stream line. Include the A2 where-share definition in one line: share = effect of patching the where-input (token, crop) / total effect.

### Dose by servo grid (Fig 6, right) spec

A 2 (servo kept / off) by 3 (dose none / small relabel 1-9 cm / large planner 19-38 cm) grid, each cell labelled with its route and where-share: kept-none RAA-sL -.007; kept-small not run (blank, cross-hatch grey); kept-large A+ -.007; off-none A0 -.001; off-small A .274 [.239, .314]; off-large A+0 pending (hatched, label "pending"; confirmatory test is the sealed fifth fresh set). Fill by value on one sequential ramp (0 to .70, the pass bar marked), route B as a footnote strip below (no dose, crop only, .022). The pending cell stays hatched and unlabelled by number until A+0's probe lands; update then and re-run only package b's grid script.

## 5. Execution split

Two work packages with no file overlap. Step 0 (an hour, one agent): write experiments/fig_style.py and the `draw_cue` stub in experiments/fig_cues.py signature; after that neither package edits fig_style.py.

### (a) Sim-frame renders and per-failure frame panels

Owner of: experiments/fig_roll_client.py, experiments/fig_roll.sh, experiments/fig_select.py (rule-based pick of the episode per entry), experiments/fig_frames.py (frame panels A2-A9 and A16 and the tile builder), experiments/fig_gallery.py (Fig 1 and Fig 2 composition, imports `draw_cue` from package b), experiments/fig_scenes.py (A1, from the existing renders), results/flawed/fig_frames/** (all outputs), paper/figs/frames/* (panels), paper/figs/fig_teaser.{pdf,png}, fig_gallery.{pdf,png}, figA1 and figA2 to figA9, figA16, figA14 frames.

Jobs (one A40 each, about 10 to 15 min including the server start and ten episodes of 300 steps per arm; GR00T and pi0 slightly longer):

| Group | Jobs | Notes |
|---|---|---|
| F1 scene 1 | 2 | pi0.5 I (tids 16+18), pi0.5 rgp (16+18); GR00T I for A2 as a third |
| F1 scene 2 | 3 | pi0.5 I, pi0.5 rgp, pi0 I (tids 2+3), plus 2 layout jobs (swapped, moved) for A3 |
| F1m | 2 | pi0.5 g and sb (tids 22+26) |
| F2 | 4 | pi0 I, pi0 rgp, SmolVLA I, SmolVLA rgp (tids 16+22) |
| F7 | 3 | pi0.5 I and rc (rgc F7 states), GR00T I |
| F6 | 3 | pi0.5 I with condition a and b, pi0.5 rc |
| New layout | 2 | pi0.5 (tids 17+27); pp_atlas `pmin` and `servo` for the scripted arms (CPU/GPU light) |
| Moved target | 4 | I, r2, raa, rgp on T_far |
| Where-module (A14) | 2 | RAA-sL and RAA-sL-W5 on N1-N6 |
| Total render jobs | about 27 | about 5 to 7 GPU-hours |
| Select + compose | 4 | CPU, sbatch: fig_select, fig_frames, fig_gallery, fig_scenes; about 15 min each |

Order: (1) a smoke job per harness, 2 episodes, checking the frame count, EE overlay alignment and file size; (2) the render jobs; (3) fig_select.py applies the selection rule and writes paper/figs/frames/selection.json (the chosen trial and outcome per entry, with the cell's registered rate); (4) composition. If the failing arm never fails with the signature in the first 10 trials of an entry, draw 10 more and state that in the caption.

### (b) Vector schematics

Owner of: experiments/fig_cues.py (all eight `draw_cue` schematics and the standalone cue figures), experiments/fig_arch.py (Fig 4), experiments/fig_probe.py (probe diagram and grid, Fig 6; also A13 and A15), experiments/fig_removal2.py (Fig 5 re-cut; leaves paper_figs.py unedited), experiments/fig_majority.py (A10), paper/figs/fig_arch.*, fig_probe.*, fig_removal2.*, fig_cues_all.*, figA10, figA13, figA15, paper/figs/cues/*.

Jobs: all CPU, matplotlib only, through sbatch: about 7 jobs (arch, probe plus grid, removal re-cut, majority, cue sheet, A13, A15), each under 10 min on a CPU node, no GPU. Data read from results/flawed json only: raaf probe json, a2_read_*.json, raasw5r_read.json, the removal-read json behind paper_figs.py fig_removal.

Dependencies: package a imports `draw_cue` from package b (stub first, real later; both agree on the axis contract above). Package b has no dependency on a. A cue-sheet figure (fig_cues_all) lets the user review all eight cue schematics alone before they enter the gallery.

### Merge

After both: one agent updates paper/figs/captions.md (new numbering, add appendix captions with the registered rates and the "development states, chosen by outcome" statement), then DRAFT.md figure references. Commit paper/figs/*, the new experiments/fig_*.py and docs/notes/fig_plan.md by name, and never `git add -A`.

## 6. Open points for the user

- Whether Fig 3 (atlas) stays in the main text or yields to Table 1 if the 8-page cap binds.
- The scene-2 F1 row in the main gallery (the first scene is cleaner; scene 2 has the stronger result, 0 of 720).
- Whether the where-module (W5) panel in Fig 5 counts as part of the main claim or as an appendix item; its confirmatory read (SR 91) did not confirm non-inferiority to ReGuide.
- The A+0 cell: update when its probe lands.

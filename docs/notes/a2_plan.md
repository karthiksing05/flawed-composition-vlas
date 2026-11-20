# A2 distributedness probe (SR 83)

Written 2026-11-19, before the prereg row, any main capture and any main probe.  Predictions with probabilities: experiments/prereg/a2_pred.txt (copied into the ledger row m3_prereg tag a2).
Paper context: paper/DRAFT.md section 4.8 placeholder "[A2] A mechanistic probe per failure is to run", section 6 "The failures sit in identifiable parts of the architecture"; docs/PLAN_PAPER.md W2 / A2;
the side-query machinery of SR 74 (steer_serve.py, steer_conf_serve.py, steer_probe.py, steer_text.py) is reused, not edited.

## Claim
In a standard VLA (frozen pi0.5, pi05_libero) the computation that binds an instruction's target role to a scene entity is distributed: no single component carries it and band ablations are unselective across
failures.  In RAA-sL (pi05_g8v_raas_l_ae, export main6000) the same binding is localised in the role channel (the 4-dim state token and the servo input that is a function of it).  Open loop only: the model is run
on recorded model inputs, nothing is executed.

## What is recorded (capture, `experiments/a2_capture.py` + `experiments/a2_cap_client.py`, jobs `a2_cap.sh`)
The stored object is the dict the model is given (images, state, prompt) at GuidedPolicy.infer, after the role views, the paint and the channel of the RAA family, with the noise seed it drew and the chunk it returned.
Nothing the policy sees or the episode executes changes (steer_serve's SideWrap restores counters and the role sequencer's state after every side query).
Hold-phase observations do not exist in the repo (results hold start states and chunk displacements only), so the capture re-runs each model closed loop on the same start states up to the hold and records there.  This is
the one deviation from "no closed-loop rollouts": the rollouts only produce the observations; every number is read open loop.
* N and K (n2_goal, `results/flawed/n2_states`, `n2_branches_main.json` = the dev states, trials 4-23): the 3rd place-phase query (the 3rd query with ProprioHeld on, >= 10 steps after the hold onset) of each episode.  At that
  query one side query per swap candidate (steer_text.cands('place'): the other targets in the scene, the trained target and the held object excluded) with the target phrase swapped (steer_text.cf_text), then the real query.
  RAA-sL: the swapped instruction goes through the whole front end (lexicon parse, role paint, channel = the estimate of the swapped-in entity minus the end effector, state token), as SR 75's RA probe.
  Tasks: K1 (tid 0), K2 (2), K4 (6), K6 (10), familiar layouts, trained pairings; N1 (16, cream cheese on the plate), N4 (22, bottle on the plate), N5 (24, bottle on the stove), N6 (26, bottle in the bowl): new pairings.
  max 240 steps.  Selection in the probe: per task, the first 12 captured episodes in trial order that have a candidate whose prompt has the same token count as the clean prompt (the first such candidate in cands order).
* F7 (rg_comp states `results/flawed/rg_comp/states`, trials 4-23 = the first 20 of br7_main.json, copied to results/flawed/a2/br7_a2.json): row 0 (the first query after the 10 settle steps) of rgc tid 0 ("put the left
  bowl on the plate") and tid 1 ("put the right bowl on the plate").  SP_R is SP_L with the two bowls swapped, identical images, so the counterfactual of a state is the other task's instruction on the same layout.  pi0.5:
  the clean model input with the partner's prompt.  RAA-sL: the partner task's recorded model input (wa_s.py arm full: resolver, paint, channel).  The referent bowl's position comes from results/flawed/rg_comp/cache_f7.pkl
  (acc[trial].lay.SP, B1 = the task's object, B2 the other bowl; tid 1 has them swapped); the end effector from the pi0.5 capture's state[:3].
The states used are the dev states of SR 70-82 (trials 4-23 for n2 and F7), the same states the rollouts of every arm used; results/l90c/split_sealed.json is not touched.

## Effect measure
d5(chunk) = sum of the first 5 xyz deltas of the unnormalised chunk (SR 74's chunk_delta).  u_cf = the unit vector from the end effector to the counterfactual entity (N / K: the chosen swap candidate's region / object; F7:
the other bowl); u_cor = to the entity the clean instruction names (N / K: the trained or new-pairing target; F7: the task's bowl).  Simulator positions, measurement only.
s_cf = d5 . u_cf (the SR 74 steering projection, unnormalised), s_cor = d5 . u_cor.  Effect of a patch = s_cf(patched) - s_cf(reference run of the same batch); T = effect of the counterfactual run itself.
Noise: one draw per observation (the seed the server drew for the real query), shared by the clean, counterfactual and every patched run.

## Components and runs (experiments/a2_probe.py; hooks on the served model, backbone.py untouched; 237 rows per observation for pi0.5, 245 for RAA-sL, in batches of 19 + a clean reference row)
Writes (a layer sublayer's output before the residual add), patched from the counterfactual run into the clean run:
* vlm.L.attn / vlm.L.mlp, L = 0-17, at all prefix positions (primary) and at the instruction-token positions only (`.ins`);
* exp.L.attn / exp.L.mlp, L = 0-17, the action expert, the counterfactual run's activation at the same denoising step (10 steps), all suffix positions;
* residual stream (cumulative, descriptive): vlm.L.resid.all / .ins, exp.L.resid.all;
* heads: o_proj-input slice of one of 8 heads, layers 3, 9, 15, VLM and expert (48 components), all positions;
* RAA-sL inputs: in.prefix (images + prompt from the counterfactual run, state token and servo clean), in.tok (state token only, servo clean), in.servo (servo input only), in.chan (token + servo, prefix clean).
Ablations on the clean run (zero the band's attn and mlp writes): VLM bands 0-3 / 4-8 / 9-13 / 14-17 at all positions and at the instruction positions, expert bands 0-8 / 9-17; RAA-sL: the channel removed (raw state zeros =
the unrouted-frame convention, token and servo), token only, servo only.
Plumbing rows on the first run only when `--ident` (patch with the clean run's own activation: must reproduce the reference within bf16 batch noise).

## Registered statistics (experiments/a2_read.py, the docstring is the definition)
U1 = the 72 writes {vlm, exp} x L x {attn, mlp}, all positions.  e_c = mean_i effect; e+ = max(e_c, 0); T = mean_i T_i.
PR = (sum e+)^2 / sum e+^2; PRn = PR / 72; top1 = max e+ / sum e+; top3; coverage = sum e+ / T; mass_vlm / mass_exp = the VLM's / the expert's share of sum e+.
RAA-sL: chan_share = e(in.chan) / T (also in.prefix, in.tok, in.servo).  Head statistics (descriptive): PRn and top1 over the 48 heads.
Pooled (ALL = F7 + N + K observations) is the registered unit; the sets F7, N, K are reported separately.
Selectivity: intact_s = mean s_cor(clean) on set s, informative if its CI lower bound > 0 (else 'n/i'); drop(b, s) = (intact_s - abl_s(b)) / intact_s; sel(b, s) = drop(b, s) - drop(b, K) for s in {F7, N}.
Paired bootstrap B = 2000, rng 0 (numpy default_rng(0)), clusters resampled with replacement within each set (F7: both instructions of a layout are one cluster), 95% percentile CI.
Labels (computed in a2_read.py, fixed here):
* gate: T CI lower bound > 0, else the set is UNTESTABLE for the share statistics;
* "pi0.5 distributed": top1 CI upper bound < .3 AND PRn CI lower bound >= .2 (pooled); "localised": top1 CI lower bound >= .5;
* "RAA-sL localised in the channel": chan_share CI lower bound >= .7 (pooled);
* "selective cell": sel CI lower bound > .2; "inverse": CI upper bound < -.2.  Distributed matrix = no SELECTIVE cell among the informative cells; RAA-sL: abl.chan SELECTIVE for N and for F7 is the registered contrast.
Labels read SUPPORTED when the CI lies in the predicted region, otherwise NOT SUPPORTED; REVERSED where the opposite region is entered entirely.

## Not registered / descriptive
Instruction-position patches, residual profiles, heads, mass by stack, coverage, per-set statistics, the per-component profile, the in.* shares per set.

## Scoring
The probabilities in a2_pred.txt are scored as written in docs/notes/sr83.md; the claim is NOT SUPPORTED for a half whose registered label is not met.

## Limits known in advance
(1) one seed per observation; the noise floor of a patched chunk against its reference is .002-.007 in the smoke (bf16, batch), the effects are read in the same units.  (2) A single-component patch of a redundant computation
yields a small effect: PR then measures how the small effects are spread, and coverage (sum e+ / T) says how much of the total they account for; both are reported.  (3) Zero-ablation of a band damages every
set; selectivity is read against K, not in absolute terms.  (4) The pi0.5 F7 set is a failure set (relR 0): the gate may leave it UNTESTABLE.  (5) The N set for pi0.5 is a new-pairing set: its intact
correct-target displacement may sit at the floor, making its ablation cell n/i.  (6) Position patches at the instruction tokens need equal token counts for the two instructions: observations without such a
candidate are skipped and listed.  (7) The state token enters only the action expert's suffix; the VLM prefix never sees the state (discrete_state_input False).  (8) F7 positions are the layout cache's, not re-measured.

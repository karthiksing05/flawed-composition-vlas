# state_proj diagnostic (EXPLORATORY, DESCRIPTIVE, NOT REGISTERED)

Question (novelty_review_adv.md, objection 4): is "the action expert ignores the where-token" an artefact of state_proj being a new, briefly trained layer?
Script experiments/stateproj_diag.py (params only, CPU job 4831411), data results/flawed/raaf/stateproj_diag.json.  No prediction was registered; nothing here is a test.

Setup facts.  state_proj = Linear(32, 1024); kernel random init (flax default, std about 1/sqrt(32)), bias zero.  The state is [dx, dy, dz, role] (normalised), zero-padded to 32, so input dims 0-3 are live and dims 4-31 always receive zero input, hence zero gradient: their columns are an exact sample of the random init (mean column norm 5.665, sd .110, W Frobenius if untouched 32.04).  All arms share the same init (live-dim norms at step 1000 agree to about .01 across arms).  The init itself was not saved, so "change from init" is measured against the untouched pad columns (norm level) and against step 1000 (movement 1000 to 6000).  Weights are not exactly init at step 1000 either, so movement before step 1000 is not in the delta column.

## Table (step 6000; step 1000 in the delta columns)
Live-dim column norm = norm of W[d, :] (1024 values) for d = dx, dy, dz, role; pad reference 5.665.  Delta = norm of (W6000 - W1000)[d, :], in brackets relative to W6000[d, :].

| arm (config, exp) | W Frobenius s1000 / s6000 | bias norm s1000 / s6000 | live col norms s6000 (dx, dy, dz, role) | delta 1000-6000 (dx, dy, dz, role) | synthetic out norm s6000 |
|---|---|---|---|---|---|
| RAA-sL (raas_l_ae, main) | 32.049 / 32.053 | .041 / .138 | 5.598, 5.841, 5.582, 5.651 | .182 (3.3%), .122 (2.1%), .117 (2.1%), .130 (2.3%) | 6.41 |
| A0 (raac_l_ae, tl) | 32.046 / 32.048 | .040 / .133 | 5.586, 5.829, 5.573, 5.658 | .074 (1.3%), .102 (1.7%), .080 (1.4%), .180 (3.2%) | 6.41 |
| A (raac_lf_ae, tl) | 32.048 / 32.097 | .038 / .133 | 5.763, 5.910, 5.586, 5.657 | .495 (8.6%), .376 (6.4%), .136 (2.4%), .146 (2.6%) | 6.48 |
| B (raac_lb_ae, tl) | 32.047 / 32.047 | .038 / .134 | 5.585, 5.830, 5.573, 5.648 | .095 (1.7%), .115 (2.0%), .099 (1.8%), .133 (2.4%) | 6.40 |
| A+0 (raac_lp_ae, tl) | 32.048 / 32.106 | .036 / .114 | 5.917, 5.856, 5.582, 5.610 | .502 (8.5%), .223 (3.8%), .118 (2.1%), .141 (2.5%) | 6.50 |
| A+ (raas_lp_ae, main) | 32.048 / 32.094 | .036 / .115 | 5.803, 5.871, 5.580, 5.647 | .416 (7.2%), .223 (3.8%), .164 (2.9%), .169 (3.0%) | 6.48 |

Synthetic out norm = mean norm of state_proj(x) for 500 draws of x uniform in [-1, 1]^4 (NOT real training frames; the real normalised range can differ).  Item (c), comparison with the other token embeddings entering the action expert, is skipped: it needs a forward pass of the whole model.  Only the state_proj output scale (about 6.4) is reported, with no reference scale.

## Reading
The layer barely moved from its random initialisation in every arm.  The Frobenius norm stays within .3% of the untouched-init level (32.04) and live column norms are within 0-5% of the pad-column mean in every arm; the bias grew from zero to a norm of about .1 (the output is about 6.4).  The largest movement is in A, A+0 and A+ on the dx and dy inputs (relative change 1000 to 6000 of 4-9%, A and A+0 largest in dx), against 1-3% in A0, B and RAA-sL, whose live rows moved about as much as each other, with no dimension standing out.  A, the arm that learned a where-share of .274, is among the arms that moved most, consistent with the layer moving where the relabel data gave a gradient, but A+0 and A+ moved similarly and this note does not link movement to their shares.  The dominant output of state_proj in all arms is therefore the random projection of the input plus small learned offsets, and the learned part is small relative to it.

Evidence for or against the artefact reading, with limits.  Weakly for: for the arms with a null share (RAA-sL, A0, B) the layer is essentially an untrained random projection after 6000 steps, so "the expert ignores the token" is also "the token was never given a learned, informative embedding".  Against reading this as decisive: small weight change does not mean small function change (the expert's attention can read a fixed random embedding of dx, dy, dz, role perfectly well, and the 1024-d random projection of 4 numbers is injective), the VLM is frozen, and Route A shows that data which require the token do move the layer and raise the share.  This diagnostic cannot say how much the share would rise under a longer schedule or a trained init; that needs a new run (for example a learning-rate multiplier on state_proj, or a pretrained-style init) and would be a separate registered test.  Single seed per arm; no uncertainty estimates.
